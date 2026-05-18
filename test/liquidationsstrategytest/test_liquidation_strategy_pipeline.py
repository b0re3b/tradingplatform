# tests/strategy/strategies/liquidations/test_liquidation_strategy_pipeline.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Awaitable, Callable

import pytest

from core.event_bus import Event, EventPriority

from analytics.liquidations.enums import (
    CascadeDirection,
    CascadeSeverity,
    LiquidationSide,
    LiquidationStatus,
)
from analytics.liquidations.models import (
    CascadeDetectionResult,
    LiquidationCluster,
    LiquidationKey,
)

from strategy.strategies.liquidations.base import (
    StrategyRejection,
    make_strategy_scope_key,
    scoped_key_to_string,
)
from strategy.strategies.liquidations.liquidation_cascade_strategy import (
    LiquidationCascadeSignal,
    LiquidationCascadeStrategy,
    LiquidationCascadeStrategyConfig,
)
from strategy.strategies.liquidations.squeeze_reversal_strategy import (
    PendingReversalCandidate,
    SqueezeReversalSignal,
    SqueezeReversalStrategy,
    SqueezeReversalStrategyConfig,
)


pytestmark = pytest.mark.asyncio


# ============================================================================
# In-memory EventBus for strategy pipeline tests
# ============================================================================


@dataclass(slots=True)
class InMemorySubscription:
    topic: str
    handler: Callable[[Event], Awaitable[None]]
    name: str | None = None


class InMemoryEventBus:
    """
    Мінімальний async EventBus double для pipeline-тестів.

    Він навмисно dispatch-ить handlers синхронно в межах await emit(...),
    щоб тести були deterministic і ловили contract-помилки без race condition.
    """

    def __init__(self) -> None:
        self.subscriptions: list[InMemorySubscription] = []
        self.unsubscribed: list[InMemorySubscription] = []
        self.records: list[dict[str, Any]] = []

    def subscribe(
        self,
        topic: str,
        handler: Callable[[Event], Awaitable[None]],
        *,
        name: str | None = None,
    ) -> InMemorySubscription:
        subscription = InMemorySubscription(
            topic=topic,
            handler=handler,
            name=name,
        )
        self.subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, subscription: InMemorySubscription) -> None:
        self.unsubscribed.append(subscription)
        self.subscriptions = [
            item for item in self.subscriptions if item is not subscription
        ]

    async def emit(
        self,
        topic: str,
        payload: Any,
        *,
        priority: EventPriority = EventPriority.NORMAL,
        source: str,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        event = Event(
            topic=topic,
            payload=payload,
            priority=priority,
            source=source,
            correlation_id=correlation_id,
            headers=headers or {},
        )

        self.records.append(
            {
                "topic": topic,
                "payload": payload,
                "priority": priority,
                "source": source,
                "correlation_id": correlation_id,
                "headers": headers or {},
                "event": event,
            }
        )

        for subscription in list(self.subscriptions):
            if subscription.topic == topic:
                await subscription.handler(event)

        return True


class RejectingEventBus(InMemoryEventBus):
    async def emit(
        self,
        topic: str,
        payload: Any,
        *,
        priority: EventPriority = EventPriority.NORMAL,
        source: str,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        self.records.append(
            {
                "topic": topic,
                "payload": payload,
                "priority": priority,
                "source": source,
                "correlation_id": correlation_id,
                "headers": headers or {},
                "event": None,
            }
        )
        return False


class FailingEventBus(InMemoryEventBus):
    async def emit(
        self,
        topic: str,
        payload: Any,
        *,
        priority: EventPriority = EventPriority.NORMAL,
        source: str,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        raise RuntimeError("pipeline EventBus emit failed")


class FakeScheduler:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.removed_job_ids: list[str] = []
        self._counter = 0

    def add_interval_job(
        self,
        *,
        name: str,
        func: Any,
        interval: float,
        run_immediately: bool,
        max_retries: int,
        retry_delay: float,
        timeout: float,
        allow_overlap: bool,
        enabled: bool,
    ) -> str:
        self._counter += 1
        job_id = f"job-{self._counter}"
        self.jobs[job_id] = {
            "name": name,
            "func": func,
            "interval": interval,
            "run_immediately": run_immediately,
            "max_retries": max_retries,
            "retry_delay": retry_delay,
            "timeout": timeout,
            "allow_overlap": allow_overlap,
            "enabled": enabled,
        }
        return job_id

    def remove_job(self, job_id: str) -> None:
        self.removed_job_ids.append(job_id)
        self.jobs.pop(job_id, None)


# ============================================================================
# Helpers
# ============================================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def scope_key(
    *,
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    symbol: str = "BTCUSDT",
    timeframe: str = "realtime",
) -> str:
    return scoped_key_to_string(
        make_strategy_scope_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
    )


def make_metadata(
    *,
    side_imbalance_ratio: Any = 0.86,
    event_imbalance_ratio: Any = 0.80,
    acceleration_ratio: Any = 1.45,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "detector": "cascade_detector",
        "side_imbalance_ratio": side_imbalance_ratio,
        "event_imbalance_ratio": event_imbalance_ratio,
        "acceleration_ratio": acceleration_ratio,
        "long_events": 9,
        "short_events": 1,
        "long_notional_usd": "900000",
        "short_notional_usd": "100000",
    }

    if extra:
        metadata.update(extra)

    return metadata


def make_cluster(
    *,
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    symbol: str = "BTCUSDT",
    timeframe: str = "realtime",
    exchange_symbol: str | None = "BTCUSDT",
    side: LiquidationSide = LiquidationSide.LONG,
    direction: CascadeDirection = CascadeDirection.UP,
    severity: CascadeSeverity = CascadeSeverity.HIGH,
    status: LiquidationStatus = LiquidationStatus.CONFIRMED,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    event_count: int = 12,
    total_notional_usd: Decimal = Decimal("900000"),
    total_quantity: Decimal = Decimal("12"),
    avg_price: Decimal = Decimal("65000"),
    min_price: Decimal = Decimal("64600"),
    max_price: Decimal = Decimal("65400"),
    metadata: dict[str, Any] | None = None,
) -> LiquidationCluster:
    now = utc_now()

    return LiquidationCluster(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
        exchange_symbol=exchange_symbol,
        side=side,
        start_time=start_time or now - timedelta(seconds=4),
        end_time=end_time or now,
        event_count=event_count,
        total_notional_usd=total_notional_usd,
        total_quantity=total_quantity,
        avg_price=avg_price,
        min_price=min_price,
        max_price=max_price,
        direction=direction,
        severity=severity,
        status=status,
        source="test.cascade_detector",
        metadata=metadata or {},
    )


def make_cascade_result(
    *,
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    symbol: str = "BTCUSDT",
    timeframe: str = "realtime",
    exchange_symbol: str | None = "BTCUSDT",
    side: LiquidationSide = LiquidationSide.LONG,
    direction: CascadeDirection = CascadeDirection.UP,
    severity: CascadeSeverity = CascadeSeverity.HIGH,
    status: LiquidationStatus = LiquidationStatus.CONFIRMED,
    detected_at: datetime | None = None,
    confidence: float = 0.86,
    intensity_score: float = 0.78,
    continuation_bias: float = 0.78,
    exhaustion_bias: float = 0.14,
    event_count: int = 12,
    total_notional_usd: Decimal = Decimal("900000"),
    window_seconds: int = 10,
    price_range_pct: float = 0.38,
    correlation_id: str | None = "cascade-corr",
    metadata: dict[str, Any] | None = None,
) -> CascadeDetectionResult:
    cluster = make_cluster(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
        exchange_symbol=exchange_symbol,
        side=side,
        direction=direction,
        severity=severity,
        status=status,
        event_count=event_count,
        total_notional_usd=total_notional_usd,
    )

    return CascadeDetectionResult(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
        exchange_symbol=exchange_symbol,
        side=side,
        direction=direction,
        detected_at=detected_at or utc_now() - timedelta(seconds=1),
        cluster=cluster,
        intensity_score=intensity_score,
        confidence=confidence,
        continuation_bias=continuation_bias,
        exhaustion_bias=exhaustion_bias,
        event_count=event_count,
        total_notional_usd=total_notional_usd,
        window_seconds=window_seconds,
        price_range_pct=price_range_pct,
        severity=severity,
        status=status,
        correlation_id=correlation_id,
        source="test.cascade_detector",
        metadata=metadata or make_metadata(),
    )


def make_exhaustion_result(
    *,
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    symbol: str = "BTCUSDT",
    timeframe: str = "realtime",
    exchange_symbol: str | None = "BTCUSDT",
    side: LiquidationSide = LiquidationSide.LONG,
    direction: CascadeDirection = CascadeDirection.DOWN,
    severity: CascadeSeverity = CascadeSeverity.HIGH,
    status: LiquidationStatus = LiquidationStatus.CONFIRMED,
    detected_at: datetime | None = None,
    confidence: float = 0.88,
    intensity_score: float = 0.80,
    continuation_bias: float = 0.16,
    exhaustion_bias: float = 0.86,
    event_count: int = 12,
    total_notional_usd: Decimal = Decimal("950000"),
    window_seconds: int = 10,
    price_range_pct: float = 0.40,
    correlation_id: str | None = "exhaustion-corr",
    metadata: dict[str, Any] | None = None,
) -> CascadeDetectionResult:
    cluster = make_cluster(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
        exchange_symbol=exchange_symbol,
        side=side,
        direction=direction,
        severity=severity,
        status=status,
        event_count=event_count,
        total_notional_usd=total_notional_usd,
    )

    return CascadeDetectionResult(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
        exchange_symbol=exchange_symbol,
        side=side,
        direction=direction,
        detected_at=detected_at or utc_now() - timedelta(seconds=1),
        cluster=cluster,
        intensity_score=intensity_score,
        confidence=confidence,
        continuation_bias=continuation_bias,
        exhaustion_bias=exhaustion_bias,
        event_count=event_count,
        total_notional_usd=total_notional_usd,
        window_seconds=window_seconds,
        price_range_pct=price_range_pct,
        severity=severity,
        status=status,
        correlation_id=correlation_id,
        source="test.cascade_detector",
        metadata=metadata or make_metadata(),
    )


def make_cascade_config(**overrides: Any) -> LiquidationCascadeStrategyConfig:
    base = {
        "symbol_cooldown_seconds": 0,
        "min_seconds_between_same_side_signals": 0,
        "max_signals_per_symbol_window": 0,
        "deduplicate_by_detected_at": False,
        "deduplicate_same_cluster_signature": False,
        "publish_rejected_events": False,
        "min_side_imbalance_ratio": None,
        "min_event_imbalance_ratio": None,
        "min_acceleration_ratio": None,
    }
    base.update(overrides)
    return LiquidationCascadeStrategyConfig(**base)


def make_squeeze_config(**overrides: Any) -> SqueezeReversalStrategyConfig:
    base = {
        "enable_pending_confirmation": True,
        "confirmation_delay_seconds": 0.0,
        "min_pending_age_seconds": 0.0,
        "pending_ttl_seconds": 30.0,
        "pending_scan_interval_seconds": 0.1,
        "symbol_cooldown_seconds": 0,
        "min_seconds_between_same_side_signals": 0,
        "max_signals_per_symbol_window": 0,
        "deduplicate_by_detected_at": False,
        "deduplicate_same_cluster_signature": False,
        "publish_pending_events": True,
        "publish_rejected_events": False,
        "min_event_imbalance_ratio": None,
    }
    base.update(overrides)

    confirmation_delay = float(base["confirmation_delay_seconds"])
    pending_ttl = float(base["pending_ttl_seconds"])

    if pending_ttl <= confirmation_delay:
        base["pending_ttl_seconds"] = confirmation_delay + 30.0

    min_pending_age = float(base["min_pending_age_seconds"])
    pending_ttl = float(base["pending_ttl_seconds"])

    if min_pending_age > pending_ttl:
        base["pending_ttl_seconds"] = min_pending_age + 30.0

    return SqueezeReversalStrategyConfig(**base)


async def start_pipeline(
    *,
    event_bus: InMemoryEventBus,
    cascade_config: LiquidationCascadeStrategyConfig | None = None,
    squeeze_config: SqueezeReversalStrategyConfig | None = None,
    scheduler: FakeScheduler | None = None,
) -> tuple[LiquidationCascadeStrategy, SqueezeReversalStrategy]:
    cascade_strategy = LiquidationCascadeStrategy(
        event_bus=event_bus,  # type: ignore[arg-type]
        config=cascade_config or make_cascade_config(),
        scheduler=scheduler,  # type: ignore[arg-type]
    )
    squeeze_strategy = SqueezeReversalStrategy(
        event_bus=event_bus,  # type: ignore[arg-type]
        config=squeeze_config or make_squeeze_config(),
        scheduler=scheduler,  # type: ignore[arg-type]
    )

    await cascade_strategy.start()
    await squeeze_strategy.start()

    return cascade_strategy, squeeze_strategy


async def emit_cascade(
    event_bus: InMemoryEventBus,
    result: CascadeDetectionResult,
    *,
    correlation_id: str = "raw-cascade-corr",
) -> bool:
    return await event_bus.emit(
        "analytics.liquidations.cascade_detected",
        result,
        priority=EventPriority.NORMAL,
        source="analytics.liquidations.cascade_detector",
        correlation_id=correlation_id,
        headers={
            "exchange": result.exchange,
            "market_type": result.market_type,
            "symbol": result.symbol,
            "timeframe": result.timeframe,
            "exchange_symbol": result.exchange_symbol,
            "scope": scoped_key_to_string(result.key),
        },
    )


async def emit_exhaustion(
    event_bus: InMemoryEventBus,
    result: CascadeDetectionResult,
    *,
    correlation_id: str = "raw-exhaustion-corr",
) -> bool:
    return await event_bus.emit(
        "analytics.liquidations.exhaustion_detected",
        result,
        priority=EventPriority.NORMAL,
        source="analytics.liquidations.cascade_detector",
        correlation_id=correlation_id,
        headers={
            "exchange": result.exchange,
            "market_type": result.market_type,
            "symbol": result.symbol,
            "timeframe": result.timeframe,
            "exchange_symbol": result.exchange_symbol,
            "scope": scoped_key_to_string(result.key),
        },
    )


def records_for(event_bus: InMemoryEventBus, topic: str) -> list[dict[str, Any]]:
    return [record for record in event_bus.records if record["topic"] == topic]


def payloads_for(event_bus: InMemoryEventBus, topic: str) -> list[Any]:
    return [record["payload"] for record in records_for(event_bus, topic)]


def signal_payloads(event_bus: InMemoryEventBus) -> list[Any]:
    return payloads_for(event_bus, "signal.generated")


def rejection_payloads(event_bus: InMemoryEventBus) -> list[Any]:
    return payloads_for(event_bus, "signal.rejected")


def strategy_output_topics(event_bus: InMemoryEventBus) -> list[str]:
    return [
        record["topic"]
        for record in event_bus.records
        if not record["topic"].startswith("analytics.")
    ]


def assert_headers_have_full_scope(
    headers: dict[str, Any],
    *,
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    symbol: str = "BTCUSDT",
    timeframe: str = "realtime",
    exchange_symbol: str = "BTCUSDT",
) -> None:
    assert headers["exchange"] == exchange
    assert headers["market_type"] == market_type
    assert headers["symbol"] == symbol
    assert headers["timeframe"] == timeframe
    assert headers["exchange_symbol"] == exchange_symbol
    assert headers["scope"] == scope_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )


def assert_no_direct_risk_or_execution_events(event_bus: InMemoryEventBus) -> None:
    forbidden_prefixes = ("risk.", "execution.", "order.", "position.")

    for record in event_bus.records:
        topic = record["topic"]
        assert not topic.startswith(forbidden_prefixes), (
            f"strategy pipeline must not emit direct risk/execution/order/position event: {topic}"
        )


# ============================================================================
# Startup / subscription wiring
# ============================================================================


async def test_pipeline_starts_both_strategies_and_subscribes_to_plural_topics() -> None:
    event_bus = InMemoryEventBus()
    scheduler = FakeScheduler()

    cascade_strategy, squeeze_strategy = await start_pipeline(
        event_bus=event_bus,
        scheduler=scheduler,
        cascade_config=make_cascade_config(publish_diagnostics_snapshots=True),
        squeeze_config=make_squeeze_config(publish_diagnostics_snapshots=True),
    )

    topics = {subscription.topic for subscription in event_bus.subscriptions}
    names = {subscription.name for subscription in event_bus.subscriptions}
    job_names = {job["name"] for job in scheduler.jobs.values()}

    assert cascade_strategy.is_running is True
    assert squeeze_strategy.is_running is True

    assert topics == {
        "analytics.liquidations.cascade_detected",
        "analytics.liquidations.exhaustion_detected",
    }

    assert names == {
        "liquidation_cascade_strategy.on_analytics_event",
        "squeeze_reversal_strategy.on_analytics_event",
    }

    assert "liquidation_cascade_strategy:diagnostics" in job_names
    assert "squeeze_reversal_strategy:diagnostics" in job_names
    assert "squeeze_reversal_strategy:pending_scan" in job_names


async def test_pipeline_stop_unsubscribes_both_strategies_and_removes_jobs() -> None:
    event_bus = InMemoryEventBus()
    scheduler = FakeScheduler()

    cascade_strategy, squeeze_strategy = await start_pipeline(
        event_bus=event_bus,
        scheduler=scheduler,
        cascade_config=make_cascade_config(publish_diagnostics_snapshots=True),
        squeeze_config=make_squeeze_config(publish_diagnostics_snapshots=True),
    )

    job_ids = set(scheduler.jobs.keys())

    await cascade_strategy.stop()
    await squeeze_strategy.stop()

    assert cascade_strategy.is_running is False
    assert squeeze_strategy.is_running is False
    assert event_bus.subscriptions == []
    assert len(event_bus.unsubscribed) == 2
    assert set(scheduler.removed_job_ids) == job_ids


# ============================================================================
# Happy path: analytics -> strategy -> signal.generated
# ============================================================================


async def test_cascade_analytics_event_flows_to_continuation_signal_generated() -> None:
    event_bus = InMemoryEventBus()
    cascade_strategy, squeeze_strategy = await start_pipeline(event_bus=event_bus)

    result = make_cascade_result(
        direction=CascadeDirection.UP,
        side=LiquidationSide.SHORT,
        correlation_id="detector-cascade-corr",
    )

    accepted = await emit_cascade(event_bus, result, correlation_id="bus-cascade-corr")

    assert accepted is True

    signals = signal_payloads(event_bus)

    assert len(signals) == 1
    signal = signals[0]

    assert isinstance(signal, LiquidationCascadeSignal)
    assert signal.strategy_name == "liquidation_cascade_strategy"
    assert signal.signal_type == "continuation"
    assert signal.side == "LONG"
    assert signal.cascade_direction == "up"
    assert signal.exchange == "binance"
    assert signal.market_type == "usdm_futures"
    assert signal.symbol == "BTCUSDT"
    assert signal.timeframe == "realtime"
    assert signal.scope_key == "binance:usdm_futures:BTCUSDT:realtime"
    assert signal.correlation_id == "bus-cascade-corr"

    emitted = records_for(event_bus, "signal.generated")[0]

    assert emitted["source"] == "liquidation_cascade_strategy"
    assert emitted["priority"] is EventPriority.HIGH
    assert emitted["correlation_id"] == "bus-cascade-corr"
    assert emitted["headers"]["signal_type"] == "continuation"
    assert emitted["headers"]["analytics_scope"] == "binance:usdm_futures:BTCUSDT:realtime"

    assert_headers_have_full_scope(emitted["headers"])

    assert cascade_strategy.get_stats()["processed_events"] == 1
    assert cascade_strategy.get_stats()["emitted_signals"] == 1
    assert squeeze_strategy.get_stats()["processed_events"] == 0

    assert_no_direct_risk_or_execution_events(event_bus)


async def test_exhaustion_event_flows_to_pending_then_reversal_signal_generated() -> None:
    event_bus = InMemoryEventBus()
    cascade_strategy, squeeze_strategy = await start_pipeline(
        event_bus=event_bus,
        squeeze_config=make_squeeze_config(
            enable_pending_confirmation=True,
            confirmation_delay_seconds=0.0,
            min_pending_age_seconds=0.0,
            publish_pending_events=True,
        ),
    )

    result = make_exhaustion_result(
        direction=CascadeDirection.DOWN,
        side=LiquidationSide.LONG,
        correlation_id="detector-exhaustion-corr",
    )

    accepted = await emit_exhaustion(
        event_bus,
        result,
        correlation_id="bus-exhaustion-corr",
    )

    assert accepted is True

    state = squeeze_strategy.get_state(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    assert state is not None
    assert state.pending is not None
    assert isinstance(state.pending, PendingReversalCandidate)

    assert strategy_output_topics(event_bus) == [
        "strategy.liquidations.squeeze.pending_created",
    ]

    await squeeze_strategy.process_pending_candidates()

    assert state.pending is None

    outputs = strategy_output_topics(event_bus)

    assert outputs == [
        "strategy.liquidations.squeeze.pending_created",
        "signal.generated",
        "strategy.liquidations.squeeze.pending_confirmed",
    ]

    signal = signal_payloads(event_bus)[0]

    assert isinstance(signal, SqueezeReversalSignal)
    assert signal.strategy_name == "squeeze_reversal_strategy"
    assert signal.signal_type == "reversal"
    assert signal.side == "LONG"
    assert signal.cascade_direction == "down"
    assert signal.scope_key == "binance:usdm_futures:BTCUSDT:realtime"
    assert signal.is_pending_confirmed is True
    assert signal.pending_started_at is not None
    assert signal.pending_confirmed_at is not None
    assert signal.correlation_id == "bus-exhaustion-corr"

    signal_record = records_for(event_bus, "signal.generated")[0]

    assert signal_record["source"] == "squeeze_reversal_strategy"
    assert signal_record["headers"]["signal_type"] == "reversal"
    assert signal_record["headers"]["pending_confirmation"] == "true"
    assert signal_record["headers"]["analytics_scope"] == "binance:usdm_futures:BTCUSDT:realtime"

    assert_headers_have_full_scope(signal_record["headers"])

    pending_created = records_for(
        event_bus,
        "strategy.liquidations.squeeze.pending_created",
    )[0]
    pending_confirmed = records_for(
        event_bus,
        "strategy.liquidations.squeeze.pending_confirmed",
    )[0]

    assert pending_created["payload"]["scope_key"] == "binance:usdm_futures:BTCUSDT:realtime"
    assert pending_confirmed["payload"]["scope_key"] == "binance:usdm_futures:BTCUSDT:realtime"

    assert cascade_strategy.get_stats()["processed_events"] == 0
    assert squeeze_strategy.get_stats()["processed_events"] == 1
    assert squeeze_strategy.get_stats()["pending_created"] == 1
    assert squeeze_strategy.get_stats()["pending_confirmed"] == 1
    assert squeeze_strategy.get_stats()["emitted_signals"] == 1

    assert_no_direct_risk_or_execution_events(event_bus)


# ============================================================================
# Routing isolation
# ============================================================================


async def test_pipeline_routes_cascade_and_exhaustion_topics_to_correct_strategy_only() -> None:
    event_bus = InMemoryEventBus()
    cascade_strategy, squeeze_strategy = await start_pipeline(event_bus=event_bus)

    cascade_result = make_cascade_result(correlation_id="cascade-result-corr")
    exhaustion_result = make_exhaustion_result(correlation_id="exhaustion-result-corr")

    await emit_cascade(event_bus, cascade_result, correlation_id="cascade-bus-corr")
    await emit_exhaustion(event_bus, exhaustion_result, correlation_id="exhaustion-bus-corr")
    await squeeze_strategy.process_pending_candidates()

    signals = signal_payloads(event_bus)

    assert len(signals) == 2
    assert isinstance(signals[0], LiquidationCascadeSignal)
    assert isinstance(signals[1], SqueezeReversalSignal)

    assert signals[0].signal_type == "continuation"
    assert signals[0].correlation_id == "cascade-bus-corr"

    assert signals[1].signal_type == "reversal"
    assert signals[1].correlation_id == "exhaustion-bus-corr"

    assert cascade_strategy.get_stats()["processed_events"] == 1
    assert squeeze_strategy.get_stats()["processed_events"] == 1


async def test_raw_dict_payload_on_analytics_topics_is_rejected_by_payload_type_guard() -> None:
    event_bus = InMemoryEventBus()
    cascade_strategy, squeeze_strategy = await start_pipeline(event_bus=event_bus)

    raw_payload = {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "this_is": "not_a_CascadeDetectionResult",
    }

    await event_bus.emit(
        "analytics.liquidations.cascade_detected",
        raw_payload,
        priority=EventPriority.NORMAL,
        source="bad.test",
        correlation_id="bad-cascade",
    )
    await event_bus.emit(
        "analytics.liquidations.exhaustion_detected",
        raw_payload,
        priority=EventPriority.NORMAL,
        source="bad.test",
        correlation_id="bad-exhaustion",
    )

    assert signal_payloads(event_bus) == []
    assert rejection_payloads(event_bus) == []

    assert cascade_strategy.get_stats()["invalid_payload_skips"] == 1
    assert squeeze_strategy.get_stats()["invalid_payload_skips"] == 1
    assert cascade_strategy.get_stats()["processed_events"] == 0
    assert squeeze_strategy.get_stats()["processed_events"] == 0


async def test_wrong_analytics_topic_does_not_trigger_either_strategy() -> None:
    event_bus = InMemoryEventBus()
    cascade_strategy, squeeze_strategy = await start_pipeline(event_bus=event_bus)

    result = make_cascade_result()

    await event_bus.emit(
        "analytics.liquidation.cascade_detected",
        result,
        priority=EventPriority.NORMAL,
        source="legacy.topic",
        correlation_id="legacy-topic",
    )

    assert signal_payloads(event_bus) == []
    assert rejection_payloads(event_bus) == []
    assert cascade_strategy.get_stats()["processed_events"] == 0
    assert squeeze_strategy.get_stats()["processed_events"] == 0


# ============================================================================
# Full-scope isolation
# ============================================================================


async def test_same_symbol_different_market_type_does_not_share_state_or_pending() -> None:
    event_bus = InMemoryEventBus()

    cascade_strategy, squeeze_strategy = await start_pipeline(
        event_bus=event_bus,
        cascade_config=make_cascade_config(
            symbol_cooldown_seconds=60,
            deduplicate_by_detected_at=False,
            deduplicate_same_cluster_signature=False,
        ),
        squeeze_config=make_squeeze_config(
            confirmation_delay_seconds=60.0,
            publish_pending_events=False,
        ),
    )

    cascade_usdm = make_cascade_result(
        market_type="usdm_futures",
        exchange_symbol="BTCUSDT",
        detected_at=utc_now() - timedelta(seconds=3),
    )
    cascade_coinm = make_cascade_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
        detected_at=utc_now() - timedelta(seconds=2),
    )

    squeeze_usdm = make_exhaustion_result(
        market_type="usdm_futures",
        exchange_symbol="BTCUSDT",
        detected_at=utc_now() - timedelta(seconds=3),
    )
    squeeze_coinm = make_exhaustion_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
        detected_at=utc_now() - timedelta(seconds=2),
    )

    await emit_cascade(event_bus, cascade_usdm, correlation_id="cascade-usdm")
    await emit_cascade(event_bus, cascade_coinm, correlation_id="cascade-coinm")

    await emit_exhaustion(event_bus, squeeze_usdm, correlation_id="squeeze-usdm")
    await emit_exhaustion(event_bus, squeeze_coinm, correlation_id="squeeze-coinm")

    signals = signal_payloads(event_bus)

    assert len(signals) == 2
    assert {signal.scope_key for signal in signals} == {
        "binance:usdm_futures:BTCUSDT:realtime",
        "binance:coinm_futures:BTCUSDT:realtime",
    }

    assert cascade_strategy.get_stats()["tracked_scopes"] == 2
    assert cascade_strategy.get_stats()["tracked_symbols"] == 1
    assert cascade_strategy.get_stats()["cooldown_skips"] == 0

    usdm_cascade_state = cascade_strategy.get_state(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    coinm_cascade_state = cascade_strategy.get_state(
        "binance",
        "BTCUSDT",
        market_type="coinm_futures",
        timeframe="realtime",
    )

    assert usdm_cascade_state is not None
    assert coinm_cascade_state is not None
    assert usdm_cascade_state is not coinm_cascade_state

    usdm_squeeze_state = squeeze_strategy.get_state(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    coinm_squeeze_state = squeeze_strategy.get_state(
        "binance",
        "BTCUSDT",
        market_type="coinm_futures",
        timeframe="realtime",
    )

    assert usdm_squeeze_state is not None
    assert coinm_squeeze_state is not None
    assert usdm_squeeze_state is not coinm_squeeze_state
    assert usdm_squeeze_state.pending is not None
    assert coinm_squeeze_state.pending is not None

    assert usdm_squeeze_state.pending.scope_key == "binance:usdm_futures:BTCUSDT:realtime"
    assert coinm_squeeze_state.pending.scope_key == "binance:coinm_futures:BTCUSDT:realtime"
    assert squeeze_strategy.get_stats()["pending_active"] == 2


async def test_same_symbol_different_timeframe_does_not_share_state_or_pending() -> None:
    event_bus = InMemoryEventBus()

    cascade_strategy, squeeze_strategy = await start_pipeline(
        event_bus=event_bus,
        cascade_config=make_cascade_config(symbol_cooldown_seconds=60),
        squeeze_config=make_squeeze_config(
            confirmation_delay_seconds=60.0,
            publish_pending_events=False,
        ),
    )

    cascade_realtime = make_cascade_result(
        timeframe="realtime",
        detected_at=utc_now() - timedelta(seconds=3),
    )
    cascade_one_minute = make_cascade_result(
        timeframe="1m",
        detected_at=utc_now() - timedelta(seconds=2),
    )

    squeeze_realtime = make_exhaustion_result(
        timeframe="realtime",
        detected_at=utc_now() - timedelta(seconds=3),
    )
    squeeze_one_minute = make_exhaustion_result(
        timeframe="1m",
        detected_at=utc_now() - timedelta(seconds=2),
    )

    await emit_cascade(event_bus, cascade_realtime, correlation_id="cascade-realtime")
    await emit_cascade(event_bus, cascade_one_minute, correlation_id="cascade-1m")

    await emit_exhaustion(event_bus, squeeze_realtime, correlation_id="squeeze-realtime")
    await emit_exhaustion(event_bus, squeeze_one_minute, correlation_id="squeeze-1m")

    signals = signal_payloads(event_bus)

    assert len(signals) == 2
    assert {signal.scope_key for signal in signals} == {
        "binance:usdm_futures:BTCUSDT:realtime",
        "binance:usdm_futures:BTCUSDT:1m",
    }

    assert cascade_strategy.get_stats()["tracked_scopes"] == 2
    assert squeeze_strategy.get_stats()["tracked_scopes"] == 2
    assert squeeze_strategy.get_stats()["pending_active"] == 2

    realtime_pending = squeeze_strategy.get_state(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    one_minute_pending = squeeze_strategy.get_state(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="1m",
    )

    assert realtime_pending is not None
    assert one_minute_pending is not None
    assert realtime_pending.pending is not None
    assert one_minute_pending.pending is not None
    assert realtime_pending.pending is not one_minute_pending.pending


async def test_same_detected_at_is_not_duplicate_across_market_types_in_pipeline() -> None:
    event_bus = InMemoryEventBus()

    cascade_strategy, squeeze_strategy = await start_pipeline(
        event_bus=event_bus,
        cascade_config=make_cascade_config(
            deduplicate_by_detected_at=True,
            deduplicate_same_cluster_signature=False,
        ),
        squeeze_config=make_squeeze_config(
            enable_pending_confirmation=False,
            deduplicate_by_detected_at=True,
            deduplicate_same_cluster_signature=False,
        ),
    )

    same_detected_at = utc_now() - timedelta(seconds=1)

    cascade_usdm = make_cascade_result(
        market_type="usdm_futures",
        exchange_symbol="BTCUSDT",
        detected_at=same_detected_at,
    )
    cascade_coinm = make_cascade_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
        detected_at=same_detected_at,
    )
    cascade_duplicate_usdm = make_cascade_result(
        market_type="usdm_futures",
        exchange_symbol="BTCUSDT",
        detected_at=same_detected_at,
    )

    squeeze_usdm = make_exhaustion_result(
        market_type="usdm_futures",
        exchange_symbol="BTCUSDT",
        detected_at=same_detected_at,
    )
    squeeze_coinm = make_exhaustion_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
        detected_at=same_detected_at,
    )
    squeeze_duplicate_usdm = make_exhaustion_result(
        market_type="usdm_futures",
        exchange_symbol="BTCUSDT",
        detected_at=same_detected_at,
    )

    await emit_cascade(event_bus, cascade_usdm)
    await emit_cascade(event_bus, cascade_coinm)
    await emit_cascade(event_bus, cascade_duplicate_usdm)

    await emit_exhaustion(event_bus, squeeze_usdm)
    await emit_exhaustion(event_bus, squeeze_coinm)
    await emit_exhaustion(event_bus, squeeze_duplicate_usdm)

    signals = signal_payloads(event_bus)

    assert len(signals) == 4

    assert cascade_strategy.get_stats()["emitted_signals"] == 2
    assert cascade_strategy.get_stats()["rejected_events"] == 1
    assert cascade_strategy.get_stats()["duplicate_skips"] == 1

    assert squeeze_strategy.get_stats()["emitted_signals"] == 2
    assert squeeze_strategy.get_stats()["rejected_events"] == 1
    assert squeeze_strategy.get_stats()["duplicate_skips"] == 1

    rejections = [
        rejection
        for rejection in cascade_strategy.get_recent_rejections(limit=10)
        + squeeze_strategy.get_recent_rejections(limit=10)
        if rejection.reason == "duplicate_detected_at"
    ]

    assert len(rejections) == 2
    assert {rejection.scope_key for rejection in rejections} == {
        "binance:usdm_futures:BTCUSDT:realtime",
    }


# ============================================================================
# Scoped filters / rejections
# ============================================================================


async def test_pipeline_rejects_out_of_scope_market_type_before_signal_or_pending() -> None:
    event_bus = InMemoryEventBus()

    cascade_strategy, squeeze_strategy = await start_pipeline(
        event_bus=event_bus,
        cascade_config=make_cascade_config(
            allowed_market_types=("usdm_futures",),
            publish_rejected_events=True,
        ),
        squeeze_config=make_squeeze_config(
            allowed_market_types=("usdm_futures",),
            publish_rejected_events=True,
        ),
    )

    cascade_coinm = make_cascade_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
    )
    squeeze_coinm = make_exhaustion_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
    )

    await emit_cascade(event_bus, cascade_coinm, correlation_id="reject-cascade-coinm")
    await emit_exhaustion(event_bus, squeeze_coinm, correlation_id="reject-squeeze-coinm")

    assert signal_payloads(event_bus) == []
    assert records_for(event_bus, "strategy.liquidations.squeeze.pending_created") == []

    rejections = rejection_payloads(event_bus)

    assert len(rejections) == 2
    assert all(isinstance(item, StrategyRejection) for item in rejections)
    assert {item.reason for item in rejections} == {"market_type_not_allowed"}
    assert {item.scope_key for item in rejections} == {
        "binance:coinm_futures:BTCUSDT:realtime",
    }

    rejection_records = records_for(event_bus, "signal.rejected")

    assert len(rejection_records) == 2

    for record in rejection_records:
        assert_headers_have_full_scope(
            record["headers"],
            market_type="coinm_futures",
            exchange_symbol="BTCUSD_PERP",
        )

    assert cascade_strategy.get_stats()["emitted_signals"] == 0
    assert cascade_strategy.get_stats()["rejected_events"] == 1
    assert squeeze_strategy.get_stats()["emitted_signals"] == 0
    assert squeeze_strategy.get_stats()["pending_created"] == 0
    assert squeeze_strategy.get_stats()["rejected_events"] == 1


async def test_pipeline_rejects_out_of_scope_timeframe_before_signal_or_pending() -> None:
    event_bus = InMemoryEventBus()

    cascade_strategy, squeeze_strategy = await start_pipeline(
        event_bus=event_bus,
        cascade_config=make_cascade_config(
            allowed_timeframes=("realtime",),
            publish_rejected_events=True,
        ),
        squeeze_config=make_squeeze_config(
            allowed_timeframes=("realtime",),
            publish_rejected_events=True,
        ),
    )

    cascade_one_minute = make_cascade_result(timeframe="1m")
    squeeze_one_minute = make_exhaustion_result(timeframe="1m")

    await emit_cascade(event_bus, cascade_one_minute, correlation_id="reject-cascade-1m")
    await emit_exhaustion(event_bus, squeeze_one_minute, correlation_id="reject-squeeze-1m")

    assert signal_payloads(event_bus) == []
    assert records_for(event_bus, "strategy.liquidations.squeeze.pending_created") == []

    rejections = rejection_payloads(event_bus)

    assert len(rejections) == 2
    assert {item.reason for item in rejections} == {"timeframe_not_allowed"}
    assert {item.scope_key for item in rejections} == {
        "binance:usdm_futures:BTCUSDT:1m",
    }

    for record in records_for(event_bus, "signal.rejected"):
        assert_headers_have_full_scope(
            record["headers"],
            timeframe="1m",
        )


# ============================================================================
# Pending replacement / expiry across full scope
# ============================================================================


async def test_squeeze_pending_replacement_is_limited_to_same_full_scope() -> None:
    event_bus = InMemoryEventBus()

    _, squeeze_strategy = await start_pipeline(
        event_bus=event_bus,
        squeeze_config=make_squeeze_config(
            confirmation_delay_seconds=60.0,
            replace_pending_if_score_improves=True,
            min_replacement_score_delta=0.03,
            publish_pending_events=True,
        ),
    )

    weak_usdm = make_exhaustion_result(
        market_type="usdm_futures",
        exchange_symbol="BTCUSDT",
        detected_at=utc_now() - timedelta(seconds=5),
        confidence=0.72,
        exhaustion_bias=0.74,
        continuation_bias=0.20,
        intensity_score=0.62,
    )
    strong_usdm = make_exhaustion_result(
        market_type="usdm_futures",
        exchange_symbol="BTCUSDT",
        detected_at=utc_now() - timedelta(seconds=1),
        confidence=0.96,
        exhaustion_bias=0.96,
        continuation_bias=0.04,
        intensity_score=0.95,
        severity=CascadeSeverity.EXTREME,
        total_notional_usd=Decimal("2000000"),
    )
    strong_coinm = make_exhaustion_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
        detected_at=utc_now(),
        confidence=0.98,
        exhaustion_bias=0.97,
        continuation_bias=0.03,
        intensity_score=0.96,
        severity=CascadeSeverity.EXTREME,
        total_notional_usd=Decimal("2500000"),
    )

    await emit_exhaustion(event_bus, weak_usdm, correlation_id="weak-usdm")

    usdm_state = squeeze_strategy.get_state(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    assert usdm_state is not None
    old_pending = usdm_state.pending
    assert old_pending is not None

    await emit_exhaustion(event_bus, strong_coinm, correlation_id="strong-coinm")

    coinm_state = squeeze_strategy.get_state(
        "binance",
        "BTCUSDT",
        market_type="coinm_futures",
        timeframe="realtime",
    )

    assert coinm_state is not None
    assert coinm_state.pending is not None
    assert usdm_state.pending is old_pending
    assert old_pending.cancelled is False

    await emit_exhaustion(event_bus, strong_usdm, correlation_id="strong-usdm")

    assert usdm_state.pending is not None
    assert usdm_state.pending is not old_pending
    assert old_pending.cancelled is True
    assert old_pending.cancel_reason == "replaced_by_newer_stronger_pending"

    assert squeeze_strategy.get_stats()["pending_created"] == 3
    assert squeeze_strategy.get_stats()["pending_replaced"] == 1
    assert squeeze_strategy.get_stats()["pending_active"] == 2

    pending_created = records_for(
        event_bus,
        "strategy.liquidations.squeeze.pending_created",
    )
    pending_replaced = records_for(
        event_bus,
        "strategy.liquidations.squeeze.pending_replaced",
    )

    assert len(pending_created) == 3
    assert len(pending_replaced) == 1
    assert pending_replaced[0]["payload"]["scope_key"] == "binance:usdm_futures:BTCUSDT:realtime"


async def test_squeeze_pending_expiry_removes_only_expired_scope() -> None:
    event_bus = InMemoryEventBus()

    _, squeeze_strategy = await start_pipeline(
        event_bus=event_bus,
        squeeze_config=make_squeeze_config(
            confirmation_delay_seconds=60.0,
            publish_pending_events=True,
        ),
    )

    usdm = make_exhaustion_result(
        market_type="usdm_futures",
        exchange_symbol="BTCUSDT",
    )
    coinm = make_exhaustion_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
    )

    await emit_exhaustion(event_bus, usdm, correlation_id="usdm")
    await emit_exhaustion(event_bus, coinm, correlation_id="coinm")

    usdm_state = squeeze_strategy.get_state(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    coinm_state = squeeze_strategy.get_state(
        "binance",
        "BTCUSDT",
        market_type="coinm_futures",
        timeframe="realtime",
    )

    assert usdm_state is not None
    assert coinm_state is not None
    assert usdm_state.pending is not None
    assert coinm_state.pending is not None

    usdm_state.pending.expires_at = utc_now() - timedelta(seconds=1)

    await squeeze_strategy.process_pending_candidates()

    assert usdm_state.pending is None
    assert coinm_state.pending is not None

    assert squeeze_strategy.get_stats()["pending_expired"] == 1
    assert squeeze_strategy.get_stats()["pending_active"] == 1

    expired_records = records_for(
        event_bus,
        "strategy.liquidations.squeeze.pending_expired",
    )

    assert len(expired_records) == 1
    assert expired_records[0]["payload"]["scope_key"] == "binance:usdm_futures:BTCUSDT:realtime"
    assert expired_records[0]["headers"]["scope"] == "binance:usdm_futures:BTCUSDT:realtime"


# ============================================================================
# Mixed burst: valid, invalid, duplicate, filtered
# ============================================================================


async def test_pipeline_mixed_burst_produces_only_expected_clean_outputs() -> None:
    event_bus = InMemoryEventBus()

    cascade_strategy, squeeze_strategy = await start_pipeline(
        event_bus=event_bus,
        cascade_config=make_cascade_config(
            allowed_market_types=("usdm_futures",),
            publish_rejected_events=True,
            deduplicate_by_detected_at=True,
            deduplicate_same_cluster_signature=False,
        ),
        squeeze_config=make_squeeze_config(
            enable_pending_confirmation=False,
            allowed_market_types=("usdm_futures",),
            publish_rejected_events=True,
            deduplicate_by_detected_at=True,
            deduplicate_same_cluster_signature=False,
        ),
    )

    same_detected_at = utc_now() - timedelta(seconds=1)

    valid_cascade = make_cascade_result(
        market_type="usdm_futures",
        detected_at=same_detected_at,
    )
    duplicate_cascade = make_cascade_result(
        market_type="usdm_futures",
        detected_at=same_detected_at,
    )
    filtered_cascade = make_cascade_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
    )

    valid_squeeze = make_exhaustion_result(
        market_type="usdm_futures",
        detected_at=same_detected_at,
    )
    duplicate_squeeze = make_exhaustion_result(
        market_type="usdm_futures",
        detected_at=same_detected_at,
    )
    filtered_squeeze = make_exhaustion_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
    )

    await emit_cascade(event_bus, valid_cascade, correlation_id="valid-cascade")
    await emit_cascade(event_bus, duplicate_cascade, correlation_id="duplicate-cascade")
    await emit_cascade(event_bus, filtered_cascade, correlation_id="filtered-cascade")

    await emit_exhaustion(event_bus, valid_squeeze, correlation_id="valid-squeeze")
    await emit_exhaustion(event_bus, duplicate_squeeze, correlation_id="duplicate-squeeze")
    await emit_exhaustion(event_bus, filtered_squeeze, correlation_id="filtered-squeeze")

    signals = signal_payloads(event_bus)
    rejections = rejection_payloads(event_bus)

    assert len(signals) == 2
    assert len(rejections) == 4

    assert {type(signal) for signal in signals} == {
        LiquidationCascadeSignal,
        SqueezeReversalSignal,
    }

    rejection_reasons = [rejection.reason for rejection in rejections]

    assert rejection_reasons.count("duplicate_detected_at") == 2
    assert rejection_reasons.count("market_type_not_allowed") == 2

    assert cascade_strategy.get_stats()["emitted_signals"] == 1
    assert cascade_strategy.get_stats()["rejected_events"] == 2
    assert cascade_strategy.get_stats()["duplicate_skips"] == 1
    assert cascade_strategy.get_stats()["filter_skips"] == 1

    assert squeeze_strategy.get_stats()["emitted_signals"] == 1
    assert squeeze_strategy.get_stats()["rejected_events"] == 2
    assert squeeze_strategy.get_stats()["duplicate_skips"] == 1
    assert squeeze_strategy.get_stats()["filter_skips"] == 1

    assert_no_direct_risk_or_execution_events(event_bus)


# ============================================================================
# Diagnostics across both strategies
# ============================================================================


async def test_pipeline_diagnostics_snapshots_are_independent_and_scope_aware() -> None:
    event_bus = InMemoryEventBus()

    cascade_strategy, squeeze_strategy = await start_pipeline(
        event_bus=event_bus,
        cascade_config=make_cascade_config(publish_diagnostics_snapshots=True),
        squeeze_config=make_squeeze_config(
            enable_pending_confirmation=False,
            publish_diagnostics_snapshots=True,
        ),
    )

    await emit_cascade(event_bus, make_cascade_result(), correlation_id="diag-cascade")
    await emit_exhaustion(event_bus, make_exhaustion_result(), correlation_id="diag-squeeze")

    await cascade_strategy.publish_diagnostics_snapshot()
    await squeeze_strategy.publish_diagnostics_snapshot()

    cascade_snapshots = records_for(
        event_bus,
        "strategy.liquidations.cascade.snapshot",
    )
    squeeze_snapshots = records_for(
        event_bus,
        "strategy.liquidations.squeeze.snapshot",
    )

    assert len(cascade_snapshots) == 1
    assert len(squeeze_snapshots) == 1

    cascade_payload = cascade_snapshots[0]["payload"]
    squeeze_payload = squeeze_snapshots[0]["payload"]

    assert cascade_payload["strategy_name"] == "liquidation_cascade_strategy"
    assert cascade_payload["signal_type"] == "continuation"
    assert cascade_payload["scope"] == "exchange:market_type:symbol:timeframe"
    assert cascade_payload["stats"]["tracked_scopes"] == 1
    assert cascade_payload["hot_symbols"][0]["scope_key"] == "binance:usdm_futures:BTCUSDT:realtime"

    assert squeeze_payload["strategy_name"] == "squeeze_reversal_strategy"
    assert squeeze_payload["signal_type"] == "reversal"
    assert squeeze_payload["scope"] == "exchange:market_type:symbol:timeframe"
    assert squeeze_payload["stats"]["tracked_scopes"] == 1
    assert squeeze_payload["hot_symbols"][0]["scope_key"] == "binance:usdm_futures:BTCUSDT:realtime"


# ============================================================================
# EventBus failure safety
# ============================================================================


async def test_pipeline_signal_emit_false_does_not_update_state_as_successful() -> None:
    event_bus = RejectingEventBus()

    cascade_strategy = LiquidationCascadeStrategy(
        event_bus=event_bus,  # type: ignore[arg-type]
        config=make_cascade_config(),
    )
    squeeze_strategy = SqueezeReversalStrategy(
        event_bus=event_bus,  # type: ignore[arg-type]
        config=make_squeeze_config(enable_pending_confirmation=False),
    )

    await cascade_strategy.start()
    await squeeze_strategy.start()

    # RejectingEventBus не dispatch-ить input events, тому тут напряму викликаємо handlers.
    cascade_result = make_cascade_result()
    squeeze_result = make_exhaustion_result()

    await cascade_strategy._on_bus_event(
        Event(
            topic="analytics.liquidations.cascade_detected",
            payload=cascade_result,
            source="test",
            correlation_id="cascade-false",
        )
    )
    await squeeze_strategy._on_bus_event(
        Event(
            topic="analytics.liquidations.exhaustion_detected",
            payload=squeeze_result,
            source="test",
            correlation_id="squeeze-false",
        )
    )

    cascade_state = cascade_strategy.get_state(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    squeeze_state = squeeze_strategy.get_state(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    assert cascade_state is not None
    assert squeeze_state is not None

    assert cascade_strategy.get_stats()["emitted_signals"] == 0
    assert squeeze_strategy.get_stats()["emitted_signals"] == 0

    assert cascade_state.total_signals_emitted == 0
    assert squeeze_state.total_signals_emitted == 0
    assert cascade_state.last_signal_at is None
    assert squeeze_state.last_signal_at is None


async def test_pipeline_eventbus_exception_is_recorded_without_success_state_mutation() -> None:
    event_bus = FailingEventBus()

    cascade_strategy = LiquidationCascadeStrategy(
        event_bus=event_bus,  # type: ignore[arg-type]
        config=make_cascade_config(),
    )
    squeeze_strategy = SqueezeReversalStrategy(
        event_bus=event_bus,  # type: ignore[arg-type]
        config=make_squeeze_config(enable_pending_confirmation=False),
    )

    await cascade_strategy.start()
    await squeeze_strategy.start()

    cascade_result = make_cascade_result()
    squeeze_result = make_exhaustion_result()

    await cascade_strategy._on_bus_event(
        Event(
            topic="analytics.liquidations.cascade_detected",
            payload=cascade_result,
            source="test",
            correlation_id="cascade-exception",
        )
    )
    await squeeze_strategy._on_bus_event(
        Event(
            topic="analytics.liquidations.exhaustion_detected",
            payload=squeeze_result,
            source="test",
            correlation_id="squeeze-exception",
        )
    )

    assert cascade_strategy.get_stats()["emitted_signals"] == 0
    assert squeeze_strategy.get_stats()["emitted_signals"] == 0
    assert "pipeline EventBus emit failed" in cascade_strategy.get_stats()["last_error"]
    assert "pipeline EventBus emit failed" in squeeze_strategy.get_stats()["last_error"]

    cascade_state = cascade_strategy.get_state(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    squeeze_state = squeeze_strategy.get_state(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    assert cascade_state is not None
    assert squeeze_state is not None
    assert cascade_state.total_signals_emitted == 0
    assert squeeze_state.total_signals_emitted == 0