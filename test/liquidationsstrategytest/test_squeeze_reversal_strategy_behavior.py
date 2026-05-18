# tests/strategy/strategies/liquidations/test_squeeze_reversal_strategy_behavior.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

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
    liquidation_key_to_dict,
)

from strategy.strategies.liquidations.base import (
    StrategyRejection,
    make_strategy_scope_key,
    scoped_key_to_string,
)
from strategy.strategies.liquidations.squeeze_reversal_strategy import (
    PendingReversalCandidate,
    SqueezeReversalSignal,
    SqueezeReversalStrategy,
    SqueezeReversalStrategyConfig,
    SymbolSqueezeStrategyState,
)


# ============================================================================
# Test doubles
# ============================================================================


@dataclass(slots=True)
class FakeSubscription:
    topic: str
    handler: Any
    name: str | None = None


class FakeEventBus:
    def __init__(self) -> None:
        self.subscriptions: list[FakeSubscription] = []
        self.unsubscribed: list[FakeSubscription] = []
        self.emitted: list[dict[str, Any]] = []

    def subscribe(
        self,
        topic: str,
        handler: Any,
        *,
        name: str | None = None,
    ) -> FakeSubscription:
        subscription = FakeSubscription(
            topic=topic,
            handler=handler,
            name=name,
        )
        self.subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, subscription: FakeSubscription) -> None:
        self.unsubscribed.append(subscription)

    async def emit(
        self,
        topic: str,
        payload: Any,
        *,
        priority: EventPriority,
        source: str,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        self.emitted.append(
            {
                "topic": topic,
                "payload": payload,
                "priority": priority,
                "source": source,
                "correlation_id": correlation_id,
                "headers": headers or {},
            }
        )
        return True


class RejectingEventBus(FakeEventBus):
    async def emit(
        self,
        topic: str,
        payload: Any,
        *,
        priority: EventPriority,
        source: str,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        return False


class FailingEventBus(FakeEventBus):
    async def emit(
        self,
        topic: str,
        payload: Any,
        *,
        priority: EventPriority,
        source: str,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        raise RuntimeError("event bus emit failed")


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
# Fixtures / helpers
# ============================================================================


@pytest.fixture()
def event_bus() -> FakeEventBus:
    return FakeEventBus()


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
    direction: CascadeDirection = CascadeDirection.DOWN,
    severity: CascadeSeverity = CascadeSeverity.HIGH,
    status: LiquidationStatus = LiquidationStatus.CONFIRMED,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    event_count: int = 10,
    total_notional_usd: Decimal = Decimal("900000"),
    total_quantity: Decimal = Decimal("10"),
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


def make_result(
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
    confidence: float = 0.86,
    intensity_score: float = 0.78,
    continuation_bias: float = 0.18,
    exhaustion_bias: float = 0.82,
    event_count: int = 10,
    total_notional_usd: Decimal = Decimal("900000"),
    window_seconds: int = 10,
    price_range_pct: float = 0.38,
    cluster: LiquidationCluster | None = None,
    correlation_id: str | None = "analytics-corr-1",
    metadata: dict[str, Any] | None = None,
) -> CascadeDetectionResult:
    resolved_cluster = cluster or make_cluster(
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
        cluster=resolved_cluster,
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


def make_event(
    payload: Any,
    *,
    topic: str = "analytics.liquidations.exhaustion_detected",
    correlation_id: str | None = "bus-corr-1",
) -> Event:
    return Event(
        topic=topic,
        payload=payload,
        priority=EventPriority.NORMAL,
        source="analytics.liquidations.cascade_detector",
        correlation_id=correlation_id,
    )


def make_fast_pending_config(**overrides: Any) -> SqueezeReversalStrategyConfig:
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

    if float(base["min_pending_age_seconds"]) > float(base["pending_ttl_seconds"]):
        base["pending_ttl_seconds"] = float(base["min_pending_age_seconds"]) + 30.0

    return SqueezeReversalStrategyConfig(**base)


def make_strategy(
    *,
    event_bus: FakeEventBus,
    config: SqueezeReversalStrategyConfig | None = None,
    scheduler: FakeScheduler | None = None,
) -> SqueezeReversalStrategy:
    return SqueezeReversalStrategy(
        event_bus=event_bus,  # type: ignore[arg-type]
        config=config,
        scheduler=scheduler,  # type: ignore[arg-type]
    )


def emitted_topics(event_bus: FakeEventBus) -> list[str]:
    return [item["topic"] for item in event_bus.emitted]


def emitted_payloads(event_bus: FakeEventBus, topic: str) -> list[Any]:
    return [
        item["payload"]
        for item in event_bus.emitted
        if item["topic"] == topic
    ]


def latest_signal(event_bus: FakeEventBus) -> SqueezeReversalSignal:
    payloads = emitted_payloads(event_bus, "signal.generated")
    assert payloads, "expected at least one signal.generated event"
    signal = payloads[-1]
    assert isinstance(signal, SqueezeReversalSignal)
    return signal


def latest_rejection(strategy: SqueezeReversalStrategy) -> StrategyRejection:
    rejections = strategy.get_recent_rejections(limit=1)
    assert rejections, "expected at least one rejection"
    return rejections[0]


def get_state(
    strategy: SqueezeReversalStrategy,
    result: CascadeDetectionResult,
) -> SymbolSqueezeStrategyState:
    state = strategy.get_state(
        result.exchange,
        result.symbol,
        market_type=result.market_type,
        timeframe=result.timeframe,
    )
    assert isinstance(state, SymbolSqueezeStrategyState)
    return state


def assert_no_risk_or_execution_events(event_bus: FakeEventBus) -> None:
    forbidden_prefixes = ("risk.", "execution.", "order.", "position.")

    for topic in emitted_topics(event_bus):
        assert not topic.startswith(forbidden_prefixes), (
            f"strategy must not emit direct risk/execution/order/position event: {topic}"
        )


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


# ============================================================================
# Lifecycle / scheduler contract
# ============================================================================


@pytest.mark.asyncio
async def test_start_subscribes_to_plural_exhaustion_topic(
    event_bus: FakeEventBus,
) -> None:
    strategy = make_strategy(event_bus=event_bus)

    await strategy.start()

    assert strategy.is_running is True
    assert strategy.is_registered is True
    assert len(event_bus.subscriptions) == 1
    assert event_bus.subscriptions[0].topic == "analytics.liquidations.exhaustion_detected"
    assert event_bus.subscriptions[0].name == "squeeze_reversal_strategy.on_analytics_event"


@pytest.mark.asyncio
async def test_start_registers_pending_scan_job_even_when_diagnostics_disabled(
    event_bus: FakeEventBus,
) -> None:
    scheduler = FakeScheduler()
    config = make_fast_pending_config(
        publish_diagnostics_snapshots=False,
        pending_scan_interval_seconds=0.25,
    )
    strategy = make_strategy(
        event_bus=event_bus,
        config=config,
        scheduler=scheduler,
    )

    await strategy.start()

    job_names = {job["name"] for job in scheduler.jobs.values()}

    assert "squeeze_reversal_strategy:pending_scan" in job_names
    assert strategy.get_stats()["pending_scan_job_registered"] is True
    assert strategy.get_stats()["diagnostics_job_registered"] is False


@pytest.mark.asyncio
async def test_start_registers_diagnostics_and_pending_scan_jobs_when_both_enabled(
    event_bus: FakeEventBus,
) -> None:
    scheduler = FakeScheduler()
    config = make_fast_pending_config(
        publish_diagnostics_snapshots=True,
        diagnostics_interval_seconds=3.0,
        pending_scan_interval_seconds=0.25,
    )
    strategy = make_strategy(
        event_bus=event_bus,
        config=config,
        scheduler=scheduler,
    )

    await strategy.start()

    job_names = {job["name"] for job in scheduler.jobs.values()}

    assert "squeeze_reversal_strategy:diagnostics" in job_names
    assert "squeeze_reversal_strategy:pending_scan" in job_names
    assert strategy.get_stats()["diagnostics_job_registered"] is True
    assert strategy.get_stats()["pending_scan_job_registered"] is True


@pytest.mark.asyncio
async def test_start_without_scheduler_is_allowed_but_pending_can_be_processed_manually(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(enable_pending_confirmation=True)
    strategy = make_strategy(
        event_bus=event_bus,
        config=config,
        scheduler=None,
    )

    await strategy.start()

    assert strategy.is_running is True
    assert strategy.get_stats()["pending_scan_job_registered"] is False

    result = make_result()
    await strategy._on_bus_event(make_event(result))

    state = get_state(strategy, result)
    assert state.pending is not None

    await strategy.process_pending_candidates()

    assert state.pending is None
    assert strategy.get_stats()["pending_confirmed"] == 1
    assert strategy.get_stats()["emitted_signals"] == 1


@pytest.mark.asyncio
async def test_stop_unsubscribes_and_removes_all_scheduler_jobs(
    event_bus: FakeEventBus,
) -> None:
    scheduler = FakeScheduler()
    config = make_fast_pending_config(publish_diagnostics_snapshots=True)
    strategy = make_strategy(
        event_bus=event_bus,
        config=config,
        scheduler=scheduler,
    )

    await strategy.start()

    subscription = event_bus.subscriptions[0]
    job_ids = set(scheduler.jobs.keys())

    await strategy.stop()

    assert strategy.is_running is False
    assert strategy.is_registered is False
    assert event_bus.unsubscribed == [subscription]
    assert set(scheduler.removed_job_ids) == job_ids
    assert strategy.get_stats()["pending_scan_job_registered"] is False
    assert strategy.get_stats()["diagnostics_job_registered"] is False


# ============================================================================
# Direction mapping / direct signal mode
# ============================================================================


def test_direction_to_trade_side_is_reversal_not_continuation(
    event_bus: FakeEventBus,
) -> None:
    strategy = make_strategy(event_bus=event_bus)

    assert strategy.direction_to_trade_side(
        make_result(direction=CascadeDirection.DOWN)
    ) == "LONG"
    assert strategy.direction_to_trade_side(
        make_result(direction=CascadeDirection.UP)
    ) == "SHORT"
    assert strategy.direction_to_trade_side(
        make_result(direction=CascadeDirection.UNKNOWN)
    ) == "FLAT"


@pytest.mark.asyncio
async def test_down_exhaustion_emits_long_signal_without_pending(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(enable_pending_confirmation=False)
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(direction=CascadeDirection.DOWN)
    event = make_event(result, correlation_id="direct-down")

    await strategy._on_bus_event(event)

    signal = latest_signal(event_bus)

    assert signal.side == "LONG"
    assert signal.is_long is True
    assert signal.is_short is False
    assert signal.signal_type == "reversal"
    assert signal.strategy_name == "squeeze_reversal_strategy"
    assert signal.source_event_id == event.event_id
    assert signal.correlation_id == "direct-down"
    assert signal.pending_started_at is None
    assert signal.pending_confirmed_at is not None
    assert signal.is_pending_confirmed is False

    assert strategy.get_stats()["emitted_signals"] == 1
    assert strategy.get_stats()["pending_created"] == 0
    assert_no_risk_or_execution_events(event_bus)


@pytest.mark.asyncio
async def test_up_exhaustion_emits_short_signal_without_pending(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(enable_pending_confirmation=False)
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(
        direction=CascadeDirection.UP,
        side=LiquidationSide.SHORT,
    )
    event = make_event(result, correlation_id="direct-up")

    await strategy._on_bus_event(event)

    signal = latest_signal(event_bus)

    assert signal.side == "SHORT"
    assert signal.is_long is False
    assert signal.is_short is True
    assert signal.cascade_direction == CascadeDirection.UP.value
    assert signal.correlation_id == "direct-up"

    assert strategy.get_stats()["emitted_signals"] == 1
    assert strategy.get_stats()["rejected_events"] == 0


@pytest.mark.asyncio
async def test_direct_signal_contract_contains_full_scope_and_exhaustion_fields(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(enable_pending_confirmation=False)
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(
        exchange="bybit",
        market_type="linear",
        symbol="ETH/USDT",
        timeframe="realtime",
        exchange_symbol="ETHUSDT",
        direction=CascadeDirection.DOWN,
        side=LiquidationSide.LONG,
        confidence=0.93,
        intensity_score=0.89,
        exhaustion_bias=0.91,
        continuation_bias=0.14,
        total_notional_usd=Decimal("1500000"),
        metadata=make_metadata(
            side_imbalance_ratio=0.92,
            event_imbalance_ratio=0.87,
            acceleration_ratio=1.75,
        ),
    )
    event = make_event(result, correlation_id="direct-contract")

    await strategy._on_bus_event(event)

    signal = latest_signal(event_bus)

    assert signal.exchange == "bybit"
    assert signal.market_type == "linear"
    assert signal.symbol == "ETHUSDT"
    assert signal.timeframe == "realtime"
    assert signal.exchange_symbol == "ETHUSDT"
    assert signal.scope_key == "bybit:linear:ETHUSDT:realtime"

    assert signal.side == "LONG"
    assert signal.confidence == 0.93
    assert 0.0 <= signal.score <= 1.0
    assert signal.intensity_score == 0.89
    assert signal.exhaustion_bias == 0.91
    assert signal.continuation_bias == 0.14
    assert signal.bias_delta == result.bias_delta
    assert signal.event_type == result.event_type.value
    assert signal.status == result.status.value
    assert signal.side_imbalance_ratio == 0.92
    assert signal.event_imbalance_ratio == 0.87
    assert signal.acceleration_ratio == 1.75
    assert signal.cluster_duration_seconds == result.cluster.duration_seconds
    assert signal.cluster_avg_notional_per_event == result.cluster.avg_notional_per_event

    assert "squeeze reversal after liquidation exhaustion" in signal.reason
    assert "scope=bybit:linear:ETHUSDT:realtime" in signal.reason
    assert "exhaustion_bias=0.910" in signal.reason
    assert "bias_delta=0.770" in signal.reason

    assert "scope" in signal.metadata
    assert "strategy" in signal.metadata
    assert "analytics" in signal.metadata
    assert "cluster" in signal.metadata
    assert "squeeze_reversal" in signal.metadata

    squeeze_meta = signal.metadata["squeeze_reversal"]
    assert squeeze_meta["strategy_model"] == "exhaustion_reversal"
    assert squeeze_meta["favors_exhaustion"] is True
    assert squeeze_meta["favors_continuation"] is False
    assert squeeze_meta["pending"]["enabled"] is False


@pytest.mark.asyncio
async def test_signal_generated_event_has_full_scope_headers_and_no_direct_risk_or_execution(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(enable_pending_confirmation=False)
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(
        exchange="okx",
        market_type="swap",
        symbol="BTC-USDT",
        timeframe="1m",
        exchange_symbol="BTC-USDT-SWAP",
    )
    event = make_event(result, correlation_id="signal-corr")

    await strategy._on_bus_event(event)

    assert emitted_topics(event_bus) == ["signal.generated"]
    assert_no_risk_or_execution_events(event_bus)

    emitted = event_bus.emitted[0]

    assert emitted["source"] == "squeeze_reversal_strategy"
    assert emitted["priority"] is EventPriority.HIGH
    assert emitted["correlation_id"] == "signal-corr"

    headers = emitted["headers"]

    assert headers["strategy"] == "squeeze_reversal_strategy"
    assert headers["signal_type"] == "reversal"
    assert headers["source_event_id"] == event.event_id
    assert headers["source_topic"] == event.topic
    assert headers["side"] == "LONG"
    assert headers["pending_confirmation"] == "false"
    assert headers["analytics_event_type"] == result.event_type.value
    assert headers["analytics_status"] == result.status.value
    assert headers["analytics_scope"] == "okx:swap:BTCUSDT:1m"

    assert_headers_have_full_scope(
        headers,
        exchange="okx",
        market_type="swap",
        symbol="BTCUSDT",
        timeframe="1m",
        exchange_symbol="BTC-USDT-SWAP",
    )


# ============================================================================
# Pending confirmation behavior
# ============================================================================


@pytest.mark.asyncio
async def test_pending_candidate_is_created_with_full_scope_and_pending_event_headers(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        enable_pending_confirmation=True,
        confirmation_delay_seconds=60.0,
        publish_pending_events=True,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(
        exchange="okx",
        market_type="swap",
        symbol="BTC-USDT",
        timeframe="1m",
        exchange_symbol="BTC-USDT-SWAP",
    )
    event = make_event(result, correlation_id="pending-corr")

    await strategy._on_bus_event(event)

    state = get_state(strategy, result)
    candidate = state.pending

    assert isinstance(candidate, PendingReversalCandidate)
    assert candidate.exchange == "okx"
    assert candidate.market_type == "swap"
    assert candidate.symbol == "BTCUSDT"
    assert candidate.timeframe == "1m"
    assert candidate.exchange_symbol == "BTC-USDT-SWAP"
    assert candidate.key == ("okx", "swap", "BTCUSDT", "1m")
    assert candidate.scope_key == "okx:swap:BTCUSDT:1m"
    assert candidate.result is result
    assert candidate.correlation_id == "pending-corr"
    assert candidate.source_event_id == event.event_id
    assert candidate.cluster_signature is not None
    assert candidate.quality_snapshot["scope_key"] == "okx:swap:BTCUSDT:1m"

    assert strategy.get_stats()["pending_created"] == 1
    assert strategy.get_stats()["pending_active"] == 1
    assert strategy.get_stats()["emitted_signals"] == 0

    assert emitted_topics(event_bus) == ["strategy.liquidations.squeeze.pending_created"]

    emitted = event_bus.emitted[0]
    payload = emitted["payload"]

    assert payload["state"] == "pending_created"
    assert payload["market_type"] == "swap"
    assert payload["timeframe"] == "1m"
    assert payload["scope_key"] == "okx:swap:BTCUSDT:1m"

    assert_headers_have_full_scope(
        emitted["headers"],
        exchange="okx",
        market_type="swap",
        symbol="BTCUSDT",
        timeframe="1m",
        exchange_symbol="BTC-USDT-SWAP",
    )


@pytest.mark.asyncio
async def test_process_pending_candidates_confirms_ready_candidate_and_emits_signal(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        enable_pending_confirmation=True,
        confirmation_delay_seconds=0.0,
        min_pending_age_seconds=0.0,
        publish_pending_events=True,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result()
    event = make_event(result, correlation_id="pending-confirm")

    await strategy._on_bus_event(event)
    state = get_state(strategy, result)
    candidate = state.pending

    assert candidate is not None

    await strategy.process_pending_candidates()

    assert state.pending is None
    assert strategy.get_stats()["pending_confirmed"] == 1
    assert strategy.get_stats()["emitted_signals"] == 1
    assert strategy.get_stats()["pending_active"] == 0

    assert emitted_topics(event_bus) == [
        "strategy.liquidations.squeeze.pending_created",
        "signal.generated",
        "strategy.liquidations.squeeze.pending_confirmed",
    ]

    signal = latest_signal(event_bus)

    assert signal.is_pending_confirmed is True
    assert signal.pending_started_at == candidate.created_at
    assert signal.pending_confirmed_at is not None
    assert signal.confirmation_delay_seconds is not None
    assert signal.source_event_id == event.event_id
    assert signal.correlation_id == "pending-confirm"
    assert signal.scope_key == "binance:usdm_futures:BTCUSDT:realtime"

    confirmed_payload = event_bus.emitted[-1]["payload"]
    confirmed_headers = event_bus.emitted[-1]["headers"]

    assert confirmed_payload["state"] == "pending_confirmed"
    assert confirmed_payload["scope_key"] == "binance:usdm_futures:BTCUSDT:realtime"
    assert confirmed_headers["state"] == "pending_confirmed"
    assert confirmed_headers["scope"] == "binance:usdm_futures:BTCUSDT:realtime"


@pytest.mark.asyncio
async def test_pending_candidate_not_confirmed_before_confirm_after(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        confirmation_delay_seconds=60.0,
        min_pending_age_seconds=0.0,
        publish_pending_events=False,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result()

    await strategy._on_bus_event(make_event(result))
    state = get_state(strategy, result)

    await strategy.process_pending_candidates()

    assert state.pending is not None
    assert strategy.get_stats()["pending_confirmed"] == 0
    assert strategy.get_stats()["emitted_signals"] == 0


@pytest.mark.asyncio
async def test_pending_candidate_expires_and_publishes_full_scope_expiration_event(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        confirmation_delay_seconds=60.0,
        pending_ttl_seconds=120.0,
        publish_pending_events=True,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result()

    await strategy._on_bus_event(make_event(result, correlation_id="expire-corr"))
    state = get_state(strategy, result)
    candidate = state.pending

    assert candidate is not None

    candidate.expires_at = utc_now() - timedelta(seconds=1)

    await strategy.process_pending_candidates()

    assert state.pending is None
    assert candidate.cancelled is True
    assert candidate.cancel_reason == "pending_expired"

    assert strategy.get_stats()["pending_expired"] == 1
    assert strategy.get_stats()["pending_active"] == 0
    assert emitted_topics(event_bus) == [
        "strategy.liquidations.squeeze.pending_created",
        "strategy.liquidations.squeeze.pending_expired",
    ]

    emitted = event_bus.emitted[-1]

    assert emitted["topic"] == "strategy.liquidations.squeeze.pending_expired"
    assert emitted["payload"]["state"] == "pending_expired"
    assert emitted["payload"]["scope_key"] == "binance:usdm_futures:BTCUSDT:realtime"
    assert emitted["headers"]["state"] == "pending_expired"
    assert emitted["headers"]["scope"] == "binance:usdm_futures:BTCUSDT:realtime"


@pytest.mark.asyncio
async def test_newer_stronger_candidate_replaces_existing_pending_only_same_scope(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        confirmation_delay_seconds=60.0,
        replace_pending_if_score_improves=True,
        min_replacement_score_delta=0.03,
        publish_pending_events=True,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    weak = make_result(
        market_type="usdm_futures",
        detected_at=utc_now() - timedelta(seconds=5),
        confidence=0.72,
        exhaustion_bias=0.74,
        continuation_bias=0.20,
        intensity_score=0.62,
        severity=CascadeSeverity.HIGH,
    )
    strong = make_result(
        market_type="usdm_futures",
        detected_at=utc_now() - timedelta(seconds=1),
        confidence=0.96,
        exhaustion_bias=0.96,
        continuation_bias=0.05,
        intensity_score=0.95,
        severity=CascadeSeverity.EXTREME,
        cluster=make_cluster(total_notional_usd=Decimal("2000000")),
    )

    await strategy._on_bus_event(make_event(weak, correlation_id="weak"))
    state = get_state(strategy, weak)
    old_pending = state.pending

    assert old_pending is not None

    await strategy._on_bus_event(make_event(strong, correlation_id="strong"))

    assert state.pending is not None
    assert state.pending is not old_pending
    assert state.pending.result is strong
    assert old_pending.cancelled is True
    assert old_pending.cancel_reason == "replaced_by_newer_stronger_pending"

    assert emitted_topics(event_bus) == [
        "strategy.liquidations.squeeze.pending_created",
        "strategy.liquidations.squeeze.pending_replaced",
        "strategy.liquidations.squeeze.pending_created",
    ]

    replaced_payload = event_bus.emitted[1]["payload"]
    replaced_headers = event_bus.emitted[1]["headers"]

    assert replaced_payload["state"] == "replaced_by_newer_stronger_pending"
    assert replaced_payload["scope_key"] == "binance:usdm_futures:BTCUSDT:realtime"
    assert replaced_headers["scope"] == "binance:usdm_futures:BTCUSDT:realtime"

    assert strategy.get_stats()["pending_created"] == 2
    assert strategy.get_stats()["pending_replaced"] == 1
    assert strategy.get_stats()["pending_cancelled"] == 1
    assert strategy.get_stats()["rejected_events"] == 0


@pytest.mark.asyncio
async def test_newer_not_stronger_candidate_is_rejected_and_existing_pending_survives(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        confirmation_delay_seconds=60.0,
        replace_pending_if_score_improves=True,
        min_replacement_score_delta=0.50,
        publish_rejected_events=False,
        publish_pending_events=False,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    strong_first = make_result(
        detected_at=utc_now() - timedelta(seconds=5),
        confidence=0.96,
        exhaustion_bias=0.96,
        continuation_bias=0.05,
        intensity_score=0.95,
    )
    weaker_newer = make_result(
        detected_at=utc_now() - timedelta(seconds=1),
        confidence=0.75,
        exhaustion_bias=0.75,
        continuation_bias=0.30,
        intensity_score=0.65,
    )

    await strategy._on_bus_event(make_event(strong_first))
    state = get_state(strategy, strong_first)
    old_pending = state.pending

    await strategy._on_bus_event(make_event(weaker_newer))

    assert state.pending is old_pending
    assert strategy.get_stats()["pending_created"] == 1
    assert strategy.get_stats()["pending_replaced"] == 0
    assert strategy.get_stats()["rejected_events"] == 1

    rejection = latest_rejection(strategy)
    assert rejection.reason == "newer_pending_not_stronger_enough"


@pytest.mark.asyncio
async def test_usdm_pending_does_not_replace_or_block_coinm_pending(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        confirmation_delay_seconds=60.0,
        publish_pending_events=True,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    usdm = make_result(
        market_type="usdm_futures",
        exchange_symbol="BTCUSDT",
        detected_at=utc_now() - timedelta(seconds=5),
    )
    coinm = make_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
        detected_at=utc_now() - timedelta(seconds=1),
    )

    await strategy._on_bus_event(make_event(usdm, correlation_id="usdm"))
    await strategy._on_bus_event(make_event(coinm, correlation_id="coinm"))

    usdm_state = get_state(strategy, usdm)
    coinm_state = get_state(strategy, coinm)

    assert usdm_state is not coinm_state
    assert usdm_state.pending is not None
    assert coinm_state.pending is not None
    assert usdm_state.pending.scope_key == "binance:usdm_futures:BTCUSDT:realtime"
    assert coinm_state.pending.scope_key == "binance:coinm_futures:BTCUSDT:realtime"

    assert strategy.get_stats()["pending_created"] == 2
    assert strategy.get_stats()["pending_replaced"] == 0
    assert strategy.get_stats()["pending_active"] == 2
    assert strategy.get_stats()["tracked_scopes"] == 2
    assert strategy.get_stats()["tracked_symbols"] == 1

    assert emitted_topics(event_bus) == [
        "strategy.liquidations.squeeze.pending_created",
        "strategy.liquidations.squeeze.pending_created",
    ]


@pytest.mark.asyncio
async def test_pending_expiry_only_removes_same_scope(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        confirmation_delay_seconds=60.0,
        pending_ttl_seconds=120.0,
        publish_pending_events=False,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    usdm = make_result(
        market_type="usdm_futures",
        exchange_symbol="BTCUSDT",
    )
    coinm = make_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
    )

    await strategy._on_bus_event(make_event(usdm))
    await strategy._on_bus_event(make_event(coinm))

    usdm_state = get_state(strategy, usdm)
    coinm_state = get_state(strategy, coinm)

    assert usdm_state.pending is not None
    assert coinm_state.pending is not None

    usdm_state.pending.expires_at = utc_now() - timedelta(seconds=1)

    await strategy.process_pending_candidates()

    assert usdm_state.pending is None
    assert coinm_state.pending is not None
    assert strategy.get_stats()["pending_expired"] == 1
    assert strategy.get_stats()["pending_active"] == 1


@pytest.mark.asyncio
async def test_same_symbol_different_timeframe_pending_is_independent(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        confirmation_delay_seconds=60.0,
        publish_pending_events=False,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    realtime = make_result(timeframe="realtime")
    one_minute = make_result(timeframe="1m")

    await strategy._on_bus_event(make_event(realtime))
    await strategy._on_bus_event(make_event(one_minute))

    realtime_state = get_state(strategy, realtime)
    one_minute_state = get_state(strategy, one_minute)

    assert realtime_state is not one_minute_state
    assert realtime_state.pending is not None
    assert one_minute_state.pending is not None
    assert realtime_state.pending.scope_key == "binance:usdm_futures:BTCUSDT:realtime"
    assert one_minute_state.pending.scope_key == "binance:usdm_futures:BTCUSDT:1m"
    assert strategy.get_stats()["pending_active"] == 2


# ============================================================================
# Exhaustion-specific filters
# ============================================================================


@pytest.mark.parametrize(
    ("config_overrides", "result_overrides", "expected_reason"),
    [
        (
            {},
            {"status": LiquidationStatus.CANDIDATE},
            "result_not_confirmed",
        ),
        (
            {},
            {"severity": CascadeSeverity.LOW},
            "severity_not_actionable",
        ),
        (
            {},
            {
                "continuation_bias": 0.80,
                "exhaustion_bias": 0.20,
            },
            "exhaustion_not_favored",
        ),
        (
            {"min_exhaustion_bias": 0.90},
            {"exhaustion_bias": 0.80},
            "exhaustion_bias_below_threshold",
        ),
        (
            {"min_bias_delta": 0.40},
            {
                "continuation_bias": 0.45,
                "exhaustion_bias": 0.70,
            },
            "bias_delta_below_threshold",
        ),
        (
            {"max_continuation_bias_after_exhaustion": 0.25},
            {
                "continuation_bias": 0.32,
                "exhaustion_bias": 0.80,
            },
            "continuation_bias_too_high_for_reversal",
        ),
        (
            {"max_future_detected_at_seconds": 1.0},
            {"detected_at": utc_now() + timedelta(seconds=20)},
            "detected_at_in_future",
        ),
        (
            {"max_result_age_seconds": 5.0},
            {"detected_at": utc_now() - timedelta(seconds=20)},
            "result_too_old",
        ),
        (
            {"max_cluster_duration_seconds": 2.0},
            {
                "cluster": make_cluster(
                    start_time=utc_now() - timedelta(seconds=10),
                    end_time=utc_now(),
                )
            },
            "cluster_duration_too_long",
        ),
        (
            {"min_avg_notional_per_event": Decimal("100000")},
            {
                "cluster": make_cluster(
                    event_count=10,
                    total_notional_usd=Decimal("300000"),
                )
            },
            "avg_notional_per_event_below_threshold",
        ),
        (
            {"min_side_imbalance_ratio": 0.90},
            {"metadata": make_metadata(side_imbalance_ratio=0.82)},
            "side_imbalance_below_threshold",
        ),
        (
            {"min_event_imbalance_ratio": 0.90},
            {"metadata": make_metadata(event_imbalance_ratio=0.76)},
            "event_imbalance_below_threshold",
        ),
        (
            {"min_climax_acceleration_ratio": 1.50},
            {"metadata": make_metadata(acceleration_ratio=1.35)},
            "acceleration_below_climax_threshold",
        ),
    ],
)
@pytest.mark.asyncio
async def test_exhaustion_specific_rejections(
    event_bus: FakeEventBus,
    config_overrides: dict[str, Any],
    result_overrides: dict[str, Any],
    expected_reason: str,
) -> None:
    config = make_fast_pending_config(
        enable_pending_confirmation=False,
        publish_rejected_events=False,
        **config_overrides,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(**result_overrides)

    await strategy._on_bus_event(make_event(result))

    stats = strategy.get_stats()

    assert stats["emitted_signals"] == 0
    assert stats["rejected_events"] == 1
    assert stats["filter_skips"] == 1
    assert event_bus.emitted == []

    rejection = latest_rejection(strategy)

    assert rejection.reason == expected_reason
    assert rejection.strategy_name == "squeeze_reversal_strategy"
    assert rejection.signal_type == "reversal"
    assert rejection.scope_key == scoped_key_to_string(result.key)


@pytest.mark.asyncio
async def test_common_filters_run_before_exhaustion_specific_filters(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        enable_pending_confirmation=False,
        min_confidence=0.95,
        min_exhaustion_bias=0.95,
        min_bias_delta=0.90,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(
        confidence=0.50,
        continuation_bias=0.80,
        exhaustion_bias=0.10,
    )

    await strategy._on_bus_event(make_event(result))

    rejection = latest_rejection(strategy)

    assert rejection.reason == "confidence_below_threshold"
    assert rejection.reason != "exhaustion_not_favored"
    assert rejection.reason != "exhaustion_bias_below_threshold"
    assert rejection.reason != "bias_delta_below_threshold"


@pytest.mark.asyncio
async def test_allowed_market_type_filter_rejects_other_futures_scope(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        enable_pending_confirmation=False,
        allowed_market_types=("usdm_futures",),
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
    )

    await strategy._on_bus_event(make_event(result))

    rejection = latest_rejection(strategy)

    assert rejection.reason == "market_type_not_allowed"
    assert rejection.market_type == "coinm_futures"
    assert rejection.scope_key == "binance:coinm_futures:BTCUSDT:realtime"


@pytest.mark.asyncio
async def test_allowed_timeframe_filter_rejects_other_timeframe(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        enable_pending_confirmation=False,
        allowed_timeframes=("realtime",),
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(timeframe="1m")

    await strategy._on_bus_event(make_event(result))

    rejection = latest_rejection(strategy)

    assert rejection.reason == "timeframe_not_allowed"
    assert rejection.timeframe == "1m"
    assert rejection.scope_key == "binance:usdm_futures:BTCUSDT:1m"


# ============================================================================
# Rejection publishing / EventBus contract
# ============================================================================


@pytest.mark.asyncio
async def test_rejected_event_is_published_with_full_scope_headers_when_enabled(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        enable_pending_confirmation=False,
        publish_rejected_events=True,
        min_exhaustion_bias=0.95,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(
        exchange="okx",
        market_type="swap",
        symbol="SOL-USDT",
        timeframe="1m",
        exchange_symbol="SOL-USDT-SWAP",
        exhaustion_bias=0.82,
    )
    event = make_event(result, correlation_id="reject-corr")

    await strategy._on_bus_event(event)

    assert emitted_topics(event_bus) == ["signal.rejected"]

    emitted = event_bus.emitted[0]
    rejection = emitted["payload"]

    assert isinstance(rejection, StrategyRejection)
    assert rejection.reason == "exhaustion_bias_below_threshold"
    assert rejection.exchange == "okx"
    assert rejection.market_type == "swap"
    assert rejection.symbol == "SOLUSDT"
    assert rejection.timeframe == "1m"
    assert rejection.exchange_symbol == "SOL-USDT-SWAP"
    assert rejection.scope_key == "okx:swap:SOLUSDT:1m"
    assert rejection.correlation_id == "reject-corr"
    assert rejection.source_event_id == event.event_id

    assert emitted["priority"] is EventPriority.LOW
    assert emitted["source"] == "squeeze_reversal_strategy"
    assert emitted["correlation_id"] == "reject-corr"

    headers = emitted["headers"]

    assert headers["strategy"] == "squeeze_reversal_strategy"
    assert headers["signal_type"] == "reversal"
    assert headers["reason"] == "exhaustion_bias_below_threshold"
    assert headers["source_event_id"] == event.event_id
    assert headers["source_topic"] == "analytics.liquidations.exhaustion_detected"

    assert_headers_have_full_scope(
        headers,
        exchange="okx",
        market_type="swap",
        symbol="SOLUSDT",
        timeframe="1m",
        exchange_symbol="SOL-USDT-SWAP",
    )


# ============================================================================
# Emit failure / EventBus robustness
# ============================================================================


@pytest.mark.asyncio
async def test_direct_emit_false_does_not_mark_signal_as_emitted() -> None:
    event_bus = RejectingEventBus()
    config = make_fast_pending_config(enable_pending_confirmation=False)
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result()

    await strategy._on_bus_event(make_event(result))

    state = get_state(strategy, result)
    stats = strategy.get_stats()

    assert stats["processed_events"] == 1
    assert stats["emitted_signals"] == 0
    assert stats["rejected_events"] == 0
    assert state.total_signals_emitted == 0
    assert state.last_signal_at is None
    assert state.last_detected_at is None


@pytest.mark.asyncio
async def test_direct_emit_exception_does_not_mark_signal_as_emitted() -> None:
    event_bus = FailingEventBus()
    config = make_fast_pending_config(enable_pending_confirmation=False)
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result()

    await strategy._on_bus_event(make_event(result))

    state = get_state(strategy, result)
    stats = strategy.get_stats()

    assert stats["processed_events"] == 1
    assert stats["emitted_signals"] == 0
    assert stats["rejected_events"] == 0
    assert stats["last_error"] is not None
    assert "event bus emit failed" in stats["last_error"]

    assert state.total_signals_emitted == 0
    assert state.last_signal_at is None
    assert state.last_detected_at is None


@pytest.mark.asyncio
async def test_pending_confirmation_emit_failure_does_not_clear_pending_or_increment_confirmed() -> None:
    event_bus = FailingEventBus()
    config = make_fast_pending_config(
        publish_pending_events=False,
        enable_pending_confirmation=True,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result()

    await strategy._on_bus_event(make_event(result))

    state = get_state(strategy, result)
    assert state.pending is not None

    await strategy.process_pending_candidates()

    assert state.pending is not None
    assert strategy.get_stats()["pending_confirmed"] == 0
    assert strategy.get_stats()["emitted_signals"] == 0
    assert strategy.get_stats()["last_error"] is not None


# ============================================================================
# Full-scope cooldown / dedup / rate-limit
# ============================================================================


@pytest.mark.asyncio
async def test_same_scope_respects_cooldown_but_other_market_type_is_independent(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        enable_pending_confirmation=False,
        symbol_cooldown_seconds=60,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
        deduplicate_by_detected_at=False,
        deduplicate_same_cluster_signature=False,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    first = make_result(
        market_type="usdm_futures",
        exchange_symbol="BTCUSDT",
        detected_at=utc_now() - timedelta(seconds=3),
    )
    second_same_scope = make_result(
        market_type="usdm_futures",
        exchange_symbol="BTCUSDT",
        detected_at=utc_now() - timedelta(seconds=2),
    )
    third_other_market_type = make_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
        detected_at=utc_now() - timedelta(seconds=1),
    )

    await strategy._on_bus_event(make_event(first))
    await strategy._on_bus_event(make_event(second_same_scope))
    await strategy._on_bus_event(make_event(third_other_market_type))

    assert strategy.get_stats()["emitted_signals"] == 2
    assert strategy.get_stats()["rejected_events"] == 1
    assert strategy.get_stats()["cooldown_skips"] == 1
    assert strategy.get_stats()["tracked_scopes"] == 2
    assert strategy.get_stats()["tracked_symbols"] == 1

    rejection = latest_rejection(strategy)
    assert rejection.reason == "scope_in_cooldown"
    assert rejection.scope_key == "binance:usdm_futures:BTCUSDT:realtime"


@pytest.mark.asyncio
async def test_same_detected_at_is_not_duplicate_across_market_types(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        enable_pending_confirmation=False,
        deduplicate_by_detected_at=True,
        deduplicate_same_cluster_signature=False,
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    same_detected_at = utc_now() - timedelta(seconds=1)

    usdm = make_result(
        market_type="usdm_futures",
        exchange_symbol="BTCUSDT",
        detected_at=same_detected_at,
    )
    coinm = make_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
        detected_at=same_detected_at,
    )
    duplicate_usdm = make_result(
        market_type="usdm_futures",
        exchange_symbol="BTCUSDT",
        detected_at=same_detected_at,
    )

    await strategy._on_bus_event(make_event(usdm))
    await strategy._on_bus_event(make_event(coinm))
    await strategy._on_bus_event(make_event(duplicate_usdm))

    assert strategy.get_stats()["emitted_signals"] == 2
    assert strategy.get_stats()["rejected_events"] == 1
    assert strategy.get_stats()["duplicate_skips"] == 1

    rejection = latest_rejection(strategy)
    assert rejection.reason == "duplicate_detected_at"
    assert rejection.market_type == "usdm_futures"


@pytest.mark.asyncio
async def test_same_side_too_soon_is_scoped_by_timeframe(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        enable_pending_confirmation=False,
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=60,
        max_signals_per_symbol_window=0,
        deduplicate_by_detected_at=False,
        deduplicate_same_cluster_signature=False,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    realtime_first = make_result(
        timeframe="realtime",
        detected_at=utc_now() - timedelta(seconds=3),
    )
    realtime_second = make_result(
        timeframe="realtime",
        detected_at=utc_now() - timedelta(seconds=2),
    )
    one_minute = make_result(
        timeframe="1m",
        detected_at=utc_now() - timedelta(seconds=1),
    )

    await strategy._on_bus_event(make_event(realtime_first))
    await strategy._on_bus_event(make_event(realtime_second))
    await strategy._on_bus_event(make_event(one_minute))

    assert strategy.get_stats()["emitted_signals"] == 2
    assert strategy.get_stats()["rejected_events"] == 1
    assert strategy.get_stats()["duplicate_skips"] == 1

    rejection = latest_rejection(strategy)
    assert rejection.reason == "same_side_signal_too_soon"
    assert rejection.timeframe == "realtime"


@pytest.mark.asyncio
async def test_scope_signal_rate_limit_does_not_block_other_market_type(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        enable_pending_confirmation=False,
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=1,
        signal_window_seconds=60,
        deduplicate_by_detected_at=False,
        deduplicate_same_cluster_signature=False,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    first = make_result(
        market_type="usdm_futures",
        detected_at=utc_now() - timedelta(seconds=3),
    )
    second_same_scope = make_result(
        market_type="usdm_futures",
        detected_at=utc_now() - timedelta(seconds=2),
    )
    third_other_scope = make_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
        detected_at=utc_now() - timedelta(seconds=1),
    )

    await strategy._on_bus_event(make_event(first))
    await strategy._on_bus_event(make_event(second_same_scope))
    await strategy._on_bus_event(make_event(third_other_scope))

    assert strategy.get_stats()["emitted_signals"] == 2
    assert strategy.get_stats()["rejected_events"] == 1
    assert strategy.get_stats()["rate_limit_skips"] == 1

    rejection = latest_rejection(strategy)
    assert rejection.reason == "scope_signal_rate_limited"


# ============================================================================
# Query / diagnostics API
# ============================================================================


@pytest.mark.asyncio
async def test_get_recent_signals_filters_by_full_scope(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        enable_pending_confirmation=False,
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
        deduplicate_by_detected_at=False,
        deduplicate_same_cluster_signature=False,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    usdm = make_result(
        market_type="usdm_futures",
        exchange_symbol="BTCUSDT",
        detected_at=utc_now() - timedelta(seconds=3),
    )
    coinm = make_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
        detected_at=utc_now() - timedelta(seconds=2),
    )
    one_minute = make_result(
        market_type="usdm_futures",
        timeframe="1m",
        exchange_symbol="BTCUSDT",
        detected_at=utc_now() - timedelta(seconds=1),
    )

    await strategy._on_bus_event(make_event(usdm))
    await strategy._on_bus_event(make_event(coinm))
    await strategy._on_bus_event(make_event(one_minute))

    all_signals = strategy.get_recent_signals(
        exchange="binance",
        symbol="BTCUSDT",
        limit=10,
    )
    usdm_realtime = strategy.get_recent_signals(
        exchange="binance",
        market_type="usdm_futures",
        symbol="BTCUSDT",
        timeframe="realtime",
        limit=10,
    )
    coinm_realtime = strategy.get_recent_signals(
        exchange="binance",
        market_type="coinm_futures",
        symbol="BTCUSDT",
        timeframe="realtime",
        limit=10,
    )

    assert len(all_signals) == 3
    assert len(usdm_realtime) == 1
    assert len(coinm_realtime) == 1
    assert usdm_realtime[0].scope_key == "binance:usdm_futures:BTCUSDT:realtime"
    assert coinm_realtime[0].scope_key == "binance:coinm_futures:BTCUSDT:realtime"


@pytest.mark.asyncio
async def test_get_recent_pending_filters_by_market_type_and_timeframe(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        enable_pending_confirmation=True,
        confirmation_delay_seconds=60.0,
        publish_pending_events=False,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    usdm = make_result(
        market_type="usdm_futures",
        exchange_symbol="BTCUSDT",
    )
    coinm = make_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
    )
    one_minute = make_result(
        market_type="usdm_futures",
        timeframe="1m",
    )

    await strategy._on_bus_event(make_event(usdm))
    await strategy._on_bus_event(make_event(coinm))
    await strategy._on_bus_event(make_event(one_minute))

    all_pending = strategy.get_recent_pending(limit=10)
    usdm_realtime = strategy.get_recent_pending(
        exchange="binance",
        market_type="usdm_futures",
        symbol="BTCUSDT",
        timeframe="realtime",
        limit=10,
    )
    coinm_realtime = strategy.get_recent_pending(
        exchange="binance",
        market_type="coinm_futures",
        symbol="BTCUSDT",
        timeframe="realtime",
        limit=10,
    )

    assert len(all_pending) == 3
    assert len(usdm_realtime) == 1
    assert len(coinm_realtime) == 1

    assert usdm_realtime[0]["candidate"]["scope_key"] == "binance:usdm_futures:BTCUSDT:realtime"
    assert coinm_realtime[0]["candidate"]["scope_key"] == "binance:coinm_futures:BTCUSDT:realtime"


@pytest.mark.asyncio
async def test_get_hot_symbols_returns_latest_signal_per_scope_sorted_by_score(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        enable_pending_confirmation=False,
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
        deduplicate_by_detected_at=False,
        deduplicate_same_cluster_signature=False,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    weak_usdm = make_result(
        market_type="usdm_futures",
        confidence=0.72,
        exhaustion_bias=0.72,
        continuation_bias=0.18,
        intensity_score=0.62,
        detected_at=utc_now() - timedelta(seconds=5),
    )
    strong_usdm = make_result(
        market_type="usdm_futures",
        confidence=0.96,
        exhaustion_bias=0.96,
        continuation_bias=0.04,
        intensity_score=0.95,
        detected_at=utc_now() - timedelta(seconds=4),
    )
    coinm = make_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
        confidence=0.90,
        exhaustion_bias=0.88,
        continuation_bias=0.12,
        detected_at=utc_now() - timedelta(seconds=3),
    )

    await strategy._on_bus_event(make_event(weak_usdm))
    await strategy._on_bus_event(make_event(strong_usdm))
    await strategy._on_bus_event(make_event(coinm))

    hot = strategy.get_hot_symbols(limit=10)

    assert len(hot) == 2
    assert hot[0]["scope_key"] == "binance:usdm_futures:BTCUSDT:realtime"
    assert hot[1]["scope_key"] == "binance:coinm_futures:BTCUSDT:realtime"
    assert hot[0]["score"] >= hot[1]["score"]


@pytest.mark.asyncio
async def test_get_symbol_state_returns_full_scope_pending_diagnostic(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        enable_pending_confirmation=True,
        confirmation_delay_seconds=60.0,
        publish_pending_events=False,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
    )

    await strategy._on_bus_event(make_event(result))

    diagnostic = strategy.get_symbol_state(
        "binance",
        "BTCUSDT",
        market_type="coinm_futures",
        timeframe="realtime",
    )

    assert diagnostic["exists"] is True
    assert diagnostic["exchange"] == "binance"
    assert diagnostic["market_type"] == "coinm_futures"
    assert diagnostic["symbol"] == "BTCUSDT"
    assert diagnostic["timeframe"] == "realtime"
    assert diagnostic["exchange_symbol"] == "BTCUSD_PERP"
    assert diagnostic["scope_key"] == "binance:coinm_futures:BTCUSDT:realtime"
    assert diagnostic["pending"] is not None
    assert diagnostic["pending"]["scope_key"] == "binance:coinm_futures:BTCUSDT:realtime"


# ============================================================================
# Scoring / metadata / model serialization
# ============================================================================


def test_extract_analytics_metadata_parses_numeric_strings_and_bad_values(
    event_bus: FakeEventBus,
) -> None:
    strategy = make_strategy(event_bus=event_bus)

    result = make_result(
        metadata=make_metadata(
            side_imbalance_ratio="0.88",
            event_imbalance_ratio=None,
            acceleration_ratio="bad-float",
        )
    )

    extracted = strategy.extract_analytics_metadata(result)

    assert extracted["side_imbalance_ratio"] == 0.88
    assert extracted["event_imbalance_ratio"] is None
    assert extracted["acceleration_ratio"] is None


def test_compute_strategy_score_uses_exhaustion_bias_delta_cluster_imbalance_and_acceleration(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        score_confidence_weight=0.20,
        score_exhaustion_bias_weight=0.25,
        score_bias_delta_weight=0.20,
        score_intensity_weight=0.15,
        score_severity_weight=0.05,
        score_cluster_quality_weight=0.05,
        score_imbalance_weight=0.05,
        score_acceleration_weight=0.05,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)

    result = make_result(
        confidence=0.90,
        exhaustion_bias=0.85,
        continuation_bias=0.15,
        intensity_score=0.80,
        severity=CascadeSeverity.EXTREME,
        metadata=make_metadata(
            side_imbalance_ratio=0.90,
            acceleration_ratio=1.50,
        ),
    )

    score = strategy.compute_strategy_score(result)

    assert 0.0 <= score <= 1.0
    assert score > 0.70


def test_build_quality_snapshot_contains_full_scope_and_filter_critical_fields(
    event_bus: FakeEventBus,
) -> None:
    strategy = make_strategy(event_bus=event_bus)

    result = make_result(
        exchange="okx",
        market_type="swap",
        symbol="BTC-USDT",
        exchange_symbol="BTC-USDT-SWAP",
        confidence=0.91,
        exhaustion_bias=0.89,
        continuation_bias=0.11,
        metadata=make_metadata(side_imbalance_ratio=0.93),
    )

    snapshot = strategy.build_quality_snapshot(result)

    assert snapshot["exchange"] == "okx"
    assert snapshot["market_type"] == "swap"
    assert snapshot["symbol"] == "BTCUSDT"
    assert snapshot["timeframe"] == "realtime"
    assert snapshot["exchange_symbol"] == "BTC-USDT-SWAP"
    assert snapshot["scope"] == {
        "exchange": "okx",
        "market_type": "swap",
        "symbol": "BTCUSDT",
        "timeframe": "realtime",
    }
    assert snapshot["scope_key"] == "okx:swap:BTCUSDT:realtime"
    assert snapshot["confidence"] == 0.91
    assert snapshot["exhaustion_bias"] == 0.89
    assert snapshot["bias_delta"] == result.bias_delta
    assert snapshot["event_type"] == "liquidation_exhaustion"
    assert snapshot["cluster"]["scope_key"] == "okx:swap:BTCUSDT:realtime"
    assert snapshot["cluster"]["avg_notional_per_event"] == str(result.cluster.avg_notional_per_event)
    assert snapshot["analytics_metadata"]["side_imbalance_ratio"] == 0.93


def test_pending_candidate_to_dict_contains_full_scope_and_serialized_values(
    event_bus: FakeEventBus,
) -> None:
    strategy = make_strategy(event_bus=event_bus)
    result = make_result(
        exchange="okx",
        market_type="swap",
        symbol="BTC-USDT",
        timeframe="1m",
        exchange_symbol="BTC-USDT-SWAP",
    )
    now = utc_now()

    candidate = PendingReversalCandidate(
        exchange=result.exchange,
        market_type=result.market_type,
        symbol=result.symbol,
        timeframe=result.timeframe,
        exchange_symbol=result.exchange_symbol,
        result=result,
        source_topic="analytics.liquidations.exhaustion_detected",
        source_event_id="event-1",
        correlation_id="corr-1",
        created_at=now,
        confirm_after=now + timedelta(seconds=1),
        expires_at=now + timedelta(seconds=10),
        cluster_signature=strategy.build_cluster_signature(result),
        score_at_creation=0.88,
        quality_snapshot=strategy.build_quality_snapshot(result),
    )

    data = candidate.to_dict()

    assert data["exchange"] == "okx"
    assert data["market_type"] == "swap"
    assert data["symbol"] == "BTCUSDT"
    assert data["timeframe"] == "1m"
    assert data["exchange_symbol"] == "BTC-USDT-SWAP"
    assert data["scope_key"] == "okx:swap:BTCUSDT:1m"
    assert isinstance(data["created_at"], str)
    assert isinstance(data["confirm_after"], str)
    assert isinstance(data["expires_at"], str)
    assert data["score_at_creation"] == 0.88


def test_squeeze_reversal_signal_to_dict_contains_full_scope_and_confirmation_metadata() -> None:
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    signal = SqueezeReversalSignal(
        strategy_name="squeeze_reversal_strategy",
        signal_type="reversal",
        exchange="BINANCE",
        market_type="USDM_FUTURES",
        symbol="btc-usdt",
        timeframe="Realtime",
        exchange_symbol="BTCUSDT",
        side="long",
        confidence=1.2,
        score=1.1,
        generated_at=now,
        detected_at=now - timedelta(seconds=1),
        reason="test",
        source_topic="analytics.liquidations.exhaustion_detected",
        severity="extreme",
        cascade_direction="down",
        liquidation_side="long",
        event_count=10,
        total_notional_usd=Decimal("900000"),
        intensity_score=1.2,
        continuation_bias=-0.1,
        exhaustion_bias=1.3,
        bias_delta=1.5,
        price_range_pct=0.38,
        window_seconds=10,
        cluster_duration_seconds=4.0,
        cluster_avg_notional_per_event=Decimal("90000"),
        pending_started_at=now - timedelta(seconds=2),
        pending_confirmed_at=now,
    )

    data = signal.to_dict()

    assert signal.exchange == "binance"
    assert signal.market_type == "usdm_futures"
    assert signal.symbol == "BTCUSDT"
    assert signal.timeframe == "realtime"
    assert signal.scope_key == "binance:usdm_futures:BTCUSDT:realtime"
    assert signal.confidence == 1.0
    assert signal.score == 1.0
    assert signal.intensity_score == 1.0
    assert signal.continuation_bias == 0.0
    assert signal.exhaustion_bias == 1.0
    assert signal.bias_delta == 1.0
    assert signal.is_pending_confirmed is True
    assert signal.confirmation_delay_seconds == 2.0

    assert data["scope_key"] == "binance:usdm_futures:BTCUSDT:realtime"
    assert data["total_notional_usd"] == "900000"
    assert data["cluster_avg_notional_per_event"] == "90000"
    assert data["is_pending_confirmed"] is True
    assert data["confirmation_delay_seconds"] == 2.0


# ============================================================================
# Diagnostics snapshot
# ============================================================================


@pytest.mark.asyncio
async def test_publish_diagnostics_snapshot_emits_scope_aware_payload(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        enable_pending_confirmation=False,
        publish_diagnostics_snapshots=True,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    await strategy._on_bus_event(make_event(make_result()))
    await strategy.publish_diagnostics_snapshot()

    assert emitted_topics(event_bus) == ["signal.generated", "strategy.liquidations.squeeze.snapshot"]

    emitted = event_bus.emitted[-1]

    assert emitted["topic"] == "strategy.liquidations.squeeze.snapshot"
    assert emitted["priority"] is EventPriority.LOW
    assert emitted["source"] == "squeeze_reversal_strategy"
    assert emitted["headers"]["event_type"] == "strategy_diagnostics"

    payload = emitted["payload"]

    assert payload["strategy_name"] == "squeeze_reversal_strategy"
    assert payload["signal_type"] == "reversal"
    assert payload["scope"] == "exchange:market_type:symbol:timeframe"
    assert payload["stats"]["tracked_scopes"] == 1
    assert payload["stats"]["pending_active"] == 0
    assert payload["hot_symbols"][0]["scope_key"] == "binance:usdm_futures:BTCUSDT:realtime"


# ============================================================================
# Config validation
# ============================================================================


@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        ({"subscribe_topic": ""}, "subscribe_topic must not be empty"),
        ({"publish_topic_signal_generated": ""}, "publish_topic_signal_generated must not be empty"),
        ({"publish_topic_signal_rejected": ""}, "publish_topic_signal_rejected must not be empty"),
        ({"diagnostics_topic": ""}, "diagnostics_topic must not be empty"),
        ({"publish_topic_pending_created": ""}, "publish_topic_pending_created must not be empty"),
        ({"publish_topic_pending_expired": ""}, "publish_topic_pending_expired must not be empty"),
        ({"publish_topic_pending_replaced": ""}, "publish_topic_pending_replaced must not be empty"),
        ({"publish_topic_pending_confirmed": ""}, "publish_topic_pending_confirmed must not be empty"),
        ({"min_confidence": -0.01}, "min_confidence must be between 0 and 1"),
        ({"min_intensity_score": 1.01}, "min_intensity_score must be between 0 and 1"),
        ({"min_exhaustion_bias": -0.01}, "min_exhaustion_bias must be between 0 and 1"),
        ({"min_bias_delta": 1.01}, "min_bias_delta must be between 0 and 1"),
        ({"max_continuation_bias_after_exhaustion": 1.01}, "max_continuation_bias_after_exhaustion must be between 0 and 1 or None"),
        ({"min_total_notional_usd": Decimal("-1")}, "min_total_notional_usd must be >= 0"),
        ({"min_event_count": -1}, "min_event_count must be >= 0"),
        ({"max_price_range_pct": -0.1}, "max_price_range_pct must be >= 0 or None"),
        ({"min_side_imbalance_ratio": 1.01}, "min_side_imbalance_ratio must be between 0 and 1 or None"),
        ({"min_event_imbalance_ratio": -0.01}, "min_event_imbalance_ratio must be between 0 and 1 or None"),
        ({"min_climax_acceleration_ratio": -0.1}, "min_climax_acceleration_ratio must be >= 0 or None"),
        ({"max_cluster_duration_seconds": 0.0}, "max_cluster_duration_seconds must be > 0 or None"),
        ({"min_avg_notional_per_event": Decimal("-1")}, "min_avg_notional_per_event must be >= 0 or None"),
        ({"max_result_age_seconds": 0.0}, "max_result_age_seconds must be > 0"),
        ({"max_future_detected_at_seconds": -1.0}, "max_future_detected_at_seconds must be >= 0"),
        ({"confirmation_delay_seconds": -1.0}, "confirmation_delay_seconds must be >= 0"),
        ({"pending_ttl_seconds": 0.0}, "pending_ttl_seconds must be > 0"),
        ({"min_pending_age_seconds": -1.0}, "min_pending_age_seconds must be >= 0"),
        ({"pending_scan_interval_seconds": 0.0}, "pending_scan_interval_seconds must be > 0"),
        (
            {
                "confirmation_delay_seconds": 10.0,
                "pending_ttl_seconds": 10.0,
            },
            "pending_ttl_seconds must be greater than confirmation_delay_seconds",
        ),
        (
            {
                "min_pending_age_seconds": 31.0,
                "pending_ttl_seconds": 30.0,
            },
            "min_pending_age_seconds must be <= pending_ttl_seconds",
        ),
        ({"symbol_cooldown_seconds": -1}, "symbol_cooldown_seconds must be >= 0"),
        ({"signal_window_seconds": 0}, "signal_window_seconds must be > 0"),
        ({"recent_pending_limit": 0}, "recent_pending_limit must be > 0"),
        (
            {
                "allowed_market_types": ("usdm_futures",),
                "blocked_market_types": ("USDM_FUTURES",),
            },
            "allowed_market_types and blocked_market_types overlap",
        ),
        (
            {
                "allowed_timeframes": ("realtime",),
                "blocked_timeframes": ("Realtime",),
            },
            "allowed_timeframes and blocked_timeframes overlap",
        ),
        (
            {
                "allowed_symbols": ("BTCUSDT",),
                "blocked_symbols": ("btc-usdt",),
            },
            "allowed_symbols and blocked_symbols overlap",
        ),
        (
            {
                "score_confidence_weight": 0.0,
                "score_exhaustion_bias_weight": 0.0,
                "score_bias_delta_weight": 0.0,
                "score_intensity_weight": 0.0,
                "score_severity_weight": 0.0,
                "score_cluster_quality_weight": 0.0,
                "score_imbalance_weight": 0.0,
                "score_acceleration_weight": 0.0,
            },
            "strategy score weights sum must be > 0",
        ),
    ],
)
def test_config_validation_rejects_invalid_values(
    overrides: dict[str, Any],
    expected_message: str,
) -> None:
    config = SqueezeReversalStrategyConfig(**overrides)

    with pytest.raises(ValueError, match=expected_message):
        config.validate()