from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest

from core.event_bus import Event, EventPriority

from analytics.liquidations.enums import CascadeDirection, CascadeSeverity

from strategy.strategies.liquidations.base import StrategyRejection
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
class FakeValue:
    value: str


@dataclass(slots=True)
class FakeDirection:
    value: str


@dataclass(slots=True)
class FakeCluster:
    start_time: datetime
    end_time: datetime
    event_count: int
    total_notional_usd: Decimal
    avg_notional_per_event: Decimal
    avg_price: Decimal = Decimal("100000")
    min_price: Decimal = Decimal("99500")
    max_price: Decimal = Decimal("100500")
    duration_seconds: float = 4.0
    price_range_pct: float = 0.35


@dataclass(slots=True)
class FakeCascadeDetectionResult:
    """
    Поведінковий test double для CascadeDetectionResult.

    Тут ми тестуємо саме SqueezeReversalStrategy, а не analytics model.
    Тому strategy.payload_type у helper-и нижче підміняється на цей клас.
    """

    exchange: str
    symbol: str
    detected_at: datetime

    confidence: float
    intensity_score: float
    continuation_bias: float
    exhaustion_bias: float
    bias_delta: float

    event_count: int
    total_notional_usd: Decimal
    price_range_pct: float
    window_seconds: int

    severity: Any
    direction: Any
    side: Any
    event_type: Any

    is_confirmed: bool
    is_actionable_severity: bool
    favors_exhaustion: bool
    favors_continuation: bool

    correlation_id: str | None
    metadata: dict[str, Any]
    cluster: FakeCluster

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.75


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
        subscription = FakeSubscription(topic=topic, handler=handler, name=name)
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


def make_cluster(**overrides: Any) -> FakeCluster:
    now = utc_now()

    base = {
        "start_time": now - timedelta(seconds=5),
        "end_time": now,
        "event_count": 12,
        "total_notional_usd": Decimal("900000"),
        "avg_notional_per_event": Decimal("75000"),
        "avg_price": Decimal("100000"),
        "min_price": Decimal("99500"),
        "max_price": Decimal("100500"),
        "duration_seconds": 4.0,
        "price_range_pct": 0.35,
    }
    base.update(overrides)
    return FakeCluster(**base)


def make_metadata(**overrides: Any) -> dict[str, Any]:
    base = {
        "detector": "liquidation_exhaustion_detector",
        "side_imbalance_ratio": 0.82,
        "event_imbalance_ratio": 0.76,
        "acceleration_ratio": 1.35,
    }
    base.update(overrides)
    return base


def make_result(**overrides: Any) -> FakeCascadeDetectionResult:
    now = utc_now()

    base = {
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "detected_at": now - timedelta(seconds=1),
        "confidence": 0.86,
        "intensity_score": 0.78,
        "continuation_bias": 0.32,
        "exhaustion_bias": 0.82,
        "bias_delta": 0.28,
        "event_count": 12,
        "total_notional_usd": Decimal("900000"),
        "price_range_pct": 0.35,
        "window_seconds": 10,
        "severity": CascadeSeverity.HIGH,
        "direction": CascadeDirection.DOWN,
        "side": FakeValue("long_liquidations"),
        "event_type": FakeValue("liquidation_exhaustion"),
        "is_confirmed": True,
        "is_actionable_severity": True,
        "favors_exhaustion": True,
        "favors_continuation": False,
        "correlation_id": "analytics-corr-1",
        "metadata": make_metadata(),
        "cluster": make_cluster(),
    }
    base.update(overrides)
    return FakeCascadeDetectionResult(**base)


def make_event(
    payload: Any,
    *,
    topic: str = "analytics.liquidation.exhaustion_detected",
    correlation_id: str | None = "bus-corr-1",
) -> Event:
    return Event(
        topic=topic,
        payload=payload,
        priority=EventPriority.NORMAL,
        source="analytics.liquidations.exhaustion_detector",
        correlation_id=correlation_id,
    )


def make_strategy(
    *,
    event_bus: FakeEventBus,
    config: SqueezeReversalStrategyConfig | None = None,
    scheduler: FakeScheduler | None = None,
) -> SqueezeReversalStrategy:
    strategy = SqueezeReversalStrategy(
        event_bus=event_bus,  # type: ignore[arg-type]
        config=config,
        scheduler=scheduler,  # type: ignore[arg-type]
    )

    strategy.payload_type = FakeCascadeDetectionResult  # type: ignore[assignment]

    return strategy


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
        "min_event_imbalance_ratio": None,
    }
    base.update(overrides)
    return SqueezeReversalStrategyConfig(**base)


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
    result: FakeCascadeDetectionResult,
) -> SymbolSqueezeStrategyState:
    state = strategy.get_or_create_state(result.exchange, result.symbol)
    assert isinstance(state, SymbolSqueezeStrategyState)
    return state


def assert_no_risk_or_execution_events(event_bus: FakeEventBus) -> None:
    forbidden_prefixes = ("risk.", "execution.", "order.", "position.")
    for topic in emitted_topics(event_bus):
        assert not topic.startswith(forbidden_prefixes), (
            f"strategy must not emit direct risk/execution/order/position event: {topic}"
        )


# ============================================================================
# Lifecycle / Scheduler contract
# ============================================================================


@pytest.mark.asyncio
async def test_start_subscribes_to_exhaustion_topic(
    event_bus: FakeEventBus,
) -> None:
    strategy = make_strategy(event_bus=event_bus)

    await strategy.start()

    assert strategy.get_stats()["running"] is True
    assert len(event_bus.subscriptions) == 1
    assert event_bus.subscriptions[0].topic == "analytics.liquidation.exhaustion_detected"
    assert event_bus.subscriptions[0].name == (
        "squeeze_reversal_strategy.on_analytics_event"
    )


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

    pending_job = next(
        job
        for job in scheduler.jobs.values()
        if job["name"] == "squeeze_reversal_strategy:pending_scan"
    )

    assert pending_job["func"] == strategy.process_pending_candidates
    assert pending_job["interval"] == 0.25
    assert pending_job["allow_overlap"] is False


@pytest.mark.asyncio
async def test_start_registers_both_diagnostics_and_pending_scan_jobs(
    event_bus: FakeEventBus,
) -> None:
    scheduler = FakeScheduler()
    config = make_fast_pending_config(
        publish_diagnostics_snapshots=True,
        diagnostics_interval_seconds=9.0,
        pending_scan_interval_seconds=0.2,
    )
    strategy = make_strategy(
        event_bus=event_bus,
        config=config,
        scheduler=scheduler,
    )

    await strategy.start()

    job_names = {job["name"] for job in scheduler.jobs.values()}

    assert job_names == {
        "squeeze_reversal_strategy:diagnostics",
        "squeeze_reversal_strategy:pending_scan",
    }
    assert strategy.get_stats()["diagnostics_job_registered"] is True
    assert strategy.get_stats()["pending_scan_job_registered"] is True


@pytest.mark.asyncio
async def test_stop_removes_pending_scan_and_diagnostics_jobs(
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

    job_ids = set(scheduler.jobs.keys())
    assert len(job_ids) == 2

    await strategy.stop()

    assert job_ids.issubset(set(scheduler.removed_job_ids))
    assert strategy.get_stats()["running"] is False
    assert strategy.get_stats()["pending_scan_job_registered"] is False
    assert strategy.get_stats()["diagnostics_job_registered"] is False


@pytest.mark.xfail(
    reason=(
        "Known vulnerability: SqueezeReversalStrategy allows start() with "
        "enable_pending_confirmation=True and scheduler=None. Pending candidates can be created "
        "but never automatically scanned/confirmed. Fix recommendation: raise ValueError in "
        "start() or __init__ when pending confirmation is enabled without Scheduler."
    ),
    strict=True,
)
@pytest.mark.asyncio
async def test_pending_confirmation_requires_scheduler_when_enabled(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(enable_pending_confirmation=True)

    strategy = make_strategy(
        event_bus=event_bus,
        config=config,
        scheduler=None,
    )

    with pytest.raises(ValueError):
        await strategy.start()


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
        make_result(direction=FakeDirection("sideways"))
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

    result = make_result(direction=CascadeDirection.UP)
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
async def test_direct_signal_contract_contains_exhaustion_and_metadata_fields(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(enable_pending_confirmation=False)
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(
        exchange="bybit",
        symbol="ETH/USDT",
        direction=CascadeDirection.DOWN,
        side=FakeValue("long_liquidations"),
        severity=CascadeSeverity.EXTREME,
        confidence=0.93,
        intensity_score=0.89,
        continuation_bias=0.21,
        exhaustion_bias=0.91,
        bias_delta=0.55,
        event_count=30,
        total_notional_usd=Decimal("2500000"),
        price_range_pct=0.28,
        window_seconds=15,
        cluster=make_cluster(
            duration_seconds=3.0,
            avg_notional_per_event=Decimal("125000"),
            total_notional_usd=Decimal("2500000"),
        ),
        metadata=make_metadata(
            side_imbalance_ratio=0.91,
            event_imbalance_ratio=0.86,
            acceleration_ratio=1.75,
        ),
    )
    event = make_event(result, correlation_id="contract-corr")

    await strategy._on_bus_event(event)

    signal = latest_signal(event_bus)

    assert signal.strategy_name == "squeeze_reversal_strategy"
    assert signal.signal_type == "reversal"
    assert signal.exchange == "bybit"
    assert signal.symbol == "ETH/USDT"
    assert signal.side == "LONG"
    assert signal.confidence == 0.93
    assert 0.0 <= signal.score <= 1.0
    assert signal.detected_at == result.detected_at
    assert signal.source_topic == event.topic
    assert signal.severity == CascadeSeverity.EXTREME.value
    assert signal.cascade_direction == CascadeDirection.DOWN.value
    assert signal.liquidation_side == "long_liquidations"
    assert signal.event_count == 30
    assert signal.total_notional_usd == Decimal("2500000")
    assert signal.intensity_score == 0.89
    assert signal.continuation_bias == 0.21
    assert signal.exhaustion_bias == 0.91
    assert signal.bias_delta == 0.55
    assert signal.price_range_pct == 0.28
    assert signal.window_seconds == 15
    assert signal.cluster_duration_seconds == 3.0
    assert signal.cluster_avg_notional_per_event == Decimal("125000")
    assert signal.side_imbalance_ratio == 0.91
    assert signal.event_imbalance_ratio == 0.86
    assert signal.acceleration_ratio == 1.75
    assert signal.correlation_id == "contract-corr"
    assert signal.source_event_id == event.event_id

    assert "squeeze reversal after liquidation exhaustion" in signal.reason
    assert "direction=down" in signal.reason
    assert "exhaustion_bias=0.910" in signal.reason
    assert "bias_delta=0.550" in signal.reason

    assert "strategy" in signal.metadata
    assert "bus_event" in signal.metadata
    assert "analytics_metadata" in signal.metadata
    assert "cluster" in signal.metadata
    assert "squeeze_reversal" in signal.metadata

    squeeze_meta = signal.metadata["squeeze_reversal"]
    assert squeeze_meta["analytics_event_type"] == "liquidation_exhaustion"
    assert squeeze_meta["favors_exhaustion"] is True
    assert squeeze_meta["bias_delta"] == 0.55
    assert squeeze_meta["pending"]["enabled"] is False
    assert squeeze_meta["quality_snapshot"]["analytics_metadata"]["side_imbalance_ratio"] == 0.91

    emitted = event_bus.emitted[0]
    assert emitted["topic"] == "signal.generated"
    assert emitted["priority"] is EventPriority.HIGH
    assert emitted["source"] == "squeeze_reversal_strategy"
    assert emitted["headers"]["pending_confirmation"] == "false"
    assert emitted["headers"]["analytics_event_type"] == "liquidation_exhaustion"


# ============================================================================
# Pending creation / confirmation
# ============================================================================


@pytest.mark.asyncio
async def test_valid_exhaustion_result_creates_pending_candidate_not_signal_immediately(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        confirmation_delay_seconds=5.0,
        min_pending_age_seconds=0.0,
        pending_ttl_seconds=30.0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result()
    event = make_event(result, correlation_id="pending-create")

    await strategy._on_bus_event(event)

    state = get_state(strategy, result)
    candidate = state.pending

    assert isinstance(candidate, PendingReversalCandidate)
    assert candidate.exchange == result.exchange
    assert candidate.symbol == result.symbol
    assert candidate.result is result
    assert candidate.source_topic == event.topic
    assert candidate.source_event_id == event.event_id
    assert candidate.correlation_id == "pending-create"
    assert candidate.confirm_after > candidate.created_at
    assert candidate.expires_at > candidate.confirm_after
    assert candidate.cluster_signature
    assert 0.0 <= candidate.score_at_creation <= 1.0
    assert candidate.quality_snapshot["confidence"] == result.confidence
    assert candidate.quality_snapshot["analytics_metadata"]["acceleration_ratio"] == 1.35

    assert emitted_topics(event_bus) == [
        "strategy.liquidations.squeeze.pending_created"
    ]

    payload = event_bus.emitted[0]["payload"]
    headers = event_bus.emitted[0]["headers"]

    assert payload["state"] == "pending_created"
    assert payload["strategy_name"] == "squeeze_reversal_strategy"
    assert payload["signal_type"] == "reversal"
    assert payload["exchange"] == result.exchange
    assert payload["symbol"] == result.symbol
    assert headers["state"] == "pending_created"

    stats = strategy.get_stats()

    assert stats["processed_events"] == 1
    assert stats["pending_created"] == 1
    assert stats["emitted_signals"] == 0
    assert stats["recent_pending"] == 1
    assert stats["pending_keys"] == 1


@pytest.mark.asyncio
async def test_pending_candidate_confirms_and_emits_signal_when_ready(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config()
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(direction=CascadeDirection.DOWN)
    event = make_event(result, correlation_id="pending-confirm")

    await strategy._on_bus_event(event)

    assert emitted_topics(event_bus) == [
        "strategy.liquidations.squeeze.pending_created"
    ]

    await strategy.process_pending_candidates()

    topics = emitted_topics(event_bus)

    assert topics == [
        "strategy.liquidations.squeeze.pending_created",
        "signal.generated",
        "strategy.liquidations.squeeze.pending_confirmed",
    ]

    signal = latest_signal(event_bus)

    assert signal.side == "LONG"
    assert signal.is_pending_confirmed is True
    assert signal.pending_started_at is not None
    assert signal.pending_confirmed_at is not None
    assert signal.confirmation_delay_seconds is not None
    assert signal.confirmation_delay_seconds >= 0.0
    assert signal.source_event_id == event.event_id
    assert signal.correlation_id == "pending-confirm"

    emitted_signal = event_bus.emitted[1]
    assert emitted_signal["headers"]["pending_confirmation"] == "true"
    assert emitted_signal["headers"]["analytics_event_type"] == "liquidation_exhaustion"

    pending_confirmed_payload = event_bus.emitted[2]["payload"]
    assert pending_confirmed_payload["state"] == "pending_confirmed"

    state = get_state(strategy, result)

    assert state.pending is None
    assert state.last_signal_side == "LONG"
    assert state.last_detected_at == result.detected_at
    assert state.last_cluster_signature is not None
    assert state.total_signals_emitted == 1

    stats = strategy.get_stats()

    assert stats["pending_created"] == 1
    assert stats["pending_confirmed"] == 1
    assert stats["emitted_signals"] == 1
    assert stats["pending_keys"] == 0
    assert stats["rejected_events"] == 0


@pytest.mark.asyncio
async def test_pending_candidate_is_not_confirmed_before_confirm_after(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        confirmation_delay_seconds=60.0,
        min_pending_age_seconds=0.0,
        pending_ttl_seconds=120.0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result()
    await strategy._on_bus_event(make_event(result))

    await strategy.process_pending_candidates()

    state = get_state(strategy, result)

    assert state.pending is not None
    assert emitted_topics(event_bus) == [
        "strategy.liquidations.squeeze.pending_created"
    ]
    assert strategy.get_stats()["pending_confirmed"] == 0
    assert strategy.get_stats()["emitted_signals"] == 0


@pytest.mark.asyncio
async def test_pending_candidate_respects_min_pending_age_even_when_confirm_after_ready(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        confirmation_delay_seconds=0.0,
        min_pending_age_seconds=60.0,
        pending_ttl_seconds=120.0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result()
    await strategy._on_bus_event(make_event(result))

    await strategy.process_pending_candidates()

    state = get_state(strategy, result)

    assert state.pending is not None
    assert strategy.get_stats()["pending_confirmed"] == 0
    assert strategy.get_stats()["emitted_signals"] == 0
    assert emitted_topics(event_bus) == [
        "strategy.liquidations.squeeze.pending_created"
    ]


# ============================================================================
# Pending expiry / cancellation
# ============================================================================


@pytest.mark.asyncio
async def test_pending_candidate_expires_when_ttl_is_passed(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        confirmation_delay_seconds=0.0,
        min_pending_age_seconds=0.0,
        pending_ttl_seconds=30.0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result()
    await strategy._on_bus_event(make_event(result, correlation_id="expire-corr"))

    state = get_state(strategy, result)
    assert state.pending is not None

    state.pending.expires_at = utc_now() - timedelta(seconds=1)

    await strategy.process_pending_candidates()

    assert state.pending is None
    assert "signal.generated" not in emitted_topics(event_bus)
    assert emitted_topics(event_bus)[-1] == "strategy.liquidations.squeeze.pending_expired"

    expired_payload = event_bus.emitted[-1]["payload"]

    assert expired_payload["state"] == "pending_expired"
    assert expired_payload["correlation_id"] == "expire-corr"

    stats = strategy.get_stats()

    assert stats["pending_expired"] == 1
    assert stats["pending_confirmed"] == 0
    assert stats["emitted_signals"] == 0
    assert stats["pending_keys"] == 0


@pytest.mark.asyncio
async def test_pending_candidate_expires_when_newer_detected_at_exists(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(cancel_if_newer_detected_at=True)
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    old_result = make_result(
        detected_at=utc_now() - timedelta(seconds=5),
    )
    await strategy._on_bus_event(make_event(old_result, correlation_id="old-corr"))

    state = get_state(strategy, old_result)
    assert state.pending is not None

    state.latest_seen_detected_at = old_result.detected_at + timedelta(seconds=2)
    state.latest_seen_score = 0.99

    await strategy.process_pending_candidates()

    assert state.pending is None
    assert "signal.generated" not in emitted_topics(event_bus)

    expired_payload = event_bus.emitted[-1]["payload"]

    assert expired_payload["state"] == "newer_detected_at_exists"

    stats = strategy.get_stats()

    assert stats["pending_expired"] == 1
    assert stats["pending_confirmed"] == 0
    assert stats["emitted_signals"] == 0


@pytest.mark.asyncio
async def test_pending_candidate_is_not_cancelled_by_newer_detected_at_when_config_disabled(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(cancel_if_newer_detected_at=False)
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(detected_at=utc_now() - timedelta(seconds=5))
    await strategy._on_bus_event(make_event(result))

    state = get_state(strategy, result)
    assert state.pending is not None

    state.latest_seen_detected_at = result.detected_at + timedelta(seconds=3)

    await strategy.process_pending_candidates()

    assert state.pending is None
    assert strategy.get_stats()["pending_confirmed"] == 1
    assert strategy.get_stats()["emitted_signals"] == 1
    assert "signal.generated" in emitted_topics(event_bus)


@pytest.mark.asyncio
async def test_late_filter_failure_expires_pending_candidate(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        max_result_age_seconds=60.0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(detected_at=utc_now() - timedelta(seconds=5))
    await strategy._on_bus_event(make_event(result))

    state = get_state(strategy, result)
    assert state.pending is not None

    result.detected_at = utc_now() - timedelta(seconds=120)

    await strategy.process_pending_candidates()

    assert state.pending is None
    assert "signal.generated" not in emitted_topics(event_bus)

    expired_payload = event_bus.emitted[-1]["payload"]

    assert expired_payload["state"] == "late_filter_failed:result_too_old"
    assert strategy.get_stats()["pending_expired"] == 1
    assert strategy.get_stats()["pending_confirmed"] == 0


@pytest.mark.asyncio
async def test_cancelled_pending_candidate_is_silently_removed_without_expiry_event(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config()
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result()
    await strategy._on_bus_event(make_event(result))

    state = get_state(strategy, result)
    assert state.pending is not None

    state.pending.cancelled = True
    state.pending.cancel_reason = "manual_test_cancel"

    event_bus.emitted.clear()

    await strategy.process_pending_candidates()

    assert state.pending is None
    assert event_bus.emitted == []
    assert strategy.get_stats()["pending_expired"] == 0
    assert strategy.get_stats()["pending_confirmed"] == 0


@pytest.mark.asyncio
async def test_stale_pending_key_is_discarded_when_state_is_missing(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config()
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    strategy._pending_keys.add(("binance", "MISSING"))

    await strategy.process_pending_candidates()

    assert ("binance", "MISSING") not in strategy._pending_keys


# ============================================================================
# Pending replacement rules
# ============================================================================


@pytest.mark.asyncio
async def test_older_incoming_candidate_is_rejected_and_existing_pending_stays(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        confirmation_delay_seconds=60.0,
        min_pending_age_seconds=0.0,
        replace_pending_if_score_improves=True,
        min_replacement_score_delta=0.03,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    newer_existing = make_result(
        detected_at=utc_now() - timedelta(seconds=1),
        exhaustion_bias=0.88,
        bias_delta=0.35,
    )
    older_incoming = make_result(
        detected_at=utc_now() - timedelta(seconds=3),
        exhaustion_bias=0.95,
        bias_delta=0.50,
    )

    await strategy._on_bus_event(make_event(newer_existing, correlation_id="first"))
    state = get_state(strategy, newer_existing)
    existing_pending = state.pending

    await strategy._on_bus_event(make_event(older_incoming, correlation_id="older"))

    assert state.pending is existing_pending
    assert latest_rejection(strategy).reason == "older_than_existing_pending"
    assert strategy.get_stats()["pending_created"] == 1
    assert strategy.get_stats()["pending_replaced"] == 0
    assert strategy.get_stats()["duplicate_skips"] == 1


@pytest.mark.asyncio
async def test_newer_but_not_stronger_candidate_is_rejected_and_existing_pending_stays(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        confirmation_delay_seconds=60.0,
        min_replacement_score_delta=0.20,
        replace_pending_if_score_improves=True,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    first = make_result(
        detected_at=utc_now() - timedelta(seconds=5),
        confidence=0.90,
        exhaustion_bias=0.90,
        bias_delta=0.40,
    )
    second = make_result(
        detected_at=utc_now() - timedelta(seconds=1),
        confidence=0.91,
        exhaustion_bias=0.91,
        bias_delta=0.41,
    )

    await strategy._on_bus_event(make_event(first))
    state = get_state(strategy, first)
    existing_pending = state.pending

    await strategy._on_bus_event(make_event(second))

    assert state.pending is existing_pending
    assert latest_rejection(strategy).reason == "newer_pending_not_stronger_enough"
    assert strategy.get_stats()["pending_created"] == 1
    assert strategy.get_stats()["pending_replaced"] == 0
    assert strategy.get_stats()["pending_cancelled"] == 0
    assert strategy.get_stats()["duplicate_skips"] == 1


@pytest.mark.asyncio
async def test_newer_and_stronger_candidate_replaces_existing_pending(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        confirmation_delay_seconds=60.0,
        min_replacement_score_delta=0.01,
        replace_pending_if_score_improves=True,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    weak = make_result(
        detected_at=utc_now() - timedelta(seconds=5),
        confidence=0.70,
        exhaustion_bias=0.72,
        bias_delta=0.13,
        intensity_score=0.62,
        severity=CascadeSeverity.HIGH,
        cluster=make_cluster(avg_notional_per_event=Decimal("50000")),
    )
    strong = make_result(
        detected_at=utc_now() - timedelta(seconds=1),
        confidence=0.96,
        exhaustion_bias=0.96,
        bias_delta=0.75,
        intensity_score=0.95,
        severity=CascadeSeverity.EXTREME,
        cluster=make_cluster(avg_notional_per_event=Decimal("200000")),
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

    topics = emitted_topics(event_bus)

    assert topics == [
        "strategy.liquidations.squeeze.pending_created",
        "strategy.liquidations.squeeze.pending_replaced",
        "strategy.liquidations.squeeze.pending_created",
    ]

    replaced_payload = event_bus.emitted[1]["payload"]

    assert replaced_payload["state"] == "replaced_by_newer_stronger_pending"

    stats = strategy.get_stats()

    assert stats["pending_created"] == 2
    assert stats["pending_replaced"] == 1
    assert stats["pending_cancelled"] == 1
    assert stats["rejected_events"] == 0


@pytest.mark.asyncio
async def test_newer_candidate_replaces_even_without_score_improvement_when_config_disabled(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        confirmation_delay_seconds=60.0,
        replace_pending_if_score_improves=False,
        min_replacement_score_delta=0.99,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    strong_first = make_result(
        detected_at=utc_now() - timedelta(seconds=5),
        confidence=0.96,
        exhaustion_bias=0.96,
        bias_delta=0.70,
    )
    weaker_newer = make_result(
        detected_at=utc_now() - timedelta(seconds=1),
        confidence=0.75,
        exhaustion_bias=0.75,
        bias_delta=0.15,
    )

    await strategy._on_bus_event(make_event(strong_first))
    state = get_state(strategy, strong_first)
    old_pending = state.pending

    await strategy._on_bus_event(make_event(weaker_newer))

    assert state.pending is not old_pending
    assert state.pending is not None
    assert state.pending.result is weaker_newer
    assert strategy.get_stats()["pending_replaced"] == 1
    assert strategy.get_stats()["pending_created"] == 2


# ============================================================================
# Exhaustion-specific filters
# ============================================================================


@pytest.mark.parametrize(
    ("config_overrides", "result_overrides", "expected_reason"),
    [
        ({}, {"is_confirmed": False}, "result_not_confirmed"),
        ({}, {"is_actionable_severity": False}, "severity_not_actionable"),
        ({}, {"favors_exhaustion": False}, "exhaustion_not_favored"),
        (
            {"min_exhaustion_bias": 0.90},
            {"exhaustion_bias": 0.80},
            "exhaustion_bias_below_threshold",
        ),
        (
            {"min_bias_delta": 0.40},
            {"bias_delta": 0.28},
            "bias_delta_below_threshold",
        ),
        (
            {"max_continuation_bias_after_exhaustion": 0.25},
            {"continuation_bias": 0.32},
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
            {"cluster": make_cluster(duration_seconds=4.0)},
            "cluster_duration_too_long",
        ),
        (
            {"min_avg_notional_per_event": Decimal("100000")},
            {"cluster": make_cluster(avg_notional_per_event=Decimal("75000"))},
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
        exhaustion_bias=0.10,
        bias_delta=0.01,
        favors_exhaustion=False,
    )

    await strategy._on_bus_event(make_event(result))

    rejection = latest_rejection(strategy)

    assert rejection.reason == "confidence_below_threshold"
    assert rejection.reason != "exhaustion_not_favored"
    assert rejection.reason != "exhaustion_bias_below_threshold"
    assert rejection.reason != "bias_delta_below_threshold"


@pytest.mark.asyncio
async def test_rejected_event_is_published_with_full_headers_when_enabled(
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
        symbol="SOL-USDT",
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
    assert rejection.symbol == "SOL-USDT"
    assert rejection.correlation_id == "reject-corr"
    assert rejection.source_event_id == event.event_id

    assert emitted["priority"] is EventPriority.LOW
    assert emitted["source"] == "squeeze_reversal_strategy"
    assert emitted["correlation_id"] == "reject-corr"

    headers = emitted["headers"]

    assert headers["strategy"] == "squeeze_reversal_strategy"
    assert headers["signal_type"] == "reversal"
    assert headers["exchange"] == "okx"
    assert headers["symbol"] == "SOL-USDT"
    assert headers["reason"] == "exhaustion_bias_below_threshold"
    assert headers["source_event_id"] == event.event_id
    assert headers["source_topic"] == "analytics.liquidation.exhaustion_detected"


# ============================================================================
# Scoring / cluster quality / metadata extraction
# ============================================================================


def test_compute_cluster_quality_score_rewards_fast_dense_clusters(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        max_cluster_duration_seconds=10.0,
        min_avg_notional_per_event=Decimal("50000"),
        max_price_range_pct=1.0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)

    strong = make_result(
        price_range_pct=0.10,
        cluster=make_cluster(
            duration_seconds=2.0,
            avg_notional_per_event=Decimal("150000"),
        ),
    )
    weak = make_result(
        price_range_pct=1.00,
        cluster=make_cluster(
            duration_seconds=10.0,
            avg_notional_per_event=Decimal("10000"),
        ),
    )

    assert strategy.compute_cluster_quality_score(strong) > strategy.compute_cluster_quality_score(weak)
    assert 0.0 <= strategy.compute_cluster_quality_score(strong) <= 1.0
    assert 0.0 <= strategy.compute_cluster_quality_score(weak) <= 1.0


def test_compute_strategy_score_uses_exhaustion_bias_delta_intensity_severity_and_cluster_quality(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        score_confidence_weight=0.25,
        score_exhaustion_bias_weight=0.30,
        score_bias_delta_weight=0.15,
        score_intensity_weight=0.12,
        score_severity_weight=0.10,
        score_cluster_quality_weight=0.08,
        max_cluster_duration_seconds=10.0,
        min_avg_notional_per_event=Decimal("50000"),
        max_price_range_pct=1.0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)

    result = make_result(
        confidence=0.80,
        exhaustion_bias=0.90,
        bias_delta=0.50,
        intensity_score=0.70,
        severity=CascadeSeverity.HIGH,
        price_range_pct=0.20,
        cluster=make_cluster(
            duration_seconds=2.0,
            avg_notional_per_event=Decimal("100000"),
        ),
    )

    cluster_quality = strategy.compute_cluster_quality_score(result)

    expected = (
        0.80 * 0.25
        + 0.90 * 0.30
        + 0.50 * 0.15
        + 0.70 * 0.12
        + 0.80 * 0.10
        + cluster_quality * 0.08
    ) / 1.0

    assert strategy.compute_strategy_score(result) == pytest.approx(expected)


def test_extract_analytics_metadata_converts_invalid_values_to_none(
    event_bus: FakeEventBus,
) -> None:
    strategy = make_strategy(event_bus=event_bus)

    result = make_result(
        metadata={
            "side_imbalance_ratio": "0.88",
            "event_imbalance_ratio": None,
            "acceleration_ratio": "bad-float",
        }
    )

    extracted = strategy.extract_analytics_metadata(result)

    assert extracted == {
        "side_imbalance_ratio": 0.88,
        "event_imbalance_ratio": None,
        "acceleration_ratio": None,
    }


def test_build_quality_snapshot_contains_filter_critical_fields(
    event_bus: FakeEventBus,
) -> None:
    strategy = make_strategy(event_bus=event_bus)

    result = make_result(
        confidence=0.91,
        exhaustion_bias=0.89,
        bias_delta=0.44,
        metadata=make_metadata(side_imbalance_ratio=0.93),
    )

    snapshot = strategy.build_quality_snapshot(result)

    assert snapshot["confidence"] == 0.91
    assert snapshot["exhaustion_bias"] == 0.89
    assert snapshot["bias_delta"] == 0.44
    assert snapshot["event_type"] == "liquidation_exhaustion"
    assert snapshot["cluster"]["avg_notional_per_event"] == str(
        result.cluster.avg_notional_per_event
    )
    assert snapshot["analytics_metadata"]["side_imbalance_ratio"] == 0.93


# ============================================================================
# Emit failure / EventBus robustness
# ============================================================================


@pytest.mark.asyncio
async def test_direct_emit_failure_does_not_mark_signal_as_emitted() -> None:
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


@pytest.mark.xfail(
    reason=(
        "Known vulnerability: process_pending_candidates() ignores the False return value "
        "from emit_confirmed_signal(). If signal.generated emit fails, it still increments "
        "pending_confirmed and removes pending candidate. Fix recommendation: only increment "
        "pending_confirmed and clear pending when emit_confirmed_signal() returns True."
    ),
    strict=True,
)
@pytest.mark.asyncio
async def test_pending_confirmation_emit_failure_must_not_clear_pending_or_increment_confirmed() -> None:
    event_bus = FailingEventBus()
    config = make_fast_pending_config(publish_pending_events=False)
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


# ============================================================================
# Diagnostics / state query API
# ============================================================================


@pytest.mark.asyncio
async def test_get_hot_symbols_returns_confirmed_reversal_signals_sorted_by_score(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        enable_pending_confirmation=False,
        hot_symbols_window_seconds=300,
        deduplicate_by_detected_at=False,
        deduplicate_same_cluster_signature=False,
        max_signals_per_symbol_window=0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    weak_btc = make_result(
        symbol="BTCUSDT",
        confidence=0.70,
        intensity_score=0.62,
        exhaustion_bias=0.72,
        bias_delta=0.13,
        severity=CascadeSeverity.HIGH,
    )
    strong_eth = make_result(
        symbol="ETHUSDT",
        confidence=0.96,
        intensity_score=0.91,
        exhaustion_bias=0.95,
        bias_delta=0.70,
        severity=CascadeSeverity.EXTREME,
    )

    await strategy._on_bus_event(make_event(weak_btc))
    await strategy._on_bus_event(make_event(strong_eth))

    rows = strategy.get_hot_symbols(limit=10)

    assert len(rows) == 2
    assert rows[0]["symbol"] == "ETHUSDT"
    assert rows[0]["score"] >= rows[1]["score"]
    assert rows[0]["side"] == "LONG"
    assert rows[0]["severity"] == CascadeSeverity.EXTREME.value


@pytest.mark.asyncio
async def test_get_symbol_state_exposes_pending_runtime_state(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(confirmation_delay_seconds=60.0)
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(exchange="binance", symbol="BTC/USDT")

    await strategy._on_bus_event(make_event(result))

    snapshot = strategy.get_symbol_state("BINANCE", "BTCUSDT")

    assert snapshot["exists"] is True
    assert snapshot["exchange"] == "binance"
    assert snapshot["symbol"] == "BTCUSDT"
    assert snapshot["pending"] is not None
    assert snapshot["pending"]["score_at_creation"] is not None
    assert snapshot["latest_seen_detected_at"] is not None
    assert snapshot["latest_seen_score"] is not None


@pytest.mark.asyncio
async def test_publish_diagnostics_snapshot_contains_pending_and_hot_symbol_context(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        publish_diagnostics_snapshots=True,
        enable_pending_confirmation=False,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    await strategy._on_bus_event(make_event(make_result(symbol="BTCUSDT")))

    event_bus.emitted.clear()

    await strategy.publish_diagnostics_snapshot()

    assert emitted_topics(event_bus) == ["strategy.liquidations.squeeze.snapshot"]

    snapshot = event_bus.emitted[0]["payload"]

    assert snapshot["strategy_name"] == "squeeze_reversal_strategy"
    assert snapshot["signal_type"] == "reversal"
    assert snapshot["stats"]["emitted_signals"] == 1
    assert len(snapshot["hot_symbols"]) == 1
    assert snapshot["hot_symbols"][0]["symbol"] == "BTCUSDT"
    assert "recent_pending" in snapshot

    headers = event_bus.emitted[0]["headers"]

    assert headers["strategy"] == "squeeze_reversal_strategy"
    assert headers["signal_type"] == "reversal"
    assert headers["event_type"] == "strategy_diagnostics"


# ============================================================================
# Payload validation / robustness
# ============================================================================


@pytest.mark.asyncio
async def test_invalid_payload_type_is_ignored_and_does_not_create_pending(
    event_bus: FakeEventBus,
) -> None:
    strategy = make_strategy(event_bus=event_bus)
    await strategy.start()

    await strategy._on_bus_event(make_event({"not": "cascade-result"}))

    assert strategy.get_stats()["processed_events"] == 0
    assert strategy.get_stats()["invalid_payload_skips"] == 1
    assert strategy.get_stats()["pending_created"] == 0
    assert event_bus.emitted == []


@pytest.mark.xfail(
    reason=(
        "Known vulnerability: unsupported non-unknown direction can pass common filters "
        "and direction_to_trade_side() returns FLAT. Fix recommendation: reject trade_side == 'FLAT'."
    ),
    strict=True,
)
@pytest.mark.asyncio
async def test_unsupported_non_unknown_direction_must_not_emit_flat_reversal_signal(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(
        enable_pending_confirmation=False,
        allowed_severities=(),
        require_actionable_severity=False,
        require_favors_exhaustion=False,
        max_continuation_bias_after_exhaustion=None,
        min_side_imbalance_ratio=None,
        min_climax_acceleration_ratio=None,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(
        direction=FakeDirection("sideways"),
        severity=FakeValue("weird"),
        event_type=FakeValue("weird_event"),
    )

    await strategy._on_bus_event(make_event(result))

    assert strategy.get_stats()["emitted_signals"] == 0
    assert strategy.get_stats()["rejected_events"] == 1
    assert latest_rejection(strategy).reason == "unsupported_trade_side"


# ============================================================================
# Serialization / mutation safety
# ============================================================================


def test_build_signal_does_not_mutate_source_result(
    event_bus: FakeEventBus,
) -> None:
    config = make_fast_pending_config(enable_pending_confirmation=False)
    strategy = make_strategy(event_bus=event_bus, config=config)

    result = make_result(
        confidence=0.84,
        intensity_score=0.79,
        continuation_bias=0.22,
        exhaustion_bias=0.88,
        bias_delta=0.33,
    )
    event = make_event(result)

    before = {
        "confidence": result.confidence,
        "intensity_score": result.intensity_score,
        "continuation_bias": result.continuation_bias,
        "exhaustion_bias": result.exhaustion_bias,
        "bias_delta": result.bias_delta,
        "metadata": dict(result.metadata),
    }

    signal = strategy.build_signal(
        result=result,
        bus_event=event,
        pending_started_at=None,
        pending_confirmed_at=utc_now(),
        source_event_id=event.event_id,
    )

    assert isinstance(signal, SqueezeReversalSignal)

    assert result.confidence == before["confidence"]
    assert result.intensity_score == before["intensity_score"]
    assert result.continuation_bias == before["continuation_bias"]
    assert result.exhaustion_bias == before["exhaustion_bias"]
    assert result.bias_delta == before["bias_delta"]
    assert result.metadata == before["metadata"]


def test_signal_to_dict_serializes_decimal_datetime_and_computed_properties(
    event_bus: FakeEventBus,
) -> None:
    strategy = make_strategy(event_bus=event_bus)

    result = make_result(total_notional_usd=Decimal("1234567.89"))
    event = make_event(result)
    pending_started = utc_now() - timedelta(seconds=2)
    pending_confirmed = utc_now()

    signal = strategy.build_signal(
        result=result,
        bus_event=event,
        pending_started_at=pending_started,
        pending_confirmed_at=pending_confirmed,
        source_event_id=event.event_id,
    )

    serialized = signal.to_dict(serialize=True)

    assert serialized["total_notional_usd"] == "1234567.89"
    assert serialized["cluster_avg_notional_per_event"] == str(
        result.cluster.avg_notional_per_event
    )
    assert isinstance(serialized["generated_at"], str)
    assert isinstance(serialized["detected_at"], str)
    assert serialized["is_pending_confirmed"] is True
    assert serialized["confirmation_delay_seconds"] is not None
    assert serialized["is_long"] is True
    assert serialized["is_short"] is False