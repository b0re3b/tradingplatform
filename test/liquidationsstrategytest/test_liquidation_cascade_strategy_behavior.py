from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest

from core.event_bus import Event, EventPriority

from analytics.liquidations.enums import CascadeDirection, CascadeSeverity

from strategy.strategies.liquidations.base import StrategyRejection
from strategy.strategies.liquidations.liquidation_cascade_strategy import (
    LiquidationCascadeSignal,
    LiquidationCascadeStrategy,
    LiquidationCascadeStrategyConfig,
    SymbolCascadeStrategyState,
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

    Ми не тестуємо analytics model/dataclass тут. Ми тестуємо саме поведінку
    LiquidationCascadeStrategy, тому підміняємо strategy.payload_type на цей клас.
    Direction/severity беремо з реальних enums, бо strategy порівнює direction через `is`.
    """

    exchange: str
    symbol: str
    detected_at: datetime

    confidence: float
    intensity_score: float
    continuation_bias: float
    exhaustion_bias: float

    event_count: int
    total_notional_usd: Decimal
    price_range_pct: float

    severity: CascadeSeverity
    direction: Any
    side: Any

    favors_continuation: bool
    correlation_id: str | None
    metadata: dict[str, Any]
    cluster: FakeCluster

    window_seconds: int = 10

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
        "event_count": 10,
        "total_notional_usd": Decimal("750000"),
        "avg_notional_per_event": Decimal("75000"),
        "avg_price": Decimal("100000"),
        "min_price": Decimal("99500"),
        "max_price": Decimal("100500"),
        "duration_seconds": 4.0,
        "price_range_pct": 0.35,
    }
    base.update(overrides)
    return FakeCluster(**base)


def make_result(**overrides: Any) -> FakeCascadeDetectionResult:
    now = utc_now()

    base = {
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "detected_at": now - timedelta(seconds=1),
        "confidence": 0.82,
        "intensity_score": 0.77,
        "continuation_bias": 0.72,
        "exhaustion_bias": 0.18,
        "event_count": 12,
        "total_notional_usd": Decimal("750000"),
        "price_range_pct": 0.35,
        "severity": CascadeSeverity.HIGH,
        "direction": CascadeDirection.UP,
        "side": FakeValue("short_liquidations"),
        "favors_continuation": True,
        "correlation_id": "analytics-corr-1",
        "metadata": {
            "detector": "cascade_detector",
            "test_case": "liquidation_cascade_strategy",
        },
        "cluster": make_cluster(),
        "window_seconds": 10,
    }
    base.update(overrides)
    return FakeCascadeDetectionResult(**base)


def make_event(
    payload: Any,
    *,
    topic: str = "analytics.liquidation.cascade_detected",
    correlation_id: str | None = "bus-corr-1",
) -> Event:
    return Event(
        topic=topic,
        payload=payload,
        priority=EventPriority.NORMAL,
        source="analytics.liquidations.cascade_detector",
        correlation_id=correlation_id,
    )


def make_strategy(
    *,
    event_bus: FakeEventBus,
    config: LiquidationCascadeStrategyConfig | None = None,
    scheduler: FakeScheduler | None = None,
) -> LiquidationCascadeStrategy:
    strategy = LiquidationCascadeStrategy(
        event_bus=event_bus,  # type: ignore[arg-type]
        config=config,
        scheduler=scheduler,  # type: ignore[arg-type]
    )

    # Цей тестовий файл не тестує analytics.models.CascadeDetectionResult.
    # Тому payload type підміняємо на поведінковий fake.
    strategy.payload_type = FakeCascadeDetectionResult  # type: ignore[assignment]

    return strategy


def emitted_topics(event_bus: FakeEventBus) -> list[str]:
    return [item["topic"] for item in event_bus.emitted]


def emitted_payloads(event_bus: FakeEventBus, topic: str) -> list[Any]:
    return [
        item["payload"]
        for item in event_bus.emitted
        if item["topic"] == topic
    ]


def latest_signal(event_bus: FakeEventBus) -> LiquidationCascadeSignal:
    payloads = emitted_payloads(event_bus, "signal.generated")
    assert payloads, "expected at least one signal.generated event"
    signal = payloads[-1]
    assert isinstance(signal, LiquidationCascadeSignal)
    return signal


def latest_rejection(strategy: LiquidationCascadeStrategy) -> StrategyRejection:
    rejections = strategy.get_recent_rejections(limit=1)
    assert rejections, "expected at least one rejection"
    return rejections[0]


def assert_no_risk_or_execution_events(event_bus: FakeEventBus) -> None:
    forbidden_prefixes = ("risk.", "execution.", "order.", "position.")
    for topic in emitted_topics(event_bus):
        assert not topic.startswith(forbidden_prefixes), (
            f"strategy must not emit direct risk/execution/order/position event: {topic}"
        )


# ============================================================================
# Lifecycle / subscription contract
# ============================================================================


@pytest.mark.asyncio
async def test_start_subscribes_to_liquidation_cascade_topic(
    event_bus: FakeEventBus,
) -> None:
    strategy = make_strategy(event_bus=event_bus)

    await strategy.start()

    assert strategy.get_stats()["running"] is True
    assert len(event_bus.subscriptions) == 1
    assert event_bus.subscriptions[0].topic == "analytics.liquidation.cascade_detected"
    assert event_bus.subscriptions[0].name == (
        "liquidation_cascade_strategy.on_analytics_event"
    )


@pytest.mark.asyncio
async def test_diagnostics_job_is_registered_only_when_enabled(
    event_bus: FakeEventBus,
) -> None:
    scheduler = FakeScheduler()
    config = LiquidationCascadeStrategyConfig(
        publish_diagnostics_snapshots=True,
        diagnostics_interval_seconds=7.5,
    )
    strategy = make_strategy(
        event_bus=event_bus,
        config=config,
        scheduler=scheduler,
    )

    await strategy.start()

    assert strategy.get_stats()["diagnostics_job_registered"] is True
    assert len(scheduler.jobs) == 1

    job = next(iter(scheduler.jobs.values()))
    assert job["name"] == "liquidation_cascade_strategy:diagnostics"
    assert job["interval"] == 7.5
    assert job["allow_overlap"] is False
    assert job["enabled"] is True


# ============================================================================
# Direction mapping / happy paths
# ============================================================================


@pytest.mark.asyncio
async def test_up_cascade_emits_long_continuation_signal(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(direction=CascadeDirection.UP)
    event = make_event(result, correlation_id="corr-up")

    await strategy._on_bus_event(event)

    signal = latest_signal(event_bus)

    assert signal.side == "LONG"
    assert signal.cascade_direction == CascadeDirection.UP.value
    assert signal.signal_type == "continuation"
    assert signal.strategy_name == "liquidation_cascade_strategy"
    assert signal.source_event_id == event.event_id
    assert signal.correlation_id == "corr-up"

    assert strategy.get_stats()["emitted_signals"] == 1
    assert strategy.get_stats()["rejected_events"] == 0

    assert_no_risk_or_execution_events(event_bus)


@pytest.mark.asyncio
async def test_down_cascade_emits_short_continuation_signal(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(direction=CascadeDirection.DOWN)
    event = make_event(result, correlation_id="corr-down")

    await strategy._on_bus_event(event)

    signal = latest_signal(event_bus)

    assert signal.side == "SHORT"
    assert signal.cascade_direction == CascadeDirection.DOWN.value
    assert signal.signal_type == "continuation"
    assert signal.source_event_id == event.event_id
    assert signal.correlation_id == "corr-down"

    assert strategy.get_stats()["emitted_signals"] == 1
    assert strategy.get_stats()["rejected_events"] == 0

    assert_no_risk_or_execution_events(event_bus)


@pytest.mark.asyncio
async def test_valid_result_emits_complete_signal_contract_and_updates_state(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        symbol_cooldown_seconds=15,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(
        exchange="bybit",
        symbol="ETH/USDT",
        direction=CascadeDirection.DOWN,
        side=FakeValue("long_liquidations"),
        confidence=0.91,
        intensity_score=0.88,
        continuation_bias=0.83,
        exhaustion_bias=0.11,
        event_count=25,
        total_notional_usd=Decimal("1750000"),
        price_range_pct=0.42,
        severity=CascadeSeverity.EXTREME,
    )
    event = make_event(result, correlation_id="contract-corr")

    await strategy._on_bus_event(event)

    signal = latest_signal(event_bus)

    assert signal.strategy_name == "liquidation_cascade_strategy"
    assert signal.signal_type == "continuation"
    assert signal.exchange == "bybit"
    assert signal.symbol == "ETH/USDT"
    assert signal.side == "SHORT"
    assert signal.confidence == 0.91
    assert 0.0 <= signal.score <= 1.0
    assert signal.detected_at == result.detected_at
    assert signal.source_topic == event.topic
    assert signal.severity == CascadeSeverity.EXTREME.value
    assert signal.cascade_direction == CascadeDirection.DOWN.value
    assert signal.liquidation_side == "long_liquidations"
    assert signal.event_count == 25
    assert signal.total_notional_usd == Decimal("1750000")
    assert signal.intensity_score == 0.88
    assert signal.continuation_bias == 0.83
    assert signal.exhaustion_bias == 0.11
    assert signal.price_range_pct == 0.42
    assert signal.correlation_id == "contract-corr"
    assert signal.source_event_id == event.event_id

    assert "liquidation cascade continuation" in signal.reason
    assert "direction=down" in signal.reason
    assert "continuation_bias=0.830" in signal.reason
    assert "confidence=0.910" in signal.reason

    assert "strategy" in signal.metadata
    assert "bus_event" in signal.metadata
    assert "analytics" in signal.metadata
    assert "cluster" in signal.metadata
    assert "liquidation_strategy" in signal.metadata

    liquidation_meta = signal.metadata["liquidation_strategy"]
    assert liquidation_meta["min_continuation_bias"] == config.min_continuation_bias
    assert liquidation_meta["require_favors_continuation"] is True
    assert (
        liquidation_meta["max_future_detected_at_seconds"]
        == config.max_future_detected_at_seconds
    )

    state = strategy.get_or_create_state("bybit", "ETH/USDT")

    assert isinstance(state, SymbolCascadeStrategyState)
    assert state.last_signal_side == "SHORT"
    assert state.last_detected_at == result.detected_at
    assert state.last_cluster_signature is not None
    assert state.total_signals_emitted == 1
    assert state.cooldown_until is not None

    stats = strategy.get_stats()

    assert stats["processed_events"] == 1
    assert stats["emitted_signals"] == 1
    assert stats["rejected_events"] == 0
    assert stats["tracked_symbols"] == 1
    assert stats["recent_signals"] == 1


@pytest.mark.asyncio
async def test_signal_values_are_clamped_but_filters_still_allow_high_values(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(
        confidence=1.25,
        intensity_score=1.10,
        continuation_bias=1.40,
        exhaustion_bias=-0.50,
    )

    await strategy._on_bus_event(make_event(result))

    signal = latest_signal(event_bus)

    assert signal.confidence == 1.0
    assert signal.intensity_score == 1.0
    assert signal.continuation_bias == 1.0
    assert signal.exhaustion_bias == 0.0
    assert signal.score == 1.0


# ============================================================================
# Domain-specific liquidation filters
# ============================================================================


@pytest.mark.parametrize(
    ("config_overrides", "result_overrides", "expected_reason"),
    [
        (
            {"require_favors_continuation": True},
            {"favors_continuation": False},
            "continuation_not_favored",
        ),
        (
            {"min_continuation_bias": 0.80},
            {"continuation_bias": 0.79},
            "continuation_bias_below_threshold",
        ),
        (
            {"max_future_detected_at_seconds": 1.0},
            {"detected_at": utc_now() + timedelta(seconds=30)},
            "detected_at_in_future",
        ),
    ],
)
@pytest.mark.asyncio
async def test_liquidation_specific_rejections(
    event_bus: FakeEventBus,
    config_overrides: dict[str, Any],
    result_overrides: dict[str, Any],
    expected_reason: str,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        publish_rejected_events=False,
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
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
    assert rejection.strategy_name == "liquidation_cascade_strategy"
    assert rejection.signal_type == "continuation"
    assert rejection.details["continuation_bias"] == result.continuation_bias


@pytest.mark.asyncio
async def test_favors_continuation_can_be_disabled_for_counterfactual_testing(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        require_favors_continuation=False,
        min_continuation_bias=0.60,
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(
        favors_continuation=False,
        continuation_bias=0.71,
    )

    await strategy._on_bus_event(make_event(result))

    assert strategy.get_stats()["emitted_signals"] == 1
    assert strategy.get_stats()["rejected_events"] == 0

    signal = latest_signal(event_bus)
    assert signal.continuation_bias == 0.71


@pytest.mark.asyncio
async def test_common_filter_runs_before_liquidation_specific_filter(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        min_confidence=0.90,
        min_continuation_bias=0.95,
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(
        confidence=0.50,
        continuation_bias=0.10,
        favors_continuation=False,
    )

    await strategy._on_bus_event(make_event(result))

    rejection = latest_rejection(strategy)

    assert rejection.reason == "confidence_below_threshold"
    assert rejection.reason != "continuation_not_favored"
    assert rejection.reason != "continuation_bias_below_threshold"


# ============================================================================
# Rejection publishing / EventBus contract
# ============================================================================


@pytest.mark.asyncio
async def test_rejected_event_is_published_with_full_headers_when_enabled(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        publish_rejected_events=True,
        min_continuation_bias=0.95,
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(
        exchange="okx",
        symbol="SOL-USDT",
        continuation_bias=0.40,
    )
    event = make_event(result, correlation_id="reject-corr")

    await strategy._on_bus_event(event)

    assert emitted_topics(event_bus) == ["signal.rejected"]

    emitted = event_bus.emitted[0]
    rejection = emitted["payload"]

    assert isinstance(rejection, StrategyRejection)
    assert rejection.reason == "continuation_bias_below_threshold"
    assert rejection.exchange == "okx"
    assert rejection.symbol == "SOL-USDT"
    assert rejection.correlation_id == "reject-corr"
    assert rejection.source_event_id == event.event_id

    assert emitted["priority"] is EventPriority.LOW
    assert emitted["source"] == "liquidation_cascade_strategy"
    assert emitted["correlation_id"] == "reject-corr"

    headers = emitted["headers"]

    assert headers["strategy"] == "liquidation_cascade_strategy"
    assert headers["signal_type"] == "continuation"
    assert headers["exchange"] == "okx"
    assert headers["symbol"] == "SOL-USDT"
    assert headers["reason"] == "continuation_bias_below_threshold"
    assert headers["source_event_id"] == event.event_id
    assert headers["source_topic"] == "analytics.liquidation.cascade_detected"


@pytest.mark.asyncio
async def test_signal_generated_event_has_only_strategy_level_topic_not_risk_or_execution(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result()
    event = make_event(result, correlation_id="signal-corr")

    await strategy._on_bus_event(event)

    assert emitted_topics(event_bus) == ["signal.generated"]
    assert_no_risk_or_execution_events(event_bus)

    emitted = event_bus.emitted[0]

    assert emitted["source"] == "liquidation_cascade_strategy"
    assert emitted["priority"] is EventPriority.HIGH
    assert emitted["correlation_id"] == "signal-corr"

    headers = emitted["headers"]

    assert headers["strategy"] == "liquidation_cascade_strategy"
    assert headers["signal_type"] == "continuation"
    assert headers["source_event_id"] == event.event_id
    assert headers["source_topic"] == event.topic
    assert headers["exchange"] == result.exchange
    assert headers["symbol"] == result.symbol
    assert headers["side"] == "LONG"


@pytest.mark.asyncio
async def test_emit_failure_does_not_update_state_or_stats_as_emitted() -> None:
    event_bus = FailingEventBus()
    config = LiquidationCascadeStrategyConfig(
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result()

    await strategy._on_bus_event(make_event(result))

    stats = strategy.get_stats()
    state = strategy.get_or_create_state(result.exchange, result.symbol)

    assert stats["processed_events"] == 1
    assert stats["emitted_signals"] == 0
    assert stats["rejected_events"] == 0
    assert stats["last_error"] is not None
    assert "event bus emit failed" in stats["last_error"]

    assert state.total_signals_emitted == 0
    assert state.last_signal_at is None
    assert state.last_detected_at is None
    assert state.last_cluster_signature is None


# ============================================================================
# Scoring
# ============================================================================


@pytest.mark.parametrize(
    ("severity", "expected_severity_score"),
    [
        (CascadeSeverity.LOW, 0.4),
        (CascadeSeverity.MEDIUM, 0.6),
        (CascadeSeverity.HIGH, 0.8),
        (CascadeSeverity.EXTREME, 1.0),
    ],
)
def test_compute_strategy_score_uses_weighted_components(
    event_bus: FakeEventBus,
    severity: CascadeSeverity,
    expected_severity_score: float,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        score_confidence_weight=0.35,
        score_continuation_bias_weight=0.35,
        score_intensity_weight=0.20,
        score_severity_weight=0.10,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)

    result = make_result(
        confidence=0.80,
        continuation_bias=0.70,
        intensity_score=0.60,
        severity=severity,
    )

    expected = (
        0.80 * 0.35
        + 0.70 * 0.35
        + 0.60 * 0.20
        + expected_severity_score * 0.10
    ) / 1.0

    assert strategy.compute_strategy_score(result) == pytest.approx(expected)


def test_compute_strategy_score_clamps_out_of_range_inputs(
    event_bus: FakeEventBus,
) -> None:
    strategy = make_strategy(event_bus=event_bus)

    result = make_result(
        confidence=9.0,
        continuation_bias=3.0,
        intensity_score=2.0,
        severity=CascadeSeverity.EXTREME,
    )

    assert strategy.compute_strategy_score(result) == 1.0


def test_severity_unknown_value_contributes_zero_to_score(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        allowed_severities=(),
        score_confidence_weight=0.0,
        score_continuation_bias_weight=0.0,
        score_intensity_weight=0.0,
        score_severity_weight=1.0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)

    result = make_result(severity=FakeValue("catastrophic"))  # type: ignore[arg-type]

    assert strategy.compute_strategy_score(result) == 0.0


# ============================================================================
# Dedup / cooldown / same-side / rate-limit
# ============================================================================


@pytest.mark.asyncio
async def test_duplicate_detected_at_is_rejected_before_second_signal(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
        deduplicate_by_detected_at=True,
        deduplicate_same_cluster_signature=False,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    detected_at = utc_now() - timedelta(seconds=1)

    first = make_result(detected_at=detected_at)
    second = make_result(detected_at=detected_at)

    await strategy._on_bus_event(make_event(first))
    await strategy._on_bus_event(make_event(second))

    assert strategy.get_stats()["emitted_signals"] == 1
    assert strategy.get_stats()["rejected_events"] == 1
    assert strategy.get_stats()["duplicate_skips"] == 1
    assert latest_rejection(strategy).reason == "duplicate_detected_at"


@pytest.mark.asyncio
async def test_duplicate_cluster_signature_is_rejected_when_detected_at_dedup_disabled(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
        deduplicate_by_detected_at=False,
        deduplicate_same_cluster_signature=True,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result()

    await strategy._on_bus_event(make_event(result))
    await strategy._on_bus_event(make_event(result))

    assert strategy.get_stats()["emitted_signals"] == 1
    assert strategy.get_stats()["rejected_events"] == 1
    assert strategy.get_stats()["duplicate_skips"] == 1
    assert latest_rejection(strategy).reason == "duplicate_cluster_signature"


@pytest.mark.asyncio
async def test_symbol_cooldown_rejects_second_signal_even_with_new_cluster(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        symbol_cooldown_seconds=60,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
        deduplicate_by_detected_at=False,
        deduplicate_same_cluster_signature=False,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    first = make_result(
        detected_at=utc_now() - timedelta(seconds=2),
        cluster=make_cluster(avg_price=Decimal("100000")),
    )
    second = make_result(
        detected_at=utc_now() - timedelta(seconds=1),
        cluster=make_cluster(avg_price=Decimal("101000")),
    )

    await strategy._on_bus_event(make_event(first))
    await strategy._on_bus_event(make_event(second))

    assert strategy.get_stats()["emitted_signals"] == 1
    assert strategy.get_stats()["rejected_events"] == 1
    assert strategy.get_stats()["cooldown_skips"] == 1
    assert latest_rejection(strategy).reason == "symbol_in_cooldown"


@pytest.mark.asyncio
async def test_same_side_signal_too_soon_rejects_second_same_direction_signal(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=60,
        max_signals_per_symbol_window=0,
        deduplicate_by_detected_at=False,
        deduplicate_same_cluster_signature=False,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    first = make_result(
        detected_at=utc_now() - timedelta(seconds=2),
        direction=CascadeDirection.UP,
        cluster=make_cluster(avg_price=Decimal("100000")),
    )
    second = make_result(
        detected_at=utc_now() - timedelta(seconds=1),
        direction=CascadeDirection.UP,
        cluster=make_cluster(avg_price=Decimal("101000")),
    )

    await strategy._on_bus_event(make_event(first))
    await strategy._on_bus_event(make_event(second))

    assert strategy.get_stats()["emitted_signals"] == 1
    assert strategy.get_stats()["rejected_events"] == 1
    assert strategy.get_stats()["duplicate_skips"] == 1
    assert latest_rejection(strategy).reason == "same_side_signal_too_soon"


@pytest.mark.asyncio
async def test_opposite_side_signal_bypasses_same_side_filter(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=60,
        max_signals_per_symbol_window=0,
        deduplicate_by_detected_at=False,
        deduplicate_same_cluster_signature=False,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    first = make_result(
        detected_at=utc_now() - timedelta(seconds=2),
        direction=CascadeDirection.UP,
        cluster=make_cluster(avg_price=Decimal("100000")),
    )
    second = make_result(
        detected_at=utc_now() - timedelta(seconds=1),
        direction=CascadeDirection.DOWN,
        cluster=make_cluster(avg_price=Decimal("101000")),
    )

    await strategy._on_bus_event(make_event(first))
    await strategy._on_bus_event(make_event(second))

    signals = emitted_payloads(event_bus, "signal.generated")

    assert strategy.get_stats()["emitted_signals"] == 2
    assert strategy.get_stats()["rejected_events"] == 0
    assert [signal.side for signal in signals] == ["LONG", "SHORT"]


@pytest.mark.asyncio
async def test_symbol_signal_rate_limit_rejects_third_signal_in_window(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=2,
        signal_window_seconds=90,
        deduplicate_by_detected_at=False,
        deduplicate_same_cluster_signature=False,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    first = make_result(
        detected_at=utc_now() - timedelta(seconds=3),
        cluster=make_cluster(avg_price=Decimal("100000")),
    )
    second = make_result(
        detected_at=utc_now() - timedelta(seconds=2),
        cluster=make_cluster(avg_price=Decimal("101000")),
    )
    third = make_result(
        detected_at=utc_now() - timedelta(seconds=1),
        cluster=make_cluster(avg_price=Decimal("102000")),
    )

    await strategy._on_bus_event(make_event(first))
    await strategy._on_bus_event(make_event(second))
    await strategy._on_bus_event(make_event(third))

    assert strategy.get_stats()["emitted_signals"] == 2
    assert strategy.get_stats()["rejected_events"] == 1
    assert strategy.get_stats()["rate_limit_skips"] == 1
    assert latest_rejection(strategy).reason == "symbol_signal_rate_limited"


@pytest.mark.asyncio
async def test_rate_limit_is_per_symbol_not_global(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=1,
        signal_window_seconds=90,
        deduplicate_by_detected_at=False,
        deduplicate_same_cluster_signature=False,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    btc = make_result(
        symbol="BTCUSDT",
        detected_at=utc_now() - timedelta(seconds=2),
        cluster=make_cluster(avg_price=Decimal("100000")),
    )
    eth = make_result(
        symbol="ETHUSDT",
        detected_at=utc_now() - timedelta(seconds=1),
        cluster=make_cluster(avg_price=Decimal("3000")),
    )

    await strategy._on_bus_event(make_event(btc))
    await strategy._on_bus_event(make_event(eth))

    assert strategy.get_stats()["emitted_signals"] == 2
    assert strategy.get_stats()["rejected_events"] == 0
    assert strategy.get_stats()["tracked_symbols"] == 2


# ============================================================================
# Hot symbols / state snapshots / diagnostics
# ============================================================================


@pytest.mark.asyncio
async def test_get_hot_symbols_returns_latest_per_symbol_sorted_by_score(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
        deduplicate_by_detected_at=False,
        deduplicate_same_cluster_signature=False,
        hot_symbols_window_seconds=300,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    weak_btc = make_result(
        symbol="BTCUSDT",
        confidence=0.65,
        intensity_score=0.60,
        continuation_bias=0.61,
        severity=CascadeSeverity.MEDIUM,
        detected_at=utc_now() - timedelta(seconds=4),
        cluster=make_cluster(avg_price=Decimal("100000")),
    )
    strong_eth = make_result(
        symbol="ETHUSDT",
        confidence=0.95,
        intensity_score=0.90,
        continuation_bias=0.92,
        severity=CascadeSeverity.EXTREME,
        detected_at=utc_now() - timedelta(seconds=3),
        cluster=make_cluster(avg_price=Decimal("3000")),
    )
    stronger_btc_later = make_result(
        symbol="BTCUSDT",
        confidence=0.85,
        intensity_score=0.82,
        continuation_bias=0.80,
        severity=CascadeSeverity.HIGH,
        detected_at=utc_now() - timedelta(seconds=2),
        cluster=make_cluster(avg_price=Decimal("101000")),
    )

    await strategy._on_bus_event(make_event(weak_btc))
    await strategy._on_bus_event(make_event(strong_eth))
    await strategy._on_bus_event(make_event(stronger_btc_later))

    rows = strategy.get_hot_symbols(limit=10)

    assert len(rows) == 2

    assert rows[0]["symbol"] == "ETHUSDT"
    assert rows[0]["score"] >= rows[1]["score"]
    assert rows[0]["continuation_bias"] == 0.92
    assert rows[0]["severity"] == CascadeSeverity.EXTREME.value

    btc_row = next(row for row in rows if row["symbol"] == "BTCUSDT")
    assert btc_row["confidence"] == 0.85
    assert btc_row["continuation_bias"] == 0.80
    assert btc_row["side"] == "LONG"


@pytest.mark.asyncio
async def test_get_hot_symbols_respects_limit_zero_and_positive_limit(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
        deduplicate_by_detected_at=False,
        deduplicate_same_cluster_signature=False,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    await strategy._on_bus_event(make_event(make_result(symbol="BTCUSDT")))
    await strategy._on_bus_event(make_event(make_result(symbol="ETHUSDT")))

    assert strategy.get_hot_symbols(limit=0) == []
    assert len(strategy.get_hot_symbols(limit=1)) == 1


@pytest.mark.asyncio
async def test_get_symbol_state_returns_non_existing_snapshot_before_any_signal(
    event_bus: FakeEventBus,
) -> None:
    strategy = make_strategy(event_bus=event_bus)

    snapshot = strategy.get_symbol_state("binance", "BTC/USDT")

    assert snapshot["exists"] is False
    assert snapshot["exchange"] == "binance"
    assert snapshot["symbol"] == "BTCUSDT"


@pytest.mark.asyncio
async def test_get_symbol_state_returns_runtime_state_after_signal(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        symbol_cooldown_seconds=20,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(exchange="binance", symbol="BTC/USDT")

    await strategy._on_bus_event(make_event(result))

    snapshot = strategy.get_symbol_state("BINANCE", "BTCUSDT")

    assert snapshot["exists"] is True
    assert snapshot["exchange"] == "binance"
    assert snapshot["symbol"] == "BTCUSDT"
    assert snapshot["last_signal_side"] == "LONG"
    assert snapshot["last_signal_score"] is not None
    assert snapshot["total_signals_emitted"] == 1
    assert snapshot["signals_in_window"] == 1
    assert snapshot["is_in_cooldown"] is True
    assert snapshot["cooldown_until"] is not None
    assert snapshot["last_cluster_signature"] is not None


@pytest.mark.asyncio
async def test_publish_diagnostics_snapshot_contains_cascade_hot_symbols(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        publish_diagnostics_snapshots=True,
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
        deduplicate_by_detected_at=False,
        deduplicate_same_cluster_signature=False,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    await strategy._on_bus_event(make_event(make_result(symbol="BTCUSDT")))

    event_bus.emitted.clear()

    await strategy.publish_diagnostics_snapshot()

    assert emitted_topics(event_bus) == ["strategy.liquidations.cascade.snapshot"]

    snapshot = event_bus.emitted[0]["payload"]

    assert snapshot["strategy_name"] == "liquidation_cascade_strategy"
    assert snapshot["signal_type"] == "continuation"
    assert snapshot["stats"]["emitted_signals"] == 1
    assert len(snapshot["hot_symbols"]) == 1
    assert snapshot["hot_symbols"][0]["symbol"] == "BTCUSDT"
    assert "continuation_bias" in snapshot["hot_symbols"][0]

    headers = event_bus.emitted[0]["headers"]

    assert headers["strategy"] == "liquidation_cascade_strategy"
    assert headers["signal_type"] == "continuation"
    assert headers["event_type"] == "strategy_diagnostics"


# ============================================================================
# Payload validation / robustness
# ============================================================================


@pytest.mark.asyncio
async def test_invalid_payload_type_is_ignored_and_does_not_emit_signal(
    event_bus: FakeEventBus,
) -> None:
    strategy = make_strategy(event_bus=event_bus)
    await strategy.start()

    event = make_event({"payload": "not-a-cascade-result"})

    await strategy._on_bus_event(event)

    assert strategy.get_stats()["processed_events"] == 0
    assert strategy.get_stats()["invalid_payload_skips"] == 1
    assert event_bus.emitted == []


@pytest.mark.asyncio
async def test_processing_error_is_recorded_when_result_is_missing_required_field(
    event_bus: FakeEventBus,
) -> None:
    strategy = make_strategy(event_bus=event_bus)
    await strategy.start()

    result = make_result()
    delattr(result, "side")

    await strategy._on_bus_event(make_event(result))

    stats = strategy.get_stats()

    assert stats["processed_events"] == 1
    assert stats["emitted_signals"] == 0
    assert stats["last_error"] is not None
    assert "side" in stats["last_error"]


# ============================================================================
# Known vulnerability detector
# ============================================================================


@pytest.mark.xfail(
    reason=(
        "Known vulnerability: BaseAnalyticsStrategy rejects direction only when "
        "direction.value == 'unknown'. LiquidationCascadeStrategy.direction_to_trade_side() "
        "returns 'FLAT' for unsupported non-unknown directions, so a FLAT signal can be emitted. "
        "Fix recommendation: reject trade_side == 'FLAT' in common filters or subclass filters."
    ),
    strict=True,
)
@pytest.mark.asyncio
async def test_unsupported_non_unknown_direction_must_not_emit_flat_signal(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        allowed_severities=(),
        require_favors_continuation=False,
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(
        direction=FakeDirection("sideways"),
        severity=FakeValue("weird"),  # type: ignore[arg-type]
        continuation_bias=0.90,
        favors_continuation=True,
    )

    await strategy._on_bus_event(make_event(result))

    assert strategy.get_stats()["emitted_signals"] == 0
    assert strategy.get_stats()["rejected_events"] == 1
    assert latest_rejection(strategy).reason == "unsupported_trade_side"


# ============================================================================
# Direct method-level checks for fragile hooks
# ============================================================================


def test_direction_to_trade_side_maps_only_up_and_down(
    event_bus: FakeEventBus,
) -> None:
    strategy = make_strategy(event_bus=event_bus)

    assert strategy.direction_to_trade_side(
        make_result(direction=CascadeDirection.UP)
    ) == "LONG"
    assert strategy.direction_to_trade_side(
        make_result(direction=CascadeDirection.DOWN)
    ) == "SHORT"
    assert strategy.direction_to_trade_side(
        make_result(direction=FakeDirection("sideways"))
    ) == "FLAT"


def test_build_signal_does_not_mutate_source_result(
    event_bus: FakeEventBus,
) -> None:
    strategy = make_strategy(event_bus=event_bus)

    result = make_result(
        confidence=0.84,
        intensity_score=0.79,
        continuation_bias=0.73,
        exhaustion_bias=0.14,
    )
    event = make_event(result)

    before = {
        "confidence": result.confidence,
        "intensity_score": result.intensity_score,
        "continuation_bias": result.continuation_bias,
        "exhaustion_bias": result.exhaustion_bias,
        "metadata": dict(result.metadata),
    }

    signal = strategy.build_signal(result=result, bus_event=event)

    assert isinstance(signal, LiquidationCascadeSignal)

    assert result.confidence == before["confidence"]
    assert result.intensity_score == before["intensity_score"]
    assert result.continuation_bias == before["continuation_bias"]
    assert result.exhaustion_bias == before["exhaustion_bias"]
    assert result.metadata == before["metadata"]


def test_signal_to_dict_serializes_decimal_and_datetime_values(
    event_bus: FakeEventBus,
) -> None:
    strategy = make_strategy(event_bus=event_bus)

    result = make_result(total_notional_usd=Decimal("1234567.89"))
    event = make_event(result)

    signal = strategy.build_signal(result=result, bus_event=event)
    serialized = signal.to_dict(serialize=True)

    assert serialized["total_notional_usd"] == "1234567.89"
    assert isinstance(serialized["generated_at"], str)
    assert isinstance(serialized["detected_at"], str)
    assert serialized["strategy_name"] == "liquidation_cascade_strategy"
    assert serialized["signal_type"] == "continuation"


def test_evaluate_filters_returns_cluster_signature_only_on_acceptance(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)

    accepted = make_result()
    accepted_state = strategy.get_or_create_state(accepted.exchange, accepted.symbol)
    accepted_result = strategy.evaluate_filters(
        result=accepted,
        state=accepted_state,
        now=utc_now(),
    )

    assert accepted_result.rejection_reason is None
    assert accepted_result.cluster_signature is not None

    rejected = make_result(continuation_bias=0.10)
    rejected_state = strategy.get_or_create_state("bybit", "ETHUSDT")
    rejected_result = strategy.evaluate_filters(
        result=rejected,
        state=rejected_state,
        now=utc_now(),
    )

    assert rejected_result.rejection_reason == "continuation_bias_below_threshold"
    assert rejected_result.cluster_signature is None