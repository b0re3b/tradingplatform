# tests/strategy/strategies/liquidations/test_liquidation_cascade_strategy_behavior.py

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
    total_notional_usd: Decimal = Decimal("750000"),
    total_quantity: Decimal = Decimal("12"),
    avg_price: Decimal = Decimal("65000"),
    min_price: Decimal = Decimal("64500"),
    max_price: Decimal = Decimal("65500"),
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
    direction: CascadeDirection = CascadeDirection.UP,
    severity: CascadeSeverity = CascadeSeverity.HIGH,
    status: LiquidationStatus = LiquidationStatus.CONFIRMED,
    detected_at: datetime | None = None,
    confidence: float = 0.82,
    intensity_score: float = 0.77,
    continuation_bias: float = 0.72,
    exhaustion_bias: float = 0.18,
    event_count: int = 12,
    total_notional_usd: Decimal = Decimal("750000"),
    window_seconds: int = 10,
    price_range_pct: float = 0.35,
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
        metadata=metadata
        or {
            "detector": "cascade_detector",
            "side_imbalance_ratio": 0.82,
            "event_imbalance_ratio": 0.76,
            "acceleration_ratio": 1.35,
        },
    )


def make_event(
    payload: Any,
    *,
    topic: str = "analytics.liquidations.cascade_detected",
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
    return LiquidationCascadeStrategy(
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
# Lifecycle / subscription contract
# ============================================================================


@pytest.mark.asyncio
async def test_start_subscribes_to_plural_liquidations_cascade_topic(
    event_bus: FakeEventBus,
) -> None:
    strategy = make_strategy(event_bus=event_bus)

    await strategy.start()

    assert strategy.is_running is True
    assert strategy.is_registered is True
    assert strategy.get_stats()["running"] is True
    assert len(event_bus.subscriptions) == 1
    assert event_bus.subscriptions[0].topic == "analytics.liquidations.cascade_detected"
    assert event_bus.subscriptions[0].name == (
        "liquidation_cascade_strategy.on_analytics_event"
    )


@pytest.mark.asyncio
async def test_start_does_not_subscribe_when_strategy_disabled(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(enabled=False)
    strategy = make_strategy(event_bus=event_bus, config=config)

    await strategy.start()

    assert strategy.is_running is False
    assert strategy.is_registered is False
    assert event_bus.subscriptions == []


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


@pytest.mark.asyncio
async def test_stop_unsubscribes_and_removes_diagnostics_job(
    event_bus: FakeEventBus,
) -> None:
    scheduler = FakeScheduler()
    config = LiquidationCascadeStrategyConfig(
        publish_diagnostics_snapshots=True,
    )
    strategy = make_strategy(
        event_bus=event_bus,
        config=config,
        scheduler=scheduler,
    )

    await strategy.start()
    subscription = event_bus.subscriptions[0]
    job_id = next(iter(scheduler.jobs.keys()))

    await strategy.stop()

    assert strategy.is_running is False
    assert strategy.is_registered is False
    assert event_bus.unsubscribed == [subscription]
    assert job_id in scheduler.removed_job_ids


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

    result = make_result(
        direction=CascadeDirection.DOWN,
        side=LiquidationSide.SHORT,
    )
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
async def test_valid_result_emits_complete_full_scope_signal_contract_and_updates_state(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        symbol_cooldown_seconds=15,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
        min_side_imbalance_ratio=0.70,
        min_event_imbalance_ratio=0.70,
        min_acceleration_ratio=1.10,
    )
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
        confidence=0.91,
        intensity_score=0.88,
        continuation_bias=0.83,
        exhaustion_bias=0.11,
        event_count=25,
        total_notional_usd=Decimal("1750000"),
        price_range_pct=0.42,
        severity=CascadeSeverity.EXTREME,
        metadata={
            "side_imbalance_ratio": 0.88,
            "event_imbalance_ratio": 0.83,
            "acceleration_ratio": 1.60,
        },
    )
    event = make_event(result, correlation_id="contract-corr")

    await strategy._on_bus_event(event)

    signal = latest_signal(event_bus)

    assert signal.strategy_name == "liquidation_cascade_strategy"
    assert signal.signal_type == "continuation"
    assert signal.exchange == "bybit"
    assert signal.market_type == "linear"
    assert signal.symbol == "ETHUSDT"
    assert signal.timeframe == "realtime"
    assert signal.exchange_symbol == "ETHUSDT"
    assert signal.scope_key == "bybit:linear:ETHUSDT:realtime"
    assert signal.side == "SHORT"
    assert signal.confidence == 0.91
    assert 0.0 <= signal.score <= 1.0
    assert signal.detected_at == result.detected_at
    assert signal.source_topic == event.topic
    assert signal.severity == CascadeSeverity.EXTREME.value
    assert signal.cascade_direction == CascadeDirection.DOWN.value
    assert signal.liquidation_side == LiquidationSide.LONG.value
    assert signal.event_type == result.event_type.value
    assert signal.status == result.status.value
    assert signal.event_count == 25
    assert signal.total_notional_usd == Decimal("1750000")
    assert signal.intensity_score == 0.88
    assert signal.continuation_bias == 0.83
    assert signal.exhaustion_bias == 0.11
    assert signal.bias_delta == result.bias_delta
    assert signal.price_range_pct == 0.42
    assert signal.side_imbalance_ratio == 0.88
    assert signal.event_imbalance_ratio == 0.83
    assert signal.acceleration_ratio == 1.60
    assert signal.correlation_id == "contract-corr"
    assert signal.source_event_id == event.event_id

    assert "liquidation cascade continuation" in signal.reason
    assert "scope=bybit:linear:ETHUSDT:realtime" in signal.reason
    assert "direction=down" in signal.reason
    assert "continuation_bias=0.830" in signal.reason
    assert "confidence=0.910" in signal.reason

    assert "scope" in signal.metadata
    assert "strategy" in signal.metadata
    assert "bus_event" in signal.metadata
    assert "analytics" in signal.metadata
    assert "cluster" in signal.metadata
    assert "liquidation_cascade_strategy" in signal.metadata

    cascade_meta = signal.metadata["liquidation_cascade_strategy"]
    assert cascade_meta["min_continuation_bias"] == config.min_continuation_bias
    assert cascade_meta["require_favors_continuation"] is True
    assert cascade_meta["min_side_imbalance_ratio"] == 0.70
    assert cascade_meta["min_event_imbalance_ratio"] == 0.70
    assert cascade_meta["min_acceleration_ratio"] == 1.10

    state = strategy.get_state(
        "bybit",
        "ETHUSDT",
        market_type="linear",
        timeframe="realtime",
    )

    assert isinstance(state, SymbolCascadeStrategyState)
    assert state.key == ("bybit", "linear", "ETHUSDT", "realtime")
    assert state.last_signal_side == "SHORT"
    assert state.last_detected_at == result.detected_at
    assert state.last_cluster_signature is not None
    assert state.total_signals_emitted == 1
    assert state.cooldown_until is not None

    stats = strategy.get_stats()

    assert stats["processed_events"] == 1
    assert stats["emitted_signals"] == 1
    assert stats["rejected_events"] == 0
    assert stats["tracked_scopes"] == 1
    assert stats["tracked_symbols"] == 1
    assert stats["recent_signals"] == 1


@pytest.mark.asyncio
async def test_signal_generated_event_has_full_scope_headers_and_no_direct_risk_or_execution(
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

    assert emitted["source"] == "liquidation_cascade_strategy"
    assert emitted["priority"] is EventPriority.HIGH
    assert emitted["correlation_id"] == "signal-corr"

    headers = emitted["headers"]

    assert headers["strategy"] == "liquidation_cascade_strategy"
    assert headers["signal_type"] == "continuation"
    assert headers["source_event_id"] == event.event_id
    assert headers["source_topic"] == event.topic
    assert headers["side"] == "LONG"
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
# Domain-specific liquidation filters
# ============================================================================


@pytest.mark.parametrize(
    ("config_overrides", "result_overrides", "expected_reason"),
    [
        (
            {"require_favors_continuation": True},
            {
                "continuation_bias": 0.30,
                "exhaustion_bias": 0.80,
            },
            "continuation_not_favored",
        ),
        (
            {"min_continuation_bias": 0.80},
            {"continuation_bias": 0.79},
            "continuation_bias_below_threshold",
        ),
        (
            {"max_exhaustion_bias_for_continuation": 0.20},
            {"exhaustion_bias": 0.21},
            "exhaustion_bias_too_high_for_continuation",
        ),
        (
            {"min_bias_delta": 0.20},
            {
                "continuation_bias": 0.61,
                "exhaustion_bias": 0.50,
            },
            "bias_delta_below_threshold",
        ),
        (
            {"max_future_detected_at_seconds": 1.0},
            {"detected_at": utc_now() + timedelta(seconds=30)},
            "detected_at_in_future",
        ),
        (
            {"max_result_age_seconds": 1.0},
            {"detected_at": utc_now() - timedelta(seconds=30)},
            "result_too_old",
        ),
        (
            {"min_side_imbalance_ratio": 0.90},
            {"metadata": {"side_imbalance_ratio": 0.89}},
            "side_imbalance_below_threshold",
        ),
        (
            {"min_event_imbalance_ratio": 0.90},
            {"metadata": {"event_imbalance_ratio": 0.89}},
            "event_imbalance_below_threshold",
        ),
        (
            {"min_acceleration_ratio": 1.50},
            {"metadata": {"acceleration_ratio": 1.49}},
            "acceleration_below_threshold",
        ),
        (
            {"max_cluster_duration_seconds": 2.0},
            {"cluster": make_cluster(start_time=utc_now() - timedelta(seconds=10), end_time=utc_now())},
            "cluster_duration_too_long",
        ),
        (
            {"min_avg_notional_per_event": Decimal("100000")},
            {
                "cluster": make_cluster(
                    event_count=10,
                    total_notional_usd=Decimal("500000"),
                )
            },
            "avg_notional_per_event_below_threshold",
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
    assert rejection.scope_key == scoped_key_to_string(result.key)
    assert rejection.details["continuation_bias"] == result.continuation_bias
    assert rejection.details["scope_key"] == scoped_key_to_string(result.key)


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
        exhaustion_bias=0.90,
    )

    await strategy._on_bus_event(make_event(result))

    rejection = latest_rejection(strategy)

    assert rejection.reason == "confidence_below_threshold"
    assert rejection.reason != "continuation_not_favored"
    assert rejection.reason != "continuation_bias_below_threshold"


@pytest.mark.asyncio
async def test_allowed_market_type_filter_rejects_other_futures_scope(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        allowed_market_types=("usdm_futures",),
        publish_rejected_events=False,
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
    config = LiquidationCascadeStrategyConfig(
        allowed_timeframes=("realtime",),
        publish_rejected_events=False,
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
        market_type="swap",
        symbol="SOL-USDT",
        timeframe="1m",
        exchange_symbol="SOL-USDT-SWAP",
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
    assert rejection.market_type == "swap"
    assert rejection.symbol == "SOLUSDT"
    assert rejection.timeframe == "1m"
    assert rejection.exchange_symbol == "SOL-USDT-SWAP"
    assert rejection.scope_key == "okx:swap:SOLUSDT:1m"
    assert rejection.correlation_id == "reject-corr"
    assert rejection.source_event_id == event.event_id

    assert emitted["priority"] is EventPriority.LOW
    assert emitted["source"] == "liquidation_cascade_strategy"
    assert emitted["correlation_id"] == "reject-corr"

    headers = emitted["headers"]

    assert headers["strategy"] == "liquidation_cascade_strategy"
    assert headers["signal_type"] == "continuation"
    assert headers["reason"] == "continuation_bias_below_threshold"
    assert headers["source_event_id"] == event.event_id
    assert headers["source_topic"] == "analytics.liquidations.cascade_detected"

    assert_headers_have_full_scope(
        headers,
        exchange="okx",
        market_type="swap",
        symbol="SOLUSDT",
        timeframe="1m",
        exchange_symbol="SOL-USDT-SWAP",
    )


# ============================================================================
# Emit failure / EventBus resilience
# ============================================================================


@pytest.mark.asyncio
async def test_emit_false_does_not_update_state_or_stats_as_emitted() -> None:
    event_bus = RejectingEventBus()
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
    state = strategy.get_or_create_state_for_result(result)

    assert stats["processed_events"] == 1
    assert stats["emitted_signals"] == 0
    assert stats["rejected_events"] == 0
    assert state.total_signals_emitted == 0
    assert state.last_signal_at is None
    assert state.last_detected_at is None


@pytest.mark.asyncio
async def test_emit_exception_is_recorded_without_marking_signal_as_emitted() -> None:
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
    state = strategy.get_or_create_state_for_result(result)

    assert stats["processed_events"] == 1
    assert stats["emitted_signals"] == 0
    assert stats["last_error"] is not None
    assert "event bus emit failed" in stats["last_error"]
    assert state.total_signals_emitted == 0
    assert state.last_signal_at is None


# ============================================================================
# Full-scope state isolation
# ============================================================================


@pytest.mark.asyncio
async def test_usdm_and_coinm_same_symbol_do_not_share_cooldown(
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

    await strategy._on_bus_event(make_event(usdm, correlation_id="usdm"))
    await strategy._on_bus_event(make_event(coinm, correlation_id="coinm"))

    assert strategy.get_stats()["emitted_signals"] == 2
    assert strategy.get_stats()["cooldown_skips"] == 0
    assert strategy.get_stats()["tracked_scopes"] == 2
    assert strategy.get_stats()["tracked_symbols"] == 1

    usdm_state = strategy.get_state(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )
    coinm_state = strategy.get_state(
        "binance",
        "BTCUSDT",
        market_type="coinm_futures",
        timeframe="realtime",
    )

    assert usdm_state is not None
    assert coinm_state is not None
    assert usdm_state is not coinm_state
    assert usdm_state.cooldown_until is not None
    assert coinm_state.cooldown_until is not None


@pytest.mark.asyncio
async def test_realtime_and_1m_same_symbol_do_not_share_cooldown(
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

    realtime = make_result(
        timeframe="realtime",
        detected_at=utc_now() - timedelta(seconds=3),
    )
    one_minute = make_result(
        timeframe="1m",
        detected_at=utc_now() - timedelta(seconds=2),
    )

    await strategy._on_bus_event(make_event(realtime, correlation_id="realtime"))
    await strategy._on_bus_event(make_event(one_minute, correlation_id="1m"))

    assert strategy.get_stats()["emitted_signals"] == 2
    assert strategy.get_stats()["cooldown_skips"] == 0
    assert strategy.get_stats()["tracked_scopes"] == 2

    assert strategy.get_state(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    ) is not None
    assert strategy.get_state(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="1m",
    ) is not None


@pytest.mark.asyncio
async def test_same_scope_respects_cooldown(
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

    first = make_result(detected_at=utc_now() - timedelta(seconds=3))
    second = make_result(detected_at=utc_now() - timedelta(seconds=2))

    await strategy._on_bus_event(make_event(first))
    await strategy._on_bus_event(make_event(second))

    assert strategy.get_stats()["emitted_signals"] == 1
    assert strategy.get_stats()["rejected_events"] == 1
    assert strategy.get_stats()["cooldown_skips"] == 1

    rejection = latest_rejection(strategy)
    assert rejection.reason == "scope_in_cooldown"
    assert rejection.scope_key == "binance:usdm_futures:BTCUSDT:realtime"


# ============================================================================
# Dedup / rate-limit scope isolation
# ============================================================================


@pytest.mark.asyncio
async def test_duplicate_detected_at_is_scoped_by_market_type(
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

    await strategy._on_bus_event(make_event(usdm, correlation_id="usdm"))
    await strategy._on_bus_event(make_event(coinm, correlation_id="coinm"))
    await strategy._on_bus_event(make_event(duplicate_usdm, correlation_id="dup-usdm"))

    assert strategy.get_stats()["emitted_signals"] == 2
    assert strategy.get_stats()["rejected_events"] == 1
    assert strategy.get_stats()["duplicate_skips"] == 1

    rejection = latest_rejection(strategy)
    assert rejection.reason == "duplicate_detected_at"
    assert rejection.market_type == "usdm_futures"


@pytest.mark.asyncio
async def test_cluster_signature_includes_market_type_and_timeframe(
    event_bus: FakeEventBus,
) -> None:
    strategy = make_strategy(event_bus=event_bus)

    usdm = make_result(
        market_type="usdm_futures",
        timeframe="realtime",
        exchange_symbol="BTCUSDT",
    )
    coinm = make_result(
        market_type="coinm_futures",
        timeframe="realtime",
        exchange_symbol="BTCUSD_PERP",
    )
    one_minute = make_result(
        market_type="usdm_futures",
        timeframe="1m",
        exchange_symbol="BTCUSDT",
    )

    usdm_signature = strategy.build_cluster_signature(usdm)
    coinm_signature = strategy.build_cluster_signature(coinm)
    one_minute_signature = strategy.build_cluster_signature(one_minute)

    assert usdm_signature != coinm_signature
    assert usdm_signature != one_minute_signature
    assert coinm_signature != one_minute_signature


@pytest.mark.asyncio
async def test_same_side_signal_too_soon_is_scoped(
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
    assert strategy.get_stats()["duplicate_skips"] == 1

    rejection = latest_rejection(strategy)
    assert rejection.reason == "same_side_signal_too_soon"
    assert rejection.market_type == "usdm_futures"


@pytest.mark.asyncio
async def test_scope_signal_rate_limit_does_not_block_other_market_type(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
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
# Recent API / hot symbols
# ============================================================================


@pytest.mark.asyncio
async def test_get_recent_signals_filters_by_full_scope(
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

    all_signals = strategy.get_recent_signals(exchange="binance", symbol="BTCUSDT", limit=10)
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
async def test_get_recent_rejections_filters_by_full_scope(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        allowed_market_types=("usdm_futures",),
        publish_rejected_events=False,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    coinm = make_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
    )
    swap = make_result(
        exchange="okx",
        market_type="swap",
        symbol="BTC-USDT",
        exchange_symbol="BTC-USDT-SWAP",
    )

    await strategy._on_bus_event(make_event(coinm))
    await strategy._on_bus_event(make_event(swap))

    all_rejections = strategy.get_recent_rejections(limit=10)
    coinm_rejections = strategy.get_recent_rejections(
        exchange="binance",
        market_type="coinm_futures",
        symbol="BTCUSDT",
        timeframe="realtime",
        limit=10,
    )
    okx_swap_rejections = strategy.get_recent_rejections(
        exchange="okx",
        market_type="swap",
        symbol="BTCUSDT",
        timeframe="realtime",
        limit=10,
    )

    assert len(all_rejections) == 2
    assert len(coinm_rejections) == 1
    assert len(okx_swap_rejections) == 1
    assert coinm_rejections[0].scope_key == "binance:coinm_futures:BTCUSDT:realtime"
    assert okx_swap_rejections[0].scope_key == "okx:swap:BTCUSDT:realtime"


@pytest.mark.asyncio
async def test_get_hot_symbols_returns_latest_signal_per_scope_sorted_by_score(
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

    weaker_usdm = make_result(
        market_type="usdm_futures",
        symbol="BTCUSDT",
        confidence=0.70,
        intensity_score=0.70,
        continuation_bias=0.70,
        detected_at=utc_now() - timedelta(seconds=5),
    )
    stronger_usdm = make_result(
        market_type="usdm_futures",
        symbol="BTCUSDT",
        confidence=0.95,
        intensity_score=0.95,
        continuation_bias=0.90,
        detected_at=utc_now() - timedelta(seconds=4),
    )
    coinm = make_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
        symbol="BTCUSDT",
        confidence=0.90,
        intensity_score=0.85,
        continuation_bias=0.80,
        detected_at=utc_now() - timedelta(seconds=3),
    )

    await strategy._on_bus_event(make_event(weaker_usdm))
    await strategy._on_bus_event(make_event(stronger_usdm))
    await strategy._on_bus_event(make_event(coinm))

    hot = strategy.get_hot_symbols(limit=10)

    assert len(hot) == 2
    assert hot[0]["scope_key"] == "binance:usdm_futures:BTCUSDT:realtime"
    assert hot[1]["scope_key"] == "binance:coinm_futures:BTCUSDT:realtime"
    assert hot[0]["score"] >= hot[1]["score"]


@pytest.mark.asyncio
async def test_get_symbol_state_returns_full_scope_diagnostic(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        symbol_cooldown_seconds=30,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(
        exchange="binance",
        market_type="usdm_futures",
        symbol="BTCUSDT",
        timeframe="realtime",
    )

    await strategy._on_bus_event(make_event(result))

    diagnostic = strategy.get_symbol_state(
        "binance",
        "BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
    )

    assert diagnostic["exists"] is True
    assert diagnostic["exchange"] == "binance"
    assert diagnostic["market_type"] == "usdm_futures"
    assert diagnostic["symbol"] == "BTCUSDT"
    assert diagnostic["timeframe"] == "realtime"
    assert diagnostic["scope_key"] == "binance:usdm_futures:BTCUSDT:realtime"
    assert diagnostic["last_signal_side"] == "LONG"
    assert diagnostic["total_signals_emitted"] == 1
    assert diagnostic["in_cooldown"] is True


# ============================================================================
# Diagnostics snapshot
# ============================================================================


@pytest.mark.asyncio
async def test_publish_diagnostics_snapshot_emits_scope_aware_payload(
    event_bus: FakeEventBus,
) -> None:
    config = LiquidationCascadeStrategyConfig(
        publish_diagnostics_snapshots=True,
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
    )
    strategy = make_strategy(event_bus=event_bus, config=config)
    await strategy.start()

    await strategy._on_bus_event(make_event(make_result()))
    await strategy.publish_diagnostics_snapshot()

    assert emitted_topics(event_bus) == ["signal.generated", "strategy.liquidations.cascade.snapshot"]

    emitted = event_bus.emitted[-1]
    assert emitted["topic"] == "strategy.liquidations.cascade.snapshot"
    assert emitted["priority"] is EventPriority.LOW
    assert emitted["source"] == "liquidation_cascade_strategy"
    assert emitted["headers"]["event_type"] == "strategy_diagnostics"

    payload = emitted["payload"]
    assert payload["strategy_name"] == "liquidation_cascade_strategy"
    assert payload["signal_type"] == "continuation"
    assert payload["scope"] == "exchange:market_type:symbol:timeframe"
    assert payload["stats"]["tracked_scopes"] == 1
    assert payload["hot_symbols"][0]["scope_key"] == "binance:usdm_futures:BTCUSDT:realtime"


# ============================================================================
# Model / serialization
# ============================================================================


def test_liquidation_cascade_signal_to_dict_contains_full_scope() -> None:
    signal = LiquidationCascadeSignal(
        strategy_name="liquidation_cascade_strategy",
        signal_type="continuation",
        exchange="BINANCE",
        market_type="USDM_FUTURES",
        symbol="btc-usdt",
        timeframe="Realtime",
        exchange_symbol="BTCUSDT",
        side="long",
        confidence=1.2,
        score=1.1,
        generated_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        detected_at=datetime(2026, 1, 1, 11, 59, 59, tzinfo=timezone.utc),
        reason="test",
        source_topic="analytics.liquidations.cascade_detected",
        severity="high",
        cascade_direction="up",
        liquidation_side="short",
        event_count=10,
        total_notional_usd=Decimal("750000"),
        intensity_score=1.2,
        continuation_bias=1.5,
        exhaustion_bias=-0.2,
        price_range_pct=0.35,
    )

    data = signal.to_dict()

    assert signal.exchange == "binance"
    assert signal.market_type == "usdm_futures"
    assert signal.symbol == "BTCUSDT"
    assert signal.timeframe == "realtime"
    assert signal.exchange_symbol == "BTCUSDT"
    assert signal.scope_key == "binance:usdm_futures:BTCUSDT:realtime"

    assert signal.confidence == 1.0
    assert signal.score == 1.0
    assert signal.intensity_score == 1.0
    assert signal.continuation_bias == 1.0
    assert signal.exhaustion_bias == 0.0

    assert data["scope"] == {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "realtime",
        "exchange_symbol": "BTCUSDT",
    }
    assert data["scope_key"] == "binance:usdm_futures:BTCUSDT:realtime"
    assert data["total_notional_usd"] == "750000"


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
        ({"min_confidence": -0.01}, "min_confidence must be between 0 and 1"),
        ({"min_intensity_score": 1.01}, "min_intensity_score must be between 0 and 1"),
        ({"min_total_notional_usd": Decimal("-1")}, "min_total_notional_usd must be >= 0"),
        ({"min_event_count": -1}, "min_event_count must be >= 0"),
        ({"max_price_range_pct": -0.1}, "max_price_range_pct must be >= 0 or None"),
        ({"max_future_detected_at_seconds": -1.0}, "max_future_detected_at_seconds must be >= 0"),
        ({"max_result_age_seconds": 0.0}, "max_result_age_seconds must be > 0 or None"),
        ({"min_side_imbalance_ratio": 1.01}, "min_side_imbalance_ratio must be between 0 and 1 or None"),
        ({"min_event_imbalance_ratio": -0.01}, "min_event_imbalance_ratio must be between 0 and 1 or None"),
        ({"min_acceleration_ratio": -0.1}, "min_acceleration_ratio must be >= 0 or None"),
        ({"max_cluster_duration_seconds": 0.0}, "max_cluster_duration_seconds must be > 0 or None"),
        ({"min_avg_notional_per_event": Decimal("-1")}, "min_avg_notional_per_event must be >= 0 or None"),
        ({"symbol_cooldown_seconds": -1}, "symbol_cooldown_seconds must be >= 0"),
        ({"signal_window_seconds": 0}, "signal_window_seconds must be > 0"),
        ({"diagnostics_interval_seconds": 0}, "diagnostics_interval_seconds must be > 0"),
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
                "score_continuation_bias_weight": 0.0,
                "score_intensity_weight": 0.0,
                "score_severity_weight": 0.0,
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
    config = LiquidationCascadeStrategyConfig(**overrides)

    with pytest.raises(ValueError, match=expected_message):
        config.validate()