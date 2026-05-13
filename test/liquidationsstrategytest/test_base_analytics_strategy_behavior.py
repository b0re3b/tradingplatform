from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

import pytest

from core.event_bus import Event, EventPriority

from strategy.strategies.liquidations.base import (
    BaseAnalyticsStrategy,
    BaseSymbolStrategyState,
    StrategyRejection,
    clamp_float,
    ensure_utc,
    normalize_symbol,
    utc_now,
)


# ============================================================================
# Test doubles
# ============================================================================


class DummyDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    UNKNOWN = "unknown"


class DummySeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass(slots=True)
class DummyCluster:
    duration_seconds: float = 5.0
    event_count: int = 10
    total_notional_usd: Decimal = Decimal("500000")
    avg_notional_per_event: Decimal = Decimal("50000")
    price_range_pct: float = 0.25


@dataclass(slots=True)
class DummyResult:
    exchange: str = "binance"
    symbol: str = "BTCUSDT"
    detected_at: datetime = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    confidence: float = 0.80
    intensity_score: float = 0.75

    event_count: int = 10
    total_notional_usd: Decimal = Decimal("500000")
    price_range_pct: float = 0.25

    severity: DummySeverity = DummySeverity.HIGH
    direction: DummyDirection = DummyDirection.UP

    continuation_bias: float = 0.70
    exhaustion_bias: float = 0.20

    correlation_id: str | None = "corr-1"
    metadata: dict[str, Any] | None = None
    cluster: DummyCluster | None = None

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.75

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}

        if self.cluster is None:
            self.cluster = DummyCluster()


@dataclass(slots=True)
class DummySignal:
    strategy_name: str
    signal_type: str
    exchange: str
    symbol: str
    side: str
    confidence: float
    score: float
    intensity_score: float
    severity: str
    generated_at: datetime
    detected_at: datetime
    total_notional_usd: Decimal
    source_event_id: str | None
    correlation_id: str | None


@dataclass(slots=True)
class DummyStrategyConfig:
    enabled: bool = True

    subscribe_topic: str = "analytics.dummy.detected"
    publish_topic_signal_generated: str = "signal.generated"
    publish_topic_signal_rejected: str = "signal.rejected"

    publish_rejected_events: bool = False
    publish_diagnostics_snapshots: bool = False

    diagnostics_topic: str = "strategy.dummy.snapshot"
    diagnostics_interval_seconds: float = 30.0

    strategy_name: str = "dummy_strategy"
    signal_type: str = "dummy"
    service_name: str = "dummy_strategy"

    signal_priority: EventPriority = EventPriority.HIGH
    rejection_priority: EventPriority = EventPriority.LOW
    diagnostics_priority: EventPriority = EventPriority.LOW

    allowed_exchanges: tuple[str, ...] = ()
    allowed_symbols: tuple[str, ...] = ()
    blocked_symbols: tuple[str, ...] = ()

    min_confidence: float = 0.60
    min_intensity_score: float = 0.50
    min_total_notional_usd: Decimal = Decimal("100000")
    min_event_count: int = 3
    max_price_range_pct: float | None = None

    allowed_severities: tuple[DummySeverity, ...] = (
        DummySeverity.MEDIUM,
        DummySeverity.HIGH,
        DummySeverity.EXTREME,
    )

    require_high_confidence_only: bool = False

    symbol_cooldown_seconds: int = 10
    min_seconds_between_same_side_signals: int = 5

    max_signals_per_symbol_window: int = 3
    signal_window_seconds: int = 60

    deduplicate_by_detected_at: bool = True
    deduplicate_same_cluster_signature: bool = True

    recent_signals_limit: int = 200
    recent_rejections_limit: int = 200

    def validate(self) -> None:
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be between 0 and 1")

        if not (0.0 <= self.min_intensity_score <= 1.0):
            raise ValueError("min_intensity_score must be between 0 and 1")

        if self.min_total_notional_usd < 0:
            raise ValueError("min_total_notional_usd must be >= 0")

        if self.min_event_count < 0:
            raise ValueError("min_event_count must be >= 0")

        if self.signal_window_seconds <= 0:
            raise ValueError("signal_window_seconds must be > 0")


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


class DummyAnalyticsStrategy(
    BaseAnalyticsStrategy[
        DummyResult,
        DummySignal,
        BaseSymbolStrategyState,
        DummyStrategyConfig,
    ]
):
    def __init__(
        self,
        *,
        event_bus: FakeEventBus,
        config: DummyStrategyConfig | None = None,
        scheduler: FakeScheduler | None = None,
    ) -> None:
        super().__init__(
            event_bus=event_bus,
            scheduler=scheduler,
            config=config or DummyStrategyConfig(),
            service_name=None,
            component="tests.dummy_strategy",
            payload_type=DummyResult,
        )

    def create_symbol_state(
        self,
        *,
        exchange: str,
        symbol: str,
    ) -> BaseSymbolStrategyState:
        return BaseSymbolStrategyState(
            exchange=exchange.lower(),
            symbol=normalize_symbol(symbol),
        )

    def direction_to_trade_side(self, result: DummyResult) -> str:
        if result.direction is DummyDirection.UP:
            return "LONG"

        if result.direction is DummyDirection.DOWN:
            return "SHORT"

        return "FLAT"

    async def process_result(
        self,
        result: DummyResult,
        *,
        bus_event: Event,
    ) -> None:
        state = self.get_or_create_state(result.exchange, result.symbol)
        now = utc_now()

        filter_result = self.evaluate_common_filters(
            result=result,
            state=state,
            now=now,
        )

        if filter_result.rejection_reason is not None:
            await self.reject_result(
                result=result,
                bus_event=bus_event,
                reason=filter_result.rejection_reason,
            )
            return

        signal = self.build_signal(result=result, bus_event=bus_event)

        emitted = await self.emit_signal(signal, bus_event=bus_event)
        if not emitted:
            return

        self.remember_emitted_signal(
            signal=signal,
            state=state,
            result=result,
            signal_side=signal.side,
            score=signal.score,
            cluster_signature=filter_result.cluster_signature,
        )

    def build_signal(
        self,
        *,
        result: DummyResult,
        bus_event: Event,
    ) -> DummySignal:
        return DummySignal(
            strategy_name=self.config.strategy_name,
            signal_type=self.config.signal_type,
            exchange=result.exchange,
            symbol=result.symbol,
            side=self.direction_to_trade_side(result),
            confidence=clamp_float(result.confidence),
            score=self.compute_strategy_score(result),
            intensity_score=clamp_float(result.intensity_score),
            severity=result.severity.value,
            generated_at=utc_now(),
            detected_at=ensure_utc(result.detected_at),
            total_notional_usd=result.total_notional_usd,
            source_event_id=bus_event.event_id,
            correlation_id=bus_event.correlation_id,
        )

    def compute_strategy_score(self, result: DummyResult) -> float:
        return clamp_float(
            (
                clamp_float(result.confidence)
                + clamp_float(result.intensity_score)
            )
            / 2.0
        )


# ============================================================================
# Fixtures / helpers
# ============================================================================


@pytest.fixture()
def event_bus() -> FakeEventBus:
    return FakeEventBus()


@pytest.fixture()
def strategy(event_bus: FakeEventBus) -> DummyAnalyticsStrategy:
    return DummyAnalyticsStrategy(event_bus=event_bus)


def make_result(**overrides: Any) -> DummyResult:
    base = {
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "detected_at": datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "confidence": 0.80,
        "intensity_score": 0.75,
        "event_count": 10,
        "total_notional_usd": Decimal("500000"),
        "price_range_pct": 0.25,
        "severity": DummySeverity.HIGH,
        "direction": DummyDirection.UP,
        "continuation_bias": 0.70,
        "exhaustion_bias": 0.20,
        "correlation_id": "corr-1",
        "metadata": {},
        "cluster": DummyCluster(),
    }
    base.update(overrides)
    return DummyResult(**base)


def make_event(
    payload: Any,
    *,
    topic: str = "analytics.dummy.detected",
    correlation_id: str | None = "corr-1",
) -> Event:
    return Event(
        topic=topic,
        payload=payload,
        priority=EventPriority.NORMAL,
        source="test.analytics",
        correlation_id=correlation_id,
    )


def emitted_topics(event_bus: FakeEventBus) -> list[str]:
    return [item["topic"] for item in event_bus.emitted]


def emitted_payloads(event_bus: FakeEventBus, topic: str) -> list[Any]:
    return [
        item["payload"]
        for item in event_bus.emitted
        if item["topic"] == topic
    ]


# ============================================================================
# Lifecycle
# ============================================================================


@pytest.mark.asyncio
async def test_start_subscribes_to_configured_topic_and_sets_running(
    event_bus: FakeEventBus,
) -> None:
    strategy = DummyAnalyticsStrategy(event_bus=event_bus)

    await strategy.start()

    assert strategy.get_stats()["running"] is True
    assert strategy.get_stats()["started_at"] is not None
    assert len(event_bus.subscriptions) == 1

    subscription = event_bus.subscriptions[0]

    assert subscription.topic == "analytics.dummy.detected"
    assert subscription.name == "dummy_strategy.on_analytics_event"
    assert callable(subscription.handler)


@pytest.mark.asyncio
async def test_start_does_not_subscribe_when_strategy_disabled(
    event_bus: FakeEventBus,
) -> None:
    config = DummyStrategyConfig(enabled=False)
    strategy = DummyAnalyticsStrategy(event_bus=event_bus, config=config)

    await strategy.start()

    assert strategy.get_stats()["running"] is False
    assert event_bus.subscriptions == []


@pytest.mark.asyncio
async def test_stop_unsubscribes_and_marks_strategy_stopped(
    event_bus: FakeEventBus,
) -> None:
    strategy = DummyAnalyticsStrategy(event_bus=event_bus)

    await strategy.start()
    await strategy.stop()

    assert strategy.get_stats()["running"] is False
    assert strategy.get_stats()["stopped_at"] is not None
    assert len(event_bus.unsubscribed) == 1
    assert event_bus.unsubscribed[0] is event_bus.subscriptions[0]


@pytest.mark.asyncio
async def test_restart_unsubscribes_old_subscription_and_registers_new_one(
    event_bus: FakeEventBus,
) -> None:
    strategy = DummyAnalyticsStrategy(event_bus=event_bus)

    await strategy.start()
    first_subscription = event_bus.subscriptions[0]

    await strategy.restart()

    assert first_subscription in event_bus.unsubscribed
    assert len(event_bus.subscriptions) == 2
    assert strategy.get_stats()["running"] is True


@pytest.mark.asyncio
async def test_start_registers_diagnostics_job_when_enabled(
    event_bus: FakeEventBus,
) -> None:
    scheduler = FakeScheduler()
    config = DummyStrategyConfig(publish_diagnostics_snapshots=True)
    strategy = DummyAnalyticsStrategy(
        event_bus=event_bus,
        config=config,
        scheduler=scheduler,
    )

    await strategy.start()

    assert strategy.get_stats()["diagnostics_job_registered"] is True
    assert len(scheduler.jobs) == 1

    job = next(iter(scheduler.jobs.values()))

    assert job["name"] == "dummy_strategy:diagnostics"
    assert job["interval"] == config.diagnostics_interval_seconds
    assert job["allow_overlap"] is False


@pytest.mark.asyncio
async def test_stop_removes_diagnostics_job_when_registered(
    event_bus: FakeEventBus,
) -> None:
    scheduler = FakeScheduler()
    config = DummyStrategyConfig(publish_diagnostics_snapshots=True)
    strategy = DummyAnalyticsStrategy(
        event_bus=event_bus,
        config=config,
        scheduler=scheduler,
    )

    await strategy.start()
    job_id = next(iter(scheduler.jobs.keys()))

    await strategy.stop()

    assert job_id in scheduler.removed_job_ids
    assert strategy.get_stats()["diagnostics_job_registered"] is False


# ============================================================================
# Event handling / payload validation
# ============================================================================


@pytest.mark.asyncio
async def test_on_bus_event_ignores_events_when_strategy_is_not_running(
    strategy: DummyAnalyticsStrategy,
    event_bus: FakeEventBus,
) -> None:
    event = make_event(make_result())

    await strategy._on_bus_event(event)

    assert strategy.get_stats()["processed_events"] == 0
    assert event_bus.emitted == []


@pytest.mark.asyncio
async def test_on_bus_event_ignores_invalid_payload_type(
    strategy: DummyAnalyticsStrategy,
    event_bus: FakeEventBus,
) -> None:
    await strategy.start()

    event = make_event({"not": "dummy-result"})

    await strategy._on_bus_event(event)

    stats = strategy.get_stats()

    assert stats["processed_events"] == 0
    assert stats["invalid_payload_skips"] == 1
    assert event_bus.emitted == []


@pytest.mark.asyncio
async def test_on_bus_event_records_processing_error_without_raising(
    event_bus: FakeEventBus,
) -> None:
    class ExplodingStrategy(DummyAnalyticsStrategy):
        async def process_result(
            self,
            result: DummyResult,
            *,
            bus_event: Event,
        ) -> None:
            raise RuntimeError("boom")

    strategy = ExplodingStrategy(event_bus=event_bus)
    await strategy.start()

    await strategy._on_bus_event(make_event(make_result()))

    stats = strategy.get_stats()

    assert stats["processed_events"] == 1
    assert stats["last_error"] is not None
    assert "boom" in stats["last_error"]


# ============================================================================
# Successful signal path
# ============================================================================


@pytest.mark.asyncio
async def test_valid_result_emits_signal_generated_and_updates_state(
    strategy: DummyAnalyticsStrategy,
    event_bus: FakeEventBus,
) -> None:
    await strategy.start()

    result = make_result()
    event = make_event(result)

    await strategy._on_bus_event(event)

    assert emitted_topics(event_bus) == ["signal.generated"]

    signal = emitted_payloads(event_bus, "signal.generated")[0]

    assert isinstance(signal, DummySignal)
    assert signal.strategy_name == "dummy_strategy"
    assert signal.signal_type == "dummy"
    assert signal.exchange == result.exchange
    assert signal.symbol == result.symbol
    assert signal.side == "LONG"
    assert signal.source_event_id == event.event_id
    assert signal.correlation_id == event.correlation_id

    state = strategy.get_or_create_state(result.exchange, result.symbol)

    assert state.last_signal_at is not None
    assert state.cooldown_until is not None
    assert state.last_signal_side == "LONG"
    assert state.last_detected_at == result.detected_at
    assert state.last_cluster_signature is not None
    assert state.total_signals_emitted == 1
    assert len(state.signal_timestamps) == 1

    stats = strategy.get_stats()

    assert stats["processed_events"] == 1
    assert stats["emitted_signals"] == 1
    assert stats["rejected_events"] == 0
    assert stats["recent_signals"] == 1
    assert stats["tracked_symbols"] == 1


@pytest.mark.asyncio
async def test_emit_signal_includes_expected_headers_and_correlation_id(
    strategy: DummyAnalyticsStrategy,
    event_bus: FakeEventBus,
) -> None:
    await strategy.start()

    result = make_result(exchange="bybit", symbol="ETHUSDT")
    event = make_event(result, correlation_id="corr-xyz")

    await strategy._on_bus_event(event)

    emitted = event_bus.emitted[0]

    assert emitted["topic"] == "signal.generated"
    assert emitted["priority"] is EventPriority.HIGH
    assert emitted["source"] == "dummy_strategy"
    assert emitted["correlation_id"] == "corr-xyz"

    headers = emitted["headers"]

    assert headers["strategy"] == "dummy_strategy"
    assert headers["signal_type"] == "dummy"
    assert headers["source_event_id"] == event.event_id
    assert headers["source_topic"] == event.topic
    assert headers["exchange"] == "bybit"
    assert headers["symbol"] == "ETHUSDT"
    assert headers["side"] == "LONG"


@pytest.mark.asyncio
async def test_emit_failure_does_not_mark_signal_as_emitted() -> None:
    event_bus = FailingEventBus()
    strategy = DummyAnalyticsStrategy(event_bus=event_bus)
    await strategy.start()

    result = make_result()
    event = make_event(result)

    await strategy._on_bus_event(event)

    stats = strategy.get_stats()
    state = strategy.get_or_create_state(result.exchange, result.symbol)

    assert stats["processed_events"] == 1
    assert stats["emitted_signals"] == 0
    assert stats["last_error"] is not None
    assert state.total_signals_emitted == 0
    assert state.last_signal_at is None
    assert state.last_detected_at is None


# ============================================================================
# Common filter rejection matrix
# ============================================================================


@pytest.mark.parametrize(
    ("config_overrides", "result_overrides", "expected_reason"),
    [
        (
            {"allowed_exchanges": ("bybit",)},
            {"exchange": "binance"},
            "exchange_not_allowed",
        ),
        (
            {"allowed_symbols": ("ETHUSDT",)},
            {"symbol": "BTCUSDT"},
            "symbol_not_allowed",
        ),
        (
            {"blocked_symbols": ("BTCUSDT",)},
            {"symbol": "BTCUSDT"},
            "symbol_blocked",
        ),
        (
            {},
            {"direction": DummyDirection.UNKNOWN},
            "unknown_direction",
        ),
        (
            {"require_high_confidence_only": True},
            {"confidence": 0.70},
            "not_high_confidence",
        ),
        (
            {"min_confidence": 0.90},
            {"confidence": 0.80},
            "confidence_below_threshold",
        ),
        (
            {"min_intensity_score": 0.90},
            {"intensity_score": 0.75},
            "intensity_below_threshold",
        ),
        (
            {"min_total_notional_usd": Decimal("1000000")},
            {"total_notional_usd": Decimal("500000")},
            "notional_below_threshold",
        ),
        (
            {"min_event_count": 20},
            {"event_count": 10},
            "event_count_below_threshold",
        ),
        (
            {"allowed_severities": (DummySeverity.EXTREME,)},
            {"severity": DummySeverity.HIGH},
            "severity_not_allowed",
        ),
        (
            {"max_price_range_pct": 0.10},
            {"price_range_pct": 0.25},
            "price_range_above_threshold",
        ),
    ],
)
@pytest.mark.asyncio
async def test_common_filter_rejections(
    event_bus: FakeEventBus,
    config_overrides: dict[str, Any],
    result_overrides: dict[str, Any],
    expected_reason: str,
) -> None:
    config = DummyStrategyConfig(**config_overrides)
    strategy = DummyAnalyticsStrategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(**result_overrides)

    await strategy._on_bus_event(make_event(result))

    stats = strategy.get_stats()

    assert stats["emitted_signals"] == 0
    assert stats["rejected_events"] == 1
    assert stats["recent_rejections"] == 1
    assert event_bus.emitted == []

    rejection = strategy.get_recent_rejections(limit=1)[0]

    assert rejection.reason == expected_reason
    assert rejection.exchange == result.exchange
    assert rejection.symbol == result.symbol


@pytest.mark.asyncio
async def test_rejected_event_is_published_when_enabled(
    event_bus: FakeEventBus,
) -> None:
    config = DummyStrategyConfig(
        publish_rejected_events=True,
        min_confidence=0.90,
    )
    strategy = DummyAnalyticsStrategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(confidence=0.80)

    await strategy._on_bus_event(make_event(result))

    assert emitted_topics(event_bus) == ["signal.rejected"]

    rejection = emitted_payloads(event_bus, "signal.rejected")[0]

    assert isinstance(rejection, StrategyRejection)
    assert rejection.reason == "confidence_below_threshold"
    assert rejection.strategy_name == "dummy_strategy"
    assert rejection.signal_type == "dummy"


# ============================================================================
# Cooldown / dedup / same-side / rate-limit
# ============================================================================


@pytest.mark.asyncio
async def test_second_signal_is_rejected_when_symbol_in_cooldown(
    event_bus: FakeEventBus,
) -> None:
    config = DummyStrategyConfig(
        symbol_cooldown_seconds=60,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
        deduplicate_by_detected_at=False,
        deduplicate_same_cluster_signature=False,
    )
    strategy = DummyAnalyticsStrategy(event_bus=event_bus, config=config)
    await strategy.start()

    first = make_result(
        detected_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    second = make_result(
        detected_at=datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
    )

    await strategy._on_bus_event(make_event(first))
    await strategy._on_bus_event(make_event(second))

    stats = strategy.get_stats()

    assert stats["emitted_signals"] == 1
    assert stats["rejected_events"] == 1
    assert stats["cooldown_skips"] == 1
    assert strategy.get_recent_rejections(limit=1)[0].reason == "symbol_in_cooldown"


@pytest.mark.asyncio
async def test_duplicate_detected_at_is_rejected(
    event_bus: FakeEventBus,
) -> None:
    config = DummyStrategyConfig(
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
        deduplicate_by_detected_at=True,
        deduplicate_same_cluster_signature=False,
    )
    strategy = DummyAnalyticsStrategy(event_bus=event_bus, config=config)
    await strategy.start()

    detected_at = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    await strategy._on_bus_event(make_event(make_result(detected_at=detected_at)))
    await strategy._on_bus_event(make_event(make_result(detected_at=detected_at)))

    stats = strategy.get_stats()

    assert stats["emitted_signals"] == 1
    assert stats["rejected_events"] == 1
    assert stats["duplicate_skips"] == 1
    assert strategy.get_recent_rejections(limit=1)[0].reason == "duplicate_detected_at"


@pytest.mark.asyncio
async def test_duplicate_cluster_signature_is_rejected_when_detected_at_dedup_disabled(
    event_bus: FakeEventBus,
) -> None:
    config = DummyStrategyConfig(
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
        deduplicate_by_detected_at=False,
        deduplicate_same_cluster_signature=True,
    )
    strategy = DummyAnalyticsStrategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result()

    await strategy._on_bus_event(make_event(result))
    await strategy._on_bus_event(make_event(result))

    stats = strategy.get_stats()

    assert stats["emitted_signals"] == 1
    assert stats["rejected_events"] == 1
    assert stats["duplicate_skips"] == 1
    assert strategy.get_recent_rejections(limit=1)[0].reason == "duplicate_cluster_signature"


@pytest.mark.asyncio
async def test_same_side_signal_too_soon_is_rejected(
    event_bus: FakeEventBus,
) -> None:
    config = DummyStrategyConfig(
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=60,
        max_signals_per_symbol_window=0,
        deduplicate_by_detected_at=False,
        deduplicate_same_cluster_signature=False,
    )
    strategy = DummyAnalyticsStrategy(event_bus=event_bus, config=config)
    await strategy.start()

    first = make_result(
        detected_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        direction=DummyDirection.UP,
    )
    second = make_result(
        detected_at=datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc),
        direction=DummyDirection.UP,
    )

    await strategy._on_bus_event(make_event(first))
    await strategy._on_bus_event(make_event(second))

    stats = strategy.get_stats()

    assert stats["emitted_signals"] == 1
    assert stats["rejected_events"] == 1
    assert stats["duplicate_skips"] == 1
    assert strategy.get_recent_rejections(limit=1)[0].reason == "same_side_signal_too_soon"


@pytest.mark.asyncio
async def test_opposite_side_signal_is_not_rejected_by_same_side_filter(
    event_bus: FakeEventBus,
) -> None:
    config = DummyStrategyConfig(
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=60,
        max_signals_per_symbol_window=0,
        deduplicate_by_detected_at=False,
        deduplicate_same_cluster_signature=False,
    )
    strategy = DummyAnalyticsStrategy(event_bus=event_bus, config=config)
    await strategy.start()

    first = make_result(
        detected_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        direction=DummyDirection.UP,
    )
    second = make_result(
        detected_at=datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc),
        direction=DummyDirection.DOWN,
    )

    await strategy._on_bus_event(make_event(first))
    await strategy._on_bus_event(make_event(second))

    stats = strategy.get_stats()

    assert stats["emitted_signals"] == 2
    assert stats["rejected_events"] == 0

    signals = emitted_payloads(event_bus, "signal.generated")

    assert [signal.side for signal in signals] == ["LONG", "SHORT"]


@pytest.mark.asyncio
async def test_symbol_signal_rate_limit_rejects_when_window_is_full(
    event_bus: FakeEventBus,
) -> None:
    config = DummyStrategyConfig(
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=1,
        signal_window_seconds=60,
        deduplicate_by_detected_at=False,
        deduplicate_same_cluster_signature=False,
    )
    strategy = DummyAnalyticsStrategy(event_bus=event_bus, config=config)
    await strategy.start()

    first = make_result(
        detected_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    second = make_result(
        detected_at=datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc),
    )

    await strategy._on_bus_event(make_event(first))
    await strategy._on_bus_event(make_event(second))

    stats = strategy.get_stats()

    assert stats["emitted_signals"] == 1
    assert stats["rejected_events"] == 1
    assert stats["rate_limit_skips"] == 1
    assert strategy.get_recent_rejections(limit=1)[0].reason == "symbol_signal_rate_limited"


# ============================================================================
# State helpers / signatures / diagnostics
# ============================================================================


def test_state_key_normalizes_exchange_and_symbol() -> None:
    assert DummyAnalyticsStrategy.state_key("Binance", "BTC/USDT") == (
        "binance",
        "BTCUSDT",
    )
    assert DummyAnalyticsStrategy.state_key("BYBIT", "eth-usdt") == (
        "bybit",
        "ETHUSDT",
    )


def test_get_or_create_state_reuses_existing_state(
    strategy: DummyAnalyticsStrategy,
) -> None:
    first = strategy.get_or_create_state("binance", "BTC/USDT")
    second = strategy.get_or_create_state("BINANCE", "BTCUSDT")

    assert first is second
    assert first.exchange == "binance"
    assert first.symbol == "BTCUSDT"


def test_build_cluster_signature_is_stable_for_same_result(
    strategy: DummyAnalyticsStrategy,
) -> None:
    result = make_result()

    first = strategy.build_cluster_signature(result)
    second = strategy.build_cluster_signature(result)

    assert first == second
    assert isinstance(first, str)
    assert len(first) == 64


def test_build_cluster_signature_changes_when_relevant_result_changes(
    strategy: DummyAnalyticsStrategy,
) -> None:
    first = make_result(
        detected_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    second = make_result(
        detected_at=datetime(2026, 1, 1, 12, 1, 0, tzinfo=timezone.utc),
    )

    assert strategy.build_cluster_signature(first) != strategy.build_cluster_signature(second)


@pytest.mark.asyncio
async def test_publish_diagnostics_snapshot_emits_snapshot_when_running(
    event_bus: FakeEventBus,
) -> None:
    config = DummyStrategyConfig(publish_diagnostics_snapshots=True)
    strategy = DummyAnalyticsStrategy(event_bus=event_bus, config=config)

    await strategy.start()
    await strategy._on_bus_event(make_event(make_result()))

    event_bus.emitted.clear()

    await strategy.publish_diagnostics_snapshot()

    assert emitted_topics(event_bus) == ["strategy.dummy.snapshot"]

    snapshot = emitted_payloads(event_bus, "strategy.dummy.snapshot")[0]

    assert snapshot["strategy_name"] == "dummy_strategy"
    assert snapshot["signal_type"] == "dummy"
    assert snapshot["stats"]["emitted_signals"] == 1
    assert "hot_symbols" in snapshot


@pytest.mark.asyncio
async def test_publish_diagnostics_snapshot_does_nothing_when_not_running(
    strategy: DummyAnalyticsStrategy,
    event_bus: FakeEventBus,
) -> None:
    await strategy.publish_diagnostics_snapshot()

    assert event_bus.emitted == []


@pytest.mark.asyncio
async def test_get_recent_signals_returns_most_recent_first(
    event_bus: FakeEventBus,
) -> None:
    config = DummyStrategyConfig(
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
        deduplicate_by_detected_at=False,
        deduplicate_same_cluster_signature=False,
    )
    strategy = DummyAnalyticsStrategy(event_bus=event_bus, config=config)
    await strategy.start()

    await strategy._on_bus_event(
        make_event(
            make_result(
                symbol="BTCUSDT",
                detected_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            )
        )
    )
    await strategy._on_bus_event(
        make_event(
            make_result(
                symbol="ETHUSDT",
                detected_at=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
            )
        )
    )

    recent = strategy.get_recent_signals(limit=10)

    assert len(recent) == 2
    assert recent[0].symbol == "ETHUSDT"
    assert recent[1].symbol == "BTCUSDT"


@pytest.mark.asyncio
async def test_get_recent_rejections_can_filter_by_exchange_and_symbol(
    event_bus: FakeEventBus,
) -> None:
    config = DummyStrategyConfig(min_confidence=0.90)
    strategy = DummyAnalyticsStrategy(event_bus=event_bus, config=config)
    await strategy.start()

    await strategy._on_bus_event(
        make_event(
            make_result(
                exchange="binance",
                symbol="BTCUSDT",
                confidence=0.80,
            )
        )
    )
    await strategy._on_bus_event(
        make_event(
            make_result(
                exchange="bybit",
                symbol="ETHUSDT",
                confidence=0.80,
            )
        )
    )

    filtered = strategy.get_recent_rejections(
        exchange="bybit",
        symbol="ETH/USDT",
        limit=10,
    )

    assert len(filtered) == 1
    assert filtered[0].exchange == "bybit"
    assert filtered[0].symbol == "ETHUSDT"
    assert filtered[0].reason == "confidence_below_threshold"


@pytest.mark.asyncio
async def test_get_hot_symbols_returns_latest_signal_per_symbol_sorted_by_score(
    event_bus: FakeEventBus,
) -> None:
    config = DummyStrategyConfig(
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
        deduplicate_by_detected_at=False,
        deduplicate_same_cluster_signature=False,
    )
    strategy = DummyAnalyticsStrategy(event_bus=event_bus, config=config)
    await strategy.start()

    await strategy._on_bus_event(
        make_event(
            make_result(
                symbol="BTCUSDT",
                confidence=0.70,
                intensity_score=0.70,
                detected_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            )
        )
    )
    await strategy._on_bus_event(
        make_event(
            make_result(
                symbol="ETHUSDT",
                confidence=0.90,
                intensity_score=0.90,
                detected_at=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
            )
        )
    )

    rows = strategy.get_hot_symbols(limit=10)

    assert len(rows) == 2
    assert rows[0]["symbol"] == "ETHUSDT"
    assert rows[1]["symbol"] == "BTCUSDT"
    assert rows[0]["score"] >= rows[1]["score"]