# tests/strategy/strategies/liquidations/test_base_analytics_strategy_behavior.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

import pytest

from core.event_bus import Event, EventPriority

from analytics.liquidations.models import (
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    LiquidationKey,
    liquidation_key_to_dict,
)

from strategy.strategies.liquidations.base import (
    BaseAnalyticsStrategy,
    BaseSymbolStrategyState,
    StrategyRejection,
    clamp_float,
    ensure_utc,
    make_strategy_scope_key,
    normalize_exchange,
    normalize_exchange_symbol,
    normalize_market_type,
    normalize_symbol,
    normalize_timeframe,
    result_scope,
    scoped_key_to_string,
    serialize_value,
    signal_scope,
    utc_now,
)


# ============================================================================
# Test doubles
# ============================================================================


class DummyDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    UNKNOWN = "unknown"

    @property
    def is_known(self) -> bool:
        return self is not DummyDirection.UNKNOWN


class DummySeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

    @property
    def rank(self) -> int:
        if self is DummySeverity.LOW:
            return 1
        if self is DummySeverity.MEDIUM:
            return 2
        if self is DummySeverity.HIGH:
            return 3
        if self is DummySeverity.EXTREME:
            return 4
        return 0


@dataclass(slots=True)
class DummyCluster:
    exchange: str = "binance"
    symbol: str = "BTCUSDT"
    market_type: str = "usdm_futures"
    timeframe: str = "realtime"
    exchange_symbol: str | None = "BTCUSDT"

    start_time: datetime = datetime(2026, 1, 1, 11, 59, 55, tzinfo=timezone.utc)
    end_time: datetime = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    duration_seconds: float = 5.0
    event_count: int = 10
    total_notional_usd: Decimal = Decimal("500000")
    avg_notional_per_event: Decimal = Decimal("50000")
    avg_price: Decimal = Decimal("100000")
    min_price: Decimal = Decimal("99500")
    max_price: Decimal = Decimal("100500")
    price_range_pct: float = 0.25

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.exchange = normalize_exchange(self.exchange)
        self.symbol = normalize_symbol(self.symbol)
        self.market_type = normalize_market_type(self.market_type)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

    @property
    def key(self) -> LiquidationKey:
        return make_strategy_scope_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def scope(self) -> dict[str, str]:
        scope = liquidation_key_to_dict(self.key)
        scope["exchange_symbol"] = self.exchange_symbol or self.symbol
        return scope


@dataclass(slots=True)
class DummyResult:
    exchange: str = "binance"
    symbol: str = "BTCUSDT"
    market_type: str = "usdm_futures"
    timeframe: str = "realtime"
    exchange_symbol: str | None = "BTCUSDT"

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

    correlation_id: str | None = "analytics-corr-1"
    metadata: dict[str, Any] | None = None
    cluster: DummyCluster | None = None

    event_type: Any = "dummy_detected"
    status: Any = "confirmed"

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.75

    @property
    def key(self) -> LiquidationKey:
        return make_strategy_scope_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def scope(self) -> dict[str, str]:
        scope = liquidation_key_to_dict(self.key)
        scope["exchange_symbol"] = self.exchange_symbol or self.symbol
        return scope

    def __post_init__(self) -> None:
        self.exchange = normalize_exchange(self.exchange)
        self.symbol = normalize_symbol(self.symbol)
        self.market_type = normalize_market_type(self.market_type)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )
        self.detected_at = ensure_utc(self.detected_at)

        if self.metadata is None:
            self.metadata = {}

        if self.cluster is None:
            self.cluster = DummyCluster(
                exchange=self.exchange,
                market_type=self.market_type,
                symbol=self.symbol,
                timeframe=self.timeframe,
                exchange_symbol=self.exchange_symbol,
            )


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

    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.exchange = normalize_exchange(self.exchange)
        self.symbol = normalize_symbol(self.symbol)
        self.market_type = normalize_market_type(self.market_type)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )
        self.side = self.side.upper()
        self.confidence = clamp_float(self.confidence)
        self.score = clamp_float(self.score)
        self.intensity_score = clamp_float(self.intensity_score)
        self.generated_at = ensure_utc(self.generated_at)
        self.detected_at = ensure_utc(self.detected_at)

    @property
    def key(self) -> LiquidationKey:
        return make_strategy_scope_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def scope(self) -> dict[str, str]:
        scope = liquidation_key_to_dict(self.key)
        scope["exchange_symbol"] = self.exchange_symbol or self.symbol
        return scope

    @property
    def scope_key(self) -> str:
        return scoped_key_to_string(self.key)


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
    allowed_market_types: tuple[str, ...] = ()
    allowed_symbols: tuple[str, ...] = ()
    allowed_timeframes: tuple[str, ...] = ()

    blocked_market_types: tuple[str, ...] = ()
    blocked_symbols: tuple[str, ...] = ()
    blocked_timeframes: tuple[str, ...] = ()

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
        if not self.subscribe_topic:
            raise ValueError("subscribe_topic must not be empty")

        if not self.publish_topic_signal_generated:
            raise ValueError("publish_topic_signal_generated must not be empty")

        if not self.publish_topic_signal_rejected:
            raise ValueError("publish_topic_signal_rejected must not be empty")

        if not self.diagnostics_topic:
            raise ValueError("diagnostics_topic must not be empty")

        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be between 0 and 1")

        if not (0.0 <= self.min_intensity_score <= 1.0):
            raise ValueError("min_intensity_score must be between 0 and 1")

        if self.min_total_notional_usd < 0:
            raise ValueError("min_total_notional_usd must be >= 0")

        if self.min_event_count < 0:
            raise ValueError("min_event_count must be >= 0")

        if self.max_price_range_pct is not None and self.max_price_range_pct < 0:
            raise ValueError("max_price_range_pct must be >= 0 or None")

        if self.symbol_cooldown_seconds < 0:
            raise ValueError("symbol_cooldown_seconds must be >= 0")

        if self.min_seconds_between_same_side_signals < 0:
            raise ValueError("min_seconds_between_same_side_signals must be >= 0")

        if self.max_signals_per_symbol_window < 0:
            raise ValueError("max_signals_per_symbol_window must be >= 0")

        if self.signal_window_seconds <= 0:
            raise ValueError("signal_window_seconds must be > 0")

        if self.diagnostics_interval_seconds <= 0:
            raise ValueError("diagnostics_interval_seconds must be > 0")


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
            event_bus=event_bus,  # type: ignore[arg-type]
            scheduler=scheduler,  # type: ignore[arg-type]
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
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        exchange_symbol: str | None = None,
    ) -> BaseSymbolStrategyState:
        return BaseSymbolStrategyState(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            exchange_symbol=exchange_symbol,
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
        state = self.get_or_create_state_for_result(result)
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
            market_type=result.market_type,
            symbol=result.symbol,
            timeframe=result.timeframe,
            exchange_symbol=result.exchange_symbol,
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
            metadata={
                "scope": result.scope,
                "source_result_key": result.key,
            },
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


def make_cluster(**overrides: Any) -> DummyCluster:
    base = {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "realtime",
        "exchange_symbol": "BTCUSDT",
        "start_time": datetime(2026, 1, 1, 11, 59, 55, tzinfo=timezone.utc),
        "end_time": datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "duration_seconds": 5.0,
        "event_count": 10,
        "total_notional_usd": Decimal("500000"),
        "avg_notional_per_event": Decimal("50000"),
        "avg_price": Decimal("100000"),
        "min_price": Decimal("99500"),
        "max_price": Decimal("100500"),
        "price_range_pct": 0.25,
    }
    base.update(overrides)
    return DummyCluster(**base)


def make_result(**overrides: Any) -> DummyResult:
    base = {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "realtime",
        "exchange_symbol": "BTCUSDT",
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
        "correlation_id": "analytics-corr-1",
        "metadata": {
            "side_imbalance_ratio": 0.80,
            "event_imbalance_ratio": 0.75,
            "acceleration_ratio": 1.30,
        },
        "cluster": None,
    }
    base.update(overrides)

    if base["cluster"] is None:
        base["cluster"] = make_cluster(
            exchange=base["exchange"],
            market_type=base["market_type"],
            symbol=base["symbol"],
            timeframe=base["timeframe"],
            exchange_symbol=base["exchange_symbol"],
        )

    return DummyResult(**base)


def make_event(
    payload: Any,
    *,
    topic: str = "analytics.dummy.detected",
    correlation_id: str | None = "bus-corr-1",
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


def latest_signal(event_bus: FakeEventBus) -> DummySignal:
    payloads = emitted_payloads(event_bus, "signal.generated")
    assert payloads, "expected signal.generated"
    signal = payloads[-1]
    assert isinstance(signal, DummySignal)
    return signal


def latest_rejection(strategy: DummyAnalyticsStrategy) -> StrategyRejection:
    rejections = strategy.get_recent_rejections(limit=1)
    assert rejections, "expected at least one rejection"
    return rejections[0]


def assert_no_risk_or_execution_events(event_bus: FakeEventBus) -> None:
    forbidden_prefixes = ("risk.", "execution.", "order.", "position.")

    for topic in emitted_topics(event_bus):
        assert not topic.startswith(forbidden_prefixes), (
            f"strategy must not emit direct risk/execution/order/position event: {topic}"
        )


def assert_full_scope_headers(
    headers: dict[str, Any],
    *,
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    symbol: str = "BTCUSDT",
    timeframe: str = "realtime",
    exchange_symbol: str = "BTCUSDT",
) -> None:
    expected_key = make_strategy_scope_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )

    assert headers["exchange"] == normalize_exchange(exchange)
    assert headers["market_type"] == normalize_market_type(market_type)
    assert headers["symbol"] == normalize_symbol(symbol)
    assert headers["timeframe"] == normalize_timeframe(timeframe)
    assert headers["exchange_symbol"] == exchange_symbol
    assert headers["scope"] == scoped_key_to_string(expected_key)


# ============================================================================
# Helper behavior
# ============================================================================


def test_scope_helpers_normalize_futures_scope() -> None:
    key = make_strategy_scope_key(
        exchange="BINANCE",
        market_type="USDM_FUTURES",
        symbol="btc-usdt",
        timeframe="Realtime",
    )

    assert key == ("binance", "usdm_futures", "BTCUSDT", "realtime")
    assert scoped_key_to_string(key) == "binance:usdm_futures:BTCUSDT:realtime"


def test_result_scope_and_signal_scope_include_exchange_symbol_and_scope_key() -> None:
    result = make_result(
        exchange="okx",
        market_type="swap",
        symbol="BTC-USDT",
        timeframe="Realtime",
        exchange_symbol="BTC-USDT-SWAP",
    )
    signal = DummySignal(
        strategy_name="dummy_strategy",
        signal_type="dummy",
        exchange=result.exchange,
        market_type=result.market_type,
        symbol=result.symbol,
        timeframe=result.timeframe,
        exchange_symbol=result.exchange_symbol,
        side="LONG",
        confidence=0.8,
        score=0.8,
        intensity_score=0.8,
        severity="high",
        generated_at=utc_now(),
        detected_at=result.detected_at,
        total_notional_usd=result.total_notional_usd,
        source_event_id="event-1",
        correlation_id="corr-1",
    )

    assert result_scope(result) == {
        "exchange": "okx",
        "market_type": "swap",
        "symbol": "BTCUSDT",
        "timeframe": "realtime",
        "exchange_symbol": "BTC-USDT-SWAP",
        "scope_key": "okx:swap:BTCUSDT:realtime",
    }

    assert signal_scope(signal) == {
        "exchange": "okx",
        "market_type": "swap",
        "symbol": "BTCUSDT",
        "timeframe": "realtime",
        "exchange_symbol": "BTC-USDT-SWAP",
        "scope_key": "okx:swap:BTCUSDT:realtime",
    }


def test_serialize_value_converts_decimal_datetime_enum_and_tuple_safely() -> None:
    dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    data = serialize_value(
        {
            "decimal": Decimal("123.45"),
            "datetime": dt,
            "enum": DummySeverity.HIGH,
            "tuple": ("binance", "usdm_futures", "BTCUSDT", "realtime"),
        }
    )

    assert data == {
        "decimal": "123.45",
        "datetime": "2026-01-01T12:00:00+00:00",
        "enum": "high",
        "tuple": ["binance", "usdm_futures", "BTCUSDT", "realtime"],
    }


# ============================================================================
# Lifecycle
# ============================================================================


@pytest.mark.asyncio
async def test_start_subscribes_to_configured_topic_and_sets_running(
    event_bus: FakeEventBus,
) -> None:
    strategy = DummyAnalyticsStrategy(event_bus=event_bus)

    await strategy.start()

    assert strategy.is_running is True
    assert strategy.is_registered is True
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

    assert strategy.is_running is False
    assert strategy.is_registered is False
    assert strategy.get_stats()["running"] is False
    assert event_bus.subscriptions == []


@pytest.mark.asyncio
async def test_start_is_idempotent(
    event_bus: FakeEventBus,
) -> None:
    strategy = DummyAnalyticsStrategy(event_bus=event_bus)

    await strategy.start()
    await strategy.start()

    assert strategy.is_running is True
    assert len(event_bus.subscriptions) == 1


@pytest.mark.asyncio
async def test_stop_unsubscribes_and_marks_strategy_stopped(
    event_bus: FakeEventBus,
) -> None:
    strategy = DummyAnalyticsStrategy(event_bus=event_bus)

    await strategy.start()
    subscription = event_bus.subscriptions[0]

    await strategy.stop()

    assert strategy.is_running is False
    assert strategy.is_registered is False
    assert strategy.get_stats()["running"] is False
    assert strategy.get_stats()["stopped_at"] is not None
    assert event_bus.unsubscribed == [subscription]


@pytest.mark.asyncio
async def test_stop_is_idempotent(
    event_bus: FakeEventBus,
) -> None:
    strategy = DummyAnalyticsStrategy(event_bus=event_bus)

    await strategy.stop()
    await strategy.stop()

    assert strategy.is_running is False
    assert event_bus.unsubscribed == []


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
    assert strategy.is_running is True
    assert strategy.is_registered is True


@pytest.mark.asyncio
async def test_close_delegates_to_stop(
    event_bus: FakeEventBus,
) -> None:
    strategy = DummyAnalyticsStrategy(event_bus=event_bus)

    await strategy.start()
    await strategy.close()

    assert strategy.is_running is False
    assert strategy.is_registered is False
    assert len(event_bus.unsubscribed) == 1


@pytest.mark.asyncio
async def test_start_registers_diagnostics_job_when_enabled(
    event_bus: FakeEventBus,
) -> None:
    scheduler = FakeScheduler()
    config = DummyStrategyConfig(
        publish_diagnostics_snapshots=True,
        diagnostics_interval_seconds=7.5,
    )
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
    assert job["func"] == strategy.publish_diagnostics_snapshot
    assert job["interval"] == 7.5
    assert job["allow_overlap"] is False
    assert job["enabled"] is True


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
    assert event_bus.emitted == []


# ============================================================================
# Successful signal path
# ============================================================================


@pytest.mark.asyncio
async def test_valid_result_emits_signal_generated_with_full_scope_and_updates_state(
    strategy: DummyAnalyticsStrategy,
    event_bus: FakeEventBus,
) -> None:
    await strategy.start()

    result = make_result(
        exchange="Bybit",
        market_type="Linear",
        symbol="ETH/USDT",
        timeframe="Realtime",
        exchange_symbol="ETHUSDT",
    )
    event = make_event(result, correlation_id="signal-corr")

    await strategy._on_bus_event(event)

    assert emitted_topics(event_bus) == ["signal.generated"]

    signal = latest_signal(event_bus)

    assert signal.strategy_name == "dummy_strategy"
    assert signal.signal_type == "dummy"
    assert signal.exchange == "bybit"
    assert signal.market_type == "linear"
    assert signal.symbol == "ETHUSDT"
    assert signal.timeframe == "realtime"
    assert signal.exchange_symbol == "ETHUSDT"
    assert signal.scope_key == "bybit:linear:ETHUSDT:realtime"
    assert signal.side == "LONG"
    assert signal.source_event_id == event.event_id
    assert signal.correlation_id == "signal-corr"

    state = strategy.get_or_create_state_for_result(result)

    assert state.key == ("bybit", "linear", "ETHUSDT", "realtime")
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
    assert stats["tracked_scopes"] == 1
    assert stats["tracked_symbols"] == 1
    assert stats["scope"] == "exchange:market_type:symbol:timeframe"


@pytest.mark.asyncio
async def test_emit_signal_includes_full_scope_headers_and_correlation_id(
    strategy: DummyAnalyticsStrategy,
    event_bus: FakeEventBus,
) -> None:
    await strategy.start()

    result = make_result(
        exchange="okx",
        market_type="swap",
        symbol="BTC-USDT",
        timeframe="1m",
        exchange_symbol="BTC-USDT-SWAP",
    )
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
    assert headers["side"] == "LONG"

    assert_full_scope_headers(
        headers,
        exchange="okx",
        market_type="swap",
        symbol="BTCUSDT",
        timeframe="1m",
        exchange_symbol="BTC-USDT-SWAP",
    )


@pytest.mark.asyncio
async def test_emit_failure_does_not_mark_signal_as_emitted_or_update_state() -> None:
    event_bus = FailingEventBus()
    strategy = DummyAnalyticsStrategy(event_bus=event_bus)
    await strategy.start()

    result = make_result()
    event = make_event(result)

    await strategy._on_bus_event(event)

    stats = strategy.get_stats()
    state = strategy.get_or_create_state_for_result(result)

    assert stats["processed_events"] == 1
    assert stats["emitted_signals"] == 0
    assert stats["last_error"] is not None
    assert "event bus emit failed" in stats["last_error"]

    assert state.total_signals_emitted == 0
    assert state.last_signal_at is None
    assert state.last_detected_at is None
    assert state.last_cluster_signature is None


@pytest.mark.asyncio
async def test_strategy_never_emits_direct_risk_or_execution_events(
    strategy: DummyAnalyticsStrategy,
    event_bus: FakeEventBus,
) -> None:
    await strategy.start()

    await strategy._on_bus_event(make_event(make_result()))

    assert_no_risk_or_execution_events(event_bus)


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
            {"allowed_market_types": ("linear",)},
            {"market_type": "usdm_futures"},
            "market_type_not_allowed",
        ),
        (
            {"blocked_market_types": ("usdm_futures",)},
            {"market_type": "usdm_futures"},
            "market_type_blocked",
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
            {"allowed_timeframes": ("1m",)},
            {"timeframe": "realtime"},
            "timeframe_not_allowed",
        ),
        (
            {"blocked_timeframes": ("realtime",)},
            {"timeframe": "realtime"},
            "timeframe_blocked",
        ),
        (
            {},
            {"direction": DummyDirection.UNKNOWN},
            "unknown_direction",
        ),
        (
            {"require_high_confidence_only": True},
            {"confidence": 0.74},
            "not_high_confidence",
        ),
        (
            {"min_confidence": 0.90},
            {"confidence": 0.89},
            "confidence_below_threshold",
        ),
        (
            {"min_intensity_score": 0.80},
            {"intensity_score": 0.79},
            "intensity_below_threshold",
        ),
        (
            {"min_total_notional_usd": Decimal("600000")},
            {"total_notional_usd": Decimal("599999")},
            "notional_below_threshold",
        ),
        (
            {"min_event_count": 11},
            {"event_count": 10},
            "event_count_below_threshold",
        ),
        (
            {"allowed_severities": (DummySeverity.EXTREME,)},
            {"severity": DummySeverity.HIGH},
            "severity_not_allowed",
        ),
        (
            {"max_price_range_pct": 0.20},
            {"price_range_pct": 0.21},
            "price_range_above_threshold",
        ),
    ],
)
@pytest.mark.asyncio
async def test_common_filter_rejection_matrix(
    event_bus: FakeEventBus,
    config_overrides: dict[str, Any],
    result_overrides: dict[str, Any],
    expected_reason: str,
) -> None:
    config = DummyStrategyConfig(
        publish_rejected_events=False,
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
        **config_overrides,
    )
    strategy = DummyAnalyticsStrategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(**result_overrides)

    await strategy._on_bus_event(make_event(result))

    stats = strategy.get_stats()

    assert stats["processed_events"] == 1
    assert stats["emitted_signals"] == 0
    assert stats["rejected_events"] == 1
    assert stats["filter_skips"] == 1
    assert event_bus.emitted == []

    rejection = latest_rejection(strategy)

    assert rejection.reason == expected_reason
    assert rejection.exchange == result.exchange
    assert rejection.market_type == result.market_type
    assert rejection.symbol == result.symbol
    assert rejection.timeframe == result.timeframe
    assert rejection.scope_key == scoped_key_to_string(result.key)


@pytest.mark.asyncio
async def test_filter_order_returns_first_rejection_only(
    event_bus: FakeEventBus,
) -> None:
    config = DummyStrategyConfig(
        allowed_market_types=("linear",),
        min_confidence=0.99,
        min_total_notional_usd=Decimal("100000000"),
    )
    strategy = DummyAnalyticsStrategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(
        market_type="usdm_futures",
        confidence=0.10,
        total_notional_usd=Decimal("1"),
    )

    await strategy._on_bus_event(make_event(result))

    rejection = latest_rejection(strategy)

    assert rejection.reason == "market_type_not_allowed"


# ============================================================================
# Rejection publishing / contract
# ============================================================================


@pytest.mark.asyncio
async def test_rejected_event_is_published_with_full_scope_headers_when_enabled(
    event_bus: FakeEventBus,
) -> None:
    config = DummyStrategyConfig(
        publish_rejected_events=True,
        min_confidence=0.90,
    )
    strategy = DummyAnalyticsStrategy(event_bus=event_bus, config=config)
    await strategy.start()

    result = make_result(
        exchange="okx",
        market_type="swap",
        symbol="SOL-USDT",
        timeframe="1m",
        exchange_symbol="SOL-USDT-SWAP",
        confidence=0.50,
    )
    event = make_event(result, correlation_id="reject-corr")

    await strategy._on_bus_event(event)

    assert emitted_topics(event_bus) == ["signal.rejected"]

    emitted = event_bus.emitted[0]
    rejection = emitted["payload"]

    assert isinstance(rejection, StrategyRejection)
    assert rejection.reason == "confidence_below_threshold"
    assert rejection.exchange == "okx"
    assert rejection.market_type == "swap"
    assert rejection.symbol == "SOLUSDT"
    assert rejection.timeframe == "1m"
    assert rejection.exchange_symbol == "SOL-USDT-SWAP"
    assert rejection.scope_key == "okx:swap:SOLUSDT:1m"
    assert rejection.correlation_id == "reject-corr"
    assert rejection.source_event_id == event.event_id
    assert rejection.details["scope_key"] == "okx:swap:SOLUSDT:1m"
    assert rejection.details["confidence"] == 0.50

    assert emitted["priority"] is EventPriority.LOW
    assert emitted["source"] == "dummy_strategy"
    assert emitted["correlation_id"] == "reject-corr"

    headers = emitted["headers"]

    assert headers["strategy"] == "dummy_strategy"
    assert headers["signal_type"] == "dummy"
    assert headers["reason"] == "confidence_below_threshold"
    assert headers["source_event_id"] == event.event_id
    assert headers["source_topic"] == event.topic

    assert_full_scope_headers(
        headers,
        exchange="okx",
        market_type="swap",
        symbol="SOLUSDT",
        timeframe="1m",
        exchange_symbol="SOL-USDT-SWAP",
    )


def test_strategy_rejection_to_dict_contains_full_scope() -> None:
    rejection = StrategyRejection(
        exchange="BINANCE",
        market_type="USDM_FUTURES",
        symbol="btc-usdt",
        timeframe="Realtime",
        exchange_symbol="BTCUSDT",
        rejected_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        reason="test_reason",
        source_topic="analytics.dummy.detected",
        strategy_name="dummy_strategy",
        signal_type="dummy",
        correlation_id="corr-1",
        source_event_id="event-1",
        details={"notional": Decimal("123.45")},
    )

    data = rejection.to_dict()

    assert data["exchange"] == "binance"
    assert data["market_type"] == "usdm_futures"
    assert data["symbol"] == "BTCUSDT"
    assert data["timeframe"] == "realtime"
    assert data["exchange_symbol"] == "BTCUSDT"
    assert data["scope"] == {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "realtime",
        "exchange_symbol": "BTCUSDT",
    }
    assert data["scope_key"] == "binance:usdm_futures:BTCUSDT:realtime"
    assert data["details"]["notional"] == "123.45"


# ============================================================================
# Full-scope state isolation
# ============================================================================


@pytest.mark.asyncio
async def test_same_symbol_different_market_type_does_not_share_cooldown(
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

    usdm = make_result(
        exchange="binance",
        market_type="usdm_futures",
        symbol="BTCUSDT",
        timeframe="realtime",
        exchange_symbol="BTCUSDT",
        detected_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    coinm = make_result(
        exchange="binance",
        market_type="coinm_futures",
        symbol="BTCUSDT",
        timeframe="realtime",
        exchange_symbol="BTCUSD_PERP",
        detected_at=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
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
async def test_same_symbol_different_timeframe_does_not_share_cooldown(
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

    realtime = make_result(
        market_type="usdm_futures",
        timeframe="realtime",
        detected_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    one_minute = make_result(
        market_type="usdm_futures",
        timeframe="1m",
        detected_at=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
    )

    await strategy._on_bus_event(make_event(realtime, correlation_id="realtime"))
    await strategy._on_bus_event(make_event(one_minute, correlation_id="1m"))

    assert strategy.get_stats()["emitted_signals"] == 2
    assert strategy.get_stats()["cooldown_skips"] == 0
    assert strategy.get_stats()["tracked_scopes"] == 2
    assert strategy.get_state("binance", "BTCUSDT", market_type="usdm_futures", timeframe="realtime") is not None
    assert strategy.get_state("binance", "BTCUSDT", market_type="usdm_futures", timeframe="1m") is not None


@pytest.mark.asyncio
async def test_same_scope_respects_cooldown(
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
        detected_at=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
    )

    await strategy._on_bus_event(make_event(first))
    await strategy._on_bus_event(make_event(second))

    assert strategy.get_stats()["emitted_signals"] == 1
    assert strategy.get_stats()["rejected_events"] == 1
    assert strategy.get_stats()["cooldown_skips"] == 1

    rejection = latest_rejection(strategy)
    assert rejection.reason == "scope_in_cooldown"


# ============================================================================
# Dedup / rate limit
# ============================================================================


@pytest.mark.asyncio
async def test_duplicate_detected_at_is_scoped_by_market_type(
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

    same_detected_at = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

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
    strategy = DummyAnalyticsStrategy(event_bus=event_bus)

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
        market_type="usdm_futures",
        detected_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    second_same_scope = make_result(
        market_type="usdm_futures",
        detected_at=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
    )
    third_other_scope = make_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
        detected_at=datetime(2026, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
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
        market_type="usdm_futures",
        detected_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    second_same_scope = make_result(
        market_type="usdm_futures",
        detected_at=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
    )
    third_other_scope = make_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
        detected_at=datetime(2026, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
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
# Recent query API
# ============================================================================


@pytest.mark.asyncio
async def test_get_recent_signals_filters_by_full_scope(
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

    usdm = make_result(
        market_type="usdm_futures",
        exchange_symbol="BTCUSDT",
        detected_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    coinm = make_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
        detected_at=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
    )
    one_minute = make_result(
        market_type="usdm_futures",
        timeframe="1m",
        exchange_symbol="BTCUSDT",
        detected_at=datetime(2026, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
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
    assert usdm_realtime[0].market_type == "usdm_futures"
    assert usdm_realtime[0].timeframe == "realtime"
    assert coinm_realtime[0].market_type == "coinm_futures"


@pytest.mark.asyncio
async def test_get_recent_rejections_filters_by_full_scope(
    event_bus: FakeEventBus,
) -> None:
    config = DummyStrategyConfig(
        publish_rejected_events=False,
        allowed_market_types=("usdm_futures",),
    )
    strategy = DummyAnalyticsStrategy(event_bus=event_bus, config=config)
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
async def test_recent_query_methods_return_empty_for_non_positive_limit(
    strategy: DummyAnalyticsStrategy,
) -> None:
    assert strategy.get_recent_signals(limit=0) == []
    assert strategy.get_recent_signals(limit=-1) == []
    assert strategy.get_recent_rejections(limit=0) == []
    assert strategy.get_recent_rejections(limit=-1) == []
    assert strategy.get_hot_symbols(limit=0) == []


# ============================================================================
# Hot symbols / diagnostics
# ============================================================================


@pytest.mark.asyncio
async def test_get_hot_symbols_returns_one_row_per_scope_sorted_by_score(
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

    weaker_usdm = make_result(
        market_type="usdm_futures",
        symbol="BTCUSDT",
        confidence=0.70,
        intensity_score=0.70,
        detected_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    stronger_usdm = make_result(
        market_type="usdm_futures",
        symbol="BTCUSDT",
        confidence=0.95,
        intensity_score=0.95,
        detected_at=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
    )
    coinm = make_result(
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
        symbol="BTCUSDT",
        confidence=0.90,
        intensity_score=0.85,
        detected_at=datetime(2026, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
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
async def test_publish_diagnostics_snapshot_emits_stats_and_hot_symbols(
    event_bus: FakeEventBus,
) -> None:
    config = DummyStrategyConfig(
        publish_diagnostics_snapshots=True,
        symbol_cooldown_seconds=0,
        min_seconds_between_same_side_signals=0,
        max_signals_per_symbol_window=0,
        deduplicate_by_detected_at=False,
        deduplicate_same_cluster_signature=False,
    )
    strategy = DummyAnalyticsStrategy(event_bus=event_bus, config=config)
    await strategy.start()

    await strategy._on_bus_event(make_event(make_result()))
    await strategy.publish_diagnostics_snapshot()

    assert emitted_topics(event_bus) == ["signal.generated", "strategy.dummy.snapshot"]

    emitted = event_bus.emitted[-1]

    assert emitted["topic"] == "strategy.dummy.snapshot"
    assert emitted["priority"] is EventPriority.LOW
    assert emitted["source"] == "dummy_strategy"
    assert emitted["headers"]["event_type"] == "strategy_diagnostics"

    payload = emitted["payload"]

    assert payload["strategy_name"] == "dummy_strategy"
    assert payload["signal_type"] == "dummy"
    assert payload["scope"] == "exchange:market_type:symbol:timeframe"
    assert payload["stats"]["tracked_scopes"] == 1
    assert payload["hot_symbols"][0]["scope_key"] == "binance:usdm_futures:BTCUSDT:realtime"


@pytest.mark.asyncio
async def test_publish_diagnostics_snapshot_noops_when_not_running(
    event_bus: FakeEventBus,
) -> None:
    strategy = DummyAnalyticsStrategy(event_bus=event_bus)

    await strategy.publish_diagnostics_snapshot()

    assert event_bus.emitted == []


# ============================================================================
# State model
# ============================================================================


def test_base_symbol_strategy_state_normalizes_and_exposes_full_scope() -> None:
    state = BaseSymbolStrategyState(
        exchange="BINANCE",
        market_type="USDM_FUTURES",
        symbol="btc-usdt",
        timeframe="Realtime",
        exchange_symbol="BTCUSDT",
    )

    assert state.exchange == "binance"
    assert state.market_type == "usdm_futures"
    assert state.symbol == "BTCUSDT"
    assert state.timeframe == "realtime"
    assert state.exchange_symbol == "BTCUSDT"
    assert state.key == ("binance", "usdm_futures", "BTCUSDT", "realtime")
    assert state.scope_key == "binance:usdm_futures:BTCUSDT:realtime"

    data = state.to_dict()

    assert data["scope"] == {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "realtime",
        "exchange_symbol": "BTCUSDT",
    }
    assert data["scope_key"] == "binance:usdm_futures:BTCUSDT:realtime"


def test_base_symbol_strategy_state_cooldown_and_rate_window() -> None:
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    state = BaseSymbolStrategyState(
        exchange="binance",
        market_type="usdm_futures",
        symbol="BTCUSDT",
        timeframe="realtime",
    )

    state.remember_signal(
        signal_at=now,
        signal_side="LONG",
        score=0.8,
        cooldown_seconds=30,
        cluster_signature="cluster-1",
        detected_at=now - timedelta(seconds=1),
        window_seconds=60,
    )

    assert state.is_in_cooldown(now + timedelta(seconds=29)) is True
    assert state.is_in_cooldown(now + timedelta(seconds=31)) is False
    assert state.signals_in_window(now + timedelta(seconds=30), 60) == 1
    assert state.signals_in_window(now + timedelta(seconds=61), 60) == 0


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
        ({"min_confidence": 1.01}, "min_confidence must be between 0 and 1"),
        ({"min_intensity_score": -0.01}, "min_intensity_score must be between 0 and 1"),
        ({"min_total_notional_usd": Decimal("-1")}, "min_total_notional_usd must be >= 0"),
        ({"min_event_count": -1}, "min_event_count must be >= 0"),
        ({"max_price_range_pct": -1.0}, "max_price_range_pct must be >= 0 or None"),
        ({"symbol_cooldown_seconds": -1}, "symbol_cooldown_seconds must be >= 0"),
        ({"min_seconds_between_same_side_signals": -1}, "min_seconds_between_same_side_signals must be >= 0"),
        ({"max_signals_per_symbol_window": -1}, "max_signals_per_symbol_window must be >= 0"),
        ({"signal_window_seconds": 0}, "signal_window_seconds must be > 0"),
        ({"diagnostics_interval_seconds": 0}, "diagnostics_interval_seconds must be > 0"),
    ],
)
def test_dummy_config_validation_rejects_invalid_values(
    overrides: dict[str, Any],
    expected_message: str,
) -> None:
    config = DummyStrategyConfig(**overrides)

    with pytest.raises(ValueError, match=expected_message):
        config.validate()