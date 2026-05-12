# tests/strategy/funding/conftest.py
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import pytest

from analytics.funding.enums import (
    FundingBias,
    FundingDivergenceType,
    FundingExtremeType,
    FundingFlipType,
    FundingPressureDirection,
    FundingPressureLevel,
    FundingRegime,
    FundingSignalType,
    FundingTimeframe,
)
from core.event_bus import Event, EventPriority

from strategy.strategies.funding.base import (
    BaseFundingStrategyConfig,
    FundingSetupStatus,
    FundingStrategyDirection,
    FundingStrategyState,
)
from strategy.strategies.funding.funding_divergence_strategy import (
    FundingDivergenceStrategy,
    FundingDivergenceStrategyConfig,
)
from strategy.strategies.funding.funding_extreme_reversal_strategy import (
    FundingExtremeReversalStrategy,
    FundingExtremeReversalStrategyConfig,
)


# =============================================================================
# Generic test helpers
# =============================================================================


UTC = timezone.utc
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_EXCHANGE = "binance"
DEFAULT_TIMEFRAME = FundingTimeframe.H1


def now_utc() -> datetime:
    return datetime.now(tz=UTC)


def iso_now() -> str:
    return now_utc().isoformat()


def iso_ago(seconds: float) -> str:
    return (now_utc() - timedelta(seconds=seconds)).isoformat()


def iso_after(seconds: float) -> str:
    return (now_utc() + timedelta(seconds=seconds)).isoformat()


def enum_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def enum_name_value(enum_cls: type[Enum], name: str, fallback: str) -> str:
    item = getattr(enum_cls, name, None)
    if isinstance(item, Enum):
        return str(item.value)
    return fallback


def merge_payload(base: dict[str, Any], overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    data = dict(base)
    if overrides:
        data.update(dict(overrides))
    return data


def last_emitted_payload(event_bus: "SpyEventBus") -> dict[str, Any]:
    assert event_bus.emitted, "Expected at least one emitted event"
    return dict(event_bus.emitted[-1].payload)


def last_emitted_topic(event_bus: "SpyEventBus") -> str:
    assert event_bus.emitted, "Expected at least one emitted event"
    return event_bus.emitted[-1].topic


def emitted_topics(event_bus: "SpyEventBus") -> list[str]:
    return [record.topic for record in event_bus.emitted]


def state_key(symbol: str = DEFAULT_SYMBOL, exchange: str = DEFAULT_EXCHANGE) -> str:
    return f"{symbol.upper()}:{exchange.lower()}"


# =============================================================================
# EventBus / Scheduler spies
# =============================================================================


@dataclass(slots=True)
class SpySubscription:
    pattern: str
    handler: Callable[..., Any]
    name: str | None = None


@dataclass(slots=True)
class EmittedEventRecord:
    topic: str
    payload: dict[str, Any]
    priority: Any = None
    source: str | None = None
    correlation_id: str | None = None
    headers: dict[str, Any] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)


class SpyEventBus:
    """
    Minimal EventBus test double.

    It intentionally keeps `_subscriptions` because the current BaseFundingStrategy
    snapshots EventBus internals during register(). Once production code stops using
    private EventBus fields, this spy can drop that compatibility attribute.
    """

    def __init__(self, *, emit_result: bool = True) -> None:
        self.emit_result = emit_result
        self.raise_on_emit: Exception | None = None

        self._subscriptions: list[SpySubscription] = []
        self.subscribed: list[SpySubscription] = []
        self.unsubscribed: list[SpySubscription] = []
        self.emitted: list[EmittedEventRecord] = []

    def subscribe(
        self,
        pattern: str,
        handler: Callable[..., Any],
        *,
        name: str | None = None,
        **_: Any,
    ) -> SpySubscription:
        subscription = SpySubscription(pattern=pattern, handler=handler, name=name)
        self._subscriptions.append(subscription)
        self.subscribed.append(subscription)
        return subscription

    def unsubscribe(self, subscription: SpySubscription) -> None:
        self.unsubscribed.append(subscription)
        if subscription in self._subscriptions:
            self._subscriptions.remove(subscription)

    async def emit(
        self,
        topic: str,
        payload: Mapping[str, Any] | None = None,
        *,
        priority: Any = None,
        source: str | None = None,
        correlation_id: str | None = None,
        headers: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> bool:
        if self.raise_on_emit is not None:
            raise self.raise_on_emit

        self.emitted.append(
            EmittedEventRecord(
                topic=topic,
                payload=dict(payload or {}),
                priority=priority,
                source=source,
                correlation_id=correlation_id,
                headers=dict(headers or {}),
                kwargs=dict(kwargs),
            )
        )
        return self.emit_result

    def clear(self) -> None:
        self.emitted.clear()
        self.subscribed.clear()
        self.unsubscribed.clear()


@dataclass(slots=True)
class SpyScheduledJob:
    job_id: str
    name: str
    func: Callable[..., Any]
    interval: float
    kwargs: dict[str, Any]
    run_immediately: bool
    max_retries: int
    retry_delay: float
    timeout: float | None
    allow_overlap: bool
    enabled: bool


class SpyScheduler:
    def __init__(self) -> None:
        self.jobs: dict[str, SpyScheduledJob] = {}
        self.added_jobs: list[SpyScheduledJob] = []
        self.removed_job_ids: list[str] = []

    def get_job_by_name(self, name: str) -> SpyScheduledJob | None:
        for job in self.jobs.values():
            if job.name == name:
                return job
        return None

    def add_interval_job(
        self,
        *,
        name: str,
        func: Callable[..., Any],
        interval: float,
        kwargs: Mapping[str, Any] | None = None,
        run_immediately: bool = False,
        max_retries: int = 0,
        retry_delay: float = 0.0,
        timeout: float | None = None,
        allow_overlap: bool = False,
        enabled: bool = True,
        **_: Any,
    ) -> str:
        job_id = f"job-{len(self.jobs) + 1}"
        job = SpyScheduledJob(
            job_id=job_id,
            name=name,
            func=func,
            interval=interval,
            kwargs=dict(kwargs or {}),
            run_immediately=run_immediately,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
            allow_overlap=allow_overlap,
            enabled=enabled,
        )
        self.jobs[job_id] = job
        self.added_jobs.append(job)
        return job_id

    def remove_job(self, job_id: str) -> None:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        self.removed_job_ids.append(job_id)
        del self.jobs[job_id]


# =============================================================================
# Event factory
# =============================================================================


def make_event(
    topic: str,
    payload: Mapping[str, Any],
    *,
    correlation_id: str = "test-correlation-id",
    priority: Any = EventPriority.NORMAL,
    source: str = "pytest",
    headers: Mapping[str, Any] | None = None,
) -> Event | SimpleNamespace:
    """
    Build core.event_bus.Event while being tolerant to small constructor differences.

    If the concrete Event signature changes, the fallback still gives strategy handlers
    the attributes they use: payload, correlation_id, topic/event_name and headers.
    """
    event_headers = dict(headers or {"test": True})

    constructor_attempts = [
        {
            "topic": topic,
            "payload": dict(payload),
            "priority": priority,
            "source": source,
            "correlation_id": correlation_id,
            "headers": event_headers,
        },
        {
            "event_type": topic,
            "payload": dict(payload),
            "priority": priority,
            "source": source,
            "correlation_id": correlation_id,
            "headers": event_headers,
        },
        {
            "name": topic,
            "payload": dict(payload),
            "priority": priority,
            "source": source,
            "correlation_id": correlation_id,
            "headers": event_headers,
        },
    ]

    try:
        signature = inspect.signature(Event)
        params = set(signature.parameters)
        for candidate in constructor_attempts:
            kwargs = {key: value for key, value in candidate.items() if key in params}
            if "payload" not in kwargs:
                continue
            try:
                return Event(**kwargs)
            except TypeError:
                continue
    except (TypeError, ValueError):
        pass

    for candidate in constructor_attempts:
        try:
            return Event(**candidate)
        except TypeError:
            continue

    try:
        return Event(topic, dict(payload))
    except TypeError:
        return SimpleNamespace(
            topic=topic,
            event_type=topic,
            name=topic,
            payload=dict(payload),
            priority=priority,
            source=source,
            correlation_id=correlation_id,
            headers=event_headers,
        )


# =============================================================================
# Payload builders
# =============================================================================


def regime_payload(
    *,
    symbol: str = DEFAULT_SYMBOL,
    exchange: str = DEFAULT_EXCHANGE,
    timeframe: Any = DEFAULT_TIMEFRAME,
    regime: Any = FundingRegime.POSITIVE,
    bias: Any = FundingBias.LONG_BIAS,
    current_rate: float = 0.0008,
    mean_rate: float = 0.0003,
    zscore: float = 2.0,
    percentile: float = 0.90,
    confidence: float = 0.85,
    changed: bool = False,
    previous_regime: Any | None = None,
    event_time: str | datetime | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    return merge_payload(
        {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": enum_value(timeframe),
            "regime": enum_value(regime),
            "bias": enum_value(bias),
            "current_rate": current_rate,
            "mean_rate": mean_rate,
            "zscore": zscore,
            "percentile": percentile,
            "confidence": confidence,
            "changed": changed,
            "previous_regime": enum_value(previous_regime) if previous_regime is not None else None,
            "event_time": event_time or iso_now(),
            "metadata": {"fixture": "regime"},
        },
        overrides,
    )


def pressure_payload(
    *,
    symbol: str = DEFAULT_SYMBOL,
    exchange: str = DEFAULT_EXCHANGE,
    timeframe: Any = DEFAULT_TIMEFRAME,
    direction: Any = FundingPressureDirection.LONG,
    level: Any = FundingPressureLevel.HIGH,
    bias: Any = FundingBias.LONG_BIAS,
    funding_rate: float = 0.0009,
    pressure_score: float = 0.82,
    oi_confirmation: bool = True,
    price_stall_confirmation: bool = True,
    squeeze_probability: float = 0.72,
    mean_reversion_probability: float = 0.68,
    event_time: str | datetime | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    return merge_payload(
        {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": enum_value(timeframe),
            "direction": enum_value(direction),
            "level": enum_value(level),
            "bias": enum_value(bias),
            "funding_rate": funding_rate,
            "pressure_score": pressure_score,
            "oi_confirmation": oi_confirmation,
            "price_stall_confirmation": price_stall_confirmation,
            "squeeze_probability": squeeze_probability,
            "mean_reversion_probability": mean_reversion_probability,
            "event_time": event_time or iso_now(),
            "metadata": {"fixture": "pressure"},
        },
        overrides,
    )


def positive_extreme_payload(
    *,
    symbol: str = DEFAULT_SYMBOL,
    exchange: str = DEFAULT_EXCHANGE,
    timeframe: Any = DEFAULT_TIMEFRAME,
    extreme_type: Any | None = None,
    regime: Any = FundingRegime.POSITIVE,
    funding_rate: float = 0.0012,
    zscore: float = 3.0,
    percentile: float = 0.98,
    severity: float = 0.90,
    is_reversal_risk: bool = True,
    is_squeeze_risk: bool = True,
    event_time: str | datetime | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    resolved_extreme_type = extreme_type or FundingExtremeType.ZSCORE_HIGH
    return merge_payload(
        {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": enum_value(timeframe),
            "extreme_type": enum_value(resolved_extreme_type),
            "regime": enum_value(regime),
            "funding_rate": funding_rate,
            "zscore": zscore,
            "percentile": percentile,
            "severity": severity,
            "is_reversal_risk": is_reversal_risk,
            "is_squeeze_risk": is_squeeze_risk,
            "event_time": event_time or iso_now(),
            "metadata": {"fixture": "positive_extreme"},
        },
        overrides,
    )


def negative_extreme_payload(
    *,
    symbol: str = DEFAULT_SYMBOL,
    exchange: str = DEFAULT_EXCHANGE,
    timeframe: Any = DEFAULT_TIMEFRAME,
    extreme_type: Any | None = None,
    regime: Any = FundingRegime.NEGATIVE,
    funding_rate: float = -0.0012,
    zscore: float = -3.0,
    percentile: float = 0.02,
    severity: float = 0.90,
    is_reversal_risk: bool = True,
    is_squeeze_risk: bool = True,
    event_time: str | datetime | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    resolved_extreme_type = extreme_type or FundingExtremeType.ZSCORE_LOW
    return merge_payload(
        {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": enum_value(timeframe),
            "extreme_type": enum_value(resolved_extreme_type),
            "regime": enum_value(regime),
            "funding_rate": funding_rate,
            "zscore": zscore,
            "percentile": percentile,
            "severity": severity,
            "is_reversal_risk": is_reversal_risk,
            "is_squeeze_risk": is_squeeze_risk,
            "event_time": event_time or iso_now(),
            "metadata": {"fixture": "negative_extreme"},
        },
        overrides,
    )


def bullish_divergence_payload(
    *,
    symbol: str = DEFAULT_SYMBOL,
    exchange: str = DEFAULT_EXCHANGE,
    timeframe: Any = DEFAULT_TIMEFRAME,
    divergence_type: Any | None = None,
    funding_rate: float = -0.0006,
    price_change_pct: float = 1.2,
    oi_change_pct: float = 2.4,
    cvd_change: float = 100_000.0,
    long_liquidations: float = 5_000.0,
    short_liquidations: float = 45_000.0,
    confidence: float = 0.82,
    event_time: str | datetime | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    resolved_type = divergence_type or FundingDivergenceType.PRICE_UP_FUNDING_DOWN
    return merge_payload(
        {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": enum_value(timeframe),
            "divergence_type": enum_value(resolved_type),
            "funding_rate": funding_rate,
            "price_change_pct": price_change_pct,
            "oi_change_pct": oi_change_pct,
            "cvd_change": cvd_change,
            "long_liquidations": long_liquidations,
            "short_liquidations": short_liquidations,
            "confidence": confidence,
            "event_time": event_time or iso_now(),
            "metadata": {"fixture": "bullish_divergence"},
        },
        overrides,
    )


def bearish_divergence_payload(
    *,
    symbol: str = DEFAULT_SYMBOL,
    exchange: str = DEFAULT_EXCHANGE,
    timeframe: Any = DEFAULT_TIMEFRAME,
    divergence_type: Any | None = None,
    funding_rate: float = 0.0008,
    price_change_pct: float = -1.2,
    oi_change_pct: float = 2.4,
    cvd_change: float = -100_000.0,
    long_liquidations: float = 45_000.0,
    short_liquidations: float = 5_000.0,
    confidence: float = 0.82,
    event_time: str | datetime | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    resolved_type = divergence_type or FundingDivergenceType.PRICE_DOWN_FUNDING_UP
    return merge_payload(
        {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": enum_value(timeframe),
            "divergence_type": enum_value(resolved_type),
            "funding_rate": funding_rate,
            "price_change_pct": price_change_pct,
            "oi_change_pct": oi_change_pct,
            "cvd_change": cvd_change,
            "long_liquidations": long_liquidations,
            "short_liquidations": short_liquidations,
            "confidence": confidence,
            "event_time": event_time or iso_now(),
            "metadata": {"fixture": "bearish_divergence"},
        },
        overrides,
    )


def positive_to_negative_flip_payload(
    *,
    symbol: str = DEFAULT_SYMBOL,
    exchange: str = DEFAULT_EXCHANGE,
    timeframe: Any = DEFAULT_TIMEFRAME,
    previous_rate: float = 0.0008,
    current_rate: float = -0.0002,
    flip_magnitude: float = 0.0010,
    confidence: float = 0.82,
    event_time: str | datetime | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    return merge_payload(
        {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": enum_value(timeframe),
            "flip_type": enum_value(FundingFlipType.POSITIVE_TO_NEGATIVE),
            "previous_rate": previous_rate,
            "current_rate": current_rate,
            "flip_magnitude": flip_magnitude,
            "confidence": confidence,
            "event_time": event_time or iso_now(),
            "metadata": {"fixture": "positive_to_negative_flip"},
        },
        overrides,
    )


def negative_to_positive_flip_payload(
    *,
    symbol: str = DEFAULT_SYMBOL,
    exchange: str = DEFAULT_EXCHANGE,
    timeframe: Any = DEFAULT_TIMEFRAME,
    previous_rate: float = -0.0008,
    current_rate: float = 0.0002,
    flip_magnitude: float = 0.0010,
    confidence: float = 0.82,
    event_time: str | datetime | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    return merge_payload(
        {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": enum_value(timeframe),
            "flip_type": enum_value(FundingFlipType.NEGATIVE_TO_POSITIVE),
            "previous_rate": previous_rate,
            "current_rate": current_rate,
            "flip_magnitude": flip_magnitude,
            "confidence": confidence,
            "event_time": event_time or iso_now(),
            "metadata": {"fixture": "negative_to_positive_flip"},
        },
        overrides,
    )


def funding_signal_payload(
    *,
    symbol: str = DEFAULT_SYMBOL,
    exchange: str = DEFAULT_EXCHANGE,
    timeframe: Any = DEFAULT_TIMEFRAME,
    signal_type: Any = FundingSignalType.REVERSION_SETUP,
    bias: Any = FundingBias.NEUTRAL,
    regime: Any = FundingRegime.UNKNOWN,
    score: float = 0.70,
    confidence: float = 0.80,
    description: str = "pytest funding signal",
    supporting_factors: list[str] | None = None,
    tags: list[str] | None = None,
    event_time: str | datetime | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    return merge_payload(
        {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": enum_value(timeframe),
            "signal_type": enum_value(signal_type),
            "bias": enum_value(bias),
            "regime": enum_value(regime),
            "score": score,
            "confidence": confidence,
            "description": description,
            "supporting_factors": supporting_factors or ["pytest"],
            "tags": tags or ["pytest"],
            "event_time": event_time or iso_now(),
            "metadata": {"fixture": "funding_signal"},
        },
        overrides,
    )


def funding_updated_payload(
    *,
    symbol: str = DEFAULT_SYMBOL,
    exchange: str = DEFAULT_EXCHANGE,
    snapshot: Mapping[str, Any] | None = None,
    statistics: Mapping[str, Any] | None = None,
    regime_state: Mapping[str, Any] | None = None,
    pressure_state: Mapping[str, Any] | None = None,
    extreme_event: Mapping[str, Any] | None = None,
    divergence_event: Mapping[str, Any] | None = None,
    flip_event: Mapping[str, Any] | None = None,
    event_time: str | datetime | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    payload = {
        "symbol": symbol,
        "exchange": exchange,
        "event_time": event_time or iso_now(),
        "payload": {
            "symbol": symbol,
            "exchange": exchange,
            "snapshot": dict(snapshot or {}),
            "statistics": dict(statistics or {}),
            "regime_state": dict(regime_state or {}),
            "pressure_state": dict(pressure_state or {}),
            "extreme_event": dict(extreme_event or {}),
            "divergence_event": dict(divergence_event or {}),
            "flip_event": dict(flip_event or {}),
        },
    }
    return merge_payload(payload, overrides)


# =============================================================================
# Event fixtures
# =============================================================================


@pytest.fixture
def event_bus_spy() -> SpyEventBus:
    return SpyEventBus()


@pytest.fixture
def scheduler_spy() -> SpyScheduler:
    return SpyScheduler()


@pytest.fixture
def make_test_event() -> Callable[..., Event | SimpleNamespace]:
    return make_event


@pytest.fixture
def make_regime_event() -> Callable[..., Event | SimpleNamespace]:
    def factory(**kwargs: Any) -> Event | SimpleNamespace:
        return make_event("analytics.funding.regime", regime_payload(**kwargs))

    return factory


@pytest.fixture
def make_pressure_event() -> Callable[..., Event | SimpleNamespace]:
    def factory(**kwargs: Any) -> Event | SimpleNamespace:
        return make_event("analytics.funding.pressure", pressure_payload(**kwargs))

    return factory


@pytest.fixture
def make_positive_extreme_event() -> Callable[..., Event | SimpleNamespace]:
    def factory(**kwargs: Any) -> Event | SimpleNamespace:
        return make_event("analytics.funding.extreme", positive_extreme_payload(**kwargs))

    return factory


@pytest.fixture
def make_negative_extreme_event() -> Callable[..., Event | SimpleNamespace]:
    def factory(**kwargs: Any) -> Event | SimpleNamespace:
        return make_event("analytics.funding.extreme", negative_extreme_payload(**kwargs))

    return factory


@pytest.fixture
def make_bullish_divergence_event() -> Callable[..., Event | SimpleNamespace]:
    def factory(**kwargs: Any) -> Event | SimpleNamespace:
        return make_event("analytics.funding.divergence", bullish_divergence_payload(**kwargs))

    return factory


@pytest.fixture
def make_bearish_divergence_event() -> Callable[..., Event | SimpleNamespace]:
    def factory(**kwargs: Any) -> Event | SimpleNamespace:
        return make_event("analytics.funding.divergence", bearish_divergence_payload(**kwargs))

    return factory


@pytest.fixture
def make_positive_to_negative_flip_event() -> Callable[..., Event | SimpleNamespace]:
    def factory(**kwargs: Any) -> Event | SimpleNamespace:
        return make_event("analytics.funding.flip", positive_to_negative_flip_payload(**kwargs))

    return factory


@pytest.fixture
def make_negative_to_positive_flip_event() -> Callable[..., Event | SimpleNamespace]:
    def factory(**kwargs: Any) -> Event | SimpleNamespace:
        return make_event("analytics.funding.flip", negative_to_positive_flip_payload(**kwargs))

    return factory


@pytest.fixture
def make_funding_signal_event() -> Callable[..., Event | SimpleNamespace]:
    def factory(**kwargs: Any) -> Event | SimpleNamespace:
        return make_event("analytics.funding.signal", funding_signal_payload(**kwargs))

    return factory


@pytest.fixture
def make_funding_updated_event() -> Callable[..., Event | SimpleNamespace]:
    def factory(**kwargs: Any) -> Event | SimpleNamespace:
        return make_event("analytics.funding.updated", funding_updated_payload(**kwargs))

    return factory


# =============================================================================
# Config fixtures
# =============================================================================


@pytest.fixture
def base_strategy_config() -> BaseFundingStrategyConfig:
    return BaseFundingStrategyConfig(
        setup_ttl_sec=120.0,
        cooldown_sec=30.0,
        event_stale_after_sec=300.0,
        state_lock_timeout_sec=0.05,
        enable_scheduler_cleanup=True,
        cleanup_interval_sec=10.0,
        cleanup_job_timeout_sec=2.0,
        strategy_namespace="strategy.funding.test",
        source_name="pytest_funding_strategy",
        service_name="pytest_funding_strategy",
        enable_funding_updated_subscription=True,
        enable_funding_signal_subscription=True,
    )


@pytest.fixture
def extreme_reversal_config() -> FundingExtremeReversalStrategyConfig:
    return FundingExtremeReversalStrategyConfig(
        setup_ttl_sec=120.0,
        cooldown_sec=30.0,
        event_stale_after_sec=300.0,
        state_lock_timeout_sec=0.05,
        cleanup_interval_sec=10.0,
        cleanup_job_timeout_sec=2.0,
        min_extreme_severity=0.60,
        min_pressure_score=0.55,
        min_regime_confidence=0.15,
        min_mean_reversion_probability=0.50,
        min_squeeze_probability=0.50,
        min_divergence_confidence=0.45,
        min_signal_confidence=0.45,
        min_signal_abs_score=0.35,
    )


@pytest.fixture
def divergence_config() -> FundingDivergenceStrategyConfig:
    return FundingDivergenceStrategyConfig(
        setup_ttl_sec=120.0,
        cooldown_sec=30.0,
        event_stale_after_sec=300.0,
        state_lock_timeout_sec=0.05,
        cleanup_interval_sec=10.0,
        cleanup_job_timeout_sec=2.0,
        min_divergence_confidence=0.50,
        min_pressure_score=0.35,
        min_regime_confidence=0.10,
        min_extreme_severity=0.45,
        min_signal_confidence=0.45,
        min_signal_abs_score=0.30,
        require_non_neutral_regime=True,
        require_pressure_alignment=False,
        require_pressure_present=False,
    )


# =============================================================================
# Strategy fixtures
# =============================================================================


@pytest.fixture
def extreme_reversal_strategy(
    event_bus_spy: SpyEventBus,
    scheduler_spy: SpyScheduler,
    extreme_reversal_config: FundingExtremeReversalStrategyConfig,
) -> FundingExtremeReversalStrategy:
    return FundingExtremeReversalStrategy(
        event_bus=event_bus_spy,  # type: ignore[arg-type]
        scheduler=scheduler_spy,  # type: ignore[arg-type]
        config=extreme_reversal_config,
    )


@pytest.fixture
def divergence_strategy(
    event_bus_spy: SpyEventBus,
    scheduler_spy: SpyScheduler,
    divergence_config: FundingDivergenceStrategyConfig,
) -> FundingDivergenceStrategy:
    return FundingDivergenceStrategy(
        event_bus=event_bus_spy,  # type: ignore[arg-type]
        scheduler=scheduler_spy,  # type: ignore[arg-type]
        config=divergence_config,
    )


# =============================================================================
# Scenario fixtures
# =============================================================================


@pytest.fixture
def crowded_longs_context() -> dict[str, dict[str, Any]]:
    return {
        "regime": regime_payload(
            regime=FundingRegime.POSITIVE,
            bias=FundingBias.OVERCROWDED_LONGS,
            confidence=0.86,
        ),
        "pressure": pressure_payload(
            direction=FundingPressureDirection.LONG,
            level=FundingPressureLevel.HIGH,
            bias=FundingBias.OVERCROWDED_LONGS,
            pressure_score=0.84,
            squeeze_probability=0.74,
            mean_reversion_probability=0.70,
        ),
        "extreme": positive_extreme_payload(severity=0.91),
    }


@pytest.fixture
def crowded_shorts_context() -> dict[str, dict[str, Any]]:
    return {
        "regime": regime_payload(
            regime=FundingRegime.NEGATIVE,
            bias=FundingBias.OVERCROWDED_SHORTS,
            confidence=0.86,
            current_rate=-0.0008,
            mean_rate=-0.0003,
            zscore=-2.0,
            percentile=0.10,
        ),
        "pressure": pressure_payload(
            direction=FundingPressureDirection.SHORT,
            level=FundingPressureLevel.HIGH,
            bias=FundingBias.OVERCROWDED_SHORTS,
            funding_rate=-0.0009,
            pressure_score=0.84,
            squeeze_probability=0.74,
            mean_reversion_probability=0.70,
        ),
        "extreme": negative_extreme_payload(severity=0.91),
    }


@pytest.fixture
def bullish_divergence_context() -> dict[str, dict[str, Any]]:
    return {
        "regime": regime_payload(
            regime=FundingRegime.NEGATIVE,
            bias=FundingBias.OVERCROWDED_SHORTS,
            confidence=0.82,
            current_rate=-0.0007,
            mean_rate=-0.0002,
            zscore=-2.1,
            percentile=0.08,
        ),
        "pressure": pressure_payload(
            direction=FundingPressureDirection.SHORT,
            level=FundingPressureLevel.HIGH,
            bias=FundingBias.OVERCROWDED_SHORTS,
            funding_rate=-0.0008,
            pressure_score=0.72,
            squeeze_probability=0.62,
            mean_reversion_probability=0.64,
        ),
        "divergence": bullish_divergence_payload(confidence=0.84),
    }


@pytest.fixture
def bearish_divergence_context() -> dict[str, dict[str, Any]]:
    return {
        "regime": regime_payload(
            regime=FundingRegime.POSITIVE,
            bias=FundingBias.OVERCROWDED_LONGS,
            confidence=0.82,
            current_rate=0.0007,
            mean_rate=0.0002,
            zscore=2.1,
            percentile=0.92,
        ),
        "pressure": pressure_payload(
            direction=FundingPressureDirection.LONG,
            level=FundingPressureLevel.HIGH,
            bias=FundingBias.OVERCROWDED_LONGS,
            funding_rate=0.0008,
            pressure_score=0.72,
            squeeze_probability=0.62,
            mean_reversion_probability=0.64,
        ),
        "divergence": bearish_divergence_payload(confidence=0.84),
    }


# =============================================================================
# Assertion helpers
# =============================================================================


@pytest.fixture
def assert_last_event() -> Callable[..., EmittedEventRecord]:
    def assertion(
        event_bus: SpyEventBus,
        *,
        topic: str | None = None,
        event_kind: str | None = None,
        symbol: str = DEFAULT_SYMBOL,
        exchange: str = DEFAULT_EXCHANGE,
    ) -> EmittedEventRecord:
        assert event_bus.emitted, "Expected at least one emitted event"

        record = event_bus.emitted[-1]
        if topic is not None:
            assert record.topic == topic

        payload = record.payload
        assert payload["symbol"] == symbol.upper()
        assert payload["exchange"] == exchange.lower()

        if event_kind is not None:
            assert payload["event_kind"] == event_kind

        return record

    return assertion


@pytest.fixture
def assert_state_status() -> Callable[..., FundingStrategyState]:
    def assertion(
        strategy: Any,
        *,
        status: FundingSetupStatus,
        symbol: str = DEFAULT_SYMBOL,
        exchange: str = DEFAULT_EXCHANGE,
        direction: FundingStrategyDirection | None = None,
    ) -> FundingStrategyState:
        state = strategy.get_state(symbol, exchange)
        assert state.status == status

        if direction is not None:
            assert state.direction == direction

        return state

    return assertion