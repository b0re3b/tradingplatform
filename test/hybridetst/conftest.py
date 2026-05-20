# tests/strategy/conftest.py

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest

from core.event_bus import EventPriority

from strategy.base import BaseStrategy
from strategy.config import (
    BuilderConfig,
    ConfluenceConfig,
    FeatureFreshnessConfig,
    FilterConfig,
    PortfolioCoordinatorConfig,
    PresetConfig,
    RoutingConfig,
    StrategyConfig,
    StrategyDefinitionConfig,
    StrategyRuntimeConfig,
)
from strategy.enums import (
    EntryType,
    ExitType,
    FeatureSource,
    MarketRegime,
    PresetMode,
    SetupType,
    SignalPriority,
    SignalSide,
    SignalStatus,
    StrategyCategory,
    Timeframe,
    TriggerType,
)
from strategy.exceptions import StrategyEvaluationError
from strategy.models import (
    EntryPlan,
    ExecutionPlanDraft,
    ExitPlan,
    FeatureSnapshot,
    InvalidationPlan,
    StrategyContext,
    StrategyEvaluation,
    StrategySignal,
    TargetPlan,
    ensure_aware_utc,
    utcnow,
)


EventHandler = Callable[..., Awaitable[None]] | Callable[..., None]


# =============================================================================
# Mock EventBus
# =============================================================================


@dataclass(slots=True)
class MockSubscription:
    topic: str
    handler: EventHandler
    kwargs: dict[str, Any] = field(default_factory=dict)
    active: bool = True


@dataclass(slots=True)
class MockEvent:
    topic: str
    payload: dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL
    source: str | None = None
    timestamp: datetime = field(default_factory=utcnow)
    kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.topic

    @property
    def event_name(self) -> str:
        return self.topic

    @property
    def type(self) -> str:
        return self.topic


class MockEventBus:
    """
    Мінімальний, але строгий EventBus double для strategy tests.

    Підтримує:
    - emit();
    - publish();
    - publish_nowait_best_effort();
    - subscribe();
    - unsubscribe();
    - dispatch helpers для ручного прогону handlers;
    - failure injection для emit/subscribe/unsubscribe.
    """

    def __init__(self) -> None:
        self.emitted: list[MockEvent] = []
        self.published: list[MockEvent] = []
        self.nowait: list[MockEvent] = []

        self.subscriptions: list[MockSubscription] = []
        self.unsubscribed: list[MockSubscription] = []

        self.emit_calls: int = 0
        self.publish_calls: int = 0
        self.nowait_calls: int = 0
        self.subscribe_calls: int = 0
        self.unsubscribe_calls: int = 0

        self.fail_emit: bool = False
        self.fail_publish: bool = False
        self.fail_nowait: bool = False
        self.fail_subscribe: bool = False
        self.fail_unsubscribe: bool = False

        self.emit_error: Exception = RuntimeError("mock emit failure")
        self.publish_error: Exception = RuntimeError("mock publish failure")
        self.nowait_error: Exception = RuntimeError("mock nowait failure")
        self.subscribe_error: Exception = RuntimeError("mock subscribe failure")
        self.unsubscribe_error: Exception = RuntimeError("mock unsubscribe failure")

    async def emit(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: EventPriority = EventPriority.NORMAL,
        source: str | None = None,
        **kwargs: Any,
    ) -> MockEvent:
        self.emit_calls += 1

        if self.fail_emit:
            raise self.emit_error

        event = MockEvent(
            topic=topic,
            payload=dict(payload),
            priority=priority,
            source=source,
            kwargs=dict(kwargs),
        )
        self.emitted.append(event)
        return event

    async def publish(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: EventPriority = EventPriority.NORMAL,
        source: str | None = None,
        **kwargs: Any,
    ) -> MockEvent:
        self.publish_calls += 1

        if self.fail_publish:
            raise self.publish_error

        event = MockEvent(
            topic=topic,
            payload=dict(payload),
            priority=priority,
            source=source,
            kwargs=dict(kwargs),
        )
        self.published.append(event)
        return event

    def publish_nowait_best_effort(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: EventPriority = EventPriority.NORMAL,
        source: str | None = None,
        **kwargs: Any,
    ) -> MockEvent:
        self.nowait_calls += 1

        if self.fail_nowait:
            raise self.nowait_error

        event = MockEvent(
            topic=topic,
            payload=dict(payload),
            priority=priority,
            source=source,
            kwargs=dict(kwargs),
        )
        self.nowait.append(event)
        return event

    def subscribe(
        self,
        topic: str,
        handler: EventHandler,
        **kwargs: Any,
    ) -> MockSubscription:
        self.subscribe_calls += 1

        if self.fail_subscribe:
            raise self.subscribe_error

        subscription = MockSubscription(
            topic=topic,
            handler=handler,
            kwargs=dict(kwargs),
        )
        self.subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, subscription: MockSubscription) -> None:
        self.unsubscribe_calls += 1

        if self.fail_unsubscribe:
            raise self.unsubscribe_error

        subscription.active = False
        self.unsubscribed.append(subscription)

    def topic_emitted(self, topic: str) -> bool:
        return any(event.topic == topic for event in self.emitted)

    def nowait_topic_emitted(self, topic: str) -> bool:
        return any(event.topic == topic for event in self.nowait)

    def emitted_payloads(self, topic: str) -> list[dict[str, Any]]:
        return [event.payload for event in self.emitted if event.topic == topic]

    def nowait_payloads(self, topic: str) -> list[dict[str, Any]]:
        return [event.payload for event in self.nowait if event.topic == topic]

    async def dispatch(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: EventPriority = EventPriority.NORMAL,
        source: str | None = "test",
        **kwargs: Any,
    ) -> MockEvent:
        """
        Ручний dispatch у підписані handlers.

        Корисно для engine/event_handler тестів, де треба перевірити, що
        subscriptions реально викликаються.
        """
        event = MockEvent(
            topic=topic,
            payload=dict(payload),
            priority=priority,
            source=source,
            kwargs=dict(kwargs),
        )

        for subscription in list(self.subscriptions):
            if not subscription.active:
                continue
            if subscription.topic != topic and subscription.topic != "*":
                continue

            result = subscription.handler(event)
            if asyncio.iscoroutine(result):
                await result

        return event

    def reset(self) -> None:
        self.emitted.clear()
        self.published.clear()
        self.nowait.clear()
        self.subscriptions.clear()
        self.unsubscribed.clear()

        self.emit_calls = 0
        self.publish_calls = 0
        self.nowait_calls = 0
        self.subscribe_calls = 0
        self.unsubscribe_calls = 0

        self.fail_emit = False
        self.fail_publish = False
        self.fail_nowait = False
        self.fail_subscribe = False
        self.fail_unsubscribe = False


# =============================================================================
# Mock Scheduler
# =============================================================================


@dataclass(slots=True)
class MockScheduledJob:
    job_id: str
    name: str
    callback: Callable[..., Any]
    interval_seconds: float | None = None
    delay_seconds: float | None = None
    enabled: bool = True
    kwargs: dict[str, Any] = field(default_factory=dict)
    run_count: int = 0

    async def run(self) -> Any:
        self.run_count += 1
        result = self.callback()
        if asyncio.iscoroutine(result):
            return await result
        return result


class MockScheduler:
    """
    Scheduler double для перевірки того, що strategy layer не створює unmanaged loops.
    """

    def __init__(self) -> None:
        self.jobs: dict[str, MockScheduledJob] = {}
        self.added_jobs: list[MockScheduledJob] = []
        self.removed_jobs: list[str] = []
        self.enabled_jobs: list[str] = []
        self.disabled_jobs: list[str] = []

        self._counter: int = 0

        self.fail_add: bool = False
        self.fail_remove: bool = False
        self.add_error: Exception = RuntimeError("mock scheduler add failure")
        self.remove_error: Exception = RuntimeError("mock scheduler remove failure")

    def add_interval_job(
        self,
        *,
        name: str,
        callback: Callable[..., Any],
        interval_seconds: float,
        **kwargs: Any,
    ) -> MockScheduledJob:
        if self.fail_add:
            raise self.add_error

        self._counter += 1
        job = MockScheduledJob(
            job_id=f"interval-{self._counter}",
            name=name,
            callback=callback,
            interval_seconds=interval_seconds,
            kwargs=dict(kwargs),
        )
        self.jobs[job.job_id] = job
        self.added_jobs.append(job)
        return job

    def add_delayed_job(
        self,
        *,
        name: str,
        callback: Callable[..., Any],
        delay_seconds: float,
        **kwargs: Any,
    ) -> MockScheduledJob:
        if self.fail_add:
            raise self.add_error

        self._counter += 1
        job = MockScheduledJob(
            job_id=f"delayed-{self._counter}",
            name=name,
            callback=callback,
            delay_seconds=delay_seconds,
            kwargs=dict(kwargs),
        )
        self.jobs[job.job_id] = job
        self.added_jobs.append(job)
        return job

    def remove_job(self, job_or_id: MockScheduledJob | str) -> None:
        if self.fail_remove:
            raise self.remove_error

        job_id = job_or_id.job_id if isinstance(job_or_id, MockScheduledJob) else job_or_id
        self.jobs.pop(job_id, None)
        self.removed_jobs.append(job_id)

    def enable_job(self, job_or_id: MockScheduledJob | str) -> None:
        job_id = job_or_id.job_id if isinstance(job_or_id, MockScheduledJob) else job_or_id
        if job_id in self.jobs:
            self.jobs[job_id].enabled = True
        self.enabled_jobs.append(job_id)

    def disable_job(self, job_or_id: MockScheduledJob | str) -> None:
        job_id = job_or_id.job_id if isinstance(job_or_id, MockScheduledJob) else job_or_id
        if job_id in self.jobs:
            self.jobs[job_id].enabled = False
        self.disabled_jobs.append(job_id)

    def get_job(self, job_id: str) -> MockScheduledJob | None:
        return self.jobs.get(job_id)

    def list_jobs(self) -> list[MockScheduledJob]:
        return list(self.jobs.values())

    def reset(self) -> None:
        self.jobs.clear()
        self.added_jobs.clear()
        self.removed_jobs.clear()
        self.enabled_jobs.clear()
        self.disabled_jobs.clear()
        self._counter = 0
        self.fail_add = False
        self.fail_remove = False


# =============================================================================
# Test strategies
# =============================================================================


class DummyStrategy(BaseStrategy):
    category = StrategyCategory.ORDERFLOW
    default_setup_type = SetupType.CVD_DIVERGENCE
    default_timeframe = Timeframe.M1
    default_trigger_type = TriggerType.PRIMARY

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: MockEventBus | None = None,
        scheduler: MockScheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        side: SignalSide = SignalSide.LONG,
        confidence: float = 0.82,
        score: float = 0.75,
        should_return_signal: bool = True,
        service_name: str | None = None,
    ) -> None:
        self.generated_contexts: list[StrategyContext] = []
        self.generate_calls: int = 0

        self.side = side
        self.generated_confidence = confidence
        self.generated_score = score
        self.should_return_signal = should_return_signal

        super().__init__(
            config=config,
            event_bus=event_bus,  # type: ignore[arg-type]
            scheduler=scheduler,  # type: ignore[arg-type]
            definition=definition,
            service_name=service_name,
        )

    async def generate_signal(self, context: StrategyContext) -> StrategySignal | None:
        self.generate_calls += 1
        self.generated_contexts.append(context)

        if not self.should_return_signal:
            return None

        return self.build_signal(
            context=context,
            side=self.side,
            confidence=self.generated_confidence,
            score=self.generated_score,
            reasons=["dummy_signal"],
            confirmations=["dummy_confirmation"],
            source_features=sorted(self.required_features()),
            metadata={
                "test_strategy": True,
                "exchange": "binance",
                "market_type": "usdm_futures",
            },
        )


class ShortDummyStrategy(DummyStrategy):
    category = StrategyCategory.ORDERFLOW

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("side", SignalSide.SHORT)
        super().__init__(*args, **kwargs)


class FlatDummyStrategy(DummyStrategy):
    category = StrategyCategory.ORDERFLOW

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("side", SignalSide.FLAT)
        super().__init__(*args, **kwargs)


class NoSignalStrategy(DummyStrategy):
    category = StrategyCategory.ORDERFLOW

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("should_return_signal", False)
        super().__init__(*args, **kwargs)


class FailingStrategy(BaseStrategy):
    category = StrategyCategory.ORDERFLOW
    default_setup_type = SetupType.CVD_DIVERGENCE
    default_timeframe = Timeframe.M1
    default_trigger_type = TriggerType.PRIMARY

    async def generate_signal(self, context: StrategyContext) -> StrategySignal | None:
        raise StrategyEvaluationError("intentional strategy failure")


# =============================================================================
# Config factories
# =============================================================================


def make_runtime_config(
    *,
    enabled: bool = True,
    symbols: Iterable[str] | None = ("BTCUSDT",),
    timeframes: Iterable[Timeframe] | None = (Timeframe.M1, Timeframe.M5, Timeframe.M15),
    allowed_regimes: Iterable[MarketRegime] | None = (MarketRegime.UNKNOWN,),
    cooldown_seconds: int = 0,
    max_signal_age_seconds: int = 60,
    min_confidence: float = 0.5,
    min_score: float = 0.0,
) -> StrategyRuntimeConfig:
    runtime = StrategyRuntimeConfig(
        enabled=enabled,
        symbols=list(symbols or []),
        timeframes=list(timeframes or [Timeframe.M1]),
        allowed_regimes=list(allowed_regimes or [MarketRegime.UNKNOWN]),
        cooldown_seconds=cooldown_seconds,
        max_signal_age_seconds=max_signal_age_seconds,
        min_confidence=min_confidence,
        min_score=min_score,
    )
    runtime.validate()
    return runtime


def make_definition_config(
    *,
    name: str = "dummy_strategy",
    category: StrategyCategory = StrategyCategory.ORDERFLOW,
    runtime: StrategyRuntimeConfig | None = None,
    required_features: Iterable[str] | None = ("orderflow_imbalance",),
    weight: float = 1.0,
    priority: int = 10,
    tags: Iterable[str] | None = ("unit", "test", "futures"),
    metadata: dict[str, object] | None = None,
) -> StrategyDefinitionConfig:
    definition = StrategyDefinitionConfig(
        name=name,
        category=category,
        runtime=runtime or make_runtime_config(),
        required_features=set(required_features or set()),
        weight=weight,
        priority=priority,
        tags=list(tags or []),
        metadata=dict(metadata or {"source": "tests"}),
    )
    definition.validate()
    return definition


def make_strategy_config(
    *,
    definitions: Iterable[StrategyDefinitionConfig] | None = None,
    enabled_strategy_names: Iterable[str] | None = None,
    runtime: StrategyRuntimeConfig | None = None,
    confluence_enabled: bool = True,
    confluence_min_agreement_count: int = 1,
    confluence_min_confidence: float = 0.5,
    confluence_min_score: float = 0.0,
) -> StrategyConfig:
    strategy_definitions = list(definitions or [make_definition_config()])

    config = StrategyConfig(
        runtime=runtime or make_runtime_config(symbols=[]),
        routing=RoutingConfig(
            reevaluate_on_any_update=False,
            route_hybrid_on_domain_signal=True,
            allow_partial_context=True,
            stale_feature_threshold_seconds=60,
            event_to_categories={
                "analytics.orderflow.updated": [StrategyCategory.ORDERFLOW],
                "analytics.open_interest.updated": [StrategyCategory.OPEN_INTEREST],
                "analytics.funding.updated": [StrategyCategory.FUNDING],
                "analytics.hybrid.updated": [StrategyCategory.HYBRID],
            },
        ),
        confluence=ConfluenceConfig(
            enabled=confluence_enabled,
            min_agreement_count=confluence_min_agreement_count,
            min_confidence=confluence_min_confidence,
            min_score=confluence_min_score,
            conflict_penalty=0.15,
            confirmation_bonus=0.10,
            max_strategies_per_side=10,
        ),
        filters=FilterConfig(),
        builders=BuilderConfig(),
        freshness=FeatureFreshnessConfig(
            default_ttl_seconds=60,
            per_feature_ttl_seconds={
                "orderflow_imbalance": 60,
                "cvd_delta": 60,
                "open_interest": 120,
                "liquidity_score": 30,
            },
        ),
        portfolio=PortfolioCoordinatorConfig(),
        preset=PresetConfig(
            mode=PresetMode.INTRADAY,
            enabled_strategy_names=list(
                enabled_strategy_names
                if enabled_strategy_names is not None
                else [definition.name for definition in strategy_definitions]
            ),
            metadata={"source": "tests"},
        ),
        strategies={definition.name: definition for definition in strategy_definitions},
    )
    config.validate()
    return config


# =============================================================================
# Model factories
# =============================================================================


def make_feature_snapshot(
    *,
    name: str = "orderflow_imbalance",
    value: Any = 0.72,
    source: FeatureSource = FeatureSource.ORDERFLOW,
    symbol: str = "BTCUSDT",
    timestamp: datetime | None = None,
    confidence: float = 0.85,
    normalized_value: float | None = 0.72,
    freshness_seconds: float | None = 60.0,
    metadata: dict[str, Any] | None = None,
) -> FeatureSnapshot:
    feature = FeatureSnapshot(
        name=name,
        value=value,
        source=source,
        symbol=symbol,
        timestamp=timestamp or utcnow(),
        confidence=confidence,
        normalized_value=normalized_value,
        freshness_seconds=freshness_seconds,
        metadata=dict(metadata or {}),
    )
    feature.validate()
    return feature


def make_strategy_context(
    *,
    symbol: str = "BTCUSDT",
    timestamp: datetime | None = None,
    timeframe: Timeframe = Timeframe.M1,
    features: Iterable[FeatureSnapshot] | None = None,
    domain_data: dict[FeatureSource, dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> StrategyContext:
    context = StrategyContext(
        symbol=symbol,
        timestamp=ensure_aware_utc(timestamp or utcnow()),
        timeframe=timeframe,
        metadata=dict(metadata or {"source": "tests"}),
    )

    for feature in features or [make_feature_snapshot(symbol=symbol)]:
        context.put_feature(feature)
        if feature.freshness_seconds is not None:
            context.freshness_map[feature.name] = feature.freshness_seconds

    for source, values in (domain_data or {}).items():
        context.domain_dict(source).update(dict(values))

    context.validate()
    return context


def make_strategy_signal(
    *,
    symbol: str = "BTCUSDT",
    side: SignalSide = SignalSide.LONG,
    strategy_name: str = "dummy_strategy",
    category: StrategyCategory = StrategyCategory.ORDERFLOW,
    timeframe: Timeframe = Timeframe.M1,
    setup_type: SetupType = SetupType.CVD_DIVERGENCE,
    timestamp: datetime | None = None,
    confidence: float = 0.82,
    score: float = 0.75,
    status: SignalStatus = SignalStatus.NEW,
    with_execution_plan: bool = False,
    metadata: dict[str, Any] | None = None,
) -> StrategySignal:
    signal = StrategySignal(
        symbol=symbol,
        side=side,
        strategy_name=strategy_name,
        category=category,
        timeframe=timeframe,
        setup_type=setup_type,
        timestamp=timestamp or utcnow(),
        confidence=confidence,
        score=score,
        status=status,
        trigger_type=TriggerType.PRIMARY,
        priority=SignalPriority.MEDIUM,
        reasons=["test_signal"],
        confirmations=["test_confirmation"],
        source_features=["orderflow_imbalance"],
        metadata={
            "source": "tests",
            "exchange": "binance",
            "market_type": "usdm_futures",
            **dict(metadata or {}),
        },
    )

    if with_execution_plan:
        attach_execution_plan(signal)

    signal.validate()
    return signal


def attach_execution_plan(
    signal: StrategySignal,
    *,
    entry_price: float = 100.0,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    leverage: float = 2.0,
) -> StrategySignal:
    if signal.side is SignalSide.LONG:
        resolved_stop = stop_loss if stop_loss is not None else entry_price * 0.99
        resolved_target = take_profit if take_profit is not None else entry_price * 1.02
    elif signal.side is SignalSide.SHORT:
        resolved_stop = stop_loss if stop_loss is not None else entry_price * 1.01
        resolved_target = take_profit if take_profit is not None else entry_price * 0.98
    else:
        resolved_stop = stop_loss if stop_loss is not None else entry_price * 0.99
        resolved_target = take_profit if take_profit is not None else entry_price * 1.02

    entry = EntryPlan(
        entry_type=EntryType.LIMIT,
        price=entry_price,
        timeout_seconds=30,
        max_slippage_bps=5.0,
        confirmation_required=False,
    )
    invalidation = InvalidationPlan(
        price=resolved_stop,
        reason="test_invalidation",
        timeout_seconds=300,
    )
    exit_plan = ExitPlan(
        exit_types=[ExitType.TAKE_PROFIT, ExitType.STOP_LOSS],
        stop_loss=resolved_stop,
        take_profit_levels=[
            TargetPlan(
                price=resolved_target,
                size_fraction=1.0,
                rr=2.0,
                label="tp1",
            )
        ],
        max_holding_seconds=3600,
    )

    signal.entry_plan = entry
    signal.invalidation_plan = invalidation
    signal.exit_plan = exit_plan
    signal.execution_plan = ExecutionPlanDraft(
        symbol=signal.symbol,
        side=signal.side,
        entry=entry,
        exit=exit_plan,
        invalidation=invalidation,
        leverage=leverage,
        reduce_only=False,
        post_only=False,
        expected_holding_seconds=1800,
        metadata={"source": "tests"},
    )
    signal.validate()
    return signal


def make_strategy_evaluation(
    *,
    signal: StrategySignal | None = None,
    strategy_name: str = "dummy_strategy",
    symbol: str = "BTCUSDT",
    timestamp: datetime | None = None,
    passed: bool = True,
    score: float = 0.75,
    confidence: float = 0.82,
    reasons: Iterable[str] | None = ("test_evaluation",),
    metadata: dict[str, Any] | None = None,
) -> StrategyEvaluation:
    evaluation = StrategyEvaluation(
        strategy_name=strategy_name,
        symbol=symbol,
        timestamp=timestamp or utcnow(),
        signal=signal,
        passed=passed,
        score=score,
        confidence=confidence,
        reasons=list(reasons or []),
        metadata=dict(metadata or {"source": "tests"}),
    )
    evaluation.validate()
    return evaluation


# =============================================================================
# Pytest fixtures
# =============================================================================


@pytest.fixture()
def mock_event_bus() -> MockEventBus:
    return MockEventBus()


@pytest.fixture()
def mock_scheduler() -> MockScheduler:
    return MockScheduler()


@pytest.fixture()
def runtime_config() -> StrategyRuntimeConfig:
    return make_runtime_config()


@pytest.fixture()
def definition_config(runtime_config: StrategyRuntimeConfig) -> StrategyDefinitionConfig:
    return make_definition_config(runtime=runtime_config)


@pytest.fixture()
def strategy_config(definition_config: StrategyDefinitionConfig) -> StrategyConfig:
    return make_strategy_config(definitions=[definition_config])


@pytest.fixture()
def feature_snapshot() -> FeatureSnapshot:
    return make_feature_snapshot()


@pytest.fixture()
def strategy_context(feature_snapshot: FeatureSnapshot) -> StrategyContext:
    return make_strategy_context(features=[feature_snapshot])


@pytest.fixture()
def strategy_signal() -> StrategySignal:
    return make_strategy_signal()


@pytest.fixture()
def risk_ready_strategy_signal() -> StrategySignal:
    return make_strategy_signal(with_execution_plan=True)


@pytest.fixture()
def dummy_strategy(
    strategy_config: StrategyConfig,
    definition_config: StrategyDefinitionConfig,
    mock_event_bus: MockEventBus,
    mock_scheduler: MockScheduler,
) -> DummyStrategy:
    return DummyStrategy(
        config=strategy_config,
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        definition=definition_config,
    )


@pytest.fixture()
def short_dummy_strategy(
    strategy_config: StrategyConfig,
    mock_event_bus: MockEventBus,
    mock_scheduler: MockScheduler,
) -> ShortDummyStrategy:
    definition = make_definition_config(
        name="short_dummy_strategy",
        category=StrategyCategory.ORDERFLOW,
        required_features=("orderflow_imbalance",),
        priority=20,
    )
    strategy_config.upsert_strategy(definition)

    return ShortDummyStrategy(
        config=strategy_config,
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        definition=definition,
    )


@pytest.fixture()
def flat_dummy_strategy(
    strategy_config: StrategyConfig,
    mock_event_bus: MockEventBus,
    mock_scheduler: MockScheduler,
) -> FlatDummyStrategy:
    definition = make_definition_config(
        name="flat_dummy_strategy",
        category=StrategyCategory.ORDERFLOW,
        required_features=("orderflow_imbalance",),
        priority=30,
    )
    strategy_config.upsert_strategy(definition)

    return FlatDummyStrategy(
        config=strategy_config,
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        definition=definition,
    )


@pytest.fixture()
def no_signal_strategy(
    strategy_config: StrategyConfig,
    mock_event_bus: MockEventBus,
    mock_scheduler: MockScheduler,
) -> NoSignalStrategy:
    definition = make_definition_config(
        name="no_signal_strategy",
        category=StrategyCategory.ORDERFLOW,
        required_features=("orderflow_imbalance",),
        priority=40,
    )
    strategy_config.upsert_strategy(definition)

    return NoSignalStrategy(
        config=strategy_config,
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        definition=definition,
    )


@pytest.fixture()
def failing_strategy(
    strategy_config: StrategyConfig,
    mock_event_bus: MockEventBus,
    mock_scheduler: MockScheduler,
) -> FailingStrategy:
    definition = make_definition_config(
        name="failing_strategy",
        category=StrategyCategory.ORDERFLOW,
        required_features=("orderflow_imbalance",),
        priority=50,
    )
    strategy_config.upsert_strategy(definition)

    return FailingStrategy(
        config=strategy_config,
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        definition=definition,
    )


@pytest.fixture()
def make_definition() -> Callable[..., StrategyDefinitionConfig]:
    return make_definition_config


@pytest.fixture()
def make_config() -> Callable[..., StrategyConfig]:
    return make_strategy_config


@pytest.fixture()
def make_feature() -> Callable[..., FeatureSnapshot]:
    return make_feature_snapshot


@pytest.fixture()
def make_context() -> Callable[..., StrategyContext]:
    return make_strategy_context


@pytest.fixture()
def make_signal() -> Callable[..., StrategySignal]:
    return make_strategy_signal


@pytest.fixture()
def make_evaluation() -> Callable[..., StrategyEvaluation]:
    return make_strategy_evaluation


@pytest.fixture()
def attach_plan() -> Callable[..., StrategySignal]:
    return attach_execution_plan