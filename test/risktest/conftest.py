# tests/risk/conftest.py
from __future__ import annotations

import asyncio
import dataclasses
import fnmatch
import inspect
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Iterable
from uuid import uuid4

import pytest

from risk.config import RiskConfig
from risk.enums import (
    ExecutionQuality,
    LiquidityClass,
    MarginMode,
    OrderIntent,
    PositionSide,
    RiskDecisionType,
    RiskMode,
    TradeTier,
)
from risk.metrics import RiskMetrics
from risk.models import (
    ExecutionCostEstimate,
    PortfolioPosition,
    RiskDecision,
    RiskEvaluationRequest,
)
from risk.risk_manager import RiskManager
from risk.state import PendingRiskReservation, RiskState


# =============================================================================
# Test constants
# =============================================================================

TEST_BALANCE = 10_000.0
TEST_EQUITY = 10_000.0
TEST_FREE_BALANCE = 10_000.0
TEST_SYMBOL = "BTCUSDT"
TEST_STRATEGY = "test_strategy"
TEST_SIGNAL_ID = "signal-test-001"


# =============================================================================
# Fake EventBus layer
# =============================================================================

@dataclass(slots=True)
class FakeEvent:
    """
    Minimal Event-compatible object for RiskManager handlers.

    RiskManager handlers mostly access:
    - event.topic
    - event.payload
    - event.source
    - event.priority
    """

    topic: str
    payload: dict[str, Any]
    source: str | None = None
    priority: Any | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FakeSubscription:
    topic: str
    handler: Callable[..., Any]
    name: str | None = None
    active: bool = True
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class EmittedEvent:
    topic: str
    payload: dict[str, Any]
    source: str | None = None
    priority: Any | None = None
    timestamp: float = field(default_factory=time.time)


class FakeEventBus:
    """
    Async EventBus test double.

    Goals:
    - record subscriptions;
    - record emitted events;
    - optionally dispatch emitted events to matching subscribers;
    - support wildcard topics like "account.*";
    - keep a call history for lifecycle/integration assertions.

    By default auto_dispatch=False to avoid accidental recursive calls in unit
    tests. Enable it only in EventBus flow tests.
    """

    def __init__(self, *, auto_dispatch: bool = False) -> None:
        self.auto_dispatch = auto_dispatch

        self.subscriptions: list[FakeSubscription] = []
        self.unsubscriptions: list[FakeSubscription] = []
        self.emitted: list[EmittedEvent] = []

        self.emit_calls: list[dict[str, Any]] = []
        self.subscribe_calls: list[dict[str, Any]] = []
        self.unsubscribe_calls: list[dict[str, Any]] = []

    def subscribe(
        self,
        topic: str,
        handler: Callable[..., Any],
        *,
        name: str | None = None,
        **_: Any,
    ) -> FakeSubscription:
        subscription = FakeSubscription(
            topic=topic,
            handler=handler,
            name=name,
        )
        self.subscriptions.append(subscription)
        self.subscribe_calls.append(
            {
                "topic": topic,
                "handler": handler,
                "name": name,
            }
        )
        return subscription

    def unsubscribe(self, subscription: FakeSubscription) -> None:
        subscription.active = False
        self.unsubscriptions.append(subscription)
        self.unsubscribe_calls.append({"subscription": subscription})

    async def emit(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
        *,
        source: str | None = None,
        priority: Any | None = None,
        **kwargs: Any,
    ) -> None:
        payload = dict(payload or {})

        event_record = EmittedEvent(
            topic=topic,
            payload=payload,
            source=source,
            priority=priority,
        )
        self.emitted.append(event_record)
        self.emit_calls.append(
            {
                "topic": topic,
                "payload": payload,
                "source": source,
                "priority": priority,
                **kwargs,
            }
        )

        if self.auto_dispatch:
            await self.dispatch(topic, payload, source=source, priority=priority)

    async def dispatch(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
        *,
        source: str | None = None,
        priority: Any | None = None,
    ) -> None:
        event = FakeEvent(
            topic=topic,
            payload=dict(payload or {}),
            source=source,
            priority=priority,
        )

        handlers = [
            subscription.handler
            for subscription in self.subscriptions
            if subscription.active and fnmatch.fnmatch(topic, subscription.topic)
        ]

        for handler in handlers:
            result = handler(event)
            if inspect.isawaitable(result):
                await result

    def topics(self) -> list[str]:
        return [event.topic for event in self.emitted]

    def events_for(self, topic: str) -> list[EmittedEvent]:
        return [event for event in self.emitted if event.topic == topic]

    def first_event(self, topic: str) -> EmittedEvent | None:
        events = self.events_for(topic)
        return events[0] if events else None

    def last_event(self, topic: str) -> EmittedEvent | None:
        events = self.events_for(topic)
        return events[-1] if events else None

    def clear(self) -> None:
        self.emitted.clear()
        self.emit_calls.clear()


# =============================================================================
# Fake Scheduler layer
# =============================================================================

@dataclass(slots=True)
class FakeScheduledJob:
    name: str
    callback: Callable[..., Any]
    interval_seconds: float
    run_immediately: bool = False
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    run_count: int = 0
    last_result: Any = None
    last_error: BaseException | None = None

    async def run(self) -> Any:
        self.run_count += 1
        try:
            result = self.callback()
            if inspect.isawaitable(result):
                result = await result
            self.last_result = result
            return result
        except BaseException as exc:
            self.last_error = exc
            raise


class FakeScheduler:
    """
    Scheduler test double compatible with RiskManager.register_scheduler_jobs().

    RiskManager calls:
        scheduler.add_interval_job(
            callback,
            interval_seconds=...,
            name=...,
            run_immediately=False,
        )
    """

    def __init__(self, *, fail_on_add: bool = False) -> None:
        self.fail_on_add = fail_on_add
        self.jobs: list[FakeScheduledJob] = []
        self.add_interval_job_calls: list[dict[str, Any]] = []

    def add_interval_job(
        self,
        callback: Callable[..., Any],
        *,
        interval_seconds: float,
        name: str,
        run_immediately: bool = False,
        **kwargs: Any,
    ) -> FakeScheduledJob:
        self.add_interval_job_calls.append(
            {
                "callback": callback,
                "interval_seconds": interval_seconds,
                "name": name,
                "run_immediately": run_immediately,
                **kwargs,
            }
        )

        if self.fail_on_add:
            raise RuntimeError(f"FakeScheduler failed to add job: {name}")

        job = FakeScheduledJob(
            name=name,
            callback=callback,
            interval_seconds=float(interval_seconds),
            run_immediately=run_immediately,
        )
        self.jobs.append(job)
        return job

    def job_names(self) -> list[str]:
        return [job.name for job in self.jobs]

    def get_job(self, name: str) -> FakeScheduledJob | None:
        for job in self.jobs:
            if job.name == name:
                return job
        return None

    async def run_job(self, name: str) -> Any:
        job = self.get_job(name)
        if job is None:
            raise AssertionError(f"Scheduler job not found: {name}")
        return await job.run()


# =============================================================================
# JSON / enum helpers
# =============================================================================

def json_safe(value: Any) -> Any:
    """
    Convert test payloads/dataclasses/enums into JSON-safe structures.

    This helper is intentionally strict enough to catch payload issues in tests,
    but flexible enough for local assertions.
    """

    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def assert_json_serializable(value: Any) -> None:
    try:
        json.dumps(json_safe(value), sort_keys=True)
    except TypeError as exc:
        raise AssertionError(f"Value is not JSON serializable: {value!r}") from exc


# =============================================================================
# Base fixtures
# =============================================================================

@pytest.fixture
def fake_event_bus() -> FakeEventBus:
    return FakeEventBus(auto_dispatch=False)


@pytest.fixture
def auto_dispatch_event_bus() -> FakeEventBus:
    return FakeEventBus(auto_dispatch=True)


@pytest.fixture
def fake_scheduler() -> FakeScheduler:
    return FakeScheduler()


@pytest.fixture
def failing_scheduler() -> FakeScheduler:
    return FakeScheduler(fail_on_add=True)


@pytest.fixture
def risk_config() -> RiskConfig:
    """
    Default production-like config.

    The defaults are intentionally not over-relaxed. Individual tests should
    override exact limits they want to stress.
    """

    config = RiskConfig()
    config.validate()
    return config


@pytest.fixture
def strict_risk_config() -> RiskConfig:
    """
    Config for hard capital-safety tests.

    Keeps reservations enabled and uses tight enough exposure settings to make
    overexposure/reservation tests meaningful.
    """

    config = RiskConfig()

    config.reservation.enabled = True
    config.reservation.reserve_on_allow = True
    config.reservation.ttl_seconds = 30.0
    config.reservation.cleanup_interval_seconds = 5.0
    config.reservation.max_pending_reservations = 10
    config.reservation.max_pending_per_symbol = 2
    config.reservation.max_pending_per_strategy = 5

    config.budget.caution_daily_loss_r = 3.0
    config.budget.soft_daily_loss_r = 6.0
    config.budget.hard_daily_loss_r = 10.0
    config.budget.weekly_hard_loss_r = 20.0
    config.budget.monthly_review_loss_r = 30.0
    config.budget.emergency_stop_loss_r = 40.0

    config.validate()
    return config


@pytest.fixture
def risk_state() -> RiskState:
    return RiskState()


@pytest.fixture
def funded_state() -> RiskState:
    state = RiskState()
    state.update_account(
        balance=TEST_BALANCE,
        equity=TEST_EQUITY,
        free_balance=TEST_FREE_BALANCE,
        used_margin=0.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
    )
    return state


@pytest.fixture
def risk_metrics() -> RiskMetrics:
    return RiskMetrics()


# =============================================================================
# Data factories
# =============================================================================

@pytest.fixture
def execution_cost_ok() -> ExecutionCostEstimate:
    return ExecutionCostEstimate(
        spread_cost=0.01,
        slippage_cost=0.01,
        fee_cost=0.02,
        funding_cost=0.0,
        spread_pct=0.0001,
        slippage_pct=0.0001,
        quality=ExecutionQuality.ACCEPTABLE,
        metadata={"fixture": "execution_cost_ok"},
    )


@pytest.fixture
def execution_cost_bad() -> ExecutionCostEstimate:
    return ExecutionCostEstimate(
        spread_cost=10.0,
        slippage_cost=10.0,
        fee_cost=10.0,
        funding_cost=0.0,
        spread_pct=0.05,
        slippage_pct=0.05,
        quality=ExecutionQuality.POOR,
        metadata={"fixture": "execution_cost_bad"},
    )


@pytest.fixture
def valid_long_request(execution_cost_ok: ExecutionCostEstimate) -> RiskEvaluationRequest:
    return RiskEvaluationRequest(
        symbol=TEST_SYMBOL,
        side=PositionSide.LONG,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=103.0,
        signal_id=TEST_SIGNAL_ID,
        strategy_name=TEST_STRATEGY,
        tier=TradeTier.T2,
        order_intent=OrderIntent.OPEN,
        liquidity_class=LiquidityClass.HIGH,
        execution_quality=ExecutionQuality.ACCEPTABLE,
        confidence=0.75,
        edge_score=0.65,
        volatility=0.20,
        expected_reward=3.0,
        expected_loss=1.0,
        expected_win_probability=0.55,
        expected_cost=0.04,
        execution_cost=execution_cost_ok,
        requested_leverage=5.0,
        margin_mode=MarginMode.ISOLATED,
        timestamp=time.time(),
        metadata={"fixture": "valid_long_request"},
    )


@pytest.fixture
def valid_short_request(execution_cost_ok: ExecutionCostEstimate) -> RiskEvaluationRequest:
    return RiskEvaluationRequest(
        symbol=TEST_SYMBOL,
        side=PositionSide.SHORT,
        entry_price=100.0,
        stop_loss=101.0,
        take_profit=97.0,
        signal_id="signal-test-short-001",
        strategy_name=TEST_STRATEGY,
        tier=TradeTier.T2,
        order_intent=OrderIntent.OPEN,
        liquidity_class=LiquidityClass.HIGH,
        execution_quality=ExecutionQuality.ACCEPTABLE,
        confidence=0.75,
        edge_score=0.65,
        volatility=0.20,
        expected_reward=3.0,
        expected_loss=1.0,
        expected_win_probability=0.55,
        expected_cost=0.04,
        execution_cost=execution_cost_ok,
        requested_leverage=5.0,
        margin_mode=MarginMode.ISOLATED,
        timestamp=time.time(),
        metadata={"fixture": "valid_short_request"},
    )


@pytest.fixture
def make_request(
    execution_cost_ok: ExecutionCostEstimate,
) -> Callable[..., RiskEvaluationRequest]:
    def _make_request(
        *,
        symbol: str = TEST_SYMBOL,
        side: PositionSide = PositionSide.LONG,
        entry_price: float = 100.0,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        signal_id: str | None = None,
        strategy_name: str | None = TEST_STRATEGY,
        tier: TradeTier | None = TradeTier.T2,
        order_intent: OrderIntent = OrderIntent.OPEN,
        liquidity_class: LiquidityClass = LiquidityClass.HIGH,
        execution_quality: ExecutionQuality = ExecutionQuality.ACCEPTABLE,
        confidence: float | None = 0.75,
        edge_score: float | None = 0.65,
        volatility: float | None = 0.20,
        expected_reward: float | None = 3.0,
        expected_loss: float | None = 1.0,
        expected_win_probability: float | None = 0.55,
        expected_cost: float | None = 0.04,
        execution_cost: ExecutionCostEstimate | None = execution_cost_ok,
        requested_size: float | None = None,
        requested_margin: float | None = None,
        requested_leverage: float | None = 5.0,
        reduce_only: bool = False,
        margin_mode: MarginMode = MarginMode.ISOLATED,
        metadata: dict[str, Any] | None = None,
    ) -> RiskEvaluationRequest:
        if stop_loss is None:
            stop_loss = 99.0 if side is PositionSide.LONG else 101.0

        if take_profit is None:
            take_profit = 103.0 if side is PositionSide.LONG else 97.0

        return RiskEvaluationRequest(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_id=signal_id or f"signal-{uuid4()}",
            strategy_name=strategy_name,
            tier=tier,
            order_intent=order_intent,
            liquidity_class=liquidity_class,
            execution_quality=execution_quality,
            confidence=confidence,
            edge_score=edge_score,
            volatility=volatility,
            expected_reward=expected_reward,
            expected_loss=expected_loss,
            expected_win_probability=expected_win_probability,
            expected_cost=expected_cost,
            execution_cost=execution_cost,
            requested_size=requested_size,
            requested_margin=requested_margin,
            requested_leverage=requested_leverage,
            reduce_only=reduce_only,
            margin_mode=margin_mode,
            timestamp=time.time(),
            metadata=dict(metadata or {}),
        )

    return _make_request


@pytest.fixture
def make_position() -> Callable[..., PortfolioPosition]:
    def _make_position(
        *,
        symbol: str = TEST_SYMBOL,
        side: PositionSide = PositionSide.LONG,
        size: float = 1.0,
        entry_price: float = 100.0,
        mark_price: float | None = None,
        notional_value: float | None = None,
        leverage: float | None = 5.0,
        margin_used: float | None = None,
        risk_amount: float = 10.0,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        tier: TradeTier | None = TradeTier.T2,
        strategy_name: str | None = TEST_STRATEGY,
        signal_id: str | None = None,
        position_id: str | None = None,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> PortfolioPosition:
        mark_price = mark_price if mark_price is not None else entry_price
        notional_value = (
            notional_value
            if notional_value is not None
            else abs(size * entry_price)
        )
        margin_used = (
            margin_used
            if margin_used is not None
            else notional_value / leverage
            if leverage and leverage > 0
            else 0.0
        )

        if stop_loss is None:
            stop_loss = 99.0 if side is PositionSide.LONG else 101.0

        if take_profit is None:
            take_profit = 103.0 if side is PositionSide.LONG else 97.0

        return PortfolioPosition(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            mark_price=mark_price,
            notional_value=notional_value,
            leverage=leverage,
            margin_used=margin_used,
            risk_amount=risk_amount,
            stop_loss=stop_loss,
            take_profit=take_profit,
            tier=tier,
            strategy_name=strategy_name,
            signal_id=signal_id or f"signal-{uuid4()}",
            position_id=position_id or f"position-{uuid4()}",
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            opened_at=time.time(),
            updated_at=time.time(),
            metadata=dict(metadata or {}),
        )

    return _make_position


@pytest.fixture
def make_reservation() -> Callable[..., PendingRiskReservation]:
    def _make_reservation(
        *,
        reservation_id: str | None = None,
        symbol: str = TEST_SYMBOL,
        side: PositionSide = PositionSide.LONG,
        signal_id: str | None = None,
        strategy_name: str | None = TEST_STRATEGY,
        tier: TradeTier | None = TradeTier.T2,
        position_id: str | None = None,
        size: float = 1.0,
        open_risk: float = 10.0,
        margin: float = 20.0,
        notional: float = 100.0,
        created_at: float | None = None,
        expires_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PendingRiskReservation:
        return PendingRiskReservation(
            reservation_id=reservation_id or f"reservation-{uuid4()}",
            symbol=symbol,
            side=side,
            signal_id=signal_id or f"signal-{uuid4()}",
            strategy_name=strategy_name,
            tier=tier,
            position_id=position_id,
            size=size,
            open_risk=open_risk,
            margin=margin,
            notional=notional,
            created_at=created_at if created_at is not None else time.time(),
            expires_at=expires_at,
            metadata=dict(metadata or {}),
        )

    return _make_reservation


# =============================================================================
# Manager factories
# =============================================================================

@pytest.fixture
def make_risk_manager(
    risk_config: RiskConfig,
    funded_state: RiskState,
    fake_event_bus: FakeEventBus,
    fake_scheduler: FakeScheduler,
) -> Callable[..., RiskManager]:
    def _make_risk_manager(
        *,
        config: RiskConfig | None = None,
        state: RiskState | None = None,
        event_bus: FakeEventBus | None = None,
        scheduler: FakeScheduler | None = None,
        metrics: RiskMetrics | None = None,
        auto_subscribe: bool = False,
        register_scheduler_jobs: bool = False,
        service_name: str = "risk_manager_test",
    ) -> RiskManager:
        return RiskManager(
            config=config or risk_config,
            state=state or funded_state,
            metrics=metrics,
            event_bus=event_bus if event_bus is not None else fake_event_bus,
            scheduler=scheduler if scheduler is not None else fake_scheduler,
            auto_subscribe=auto_subscribe,
            register_scheduler_jobs=register_scheduler_jobs,
            service_name=service_name,
        )

    return _make_risk_manager


@pytest.fixture
def risk_manager(make_risk_manager: Callable[..., RiskManager]) -> RiskManager:
    """
    Default RiskManager fixture for unit/integration tests.

    auto_subscribe/register_scheduler_jobs are disabled by default to keep tests
    deterministic. Lifecycle tests should construct their own manager with the
    flags enabled.
    """

    return make_risk_manager(
        auto_subscribe=False,
        register_scheduler_jobs=False,
    )


@pytest.fixture
def running_risk_manager(
    make_risk_manager: Callable[..., RiskManager],
) -> Callable[..., Awaitable[RiskManager]]:
    async def _running_risk_manager(**kwargs: Any) -> RiskManager:
        manager = make_risk_manager(
            auto_subscribe=kwargs.pop("auto_subscribe", True),
            register_scheduler_jobs=kwargs.pop("register_scheduler_jobs", True),
            **kwargs,
        )
        await manager.start()
        return manager

    return _running_risk_manager


# =============================================================================
# Event payload factories
# =============================================================================

@pytest.fixture
def make_signal_payload() -> Callable[..., dict[str, Any]]:
    def _make_signal_payload(
        *,
        symbol: str = TEST_SYMBOL,
        side: str = "long",
        entry_price: float = 100.0,
        stop_loss: float | None = 99.0,
        take_profit: float | None = 103.0,
        signal_id: str = TEST_SIGNAL_ID,
        strategy_name: str = TEST_STRATEGY,
        tier: str = "t2",
        order_intent: str = "open",
        liquidity_class: str = "high",
        execution_quality: str = "acceptable",
        confidence: float = 0.75,
        edge_score: float = 0.65,
        volatility: float = 0.20,
        expected_reward: float = 3.0,
        expected_loss: float = 1.0,
        expected_win_probability: float = 0.55,
        expected_cost: float = 0.04,
        requested_size: float | None = None,
        requested_margin: float | None = None,
        requested_leverage: float | None = 5.0,
        reduce_only: bool = False,
        margin_mode: str = "isolated",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "signal_id": signal_id,
            "strategy_name": strategy_name,
            "tier": tier,
            "order_intent": order_intent,
            "liquidity_class": liquidity_class,
            "execution_quality": execution_quality,
            "confidence": confidence,
            "edge_score": edge_score,
            "volatility": volatility,
            "expected_reward": expected_reward,
            "expected_loss": expected_loss,
            "expected_win_probability": expected_win_probability,
            "expected_cost": expected_cost,
            "execution_cost": {
                "spread_cost": 0.01,
                "slippage_cost": 0.01,
                "fee_cost": 0.02,
                "funding_cost": 0.0,
                "spread_pct": 0.0001,
                "slippage_pct": 0.0001,
                "quality": execution_quality,
            },
            "requested_size": requested_size,
            "requested_margin": requested_margin,
            "requested_leverage": requested_leverage,
            "reduce_only": reduce_only,
            "margin_mode": margin_mode,
            "timestamp": time.time(),
            "metadata": dict(metadata or {}),
        }

    return _make_signal_payload


@pytest.fixture
def make_position_payload() -> Callable[..., dict[str, Any]]:
    def _make_position_payload(
        *,
        symbol: str = TEST_SYMBOL,
        side: str = "long",
        size: float = 1.0,
        entry_price: float = 100.0,
        mark_price: float | None = None,
        notional_value: float | None = None,
        leverage: float = 5.0,
        margin_used: float | None = None,
        risk_amount: float = 10.0,
        stop_loss: float | None = 99.0,
        take_profit: float | None = 103.0,
        tier: str = "t2",
        strategy_name: str = TEST_STRATEGY,
        signal_id: str = TEST_SIGNAL_ID,
        position_id: str = "position-test-001",
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        mark_price = mark_price if mark_price is not None else entry_price
        notional_value = (
            notional_value
            if notional_value is not None
            else abs(size * entry_price)
        )
        margin_used = (
            margin_used
            if margin_used is not None
            else notional_value / leverage
        )

        return {
            "symbol": symbol,
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "mark_price": mark_price,
            "notional_value": notional_value,
            "leverage": leverage,
            "margin_used": margin_used,
            "risk_amount": risk_amount,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "tier": tier,
            "strategy_name": strategy_name,
            "signal_id": signal_id,
            "position_id": position_id,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "opened_at": time.time(),
            "updated_at": time.time(),
            "metadata": dict(metadata or {}),
        }

    return _make_position_payload


@pytest.fixture
def make_account_payload() -> Callable[..., dict[str, Any]]:
    def _make_account_payload(
        *,
        balance: float = TEST_BALANCE,
        equity: float = TEST_EQUITY,
        free_balance: float = TEST_FREE_BALANCE,
        used_margin: float = 0.0,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "balance": balance,
            "equity": equity,
            "free_balance": free_balance,
            "used_margin": used_margin,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "metadata": dict(metadata or {}),
        }

    return _make_account_payload


# =============================================================================
# Assertion helpers
# =============================================================================

@pytest.fixture
def assert_decision_allowed() -> Callable[..., None]:
    def _assert_decision_allowed(
        decision: RiskDecision,
        *,
        decision_type: RiskDecisionType | None = None,
        symbol: str | None = None,
        strategy_name: str | None = None,
        require_size: bool = True,
        require_leverage: bool = True,
        require_tier: bool = True,
    ) -> None:
        assert decision.allowed is True
        assert decision.decision is not RiskDecisionType.DENY

        if decision_type is not None:
            assert decision.decision is decision_type

        if symbol is not None:
            assert decision.symbol == symbol

        if strategy_name is not None:
            assert decision.strategy_name == strategy_name

        if require_size:
            assert decision.final_size is not None
            assert decision.final_size > 0

        if require_leverage:
            assert decision.final_leverage is not None
            assert decision.final_leverage > 0

        if require_tier:
            assert decision.final_tier is not None

        assert_json_serializable(decision)

    return _assert_decision_allowed


@pytest.fixture
def assert_decision_denied() -> Callable[..., None]:
    def _assert_decision_denied(
        decision: RiskDecision,
        *,
        decision_type: RiskDecisionType | None = None,
        reason_contains: str | None = None,
        symbol: str | None = None,
        strategy_name: str | None = None,
    ) -> None:
        assert decision.allowed is False

        if decision_type is not None:
            assert decision.decision is decision_type

        if reason_contains is not None:
            assert decision.reason is not None
            assert reason_contains.lower() in decision.reason.lower()

        if symbol is not None:
            assert decision.symbol == symbol

        if strategy_name is not None:
            assert decision.strategy_name == strategy_name

        assert_json_serializable(decision)

    return _assert_decision_denied


@pytest.fixture
def assert_event_emitted() -> Callable[..., EmittedEvent]:
    def _assert_event_emitted(
        event_bus: FakeEventBus,
        topic: str,
        *,
        count: int | None = None,
        payload_contains: dict[str, Any] | None = None,
    ) -> EmittedEvent:
        events = event_bus.events_for(topic)
        assert events, f"Expected event topic={topic!r}, emitted={event_bus.topics()}"

        if count is not None:
            assert len(events) == count, (
                f"Expected {count} events for topic={topic!r}, got {len(events)}"
            )

        event = events[-1]

        if payload_contains:
            for key, expected_value in payload_contains.items():
                assert key in event.payload, (
                    f"Expected key={key!r} in payload for topic={topic!r}. "
                    f"Payload={event.payload!r}"
                )
                assert event.payload[key] == expected_value

        assert_json_serializable(event.payload)
        return event

    return _assert_event_emitted


@pytest.fixture
def assert_no_event_emitted() -> Callable[..., None]:
    def _assert_no_event_emitted(event_bus: FakeEventBus, topic: str) -> None:
        events = event_bus.events_for(topic)
        assert not events, f"Did not expect topic={topic!r}, got={events!r}"

    return _assert_no_event_emitted


@pytest.fixture
def assert_subscription_registered() -> Callable[..., FakeSubscription]:
    def _assert_subscription_registered(
        event_bus: FakeEventBus,
        topic: str,
        *,
        name: str | None = None,
    ) -> FakeSubscription:
        for subscription in event_bus.subscriptions:
            if subscription.topic != topic:
                continue
            if name is not None and subscription.name != name:
                continue
            return subscription

        raise AssertionError(
            f"Expected subscription topic={topic!r} name={name!r}. "
            f"Registered={[(s.topic, s.name) for s in event_bus.subscriptions]!r}"
        )

    return _assert_subscription_registered


@pytest.fixture
def assert_scheduler_job_registered() -> Callable[..., FakeScheduledJob]:
    def _assert_scheduler_job_registered(
        scheduler: FakeScheduler,
        name: str,
        *,
        interval_seconds: float | None = None,
    ) -> FakeScheduledJob:
        job = scheduler.get_job(name)
        assert job is not None, (
            f"Expected scheduler job {name!r}. Registered={scheduler.job_names()!r}"
        )

        if interval_seconds is not None:
            assert job.interval_seconds == pytest.approx(interval_seconds)

        return job

    return _assert_scheduler_job_registered


# =============================================================================
# Async helpers
# =============================================================================

@pytest.fixture
def run_event_handler() -> Callable[..., Awaitable[None]]:
    async def _run_event_handler(
        handler: Callable[..., Any],
        *,
        topic: str,
        payload: dict[str, Any] | None = None,
        source: str | None = "test",
        priority: Any | None = None,
    ) -> None:
        event = FakeEvent(
            topic=topic,
            payload=dict(payload or {}),
            source=source,
            priority=priority,
        )
        result = handler(event)
        if inspect.isawaitable(result):
            await result

    return _run_event_handler


@pytest.fixture
def evaluate_many_concurrently() -> Callable[..., Awaitable[list[RiskDecision]]]:
    async def _evaluate_many_concurrently(
        manager: RiskManager,
        requests: Iterable[RiskEvaluationRequest],
    ) -> list[RiskDecision]:
        return list(
            await asyncio.gather(
                *(manager.evaluate_request(request) for request in requests)
            )
        )

    return _evaluate_many_concurrently


# =============================================================================
# State mutation helpers
# =============================================================================

@pytest.fixture
def add_position_to_state() -> Callable[..., PortfolioPosition]:
    def _add_position_to_state(
        state: RiskState,
        position: PortfolioPosition,
    ) -> PortfolioPosition:
        state.add_position(position)
        return position

    return _add_position_to_state


@pytest.fixture
def add_reservation_to_state() -> Callable[..., PendingRiskReservation]:
    def _add_reservation_to_state(
        state: RiskState,
        reservation: PendingRiskReservation,
    ) -> PendingRiskReservation:
        state.pending_reservations[reservation.reservation_id] = reservation
        state.updated_at = time.time()
        return reservation

    return _add_reservation_to_state


@pytest.fixture
def put_state_in_mode() -> Callable[..., RiskState]:
    def _put_state_in_mode(
        state: RiskState,
        mode: RiskMode,
        *,
        reason: str | None = None,
    ) -> RiskState:
        state.set_risk_mode(mode, reason=reason)
        return state

    return _put_state_in_mode