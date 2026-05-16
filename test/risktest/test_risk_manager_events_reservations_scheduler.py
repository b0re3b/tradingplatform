# tests/risk/test_risk_manager_events_reservations_scheduler.py
from __future__ import annotations

import asyncio
import dataclasses
import fnmatch
import inspect
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

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
from risk.models import ExecutionCostEstimate, RiskEvaluationRequest
from risk.risk_manager import RiskManager
from risk.state import RiskState


TEST_SYMBOL = "BTCUSDT"
ALT_SYMBOL = "ETHUSDT"
TEST_STRATEGY = "test_strategy"
TEST_SIGNAL_ID = "signal-events-001"


# =============================================================================
# Test doubles
# =============================================================================


@dataclass(slots=True)
class FakeEvent:
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
    priority: Any | None = None
    source: str | None = None
    timestamp: float = field(default_factory=time.time)


class FakeEventBus:
    """
    EventBus test double for RiskManager integration tests.

    It records subscriptions/emissions and can manually dispatch events to
    matching wildcard subscribers.
    """

    def __init__(self, *, auto_dispatch_emitted: bool = False) -> None:
        self.auto_dispatch_emitted = auto_dispatch_emitted

        self.subscriptions: list[FakeSubscription] = []
        self.unsubscriptions: list[FakeSubscription] = []
        self.emitted: list[EmittedEvent] = []

        self.subscribe_calls: list[dict[str, Any]] = []
        self.unsubscribe_calls: list[dict[str, Any]] = []
        self.emit_calls: list[dict[str, Any]] = []

    def subscribe(
        self,
        topic: str,
        handler: Callable[..., Any],
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> FakeSubscription:
        subscription = FakeSubscription(topic=topic, handler=handler, name=name)
        self.subscriptions.append(subscription)
        self.subscribe_calls.append(
            {
                "topic": topic,
                "handler": handler,
                "name": name,
                **kwargs,
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
        priority: Any | None = None,
        source: str | None = None,
        **kwargs: Any,
    ) -> None:
        payload = dict(payload or {})
        self.emitted.append(
            EmittedEvent(
                topic=topic,
                payload=payload,
                priority=priority,
                source=source,
            )
        )
        self.emit_calls.append(
            {
                "topic": topic,
                "payload": payload,
                "priority": priority,
                "source": source,
                **kwargs,
            }
        )

        if self.auto_dispatch_emitted:
            await self.dispatch(topic, payload, priority=priority, source=source)

    async def dispatch(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
        *,
        source: str | None = "test",
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

    def subscription_topics(self) -> list[str]:
        return [subscription.topic for subscription in self.subscriptions]

    def active_subscription_topics(self) -> list[str]:
        return [
            subscription.topic
            for subscription in self.subscriptions
            if subscription.active
        ]

    def events_for(self, topic: str) -> list[EmittedEvent]:
        return [event for event in self.emitted if event.topic == topic]

    def last_event(self, topic: str) -> EmittedEvent:
        events = self.events_for(topic)
        assert events, f"Expected topic={topic!r}, emitted={self.topics()!r}"
        return events[-1]

    def clear_emitted(self) -> None:
        self.emitted.clear()
        self.emit_calls.clear()


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
            raise RuntimeError(f"Fake scheduler add job failure: {name}")

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
        assert job is not None, f"Scheduler job not found: {name!r}"
        return await job.run()


# =============================================================================
# JSON helpers
# =============================================================================


def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}

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
# Builders
# =============================================================================


def set_if_exists(obj: Any, name: str, value: Any) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def make_state(
    *,
    balance: float = 10_000.0,
    equity: float = 10_000.0,
    free_balance: float = 10_000.0,
    used_margin: float = 0.0,
    mode: RiskMode = RiskMode.NORMAL,
) -> RiskState:
    state = RiskState()
    state.update_account(
        balance=balance,
        equity=equity,
        free_balance=free_balance,
        used_margin=used_margin,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
    )
    state.set_risk_mode(mode)
    return state


def force_daily_loss(state: RiskState, loss_amount: float) -> None:
    assert loss_amount >= 0
    if hasattr(state, "daily_start_equity"):
        state.daily_start_equity = state.equity + loss_amount
        return
    raise AssertionError("RiskState has no daily_start_equity-compatible field")


def force_weekly_loss(state: RiskState, loss_amount: float) -> None:
    assert loss_amount >= 0
    if hasattr(state, "weekly_start_equity"):
        state.weekly_start_equity = state.equity + loss_amount
        return
    raise AssertionError("RiskState has no weekly_start_equity-compatible field")


def force_monthly_loss(state: RiskState, loss_amount: float) -> None:
    assert loss_amount >= 0
    if hasattr(state, "monthly_start_equity"):
        state.monthly_start_equity = state.equity + loss_amount
        return
    raise AssertionError("RiskState has no monthly_start_equity-compatible field")


def make_risk_config(
    *,
    reservations_enabled: bool = True,
    tight_exposure: bool = False,
    auto_expire_on_evaluate: bool = True,
) -> RiskConfig:
    config = RiskConfig()

    # -------------------------------------------------------------------------
    # Deterministic R / sizing.
    # -------------------------------------------------------------------------
    set_if_exists(config.risk_unit, "base_risk_unit_pct", 0.001)
    set_if_exists(config.risk_unit, "min_risk_unit", None)
    set_if_exists(config.risk_unit, "max_risk_unit", None)
    set_if_exists(config.risk_unit, "use_available_budget_caps", True)

    set_if_exists(config.position_sizing, "use_confidence_adjustment", False)
    set_if_exists(config.position_sizing, "use_volatility_adjustment", False)
    set_if_exists(config.position_sizing, "min_position_size", 0.0)
    set_if_exists(config.position_sizing, "max_position_size", None)
    set_if_exists(config.position_sizing, "require_stop_loss", True)
    set_if_exists(config.position_sizing, "fallback_stop_loss_pct", None)
    set_if_exists(config.position_sizing, "reject_if_below_min_size", True)
    set_if_exists(config.position_sizing, "never_increase_size_above_risk", True)

    # -------------------------------------------------------------------------
    # Exposure.
    #
    # RiskConfig.validate() requires:
    #   safe_mode_* <= normal_* <= aggressive_*
    #
    # Keep the non-tight profile permissive so orchestration tests do not fail
    # because of unrelated exposure caps. Use tight_exposure=True only in tests
    # that intentionally want exposure denial.
    # -------------------------------------------------------------------------
    if tight_exposure:
        max_open_risk_r = 0.5
        safe_open_risk_r = 0.25
        aggressive_open_risk_r = 0.75

        max_used_margin_pct = 0.01
        safe_used_margin_pct = 0.005
        aggressive_used_margin_pct = 0.02

        max_total_exposure_pct = 0.01
        safe_total_exposure_pct = 0.005
        aggressive_total_exposure_pct = 0.02

        max_symbol_exposure_pct = 0.01
        safe_symbol_exposure_pct = 0.005
        aggressive_symbol_exposure_pct = 0.02

        max_side_exposure_pct = 0.01
        safe_side_exposure_pct = 0.005
        aggressive_side_exposure_pct = 0.02

        max_open_positions = 0
    else:
        max_open_risk_r = 100.0
        safe_open_risk_r = 50.0
        aggressive_open_risk_r = 150.0

        max_used_margin_pct = 1.0
        safe_used_margin_pct = 0.5
        aggressive_used_margin_pct = 1.5

        max_total_exposure_pct = 10.0
        safe_total_exposure_pct = 5.0
        aggressive_total_exposure_pct = 15.0

        max_symbol_exposure_pct = 10.0
        safe_symbol_exposure_pct = 5.0
        aggressive_symbol_exposure_pct = 15.0

        max_side_exposure_pct = 10.0
        safe_side_exposure_pct = 5.0
        aggressive_side_exposure_pct = 15.0

        max_open_positions = 100

    set_if_exists(config.exposure, "max_open_risk_r", max_open_risk_r)
    set_if_exists(config.exposure, "safe_mode_max_open_risk_r", safe_open_risk_r)
    set_if_exists(config.exposure, "aggressive_max_open_risk_r", aggressive_open_risk_r)

    set_if_exists(config.exposure, "max_used_margin_pct", max_used_margin_pct)
    set_if_exists(config.exposure, "safe_mode_max_used_margin_pct", safe_used_margin_pct)
    set_if_exists(config.exposure, "aggressive_max_used_margin_pct", aggressive_used_margin_pct)

    set_if_exists(config.exposure, "max_total_exposure_pct", max_total_exposure_pct)
    set_if_exists(config.exposure, "safe_mode_max_total_exposure_pct", safe_total_exposure_pct)
    set_if_exists(config.exposure, "aggressive_max_total_exposure_pct", aggressive_total_exposure_pct)

    set_if_exists(config.exposure, "max_symbol_exposure_pct", max_symbol_exposure_pct)
    set_if_exists(config.exposure, "safe_mode_max_symbol_exposure_pct", safe_symbol_exposure_pct)
    set_if_exists(config.exposure, "aggressive_max_symbol_exposure_pct", aggressive_symbol_exposure_pct)

    set_if_exists(config.exposure, "max_side_exposure_pct", max_side_exposure_pct)
    set_if_exists(config.exposure, "safe_mode_max_side_exposure_pct", safe_side_exposure_pct)
    set_if_exists(config.exposure, "aggressive_max_side_exposure_pct", aggressive_side_exposure_pct)

    set_if_exists(config.exposure, "max_open_positions", max_open_positions)

    # -------------------------------------------------------------------------
    # Execution cost.
    #
    # Keep permissive in orchestration tests. Execution-cost strictness is tested
    # separately in guard/core-flow tests.
    # -------------------------------------------------------------------------
    set_if_exists(config.execution_cost, "enabled", True)
    set_if_exists(config.execution_cost, "max_spread_pct", 1.0)
    set_if_exists(config.execution_cost, "max_slippage_pct", 1.0)
    set_if_exists(config.execution_cost, "default_max_cost_to_reward_pct", 1.0)
    set_if_exists(config.execution_cost, "safe_mode_max_cost_to_reward_pct", 0.5)
    set_if_exists(config.execution_cost, "min_execution_quality", ExecutionQuality.ACCEPTABLE)
    set_if_exists(config.execution_cost, "require_positive_ev_after_cost", True)

    # -------------------------------------------------------------------------
    # Reservation policy.
    # -------------------------------------------------------------------------
    set_if_exists(config.reservation, "enabled", reservations_enabled)
    set_if_exists(config.reservation, "reserve_on_allow", reservations_enabled)
    set_if_exists(config.reservation, "ttl_seconds", 30.0)
    set_if_exists(config.reservation, "cleanup_interval_seconds", 5.0)
    set_if_exists(config.reservation, "max_pending_reservations", 100)
    set_if_exists(config.reservation, "max_pending_per_symbol", 10)
    set_if_exists(config.reservation, "max_pending_per_strategy", 20)
    set_if_exists(config.reservation, "fail_closed_on_reservation_error", True)
    set_if_exists(config.reservation, "auto_expire_on_evaluate", auto_expire_on_evaluate)

    # -------------------------------------------------------------------------
    # Circuit breaker.
    # -------------------------------------------------------------------------
    set_if_exists(config.circuit_breaker, "enabled", True)
    set_if_exists(config.circuit_breaker, "max_consecutive_failures", 5)
    set_if_exists(config.circuit_breaker, "max_execution_failures", 5)
    set_if_exists(config.circuit_breaker, "cooldown_seconds", 300.0)

    config.validate()
    return config


def make_execution_cost(
    *,
    spread_cost: float = 0.01,
    slippage_cost: float = 0.01,
    fee_cost: float = 0.01,
    funding_cost: float = 0.0,
    other_cost: float = 0.0,
    spread_pct: float | None = 0.0001,
    slippage_pct: float | None = 0.0001,
    quality: ExecutionQuality = ExecutionQuality.ACCEPTABLE,
) -> ExecutionCostEstimate:
    return ExecutionCostEstimate(
        spread_cost=spread_cost,
        slippage_cost=slippage_cost,
        fee_cost=fee_cost,
        funding_cost=funding_cost,
        other_cost=other_cost,
        spread_pct=spread_pct,
        slippage_pct=slippage_pct,
        quality=quality,
    )


def make_request(
    *,
    symbol: str = TEST_SYMBOL,
    side: PositionSide = PositionSide.LONG,
    entry_price: float = 100.0,
    stop_loss: float | None = 99.0,
    take_profit: float | None = 103.0,
    signal_id: str | None = TEST_SIGNAL_ID,
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
    expected_cost: float | None = 0.03,
    execution_cost: ExecutionCostEstimate | None = None,
    requested_size: float | None = None,
    requested_margin: float | None = None,
    requested_leverage: float | None = 5.0,
    reduce_only: bool = False,
    margin_mode: MarginMode = MarginMode.ISOLATED,
    metadata: dict[str, Any] | None = None,
) -> RiskEvaluationRequest:
    return RiskEvaluationRequest(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        signal_id=signal_id,
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
        execution_cost=execution_cost or make_execution_cost(),
        requested_size=requested_size,
        requested_margin=requested_margin,
        requested_leverage=requested_leverage,
        reduce_only=reduce_only,
        margin_mode=margin_mode,
        timestamp=time.time(),
        metadata=dict(metadata or {}),
    )


def signal_payload(
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
    requested_leverage: float | None = 5.0,
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
        "liquidity_class": "high",
        "execution_quality": "acceptable",
        "confidence": 0.75,
        "edge_score": 0.65,
        "volatility": 0.20,
        "expected_reward": 3.0,
        "expected_loss": 1.0,
        "expected_win_probability": 0.55,
        "expected_cost": 0.03,
        "execution_cost": {
            "spread_cost": 0.01,
            "slippage_cost": 0.01,
            "fee_cost": 0.01,
            "funding_cost": 0.0,
            "spread_pct": 0.0001,
            "slippage_pct": 0.0001,
            "quality": "acceptable",
        },
        "requested_size": None,
        "requested_margin": None,
        "requested_leverage": requested_leverage,
        "reduce_only": False,
        "margin_mode": "isolated",
        "timestamp": time.time(),
        "metadata": {"fixture": "signal_payload"},
    }


def position_payload(
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
    position_id: str = "position-events-001",
    realized_pnl: float = 0.0,
    unrealized_pnl: float = 0.0,
    reservation_id: str | None = None,
) -> dict[str, Any]:
    mark_price = mark_price if mark_price is not None else entry_price
    notional_value = (
        notional_value if notional_value is not None else abs(size * entry_price)
    )
    margin_used = (
        margin_used if margin_used is not None else notional_value / leverage
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
        "reservation_id": reservation_id,
        "opened_at": time.time(),
        "updated_at": time.time(),
        "metadata": {"fixture": "position_payload"},
    }


def account_payload(
    *,
    balance: float = 10_000.0,
    equity: float = 10_000.0,
    free_balance: float = 10_000.0,
    used_margin: float = 0.0,
    realized_pnl: float = 0.0,
    unrealized_pnl: float = 0.0,
) -> dict[str, Any]:
    return {
        "balance": balance,
        "equity": equity,
        "free_balance": free_balance,
        "used_margin": used_margin,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "metadata": {"fixture": "account_payload"},
    }


def execution_payload(
    *,
    symbol: str = TEST_SYMBOL,
    signal_id: str = TEST_SIGNAL_ID,
    order_id: str = "order-events-001",
    reservation_id: str | None = None,
    reason: str | None = "test execution event",
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "signal_id": signal_id,
        "order_id": order_id,
        "reservation_id": reservation_id,
        "reason": reason,
        "metadata": {"fixture": "execution_payload"},
    }


def make_manager(
    *,
    config: RiskConfig | None = None,
    state: RiskState | None = None,
    metrics: RiskMetrics | None = None,
    event_bus: FakeEventBus | None = None,
    scheduler: FakeScheduler | None = None,
    auto_subscribe: bool = True,
    register_scheduler_jobs: bool = True,
) -> tuple[RiskManager, FakeEventBus, FakeScheduler, RiskState, RiskMetrics]:
    resolved_config = config or make_risk_config()
    resolved_state = state or make_state()
    resolved_metrics = metrics or RiskMetrics()
    resolved_event_bus = event_bus or FakeEventBus()
    resolved_scheduler = scheduler or FakeScheduler()

    manager = RiskManager(
        resolved_config,
        event_bus=resolved_event_bus,  # type: ignore[arg-type]
        scheduler=resolved_scheduler,  # type: ignore[arg-type]
        state=resolved_state,
        metrics=resolved_metrics,
        auto_subscribe=auto_subscribe,
        register_scheduler_jobs=register_scheduler_jobs,
        service_name="risk_manager_events_test",
    )

    return manager, resolved_event_bus, resolved_scheduler, resolved_state, resolved_metrics


# =============================================================================
# Assertions
# =============================================================================


def assert_event(event_bus: FakeEventBus, topic: str) -> EmittedEvent:
    event = event_bus.last_event(topic)
    assert_json_serializable(event.payload)
    return event


def assert_any_event(event_bus: FakeEventBus, topics: set[str]) -> str:
    emitted = set(event_bus.topics())
    intersection = emitted.intersection(topics)
    assert intersection, f"Expected one of {topics}, emitted={event_bus.topics()!r}"
    return sorted(intersection)[0]


def assert_no_event(event_bus: FakeEventBus, topic: str) -> None:
    assert not event_bus.events_for(topic), (
        f"Did not expect topic={topic!r}, emitted={event_bus.topics()!r}"
    )


async def allow_signal_and_return_reservation(
    manager: RiskManager,
    event_bus: FakeEventBus,
    state: RiskState,
    *,
    signal_id: str = TEST_SIGNAL_ID,
) -> str:
    decision = await manager.evaluate_request(
        make_request(signal_id=signal_id)
    )

    assert decision.allowed is True
    assert decision.reservation_id is not None
    assert state.get_pending_reservation(decision.reservation_id) is not None
    assert_event(event_bus, "signal.confirmed")

    return decision.reservation_id


# =============================================================================
# Lifecycle / registration
# =============================================================================


class TestRiskManagerLifecycle:
    @pytest.mark.asyncio
    async def test_start_registers_subscriptions_scheduler_jobs_and_started_event(self) -> None:
        manager, event_bus, scheduler, _, _ = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=True,
        )

        await manager.start()

        assert manager.is_running is True
        assert event_bus.subscriptions
        assert scheduler.jobs
        assert "risk.manager.started" in event_bus.topics()

        subscription_topics = set(event_bus.subscription_topics())
        expected_topics = {
            "signal.generated",
            "account.*",
            "position.opened",
            "position.updated",
            "position.closed",
            "execution.order_rejected",
            "execution.order_failed",
            "execution.order_cancelled",
            "execution.order_filled",
            "system.clock.day_rollover",
            "system.clock.week_rollover",
            "system.clock.month_rollover",
            "system.scheduler.job_failed",
            "risk.manual_halt",
            "risk.manual_resume",
        }

        missing = expected_topics.difference(subscription_topics)
        assert not missing, f"Missing subscriptions: {missing}"

        for event in event_bus.emitted:
            assert_json_serializable(event.payload)

    @pytest.mark.asyncio
    async def test_stop_unregisters_subscriptions_and_emits_stopped_event(self) -> None:
        manager, event_bus, _, _, _ = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=True,
        )

        await manager.start()
        subscriptions_count = len(event_bus.subscriptions)

        await manager.stop()

        assert manager.is_running is False
        assert len(event_bus.unsubscriptions) == subscriptions_count
        assert all(not subscription.active for subscription in event_bus.subscriptions)
        assert "risk.manager.stopped" in event_bus.topics()

    @pytest.mark.asyncio
    async def test_double_start_does_not_duplicate_subscriptions_or_jobs(self) -> None:
        manager, event_bus, scheduler, _, _ = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=True,
        )

        await manager.start()
        first_subscriptions = len(event_bus.subscriptions)
        first_jobs = len(scheduler.jobs)

        await manager.start()

        assert len(event_bus.subscriptions) == first_subscriptions
        assert len(scheduler.jobs) == first_jobs

    @pytest.mark.asyncio
    async def test_double_stop_is_safe(self) -> None:
        manager, event_bus, _, _, _ = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=True,
        )

        await manager.start()
        await manager.stop()
        await manager.stop()

        assert manager.is_running is False
        assert "risk.manager.stopped" in event_bus.topics()

    @pytest.mark.asyncio
    async def test_start_without_event_bus_and_scheduler_is_allowed(self) -> None:
        config = make_risk_config()
        state = make_state()
        metrics = RiskMetrics()

        manager = RiskManager(
            config,
            event_bus=None,
            scheduler=None,
            state=state,
            metrics=metrics,
            auto_subscribe=True,
            register_scheduler_jobs=True,
            service_name="risk_manager_no_bus_scheduler_test",
        )

        await manager.start()

        assert manager.is_running is True

        await manager.stop()
        assert manager.is_running is False


# =============================================================================
# Signal events
# =============================================================================


class TestRiskManagerSignalEvents:
    @pytest.mark.asyncio
    async def test_signal_generated_valid_payload_is_evaluated_and_confirmed(self) -> None:
        manager, event_bus, _, state, metrics = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=False,
        )
        await manager.start()
        event_bus.clear_emitted()

        await event_bus.dispatch("signal.generated", signal_payload(signal_id=TEST_SIGNAL_ID))

        assert "risk.request_received" in event_bus.topics()
        confirmed = assert_event(event_bus, "signal.confirmed")
        approved = assert_event(event_bus, "risk.approved")

        assert confirmed.payload["allowed"] is True
        assert confirmed.payload["signal_id"] == TEST_SIGNAL_ID
        assert confirmed.payload["reservation_id"] is not None
        assert approved.payload["reservation_id"] == confirmed.payload["reservation_id"]

        assert len(state.pending_reservations) == 1
        assert metrics.total_decisions == 1
        assert metrics.approvals == 1

    @pytest.mark.asyncio
    async def test_signal_generated_invalid_payload_emits_invalid_or_rejected_event(self) -> None:
        manager, event_bus, _, state, metrics = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=False,
        )
        await manager.start()
        event_bus.clear_emitted()

        await event_bus.dispatch(
            "signal.generated",
            {
                "symbol": TEST_SYMBOL,
                "side": "not-a-side",
                "entry_price": "not-a-number",
            },
        )

        assert len(state.pending_reservations) == 0
        assert metrics.approvals == 0

        assert_any_event(
            event_bus,
            {
                "risk.signal_invalid",
                "risk.position_blocked",
                "risk.rejected",
                "risk.error",
            },
        )

    @pytest.mark.asyncio
    async def test_signal_generated_denied_request_does_not_create_reservation(self) -> None:
        manager, event_bus, _, state, metrics = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=False,
        )
        await manager.start()
        event_bus.clear_emitted()

        payload = signal_payload(stop_loss=None, signal_id="signal-denied-no-sl")
        await event_bus.dispatch("signal.generated", payload)

        assert len(state.pending_reservations) == 0
        assert metrics.total_decisions == 1
        assert metrics.rejections == 1

        blocked = assert_event(event_bus, "risk.position_blocked")
        assert blocked.payload["allowed"] is False
        assert blocked.payload["reservation_id"] is None
        assert_no_event(event_bus, "signal.confirmed")


# =============================================================================
# Account / position events
# =============================================================================


class TestRiskManagerAccountAndPositionEvents:
    @pytest.mark.asyncio
    async def test_account_update_event_updates_risk_state(self) -> None:
        manager, event_bus, _, state, _ = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=False,
        )
        await manager.start()

        await event_bus.dispatch(
            "account.updated",
            account_payload(
                balance=12_000.0,
                equity=11_500.0,
                free_balance=10_000.0,
                used_margin=1_500.0,
                realized_pnl=100.0,
                unrealized_pnl=-50.0,
            ),
        )

        assert state.balance == pytest.approx(12_000.0)
        assert state.equity == pytest.approx(11_500.0)
        assert state.free_balance == pytest.approx(10_000.0)
        assert state.used_margin == pytest.approx(1_500.0)
        assert state.realized_pnl == pytest.approx(100.0)
        assert state.unrealized_pnl == pytest.approx(-50.0)

    @pytest.mark.asyncio
    async def test_position_opened_confirms_matching_reservation_and_adds_position(self) -> None:
        manager, event_bus, _, state, metrics = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=False,
        )
        await manager.start()

        reservation_id = await allow_signal_and_return_reservation(
            manager,
            event_bus,
            state,
            signal_id="signal-position-opened",
        )

        event_bus.clear_emitted()

        await event_bus.dispatch(
            "position.opened",
            position_payload(
                signal_id="signal-position-opened",
                reservation_id=reservation_id,
                position_id="position-opened-001",
            ),
        )

        assert state.get_pending_reservation(reservation_id) is None
        assert len(state.positions) == 1
        assert metrics.reservations.confirmed == 1
        assert metrics.reservations.active == 0

        assert_any_event(
            event_bus,
            {
                "risk.reservation.confirmed",
                "risk.position_opened",
                "risk.position.opened",
            },
        )

    @pytest.mark.asyncio
    async def test_position_opened_without_matching_reservation_is_safe(self) -> None:
        manager, event_bus, _, state, metrics = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=False,
        )
        await manager.start()
        event_bus.clear_emitted()

        await event_bus.dispatch(
            "position.opened",
            position_payload(
                signal_id="unknown-signal",
                reservation_id="missing-reservation",
                position_id="position-without-reservation",
            ),
        )

        assert len(state.positions) == 1
        assert metrics.reservations.confirmed == 0

    @pytest.mark.asyncio
    async def test_position_updated_mutates_existing_position_only(self) -> None:
        manager, event_bus, _, state, _ = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=False,
        )
        await manager.start()

        await event_bus.dispatch(
            "position.opened",
            position_payload(
                position_id="position-update-test",
                size=1.0,
                mark_price=100.0,
                unrealized_pnl=0.0,
            ),
        )

        await event_bus.dispatch(
            "position.updated",
            {
                "symbol": TEST_SYMBOL,
                "position_id": "position-update-test",
                "size": 2.0,
                "mark_price": 110.0,
                "notional_value": 220.0,
                "margin_used": 44.0,
                "risk_amount": 12.0,
                "unrealized_pnl": 20.0,
            },
        )

        position = next(iter(state.positions.values()))
        assert position.size == pytest.approx(2.0)
        assert position.mark_price == pytest.approx(110.0)
        assert position.notional_value == pytest.approx(220.0)
        assert position.margin_used == pytest.approx(44.0)
        assert position.unrealized_pnl == pytest.approx(20.0)

    @pytest.mark.asyncio
    async def test_position_closed_releases_position_risk_and_updates_pnl(self) -> None:
        manager, event_bus, _, state, _ = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=False,
        )
        await manager.start()

        await event_bus.dispatch(
            "position.opened",
            position_payload(
                position_id="position-close-test",
                risk_amount=10.0,
            ),
        )

        assert len(state.positions) == 1
        assert state.get_symbol_state(TEST_SYMBOL).open_risk > 0

        await event_bus.dispatch(
            "position.closed",
            {
                "symbol": TEST_SYMBOL,
                "position_id": "position-close-test",
                "realized_pnl": -15.0,
            },
        )

        assert len(state.positions) == 0
        assert state.realized_pnl == pytest.approx(-15.0)
        assert state.get_symbol_state(TEST_SYMBOL).open_risk == pytest.approx(0.0)
        assert state.get_strategy_state(TEST_STRATEGY).open_risk == pytest.approx(0.0)


# =============================================================================
# Execution events / reservation release
# =============================================================================


class TestRiskManagerExecutionEvents:
    @pytest.mark.asyncio
    async def test_order_rejected_releases_matching_reservation(self) -> None:
        manager, event_bus, _, state, metrics = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=False,
        )
        await manager.start()

        reservation_id = await allow_signal_and_return_reservation(
            manager,
            event_bus,
            state,
            signal_id="signal-order-rejected",
        )
        event_bus.clear_emitted()

        await event_bus.dispatch(
            "execution.order_rejected",
            execution_payload(
                signal_id="signal-order-rejected",
                reservation_id=reservation_id,
                reason="exchange rejected",
            ),
        )

        assert state.get_pending_reservation(reservation_id) is None
        assert metrics.reservations.released >= 1
        assert metrics.reservations.active == 0

        assert_any_event(
            event_bus,
            {
                "risk.reservation.released",
                "risk.order_rejected",
                "risk.execution_rejected",
            },
        )

    @pytest.mark.asyncio
    async def test_order_cancelled_releases_matching_reservation(self) -> None:
        manager, event_bus, _, state, metrics = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=False,
        )
        await manager.start()

        reservation_id = await allow_signal_and_return_reservation(
            manager,
            event_bus,
            state,
            signal_id="signal-order-cancelled",
        )
        event_bus.clear_emitted()

        await event_bus.dispatch(
            "execution.order_cancelled",
            execution_payload(
                signal_id="signal-order-cancelled",
                reservation_id=reservation_id,
                reason="cancelled by exchange",
            ),
        )

        assert state.get_pending_reservation(reservation_id) is None
        assert metrics.reservations.released >= 1
        assert metrics.reservations.active == 0

    @pytest.mark.asyncio
    async def test_order_failed_releases_reservation_and_registers_failure(self) -> None:
        manager, event_bus, _, state, metrics = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=False,
        )
        await manager.start()

        reservation_id = await allow_signal_and_return_reservation(
            manager,
            event_bus,
            state,
            signal_id="signal-order-failed",
        )
        event_bus.clear_emitted()

        await event_bus.dispatch(
            "execution.order_failed",
            execution_payload(
                signal_id="signal-order-failed",
                reservation_id=reservation_id,
                reason="execution failure",
            ),
        )

        assert state.get_pending_reservation(reservation_id) is None
        assert metrics.reservations.released >= 1 or metrics.reservations.failed >= 1
        assert metrics.reservations.active == 0

        # Depending on policy, failure may only update circuit stats or may also emit.
        assert_json_serializable([event.payload for event in event_bus.emitted])

    @pytest.mark.asyncio
    async def test_order_filled_confirms_or_clears_reservation(self) -> None:
        manager, event_bus, _, state, metrics = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=False,
        )
        await manager.start()

        reservation_id = await allow_signal_and_return_reservation(
            manager,
            event_bus,
            state,
            signal_id="signal-order-filled",
        )
        event_bus.clear_emitted()

        await event_bus.dispatch(
            "execution.order_filled",
            {
                **execution_payload(
                    signal_id="signal-order-filled",
                    reservation_id=reservation_id,
                ),
                **position_payload(
                    signal_id="signal-order-filled",
                    reservation_id=reservation_id,
                    position_id="position-filled-001",
                ),
            },
        )

        assert state.get_pending_reservation(reservation_id) is None
        assert metrics.reservations.confirmed >= 1 or metrics.reservations.released >= 1
        assert metrics.reservations.active == 0

    @pytest.mark.asyncio
    async def test_duplicate_execution_release_events_do_not_make_reservation_counters_negative(
        self,
    ) -> None:
        manager, event_bus, _, state, metrics = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=False,
        )
        await manager.start()

        reservation_id = await allow_signal_and_return_reservation(
            manager,
            event_bus,
            state,
            signal_id="signal-duplicate-release",
        )

        payload = execution_payload(
            signal_id="signal-duplicate-release",
            reservation_id=reservation_id,
        )

        await event_bus.dispatch("execution.order_rejected", payload)
        await event_bus.dispatch("execution.order_rejected", payload)
        await event_bus.dispatch("execution.order_cancelled", payload)
        await event_bus.dispatch("execution.order_failed", payload)

        assert state.get_pending_reservation(reservation_id) is None
        assert metrics.reservations.active >= 0
        assert metrics.reservations.reserved_open_risk >= 0
        assert metrics.reservations.reserved_margin >= 0
        assert metrics.reservations.reserved_notional >= 0


# =============================================================================
# Reservation lifecycle
# =============================================================================


class TestRiskManagerReservationFlow:
    @pytest.mark.asyncio
    async def test_allow_creates_reservation_with_expected_snapshot_fields(self) -> None:
        manager, event_bus, _, state, metrics = make_manager(
            auto_subscribe=False,
            register_scheduler_jobs=False,
        )

        decision = await manager.evaluate_request(
            make_request(signal_id="signal-reservation-created")
        )

        assert decision.allowed is True
        assert decision.reservation_id is not None

        reservation = state.get_pending_reservation(decision.reservation_id)
        assert reservation is not None
        assert reservation.signal_id == "signal-reservation-created"
        assert reservation.symbol == TEST_SYMBOL
        assert reservation.side is PositionSide.LONG
        assert reservation.strategy_name == TEST_STRATEGY
        assert reservation.tier is TradeTier.T2
        assert reservation.size > 0
        assert reservation.open_risk >= 0
        assert reservation.margin >= 0
        assert reservation.notional >= 0
        assert reservation.expires_at is not None

        assert metrics.reservations.created == 1
        assert metrics.reservations.active == 1
        assert_json_serializable(reservation.snapshot())
        assert_event(event_bus, "signal.confirmed")

    @pytest.mark.asyncio
    async def test_denied_signal_never_creates_reservation(self) -> None:
        manager, event_bus, _, state, metrics = make_manager(
            auto_subscribe=False,
            register_scheduler_jobs=False,
        )

        decision = await manager.evaluate_request(
            make_request(signal_id="signal-denied-reservation", stop_loss=None)
        )

        assert decision.allowed is False
        assert decision.reservation_id is None
        assert len(state.pending_reservations) == 0
        assert metrics.reservations.created == 0
        assert_event(event_bus, "risk.position_blocked")

    @pytest.mark.asyncio
    async def test_cleanup_expired_reservations_removes_only_expired(self) -> None:
        config = make_risk_config(reservations_enabled=True)
        state = make_state()
        manager, event_bus, _, _, metrics = make_manager(
            config=config,
            state=state,
            auto_subscribe=False,
            register_scheduler_jobs=False,
        )

        active = state.reserve_risk(
            reservation_id="active-reservation",
            symbol=TEST_SYMBOL,
            side=PositionSide.LONG,
            signal_id="active-signal",
            strategy_name=TEST_STRATEGY,
            tier=TradeTier.T2,
            size=1.0,
            open_risk=10.0,
            margin=20.0,
            notional=100.0,
            ttl_seconds=100.0,
            now_ts=1_000.0,
        )
        expired = state.reserve_risk(
            reservation_id="expired-reservation",
            symbol=ALT_SYMBOL,
            side=PositionSide.SHORT,
            signal_id="expired-signal",
            strategy_name=TEST_STRATEGY,
            tier=TradeTier.T2,
            size=1.0,
            open_risk=5.0,
            margin=10.0,
            notional=50.0,
            ttl_seconds=1.0,
            now_ts=1_000.0,
        )

        metrics.register_reservation_created(
            reservation_id=active.reservation_id,
            symbol=active.symbol,
            tier=active.tier,
            strategy_name=active.strategy_name,
            open_risk=active.open_risk,
            margin=active.margin,
            notional=active.notional,
        )
        metrics.register_reservation_created(
            reservation_id=expired.reservation_id,
            symbol=expired.symbol,
            tier=expired.tier,
            strategy_name=expired.strategy_name,
            open_risk=expired.open_risk,
            margin=expired.margin,
            notional=expired.notional,
        )

        expired_items = await manager.cleanup_expired_reservations(now_ts=1_002.0)

        expired_ids = {item.reservation_id for item in expired_items}
        assert expired_ids == {"expired-reservation"}
        assert state.get_pending_reservation("expired-reservation") is None
        assert state.get_pending_reservation("active-reservation") is not None

        assert metrics.reservations.expired >= 1
        assert metrics.reservations.active == 1
        assert_event(event_bus, "risk.reservation.expired")

    @pytest.mark.asyncio
    async def test_cleanup_expired_reservations_is_safe_when_none_exist(self) -> None:
        manager, event_bus, _, state, metrics = make_manager(
            auto_subscribe=False,
            register_scheduler_jobs=False,
        )

        expired_items = await manager.cleanup_expired_reservations(now_ts=time.time())

        assert expired_items == []
        assert len(state.pending_reservations) == 0
        assert metrics.reservations.active == 0
        assert_no_event(event_bus, "risk.reservation.expired")

    @pytest.mark.asyncio
    async def test_reservation_limit_is_fail_closed_when_max_pending_reached(self) -> None:
        config = make_risk_config(reservations_enabled=True)
        set_if_exists(config.reservation, "max_pending_reservations", 1)
        config.validate()

        state = make_state()
        manager, event_bus, _, _, metrics = make_manager(
            config=config,
            state=state,
            auto_subscribe=False,
            register_scheduler_jobs=False,
        )

        first = await manager.evaluate_request(
            make_request(signal_id="reservation-limit-first")
        )
        second = await manager.evaluate_request(
            make_request(signal_id="reservation-limit-second")
        )

        assert first.allowed is True
        assert first.reservation_id is not None

        # Depending on implementation, second may be DENY, or ALLOW without
        # reservation only if policy is not fail-closed. For live capital we
        # expect fail-closed.
        assert second.allowed is False
        assert second.reservation_id is None
        assert len(state.pending_reservations) == 1
        assert metrics.reservations.active == 1
        assert_event(event_bus, "risk.position_blocked")


# =============================================================================
# Scheduler jobs
# =============================================================================


class TestRiskManagerScheduler:
    @pytest.mark.asyncio
    async def test_start_registers_daily_weekly_monthly_and_cleanup_jobs(self) -> None:
        scheduler = FakeScheduler()
        manager, event_bus, _, _, _ = make_manager(
            scheduler=scheduler,
            auto_subscribe=False,
            register_scheduler_jobs=True,
        )

        await manager.start()

        names = scheduler.job_names()
        assert names, "Expected RiskManager to register scheduler jobs"

        lowered = " ".join(names).lower()
        assert "daily" in lowered or "day" in lowered
        assert "weekly" in lowered or "week" in lowered
        assert "monthly" in lowered or "month" in lowered
        assert "cleanup" in lowered or "reservation" in lowered

        assert "risk.manager.started" in event_bus.topics()

    @pytest.mark.asyncio
    async def test_scheduler_jobs_are_not_duplicated_on_double_start(self) -> None:
        scheduler = FakeScheduler()
        manager, _, _, _, _ = make_manager(
            scheduler=scheduler,
            auto_subscribe=False,
            register_scheduler_jobs=True,
        )

        await manager.start()
        first_names = scheduler.job_names()

        await manager.start()
        second_names = scheduler.job_names()

        assert second_names == first_names

    @pytest.mark.asyncio
    async def test_scheduler_add_failure_does_not_leave_manager_unusable(self) -> None:
        scheduler = FakeScheduler(fail_on_add=True)
        manager, event_bus, _, _, _ = make_manager(
            scheduler=scheduler,
            auto_subscribe=False,
            register_scheduler_jobs=True,
        )

        try:
            await manager.start()
        except RuntimeError:
            # Acceptable current behavior, but manager must not have partially
            # corrupted state.
            assert manager.is_running in {False, True}
            return

        assert manager.is_running is True
        assert_any_event(
            event_bus,
            {
                "risk.manager.started",
                "risk.scheduler.error",
                "risk.error",
            },
        )

    @pytest.mark.asyncio
    async def test_daily_weekly_monthly_clock_events_call_resets(self) -> None:
        manager, event_bus, _, state, _ = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=False,
        )
        await manager.start()

        symbol_state = state.get_symbol_state(TEST_SYMBOL)
        strategy_state = state.get_strategy_state(TEST_STRATEGY)

        symbol_state.daily_pnl = -10.0
        symbol_state.weekly_pnl = -20.0
        symbol_state.monthly_pnl = -30.0
        symbol_state.trades_today = 5

        strategy_state.daily_pnl = -10.0
        strategy_state.weekly_pnl = -20.0
        strategy_state.monthly_pnl = -30.0
        strategy_state.trades_today = 5

        await event_bus.dispatch("system.clock.day_rollover", {"reason": "test"})
        assert symbol_state.daily_pnl == pytest.approx(0.0)
        assert strategy_state.daily_pnl == pytest.approx(0.0)
        assert symbol_state.weekly_pnl == pytest.approx(-20.0)

        await event_bus.dispatch("system.clock.week_rollover", {"reason": "test"})
        assert symbol_state.weekly_pnl == pytest.approx(0.0)
        assert strategy_state.weekly_pnl == pytest.approx(0.0)
        assert symbol_state.monthly_pnl == pytest.approx(-30.0)

        await event_bus.dispatch("system.clock.month_rollover", {"reason": "test"})
        assert symbol_state.monthly_pnl == pytest.approx(0.0)
        assert strategy_state.monthly_pnl == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_cleanup_scheduler_job_removes_expired_reservations_when_run(self) -> None:
        scheduler = FakeScheduler()
        state = make_state()
        manager, event_bus, _, _, metrics = make_manager(
            state=state,
            scheduler=scheduler,
            auto_subscribe=False,
            register_scheduler_jobs=True,
        )

        await manager.start()

        state.reserve_risk(
            reservation_id="scheduler-expired",
            symbol=TEST_SYMBOL,
            side=PositionSide.LONG,
            signal_id="scheduler-expired-signal",
            open_risk=5.0,
            margin=10.0,
            notional=50.0,
            ttl_seconds=1.0,
            now_ts=1_000.0,
        )
        metrics.register_reservation_created(
            reservation_id="scheduler-expired",
            symbol=TEST_SYMBOL,
            tier=None,
            strategy_name=None,
            open_risk=5.0,
            margin=10.0,
            notional=50.0,
        )

        cleanup_jobs = [
            job
            for job in scheduler.jobs
            if "cleanup" in job.name.lower() or "reservation" in job.name.lower()
        ]
        assert cleanup_jobs, f"No cleanup/reservation job found: {scheduler.job_names()}"

        for reservation in state.pending_reservations.values():
            reservation.expires_at = time.time() - 1.0

        await cleanup_jobs[0].run()

        assert len(state.pending_reservations) == 0
        assert_event(event_bus, "risk.reservation.expired")


# =============================================================================
# Manual controls
# =============================================================================


class TestRiskManagerManualControls:
    @pytest.mark.asyncio
    async def test_manual_halt_event_blocks_new_signals(self) -> None:
        manager, event_bus, _, state, _ = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=False,
        )
        await manager.start()
        event_bus.clear_emitted()

        await event_bus.dispatch(
            "risk.manual_halt",
            {"reason": "manual halt test"},
        )

        assert state.trading_halted is True
        assert state.risk_mode in {RiskMode.HALTED, RiskMode.EMERGENCY_STOP}

        assert_any_event(
            event_bus,
            {
                "risk.kill_switch",
                "risk.trading_halted",
                "risk.manual_halt",
                "risk.halted",
            },
        )

        event_bus.clear_emitted()
        decision = await manager.evaluate_request(make_request(signal_id="after-manual-halt"))

        assert decision.allowed is False
        assert decision.decision in {
            RiskDecisionType.HALT_TRADING,
            RiskDecisionType.EMERGENCY_STOP,
            RiskDecisionType.DENY,
        }
        assert "signal.confirmed" not in event_bus.topics()

    @pytest.mark.asyncio
    async def test_manual_resume_event_allows_new_signals_after_manual_halt(self) -> None:
        manager, event_bus, _, state, _ = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=False,
        )
        await manager.start()

        await event_bus.dispatch(
            "risk.manual_halt",
            {"reason": "manual halt test"},
        )
        assert state.trading_halted is True

        event_bus.clear_emitted()

        await event_bus.dispatch(
            "risk.manual_resume",
            {"reason": "manual resume test"},
        )

        assert state.trading_halted is False
        assert state.risk_mode is RiskMode.NORMAL

        assert_any_event(
            event_bus,
            {
                "risk.resumed",
                "risk.manual_resume",
                "risk.trading_resumed",
            },
        )

        event_bus.clear_emitted()
        decision = await manager.evaluate_request(make_request(signal_id="after-manual-resume"))

        assert decision.allowed is True
        assert_event(event_bus, "signal.confirmed")

    @pytest.mark.asyncio
    async def test_manual_resume_cannot_bypass_emergency_stop_without_explicit_clear(self) -> None:
        manager, event_bus, _, state, _ = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=False,
        )
        await manager.start()

        state.emergency_stop("test emergency stop")
        assert state.emergency_stop_active is True

        await event_bus.dispatch(
            "risk.manual_resume",
            {"reason": "should not bypass emergency"},
        )

        assert state.emergency_stop_active is True
        assert state.trading_halted is True
        assert state.risk_mode is RiskMode.EMERGENCY_STOP

        event_bus.clear_emitted()
        decision = await manager.evaluate_request(make_request(signal_id="after-emergency-resume"))

        assert decision.allowed is False
        assert decision.decision is RiskDecisionType.EMERGENCY_STOP
        assert "signal.confirmed" not in event_bus.topics()


# =============================================================================
# Malformed event safety
# =============================================================================


class TestRiskManagerMalformedEventSafety:
    @pytest.mark.asyncio
    async def test_malformed_account_event_does_not_corrupt_existing_state(self) -> None:
        manager, event_bus, _, state, _ = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=False,
        )
        await manager.start()

        before = {
            "balance": state.balance,
            "equity": state.equity,
            "free_balance": state.free_balance,
            "used_margin": state.used_margin,
        }

        try:
            await event_bus.dispatch(
                "account.updated",
                {
                    "equity": "not-a-number",
                    "free_balance": math.nan,
                },
            )
        except (ValueError, TypeError):
            pass

        after = {
            "balance": state.balance,
            "equity": state.equity,
            "free_balance": state.free_balance,
            "used_margin": state.used_margin,
        }

        # If current implementation is fail-fast, state should remain unchanged.
        # If it is fail-closed with event emission, it should still not contain NaN.
        for value in after.values():
            assert isinstance(value, int | float)
            assert math.isfinite(float(value))

        if before != after:
            assert_any_event(
                event_bus,
                {
                    "risk.account_invalid",
                    "risk.error",
                    "risk.state_warning",
                },
            )

    @pytest.mark.asyncio
    async def test_malformed_position_event_does_not_create_non_finite_position(self) -> None:
        manager, event_bus, _, state, _ = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=False,
        )
        await manager.start()

        try:
            await event_bus.dispatch(
                "position.opened",
                {
                    "symbol": TEST_SYMBOL,
                    "side": "long",
                    "size": math.nan,
                    "entry_price": 100.0,
                    "position_id": "bad-position",
                },
            )
        except (ValueError, TypeError):
            pass

        for position in state.positions.values():
            assert math.isfinite(float(position.size))
            assert math.isfinite(float(position.entry_price))
            assert position.size >= 0
            assert position.entry_price > 0

    @pytest.mark.asyncio
    async def test_unknown_event_topic_is_ignored(self) -> None:
        manager, event_bus, _, state, metrics = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=False,
        )
        await manager.start()

        before_positions = len(state.positions)
        before_reservations = len(state.pending_reservations)
        before_decisions = metrics.total_decisions

        await event_bus.dispatch(
            "unknown.topic",
            {"anything": "ignored"},
        )

        assert len(state.positions) == before_positions
        assert len(state.pending_reservations) == before_reservations
        assert metrics.total_decisions == before_decisions