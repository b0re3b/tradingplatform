# tests/risk/test_risk_scenarios_stress_regression.py
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
    CircuitBreakerReason,
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
from risk.models import ExecutionCostEstimate, RiskDecision, RiskEvaluationRequest
from risk.risk_manager import RiskManager
from risk.state import RiskState


TEST_SYMBOL = "BTCUSDT"
ALT_SYMBOL = "ETHUSDT"
THIRD_SYMBOL = "SOLUSDT"
TEST_STRATEGY = "test_strategy"
ALT_STRATEGY = "alt_strategy"


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


@dataclass(slots=True)
class EmittedEvent:
    topic: str
    payload: dict[str, Any]
    priority: Any | None = None
    source: str | None = None
    timestamp: float = field(default_factory=time.time)


class FakeEventBus:
    def __init__(self) -> None:
        self.subscriptions: list[FakeSubscription] = []
        self.unsubscriptions: list[FakeSubscription] = []
        self.emitted: list[EmittedEvent] = []

    def subscribe(
        self,
        topic: str,
        handler: Callable[..., Any],
        *,
        name: str | None = None,
        **_: Any,
    ) -> FakeSubscription:
        subscription = FakeSubscription(topic=topic, handler=handler, name=name)
        self.subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, subscription: FakeSubscription) -> None:
        subscription.active = False
        self.unsubscriptions.append(subscription)

    async def emit(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
        *,
        priority: Any | None = None,
        source: str | None = None,
        **_: Any,
    ) -> None:
        self.emitted.append(
            EmittedEvent(
                topic=topic,
                payload=dict(payload or {}),
                priority=priority,
                source=source,
            )
        )

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

    def events_for(self, topic: str) -> list[EmittedEvent]:
        return [event for event in self.emitted if event.topic == topic]

    def last_event(self, topic: str) -> EmittedEvent:
        events = self.events_for(topic)
        assert events, f"Expected topic={topic!r}, emitted={self.topics()!r}"
        return events[-1]

    def clear(self) -> None:
        self.emitted.clear()


@dataclass(slots=True)
class FakeScheduledJob:
    name: str
    callback: Callable[..., Any]
    interval_seconds: float
    run_immediately: bool = False
    run_count: int = 0

    async def run(self) -> Any:
        self.run_count += 1
        result = self.callback()
        if inspect.isawaitable(result):
            result = await result
        return result


class FakeScheduler:
    def __init__(self) -> None:
        self.jobs: list[FakeScheduledJob] = []

    def add_interval_job(
        self,
        callback: Callable[..., Any],
        *,
        interval_seconds: float,
        name: str,
        run_immediately: bool = False,
        **_: Any,
    ) -> FakeScheduledJob:
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
    assert loss_amount >= 0.0
    state.daily_start_equity = state.equity + loss_amount


def force_weekly_loss(state: RiskState, loss_amount: float) -> None:
    assert loss_amount >= 0.0
    state.weekly_start_equity = state.equity + loss_amount


def force_monthly_loss(state: RiskState, loss_amount: float) -> None:
    assert loss_amount >= 0.0
    state.monthly_start_equity = state.equity + loss_amount


def make_risk_config(
    *,
    reservations_enabled: bool = True,
    max_open_risk_r: float = 100.0,
    max_pending_reservations: int = 100,
    max_pending_per_symbol: int = 100,
    max_pending_per_strategy: int = 100,
    strict_execution_cost: bool = False,
    tight_exposure: bool = False,
    auto_expire_on_evaluate: bool = True,
    circuit_breaker_failures: int = 3,
) -> RiskConfig:
    config = RiskConfig()

    # Deterministic R/sizing.
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

    if tight_exposure:
        normal_open_risk = max_open_risk_r
        safe_open_risk = max(0.01, max_open_risk_r * 0.5)
        aggressive_open_risk = max_open_risk_r * 1.5

        normal_margin = 0.01
        safe_margin = 0.005
        aggressive_margin = 0.02

        normal_total = 0.01
        safe_total = 0.005
        aggressive_total = 0.02

        normal_symbol = 0.01
        safe_symbol = 0.005
        aggressive_symbol = 0.02

        normal_side = 0.01
        safe_side = 0.005
        aggressive_side = 0.02

        max_open_positions = 1
    else:
        normal_open_risk = max_open_risk_r
        safe_open_risk = max(0.01, max_open_risk_r * 0.5)
        aggressive_open_risk = max_open_risk_r * 1.5

        normal_margin = 1.0
        safe_margin = 0.5
        aggressive_margin = 1.5

        normal_total = 10.0
        safe_total = 5.0
        aggressive_total = 15.0

        normal_symbol = 10.0
        safe_symbol = 5.0
        aggressive_symbol = 15.0

        normal_side = 10.0
        safe_side = 5.0
        aggressive_side = 15.0

        max_open_positions = 100

    set_if_exists(config.exposure, "max_open_risk_r", normal_open_risk)
    set_if_exists(config.exposure, "safe_mode_max_open_risk_r", safe_open_risk)
    set_if_exists(config.exposure, "aggressive_max_open_risk_r", aggressive_open_risk)

    set_if_exists(config.exposure, "max_used_margin_pct", normal_margin)
    set_if_exists(config.exposure, "safe_mode_max_used_margin_pct", safe_margin)
    set_if_exists(config.exposure, "aggressive_max_used_margin_pct", aggressive_margin)

    set_if_exists(config.exposure, "max_total_exposure_pct", normal_total)
    set_if_exists(config.exposure, "safe_mode_max_total_exposure_pct", safe_total)
    set_if_exists(config.exposure, "aggressive_max_total_exposure_pct", aggressive_total)

    set_if_exists(config.exposure, "max_symbol_exposure_pct", normal_symbol)
    set_if_exists(config.exposure, "safe_mode_max_symbol_exposure_pct", safe_symbol)
    set_if_exists(config.exposure, "aggressive_max_symbol_exposure_pct", aggressive_symbol)

    set_if_exists(config.exposure, "max_side_exposure_pct", normal_side)
    set_if_exists(config.exposure, "safe_mode_max_side_exposure_pct", safe_side)
    set_if_exists(config.exposure, "aggressive_max_side_exposure_pct", aggressive_side)

    set_if_exists(config.exposure, "max_open_positions", max_open_positions)

    if strict_execution_cost:
        set_if_exists(config.execution_cost, "enabled", True)
        set_if_exists(config.execution_cost, "max_spread_pct", 0.001)
        set_if_exists(config.execution_cost, "max_slippage_pct", 0.001)
        set_if_exists(config.execution_cost, "default_max_cost_to_reward_pct", 0.05)
        set_if_exists(config.execution_cost, "safe_mode_max_cost_to_reward_pct", 0.03)
        set_if_exists(config.execution_cost, "min_execution_quality", ExecutionQuality.ACCEPTABLE)
        set_if_exists(config.execution_cost, "require_positive_ev_after_cost", True)
    else:
        set_if_exists(config.execution_cost, "enabled", True)
        set_if_exists(config.execution_cost, "max_spread_pct", 1.0)
        set_if_exists(config.execution_cost, "max_slippage_pct", 1.0)
        set_if_exists(config.execution_cost, "default_max_cost_to_reward_pct", 1.0)
        set_if_exists(config.execution_cost, "safe_mode_max_cost_to_reward_pct", 0.5)
        set_if_exists(config.execution_cost, "min_execution_quality", ExecutionQuality.ACCEPTABLE)
        set_if_exists(config.execution_cost, "require_positive_ev_after_cost", True)

    set_if_exists(config.reservation, "enabled", reservations_enabled)
    set_if_exists(config.reservation, "reserve_on_allow", reservations_enabled)
    set_if_exists(config.reservation, "ttl_seconds", 30.0)
    set_if_exists(config.reservation, "cleanup_interval_seconds", 5.0)
    set_if_exists(config.reservation, "max_pending_reservations", max_pending_reservations)
    set_if_exists(config.reservation, "max_pending_per_symbol", max_pending_per_symbol)
    set_if_exists(config.reservation, "max_pending_per_strategy", max_pending_per_strategy)
    set_if_exists(config.reservation, "fail_closed_on_reservation_error", True)
    set_if_exists(config.reservation, "auto_expire_on_evaluate", auto_expire_on_evaluate)

    set_if_exists(config.circuit_breaker, "enabled", True)
    set_if_exists(config.circuit_breaker, "max_consecutive_failures", circuit_breaker_failures)
    set_if_exists(config.circuit_breaker, "max_execution_failures", circuit_breaker_failures)
    set_if_exists(config.circuit_breaker, "cooldown_seconds", 300.0)
    set_if_exists(config.circuit_breaker, "trigger_on_execution_cost_spike", True)
    set_if_exists(config.circuit_breaker, "trigger_on_data_feed_failure", True)

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
    signal_id: str | None = "signal-scenario",
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


def signal_payload_from_request(request: RiskEvaluationRequest) -> dict[str, Any]:
    return {
        "symbol": request.symbol,
        "side": request.side.value,
        "entry_price": request.entry_price,
        "stop_loss": request.stop_loss,
        "take_profit": request.take_profit,
        "signal_id": request.signal_id,
        "strategy_name": request.strategy_name,
        "tier": request.tier.value if request.tier else None,
        "order_intent": request.order_intent.value,
        "liquidity_class": request.liquidity_class.value,
        "execution_quality": request.execution_quality.value,
        "confidence": request.confidence,
        "edge_score": request.edge_score,
        "volatility": request.volatility,
        "expected_reward": request.expected_reward,
        "expected_loss": request.expected_loss,
        "expected_win_probability": request.expected_win_probability,
        "expected_cost": request.expected_cost,
        "execution_cost": dataclasses.asdict(request.execution_cost)
        if request.execution_cost is not None
        else None,
        "requested_size": request.requested_size,
        "requested_margin": request.requested_margin,
        "requested_leverage": request.requested_leverage,
        "reduce_only": request.reduce_only,
        "margin_mode": request.margin_mode.value,
        "timestamp": request.timestamp,
        "metadata": dict(request.metadata or {}),
    }


def execution_payload(
    *,
    symbol: str = TEST_SYMBOL,
    signal_id: str = "signal-scenario",
    reservation_id: str | None = None,
    order_id: str = "order-scenario",
    reason: str = "scenario execution event",
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "signal_id": signal_id,
        "reservation_id": reservation_id,
        "order_id": order_id,
        "reason": reason,
        "metadata": {"fixture": "execution_payload"},
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
    signal_id: str = "signal-scenario",
    position_id: str = "position-scenario",
    reservation_id: str | None = None,
    realized_pnl: float = 0.0,
    unrealized_pnl: float = 0.0,
) -> dict[str, Any]:
    mark_price = mark_price if mark_price is not None else entry_price
    notional_value = notional_value if notional_value is not None else abs(size * entry_price)
    margin_used = margin_used if margin_used is not None else notional_value / leverage

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
        "reservation_id": reservation_id,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "opened_at": time.time(),
        "updated_at": time.time(),
    }


def make_manager(
    *,
    config: RiskConfig | None = None,
    state: RiskState | None = None,
    metrics: RiskMetrics | None = None,
    event_bus: FakeEventBus | None = None,
    scheduler: FakeScheduler | None = None,
    auto_subscribe: bool = False,
    register_scheduler_jobs: bool = False,
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
        service_name="risk_manager_scenarios_test",
    )

    return manager, resolved_event_bus, resolved_scheduler, resolved_state, resolved_metrics


# =============================================================================
# Assertions
# =============================================================================


def assert_allowed(
    decision: RiskDecision,
    *,
    require_notional: bool = True,
) -> None:
    assert decision.allowed is True
    assert decision.decision in {
        RiskDecisionType.ALLOW,
        RiskDecisionType.REDUCE_SIZE,
        RiskDecisionType.REDUCE_RISK,
        RiskDecisionType.DOWNGRADE_TIER,
    }
    assert decision.final_size is not None
    assert decision.final_size > 0
    assert decision.final_margin is not None
    assert decision.final_margin >= 0

    if require_notional:
        assert decision.final_notional is not None
        assert decision.final_notional >= 0

    assert_json_serializable(decision)


def assert_denied(decision: RiskDecision) -> None:
    assert decision.allowed is False
    assert decision.final_size is None
    assert decision.reason is not None
    assert_json_serializable(decision)


def assert_no_confirmed(event_bus: FakeEventBus) -> None:
    assert "signal.confirmed" not in event_bus.topics()


def assert_any_event(event_bus: FakeEventBus, topics: set[str]) -> str:
    emitted = set(event_bus.topics())
    matched = emitted.intersection(topics)
    assert matched, f"Expected one of {topics}, emitted={event_bus.topics()!r}"
    return sorted(matched)[0]


def assert_no_negative_reservation_metrics(metrics: RiskMetrics) -> None:
    assert metrics.reservations.active >= 0
    assert metrics.reservations.reserved_open_risk >= 0
    assert metrics.reservations.reserved_margin >= 0
    assert metrics.reservations.reserved_notional >= 0


async def run_signal_event(manager: RiskManager, event_bus: FakeEventBus, request: RiskEvaluationRequest) -> None:
    payload = signal_payload_from_request(request)
    await event_bus.dispatch("signal.generated", payload)


# =============================================================================
# Capital safety scenarios
# =============================================================================


class TestCapitalSafetyScenarios:
    @pytest.mark.asyncio
    async def test_happy_path_trade_is_allowed_reserved_and_json_safe(self) -> None:
        manager, event_bus, _, state, metrics = make_manager(
            config=make_risk_config(reservations_enabled=True),
        )

        decision = await manager.evaluate_request(
            make_request(signal_id="scenario-happy-path")
        )

        assert_allowed(decision)
        assert decision.reservation_id is not None
        assert state.get_pending_reservation(decision.reservation_id) is not None
        assert metrics.approvals == 1
        assert metrics.reservations.active == 1

        assert_any_event(event_bus, {"signal.confirmed"})
        for event in event_bus.emitted:
            assert_json_serializable(event.payload)

    @pytest.mark.asyncio
    async def test_trade_without_stop_loss_is_blocked_and_never_reserved(self) -> None:
        manager, event_bus, _, state, metrics = make_manager()

        decision = await manager.evaluate_request(
            make_request(
                signal_id="scenario-no-stop",
                stop_loss=None,
            )
        )

        assert_denied(decision)
        assert len(state.pending_reservations) == 0
        assert metrics.rejections == 1
        assert metrics.reservations.created == 0
        assert_any_event(event_bus, {"risk.position_blocked", "risk.rejected"})
        assert_no_confirmed(event_bus)

    @pytest.mark.asyncio
    async def test_bad_risk_reward_is_blocked(self) -> None:
        manager, event_bus, _, state, _ = make_manager()

        decision = await manager.evaluate_request(
            make_request(
                signal_id="scenario-bad-rr",
                entry_price=100.0,
                stop_loss=99.0,
                take_profit=100.20,
                expected_reward=0.20,
                expected_loss=1.0,
                expected_win_probability=0.95,
            )
        )

        assert_denied(decision)
        assert len(state.pending_reservations) == 0
        assert "risk_reward" in decision.checks
        assert decision.checks["risk_reward"].passed is False
        assert_no_confirmed(event_bus)

    @pytest.mark.asyncio
    async def test_negative_ev_after_cost_is_blocked_even_with_large_take_profit(self) -> None:
        manager, event_bus, _, state, _ = make_manager()

        decision = await manager.evaluate_request(
            make_request(
                signal_id="scenario-negative-ev",
                entry_price=100.0,
                stop_loss=99.0,
                take_profit=120.0,
                expected_reward=20.0,
                expected_loss=1.0,
                expected_win_probability=0.001,
                expected_cost=1.0,
            )
        )

        assert_denied(decision)
        assert len(state.pending_reservations) == 0
        assert_no_confirmed(event_bus)

    @pytest.mark.asyncio
    async def test_execution_cost_spike_is_blocked_under_strict_config(self) -> None:
        manager, event_bus, _, state, _ = make_manager(
            config=make_risk_config(strict_execution_cost=True),
        )

        decision = await manager.evaluate_request(
            make_request(
                signal_id="scenario-cost-spike",
                execution_cost=make_execution_cost(
                    spread_cost=0.05,
                    slippage_cost=0.05,
                    fee_cost=0.05,
                    spread_pct=0.05,
                    slippage_pct=0.05,
                    quality=ExecutionQuality.POOR,
                ),
                # Важливо: не ставимо expected_cost=6.0, бо тоді risk_reward
                # блокує pipeline раніше за execution_cost.
                expected_cost=0.03,
            )
        )

        assert_denied(decision)
        assert len(state.pending_reservations) == 0
        assert "execution_cost" in decision.checks
        assert decision.checks["execution_cost"].passed is False
        assert_no_confirmed(event_bus)

    @pytest.mark.asyncio
    async def test_exposure_breach_is_blocked_after_sizing_and_no_reservation_created(self) -> None:
        manager, event_bus, _, state, _ = make_manager(
            config=make_risk_config(
                reservations_enabled=True,
                tight_exposure=True,
                max_open_risk_r=0.01,
            )
        )

        decision = await manager.evaluate_request(
            make_request(signal_id="scenario-exposure-breach")
        )

        assert_denied(decision)
        assert "exposure" in decision.checks
        assert decision.checks["exposure"].passed is False
        assert len(state.pending_reservations) == 0
        assert_no_confirmed(event_bus)

    @pytest.mark.asyncio
    async def test_reduce_or_close_order_should_not_be_blocked_by_new_risk_exposure_limits(self) -> None:
        manager, event_bus, _, state, _ = make_manager(
            config=make_risk_config(
                reservations_enabled=False,
                tight_exposure=True,
                max_open_risk_r=0.01,
            )
        )

        decision = await manager.evaluate_request(
            make_request(
                signal_id="scenario-close-under-tight-exposure",
                order_intent=OrderIntent.CLOSE,
                reduce_only=True,
            )
        )

        # If this fails, exposure/tier/budget pipeline is blocking exits.
        assert_allowed(decision)
        assert_any_event(event_bus, {"signal.confirmed"})
        assert len(state.pending_reservations) == 0


# =============================================================================
# Reservation safety scenarios
# =============================================================================


class TestReservationSafetyScenarios:
    @pytest.mark.asyncio
    async def test_pending_reservation_prevents_second_signal_from_overexposing_account(self) -> None:
        config = make_risk_config(
            reservations_enabled=True,
            max_open_risk_r=0.75,
            max_pending_reservations=10,
        )
        manager, event_bus, _, state, metrics = make_manager(config=config)

        first = await manager.evaluate_request(
            make_request(
                signal_id="scenario-reservation-first",
                symbol=TEST_SYMBOL,
            )
        )

        assert_allowed(first)
        assert first.reservation_id is not None
        assert state.get_pending_open_risk() > 0.0

        second = await manager.evaluate_request(
            make_request(
                signal_id="scenario-reservation-second",
                symbol=ALT_SYMBOL,
            )
        )

        assert_denied(second)
        assert "exposure" in second.checks
        assert second.checks["exposure"].passed is False
        assert len(state.pending_reservations) == 1
        assert metrics.reservations.active == 1
        assert_any_event(event_bus, {"risk.position_blocked", "risk.rejected"})

    @pytest.mark.asyncio
    async def test_order_rejected_releases_reservation_and_frees_budget(self) -> None:
        manager, event_bus, _, state, metrics = make_manager(
            auto_subscribe=True,
            config=make_risk_config(reservations_enabled=True),
        )
        await manager.start()

        decision = await manager.evaluate_request(
            make_request(signal_id="scenario-reject-release")
        )
        assert_allowed(decision)
        assert decision.reservation_id is not None

        await event_bus.dispatch(
            "execution.order_rejected",
            execution_payload(
                signal_id="scenario-reject-release",
                reservation_id=decision.reservation_id,
            ),
        )

        assert state.get_pending_reservation(decision.reservation_id) is None
        assert metrics.reservations.active == 0
        assert_no_negative_reservation_metrics(metrics)
        assert_any_event(
            event_bus,
            {
                "risk.reservation_released",
                "risk.reservation.released",
                "risk.order_rejected",
            },
        )

    @pytest.mark.asyncio
    async def test_expired_reservation_cleanup_frees_reserved_risk(self) -> None:
        manager, event_bus, scheduler, state, metrics = make_manager(
            config=make_risk_config(
                reservations_enabled=True,
                auto_expire_on_evaluate=True,
            ),
            register_scheduler_jobs=True,
        )

        await manager.start()

        decision = await manager.evaluate_request(
            make_request(signal_id="scenario-expiry-cleanup")
        )
        assert_allowed(decision)
        assert decision.reservation_id is not None

        for reservation in state.pending_reservations.values():
            reservation.expires_at = time.time() - 1.0

        expired = await manager.cleanup_expired_reservations()

        assert expired
        assert state.get_pending_reservation(decision.reservation_id) is None
        assert metrics.reservations.active == 0
        assert_no_negative_reservation_metrics(metrics)
        assert_any_event(
            event_bus,
            {
                "risk.reservation_expired",
                "risk.reservation.expired",
            },
        )

        assert scheduler.job_names()  # scheduler integration still registered jobs

    @pytest.mark.asyncio
    async def test_duplicate_release_events_do_not_make_counters_negative(self) -> None:
        manager, event_bus, _, state, metrics = make_manager(
            auto_subscribe=True,
            config=make_risk_config(reservations_enabled=True),
        )
        await manager.start()

        decision = await manager.evaluate_request(
            make_request(signal_id="scenario-duplicate-release")
        )
        assert_allowed(decision)
        assert decision.reservation_id is not None

        payload = execution_payload(
            signal_id="scenario-duplicate-release",
            reservation_id=decision.reservation_id,
        )

        await event_bus.dispatch("execution.order_rejected", payload)
        await event_bus.dispatch("execution.order_rejected", payload)
        await event_bus.dispatch("execution.order_failed", payload)
        await event_bus.dispatch("execution.order_cancelled", payload)

        assert state.get_pending_reservation(decision.reservation_id) is None
        assert_no_negative_reservation_metrics(metrics)


# =============================================================================
# Risk mode scenarios
# =============================================================================


class TestRiskModeScenarios:
    @pytest.mark.asyncio
    async def test_daily_loss_moves_to_caution_but_still_allows_reduced_risk(self) -> None:
        state = make_state()
        force_daily_loss(state, 30.0)

        manager, event_bus, _, _, metrics = make_manager(state=state)

        decision = await manager.evaluate_request(
            make_request(signal_id="scenario-caution")
        )

        assert_allowed(decision)
        assert decision.risk_mode is RiskMode.CAUTION
        assert state.risk_mode is RiskMode.CAUTION
        assert metrics.approvals == 1
        assert_any_event(event_bus, {"signal.confirmed"})

    @pytest.mark.asyncio
    async def test_safe_mode_downgrades_high_tier_before_confirming_signal(self) -> None:
        state = make_state()
        force_daily_loss(state, 60.0)

        manager, event_bus, _, _, _ = make_manager(state=state)

        decision = await manager.evaluate_request(
            make_request(
                signal_id="scenario-safe-mode-downgrade",
                tier=TradeTier.T4,
            )
        )

        assert_allowed(decision)
        assert decision.risk_mode is RiskMode.SAFE_MODE
        assert decision.final_tier is not TradeTier.T4
        assert_any_event(
            event_bus,
            {
                "risk.limit_warning",
                "risk.size_adjusted",
                "signal.confirmed",
            },
        )

    @pytest.mark.asyncio
    async def test_daily_hard_loss_halts_new_risk(self) -> None:
        state = make_state()
        force_daily_loss(state, 100.0)

        manager, event_bus, _, _, metrics = make_manager(state=state)

        decision = await manager.evaluate_request(
            make_request(signal_id="scenario-daily-hard-halt")
        )

        assert_denied(decision)
        assert decision.decision is RiskDecisionType.HALT_TRADING
        assert decision.risk_mode is RiskMode.HALTED
        assert metrics.halts == 1
        assert_any_event(event_bus, {"risk.kill_switch", "risk.trading_halted"})
        assert_no_confirmed(event_bus)

    @pytest.mark.asyncio
    async def test_monthly_emergency_loss_triggers_emergency_stop(self) -> None:
        state = make_state()
        force_monthly_loss(state, 500.0)

        manager, event_bus, _, _, metrics = make_manager(state=state)

        decision = await manager.evaluate_request(
            make_request(signal_id="scenario-monthly-emergency")
        )

        assert_denied(decision)
        assert decision.decision is RiskDecisionType.EMERGENCY_STOP
        assert decision.risk_mode is RiskMode.EMERGENCY_STOP
        assert state.emergency_stop_active is True
        assert metrics.emergency_stops == 1
        assert_any_event(event_bus, {"risk.kill_switch", "risk.emergency_stop"})
        assert_no_confirmed(event_bus)

    @pytest.mark.asyncio
    async def test_manual_halt_blocks_new_signals_via_circuit_breaker(self) -> None:
        manager, event_bus, _, state, _ = make_manager(
            auto_subscribe=True,
        )
        await manager.start()

        await event_bus.dispatch(
            "risk.manual_halt",
            {"reason": "scenario manual halt"},
        )

        assert state.is_circuit_breaker_active() is True

        event_bus.clear()

        decision = await manager.evaluate_request(
            make_request(signal_id="scenario-after-manual-halt")
        )

        assert_denied(decision)
        assert decision.decision in {
            RiskDecisionType.HALT_TRADING,
            RiskDecisionType.EMERGENCY_STOP,
            RiskDecisionType.DENY,
        }
        assert_no_confirmed(event_bus)

    @pytest.mark.asyncio
    async def test_emergency_stop_cannot_be_bypassed_by_manual_resume_event(self) -> None:
        manager, event_bus, _, state, _ = make_manager(
            auto_subscribe=True,
        )
        await manager.start()

        state.emergency_stop("scenario emergency stop")

        await event_bus.dispatch(
            "risk.manual_resume",
            {"reason": "attempted unsafe resume"},
        )

        assert state.emergency_stop_active is True
        assert state.risk_mode is RiskMode.EMERGENCY_STOP

        event_bus.clear()

        decision = await manager.evaluate_request(
            make_request(signal_id="scenario-after-emergency-resume")
        )

        assert_denied(decision)
        assert decision.decision is RiskDecisionType.EMERGENCY_STOP
        assert_no_confirmed(event_bus)


# =============================================================================
# Circuit breaker scenarios
# =============================================================================


class TestCircuitBreakerScenarios:
    @pytest.mark.asyncio
    async def test_repeated_execution_failures_trigger_breaker_and_block_next_signal(self) -> None:
        manager, event_bus, _, state, _ = make_manager(
            auto_subscribe=True,
            config=make_risk_config(circuit_breaker_failures=2),
        )
        await manager.start()

        await event_bus.dispatch(
            "execution.order_failed",
            execution_payload(signal_id="failure-1"),
        )
        await event_bus.dispatch(
            "execution.order_failed",
            execution_payload(signal_id="failure-2"),
        )

        assert state.is_circuit_breaker_active() is True

        event_bus.clear()

        decision = await manager.evaluate_request(
            make_request(signal_id="scenario-after-breaker")
        )

        assert_denied(decision)
        assert decision.decision in {
            RiskDecisionType.HALT_TRADING,
            RiskDecisionType.EMERGENCY_STOP,
            RiskDecisionType.DENY,
        }
        assert_no_confirmed(event_bus)

    @pytest.mark.asyncio
    async def test_direct_circuit_breaker_activation_blocks_before_position_sizing(self) -> None:
        state = make_state()
        state.activate_circuit_breaker(
            CircuitBreakerReason.MANUAL_HALT,
            message="scenario direct breaker",
            manual_release_required=True,
        )

        manager, event_bus, _, _, _ = make_manager(state=state)

        decision = await manager.evaluate_request(
            make_request(signal_id="scenario-direct-breaker")
        )

        assert_denied(decision)
        assert "circuit_breaker" in decision.checks
        assert decision.checks["circuit_breaker"].passed is False
        assert "position_sizing" not in decision.checks
        assert_no_confirmed(event_bus)


# =============================================================================
# Stress / concurrency
# =============================================================================


class TestConcurrencyStress:
    @pytest.mark.asyncio
    async def test_concurrent_valid_signals_cannot_exceed_open_risk_budget(self) -> None:
        """
        50 concurrent signals compete for a small open-risk budget.

        The critical invariant is not exact allowed count; it is:
        - pending open risk never exceeds configured limit;
        - reservation IDs are unique;
        - metrics/counters stay non-negative;
        - not every concurrent request is allowed.
        """
        config = make_risk_config(
            reservations_enabled=True,
            max_open_risk_r=2.0,
            max_pending_reservations=100,
        )
        state = make_state()
        manager, _, _, _, metrics = make_manager(config=config, state=state)

        requests = [
            make_request(
                signal_id=f"concurrent-signal-{index}",
                symbol=f"SYM{index}USDT",
            )
            for index in range(50)
        ]

        decisions = list(
            await asyncio.gather(
                *(manager.evaluate_request(request) for request in requests)
            )
        )

        allowed = [decision for decision in decisions if decision.allowed]
        denied = [decision for decision in decisions if not decision.allowed]

        assert allowed
        assert denied

        reservation_ids = [
            decision.reservation_id
            for decision in allowed
            if decision.reservation_id is not None
        ]
        assert len(reservation_ids) == len(set(reservation_ids))

        total_pending_open_risk = state.get_pending_open_risk()
        base_risk_unit = state.equity * config.risk_unit.base_risk_unit_pct
        max_open_risk_amount = config.exposure.max_open_risk_r * base_risk_unit

        assert total_pending_open_risk <= max_open_risk_amount + 1e-9
        assert metrics.total_decisions == 50
        assert metrics.approvals == len(allowed)
        assert metrics.rejections == len(denied)
        assert_no_negative_reservation_metrics(metrics)

        for decision in decisions:
            assert_json_serializable(decision)

    @pytest.mark.asyncio
    async def test_concurrent_cleanup_and_evaluation_do_not_corrupt_reservation_state(self) -> None:
        config = make_risk_config(
            reservations_enabled=True,
            max_open_risk_r=10.0,
            max_pending_reservations=100,
        )
        manager, _, _, state, metrics = make_manager(config=config)

        initial_decisions = await asyncio.gather(
            *(
                manager.evaluate_request(
                    make_request(
                        signal_id=f"cleanup-race-initial-{index}",
                        symbol=f"RACE{index}USDT",
                    )
                )
                for index in range(10)
            )
        )

        assert any(decision.allowed for decision in initial_decisions)

        for reservation in state.pending_reservations.values():
            reservation.expires_at = time.time() - 1.0

        cleanup_task = asyncio.create_task(manager.cleanup_expired_reservations())

        evaluation_tasks = [
            asyncio.create_task(
                manager.evaluate_request(
                    make_request(
                        signal_id=f"cleanup-race-new-{index}",
                        symbol=f"NEWRACE{index}USDT",
                    )
                )
            )
            for index in range(10)
        ]

        expired, *new_decisions = await asyncio.gather(cleanup_task, *evaluation_tasks)

        assert expired is not None
        assert all(isinstance(decision, RiskDecision) for decision in new_decisions)
        assert_no_negative_reservation_metrics(metrics)

        for reservation in state.pending_reservations.values():
            assert math.isfinite(float(reservation.open_risk))
            assert reservation.open_risk >= 0.0
            assert math.isfinite(float(reservation.margin))
            assert reservation.margin >= 0.0

    @pytest.mark.asyncio
    async def test_high_volume_order_failures_do_not_make_metrics_or_state_negative(self) -> None:
        manager, event_bus, _, state, metrics = make_manager(
            auto_subscribe=True,
            config=make_risk_config(circuit_breaker_failures=10_000),
        )
        await manager.start()

        for index in range(1_000):
            await event_bus.dispatch(
                "execution.order_failed",
                execution_payload(
                    signal_id=f"stress-failure-{index}",
                    order_id=f"stress-order-{index}",
                ),
            )

        assert_no_negative_reservation_metrics(metrics)

        assert len(state.pending_reservations) == 0
        assert state.get_pending_open_risk() >= 0.0
        assert state.get_pending_margin() >= 0.0
        assert state.get_pending_notional() >= 0.0


# =============================================================================
# Regression safety bugs
# =============================================================================


class TestRegressionSafetyBugs:
    @pytest.mark.asyncio
    async def test_min_notional_never_auto_upsizes_position_above_allowed_risk(self) -> None:
        config = make_risk_config()
        set_if_exists(config.position_sizing, "min_position_size", 0.0)

        manager, event_bus, _, state, _ = make_manager(config=config)

        decision = await manager.evaluate_request(
            make_request(
                signal_id="regression-min-notional",
                entry_price=100.0,
                stop_loss=99.0,
                requested_size=None,
                metadata={
                    # Depending on implementation, metadata may be ignored by
                    # PositionSizer. This test still guards final risk invariant.
                    "min_notional": 10_000_000.0,
                },
            )
        )

        if decision.allowed:
            assert decision.final_size is not None
            assert decision.final_risk_amount is not None
            assert decision.final_risk_amount <= 10.0 + 1e-9
            assert decision.final_notional is not None
            assert math.isfinite(float(decision.final_notional))
        else:
            assert len(state.pending_reservations) == 0
            assert_no_confirmed(event_bus)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("side", "stop_loss"),
        [
            (PositionSide.LONG, 100.0),
            (PositionSide.LONG, 101.0),
            (PositionSide.SHORT, 100.0),
            (PositionSide.SHORT, 99.0),
        ],
    )
    async def test_wrong_side_stop_loss_is_never_allowed(
        self,
        side: PositionSide,
        stop_loss: float,
    ) -> None:
        manager, event_bus, _, state, _ = make_manager()

        decision = await manager.evaluate_request(
            make_request(
                signal_id=f"regression-wrong-stop-{side.value}-{stop_loss}",
                side=side,
                stop_loss=stop_loss,
                take_profit=103.0 if side is PositionSide.LONG else 97.0,
            )
        )

        assert_denied(decision)
        assert len(state.pending_reservations) == 0
        assert_no_confirmed(event_bus)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("field_name", "bad_value"),
        [
            ("entry_price", math.nan),
            ("entry_price", math.inf),
            ("stop_loss", math.nan),
            ("take_profit", math.inf),
            ("expected_win_probability", math.nan),
            ("expected_cost", math.inf),
            ("requested_leverage", math.nan),
            ("requested_leverage", math.inf),
        ],
    )
    async def test_nan_or_inf_request_values_never_result_in_confirmed_signal(
        self,
        field_name: str,
        bad_value: float,
    ) -> None:
        manager, event_bus, _, state, _ = make_manager()

        try:
            decision = await manager.evaluate_request(
                make_request(
                    signal_id=f"regression-non-finite-{field_name}",
                    **{field_name: bad_value},
                )
            )
        except (ValueError, ArithmeticError):
            assert len(state.pending_reservations) == 0
            assert_no_confirmed(event_bus)
            return

        assert_denied(decision)
        assert len(state.pending_reservations) == 0
        assert_no_confirmed(event_bus)

    @pytest.mark.asyncio
    async def test_events_are_emitted_after_lock_release_not_under_manager_lock(self) -> None:
        manager, event_bus, _, _, _ = make_manager()

        original_emit = manager._emit_event
        observed_topics: list[str] = []

        async def strict_emit(
            topic: str,
            payload: dict[str, Any] | None = None,
            *,
            priority: Any | None = None,
            **kwargs: Any,
        ) -> None:
            assert not manager._lock.locked(), (
                f"RiskManager emitted {topic!r} while _lock was held"
            )
            observed_topics.append(topic)
            await original_emit(topic, payload, priority=priority, **kwargs)

        manager._emit_event = strict_emit  # type: ignore[method-assign]

        decision = await manager.evaluate_request(
            make_request(signal_id="regression-lock-release")
        )

        assert_allowed(decision)
        assert "risk.request_received" in observed_topics
        assert "signal.confirmed" in observed_topics
        assert event_bus.events_for("signal.confirmed")

    @pytest.mark.asyncio
    async def test_reentrant_emit_can_call_evaluate_request_without_deadlock(self) -> None:
        manager, event_bus, _, _, _ = make_manager()

        original_emit = manager._emit_event
        reentered = False

        async def reentrant_emit(
            topic: str,
            payload: dict[str, Any] | None = None,
            *,
            priority: Any | None = None,
            **kwargs: Any,
        ) -> None:
            nonlocal reentered

            assert not manager._lock.locked(), (
                f"RiskManager emitted {topic!r} while _lock was held"
            )

            await original_emit(topic, payload, priority=priority, **kwargs)

            if topic == "signal.confirmed" and not reentered:
                reentered = True
                nested = await manager.evaluate_request(
                    make_request(
                        signal_id="regression-reentrant-deny",
                        stop_loss=None,
                    )
                )
                assert nested.allowed is False

        manager._emit_event = reentrant_emit  # type: ignore[method-assign]

        decision = await manager.evaluate_request(
            make_request(signal_id="regression-reentrant-allow")
        )

        assert_allowed(decision)
        assert reentered is True
        assert "signal.confirmed" in event_bus.topics()
        assert_any_event(event_bus, {"risk.position_blocked", "risk.rejected"})

    @pytest.mark.asyncio
    async def test_malformed_position_event_with_nan_does_not_poison_state(self) -> None:
        """
        This is intentionally strict. Previous integration run showed that
        position.opened with size=NaN could create PortfolioPosition(size=nan).
        For live capital, this must be rejected or ignored.
        """
        manager, event_bus, _, state, _ = make_manager(
            auto_subscribe=True,
        )
        await manager.start()

        try:
            await event_bus.dispatch(
                "position.opened",
                position_payload(
                    signal_id="regression-nan-position",
                    position_id="nan-position",
                    size=math.nan,
                    notional_value=math.nan,
                ),
            )
        except (ValueError, TypeError):
            pass

        for position in state.positions.values():
            assert math.isfinite(float(position.size))
            assert position.size >= 0.0
            assert math.isfinite(float(position.entry_price))
            assert position.entry_price > 0.0
            assert math.isfinite(float(position.notional_value))
            assert position.notional_value >= 0.0

    @pytest.mark.asyncio
    async def test_scheduler_jobs_are_not_duplicated_on_double_start(self) -> None:
        manager, _, scheduler, _, _ = make_manager(
            auto_subscribe=True,
            register_scheduler_jobs=True,
        )

        await manager.start()
        first_jobs = scheduler.job_names()

        await manager.start()
        second_jobs = scheduler.job_names()

        assert second_jobs == first_jobs

    @pytest.mark.asyncio
    async def test_no_duplicate_reservation_ids_under_concurrent_same_symbol_signals(self) -> None:
        manager, _, _, state, metrics = make_manager(
            config=make_risk_config(
                reservations_enabled=True,
                max_open_risk_r=20.0,
                max_pending_reservations=100,
                max_pending_per_symbol=100,
            )
        )

        requests = [
            make_request(
                signal_id=f"same-symbol-concurrent-{index}",
                symbol=TEST_SYMBOL,
            )
            for index in range(25)
        ]

        decisions = list(
            await asyncio.gather(
                *(manager.evaluate_request(request) for request in requests)
            )
        )

        allowed = [decision for decision in decisions if decision.allowed]
        reservation_ids = [
            decision.reservation_id
            for decision in allowed
            if decision.reservation_id is not None
        ]

        assert reservation_ids
        assert len(reservation_ids) == len(set(reservation_ids))
        assert len(state.pending_reservations) == len(reservation_ids)
        assert metrics.reservations.active == len(reservation_ids)
        assert_no_negative_reservation_metrics(metrics)

    @pytest.mark.asyncio
    async def test_fail_closed_when_reservation_capacity_is_reached(self) -> None:
        manager, event_bus, _, state, metrics = make_manager(
            config=make_risk_config(
                reservations_enabled=True,
                max_open_risk_r=100.0,
                max_pending_reservations=1,
            )
        )

        first = await manager.evaluate_request(
            make_request(signal_id="capacity-first")
        )
        second = await manager.evaluate_request(
            make_request(signal_id="capacity-second")
        )

        assert_allowed(first)
        assert first.reservation_id is not None
        assert_denied(second)

        assert len(state.pending_reservations) == 1
        assert metrics.reservations.active == 1
        assert_no_negative_reservation_metrics(metrics)
        assert_any_event(event_bus, {"risk.position_blocked", "risk.rejected"})