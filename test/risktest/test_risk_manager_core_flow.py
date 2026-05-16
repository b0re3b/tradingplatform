# tests/risk/test_risk_manager_core_flow.py
from __future__ import annotations

import dataclasses
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
    RiskLevel,
    RiskMode,
    RiskViolationType,
    TradeTier,
)
from risk.metrics import RiskMetrics
from risk.models import (
    ExecutionCostEstimate,
    RiskCheckResult,
    RiskDecision,
    RiskEvaluationRequest,
    RiskViolation,
)
from risk.risk_manager import RiskManager
from risk.state import RiskState


TEST_SYMBOL = "BTCUSDT"
ALT_SYMBOL = "ETHUSDT"
TEST_STRATEGY = "test_strategy"
TEST_SIGNAL_ID = "signal-core-flow-001"


# =============================================================================
# Test doubles
# =============================================================================


@dataclass(slots=True)
class EmittedEvent:
    topic: str
    payload: dict[str, Any]
    priority: Any | None = None
    source: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class FakeSubscription:
    topic: str
    handler: Callable[..., Any]
    name: str | None = None
    active: bool = True


class FakeEventBus:
    def __init__(self) -> None:
        self.emitted: list[EmittedEvent] = []
        self.subscriptions: list[FakeSubscription] = []
        self.unsubscriptions: list[FakeSubscription] = []
        self.emit_calls: list[dict[str, Any]] = []

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
        self.emit_calls.clear()


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
    """
    Force daily loss through equity anchors, because RiskState does not expose
    daily_pnl as a direct field in this implementation.
    """
    assert loss_amount >= 0

    if hasattr(state, "daily_start_equity"):
        state.daily_start_equity = state.equity + loss_amount
    elif hasattr(state, "metadata"):
        state.metadata["test_daily_loss"] = loss_amount
    else:
        raise AssertionError("RiskState has no daily_start_equity-compatible field")


def force_weekly_loss(state: RiskState, loss_amount: float) -> None:
    assert loss_amount >= 0

    if hasattr(state, "weekly_start_equity"):
        state.weekly_start_equity = state.equity + loss_amount
    elif hasattr(state, "metadata"):
        state.metadata["test_weekly_loss"] = loss_amount
    else:
        raise AssertionError("RiskState has no weekly_start_equity-compatible field")


def force_monthly_loss(state: RiskState, loss_amount: float) -> None:
    assert loss_amount >= 0

    if hasattr(state, "monthly_start_equity"):
        state.monthly_start_equity = state.equity + loss_amount
    elif hasattr(state, "metadata"):
        state.metadata["test_monthly_loss"] = loss_amount
    else:
        raise AssertionError("RiskState has no monthly_start_equity-compatible field")

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


def make_risk_config(
    *,
    reservations_enabled: bool = True,
    strict_execution_cost: bool = False,
    tight_exposure: bool = False,
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

    # Exposure limits must satisfy RiskConfig.validate():
    # safe_mode_max_open_risk_r <= max_open_risk_r <= aggressive_max_open_risk_r
    if tight_exposure:
        max_open_risk_r = 0.5
        safe_open_risk_r = 0.25
        aggressive_open_risk_r = 0.5

        max_used_margin_pct = 0.01
        max_total_exposure_pct = 0.01
        max_symbol_exposure_pct = 0.01
        max_side_exposure_pct = 0.01
        max_open_positions = 0
    else:
        max_open_risk_r = 100.0
        safe_open_risk_r = 50.0
        aggressive_open_risk_r = 150.0

        max_used_margin_pct = 1.0
        max_total_exposure_pct = 10.0
        max_symbol_exposure_pct = 10.0
        max_side_exposure_pct = 10.0
        max_open_positions = 100

    set_if_exists(config.exposure, "max_open_risk_r", max_open_risk_r)
    set_if_exists(config.exposure, "safe_mode_max_open_risk_r", safe_open_risk_r)
    set_if_exists(config.exposure, "aggressive_max_open_risk_r", aggressive_open_risk_r)
    set_if_exists(config.exposure, "max_used_margin_pct", max_used_margin_pct)
    set_if_exists(config.exposure, "max_total_exposure_pct", max_total_exposure_pct)
    set_if_exists(config.exposure, "max_symbol_exposure_pct", max_symbol_exposure_pct)
    set_if_exists(config.exposure, "max_side_exposure_pct", max_side_exposure_pct)
    set_if_exists(config.exposure, "max_open_positions", max_open_positions)

    # Execution-cost happy path is permissive unless a test asks for strict.
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

    # Reservation policy.
    set_if_exists(config.reservation, "enabled", reservations_enabled)
    set_if_exists(config.reservation, "reserve_on_allow", reservations_enabled)
    set_if_exists(config.reservation, "ttl_seconds", 30.0)
    set_if_exists(config.reservation, "cleanup_interval_seconds", 5.0)
    set_if_exists(config.reservation, "max_pending_reservations", 100)
    set_if_exists(config.reservation, "max_pending_per_symbol", 10)
    set_if_exists(config.reservation, "max_pending_per_strategy", 20)
    set_if_exists(config.reservation, "fail_closed_on_reservation_error", True)
    set_if_exists(config.reservation, "auto_expire_on_evaluate", True)

    # Circuit breaker is enabled but not aggressive in happy path.
    set_if_exists(config.circuit_breaker, "enabled", True)
    set_if_exists(config.circuit_breaker, "max_consecutive_failures", 5)
    set_if_exists(config.circuit_breaker, "max_execution_failures", 5)
    set_if_exists(config.circuit_breaker, "cooldown_seconds", 300.0)

    config.validate()
    return config


def make_manager(
    *,
    config: RiskConfig | None = None,
    state: RiskState | None = None,
    event_bus: FakeEventBus | None = None,
    metrics: RiskMetrics | None = None,
    auto_subscribe: bool = False,
    register_scheduler_jobs: bool = False,
) -> tuple[RiskManager, FakeEventBus, RiskState, RiskMetrics]:
    resolved_config = config or make_risk_config()
    resolved_state = state or make_state()
    resolved_event_bus = event_bus or FakeEventBus()
    resolved_metrics = metrics or RiskMetrics()

    manager = RiskManager(
        resolved_config,
        event_bus=resolved_event_bus,  # type: ignore[arg-type]
        scheduler=None,
        state=resolved_state,
        metrics=resolved_metrics,
        auto_subscribe=auto_subscribe,
        register_scheduler_jobs=register_scheduler_jobs,
        service_name="risk_manager_core_flow_test",
    )

    return manager, resolved_event_bus, resolved_state, resolved_metrics


def make_violation(
    violation_type: RiskViolationType = RiskViolationType.INVALID_REQUEST,
) -> RiskViolation:
    return RiskViolation(
        violation_type=violation_type,
        level=RiskLevel.CRITICAL,
        message=f"test violation: {violation_type.value}",
        symbol=TEST_SYMBOL,
        strategy_name=TEST_STRATEGY,
        tier=TradeTier.T2,
        metadata={"violation": True},
    )


def make_decision(
    *,
    allowed: bool = True,
    decision: RiskDecisionType = RiskDecisionType.ALLOW,
    checks: dict[str, RiskCheckResult] | None = None,
    violations: list[RiskViolation] | None = None,
) -> RiskDecision:
    return RiskDecision(
        allowed=allowed,
        decision=decision,
        final_size=1.0 if allowed else None,
        final_leverage=5.0 if allowed else None,
        final_tier=TradeTier.T2,
        final_risk_amount=10.0 if allowed else None,
        final_margin=20.0 if allowed else None,
        final_notional=100.0 if allowed else None,
        reservation_id="reservation-test" if allowed else None,
        reservation_expires_at=time.time() + 30.0 if allowed else None,
        risk_mode=RiskMode.NORMAL,
        risk_reward_ratio=3.0,
        expected_value=1.0,
        expected_value_after_cost=0.9,
        expected_cost=0.1,
        cost_to_reward_ratio=0.03,
        reason="test decision",
        signal_id=TEST_SIGNAL_ID,
        strategy_name=TEST_STRATEGY,
        symbol=TEST_SYMBOL,
        side=PositionSide.LONG,
        order_intent=OrderIntent.OPEN,
        violations=list(violations or []),
        checks=dict(checks or {}),
        metadata={"test": True, "enum": TradeTier.T2},
    )


# =============================================================================
# Assertions
# =============================================================================


def assert_allowed_decision(decision: RiskDecision) -> None:
    assert decision.allowed is True
    assert decision.decision in {
        RiskDecisionType.ALLOW,
        RiskDecisionType.REDUCE_SIZE,
        RiskDecisionType.REDUCE_RISK,
        RiskDecisionType.DOWNGRADE_TIER,
    }
    assert decision.final_size is not None
    assert decision.final_size > 0
    assert decision.final_leverage is not None
    assert decision.final_leverage > 0
    assert decision.final_tier is not None
    assert decision.final_risk_amount is not None
    assert decision.final_risk_amount >= 0
    assert decision.final_margin is not None
    assert decision.final_margin >= 0
    assert decision.final_notional is not None
    assert decision.final_notional >= 0
    assert decision.symbol == TEST_SYMBOL
    assert_json_serializable(decision)


def assert_denied_decision(decision: RiskDecision) -> None:
    assert decision.allowed is False
    assert decision.final_size is None
    assert decision.reason is not None
    assert_json_serializable(decision)


def assert_event(event_bus: FakeEventBus, topic: str) -> EmittedEvent:
    event = event_bus.last_event(topic)
    assert_json_serializable(event.payload)
    return event


def assert_no_event(event_bus: FakeEventBus, topic: str) -> None:
    assert not event_bus.events_for(topic), (
        f"Did not expect topic={topic!r}, emitted={event_bus.topics()!r}"
    )


def assert_payload_has_decision_shape(payload: dict[str, Any]) -> None:
    required = {
        "symbol",
        "side",
        "signal_id",
        "strategy_name",
        "allowed",
        "decision",
        "risk_mode",
        "final_tier",
        "final_size",
        "final_leverage",
        "final_risk_amount",
        "final_margin",
        "final_notional",
        "reservation_id",
        "reason",
        "violations",
        "checks",
        "metadata",
    }
    missing = required.difference(payload)
    assert not missing, f"Missing decision payload keys: {missing}"
    assert_json_serializable(payload)


# =============================================================================
# Happy path
# =============================================================================


class TestRiskManagerEvaluateHappyPath:
    @pytest.mark.asyncio
    async def test_valid_long_request_is_allowed_reserved_and_emitted(self) -> None:
        manager, event_bus, state, metrics = make_manager(
            config=make_risk_config(reservations_enabled=True),
            state=make_state(),
        )
        request = make_request()

        decision = await manager.evaluate_request(request)

        assert_allowed_decision(decision)
        assert decision.decision is RiskDecisionType.ALLOW
        assert decision.reservation_id is not None
        assert decision.reservation_expires_at is not None
        assert len(state.pending_reservations) == 1
        assert state.get_pending_reservation(decision.reservation_id) is not None

        assert_event(event_bus, "risk.request_received")
        confirmed = assert_event(event_bus, "signal.confirmed")
        approved = assert_event(event_bus, "risk.approved")

        assert_payload_has_decision_shape(confirmed.payload)
        assert confirmed.payload["allowed"] is True
        assert confirmed.payload["decision"] == RiskDecisionType.ALLOW.value
        assert confirmed.payload["reservation_id"] == decision.reservation_id
        assert approved.payload["reservation_id"] == decision.reservation_id

        assert metrics.total_decisions == 1
        assert metrics.approvals == 1
        assert metrics.rejections == 0
        assert metrics.last_reservation_id == decision.reservation_id
        assert metrics.reservations.created == 1
        assert metrics.reservations.active == 1

    @pytest.mark.asyncio
    async def test_valid_short_request_is_allowed_and_side_safe(self) -> None:
        manager, event_bus, _, _ = make_manager()
        request = make_request(
            side=PositionSide.SHORT,
            stop_loss=101.0,
            take_profit=97.0,
            signal_id="signal-short-core-flow",
        )

        decision = await manager.evaluate_request(request)

        assert decision.allowed is True
        assert decision.side is PositionSide.SHORT
        assert decision.final_size is not None
        assert decision.final_size > 0
        assert_event(event_bus, "signal.confirmed")

    @pytest.mark.asyncio
    async def test_reservations_can_be_disabled_without_blocking_allow(self) -> None:
        manager, event_bus, state, metrics = make_manager(
            config=make_risk_config(reservations_enabled=False),
        )

        decision = await manager.evaluate_request(make_request())

        assert_allowed_decision(decision)
        assert decision.reservation_id is None
        assert len(state.pending_reservations) == 0
        assert metrics.reservations.created == 0

        confirmed = assert_event(event_bus, "signal.confirmed")
        assert confirmed.payload["reservation_id"] is None

    @pytest.mark.asyncio
    async def test_decision_contains_all_pipeline_checks_on_success(self) -> None:
        manager, _, _, _ = make_manager()

        decision = await manager.evaluate_request(make_request())

        assert_allowed_decision(decision)

        expected_checks = {
            "circuit_breaker",
            "budget",
            "tier",
            "risk_reward",
            "execution_cost",
            "leverage",
            "position_sizing",
            "exposure",
            "symbol",
            "strategy",
        }
        assert expected_checks.issubset(decision.checks.keys())

        for name, check in decision.checks.items():
            assert isinstance(name, str)
            assert isinstance(check, RiskCheckResult)
            assert_json_serializable(check)

    @pytest.mark.asyncio
    async def test_request_received_event_is_emitted_before_final_decision_events(self) -> None:
        manager, event_bus, _, _ = make_manager()

        await manager.evaluate_request(make_request())

        topics = event_bus.topics()
        assert topics[0] == "risk.request_received"
        assert "signal.confirmed" in topics
        assert topics.index("risk.request_received") < topics.index("signal.confirmed")


# =============================================================================
# Denial pipeline
# =============================================================================


class TestRiskManagerEvaluateDenials:
    @pytest.mark.asyncio
    async def test_missing_stop_loss_is_denied_and_does_not_reserve_risk(self) -> None:
        manager, event_bus, state, metrics = make_manager()
        request = make_request(stop_loss=None)

        decision = await manager.evaluate_request(request)

        assert_denied_decision(decision)
        assert decision.decision is RiskDecisionType.DENY
        assert "stop" in decision.reason.lower()
        assert len(state.pending_reservations) == 0

        blocked = assert_event(event_bus, "risk.position_blocked")
        rejected = assert_event(event_bus, "risk.rejected")

        assert blocked.payload["allowed"] is False
        assert rejected.payload["allowed"] is False
        assert blocked.payload["reservation_id"] is None
        assert_payload_has_decision_shape(blocked.payload)

        assert metrics.total_decisions == 1
        assert metrics.rejections == 1
        assert metrics.approvals == 0
        assert metrics.reservations.created == 0

    @pytest.mark.asyncio
    async def test_wrong_side_long_stop_is_denied_before_sizing(self) -> None:
        manager, event_bus, state, _ = make_manager()
        request = make_request(
            side=PositionSide.LONG,
            entry_price=100.0,
            stop_loss=101.0,
            take_profit=103.0,
        )

        decision = await manager.evaluate_request(request)

        assert_denied_decision(decision)
        assert len(state.pending_reservations) == 0
        assert "risk_reward" in decision.checks
        assert decision.checks["risk_reward"].passed is False
        assert_event(event_bus, "risk.position_blocked")
        assert_no_event(event_bus, "signal.confirmed")

    @pytest.mark.asyncio
    async def test_bad_risk_reward_is_denied(self) -> None:
        manager, event_bus, state, _ = make_manager()
        request = make_request(
            entry_price=100.0,
            stop_loss=99.0,
            take_profit=100.25,
            expected_reward=0.25,
            expected_loss=1.0,
            expected_win_probability=0.9,
        )

        decision = await manager.evaluate_request(request)

        assert_denied_decision(decision)
        assert len(state.pending_reservations) == 0
        assert decision.checks["risk_reward"].passed is False
        assert any(
            violation.violation_type is RiskViolationType.RISK_REWARD_TOO_LOW
            for violation in decision.violations
        )
        assert_event(event_bus, "risk.position_blocked")

    @pytest.mark.asyncio
    async def test_negative_ev_after_cost_is_denied_even_if_rr_is_good(self) -> None:
        manager, event_bus, state, _ = make_manager()
        request = make_request(
            entry_price=100.0,
            stop_loss=99.0,
            take_profit=110.0,
            expected_reward=10.0,
            expected_loss=1.0,
            expected_win_probability=0.01,
            expected_cost=0.5,
        )

        decision = await manager.evaluate_request(request)

        assert_denied_decision(decision)
        assert len(state.pending_reservations) == 0
        assert decision.checks["risk_reward"].passed is False
        assert_event(event_bus, "risk.position_blocked")

    @pytest.mark.asyncio
    async def test_execution_cost_spike_is_denied_under_strict_config(self) -> None:
        manager, event_bus, state, _ = make_manager(
            config=make_risk_config(strict_execution_cost=True),
        )
        request = make_request(
            execution_cost=make_execution_cost(
                spread_pct=0.05,
                slippage_pct=0.05,
                spread_cost=2.0,
                slippage_cost=2.0,
                fee_cost=2.0,
                quality=ExecutionQuality.POOR,
            ),
            expected_cost=6.0,
        )

        decision = await manager.evaluate_request(request)

        assert_denied_decision(decision)
        assert len(state.pending_reservations) == 0
        assert "execution_cost" in decision.checks
        assert decision.checks["execution_cost"].passed is False
        assert_event(event_bus, "risk.position_blocked")

    @pytest.mark.asyncio
    async def test_exposure_breach_is_denied_after_sizing_without_creating_reservation(self) -> None:
        manager, event_bus, state, _ = make_manager(
            config=make_risk_config(tight_exposure=True),
        )
        request = make_request()

        decision = await manager.evaluate_request(request)

        assert_denied_decision(decision)
        assert "exposure" in decision.checks
        assert decision.checks["exposure"].passed is False
        assert len(state.pending_reservations) == 0
        assert_event(event_bus, "risk.position_blocked")

    @pytest.mark.asyncio
    async def test_symbol_disabled_blocks_new_risk(self) -> None:
        state = make_state()
        state.get_symbol_state(TEST_SYMBOL).disable(reason="test symbol disabled")
        manager, event_bus, _, _ = make_manager(state=state)

        decision = await manager.evaluate_request(make_request())

        assert_denied_decision(decision)
        assert "symbol" in decision.checks
        assert decision.checks["symbol"].passed is False
        assert_event(event_bus, "risk.position_blocked")

    @pytest.mark.asyncio
    async def test_strategy_disabled_blocks_new_risk(self) -> None:
        state = make_state()
        state.get_strategy_state(TEST_STRATEGY).disable(reason="test strategy disabled")
        manager, event_bus, _, _ = make_manager(state=state)

        decision = await manager.evaluate_request(make_request())

        assert_denied_decision(decision)
        assert "strategy" in decision.checks
        assert decision.checks["strategy"].passed is False
        assert_event(event_bus, "risk.position_blocked")


# =============================================================================
# Risk modes
# =============================================================================


class TestRiskManagerRiskModes:
    @pytest.mark.asyncio
    async def test_caution_mode_allows_but_sets_caution_risk_mode(self) -> None:
        state = make_state()
        force_daily_loss(state, 30.0)  # 3R with default R=10
        manager, event_bus, _, _ = make_manager(state=state)

        decision = await manager.evaluate_request(make_request(tier=TradeTier.T2))

        assert decision.allowed is True
        assert decision.risk_mode is RiskMode.CAUTION
        assert state.risk_mode is RiskMode.CAUTION
        assert_event(event_bus, "signal.confirmed")

    @pytest.mark.asyncio
    async def test_safe_mode_downgrades_high_tier_and_emits_warning_events(self) -> None:
        state = make_state()
        force_daily_loss(state, 60.0)  # soft daily loss => SAFE_MODE by default config
        manager, event_bus, _, metrics = make_manager(state=state)

        decision = await manager.evaluate_request(make_request(tier=TradeTier.T4))

        assert decision.allowed is True
        assert decision.risk_mode is RiskMode.SAFE_MODE
        assert decision.final_tier is TradeTier.T2
        assert decision.decision in {
            RiskDecisionType.DOWNGRADE_TIER,
            RiskDecisionType.REDUCE_RISK,
            RiskDecisionType.REDUCE_SIZE,
        }

        assert_event(event_bus, "signal.confirmed")
        assert_event(event_bus, "risk.limit_warning")
        assert_event(event_bus, "risk.size_adjusted")

        assert metrics.total_decisions == 1
        assert metrics.approvals == 1
        assert metrics.tier_downgrades >= 1 or metrics.risk_reductions >= 1

    @pytest.mark.asyncio
    async def test_hard_daily_loss_halts_and_emits_kill_switch(self) -> None:
        state = make_state()
        force_daily_loss(state, 100.0)  # hard daily loss => HALTED
        manager, event_bus, _, metrics = make_manager(state=state)

        decision = await manager.evaluate_request(make_request())

        assert_denied_decision(decision)
        assert decision.decision is RiskDecisionType.HALT_TRADING
        assert decision.risk_mode is RiskMode.HALTED
        assert state.trading_halted is True

        kill = assert_event(event_bus, "risk.kill_switch")
        halted = assert_event(event_bus, "risk.trading_halted")
        assert kill.payload["decision"] == RiskDecisionType.HALT_TRADING.value
        assert halted.payload["decision"] == RiskDecisionType.HALT_TRADING.value
        assert_no_event(event_bus, "risk.position_blocked")

        assert metrics.halts == 1
        assert metrics.rejections == 1

    @pytest.mark.asyncio
    async def test_monthly_emergency_loss_emits_emergency_stop(self) -> None:
        state = make_state()
        force_monthly_loss(state, 500.0)  # emergency monthly loss with R=10
        manager, event_bus, _, metrics = make_manager(state=state)

        decision = await manager.evaluate_request(make_request())

        assert_denied_decision(decision)
        assert decision.decision is RiskDecisionType.EMERGENCY_STOP
        assert decision.risk_mode is RiskMode.EMERGENCY_STOP
        assert state.emergency_stop_active is True

        kill = assert_event(event_bus, "risk.kill_switch")
        emergency = assert_event(event_bus, "risk.emergency_stop")
        assert kill.payload["decision"] == RiskDecisionType.EMERGENCY_STOP.value
        assert emergency.payload["decision"] == RiskDecisionType.EMERGENCY_STOP.value

        assert metrics.emergency_stops == 1
        assert metrics.rejections == 1

    @pytest.mark.asyncio
    async def test_active_circuit_breaker_blocks_before_budget_and_sizing(self) -> None:
        state = make_state()
        state.activate_circuit_breaker(
            reason="manual_halt",
            message="manual halt from test",
            manual_release_required=True,
        )
        manager, event_bus, _, _ = make_manager(state=state)

        decision = await manager.evaluate_request(make_request())

        assert_denied_decision(decision)
        assert decision.decision is RiskDecisionType.HALT_TRADING
        assert "circuit_breaker" in decision.checks
        assert decision.checks["circuit_breaker"].passed is False
        assert "budget" not in decision.checks
        assert "position_sizing" not in decision.checks
        assert_event(event_bus, "risk.kill_switch")


# =============================================================================
# Sizing / leverage adjustments
# =============================================================================


class TestRiskManagerAdjustments:
    @pytest.mark.asyncio
    async def test_requested_size_above_risk_based_size_is_capped_not_allowed_as_requested(
        self,
    ) -> None:
        manager, event_bus, _, _ = make_manager()
        request = make_request(requested_size=999_999.0)

        decision = await manager.evaluate_request(request)

        assert_allowed_decision(decision)
        assert decision.final_size is not None
        assert decision.final_size < 999_999.0
        assert_event(event_bus, "signal.confirmed")

    @pytest.mark.asyncio
    async def test_high_requested_leverage_is_reduced_and_emits_warning(self) -> None:
        manager, event_bus, _, _ = make_manager()
        request = make_request(
            tier=TradeTier.T3,
            requested_leverage=100.0,
            liquidity_class=LiquidityClass.NORMAL,
        )

        decision = await manager.evaluate_request(request)

        assert decision.allowed is True
        assert decision.decision is RiskDecisionType.REDUCE_SIZE
        assert decision.final_leverage is not None
        assert decision.final_leverage < 100.0
        assert_event(event_bus, "risk.limit_warning")
        assert_event(event_bus, "risk.size_adjusted")

    @pytest.mark.asyncio
    async def test_requested_margin_caps_final_size_and_margin(self) -> None:
        manager, _, _, _ = make_manager()
        request = make_request(
            requested_margin=50.0,
            requested_leverage=5.0,
        )

        decision = await manager.evaluate_request(request)

        assert decision.allowed is True
        assert decision.final_margin is not None
        assert decision.final_margin <= 50.0 + 1e-9
        assert decision.final_size is not None
        assert decision.final_size > 0


# =============================================================================
# Reservation safety
# =============================================================================


class TestRiskManagerReservationFlow:
    @pytest.mark.asyncio
    async def test_second_allowed_request_sees_first_pending_reservation(self) -> None:
        config = make_risk_config(reservations_enabled=True)
        set_if_exists(config.exposure, "max_open_risk_r", 1.5)
        config.validate()

        state = make_state()
        manager, event_bus, _, _ = make_manager(config=config, state=state)

        first = await manager.evaluate_request(
            make_request(signal_id="signal-first-reservation")
        )
        assert first.allowed is True
        assert first.reservation_id is not None
        assert state.get_pending_open_risk() > 0

        second = await manager.evaluate_request(
            make_request(signal_id="signal-second-reservation")
        )

        assert second.allowed is False
        assert "exposure" in second.checks
        assert second.checks["exposure"].passed is False
        assert len(state.pending_reservations) == 1
        assert_event(event_bus, "risk.position_blocked")

    @pytest.mark.asyncio
    async def test_expired_reservations_are_cleaned_before_new_evaluation(self) -> None:
        config = make_risk_config(reservations_enabled=True)
        set_if_exists(config.reservation, "ttl_seconds", 0.001)
        set_if_exists(config.reservation, "auto_expire_on_evaluate", True)
        config.validate()

        state = make_state()
        manager, event_bus, _, _ = make_manager(config=config, state=state)

        first = await manager.evaluate_request(
            make_request(signal_id="reservation-expiry-first")
        )
        assert first.allowed is True
        assert len(state.pending_reservations) == 1

        # Make expiration deterministic without sleeping.
        for reservation in state.pending_reservations.values():
            reservation.expires_at = time.time() - 1.0

        second = await manager.evaluate_request(
            make_request(signal_id="reservation-expiry-second")
        )

        assert second.allowed is True
        assert len(state.pending_reservations) == 1

        topics = event_bus.topics()
        assert "risk.reservation.expired" in topics or "risk.reservation.released" in topics


# =============================================================================
# Serialization / payload shape
# =============================================================================


class TestRiskManagerDecisionSerialization:
    def test_serialize_decision_is_json_safe_and_uses_enum_values(self) -> None:
        request = make_request()
        checks = {
            "unit": RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[make_violation(RiskViolationType.INVALID_REQUEST)],
                adjusted_tier=TradeTier.T2,
                adjusted_size=1.0,
                adjusted_margin=20.0,
                adjusted_leverage=5.0,
                adjusted_risk_amount=10.0,
                risk_mode=RiskMode.NORMAL,
                reason="unit check failed",
                metadata={"tier": TradeTier.T2, "nested": {"side": PositionSide.LONG}},
            )
        }
        decision = make_decision(
            allowed=False,
            decision=RiskDecisionType.DENY,
            checks=checks,
            violations=[make_violation(RiskViolationType.INVALID_REQUEST)],
        )

        payload = RiskManager._serialize_decision(request, decision)

        assert_payload_has_decision_shape(payload)
        assert payload["decision"] == RiskDecisionType.DENY.value
        assert payload["risk_mode"] == RiskMode.NORMAL.value
        assert payload["side"] == PositionSide.LONG.value
        assert payload["final_tier"] == TradeTier.T2.value
        assert payload["violations"][0]["type"] == RiskViolationType.INVALID_REQUEST.value
        assert payload["checks"]["unit"]["decision"] == RiskDecisionType.DENY.value
        assert payload["checks"]["unit"]["metadata"]["tier"] == TradeTier.T2.value
        assert_json_serializable(payload)

    @pytest.mark.parametrize(
        ("decision_type", "allowed", "expected_primary"),
        [
            (RiskDecisionType.ALLOW, True, "signal.confirmed"),
            (RiskDecisionType.REDUCE_SIZE, True, "signal.confirmed"),
            (RiskDecisionType.REDUCE_RISK, True, "signal.confirmed"),
            (RiskDecisionType.DOWNGRADE_TIER, True, "signal.confirmed"),
            (RiskDecisionType.DENY, False, "risk.position_blocked"),
            (RiskDecisionType.HALT_TRADING, False, "risk.kill_switch"),
            (RiskDecisionType.EMERGENCY_STOP, False, "risk.kill_switch"),
        ],
    )
    def test_topic_for_decision_is_stable(
        self,
        decision_type: RiskDecisionType,
        allowed: bool,
        expected_primary: str,
    ) -> None:
        decision = make_decision(allowed=allowed, decision=decision_type)

        topic, priority = RiskManager._topic_for_decision(decision)

        assert topic == expected_primary
        if expected_primary == "risk.kill_switch":
            assert priority is not None

    def test_events_for_decision_adds_expected_legacy_topics(self) -> None:
        manager, _, _, _ = make_manager()
        request = make_request()

        approved = make_decision(allowed=True, decision=RiskDecisionType.ALLOW)
        approved_payload = RiskManager._serialize_decision(request, approved)
        approved_events = manager._events_for_decision(approved, approved_payload)
        assert [topic for topic, _, _ in approved_events] == [
            "signal.confirmed",
            "risk.approved",
        ]

        adjusted = make_decision(allowed=True, decision=RiskDecisionType.REDUCE_SIZE)
        adjusted_payload = RiskManager._serialize_decision(request, adjusted)
        adjusted_events = manager._events_for_decision(adjusted, adjusted_payload)
        assert [topic for topic, _, _ in adjusted_events] == [
            "signal.confirmed",
            "risk.limit_warning",
            "risk.size_adjusted",
        ]

        denied = make_decision(allowed=False, decision=RiskDecisionType.DENY)
        denied_payload = RiskManager._serialize_decision(request, denied)
        denied_events = manager._events_for_decision(denied, denied_payload)
        assert [topic for topic, _, _ in denied_events] == [
            "risk.position_blocked",
            "risk.rejected",
        ]


class TestRiskManagerPayloadParsing:
    def test_request_from_payload_accepts_string_enums_and_execution_cost_dict(self) -> None:
        payload = {
            "symbol": TEST_SYMBOL,
            "side": "long",
            "entry_price": "100.0",
            "stop_loss": "99.0",
            "take_profit": "103.0",
            "signal_id": TEST_SIGNAL_ID,
            "strategy_name": TEST_STRATEGY,
            "tier": "t2",
            "order_intent": "open",
            "liquidity_class": "high",
            "execution_quality": "acceptable",
            "confidence": "0.75",
            "edge_score": "0.65",
            "volatility": "0.2",
            "expected_reward": "3.0",
            "expected_loss": "1.0",
            "expected_win_probability": "0.55",
            "expected_cost": "0.03",
            "requested_size": None,
            "requested_margin": None,
            "requested_leverage": "5.0",
            "margin_mode": "isolated",
            "execution_cost": {
                "spread_cost": "0.01",
                "slippage_cost": "0.01",
                "fee_cost": "0.01",
                "funding_cost": "0.0",
                "spread_pct": "0.0001",
                "slippage_pct": "0.0001",
                "quality": "acceptable",
            },
            "metadata": {"source": "payload-test"},
        }

        request = RiskManager._request_from_payload(payload)

        assert request.symbol == TEST_SYMBOL
        assert request.side is PositionSide.LONG
        assert request.entry_price == pytest.approx(100.0)
        assert request.stop_loss == pytest.approx(99.0)
        assert request.take_profit == pytest.approx(103.0)
        assert request.tier is TradeTier.T2
        assert request.order_intent is OrderIntent.OPEN
        assert request.execution_cost is not None
        assert request.execution_cost.quality is ExecutionQuality.ACCEPTABLE

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"side": "long", "entry_price": 100.0},
            {"symbol": TEST_SYMBOL, "entry_price": 100.0},
            {"symbol": TEST_SYMBOL, "side": "long"},
            {"symbol": TEST_SYMBOL, "side": "not-a-side", "entry_price": 100.0},
            {"symbol": TEST_SYMBOL, "side": "long", "entry_price": "not-a-number"},
        ],
    )
    def test_request_from_payload_rejects_malformed_payloads(
        self,
        payload: dict[str, Any],
    ) -> None:
        with pytest.raises((ValueError, KeyError)):
            RiskManager._request_from_payload(payload)


# =============================================================================
# Metrics
# =============================================================================


class TestRiskManagerMetrics:
    @pytest.mark.asyncio
    async def test_metrics_are_updated_for_allow_deny_and_adjustment_paths(self) -> None:
        metrics = RiskMetrics()
        manager, _, _, _ = make_manager(metrics=metrics)

        allow = await manager.evaluate_request(
            make_request(signal_id="metrics-allow")
        )
        deny = await manager.evaluate_request(
            make_request(signal_id="metrics-deny", stop_loss=None)
        )
        adjust = await manager.evaluate_request(
            make_request(
                signal_id="metrics-adjust",
                requested_leverage=100.0,
                liquidity_class=LiquidityClass.NORMAL,
            )
        )

        assert allow.allowed is True
        assert deny.allowed is False
        assert adjust.allowed is True

        assert metrics.total_decisions == 3
        assert metrics.approvals == 2
        assert metrics.rejections == 1
        assert metrics.decision_latency_ms.count == 3
        assert metrics.last_symbol == TEST_SYMBOL
        assert metrics.last_strategy_name == TEST_STRATEGY
        assert metrics.decisions_by_symbol[TEST_SYMBOL].decisions == 3
        assert metrics.decisions_by_strategy[TEST_STRATEGY].decisions == 3


# =============================================================================
# Lock / re-entrancy safety
# =============================================================================


class TestRiskManagerLockSafety:
    @pytest.mark.asyncio
    async def test_events_are_not_emitted_while_manager_lock_is_held(self) -> None:
        manager, event_bus, _, _ = make_manager()
        emitted: list[tuple[str, dict[str, Any]]] = []

        async def strict_emit(
            topic: str,
            payload: dict[str, Any] | None = None,
            *,
            priority: Any | None = None,
            **kwargs: Any,
        ) -> None:
            assert not manager._lock.locked(), (
                f"RiskManager emitted topic={topic!r} while _lock was held"
            )
            emitted.append((topic, dict(payload or {})))
            await event_bus.emit(topic, payload, priority=priority, **kwargs)

        manager._emit_event = strict_emit  # type: ignore[method-assign]

        decision = await manager.evaluate_request(make_request())

        assert decision.allowed is True
        assert emitted
        assert "risk.request_received" in [topic for topic, _ in emitted]
        assert "signal.confirmed" in [topic for topic, _ in emitted]

    @pytest.mark.asyncio
    async def test_emit_hook_can_reenter_evaluate_request_without_deadlock(self) -> None:
        manager, event_bus, _, _ = make_manager()
        reentered = False

        original_emit = manager._emit_event

        async def reentrant_emit(
            topic: str,
            payload: dict[str, Any] | None = None,
            *,
            priority: Any | None = None,
            **kwargs: Any,
        ) -> None:
            nonlocal reentered

            assert not manager._lock.locked(), (
                f"RiskManager emitted topic={topic!r} while _lock was held"
            )

            await original_emit(topic, payload, priority=priority, **kwargs)

            if topic == "signal.confirmed" and not reentered:
                reentered = True
                nested = make_request(
                    signal_id="nested-reentrant-deny",
                    stop_loss=None,
                )
                nested_decision = await manager.evaluate_request(nested)
                assert nested_decision.allowed is False

        manager._emit_event = reentrant_emit  # type: ignore[method-assign]

        decision = await manager.evaluate_request(
            make_request(signal_id="outer-reentrant-allow")
        )

        assert decision.allowed is True
        assert reentered is True
        assert "signal.confirmed" in event_bus.topics()
        assert "risk.position_blocked" in event_bus.topics()


# =============================================================================
# Adversarial / fail-closed behavior
# =============================================================================


class TestRiskManagerAdversarialInputs:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("field_name", "bad_value"),
        [
            ("entry_price", math.nan),
            ("entry_price", math.inf),
            ("risk_nan_stop_loss", math.nan),
            ("expected_win_probability", math.nan),
            ("expected_cost", math.inf),
            ("requested_leverage", math.nan),
        ],
    )
    async def test_non_finite_request_values_do_not_result_in_allowed_decision(
        self,
        field_name: str,
        bad_value: float,
    ) -> None:
        """
        Цей тест навмисно суворий. Якщо він падає — треба додати finite validation
        в нижніх guard/utils/sizing шарах, а не дозволяти NaN/inf пройти в ALLOW.
        """
        manager, event_bus, state, _ = make_manager()

        if field_name == "risk_nan_stop_loss":
            request = make_request(stop_loss=bad_value)
        else:
            request = make_request(**{field_name: bad_value})

        try:
            decision = await manager.evaluate_request(request)
        except (ValueError, ArithmeticError):
            # Прийнятно для низькорівневої hard validation, але у фінальній
            # production-версії краще конвертувати це в DENY RiskDecision.
            return

        assert decision.allowed is False
        assert len(state.pending_reservations) == 0
        assert "signal.confirmed" not in event_bus.topics()

    @pytest.mark.asyncio
    async def test_manager_does_not_allow_when_base_equity_is_zero(self) -> None:
        state = make_state(balance=0.0, equity=0.0, free_balance=0.0)
        manager, event_bus, _, _ = make_manager(state=state)

        decision = await manager.evaluate_request(make_request())

        assert decision.allowed is False
        assert "signal.confirmed" not in event_bus.topics()
        assert len(state.pending_reservations) == 0

    @pytest.mark.asyncio
    async def test_unexpected_internal_exception_should_not_create_reservation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Якщо цей тест падає uncaught exception-ом — це корисний сигнал:
        RiskManager.evaluate_request() варто обгорнути fail-closed boundary,
        щоб internal guard/sizing exception повертав DENY і не валив event loop.
        """
        manager, _, state, _ = make_manager()

        def broken_check(*_: Any, **__: Any) -> RiskCheckResult:
            raise RuntimeError("synthetic guard failure")

        monkeypatch.setattr(manager._tier_guard, "check", broken_check)

        try:
            decision = await manager.evaluate_request(make_request())
        except RuntimeError:
            assert len(state.pending_reservations) == 0
            return

        assert decision.allowed is False
        assert len(state.pending_reservations) == 0