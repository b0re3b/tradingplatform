# tests/risk/test_risk_guards.py
from __future__ import annotations

import copy
import math
import time
from dataclasses import replace
from typing import Any

import pytest

from risk.budget import (
    RiskBudgetGuard,
    RiskModeResolver,
    StrategyRiskGuard,
    SymbolRiskGuard,
)
from risk.config import (
    ExecutionCostConfig,
    ExposureConfig,
    LeveragePolicyConfig,
    RiskBudgetConfig,
    StrategyRiskConfig,
    SymbolRiskConfig,
    TierModelConfig,
)
from risk.enums import (
    ExecutionQuality,
    LiquidityClass,
    MarginMode,
    OrderIntent,
    PositionSide,
    RiskDecisionType,
    RiskMode,
    RiskViolationType,
    StrategyRiskStatus,
    SymbolRiskStatus,
    TradeTier,
)
from risk.exposure_control import ExposureControl
from risk.guards import (
    ExecutionCostGuard,
    LeverageGuard,
    RiskRewardGuard,
    TierRiskGuard,
)
from risk.models import (
    ExecutionCostEstimate,
    ExpectedValueSnapshot,
    PortfolioPosition,
    RiskCheckResult,
    RiskEvaluationRequest,
    TierRiskProfile,
)
from risk.state import PendingRiskReservation, RiskState


TEST_SYMBOL = "BTCUSDT"
ALT_SYMBOL = "ETHUSDT"
TEST_STRATEGY = "test_strategy"
ALT_STRATEGY = "alt_strategy"


# =============================================================================
# Local builders
# =============================================================================


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


def make_request(
    *,
    symbol: str = TEST_SYMBOL,
    side: PositionSide = PositionSide.LONG,
    entry_price: float = 100.0,
    stop_loss: float | None = 99.0,
    take_profit: float | None = 103.0,
    signal_id: str | None = "signal-guard-test",
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
    expected_cost: float | None = 0.02,
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
        execution_cost=execution_cost,
        requested_size=requested_size,
        requested_margin=requested_margin,
        requested_leverage=requested_leverage,
        reduce_only=reduce_only,
        margin_mode=margin_mode,
        timestamp=time.time(),
        metadata=dict(metadata or {}),
    )


def make_tier_profile(
    *,
    requested_tier: TradeTier = TradeTier.T2,
    final_tier: TradeTier = TradeTier.T2,
    risk_units: float = 0.5,
    min_rr: float = 1.8,
    min_expected_value: float = 0.03,
    max_cost_to_reward_pct: float = 0.10,
    default_leverage: float = 5.0,
    max_leverage: float = 10.0,
) -> TierRiskProfile:
    return TierRiskProfile(
        requested_tier=requested_tier,
        final_tier=final_tier,
        risk_units=risk_units,
        min_rr=min_rr,
        min_expected_value=min_expected_value,
        max_cost_to_reward_pct=max_cost_to_reward_pct,
        default_leverage=default_leverage,
        max_leverage=max_leverage,
    )


def make_ev_snapshot(
    *,
    expected_reward: float = 3.0,
    expected_loss: float = 1.0,
    expected_cost: float = 0.05,
    win_probability: float | None = 0.55,
    risk_reward_ratio: float | None = 3.0,
    cost_to_reward_ratio: float | None = 0.02,
    expected_value: float | None = 1.2,
    expected_value_after_cost: float | None = 1.15,
) -> ExpectedValueSnapshot:
    return ExpectedValueSnapshot(
        expected_reward=expected_reward,
        expected_loss=expected_loss,
        expected_cost=expected_cost,
        win_probability=win_probability,
        risk_reward_ratio=risk_reward_ratio,
        cost_to_reward_ratio=cost_to_reward_ratio,
        expected_value=expected_value,
        expected_value_after_cost=expected_value_after_cost,
    )


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


def make_position(
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
    stop_loss: float | None = 99.0,
    take_profit: float | None = 103.0,
    tier: TradeTier | None = TradeTier.T2,
    strategy_name: str | None = TEST_STRATEGY,
    signal_id: str | None = "signal-position-test",
    position_id: str | None = "position-test",
    realized_pnl: float = 0.0,
    unrealized_pnl: float = 0.0,
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
        signal_id=signal_id,
        position_id=position_id,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        opened_at=time.time(),
        updated_at=time.time(),
    )


def make_reservation(
    *,
    reservation_id: str = "reservation-test",
    symbol: str = TEST_SYMBOL,
    side: PositionSide = PositionSide.LONG,
    signal_id: str | None = "signal-reservation-test",
    strategy_name: str | None = TEST_STRATEGY,
    tier: TradeTier | None = TradeTier.T2,
    position_id: str | None = None,
    size: float = 1.0,
    open_risk: float = 10.0,
    margin: float = 20.0,
    notional: float = 100.0,
    expires_at: float | None = None,
) -> PendingRiskReservation:
    return PendingRiskReservation(
        reservation_id=reservation_id,
        symbol=symbol,
        side=side,
        signal_id=signal_id,
        strategy_name=strategy_name,
        tier=tier,
        position_id=position_id,
        size=size,
        open_risk=open_risk,
        margin=margin,
        notional=notional,
        created_at=time.time(),
        expires_at=expires_at,
    )


def add_position(state: RiskState, position: PortfolioPosition) -> None:
    state.add_position(position)


def add_reservation(state: RiskState, reservation: PendingRiskReservation) -> None:
    state.pending_reservations[reservation.reservation_id] = reservation
    state.updated_at = time.time()


def violation_types(result: RiskCheckResult) -> set[RiskViolationType]:
    return {violation.violation_type for violation in result.violations}


def assert_allowed(result: RiskCheckResult) -> None:
    assert result.passed is True
    assert result.decision in {
        RiskDecisionType.ALLOW,
        RiskDecisionType.REDUCE_RISK,
        RiskDecisionType.REDUCE_SIZE,
        RiskDecisionType.DOWNGRADE_TIER,
    }


def assert_denied(
    result: RiskCheckResult,
    *,
    decision: RiskDecisionType | None = None,
    violation_type: RiskViolationType | None = None,
) -> None:
    assert result.passed is False
    if decision is not None:
        assert result.decision is decision
    if violation_type is not None:
        assert violation_type in violation_types(result)
    assert result.reason is not None or result.violations


def snapshot_state_for_readonly_assertion(state: RiskState) -> dict[str, Any]:
    return {
        "equity": state.equity,
        "free_balance": state.free_balance,
        "used_margin": state.used_margin,
        "risk_mode": state.risk_mode,
        "positions": copy.deepcopy(getattr(state, "positions", {})),
        "pending_reservations": copy.deepcopy(getattr(state, "pending_reservations", {})),
        "symbols": copy.deepcopy(getattr(state, "symbols", {})),
        "strategies": copy.deepcopy(getattr(state, "strategies", {})),
    }


# =============================================================================
# TierRiskGuard
# =============================================================================


class TestTierRiskGuard:
    def test_missing_tier_uses_config_default_tier(self) -> None:
        config = TierModelConfig()
        guard = TierRiskGuard(config)
        state = make_state()
        request = make_request(tier=None)

        result = guard.check(request, state)

        assert result.passed is True
        assert result.decision is RiskDecisionType.ALLOW
        assert result.adjusted_tier is config.default_tier
        assert result.metadata["tier_profile"].final_tier is config.default_tier

    def test_normal_mode_allows_highest_configured_tier(self) -> None:
        guard = TierRiskGuard(TierModelConfig())
        state = make_state(mode=RiskMode.NORMAL)
        request = make_request(tier=TradeTier.T4)

        result = guard.check(request, state, mode=RiskMode.NORMAL)

        assert result.passed is True
        assert result.decision is RiskDecisionType.ALLOW
        assert result.adjusted_tier is TradeTier.T4
        assert not result.violations

    @pytest.mark.parametrize(
        ("mode", "requested_tier", "expected_tier"),
        [
            (RiskMode.CAUTION, TradeTier.T4, TradeTier.T3),
            (RiskMode.SAFE_MODE, TradeTier.T4, TradeTier.T2),
            (RiskMode.SAFE_MODE, TradeTier.T3, TradeTier.T2),
        ],
    )
    def test_risk_modes_downgrade_tier_instead_of_silently_allowing_high_risk(
        self,
        mode: RiskMode,
        requested_tier: TradeTier,
        expected_tier: TradeTier,
    ) -> None:
        guard = TierRiskGuard(TierModelConfig())
        state = make_state(mode=mode)
        request = make_request(tier=requested_tier)

        result = guard.check(request, state, mode=mode)

        assert result.passed is True
        assert result.decision is RiskDecisionType.DOWNGRADE_TIER
        assert result.adjusted_tier is expected_tier
        assert RiskViolationType.TIER_DOWNGRADED in violation_types(result)

        profile = result.metadata["tier_profile"]
        assert profile.requested_tier is requested_tier
        assert profile.final_tier is expected_tier
        assert profile.downgraded is True

    @pytest.mark.parametrize(
        "intent",
        [OrderIntent.OPEN, OrderIntent.INCREASE, OrderIntent.FLIP],
    )
    def test_reduce_only_denies_risk_increasing_orders(self, intent: OrderIntent) -> None:
        guard = TierRiskGuard(TierModelConfig())
        state = make_state(mode=RiskMode.REDUCE_ONLY)
        request = make_request(order_intent=intent, tier=TradeTier.T1)

        result = guard.check(request, state, mode=RiskMode.REDUCE_ONLY)

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=RiskViolationType.TIER_NOT_ALLOWED,
        )
        assert "REDUCE_ONLY" in result.reason

    @pytest.mark.parametrize(
        "intent",
        [OrderIntent.REDUCE, OrderIntent.CLOSE],
    )
    def test_reduce_only_currently_denies_tier_check_even_for_reduce_orders_to_expose_pipeline_requirement(
        self,
        intent: OrderIntent,
    ) -> None:
        """
        This test documents current TierRiskGuard behavior.

        If RiskManager bypasses tier checks for reducing orders, this is okay.
        If it sends REDUCE/CLOSE through TierRiskGuard, this test reminds us that
        TierRiskGuard itself still denies REDUCE_ONLY because final tier validation
        rejects REDUCE_ONLY modes.
        """
        guard = TierRiskGuard(TierModelConfig())
        state = make_state(mode=RiskMode.REDUCE_ONLY)
        request = make_request(order_intent=intent, tier=TradeTier.T1)

        result = guard.check(request, state, mode=RiskMode.REDUCE_ONLY)

        assert_denied(result, violation_type=RiskViolationType.TIER_NOT_ALLOWED)

    @pytest.mark.parametrize("mode", [RiskMode.HALTED, RiskMode.EMERGENCY_STOP])
    def test_terminal_modes_deny_all_tier_profiles(self, mode: RiskMode) -> None:
        guard = TierRiskGuard(TierModelConfig())
        state = make_state(mode=mode)
        request = make_request(tier=TradeTier.T1)

        result = guard.check(request, state, mode=mode)

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=RiskViolationType.TIER_NOT_ALLOWED,
        )
        assert result.risk_mode is mode

    def test_resolve_profile_returns_exact_tier_policy_values(self) -> None:
        config = TierModelConfig()
        guard = TierRiskGuard(config)
        state = make_state()
        request = make_request(tier=TradeTier.T2)

        profile = guard.resolve_profile(request, state)

        tier_config = config.tiers[TradeTier.T2]
        assert profile.final_tier is TradeTier.T2
        assert profile.risk_units == pytest.approx(tier_config.risk_units)
        assert profile.min_rr == pytest.approx(tier_config.min_rr)
        assert profile.min_expected_value == pytest.approx(tier_config.min_expected_value)
        assert profile.max_cost_to_reward_pct == pytest.approx(
            tier_config.max_cost_to_reward_pct
        )
        assert profile.default_leverage == pytest.approx(tier_config.default_leverage)
        assert profile.max_leverage == pytest.approx(tier_config.max_leverage)


# =============================================================================
# RiskRewardGuard
# =============================================================================


class TestRiskRewardGuard:
    def test_valid_rr_and_positive_ev_passes(self) -> None:
        guard = RiskRewardGuard()
        request = make_request(
            entry_price=100.0,
            stop_loss=99.0,
            take_profit=103.0,
            expected_reward=3.0,
            expected_loss=1.0,
            expected_win_probability=0.55,
            expected_cost=0.02,
        )
        profile = make_tier_profile(min_rr=1.8, min_expected_value=0.03)

        result = guard.check(request, profile)

        assert result.passed is True
        assert result.decision is RiskDecisionType.ALLOW
        assert "expected_value_snapshot" in result.metadata
        snapshot = result.metadata["expected_value_snapshot"]
        assert snapshot.risk_reward_ratio == pytest.approx(3.0)
        assert snapshot.expected_value_after_cost is not None
        assert snapshot.expected_value_after_cost > 0

    @pytest.mark.parametrize(
        ("stop_loss", "expected_violation"),
        [
            (None, RiskViolationType.STOP_LOSS_MISSING),
            (100.0, RiskViolationType.STOP_LOSS_SIDE_INVALID),
            (101.0, RiskViolationType.STOP_LOSS_SIDE_INVALID),
        ],
    )
    def test_missing_or_wrong_long_stop_loss_is_denied(
        self,
        stop_loss: float | None,
        expected_violation: RiskViolationType,
    ) -> None:
        guard = RiskRewardGuard()
        request = make_request(
            side=PositionSide.LONG,
            entry_price=100.0,
            stop_loss=stop_loss,
            take_profit=103.0,
        )

        result = guard.check(request, make_tier_profile())

        assert_denied(result, violation_type=expected_violation)

    @pytest.mark.parametrize(
        ("take_profit", "expected_violation"),
        [
            (None, RiskViolationType.TAKE_PROFIT_MISSING),
            (100.0, RiskViolationType.TAKE_PROFIT_SIDE_INVALID),
            (99.0, RiskViolationType.TAKE_PROFIT_SIDE_INVALID),
        ],
    )
    def test_missing_or_wrong_long_take_profit_is_denied(
        self,
        take_profit: float | None,
        expected_violation: RiskViolationType,
    ) -> None:
        guard = RiskRewardGuard()
        request = make_request(
            side=PositionSide.LONG,
            entry_price=100.0,
            stop_loss=99.0,
            take_profit=take_profit,
        )

        result = guard.check(request, make_tier_profile())

        assert_denied(result, violation_type=expected_violation)

    @pytest.mark.parametrize(
        ("stop_loss", "take_profit", "expected_violation"),
        [
            (99.0, 97.0, RiskViolationType.STOP_LOSS_SIDE_INVALID),
            (101.0, 101.0, RiskViolationType.TAKE_PROFIT_SIDE_INVALID),
        ],
    )
    def test_short_side_validation_is_strict(
        self,
        stop_loss: float,
        take_profit: float,
        expected_violation: RiskViolationType,
    ) -> None:
        guard = RiskRewardGuard()
        request = make_request(
            side=PositionSide.SHORT,
            entry_price=100.0,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        result = guard.check(request, make_tier_profile())

        assert_denied(result, violation_type=expected_violation)

    def test_rr_below_tier_minimum_is_denied(self) -> None:
        guard = RiskRewardGuard()
        request = make_request(
            entry_price=100.0,
            stop_loss=99.0,
            take_profit=100.5,
            expected_reward=0.5,
            expected_loss=1.0,
            expected_win_probability=0.80,
        )
        profile = make_tier_profile(min_rr=2.0)

        result = guard.check(request, profile)

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=RiskViolationType.RISK_REWARD_TOO_LOW,
        )
        assert result.metadata["expected_value_snapshot"].risk_reward_ratio == pytest.approx(0.5)

    def test_expected_value_after_cost_below_minimum_is_denied_even_when_rr_is_good(self) -> None:
        guard = RiskRewardGuard()
        request = make_request(
            entry_price=100.0,
            stop_loss=99.0,
            take_profit=105.0,
            expected_reward=5.0,
            expected_loss=1.0,
            expected_win_probability=0.10,
            expected_cost=0.10,
        )
        profile = make_tier_profile(min_rr=2.0, min_expected_value=0.03)

        result = guard.check(request, profile)

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=RiskViolationType.EXPECTED_VALUE_NEGATIVE,
        )

    @pytest.mark.parametrize(
        ("win_probability", "expected_cost"),
        [
            (math.nan, 0.01),
            (math.inf, 0.01),
            (0.55, math.nan),
            (0.55, math.inf),
        ],
    )
    def test_non_finite_probability_or_cost_should_not_be_allowed(
        self,
        win_probability: float,
        expected_cost: float,
    ) -> None:
        """
        This is intentionally harsh. If it fails, add finite validation in
        RiskRewardGuard.evaluate() / risk.utils.calculate_expected_value().
        """
        guard = RiskRewardGuard()
        request = make_request(
            expected_win_probability=win_probability,
            expected_cost=expected_cost,
        )

        result = guard.check(request, make_tier_profile())

        assert result.passed is False


# =============================================================================
# ExecutionCostGuard
# =============================================================================


class TestExecutionCostGuard:
    def test_disabled_execution_cost_guard_allows_without_cost_validation(self) -> None:
        config = ExecutionCostConfig(enabled=False)
        guard = ExecutionCostGuard(config)
        request = make_request(
            execution_cost=make_execution_cost(
                spread_pct=999.0,
                slippage_pct=999.0,
                quality=ExecutionQuality.BLOCKED,
            )
        )
        ev = make_ev_snapshot(cost_to_reward_ratio=999.0, expected_value_after_cost=-999.0)

        result = guard.check(request, make_tier_profile(), ev)

        assert result.passed is True
        assert result.decision is RiskDecisionType.ALLOW
        assert result.metadata["enabled"] is False

    def test_acceptable_cost_and_positive_ev_passes(self) -> None:
        config = ExecutionCostConfig(
            max_spread_pct=0.001,
            max_slippage_pct=0.001,
            min_execution_quality=ExecutionQuality.ACCEPTABLE,
            default_max_cost_to_reward_pct=0.10,
        )
        guard = ExecutionCostGuard(config)
        request = make_request(execution_cost=make_execution_cost())
        ev = make_ev_snapshot(cost_to_reward_ratio=0.02, expected_value_after_cost=1.0)

        result = guard.check(request, make_tier_profile(), ev)

        assert result.passed is True
        assert result.decision is RiskDecisionType.ALLOW

    @pytest.mark.parametrize(
        ("cost", "expected_violation"),
        [
            (
                make_execution_cost(spread_pct=0.05),
                RiskViolationType.SPREAD_TOO_WIDE,
            ),
            (
                make_execution_cost(slippage_pct=0.05),
                RiskViolationType.SLIPPAGE_TOO_HIGH,
            ),
            (
                make_execution_cost(quality=ExecutionQuality.POOR),
                RiskViolationType.EXECUTION_QUALITY_TOO_LOW,
            ),
            (
                make_execution_cost(quality=ExecutionQuality.BLOCKED),
                RiskViolationType.EXECUTION_QUALITY_TOO_LOW,
            ),
        ],
    )
    def test_bad_execution_inputs_are_denied(
        self,
        cost: ExecutionCostEstimate,
        expected_violation: RiskViolationType,
    ) -> None:
        config = ExecutionCostConfig(
            max_spread_pct=0.001,
            max_slippage_pct=0.001,
            min_execution_quality=ExecutionQuality.ACCEPTABLE,
            default_max_cost_to_reward_pct=0.10,
        )
        guard = ExecutionCostGuard(config)
        request = make_request(execution_cost=cost)
        ev = make_ev_snapshot(cost_to_reward_ratio=0.02, expected_value_after_cost=1.0)

        result = guard.check(request, make_tier_profile(), ev)

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=expected_violation,
        )

    def test_cost_to_reward_above_limit_is_denied(self) -> None:
        config = ExecutionCostConfig(default_max_cost_to_reward_pct=0.10)
        guard = ExecutionCostGuard(config)
        request = make_request(execution_cost=make_execution_cost())
        ev = make_ev_snapshot(
            expected_reward=3.0,
            expected_cost=1.0,
            cost_to_reward_ratio=0.333,
            expected_value_after_cost=1.0,
        )

        result = guard.check(request, make_tier_profile(max_cost_to_reward_pct=0.10), ev)

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=RiskViolationType.EXECUTION_COST_TOO_HIGH,
        )
        assert result.metadata["cost_to_reward_ratio"] == pytest.approx(0.333)

    def test_safe_mode_uses_stricter_cost_to_reward_limit(self) -> None:
        config = ExecutionCostConfig(
            default_max_cost_to_reward_pct=0.10,
            safe_mode_max_cost_to_reward_pct=0.08,
        )
        guard = ExecutionCostGuard(config)
        request = make_request(execution_cost=make_execution_cost())
        ev = make_ev_snapshot(cost_to_reward_ratio=0.09, expected_value_after_cost=1.0)

        result = guard.check(
            request,
            make_tier_profile(max_cost_to_reward_pct=0.10),
            ev,
            mode=RiskMode.SAFE_MODE,
        )

        assert_denied(result, violation_type=RiskViolationType.EXECUTION_COST_TOO_HIGH)
        assert result.metadata["max_cost_to_reward"] == pytest.approx(0.08)

    def test_non_positive_ev_after_cost_is_denied(self) -> None:
        config = ExecutionCostConfig(require_positive_ev_after_cost=True)
        guard = ExecutionCostGuard(config)
        request = make_request(execution_cost=make_execution_cost())
        ev = make_ev_snapshot(cost_to_reward_ratio=0.01, expected_value_after_cost=0.0)

        result = guard.check(request, make_tier_profile(), ev)

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=RiskViolationType.EXPECTED_VALUE_NEGATIVE,
        )

    @pytest.mark.parametrize(
        "cost_ratio",
        [math.nan, math.inf],
    )
    def test_non_finite_cost_ratio_should_not_pass(
        self,
        cost_ratio: float,
    ) -> None:
        guard = ExecutionCostGuard(ExecutionCostConfig())
        request = make_request(execution_cost=make_execution_cost())
        ev = make_ev_snapshot(cost_to_reward_ratio=cost_ratio, expected_value_after_cost=1.0)

        result = guard.check(request, make_tier_profile(), ev)

        assert result.passed is False


# =============================================================================
# LeverageGuard
# =============================================================================


class TestLeverageGuard:
    def test_default_leverage_is_used_when_request_has_no_leverage(self) -> None:
        config = LeveragePolicyConfig(default_leverage=5.0)
        guard = LeverageGuard(config)
        request = make_request(requested_leverage=None)
        profile = make_tier_profile(default_leverage=3.0, max_leverage=10.0)

        result = guard.check(request, profile, mode=RiskMode.NORMAL)

        assert result.passed is True
        assert result.decision is RiskDecisionType.ALLOW
        assert result.adjusted_leverage == pytest.approx(3.0)

    def test_requested_leverage_below_all_caps_passes(self) -> None:
        guard = LeverageGuard(LeveragePolicyConfig(default_leverage=10.0))
        request = make_request(requested_leverage=3.0, liquidity_class=LiquidityClass.HIGH)
        profile = make_tier_profile(max_leverage=10.0)

        result = guard.check(request, profile)

        assert result.passed is True
        assert result.decision is RiskDecisionType.ALLOW
        assert result.adjusted_leverage == pytest.approx(3.0)

    def test_requested_leverage_above_cap_is_reduced_not_silently_allowed(self) -> None:
        config = LeveragePolicyConfig(default_leverage=10.0)
        guard = LeverageGuard(config)
        request = make_request(requested_leverage=20.0, liquidity_class=LiquidityClass.NORMAL)
        profile = make_tier_profile(final_tier=TradeTier.T3, max_leverage=5.0)

        result = guard.check(request, profile)

        assert result.passed is True
        assert result.decision is RiskDecisionType.REDUCE_SIZE
        assert result.adjusted_leverage == pytest.approx(5.0)
        assert RiskViolationType.MAX_LEVERAGE_EXCEEDED in violation_types(result)

    @pytest.mark.parametrize(
        ("liquidity", "expected_max"),
        [
            (LiquidityClass.TOP, 10.0),
            (LiquidityClass.HIGH, 10.0),
            (LiquidityClass.NORMAL, 5.0),
            (LiquidityClass.LOW, 3.0),
            (LiquidityClass.ILLIQUID, 2.0),
            (LiquidityClass.SHITCOIN, 3.0),
        ],
    )
    def test_liquidity_class_caps_max_leverage(
        self,
        liquidity: LiquidityClass,
        expected_max: float,
    ) -> None:
        config = LeveragePolicyConfig(default_leverage=10.0)
        guard = LeverageGuard(config)
        request = make_request(
            requested_leverage=100.0,
            liquidity_class=liquidity,
            execution_quality=ExecutionQuality.ACCEPTABLE,
        )
        profile = make_tier_profile(final_tier=TradeTier.T2, max_leverage=10.0)

        result = guard.check(request, profile)

        assert result.passed is True
        assert result.decision is RiskDecisionType.REDUCE_SIZE
        assert result.adjusted_leverage == pytest.approx(expected_max)

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            (RiskMode.CAUTION, 5.0),
            (RiskMode.SAFE_MODE, 3.0),
            (RiskMode.REDUCE_ONLY, 1.0),
        ],
    )
    def test_risk_mode_caps_max_leverage(self, mode: RiskMode, expected: float) -> None:
        guard = LeverageGuard(LeveragePolicyConfig(default_leverage=10.0))
        request = make_request(requested_leverage=100.0, liquidity_class=LiquidityClass.HIGH)
        profile = make_tier_profile(max_leverage=10.0)

        result = guard.check(request, profile, mode=mode)

        assert result.passed is True
        assert result.decision is RiskDecisionType.REDUCE_SIZE
        assert result.adjusted_leverage == pytest.approx(expected)

    @pytest.mark.parametrize("mode", [RiskMode.HALTED, RiskMode.EMERGENCY_STOP])
    def test_terminal_modes_deny_leverage(self, mode: RiskMode) -> None:
        guard = LeverageGuard(LeveragePolicyConfig())
        request = make_request(requested_leverage=1.0)
        profile = make_tier_profile()

        result = guard.check(request, profile, mode=mode)

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=RiskViolationType.LEVERAGE_NOT_ALLOWED,
        )

    @pytest.mark.parametrize("bad_leverage", [0.0, -1.0])
    def test_non_positive_leverage_is_denied(self, bad_leverage: float) -> None:
        guard = LeverageGuard(LeveragePolicyConfig())
        request = make_request(requested_leverage=bad_leverage)

        result = guard.check(request, make_tier_profile())

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=RiskViolationType.LEVERAGE_NOT_ALLOWED,
        )

    @pytest.mark.parametrize("bad_leverage", [math.nan, math.inf])
    def test_non_finite_leverage_should_not_pass(self, bad_leverage: float) -> None:
        guard = LeverageGuard(LeveragePolicyConfig())
        request = make_request(requested_leverage=bad_leverage)

        result = guard.check(request, make_tier_profile())

        assert result.passed is False


# =============================================================================
# RiskModeResolver / RiskBudgetGuard
# =============================================================================


class TestRiskModeResolverAndBudgetGuard:
    def test_no_losses_resolves_normal_mode(self) -> None:
        resolver = RiskModeResolver(RiskBudgetConfig())
        state = make_state()

        mode, reason = resolver.resolve(state, risk_unit=10.0)

        assert mode is RiskMode.NORMAL
        assert reason is None

    @pytest.mark.parametrize(
        ("daily_pnl", "expected_mode"),
        [
            (-30.0, RiskMode.CAUTION),
            (-60.0, RiskMode.SAFE_MODE),
            (-100.0, RiskMode.HALTED),
        ],
    )
    def test_daily_loss_thresholds_resolve_expected_modes(
        self,
        daily_pnl: float,
        expected_mode: RiskMode,
    ) -> None:
        resolver = RiskModeResolver(RiskBudgetConfig())
        state = make_state()
        state.daily_pnl = daily_pnl

        mode, reason = resolver.resolve(state, risk_unit=10.0)

        assert mode is expected_mode
        assert reason is not None

    def test_soft_daily_loss_can_resolve_reduce_only_when_new_positions_not_allowed(self) -> None:
        config = RiskBudgetConfig(allow_new_positions_after_soft_daily_loss=False)
        resolver = RiskModeResolver(config)
        state = make_state()
        state.daily_pnl = -60.0

        mode, reason = resolver.resolve(state, risk_unit=10.0)

        assert mode is RiskMode.REDUCE_ONLY
        assert reason is not None

    @pytest.mark.parametrize(
        ("weekly_pnl", "monthly_pnl", "expected_mode"),
        [
            (-250.0, 0.0, RiskMode.HALTED),
            (0.0, -400.0, RiskMode.REDUCE_ONLY),
            (0.0, -500.0, RiskMode.EMERGENCY_STOP),
        ],
    )
    def test_weekly_and_monthly_thresholds_resolve_expected_modes(
        self,
        weekly_pnl: float,
        monthly_pnl: float,
        expected_mode: RiskMode,
    ) -> None:
        resolver = RiskModeResolver(RiskBudgetConfig())
        state = make_state()
        state.weekly_pnl = weekly_pnl
        state.monthly_pnl = monthly_pnl

        mode, reason = resolver.resolve(state, risk_unit=10.0)

        assert mode is expected_mode
        assert reason is not None

    def test_manual_halt_overrides_budget_mode(self) -> None:
        resolver = RiskModeResolver(RiskBudgetConfig())
        state = make_state()
        state.halt_trading(reason="manual halt test")

        mode, reason = resolver.resolve(state, risk_unit=10.0)

        assert mode is RiskMode.HALTED
        assert "manual halt test" in reason

    def test_emergency_stop_flag_overrides_budget_mode(self) -> None:
        resolver = RiskModeResolver(RiskBudgetConfig())
        state = make_state()
        state.activate_emergency_stop(reason="emergency test")

        mode, reason = resolver.resolve(state, risk_unit=10.0)

        assert mode is RiskMode.EMERGENCY_STOP
        assert reason is not None

    @pytest.mark.parametrize("bad_risk_unit", [0.0, -1.0, math.nan, math.inf])
    def test_invalid_risk_unit_resolves_halted(self, bad_risk_unit: float) -> None:
        resolver = RiskModeResolver(RiskBudgetConfig())
        state = make_state()

        mode, reason = resolver.resolve(state, risk_unit=bad_risk_unit)

        assert mode is RiskMode.HALTED
        assert reason is not None

    def test_budget_guard_allows_normal_mode(self) -> None:
        guard = RiskBudgetGuard(RiskBudgetConfig())
        result = guard.check(make_request(), make_state(), risk_unit=10.0)

        assert result.passed is True
        assert result.decision is RiskDecisionType.ALLOW
        assert result.risk_mode is RiskMode.NORMAL

    @pytest.mark.parametrize(
        ("daily_pnl", "expected_decision", "expected_mode"),
        [
            (-30.0, RiskDecisionType.REDUCE_RISK, RiskMode.CAUTION),
            (-60.0, RiskDecisionType.REDUCE_RISK, RiskMode.SAFE_MODE),
            (-100.0, RiskDecisionType.HALT_TRADING, RiskMode.HALTED),
        ],
    )
    def test_budget_guard_decisions_follow_risk_mode(
        self,
        daily_pnl: float,
        expected_decision: RiskDecisionType,
        expected_mode: RiskMode,
    ) -> None:
        state = make_state()
        state.daily_pnl = daily_pnl
        guard = RiskBudgetGuard(RiskBudgetConfig())

        result = guard.check(make_request(), state, risk_unit=10.0)

        assert result.risk_mode is expected_mode
        assert result.decision is expected_decision
        if expected_mode is RiskMode.HALTED:
            assert result.passed is False
        else:
            assert result.passed is True

    def test_budget_guard_emergency_stop_denies(self) -> None:
        state = make_state()
        state.monthly_pnl = -500.0
        guard = RiskBudgetGuard(RiskBudgetConfig())

        result = guard.check(make_request(), state, risk_unit=10.0)

        assert_denied(
            result,
            decision=RiskDecisionType.EMERGENCY_STOP,
            violation_type=RiskViolationType.EMERGENCY_STOP_TRIGGERED,
        )
        assert result.risk_mode is RiskMode.EMERGENCY_STOP

    @pytest.mark.parametrize(
        "restricted_mode",
        [RiskMode.REDUCE_ONLY, RiskMode.HALTED, RiskMode.EMERGENCY_STOP],
    )
    def test_budget_guard_allows_reduce_orders_even_when_mode_is_restricted(
        self,
        restricted_mode: RiskMode,
    ) -> None:
        class StaticResolver:
            def resolve(self, state: RiskState, *, risk_unit: float) -> tuple[RiskMode, str]:
                return restricted_mode, f"forced {restricted_mode.value}"

        guard = RiskBudgetGuard(
            RiskBudgetConfig(),
            mode_resolver=StaticResolver(),  # type: ignore[arg-type]
        )
        request = make_request(order_intent=OrderIntent.CLOSE)
        state = make_state(mode=restricted_mode)

        result = guard.check(request, state, risk_unit=10.0)

        assert result.passed is True
        assert result.decision is RiskDecisionType.ALLOW
        assert result.metadata["reduce_order_allowed"] is True


# =============================================================================
# SymbolRiskGuard
# =============================================================================


class TestSymbolRiskGuard:
    def test_clean_symbol_state_allows_request(self) -> None:
        guard = SymbolRiskGuard(SymbolRiskConfig())
        state = make_state()
        request = make_request()

        result = guard.check(
            request,
            state,
            risk_unit=10.0,
            candidate_open_risk=5.0,
        )

        assert result.passed is True
        assert result.decision is RiskDecisionType.ALLOW

    @pytest.mark.parametrize("bad_risk_unit", [0.0, -1.0, math.nan, math.inf])
    def test_invalid_risk_unit_is_denied(self, bad_risk_unit: float) -> None:
        guard = SymbolRiskGuard(SymbolRiskConfig())
        result = guard.check(
            make_request(),
            make_state(),
            risk_unit=bad_risk_unit,
            candidate_open_risk=1.0,
        )

        assert_denied(result, decision=RiskDecisionType.DENY)

    @pytest.mark.parametrize("bad_open_risk", [-1.0, math.nan, math.inf])
    def test_invalid_candidate_open_risk_is_denied(self, bad_open_risk: float) -> None:
        guard = SymbolRiskGuard(SymbolRiskConfig())
        result = guard.check(
            make_request(),
            make_state(),
            risk_unit=10.0,
            candidate_open_risk=bad_open_risk,
        )

        assert_denied(result, decision=RiskDecisionType.DENY)

    def test_disabled_symbol_is_denied(self) -> None:
        guard = SymbolRiskGuard(SymbolRiskConfig())
        state = make_state()
        state.get_symbol_state(TEST_SYMBOL).disable(reason="symbol disabled test")

        result = guard.check(make_request(), state, risk_unit=10.0, candidate_open_risk=1.0)

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=RiskViolationType.SYMBOL_DISABLED,
        )

    def test_symbol_cooldown_reduces_risk_not_hard_deny(self) -> None:
        guard = SymbolRiskGuard(SymbolRiskConfig())
        state = make_state()
        state.get_symbol_state(TEST_SYMBOL).activate_cooldown(
            cooldown_seconds=60.0,
            reason="cooldown test",
        )

        result = guard.check(make_request(), state, risk_unit=10.0, candidate_open_risk=1.0)

        assert result.passed is True
        assert result.decision is RiskDecisionType.REDUCE_RISK
        assert RiskViolationType.SYMBOL_COOLDOWN_ACTIVE in violation_types(result)

    def test_symbol_daily_loss_breach_is_denied(self) -> None:
        config = SymbolRiskConfig(max_symbol_daily_loss_r=2.0)
        guard = SymbolRiskGuard(config)
        state = make_state()
        symbol_state = state.get_symbol_state(TEST_SYMBOL)
        symbol_state.daily_pnl = -20.0

        result = guard.check(make_request(), state, risk_unit=10.0, candidate_open_risk=1.0)

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=RiskViolationType.SYMBOL_DAILY_LOSS_EXCEEDED,
        )

    def test_symbol_open_risk_breach_includes_candidate_open_risk(self) -> None:
        config = SymbolRiskConfig(max_symbol_open_risk_r=2.0)
        guard = SymbolRiskGuard(config)
        state = make_state()
        state.get_symbol_state(TEST_SYMBOL).open_risk = 15.0

        result = guard.check(make_request(), state, risk_unit=10.0, candidate_open_risk=6.0)

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=RiskViolationType.SYMBOL_OPEN_RISK_EXCEEDED,
        )
        assert result.metadata["projected_open_risk_r"] > 2.0

    def test_pending_reservations_count_against_symbol_position_limit(self) -> None:
        config = SymbolRiskConfig(max_positions_per_symbol=1)
        guard = SymbolRiskGuard(config)
        state = make_state()
        add_reservation(
            state,
            make_reservation(
                reservation_id="pending-symbol-1",
                symbol=TEST_SYMBOL,
                open_risk=1.0,
            ),
        )

        result = guard.check(make_request(), state, risk_unit=10.0, candidate_open_risk=1.0)

        assert result.passed is False or result.decision is RiskDecisionType.REDUCE_RISK
        assert result.metadata["pending_reservations_count"] == 1

    def test_reduce_order_bypasses_symbol_limits(self) -> None:
        config = SymbolRiskConfig(max_symbol_open_risk_r=0.1, max_positions_per_symbol=0)
        guard = SymbolRiskGuard(config)
        state = make_state()
        state.get_symbol_state(TEST_SYMBOL).disable(reason="still should allow close")
        request = make_request(order_intent=OrderIntent.CLOSE)

        result = guard.check(request, state, risk_unit=10.0, candidate_open_risk=999.0)

        assert result.passed is True
        assert result.decision is RiskDecisionType.ALLOW
        assert result.metadata["reduce_order_allowed"] is True


# =============================================================================
# StrategyRiskGuard
# =============================================================================


class TestStrategyRiskGuard:
    def test_missing_strategy_name_passes_with_metadata(self) -> None:
        guard = StrategyRiskGuard(StrategyRiskConfig())
        state = make_state()
        request = make_request(strategy_name=None)

        result = guard.check(request, state, risk_unit=10.0, candidate_open_risk=1.0)

        assert result.passed is True
        assert result.decision is RiskDecisionType.ALLOW
        assert result.metadata["strategy_name_missing"] is True

    def test_clean_strategy_allows_request(self) -> None:
        guard = StrategyRiskGuard(StrategyRiskConfig())
        state = make_state()

        result = guard.check(make_request(), state, risk_unit=10.0, candidate_open_risk=1.0)

        assert result.passed is True
        assert result.decision is RiskDecisionType.ALLOW

    @pytest.mark.parametrize("bad_risk_unit", [0.0, -1.0, math.nan, math.inf])
    def test_invalid_risk_unit_is_denied(self, bad_risk_unit: float) -> None:
        guard = StrategyRiskGuard(StrategyRiskConfig())

        result = guard.check(
            make_request(),
            make_state(),
            risk_unit=bad_risk_unit,
            candidate_open_risk=1.0,
        )

        assert_denied(result, decision=RiskDecisionType.DENY)

    @pytest.mark.parametrize("bad_open_risk", [-1.0, math.nan, math.inf])
    def test_invalid_candidate_open_risk_is_denied(self, bad_open_risk: float) -> None:
        guard = StrategyRiskGuard(StrategyRiskConfig())

        result = guard.check(
            make_request(),
            make_state(),
            risk_unit=10.0,
            candidate_open_risk=bad_open_risk,
        )

        assert_denied(result, decision=RiskDecisionType.DENY)

    def test_disabled_strategy_is_denied(self) -> None:
        guard = StrategyRiskGuard(StrategyRiskConfig())
        state = make_state()
        state.get_strategy_state(TEST_STRATEGY).disable(reason="strategy disabled test")

        result = guard.check(make_request(), state, risk_unit=10.0, candidate_open_risk=1.0)

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=RiskViolationType.STRATEGY_DISABLED,
        )

    def test_strategy_cooldown_reduces_risk_not_hard_deny(self) -> None:
        guard = StrategyRiskGuard(StrategyRiskConfig())
        state = make_state()
        state.get_strategy_state(TEST_STRATEGY).activate_cooldown(
            cooldown_seconds=60.0,
            reason="strategy cooldown test",
        )

        result = guard.check(make_request(), state, risk_unit=10.0, candidate_open_risk=1.0)

        assert result.passed is True
        assert result.decision is RiskDecisionType.REDUCE_RISK
        assert RiskViolationType.STRATEGY_COOLDOWN_ACTIVE in violation_types(result)

    def test_strategy_daily_loss_breach_is_denied(self) -> None:
        config = StrategyRiskConfig(default_daily_loss_budget_r=2.0)
        guard = StrategyRiskGuard(config)
        state = make_state()
        state.get_strategy_state(TEST_STRATEGY).daily_pnl = -20.0

        result = guard.check(make_request(), state, risk_unit=10.0, candidate_open_risk=1.0)

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=RiskViolationType.STRATEGY_DAILY_LOSS_EXCEEDED,
        )

    def test_strategy_open_risk_breach_is_denied(self) -> None:
        config = StrategyRiskConfig(default_open_risk_budget_r=2.0)
        guard = StrategyRiskGuard(config)
        state = make_state()
        state.get_strategy_state(TEST_STRATEGY).open_risk = 15.0

        result = guard.check(make_request(), state, risk_unit=10.0, candidate_open_risk=6.0)

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=RiskViolationType.STRATEGY_OPEN_RISK_EXCEEDED,
        )

    def test_negative_rolling_expectancy_can_disable_strategy(self) -> None:
        config = StrategyRiskConfig(
            rolling_expectancy_window=3,
            disable_on_negative_expectancy=True,
            disable_when_expectancy_below=-0.05,
        )
        guard = StrategyRiskGuard(config)
        state = make_state()
        strategy_state = state.get_strategy_state(TEST_STRATEGY)

        strategy_state.rolling_pnls.extend([-1.0, -1.0, -1.0])

        result = guard.check(make_request(), state, risk_unit=10.0, candidate_open_risk=1.0)

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=RiskViolationType.STRATEGY_EXPECTANCY_NEGATIVE,
        )

    def test_weak_but_not_disable_level_expectancy_reduces_risk(self) -> None:
        config = StrategyRiskConfig(
            rolling_expectancy_window=3,
            reduce_when_expectancy_below=0.0,
            disable_when_expectancy_below=-10.0,
            reduced_risk_multiplier=0.5,
        )
        guard = StrategyRiskGuard(config)
        state = make_state()
        strategy_state = state.get_strategy_state(TEST_STRATEGY)
        strategy_state.rolling_pnls.extend([-0.01, -0.01, 0.0])

        result = guard.check(make_request(), state, risk_unit=10.0, candidate_open_risk=1.0)

        assert result.passed is True
        assert result.decision is RiskDecisionType.REDUCE_RISK
        assert result.metadata["suggested_multiplier"] == pytest.approx(0.5)

    def test_pending_reservations_are_counted_for_strategy_open_risk(self) -> None:
        config = StrategyRiskConfig(default_open_risk_budget_r=1.0)
        guard = StrategyRiskGuard(config)
        state = make_state()
        add_reservation(
            state,
            make_reservation(
                reservation_id="pending-strategy-1",
                strategy_name=TEST_STRATEGY,
                open_risk=9.0,
            ),
        )

        result = guard.check(make_request(), state, risk_unit=10.0, candidate_open_risk=2.0)

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=RiskViolationType.STRATEGY_OPEN_RISK_EXCEEDED,
        )
        assert result.metadata["pending_open_risk"] == pytest.approx(9.0)

    def test_reduce_order_bypasses_strategy_limits(self) -> None:
        config = StrategyRiskConfig(default_open_risk_budget_r=0.1)
        guard = StrategyRiskGuard(config)
        state = make_state()
        state.get_strategy_state(TEST_STRATEGY).disable(reason="still should allow close")
        request = make_request(order_intent=OrderIntent.CLOSE)

        result = guard.check(request, state, risk_unit=10.0, candidate_open_risk=999.0)

        assert result.passed is True
        assert result.decision is RiskDecisionType.ALLOW
        assert result.metadata["reduce_order_allowed"] is True


# =============================================================================
# ExposureControl
# =============================================================================


class TestExposureControl:
    def test_clean_portfolio_allows_candidate(self) -> None:
        config = ExposureConfig(
            max_open_risk_r=6.0,
            max_used_margin_pct=0.25,
            max_total_exposure_pct=1.0,
            max_symbol_exposure_pct=0.3,
            max_side_exposure_pct=0.6,
        )
        guard = ExposureControl(config)
        state = make_state(equity=10_000.0)
        before = snapshot_state_for_readonly_assertion(state)

        result = guard.check(
            make_request(),
            state,
            candidate_size=1.0,
            candidate_open_risk=10.0,
            candidate_leverage=5.0,
            risk_unit=10.0,
        )

        assert result.passed is True
        assert result.decision is RiskDecisionType.ALLOW
        assert snapshot_state_for_readonly_assertion(state) == before

    @pytest.mark.parametrize(
        ("kwargs", "expected_reason"),
        [
            ({"candidate_size": 0.0}, "Invalid candidate size"),
            ({"candidate_size": -1.0}, "Invalid candidate size"),
            ({"candidate_size": math.nan}, "Invalid candidate size"),
            ({"candidate_open_risk": -1.0}, "Invalid candidate open risk"),
            ({"candidate_open_risk": math.nan}, "Invalid candidate open risk"),
            ({"candidate_leverage": 0.0}, "Invalid candidate leverage"),
            ({"candidate_leverage": math.nan}, "Invalid candidate leverage"),
            ({"risk_unit": 0.0}, "Invalid risk unit"),
            ({"risk_unit": math.nan}, "Invalid risk unit"),
        ],
    )
    def test_invalid_exposure_inputs_are_denied(
        self,
        kwargs: dict[str, float],
        expected_reason: str,
    ) -> None:
        guard = ExposureControl(ExposureConfig())
        state = make_state()

        params = {
            "candidate_size": 1.0,
            "candidate_open_risk": 1.0,
            "candidate_leverage": 5.0,
            "risk_unit": 10.0,
        }
        params.update(kwargs)

        result = guard.check(make_request(), state, **params)

        assert_denied(result, decision=RiskDecisionType.DENY)
        assert result.reason == expected_reason

    def test_invalid_state_equity_is_denied(self) -> None:
        guard = ExposureControl(ExposureConfig())
        state = make_state()
        state.equity = 0.0

        result = guard.check(
            make_request(),
            state,
            candidate_size=1.0,
            candidate_open_risk=1.0,
            candidate_leverage=5.0,
            risk_unit=10.0,
        )

        assert_denied(result, decision=RiskDecisionType.DENY)
        assert result.reason == "Invalid state equity"

    def test_open_risk_limit_is_denied(self) -> None:
        config = ExposureConfig(max_open_risk_r=1.0)
        guard = ExposureControl(config)
        state = make_state(equity=10_000.0)

        result = guard.check(
            make_request(),
            state,
            candidate_size=1.0,
            candidate_open_risk=20.0,
            candidate_leverage=5.0,
            risk_unit=10.0,
        )

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=RiskViolationType.OPEN_RISK_EXCEEDED,
        )

    def test_used_margin_limit_is_denied(self) -> None:
        config = ExposureConfig(max_used_margin_pct=0.01)
        guard = ExposureControl(config)
        state = make_state(equity=10_000.0)
        request = make_request(entry_price=100.0)

        result = guard.check(
            request,
            state,
            candidate_size=100.0,
            candidate_open_risk=1.0,
            candidate_leverage=1.0,
            risk_unit=10.0,
        )

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=RiskViolationType.USED_MARGIN_EXCEEDED,
        )

    def test_total_exposure_limit_is_denied(self) -> None:
        config = ExposureConfig(max_total_exposure_pct=0.01)
        guard = ExposureControl(config)
        state = make_state(equity=10_000.0)
        request = make_request(entry_price=100.0)

        result = guard.check(
            request,
            state,
            candidate_size=2.0,
            candidate_open_risk=1.0,
            candidate_leverage=10.0,
            risk_unit=10.0,
        )

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=RiskViolationType.MAX_EXPOSURE_EXCEEDED,
        )

    def test_symbol_exposure_limit_is_denied(self) -> None:
        config = ExposureConfig(max_symbol_exposure_pct=0.01)
        guard = ExposureControl(config)
        state = make_state(equity=10_000.0)
        add_position(
            state,
            make_position(
                symbol=TEST_SYMBOL,
                notional_value=90.0,
                margin_used=10.0,
                risk_amount=1.0,
            ),
        )

        result = guard.check(
            make_request(symbol=TEST_SYMBOL),
            state,
            candidate_size=1.0,
            candidate_open_risk=1.0,
            candidate_leverage=10.0,
            risk_unit=10.0,
        )

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=RiskViolationType.MAX_SYMBOL_EXPOSURE_EXCEEDED,
        )

    def test_side_exposure_limit_is_denied(self) -> None:
        config = ExposureConfig(max_side_exposure_pct=0.01)
        guard = ExposureControl(config)
        state = make_state(equity=10_000.0)
        add_position(
            state,
            make_position(
                symbol=ALT_SYMBOL,
                side=PositionSide.LONG,
                notional_value=90.0,
                margin_used=10.0,
                risk_amount=1.0,
            ),
        )

        result = guard.check(
            make_request(symbol=TEST_SYMBOL, side=PositionSide.LONG),
            state,
            candidate_size=1.0,
            candidate_open_risk=1.0,
            candidate_leverage=10.0,
            risk_unit=10.0,
        )

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=RiskViolationType.MAX_SIDE_EXPOSURE_EXCEEDED,
        )

    def test_max_open_positions_limit_is_denied(self) -> None:
        config = ExposureConfig(max_open_positions=1)
        guard = ExposureControl(config)
        state = make_state(equity=10_000.0)
        add_position(
            state,
            make_position(
                symbol=ALT_SYMBOL,
                position_id="existing-position",
                risk_amount=1.0,
                margin_used=10.0,
                notional_value=100.0,
            ),
        )

        result = guard.check(
            make_request(symbol=TEST_SYMBOL),
            state,
            candidate_size=1.0,
            candidate_open_risk=1.0,
            candidate_leverage=10.0,
            risk_unit=10.0,
        )

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=RiskViolationType.MAX_OPEN_POSITIONS_EXCEEDED,
        )

    def test_pending_reservations_are_included_in_exposure_projection(self) -> None:
        config = ExposureConfig(max_open_risk_r=1.0)
        guard = ExposureControl(config)
        state = make_state(equity=10_000.0)
        add_reservation(
            state,
            make_reservation(
                reservation_id="pending-exposure-1",
                open_risk=9.0,
                margin=10.0,
                notional=100.0,
            ),
        )

        result = guard.check(
            make_request(),
            state,
            candidate_size=1.0,
            candidate_open_risk=2.0,
            candidate_leverage=10.0,
            risk_unit=10.0,
        )

        assert_denied(
            result,
            decision=RiskDecisionType.DENY,
            violation_type=RiskViolationType.OPEN_RISK_EXCEEDED,
        )
        assert result.metadata["pending_reservations_count"] == 1
        assert result.metadata["projected_open_risk"] >= 11.0

    def test_reduce_order_bypasses_exposure_limits(self) -> None:
        config = ExposureConfig(
            max_open_risk_r=0.01,
            max_used_margin_pct=0.01,
            max_total_exposure_pct=0.01,
            max_symbol_exposure_pct=0.01,
            max_side_exposure_pct=0.01,
            max_open_positions=0,
        )
        guard = ExposureControl(config)
        state = make_state(equity=10_000.0)
        request = make_request(order_intent=OrderIntent.CLOSE)

        result = guard.check(
            request,
            state,
            candidate_size=999.0,
            candidate_open_risk=999.0,
            candidate_leverage=1.0,
            risk_unit=10.0,
        )

        assert result.passed is True
        assert result.decision is RiskDecisionType.ALLOW
        assert result.metadata["reduce_order_allowed"] is True

    def test_exposure_guard_is_read_only_even_when_denied(self) -> None:
        config = ExposureConfig(max_open_risk_r=0.1)
        guard = ExposureControl(config)
        state = make_state(equity=10_000.0)
        add_reservation(
            state,
            make_reservation(
                reservation_id="readonly-pending",
                open_risk=1.0,
                margin=1.0,
                notional=10.0,
            ),
        )
        before = snapshot_state_for_readonly_assertion(state)

        result = guard.check(
            make_request(),
            state,
            candidate_size=1.0,
            candidate_open_risk=100.0,
            candidate_leverage=5.0,
            risk_unit=10.0,
        )

        assert result.passed is False
        assert snapshot_state_for_readonly_assertion(state) == before