# tests/risk/test_risk_state_metrics_circuit_breaker.py
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import pytest

from risk.circuit_breaker import CircuitBreaker, CircuitBreakerStats
from risk.config import CircuitBreakerConfig
from risk.enums import (
    CircuitBreakerReason,
    PositionSide,
    RiskDecisionType,
    RiskLevel,
    RiskMode,
    RiskViolationType,
    StrategyRiskStatus,
    SymbolRiskStatus,
    TradeTier,
    TradingMode,
)
from risk.metrics import GroupMetrics, MetricStats, ReservationMetrics, RiskMetrics
from risk.models import PortfolioPosition, RiskDecision, RiskViolation
from risk.state import (
    CooldownState,
    PendingRiskReservation,
    RiskState,
    StrategyRiskState,
    SymbolRiskState,
    TierRuntimeStats,
)


TEST_SYMBOL = "BTCUSDT"
ALT_SYMBOL = "ETHUSDT"
TEST_STRATEGY = "test_strategy"
ALT_STRATEGY = "alt_strategy"


# =============================================================================
# Helpers
# =============================================================================


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))

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
    signal_id: str | None = "signal-test",
    position_id: str | None = "position-test",
    realized_pnl: float = 0.0,
    unrealized_pnl: float = 0.0,
) -> PortfolioPosition:
    mark_price = mark_price if mark_price is not None else entry_price
    notional_value = (
        notional_value if notional_value is not None else abs(size * entry_price)
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
    signal_id: str | None = "signal-reservation",
    strategy_name: str | None = TEST_STRATEGY,
    tier: TradeTier | None = TradeTier.T2,
    position_id: str | None = None,
    size: float = 1.0,
    open_risk: float = 10.0,
    margin: float = 20.0,
    notional: float = 100.0,
    created_at: float | None = None,
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
        created_at=created_at if created_at is not None else time.time(),
        expires_at=expires_at,
    )


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
    )


def make_decision(
    *,
    allowed: bool = True,
    decision: RiskDecisionType = RiskDecisionType.ALLOW,
    symbol: str | None = TEST_SYMBOL,
    strategy_name: str | None = TEST_STRATEGY,
    tier: TradeTier | None = TradeTier.T2,
    reason: str | None = "test decision",
    violations: list[RiskViolation] | None = None,
    rr: float | None = 2.5,
    ev: float | None = 0.5,
    ev_after_cost: float | None = 0.4,
    cost_to_reward: float | None = 0.05,
    reservation_id: str | None = None,
) -> RiskDecision:
    return RiskDecision(
        allowed=allowed,
        decision=decision,
        final_size=1.0 if allowed else None,
        final_leverage=5.0 if allowed else None,
        reason=reason,
        final_tier=tier,
        final_risk_amount=10.0 if allowed else None,
        final_margin=20.0 if allowed else None,
        final_notional=100.0 if allowed else None,
        reservation_id=reservation_id,
        risk_mode=RiskMode.NORMAL,
        risk_reward_ratio=rr,
        expected_value=ev,
        expected_value_after_cost=ev_after_cost,
        expected_cost=0.1,
        cost_to_reward_ratio=cost_to_reward,
        signal_id="signal-decision",
        strategy_name=strategy_name,
        symbol=symbol,
        side=PositionSide.LONG,
        violations=list(violations or []),
        metadata={"test": True},
    )


def assert_non_negative_metric_fields(snapshot: dict[str, Any], fields: list[str]) -> None:
    for field_name in fields:
        assert snapshot[field_name] >= 0, f"{field_name} went negative: {snapshot}"


# =============================================================================
# CooldownState
# =============================================================================


class TestCooldownState:
    def test_activate_sets_active_reason_and_deadline(self) -> None:
        cooldown = CooldownState()
        now_ts = 1_000.0

        cooldown.activate(cooldown_seconds=60.0, reason="test", now_ts=now_ts)

        assert cooldown.active is True
        assert cooldown.reason == "test"
        assert cooldown.started_at == pytest.approx(now_ts)
        assert cooldown.cooldown_until == pytest.approx(now_ts + 60.0)

    def test_negative_cooldown_seconds_expire_immediately(self) -> None:
        cooldown = CooldownState()
        now_ts = 1_000.0

        cooldown.activate(cooldown_seconds=-10.0, reason="negative", now_ts=now_ts)

        assert cooldown.cooldown_until == pytest.approx(now_ts)
        assert cooldown.has_expired(now_ts=now_ts) is True

    def test_is_active_is_side_effect_free_even_after_expiry(self) -> None:
        cooldown = CooldownState()
        cooldown.activate(cooldown_seconds=10.0, reason="expires", now_ts=1_000.0)

        assert cooldown.is_active(now_ts=1_011.0) is False

        assert cooldown.active is True
        assert cooldown.reason == "expires"
        assert cooldown.cooldown_until == pytest.approx(1_010.0)

    def test_expire_if_needed_is_mutating_counterpart(self) -> None:
        cooldown = CooldownState()
        cooldown.activate(cooldown_seconds=10.0, reason="expires", now_ts=1_000.0)

        assert cooldown.expire_if_needed(now_ts=1_009.0) is False
        assert cooldown.active is True

        assert cooldown.expire_if_needed(now_ts=1_010.0) is True
        assert cooldown.active is False
        assert cooldown.reason is None
        assert cooldown.started_at is None
        assert cooldown.cooldown_until is None

    def test_deactivate_is_idempotent(self) -> None:
        cooldown = CooldownState()
        cooldown.activate(cooldown_seconds=60.0, reason="test", now_ts=1_000.0)

        cooldown.deactivate()
        cooldown.deactivate()

        assert cooldown.active is False
        assert cooldown.reason is None
        assert cooldown.started_at is None
        assert cooldown.cooldown_until is None


# =============================================================================
# SymbolRiskState / StrategyRiskState
# =============================================================================


class TestSymbolRiskState:
    def test_trade_opened_increments_trade_count_and_open_risk(self) -> None:
        state = SymbolRiskState(symbol=TEST_SYMBOL)

        state.register_trade_opened(open_risk=10.0)
        state.register_trade_opened(open_risk=-999.0)

        assert state.trades_today == 2
        assert state.open_risk == pytest.approx(10.0)

    def test_trade_closed_updates_pnl_and_never_negative_open_risk(self) -> None:
        state = SymbolRiskState(symbol=TEST_SYMBOL)
        state.open_risk = 10.0

        state.register_trade_closed(realized_pnl=-5.0, released_risk=999.0)

        assert state.daily_pnl == pytest.approx(-5.0)
        assert state.weekly_pnl == pytest.approx(-5.0)
        assert state.monthly_pnl == pytest.approx(-5.0)
        assert state.open_risk == pytest.approx(0.0)
        assert state.consecutive_losses == 1

        state.register_trade_closed(realized_pnl=1.0, released_risk=1.0)
        assert state.consecutive_losses == 0

    def test_disable_reduce_activate_and_refresh_status(self) -> None:
        state = SymbolRiskState(symbol=TEST_SYMBOL)

        state.disable(reason="disabled", until=1_100.0)
        assert state.status is SymbolRiskStatus.DISABLED
        assert state.disabled_reason == "disabled"

        state.refresh_status(now_ts=1_099.0)
        assert state.status is SymbolRiskStatus.DISABLED

        state.refresh_status(now_ts=1_100.0)
        assert state.status is SymbolRiskStatus.ACTIVE
        assert state.disabled_reason is None

        state.reduce(reason="reduce")
        assert state.status is SymbolRiskStatus.REDUCED
        assert state.metadata["reduced_reason"] == "reduce"

        state.activate()
        assert state.status is SymbolRiskStatus.ACTIVE

    def test_cooldown_refresh_reactivates_after_expiry(self) -> None:
        state = SymbolRiskState(symbol=TEST_SYMBOL)
        state.cooldown.activate(cooldown_seconds=10.0, reason="cooldown", now_ts=1_000.0)
        state.status = SymbolRiskStatus.COOLDOWN

        state.refresh_status(now_ts=1_009.0)
        assert state.status is SymbolRiskStatus.COOLDOWN

        state.refresh_status(now_ts=1_011.0)
        assert state.status is SymbolRiskStatus.ACTIVE
        assert state.cooldown.active is False

    def test_reset_methods_only_reset_their_own_period(self) -> None:
        state = SymbolRiskState(symbol=TEST_SYMBOL)
        state.daily_pnl = -1.0
        state.weekly_pnl = -2.0
        state.monthly_pnl = -3.0
        state.trades_today = 5
        state.consecutive_losses = 4

        state.reset_daily()
        assert state.daily_pnl == 0.0
        assert state.weekly_pnl == -2.0
        assert state.monthly_pnl == -3.0
        assert state.trades_today == 0
        assert state.consecutive_losses == 0

        state.reset_weekly()
        assert state.weekly_pnl == 0.0
        assert state.monthly_pnl == -3.0

        state.reset_monthly()
        assert state.monthly_pnl == 0.0

    def test_snapshot_does_not_mutate_expired_cooldown(self) -> None:
        state = SymbolRiskState(symbol=TEST_SYMBOL)
        state.status = SymbolRiskStatus.COOLDOWN
        state.cooldown.activate(cooldown_seconds=1.0, reason="expired", now_ts=1.0)

        snapshot = state.snapshot(risk_unit=10.0)

        assert snapshot.status is SymbolRiskStatus.ACTIVE
        assert state.status is SymbolRiskStatus.COOLDOWN
        assert state.cooldown.active is True
        assert_json_serializable(snapshot)


class TestStrategyRiskState:
    def test_trade_closed_updates_rolling_expectancy_and_trims_window(self) -> None:
        state = StrategyRiskState(strategy_name=TEST_STRATEGY)
        state.open_risk = 30.0

        state.register_trade_closed(realized_pnl=-3.0, released_risk=10.0, rolling_window=3)
        state.register_trade_closed(realized_pnl=6.0, released_risk=10.0, rolling_window=3)
        state.register_trade_closed(realized_pnl=-9.0, released_risk=10.0, rolling_window=3)
        state.register_trade_closed(realized_pnl=12.0, released_risk=10.0, rolling_window=3)

        assert state.open_risk == pytest.approx(0.0)
        assert state.rolling_pnls == [6.0, -9.0, 12.0]
        assert state.rolling_expectancy == pytest.approx(3.0)
        assert state.consecutive_losses == 0

    def test_disable_sets_multiplier_to_zero_and_activate_restores_it(self) -> None:
        state = StrategyRiskState(strategy_name=TEST_STRATEGY)

        state.disable(reason="bad expectancy", until=1_100.0)

        assert state.status is StrategyRiskStatus.DISABLED
        assert state.risk_multiplier == pytest.approx(0.0)

        state.refresh_status(now_ts=1_100.0)

        assert state.status is StrategyRiskStatus.ACTIVE
        assert state.risk_multiplier == pytest.approx(1.0)

    def test_reduce_clamps_negative_multiplier_to_zero(self) -> None:
        state = StrategyRiskState(strategy_name=TEST_STRATEGY)

        state.reduce(multiplier=-5.0, reason="bad")

        assert state.status is StrategyRiskStatus.REDUCED
        assert state.risk_multiplier == pytest.approx(0.0)
        assert state.metadata["reduced_reason"] == "bad"

    def test_strategy_snapshot_is_side_effect_free_for_expired_cooldown(self) -> None:
        state = StrategyRiskState(strategy_name=TEST_STRATEGY)
        state.status = StrategyRiskStatus.COOLDOWN
        state.cooldown.activate(cooldown_seconds=1.0, reason="expired", now_ts=1.0)

        snapshot = state.snapshot(risk_unit=10.0)

        assert snapshot.status is StrategyRiskStatus.ACTIVE
        assert state.status is StrategyRiskStatus.COOLDOWN
        assert state.cooldown.active is True
        assert_json_serializable(snapshot)


# =============================================================================
# TierRuntimeStats
# =============================================================================


class TestTierRuntimeStats:
    def test_register_approval_rejection_and_close_never_negative_open_risk(self) -> None:
        stats = TierRuntimeStats(tier=TradeTier.T2)

        stats.register_approval(open_risk=10.0, rr=2.0, expected_value=0.5, cost_to_reward=0.1)
        stats.register_rejection()
        stats.register_close(realized_pnl=-5.0, released_risk=999.0)

        assert stats.trades == 1
        assert stats.approvals == 1
        assert stats.rejections == 1
        assert stats.realized_pnl == pytest.approx(-5.0)
        assert stats.open_risk == pytest.approx(0.0)

        snapshot = stats.snapshot()
        assert snapshot.avg_rr == pytest.approx(2.0)
        assert snapshot.avg_expected_value == pytest.approx(0.5)
        assert snapshot.avg_cost_to_reward == pytest.approx(0.1)
        assert_json_serializable(snapshot)


# =============================================================================
# PendingRiskReservation
# =============================================================================


class TestPendingRiskReservation:
    def test_expiration_logic_and_snapshot_are_stable(self) -> None:
        reservation = make_reservation(
            reservation_id="r1",
            expires_at=1_010.0,
        )

        assert reservation.is_expired(now_ts=1_009.0) is False
        assert reservation.is_expired(now_ts=1_010.0) is True

        snapshot = reservation.snapshot()
        assert snapshot["reservation_id"] == "r1"
        assert snapshot["side"] == PositionSide.LONG.value
        assert snapshot["tier"] == TradeTier.T2.value
        assert_json_serializable(snapshot)

    def test_no_expiry_means_not_expired(self) -> None:
        reservation = make_reservation(expires_at=None)

        assert reservation.is_expired(now_ts=999_999.0) is False


# =============================================================================
# RiskState account/mode/accounting
# =============================================================================


class TestRiskStateAccountAndModes:
    def test_update_account_initializes_anchors_and_peak_equity(self) -> None:
        state = RiskState()

        state.update_account(
            balance=10_000.0,
            equity=10_000.0,
            free_balance=9_000.0,
            used_margin=1_000.0,
            realized_pnl=10.0,
            unrealized_pnl=-5.0,
        )

        assert state.balance == pytest.approx(10_000.0)
        assert state.equity == pytest.approx(10_000.0)
        assert state.free_balance == pytest.approx(9_000.0)
        assert state.used_margin == pytest.approx(1_000.0)
        assert state.realized_pnl == pytest.approx(10.0)
        assert state.unrealized_pnl == pytest.approx(-5.0)
        assert state.peak_equity == pytest.approx(10_000.0)
        assert state.daily_start_equity == pytest.approx(10_000.0)
        assert state.weekly_start_equity == pytest.approx(10_000.0)
        assert state.monthly_start_equity == pytest.approx(10_000.0)

        state.update_account(equity=11_000.0)
        assert state.peak_equity == pytest.approx(11_000.0)

        state.update_account(equity=9_000.0)
        assert state.peak_equity == pytest.approx(11_000.0)

    @pytest.mark.parametrize(
        ("mode", "halted"),
        [
            (RiskMode.NORMAL, False),
            (RiskMode.CAUTION, False),
            (RiskMode.SAFE_MODE, False),
            (RiskMode.REDUCE_ONLY, False),
            (RiskMode.HALTED, True),
            (RiskMode.EMERGENCY_STOP, True),
        ],
    )
    def test_set_risk_mode_updates_halt_flags_without_assuming_trading_mode_mirror_enum(
            self,
            mode: RiskMode,
            halted: bool,
    ) -> None:
        state = make_state()

        state.set_risk_mode(mode, reason="mode test")

        assert state.risk_mode is mode
        assert state.trading_halted is halted

        if halted:
            assert state.halt_reason == "mode test"
            assert state.trading_mode in {
                TradingMode.HALTED,
                TradingMode.EMERGENCY_STOP,
            }
        else:
            assert state.halt_reason is None
            assert state.trading_mode is not TradingMode.HALTED
            assert state.trading_mode is not TradingMode.EMERGENCY_STOP

    def test_resume_trading_does_not_clear_emergency_stop(self) -> None:
        state = make_state()
        state.emergency_stop("emergency")

        state.resume_trading()

        assert state.emergency_stop_active is True
        assert state.trading_halted is True
        assert state.risk_mode is RiskMode.EMERGENCY_STOP

    def test_clear_emergency_stop_allows_resume_to_normal(self) -> None:
        state = make_state()
        state.emergency_stop("emergency")

        state.clear_emergency_stop()
        state.resume_trading()

        assert state.emergency_stop_active is False
        assert state.trading_halted is False
        assert state.risk_mode is RiskMode.NORMAL

    def test_disable_protection_modes_does_not_override_emergency_stop(self) -> None:
        state = make_state()
        state.emergency_stop("emergency")

        state.disable_protection_modes()

        assert state.emergency_stop_active is True
        assert state.risk_mode is RiskMode.EMERGENCY_STOP
        assert state.trading_halted is True


class TestRiskStatePositionAccounting:
    def test_add_position_updates_symbol_strategy_and_tier_accounting(self) -> None:
        state = make_state()
        position = make_position(
            symbol=TEST_SYMBOL,
            strategy_name=TEST_STRATEGY,
            tier=TradeTier.T2,
            risk_amount=10.0,
        )

        state.add_position(position)

        assert len(state.positions) == 1
        assert state.get_symbol_state(TEST_SYMBOL).trades_today == 1
        assert state.get_symbol_state(TEST_SYMBOL).open_risk == pytest.approx(10.0)
        assert state.get_strategy_state(TEST_STRATEGY).trades_today == 1
        assert state.get_strategy_state(TEST_STRATEGY).open_risk == pytest.approx(10.0)
        assert state.get_tier_stats(TradeTier.T2).approvals == 1
        assert state.get_tier_stats(TradeTier.T2).open_risk == pytest.approx(10.0)

    def test_update_position_changes_only_existing_position(self) -> None:
        state = make_state()
        position = make_position(position_id="p1")
        state.add_position(position)

        state.update_position(
            TEST_SYMBOL,
            position_id="p1",
            size=2.0,
            mark_price=110.0,
            notional_value=220.0,
            leverage=10.0,
            margin_used=22.0,
            risk_amount=12.0,
            stop_loss=98.0,
            take_profit=120.0,
            unrealized_pnl=20.0,
        )

        key = next(iter(state.positions))
        updated = state.positions[key]
        assert updated.size == pytest.approx(2.0)
        assert updated.mark_price == pytest.approx(110.0)
        assert updated.notional_value == pytest.approx(220.0)
        assert updated.leverage == pytest.approx(10.0)
        assert updated.margin_used == pytest.approx(22.0)
        assert updated.risk_amount == pytest.approx(12.0)
        assert updated.unrealized_pnl == pytest.approx(20.0)

        state.update_position("UNKNOWN", position_id="missing", size=999.0)
        assert len(state.positions) == 1

    def test_remove_position_releases_risk_updates_pnl_and_never_negative_accounting(self) -> None:
        state = make_state()
        position = make_position(
            position_id="p1",
            risk_amount=10.0,
            symbol=TEST_SYMBOL,
            strategy_name=TEST_STRATEGY,
            tier=TradeTier.T2,
        )
        state.add_position(position)

        removed = state.remove_position(TEST_SYMBOL, position_id="p1", realized_pnl=-15.0)

        assert removed is not None
        assert removed.realized_pnl == pytest.approx(-15.0)
        assert not state.positions
        assert state.realized_pnl == pytest.approx(-15.0)
        assert state.loss_streak == 1
        assert state.get_symbol_state(TEST_SYMBOL).daily_pnl == pytest.approx(-15.0)
        assert state.get_symbol_state(TEST_SYMBOL).open_risk == pytest.approx(0.0)
        assert state.get_strategy_state(TEST_STRATEGY).daily_pnl == pytest.approx(-15.0)
        assert state.get_strategy_state(TEST_STRATEGY).open_risk == pytest.approx(0.0)
        assert state.get_tier_stats(TradeTier.T2).open_risk == pytest.approx(0.0)

        missing = state.remove_position(TEST_SYMBOL, position_id="p1", realized_pnl=999.0)
        assert missing is None
        assert state.realized_pnl == pytest.approx(-15.0)

    def test_register_trade_outcome_loss_streak_resets_on_profit(self) -> None:
        state = make_state()

        state.register_trade_outcome(-1.0)
        state.register_trade_outcome(-2.0)
        assert state.loss_streak == 2

        state.register_trade_outcome(1.0)
        assert state.loss_streak == 0

    def test_register_rejection_updates_reason_and_group_metadata(self) -> None:
        state = make_state()

        state.register_rejection(
            reason="bad rr",
            symbol=TEST_SYMBOL,
            strategy_name=TEST_STRATEGY,
            tier=TradeTier.T2,
        )

        assert state.last_rejected_reason == "bad rr"
        assert state.get_symbol_state(TEST_SYMBOL).metadata["last_rejection_reason"] == "bad rr"
        assert (
            state.get_strategy_state(TEST_STRATEGY).metadata["last_rejection_reason"]
            == "bad rr"
        )
        assert state.get_tier_stats(TradeTier.T2).rejections == 1


class TestRiskStateReservations:
    def test_reserve_get_release_and_confirm_reservation(self) -> None:
        state = make_state()

        reservation = state.reserve_risk(
            reservation_id="r1",
            symbol=TEST_SYMBOL,
            side=PositionSide.LONG,
            signal_id="s1",
            strategy_name=TEST_STRATEGY,
            tier=TradeTier.T2,
            size=1.0,
            open_risk=10.0,
            margin=20.0,
            notional=100.0,
            ttl_seconds=30.0,
            now_ts=1_000.0,
        )

        assert reservation.reservation_id == "r1"
        assert reservation.expires_at == pytest.approx(1_030.0)
        assert state.get_pending_reservation("r1") is reservation
        assert state.get_pending_reservation(signal_id="s1") is reservation
        assert state.get_pending_reservation(symbol=TEST_SYMBOL) is reservation
        assert state.get_pending_open_risk() == pytest.approx(10.0)
        assert state.get_pending_margin() == pytest.approx(20.0)
        assert state.get_pending_notional() == pytest.approx(100.0)

        released = state.release_risk_reservation("r1")
        assert released is reservation
        assert state.get_pending_reservation("r1") is None

        assert state.release_risk_reservation("r1") is None

        reservation2 = state.reserve_risk(
            reservation_id="r2",
            symbol=TEST_SYMBOL,
            side=PositionSide.LONG,
            signal_id="s2",
            open_risk=5.0,
            ttl_seconds=None,
        )
        confirmed = state.confirm_risk_reservation(signal_id="s2")
        assert confirmed is reservation2
        assert state.get_pending_open_risk() == pytest.approx(0.0)

    def test_reservation_negative_amounts_are_clamped_or_absed(self) -> None:
        state = make_state()

        reservation = state.reserve_risk(
            reservation_id="negative",
            symbol=TEST_SYMBOL,
            side=PositionSide.LONG,
            size=-1.0,
            open_risk=-10.0,
            margin=-20.0,
            notional=-100.0,
            ttl_seconds=1.0,
        )

        assert reservation.size == pytest.approx(0.0)
        assert reservation.open_risk == pytest.approx(0.0)
        assert reservation.margin == pytest.approx(0.0)
        assert reservation.notional == pytest.approx(100.0)

    def test_expire_pending_reservations_removes_only_expired(self) -> None:
        state = make_state()
        state.reserve_risk(
            reservation_id="expired",
            symbol=TEST_SYMBOL,
            side=PositionSide.LONG,
            open_risk=10.0,
            ttl_seconds=10.0,
            now_ts=1_000.0,
        )
        state.reserve_risk(
            reservation_id="active",
            symbol=ALT_SYMBOL,
            side=PositionSide.SHORT,
            open_risk=20.0,
            ttl_seconds=100.0,
            now_ts=1_000.0,
        )

        expired = state.expire_pending_reservations(now_ts=1_011.0)

        assert [item.reservation_id for item in expired] == ["expired"]
        assert state.get_pending_reservation("expired") is None
        assert state.get_pending_reservation("active") is not None
        assert state.get_pending_open_risk(include_expired=True) == pytest.approx(20.0)

    def test_pending_filters_by_symbol_strategy_tier_and_side(self) -> None:
        state = make_state()
        state.reserve_risk(
            reservation_id="r1",
            symbol=TEST_SYMBOL,
            side=PositionSide.LONG,
            strategy_name=TEST_STRATEGY,
            tier=TradeTier.T2,
            open_risk=10.0,
            margin=20.0,
            notional=100.0,
        )
        state.reserve_risk(
            reservation_id="r2",
            symbol=ALT_SYMBOL,
            side=PositionSide.SHORT,
            strategy_name=ALT_STRATEGY,
            tier=TradeTier.T3,
            open_risk=30.0,
            margin=40.0,
            notional=500.0,
        )

        assert state.get_pending_open_risk(symbol=TEST_SYMBOL) == pytest.approx(10.0)
        assert state.get_pending_open_risk(symbol=ALT_SYMBOL) == pytest.approx(30.0)
        assert state.get_pending_open_risk(strategy_name=TEST_STRATEGY) == pytest.approx(10.0)
        assert state.get_pending_open_risk(strategy_name=ALT_STRATEGY) == pytest.approx(30.0)
        assert state.get_pending_open_risk(tier=TradeTier.T2) == pytest.approx(10.0)
        assert state.get_pending_notional(side=PositionSide.LONG) == pytest.approx(100.0)
        assert state.get_pending_notional(side=PositionSide.SHORT) == pytest.approx(500.0)

    def test_projected_open_risk_includes_actual_pending_and_candidate(self) -> None:
        state = make_state()
        state.add_position(make_position(symbol=TEST_SYMBOL, risk_amount=10.0))
        state.reserve_risk(
            reservation_id="r1",
            symbol=TEST_SYMBOL,
            side=PositionSide.LONG,
            open_risk=5.0,
        )

        projected = state.get_projected_open_risk(
            symbol=TEST_SYMBOL,
            candidate_open_risk=7.0,
        )

        assert projected == pytest.approx(22.0)


class TestRiskStateSnapshotsAndResets:
    def test_open_risk_snapshot_includes_actual_and_pending(self) -> None:
        state = make_state(equity=10_000.0, used_margin=100.0)
        state.add_position(
            make_position(
                symbol=TEST_SYMBOL,
                risk_amount=10.0,
                margin_used=100.0,
                notional_value=1_000.0,
            )
        )
        state.reserve_risk(
            reservation_id="r1",
            symbol=TEST_SYMBOL,
            side=PositionSide.LONG,
            strategy_name=TEST_STRATEGY,
            tier=TradeTier.T2,
            open_risk=5.0,
            margin=50.0,
            notional=500.0,
        )

        snapshot = state.get_open_risk_snapshot(risk_unit=10.0)

        assert snapshot.actual_open_risk == pytest.approx(10.0)
        assert snapshot.pending_orders_risk == pytest.approx(5.0)
        assert snapshot.total_open_risk == pytest.approx(15.0)
        assert snapshot.total_open_risk_r == pytest.approx(1.5)
        assert snapshot.pending_reservations_count == 1
        assert_json_serializable(snapshot)

    def test_exposure_snapshot_includes_pending_notional_and_margin(self) -> None:
        state = make_state(equity=10_000.0)
        state.add_position(
            make_position(
                symbol=TEST_SYMBOL,
                side=PositionSide.LONG,
                notional_value=1_000.0,
                margin_used=200.0,
            )
        )
        state.reserve_risk(
            reservation_id="r1",
            symbol=ALT_SYMBOL,
            side=PositionSide.SHORT,
            open_risk=5.0,
            margin=100.0,
            notional=500.0,
        )

        snapshot = state.get_exposure_snapshot()

        assert snapshot.actual_notional == pytest.approx(1_000.0)
        assert snapshot.pending_notional == pytest.approx(500.0)
        assert snapshot.pending_margin == pytest.approx(100.0)
        assert snapshot.gross_exposure == pytest.approx(1_500.0)
        assert snapshot.margin_used == pytest.approx(300.0)
        assert snapshot.symbol_exposure[TEST_SYMBOL] == pytest.approx(1_000.0)
        assert snapshot.symbol_exposure[ALT_SYMBOL] == pytest.approx(500.0)
        assert_json_serializable(snapshot)

    def test_daily_weekly_monthly_resets_are_scoped(self) -> None:
        state = make_state(equity=10_000.0)
        symbol = state.get_symbol_state(TEST_SYMBOL)
        strategy = state.get_strategy_state(TEST_STRATEGY)

        for item in (symbol, strategy):
            item.daily_pnl = -1.0
            item.weekly_pnl = -2.0
            item.monthly_pnl = -3.0
            item.trades_today = 10
            item.consecutive_losses = 5

        state.loss_streak = 5
        state.manual_review_required = True

        state.reset_daily_state()
        assert state.daily_start_equity == pytest.approx(10_000.0)
        assert state.loss_streak == 0
        assert symbol.daily_pnl == 0.0
        assert strategy.daily_pnl == 0.0
        assert symbol.weekly_pnl == -2.0
        assert strategy.weekly_pnl == -2.0

        state.reset_weekly_state()
        assert symbol.weekly_pnl == 0.0
        assert strategy.weekly_pnl == 0.0
        assert symbol.monthly_pnl == -3.0
        assert strategy.monthly_pnl == -3.0

        state.reset_monthly_state()
        assert symbol.monthly_pnl == 0.0
        assert strategy.monthly_pnl == 0.0
        assert state.manual_review_required is False


# =============================================================================
# MetricStats / ReservationMetrics / GroupMetrics / RiskMetrics
# =============================================================================


class TestMetricStats:
    def test_metric_stats_ignore_none_nan_and_infinity(self) -> None:
        stats = MetricStats()

        for value in [None, math.nan, math.inf, -math.inf]:
            stats.add(value)

        assert stats.count == 0
        assert stats.average is None

        stats.add(2.0)
        stats.add(4.0)

        assert stats.count == 2
        assert stats.total == pytest.approx(6.0)
        assert stats.minimum == pytest.approx(2.0)
        assert stats.maximum == pytest.approx(4.0)
        assert stats.last == pytest.approx(4.0)
        assert stats.average == pytest.approx(3.0)
        assert_json_serializable(stats.snapshot())


class TestReservationMetrics:
    def test_lifecycle_counters_and_amounts_never_go_negative(self) -> None:
        metrics = ReservationMetrics()

        metrics.register_created(
            reservation_id="r1",
            open_risk=10.0,
            margin=20.0,
            notional=100.0,
        )
        assert metrics.created == 1
        assert metrics.active == 1
        assert metrics.peak_active == 1
        assert metrics.reserved_open_risk == pytest.approx(10.0)

        metrics.register_confirmed(
            reservation_id="r1",
            open_risk=999.0,
            margin=999.0,
            notional=999.0,
            age_ms=50.0,
        )
        assert metrics.confirmed == 1
        assert metrics.active == 0
        assert metrics.reserved_open_risk == pytest.approx(0.0)
        assert metrics.reserved_margin == pytest.approx(0.0)
        assert metrics.reserved_notional == pytest.approx(0.0)

        metrics.register_released(open_risk=999.0, margin=999.0, notional=999.0)
        metrics.register_expired(open_risk=999.0, margin=999.0, notional=999.0)
        metrics.register_failed(open_risk=999.0, margin=999.0, notional=999.0)

        snapshot = metrics.snapshot()
        assert_non_negative_metric_fields(
            snapshot,
            [
                "active",
                "reserved_open_risk",
                "reserved_margin",
                "reserved_notional",
                "confirmed_open_risk",
                "released_open_risk",
                "expired_open_risk",
                "failed_open_risk",
            ],
        )
        assert metrics.active == 0
        assert metrics.reservation_age_ms.count == 1
        assert_json_serializable(snapshot)

    def test_rates_do_not_divide_by_zero(self) -> None:
        metrics = ReservationMetrics()

        assert metrics.completion_rate == 0.0
        assert metrics.expiration_rate == 0.0
        assert metrics.release_rate == 0.0
        assert metrics.failure_rate == 0.0

    def test_set_active_snapshot_reconciles_to_non_negative_values(self) -> None:
        metrics = ReservationMetrics()

        metrics.set_active_snapshot(
            active=-100,
            reserved_open_risk=-1.0,
            reserved_margin=-2.0,
            reserved_notional=-3.0,
        )

        assert metrics.active == 0
        assert metrics.reserved_open_risk == pytest.approx(0.0)
        assert metrics.reserved_margin == pytest.approx(0.0)
        assert metrics.reserved_notional == pytest.approx(0.0)


class TestGroupMetrics:
    def test_group_metrics_register_decisions_and_violations(self) -> None:
        group = GroupMetrics()

        allow = make_decision(allowed=True, decision=RiskDecisionType.ALLOW)
        deny = make_decision(
            allowed=False,
            decision=RiskDecisionType.DENY,
            violations=[make_violation(RiskViolationType.RISK_REWARD_TOO_LOW)],
        )
        reduce_size = make_decision(allowed=True, decision=RiskDecisionType.REDUCE_SIZE)
        downgrade = make_decision(allowed=True, decision=RiskDecisionType.DOWNGRADE_TIER)
        reduce_risk = make_decision(allowed=True, decision=RiskDecisionType.REDUCE_RISK)
        halt = make_decision(allowed=False, decision=RiskDecisionType.HALT_TRADING)
        emergency = make_decision(allowed=False, decision=RiskDecisionType.EMERGENCY_STOP)

        for decision in [allow, deny, reduce_size, downgrade, reduce_risk, halt, emergency]:
            group.register_decision(decision)

        assert group.decisions == 7
        assert group.approvals == 4
        assert group.rejections == 3
        assert group.size_adjustments == 1
        assert group.tier_downgrades == 1
        assert group.risk_reductions == 1
        assert group.halts == 1
        assert group.emergency_stops == 1
        assert group.violation_counts[RiskViolationType.RISK_REWARD_TOO_LOW.value] == 1

        group.register_open_risk(10.0)
        group.register_pending_reservation(open_risk=5.0)
        group.release_pending_reservation(open_risk=999.0)
        group.register_closed_pnl(-3.0, released_risk=999.0)

        snapshot = group.snapshot()
        assert snapshot["open_risk"] == pytest.approx(0.0)
        assert snapshot["pending_open_risk"] == pytest.approx(0.0)
        assert snapshot["pending_reservations"] == 0
        assert_json_serializable(snapshot)


class TestRiskMetrics:
    def test_register_decision_updates_global_and_group_metrics(self) -> None:
        metrics = RiskMetrics()

        decisions = [
            make_decision(allowed=True, decision=RiskDecisionType.ALLOW),
            make_decision(allowed=False, decision=RiskDecisionType.DENY),
            make_decision(allowed=True, decision=RiskDecisionType.REDUCE_SIZE),
            make_decision(allowed=True, decision=RiskDecisionType.DOWNGRADE_TIER),
            make_decision(allowed=True, decision=RiskDecisionType.REDUCE_RISK),
            make_decision(allowed=False, decision=RiskDecisionType.HALT_TRADING),
            make_decision(allowed=False, decision=RiskDecisionType.EMERGENCY_STOP),
            make_decision(allowed=True, decision=RiskDecisionType.ONLY_REDUCE),
            make_decision(allowed=False, decision=RiskDecisionType.FORCE_CLOSE),
        ]

        for decision in decisions:
            metrics.register_decision(decision, latency_ms=12.5)

        assert metrics.total_decisions == len(decisions)
        assert metrics.approvals == 5
        assert metrics.rejections == 4
        assert metrics.size_adjustments == 1
        assert metrics.tier_downgrades == 1
        assert metrics.risk_reductions == 1
        assert metrics.halts == 1
        assert metrics.emergency_stops == 1
        assert metrics.only_reduce_decisions == 1
        assert metrics.force_close_requests == 1
        assert metrics.decision_latency_ms.count == len(decisions)

        assert metrics.decisions_by_tier[TradeTier.T2.value].decisions == len(decisions)
        assert metrics.decisions_by_symbol[TEST_SYMBOL].decisions == len(decisions)
        assert metrics.decisions_by_strategy[TEST_STRATEGY].decisions == len(decisions)

        snapshot = metrics.snapshot()
        assert snapshot["approval_rate"] == pytest.approx(metrics.approvals / metrics.total_decisions)
        assert_json_serializable(snapshot)

    def test_register_decision_with_violations_counts_by_type(self) -> None:
        metrics = RiskMetrics()
        violation = make_violation(RiskViolationType.EXECUTION_COST_TOO_HIGH)

        metrics.register_decision(
            make_decision(
                allowed=False,
                decision=RiskDecisionType.DENY,
                violations=[violation],
            )
        )

        assert metrics.violation_counts[RiskViolationType.EXECUTION_COST_TOO_HIGH.value] == 1

    def test_reservation_metric_methods_update_global_and_groups(self) -> None:
        metrics = RiskMetrics()

        metrics.register_reservation_created(
            reservation_id="r1",
            symbol=TEST_SYMBOL,
            tier=TradeTier.T2,
            strategy_name=TEST_STRATEGY,
            open_risk=10.0,
            margin=20.0,
            notional=100.0,
        )
        assert metrics.reservations.created == 1
        assert metrics.reservations.active == 1
        assert metrics.decisions_by_symbol[TEST_SYMBOL].pending_open_risk == pytest.approx(10.0)

        metrics.register_reservation_confirmed(
            reservation_id="r1",
            symbol=TEST_SYMBOL,
            tier=TradeTier.T2,
            strategy_name=TEST_STRATEGY,
            open_risk=10.0,
            margin=20.0,
            notional=100.0,
            age_ms=10.0,
        )
        assert metrics.reservations.confirmed == 1
        assert metrics.reservations.active == 0
        assert metrics.decisions_by_symbol[TEST_SYMBOL].pending_open_risk == pytest.approx(0.0)

        metrics.register_reservation_created(
            reservation_id="r2",
            symbol=TEST_SYMBOL,
            tier=TradeTier.T2,
            strategy_name=TEST_STRATEGY,
            open_risk=5.0,
            margin=10.0,
            notional=50.0,
        )
        metrics.register_reservation_released(
            reservation_id="r2",
            symbol=TEST_SYMBOL,
            tier=TradeTier.T2,
            strategy_name=TEST_STRATEGY,
            open_risk=999.0,
            margin=999.0,
            notional=999.0,
            reason="test",
            age_ms=20.0,
        )

        assert metrics.reservations.active == 0
        assert metrics.decisions_by_symbol[TEST_SYMBOL].pending_open_risk == pytest.approx(0.0)

    def test_reset_runtime_counters_clears_metrics_but_preserves_startup_semantics(self) -> None:
        metrics = RiskMetrics()
        metrics.register_decision(make_decision())

        assert metrics.total_decisions == 1

        metrics.reset()

        assert metrics.total_decisions == 0
        assert metrics.approvals == 0
        assert metrics.rejections == 0
        assert metrics.decision_counts == {}
        assert metrics.decisions_by_symbol == {}
        assert metrics.reservations.created == 0
        assert metrics.started_at > 0


# =============================================================================
# CircuitBreakerStats
# =============================================================================


class TestCircuitBreakerStats:
    def test_register_failure_classifies_execution_data_system_and_cost_failures(self) -> None:
        stats = CircuitBreakerStats()

        stats.register_failure(CircuitBreakerReason.EXECUTION_FAILURES, now_ts=1_000.0)
        stats.register_failure(CircuitBreakerReason.DATA_FEED_FAILURE, now_ts=1_001.0)
        stats.register_failure(CircuitBreakerReason.SYSTEM_ERROR_RATE, now_ts=1_002.0)
        stats.register_failure(CircuitBreakerReason.EXECUTION_COST_SPIKE, now_ts=1_003.0)

        assert stats.consecutive_failures == 4
        assert stats.total_failures == 4
        assert stats.execution_failures == 2
        assert stats.data_failures == 1
        assert stats.system_failures == 1
        assert stats.cost_spikes == 1
        assert stats.last_failure_reason == CircuitBreakerReason.EXECUTION_COST_SPIKE.value
        assert stats.failure_counts[CircuitBreakerReason.EXECUTION_FAILURES.value] == 1

    def test_register_trigger_and_release_counters(self) -> None:
        stats = CircuitBreakerStats()

        stats.register_trigger(CircuitBreakerReason.MANUAL_HALT, now_ts=1_000.0)
        stats.register_trigger(CircuitBreakerReason.EMERGENCY_STOP, now_ts=1_001.0)
        stats.register_release(forced=False, reason="auto", now_ts=1_010.0)
        stats.register_release(forced=True, reason="force", now_ts=1_011.0)

        assert stats.total_triggers == 2
        assert stats.manual_triggers == 1
        assert stats.emergency_triggers == 1
        assert stats.auto_releases == 1
        assert stats.force_releases == 1
        assert stats.last_release_reason == "force"

    def test_register_success_resets_only_consecutive_failures(self) -> None:
        stats = CircuitBreakerStats()
        stats.register_failure(CircuitBreakerReason.EXECUTION_FAILURES)
        stats.register_failure(CircuitBreakerReason.EXECUTION_FAILURES)

        stats.register_success()

        assert stats.consecutive_failures == 0
        assert stats.total_failures == 2
        assert stats.execution_failures == 2

    def test_reset_failure_counters_keeps_history(self) -> None:
        stats = CircuitBreakerStats()
        stats.register_failure(CircuitBreakerReason.EXECUTION_FAILURES)
        stats.register_trigger(CircuitBreakerReason.MANUAL_HALT)

        stats.reset_failure_counters()

        assert stats.consecutive_failures == 0
        assert stats.execution_failures == 0
        assert stats.total_failures == 1
        assert stats.total_triggers == 1
        assert stats.manual_triggers == 1

    def test_reset_all_clears_everything(self) -> None:
        stats = CircuitBreakerStats()
        stats.register_failure(CircuitBreakerReason.EXECUTION_FAILURES)
        stats.register_trigger(CircuitBreakerReason.MANUAL_HALT)
        stats.register_release(forced=True, reason="force")

        stats.reset_all()

        snapshot = stats.snapshot(enabled=True)
        numeric_fields = [
            "consecutive_failures",
            "execution_failures",
            "data_failures",
            "system_failures",
            "cost_spikes",
            "manual_triggers",
            "emergency_triggers",
            "total_failures",
            "total_triggers",
            "auto_releases",
            "force_releases",
        ]
        for field_name in numeric_fields:
            assert snapshot[field_name] == 0
        assert snapshot["trigger_counts"] == {}
        assert snapshot["failure_counts"] == {}

    def test_from_snapshot_sanitizes_bad_values(self) -> None:
        payload = {
            "consecutive_failures": -10,
            "execution_failures": True,
            "data_failures": 2.9,
            "system_failures": math.nan,
            "cost_spikes": "bad",
            "manual_triggers": 1,
            "emergency_triggers": 2,
            "total_failures": 3,
            "total_triggers": 4,
            "auto_releases": 5,
            "force_releases": 6,
            "last_failure_reason": 123,
            "last_failure_at": math.inf,
            "trigger_counts": {"manual_halt": 1, "bad": -5, "float": 2.9},
            "failure_counts": "bad",
        }

        stats = CircuitBreakerStats.from_snapshot(payload)
        snapshot = stats.snapshot(enabled=True)

        assert snapshot["consecutive_failures"] == 0
        assert snapshot["execution_failures"] == 0
        assert snapshot["data_failures"] == 2
        assert snapshot["system_failures"] == 0
        assert snapshot["cost_spikes"] == 0
        assert snapshot["manual_triggers"] == 1
        assert snapshot["emergency_triggers"] == 2
        assert snapshot["last_failure_reason"] == "123"
        assert snapshot["last_failure_at"] is None
        assert snapshot["trigger_counts"] == {"manual_halt": 1, "float": 2}
        assert snapshot["failure_counts"] == {}
        assert_json_serializable(snapshot)


# =============================================================================
# CircuitBreaker
# =============================================================================


class TestCircuitBreaker:
    def test_disabled_breaker_always_allows_and_does_not_register_failures(self) -> None:
        state = make_state()
        breaker = CircuitBreaker(CircuitBreakerConfig(enabled=False))

        triggered = breaker.register_failure(
            state,
            reason=CircuitBreakerReason.EXECUTION_FAILURES,
        )
        result = breaker.check(state)

        assert triggered is False
        assert result.passed is True
        assert result.decision is RiskDecisionType.ALLOW
        assert result.metadata["enabled"] is False
        assert breaker.stats_state.total_failures == 0

    def test_inactive_breaker_check_allows(self) -> None:
        state = make_state()
        breaker = CircuitBreaker(CircuitBreakerConfig(enabled=True))

        result = breaker.check(state)

        assert result.passed is True
        assert result.decision is RiskDecisionType.ALLOW
        assert result.metadata["circuit_breaker_active"] is False
        assert_json_serializable(result.metadata)

    def test_register_failure_below_threshold_does_not_trigger(self) -> None:
        state = make_state()
        breaker = CircuitBreaker(
            CircuitBreakerConfig(
                max_consecutive_failures=3,
                max_execution_failures=3,
                trigger_on_execution_cost_spike=False,
                trigger_on_data_feed_failure=False,
            )
        )

        assert breaker.register_failure(
            state,
            reason=CircuitBreakerReason.EXECUTION_FAILURES,
            now_ts=1_000.0,
        ) is False

        assert state.is_circuit_breaker_active() is False
        assert breaker.stats_state.consecutive_failures == 1

    def test_consecutive_failures_trigger_breaker_and_check_denies(self) -> None:
        state = make_state()
        breaker = CircuitBreaker(
            CircuitBreakerConfig(
                max_consecutive_failures=2,
                max_execution_failures=99,
                cooldown_seconds=60.0,
                trigger_on_execution_cost_spike=False,
                trigger_on_data_feed_failure=False,
            )
        )

        assert breaker.register_failure(
            state,
            reason=CircuitBreakerReason.SYSTEM_ERROR_RATE,
            now_ts=1_000.0,
        ) is False
        assert breaker.register_failure(
            state,
            reason=CircuitBreakerReason.SYSTEM_ERROR_RATE,
            now_ts=1_001.0,
        ) is True

        assert state.is_circuit_breaker_active() is True
        assert state.circuit_breaker.reason is CircuitBreakerReason.SYSTEM_ERROR_RATE
        assert state.circuit_breaker.cooldown_until == pytest.approx(1_001.0 + 60.0)

        result = breaker.check(state)

        assert result.passed is False
        assert result.decision is RiskDecisionType.HALT_TRADING
        assert result.violations[0].violation_type is RiskViolationType.CIRCUIT_BREAKER_TRIGGERED
        assert result.metadata["circuit_breaker_active"] is True
        assert_json_serializable(result.metadata)

    def test_execution_failure_threshold_triggers_even_without_consecutive_threshold(self) -> None:
        state = make_state()
        breaker = CircuitBreaker(
            CircuitBreakerConfig(
                max_consecutive_failures=99,
                max_execution_failures=2,
                cooldown_seconds=60.0,
            )
        )

        breaker.register_failure(state, reason=CircuitBreakerReason.EXECUTION_FAILURES)
        triggered = breaker.register_failure(state, reason=CircuitBreakerReason.EXECUTION_FAILURES)

        assert triggered is True
        assert state.is_circuit_breaker_active() is True
        assert breaker.stats_state.execution_failures == 2

    def test_data_failure_and_cost_spike_trigger_immediately_when_configured(self) -> None:
        state = make_state()
        breaker = CircuitBreaker(
            CircuitBreakerConfig(
                trigger_on_data_feed_failure=True,
                trigger_on_execution_cost_spike=True,
                cooldown_seconds=60.0,
                max_consecutive_failures=99,
                max_execution_failures=99,
            )
        )

        data_triggered = breaker.register_failure(
            state,
            reason=CircuitBreakerReason.DATA_FEED_FAILURE,
        )
        assert data_triggered is True
        assert state.is_circuit_breaker_active() is True

        state.deactivate_circuit_breaker(force=True)

        cost_triggered = breaker.register_failure(
            state,
            reason=CircuitBreakerReason.EXECUTION_COST_SPIKE,
        )
        assert cost_triggered is True
        assert state.is_circuit_breaker_active() is True

    def test_manual_halt_always_triggers_and_requires_manual_release(self) -> None:
        state = make_state()
        breaker = CircuitBreaker(CircuitBreakerConfig(cooldown_seconds=60.0))

        triggered = breaker.register_failure(
            state,
            reason=CircuitBreakerReason.MANUAL_HALT,
            message="manual stop",
            count_as_failure=False,
            now_ts=1_000.0,
        )

        assert triggered is True
        assert breaker.stats_state.total_failures == 0
        assert breaker.stats_state.manual_triggers == 1
        assert state.is_circuit_breaker_active() is True
        assert state.circuit_breaker.manual_release_required is True

        state.deactivate_circuit_breaker(force=False)
        assert state.is_circuit_breaker_active() is True

        state.deactivate_circuit_breaker(force=True)
        assert state.is_circuit_breaker_active() is False

    def test_emergency_stop_check_returns_emergency_decision(self) -> None:
        state = make_state()
        breaker = CircuitBreaker(
            CircuitBreakerConfig(
                trigger_on_emergency_stop=True,
                require_manual_release_for_emergency=True,
                cooldown_seconds=60.0,
            )
        )

        triggered = breaker.register_failure(
            state,
            reason=CircuitBreakerReason.EMERGENCY_STOP,
            message="emergency",
            count_as_failure=False,
        )

        assert triggered is True
        assert state.emergency_stop_active is True

        result = breaker.check(state)

        assert result.passed is False
        assert result.decision is RiskDecisionType.EMERGENCY_STOP
        assert result.violations[0].violation_type is RiskViolationType.EMERGENCY_STOP_TRIGGERED

    def test_register_success_resets_streak_but_not_history(self) -> None:
        state = make_state()
        breaker = CircuitBreaker(CircuitBreakerConfig(max_consecutive_failures=99))

        breaker.register_failure(state, reason=CircuitBreakerReason.EXECUTION_FAILURES)
        breaker.register_failure(state, reason=CircuitBreakerReason.EXECUTION_FAILURES)

        breaker.register_success()

        assert breaker.stats_state.consecutive_failures == 0
        assert breaker.stats_state.total_failures == 2
        assert breaker.stats_state.execution_failures == 2

    def test_reset_counters_can_keep_or_clear_history(self) -> None:
        state = make_state()
        breaker = CircuitBreaker(CircuitBreakerConfig(max_consecutive_failures=99))

        breaker.register_failure(state, reason=CircuitBreakerReason.EXECUTION_FAILURES)
        breaker.register_failure(state, reason=CircuitBreakerReason.EXECUTION_FAILURES)

        breaker.reset_counters(include_history=False)

        assert breaker.stats_state.consecutive_failures == 0
        assert breaker.stats_state.total_failures == 2

        breaker.reset_counters(include_history=True)

        assert breaker.stats_state.total_failures == 0
        assert breaker.stats_state.execution_failures == 0

    def test_snapshot_and_restore_preserve_diagnostic_counters_without_mutating_state(self) -> None:
        state = make_state()
        breaker = CircuitBreaker(CircuitBreakerConfig(max_consecutive_failures=99))

        breaker.register_failure(
            state,
            reason=CircuitBreakerReason.EXECUTION_FAILURES,
            now_ts=1_000.0,
        )
        breaker.register_failure(
            state,
            reason=CircuitBreakerReason.DATA_FEED_FAILURE,
            now_ts=1_001.0,
        )

        snapshot = breaker.snapshot(state)
        assert_json_serializable(snapshot)

        new_state = make_state()
        restored = CircuitBreaker(CircuitBreakerConfig())
        restored.restore_from_snapshot(snapshot)

        assert restored.stats_state.total_failures == 2
        assert restored.stats_state.execution_failures == 1
        assert restored.stats_state.data_failures == 1

        assert new_state.is_circuit_breaker_active() is False

    def test_restore_from_corrupted_snapshot_sanitizes_counters(self) -> None:
        breaker = CircuitBreaker(CircuitBreakerConfig())

        breaker.restore_from_snapshot(
            {
                "stats": {
                    "consecutive_failures": -10,
                    "execution_failures": True,
                    "data_failures": 2.9,
                    "last_failure_at": math.inf,
                    "failure_counts": {"x": 1, "bad": -100},
                }
            }
        )

        assert breaker.stats_state.consecutive_failures == 0
        assert breaker.stats_state.execution_failures == 0
        assert breaker.stats_state.data_failures == 2
        assert breaker.stats_state.last_failure_at is None
        assert breaker.stats_state.failure_counts == {"x": 1}

    def test_repeated_failures_do_not_create_negative_or_non_finite_stats(self) -> None:
        state = make_state()
        breaker = CircuitBreaker(
            CircuitBreakerConfig(
                max_consecutive_failures=10_000,
                max_execution_failures=10_000,
                trigger_on_data_feed_failure=False,
                trigger_on_execution_cost_spike=False,
            )
        )

        for _ in range(1_000):
            triggered = breaker.register_failure(
                state,
                reason=CircuitBreakerReason.EXECUTION_FAILURES,
            )
            assert triggered is False

        snapshot = breaker.stats()
        numeric_fields = [
            "consecutive_failures",
            "execution_failures",
            "data_failures",
            "system_failures",
            "cost_spikes",
            "total_failures",
            "total_triggers",
            "auto_releases",
            "force_releases",
        ]
        for field_name in numeric_fields:
            value = snapshot[field_name]
            assert isinstance(value, int)
            assert value >= 0
            assert math.isfinite(float(value))

        assert snapshot["total_failures"] == 1_000
        assert snapshot["execution_failures"] == 1_000
        assert_json_serializable(snapshot)


# =============================================================================
# Integrated state/metrics/circuit-breaker safety invariants
# =============================================================================


class TestStateMetricsCircuitBreakerInvariants:
    def test_pending_reservation_state_and_metrics_can_be_reconciled_after_cleanup(self) -> None:
        state = make_state()
        metrics = ReservationMetrics()

        active = state.reserve_risk(
            reservation_id="active",
            symbol=TEST_SYMBOL,
            side=PositionSide.LONG,
            open_risk=10.0,
            margin=20.0,
            notional=100.0,
            ttl_seconds=100.0,
            now_ts=1_000.0,
        )
        expired = state.reserve_risk(
            reservation_id="expired",
            symbol=ALT_SYMBOL,
            side=PositionSide.SHORT,
            open_risk=5.0,
            margin=10.0,
            notional=50.0,
            ttl_seconds=1.0,
            now_ts=1_000.0,
        )

        metrics.register_created(
            reservation_id=active.reservation_id,
            open_risk=active.open_risk,
            margin=active.margin,
            notional=active.notional,
        )
        metrics.register_created(
            reservation_id=expired.reservation_id,
            open_risk=expired.open_risk,
            margin=expired.margin,
            notional=expired.notional,
        )

        expired_items = state.expire_pending_reservations(now_ts=1_002.0)
        for reservation in expired_items:
            metrics.register_expired(
                reservation_id=reservation.reservation_id,
                open_risk=reservation.open_risk,
                margin=reservation.margin,
                notional=reservation.notional,
            )

        metrics.set_active_snapshot(
            active=len(state.pending_reservations),
            reserved_open_risk=state.get_pending_open_risk(),
            reserved_margin=state.get_pending_margin(),
            reserved_notional=state.get_pending_notional(),
        )

        assert len(state.pending_reservations) == 1
        assert metrics.active == 1
        assert metrics.reserved_open_risk == pytest.approx(10.0)
        assert metrics.reserved_margin == pytest.approx(20.0)
        assert metrics.reserved_notional == pytest.approx(100.0)

    def test_state_snapshot_reads_do_not_clear_expired_symbol_or_strategy_cooldowns(self) -> None:
        state = make_state()
        symbol_state = state.get_symbol_state(TEST_SYMBOL)
        strategy_state = state.get_strategy_state(TEST_STRATEGY)

        symbol_state.status = SymbolRiskStatus.COOLDOWN
        symbol_state.cooldown.activate(cooldown_seconds=1.0, now_ts=1.0)

        strategy_state.status = StrategyRiskStatus.COOLDOWN
        strategy_state.cooldown.activate(cooldown_seconds=1.0, now_ts=1.0)

        symbol_snapshot = symbol_state.snapshot(risk_unit=10.0)
        strategy_snapshot = strategy_state.snapshot(risk_unit=10.0)

        assert symbol_snapshot.status is SymbolRiskStatus.ACTIVE
        assert strategy_snapshot.status is StrategyRiskStatus.ACTIVE

        assert symbol_state.status is SymbolRiskStatus.COOLDOWN
        assert symbol_state.cooldown.active is True
        assert strategy_state.status is StrategyRiskStatus.COOLDOWN
        assert strategy_state.cooldown.active is True

    def test_circuit_breaker_blocks_after_state_activation_even_without_local_failure_history(self) -> None:
        state = make_state()
        breaker = CircuitBreaker(CircuitBreakerConfig())

        state.activate_circuit_breaker(
            CircuitBreakerReason.MANUAL_HALT,
            message="external halt",
            manual_release_required=True,
        )

        result = breaker.check(state)

        assert result.passed is False
        assert result.decision is RiskDecisionType.HALT_TRADING
        assert result.violations[0].violation_type is RiskViolationType.CIRCUIT_BREAKER_TRIGGERED
        assert result.metadata["stats"]["total_failures"] == 0