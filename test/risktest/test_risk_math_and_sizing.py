# tests/risk/test_risk_math_and_sizing.py
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import pytest

from risk.config import PositionSizingConfig, RiskUnitConfig
from risk.enums import MarginMode, PositionSide, RiskDecisionType, RiskMode, TradeTier
from risk.exceptions import InvalidPositionSizeError, InvalidRiskRequestError
from risk.models import PositionSizeRequest, RiskUnitSnapshot, TierRiskProfile
from risk.position_sizing import PositionSizer, RiskUnitCalculator, SymbolConstraints
from risk.state import RiskState
from risk.utils import (
    apply_cap,
    apply_confidence_scale,
    apply_volatility_scale,
    calculate_cost_to_reward_ratio,
    calculate_drawdown_pct,
    calculate_expected_value,
    calculate_loss_r,
    calculate_margin_from_notional,
    calculate_margin_required,
    calculate_notional,
    calculate_pct,
    calculate_pnl,
    calculate_position_size_by_risk,
    calculate_reward_distance,
    calculate_r_units,
    calculate_risk_amount_from_size,
    calculate_risk_reward_ratio,
    calculate_side_aware_stop_distance,
    calculate_stop_distance,
    clamp,
    coalesce_float,
    is_finite_number,
    normalize_confidence,
    normalize_probability,
    round_down_to_step,
    safe_div,
)


# =============================================================================
# Local test builders
# =============================================================================

TEST_SYMBOL = "BTCUSDT"


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


def make_risk_unit_snapshot(
    *,
    base_risk_unit: float = 10.0,
    effective_risk_unit: float = 10.0,
    mode: RiskMode = RiskMode.NORMAL,
) -> RiskUnitSnapshot:
    return RiskUnitSnapshot(
        base_risk_unit=base_risk_unit,
        effective_risk_unit=effective_risk_unit,
        mode=mode,
        mode_multiplier=1.0,
        strategy_multiplier=1.0,
        symbol_multiplier=1.0,
        confidence_multiplier=1.0,
        volatility_multiplier=1.0,
    )


def make_tier_profile(
    *,
    requested_tier: TradeTier = TradeTier.T2,
    final_tier: TradeTier = TradeTier.T2,
    risk_units: float = 1.0,
    min_rr: float = 2.0,
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


def make_position_request(
    *,
    symbol: str = TEST_SYMBOL,
    side: PositionSide = PositionSide.LONG,
    entry_price: float = 100.0,
    stop_loss: float | None = 99.0,
    account_equity: float = 10_000.0,
    free_balance: float = 10_000.0,
    risk_amount: float = 10.0,
    risk_unit_snapshot: RiskUnitSnapshot | None = None,
    tier_profile: TierRiskProfile | None = None,
    leverage: float = 5.0,
    margin_mode: MarginMode = MarginMode.ISOLATED,
    requested_size: float | None = None,
    requested_margin: float | None = None,
    confidence: float | None = None,
    volatility: float | None = None,
    min_size: float | None = None,
    max_size: float | None = None,
    step_size: float | None = None,
    min_notional: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> PositionSizeRequest:
    return PositionSizeRequest(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        stop_loss=stop_loss,
        account_equity=account_equity,
        free_balance=free_balance,
        risk_amount=risk_amount,
        risk_unit_snapshot=risk_unit_snapshot or make_risk_unit_snapshot(),
        tier_profile=tier_profile or make_tier_profile(),
        leverage=leverage,
        margin_mode=margin_mode,
        requested_size=requested_size,
        requested_margin=requested_margin,
        confidence=confidence,
        volatility=volatility,
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        min_notional=min_notional,
        metadata=dict(metadata or {}),
    )


def deterministic_position_sizing_config(**overrides: Any) -> PositionSizingConfig:
    config = PositionSizingConfig(
        min_position_size=0.0,
        max_position_size=None,
        require_stop_loss=True,
        fallback_stop_loss_pct=None,
        use_confidence_adjustment=False,
        confidence_scale_min=1.0,
        confidence_scale_max=1.0,
        use_volatility_adjustment=False,
        volatility_scale_min=1.0,
        volatility_scale_max=1.0,
        reject_if_below_min_size=True,
        never_increase_size_above_risk=True,
    )

    for key, value in overrides.items():
        setattr(config, key, value)

    return config


def make_sizer(
    *,
    config: PositionSizingConfig | None = None,
    symbol_constraints: dict[str, SymbolConstraints] | None = None,
) -> PositionSizer:
    return PositionSizer(
        config or deterministic_position_sizing_config(),
        symbol_constraints=symbol_constraints,
        service_name="risk.position_sizer.tests",
    )


def assert_finite_non_negative(value: float, *, field_name: str) -> None:
    assert math.isfinite(value), f"{field_name} must be finite, got={value!r}"
    assert value >= 0.0, f"{field_name} must be non-negative, got={value!r}"


def assert_position_result_is_capital_safe(
    *,
    result: Any,
    request: PositionSizeRequest,
) -> None:
    assert_finite_non_negative(result.size, field_name="size")
    assert_finite_non_negative(result.notional_value, field_name="notional_value")
    assert_finite_non_negative(result.margin_required, field_name="margin_required")
    assert_finite_non_negative(result.risk_amount, field_name="risk_amount")
    assert_finite_non_negative(result.risk_unit_used, field_name="risk_unit_used")
    assert_finite_non_negative(result.risk_units_used, field_name="risk_units_used")
    assert_finite_non_negative(result.leverage_used, field_name="leverage_used")

    assert result.size > 0.0
    assert result.margin_required <= request.free_balance + 1e-12

    stop_distance = calculate_side_aware_stop_distance(
        side=request.side,
        entry_price=request.entry_price,
        stop_loss=request.stop_loss,
    )
    assert stop_distance is not None
    actual_risk = calculate_risk_amount_from_size(
        size=result.size,
        stop_distance=stop_distance,
    )

    # Ключовий invariant: sizing не має збільшити фактичний risk понад
    # request.risk_amount. Через down-rounding може бути менше, але не більше.
    assert actual_risk <= request.risk_amount + 1e-12


# =============================================================================
# risk.utils
# =============================================================================

class TestRiskUtilsMath:
    def test_safe_div_returns_default_only_for_zero_denominator(self) -> None:
        assert safe_div(10.0, 2.0) == pytest.approx(5.0)
        assert safe_div(10.0, 0.0, default=-1.0) == pytest.approx(-1.0)
        assert calculate_pct(25.0, 100.0) == pytest.approx(0.25)
        assert calculate_pct(25.0, 0.0, default=9.99) == pytest.approx(9.99)

    def test_clamp_rejects_reversed_range(self) -> None:
        assert clamp(5.0, 1.0, 10.0) == pytest.approx(5.0)
        assert clamp(-5.0, 1.0, 10.0) == pytest.approx(1.0)
        assert clamp(50.0, 1.0, 10.0) == pytest.approx(10.0)

        with pytest.raises(ValueError, match="min_value"):
            clamp(5.0, 10.0, 1.0)

    @pytest.mark.parametrize(
        ("current_equity", "peak_equity", "expected"),
        [
            (10_000.0, 10_000.0, 0.0),
            (9_000.0, 10_000.0, 0.10),
            (11_000.0, 10_000.0, 0.0),
            (9_000.0, 0.0, 0.0),
            (9_000.0, -10_000.0, 0.0),
        ],
    )
    def test_calculate_drawdown_pct_is_non_negative(
        self,
        current_equity: float,
        peak_equity: float,
        expected: float,
    ) -> None:
        assert calculate_drawdown_pct(current_equity, peak_equity) == pytest.approx(expected)

    def test_r_units_use_loss_magnitude_only(self) -> None:
        assert calculate_loss_r(-50.0, 10.0) == pytest.approx(5.0)
        assert calculate_loss_r(50.0, 10.0) == pytest.approx(0.0)
        assert calculate_loss_r(-50.0, 0.0, default=999.0) == pytest.approx(999.0)

        assert calculate_r_units(-50.0, 10.0) == pytest.approx(5.0)
        assert calculate_r_units(50.0, 10.0) == pytest.approx(5.0)
        assert calculate_r_units(50.0, 0.0, default=999.0) == pytest.approx(999.0)

    def test_backward_compatible_stop_distance_is_absolute_and_not_side_safe(self) -> None:
        assert calculate_stop_distance(100.0, 99.0) == pytest.approx(1.0)
        assert calculate_stop_distance(100.0, 101.0) == pytest.approx(1.0)
        assert calculate_stop_distance(100.0, None) is None

    @pytest.mark.parametrize(
        ("side", "entry_price", "stop_loss", "expected"),
        [
            (PositionSide.LONG, 100.0, 99.0, 1.0),
            (PositionSide.SHORT, 100.0, 101.0, 1.0),
            (PositionSide.LONG, 100.0, None, None),
            (PositionSide.SHORT, 100.0, None, None),
        ],
    )
    def test_side_aware_stop_distance_valid_cases(
        self,
        side: PositionSide,
        entry_price: float,
        stop_loss: float | None,
        expected: float | None,
    ) -> None:
        assert (
            calculate_side_aware_stop_distance(
                side=side,
                entry_price=entry_price,
                stop_loss=stop_loss,
            )
            == expected
        )

    @pytest.mark.parametrize(
        ("side", "entry_price", "stop_loss"),
        [
            (PositionSide.LONG, 100.0, 100.0),
            (PositionSide.LONG, 100.0, 101.0),
            (PositionSide.SHORT, 100.0, 100.0),
            (PositionSide.SHORT, 100.0, 99.0),
            (PositionSide.LONG, 0.0, 99.0),
            (PositionSide.LONG, -100.0, 99.0),
            (PositionSide.LONG, 100.0, 0.0),
            (PositionSide.LONG, 100.0, -99.0),
        ],
    )
    def test_side_aware_stop_distance_rejects_wrong_side_or_invalid_price(
        self,
        side: PositionSide,
        entry_price: float,
        stop_loss: float | None,
    ) -> None:
        with pytest.raises(ValueError):
            calculate_side_aware_stop_distance(
                side=side,
                entry_price=entry_price,
                stop_loss=stop_loss,
            )

    @pytest.mark.parametrize(
        ("side", "entry_price", "take_profit", "expected"),
        [
            (PositionSide.LONG, 100.0, 103.0, 3.0),
            (PositionSide.SHORT, 100.0, 97.0, 3.0),
            (PositionSide.LONG, 100.0, None, None),
            (PositionSide.SHORT, 100.0, None, None),
        ],
    )
    def test_reward_distance_valid_cases(
        self,
        side: PositionSide,
        entry_price: float,
        take_profit: float | None,
        expected: float | None,
    ) -> None:
        assert (
            calculate_reward_distance(
                side=side,
                entry_price=entry_price,
                take_profit=take_profit,
            )
            == expected
        )

    @pytest.mark.parametrize(
        ("side", "entry_price", "take_profit"),
        [
            (PositionSide.LONG, 100.0, 100.0),
            (PositionSide.LONG, 100.0, 99.0),
            (PositionSide.SHORT, 100.0, 100.0),
            (PositionSide.SHORT, 100.0, 101.0),
            (PositionSide.LONG, 0.0, 101.0),
            (PositionSide.LONG, 100.0, 0.0),
        ],
    )
    def test_reward_distance_rejects_wrong_side_or_invalid_price(
        self,
        side: PositionSide,
        entry_price: float,
        take_profit: float | None,
    ) -> None:
        with pytest.raises(ValueError):
            calculate_reward_distance(
                side=side,
                entry_price=entry_price,
                take_profit=take_profit,
            )

    @pytest.mark.parametrize(
        ("side", "entry_price", "stop_loss", "take_profit", "expected"),
        [
            (PositionSide.LONG, 100.0, 99.0, 103.0, 3.0),
            (PositionSide.SHORT, 100.0, 101.0, 97.0, 3.0),
            (PositionSide.LONG, 100.0, None, 103.0, None),
            (PositionSide.LONG, 100.0, 99.0, None, None),
        ],
    )
    def test_risk_reward_ratio(
        self,
        side: PositionSide,
        entry_price: float,
        stop_loss: float | None,
        take_profit: float | None,
        expected: float | None,
    ) -> None:
        assert (
            calculate_risk_reward_ratio(
                side=side,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            == expected
        )

    def test_expected_value_and_cost_to_reward(self) -> None:
        assert calculate_expected_value(
            expected_reward=3.0,
            expected_loss=1.0,
            win_probability=0.5,
            expected_cost=0.1,
        ) == pytest.approx(0.9)

        assert calculate_cost_to_reward_ratio(
            expected_cost=0.3,
            expected_reward=3.0,
        ) == pytest.approx(0.1)

        assert math.isinf(
            calculate_cost_to_reward_ratio(
                expected_cost=0.3,
                expected_reward=0.0,
            )
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"expected_reward": -1.0, "expected_loss": 1.0, "win_probability": 0.5},
            {"expected_reward": 1.0, "expected_loss": -1.0, "win_probability": 0.5},
            {
                "expected_reward": 1.0,
                "expected_loss": 1.0,
                "win_probability": 0.5,
                "expected_cost": -0.1,
            },
        ],
    )
    def test_expected_value_rejects_negative_magnitudes(
        self,
        kwargs: dict[str, float],
    ) -> None:
        with pytest.raises(ValueError):
            calculate_expected_value(**kwargs)

    def test_notional_margin_position_size_and_risk_amount_are_consistent(self) -> None:
        assert calculate_notional(100.0, 2.5) == pytest.approx(250.0)
        assert calculate_margin_required(100.0, 2.5, 5.0) == pytest.approx(50.0)
        assert calculate_margin_required(100.0, 2.5, None) == pytest.approx(250.0)
        assert calculate_margin_from_notional(250.0, 5.0) == pytest.approx(50.0)
        assert calculate_margin_from_notional(250.0, None) == pytest.approx(250.0)

        assert calculate_position_size_by_risk(
            risk_amount=10.0,
            stop_distance=2.0,
        ) == pytest.approx(5.0)

        assert calculate_risk_amount_from_size(
            size=5.0,
            stop_distance=2.0,
        ) == pytest.approx(10.0)

    @pytest.mark.parametrize(
        ("fn", "kwargs"),
        [
            (calculate_notional, {"entry_price": 0.0, "size": 1.0}),
            (calculate_notional, {"entry_price": -1.0, "size": 1.0}),
            (calculate_notional, {"entry_price": 100.0, "size": -1.0}),
            (calculate_margin_required, {"entry_price": 100.0, "size": 1.0, "leverage": 0.0}),
            (calculate_margin_required, {"entry_price": 100.0, "size": 1.0, "leverage": -1.0}),
            (calculate_margin_from_notional, {"notional_value": -1.0, "leverage": 1.0}),
            (calculate_margin_from_notional, {"notional_value": 1.0, "leverage": 0.0}),
            (calculate_position_size_by_risk, {"risk_amount": -1.0, "stop_distance": 1.0}),
            (calculate_position_size_by_risk, {"risk_amount": 1.0, "stop_distance": 0.0}),
            (calculate_position_size_by_risk, {"risk_amount": 1.0, "stop_distance": -1.0}),
            (calculate_risk_amount_from_size, {"size": -1.0, "stop_distance": 1.0}),
            (calculate_risk_amount_from_size, {"size": 1.0, "stop_distance": 0.0}),
            (calculate_risk_amount_from_size, {"size": 1.0, "stop_distance": -1.0}),
        ],
    )
    def test_money_and_size_helpers_reject_invalid_values(
        self,
        fn: Any,
        kwargs: dict[str, Any],
    ) -> None:
        with pytest.raises(ValueError):
            fn(**kwargs)

    @pytest.mark.parametrize(
        ("side", "entry", "exit", "size", "expected"),
        [
            (PositionSide.LONG, 100.0, 110.0, 2.0, 20.0),
            (PositionSide.LONG, 100.0, 90.0, 2.0, -20.0),
            (PositionSide.SHORT, 100.0, 90.0, 2.0, 20.0),
            (PositionSide.SHORT, 100.0, 110.0, 2.0, -20.0),
        ],
    )
    def test_calculate_pnl_side_aware(
        self,
        side: PositionSide,
        entry: float,
        exit: float,
        size: float,
        expected: float,
    ) -> None:
        assert calculate_pnl(
            side=side,
            entry_price=entry,
            exit_price=exit,
            size=size,
        ) == pytest.approx(expected)

    def test_normalize_probability_and_confidence_clamp_values(self) -> None:
        assert normalize_probability(None) == pytest.approx(0.5)
        assert normalize_probability(-10.0) == pytest.approx(0.0)
        assert normalize_probability(0.25) == pytest.approx(0.25)
        assert normalize_probability(10.0) == pytest.approx(1.0)

        assert normalize_confidence(None) == pytest.approx(0.5)
        assert normalize_confidence(-10.0) == pytest.approx(0.0)
        assert normalize_confidence(10.0) == pytest.approx(1.0)

    def test_confidence_scale_and_volatility_scale(self) -> None:
        assert apply_confidence_scale(
            100.0,
            confidence=0.5,
            scale_min=0.5,
            scale_max=1.25,
        ) == pytest.approx(87.5)

        assert apply_confidence_scale(
            100.0,
            confidence=None,
            scale_min=0.5,
            scale_max=1.25,
        ) == pytest.approx(87.5)

        assert apply_volatility_scale(
            100.0,
            volatility=1.0,
            scale_min=0.25,
            scale_max=1.0,
        ) == pytest.approx(50.0)

        assert apply_volatility_scale(
            100.0,
            volatility=None,
            scale_min=0.25,
            scale_max=1.0,
        ) == pytest.approx(100.0)

    @pytest.mark.parametrize(
        ("value", "cap", "expected"),
        [
            (10.0, None, 10.0),
            (10.0, 15.0, 10.0),
            (10.0, 5.0, 5.0),
        ],
    )
    def test_apply_cap(self, value: float, cap: float | None, expected: float) -> None:
        assert apply_cap(value, cap) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("value", "step", "expected"),
        [
            (10.999, None, 10.999),
            (10.999, 0.0, 10.999),
            (10.999, -0.1, 10.999),
            (10.999, 1.0, 10.0),
            (10.999, 0.1, 10.9),
            (10.999, 0.01, 10.99),
            (0.009, 0.01, 0.0),
        ],
    )
    def test_round_down_to_step_never_rounds_up(
        self,
        value: float,
        step: float | None,
        expected: float,
    ) -> None:
        rounded = round_down_to_step(value, step)
        assert rounded == pytest.approx(expected)
        assert rounded <= value + 1e-12

    def test_round_down_to_step_rejects_negative_value(self) -> None:
        with pytest.raises(ValueError):
            round_down_to_step(-0.001, 0.01)

    def test_coalesce_float_returns_first_non_none_value(self) -> None:
        assert coalesce_float(None, None, 3.0, default=7.0) == pytest.approx(3.0)
        assert coalesce_float(None, 0.0, 3.0, default=7.0) == pytest.approx(0.0)
        assert coalesce_float(None, None, default=7.0) == pytest.approx(7.0)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1.0, True),
            (0.0, True),
            (-1.0, True),
            (None, False),
            (math.nan, False),
            (math.inf, False),
            (-math.inf, False),
        ],
    )
    def test_is_finite_number(self, value: float | None, expected: bool) -> None:
        assert is_finite_number(value) is expected


class TestRiskUtilsAdversarialFiniteInputs:
    """
    Ці тести — safety gate проти silent NaN/inf.

    Якщо частина з них падає, це не "поганий тест", а сигнал, що helper може
    пропустити нечисловий market/account input далі в risk pipeline.
    """

    @pytest.mark.parametrize(
        ("fn", "kwargs"),
        [
            (
                calculate_side_aware_stop_distance,
                {"side": PositionSide.LONG, "entry_price": math.nan, "stop_loss": 99.0},
            ),
            (
                calculate_side_aware_stop_distance,
                {"side": PositionSide.LONG, "entry_price": math.inf, "stop_loss": 99.0},
            ),
            (
                calculate_side_aware_stop_distance,
                {"side": PositionSide.LONG, "entry_price": 100.0, "stop_loss": math.nan},
            ),
            (
                calculate_reward_distance,
                {"side": PositionSide.LONG, "entry_price": math.nan, "take_profit": 101.0},
            ),
            (
                calculate_reward_distance,
                {"side": PositionSide.LONG, "entry_price": 100.0, "take_profit": math.nan},
            ),
            (
                calculate_notional,
                {"entry_price": math.nan, "size": 1.0},
            ),
            (
                calculate_notional,
                {"entry_price": 100.0, "size": math.nan},
            ),
            (
                calculate_margin_required,
                {"entry_price": 100.0, "size": 1.0, "leverage": math.nan},
            ),
            (
                calculate_margin_from_notional,
                {"notional_value": math.nan, "leverage": 1.0},
            ),
            (
                calculate_position_size_by_risk,
                {"risk_amount": math.nan, "stop_distance": 1.0},
            ),
            (
                calculate_position_size_by_risk,
                {"risk_amount": 1.0, "stop_distance": math.nan},
            ),
            (
                calculate_risk_amount_from_size,
                {"size": math.nan, "stop_distance": 1.0},
            ),
            (
                calculate_risk_amount_from_size,
                {"size": 1.0, "stop_distance": math.nan},
            ),
        ],
    )
    def test_core_money_math_should_reject_nan_inf_inputs(
        self,
        fn: Any,
        kwargs: dict[str, Any],
    ) -> None:
        with pytest.raises(ValueError):
            fn(**kwargs)


# =============================================================================
# RiskUnitCalculator
# =============================================================================

class TestRiskUnitCalculator:
    def test_normal_mode_calculates_base_r_from_equity(self) -> None:
        state = make_state(equity=10_000.0)
        calculator = RiskUnitCalculator(RiskUnitConfig(base_risk_unit_pct=0.001))

        snapshot = calculator.calculate(state, mode=RiskMode.NORMAL)

        assert snapshot.base_risk_unit == pytest.approx(10.0)
        assert snapshot.effective_risk_unit == pytest.approx(10.0)
        assert snapshot.mode is RiskMode.NORMAL
        assert snapshot.mode_multiplier == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("mode", "expected_multiplier", "expected_effective"),
        [
            (RiskMode.NORMAL, 1.0, 10.0),
            (RiskMode.CAUTION, 0.75, 7.5),
            (RiskMode.SAFE_MODE, 0.50, 5.0),
            (RiskMode.REDUCE_ONLY, 0.0, 0.0),
            (RiskMode.HALTED, 0.0, 0.0),
            (RiskMode.EMERGENCY_STOP, 0.0, 0.0),
        ],
    )
    def test_mode_multipliers(
        self,
        mode: RiskMode,
        expected_multiplier: float,
        expected_effective: float,
    ) -> None:
        state = make_state(equity=10_000.0)
        calculator = RiskUnitCalculator(RiskUnitConfig(base_risk_unit_pct=0.001))

        snapshot = calculator.calculate(state, mode=mode)

        assert snapshot.mode_multiplier == pytest.approx(expected_multiplier)
        assert snapshot.effective_risk_unit == pytest.approx(expected_effective)

    def test_negative_multipliers_are_clamped_to_zero(self) -> None:
        state = make_state(equity=10_000.0)
        calculator = RiskUnitCalculator(RiskUnitConfig(base_risk_unit_pct=0.001))

        snapshot = calculator.calculate(
            state,
            strategy_multiplier=-10.0,
            symbol_multiplier=1.0,
            confidence_multiplier=1.0,
            volatility_multiplier=1.0,
        )

        assert snapshot.effective_risk_unit == pytest.approx(0.0)
        assert snapshot.strategy_multiplier == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("kwargs", "expected", "flag"),
        [
            ({"available_daily_r": 0.25}, 2.5, "capped_by_daily_budget"),
            ({"available_open_r": 0.25}, 2.5, "capped_by_open_risk"),
            ({"available_strategy_r": 0.25}, 2.5, "capped_by_strategy_budget"),
            ({"available_symbol_r": 0.25}, 2.5, "capped_by_symbol_budget"),
        ],
    )
    def test_available_budget_caps_effective_risk_unit(
        self,
        kwargs: dict[str, float],
        expected: float,
        flag: str,
    ) -> None:
        state = make_state(equity=10_000.0)
        calculator = RiskUnitCalculator(RiskUnitConfig(base_risk_unit_pct=0.001))

        snapshot = calculator.calculate(state, **kwargs)

        assert snapshot.effective_risk_unit == pytest.approx(expected)
        assert getattr(snapshot, flag) is True

    def test_negative_available_budget_is_ignored_not_used_as_negative_cap(self) -> None:
        state = make_state(equity=10_000.0)
        calculator = RiskUnitCalculator(RiskUnitConfig(base_risk_unit_pct=0.001))

        snapshot = calculator.calculate(
            state,
            available_daily_r=-100.0,
            available_open_r=-100.0,
            available_strategy_r=-100.0,
            available_symbol_r=-100.0,
        )

        assert snapshot.effective_risk_unit == pytest.approx(10.0)
        assert snapshot.capped_by_daily_budget is False
        assert snapshot.capped_by_open_risk is False
        assert snapshot.capped_by_strategy_budget is False
        assert snapshot.capped_by_symbol_budget is False

    def test_min_and_max_risk_unit_caps_base_and_effective(self) -> None:
        state = make_state(equity=10_000.0)
        calculator = RiskUnitCalculator(
            RiskUnitConfig(
                base_risk_unit_pct=0.001,
                min_risk_unit=20.0,
                max_risk_unit=30.0,
            )
        )

        snapshot = calculator.calculate(state, mode=RiskMode.NORMAL)

        assert snapshot.base_risk_unit == pytest.approx(20.0)
        assert snapshot.effective_risk_unit == pytest.approx(20.0)

    def test_max_risk_unit_caps_large_account(self) -> None:
        state = make_state(equity=1_000_000.0)
        calculator = RiskUnitCalculator(
            RiskUnitConfig(
                base_risk_unit_pct=0.001,
                max_risk_unit=100.0,
            )
        )

        snapshot = calculator.calculate(state)

        assert snapshot.base_risk_unit == pytest.approx(100.0)
        assert snapshot.effective_risk_unit == pytest.approx(100.0)

    def test_use_equity_for_r_false_uses_absolute_base_value(self) -> None:
        state = make_state(equity=1_000_000.0)
        calculator = RiskUnitCalculator(
            RiskUnitConfig(
                base_risk_unit_pct=25.0,
                use_equity_for_r=False,
            )
        )

        snapshot = calculator.calculate(state)

        assert snapshot.base_risk_unit == pytest.approx(25.0)
        assert snapshot.effective_risk_unit == pytest.approx(25.0)

    def test_zero_equity_gives_zero_risk_unit(self) -> None:
        state = make_state(balance=0.0, equity=0.0, free_balance=0.0)
        calculator = RiskUnitCalculator(RiskUnitConfig(base_risk_unit_pct=0.001))

        snapshot = calculator.calculate(state)

        assert snapshot.base_risk_unit == pytest.approx(0.0)
        assert snapshot.effective_risk_unit == pytest.approx(0.0)

    def test_negative_equity_is_rejected(self) -> None:
        state = make_state(equity=10_000.0)
        state.equity = -1.0

        calculator = RiskUnitCalculator(RiskUnitConfig(base_risk_unit_pct=0.001))

        with pytest.raises(InvalidRiskRequestError, match="equity"):
            calculator.calculate(state)

    @pytest.mark.parametrize("bad_equity", [math.nan, math.inf, -math.inf])
    def test_non_finite_equity_should_not_produce_non_finite_risk_unit(
        self,
        bad_equity: float,
    ) -> None:
        state = make_state(equity=10_000.0)
        state.equity = bad_equity

        calculator = RiskUnitCalculator(RiskUnitConfig(base_risk_unit_pct=0.001))

        with pytest.raises(InvalidRiskRequestError):
            calculator.calculate(state)


# =============================================================================
# PositionSizer
# =============================================================================

class TestPositionSizerHappyPath:
    def test_valid_long_position_size_is_risk_amount_divided_by_stop_distance(
        self,
    ) -> None:
        state = make_state()
        sizer = make_sizer()
        request = make_position_request(
            side=PositionSide.LONG,
            entry_price=100.0,
            stop_loss=99.0,
            risk_amount=10.0,
            leverage=5.0,
        )

        result = sizer.calculate(request, state)

        assert result.size == pytest.approx(10.0)
        assert result.notional_value == pytest.approx(1_000.0)
        assert result.margin_required == pytest.approx(200.0)
        assert result.risk_amount == pytest.approx(10.0)
        assert result.risk_unit_used == pytest.approx(10.0)
        assert result.risk_units_used == pytest.approx(1.0)
        assert result.leverage_used == pytest.approx(5.0)
        assert result.tier is TradeTier.T2
        assert result.stop_distance == pytest.approx(1.0)
        assert result.capped is False
        assert result.reason is None

        assert_position_result_is_capital_safe(result=result, request=request)

    def test_valid_short_position_size_is_side_aware(self) -> None:
        state = make_state()
        sizer = make_sizer()
        request = make_position_request(
            side=PositionSide.SHORT,
            entry_price=100.0,
            stop_loss=101.0,
            risk_amount=10.0,
            leverage=5.0,
        )

        result = sizer.calculate(request, state)

        assert result.size == pytest.approx(10.0)
        assert result.stop_distance == pytest.approx(1.0)
        assert result.notional_value == pytest.approx(1_000.0)
        assert result.margin_required == pytest.approx(200.0)
        assert_position_result_is_capital_safe(result=result, request=request)

    def test_position_sizer_check_returns_allow_for_uncapped_valid_request(self) -> None:
        state = make_state()
        sizer = make_sizer()
        request = make_position_request()

        result = sizer.check(request, state)

        assert result.passed is True
        assert result.decision is RiskDecisionType.ALLOW
        assert result.adjusted_size == pytest.approx(10.0)
        assert result.adjusted_margin == pytest.approx(200.0)
        assert result.adjusted_leverage == pytest.approx(5.0)
        assert result.adjusted_risk_amount == pytest.approx(10.0)
        assert result.adjusted_tier is TradeTier.T2

    def test_fallback_stop_loss_pct_is_used_only_when_stop_loss_missing_and_allowed(
        self,
    ) -> None:
        state = make_state()
        config = deterministic_position_sizing_config(
            require_stop_loss=False,
            fallback_stop_loss_pct=0.02,
        )
        sizer = make_sizer(config=config)
        request = make_position_request(
            entry_price=100.0,
            stop_loss=None,
            risk_amount=10.0,
            leverage=5.0,
        )

        result = sizer.calculate(request, state)

        assert result.stop_distance == pytest.approx(2.0)
        assert result.size == pytest.approx(5.0)
        assert result.notional_value == pytest.approx(500.0)
        assert result.margin_required == pytest.approx(100.0)

    def test_confidence_and_volatility_adjustments_can_only_reduce_or_scale_from_raw_risk_size(
        self,
    ) -> None:
        state = make_state()
        config = deterministic_position_sizing_config(
            use_confidence_adjustment=True,
            confidence_scale_min=0.50,
            confidence_scale_max=1.25,
            use_volatility_adjustment=True,
            volatility_scale_min=0.25,
            volatility_scale_max=1.00,
        )
        sizer = make_sizer(config=config)
        request = make_position_request(
            risk_amount=10.0,
            entry_price=100.0,
            stop_loss=99.0,
            confidence=0.0,
            volatility=1.0,
        )

        result = sizer.calculate(request, state)

        # raw size = 10
        # confidence=0 -> scale_min=0.5 => 5
        # volatility=1 -> 1/(1+1)=0.5 => 2.5
        assert result.size == pytest.approx(2.5)
        assert_position_result_is_capital_safe(result=result, request=request)


class TestPositionSizerValidation:
    @pytest.mark.parametrize(
        ("field_name", "bad_value", "expected_error"),
        [
            ("symbol", "", "symbol"),
            ("entry_price", 0.0, "entry_price"),
            ("entry_price", -1.0, "entry_price"),
            ("account_equity", -1.0, "account_equity"),
            ("free_balance", -1.0, "free_balance"),
            ("risk_amount", 0.0, "risk_amount"),
            ("risk_amount", -1.0, "risk_amount"),
            ("leverage", 0.0, "leverage"),
            ("leverage", -1.0, "leverage"),
        ],
    )
    def test_invalid_position_size_request_fields_are_rejected(
        self,
        field_name: str,
        bad_value: Any,
        expected_error: str,
    ) -> None:
        state = make_state()
        sizer = make_sizer()
        request = make_position_request()
        request = replace(request, **{field_name: bad_value})

        with pytest.raises(InvalidRiskRequestError, match=expected_error):
            sizer.calculate(request, state)

    def test_negative_state_equity_is_rejected(self) -> None:
        state = make_state()
        state.equity = -1.0

        sizer = make_sizer()
        request = make_position_request()

        with pytest.raises(InvalidRiskRequestError, match="equity"):
            sizer.calculate(request, state)

    def test_negative_effective_risk_unit_is_rejected(self) -> None:
        state = make_state()
        sizer = make_sizer()
        request = make_position_request(
            risk_unit_snapshot=make_risk_unit_snapshot(effective_risk_unit=-1.0),
        )

        with pytest.raises(InvalidRiskRequestError, match="effective_risk_unit"):
            sizer.calculate(request, state)

    def test_non_positive_tier_risk_units_are_rejected(self) -> None:
        state = make_state()
        sizer = make_sizer()
        request = make_position_request(
            tier_profile=make_tier_profile(risk_units=0.0),
        )

        with pytest.raises(InvalidRiskRequestError, match="tier risk_units"):
            sizer.calculate(request, state)

    def test_missing_stop_loss_is_rejected_when_required(self) -> None:
        state = make_state()
        sizer = make_sizer()
        request = make_position_request(stop_loss=None)

        with pytest.raises(InvalidPositionSizeError, match="Stop loss"):
            sizer.calculate(request, state)

    def test_missing_stop_loss_without_valid_fallback_is_rejected(self) -> None:
        state = make_state()
        sizer = make_sizer(
            config=deterministic_position_sizing_config(
                require_stop_loss=False,
                fallback_stop_loss_pct=None,
            )
        )
        request = make_position_request(stop_loss=None)

        with pytest.raises(InvalidPositionSizeError, match="fallback"):
            sizer.calculate(request, state)

    @pytest.mark.parametrize(
        ("side", "entry_price", "stop_loss"),
        [
            (PositionSide.LONG, 100.0, 100.0),
            (PositionSide.LONG, 100.0, 101.0),
            (PositionSide.SHORT, 100.0, 100.0),
            (PositionSide.SHORT, 100.0, 99.0),
        ],
    )
    def test_wrong_side_stop_loss_is_rejected(
        self,
        side: PositionSide,
        entry_price: float,
        stop_loss: float,
    ) -> None:
        state = make_state()
        sizer = make_sizer()
        request = make_position_request(
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
        )

        with pytest.raises(ValueError):
            sizer.calculate(request, state)

    def test_check_translates_invalid_request_to_denied_result(self) -> None:
        state = make_state()
        sizer = make_sizer()
        request = make_position_request(risk_amount=0.0)

        result = sizer.check(request, state)

        assert result.passed is False
        assert result.decision is RiskDecisionType.DENY
        assert result.reason is not None
        assert result.violations

    @pytest.mark.parametrize(
        ("field_name", "bad_value"),
        [
            ("entry_price", math.nan),
            ("entry_price", math.inf),
            ("account_equity", math.nan),
            ("free_balance", math.nan),
            ("risk_amount", math.nan),
            ("risk_amount", math.inf),
            ("leverage", math.nan),
            ("leverage", math.inf),
        ],
    )
    def test_non_finite_request_values_are_rejected(
        self,
        field_name: str,
        bad_value: float,
    ) -> None:
        state = make_state()
        sizer = make_sizer()
        request = replace(make_position_request(), **{field_name: bad_value})

        with pytest.raises((InvalidRiskRequestError, InvalidPositionSizeError, ValueError)):
            sizer.calculate(request, state)


class TestPositionSizerCapsAndConstraints:
    def test_requested_size_can_reduce_but_never_increase_risk_based_size(self) -> None:
        state = make_state()
        sizer = make_sizer()

        risk_based_request = make_position_request(risk_amount=10.0, requested_size=None)
        risk_based_result = sizer.calculate(risk_based_request, state)
        assert risk_based_result.size == pytest.approx(10.0)

        smaller_request = make_position_request(risk_amount=10.0, requested_size=3.0)
        smaller_result = sizer.calculate(smaller_request, state)
        assert smaller_result.size == pytest.approx(3.0)
        assert_position_result_is_capital_safe(result=smaller_result, request=smaller_request)

        larger_request = make_position_request(risk_amount=10.0, requested_size=999.0)
        larger_result = sizer.calculate(larger_request, state)
        assert larger_result.size == pytest.approx(10.0)
        assert_position_result_is_capital_safe(result=larger_result, request=larger_request)

    def test_max_size_caps_position_and_returns_reduce_size_check(self) -> None:
        state = make_state()
        sizer = make_sizer()
        request = make_position_request(
            risk_amount=10.0,
            max_size=4.0,
        )

        result = sizer.calculate(request, state)

        assert result.size == pytest.approx(4.0)
        assert result.capped is True
        assert result.reason is not None
        assert "max_size" in result.reason
        assert_position_result_is_capital_safe(result=result, request=request)

        check = sizer.check(request, state)
        assert check.passed is True
        assert check.decision is RiskDecisionType.REDUCE_SIZE
        assert check.adjusted_size == pytest.approx(4.0)

    def test_symbol_constraints_max_size_caps_request_level_size(self) -> None:
        state = make_state()
        sizer = make_sizer(
            symbol_constraints={
                TEST_SYMBOL: SymbolConstraints(max_size=3.0),
            }
        )
        request = make_position_request(risk_amount=10.0)

        result = sizer.calculate(request, state)

        assert result.size == pytest.approx(3.0)
        assert result.capped is True
        assert_position_result_is_capital_safe(result=result, request=request)

    def test_request_max_size_tightens_symbol_max_size(self) -> None:
        state = make_state()
        sizer = make_sizer(
            symbol_constraints={
                TEST_SYMBOL: SymbolConstraints(max_size=7.0),
            }
        )
        request = make_position_request(risk_amount=10.0, max_size=4.0)

        result = sizer.calculate(request, state)

        assert result.size == pytest.approx(4.0)
        assert result.capped is True
        assert_position_result_is_capital_safe(result=result, request=request)

    def test_step_size_rounds_down_and_never_increases_actual_risk(self) -> None:
        state = make_state()
        sizer = make_sizer()
        request = make_position_request(
            risk_amount=10.0,
            entry_price=100.0,
            stop_loss=97.0,
            step_size=0.1,
        )

        result = sizer.calculate(request, state)

        # raw size = 10 / 3 = 3.333..., step 0.1 => 3.3
        assert result.size == pytest.approx(3.3)
        assert result.capped is True
        assert result.reason is not None
        assert "step_size" in result.reason
        assert_position_result_is_capital_safe(result=result, request=request)

        actual_risk = result.size * result.stop_distance
        assert actual_risk == pytest.approx(9.9)
        assert actual_risk < request.risk_amount

    def test_requested_margin_caps_size(self) -> None:
        state = make_state()
        sizer = make_sizer()
        request = make_position_request(
            risk_amount=10.0,
            entry_price=100.0,
            stop_loss=99.0,
            leverage=5.0,
            requested_margin=50.0,
            step_size=0.01,
        )

        result = sizer.calculate(request, state)

        # max notional by requested margin = 50 * 5 = 250
        # size = 250 / 100 = 2.5
        assert result.size == pytest.approx(2.5)
        assert result.margin_required == pytest.approx(50.0)
        assert result.capped is True
        assert result.reason is not None
        assert "requested_margin_cap_applied" in result.reason
        assert_position_result_is_capital_safe(result=result, request=request)

    def test_state_free_balance_caps_size(self) -> None:
        state = make_state(free_balance=50.0)
        sizer = make_sizer()
        request = make_position_request(
            risk_amount=10.0,
            entry_price=100.0,
            stop_loss=99.0,
            leverage=5.0,
            free_balance=50.0,
            step_size=0.01,
        )

        result = sizer.calculate(request, state)

        assert result.size == pytest.approx(2.5)
        assert result.margin_required == pytest.approx(50.0)
        assert result.capped is True
        assert result.reason is not None
        assert "free_balance_cap_applied" in result.reason
        assert_position_result_is_capital_safe(result=result, request=request)

    def test_zero_free_balance_does_not_silently_allow_positive_margin_requirement(self) -> None:
        state = make_state(free_balance=0.0)
        sizer = make_sizer()
        request = make_position_request(
            risk_amount=10.0,
            entry_price=100.0,
            stop_loss=99.0,
            leverage=5.0,
            free_balance=0.0,
        )

        with pytest.raises(InvalidPositionSizeError):
            sizer.calculate(request, state)

    def test_max_notional_rejects_position_instead_of_silently_resizing_by_notional(self) -> None:
        state = make_state()
        sizer = make_sizer(
            symbol_constraints={
                TEST_SYMBOL: SymbolConstraints(max_notional=500.0),
            }
        )
        request = make_position_request(
            risk_amount=10.0,
            entry_price=100.0,
            stop_loss=99.0,
        )

        with pytest.raises(InvalidPositionSizeError, match="max_notional"):
            sizer.calculate(request, state)


class TestPositionSizerMinimumRegression:
    def test_min_size_does_not_auto_upsize_above_risk_based_size(self) -> None:
        state = make_state()
        sizer = make_sizer()
        request = make_position_request(
            risk_amount=1.0,
            entry_price=100.0,
            stop_loss=99.0,
            min_size=2.0,
        )

        # raw size = 1.0. min_size=2.0 must DENY, not increase to 2.0.
        with pytest.raises(InvalidPositionSizeError, match="min_size"):
            sizer.calculate(request, state)

        check = sizer.check(request, state)
        assert check.passed is False
        assert check.decision is RiskDecisionType.DENY
        assert check.adjusted_size is None

    def test_symbol_min_size_does_not_auto_upsize_above_risk_based_size(self) -> None:
        state = make_state()
        sizer = make_sizer(
            symbol_constraints={
                TEST_SYMBOL: SymbolConstraints(min_size=2.0),
            }
        )
        request = make_position_request(
            risk_amount=1.0,
            entry_price=100.0,
            stop_loss=99.0,
        )

        with pytest.raises(InvalidPositionSizeError, match="min_size"):
            sizer.calculate(request, state)

    def test_global_min_position_size_does_not_auto_upsize_above_risk_based_size(self) -> None:
        state = make_state()
        sizer = make_sizer(
            config=deterministic_position_sizing_config(min_position_size=2.0),
        )
        request = make_position_request(
            risk_amount=1.0,
            entry_price=100.0,
            stop_loss=99.0,
        )

        with pytest.raises(InvalidPositionSizeError, match="min_size"):
            sizer.calculate(request, state)

    def test_min_notional_does_not_auto_upsize_above_risk_based_size(self) -> None:
        state = make_state()
        sizer = make_sizer()
        request = make_position_request(
            risk_amount=1.0,
            entry_price=100.0,
            stop_loss=99.0,
            min_notional=200.0,
        )

        # raw size=1.0, notional=100. min_notional=200 must DENY,
        # not increase size to 2.0 and double actual risk.
        with pytest.raises(InvalidPositionSizeError, match="min_notional"):
            sizer.calculate(request, state)

        check = sizer.check(request, state)
        assert check.passed is False
        assert check.decision is RiskDecisionType.DENY
        assert check.adjusted_size is None

    def test_symbol_min_notional_does_not_auto_upsize_above_risk_based_size(self) -> None:
        state = make_state()
        sizer = make_sizer(
            symbol_constraints={
                TEST_SYMBOL: SymbolConstraints(min_notional=200.0),
            }
        )
        request = make_position_request(
            risk_amount=1.0,
            entry_price=100.0,
            stop_loss=99.0,
        )

        with pytest.raises(InvalidPositionSizeError, match="min_notional"):
            sizer.calculate(request, state)

    def test_minimums_that_are_already_satisfied_pass_without_size_increase(self) -> None:
        state = make_state()
        sizer = make_sizer()
        request = make_position_request(
            risk_amount=10.0,
            entry_price=100.0,
            stop_loss=99.0,
            min_size=1.0,
            min_notional=100.0,
        )

        result = sizer.calculate(request, state)

        assert result.size == pytest.approx(10.0)
        assert result.notional_value == pytest.approx(1_000.0)
        assert_position_result_is_capital_safe(result=result, request=request)


class TestPositionSizerNormalization:
    def test_normalize_size_without_constraints_returns_raw_size(self) -> None:
        sizer = make_sizer()

        size, capped, reasons = sizer.normalize_size(TEST_SYMBOL, 10.123)

        assert size == pytest.approx(10.123)
        assert capped is False
        assert reasons == []

    def test_normalize_size_applies_max_size_before_step_rounding(self) -> None:
        sizer = make_sizer()
        constraints = SymbolConstraints(max_size=10.0, step_size=0.3)

        size, capped, reasons = sizer.normalize_size(TEST_SYMBOL, 10.9, constraints)

        assert size == pytest.approx(9.9)
        assert capped is True
        assert "max_size_cap_applied" in reasons
        assert "step_size_round_down_applied" in reasons
        assert size <= 10.0

    def test_normalize_size_negative_raw_size_becomes_zero_without_exception(self) -> None:
        sizer = make_sizer()

        size, capped, reasons = sizer.normalize_size(TEST_SYMBOL, -1.0)

        assert size == pytest.approx(0.0)
        assert capped is False
        assert reasons == []


class TestPositionSizerAdversarialInvariants:
    def test_actual_risk_never_exceeds_requested_risk_over_many_stop_distances_and_steps(
        self,
    ) -> None:
        state = make_state()
        sizer = make_sizer()

        stop_distances = [0.01, 0.05, 0.1, 0.333, 0.5, 1.0, 2.5, 10.0]
        step_sizes = [None, 0.0001, 0.001, 0.01, 0.1, 1.0]

        for stop_distance in stop_distances:
            for step_size in step_sizes:
                request = make_position_request(
                    side=PositionSide.LONG,
                    entry_price=100.0,
                    stop_loss=100.0 - stop_distance,
                    risk_amount=10.0,
                    leverage=10.0,
                    step_size=step_size,
                )

                result = sizer.calculate(request, state)

                assert_position_result_is_capital_safe(result=result, request=request)

    def test_check_never_returns_passed_with_non_positive_adjusted_size(self) -> None:
        state = make_state()
        sizer = make_sizer()
        request = make_position_request(
            risk_amount=0.0001,
            entry_price=100.0,
            stop_loss=99.0,
            step_size=1.0,
        )

        result = sizer.check(request, state)

        assert result.passed is False
        assert result.decision is RiskDecisionType.DENY
        assert result.adjusted_size is None

    def test_non_finite_output_is_never_acceptable_from_sizer(self) -> None:
        state = make_state()
        sizer = make_sizer()
        request = make_position_request(
            risk_amount=10.0,
            entry_price=100.0,
            stop_loss=99.0,
            leverage=5.0,
        )

        result = sizer.calculate(request, state)

        for field_name in (
            "size",
            "notional_value",
            "margin_required",
            "risk_amount",
            "risk_unit_used",
            "risk_units_used",
            "leverage_used",
        ):
            assert math.isfinite(float(getattr(result, field_name))), field_name

    def test_nan_step_size_should_not_be_accepted_as_valid_exchange_constraint(self) -> None:
        state = make_state()
        sizer = make_sizer()
        request = make_position_request(
            risk_amount=10.0,
            entry_price=100.0,
            stop_loss=99.0,
            step_size=math.nan,
        )

        with pytest.raises((ValueError, InvalidPositionSizeError, InvalidRiskRequestError)):
            sizer.calculate(request, state)

    def test_nan_min_size_should_not_be_accepted_as_valid_exchange_constraint(self) -> None:
        state = make_state()
        sizer = make_sizer()
        request = make_position_request(
            risk_amount=10.0,
            entry_price=100.0,
            stop_loss=99.0,
            min_size=math.nan,
        )

        with pytest.raises((ValueError, InvalidPositionSizeError, InvalidRiskRequestError)):
            sizer.calculate(request, state)

    def test_nan_min_notional_should_not_be_accepted_as_valid_exchange_constraint(self) -> None:
        state = make_state()
        sizer = make_sizer()
        request = make_position_request(
            risk_amount=10.0,
            entry_price=100.0,
            stop_loss=99.0,
            min_notional=math.nan,
        )

        with pytest.raises((ValueError, InvalidPositionSizeError, InvalidRiskRequestError)):
            sizer.calculate(request, state)