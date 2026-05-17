from __future__ import annotations

import math
from typing import Final

from risk.enums import PositionSide


_EPSILON: Final[float] = 1e-12


def is_finite_number(value: float | int | None) -> bool:
    """
    Check that value is a finite int/float.

    This helper is intentionally strict: None, NaN and +/-inf are not valid
    runtime risk inputs. Risk code should reject them before they can reach
    sizing, exposure or PnL calculations.
    """
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _require_finite_number(value: float | int, field_name: str) -> float:
    """Return value as float or raise ValueError for NaN/inf/non-numeric input."""
    if not is_finite_number(value):
        raise ValueError(f"{field_name} must be finite")
    return float(value)


def _validate_positive_number(value: float | int, field_name: str) -> float:
    value_f = _require_finite_number(value, field_name)
    if value_f <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return value_f


def _validate_non_negative_number(value: float | int, field_name: str) -> float:
    value_f = _require_finite_number(value, field_name)
    if value_f < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return value_f


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp value into [min_value, max_value].

    Pure helper. No side effects.
    """
    value_f = _require_finite_number(value, "value")
    min_f = _require_finite_number(min_value, "min_value")
    max_f = _require_finite_number(max_value, "max_value")

    if min_f > max_f:
        raise ValueError("min_value must be <= max_value")
    return max(min_f, min(value_f, max_f))


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division helper.

    Returns default when denominator is zero.
    """
    numerator_f = _require_finite_number(numerator, "numerator")
    denominator_f = _require_finite_number(denominator, "denominator")
    default_f = _require_finite_number(default, "default")

    if denominator_f == 0:
        return default_f
    return numerator_f / denominator_f


def calculate_pct(value: float, base: float, default: float = 0.0) -> float:
    """
    Calculate value / base with zero-safe denominator.
    """
    return safe_div(value, base, default=default)


def calculate_drawdown_pct(current_equity: float, peak_equity: float) -> float:
    """
    Calculate drawdown percent from peak equity.

    Returns 0 if peak_equity is not positive.
    """
    current_equity_f = _require_finite_number(current_equity, "current_equity")
    peak_equity_f = _require_finite_number(peak_equity, "peak_equity")

    if peak_equity_f <= 0:
        return 0.0

    drawdown = max(0.0, peak_equity_f - current_equity_f)
    return drawdown / peak_equity_f


def calculate_loss_r(loss_amount: float, risk_unit: float, default: float = 0.0) -> float:
    """
    Convert absolute loss amount to R units.

    loss_amount may be positive or negative. The returned value is always
    non-negative and represents loss magnitude in R.
    """
    loss_amount_f = _require_finite_number(loss_amount, "loss_amount")
    return safe_div(abs(min(0.0, loss_amount_f)), risk_unit, default=default)


def calculate_r_units(amount: float, risk_unit: float, default: float = 0.0) -> float:
    """
    Convert any absolute amount into R units by magnitude.
    """
    amount_f = _require_finite_number(amount, "amount")
    return safe_div(abs(amount_f), risk_unit, default=default)


def calculate_stop_distance(entry_price: float, stop_loss: float | None) -> float | None:
    """
    Backward-compatible absolute stop distance.

    Prefer calculate_side_aware_stop_distance() for new risk logic.
    """
    entry_price_f = _validate_positive_number(entry_price, "entry_price")
    if stop_loss is None:
        return None
    stop_loss_f = _validate_positive_number(stop_loss, "stop_loss")
    return abs(entry_price_f - stop_loss_f)


def calculate_side_aware_stop_distance(
    *,
    side: PositionSide,
    entry_price: float,
    stop_loss: float | None,
) -> float | None:
    """
    Calculate stop distance with side validation.

    LONG requires stop_loss < entry_price.
    SHORT requires stop_loss > entry_price.

    Returns:
        None if stop_loss is missing.

    Raises:
        ValueError if prices are invalid, non-finite or stop is on the wrong side.
    """
    entry_price_f = _validate_positive_number(entry_price, "entry_price")

    if stop_loss is None:
        return None

    stop_loss_f = _validate_positive_number(stop_loss, "stop_loss")

    if side is PositionSide.LONG:
        if stop_loss_f >= entry_price_f:
            raise ValueError("LONG stop_loss must be below entry_price")
        return entry_price_f - stop_loss_f

    if side is PositionSide.SHORT:
        if stop_loss_f <= entry_price_f:
            raise ValueError("SHORT stop_loss must be above entry_price")
        return stop_loss_f - entry_price_f

    raise ValueError(f"Unsupported position side: {side!r}")


def calculate_reward_distance(
    *,
    side: PositionSide,
    entry_price: float,
    take_profit: float | None,
) -> float | None:
    """
    Calculate reward distance with side validation.

    LONG requires take_profit > entry_price.
    SHORT requires take_profit < entry_price.

    Returns:
        None if take_profit is missing.

    Raises:
        ValueError if prices are invalid, non-finite or take profit is on the wrong side.
    """
    entry_price_f = _validate_positive_number(entry_price, "entry_price")

    if take_profit is None:
        return None

    take_profit_f = _validate_positive_number(take_profit, "take_profit")

    if side is PositionSide.LONG:
        if take_profit_f <= entry_price_f:
            raise ValueError("LONG take_profit must be above entry_price")
        return take_profit_f - entry_price_f

    if side is PositionSide.SHORT:
        if take_profit_f >= entry_price_f:
            raise ValueError("SHORT take_profit must be below entry_price")
        return entry_price_f - take_profit_f

    raise ValueError(f"Unsupported position side: {side!r}")


def calculate_risk_reward_ratio(
    *,
    side: PositionSide,
    entry_price: float,
    stop_loss: float | None,
    take_profit: float | None,
    default: float | None = None,
) -> float | None:
    """
    Calculate RR = reward_distance / stop_distance.

    Returns default when stop_loss or take_profit is missing.
    Raises ValueError when stop/take-profit are on the wrong side.
    """
    stop_distance = calculate_side_aware_stop_distance(
        side=side,
        entry_price=entry_price,
        stop_loss=stop_loss,
    )
    reward_distance = calculate_reward_distance(
        side=side,
        entry_price=entry_price,
        take_profit=take_profit,
    )

    if stop_distance is None or reward_distance is None:
        return default

    if stop_distance <= 0:
        raise ValueError("stop_distance must be > 0")

    return reward_distance / stop_distance


def calculate_expected_value(
    *,
    expected_reward: float,
    expected_loss: float,
    win_probability: float,
    expected_cost: float = 0.0,
) -> float:
    """
    Calculate expected value after costs.

    Formula:
        EV = p(win) * expected_reward
             - p(loss) * expected_loss
             - expected_cost

    expected_reward and expected_loss should be positive magnitudes.
    """
    expected_reward_f = _validate_non_negative_number(expected_reward, "expected_reward")
    expected_loss_f = _validate_non_negative_number(expected_loss, "expected_loss")
    expected_cost_f = _validate_non_negative_number(expected_cost, "expected_cost")

    probability = normalize_probability(win_probability)
    loss_probability = 1.0 - probability

    return probability * expected_reward_f - loss_probability * expected_loss_f - expected_cost_f


def calculate_cost_to_reward_ratio(
    *,
    expected_cost: float,
    expected_reward: float,
    default: float = math.inf,
) -> float:
    """
    Calculate expected_cost / expected_reward.

    Returns default when expected_reward is not positive.
    """
    expected_cost_f = _validate_non_negative_number(expected_cost, "expected_cost")
    expected_reward_f = _require_finite_number(expected_reward, "expected_reward")
    # math.inf is an intentional default used by callers/tests when reward <= 0.
    default_f = default if default == math.inf else _require_finite_number(default, "default")

    if expected_reward_f <= 0:
        return default_f
    return expected_cost_f / expected_reward_f


def calculate_notional(entry_price: float, size: float) -> float:
    """
    Calculate absolute notional value.
    """
    entry_price_f = _validate_positive_number(entry_price, "entry_price")
    size_f = _validate_non_negative_number(size, "size")
    return abs(entry_price_f * size_f)


def calculate_margin_required(
    entry_price: float,
    size: float,
    leverage: float | None,
) -> float:
    """
    Calculate margin required for position.

    If leverage is None, margin equals notional.
    """
    notional = calculate_notional(entry_price, size)
    if leverage is None:
        return notional
    leverage_f = _validate_positive_number(leverage, "leverage")
    return notional / leverage_f


def calculate_margin_from_notional(notional_value: float, leverage: float | None) -> float:
    """
    Calculate margin from already known notional.
    """
    notional_value_f = _validate_non_negative_number(notional_value, "notional_value")
    if leverage is None:
        return notional_value_f
    leverage_f = _validate_positive_number(leverage, "leverage")
    return notional_value_f / leverage_f


def calculate_position_size_by_risk(
    *,
    risk_amount: float,
    stop_distance: float,
) -> float:
    """
    Calculate position size from risk amount and stop distance.

    size = risk_amount / stop_distance
    """
    risk_amount_f = _validate_non_negative_number(risk_amount, "risk_amount")
    stop_distance_f = _validate_positive_number(stop_distance, "stop_distance")
    return risk_amount_f / stop_distance_f


def calculate_risk_amount_from_size(
    *,
    size: float,
    stop_distance: float,
) -> float:
    """
    Calculate risk amount from size and stop distance.
    """
    size_f = _validate_non_negative_number(size, "size")
    stop_distance_f = _validate_positive_number(stop_distance, "stop_distance")
    return size_f * stop_distance_f


def calculate_pnl(
    *,
    side: PositionSide,
    entry_price: float,
    exit_price: float,
    size: float,
) -> float:
    """
    Calculate realized/unrealized PnL for long or short position.
    """
    entry_price_f = _validate_positive_number(entry_price, "entry_price")
    exit_price_f = _validate_positive_number(exit_price, "exit_price")
    size_f = _validate_non_negative_number(size, "size")

    price_delta = exit_price_f - entry_price_f
    return price_delta * size_f * side.sign


def normalize_probability(value: float | None, default: float = 0.5) -> float:
    """
    Normalize probability to [0, 1].
    """
    if value is None:
        return _require_finite_number(default, "default")
    value_f = _require_finite_number(value, "probability")
    return clamp(value_f, 0.0, 1.0)


def normalize_confidence(confidence: float | None, default: float = 0.5) -> float:
    """
    Normalize confidence score to [0, 1].
    """
    return normalize_probability(confidence, default=default)


def apply_confidence_scale(
    value: float,
    confidence: float | None,
    scale_min: float,
    scale_max: float,
) -> float:
    """
    Scale value by normalized confidence.

    confidence=0 → scale_min
    confidence=1 → scale_max
    """
    value_f = _require_finite_number(value, "value")
    scale_min_f = _validate_non_negative_number(scale_min, "scale_min")
    scale_max_f = _require_finite_number(scale_max, "scale_max")

    if scale_max_f < scale_min_f:
        raise ValueError("scale_max must be >= scale_min")

    normalized_confidence = normalize_confidence(confidence)
    scale = scale_min_f + (scale_max_f - scale_min_f) * normalized_confidence
    return value_f * scale


def apply_volatility_scale(
    value: float,
    volatility: float | None,
    *,
    scale_min: float,
    scale_max: float,
) -> float:
    """
    Reduce value when volatility is high.

    volatility is expected as non-negative normalized value.
    Higher volatility means lower multiplier.
    """
    value_f = _require_finite_number(value, "value")
    scale_min_f = _validate_non_negative_number(scale_min, "scale_min")
    scale_max_f = _require_finite_number(scale_max, "scale_max")

    if scale_max_f < scale_min_f:
        raise ValueError("scale_max must be >= scale_min")

    if volatility is None:
        return value_f

    volatility_f = _require_finite_number(volatility, "volatility")
    if volatility_f <= 0:
        return value_f

    volatility_scale = 1.0 / (1.0 + volatility_f)
    volatility_scale = clamp(volatility_scale, scale_min_f, scale_max_f)
    return value_f * volatility_scale


def apply_cap(value: float, cap: float | None) -> float:
    """
    Apply upper cap if provided.
    """
    value_f = _require_finite_number(value, "value")
    if cap is None:
        return value_f
    cap_f = _require_finite_number(cap, "cap")
    return min(value_f, cap_f)


def round_down_to_step(value: float, step: float | None) -> float:
    """
    Round value down to exchange step size.

    This never rounds up, which is important for risk safety.
    """
    value_f = _validate_non_negative_number(value, "value")
    if step is None:
        return value_f

    step_f = _require_finite_number(step, "step")
    if step_f <= 0:
        return value_f
    return math.floor(value_f / step_f) * step_f


def coalesce_float(*values: float | None, default: float = 0.0) -> float:
    """
    Return first non-None finite float-like value.
    """
    for value in values:
        if value is not None:
            return _require_finite_number(value, "value")
    return _require_finite_number(default, "default")


# Backward-compatible alias for internal validators used by older code.
def _validate_positive_price(value: float, field_name: str) -> None:
    _validate_positive_number(value, field_name)


__all__ = [
    "apply_cap",
    "apply_confidence_scale",
    "apply_volatility_scale",
    "calculate_cost_to_reward_ratio",
    "calculate_drawdown_pct",
    "calculate_expected_value",
    "calculate_loss_r",
    "calculate_margin_from_notional",
    "calculate_margin_required",
    "calculate_notional",
    "calculate_pct",
    "calculate_pnl",
    "calculate_position_size_by_risk",
    "calculate_reward_distance",
    "calculate_r_units",
    "calculate_risk_amount_from_size",
    "calculate_risk_reward_ratio",
    "calculate_side_aware_stop_distance",
    "calculate_stop_distance",
    "clamp",
    "coalesce_float",
    "is_finite_number",
    "normalize_confidence",
    "normalize_probability",
    "round_down_to_step",
    "safe_div",
]