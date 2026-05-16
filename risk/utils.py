from __future__ import annotations

import math

from risk.enums import PositionSide


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp value into [min_value, max_value].

    Pure helper. No side effects.
    """
    if min_value > max_value:
        raise ValueError("min_value must be <= max_value")
    return max(min_value, min(value, max_value))


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division helper.

    Returns default when denominator is zero.
    """
    if denominator == 0:
        return default
    return numerator / denominator


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
    if peak_equity <= 0:
        return 0.0

    drawdown = max(0.0, peak_equity - current_equity)
    return drawdown / peak_equity


def calculate_loss_r(loss_amount: float, risk_unit: float, default: float = 0.0) -> float:
    """
    Convert absolute loss amount to R units.

    loss_amount may be positive or negative. The returned value is always
    non-negative and represents loss magnitude in R.
    """
    return safe_div(abs(min(0.0, loss_amount)), risk_unit, default=default)


def calculate_r_units(amount: float, risk_unit: float, default: float = 0.0) -> float:
    """
    Convert any absolute amount into R units by magnitude.
    """
    return safe_div(abs(amount), risk_unit, default=default)


def calculate_stop_distance(entry_price: float, stop_loss: float | None) -> float | None:
    """
    Backward-compatible absolute stop distance.

    Prefer calculate_side_aware_stop_distance() for new risk logic.
    """
    if stop_loss is None:
        return None
    return abs(entry_price - stop_loss)


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
        ValueError if prices are invalid or stop is on the wrong side.
    """
    _validate_positive_price(entry_price, "entry_price")

    if stop_loss is None:
        return None

    _validate_positive_price(stop_loss, "stop_loss")

    if side is PositionSide.LONG:
        if stop_loss >= entry_price:
            raise ValueError("LONG stop_loss must be below entry_price")
        return entry_price - stop_loss

    if side is PositionSide.SHORT:
        if stop_loss <= entry_price:
            raise ValueError("SHORT stop_loss must be above entry_price")
        return stop_loss - entry_price

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
        ValueError if prices are invalid or take profit is on the wrong side.
    """
    _validate_positive_price(entry_price, "entry_price")

    if take_profit is None:
        return None

    _validate_positive_price(take_profit, "take_profit")

    if side is PositionSide.LONG:
        if take_profit <= entry_price:
            raise ValueError("LONG take_profit must be above entry_price")
        return take_profit - entry_price

    if side is PositionSide.SHORT:
        if take_profit >= entry_price:
            raise ValueError("SHORT take_profit must be below entry_price")
        return entry_price - take_profit

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
    if expected_reward < 0:
        raise ValueError("expected_reward must be >= 0")
    if expected_loss < 0:
        raise ValueError("expected_loss must be >= 0")
    if expected_cost < 0:
        raise ValueError("expected_cost must be >= 0")

    probability = normalize_probability(win_probability)
    loss_probability = 1.0 - probability

    return probability * expected_reward - loss_probability * expected_loss - expected_cost


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
    if expected_cost < 0:
        raise ValueError("expected_cost must be >= 0")
    if expected_reward <= 0:
        return default
    return expected_cost / expected_reward


def calculate_notional(entry_price: float, size: float) -> float:
    """
    Calculate absolute notional value.
    """
    _validate_positive_price(entry_price, "entry_price")
    if size < 0:
        raise ValueError("size must be >= 0")
    return abs(entry_price * size)


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
    if leverage <= 0:
        raise ValueError("leverage must be > 0")
    return notional / leverage


def calculate_margin_from_notional(notional_value: float, leverage: float | None) -> float:
    """
    Calculate margin from already known notional.
    """
    if notional_value < 0:
        raise ValueError("notional_value must be >= 0")
    if leverage is None:
        return notional_value
    if leverage <= 0:
        raise ValueError("leverage must be > 0")
    return notional_value / leverage


def calculate_position_size_by_risk(
    *,
    risk_amount: float,
    stop_distance: float,
) -> float:
    """
    Calculate position size from risk amount and stop distance.

    size = risk_amount / stop_distance
    """
    if risk_amount < 0:
        raise ValueError("risk_amount must be >= 0")
    if stop_distance <= 0:
        raise ValueError("stop_distance must be > 0")
    return risk_amount / stop_distance


def calculate_risk_amount_from_size(
    *,
    size: float,
    stop_distance: float,
) -> float:
    """
    Calculate risk amount from size and stop distance.
    """
    if size < 0:
        raise ValueError("size must be >= 0")
    if stop_distance <= 0:
        raise ValueError("stop_distance must be > 0")
    return size * stop_distance


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
    _validate_positive_price(entry_price, "entry_price")
    _validate_positive_price(exit_price, "exit_price")

    if size < 0:
        raise ValueError("size must be >= 0")

    price_delta = exit_price - entry_price
    return price_delta * size * side.sign


def normalize_probability(value: float | None, default: float = 0.5) -> float:
    """
    Normalize probability to [0, 1].
    """
    if value is None:
        return default
    return clamp(value, 0.0, 1.0)


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
    if scale_min < 0:
        raise ValueError("scale_min must be >= 0")
    if scale_max < scale_min:
        raise ValueError("scale_max must be >= scale_min")

    normalized_confidence = normalize_confidence(confidence)
    scale = scale_min + (scale_max - scale_min) * normalized_confidence
    return value * scale


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
    if scale_min < 0:
        raise ValueError("scale_min must be >= 0")
    if scale_max < scale_min:
        raise ValueError("scale_max must be >= scale_min")

    if volatility is None or volatility <= 0:
        return value

    volatility_scale = 1.0 / (1.0 + volatility)
    volatility_scale = clamp(volatility_scale, scale_min, scale_max)
    return value * volatility_scale


def apply_cap(value: float, cap: float | None) -> float:
    """
    Apply upper cap if provided.
    """
    if cap is None:
        return value
    return min(value, cap)


def round_down_to_step(value: float, step: float | None) -> float:
    """
    Round value down to exchange step size.

    This never rounds up, which is important for risk safety.
    """
    if step is None or step <= 0:
        return value
    if value < 0:
        raise ValueError("value must be >= 0")
    return math.floor(value / step) * step


def coalesce_float(*values: float | None, default: float = 0.0) -> float:
    """
    Return first non-None float-like value.
    """
    for value in values:
        if value is not None:
            return value
    return default


def is_finite_number(value: float | int | None) -> bool:
    """
    Check that value is a finite int/float.
    """
    if value is None:
        return False
    return math.isfinite(float(value))


def _validate_positive_price(value: float, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0")


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