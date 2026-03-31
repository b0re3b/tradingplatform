from __future__ import annotations

import math


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def calculate_pct(value: float, base: float, default: float = 0.0) -> float:
    return safe_div(value, base, default=default)


def calculate_drawdown_pct(current_equity: float, peak_equity: float) -> float:
    if peak_equity <= 0:
        return 0.0
    drawdown = max(0.0, peak_equity - current_equity)
    return drawdown / peak_equity


def calculate_stop_distance(entry_price: float, stop_loss: float | None) -> float | None:
    if stop_loss is None:
        return None
    return abs(entry_price - stop_loss)


def calculate_notional(entry_price: float, size: float) -> float:
    return abs(entry_price * size)


def calculate_margin_required(
    entry_price: float,
    size: float,
    leverage: float | None,
) -> float:
    notional = calculate_notional(entry_price, size)
    if leverage is None or leverage <= 0:
        return notional
    return notional / leverage


def normalize_confidence(confidence: float | None, default: float = 0.5) -> float:
    if confidence is None:
        return default
    return clamp(confidence, 0.0, 1.0)


def apply_confidence_scale(
    value: float,
    confidence: float | None,
    scale_min: float,
    scale_max: float,
) -> float:
    normalized_confidence = normalize_confidence(confidence)
    scale = scale_min + (scale_max - scale_min) * normalized_confidence
    return value * scale


def apply_cap(value: float, cap: float | None) -> float:
    if cap is None:
        return value
    return min(value, cap)


def round_down_to_step(value: float, step: float | None) -> float:
    if step is None or step <= 0:
        return value
    return math.floor(value / step) * step


def coalesce_float(*values: float | None, default: float = 0.0) -> float:
    for value in values:
        if value is not None:
            return value
    return default