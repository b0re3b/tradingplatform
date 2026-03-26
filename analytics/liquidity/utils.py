from __future__ import annotations

from statistics import mean
from typing import Iterable, Sequence


def safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(mean(values))


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def pct_distance(price_a: float, price_b: float) -> float:
    """
    Абсолютна процентна відстань між двома цінами.
    """
    if price_b == 0:
        return 0.0
    return abs(price_a - price_b) / abs(price_b)


def signed_pct_distance(from_price: float, to_price: float) -> float:
    """
    Знакова процентна відстань від from_price до to_price.
    """
    if from_price == 0:
        return 0.0
    return (to_price - from_price) / abs(from_price)


def is_within_tolerance(price_a: float, price_b: float, tolerance_pct: float) -> bool:
    return pct_distance(price_a, price_b) <= tolerance_pct


def midpoint(low: float, high: float) -> float:
    return (low + high) / 2.0


def merge_price_ranges(
    ranges: Iterable[tuple[float, float]],
    merge_distance_pct: float = 0.0,
) -> list[tuple[float, float]]:
    """
    Об'єднує price ranges, які перетинаються або майже торкаються.
    """
    normalized: list[tuple[float, float]] = []
    for low, high in ranges:
        if low > high:
            low, high = high, low
        normalized.append((low, high))

    if not normalized:
        return []

    normalized.sort(key=lambda item: item[0])
    merged: list[list[float]] = [[normalized[0][0], normalized[0][1]]]

    for low, high in normalized[1:]:
        current_low, current_high = merged[-1]
        allowed_gap = current_high * merge_distance_pct if current_high != 0 else 0.0

        if low <= current_high + allowed_gap:
            merged[-1][1] = max(current_high, high)
        else:
            merged.append([low, high])

    return [(low, high) for low, high in merged]


def infer_liquidity_side_from_level_type(level_type: str) -> str:
    """
    Спрощена евристика для первинної класифікації сторони ліквідності.
    """
    if level_type in {"equal_highs", "swing_high", "range_high"}:
        return "buy_side"
    if level_type in {"equal_lows", "swing_low", "range_low"}:
        return "sell_side"
    return "unknown"


def calculate_range_width_pct(low: float, high: float, reference_price: float | None = None) -> float:
    if reference_price is None:
        reference_price = midpoint(low, high)

    if reference_price == 0:
        return 0.0

    return abs(high - low) / abs(reference_price)


def normalize_confidence(raw_score: float) -> float:
    return clamp(raw_score, 0.0, 1.0)


def pick_nearest_above(prices: Sequence[float], current_price: float) -> float | None:
    candidates = [price for price in prices if price > current_price]
    if not candidates:
        return None
    return min(candidates)


def pick_nearest_below(prices: Sequence[float], current_price: float) -> float | None:
    candidates = [price for price in prices if price < current_price]
    if not candidates:
        return None
    return max(candidates)


def is_pivot_high(highs: Sequence[float], index: int, left: int, right: int) -> bool:
    if index - left < 0 or index + right >= len(highs):
        return False

    pivot_value = highs[index]
    for i in range(index - left, index + right + 1):
        if i == index:
            continue
        if highs[i] >= pivot_value:
            return False

    return True


def is_pivot_low(lows: Sequence[float], index: int, left: int, right: int) -> bool:
    if index - left < 0 or index + right >= len(lows):
        return False

    pivot_value = lows[index]
    for i in range(index - left, index + right + 1):
        if i == index:
            continue
        if lows[i] <= pivot_value:
            return False

    return True


def calculate_atr_from_ohlc(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float:
    """
    Простий ATR для адаптивних tolerance / zone width.
    """
    if len(highs) != len(lows) or len(lows) != len(closes):
        raise ValueError("highs, lows, closes must have the same length")

    if len(highs) < 2:
        return 0.0

    true_ranges: list[float] = []
    for i in range(1, len(highs)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        true_ranges.append(tr)

    if not true_ranges:
        return 0.0

    if len(true_ranges) < period:
        return safe_mean(true_ranges)

    return safe_mean(true_ranges[-period:])