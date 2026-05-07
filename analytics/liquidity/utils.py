from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from math import isfinite
from statistics import mean
from typing import Any


PriceRange = tuple[float, float]


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Безпечно приводить значення до float.

    Використовується для candle/orderbook payload-ів, які можуть приходити
    як dict, object або сирий exchange payload зі string-значеннями.
    """
    if value is None:
        return default

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not isfinite(result):
        return default

    return result


def safe_int(value: Any, default: int = 0) -> int:
    """
    Безпечно приводить значення до int.
    """
    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_mean(values: Sequence[float]) -> float:
    """
    Середнє значення з fallback на 0.0.
    """
    cleaned = [float(value) for value in values if isfinite(float(value))]
    if not cleaned:
        return 0.0
    return float(mean(cleaned))


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Обмежує value діапазоном [min_value, max_value].
    """
    if min_value > max_value:
        min_value, max_value = max_value, min_value

    value = safe_float(value, default=min_value)
    return max(min_value, min(value, max_value))


def normalize_confidence(raw_score: float) -> float:
    """
    Нормалізує confidence/score до діапазону [0.0, 1.0].
    """
    return clamp(raw_score, 0.0, 1.0)


def pct_distance(price_a: float, price_b: float) -> float:
    """
    Абсолютна процентна відстань між двома цінами.

    Формула:
        abs(price_a - price_b) / abs(price_b)
    """
    price_a = safe_float(price_a)
    price_b = safe_float(price_b)

    if price_b == 0:
        return 0.0

    return abs(price_a - price_b) / abs(price_b)


def signed_pct_distance(from_price: float, to_price: float) -> float:
    """
    Знакова процентна відстань від from_price до to_price.

    Додатне значення означає, що to_price вище from_price.
    Від'ємне — що to_price нижче from_price.
    """
    from_price = safe_float(from_price)
    to_price = safe_float(to_price)

    if from_price == 0:
        return 0.0

    return (to_price - from_price) / abs(from_price)


def is_within_tolerance(
    price_a: float,
    price_b: float,
    tolerance_pct: float,
) -> bool:
    """
    Перевіряє, чи дві ціни знаходяться в межах tolerance_pct.
    """
    tolerance_pct = max(0.0, safe_float(tolerance_pct))
    return pct_distance(price_a, price_b) <= tolerance_pct


def midpoint(low: float, high: float) -> float:
    """
    Середина між low і high.
    """
    low = safe_float(low)
    high = safe_float(high)
    return (low + high) / 2.0


def normalize_price_range(low: float, high: float) -> PriceRange:
    """
    Повертає price range у правильному порядку: (low, high).
    """
    low = safe_float(low)
    high = safe_float(high)

    if low > high:
        low, high = high, low

    return low, high


def calculate_range_width_pct(
    low: float,
    high: float,
    reference_price: float | None = None,
) -> float:
    """
    Розраховує ширину price range у процентах від reference_price.

    Якщо reference_price не передано, використовується midpoint(low, high).
    """
    low, high = normalize_price_range(low, high)

    if reference_price is None:
        reference_price = midpoint(low, high)

    reference_price = safe_float(reference_price)
    if reference_price == 0:
        return 0.0

    return abs(high - low) / abs(reference_price)


def merge_price_ranges(
    ranges: Iterable[PriceRange],
    merge_distance_pct: float = 0.0,
) -> list[PriceRange]:
    """
    Об'єднує price ranges, які перетинаються або майже торкаються.

    merge_distance_pct:
        Дозволена відстань між зонами у процентах від поточного high
        вже злитої зони.
    """
    normalized: list[PriceRange] = []

    for low, high in ranges:
        normalized_low, normalized_high = normalize_price_range(low, high)

        if normalized_low == 0.0 and normalized_high == 0.0:
            continue

        normalized.append((normalized_low, normalized_high))

    if not normalized:
        return []

    merge_distance_pct = max(0.0, safe_float(merge_distance_pct))
    normalized.sort(key=lambda item: item[0])

    merged: list[list[float]] = [[normalized[0][0], normalized[0][1]]]

    for low, high in normalized[1:]:
        current_low, current_high = merged[-1]
        allowed_gap = abs(current_high) * merge_distance_pct if current_high != 0 else 0.0

        if low <= current_high + allowed_gap:
            merged[-1][1] = max(current_high, high)
        else:
            merged.append([low, high])

    return [(low, high) for low, high in merged]


def infer_liquidity_side_from_level_type(level_type: str) -> str:
    """
    Спрощена евристика для первинної класифікації сторони ліквідності.

    Повертає string, а не LiquiditySide, щоб utils.py залишався легким
    і не створював зайвих імпортних циклів.
    """
    normalized = str(level_type).strip().lower()

    if normalized in {"equal_highs", "swing_high", "range_high"}:
        return "buy_side"

    if normalized in {"equal_lows", "swing_low", "range_low"}:
        return "sell_side"

    return "unknown"


def pick_nearest_above(
    prices: Sequence[float],
    current_price: float,
) -> float | None:
    """
    Повертає найближчу ціну вище current_price.
    """
    current_price = safe_float(current_price)
    candidates = [
        safe_float(price)
        for price in prices
        if safe_float(price) > current_price
    ]

    if not candidates:
        return None

    return min(candidates)


def pick_nearest_below(
    prices: Sequence[float],
    current_price: float,
) -> float | None:
    """
    Повертає найближчу ціну нижче current_price.
    """
    current_price = safe_float(current_price)
    candidates = [
        safe_float(price)
        for price in prices
        if safe_float(price) < current_price
    ]

    if not candidates:
        return None

    return max(candidates)


def is_pivot_high(
    highs: Sequence[float],
    index: int,
    left: int,
    right: int,
) -> bool:
    """
    Перевіряє, чи highs[index] є pivot high.

    Pivot high означає, що значення строго вище за сусідні значення
    в межах left/right window.
    """
    if left < 1 or right < 1:
        return False

    if index - left < 0 or index + right >= len(highs):
        return False

    pivot_value = safe_float(highs[index])

    for i in range(index - left, index + right + 1):
        if i == index:
            continue

        if safe_float(highs[i]) >= pivot_value:
            return False

    return True


def is_pivot_low(
    lows: Sequence[float],
    index: int,
    left: int,
    right: int,
) -> bool:
    """
    Перевіряє, чи lows[index] є pivot low.

    Pivot low означає, що значення строго нижче за сусідні значення
    в межах left/right window.
    """
    if left < 1 or right < 1:
        return False

    if index - left < 0 or index + right >= len(lows):
        return False

    pivot_value = safe_float(lows[index])

    for i in range(index - left, index + right + 1):
        if i == index:
            continue

        if safe_float(lows[i]) <= pivot_value:
            return False

    return True


def calculate_true_ranges(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> list[float]:
    """
    Розраховує True Range series для OHLC-даних.
    """
    if len(highs) != len(lows) or len(lows) != len(closes):
        raise ValueError("highs, lows, closes must have the same length")

    if len(highs) < 2:
        return []

    true_ranges: list[float] = []

    for i in range(1, len(highs)):
        high = safe_float(highs[i])
        low = safe_float(lows[i])
        prev_close = safe_float(closes[i - 1])

        true_range = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )

        true_ranges.append(max(0.0, true_range))

    return true_ranges


def calculate_atr_from_ohlc(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float:
    """
    Простий ATR для adaptive tolerance / zone width.

    Якщо даних менше ніж period, повертає середнє за доступними true ranges.
    """
    if period < 1:
        raise ValueError("period must be >= 1")

    true_ranges = calculate_true_ranges(
        highs=highs,
        lows=lows,
        closes=closes,
    )

    if not true_ranges:
        return 0.0

    if len(true_ranges) < period:
        return safe_mean(true_ranges)

    return safe_mean(true_ranges[-period:])


def calculate_atr_pct_from_ohlc(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float:
    """
    Розраховує ATR як процент від останнього close.

    Корисно для adaptive tolerance у EqualHighsLowsDetector.
    """
    atr = calculate_atr_from_ohlc(
        highs=highs,
        lows=lows,
        closes=closes,
        period=period,
    )

    if not closes:
        return 0.0

    last_close = safe_float(closes[-1])
    if last_close == 0:
        return 0.0

    return atr / abs(last_close)


def get_value(
    obj: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    Універсальний getter для dict/object candle або exchange payload.

    Спочатку пробує dict access, потім getattr.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def get_first_value(
    obj: Any,
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    """
    Повертає перше доступне значення з набору ключів.
    """
    for key in keys:
        value = get_value(obj, key, None)
        if value is not None:
            return value

    return default


def get_candle_open(candle: Any, default: float = 0.0) -> float:
    return safe_float(
        get_first_value(candle, ("open", "o")),
        default=default,
    )


def get_candle_high(candle: Any, default: float = 0.0) -> float:
    return safe_float(
        get_first_value(candle, ("high", "h")),
        default=default,
    )


def get_candle_low(candle: Any, default: float = 0.0) -> float:
    return safe_float(
        get_first_value(candle, ("low", "l")),
        default=default,
    )


def get_candle_close(candle: Any, default: float = 0.0) -> float:
    return safe_float(
        get_first_value(candle, ("close", "c", "price")),
        default=default,
    )


def get_candle_volume(candle: Any, default: float = 0.0) -> float:
    return safe_float(
        get_first_value(candle, ("volume", "v", "qty", "quantity")),
        default=default,
    )


def get_candle_timestamp(candle: Any) -> datetime | None:
    """
    Дістає timestamp зі свічки, якщо він уже є datetime.

    Якщо timestamp приходить як int/float/string, його краще нормалізувати
    у data layer до потрапляння в analytics.
    """
    value = get_first_value(
        candle,
        (
            "timestamp",
            "ts",
            "time",
            "open_time",
            "close_time",
            "openTime",
            "closeTime",
        ),
    )

    if isinstance(value, datetime):
        return value

    return None


def extract_ohlc(
    candles: Sequence[Any],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Витягує open/high/low/close списки зі свічок.
    """
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []

    for candle in candles:
        opens.append(get_candle_open(candle))
        highs.append(get_candle_high(candle))
        lows.append(get_candle_low(candle))
        closes.append(get_candle_close(candle))

    return opens, highs, lows, closes


def extract_hlc(
    candles: Sequence[Any],
) -> tuple[list[float], list[float], list[float]]:
    """
    Витягує high/low/close списки зі свічок.
    """
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []

    for candle in candles:
        highs.append(get_candle_high(candle))
        lows.append(get_candle_low(candle))
        closes.append(get_candle_close(candle))

    return highs, lows, closes


def filter_positive_prices(prices: Sequence[float]) -> list[float]:
    """
    Залишає тільки додатні валідні ціни.
    """
    return [
        price
        for price in (safe_float(value) for value in prices)
        if price > 0
    ]


def sort_unique_prices(
    prices: Sequence[float],
    tolerance_pct: float = 0.0,
) -> list[float]:
    """
    Сортує ціни і прибирає майже дублікати.

    Якщо tolerance_pct == 0, дублікати прибираються точним порівнянням.
    """
    cleaned = sorted(filter_positive_prices(prices))

    if not cleaned:
        return []

    tolerance_pct = max(0.0, safe_float(tolerance_pct))
    unique: list[float] = [cleaned[0]]

    for price in cleaned[1:]:
        previous = unique[-1]

        if tolerance_pct == 0:
            if price != previous:
                unique.append(price)
            continue

        if pct_distance(price, previous) > tolerance_pct:
            unique.append(price)

    return unique