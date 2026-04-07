from __future__ import annotations

from collections import deque
from datetime import datetime
from decimal import Decimal, InvalidOperation
from math import sqrt
from typing import Iterable, Sequence

from .enums import InstrumentType, QuoteValidity, SpreadDirection, SpreadRegime
from .models import QuoteSnapshot, RollingStats


DECIMAL_ZERO = Decimal("0")
DECIMAL_ONE = Decimal("1")
DECIMAL_100 = Decimal("100")
DECIMAL_10_000 = Decimal("10000")
DEFAULT_QUANT = Decimal("0.00000001")


# ============================================================
# Decimal helpers
# ============================================================

def to_decimal(value: object, default: Decimal | None = None) -> Decimal | None:
    if value is None:
        return default
    if isinstance(value, Decimal):
        return value

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def quantize_decimal(
    value: Decimal | None,
    quant: Decimal = DEFAULT_QUANT,
) -> Decimal | None:
    if value is None:
        return None
    return value.quantize(quant)


def safe_div(
    numerator: Decimal | None,
    denominator: Decimal | None,
    default: Decimal | None = None,
) -> Decimal | None:
    if numerator is None or denominator is None:
        return default
    if denominator == DECIMAL_ZERO:
        return default
    return numerator / denominator


# ============================================================
# Basic market math
# ============================================================

def midpoint(
    bid: Decimal | None,
    ask: Decimal | None,
) -> Decimal | None:
    if bid is None or ask is None:
        return None
    return (bid + ask) / Decimal("2")


def spread_abs(
    value_a: Decimal | None,
    value_b: Decimal | None,
) -> Decimal | None:
    if value_a is None or value_b is None:
        return None
    return value_a - value_b


def spread_pct(
    numerator_value: Decimal | None,
    reference_value: Decimal | None,
) -> Decimal | None:
    ratio = safe_div(numerator_value, reference_value)
    if ratio is None:
        return None
    return ratio * DECIMAL_100


def spread_bps(
    numerator_value: Decimal | None,
    reference_value: Decimal | None,
) -> Decimal | None:
    ratio = safe_div(numerator_value, reference_value)
    if ratio is None:
        return None
    return ratio * DECIMAL_10_000


def basis_from_prices(
    futures_price: Decimal | None,
    spot_price: Decimal | None,
) -> Decimal | None:
    if futures_price is None or spot_price is None:
        return None
    return futures_price - spot_price


def funding_adjusted_spread(
    raw_spread: Decimal | None,
    funding_rate: Decimal | None,
    notional: Decimal | None = None,
) -> Decimal | None:
    if raw_spread is None:
        return None

    if funding_rate is None:
        return raw_spread

    if notional is None:
        return raw_spread - funding_rate

    return raw_spread - (notional * funding_rate)


# ============================================================
# Spread interpretation
# ============================================================

def infer_direction(value: Decimal | None) -> SpreadDirection:
    if value is None:
        return SpreadDirection.FLAT
    if value > DECIMAL_ZERO:
        return SpreadDirection.POSITIVE
    if value < DECIMAL_ZERO:
        return SpreadDirection.NEGATIVE
    return SpreadDirection.FLAT


def infer_regime(
    zscore: Decimal | None,
    elevated_threshold: Decimal = Decimal("1.5"),
    extreme_threshold: Decimal = Decimal("2.5"),
    compressed_threshold: Decimal = Decimal("0.5"),
) -> SpreadRegime:
    if zscore is None:
        return SpreadRegime.NORMAL

    abs_zscore = abs(zscore)

    if abs_zscore >= extreme_threshold:
        return SpreadRegime.EXTREME
    if abs_zscore >= elevated_threshold:
        return SpreadRegime.ELEVATED
    if abs_zscore <= compressed_threshold:
        return SpreadRegime.COMPRESSED

    return SpreadRegime.NORMAL


# ============================================================
# Normalization helpers
# ============================================================

def normalize_symbol(symbol: str) -> str:
    return symbol.replace("-", "").replace("/", "").replace("_", "").upper().strip()


def normalize_exchange(exchange: str) -> str:
    return exchange.strip().lower()


def infer_instrument_type(raw_value: str | None) -> InstrumentType:
    if not raw_value:
        return InstrumentType.UNKNOWN

    value = raw_value.strip().lower()

    if value == "spot":
        return InstrumentType.SPOT

    if value in {"perp", "perpetual", "swap"}:
        return InstrumentType.PERPETUAL

    if value in {"future", "futures", "delivery"}:
        return InstrumentType.FUTURES

    return InstrumentType.UNKNOWN


# ============================================================
# Quote validation / timing helpers
# ============================================================

def quote_age_ms(
    quote: QuoteSnapshot,
    now: datetime | None = None,
) -> int:
    current_time = now or datetime.utcnow()
    delta = current_time - quote.timestamp
    return max(int(delta.total_seconds() * 1000), 0)


def is_quote_stale(
    quote: QuoteSnapshot,
    max_age_ms: int,
    now: datetime | None = None,
) -> bool:
    return quote_age_ms(quote, now=now) > max_age_ms


def validate_quote_snapshot(
    quote: QuoteSnapshot,
    max_age_ms: int | None = None,
    now: datetime | None = None,
) -> QuoteValidity:
    if quote.bid is None or quote.ask is None:
        return QuoteValidity.INCOMPLETE

    if quote.bid <= DECIMAL_ZERO or quote.ask <= DECIMAL_ZERO:
        return QuoteValidity.INVALID

    if quote.bid > quote.ask:
        return QuoteValidity.INVALID

    if max_age_ms is not None and is_quote_stale(quote, max_age_ms=max_age_ms, now=now):
        return QuoteValidity.STALE

    return QuoteValidity.VALID


def aligned_quotes(
    quote_a: QuoteSnapshot,
    quote_b: QuoteSnapshot,
    max_age_diff_ms: int,
) -> bool:
    diff_ms = abs(int((quote_a.timestamp - quote_b.timestamp).total_seconds() * 1000))
    return diff_ms <= max_age_diff_ms


# ============================================================
# Statistical helpers
# ============================================================

def decimal_mean(values: Sequence[Decimal]) -> Decimal | None:
    if not values:
        return None
    return sum(values, DECIMAL_ZERO) / Decimal(len(values))


def decimal_std(
    values: Sequence[Decimal],
    mean_value: Decimal | None = None,
) -> Decimal | None:
    if not values:
        return None

    if len(values) < 2:
        return DECIMAL_ZERO

    mean_val = mean_value if mean_value is not None else decimal_mean(values)
    if mean_val is None:
        return None

    variance = sum((value - mean_val) ** 2 for value in values) / Decimal(len(values))
    return Decimal(str(sqrt(float(variance))))


def compute_zscore(
    current_value: Decimal | None,
    mean_value: Decimal | None,
    std_value: Decimal | None,
) -> Decimal | None:
    if current_value is None or mean_value is None or std_value is None:
        return None

    if std_value == DECIMAL_ZERO:
        return DECIMAL_ZERO

    return (current_value - mean_value) / std_value


def ema_next(
    value: Decimal,
    previous_ema: Decimal | None,
    alpha: Decimal,
) -> Decimal:
    if previous_ema is None:
        return value

    return (alpha * value) + ((DECIMAL_ONE - alpha) * previous_ema)


def percentile_rank(
    value: Decimal,
    values: Sequence[Decimal],
) -> Decimal | None:
    if not values:
        return None

    less_or_equal = sum(1 for item in values if item <= value)
    return (Decimal(less_or_equal) / Decimal(len(values))) * DECIMAL_100


def build_rolling_stats(
    values: Sequence[Decimal],
    ema_value: Decimal | None = None,
) -> RollingStats:
    if not values:
        return RollingStats()

    mean_val = decimal_mean(values)
    std_val = decimal_std(values, mean_val)
    last_val = values[-1]
    zscore_val = compute_zscore(last_val, mean_val, std_val)
    percentile_val = percentile_rank(last_val, values)

    return RollingStats(
        count=len(values),
        mean=mean_val,
        std=std_val,
        min_value=min(values),
        max_value=max(values),
        ema=ema_value,
        last_value=last_val,
        zscore=zscore_val,
        percentile_rank=percentile_val,
    )


# ============================================================
# Rolling window
# ============================================================

class RollingDecimalWindow:
    """
    Легка rolling-window структура для Decimal-значень.
    Добре підходить для історії spread, z-score та mean reversion логіки.
    """

    __slots__ = ("_values", "_ema", "_alpha")

    def __init__(
        self,
        maxlen: int,
        ema_alpha: Decimal = Decimal("0.2"),
    ) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be > 0")

        if ema_alpha <= DECIMAL_ZERO or ema_alpha > DECIMAL_ONE:
            raise ValueError("ema_alpha must be in range (0, 1]")

        self._values: deque[Decimal] = deque(maxlen=maxlen)
        self._ema: Decimal | None = None
        self._alpha = ema_alpha

    def append(self, value: Decimal) -> None:
        self._values.append(value)
        self._ema = ema_next(value, self._ema, self._alpha)

    def extend(self, values: Iterable[Decimal]) -> None:
        for value in values:
            self.append(value)

    def clear(self) -> None:
        self._values.clear()
        self._ema = None

    def values(self) -> list[Decimal]:
        return list(self._values)

    @property
    def ema(self) -> Decimal | None:
        return self._ema

    @property
    def last(self) -> Decimal | None:
        if not self._values:
            return None
        return self._values[-1]

    @property
    def count(self) -> int:
        return len(self._values)

    def stats(self) -> RollingStats:
        return build_rolling_stats(self.values(), ema_value=self._ema)