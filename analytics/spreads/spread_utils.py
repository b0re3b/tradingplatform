from __future__ import annotations
from core.logger import get_logger

from collections import deque
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from math import sqrt
from typing import Iterable, Sequence

from .enums import InstrumentType, QuoteValidity, SpreadDirection, SpreadRegime
from .models import QuoteSnapshot, RollingStats


# ============================================================
# Decimal constants
# ============================================================

DECIMAL_ZERO = Decimal("0")
DECIMAL_ONE = Decimal("1")
DECIMAL_TWO = Decimal("2")
DECIMAL_100 = Decimal("100")
DECIMAL_10_000 = Decimal("10000")

DEFAULT_QUANT = Decimal("0.00000001")


# ============================================================
# Decimal helpers
# ============================================================

def to_decimal(
    value: object,
    default: Decimal | None = None,
) -> Decimal | None:
    """
    Безпечно конвертує value у Decimal.

    Не кидає exception для невалідних input values.
    """
    if value is None:
        return default

    if isinstance(value, Decimal):
        return value

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def require_decimal(
    value: object,
    *,
    field_name: str = "value",
) -> Decimal:
    """
    Конвертує value у Decimal або кидає ValueError.

    Корисно для місць, де None/default неприпустимі.
    """
    result = to_decimal(value)
    if result is None:
        raise ValueError(f"{field_name} must be a valid Decimal-compatible value")
    return result


def quantize_decimal(
    value: Decimal | None,
    quant: Decimal = DEFAULT_QUANT,
    rounding: str = ROUND_HALF_UP,
) -> Decimal | None:
    if value is None:
        return None

    try:
        return value.quantize(quant, rounding=rounding)
    except (InvalidOperation, ValueError):
        return None


def safe_abs(value: Decimal | None) -> Decimal | None:
    return abs(value) if value is not None else None


def safe_min(values: Sequence[Decimal]) -> Decimal | None:
    return min(values) if values else None


def safe_max(values: Sequence[Decimal]) -> Decimal | None:
    return max(values) if values else None


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


def clamp_decimal(
    value: Decimal,
    *,
    min_value: Decimal | None = None,
    max_value: Decimal | None = None,
) -> Decimal:
    if min_value is not None and value < min_value:
        return min_value

    if max_value is not None and value > max_value:
        return max_value

    return value


def is_positive(value: Decimal | None) -> bool:
    return value is not None and value > DECIMAL_ZERO


def is_non_negative(value: Decimal | None) -> bool:
    return value is not None and value >= DECIMAL_ZERO


# ============================================================
# Basic market math
# ============================================================

def midpoint(
    bid: Decimal | None,
    ask: Decimal | None,
) -> Decimal | None:
    if bid is None or ask is None:
        return None

    if bid <= DECIMAL_ZERO or ask <= DECIMAL_ZERO:
        return None

    if bid > ask:
        return None

    return (bid + ask) / DECIMAL_TWO


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


def abs_spread_bps(
    numerator_value: Decimal | None,
    reference_value: Decimal | None,
) -> Decimal | None:
    value = spread_bps(numerator_value, reference_value)
    return abs(value) if value is not None else None


def basis_from_prices(
    futures_price: Decimal | None,
    spot_price: Decimal | None,
) -> Decimal | None:
    if futures_price is None or spot_price is None:
        return None
    return futures_price - spot_price


def basis_pct_from_prices(
    futures_price: Decimal | None,
    spot_price: Decimal | None,
) -> Decimal | None:
    basis = basis_from_prices(futures_price, spot_price)
    return spread_pct(basis, spot_price)


def basis_bps_from_prices(
    futures_price: Decimal | None,
    spot_price: Decimal | None,
) -> Decimal | None:
    basis = basis_from_prices(futures_price, spot_price)
    return spread_bps(basis, spot_price)


def funding_adjusted_spread(
    raw_spread: Decimal | None,
    funding_rate: Decimal | None,
    notional: Decimal | None = None,
) -> Decimal | None:
    """
    Funding-adjusted spread.

    Якщо notional не передано, funding_rate трактується як absolute adjustment.
    Якщо notional передано, funding_rate трактується як rate.
    """
    if raw_spread is None:
        return None

    if funding_rate is None:
        return raw_spread

    if notional is None:
        return raw_spread - funding_rate

    return raw_spread - (notional * funding_rate)


def notional(
    price: Decimal | None,
    quantity: Decimal | None,
) -> Decimal | None:
    if price is None or quantity is None:
        return None

    if price <= DECIMAL_ZERO or quantity <= DECIMAL_ZERO:
        return None

    return price * quantity


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
    *,
    elevated_threshold: Decimal = Decimal("1.5"),
    extreme_threshold: Decimal = Decimal("2.5"),
    compressed_threshold: Decimal = Decimal("0.5"),
    dislocated_threshold: Decimal | None = None,
) -> SpreadRegime:
    if zscore is None:
        return SpreadRegime.NORMAL

    abs_zscore = abs(zscore)
    resolved_dislocated_threshold = dislocated_threshold or (extreme_threshold * Decimal("1.5"))

    if abs_zscore >= resolved_dislocated_threshold:
        return SpreadRegime.DISLOCATED

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
    """
    BTC/USDT, BTC-USDT, btc_usdt -> BTCUSDT
    """
    return symbol.replace("-", "").replace("/", "").replace("_", "").upper().strip()


def normalize_exchange(exchange: str) -> str:
    return exchange.strip().lower()


def normalize_pair_key(*parts: object) -> tuple[str, ...]:
    """
    Створює normalized tuple key для dict-cache.
    """
    normalized: list[str] = []

    for part in parts:
        if isinstance(part, InstrumentType):
            normalized.append(part.value)
        else:
            normalized.append(str(part).strip().lower())

    return tuple(normalized)


def infer_instrument_type(raw_value: str | InstrumentType | None) -> InstrumentType:
    if isinstance(raw_value, InstrumentType):
        return raw_value

    if not raw_value:
        return InstrumentType.UNKNOWN

    value = raw_value.strip().lower()

    if value in {"spot", "cash"}:
        return InstrumentType.SPOT

    if value in {
        "perp",
        "perps",
        "perpetual",
        "perpetuals",
        "swap",
        "swaps",
        "linear_perpetual",
        "inverse_perpetual",
    }:
        return InstrumentType.PERPETUAL

    if value in {
        "future",
        "futures",
        "delivery",
        "quarterly",
        "biquarterly",
        "dated_future",
        "dated_futures",
    }:
        return InstrumentType.FUTURES

    return InstrumentType.UNKNOWN


# ============================================================
# Quote validation / timing helpers
# ============================================================

def now_utc() -> datetime:
    """
    Єдина точка для naive UTC timestamp.

    Пакет наразі використовує datetime.utcnow(), тому залишаємо сумісність.
    """
    return datetime.utcnow()


def age_ms(
    timestamp: datetime,
    now: datetime | None = None,
) -> int:
    current_time = now or now_utc()
    delta = current_time - timestamp
    return max(int(delta.total_seconds() * 1000), 0)


def quote_age_ms(
    quote: QuoteSnapshot,
    now: datetime | None = None,
) -> int:
    return age_ms(quote.timestamp, now=now)


def is_quote_stale(
    quote: QuoteSnapshot,
    max_age_ms: int,
    now: datetime | None = None,
) -> bool:
    if max_age_ms < 0:
        raise ValueError("max_age_ms must be >= 0")

    return quote_age_ms(quote, now=now) > max_age_ms


def validate_quote_snapshot(
    quote: QuoteSnapshot,
    max_age_ms: int | None = None,
    now: datetime | None = None,
) -> QuoteValidity:
    """
    Валідатор quote snapshot для analyzer-ів.

    Порядок:
    - incomplete, якщо немає bid/ask;
    - invalid, якщо bid/ask <= 0 або bid > ask;
    - stale, якщо quote старіший за max_age_ms;
    - valid.
    """
    if quote.bid is None or quote.ask is None:
        return QuoteValidity.INCOMPLETE

    if quote.bid <= DECIMAL_ZERO or quote.ask <= DECIMAL_ZERO:
        return QuoteValidity.INVALID

    if quote.bid > quote.ask:
        return QuoteValidity.INVALID

    if quote.bid_size is not None and quote.bid_size < DECIMAL_ZERO:
        return QuoteValidity.INVALID

    if quote.ask_size is not None and quote.ask_size < DECIMAL_ZERO:
        return QuoteValidity.INVALID

    if max_age_ms is not None and is_quote_stale(
        quote,
        max_age_ms=max_age_ms,
        now=now,
    ):
        return QuoteValidity.STALE

    return QuoteValidity.VALID


def aligned_quotes(
    quote_a: QuoteSnapshot,
    quote_b: QuoteSnapshot,
    max_age_diff_ms: int,
) -> bool:
    if max_age_diff_ms < 0:
        raise ValueError("max_age_diff_ms must be >= 0")

    diff_ms = abs(int((quote_a.timestamp - quote_b.timestamp).total_seconds() * 1000))
    return diff_ms <= max_age_diff_ms


def quote_time_diff_ms(
    quote_a: QuoteSnapshot,
    quote_b: QuoteSnapshot,
) -> int:
    return abs(int((quote_a.timestamp - quote_b.timestamp).total_seconds() * 1000))


# ============================================================
# Statistical helpers
# ============================================================

def decimal_mean(values: Sequence[Decimal]) -> Decimal | None:
    if not values:
        return None
    return sum(values, DECIMAL_ZERO) / Decimal(len(values))


def decimal_variance(
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

    return sum((value - mean_val) ** 2 for value in values) / Decimal(len(values))


def decimal_std(
    values: Sequence[Decimal],
    mean_value: Decimal | None = None,
) -> Decimal | None:
    variance = decimal_variance(values, mean_value)
    if variance is None:
        return None

    if variance == DECIMAL_ZERO:
        return DECIMAL_ZERO

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
    if alpha <= DECIMAL_ZERO or alpha > DECIMAL_ONE:
        raise ValueError("alpha must be in range (0, 1]")

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
        min_value=safe_min(values),
        max_value=safe_max(values),
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
    Lightweight rolling-window структура для Decimal-значень.

    Використовується для:
    - spread history;
    - z-score;
    - mean reversion;
    - regime detection.

    Не містить EventBus/Scheduler/logger і не має side effects.
    """

    __slots__ = ("_values", "_ema", "_alpha")

    def __init__(
        self,
        maxlen: int,
        ema_alpha: Decimal = Decimal("0.2"),
    ) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__init__", _analytics_args)
        except Exception:
            pass
        if maxlen <= 0:
            raise ValueError("maxlen must be > 0")

        if ema_alpha <= DECIMAL_ZERO or ema_alpha > DECIMAL_ONE:
            raise ValueError("ema_alpha must be in range (0, 1]")

        self._values: deque[Decimal] = deque(maxlen=maxlen)
        self._ema: Decimal | None = None
        self._alpha = ema_alpha

    def append(self, value: Decimal) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "append", _analytics_args)
        except Exception:
            pass
        if not isinstance(value, Decimal):
            value = require_decimal(value, field_name="value")

        self._values.append(value)
        self._ema = ema_next(value, self._ema, self._alpha)

    def extend(self, values: Iterable[Decimal]) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "extend", _analytics_args)
        except Exception:
            pass
        for value in values:
            self.append(value)

    def clear(self) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "clear", _analytics_args)
        except Exception:
            pass
        self._values.clear()
        self._ema = None

    def values(self) -> list[Decimal]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "values", _analytics_args)
        except Exception:
            pass
        return list(self._values)

    def stats(self) -> RollingStats:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "stats", _analytics_args)
        except Exception:
            pass
        return build_rolling_stats(self.values(), ema_value=self._ema)

    def to_payload(self) -> dict[str, object]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_payload", _analytics_args)
        except Exception:
            pass
        stats = self.stats()
        to_payload = getattr(stats, "to_payload", None)

        return {
            "count": self.count,
            "maxlen": self.maxlen,
            "ema": str(self._ema) if self._ema is not None else None,
            "last": str(self.last) if self.last is not None else None,
            "stats": to_payload() if callable(to_payload) else stats,
        }

    @property
    def ema(self) -> Decimal | None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "ema", _analytics_args)
        except Exception:
            pass
        return self._ema

    @property
    def last(self) -> Decimal | None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "last", _analytics_args)
        except Exception:
            pass
        if not self._values:
            return None
        return self._values[-1]

    @property
    def first(self) -> Decimal | None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "first", _analytics_args)
        except Exception:
            pass
        if not self._values:
            return None
        return self._values[0]

    @property
    def count(self) -> int:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "count", _analytics_args)
        except Exception:
            pass
        return len(self._values)

    @property
    def maxlen(self) -> int | None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "maxlen", _analytics_args)
        except Exception:
            pass
        return self._values.maxlen

    @property
    def is_full(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_full", _analytics_args)
        except Exception:
            pass
        return self._values.maxlen is not None and len(self._values) >= self._values.maxlen

    @property
    def is_empty(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_empty", _analytics_args)
        except Exception:
            pass
        return not self._values

    def __len__(self) -> int:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__len__", _analytics_args)
        except Exception:
            pass
        return len(self._values)

    def __bool__(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__bool__", _analytics_args)
        except Exception:
            pass
        return bool(self._values)


__all__ = [
    "DECIMAL_ZERO",
    "DECIMAL_ONE",
    "DECIMAL_TWO",
    "DECIMAL_100",
    "DECIMAL_10_000",
    "DEFAULT_QUANT",
    "to_decimal",
    "require_decimal",
    "quantize_decimal",
    "safe_abs",
    "safe_min",
    "safe_max",
    "safe_div",
    "clamp_decimal",
    "is_positive",
    "is_non_negative",
    "midpoint",
    "spread_abs",
    "spread_pct",
    "spread_bps",
    "abs_spread_bps",
    "basis_from_prices",
    "basis_pct_from_prices",
    "basis_bps_from_prices",
    "funding_adjusted_spread",
    "notional",
    "infer_direction",
    "infer_regime",
    "normalize_symbol",
    "normalize_exchange",
    "normalize_pair_key",
    "infer_instrument_type",
    "now_utc",
    "age_ms",
    "quote_age_ms",
    "is_quote_stale",
    "validate_quote_snapshot",
    "aligned_quotes",
    "quote_time_diff_ms",
    "decimal_mean",
    "decimal_variance",
    "decimal_std",
    "compute_zscore",
    "ema_next",
    "percentile_rank",
    "build_rolling_stats",
    "RollingDecimalWindow",
]