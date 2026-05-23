# trading_system/strategy/strategies/funding/utils.py

from __future__ import annotations
import logging

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from ...enums import FeatureSource, SignalSide
from ...models import StrategyContext, clamp, ensure_aware_utc, utcnow


# =============================================================================
# Time / serialization helpers
# =============================================================================


def utc_now() -> datetime:
    return utcnow()


def ensure_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return ensure_aware_utc(value)


def parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        return ensure_aware_utc(value)

    if isinstance(value, (int, float, Decimal)):
        try:
            raw = float(value)
            timestamp = raw / 1000.0 if raw > 10_000_000_000 else raw
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None

        try:
            return ensure_aware_utc(datetime.fromisoformat(raw.replace("Z", "+00:00")))
        except ValueError:
            try:
                return parse_datetime(float(raw))
            except ValueError:
                return None

    return None


def serialize_for_metadata(value: Any) -> Any:
    """
    Serialize nested values for StrategySignal.metadata.

    This is not a RiskReadySignalPayload builder. Final risk-ready conversion
    belongs to SignalProcessor / SignalBuilder.
    """
    if isinstance(value, datetime):
        return ensure_aware_utc(value).isoformat()

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return serialize_for_metadata(value.to_dict())

    if isinstance(value, Mapping):
        return {
            str(key): serialize_for_metadata(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [serialize_for_metadata(item) for item in value]

    return value


# =============================================================================
# Mapping / nested payload helpers
# =============================================================================


def as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return converted

    return None


def as_dict(value: Any) -> dict[str, Any]:
    mapping = as_mapping(value)
    return dict(mapping) if mapping is not None else {}


def get_attr_or_key(value: Any, key: str, default: Any = None) -> Any:
    mapping = as_mapping(value)
    if mapping is not None:
        return mapping.get(key, default)

    return getattr(value, key, default)


def get_path(value: Any, path: str, default: Any = None) -> Any:
    """
    Read dotted path from dict-like or object-like nested data.

    Examples:
        "divergence.confidence"
        "pressure.score"
        "extreme.mean_reversion_probability"
    """
    if not isinstance(path, str) or not path.strip():
        return default

    current = value

    for part in path.split("."):
        if current is None:
            return default

        part = part.strip()
        if not part:
            return default

        current = get_attr_or_key(current, part, default=None)

    return default if current is None else current


def first_present(
    value: Any,
    paths: Sequence[str],
    *,
    default: Any = None,
) -> Any:
    for path in paths:
        item = get_path(value, path, default=None)
        if item is not None:
            return item
    return default


def unwrap_analytics_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """
    Backward-compatible unwrap for analytics.funding.* envelopes.

    Concrete strategies should receive normalized StrategyContext. This helper is
    useful for SignalNormalizer / migration adapters or for nested funding domain
    data that still contains analytics envelopes.
    """
    raw = dict(payload)
    inner = raw.get("payload")

    if isinstance(inner, Mapping):
        inner_dict = dict(inner)

        for key in (
            "snapshot",
            "statistics",
            "regime_state",
            "pressure_state",
            "extreme_event",
            "divergence_event",
            "flip_event",
            "signal",
        ):
            nested_value = inner_dict.get(key)
            if isinstance(nested_value, Mapping):
                nested = dict(nested_value)
                nested.setdefault("_envelope", raw)
                return nested

        inner_dict.setdefault("_envelope", raw)
        return inner_dict

    return raw


# =============================================================================
# Primitive conversion helpers
# =============================================================================


def to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float, Decimal)):
        return float(value)

    if isinstance(value, Enum):
        return to_float(value.value, default)

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return default

        try:
            return float(raw)
        except ValueError:
            return default

    return default


def to_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    if isinstance(value, (float, Decimal)):
        return int(value)

    if isinstance(value, Enum):
        return to_int(value.value, default)

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return default

        try:
            return int(raw)
        except ValueError:
            try:
                return int(float(raw))
            except ValueError:
                return default

    return default


def to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, Enum):
        return to_bool(value.value, default)

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {
            "1",
            "true",
            "yes",
            "y",
            "on",
            "long",
            "bullish",
            "positive",
        }:
            return True

        if normalized in {
            "0",
            "false",
            "no",
            "n",
            "off",
            "short",
            "bearish",
            "negative",
        }:
            return False

    if isinstance(value, (int, float, Decimal)):
        return bool(value)

    return default


def to_str(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default

    if isinstance(value, Enum):
        return str(value.value)

    text = str(value).strip()
    return text if text else default


def enum_value(value: Any, default: str | None = None) -> str | None:
    if isinstance(value, Enum):
        return str(value.value)
    return to_str(value, default)


def unit_score(value: Any, default: float = 0.0) -> float:
    parsed = to_float(value, default)
    return clamp(float(parsed if parsed is not None else default), 0.0, 1.0)


def signed_score(value: Any, default: float = 0.0) -> float:
    parsed = to_float(value, default)
    return clamp(float(parsed if parsed is not None else default), -1.0, 1.0)


def abs_score(value: Any, default: float = 0.0) -> float:
    return abs(signed_score(value, default))


# =============================================================================
# StrategyContext funding helpers
# =============================================================================


FUNDING_DOMAIN_ALIASES: dict[str, tuple[str, ...]] = {
    "snapshot": ("snapshot", "funding_snapshot"),
    "statistics": ("statistics", "stats", "funding_statistics"),
    "regime": ("regime", "regime_state"),
    "pressure": ("pressure", "pressure_state"),
    "extreme": ("extreme", "extreme_event"),
    "divergence": ("divergence", "divergence_event"),
    "flip": ("flip", "flip_event"),
    "signal": ("signal", "funding_signal"),
}


def funding_domain(context: StrategyContext) -> dict[str, Any]:
    return dict(context.domain_dict(FeatureSource.FUNDING))


def funding_item(
    context: StrategyContext,
    key: str,
    default: Any = None,
) -> Any:
    domain = funding_domain(context)

    if key in domain:
        return domain[key]

    for alias in FUNDING_DOMAIN_ALIASES.get(key, ()):
        if alias in domain:
            return domain[alias]

    return default


def funding_path(
    context: StrategyContext,
    path: str,
    default: Any = None,
) -> Any:
    """
    Read funding value from StrategyContext.

    Priority:
    1. exact feature name;
    2. funding-prefixed feature name;
    3. funding domain dotted path.
    """
    if not isinstance(path, str) or not path.strip():
        return default

    normalized = path.strip()
    feature_name = (
        normalized
        if normalized.startswith("funding.")
        else f"funding.{normalized}"
    )

    if context.has_feature(normalized):
        return context.get_feature(normalized)

    if context.has_feature(feature_name):
        return context.get_feature(feature_name)

    domain = funding_domain(context)

    if normalized.startswith("funding."):
        normalized = normalized.removeprefix("funding.")

    return get_path(domain, normalized, default)


def funding_float(
    context: StrategyContext,
    path: str,
    *,
    default: float | None = None,
) -> float | None:
    return to_float(funding_path(context, path, default), default)


def funding_unit_score(
    context: StrategyContext,
    path: str,
    *,
    default: float = 0.0,
) -> float:
    return unit_score(funding_path(context, path, default), default)


def funding_signed_score(
    context: StrategyContext,
    path: str,
    *,
    default: float = 0.0,
) -> float:
    return signed_score(funding_path(context, path, default), default)


def funding_abs_score(
    context: StrategyContext,
    path: str,
    *,
    default: float = 0.0,
) -> float:
    return abs_score(funding_path(context, path, default), default)


def funding_str(
    context: StrategyContext,
    path: str,
    *,
    default: str | None = None,
) -> str | None:
    return to_str(funding_path(context, path, default), default)


def funding_datetime(
    context: StrategyContext,
    path: str,
    *,
    default: datetime | None = None,
) -> datetime | None:
    return parse_datetime(funding_path(context, path, default))


# =============================================================================
# Bias / direction helpers
# =============================================================================


BULLISH_VALUES: set[str] = {
    "long",
    "bullish",
    "buy",
    "up",
    "positive_long",
    "negative_extreme_reversal",
    "bullish_divergence",
    "negative_funding_reversal",
    "short_squeeze",
    "reversion_long",
}

BEARISH_VALUES: set[str] = {
    "short",
    "bearish",
    "sell",
    "down",
    "negative_short",
    "positive_extreme_reversal",
    "bearish_divergence",
    "positive_funding_reversal",
    "long_squeeze",
    "reversion_short",
}

NEUTRAL_VALUES: set[str] = {
    "neutral",
    "flat",
    "none",
    "unknown",
    "no_bias",
    "mixed",
}


def normalize_label(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value

    if value is None:
        return ""

    return str(value).strip().lower()


def side_from_bias(value: Any) -> SignalSide:
    label = normalize_label(value)

    if not label:
        return SignalSide.UNKNOWN

    if label in BULLISH_VALUES:
        return SignalSide.LONG

    if label in BEARISH_VALUES:
        return SignalSide.SHORT

    if label in NEUTRAL_VALUES:
        return SignalSide.UNKNOWN

    if "bull" in label or label.endswith("_long") or label.startswith("long_"):
        return SignalSide.LONG

    if "bear" in label or label.endswith("_short") or label.startswith("short_"):
        return SignalSide.SHORT

    return SignalSide.UNKNOWN


def opposite_side(side: SignalSide) -> SignalSide:
    if side is SignalSide.LONG:
        return SignalSide.SHORT

    if side is SignalSide.SHORT:
        return SignalSide.LONG

    return SignalSide.UNKNOWN


def contrarian_side_from_bias(value: Any) -> SignalSide:
    return opposite_side(side_from_bias(value))


def side_from_signed_value(
    value: Any,
    *,
    positive_side: SignalSide = SignalSide.LONG,
    negative_side: SignalSide = SignalSide.SHORT,
    dead_zone: float = 0.0,
) -> SignalSide:
    parsed = to_float(value)

    if parsed is None:
        return SignalSide.UNKNOWN

    if parsed > dead_zone:
        return positive_side

    if parsed < -dead_zone:
        return negative_side

    return SignalSide.UNKNOWN


def is_directional_side(side: SignalSide) -> bool:
    return side in {SignalSide.LONG, SignalSide.SHORT}


def sides_aligned(first: SignalSide, second: SignalSide) -> bool:
    return is_directional_side(first) and first is second


def sides_opposed(first: SignalSide, second: SignalSide) -> bool:
    return (
        is_directional_side(first)
        and is_directional_side(second)
        and first is opposite_side(second)
    )


# =============================================================================
# Scoring helpers
# =============================================================================


@dataclass(slots=True)
class ScoreBreakdown:
    """
    Small reusable score DTO for concrete funding strategies.

    Concrete strategies may put this into StrategySignal.metadata after
    serialize_for_metadata().
    """
    _logger = logging.getLogger(__name__ + ".ScoreBreakdown")

    score: float = 0.0
    confidence: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    confirmations: list[str] = field(default_factory=list)

    def normalize(self) -> "ScoreBreakdown":
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ScoreBreakdown.normalize")
        self.score = unit_score(self.score)
        self.confidence = unit_score(self.confidence)
        self.components = {
            str(key): unit_score(value)
            for key, value in self.components.items()
        }
        self.weights = {
            str(key): max(0.0, float(value))
            for key, value in self.weights.items()
        }
        return self

    def to_dict(self) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ScoreBreakdown.to_dict")
        self.normalize()
        return {
            "score": self.score,
            "confidence": self.confidence,
            "components": dict(self.components),
            "weights": dict(self.weights),
            "reasons": list(self.reasons),
            "confirmations": list(self.confirmations),
        }


def weighted_score(
    values: Mapping[str, float],
    weights: Mapping[str, float],
    *,
    default: float = 0.0,
) -> float:
    total_weight = 0.0
    total_value = 0.0

    for key, weight in weights.items():
        weight_f = max(0.0, float(weight))
        if weight_f <= 0.0:
            continue

        total_weight += weight_f
        total_value += unit_score(values.get(key, default), default) * weight_f

    if total_weight <= 0.0:
        return unit_score(default)

    return unit_score(total_value / total_weight)


def average_score(*values: float | None, default: float = 0.0) -> float:
    valid = [unit_score(value, default) for value in values if value is not None]

    if not valid:
        return unit_score(default)

    return unit_score(sum(valid) / len(valid))


def confidence_from_components(
    *,
    primary: float,
    context: float = 0.0,
    confirmation: float = 0.0,
    freshness: float = 1.0,
    primary_weight: float = 0.55,
    context_weight: float = 0.25,
    confirmation_weight: float = 0.15,
    freshness_weight: float = 0.05,
) -> float:
    return weighted_score(
        {
            "primary": primary,
            "context": context,
            "confirmation": confirmation,
            "freshness": freshness,
        },
        {
            "primary": primary_weight,
            "context": context_weight,
            "confirmation": confirmation_weight,
            "freshness": freshness_weight,
        },
    )


def alignment_score(
    *,
    target_side: SignalSide,
    observed_side: SignalSide,
    score: float,
    unknown_multiplier: float = 0.5,
) -> float:
    if not is_directional_side(target_side):
        return 0.0

    score_f = unit_score(score)

    if observed_side is SignalSide.UNKNOWN:
        return unit_score(score_f * unknown_multiplier)

    if observed_side is target_side:
        return score_f

    return 0.0


def freshness_score(
    *,
    event_time: datetime | None,
    now: datetime | None = None,
    stale_after_seconds: float | None = None,
) -> float:
    if event_time is None or stale_after_seconds is None:
        return 1.0

    if stale_after_seconds <= 0:
        return 0.0

    current = ensure_aware_utc(now or utcnow())
    normalized_event_time = ensure_aware_utc(event_time)
    age = max(0.0, (current - normalized_event_time).total_seconds())

    if age <= stale_after_seconds * 0.5:
        return 1.0

    if age <= stale_after_seconds:
        return 0.5

    if age <= stale_after_seconds * 2:
        return 0.25

    return 0.0


def is_stale(
    *,
    event_time: datetime | None,
    now: datetime | None = None,
    stale_after_seconds: float | None = None,
) -> bool:
    if event_time is None or stale_after_seconds is None:
        return False

    if stale_after_seconds <= 0:
        return True

    current = ensure_aware_utc(now or utcnow())
    normalized_event_time = ensure_aware_utc(event_time)
    age = max(0.0, (current - normalized_event_time).total_seconds())
    return age > stale_after_seconds


# =============================================================================
# Funding-specific extraction helpers
# =============================================================================


def extract_confidence(value: Any, *, default: float = 0.0) -> float:
    return unit_score(
        first_present(
            value,
            (
                "confidence",
                "probability",
                "score_confidence",
                "metadata.confidence",
            ),
            default=default,
        ),
        default,
    )


def extract_score(value: Any, *, default: float = 0.0) -> float:
    return unit_score(
        first_present(
            value,
            (
                "score",
                "strength",
                "severity",
                "normalized_score",
                "metadata.score",
            ),
            default=default,
        ),
        default,
    )


def extract_signed_score(value: Any, *, default: float = 0.0) -> float:
    return signed_score(
        first_present(
            value,
            (
                "signed_score",
                "score",
                "bias_score",
                "normalized_value",
                "metadata.signed_score",
            ),
            default=default,
        ),
        default,
    )


def extract_event_time(value: Any) -> datetime | None:
    return parse_datetime(
        first_present(
            value,
            (
                "timestamp",
                "event_time",
                "created_at",
                "updated_at",
                "open_time",
                "close_time",
                "metadata.timestamp",
                "_envelope.timestamp",
            ),
            default=None,
        )
    )


def extract_bias(value: Any, *, default: Any = None) -> Any:
    return first_present(
        value,
        (
            "bias",
            "direction",
            "side",
            "signal_side",
            "expected_side",
            "pressure_direction",
            "metadata.bias",
            "metadata.direction",
        ),
        default=default,
    )


def extract_side(value: Any) -> SignalSide:
    return side_from_bias(extract_bias(value))


def extract_symbol(value: Any, *, default: str | None = None) -> str | None:
    raw = first_present(
        value,
        (
            "symbol",
            "base_symbol",
            "asset",
            "metadata.symbol",
            "_envelope.symbol",
            "snapshot.symbol",
        ),
        default=default,
    )

    if raw is None:
        return default

    symbol = str(raw).strip().upper()
    return symbol or default


def extract_exchange(value: Any, *, default: str = "unknown") -> str:
    raw = first_present(
        value,
        (
            "exchange",
            "venue",
            "metadata.exchange",
            "_envelope.exchange",
            "snapshot.exchange",
        ),
        default=default,
    )

    exchange = str(raw or default).strip().lower()
    return exchange or default


def extract_market_type(value: Any, *, default: str = "usdm_futures") -> str:
    raw = first_present(
        value,
        (
            "market_type",
            "instrument_type",
            "metadata.market_type",
            "_envelope.market_type",
            "snapshot.market_type",
        ),
        default=default,
    )

    market_type = str(raw or default).strip().lower()
    return market_type or default


def extract_timeframe(value: Any, *, default: str = "1h") -> str:
    raw = first_present(
        value,
        (
            "timeframe",
            "interval",
            "metadata.timeframe",
            "_envelope.timeframe",
            "snapshot.timeframe",
        ),
        default=default,
    )

    timeframe = str(raw or default).strip().lower()
    return timeframe or default


__all__ = [
    "BEARISH_VALUES",
    "BULLISH_VALUES",
    "FUNDING_DOMAIN_ALIASES",
    "NEUTRAL_VALUES",
    "ScoreBreakdown",
    "abs_score",
    "alignment_score",
    "as_dict",
    "as_mapping",
    "average_score",
    "confidence_from_components",
    "contrarian_side_from_bias",
    "ensure_utc",
    "enum_value",
    "extract_bias",
    "extract_confidence",
    "extract_event_time",
    "extract_exchange",
    "extract_market_type",
    "extract_score",
    "extract_side",
    "extract_signed_score",
    "extract_symbol",
    "extract_timeframe",
    "first_present",
    "freshness_score",
    "funding_abs_score",
    "funding_datetime",
    "funding_domain",
    "funding_float",
    "funding_item",
    "funding_path",
    "funding_signed_score",
    "funding_str",
    "funding_unit_score",
    "get_attr_or_key",
    "get_path",
    "is_directional_side",
    "is_stale",
    "normalize_label",
    "opposite_side",
    "parse_datetime",
    "serialize_for_metadata",
    "side_from_bias",
    "side_from_signed_value",
    "signed_score",
    "sides_aligned",
    "sides_opposed",
    "to_bool",
    "to_float",
    "to_int",
    "to_str",
    "unit_score",
    "unwrap_analytics_payload",
    "utc_now",
    "weighted_score",
]