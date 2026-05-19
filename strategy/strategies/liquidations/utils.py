# trading_system/strategy/strategies/liquidations/utils.py

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from ...enums import FeatureSource, SignalSide
from ...models import StrategyContext, clamp, ensure_aware_utc, utcnow


# =============================================================================
# Time / serialization helpers
# =============================================================================


DECIMAL_ZERO = Decimal("0")


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


def safe_decimal(value: Any, default: Decimal = DECIMAL_ZERO) -> Decimal:
    if value is None:
        return default

    if isinstance(value, Decimal):
        return value

    if isinstance(value, bool):
        return Decimal(int(value))

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def serialize_for_metadata(value: Any) -> Any:
    """
    Serialize nested values for StrategySignal.metadata.

    Final RiskReadySignalPayload conversion belongs to SignalProcessor /
    SignalBuilder, not to liquidation strategies.
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
        cascade.confidence
        cascade.intensity_score
        exhaustion.exhaustion_bias
        squeeze.confirmed
        cluster.side_imbalance_ratio
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
    Backward-compatible unwrap for analytics.liquidations.* envelopes.

    Concrete strategies should usually receive normalized StrategyContext.
    This helper is useful when generic domain_data still contains analytics
    envelopes or result-like nested payloads.
    """
    raw = dict(payload)
    inner = raw.get("payload")

    if isinstance(inner, Mapping):
        inner_dict = dict(inner)

        for key in (
            "cascade",
            "cascade_result",
            "cascade_detection",
            "cascade_detected",
            "exhaustion",
            "exhaustion_result",
            "exhaustion_detection",
            "exhaustion_detected",
            "squeeze",
            "squeeze_result",
            "squeeze_reversal",
            "cluster",
            "liquidation_cluster",
            "result",
        ):
            nested_value = inner_dict.get(key)
            if isinstance(nested_value, Mapping):
                nested = dict(nested_value)
                nested.setdefault("_envelope", raw)
                nested.setdefault("_container", inner_dict)
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
            "confirmed",
            "valid",
            "active",
        }:
            return True

        if normalized in {
            "0",
            "false",
            "no",
            "n",
            "off",
            "rejected",
            "invalid",
            "expired",
            "inactive",
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


def normalize_label(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value

    if value is None:
        return ""

    return str(value).strip().lower()


def unit_score(value: Any, default: float = 0.0) -> float:
    parsed = to_float(value, default)
    return clamp(float(parsed if parsed is not None else default), 0.0, 1.0)


def signed_score(value: Any, default: float = 0.0) -> float:
    parsed = to_float(value, default)
    return clamp(float(parsed if parsed is not None else default), -1.0, 1.0)


def abs_score(value: Any, default: float = 0.0) -> float:
    return abs(signed_score(value, default))


# =============================================================================
# StrategyContext liquidations helpers
# =============================================================================


LIQUIDATIONS_DOMAIN_ALIASES: dict[str, tuple[str, ...]] = {
    "cascade": (
        "cascade",
        "cascade_result",
        "cascade_detection",
        "cascade_detected",
        "result",
    ),
    "exhaustion": (
        "exhaustion",
        "exhaustion_result",
        "exhaustion_detection",
        "exhaustion_detected",
        "reversal_context",
    ),
    "squeeze": (
        "squeeze",
        "squeeze_result",
        "squeeze_reversal",
        "squeeze_context",
        "pending_confirmation",
    ),
    "cluster": (
        "cluster",
        "liquidation_cluster",
        "cluster_stats",
        "metadata.cluster",
    ),
    "signal": (
        "signal",
        "liquidation_signal",
        "analytics_signal",
    ),
}


def liquidations_domain(context: StrategyContext) -> dict[str, Any]:
    return dict(context.domain_dict(FeatureSource.LIQUIDATIONS))


def liquidations_item(
    context: StrategyContext,
    key: str,
    default: Any = None,
) -> Any:
    domain = liquidations_domain(context)

    if key in domain:
        return domain[key]

    for alias in LIQUIDATIONS_DOMAIN_ALIASES.get(key, ()):
        value = get_path(domain, alias, default=None)
        if value is not None:
            return value

    return default


def liquidations_path(
    context: StrategyContext,
    path: str,
    default: Any = None,
) -> Any:
    """
    Read liquidation value from StrategyContext.

    Priority:
    1. exact feature name;
    2. liquidations-prefixed feature name;
    3. liquidation domain dotted path.
    """
    if not isinstance(path, str) or not path.strip():
        return default

    normalized = path.strip()
    feature_name = (
        normalized
        if normalized.startswith("liquidations.")
        else f"liquidations.{normalized}"
    )

    if context.has_feature(normalized):
        return context.get_feature(normalized)

    if context.has_feature(feature_name):
        return context.get_feature(feature_name)

    domain = liquidations_domain(context)

    if normalized.startswith("liquidations."):
        normalized = normalized.removeprefix("liquidations.")

    return get_path(domain, normalized, default)


def liquidations_float(
    context: StrategyContext,
    path: str,
    *,
    default: float | None = None,
) -> float | None:
    return to_float(liquidations_path(context, path, default), default)


def liquidations_unit_score(
    context: StrategyContext,
    path: str,
    *,
    default: float = 0.0,
) -> float:
    return unit_score(liquidations_path(context, path, default), default)


def liquidations_signed_score(
    context: StrategyContext,
    path: str,
    *,
    default: float = 0.0,
) -> float:
    return signed_score(liquidations_path(context, path, default), default)


def liquidations_abs_score(
    context: StrategyContext,
    path: str,
    *,
    default: float = 0.0,
) -> float:
    return abs_score(liquidations_path(context, path, default), default)


def liquidations_bool(
    context: StrategyContext,
    path: str,
    *,
    default: bool = False,
) -> bool:
    return to_bool(liquidations_path(context, path, default), default)


def liquidations_str(
    context: StrategyContext,
    path: str,
    *,
    default: str | None = None,
) -> str | None:
    return to_str(liquidations_path(context, path, default), default)


def liquidations_datetime(
    context: StrategyContext,
    path: str,
    *,
    default: datetime | None = None,
) -> datetime | None:
    return parse_datetime(liquidations_path(context, path, default))


# =============================================================================
# Direction helpers
# =============================================================================


UP_VALUES: set[str] = {
    "up",
    "long",
    "bullish",
    "buy",
    "upward",
    "cascade_up",
    "liquidations_up",
    "short_liquidations",
    "shorts_liquidated",
}

DOWN_VALUES: set[str] = {
    "down",
    "short",
    "bearish",
    "sell",
    "downward",
    "cascade_down",
    "liquidations_down",
    "long_liquidations",
    "longs_liquidated",
}

UNKNOWN_VALUES: set[str] = {
    "unknown",
    "none",
    "neutral",
    "flat",
    "mixed",
}


def continuation_side_from_direction(value: Any) -> SignalSide:
    """
    Continuation side from liquidation cascade direction.

    Cascade UP   -> LONG
    Cascade DOWN -> SHORT
    """
    label = normalize_label(value)

    if not label or label in UNKNOWN_VALUES:
        return SignalSide.UNKNOWN

    if label in UP_VALUES:
        return SignalSide.LONG

    if label in DOWN_VALUES:
        return SignalSide.SHORT

    if "up" in label or label.endswith("_long") or label.startswith("long_"):
        return SignalSide.LONG

    if "down" in label or label.endswith("_short") or label.startswith("short_"):
        return SignalSide.SHORT

    return SignalSide.UNKNOWN


def reversal_side_from_direction(value: Any) -> SignalSide:
    """
    Reversal side from liquidation cascade/exhaustion direction.

    Cascade/Exhaustion UP   -> SHORT
    Cascade/Exhaustion DOWN -> LONG
    """
    continuation = continuation_side_from_direction(value)

    if continuation is SignalSide.LONG:
        return SignalSide.SHORT

    if continuation is SignalSide.SHORT:
        return SignalSide.LONG

    return SignalSide.UNKNOWN


def opposite_side(side: SignalSide) -> SignalSide:
    if side is SignalSide.LONG:
        return SignalSide.SHORT

    if side is SignalSide.SHORT:
        return SignalSide.LONG

    return SignalSide.UNKNOWN


def is_directional_side(side: SignalSide) -> bool:
    return side in {SignalSide.LONG, SignalSide.SHORT}


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


def sides_aligned(first: SignalSide, second: SignalSide) -> bool:
    return is_directional_side(first) and first is second


def sides_opposed(first: SignalSide, second: SignalSide) -> bool:
    return (
        is_directional_side(first)
        and is_directional_side(second)
        and first is opposite_side(second)
    )


# =============================================================================
# Score DTO / scoring helpers
# =============================================================================


@dataclass(slots=True)
class ScoreBreakdown:
    """
    Small reusable score DTO for concrete liquidation strategies.
    """

    score: float = 0.0
    confidence: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    confirmations: list[str] = field(default_factory=list)

    def normalize(self) -> "ScoreBreakdown":
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
# Liquidation-specific extraction helpers
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
                "metadata.probability",
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
                "intensity_score",
                "strength",
                "severity_score",
                "normalized_score",
                "metadata.score",
                "metadata.intensity_score",
            ),
            default=default,
        ),
        default,
    )


def extract_intensity_score(value: Any, *, default: float = 0.0) -> float:
    return unit_score(
        first_present(
            value,
            (
                "intensity_score",
                "intensity",
                "score",
                "metadata.intensity_score",
                "metadata.intensity",
            ),
            default=default,
        ),
        default,
    )


def extract_continuation_bias(value: Any, *, default: float = 0.0) -> float:
    return unit_score(
        first_present(
            value,
            (
                "continuation_bias",
                "trend_continuation_bias",
                "metadata.continuation_bias",
                "metadata.trend_continuation_bias",
            ),
            default=default,
        ),
        default,
    )


def extract_exhaustion_bias(value: Any, *, default: float = 0.0) -> float:
    return unit_score(
        first_present(
            value,
            (
                "exhaustion_bias",
                "reversal_bias",
                "squeeze_reversal_bias",
                "metadata.exhaustion_bias",
                "metadata.reversal_bias",
            ),
            default=default,
        ),
        default,
    )


def extract_bias_delta(value: Any, *, default: float = 0.0) -> float:
    return unit_score(
        first_present(
            value,
            (
                "bias_delta",
                "delta_bias",
                "continuation_exhaustion_delta",
                "metadata.bias_delta",
                "metadata.delta_bias",
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
                "detected_at",
                "timestamp",
                "event_time",
                "created_at",
                "updated_at",
                "metadata.detected_at",
                "metadata.timestamp",
                "_envelope.timestamp",
            ),
            default=None,
        )
    )


def extract_direction(value: Any) -> Any:
    return first_present(
        value,
        (
            "direction",
            "cascade_direction",
            "side",
            "bias",
            "metadata.direction",
            "metadata.cascade_direction",
        ),
        default=None,
    )


def extract_continuation_side(value: Any) -> SignalSide:
    return continuation_side_from_direction(extract_direction(value))


def extract_reversal_side(value: Any) -> SignalSide:
    return reversal_side_from_direction(extract_direction(value))


def extract_severity_label(value: Any, *, default: str = "unknown") -> str:
    raw = first_present(
        value,
        (
            "severity",
            "level",
            "risk_level",
            "metadata.severity",
            "metadata.level",
        ),
        default=default,
    )
    label = normalize_label(raw)
    return label or default


def severity_score(value: Any, *, default: float = 0.0) -> float:
    label = extract_severity_label(value, default="unknown")

    mapping = {
        "low": 0.25,
        "medium": 0.50,
        "moderate": 0.50,
        "high": 0.75,
        "extreme": 1.00,
        "critical": 1.00,
    }

    return mapping.get(label, default)


def extract_notional_usd(value: Any, *, default: Decimal = DECIMAL_ZERO) -> Decimal:
    return safe_decimal(
        first_present(
            value,
            (
                "total_notional_usd",
                "notional_usd",
                "notional",
                "cluster_total_notional_usd",
                "metadata.total_notional_usd",
                "metadata.notional_usd",
            ),
            default=default,
        ),
        default=default,
    )


def extract_event_count(value: Any, *, default: int = 0) -> int:
    parsed = to_int(
        first_present(
            value,
            (
                "event_count",
                "count",
                "liquidation_count",
                "metadata.event_count",
            ),
            default=default,
        ),
        default=default,
    )
    return max(0, int(parsed if parsed is not None else default))


def extract_price_range_pct(value: Any, *, default: float = 0.0) -> float:
    return max(
        0.0,
        float(
            to_float(
                first_present(
                    value,
                    (
                        "price_range_pct",
                        "range_pct",
                        "metadata.price_range_pct",
                        "metadata.range_pct",
                    ),
                    default=default,
                ),
                default=default,
            )
            or default
        ),
    )


def extract_side_imbalance_ratio(value: Any, *, default: float = 0.0) -> float:
    return unit_score(
        first_present(
            value,
            (
                "side_imbalance_ratio",
                "imbalance_ratio",
                "cluster.side_imbalance_ratio",
                "metadata.side_imbalance_ratio",
            ),
            default=default,
        ),
        default,
    )


def extract_event_imbalance_ratio(value: Any, *, default: float = 0.0) -> float:
    return unit_score(
        first_present(
            value,
            (
                "event_imbalance_ratio",
                "count_imbalance_ratio",
                "cluster.event_imbalance_ratio",
                "metadata.event_imbalance_ratio",
            ),
            default=default,
        ),
        default,
    )


def extract_acceleration_ratio(value: Any, *, default: float = 0.0) -> float:
    parsed = to_float(
        first_present(
            value,
            (
                "acceleration_ratio",
                "climax_acceleration_ratio",
                "cluster.acceleration_ratio",
                "metadata.acceleration_ratio",
            ),
            default=default,
        ),
        default=default,
    )
    return max(0.0, float(parsed if parsed is not None else default))


def extract_cluster_duration_seconds(value: Any, *, default: float = 0.0) -> float:
    parsed = to_float(
        first_present(
            value,
            (
                "cluster_duration_seconds",
                "duration_seconds",
                "cluster.duration_seconds",
                "metadata.cluster_duration_seconds",
            ),
            default=default,
        ),
        default=default,
    )
    return max(0.0, float(parsed if parsed is not None else default))


def extract_cluster_avg_notional_per_event(
    value: Any,
    *,
    default: Decimal = DECIMAL_ZERO,
) -> Decimal:
    return safe_decimal(
        first_present(
            value,
            (
                "cluster_avg_notional_per_event",
                "avg_notional_per_event",
                "cluster.avg_notional_per_event",
                "metadata.cluster_avg_notional_per_event",
            ),
            default=default,
        ),
        default=default,
    )


def extract_status_label(value: Any, *, default: str = "unknown") -> str:
    raw = first_present(
        value,
        (
            "status",
            "state",
            "metadata.status",
            "metadata.state",
        ),
        default=default,
    )
    label = normalize_label(raw)
    return label or default


def is_confirmed_status(value: Any) -> bool:
    status = extract_status_label(value)
    explicit = first_present(
        value,
        (
            "confirmed",
            "is_confirmed",
            "metadata.confirmed",
            "metadata.is_confirmed",
        ),
        default=None,
    )

    if explicit is not None:
        return to_bool(explicit)

    return status in {
        "confirmed",
        "active",
        "valid",
        "ready",
        "completed",
    }


def is_actionable_direction(value: Any) -> bool:
    return is_directional_side(extract_continuation_side(value))


def quality_filter_reason(
    value: Any,
    *,
    min_confidence: float = 0.0,
    min_intensity_score: float = 0.0,
    min_total_notional_usd: Decimal = DECIMAL_ZERO,
    min_event_count: int = 0,
    max_price_range_pct: float | None = None,
    require_confirmed: bool = False,
    require_actionable_direction: bool = True,
) -> str | None:
    """
    Common liquidation analytics quality filter.

    Returns rejection reason or None.
    """
    if require_confirmed and not is_confirmed_status(value):
        return "not_confirmed"

    if require_actionable_direction and not is_actionable_direction(value):
        return "unknown_direction"

    if extract_confidence(value) < min_confidence:
        return "confidence_below_threshold"

    if extract_intensity_score(value) < min_intensity_score:
        return "intensity_below_threshold"

    if extract_notional_usd(value) < min_total_notional_usd:
        return "notional_below_threshold"

    if extract_event_count(value) < min_event_count:
        return "event_count_below_threshold"

    if max_price_range_pct is not None:
        if extract_price_range_pct(value) > max_price_range_pct:
            return "price_range_above_threshold"

    return None


__all__ = [
    "DECIMAL_ZERO",
    "DOWN_VALUES",
    "LIQUIDATIONS_DOMAIN_ALIASES",
    "UNKNOWN_VALUES",
    "UP_VALUES",
    "ScoreBreakdown",
    "abs_score",
    "alignment_score",
    "as_dict",
    "as_mapping",
    "average_score",
    "confidence_from_components",
    "continuation_side_from_direction",
    "ensure_utc",
    "enum_value",
    "extract_acceleration_ratio",
    "extract_bias_delta",
    "extract_cluster_avg_notional_per_event",
    "extract_cluster_duration_seconds",
    "extract_confidence",
    "extract_continuation_bias",
    "extract_continuation_side",
    "extract_direction",
    "extract_event_count",
    "extract_event_imbalance_ratio",
    "extract_event_time",
    "extract_exhaustion_bias",
    "extract_intensity_score",
    "extract_notional_usd",
    "extract_price_range_pct",
    "extract_reversal_side",
    "extract_score",
    "extract_severity_label",
    "extract_side_imbalance_ratio",
    "extract_status_label",
    "first_present",
    "freshness_score",
    "get_attr_or_key",
    "get_path",
    "is_actionable_direction",
    "is_confirmed_status",
    "is_directional_side",
    "is_stale",
    "liquidations_abs_score",
    "liquidations_bool",
    "liquidations_datetime",
    "liquidations_domain",
    "liquidations_float",
    "liquidations_item",
    "liquidations_path",
    "liquidations_signed_score",
    "liquidations_str",
    "liquidations_unit_score",
    "normalize_label",
    "opposite_side",
    "parse_datetime",
    "quality_filter_reason",
    "reversal_side_from_direction",
    "safe_decimal",
    "serialize_for_metadata",
    "severity_score",
    "side_from_signed_value",
    "sides_aligned",
    "sides_opposed",
    "signed_score",
    "to_bool",
    "to_float",
    "to_int",
    "to_str",
    "unit_score",
    "unwrap_analytics_payload",
    "utc_now",
    "weighted_score",
]