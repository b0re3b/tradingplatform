# trading_system/strategy/strategies/open_interest/utils.py

from __future__ import annotations
import logging

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from analytics.open_interest.enums import (
    OIAnomalyType,
    OIDivergenceType,
    OIRegime,
)
from analytics.open_interest.models import (
    OIAnomalyResult,
    OIDivergenceResult,
    OIFeatures,
    OIRegimeResult,
)

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
    SignalBuilder, not to open-interest strategies.
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
        regime.confidence
        divergence.divergence_type
        anomaly.anomaly_type
        features.oi_delta_pct
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
    Backward-compatible unwrap for analytics.open_interest.* envelopes.

    Concrete strategies should usually receive normalized StrategyContext.
    This helper is useful when generic domain_data still contains analytics
    envelopes or result-like nested payloads.
    """
    raw = dict(payload)
    inner = raw.get("payload")

    if isinstance(inner, Mapping):
        inner_dict = dict(inner)

        for key in (
            "analysis",
            "oi_analysis",
            "open_interest_analysis",
            "result",
            "snapshot",
            "context",
            "market_context",
            "features",
            "regime",
            "regime_result",
            "divergence",
            "divergence_result",
            "anomaly",
            "anomaly_result",
            "signal",
            "oi_signal",
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
            "detected",
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
            "none",
            "not_detected",
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
# Enum parsing helpers
# =============================================================================


def parse_oi_regime(value: Any, default: OIRegime = OIRegime.NEUTRAL) -> OIRegime:
    if isinstance(value, OIRegime):
        return value

    if value is None:
        return default

    text = str(value).strip()
    if not text:
        return default

    for item in OIRegime:
        if item.value.lower() == text.lower() or item.name.lower() == text.lower():
            return item

    return default


def parse_divergence_type(
    value: Any,
    default: OIDivergenceType = OIDivergenceType.NONE,
) -> OIDivergenceType:
    if isinstance(value, OIDivergenceType):
        return value

    if value is None:
        return default

    text = str(value).strip()
    if not text:
        return default

    for item in OIDivergenceType:
        if item.value.lower() == text.lower() or item.name.lower() == text.lower():
            return item

    return default


def parse_anomaly_type(
    value: Any,
    default: OIAnomalyType = OIAnomalyType.NONE,
) -> OIAnomalyType:
    if isinstance(value, OIAnomalyType):
        return value

    if value is None:
        return default

    text = str(value).strip()
    if not text:
        return default

    for item in OIAnomalyType:
        if item.value.lower() == text.lower() or item.name.lower() == text.lower():
            return item

    return default


# =============================================================================
# StrategyContext open-interest helpers
# =============================================================================


OPEN_INTEREST_DOMAIN_ALIASES: dict[str, tuple[str, ...]] = {
    "analysis": (
        "analysis",
        "oi_analysis",
        "open_interest_analysis",
        "result",
    ),
    "snapshot": (
        "snapshot",
        "oi_snapshot",
        "open_interest_snapshot",
        "analysis.snapshot",
    ),
    "market_context": (
        "context",
        "market_context",
        "oi_context",
        "open_interest_context",
        "analysis.context",
    ),
    "features": (
        "features",
        "oi_features",
        "open_interest_features",
        "analysis.features",
    ),
    "regime": (
        "regime",
        "regime_result",
        "oi_regime",
        "open_interest_regime",
        "new_regime",
        "analysis.regime",
    ),
    "divergence": (
        "divergence",
        "divergence_result",
        "oi_divergence",
        "open_interest_divergence",
        "analysis.divergence",
    ),
    "anomaly": (
        "anomaly",
        "anomaly_result",
        "oi_anomaly",
        "open_interest_anomaly",
        "analysis.anomaly",
    ),
    "signal": (
        "signal",
        "oi_signal",
        "open_interest_signal",
        "analytics_signal",
    ),
}


def open_interest_domain(context: StrategyContext) -> dict[str, Any]:
    return dict(context.domain_dict(FeatureSource.OPEN_INTEREST))


def open_interest_item(
    context: StrategyContext,
    key: str,
    default: Any = None,
) -> Any:
    domain = open_interest_domain(context)

    if key in domain:
        return domain[key]

    for alias in OPEN_INTEREST_DOMAIN_ALIASES.get(key, ()):
        value = get_path(domain, alias, default=None)
        if value is not None:
            return value

    return default


def open_interest_path(
    context: StrategyContext,
    path: str,
    default: Any = None,
) -> Any:
    """
    Read open-interest value from StrategyContext.

    Priority:
    1. exact feature name;
    2. open_interest-prefixed feature name;
    3. legacy oi-prefixed feature name;
    4. open-interest domain dotted path.
    """
    if not isinstance(path, str) or not path.strip():
        return default

    normalized = path.strip()

    open_interest_feature_name = (
        normalized
        if normalized.startswith("open_interest.")
        else f"open_interest.{normalized}"
    )
    legacy_feature_name = (
        normalized
        if normalized.startswith("oi.")
        else f"oi.{normalized}"
    )

    if context.has_feature(normalized):
        return context.get_feature(normalized)

    if context.has_feature(open_interest_feature_name):
        return context.get_feature(open_interest_feature_name)

    if context.has_feature(legacy_feature_name):
        return context.get_feature(legacy_feature_name)

    domain = open_interest_domain(context)

    if normalized.startswith("open_interest."):
        normalized = normalized.removeprefix("open_interest.")
    elif normalized.startswith("oi."):
        normalized = normalized.removeprefix("oi.")

    return get_path(domain, normalized, default)


def open_interest_float(
    context: StrategyContext,
    path: str,
    *,
    default: float | None = None,
) -> float | None:
    return to_float(open_interest_path(context, path, default), default)


def open_interest_unit_score(
    context: StrategyContext,
    path: str,
    *,
    default: float = 0.0,
) -> float:
    return unit_score(open_interest_path(context, path, default), default)


def open_interest_signed_score(
    context: StrategyContext,
    path: str,
    *,
    default: float = 0.0,
) -> float:
    return signed_score(open_interest_path(context, path, default), default)


def open_interest_abs_score(
    context: StrategyContext,
    path: str,
    *,
    default: float = 0.0,
) -> float:
    return abs_score(open_interest_path(context, path, default), default)


def open_interest_bool(
    context: StrategyContext,
    path: str,
    *,
    default: bool = False,
) -> bool:
    return to_bool(open_interest_path(context, path, default), default)


def open_interest_str(
    context: StrategyContext,
    path: str,
    *,
    default: str | None = None,
) -> str | None:
    return to_str(open_interest_path(context, path, default), default)


def open_interest_datetime(
    context: StrategyContext,
    path: str,
    *,
    default: datetime | None = None,
) -> datetime | None:
    return parse_datetime(open_interest_path(context, path, default))


# =============================================================================
# OI result extractors
# =============================================================================


def extract_regime_type(value: Any) -> OIRegime:
    raw = first_present(
        value,
        (
            "regime",
            "type",
            "regime_type",
            "new_regime",
            "oi_regime",
            "open_interest_regime",
        ),
    )
    return parse_oi_regime(raw)


def extract_divergence_type(value: Any) -> OIDivergenceType:
    raw = first_present(
        value,
        (
            "divergence_type",
            "type",
            "oi_divergence_type",
            "open_interest_divergence_type",
        ),
    )
    return parse_divergence_type(raw)


def extract_anomaly_type(value: Any) -> OIAnomalyType:
    raw = first_present(
        value,
        (
            "anomaly_type",
            "type",
            "oi_anomaly_type",
            "open_interest_anomaly_type",
        ),
    )
    return parse_anomaly_type(raw)


def extract_detected(value: Any, *, default: bool = False) -> bool:
    return to_bool(
        first_present(
            value,
            (
                "detected",
                "confirmed",
                "is_detected",
                "is_confirmed",
                "active",
                "valid",
            ),
            default=default,
        ),
        default,
    )


def extract_confidence(value: Any, *, default: float = 0.0) -> float:
    return unit_score(
        first_present(
            value,
            (
                "confidence",
                "regime_confidence",
                "divergence_confidence",
                "anomaly_confidence",
                "analysis_confidence",
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
                "regime_score",
                "divergence_score",
                "anomaly_score",
                "normalized_score",
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
                "event_time",
                "detected_at",
                "timestamp",
                "created_at",
                "updated_at",
                "time",
            ),
        )
    )


def extract_window_size(value: Any) -> int | None:
    return to_int(
        first_present(
            value,
            (
                "window_size",
                "lookback",
                "lookback_periods",
                "periods",
            ),
        )
    )


def extract_reasons(value: Any) -> list[str]:
    raw = first_present(value, ("reasons", "reason", "messages", "notes"), default=[])

    if raw is None:
        return []

    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []

    if isinstance(raw, (list, tuple, set)):
        result: list[str] = []
        for item in raw:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                result.append(text)
        return list(dict.fromkeys(result))

    text = str(raw).strip()
    return [text] if text else []


def extract_oi_delta_pct(value: Any) -> float:
    return signed_score(
        first_present(
            value,
            (
                "oi_delta_pct",
                "open_interest_delta_pct",
                "features.oi_delta_pct",
                "features.open_interest_delta_pct",
            ),
            default=0.0,
        )
    )


def extract_price_delta_pct(value: Any) -> float:
    return signed_score(
        first_present(
            value,
            (
                "price_delta_pct",
                "features.price_delta_pct",
                "price_change_pct",
                "features.price_change_pct",
            ),
            default=0.0,
        )
    )


def extract_oi_pressure_score(value: Any) -> float:
    return signed_score(
        first_present(
            value,
            (
                "oi_pressure_score",
                "features.oi_pressure_score",
                "pressure_score",
                "features.pressure_score",
            ),
            default=0.0,
        )
    )


def extract_aggressive_flow_imbalance(value: Any) -> float:
    return signed_score(
        first_present(
            value,
            (
                "aggressive_flow_imbalance",
                "features.aggressive_flow_imbalance",
                "flow_imbalance",
                "features.flow_imbalance",
            ),
            default=0.0,
        )
    )


def extract_funding_rate(value: Any) -> float | None:
    return to_float(
        first_present(
            value,
            (
                "funding_rate",
                "features.funding_rate",
                "funding",
                "features.funding",
            ),
            default=None,
        )
    )


def extract_liquidation_pressure(value: Any) -> float:
    return signed_score(
        first_present(
            value,
            (
                "liquidation_pressure",
                "features.liquidation_pressure",
                "liq_pressure",
                "features.liq_pressure",
            ),
            default=0.0,
        )
    )


# =============================================================================
# Direction helpers
# =============================================================================


LONG_VALUES: set[str] = {
    "long",
    "buy",
    "bull",
    "bullish",
    "up",
    "upside",
    "positive",
    "long_buildup",
    "short_covering",
}

SHORT_VALUES: set[str] = {
    "short",
    "sell",
    "bear",
    "bearish",
    "down",
    "downside",
    "negative",
    "short_buildup",
    "long_unwind",
}

UNKNOWN_VALUES: set[str] = {
    "unknown",
    "none",
    "neutral",
    "flat",
    "mixed",
}


def parse_side(value: Any) -> SignalSide:
    if isinstance(value, SignalSide):
        return value

    label = normalize_label(value)
    if not label or label in UNKNOWN_VALUES:
        return SignalSide.UNKNOWN

    if label in LONG_VALUES:
        return SignalSide.LONG

    if label in SHORT_VALUES:
        return SignalSide.SHORT

    if "bull" in label or label.startswith("long_") or label.endswith("_long"):
        return SignalSide.LONG

    if "bear" in label or label.startswith("short_") or label.endswith("_short"):
        return SignalSide.SHORT

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


def regime_side_hint(regime: OIRegime | Any) -> str:
    """
    Return semantic side hint from OIRegime.

    Returns:
        bullish | bearish | contextual | neutral
    """
    parsed = parse_oi_regime(regime)
    label = normalize_label(parsed)

    if label in {
        "long_buildup",
        "short_covering",
        "bullish_oi_expansion",
    }:
        return "bullish"

    if label in {
        "short_buildup",
        "long_unwind",
        "bearish_oi_expansion",
    }:
        return "bearish"

    if label in {
        "trend_confirmation",
        "squeeze_setup",
        "capitulation",
        "trend_exhaustion",
        "overheated",
    }:
        return "contextual"

    return "neutral"


def side_from_oi_regime(
    regime: OIRegime | Any,
    *,
    features: OIFeatures | Any | None = None,
) -> SignalSide:
    hint = regime_side_hint(regime)

    if hint == "bullish":
        return SignalSide.LONG

    if hint == "bearish":
        return SignalSide.SHORT

    if hint == "contextual":
        return side_from_oi_features(features)

    return SignalSide.UNKNOWN


def divergence_side_hint(divergence_type: OIDivergenceType | Any) -> str:
    """
    Return semantic side hint from OIDivergenceType.

    Uses enum properties when analytics.open_interest provides them, with
    label-based fallback for compatibility.
    """
    parsed = parse_divergence_type(divergence_type)

    is_bullish = getattr(parsed, "is_bullish_context", None)
    if is_bullish is True:
        return "bullish"

    is_bearish = getattr(parsed, "is_bearish_context", None)
    if is_bearish is True:
        return "bearish"

    label = normalize_label(parsed)

    if not label or label in UNKNOWN_VALUES:
        return "neutral"

    if "bull" in label or "positive" in label or "long" in label:
        return "bullish"

    if "bear" in label or "negative" in label or "short" in label:
        return "bearish"

    if "exhaustion" in label or "reversal" in label:
        return "contextual"

    return "neutral"


def side_from_oi_divergence(
    divergence_type: OIDivergenceType | Any,
    *,
    features: OIFeatures | Any | None = None,
) -> SignalSide:
    hint = divergence_side_hint(divergence_type)

    if hint == "bullish":
        return SignalSide.LONG

    if hint == "bearish":
        return SignalSide.SHORT

    if hint == "contextual":
        feature_side = side_from_oi_features(features)
        return opposite_side(feature_side)

    return SignalSide.UNKNOWN


def anomaly_setup_hint(anomaly_type: OIAnomalyType | Any) -> str:
    parsed = parse_anomaly_type(anomaly_type)
    label = normalize_label(parsed)

    if label in {
        "oi_collapse",
        "liquidation_driven_oi_drop",
        "sudden_deleveraging",
        "overheated_buildup",
        "extreme_crowding",
        "funding_oi_imbalance",
        "oi_price_dislocation",
    }:
        return "reversal"

    if label in {
        "oi_spike",
        "oi_volume_dislocation",
    }:
        return "continuation"

    return "unknown"


def anomaly_is_risk_critical(anomaly_type: OIAnomalyType | Any) -> bool:
    parsed = parse_anomaly_type(anomaly_type)

    is_risk_anomaly = getattr(parsed, "is_risk_anomaly", None)
    if is_risk_anomaly is True:
        return True

    label = normalize_label(parsed)
    return label in {
        "liquidation_driven_oi_drop",
        "sudden_deleveraging",
        "oi_collapse",
        "extreme_crowding",
    }


def side_from_oi_anomaly(
    anomaly_type: OIAnomalyType | Any,
    *,
    features: OIFeatures | Any | None = None,
    regime: OIRegime | Any | None = None,
) -> SignalSide:
    parsed = parse_anomaly_type(anomaly_type)
    label = normalize_label(parsed)

    if label in {
        "oi_spike",
        "oi_volume_dislocation",
    }:
        side = side_from_oi_regime(regime, features=features)
        if is_directional_side(side):
            return side
        return side_from_oi_features(features)

    if label in {
        "oi_collapse",
        "liquidation_driven_oi_drop",
        "sudden_deleveraging",
    }:
        return reversal_side_from_flush(features)

    if label in {
        "overheated_buildup",
        "extreme_crowding",
        "funding_oi_imbalance",
        "oi_price_dislocation",
    }:
        return contrarian_side_from_crowding(features=features, regime=regime)

    return SignalSide.UNKNOWN


def side_from_oi_features(
    features: OIFeatures | Any | None,
    *,
    dead_zone: float = 0.0,
) -> SignalSide:
    if features is None:
        return SignalSide.UNKNOWN

    price_delta = to_float(get_attr_or_key(features, "price_delta_pct"))
    oi_delta = to_float(get_attr_or_key(features, "oi_delta_pct"))
    pressure = to_float(get_attr_or_key(features, "oi_pressure_score"))
    flow = to_float(get_attr_or_key(features, "aggressive_flow_imbalance"))

    if price_delta is not None and oi_delta is not None:
        if price_delta > dead_zone and oi_delta > dead_zone:
            if pressure is None or pressure >= -dead_zone:
                if flow is None or flow >= -dead_zone:
                    return SignalSide.LONG

        if price_delta < -dead_zone and oi_delta > dead_zone:
            if pressure is None or pressure <= dead_zone:
                if flow is None or flow <= dead_zone:
                    return SignalSide.SHORT

    if pressure is not None:
        if pressure > dead_zone:
            return SignalSide.LONG
        if pressure < -dead_zone:
            return SignalSide.SHORT

    if flow is not None:
        if flow > dead_zone:
            return SignalSide.LONG
        if flow < -dead_zone:
            return SignalSide.SHORT

    return SignalSide.UNKNOWN


def reversal_side_from_flush(features: OIFeatures | Any | None) -> SignalSide:
    """
    Reversal side after OI flush / forced deleveraging.

    If price/OI pressure suggests downside flush, reversal is LONG.
    If price/OI pressure suggests upside squeeze exhaustion, reversal is SHORT.
    """
    feature_side = side_from_oi_features(features)

    if feature_side is SignalSide.SHORT:
        return SignalSide.LONG

    if feature_side is SignalSide.LONG:
        return SignalSide.SHORT

    if features is None:
        return SignalSide.UNKNOWN

    price_delta = to_float(get_attr_or_key(features, "price_delta_pct"))
    pressure = to_float(get_attr_or_key(features, "oi_pressure_score"))

    if price_delta is not None:
        if price_delta < 0:
            return SignalSide.LONG
        if price_delta > 0:
            return SignalSide.SHORT

    if pressure is not None:
        if pressure < 0:
            return SignalSide.LONG
        if pressure > 0:
            return SignalSide.SHORT

    return SignalSide.UNKNOWN


def contrarian_side_from_crowding(
    *,
    features: OIFeatures | Any | None,
    regime: OIRegime | Any | None = None,
) -> SignalSide:
    regime_side = side_from_oi_regime(regime, features=features)
    if is_directional_side(regime_side):
        return opposite_side(regime_side)

    feature_side = side_from_oi_features(features)
    if is_directional_side(feature_side):
        return opposite_side(feature_side)

    return SignalSide.UNKNOWN


# =============================================================================
# Score DTO / scoring helpers
# =============================================================================


@dataclass(slots=True)
class ScoreBreakdown:
    """
    Small reusable score DTO for concrete open-interest strategies.
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
        self.reasons = list(dict.fromkeys(str(item) for item in self.reasons if str(item).strip()))
        self.confirmations = list(
            dict.fromkeys(str(item) for item in self.confirmations if str(item).strip())
        )
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


def freshness_score(
    *,
    event_time: datetime | None,
    now: datetime | None = None,
    stale_after_seconds: float | None = None,
) -> float:
    if event_time is None or stale_after_seconds is None:
        return 1.0

    current = ensure_aware_utc(now or utc_now())
    event_ts = ensure_aware_utc(event_time)

    age = max(0.0, (current - event_ts).total_seconds())

    if age <= stale_after_seconds * 0.5:
        return 1.0

    if age <= stale_after_seconds:
        return clamp(1.0 - ((age - stale_after_seconds * 0.5) / (stale_after_seconds * 0.5)) * 0.5, 0.5, 1.0)

    if age <= stale_after_seconds * 2:
        return clamp(0.5 - ((age - stale_after_seconds) / stale_after_seconds) * 0.5, 0.0, 0.5)

    return 0.0


def is_stale(
    *,
    event_time: datetime | None,
    now: datetime | None = None,
    stale_after_seconds: float | None = None,
) -> bool:
    if event_time is None or stale_after_seconds is None:
        return False

    current = ensure_aware_utc(now or utc_now())
    event_ts = ensure_aware_utc(event_time)

    age = max(0.0, (current - event_ts).total_seconds())
    return age > stale_after_seconds


# =============================================================================
# Quality filters
# =============================================================================


def quality_filter_reason(
    value: Any,
    *,
    min_confidence: float = 0.0,
    min_score: float = 0.0,
    require_detected: bool = False,
    stale_after_seconds: float | None = None,
    now: datetime | None = None,
) -> str | None:
    if value is None:
        return "missing_open_interest_context"

    if require_detected and not extract_detected(value, default=False):
        return "open_interest_context_not_detected"

    confidence = extract_confidence(value)
    if confidence < min_confidence:
        return "open_interest_confidence_below_threshold"

    score = extract_score(value, default=confidence)
    if score < min_score:
        return "open_interest_score_below_threshold"

    event_time = extract_event_time(value)
    if is_stale(
        event_time=event_time,
        now=now,
        stale_after_seconds=stale_after_seconds,
    ):
        return "open_interest_context_stale"

    return None


def regime_filter_reason(
    regime: OIRegimeResult | Any,
    *,
    min_confidence: float = 0.0,
    min_score: float = 0.0,
    allowed_regimes: set[OIRegime] | None = None,
    require_directional: bool = False,
) -> str | None:
    if regime is None:
        return "missing_oi_regime"

    oi_regime = extract_regime_type(regime)

    if allowed_regimes is not None and oi_regime not in allowed_regimes:
        return "oi_regime_not_allowed"

    confidence = extract_confidence(regime)
    if confidence < min_confidence:
        return "oi_regime_confidence_below_threshold"

    score = extract_score(regime, default=confidence)
    if score < min_score:
        return "oi_regime_score_below_threshold"

    if require_directional and not is_directional_side(side_from_oi_regime(oi_regime)):
        return "oi_regime_not_directional"

    return None


def divergence_filter_reason(
    divergence: OIDivergenceResult | Any,
    *,
    min_confidence: float = 0.0,
    min_score: float = 0.0,
    require_detected: bool = True,
    require_directional: bool = True,
) -> str | None:
    if divergence is None:
        return "missing_oi_divergence"

    divergence_type = extract_divergence_type(divergence)

    if require_detected and not extract_detected(divergence, default=False):
        return "oi_divergence_not_detected"

    if divergence_type is OIDivergenceType.NONE:
        return "oi_divergence_type_none"

    confidence = extract_confidence(divergence)
    if confidence < min_confidence:
        return "oi_divergence_confidence_below_threshold"

    score = extract_score(divergence, default=confidence)
    if score < min_score:
        return "oi_divergence_score_below_threshold"

    if require_directional:
        side = side_from_oi_divergence(divergence_type)
        if not is_directional_side(side):
            return "oi_divergence_not_directional"

    return None


def anomaly_filter_reason(
    anomaly: OIAnomalyResult | Any,
    *,
    min_confidence: float = 0.0,
    min_score: float = 0.0,
    require_detected: bool = True,
    require_actionable: bool = True,
) -> str | None:
    if anomaly is None:
        return "missing_oi_anomaly"

    anomaly_type = extract_anomaly_type(anomaly)

    if require_detected and not extract_detected(anomaly, default=False):
        return "oi_anomaly_not_detected"

    if anomaly_type is OIAnomalyType.NONE:
        return "oi_anomaly_type_none"

    confidence = extract_confidence(anomaly)
    if confidence < min_confidence:
        return "oi_anomaly_confidence_below_threshold"

    score = extract_score(anomaly, default=confidence)
    if score < min_score:
        return "oi_anomaly_score_below_threshold"

    if require_actionable and anomaly_setup_hint(anomaly_type) == "unknown":
        return "oi_anomaly_not_actionable"

    return None


# =============================================================================
# Convenience source-feature helpers
# =============================================================================


def source_features_from_paths(*paths: str) -> list[str]:
    result: list[str] = []

    for path in paths:
        if not isinstance(path, str) or not path.strip():
            continue

        normalized = path.strip()
        if normalized.startswith("open_interest."):
            result.append(normalized)
        elif normalized.startswith("oi."):
            result.append(f"open_interest.{normalized.removeprefix('oi.')}")
        else:
            result.append(f"open_interest.{normalized}")

    return list(dict.fromkeys(result))


def base_open_interest_source_features() -> list[str]:
    return source_features_from_paths(
        "analysis",
        "features",
        "regime",
        "divergence",
        "anomaly",
    )