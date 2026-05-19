# trading_system/strategy/strategies/spoofing/utils.py

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from analytics.spoofing.enums import (
    SpoofingComponent,
    SpoofingPattern,
    SpoofingSeverity,
    SpoofingSide,
    SpoofingStatus,
    SpoofingType,
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
    SignalBuilder, not to concrete spoofing strategies.
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
        features.pull_ratio
        detector_results.order_pull.score
        score_breakdown.agreement_ratio
        metadata.wall_notional
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


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue

        if isinstance(value, str) and not value.strip():
            continue

        return value

    return None


def unwrap_analytics_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """
    Backward-compatible unwrap for analytics.spoofing.* envelopes.

    Concrete strategies should usually receive normalized StrategyContext.
    This helper is useful while domain_data still contains analytics envelopes
    or result-like nested payloads.
    """
    raw = dict(payload)
    inner = raw.get("payload")

    if isinstance(inner, Mapping):
        inner_dict = dict(inner)

        for key in (
            "signal",
            "spoofing_signal",
            "composite",
            "spoofing",
            "result",
            "snapshot",
            "event",
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


def normalize_signal_payload(payload: Any) -> dict[str, Any]:
    """
    Normalize common analytics.spoofing payload shapes:

    - SpoofingSignal-like object with to_dict()
    - direct signal dict
    - {"signal": {...}}
    - {"payload": {"signal": {...}}}
    """
    mapping = as_mapping(payload)
    if mapping is None:
        return {}

    data = dict(mapping)

    signal = data.get("signal") or data.get("spoofing_signal")
    if isinstance(signal, Mapping):
        result = dict(signal)
        result.setdefault("_source_feature", data.get("_source_feature"))
        result.setdefault("_container", data)
        return result

    payload_value = data.get("payload")
    if isinstance(payload_value, Mapping):
        nested = normalize_signal_payload(payload_value)
        if nested:
            nested.setdefault("_envelope", data)
            return nested

    return data


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
            "passed",
            "triggered",
            "pulled",
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
            "failed",
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


def enum_value(value: Any, default: str = "") -> str:
    if value is None:
        return default

    if isinstance(value, Enum):
        return str(value.value).strip().lower()

    raw = getattr(value, "value", value)
    text = str(raw).strip().lower()
    return text if text else default


def normalize_label(value: Any) -> str:
    return enum_value(value, "")


def unit_score(value: Any, default: float = 0.0) -> float:
    parsed = to_float(value, default)
    return clamp(float(parsed if parsed is not None else default), 0.0, 1.0)


def signed_score(value: Any, default: float = 0.0) -> float:
    parsed = to_float(value, default)
    return clamp(float(parsed if parsed is not None else default), -1.0, 1.0)


def abs_score(value: Any, default: float = 0.0) -> float:
    return abs(signed_score(value, default))


def ratio_score(
    value: Any,
    *,
    scale: float = 1.0,
    default: float = 0.0,
) -> float:
    parsed = abs(to_float(value, default) or default)
    if scale <= 0:
        return unit_score(parsed)
    return unit_score(parsed / scale)


# =============================================================================
# Bps / price helpers
# =============================================================================


def bps_move(
    *,
    from_price: Any,
    to_price: Any,
) -> float:
    start = to_float(from_price)
    end = to_float(to_price)

    if start is None or end is None or start <= 0:
        return 0.0

    return abs((end - start) / start) * 10_000.0


def signed_bps_move(
    *,
    from_price: Any,
    to_price: Any,
) -> float:
    start = to_float(from_price)
    end = to_float(to_price)

    if start is None or end is None or start <= 0:
        return 0.0

    return ((end - start) / start) * 10_000.0


def distance_bps(
    *,
    price: Any,
    reference_price: Any,
) -> float:
    return bps_move(from_price=reference_price, to_price=price)


# =============================================================================
# Enum parsing helpers
# =============================================================================


def parse_enum(
    value: Any,
    enum_cls: type[Enum],
    default: Enum | None = None,
) -> Enum | None:
    if isinstance(value, enum_cls):
        return value

    if value is None:
        return default

    text = str(getattr(value, "value", value)).strip()
    if not text:
        return default

    for item in enum_cls:
        if item.value.lower() == text.lower() or item.name.lower() == text.lower():
            return item

    return default


def parse_spoofing_side(
    value: Any,
    default: SpoofingSide | None = None,
) -> SpoofingSide | None:
    return parse_enum(value, SpoofingSide, default)  # type: ignore[return-value]


def parse_spoofing_type(
    value: Any,
    default: SpoofingType | None = None,
) -> SpoofingType | None:
    return parse_enum(value, SpoofingType, default)  # type: ignore[return-value]


def parse_spoofing_pattern(
    value: Any,
    default: SpoofingPattern | None = None,
) -> SpoofingPattern | None:
    return parse_enum(value, SpoofingPattern, default)  # type: ignore[return-value]


def parse_spoofing_severity(
    value: Any,
    default: SpoofingSeverity | None = None,
) -> SpoofingSeverity | None:
    return parse_enum(value, SpoofingSeverity, default)  # type: ignore[return-value]


def parse_spoofing_status(
    value: Any,
    default: SpoofingStatus | None = None,
) -> SpoofingStatus | None:
    return parse_enum(value, SpoofingStatus, default)  # type: ignore[return-value]


def parse_spoofing_component(
    value: Any,
    default: SpoofingComponent | None = None,
) -> SpoofingComponent | None:
    return parse_enum(value, SpoofingComponent, default)  # type: ignore[return-value]


# =============================================================================
# StrategyContext spoofing helpers
# =============================================================================


SPOOFING_DOMAIN_ALIASES: dict[str, tuple[str, ...]] = {
    "composite": (
        "composite",
        "spoofing",
        "analytics.spoofing",
        "result",
        "snapshot",
    ),
    "signal": (
        "signal",
        "spoofing_signal",
        "analytics_signal",
        "event",
    ),
    "features": (
        "features",
        "spoofing_features",
        "signal.features",
    ),
    "detector_results": (
        "detector_results",
        "detectors",
        "signal.detector_results",
    ),
    "score_breakdown": (
        "score_breakdown",
        "scoring",
        "signal.score_breakdown",
    ),
    "analytics_metadata": (
        "analytics_metadata",
        "metadata",
        "signal.analytics_metadata",
    ),
}


def spoofing_domain(context: StrategyContext) -> dict[str, Any]:
    return dict(context.domain_dict(FeatureSource.SPOOFING))


def spoofing_item(
    context: StrategyContext,
    key: str,
    default: Any = None,
) -> Any:
    domain = spoofing_domain(context)

    if key in domain:
        return domain[key]

    for alias in SPOOFING_DOMAIN_ALIASES.get(key, ()):
        value = get_path(domain, alias, default=None)
        if value is not None:
            return value

    return default


def spoofing_path(
    context: StrategyContext,
    path: str,
    default: Any = None,
) -> Any:
    """
    Read spoofing value from StrategyContext.

    Priority:
    1. exact feature name;
    2. spoofing-prefixed feature name;
    3. analytics.spoofing-prefixed feature name;
    4. FeatureSource.SPOOFING domain dotted path.
    """
    if not isinstance(path, str) or not path.strip():
        return default

    normalized = path.strip()
    spoofing_feature_name = (
        normalized
        if normalized.startswith("spoofing.")
        else f"spoofing.{normalized}"
    )
    analytics_feature_name = (
        normalized
        if normalized.startswith("analytics.spoofing.")
        else f"analytics.spoofing.{normalized}"
    )

    if context.has_feature(normalized):
        return context.get_feature(normalized)

    if context.has_feature(spoofing_feature_name):
        return context.get_feature(spoofing_feature_name)

    if context.has_feature(analytics_feature_name):
        return context.get_feature(analytics_feature_name)

    domain = spoofing_domain(context)

    if normalized.startswith("spoofing."):
        normalized = normalized.removeprefix("spoofing.")
    elif normalized.startswith("analytics.spoofing."):
        normalized = normalized.removeprefix("analytics.spoofing.")

    return get_path(domain, normalized, default)


def spoofing_float(
    context: StrategyContext,
    path: str,
    *,
    default: float | None = None,
) -> float | None:
    return to_float(spoofing_path(context, path, default), default)


def spoofing_int(
    context: StrategyContext,
    path: str,
    *,
    default: int | None = None,
) -> int | None:
    return to_int(spoofing_path(context, path, default), default)


def spoofing_unit_score(
    context: StrategyContext,
    path: str,
    *,
    default: float = 0.0,
) -> float:
    return unit_score(spoofing_path(context, path, default), default)


def spoofing_signed_score(
    context: StrategyContext,
    path: str,
    *,
    default: float = 0.0,
) -> float:
    return signed_score(spoofing_path(context, path, default), default)


def spoofing_bool(
    context: StrategyContext,
    path: str,
    *,
    default: bool = False,
) -> bool:
    return to_bool(spoofing_path(context, path, default), default)


def spoofing_str(
    context: StrategyContext,
    path: str,
    *,
    default: str | None = None,
) -> str | None:
    return to_str(spoofing_path(context, path, default), default)


def spoofing_datetime(
    context: StrategyContext,
    path: str,
    *,
    default: datetime | None = None,
) -> datetime | None:
    return parse_datetime(spoofing_path(context, path, default))


# =============================================================================
# Spoofing payload extraction
# =============================================================================


def extract_signal_payload(value: Any) -> dict[str, Any]:
    return normalize_signal_payload(value)


def extract_features(value: Any) -> dict[str, Any]:
    payload = normalize_signal_payload(value)

    features = (
        payload.get("features")
        or payload.get("spoofing_features")
        or get_path(payload, "signal.features")
    )
    return as_dict(features)


def extract_detector_results(value: Any) -> dict[str, Any]:
    payload = normalize_signal_payload(value)

    detectors = (
        payload.get("detector_results")
        or payload.get("detectors")
        or get_path(payload, "signal.detector_results")
    )
    return as_dict(detectors)


def extract_score_breakdown(value: Any) -> dict[str, Any]:
    payload = normalize_signal_payload(value)

    score_breakdown = (
        payload.get("score_breakdown")
        or payload.get("scoring")
        or get_path(payload, "signal.score_breakdown")
    )
    return as_dict(score_breakdown)


def extract_analytics_metadata(value: Any) -> dict[str, Any]:
    payload = normalize_signal_payload(value)

    metadata = (
        payload.get("analytics_metadata")
        or payload.get("metadata")
        or get_path(payload, "signal.analytics_metadata")
    )
    return as_dict(metadata)


def extract_event_time(value: Any) -> datetime | None:
    payload = normalize_signal_payload(value)

    return parse_datetime(
        first_present(
            payload,
            (
                "event_time",
                "detected_at",
                "timestamp",
                "created_at",
                "updated_at",
                "time",
                "metadata.event_time",
                "metadata.detected_at",
            ),
        )
    )


def extract_spoofing_type(value: Any) -> SpoofingType | None:
    payload = normalize_signal_payload(value)
    return parse_spoofing_type(
        payload.get("spoofing_type")
        or payload.get("type")
        or get_path(payload, "metadata.spoofing_type")
    )


def extract_spoofing_pattern(value: Any) -> SpoofingPattern | None:
    payload = normalize_signal_payload(value)
    return parse_spoofing_pattern(
        payload.get("pattern")
        or payload.get("spoofing_pattern")
        or get_path(payload, "metadata.pattern")
    )


def extract_spoofing_side(value: Any) -> SpoofingSide | None:
    payload = normalize_signal_payload(value)
    return parse_spoofing_side(
        payload.get("side")
        or payload.get("spoofing_side")
        or get_path(payload, "features.side")
        or get_path(payload, "metadata.side")
    )


def extract_spoofing_status(value: Any) -> SpoofingStatus | None:
    payload = normalize_signal_payload(value)
    return parse_spoofing_status(
        payload.get("status")
        or payload.get("spoofing_status")
        or get_path(payload, "metadata.status")
    )


def extract_spoofing_severity(value: Any) -> SpoofingSeverity | None:
    payload = normalize_signal_payload(value)
    return parse_spoofing_severity(
        payload.get("severity")
        or payload.get("spoofing_severity")
        or get_path(payload, "metadata.severity")
    )


def extract_score(value: Any, default: float = 0.0) -> float:
    payload = normalize_signal_payload(value)
    return unit_score(
        payload.get("score")
        or payload.get("spoofing_score")
        or get_path(payload, "score_breakdown.final_score")
        or get_path(payload, "score_breakdown.score")
        or get_path(payload, "metadata.score"),
        default,
    )


def extract_confidence(value: Any, default: float = 0.0) -> float:
    payload = normalize_signal_payload(value)
    return unit_score(
        payload.get("confidence")
        or payload.get("spoofing_confidence")
        or get_path(payload, "score_breakdown.confidence")
        or get_path(payload, "metadata.confidence"),
        default,
    )


def extract_price_level(value: Any) -> float | None:
    payload = normalize_signal_payload(value)
    return to_float(
        payload.get("price_level")
        or payload.get("level_price")
        or payload.get("wall_price")
        or get_path(payload, "features.price_level")
        or get_path(payload, "metadata.price_level")
    )


def extract_wall_id(value: Any) -> str | None:
    payload = normalize_signal_payload(value)
    return to_str(
        payload.get("wall_id")
        or payload.get("source_wall_id")
        or get_path(payload, "features.wall_id")
        or get_path(payload, "metadata.wall_id")
    )


# =============================================================================
# Feature helpers
# =============================================================================


def feature_value(value: Any, name: str, default: Any = None) -> Any:
    payload = normalize_signal_payload(value)
    features = extract_features(payload)

    if name in features:
        return features[name]

    if name in payload:
        return payload[name]

    return get_path(features, name, get_path(payload, f"features.{name}", default))


def feature_float(value: Any, name: str, default: float | None = None) -> float | None:
    return to_float(feature_value(value, name, default), default)


def feature_int(value: Any, name: str, default: int | None = None) -> int | None:
    return to_int(feature_value(value, name, default), default)


def feature_bool(value: Any, name: str, default: bool = False) -> bool:
    return to_bool(feature_value(value, name, default), default)


def feature_str(value: Any, name: str, default: str | None = None) -> str | None:
    return to_str(feature_value(value, name, default), default)


def metadata_value(value: Any, name: str, default: Any = None) -> Any:
    metadata = extract_analytics_metadata(value)

    if name in metadata:
        return metadata[name]

    return get_path(metadata, name, default)


def metadata_float(value: Any, name: str, default: float | None = None) -> float | None:
    return to_float(metadata_value(value, name, default), default)


def metadata_int(value: Any, name: str, default: int | None = None) -> int | None:
    return to_int(metadata_value(value, name, default), default)


def metadata_bool(value: Any, name: str, default: bool = False) -> bool:
    return to_bool(metadata_value(value, name, default), default)


def metadata_str(value: Any, name: str, default: str | None = None) -> str | None:
    return to_str(metadata_value(value, name, default), default)


def analytics_value(value: Any, name: str, default: Any = None) -> Any:
    payload = normalize_signal_payload(value)

    if name in payload:
        return payload[name]

    return get_path(payload, name, default)


def analytics_float(value: Any, name: str, default: float | None = None) -> float | None:
    return to_float(analytics_value(value, name, default), default)


def analytics_int(value: Any, name: str, default: int | None = None) -> int | None:
    return to_int(analytics_value(value, name, default), default)


def analytics_bool(value: Any, name: str, default: bool = False) -> bool:
    return to_bool(analytics_value(value, name, default), default)


def analytics_str(value: Any, name: str, default: str | None = None) -> str | None:
    return to_str(analytics_value(value, name, default), default)


# =============================================================================
# Common spoofing metric extractors
# =============================================================================


def extract_pull_ratio(value: Any) -> float:
    return unit_score(
        first_non_empty(
            feature_value(value, "pull_ratio"),
            feature_value(value, "pulled_ratio"),
            metadata_value(value, "pull_ratio"),
            analytics_value(value, "pull_ratio"),
        )
    )


def extract_fill_ratio(value: Any) -> float:
    return unit_score(
        first_non_empty(
            feature_value(value, "fill_ratio"),
            feature_value(value, "filled_ratio"),
            metadata_value(value, "fill_ratio"),
            analytics_value(value, "fill_ratio"),
        )
    )


def extract_price_reaction_bps(value: Any) -> float:
    return abs(
        to_float(
            first_non_empty(
                feature_value(value, "price_reaction_bps"),
                feature_value(value, "reaction_bps"),
                metadata_value(value, "price_reaction_bps"),
                analytics_value(value, "price_reaction_bps"),
            ),
            0.0,
        )
        or 0.0
    )


def extract_signed_price_reaction_bps(value: Any) -> float:
    return to_float(
        first_non_empty(
            feature_value(value, "signed_price_reaction_bps"),
            feature_value(value, "price_reaction_bps"),
            feature_value(value, "reaction_bps"),
            metadata_value(value, "signed_price_reaction_bps"),
            metadata_value(value, "price_reaction_bps"),
        ),
        0.0,
    ) or 0.0


def extract_lifetime_ms(value: Any) -> float:
    return max(
        0.0,
        to_float(
            first_non_empty(
                feature_value(value, "lifetime_ms"),
                feature_value(value, "wall_lifetime_ms"),
                metadata_value(value, "lifetime_ms"),
                analytics_value(value, "lifetime_ms"),
            ),
            0.0,
        )
        or 0.0,
    )


def extract_wall_notional(value: Any) -> float:
    return max(
        0.0,
        to_float(
            first_non_empty(
                feature_value(value, "wall_notional"),
                feature_value(value, "notional"),
                metadata_value(value, "wall_notional"),
                analytics_value(value, "wall_notional"),
            ),
            0.0,
        )
        or 0.0,
    )


def extract_pulled_notional(value: Any) -> float:
    return max(
        0.0,
        to_float(
            first_non_empty(
                feature_value(value, "pulled_notional"),
                feature_value(value, "cancelled_notional"),
                metadata_value(value, "pulled_notional"),
                analytics_value(value, "pulled_notional"),
            ),
            0.0,
        )
        or 0.0,
    )


def extract_cancel_to_fill_ratio(value: Any) -> float:
    return unit_score(
        first_non_empty(
            feature_value(value, "cancel_to_fill_ratio"),
            metadata_value(value, "cancel_to_fill_ratio"),
            analytics_value(value, "cancel_to_fill_ratio"),
        )
    )


def extract_distance_from_mid_bps(value: Any) -> float:
    return abs(
        to_float(
            first_non_empty(
                feature_value(value, "distance_from_mid_bps"),
                feature_value(value, "distance_to_mid_bps"),
                metadata_value(value, "distance_from_mid_bps"),
                analytics_value(value, "distance_from_mid_bps"),
            ),
            0.0,
        )
        or 0.0
    )


def extract_layer_count(value: Any) -> int:
    return max(
        0,
        to_int(
            first_non_empty(
                feature_value(value, "layer_count"),
                feature_value(value, "layers"),
                metadata_value(value, "layer_count"),
                analytics_value(value, "layer_count"),
            ),
            0,
        )
        or 0,
    )


def extract_layer_price_span_bps(value: Any) -> float:
    return abs(
        to_float(
            first_non_empty(
                feature_value(value, "layer_price_span_bps"),
                feature_value(value, "price_span_bps"),
                metadata_value(value, "layer_price_span_bps"),
                analytics_value(value, "layer_price_span_bps"),
            ),
            0.0,
        )
        or 0.0
    )


def extract_pressure_flip_strength(value: Any) -> float:
    return unit_score(
        first_non_empty(
            feature_value(value, "pressure_flip_strength"),
            feature_value(value, "flip_strength"),
            metadata_value(value, "pressure_flip_strength"),
            analytics_value(value, "pressure_flip_strength"),
        )
    )


# =============================================================================
# Detector helpers
# =============================================================================


def _component_keys(component: SpoofingComponent | str) -> tuple[str, ...]:
    if isinstance(component, SpoofingComponent):
        value = component.value
        name = component.name
    else:
        value = str(component)
        name = str(component)

    normalized_value = value.strip()
    normalized_name = name.strip()

    return tuple(
        dict.fromkeys(
            key
            for key in (
                normalized_value,
                normalized_value.lower(),
                normalized_name,
                normalized_name.lower(),
                normalized_value.replace("-", "_").lower(),
                normalized_name.replace("-", "_").lower(),
            )
            if key
        )
    )


def detector_payload(value: Any, component: SpoofingComponent | str) -> dict[str, Any]:
    detectors = extract_detector_results(value)

    for key in _component_keys(component):
        candidate = detectors.get(key)
        if isinstance(candidate, Mapping):
            return dict(candidate)

    for key, candidate in detectors.items():
        label = normalize_label(key)
        if label in {normalize_label(item) for item in _component_keys(component)}:
            if isinstance(candidate, Mapping):
                return dict(candidate)

    return {}


def has_detector(value: Any, component: SpoofingComponent | str) -> bool:
    return bool(detector_payload(value, component))


def detector_metadata(value: Any, component: SpoofingComponent | str) -> dict[str, Any]:
    payload = detector_payload(value, component)
    metadata = payload.get("metadata") or payload.get("details")
    return as_dict(metadata)


def detector_score(value: Any, component: SpoofingComponent | str) -> float:
    payload = detector_payload(value, component)
    return unit_score(
        payload.get("score")
        or payload.get("detector_score")
        or get_path(payload, "metadata.score")
    )


def detector_confidence(value: Any, component: SpoofingComponent | str) -> float:
    payload = detector_payload(value, component)
    return unit_score(
        payload.get("confidence")
        or payload.get("detector_confidence")
        or get_path(payload, "metadata.confidence")
    )


def detector_passed(value: Any, component: SpoofingComponent | str) -> bool:
    payload = detector_payload(value, component)
    if not payload:
        return False

    return to_bool(
        payload.get("passed")
        or payload.get("detected")
        or payload.get("is_detected")
        or payload.get("valid"),
        default=True,
    )


def detector_count(value: Any) -> int:
    detectors = extract_detector_results(value)
    if not detectors:
        return 0

    return len(
        [
            item
            for item in detectors.values()
            if isinstance(item, Mapping)
        ]
    )


def passed_detector_count(value: Any) -> int:
    detectors = extract_detector_results(value)
    if not detectors:
        return 0

    total = 0
    for detector in detectors.values():
        if not isinstance(detector, Mapping):
            continue
        if to_bool(
            detector.get("passed")
            or detector.get("detected")
            or detector.get("is_detected")
            or detector.get("valid"),
            default=True,
        ):
            total += 1

    return total


def detector_average_confidence(value: Any) -> float:
    detectors = extract_detector_results(value)

    values = [
        unit_score(
            detector.get("confidence")
            or detector.get("detector_confidence")
            or get_path(detector, "metadata.confidence")
        )
        for detector in detectors.values()
        if isinstance(detector, Mapping)
    ]

    if not values:
        return 0.0

    return unit_score(sum(values) / len(values))


def detector_agreement_ratio(value: Any) -> float:
    score_breakdown = extract_score_breakdown(value)

    explicit = (
        score_breakdown.get("agreement_ratio")
        or score_breakdown.get("detector_agreement_ratio")
        or metadata_value(value, "agreement_ratio")
    )
    if explicit is not None:
        return unit_score(explicit)

    total = detector_count(value)
    if total <= 0:
        return 0.0

    return unit_score(passed_detector_count(value) / total)


# =============================================================================
# Direction helpers
# =============================================================================


def spoofing_side_to_signal_side(side: SpoofingSide | str | None) -> SignalSide:
    """
    Spoofing strategy direction convention:

    ASK fake liquidity / pulled ask wall -> fake resistance removed -> LONG.
    BID fake liquidity / pulled bid wall -> fake support removed -> SHORT.
    """
    parsed = parse_spoofing_side(side)
    label = normalize_label(parsed or side)

    if label in {"ask", "sell", "sell_side", "asks", "offer", "supply"}:
        return SignalSide.LONG

    if label in {"bid", "buy", "buy_side", "bids", "demand"}:
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


def reaction_aligns_with_side(
    *,
    signed_reaction_bps: float,
    side: SignalSide,
    min_reaction_bps: float = 0.0,
) -> bool:
    if abs(signed_reaction_bps) < min_reaction_bps:
        return False

    if side is SignalSide.LONG:
        return signed_reaction_bps > 0

    if side is SignalSide.SHORT:
        return signed_reaction_bps < 0

    return False


# =============================================================================
# Score DTO / scoring helpers
# =============================================================================


@dataclass(slots=True)
class ScoreBreakdown:
    """
    Small reusable score DTO for concrete spoofing strategies.
    """

    score: float = 0.0
    confidence: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    confirmations: list[str] = field(default_factory=list)

    def normalize(self) -> ScoreBreakdown:
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
        self.reasons = list(
            dict.fromkeys(
                str(item).strip()
                for item in self.reasons
                if str(item).strip()
            )
        )
        self.confirmations = list(
            dict.fromkeys(
                str(item).strip()
                for item in self.confirmations
                if str(item).strip()
            )
        )
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
        return clamp(
            1.0
            - ((age - stale_after_seconds * 0.5) / (stale_after_seconds * 0.5)) * 0.5,
            0.5,
            1.0,
        )

    if age <= stale_after_seconds * 2:
        return clamp(
            0.5 - ((age - stale_after_seconds) / stale_after_seconds) * 0.5,
            0.0,
            0.5,
        )

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
    min_score: float = 0.0,
    min_confidence: float = 0.0,
    allowed_severities: tuple[SpoofingSeverity, ...] = (),
    min_detector_count: int = 0,
    min_agreement_ratio: float = 0.0,
    min_average_confidence: float = 0.0,
    require_score_passed: bool = False,
    stale_after_seconds: float | None = None,
    now: datetime | None = None,
) -> str | None:
    if value is None:
        return "missing_spoofing_context"

    score = extract_score(value)
    if score < min_score:
        return "spoofing_score_below_threshold"

    confidence = extract_confidence(value)
    if confidence < min_confidence:
        return "spoofing_confidence_below_threshold"

    severity = extract_spoofing_severity(value)
    if allowed_severities and severity not in allowed_severities:
        return "spoofing_severity_not_allowed"

    if detector_count(value) < min_detector_count:
        return "spoofing_detector_count_below_threshold"

    if detector_agreement_ratio(value) < min_agreement_ratio:
        return "spoofing_detector_agreement_below_threshold"

    if detector_average_confidence(value) < min_average_confidence:
        return "spoofing_detector_average_confidence_below_threshold"

    if require_score_passed:
        score_breakdown = extract_score_breakdown(value)
        if not to_bool(
            score_breakdown.get("passed")
            or score_breakdown.get("score_passed")
            or metadata_value(value, "score_passed"),
            default=True,
        ):
            return "spoofing_score_not_passed"

    event_time = extract_event_time(value)
    if is_stale(
        event_time=event_time,
        now=now,
        stale_after_seconds=stale_after_seconds,
    ):
        return "spoofing_context_stale"

    return None


# =============================================================================
# Pattern helpers
# =============================================================================


def is_order_pull_signal(value: Any) -> bool:
    spoofing_type = extract_spoofing_type(value)
    pattern = extract_spoofing_pattern(value)

    return (
        spoofing_type is SpoofingType.ORDER_PULL
        or pattern is SpoofingPattern.PULL_AND_REVERSAL
        or has_detector(value, SpoofingComponent.ORDER_PULL_DETECTOR)
    )


def is_pressure_bluff_signal(value: Any) -> bool:
    spoofing_type = extract_spoofing_type(value)
    pattern = extract_spoofing_pattern(value)

    return (
        spoofing_type is SpoofingType.FLIP_PRESSURE
        or pattern is SpoofingPattern.PRESSURE_BLUFF
        or has_detector(value, SpoofingComponent.FLIP_PRESSURE_DETECTOR)
    )


def is_layering_signal(value: Any) -> bool:
    spoofing_type = extract_spoofing_type(value)
    pattern = extract_spoofing_pattern(value)

    return (
        spoofing_type is SpoofingType.LAYERING
        or pattern is SpoofingPattern.MULTI_LEVEL_LAYERING
        or has_detector(value, SpoofingComponent.LAYERING_DETECTOR)
    )


def is_fake_liquidity_signal(value: Any) -> bool:
    spoofing_type = extract_spoofing_type(value)
    pattern = extract_spoofing_pattern(value)

    return (
        spoofing_type is SpoofingType.FAKE_LIQUIDITY
        or pattern is SpoofingPattern.FAKE_ABSORPTION
        or has_detector(value, SpoofingComponent.FAKE_LIQUIDITY_DETECTOR)
        or feature_bool(value, "is_fake_liquidity")
        or metadata_bool(value, "is_fake_liquidity")
    )


def is_composite_signal(value: Any) -> bool:
    spoofing_type = extract_spoofing_type(value)
    pattern = extract_spoofing_pattern(value)

    return (
        spoofing_type is SpoofingType.COMPOSITE
        or pattern is SpoofingPattern.COMPOSITE
        or detector_count(value) >= 2
    )


# =============================================================================
# Source-feature helpers
# =============================================================================


def source_features_from_paths(*paths: str) -> list[str]:
    result: list[str] = []

    for path in paths:
        if not isinstance(path, str) or not path.strip():
            continue

        normalized = path.strip()

        if normalized.startswith("spoofing."):
            result.append(normalized)
        elif normalized.startswith("analytics.spoofing."):
            result.append(normalized.replace("analytics.", "", 1))
        else:
            result.append(f"spoofing.{normalized}")

    return list(dict.fromkeys(result))


def base_spoofing_source_features() -> list[str]:
    return source_features_from_paths(
        "composite",
        "signal",
        "features",
        "detector_results",
        "score_breakdown",
        "analytics_metadata",
        "type",
        "pattern",
        "side",
        "score",
        "confidence",
    )


def order_pull_source_features() -> list[str]:
    return source_features_from_paths(
        "signal",
        "type",
        "pattern",
        "side",
        "features.pull_ratio",
        "features.fill_ratio",
        "features.price_reaction_bps",
        "features.lifetime_ms",
        "features.wall_notional",
        "features.pulled_notional",
        "detector_results.order_pull",
    )


def fake_liquidity_source_features() -> list[str]:
    return source_features_from_paths(
        "signal",
        "type",
        "pattern",
        "side",
        "features.pull_ratio",
        "features.fill_ratio",
        "features.price_reaction_bps",
        "features.cancel_to_fill_ratio",
        "features.wall_notional",
        "features.pulled_notional",
        "features.distance_from_mid_bps",
        "detector_results.fake_liquidity",
    )


def pressure_bluff_source_features() -> list[str]:
    return source_features_from_paths(
        "signal",
        "type",
        "pattern",
        "side",
        "features.pressure_flip_strength",
        "features.price_reaction_bps",
        "features.distance_from_mid_bps",
        "detector_results.flip_pressure",
    )


def layering_source_features() -> list[str]:
    return source_features_from_paths(
        "signal",
        "type",
        "pattern",
        "side",
        "features.layer_count",
        "features.layer_price_span_bps",
        "features.pull_ratio",
        "features.wall_notional",
        "features.pulled_notional",
        "detector_results.layering",
    )


def composite_spoofing_source_features() -> list[str]:
    return source_features_from_paths(
        "signal",
        "type",
        "pattern",
        "side",
        "detector_results",
        "score_breakdown",
        "score_breakdown.agreement_ratio",
        "score_breakdown.average_confidence",
        "analytics_metadata",
    )