# trading_system/strategy/strategies/whales/utils.py

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


FUTURES_MARKET_TYPES: frozenset[str] = frozenset(
    {
        "perpetual",
        "futures",
        "linear",
        "inverse",
        "swap",
        "usdm_futures",
        "coinm_futures",
    }
)

DEFAULT_WHALE_FEATURE_MAX_AGE_SECONDS = 90.0


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


def timestamp_ms(value: datetime | None) -> int | None:
    parsed = parse_datetime(value)
    if parsed is None:
        return None
    return int(parsed.timestamp() * 1000)


def serialize_for_metadata(value: Any) -> Any:
    if isinstance(value, datetime):
        return ensure_aware_utc(value).isoformat()

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    if hasattr(value, "to_payload") and callable(value.to_payload):
        return serialize_for_metadata(value.to_payload())

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
# Generic mapping / nested payload helpers
# =============================================================================


def as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value

    if hasattr(value, "to_payload") and callable(value.to_payload):
        converted = value.to_payload()
        if isinstance(converted, Mapping):
            return converted

    if hasattr(value, "to_dict") and callable(value.to_dict):
        converted = value.to_dict()
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
    if not isinstance(path, str) or not path.strip():
        return default

    normalized = path.strip()

    # Normalized StrategyContext domain_data may contain literal dotted keys
    # such as ``whales.large_trade.notional`` next to nested dictionaries.
    mapping = as_mapping(value)
    if mapping is not None and normalized in mapping:
        item = mapping.get(normalized)
        return default if item is None else item

    current = value

    for index, part in enumerate(normalized.split(".")):
        if current is None:
            return default

        # Support remaining literal dotted suffixes at each nesting level.
        current_mapping = as_mapping(current)
        if current_mapping is not None:
            suffix = ".".join(normalized.split(".")[index:])
            if suffix in current_mapping:
                item = current_mapping.get(suffix)
                return default if item is None else item

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


def unwrap_analytics_payload(payload: Any) -> dict[str, Any]:
    """
    Backward-compatible unwrap for analytics.whales.* envelopes.

    Supports:
    - direct dict payload;
    - dataclass/model with to_dict()/to_payload();
    - {"payload": {...}};
    - {"payload": {"pressure": {...}}};
    - {"payload": {"activity": {...}}};
    - {"payload": {"large_trade": {...}}};
    - {"payload": {"cluster": {...}}};
    - {"payload": {"liquidation_context": {...}}}.
    """
    mapping = as_mapping(payload)
    if mapping is None:
        return {}

    raw = dict(mapping)
    inner = raw.get("payload")

    if isinstance(inner, Mapping):
        inner_dict = dict(inner)

        for key in (
            "pressure",
            "whale_pressure",
            "activity",
            "whale_activity",
            "large_trade",
            "cluster",
            "whale_cluster",
            "cluster_update",
            "whale_cluster_update",
            "cluster_exhaustion",
            "whale_cluster_exhaustion",
            "liquidation_context",
            "whale_liquidation_context",
            "result",
            "event",
            "signal",
        ):
            nested = inner_dict.get(key)
            if isinstance(nested, Mapping):
                nested_dict = dict(nested)
                nested_dict.setdefault("_envelope", raw)
                nested_dict.setdefault("_container", inner_dict)
                return nested_dict

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
            "valid",
            "active",
            "tradeable",
            "passed",
            "confirmed",
            "enabled",
        }:
            return True

        if normalized in {
            "0",
            "false",
            "no",
            "n",
            "off",
            "invalid",
            "inactive",
            "failed",
            "disabled",
            "none",
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


def normalize_symbol(value: Any) -> str:
    text = to_str(value, "") or ""
    return text.replace("-", "").replace("/", "").replace("_", "").upper().strip()


def normalize_exchange(value: Any) -> str:
    return (to_str(value, "") or "").strip().lower()


def normalize_market_type(value: Any) -> str:
    return (to_str(value, "") or "").strip().lower()


def unit_score(value: Any, default: float = 0.0) -> float:
    parsed = to_float(value, default)
    return clamp(float(parsed if parsed is not None else default), 0.0, 1.0)


def signed_score(value: Any, default: float = 0.0) -> float:
    parsed = to_float(value, default)
    return clamp(float(parsed if parsed is not None else default), -1.0, 1.0)


def abs_score(value: Any, default: float = 0.0) -> float:
    return abs(signed_score(value, default))


def positive_float(value: Any, default: float = 0.0) -> float:
    parsed = to_float(value, default)
    return max(0.0, float(parsed if parsed is not None else default))


def first_float(*values: Any, default: float | None = None) -> float | None:
    for value in values:
        parsed = to_float(value, None)
        if parsed is not None:
            return parsed
    return default


def first_int(*values: Any, default: int | None = None) -> int | None:
    for value in values:
        parsed = to_int(value, None)
        if parsed is not None:
            return parsed
    return default


def normalize_notional(
    value: Any,
    threshold: float,
    *,
    default: float = 0.0,
) -> float:
    parsed = positive_float(value, default)
    return unit_score(parsed / max(float(threshold), 1.0))


# =============================================================================
# StrategyContext whales helpers
# =============================================================================


WHALES_DOMAIN_ALIASES: dict[str, tuple[str, ...]] = {
    "pressure": (
        "pressure",
        "whale_pressure",
        "whale_pressure_signal",
        "analytics.whales.whale_pressure",
    ),
    "activity": (
        "activity",
        "whale_activity",
        "whale_activity_signal",
        "analytics.whales.whale_activity",
    ),
    "large_trade": (
        "large_trade",
        "whale_large_trade",
        "large_trade_signal",
        "whale_large_trade_signal",
        "analytics.whales.large_trade",
        "whales.large_trade",
    ),
    "cluster": (
        "cluster",
        "whale_cluster",
        "whale_cluster_signal",
        "analytics.whales.whale_cluster",
    ),
    "cluster_update": (
        "cluster_update",
        "whale_cluster_update",
        "whale_cluster_update_signal",
        "analytics.whales.whale_cluster_update",
    ),
    "cluster_exhaustion": (
        "cluster_exhaustion",
        "whale_cluster_exhaustion",
        "whale_cluster_exhaustion_signal",
        "analytics.whales.whale_cluster_exhaustion",
    ),
    "liquidation_context": (
        "liquidation_context",
        "whale_liquidation_context",
        "whale_liquidation_context_signal",
        "analytics.whales.whale_liquidation_context",
    ),
    "metadata": (
        "metadata",
        "analytics_metadata",
        "pressure.metadata",
        "activity.metadata",
        "large_trade.metadata",
        "cluster.metadata",
        "cluster_update.metadata",
        "cluster_exhaustion.metadata",
        "liquidation_context.metadata",
    ),
}


def whales_domain(context: StrategyContext) -> dict[str, Any]:
    return dict(context.domain_dict(FeatureSource.WHALES))


def whales_item(
    context: StrategyContext,
    key: str,
    default: Any = None,
) -> Any:
    domain = whales_domain(context)

    if key in domain:
        return domain[key]

    for alias in WHALES_DOMAIN_ALIASES.get(key, ()):
        value = get_path(domain, alias, default=None)
        if value is not None:
            return value

    return default


def whales_path(
    context: StrategyContext,
    path: str,
    default: Any = None,
) -> Any:
    """
    Read whales value from StrategyContext.

    Priority:
    1. exact feature name;
    2. whales-prefixed feature name;
    3. analytics.whales-prefixed feature name;
    4. FeatureSource.WHALES domain dotted path.
    """
    if not isinstance(path, str) or not path.strip():
        return default

    normalized = path.strip()
    whales_feature_name = (
        normalized
        if normalized.startswith("whales.")
        else f"whales.{normalized}"
    )
    analytics_feature_name = (
        normalized
        if normalized.startswith("analytics.whales.")
        else f"analytics.whales.{normalized}"
    )

    if context.has_feature(normalized):
        return context.get_feature(normalized)

    if context.has_feature(whales_feature_name):
        return context.get_feature(whales_feature_name)

    if context.has_feature(analytics_feature_name):
        return context.get_feature(analytics_feature_name)

    domain = whales_domain(context)

    if normalized.startswith("whales."):
        normalized = normalized.removeprefix("whales.")
    elif normalized.startswith("analytics.whales."):
        normalized = normalized.removeprefix("analytics.whales.")

    return get_path(domain, normalized, default)


def whales_float(
    context: StrategyContext,
    path: str,
    *,
    default: float | None = None,
) -> float | None:
    return to_float(whales_path(context, path, default), default)


def whales_int(
    context: StrategyContext,
    path: str,
    *,
    default: int | None = None,
) -> int | None:
    return to_int(whales_path(context, path, default), default)


def whales_bool(
    context: StrategyContext,
    path: str,
    *,
    default: bool = False,
) -> bool:
    return to_bool(whales_path(context, path, default), default)


def whales_str(
    context: StrategyContext,
    path: str,
    *,
    default: str | None = None,
) -> str | None:
    return to_str(whales_path(context, path, default), default)


def whales_datetime(
    context: StrategyContext,
    path: str,
    *,
    default: datetime | None = None,
) -> datetime | None:
    return parse_datetime(whales_path(context, path, default))


# =============================================================================
# Payload normalization / extraction
# =============================================================================


def normalize_feature_payload(value: Any) -> dict[str, Any]:
    payload = unwrap_analytics_payload(value)

    nested_keys = (
        "pressure",
        "whale_pressure",
        "activity",
        "whale_activity",
        "large_trade",
        "cluster",
        "whale_cluster",
        "cluster_update",
        "whale_cluster_update",
        "cluster_exhaustion",
        "whale_cluster_exhaustion",
        "liquidation_context",
        "whale_liquidation_context",
        "signal",
        "result",
    )

    for key in nested_keys:
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            result = dict(nested)
            result.setdefault("_container", payload)
            return result

    return payload


def extract_whale_pressure_payload(value: Any) -> dict[str, Any]:
    return normalize_feature_payload(value)


def extract_whale_activity_payload(value: Any) -> dict[str, Any]:
    return normalize_feature_payload(value)


def extract_large_trade_payload(value: Any) -> dict[str, Any]:
    return normalize_feature_payload(value)


def extract_whale_cluster_payload(value: Any) -> dict[str, Any]:
    return normalize_feature_payload(value)


def extract_whale_cluster_update_payload(value: Any) -> dict[str, Any]:
    return normalize_feature_payload(value)


def extract_whale_cluster_exhaustion_payload(value: Any) -> dict[str, Any]:
    return normalize_feature_payload(value)


def extract_whale_liquidation_context_payload(value: Any) -> dict[str, Any]:
    return normalize_feature_payload(value)


def extract_metadata(value: Any) -> dict[str, Any]:
    payload = unwrap_analytics_payload(value)
    metadata = (
        payload.get("metadata")
        or payload.get("analytics_metadata")
        or get_path(payload, "signal.metadata")
        or get_path(payload, "result.metadata")
    )
    return as_dict(metadata)


def extract_event_time(value: Any) -> datetime | None:
    payload = unwrap_analytics_payload(value)
    return parse_datetime(
        first_non_empty(
            payload.get("timestamp"),
            payload.get("timestamp_ms"),
            payload.get("event_time"),
            payload.get("detected_at"),
            payload.get("created_at"),
            payload.get("updated_at"),
            payload.get("time"),
            get_path(payload, "metadata.timestamp"),
            get_path(payload, "metadata.event_time"),
            get_path(payload, "metadata.detected_at"),
        )
    )


def extract_symbol(value: Any) -> str:
    payload = unwrap_analytics_payload(value)
    return normalize_symbol(
        first_non_empty(
            payload.get("symbol"),
            payload.get("base_symbol"),
            payload.get("exchange_symbol"),
            get_path(payload, "scope.symbol"),
            get_path(payload, "metadata.symbol"),
        )
    )


def extract_exchange(value: Any) -> str:
    payload = unwrap_analytics_payload(value)
    return normalize_exchange(
        first_non_empty(
            payload.get("exchange"),
            payload.get("venue"),
            get_path(payload, "scope.exchange"),
            get_path(payload, "metadata.exchange"),
        )
    )


def extract_market_type(value: Any) -> str:
    payload = unwrap_analytics_payload(value)
    return normalize_market_type(
        first_non_empty(
            payload.get("market_type"),
            payload.get("instrument_type"),
            payload.get("contract_type"),
            get_path(payload, "scope.market_type"),
            get_path(payload, "metadata.market_type"),
        )
    )


def extract_timeframe(value: Any, default: str = "1m") -> str:
    payload = unwrap_analytics_payload(value)
    return (
        to_str(
            first_non_empty(
                payload.get("timeframe"),
                get_path(payload, "scope.timeframe"),
                get_path(payload, "metadata.timeframe"),
            ),
            default,
        )
        or default
    )


def extract_exchange_symbol(value: Any) -> str | None:
    payload = unwrap_analytics_payload(value)
    return to_str(
        first_non_empty(
            payload.get("exchange_symbol"),
            payload.get("symbol"),
            get_path(payload, "scope.exchange_symbol"),
            get_path(payload, "metadata.exchange_symbol"),
        )
    )


# =============================================================================
# Side helpers
# =============================================================================


BUY_LABELS: frozenset[str] = frozenset(
    {
        "buy",
        "bid",
        "buyer",
        "buyers",
        "long",
        "bull",
        "bullish",
        "ask_lift",
        "taker_buy",
        "aggressive_buy",
    }
)

SELL_LABELS: frozenset[str] = frozenset(
    {
        "sell",
        "ask",
        "seller",
        "sellers",
        "short",
        "bear",
        "bearish",
        "bid_hit",
        "taker_sell",
        "aggressive_sell",
    }
)


def side_value(value: Any, default: str = "unknown") -> str:
    label = normalize_label(value)

    if label in BUY_LABELS:
        return "buy"

    if label in SELL_LABELS:
        return "sell"

    if label in {"none", "neutral", "unknown", ""}:
        return default

    return label or default


def is_buy_side(value: Any) -> bool:
    return side_value(value) == "buy"


def is_sell_side(value: Any) -> bool:
    return side_value(value) == "sell"


def opposite_side(value: Any) -> str:
    side = side_value(value)

    if side == "buy":
        return "sell"

    if side == "sell":
        return "buy"

    return "unknown"


def side_label_to_signal_side(value: Any) -> SignalSide:
    side = side_value(value)

    if side == "buy":
        return SignalSide.LONG

    if side == "sell":
        return SignalSide.SHORT

    return SignalSide.UNKNOWN


def signal_side_to_whale_side(side: SignalSide) -> str:
    if side is SignalSide.LONG:
        return "buy"

    if side is SignalSide.SHORT:
        return "sell"

    return "unknown"


def is_directional_side(side: SignalSide) -> bool:
    return side in {SignalSide.LONG, SignalSide.SHORT}


def extract_side(value: Any) -> str:
    payload = unwrap_analytics_payload(value)
    return side_value(
        first_non_empty(
            payload.get("side"),
            payload.get("trade_side"),
            payload.get("taker_side"),
            payload.get("whale_side"),
            payload.get("dominant_side"),
            get_path(payload, "metadata.side"),
        )
    )


def extract_dominant_side(value: Any) -> str:
    payload = unwrap_analytics_payload(value)
    return side_value(
        first_non_empty(
            payload.get("dominant_side"),
            payload.get("side"),
            payload.get("pressure_side"),
            get_path(payload, "metadata.dominant_side"),
        )
    )


def extract_whale_side(value: Any) -> str:
    payload = unwrap_analytics_payload(value)
    return side_value(
        first_non_empty(
            payload.get("whale_side"),
            payload.get("absorbing_side"),
            payload.get("dominant_side"),
            payload.get("side"),
            get_path(payload, "metadata.whale_side"),
        )
    )


def extract_liquidation_side(value: Any) -> str:
    payload = unwrap_analytics_payload(value)
    return side_value(
        first_non_empty(
            payload.get("liquidation_side"),
            payload.get("liquidated_side"),
            payload.get("opposite_side"),
            payload.get("side"),
            get_path(payload, "metadata.liquidation_side"),
        )
    )


def extract_exhausted_side(value: Any) -> str:
    payload = unwrap_analytics_payload(value)
    return side_value(
        first_non_empty(
            payload.get("exhausted_side"),
            payload.get("side"),
            payload.get("cluster_side"),
            get_path(payload, "metadata.exhausted_side"),
        )
    )


def extract_cluster_side(value: Any) -> str:
    payload = unwrap_analytics_payload(value)
    return side_value(
        first_non_empty(
            payload.get("cluster_side"),
            payload.get("side"),
            payload.get("dominant_side"),
            get_path(payload, "metadata.cluster_side"),
        )
    )


def resolve_cluster_side(inputs: Mapping[str, Mapping[str, Any]]) -> str:
    cluster = inputs.get("cluster") or {}
    cluster_update = inputs.get("cluster_update") or {}
    cluster_exhaustion = inputs.get("cluster_exhaustion") or {}

    for candidate in (
        extract_cluster_side(cluster),
        extract_cluster_side(cluster_update),
        extract_cluster_side(cluster_exhaustion),
    ):
        if candidate not in {"unknown", ""}:
            return candidate

    return "unknown"


def resolve_exhausted_side(inputs: Mapping[str, Mapping[str, Any]]) -> str:
    cluster_exhaustion = inputs.get("cluster_exhaustion") or {}
    cluster_update = inputs.get("cluster_update") or {}
    cluster = inputs.get("cluster") or {}

    for candidate in (
        extract_exhausted_side(cluster_exhaustion),
        extract_exhausted_side(cluster_update),
        extract_exhausted_side(cluster),
    ):
        if candidate not in {"unknown", ""}:
            return candidate

    return "unknown"


# =============================================================================
# Whale analytics field extractors
# =============================================================================


def extract_imbalance_ratio(value: Any) -> float:
    payload = unwrap_analytics_payload(value)
    return unit_score(
        first_non_empty(
            payload.get("imbalance_ratio"),
            payload.get("pressure_imbalance_ratio"),
            payload.get("imbalance"),
            payload.get("ratio"),
            get_path(payload, "metadata.imbalance_ratio"),
        )
    )


def extract_pressure_score(value: Any) -> float:
    payload = unwrap_analytics_payload(value)

    # analytics.whales.whale_pressure does not always emit an explicit
    # pressure_score. For that canonical event, imbalance_ratio is the
    # normalized strength of the dominant whale pressure and must be accepted
    # as the pressure score fallback. Without this, pressure-only whale
    # snapshots unwrap correctly but every pressure strategy sees score=0.0.
    return unit_score(
        first_non_empty(
            payload.get("pressure_score"),
            payload.get("score"),
            payload.get("confidence"),
            payload.get("imbalance_ratio"),
            payload.get("pressure_imbalance_ratio"),
            get_path(payload, "metadata.pressure_score"),
            get_path(payload, "metadata.imbalance_ratio"),
        )
    )


def extract_context_strength(value: Any) -> float:
    payload = unwrap_analytics_payload(value)
    return unit_score(
        first_non_empty(
            payload.get("context_strength"),
            payload.get("liquidation_context_strength"),
            payload.get("strength"),
            payload.get("score"),
            get_path(payload, "metadata.context_strength"),
        )
    )


def extract_cluster_score(value: Any) -> float | None:
    payload = unwrap_analytics_payload(value)
    return first_float(
        payload.get("cluster_score"),
        payload.get("score"),
        payload.get("confidence"),
        get_path(payload, "metadata.cluster_score"),
        default=None,
    )


def resolve_cluster_score(inputs: Mapping[str, Mapping[str, Any]]) -> float | None:
    cluster = inputs.get("cluster") or {}
    cluster_update = inputs.get("cluster_update") or {}
    cluster_exhaustion = inputs.get("cluster_exhaustion") or {}

    return first_float(
        extract_cluster_score(cluster),
        extract_cluster_score(cluster_update),
        extract_cluster_score(cluster_exhaustion),
        default=None,
    )


def extract_continuation_probability(value: Any) -> float | None:
    payload = unwrap_analytics_payload(value)
    return first_float(
        payload.get("continuation_probability"),
        payload.get("continuation_prob"),
        payload.get("breakout_probability"),
        get_path(payload, "metadata.continuation_probability"),
        default=None,
    )


def resolve_continuation_probability(
    inputs: Mapping[str, Mapping[str, Any]],
) -> float | None:
    cluster = inputs.get("cluster") or {}
    cluster_update = inputs.get("cluster_update") or {}

    return first_float(
        extract_continuation_probability(cluster),
        extract_continuation_probability(cluster_update),
        default=None,
    )


def extract_exhaustion_probability(value: Any) -> float | None:
    payload = unwrap_analytics_payload(value)
    return first_float(
        payload.get("exhaustion_probability"),
        payload.get("probability"),
        payload.get("exhaustion_prob"),
        get_path(payload, "metadata.exhaustion_probability"),
        default=None,
    )


def resolve_exhaustion_probability(
    inputs: Mapping[str, Mapping[str, Any]],
) -> float | None:
    cluster = inputs.get("cluster") or {}
    cluster_update = inputs.get("cluster_update") or {}
    cluster_exhaustion = inputs.get("cluster_exhaustion") or {}

    return first_float(
        extract_exhaustion_probability(cluster_exhaustion),
        extract_exhaustion_probability(cluster_update),
        extract_exhaustion_probability(cluster),
        default=None,
    )


def extract_notional(value: Any) -> float:
    payload = unwrap_analytics_payload(value)
    return positive_float(
        first_non_empty(
            payload.get("notional"),
            payload.get("trade_notional"),
            payload.get("total_notional"),
            payload.get("volume_notional"),
            payload.get("usd_notional"),
            payload.get("quote_notional"),
            get_path(payload, "metadata.notional"),
        )
    )


def extract_total_notional(value: Any) -> float:
    payload = unwrap_analytics_payload(value)
    return positive_float(
        first_non_empty(
            payload.get("total_notional"),
            payload.get("notional"),
            payload.get("volume_notional"),
            payload.get("usd_notional"),
            payload.get("quote_notional"),
            get_path(payload, "metadata.total_notional"),
        )
    )


def extract_liquidation_notional(value: Any) -> float:
    payload = unwrap_analytics_payload(value)
    return positive_float(
        first_non_empty(
            payload.get("liquidation_notional"),
            payload.get("total_liquidation_notional"),
            payload.get("liquidated_notional"),
            payload.get("notional"),
            payload.get("total_notional"),
            get_path(payload, "metadata.liquidation_notional"),
        )
    )


def extract_trade_count(value: Any) -> int:
    payload = unwrap_analytics_payload(value)

    explicit_count = first_int(
        payload.get("trade_count"),
        payload.get("large_trade_count"),
        payload.get("count"),
        payload.get("trades_count"),
        get_path(payload, "metadata.trade_count"),
        default=None,
    )
    if explicit_count is not None:
        return max(0, explicit_count)

    # WhalePressureSignal/WhalePressureRecord expose side-specific counts,
    # not a single trade_count. Strategy filters use snapshot.trade_count, so
    # sum the side counts as the canonical activity count fallback.
    buy_count = first_int(
        payload.get("buy_trade_count"),
        payload.get("buy_trades_count"),
        get_path(payload, "metadata.buy_trade_count"),
        default=0,
    ) or 0
    sell_count = first_int(
        payload.get("sell_trade_count"),
        payload.get("sell_trades_count"),
        get_path(payload, "metadata.sell_trade_count"),
        default=0,
    ) or 0
    return max(0, buy_count + sell_count)


def extract_large_trade_notional(value: Any) -> float:
    payload = unwrap_analytics_payload(value)
    return positive_float(
        first_non_empty(
            payload.get("large_trade_notional"),
            payload.get("trade_notional"),
            payload.get("notional"),
            payload.get("total_notional"),
            get_path(payload, "whales.large_trade.notional"),
            get_path(payload, "whales.notional"),
            get_path(payload, "metadata.large_trade_notional"),
        )
    )


def extract_large_trade_zscore(value: Any) -> float | None:
    payload = unwrap_analytics_payload(value)
    return first_float(
        payload.get("zscore"),
        payload.get("z_score"),
        payload.get("notional_zscore"),
        payload.get("large_trade_zscore"),
        get_path(payload, "whales.large_trade.zscore"),
        get_path(payload, "whales.zscore"),
        get_path(payload, "metadata.large_trade_zscore"),
        default=None,
    )


def extract_reference_price(value: Any) -> float | None:
    payload = unwrap_analytics_payload(value)
    return first_float(
        payload.get("price"),
        payload.get("reference_price"),
        payload.get("mark_price"),
        payload.get("last_price"),
        payload.get("current_price"),
        payload.get("close"),
        get_path(payload, "whales.reference_price"),
        get_path(payload, "metadata.reference_price"),
        default=None,
    )


def resolve_reference_price_from_inputs(
    inputs: Mapping[str, Mapping[str, Any]],
    *,
    default: float | None = None,
) -> float | None:
    for key in (
        "large_trade",
        "activity",
        "pressure",
        "liquidation_context",
        "cluster",
        "cluster_update",
        "cluster_exhaustion",
    ):
        price = extract_reference_price(inputs.get(key) or {})
        if price is not None and price > 0:
            return price

    return default


# =============================================================================
# Confirmation / score helpers
# =============================================================================


def whale_activity_score(
    activity: Mapping[str, Any],
    *,
    min_notional: float,
    min_trade_count: int = 1,
) -> float:
    if not activity:
        return 0.0

    total_notional = extract_total_notional(activity)
    trade_count = extract_trade_count(activity)

    notional_score = normalize_notional(total_notional, min_notional)
    count_score = unit_score(trade_count / max(min_trade_count, 1))

    return weighted_score(
        {
            "notional": notional_score,
            "count": count_score,
        },
        {
            "notional": 0.75,
            "count": 0.25,
        },
    )


def large_trade_score(
    large_trade: Mapping[str, Any],
    *,
    min_notional: float,
    min_zscore: float = 0.0,
) -> float:
    if not large_trade:
        return 0.0

    notional = extract_large_trade_notional(large_trade)
    zscore = extract_large_trade_zscore(large_trade) or 0.0

    notional_score = normalize_notional(notional, min_notional)
    zscore_score = (
        unit_score(zscore / max(min_zscore * 2.0, 1.0))
        if min_zscore > 0
        else unit_score(zscore / 5.0)
    )

    return weighted_score(
        {
            "notional": notional_score,
            "zscore": zscore_score,
        },
        {
            "notional": 0.70,
            "zscore": 0.30,
        },
    )


def whale_pressure_score(
    pressure: Mapping[str, Any],
    *,
    min_imbalance_ratio: float = 0.0,
) -> float:
    if not pressure:
        return 0.0

    imbalance = extract_imbalance_ratio(pressure)
    score = extract_pressure_score(pressure)

    if min_imbalance_ratio > 0:
        threshold_component = unit_score(imbalance / max(min_imbalance_ratio, 0.0001))
    else:
        threshold_component = imbalance

    return average_score(imbalance, score, threshold_component)


def liquidation_context_score(
    liquidation_context: Mapping[str, Any],
    *,
    min_notional: float,
    min_context_strength: float = 0.0,
) -> float:
    if not liquidation_context:
        return 0.0

    strength = extract_context_strength(liquidation_context)
    notional = extract_liquidation_notional(liquidation_context)

    strength_component = (
        unit_score(strength / max(min_context_strength, 0.0001))
        if min_context_strength > 0
        else strength
    )
    notional_component = normalize_notional(notional, min_notional)

    return weighted_score(
        {
            "strength": strength_component,
            "notional": notional_component,
        },
        {
            "strength": 0.65,
            "notional": 0.35,
        },
    )


def cluster_context_score(
    inputs: Mapping[str, Mapping[str, Any]],
    *,
    min_cluster_score: float = 0.0,
) -> float:
    score = resolve_cluster_score(inputs)
    if score is None:
        return 0.0

    if min_cluster_score > 0:
        return unit_score(score / max(min_cluster_score, 0.0001))

    return unit_score(score)


def continuation_context_score(
    inputs: Mapping[str, Mapping[str, Any]],
    *,
    min_continuation_probability: float = 0.0,
) -> float:
    continuation = resolve_continuation_probability(inputs)
    if continuation is None:
        return 0.0

    if min_continuation_probability > 0:
        return unit_score(continuation / max(min_continuation_probability, 0.0001))

    return unit_score(continuation)


def exhaustion_context_score(
    inputs: Mapping[str, Mapping[str, Any]],
    *,
    min_exhaustion_probability: float = 0.0,
) -> float:
    exhaustion = resolve_exhaustion_probability(inputs)
    if exhaustion is None:
        return 0.0

    if min_exhaustion_probability > 0:
        return unit_score(exhaustion / max(min_exhaustion_probability, 0.0001))

    return unit_score(exhaustion)


def low_exhaustion_score(
    inputs: Mapping[str, Mapping[str, Any]],
    *,
    max_exhaustion_probability: float,
) -> float:
    exhaustion = resolve_exhaustion_probability(inputs)
    if exhaustion is None:
        return 1.0

    return unit_score(1.0 - (exhaustion / max(max_exhaustion_probability, 0.0001)))


# =============================================================================
# Freshness / validation helpers
# =============================================================================


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


def market_type_is_futures(value: Any) -> bool:
    market_type = extract_market_type(value)
    if not market_type:
        return True
    return market_type in FUTURES_MARKET_TYPES


def scope_matches_context(
    payload: Mapping[str, Any],
    context: StrategyContext,
    *,
    strict_timeframe: bool = False,
) -> bool:
    payload_symbol = extract_symbol(payload)
    payload_exchange = extract_exchange(payload)
    payload_timeframe = extract_timeframe(payload, "")

    context_symbol = normalize_symbol(context.symbol)
    context_timeframe = str(getattr(context.timeframe, "value", context.timeframe) or "").lower()

    if payload_symbol and context_symbol and payload_symbol != context_symbol:
        return False

    context_exchange = normalize_exchange(context.metadata.get("exchange"))
    if payload_exchange and context_exchange and payload_exchange != context_exchange:
        return False

    if strict_timeframe and payload_timeframe and context_timeframe:
        if payload_timeframe.lower() != context_timeframe:
            return False

    return True


def whale_payload_validation_reason(
    payload: Mapping[str, Any],
    *,
    context: StrategyContext | None = None,
    require_futures_market_type: bool = True,
    validate_scope: bool = True,
    strict_timeframe: bool = False,
    stale_after_seconds: float | None = DEFAULT_WHALE_FEATURE_MAX_AGE_SECONDS,
    now: datetime | None = None,
) -> str | None:
    if not payload:
        return "whale_payload_missing"

    if require_futures_market_type and not market_type_is_futures(payload):
        return "whale_payload_market_type_not_futures"

    if context is not None and validate_scope:
        if not scope_matches_context(
            payload,
            context,
            strict_timeframe=strict_timeframe,
        ):
            return "whale_payload_scope_mismatch"

    if is_stale(
        event_time=extract_event_time(payload),
        now=now,
        stale_after_seconds=stale_after_seconds,
    ):
        return "whale_payload_stale"

    return None


def whale_quality_filter_reason(
    *,
    inputs: Mapping[str, Mapping[str, Any]],
    min_score: float = 0.0,
    min_confidence: float = 0.0,
    score: float = 0.0,
    confidence: float = 0.0,
    required_keys: tuple[str, ...] = (),
) -> str | None:
    for key in required_keys:
        if not inputs.get(key):
            return f"missing_required_whale_input:{key}"

    if score < min_score:
        return "whale_score_below_threshold"

    if confidence < min_confidence:
        return "whale_confidence_below_threshold"

    return None


# =============================================================================
# Scoring DTO / helpers
# =============================================================================


@dataclass(slots=True)
class ScoreBreakdown:
    _logger = logging.getLogger(__name__ + ".ScoreBreakdown")
    score: float = 0.0
    confidence: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    confirmations: list[str] = field(default_factory=list)

    def normalize(self) -> ScoreBreakdown:
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
        if weight_f <= 0:
            continue

        total_weight += weight_f
        total_value += unit_score(values.get(key, default), default) * weight_f

    if total_weight <= 0:
        return unit_score(default)

    return unit_score(total_value / total_weight)


def average_score(*values: float | Decimal | None, default: float = 0.0) -> float:
    valid = [
        unit_score(float(value))
        for value in values
        if value is not None
    ]

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


# =============================================================================
# Source-feature helpers
# =============================================================================


def source_features_from_paths(*paths: str) -> list[str]:
    result: list[str] = []

    for path in paths:
        if not isinstance(path, str) or not path.strip():
            continue

        normalized = path.strip()

        if normalized.startswith("whales."):
            result.append(normalized)
        elif normalized.startswith("analytics.whales."):
            result.append(normalized.replace("analytics.", "", 1))
        else:
            result.append(f"whales.{normalized}")

    return list(dict.fromkeys(result))


def base_whales_source_features() -> list[str]:
    return source_features_from_paths(
        "pressure",
        "activity",
        "large_trade",
        "cluster",
        "cluster_update",
        "cluster_exhaustion",
        "liquidation_context",
        "dominant_side",
        "whale_side",
        "liquidation_side",
        "exhausted_side",
        "cluster_side",
        "imbalance_ratio",
        "context_strength",
        "cluster_score",
        "continuation_probability",
        "exhaustion_probability",
        "total_notional",
        "trade_count",
        "large_trade_notional",
        "large_trade_zscore",
    )


def whale_pressure_source_features() -> list[str]:
    return source_features_from_paths(
        "pressure",
        "dominant_side",
        "imbalance_ratio",
        "pressure_score",
    )


def whale_activity_source_features() -> list[str]:
    return source_features_from_paths(
        "activity",
        "total_notional",
        "trade_count",
        "side",
    )


def large_trade_source_features() -> list[str]:
    return source_features_from_paths(
        "large_trade",
        "large_trade_notional",
        "large_trade_zscore",
        "side",
        "price",
    )


def whale_cluster_source_features() -> list[str]:
    return source_features_from_paths(
        "cluster",
        "cluster_update",
        "cluster_exhaustion",
        "cluster_score",
        "cluster_side",
        "continuation_probability",
        "exhaustion_probability",
        "exhausted_side",
    )


def whale_liquidation_context_source_features() -> list[str]:
    return source_features_from_paths(
        "liquidation_context",
        "liquidation_side",
        "whale_side",
        "context_strength",
        "liquidation_notional",
    )


def whale_absorption_source_features() -> list[str]:
    return list(
        dict.fromkeys(
            [
                *whale_pressure_source_features(),
                *whale_liquidation_context_source_features(),
                *whale_cluster_source_features(),
                *whale_activity_source_features(),
                *large_trade_source_features(),
            ]
        )
    )


def whale_breakout_source_features() -> list[str]:
    return list(
        dict.fromkeys(
            [
                *whale_activity_source_features(),
                *whale_pressure_source_features(),
                *large_trade_source_features(),
                *whale_cluster_source_features(),
                *whale_liquidation_context_source_features(),
            ]
        )
    )


def whale_accumulation_source_features() -> list[str]:
    return source_features_from_paths(
        "activity",
        "pressure",
        "cluster",
        "cluster_update",
        "dominant_side",
        "cluster_side",
        "total_notional",
        "trade_count",
        "continuation_probability",
        "exhaustion_probability",
    )


def whale_distribution_source_features() -> list[str]:
    return source_features_from_paths(
        "activity",
        "pressure",
        "cluster",
        "cluster_update",
        "large_trade",
        "dominant_side",
        "cluster_side",
        "total_notional",
        "large_trade_notional",
        "continuation_probability",
        "exhaustion_probability",
    )


def whale_liquidation_reversal_source_features() -> list[str]:
    return source_features_from_paths(
        "liquidation_context",
        "pressure",
        "cluster_exhaustion",
        "activity",
        "large_trade",
        "liquidation_side",
        "whale_side",
        "context_strength",
        "liquidation_notional",
        "exhaustion_probability",
    )

def whale_large_trade_source_features() -> list[str]:
    return source_features_from_paths(
        "large_trade",
        "large_trade.notional",
        "large_trade.zscore",
        "large_trade.side",
        "large_trade.quantity",
        "side",
        "notional",
        "zscore",
        "reference_price",
        "price",
    )
