# trading_system/strategy/strategies/price_action/utils.py

from __future__ import annotations
import logging

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from analytics.price_action.enums import (
    FVGDirection,
    FVGEventType,
    FVGStatus,
    LevelStatus,
    LevelType,
    MarketBias,
    SREventType,
    StructureEventType,
    StructureLayer,
    SwingType,
    TrendDirection,
    TrendEventType,
    TrendRegime,
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
    SignalBuilder, not to concrete price-action strategies.
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
        market_structure.external.bias
        support_resistance.internal.nearest_support
        fair_value_gap.external.nearest_bullish_gap
        trend.internal.continuation_probability
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
    Backward-compatible unwrap for analytics.price_action.* envelopes.

    Concrete strategies should usually receive normalized StrategyContext.
    This helper is useful while domain_data still contains analytics envelopes
    or result-like nested payloads.
    """
    raw = dict(payload)
    inner = raw.get("payload")

    if isinstance(inner, Mapping):
        inner_dict = dict(inner)

        for key in (
            "state",
            "composite",
            "price_action",
            "market_structure",
            "structure",
            "support_resistance",
            "sr",
            "fair_value_gap",
            "fvg",
            "trend",
            "result",
            "snapshot",
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


def normalize_state_payload(payload: Any) -> dict[str, Any]:
    """
    Normalize common analytics payload shapes:

    - dataclass / object with to_dict()
    - direct state dict
    - {"state": {...}}
    - {"payload": {"state": {...}}}
    """
    mapping = as_mapping(payload)
    if mapping is None:
        return {}

    data = dict(mapping)

    state = data.get("state")
    if isinstance(state, Mapping):
        result = dict(state)
        result.setdefault("_source_feature", data.get("_source_feature"))
        result.setdefault("_container", data)
        return result

    payload_value = data.get("payload")
    if isinstance(payload_value, Mapping):
        nested = normalize_state_payload(payload_value)
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
            "respected",
            "retested",
            "aligned",
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
            "misaligned",
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


def distance_score(
    distance_pct: Any,
    *,
    max_distance_pct: float,
) -> float:
    distance = abs(to_float(distance_pct, 0.0) or 0.0)

    if max_distance_pct <= 0:
        return 1.0 if distance <= 0 else 0.0

    return clamp(1.0 - (distance / max_distance_pct), 0.0, 1.0)


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


def parse_structure_layer(
    value: Any,
    default: StructureLayer | None = None,
) -> StructureLayer | None:
    return parse_enum(value, StructureLayer, default)  # type: ignore[return-value]


def parse_market_bias(
    value: Any,
    default: MarketBias = MarketBias.UNKNOWN,
) -> MarketBias:
    return parse_enum(value, MarketBias, default)  # type: ignore[return-value]


def parse_swing_type(
    value: Any,
    default: SwingType | None = None,
) -> SwingType | None:
    return parse_enum(value, SwingType, default)  # type: ignore[return-value]


def parse_structure_event_type(
    value: Any,
    default: StructureEventType | None = None,
) -> StructureEventType | None:
    return parse_enum(value, StructureEventType, default)  # type: ignore[return-value]


def parse_fvg_direction(
    value: Any,
    default: FVGDirection | None = None,
) -> FVGDirection | None:
    return parse_enum(value, FVGDirection, default)  # type: ignore[return-value]


def parse_fvg_status(
    value: Any,
    default: FVGStatus | None = None,
) -> FVGStatus | None:
    return parse_enum(value, FVGStatus, default)  # type: ignore[return-value]


def parse_fvg_event_type(
    value: Any,
    default: FVGEventType | None = None,
) -> FVGEventType | None:
    return parse_enum(value, FVGEventType, default)  # type: ignore[return-value]


def parse_level_type(
    value: Any,
    default: LevelType | None = None,
) -> LevelType | None:
    return parse_enum(value, LevelType, default)  # type: ignore[return-value]


def parse_level_status(
    value: Any,
    default: LevelStatus | None = None,
) -> LevelStatus | None:
    return parse_enum(value, LevelStatus, default)  # type: ignore[return-value]


def parse_sr_event_type(
    value: Any,
    default: SREventType | None = None,
) -> SREventType | None:
    return parse_enum(value, SREventType, default)  # type: ignore[return-value]


def parse_trend_direction(
    value: Any,
    default: TrendDirection | None = None,
) -> TrendDirection | None:
    return parse_enum(value, TrendDirection, default)  # type: ignore[return-value]


def parse_trend_regime(
    value: Any,
    default: TrendRegime | None = None,
) -> TrendRegime | None:
    return parse_enum(value, TrendRegime, default)  # type: ignore[return-value]


def parse_trend_event_type(
    value: Any,
    default: TrendEventType | None = None,
) -> TrendEventType | None:
    return parse_enum(value, TrendEventType, default)  # type: ignore[return-value]


# =============================================================================
# StrategyContext price-action helpers
# =============================================================================


PRICE_ACTION_DOMAIN_ALIASES: dict[str, tuple[str, ...]] = {
    "composite": (
        "composite",
        "price_action",
        "analytics.price_action",
        "state",
        "result",
        "snapshot",
    ),
    "market_structure": (
        "market_structure",
        "structure",
        "price_action.market_structure",
        "analytics.price_action.market_structure",
    ),
    "support_resistance": (
        "support_resistance",
        "sr",
        "price_action.support_resistance",
        "analytics.price_action.support_resistance",
    ),
    "fair_value_gap": (
        "fair_value_gap",
        "fvg",
        "price_action.fair_value_gap",
        "price_action.fvg",
        "analytics.price_action.fair_value_gap",
        "analytics.price_action.fvg",
    ),
    "trend": (
        "trend",
        "price_action.trend",
        "analytics.price_action.trend",
    ),
    "liquidity_levels": (
        "liquidity_levels",
        "liquidity",
        "price_action.liquidity_levels",
        "price_action.liquidity",
        "analytics.price_action.liquidity_levels",
        "analytics.price_action.liquidity",
    ),
}


def price_action_domain(context: StrategyContext) -> dict[str, Any]:
    return dict(context.domain_dict(FeatureSource.PRICE_ACTION))


def price_action_item(
    context: StrategyContext,
    key: str,
    default: Any = None,
) -> Any:
    domain = price_action_domain(context)

    if key in domain:
        return domain[key]

    for alias in PRICE_ACTION_DOMAIN_ALIASES.get(key, ()):
        value = get_path(domain, alias, default=None)
        if value is not None:
            return value

    return default


def price_action_path(
    context: StrategyContext,
    path: str,
    default: Any = None,
) -> Any:
    """
    Read price-action value from StrategyContext.

    Priority:
    1. exact feature name;
    2. price_action-prefixed feature name;
    3. analytics.price_action-prefixed feature name;
    4. FeatureSource.PRICE_ACTION domain dotted path.
    """
    if not isinstance(path, str) or not path.strip():
        return default

    normalized = path.strip()
    price_action_feature_name = (
        normalized
        if normalized.startswith("price_action.")
        else f"price_action.{normalized}"
    )
    analytics_feature_name = (
        normalized
        if normalized.startswith("analytics.price_action.")
        else f"analytics.price_action.{normalized}"
    )

    if context.has_feature(normalized):
        return context.get_feature(normalized)

    if context.has_feature(price_action_feature_name):
        return context.get_feature(price_action_feature_name)

    if context.has_feature(analytics_feature_name):
        return context.get_feature(analytics_feature_name)

    domain = price_action_domain(context)

    if normalized.startswith("price_action."):
        normalized = normalized.removeprefix("price_action.")
    elif normalized.startswith("analytics.price_action."):
        normalized = normalized.removeprefix("analytics.price_action.")

    return get_path(domain, normalized, default)


def price_action_float(
    context: StrategyContext,
    path: str,
    *,
    default: float | None = None,
) -> float | None:
    return to_float(price_action_path(context, path, default), default)


def price_action_int(
    context: StrategyContext,
    path: str,
    *,
    default: int | None = None,
) -> int | None:
    return to_int(price_action_path(context, path, default), default)


def price_action_unit_score(
    context: StrategyContext,
    path: str,
    *,
    default: float = 0.0,
) -> float:
    return unit_score(price_action_path(context, path, default), default)


def price_action_signed_score(
    context: StrategyContext,
    path: str,
    *,
    default: float = 0.0,
) -> float:
    return signed_score(price_action_path(context, path, default), default)


def price_action_bool(
    context: StrategyContext,
    path: str,
    *,
    default: bool = False,
) -> bool:
    return to_bool(price_action_path(context, path, default), default)


def price_action_str(
    context: StrategyContext,
    path: str,
    *,
    default: str | None = None,
) -> str | None:
    return to_str(price_action_path(context, path, default), default)


def price_action_datetime(
    context: StrategyContext,
    path: str,
    *,
    default: datetime | None = None,
) -> datetime | None:
    return parse_datetime(price_action_path(context, path, default))


def extract_price_action_state(context: StrategyContext) -> dict[str, Any]:
    """
    Return normalized composite price-action state from StrategyContext.
    """
    candidates: list[Any] = [
        price_action_item(context, "composite"),
        price_action_path(context, "composite"),
        price_action_path(context, "analytics.price_action"),
        price_action_path(context, "price_action"),
    ]

    for candidate in candidates:
        state = normalize_state_payload(candidate)
        if state:
            return state

    domain = price_action_domain(context)
    if domain:
        return normalize_state_payload(domain)

    return {}


def extract_price_action_module(
    context: StrategyContext,
    module_name: str,
    *,
    aliases: tuple[str, ...] = (),
) -> dict[str, Any]:
    """
    Return normalized module payload from composite state, direct module feature,
    or backward-compatible aliases.
    """
    if not isinstance(module_name, str) or not module_name.strip():
        return {}

    canonical = module_name.strip()
    candidate_names = (
        canonical,
        *PRICE_ACTION_DOMAIN_ALIASES.get(canonical, ()),
        *aliases,
    )

    composite = extract_price_action_state(context)
    for name in candidate_names:
        module_payload = get_path(composite, name, None)
        state = normalize_state_payload(module_payload)
        if state:
            state.setdefault("_source_feature", composite.get("_source_feature", "price_action.composite"))
            state.setdefault("_source_module", canonical)
            return state

    for name in candidate_names:
        direct = price_action_path(context, name, None)
        state = normalize_state_payload(direct)
        if state:
            state.setdefault("_source_feature", name)
            state.setdefault("_source_module", canonical)
            return state

    item = price_action_item(context, canonical)
    state = normalize_state_payload(item)
    if state:
        state.setdefault("_source_feature", canonical)
        state.setdefault("_source_module", canonical)
        return state

    return {}


# =============================================================================
# Scope / freshness helpers
# =============================================================================


def extract_scope(payload: Any) -> dict[str, Any]:
    mapping = as_mapping(payload)
    if mapping is None:
        return {}

    data = dict(mapping)
    scope = as_dict(data.get("scope"))

    key = (
        data.get("key")
        or scope.get("key")
        or data.get("scope_key")
        or scope.get("scope_key")
    )

    return {
        "exchange": first_non_empty(
            data.get("exchange"),
            scope.get("exchange"),
            get_path(data, "metadata.exchange"),
        ),
        "market_type": first_non_empty(
            data.get("market_type"),
            scope.get("market_type"),
            get_path(data, "metadata.market_type"),
        ),
        "symbol": first_non_empty(
            data.get("symbol"),
            scope.get("symbol"),
            get_path(data, "metadata.symbol"),
        ),
        "exchange_symbol": first_non_empty(
            data.get("exchange_symbol"),
            scope.get("exchange_symbol"),
            get_path(data, "metadata.exchange_symbol"),
        ),
        "timeframe": first_non_empty(
            data.get("timeframe"),
            scope.get("timeframe"),
            get_path(data, "metadata.timeframe"),
        ),
        "key": key,
    }


def scope_matches_context(
    context: StrategyContext,
    payload: Any,
    *,
    require_exchange: bool = False,
    require_market_type: bool = False,
    require_timeframe: bool = False,
) -> bool:
    scope = extract_scope(payload)

    symbol = to_str(scope.get("symbol"))
    if symbol and symbol.upper() != context.symbol.upper():
        return False

    context_exchange = to_str(context.metadata.get("exchange"))
    payload_exchange = to_str(scope.get("exchange"))
    if require_exchange and context_exchange and payload_exchange:
        if payload_exchange.lower() != context_exchange.lower():
            return False

    context_market_type = to_str(context.metadata.get("market_type"))
    payload_market_type = to_str(scope.get("market_type"))
    if require_market_type and context_market_type and payload_market_type:
        if payload_market_type != context_market_type:
            return False

    context_timeframe = str(getattr(context.timeframe, "value", context.timeframe))
    payload_timeframe = to_str(scope.get("timeframe"))
    if require_timeframe and context_timeframe and payload_timeframe:
        if payload_timeframe.lower() != context_timeframe.lower():
            return False

    return True


def extract_last_update(value: Any) -> datetime | None:
    return parse_datetime(
        first_present(
            value,
            (
                "last_update",
                "updated_at",
                "timestamp",
                "event_time",
                "created_at",
                "time",
                "metadata.last_update",
                "metadata.updated_at",
            ),
        )
    )


def extract_last_event(value: Any) -> dict[str, Any] | None:
    for path in (
        "last_event",
        "internal.last_event",
        "external.last_event",
        "last_break_event",
        "last_signal",
    ):
        event = as_mapping(get_path(value, path))
        if event:
            return dict(event)
    return None


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
    "support",
    "demand",
}

SHORT_VALUES: set[str] = {
    "short",
    "sell",
    "bear",
    "bearish",
    "down",
    "downside",
    "negative",
    "resistance",
    "supply",
}

UNKNOWN_VALUES: set[str] = {
    "unknown",
    "none",
    "neutral",
    "flat",
    "mixed",
    "sideways",
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


def market_bias_to_side(value: Any) -> SignalSide:
    bias = parse_market_bias(value)

    label = normalize_label(bias)

    if "bull" in label or label in {"uptrend", "up", "long"}:
        return SignalSide.LONG

    if "bear" in label or label in {"downtrend", "down", "short"}:
        return SignalSide.SHORT

    return SignalSide.UNKNOWN


def trend_direction_to_side(value: Any) -> SignalSide:
    direction = parse_trend_direction(value)

    label = normalize_label(direction)

    if label in {"up", "uptrend", "bullish", "long"}:
        return SignalSide.LONG

    if label in {"down", "downtrend", "bearish", "short"}:
        return SignalSide.SHORT

    return SignalSide.UNKNOWN


def fvg_direction_to_side(value: Any) -> SignalSide:
    direction = parse_fvg_direction(value)

    label = normalize_label(direction)

    if "bull" in label or label in {"up", "long", "demand"}:
        return SignalSide.LONG

    if "bear" in label or label in {"down", "short", "supply"}:
        return SignalSide.SHORT

    return SignalSide.UNKNOWN


def structure_event_to_side(
    event: Any,
    *,
    reverse_on_choch: bool = True,
    reverse_on_mss: bool = True,
) -> SignalSide:
    event_type = parse_structure_event_type(get_attr_or_key(event, "event_type"))
    direction = market_bias_to_side(
        first_non_empty(
            get_attr_or_key(event, "direction"),
            get_attr_or_key(event, "bias"),
            get_attr_or_key(event, "side"),
        )
    )

    if event_type is None:
        return direction

    label = normalize_label(event_type)

    if label == "bos":
        return direction

    if label == "choch" and reverse_on_choch:
        return direction

    if label == "mss" and reverse_on_mss:
        return direction

    return direction


def level_reaction_to_side(level: Any, event: Any | None = None) -> SignalSide:
    level_type = parse_level_type(get_attr_or_key(level, "level_type"))
    status = parse_level_status(get_attr_or_key(level, "status"))
    event_type = parse_sr_event_type(get_attr_or_key(event, "event_type")) if event else None

    level_label = normalize_label(level_type)
    status_label = normalize_label(status)
    event_label = normalize_label(event_type)

    if event_label in {"support_rejection", "support_hold", "support_retest"}:
        return SignalSide.LONG

    if event_label in {"resistance_rejection", "resistance_hold", "resistance_retest"}:
        return SignalSide.SHORT

    if event_label in {"resistance_break", "resistance_breakout", "flip_support"}:
        return SignalSide.LONG

    if event_label in {"support_break", "support_breakdown", "flip_resistance"}:
        return SignalSide.SHORT

    if "support" in status_label and "flip" in status_label:
        return SignalSide.LONG

    if "resistance" in status_label and "flip" in status_label:
        return SignalSide.SHORT

    if "support" in level_label:
        return SignalSide.LONG

    if "resistance" in level_label:
        return SignalSide.SHORT

    return SignalSide.UNKNOWN


# =============================================================================
# Layer selection helpers
# =============================================================================


def select_primary_layer(
    view: Any,
    *,
    prefer_external_layer: bool = True,
) -> dict[str, Any]:
    external = as_dict(get_attr_or_key(view, "external"))
    internal = as_dict(get_attr_or_key(view, "internal"))

    if prefer_external_layer:
        return external or internal

    return internal or external


def select_secondary_layer(
    view: Any,
    *,
    prefer_external_layer: bool = True,
) -> dict[str, Any]:
    external = as_dict(get_attr_or_key(view, "external"))
    internal = as_dict(get_attr_or_key(view, "internal"))

    if prefer_external_layer:
        return internal if external else {}

    return external if internal else {}


def layer_confidence(layer: Any) -> float:
    return unit_score(get_path(layer, "confidence", 0.0))


def layer_strength(layer: Any) -> float:
    return unit_score(
        first_present(
            layer,
            (
                "strength",
                "trend_strength",
                "score",
                "structure_score",
                "level_strength",
                "gap_strength",
            ),
            default=0.0,
        )
    )


# =============================================================================
# Score DTO / scoring helpers
# =============================================================================


@dataclass(slots=True)
class ScoreBreakdown:
    """
    Small reusable score DTO for concrete price-action strategies.
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


# =============================================================================
# Quality filters
# =============================================================================


def quality_filter_reason(
    value: Any,
    *,
    min_confidence: float = 0.0,
    min_score: float = 0.0,
    stale_after_seconds: float | None = None,
    now: datetime | None = None,
) -> str | None:
    if value is None:
        return "missing_price_action_context"

    confidence = unit_score(
        first_present(
            value,
            (
                "confidence",
                "score",
                "strength",
                "layer.confidence",
                "primary.confidence",
            ),
            default=0.0,
        )
    )
    if confidence < min_confidence:
        return "price_action_confidence_below_threshold"

    score = unit_score(
        first_present(
            value,
            (
                "score",
                "strength",
                "trend_strength",
                "structure_score",
                "level_strength",
                "gap_strength",
                "confidence",
            ),
            default=confidence,
        )
    )
    if score < min_score:
        return "price_action_score_below_threshold"

    event_time = extract_last_update(value)
    if is_stale(
        event_time=event_time,
        now=now,
        stale_after_seconds=stale_after_seconds,
    ):
        return "price_action_context_stale"

    return None


# =============================================================================
# Source-feature helpers
# =============================================================================


def source_features_from_paths(*paths: str) -> list[str]:
    result: list[str] = []

    for path in paths:
        if not isinstance(path, str) or not path.strip():
            continue

        normalized = path.strip()

        if normalized.startswith("price_action."):
            result.append(normalized)
        elif normalized.startswith("analytics.price_action."):
            result.append(normalized.replace("analytics.", "", 1))
        else:
            result.append(f"price_action.{normalized}")

    return list(dict.fromkeys(result))


def base_price_action_source_features() -> list[str]:
    return source_features_from_paths(
        "composite",
        "market_structure",
        "support_resistance",
        "fair_value_gap",
        "trend",
    )


def market_structure_source_features() -> list[str]:
    return source_features_from_paths(
        "market_structure",
        "market_structure.internal",
        "market_structure.external",
        "market_structure.last_break_event",
        "market_structure.mtf_alignment",
    )


def trend_source_features() -> list[str]:
    return source_features_from_paths(
        "trend",
        "trend.internal",
        "trend.external",
        "trend.last_signal",
        "trend.internal_external_alignment",
        "trend.higher_timeframe_alignment",
        "trend.overall_trend_score",
    )


def support_resistance_source_features() -> list[str]:
    return source_features_from_paths(
        "support_resistance",
        "support_resistance.internal",
        "support_resistance.external",
        "support_resistance.last_event",
        "support_resistance.nearest_support",
        "support_resistance.nearest_resistance",
    )


def fvg_source_features() -> list[str]:
    return source_features_from_paths(
        "fair_value_gap",
        "fair_value_gap.internal",
        "fair_value_gap.external",
        "fair_value_gap.last_event",
        "fair_value_gap.nearest_bullish_gap",
        "fair_value_gap.nearest_bearish_gap",
    )