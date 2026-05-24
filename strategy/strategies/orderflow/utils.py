# trading_system/strategy/strategies/orderflow/utils.py

from __future__ import annotations
import logging

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
    SignalBuilder, not to orderflow strategies.
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

    Supports both nested dictionaries and literal dotted keys. This is important
    for StrategyContext domain payloads where the normalizer may contain both
    forms at the same time, for example::

        {"orderflow.cvd.delta_ratio": 0.8}
        {"orderflow": {"cvd": {"delta_ratio": 0.8}}}
    """
    if not isinstance(path, str) or not path.strip():
        return default

    normalized_path = path.strip()
    root_mapping = as_mapping(value)
    if root_mapping is not None and normalized_path in root_mapping:
        item = root_mapping.get(normalized_path)
        return default if item is None else item

    current = value
    parts = [part.strip() for part in normalized_path.split(".") if part.strip()]

    for index, part in enumerate(parts):
        if current is None:
            return default

        current_mapping = as_mapping(current)
        if current_mapping is not None:
            remaining = ".".join(parts[index:])
            if remaining in current_mapping:
                item = current_mapping.get(remaining)
                return default if item is None else item

            if part in current_mapping:
                current = current_mapping.get(part)
                continue

            return default

        current = getattr(current, part, None)

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
    Backward-compatible unwrap for analytics.orderflow.* envelopes.

    Concrete strategies should usually receive normalized StrategyContext.
    This helper is useful while domain_data still contains analytics envelopes
    or result-like nested payloads.
    """
    raw = dict(payload)
    inner = raw.get("payload")

    if isinstance(inner, Mapping):
        inner_dict = dict(inner)

        for key in (
            "composite",
            "snapshot",
            "orderflow",
            "cvd",
            "cvd_stats",
            "volume_delta",
            "volume_delta_stats",
            "aggressive_trades",
            "aggressive",
            "orderbook_imbalance",
            "imbalance",
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


def percent_score(
    value: Any,
    *,
    scale: float = 1.0,
    default: float = 0.0,
) -> float:
    parsed = abs(to_float(value, default) or default)
    if scale <= 0:
        return unit_score(parsed)
    return unit_score(parsed / scale)


def magnitude_score(
    value: Any,
    *,
    scale: float,
    default: float = 0.0,
) -> float:
    parsed = abs(to_float(value, default) or default)
    if scale <= 0:
        return unit_score(parsed)
    return unit_score(parsed / scale)


# =============================================================================
# StrategyContext orderflow helpers
# =============================================================================


ORDERFLOW_DOMAIN_ALIASES: dict[str, tuple[str, ...]] = {
    "composite": (
        "composite",
        "snapshot",
        "orderflow",
        "orderflow_snapshot",
        "composite_snapshot",
        "result",
    ),
    "cvd": (
        "cvd",
        "cvd_stats",
        "cvd_result",
        "metrics.cvd",
    ),
    "volume_delta": (
        "volume_delta",
        "volume_delta_stats",
        "volume_delta_result",
        "delta",
        "metrics.volume_delta",
    ),
    "aggressive_trades": (
        "aggressive_trades",
        "aggressive",
        "aggressive_stats",
        "aggressive_trades_stats",
        "metrics.aggressive_trades",
    ),
    "orderbook_imbalance": (
        "orderbook_imbalance",
        "imbalance",
        "orderbook",
        "orderbook_stats",
        "metrics.orderbook_imbalance",
    ),
    "signal": (
        "signal",
        "orderflow_signal",
        "analytics_signal",
    ),
}


def orderflow_domain(context: StrategyContext) -> dict[str, Any]:
    return dict(context.domain_dict(FeatureSource.ORDERFLOW))


def orderflow_item(
    context: StrategyContext,
    key: str,
    default: Any = None,
) -> Any:
    domain = orderflow_domain(context)

    if key in domain:
        return domain[key]

    for alias in ORDERFLOW_DOMAIN_ALIASES.get(key, ()):
        value = get_path(domain, alias, default=None)
        if value is not None:
            return value

    return default


def orderflow_path(
    context: StrategyContext,
    path: str,
    default: Any = None,
) -> Any:
    """
    Read orderflow value from StrategyContext.

    Priority:
    1. exact feature name;
    2. orderflow-prefixed feature name;
    3. orderflow domain dotted path.
    """
    if not isinstance(path, str) or not path.strip():
        return default

    normalized = path.strip()
    feature_name = (
        normalized
        if normalized.startswith("orderflow.")
        else f"orderflow.{normalized}"
    )

    if context.has_feature(normalized):
        return context.get_feature(normalized)

    if context.has_feature(feature_name):
        return context.get_feature(feature_name)

    domain = orderflow_domain(context)

    if normalized.startswith("orderflow."):
        normalized = normalized.removeprefix("orderflow.")

    return get_path(domain, normalized, default)


def orderflow_float(
    context: StrategyContext,
    path: str,
    *,
    default: float | None = None,
) -> float | None:
    return to_float(orderflow_path(context, path, default), default)


def orderflow_int(
    context: StrategyContext,
    path: str,
    *,
    default: int | None = None,
) -> int | None:
    return to_int(orderflow_path(context, path, default), default)


def orderflow_unit_score(
    context: StrategyContext,
    path: str,
    *,
    default: float = 0.0,
) -> float:
    return unit_score(orderflow_path(context, path, default), default)


def orderflow_signed_score(
    context: StrategyContext,
    path: str,
    *,
    default: float = 0.0,
) -> float:
    return signed_score(orderflow_path(context, path, default), default)


def orderflow_abs_score(
    context: StrategyContext,
    path: str,
    *,
    default: float = 0.0,
) -> float:
    return abs_score(orderflow_path(context, path, default), default)


def orderflow_bool(
    context: StrategyContext,
    path: str,
    *,
    default: bool = False,
) -> bool:
    return to_bool(orderflow_path(context, path, default), default)


def orderflow_str(
    context: StrategyContext,
    path: str,
    *,
    default: str | None = None,
) -> str | None:
    return to_str(orderflow_path(context, path, default), default)


def orderflow_datetime(
    context: StrategyContext,
    path: str,
    *,
    default: datetime | None = None,
) -> datetime | None:
    return parse_datetime(orderflow_path(context, path, default))


# =============================================================================
# Snapshot / metric extractors
# =============================================================================


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


def extract_price_change_pct(value: Any) -> float:
    return signed_score(
        first_present(
            value,
            (
                "price_change_pct",
                "price.change_pct",
                "cvd.price_change_pct",
                "last_price_change_pct",
            ),
            default=0.0,
        )
    )


def extract_last_price(value: Any) -> float | None:
    return to_float(
        first_present(
            value,
            (
                "last_price",
                "price",
                "mid_price",
                "cvd.last_price",
                "orderbook_imbalance.mid_price",
            ),
            default=None,
        )
    )


def extract_trades_count(value: Any) -> int:
    return to_int(
        first_present(
            value,
            (
                "trades_count",
                "cvd.trades_count",
                "volume_delta.trades_count",
                "aggressive_trades.trades_count",
            ),
            default=0,
        ),
        default=0,
    ) or 0


def extract_total_volume(value: Any) -> float:
    return to_float(
        first_present(
            value,
            (
                "total_volume",
                "cvd.total_volume",
                "volume_delta.total_volume",
                "aggressive_trades.total_volume",
            ),
            default=0.0,
        ),
        default=0.0,
    ) or 0.0


def extract_total_notional(value: Any) -> float:
    return to_float(
        first_present(
            value,
            (
                "total_notional",
                "cvd.total_notional",
                "volume_delta.total_notional",
                "aggressive_trades.total_notional",
            ),
            default=0.0,
        ),
        default=0.0,
    ) or 0.0


def extract_cvd_delta_ratio(value: Any) -> float:
    return signed_score(
        first_present(
            value,
            (
                "cvd_delta_ratio",
                "delta_ratio",
                "cvd.delta_ratio",
                "cvd.cvd_delta_ratio",
            ),
            default=0.0,
        )
    )


def extract_cvd_change_pct(value: Any) -> float:
    return signed_score(
        first_present(
            value,
            (
                "cvd_change_pct",
                "cvd.change_pct",
                "cvd.cvd_change_pct",
                "change_pct",
            ),
            default=0.0,
        )
    )


def extract_cvd_slope(value: Any) -> float:
    return signed_score(
        first_present(
            value,
            (
                "cvd_slope",
                "slope",
                "cvd.cvd_slope",
                "cvd.slope",
            ),
            default=0.0,
        )
    )


def extract_cvd_value(value: Any) -> float:
    return to_float(
        first_present(
            value,
            (
                "cvd_value",
                "cvd.value",
                "cvd.cvd_value",
                "cvd_close",
                "cvd.cvd_close",
            ),
            default=0.0,
        ),
        default=0.0,
    ) or 0.0


def extract_volume_delta(value: Any) -> float:
    return to_float(
        first_present(
            value,
            (
                "volume_delta",
                "volume_delta.volume_delta",
                "delta.volume_delta",
            ),
            default=0.0,
        ),
        default=0.0,
    ) or 0.0


def extract_volume_delta_ratio(value: Any) -> float:
    return signed_score(
        first_present(
            value,
            (
                "volume_delta_ratio",
                "volume_delta.delta_ratio",
                "volume_delta.ratio",
                "delta_ratio",
            ),
            default=0.0,
        )
    )


def extract_cumulative_volume_delta(value: Any) -> float:
    return to_float(
        first_present(
            value,
            (
                "cumulative_volume_delta",
                "volume_delta.cumulative_volume_delta",
                "cumulative_delta",
            ),
            default=0.0,
        ),
        default=0.0,
    ) or 0.0


def extract_notional_delta(value: Any) -> float:
    return to_float(
        first_present(
            value,
            (
                "notional_delta",
                "volume_delta.notional_delta",
                "delta.notional_delta",
            ),
            default=0.0,
        ),
        default=0.0,
    ) or 0.0


def extract_cumulative_notional_delta(value: Any) -> float:
    return to_float(
        first_present(
            value,
            (
                "cumulative_notional_delta",
                "volume_delta.cumulative_notional_delta",
                "cumulative_notional",
            ),
            default=0.0,
        ),
        default=0.0,
    ) or 0.0


def extract_buy_volume(value: Any) -> float:
    return to_float(
        first_present(
            value,
            (
                "buy_volume",
                "volume_delta.buy_volume",
                "cvd.buy_volume",
            ),
            default=0.0,
        ),
        default=0.0,
    ) or 0.0


def extract_sell_volume(value: Any) -> float:
    return to_float(
        first_present(
            value,
            (
                "sell_volume",
                "volume_delta.sell_volume",
                "cvd.sell_volume",
            ),
            default=0.0,
        ),
        default=0.0,
    ) or 0.0


def extract_aggressive_buy_ratio(value: Any) -> float:
    return unit_score(
        first_present(
            value,
            (
                "aggressive_buy_ratio",
                "aggressive_trades.buy_ratio",
                "aggressive_trades.aggressive_buy_ratio",
                "buy_ratio",
            ),
            default=0.0,
        )
    )


def extract_aggressive_sell_ratio(value: Any) -> float:
    return unit_score(
        first_present(
            value,
            (
                "aggressive_sell_ratio",
                "aggressive_trades.sell_ratio",
                "aggressive_trades.aggressive_sell_ratio",
                "sell_ratio",
            ),
            default=0.0,
        )
    )


def extract_aggressive_burst_score(value: Any) -> float:
    return unit_score(
        first_present(
            value,
            (
                "aggressive_burst_score",
                "aggressive_trades.burst_score",
                "aggressive_trades.aggressive_burst_score",
                "burst_score",
            ),
            default=0.0,
        )
    )


def extract_aggressive_net_notional_delta(value: Any) -> float:
    return to_float(
        first_present(
            value,
            (
                "aggressive_net_notional_delta",
                "aggressive_trades.net_notional_delta",
                "aggressive_trades.aggressive_net_notional_delta",
            ),
            default=0.0,
        ),
        default=0.0,
    ) or 0.0


def extract_aggressive_net_volume_delta(value: Any) -> float:
    return to_float(
        first_present(
            value,
            (
                "aggressive_net_volume_delta",
                "aggressive_trades.net_volume_delta",
                "aggressive_trades.aggressive_net_volume_delta",
            ),
            default=0.0,
        ),
        default=0.0,
    ) or 0.0


def extract_large_buy_trades(value: Any) -> int:
    return to_int(
        first_present(
            value,
            (
                "large_buy_trades",
                "aggressive_trades.large_buy_trades",
            ),
            default=0,
        ),
        default=0,
    ) or 0


def extract_large_sell_trades(value: Any) -> int:
    return to_int(
        first_present(
            value,
            (
                "large_sell_trades",
                "aggressive_trades.large_sell_trades",
            ),
            default=0,
        ),
        default=0,
    ) or 0


def extract_orderbook_imbalance_ratio(value: Any) -> float:
    return signed_score(
        first_present(
            value,
            (
                "orderbook_imbalance_ratio",
                "orderbook_imbalance.ratio",
                "orderbook_imbalance.imbalance_ratio",
                "imbalance_ratio",
                "ratio",
            ),
            default=0.0,
        )
    )


def extract_orderbook_imbalance_diff(value: Any) -> float:
    explicit = first_present(
        value,
        (
            "signed_orderbook_imbalance",
            "orderbook_imbalance_diff",
            "orderbook_imbalance.diff",
            "orderbook_imbalance.imbalance_diff",
            "imbalance_diff",
            "diff",
        ),
        default=None,
    )
    if explicit is not None:
        return signed_score(explicit)

    ratio = extract_orderbook_imbalance_ratio(value)
    return signed_imbalance_from_ratio(ratio)


def signed_imbalance_from_ratio(value: Any) -> float:
    """
    Normalize orderbook imbalance to signed -1..1.

    If incoming ratio is already signed, keep it.
    If it looks like 0..1 bid ratio, convert:
        0.50 -> 0.0
        1.00 -> +1.0
        0.00 -> -1.0
    """
    ratio = to_float(value, 0.0) or 0.0

    if -1.0 <= ratio <= 1.0:
        if 0.0 <= ratio <= 1.0:
            return clamp((ratio - 0.5) * 2.0, -1.0, 1.0)
        return clamp(ratio, -1.0, 1.0)

    return clamp(ratio, -1.0, 1.0)


def directional_aggressive_ratio(value: Any, side: SignalSide) -> float:
    if side is SignalSide.LONG:
        return extract_aggressive_buy_ratio(value)

    if side is SignalSide.SHORT:
        return extract_aggressive_sell_ratio(value)

    return 0.0


def directional_large_trades(value: Any, side: SignalSide) -> int:
    if side is SignalSide.LONG:
        return extract_large_buy_trades(value)

    if side is SignalSide.SHORT:
        return extract_large_sell_trades(value)

    return 0


def directional_aggressive_notional_delta(value: Any, side: SignalSide) -> float:
    delta = extract_aggressive_net_notional_delta(value)

    if side is SignalSide.LONG:
        return delta

    if side is SignalSide.SHORT:
        return -delta

    return 0.0


def notional_delta_ratio(value: Any) -> float:
    total_notional = extract_total_notional(value)
    if total_notional <= 0:
        return 0.0

    return clamp(extract_notional_delta(value) / total_notional, -1.0, 1.0)


def volume_delta_ratio_from_volumes(value: Any) -> float:
    buy_volume = extract_buy_volume(value)
    sell_volume = extract_sell_volume(value)
    total = buy_volume + sell_volume

    if total <= 0:
        return extract_volume_delta_ratio(value)

    return clamp((buy_volume - sell_volume) / total, -1.0, 1.0)


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
    "bid",
    "bid_side",
}

SHORT_VALUES: set[str] = {
    "short",
    "sell",
    "bear",
    "bearish",
    "down",
    "downside",
    "negative",
    "ask",
    "ask_side",
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


def continuation_side_from_snapshot(
    snapshot: Any,
    *,
    min_abs_price_change_pct: float = 0.0,
    min_cvd_delta_ratio: float = 0.0,
    min_volume_delta_ratio: float = 0.0,
    min_aggressive_buy_ratio: float = 0.50,
    min_aggressive_sell_ratio: float = 0.50,
    max_orderbook_contradiction: float | None = None,
) -> SignalSide:
    price = extract_price_change_pct(snapshot)
    cvd = extract_cvd_delta_ratio(snapshot)
    volume_delta = extract_volume_delta_ratio(snapshot)
    buy_ratio = extract_aggressive_buy_ratio(snapshot)
    sell_ratio = extract_aggressive_sell_ratio(snapshot)
    imbalance = extract_orderbook_imbalance_diff(snapshot)

    long_ok = (
        price >= min_abs_price_change_pct
        and cvd >= min_cvd_delta_ratio
        and volume_delta >= min_volume_delta_ratio
        and buy_ratio >= min_aggressive_buy_ratio
        and buy_ratio > sell_ratio
    )

    short_ok = (
        price <= -min_abs_price_change_pct
        and cvd <= -min_cvd_delta_ratio
        and volume_delta <= -min_volume_delta_ratio
        and sell_ratio >= min_aggressive_sell_ratio
        and sell_ratio > buy_ratio
    )

    if max_orderbook_contradiction is not None:
        if long_ok and imbalance < -abs(max_orderbook_contradiction):
            long_ok = False
        if short_ok and imbalance > abs(max_orderbook_contradiction):
            short_ok = False

    if long_ok and not short_ok:
        return SignalSide.LONG

    if short_ok and not long_ok:
        return SignalSide.SHORT

    return SignalSide.UNKNOWN


def reversal_side_from_snapshot(
    snapshot: Any,
    *,
    min_abs_price_change_pct: float = 0.0,
    min_abs_cvd_delta_ratio: float = 0.0,
    min_abs_volume_delta_ratio: float = 0.0,
    min_aggressive_buy_ratio_for_long: float = 0.50,
    min_aggressive_sell_ratio_for_short: float = 0.50,
    require_aggressive_confirmation: bool = True,
    require_orderbook_confirmation: bool = False,
    min_bullish_imbalance_for_long: float = 0.0,
    min_bearish_imbalance_for_short: float = 0.0,
) -> SignalSide:
    price = extract_price_change_pct(snapshot)
    cvd = extract_cvd_delta_ratio(snapshot)
    volume_delta = extract_volume_delta_ratio(snapshot)
    notional_delta = extract_notional_delta(snapshot)
    aggressive_delta = extract_aggressive_net_notional_delta(snapshot)
    buy_ratio = extract_aggressive_buy_ratio(snapshot)
    sell_ratio = extract_aggressive_sell_ratio(snapshot)
    large_buy = extract_large_buy_trades(snapshot)
    large_sell = extract_large_sell_trades(snapshot)
    imbalance = extract_orderbook_imbalance_diff(snapshot)

    long_ok = (
        price <= -min_abs_price_change_pct
        and cvd >= min_abs_cvd_delta_ratio
        and volume_delta >= min_abs_volume_delta_ratio
        and notional_delta >= 0
        and aggressive_delta >= 0
    )

    short_ok = (
        price >= min_abs_price_change_pct
        and cvd <= -min_abs_cvd_delta_ratio
        and volume_delta <= -min_abs_volume_delta_ratio
        and notional_delta <= 0
        and aggressive_delta <= 0
    )

    if require_aggressive_confirmation:
        long_ok = (
            long_ok
            and buy_ratio >= min_aggressive_buy_ratio_for_long
            and buy_ratio > sell_ratio
            and large_buy >= large_sell
        )
        short_ok = (
            short_ok
            and sell_ratio >= min_aggressive_sell_ratio_for_short
            and sell_ratio > buy_ratio
            and large_sell >= large_buy
        )

    if require_orderbook_confirmation:
        long_ok = long_ok and imbalance >= min_bullish_imbalance_for_long
        short_ok = short_ok and imbalance <= -min_bearish_imbalance_for_short

    if long_ok and not short_ok:
        return SignalSide.LONG

    if short_ok and not long_ok:
        return SignalSide.SHORT

    return SignalSide.UNKNOWN


def cvd_divergence_side_from_snapshot(
    snapshot: Any,
    *,
    min_abs_price_change_pct: float = 0.0,
    min_abs_cvd_change_pct: float = 0.0,
    min_abs_delta_ratio: float = 0.0,
    min_abs_cvd_slope: float = 0.0,
) -> SignalSide:
    price = extract_price_change_pct(snapshot)
    cvd_change = extract_cvd_change_pct(snapshot)
    delta_ratio = extract_cvd_delta_ratio(snapshot)
    cvd_slope = extract_cvd_slope(snapshot)

    bullish_divergence = (
        price <= -abs(min_abs_price_change_pct)
        and cvd_change >= abs(min_abs_cvd_change_pct)
        and delta_ratio >= abs(min_abs_delta_ratio)
        and cvd_slope >= abs(min_abs_cvd_slope)
    )

    bearish_divergence = (
        price >= abs(min_abs_price_change_pct)
        and cvd_change <= -abs(min_abs_cvd_change_pct)
        and delta_ratio <= -abs(min_abs_delta_ratio)
        and cvd_slope <= -abs(min_abs_cvd_slope)
    )

    if bullish_divergence and not bearish_divergence:
        return SignalSide.LONG

    if bearish_divergence and not bullish_divergence:
        return SignalSide.SHORT

    return SignalSide.UNKNOWN


# =============================================================================
# Score DTO / scoring helpers
# =============================================================================


@dataclass(slots=True)
class ScoreBreakdown:
    """
    Small reusable score DTO for concrete orderflow strategies.
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
    min_trades_count: int = 0,
    min_total_volume: float = 0.0,
    min_total_notional: float = 0.0,
    stale_after_seconds: float | None = None,
    now: datetime | None = None,
) -> str | None:
    if value is None:
        return "missing_orderflow_context"

    trades_count = extract_trades_count(value)
    if trades_count < min_trades_count:
        return "orderflow_trades_count_below_threshold"

    total_volume = extract_total_volume(value)
    if total_volume < min_total_volume:
        return "orderflow_total_volume_below_threshold"

    total_notional = extract_total_notional(value)
    if total_notional < min_total_notional:
        return "orderflow_total_notional_below_threshold"

    event_time = extract_event_time(value)
    if is_stale(
        event_time=event_time,
        now=now,
        stale_after_seconds=stale_after_seconds,
    ):
        return "orderflow_context_stale"

    return None


def cvd_divergence_filter_reason(
    value: Any,
    *,
    min_abs_price_change_pct: float = 0.0,
    min_abs_cvd_change_pct: float = 0.0,
    min_abs_delta_ratio: float = 0.0,
    min_abs_cvd_slope: float = 0.0,
    min_trades_count: int = 0,
    min_strength_for_signal: float = 0.0,
) -> str | None:
    common = quality_filter_reason(value, min_trades_count=min_trades_count)
    if common is not None:
        return common

    side = cvd_divergence_side_from_snapshot(
        value,
        min_abs_price_change_pct=min_abs_price_change_pct,
        min_abs_cvd_change_pct=min_abs_cvd_change_pct,
        min_abs_delta_ratio=min_abs_delta_ratio,
        min_abs_cvd_slope=min_abs_cvd_slope,
    )
    if not is_directional_side(side):
        return "cvd_divergence_not_detected"

    strength = average_score(
        percent_score(abs(extract_price_change_pct(value)), scale=1.0),
        percent_score(abs(extract_cvd_change_pct(value)), scale=1.0),
        ratio_score(abs(extract_cvd_delta_ratio(value)), scale=0.35),
        ratio_score(abs(extract_cvd_slope(value)), scale=1.0),
    )
    if strength < min_strength_for_signal:
        return "cvd_divergence_strength_below_threshold"

    return None


def continuation_filter_reason(
    value: Any,
    *,
    min_trades_count: int = 0,
    min_total_volume: float = 0.0,
    min_abs_price_change_pct: float = 0.0,
    min_cvd_delta_ratio: float = 0.0,
    min_volume_delta_ratio: float = 0.0,
    min_aggressive_buy_ratio: float = 0.50,
    min_aggressive_sell_ratio: float = 0.50,
) -> str | None:
    common = quality_filter_reason(
        value,
        min_trades_count=min_trades_count,
        min_total_volume=min_total_volume,
    )
    if common is not None:
        return common

    side = continuation_side_from_snapshot(
        value,
        min_abs_price_change_pct=min_abs_price_change_pct,
        min_cvd_delta_ratio=min_cvd_delta_ratio,
        min_volume_delta_ratio=min_volume_delta_ratio,
        min_aggressive_buy_ratio=min_aggressive_buy_ratio,
        min_aggressive_sell_ratio=min_aggressive_sell_ratio,
    )
    if not is_directional_side(side):
        return "orderflow_continuation_not_detected"

    return None


def reversal_filter_reason(
    value: Any,
    *,
    min_trades_count: int = 0,
    min_total_volume: float = 0.0,
    min_abs_price_change_pct: float = 0.0,
    min_abs_cvd_delta_ratio: float = 0.0,
    min_abs_volume_delta_ratio: float = 0.0,
    min_aggressive_buy_ratio_for_long: float = 0.50,
    min_aggressive_sell_ratio_for_short: float = 0.50,
    require_aggressive_confirmation: bool = True,
    require_orderbook_confirmation: bool = False,
) -> str | None:
    common = quality_filter_reason(
        value,
        min_trades_count=min_trades_count,
        min_total_volume=min_total_volume,
    )
    if common is not None:
        return common

    side = reversal_side_from_snapshot(
        value,
        min_abs_price_change_pct=min_abs_price_change_pct,
        min_abs_cvd_delta_ratio=min_abs_cvd_delta_ratio,
        min_abs_volume_delta_ratio=min_abs_volume_delta_ratio,
        min_aggressive_buy_ratio_for_long=min_aggressive_buy_ratio_for_long,
        min_aggressive_sell_ratio_for_short=min_aggressive_sell_ratio_for_short,
        require_aggressive_confirmation=require_aggressive_confirmation,
        require_orderbook_confirmation=require_orderbook_confirmation,
    )
    if not is_directional_side(side):
        return "orderflow_reversal_not_detected"

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
        if normalized.startswith("orderflow."):
            result.append(normalized)
        else:
            result.append(f"orderflow.{normalized}")

    return list(dict.fromkeys(result))


def base_orderflow_source_features() -> list[str]:
    return source_features_from_paths(
        "composite",
        "cvd",
        "volume_delta",
        "aggressive_trades",
        "orderbook_imbalance",
    )


def cvd_source_features() -> list[str]:
    return source_features_from_paths(
        "cvd",
        "cvd.delta_ratio",
        "cvd.cvd_change_pct",
        "cvd.cvd_slope",
        "cvd.price_change_pct",
    )


def continuation_source_features() -> list[str]:
    return source_features_from_paths(
        "cvd.delta_ratio",
        "cvd.cvd_change_pct",
        "cvd.cvd_slope",
        "volume_delta.delta_ratio",
        "volume_delta.volume_delta",
        "volume_delta.cumulative_volume_delta",
        "aggressive_trades.buy_ratio",
        "aggressive_trades.sell_ratio",
        "aggressive_trades.burst_score",
        "orderbook_imbalance.imbalance_diff",
    )


def reversal_source_features() -> list[str]:
    return source_features_from_paths(
        "cvd.delta_ratio",
        "cvd.cvd_change_pct",
        "volume_delta.delta_ratio",
        "notional_delta",
        "aggressive_trades.buy_ratio",
        "aggressive_trades.sell_ratio",
        "aggressive_trades.net_notional_delta",
        "orderbook_imbalance.imbalance_diff",
    )