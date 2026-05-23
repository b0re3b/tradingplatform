# trading_system/strategy/strategies/liquidity/utils.py

from __future__ import annotations
import logging

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from math import isfinite
from typing import Any

from analytics.liquidity.enums import (
    LiquidityBias,
    LiquidityLevelType,
    LiquiditySide,
)
from analytics.liquidity.models import (
    LiquidityLevel,
    LiquidityMapSnapshot,
    LiquidityZone,
    StopCluster,
)
from ...enums import FeatureSource, SignalSide
from ...models import StrategyContext, clamp, ensure_aware_utc, utcnow

DECIMAL_ZERO = Decimal("0")


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
    SignalBuilder, not to liquidity strategies.
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
# Mapping / nested helpers
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
    Backward-compatible unwrap for analytics.liquidity.* envelopes.

    Concrete strategies should normally receive StrategyContext with
    FeatureSource.LIQUIDITY domain data. This helper is useful when generic
    domain_data still contains analytics envelopes or nested snapshot wrappers.
    """
    raw = dict(payload)
    inner = raw.get("payload")

    if isinstance(inner, Mapping):
        inner_dict = dict(inner)

        for key in (
            "snapshot",
            "liquidity_map_snapshot",
            "map_snapshot",
            "last_snapshot",
            "liquidity",
            "data",
            "result",
        ):
            nested_value = inner_dict.get(key)
            if isinstance(nested_value, Mapping):
                nested = dict(nested_value)
                nested.setdefault("_envelope", raw)
                nested.setdefault("_container", inner_dict)
                return nested

            if isinstance(nested_value, LiquidityMapSnapshot):
                return {
                    "snapshot": nested_value,
                    "_envelope": raw,
                    "_container": inner_dict,
                }

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
        parsed = float(value)
        return parsed if isfinite(parsed) else default

    if isinstance(value, Enum):
        return to_float(value.value, default)

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return default

        try:
            parsed = float(raw)
        except ValueError:
            return default

        return parsed if isfinite(parsed) else default

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
            "active",
            "valid",
            "confirmed",
        }:
            return True

        if normalized in {
            "0",
            "false",
            "no",
            "n",
            "off",
            "inactive",
            "invalid",
            "expired",
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
# StrategyContext liquidity helpers
# =============================================================================


LIQUIDITY_DOMAIN_ALIASES: dict[str, tuple[str, ...]] = {
    "snapshot": (
        "snapshot",
        "liquidity_map_snapshot",
        "map_snapshot",
        "last_snapshot",
    ),
    "current_price": (
        "current_price",
        "price",
        "mark_price",
        "last_price",
        "close",
    ),
    "signal": (
        "signal",
        "liquidity_signal",
        "analytics_signal",
    ),
    "levels": (
        "levels",
        "active_levels",
        "liquidity_levels",
    ),
    "clusters": (
        "clusters",
        "stop_clusters",
        "liquidity_clusters",
    ),
    "zones": (
        "zones",
        "liquidity_zones",
    ),
}


SNAPSHOT_FEATURE_KEYS: tuple[str, ...] = (
    "liquidity_map_snapshot",
    "liquidity.snapshot",
    "liquidity_snapshot",
    "liquidity.map.snapshot",
    "analytics.liquidity.map.snapshot",
    "analytics.liquidity.map.updated",
)


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


def liquidity_domain(context: StrategyContext) -> dict[str, Any]:
    return dict(context.domain_dict(FeatureSource.LIQUIDITY))


def liquidity_item(
    context: StrategyContext,
    key: str,
    default: Any = None,
) -> Any:
    domain = liquidity_domain(context)

    if key in domain:
        return domain[key]

    for alias in LIQUIDITY_DOMAIN_ALIASES.get(key, ()):
        value = get_path(domain, alias, default=None)
        if value is not None:
            return value

    return default


def liquidity_path(
    context: StrategyContext,
    path: str,
    default: Any = None,
) -> Any:
    """
    Read liquidity value from StrategyContext.

    Priority:
    1. exact feature name;
    2. liquidity-prefixed feature name;
    3. liquidity domain dotted path.
    """
    if not isinstance(path, str) or not path.strip():
        return default

    normalized = path.strip()
    feature_name = (
        normalized
        if normalized.startswith("liquidity.")
        else f"liquidity.{normalized}"
    )

    if context.has_feature(normalized):
        return context.get_feature(normalized)

    if context.has_feature(feature_name):
        return context.get_feature(feature_name)

    domain = liquidity_domain(context)

    if normalized.startswith("liquidity."):
        normalized = normalized.removeprefix("liquidity.")

    return get_path(domain, normalized, default)


def liquidity_float(
    context: StrategyContext,
    path: str,
    *,
    default: float | None = None,
) -> float | None:
    return to_float(liquidity_path(context, path, default), default)


def liquidity_unit_score(
    context: StrategyContext,
    path: str,
    *,
    default: float = 0.0,
) -> float:
    return unit_score(liquidity_path(context, path, default), default)


def liquidity_signed_score(
    context: StrategyContext,
    path: str,
    *,
    default: float = 0.0,
) -> float:
    return signed_score(liquidity_path(context, path, default), default)


def liquidity_abs_score(
    context: StrategyContext,
    path: str,
    *,
    default: float = 0.0,
) -> float:
    return abs_score(liquidity_path(context, path, default), default)


def liquidity_bool(
    context: StrategyContext,
    path: str,
    *,
    default: bool = False,
) -> bool:
    return to_bool(liquidity_path(context, path, default), default)


def liquidity_str(
    context: StrategyContext,
    path: str,
    *,
    default: str | None = None,
) -> str | None:
    return to_str(liquidity_path(context, path, default), default)


def liquidity_datetime(
    context: StrategyContext,
    path: str,
    *,
    default: datetime | None = None,
) -> datetime | None:
    return parse_datetime(liquidity_path(context, path, default))


def unwrap_snapshot_candidate(candidate: Any) -> LiquidityMapSnapshot | None:
    if isinstance(candidate, LiquidityMapSnapshot):
        return candidate

    if candidate is None:
        return None

    if isinstance(candidate, Mapping):
        for key in ("snapshot", "value", "data", "payload"):
            nested = candidate.get(key)
            snapshot = unwrap_snapshot_candidate(nested)
            if snapshot is not None:
                return snapshot

    for attr in ("snapshot", "value", "data", "payload"):
        nested = getattr(candidate, attr, None)
        snapshot = unwrap_snapshot_candidate(nested)
        if snapshot is not None:
            return snapshot

    return None


def liquidity_snapshot_from_context(
    context: StrategyContext,
) -> LiquidityMapSnapshot | None:
    domain = liquidity_domain(context)
    candidates: list[Any] = []

    for key in LIQUIDITY_DOMAIN_ALIASES["snapshot"]:
        candidates.append(get_path(domain, key, default=None))

    legacy_liquidity_context = getattr(context, "liquidity", None)
    if legacy_liquidity_context is not None:
        for key in LIQUIDITY_DOMAIN_ALIASES["snapshot"]:
            candidates.append(get_path(legacy_liquidity_context, key, default=None))

    for key in SNAPSHOT_FEATURE_KEYS:
        getter = getattr(context, "get_feature", None)
        if callable(getter):
            try:
                candidates.append(getter(key))
            except Exception:
                pass

        snapshot_getter = getattr(context, "get_feature_snapshot", None)
        if callable(snapshot_getter):
            try:
                candidates.append(snapshot_getter(key))
            except Exception:
                pass

    for candidate in candidates:
        snapshot = unwrap_snapshot_candidate(candidate)
        if snapshot is not None:
            return snapshot

    return None


# =============================================================================
# Price / distance / side helpers
# =============================================================================


def value_of(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def timeframe_value(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value
    return str(value or "").strip().lower()


def is_futures_market_type(value: Any) -> bool:
    return normalize_label(value) in FUTURES_MARKET_TYPES


def reference_price(item: Any) -> float:
    for attr in (
        "price",
        "center_price",
        "level_price",
        "reference_price",
        "mid_price",
    ):
        value = to_float(getattr(item, attr, None))
        if value is not None and value > 0:
            return value

    if isinstance(item, Mapping):
        for key in (
            "price",
            "center_price",
            "level_price",
            "reference_price",
            "mid_price",
        ):
            value = to_float(item.get(key))
            if value is not None and value > 0:
                return value

    return 0.0


def distance_pct(price: float, current_price: float) -> float:
    if price <= 0 or current_price <= 0:
        return float("inf")
    return abs(price - current_price) / current_price


def distance_score(
    *,
    price: float,
    current_price: float,
    max_distance_pct: float,
    min_distance_pct: float = 0.0,
) -> float:
    if current_price <= 0 or price <= 0 or max_distance_pct <= 0:
        return 0.0

    distance = distance_pct(price, current_price)

    if distance < min_distance_pct:
        return 0.0

    if distance > max_distance_pct:
        return 0.0

    return unit_score(1.0 - distance / max_distance_pct)


def is_above_price(item: Any, current_price: float) -> bool:
    return reference_price(item) > current_price


def is_below_price(item: Any, current_price: float) -> bool:
    return reference_price(item) < current_price


def side_to_liquidity_side(side: SignalSide) -> LiquiditySide | None:
    if side is SignalSide.LONG:
        return LiquiditySide.BUY_SIDE

    if side is SignalSide.SHORT:
        return LiquiditySide.SELL_SIDE

    return None


def opposite_liquidity_side(side: LiquiditySide) -> LiquiditySide:
    if side is LiquiditySide.BUY_SIDE:
        return LiquiditySide.SELL_SIDE

    if side is LiquiditySide.SELL_SIDE:
        return LiquiditySide.BUY_SIDE

    return side


def opposite_signal_side(side: SignalSide) -> SignalSide:
    if side is SignalSide.LONG:
        return SignalSide.SHORT

    if side is SignalSide.SHORT:
        return SignalSide.LONG

    return SignalSide.UNKNOWN


def is_directional_side(side: SignalSide) -> bool:
    return side in {SignalSide.LONG, SignalSide.SHORT}


# =============================================================================
# Liquidity item state helpers
# =============================================================================


def sweep_status_label(item: Any) -> str:
    return normalize_label(getattr(item, "sweep_status", None))


def is_swept_item(item: Any) -> bool:
    method = getattr(item, "is_swept", None)
    if callable(method):
        try:
            return bool(method())
        except Exception:
            pass

    status = sweep_status_label(item)
    return status == "swept"


def is_partially_swept_item(item: Any) -> bool:
    method = getattr(item, "is_partially_swept", None)
    if callable(method):
        try:
            return bool(method())
        except Exception:
            pass

    status = sweep_status_label(item)
    return status == "partially_swept"


def is_invalidated_item(item: Any) -> bool:
    method = getattr(item, "is_invalidated", None)
    if callable(method):
        try:
            return bool(method())
        except Exception:
            pass

    status = normalize_label(getattr(item, "status", None))
    return status == "invalidated"


def is_expired_item(item: Any) -> bool:
    method = getattr(item, "is_expired", None)
    if callable(method):
        try:
            return bool(method())
        except Exception:
            pass

    status = normalize_label(getattr(item, "status", None))
    return status == "expired"


def is_terminal_item(item: Any) -> bool:
    if is_swept_item(item) or is_partially_swept_item(item):
        return True

    if is_invalidated_item(item) or is_expired_item(item):
        return True

    status = sweep_status_label(item)
    return status in {"swept", "partially_swept", "invalidated", "expired"}


def is_active_item(item: Any) -> bool:
    if is_terminal_item(item):
        return False

    method = getattr(item, "is_active", None)
    if callable(method):
        try:
            return bool(method())
        except Exception:
            return True

    status = normalize_label(getattr(item, "status", None))
    if status in {"inactive", "invalidated", "expired"}:
        return False

    return True


def is_valid_price_item(item: Any) -> bool:
    return reference_price(item) > 0


def item_key(item: Any) -> str:
    return str(
        getattr(item, "key", None)
        or getattr(item, "id", None)
        or f"{item.__class__.__name__}:{reference_price(item):.12f}"
    )


def item_strength(item: Any) -> float:
    candidates = [
        getattr(item, "score", None),
        getattr(item, "strength", None),
        getattr(item, "confidence", None),
        getattr(item, "volume_score", None),
        getattr(item, "notional_score", None),
    ]

    for candidate in candidates:
        value = to_float(candidate)
        if value is not None:
            return unit_score(value)

    notional = safe_decimal(getattr(item, "total_notional", None))
    if notional <= 0:
        notional = safe_decimal(getattr(item, "notional", None))
    if notional <= 0:
        notional = safe_decimal(getattr(item, "total_notional_usd", None))

    if notional > 0:
        return unit_score(float(min(notional / Decimal("1000000"), Decimal("1"))))

    return 0.0


def dedupe_liquidity_items(
    items: Sequence[LiquidityLevel | StopCluster | LiquidityZone],
) -> list[LiquidityLevel | StopCluster | LiquidityZone]:
    result: dict[str, LiquidityLevel | StopCluster | LiquidityZone] = {}

    for item in items:
        key = item_key(item)
        existing = result.get(key)

        if existing is None:
            result[key] = item
            continue

        if item_strength(item) > item_strength(existing):
            result[key] = item

    return list(result.values())


# =============================================================================
# Snapshot collection helpers
# =============================================================================


def active_levels(snapshot: LiquidityMapSnapshot) -> list[LiquidityLevel]:
    levels = list(getattr(snapshot, "active_levels", []) or [])
    return [level for level in levels if is_active_item(level)]


def equal_levels(snapshot: LiquidityMapSnapshot) -> list[LiquidityLevel]:
    levels = list(getattr(snapshot, "equal_levels", []) or [])
    return [level for level in levels if is_active_item(level)]


def all_levels(snapshot: LiquidityMapSnapshot) -> list[LiquidityLevel]:
    return list(dict.fromkeys([*active_levels(snapshot), *equal_levels(snapshot)]))


def stop_clusters(snapshot: LiquidityMapSnapshot) -> list[StopCluster]:
    return list(getattr(snapshot, "stop_clusters", []) or [])


def zones(snapshot: LiquidityMapSnapshot) -> list[LiquidityZone]:
    return list(getattr(snapshot, "zones", []) or [])


def directional_levels(
    snapshot: LiquidityMapSnapshot,
    side: LiquiditySide,
) -> list[LiquidityLevel]:
    return [
        level
        for level in [*active_levels(snapshot), *equal_levels(snapshot)]
        if getattr(level, "side", None) == side
    ]


def directional_clusters(
    snapshot: LiquidityMapSnapshot,
    side: LiquiditySide,
) -> list[StopCluster]:
    return [
        cluster
        for cluster in stop_clusters(snapshot)
        if getattr(cluster, "side", None) == side
    ]


def directional_zones(
    snapshot: LiquidityMapSnapshot,
    side: LiquiditySide,
) -> list[LiquidityZone]:
    return [
        zone
        for zone in zones(snapshot)
        if getattr(zone, "side", None) == side
    ]


def collect_targets_above(
    snapshot: LiquidityMapSnapshot,
    current_price: float,
) -> list[LiquidityLevel | StopCluster]:
    candidates: list[LiquidityLevel | StopCluster] = []

    nearest = getattr(snapshot, "nearest_above_level", None)
    strongest = getattr(snapshot, "strongest_cluster_above", None)

    for item in (nearest, strongest):
        if item is not None and reference_price(item) > current_price:
            candidates.append(item)

    candidates.extend(
        level
        for level in active_levels(snapshot)
        if reference_price(level) > current_price
    )
    candidates.extend(
        cluster
        for cluster in stop_clusters(snapshot)
        if reference_price(cluster) > current_price
    )

    deduped = dedupe_liquidity_items(candidates)
    return sorted(deduped, key=reference_price)


def collect_targets_below(
    snapshot: LiquidityMapSnapshot,
    current_price: float,
) -> list[LiquidityLevel | StopCluster]:
    candidates: list[LiquidityLevel | StopCluster] = []

    nearest = getattr(snapshot, "nearest_below_level", None)
    strongest = getattr(snapshot, "strongest_cluster_below", None)

    for item in (nearest, strongest):
        if item is not None and reference_price(item) < current_price:
            candidates.append(item)

    candidates.extend(
        level
        for level in active_levels(snapshot)
        if reference_price(level) < current_price
    )
    candidates.extend(
        cluster
        for cluster in stop_clusters(snapshot)
        if reference_price(cluster) < current_price
    )

    deduped = dedupe_liquidity_items(candidates)
    return sorted(deduped, key=reference_price, reverse=True)


def best_zone_for_side(
    *,
    snapshot: LiquidityMapSnapshot,
    side: LiquiditySide,
    current_price: float,
) -> LiquidityZone | None:
    candidates = [
        zone
        for zone in directional_zones(snapshot, side)
        if reference_price(zone) > 0
    ]

    if not candidates:
        return None

    def rank(zone: LiquidityZone) -> tuple[float, float]:
        distance = distance_pct(reference_price(zone), current_price)
        return (unit_score(getattr(zone, "score", 0.0)), -distance)

    return max(candidates, key=rank)


# =============================================================================
# Snapshot score helpers
# =============================================================================


def snapshot_signal_bias(snapshot: LiquidityMapSnapshot) -> LiquidityBias | None:
    signal = getattr(snapshot, "signal", None)
    if signal is None:
        return None

    bias = getattr(signal, "bias", None)
    if isinstance(bias, LiquidityBias):
        return bias

    try:
        return LiquidityBias(str(bias))
    except Exception:
        return None


def analytics_signal_confidence(snapshot: LiquidityMapSnapshot) -> float:
    signal = getattr(snapshot, "signal", None)
    if signal is None:
        metadata_confidence = get_path(getattr(snapshot, "metadata", None), "confidence")
        return unit_score(metadata_confidence)

    return unit_score(getattr(signal, "confidence", 0.0))


def snapshot_liquidity_strength(snapshot: LiquidityMapSnapshot) -> float:
    above = unit_score(getattr(snapshot, "above_liquidity_score", 0.0))
    below = unit_score(getattr(snapshot, "below_liquidity_score", 0.0))
    pressure = abs(signed_score(getattr(snapshot, "liquidity_pressure_score", 0.0)))
    return unit_score(max(above, below, pressure))


def magnet_score_up(snapshot: LiquidityMapSnapshot) -> float:
    candidates = [
        getattr(snapshot, "upside_magnet_score", None),
        get_path(getattr(snapshot, "metadata", None), "upside_magnet_score"),
        getattr(snapshot, "above_liquidity_score", None),
    ]
    return max(unit_score(item) for item in candidates if item is not None) if any(
        item is not None for item in candidates
    ) else 0.0


def magnet_score_down(snapshot: LiquidityMapSnapshot) -> float:
    candidates = [
        getattr(snapshot, "downside_magnet_score", None),
        get_path(getattr(snapshot, "metadata", None), "downside_magnet_score"),
        getattr(snapshot, "below_liquidity_score", None),
    ]
    return max(unit_score(item) for item in candidates if item is not None) if any(
        item is not None for item in candidates
    ) else 0.0


def sweep_risk_up(snapshot: LiquidityMapSnapshot) -> float:
    candidates = [
        getattr(snapshot, "upside_sweep_risk", None),
        getattr(snapshot, "sweep_risk_up", None),
        get_path(getattr(snapshot, "metadata", None), "upside_sweep_risk"),
    ]
    return max(unit_score(item) for item in candidates if item is not None) if any(
        item is not None for item in candidates
    ) else 0.0


def sweep_risk_down(snapshot: LiquidityMapSnapshot) -> float:
    candidates = [
        getattr(snapshot, "downside_sweep_risk", None),
        getattr(snapshot, "sweep_risk_down", None),
        get_path(getattr(snapshot, "metadata", None), "downside_sweep_risk"),
    ]
    return max(unit_score(item) for item in candidates if item is not None) if any(
        item is not None for item in candidates
    ) else 0.0


def zone_score(
    snapshot: LiquidityMapSnapshot,
    side: LiquiditySide,
) -> float:
    directional = directional_zones(snapshot, side)
    if not directional:
        return 0.0

    return max(unit_score(getattr(zone, "score", 0.0)) for zone in directional)


def upside_bias_edge(snapshot: LiquidityMapSnapshot) -> float:
    pressure = max(signed_score(getattr(snapshot, "liquidity_pressure_score", 0.0)), 0.0)

    bias_bonus = 0.10 if getattr(snapshot, "bias", None) == LiquidityBias.UP else 0.0
    signal_bias_bonus = 0.05 if snapshot_signal_bias(snapshot) == LiquidityBias.UP else 0.0

    return unit_score(
        0.24 * unit_score(getattr(snapshot, "above_liquidity_score", 0.0))
        + 0.22 * magnet_score_up(snapshot)
        + 0.18 * sweep_risk_up(snapshot)
        + 0.16 * pressure
        + 0.10 * zone_score(snapshot, LiquiditySide.BUY_SIDE)
        + bias_bonus
        + signal_bias_bonus
    )


def downside_bias_edge(snapshot: LiquidityMapSnapshot) -> float:
    pressure = max(-signed_score(getattr(snapshot, "liquidity_pressure_score", 0.0)), 0.0)

    bias_bonus = 0.10 if getattr(snapshot, "bias", None) == LiquidityBias.DOWN else 0.0
    signal_bias_bonus = 0.05 if snapshot_signal_bias(snapshot) == LiquidityBias.DOWN else 0.0

    return unit_score(
        0.24 * unit_score(getattr(snapshot, "below_liquidity_score", 0.0))
        + 0.22 * magnet_score_down(snapshot)
        + 0.18 * sweep_risk_down(snapshot)
        + 0.16 * pressure
        + 0.10 * zone_score(snapshot, LiquiditySide.SELL_SIDE)
        + bias_bonus
        + signal_bias_bonus
    )


def sweep_edge_up(snapshot: LiquidityMapSnapshot) -> float:
    pressure = max(signed_score(getattr(snapshot, "liquidity_pressure_score", 0.0)), 0.0)

    return unit_score(
        0.34 * magnet_score_up(snapshot)
        + 0.28 * sweep_risk_up(snapshot)
        + 0.22 * unit_score(getattr(snapshot, "above_liquidity_score", 0.0))
        + 0.16 * pressure
    )


def sweep_edge_down(snapshot: LiquidityMapSnapshot) -> float:
    pressure = max(-signed_score(getattr(snapshot, "liquidity_pressure_score", 0.0)), 0.0)

    return unit_score(
        0.34 * magnet_score_down(snapshot)
        + 0.28 * sweep_risk_down(snapshot)
        + 0.22 * unit_score(getattr(snapshot, "below_liquidity_score", 0.0))
        + 0.16 * pressure
    )


# =============================================================================
# Equal levels helpers
# =============================================================================


def is_equal_level(item: Any) -> bool:
    level_type = getattr(item, "level_type", None)

    return level_type in {
        LiquidityLevelType.EQUAL_HIGHS,
        LiquidityLevelType.EQUAL_LOWS,
    }


def expected_equal_level_side(item: Any) -> LiquiditySide | None:
    level_type = getattr(item, "level_type", None)

    if level_type == LiquidityLevelType.EQUAL_HIGHS:
        return LiquiditySide.BUY_SIDE

    if level_type == LiquidityLevelType.EQUAL_LOWS:
        return LiquiditySide.SELL_SIDE

    return None


def is_valid_equal_reaction_level(
    item: Any,
    *,
    allow_swept: bool = False,
) -> bool:
    if not is_equal_level(item):
        return False

    if reference_price(item) <= 0:
        return False

    if is_invalidated_item(item) or is_expired_item(item):
        return False

    if not allow_swept and (
        is_swept_item(item) or is_partially_swept_item(item)
    ):
        return False

    expected_side = expected_equal_level_side(item)
    if expected_side is not None and getattr(item, "side", None) != expected_side:
        return False

    return True


def compactness_width_pct(item: Any) -> float:
    for attr in ("width_pct", "compactness_width_pct", "range_pct"):
        value = to_float(getattr(item, attr, None))
        if value is not None and value >= 0:
            return value

    low = to_float(getattr(item, "low", None))
    high = to_float(getattr(item, "high", None))
    price = reference_price(item)

    if low is not None and high is not None and price > 0 and high >= low:
        return abs(high - low) / price

    return 0.0


def compactness_score(item: Any) -> float:
    width = compactness_width_pct(item)
    if width <= 0:
        return 0.50

    # <= 0.1% = excellent, >= 1% = weak.
    return unit_score(1.0 - min(width / 0.01, 1.0))


def level_quality(item: Any) -> tuple[float, int, int, float]:
    confidence = unit_score(getattr(item, "confidence", 0.0))
    touches = max(to_int(getattr(item, "touches_count", 0), 0) or 0, 0)
    reactions = max(to_int(getattr(item, "reaction_count", 0), 0) or 0, 0)

    return (
        confidence,
        touches,
        reactions,
        -compactness_width_pct(item),
    )


def level_distance_ok(
    item: Any,
    current_price: float,
    *,
    max_distance_pct: float,
) -> bool:
    if current_price <= 0 or reference_price(item) <= 0:
        return False

    return distance_pct(reference_price(item), current_price) <= max_distance_pct


# =============================================================================
# Swept / stop-hunt helpers
# =============================================================================


def is_valid_swept_level(item: Any) -> bool:
    if reference_price(item) <= 0:
        return False

    if is_invalidated_item(item) or is_expired_item(item):
        return False

    return is_swept_item(item) or is_partially_swept_item(item)


def cluster_is_swept(cluster: Any) -> bool:
    method = getattr(cluster, "is_swept", None)
    if callable(method):
        try:
            return bool(method())
        except Exception:
            pass

    status = normalize_label(getattr(cluster, "sweep_status", None))
    if status in {"swept", "partially_swept"}:
        return True

    return to_bool(getattr(cluster, "swept", None), default=False)


def swept_levels(
    snapshot: LiquidityMapSnapshot,
    side: LiquiditySide,
) -> list[LiquidityLevel]:
    candidates: list[LiquidityLevel] = []

    for level in [*equal_levels(snapshot), *active_levels(snapshot)]:
        if getattr(level, "side", None) != side:
            continue

        if not is_valid_swept_level(level):
            continue

        candidates.append(level)

    result: dict[str, LiquidityLevel] = {}

    for level in candidates:
        existing = result.get(item_key(level))
        if existing is None:
            result[item_key(level)] = level
            continue

        if swept_evidence_rank(level) > swept_evidence_rank(existing):
            result[item_key(level)] = level

    return list(result.values())


def swept_clusters(
    snapshot: LiquidityMapSnapshot,
    side: LiquiditySide,
) -> list[StopCluster]:
    getter = getattr(snapshot, "get_swept_clusters", None)

    if callable(getter):
        try:
            raw_clusters = list(getter())
        except Exception:
            raw_clusters = []
    else:
        raw_clusters = [
            cluster
            for cluster in stop_clusters(snapshot)
            if cluster_is_swept(cluster)
        ]

    return [
        cluster
        for cluster in raw_clusters
        if getattr(cluster, "side", None) == side and cluster_is_swept(cluster)
    ]


def swept_evidence_rank(item: Any) -> tuple[int, float, float, float]:
    if is_swept_item(item):
        sweep_rank = 3
    elif is_partially_swept_item(item):
        sweep_rank = 2
    elif is_active_item(item):
        sweep_rank = 1
    else:
        sweep_rank = 0

    return (
        sweep_rank,
        unit_score(getattr(item, "confidence", 0.0)),
        item_strength(item),
        reference_price(item),
    )


def evidence_type(item: Any) -> str:
    if isinstance(item, StopCluster):
        return "stop_cluster"

    if isinstance(item, LiquidityLevel):
        if is_equal_level(item):
            return "equal_level"
        return "liquidity_level"

    return item.__class__.__name__


def reclaim_score_from_reference(
    *,
    current_price: float,
    reference_price_value: float,
    side: SignalSide,
) -> float:
    if current_price <= 0 or reference_price_value <= 0:
        return 0.0

    if side is SignalSide.LONG:
        if current_price <= reference_price_value:
            return 0.0
        return unit_score((current_price - reference_price_value) / reference_price_value / 0.01)

    if side is SignalSide.SHORT:
        if current_price >= reference_price_value:
            return 0.0
        return unit_score((reference_price_value - current_price) / reference_price_value / 0.01)

    return 0.0


# =============================================================================
# Score DTO / scoring helpers
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


__all__ = [
    "DECIMAL_ZERO",
    "FUTURES_MARKET_TYPES",
    "LIQUIDITY_DOMAIN_ALIASES",
    "SNAPSHOT_FEATURE_KEYS",
    "ScoreBreakdown",
    "abs_score",
    "active_levels",
    "all_levels",
    "analytics_signal_confidence",
    "as_dict",
    "as_mapping",
    "average_score",
    "best_zone_for_side",
    "cluster_is_swept",
    "collect_targets_above",
    "collect_targets_below",
    "compactness_score",
    "compactness_width_pct",
    "confidence_from_components",
    "dedupe_liquidity_items",
    "directional_clusters",
    "directional_levels",
    "directional_zones",
    "distance_pct",
    "distance_score",
    "enum_value",
    "ensure_utc",
    "equal_levels",
    "evidence_type",
    "expected_equal_level_side",
    "first_present",
    "freshness_score",
    "get_attr_or_key",
    "get_path",
    "is_above_price",
    "is_active_item",
    "is_below_price",
    "is_directional_side",
    "is_equal_level",
    "is_expired_item",
    "is_futures_market_type",
    "is_invalidated_item",
    "is_partially_swept_item",
    "is_stale",
    "is_swept_item",
    "is_terminal_item",
    "is_valid_equal_reaction_level",
    "is_valid_price_item",
    "is_valid_swept_level",
    "item_key",
    "item_strength",
    "level_distance_ok",
    "level_quality",
    "liquidity_abs_score",
    "liquidity_bool",
    "liquidity_datetime",
    "liquidity_domain",
    "liquidity_float",
    "liquidity_item",
    "liquidity_path",
    "liquidity_signed_score",
    "liquidity_snapshot_from_context",
    "liquidity_str",
    "liquidity_unit_score",
    "magnet_score_down",
    "magnet_score_up",
    "normalize_label",
    "opposite_liquidity_side",
    "opposite_signal_side",
    "parse_datetime",
    "reclaim_score_from_reference",
    "reference_price",
    "safe_decimal",
    "serialize_for_metadata",
    "side_to_liquidity_side",
    "signed_score",
    "snapshot_liquidity_strength",
    "snapshot_signal_bias",
    "stop_clusters",
    "sweep_edge_down",
    "sweep_edge_up",
    "sweep_risk_down",
    "sweep_risk_up",
    "sweep_status_label",
    "swept_clusters",
    "swept_evidence_rank",
    "swept_levels",
    "timeframe_value",
    "to_bool",
    "to_float",
    "to_int",
    "to_str",
    "unit_score",
    "unwrap_analytics_payload",
    "unwrap_snapshot_candidate",
    "upside_bias_edge",
    "utc_now",
    "value_of",
    "weighted_score",
    "zone_score",
    "zones",
]