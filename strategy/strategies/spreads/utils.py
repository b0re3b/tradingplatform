# trading_system/strategy/strategies/spreads/utils.py

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from analytics.spreads.enums import (
    InstrumentType,
    OpportunityStatus,
    QuoteValidity,
    SpreadDirection,
    SpreadRegime,
    SpreadSignalType,
    SpreadType,
    parse_instrument_type as analytics_parse_instrument_type,
    parse_spread_type as analytics_parse_spread_type,
)
from analytics.spreads.models import ArbitrageOpportunity, SpreadSignal, SpreadSnapshot

from ...enums import FeatureSource, SignalSide
from ...models import StrategyContext, clamp, ensure_aware_utc, utcnow


DECIMAL_ZERO = Decimal("0")
DECIMAL_ONE = Decimal("1")
DECIMAL_100 = Decimal("100")
DECIMAL_10K = Decimal("10000")


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

    Final RiskReadySignalPayload conversion belongs to SignalProcessor /
    SignalBuilder, not to concrete spreads strategies.
    """
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
    """
    Read dotted path from dict-like or object-like nested data.

    Examples:
        snapshot.zscore
        signal.signal_type
        opportunity.net_edge_bps
        metadata.leg_a.exchange
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


def unwrap_analytics_payload(payload: Any) -> dict[str, Any]:
    """
    Backward-compatible unwrap for analytics.spreads.* envelopes.

    Supports:
    - direct SpreadSnapshot / SpreadSignal / ArbitrageOpportunity models;
    - direct dict payload;
    - {"payload": {...}};
    - {"payload": {"snapshot": {...}}};
    - {"payload": {"signal": {...}}};
    - {"payload": {"opportunity": {...}}}.
    """
    mapping = as_mapping(payload)
    if mapping is None:
        return {}

    raw = dict(mapping)
    inner = raw.get("payload")

    if isinstance(inner, Mapping):
        inner_dict = dict(inner)

        for key in (
            "snapshot",
            "spread_snapshot",
            "signal",
            "spread_signal",
            "opportunity",
            "arbitrage_opportunity",
            "result",
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


# =============================================================================
# Primitive conversion helpers
# =============================================================================


def to_decimal(value: Any, default: Decimal | None = None) -> Decimal | None:
    if value is None:
        return default

    if isinstance(value, Decimal):
        return value

    if isinstance(value, bool):
        return Decimal(int(value))

    if isinstance(value, Enum):
        return to_decimal(value.value, default)

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


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
            "open",
            "confirmed",
            "has_edge",
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
            "expired",
            "closed",
            "failed",
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


def unit_score(value: Any, default: float = 0.0) -> float:
    parsed = to_float(value, default)
    return clamp(float(parsed if parsed is not None else default), 0.0, 1.0)


def signed_score(value: Any, default: float = 0.0) -> float:
    parsed = to_float(value, default)
    return clamp(float(parsed if parsed is not None else default), -1.0, 1.0)


def abs_score(value: Any, default: float = 0.0) -> float:
    return abs(signed_score(value, default))


def decimal_abs(value: Decimal | None, default: Decimal = DECIMAL_ZERO) -> Decimal:
    return abs(value) if value is not None else default


def decimal_to_float(value: Decimal | None, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def decimal_unit_score(
    value: Decimal | None,
    *,
    scale: Decimal,
    default: float = 0.0,
) -> float:
    if value is None:
        return default

    if scale <= DECIMAL_ZERO:
        return unit_score(value)

    return unit_score(abs(value) / scale)


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


def parse_spread_type(
    value: Any,
    default: SpreadType | None = None,
) -> SpreadType | None:
    if isinstance(value, SpreadType):
        return value

    if value is None:
        return default

    try:
        parsed = analytics_parse_spread_type(value)
        if isinstance(parsed, SpreadType):
            return parsed
    except Exception:
        pass

    return parse_enum(value, SpreadType, default)  # type: ignore[return-value]


def parse_instrument_type(
    value: Any,
    default: InstrumentType | None = None,
) -> InstrumentType | None:
    if isinstance(value, InstrumentType):
        return value

    if value is None:
        return default

    try:
        parsed = analytics_parse_instrument_type(value)
        if isinstance(parsed, InstrumentType):
            return parsed
    except Exception:
        pass

    return parse_enum(value, InstrumentType, default)  # type: ignore[return-value]


def parse_spread_direction(
    value: Any,
    default: SpreadDirection | None = None,
) -> SpreadDirection | None:
    return parse_enum(value, SpreadDirection, default)  # type: ignore[return-value]


def parse_spread_regime(
    value: Any,
    default: SpreadRegime | None = None,
) -> SpreadRegime | None:
    return parse_enum(value, SpreadRegime, default)  # type: ignore[return-value]


def parse_spread_signal_type(
    value: Any,
    default: SpreadSignalType | None = None,
) -> SpreadSignalType | None:
    return parse_enum(value, SpreadSignalType, default)  # type: ignore[return-value]


def parse_quote_validity(
    value: Any,
    default: QuoteValidity | None = None,
) -> QuoteValidity | None:
    return parse_enum(value, QuoteValidity, default)  # type: ignore[return-value]


def parse_opportunity_status(
    value: Any,
    default: OpportunityStatus | None = None,
) -> OpportunityStatus | None:
    return parse_enum(value, OpportunityStatus, default)  # type: ignore[return-value]


# =============================================================================
# StrategyContext spreads helpers
# =============================================================================


SPREADS_DOMAIN_ALIASES: dict[str, tuple[str, ...]] = {
    "snapshot": (
        "snapshot",
        "spread_snapshot",
        "spot_futures",
        "cross_exchange",
        "result",
    ),
    "signal": (
        "signal",
        "spread_signal",
        "analytics_signal",
        "event",
    ),
    "opportunity": (
        "opportunity",
        "arbitrage_opportunity",
        "arb_opportunity",
        "arbitrage",
    ),
    "metadata": (
        "metadata",
        "analytics_metadata",
        "snapshot.metadata",
        "signal.metadata",
        "opportunity.metadata",
    ),
}


def spreads_domain(context: StrategyContext) -> dict[str, Any]:
    return dict(context.domain_dict(FeatureSource.SPREADS))


def spreads_item(
    context: StrategyContext,
    key: str,
    default: Any = None,
) -> Any:
    domain = spreads_domain(context)

    if key in domain:
        return domain[key]

    for alias in SPREADS_DOMAIN_ALIASES.get(key, ()):
        value = get_path(domain, alias, default=None)
        if value is not None:
            return value

    return default


def spreads_path(
    context: StrategyContext,
    path: str,
    default: Any = None,
) -> Any:
    """
    Read spreads value from StrategyContext.

    Priority:
    1. exact feature name;
    2. spreads-prefixed feature name;
    3. analytics.spreads-prefixed feature name;
    4. FeatureSource.SPREADS domain dotted path.
    """
    if not isinstance(path, str) or not path.strip():
        return default

    normalized = path.strip()
    spreads_feature_name = (
        normalized
        if normalized.startswith("spreads.")
        else f"spreads.{normalized}"
    )
    analytics_feature_name = (
        normalized
        if normalized.startswith("analytics.spreads.")
        else f"analytics.spreads.{normalized}"
    )

    if context.has_feature(normalized):
        return context.get_feature(normalized)

    if context.has_feature(spreads_feature_name):
        return context.get_feature(spreads_feature_name)

    if context.has_feature(analytics_feature_name):
        return context.get_feature(analytics_feature_name)

    domain = spreads_domain(context)

    if normalized.startswith("spreads."):
        normalized = normalized.removeprefix("spreads.")
    elif normalized.startswith("analytics.spreads."):
        normalized = normalized.removeprefix("analytics.spreads.")

    return get_path(domain, normalized, default)


def spreads_decimal(
    context: StrategyContext,
    path: str,
    *,
    default: Decimal | None = None,
) -> Decimal | None:
    return to_decimal(spreads_path(context, path, default), default)


def spreads_float(
    context: StrategyContext,
    path: str,
    *,
    default: float | None = None,
) -> float | None:
    return to_float(spreads_path(context, path, default), default)


def spreads_int(
    context: StrategyContext,
    path: str,
    *,
    default: int | None = None,
) -> int | None:
    return to_int(spreads_path(context, path, default), default)


def spreads_unit_score(
    context: StrategyContext,
    path: str,
    *,
    default: float = 0.0,
) -> float:
    return unit_score(spreads_path(context, path, default), default)


def spreads_bool(
    context: StrategyContext,
    path: str,
    *,
    default: bool = False,
) -> bool:
    return to_bool(spreads_path(context, path, default), default)


def spreads_str(
    context: StrategyContext,
    path: str,
    *,
    default: str | None = None,
) -> str | None:
    return to_str(spreads_path(context, path, default), default)


def spreads_datetime(
    context: StrategyContext,
    path: str,
    *,
    default: datetime | None = None,
) -> datetime | None:
    return parse_datetime(spreads_path(context, path, default))


# =============================================================================
# Analytics payload extraction
# =============================================================================


def normalize_snapshot_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, SpreadSnapshot):
        return as_dict(value)

    payload = unwrap_analytics_payload(value)

    nested = (
        payload.get("snapshot")
        or payload.get("spread_snapshot")
        or payload.get("spot_futures")
        or payload.get("cross_exchange")
    )
    if isinstance(nested, Mapping):
        result = dict(nested)
        result.setdefault("_container", payload)
        return result

    return payload


def normalize_signal_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, SpreadSignal):
        return as_dict(value)

    payload = unwrap_analytics_payload(value)

    nested = payload.get("signal") or payload.get("spread_signal")
    if isinstance(nested, Mapping):
        result = dict(nested)
        result.setdefault("_container", payload)
        return result

    return payload


def normalize_opportunity_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, ArbitrageOpportunity):
        return as_dict(value)

    payload = unwrap_analytics_payload(value)

    nested = (
        payload.get("opportunity")
        or payload.get("arbitrage_opportunity")
        or payload.get("arb_opportunity")
    )
    if isinstance(nested, Mapping):
        result = dict(nested)
        result.setdefault("_container", payload)
        return result

    return payload


def extract_spread_snapshot_payload(value: Any) -> dict[str, Any]:
    return normalize_snapshot_payload(value)


def extract_spread_signal_payload(value: Any) -> dict[str, Any]:
    return normalize_signal_payload(value)


def extract_arbitrage_opportunity_payload(value: Any) -> dict[str, Any]:
    return normalize_opportunity_payload(value)


def extract_metadata(value: Any) -> dict[str, Any]:
    payload = unwrap_analytics_payload(value)
    metadata = (
        payload.get("metadata")
        or payload.get("analytics_metadata")
        or get_path(payload, "snapshot.metadata")
        or get_path(payload, "signal.metadata")
        or get_path(payload, "opportunity.metadata")
    )
    return as_dict(metadata)


# =============================================================================
# Common spread field extractors
# =============================================================================


def extract_spread_type(value: Any) -> SpreadType | None:
    payload = unwrap_analytics_payload(value)
    return parse_spread_type(
        first_non_empty(
            payload.get("spread_type"),
            payload.get("type"),
            get_path(payload, "metadata.spread_type"),
            get_path(payload, "scope.spread_type"),
        )
    )


def extract_symbol(value: Any) -> str:
    payload = unwrap_analytics_payload(value)
    return normalize_symbol(
        first_non_empty(
            payload.get("symbol"),
            payload.get("base_symbol"),
            get_path(payload, "scope.symbol"),
            get_path(payload, "metadata.symbol"),
        )
    )


def extract_exchange_a(value: Any) -> str:
    payload = unwrap_analytics_payload(value)
    return normalize_exchange(
        first_non_empty(
            payload.get("exchange_a"),
            payload.get("leg_a_exchange"),
            payload.get("spot_exchange"),
            payload.get("buy_exchange"),
            get_path(payload, "leg_a.exchange"),
            get_path(payload, "buy_leg.exchange"),
            get_path(payload, "scope.exchange_a"),
        )
    )


def extract_exchange_b(value: Any) -> str:
    payload = unwrap_analytics_payload(value)
    return normalize_exchange(
        first_non_empty(
            payload.get("exchange_b"),
            payload.get("leg_b_exchange"),
            payload.get("futures_exchange"),
            payload.get("sell_exchange"),
            get_path(payload, "leg_b.exchange"),
            get_path(payload, "sell_leg.exchange"),
            get_path(payload, "scope.exchange_b"),
        )
    )


def extract_market_type_a(value: Any) -> str | None:
    payload = unwrap_analytics_payload(value)
    return to_str(
        first_non_empty(
            payload.get("market_type_a"),
            payload.get("leg_a_market_type"),
            payload.get("spot_market_type"),
            payload.get("buy_market_type"),
            get_path(payload, "leg_a.market_type"),
            get_path(payload, "buy_leg.market_type"),
            get_path(payload, "scope.market_type_a"),
        )
    )


def extract_market_type_b(value: Any) -> str | None:
    payload = unwrap_analytics_payload(value)
    return to_str(
        first_non_empty(
            payload.get("market_type_b"),
            payload.get("leg_b_market_type"),
            payload.get("futures_market_type"),
            payload.get("sell_market_type"),
            get_path(payload, "leg_b.market_type"),
            get_path(payload, "sell_leg.market_type"),
            get_path(payload, "scope.market_type_b"),
        )
    )


def extract_exchange_symbol_a(value: Any) -> str | None:
    payload = unwrap_analytics_payload(value)
    return to_str(
        first_non_empty(
            payload.get("exchange_symbol_a"),
            payload.get("leg_a_symbol"),
            payload.get("spot_exchange_symbol"),
            payload.get("buy_exchange_symbol"),
            get_path(payload, "leg_a.exchange_symbol"),
            get_path(payload, "buy_leg.exchange_symbol"),
            get_path(payload, "scope.exchange_symbol_a"),
        )
    )


def extract_exchange_symbol_b(value: Any) -> str | None:
    payload = unwrap_analytics_payload(value)
    return to_str(
        first_non_empty(
            payload.get("exchange_symbol_b"),
            payload.get("leg_b_symbol"),
            payload.get("futures_exchange_symbol"),
            payload.get("sell_exchange_symbol"),
            get_path(payload, "leg_b.exchange_symbol"),
            get_path(payload, "sell_leg.exchange_symbol"),
            get_path(payload, "scope.exchange_symbol_b"),
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


def extract_timestamp(value: Any) -> datetime | None:
    payload = unwrap_analytics_payload(value)
    return parse_datetime(
        first_non_empty(
            payload.get("timestamp"),
            payload.get("event_time"),
            payload.get("created_at"),
            payload.get("updated_at"),
            payload.get("time"),
            get_path(payload, "metadata.timestamp"),
            get_path(payload, "metadata.event_time"),
        )
    )


def extract_spread_bps(value: Any) -> Decimal | None:
    payload = unwrap_analytics_payload(value)
    return to_decimal(
        first_non_empty(
            payload.get("spread_bps"),
            payload.get("spread"),
            payload.get("basis_bps"),
            get_path(payload, "snapshot.spread_bps"),
            get_path(payload, "metadata.spread_bps"),
        )
    )


def extract_basis(value: Any) -> Decimal | None:
    payload = unwrap_analytics_payload(value)
    return to_decimal(
        first_non_empty(
            payload.get("basis"),
            payload.get("basis_bps"),
            payload.get("raw_basis"),
            get_path(payload, "snapshot.basis"),
            get_path(payload, "metadata.basis"),
        )
    )


def extract_funding_adjusted_spread(value: Any) -> Decimal | None:
    payload = unwrap_analytics_payload(value)
    return to_decimal(
        first_non_empty(
            payload.get("funding_adjusted_spread"),
            payload.get("funding_adjusted_basis"),
            payload.get("funding_adjusted_edge"),
            payload.get("net_funding_adjusted_spread"),
            get_path(payload, "snapshot.funding_adjusted_spread"),
            get_path(payload, "metadata.funding_adjusted_spread"),
        )
    )


def extract_zscore(value: Any) -> Decimal | None:
    payload = unwrap_analytics_payload(value)
    return to_decimal(
        first_non_empty(
            payload.get("zscore"),
            payload.get("z_score"),
            payload.get("spread_zscore"),
            get_path(payload, "rolling_stats.zscore"),
            get_path(payload, "stats.zscore"),
            get_path(payload, "snapshot.zscore"),
            get_path(payload, "metadata.zscore"),
        )
    )


def extract_net_edge(value: Any) -> Decimal | None:
    payload = unwrap_analytics_payload(value)
    return to_decimal(
        first_non_empty(
            payload.get("net_edge"),
            payload.get("edge"),
            payload.get("net_profit"),
            payload.get("expected_net_edge"),
            get_path(payload, "opportunity.net_edge"),
            get_path(payload, "metadata.net_edge"),
        )
    )


def extract_net_edge_bps(value: Any) -> Decimal | None:
    payload = unwrap_analytics_payload(value)
    explicit = to_decimal(
        first_non_empty(
            payload.get("net_edge_bps"),
            payload.get("edge_bps"),
            payload.get("net_profit_bps"),
            payload.get("expected_net_edge_bps"),
            get_path(payload, "opportunity.net_edge_bps"),
            get_path(payload, "metadata.net_edge_bps"),
        )
    )
    if explicit is not None:
        return explicit

    return extract_spread_bps(payload)


def extract_regime(value: Any) -> SpreadRegime | None:
    payload = unwrap_analytics_payload(value)
    return parse_spread_regime(
        first_non_empty(
            payload.get("regime"),
            payload.get("spread_regime"),
            get_path(payload, "snapshot.regime"),
            get_path(payload, "metadata.regime"),
        )
    )


def extract_direction(value: Any) -> SpreadDirection | None:
    payload = unwrap_analytics_payload(value)
    return parse_spread_direction(
        first_non_empty(
            payload.get("direction"),
            payload.get("spread_direction"),
            get_path(payload, "snapshot.direction"),
            get_path(payload, "metadata.direction"),
        )
    )


def extract_signal_type(value: Any) -> SpreadSignalType | None:
    payload = unwrap_analytics_payload(value)
    return parse_spread_signal_type(
        first_non_empty(
            payload.get("signal_type"),
            payload.get("type"),
            get_path(payload, "signal.signal_type"),
            get_path(payload, "metadata.signal_type"),
        )
    )


def extract_quote_validity(value: Any) -> QuoteValidity | None:
    payload = unwrap_analytics_payload(value)
    return parse_quote_validity(
        first_non_empty(
            payload.get("quote_validity"),
            payload.get("validity"),
            payload.get("quote_status"),
            get_path(payload, "snapshot.quote_validity"),
            get_path(payload, "metadata.quote_validity"),
        )
    )


def extract_has_edge(value: Any) -> bool | None:
    payload = unwrap_analytics_payload(value)

    raw = first_non_empty(
        payload.get("has_edge"),
        payload.get("edge_passed"),
        payload.get("is_tradeable"),
        payload.get("tradeable"),
        get_path(payload, "snapshot.has_edge"),
        get_path(payload, "metadata.has_edge"),
    )
    if raw is None:
        return None

    return to_bool(raw)


def extract_confidence(value: Any, default: float = 0.0) -> float:
    payload = unwrap_analytics_payload(value)
    return unit_score(
        first_non_empty(
            payload.get("confidence"),
            payload.get("score"),
            payload.get("signal_confidence"),
            get_path(payload, "snapshot.confidence"),
            get_path(payload, "signal.confidence"),
            get_path(payload, "opportunity.confidence"),
            get_path(payload, "metadata.confidence"),
        ),
        default,
    )


def extract_status(value: Any) -> OpportunityStatus | None:
    payload = unwrap_analytics_payload(value)
    return parse_opportunity_status(
        first_non_empty(
            payload.get("status"),
            payload.get("opportunity_status"),
            get_path(payload, "opportunity.status"),
            get_path(payload, "metadata.status"),
        )
    )


def extract_instrument_type(value: Any) -> InstrumentType | None:
    payload = unwrap_analytics_payload(value)
    return parse_instrument_type(
        first_non_empty(
            payload.get("instrument_type"),
            payload.get("type"),
            get_path(payload, "instrument.type"),
            get_path(payload, "metadata.instrument_type"),
            get_path(payload, "scope.instrument_type"),
        )
    )


def extract_leg_a_instrument_type(value: Any) -> InstrumentType | None:
    payload = unwrap_analytics_payload(value)
    return parse_instrument_type(
        first_non_empty(
            payload.get("leg_a_instrument_type"),
            payload.get("spot_instrument_type"),
            get_path(payload, "leg_a.instrument_type"),
            get_path(payload, "metadata.leg_a_instrument_type"),
        )
    )


def extract_leg_b_instrument_type(value: Any) -> InstrumentType | None:
    payload = unwrap_analytics_payload(value)
    return parse_instrument_type(
        first_non_empty(
            payload.get("leg_b_instrument_type"),
            payload.get("futures_instrument_type"),
            get_path(payload, "leg_b.instrument_type"),
            get_path(payload, "metadata.leg_b_instrument_type"),
        )
    )


def extract_persistence_ms(value: Any) -> int:
    payload = unwrap_analytics_payload(value)
    return max(
        0,
        to_int(
            first_non_empty(
                payload.get("persistence_ms"),
                payload.get("age_ms"),
                payload.get("duration_ms"),
                get_path(payload, "metadata.persistence_ms"),
            ),
            0,
        )
        or 0,
    )


# =============================================================================
# Opportunity-specific helpers
# =============================================================================


def extract_buy_exchange(value: Any) -> str:
    payload = normalize_opportunity_payload(value)
    return normalize_exchange(
        first_non_empty(
            payload.get("buy_exchange"),
            payload.get("exchange_a"),
            get_path(payload, "buy_leg.exchange"),
        )
    )


def extract_sell_exchange(value: Any) -> str:
    payload = normalize_opportunity_payload(value)
    return normalize_exchange(
        first_non_empty(
            payload.get("sell_exchange"),
            payload.get("exchange_b"),
            get_path(payload, "sell_leg.exchange"),
        )
    )


def extract_buy_market_type(value: Any) -> str | None:
    payload = normalize_opportunity_payload(value)
    return to_str(
        first_non_empty(
            payload.get("buy_market_type"),
            payload.get("market_type_a"),
            get_path(payload, "buy_leg.market_type"),
        )
    )


def extract_sell_market_type(value: Any) -> str | None:
    payload = normalize_opportunity_payload(value)
    return to_str(
        first_non_empty(
            payload.get("sell_market_type"),
            payload.get("market_type_b"),
            get_path(payload, "sell_leg.market_type"),
        )
    )


def extract_opportunity_key(value: Any) -> str | None:
    payload = normalize_opportunity_payload(value)
    return to_str(
        first_non_empty(
            payload.get("opportunity_key"),
            payload.get("key"),
            payload.get("state_key"),
            get_path(payload, "metadata.opportunity_key"),
        )
    )


def opportunity_is_active(value: Any) -> bool:
    status = extract_status(value)
    if status is None:
        return to_bool(get_path(value, "active"), default=True)

    label = normalize_label(status)
    return label in {"active", "open", "detected", "tradeable"}


def opportunity_is_tradeable(value: Any) -> bool:
    payload = normalize_opportunity_payload(value)

    explicit = first_non_empty(
        payload.get("is_tradeable"),
        payload.get("tradeable"),
        payload.get("can_trade"),
        get_path(payload, "metadata.tradeable"),
    )
    if explicit is not None:
        return to_bool(explicit)

    return opportunity_is_active(payload)


# =============================================================================
# Direction / side mapping
# =============================================================================


def basis_to_bias(value: Any) -> str | None:
    """
    Spread-specific basis semantics.

    Positive edge/basis:
        SHORT_BASIS — short expensive futures/basis side.
    Negative edge/basis:
        LONG_BASIS — long discounted basis side.
    """
    edge = first_non_empty(
        extract_funding_adjusted_spread(value),
        extract_basis(value),
        extract_spread_bps(value),
    )
    edge_decimal = to_decimal(edge)

    if edge_decimal is None or edge_decimal == DECIMAL_ZERO:
        return None

    return "SHORT_BASIS" if edge_decimal > DECIMAL_ZERO else "LONG_BASIS"


def basis_to_signal_side(value: Any) -> SignalSide:
    bias = basis_to_bias(value)

    if bias == "SHORT_BASIS":
        return SignalSide.SHORT

    if bias == "LONG_BASIS":
        return SignalSide.LONG

    return SignalSide.UNKNOWN


def spread_direction_to_signal_side(value: Any) -> SignalSide:
    direction = parse_spread_direction(value)

    if direction is None:
        return SignalSide.UNKNOWN

    label = normalize_label(direction)

    if label in {
        "widening",
        "long_a_short_b",
        "buy_a_sell_b",
        "long_basis",
        "positive",
    }:
        return SignalSide.LONG

    if label in {
        "compressing",
        "short_a_long_b",
        "sell_a_buy_b",
        "short_basis",
        "negative",
    }:
        return SignalSide.SHORT

    return SignalSide.UNKNOWN


def cross_exchange_direction(value: Any) -> str:
    """
    Return spread-leg semantics for cross-exchange arbitrage.

    Canonical output:
    - LONG_A_SHORT_B
    - SHORT_A_LONG_B

    For ArbitrageOpportunity, buy_exchange is treated as long leg and
    sell_exchange as short leg.
    """
    payload = normalize_opportunity_payload(value)

    explicit = to_str(
        first_non_empty(
            payload.get("spread_direction"),
            payload.get("direction"),
            get_path(payload, "metadata.spread_direction"),
        )
    )
    if explicit:
        normalized = explicit.strip().upper()
        if normalized in {"LONG_A_SHORT_B", "SHORT_A_LONG_B"}:
            return normalized

    buy_exchange = extract_buy_exchange(payload)
    sell_exchange = extract_sell_exchange(payload)

    if buy_exchange and sell_exchange:
        return "LONG_A_SHORT_B"

    return "UNKNOWN"


def cross_exchange_to_signal_side(value: Any) -> SignalSide:
    direction = cross_exchange_direction(value)

    if direction == "LONG_A_SHORT_B":
        return SignalSide.LONG

    if direction == "SHORT_A_LONG_B":
        return SignalSide.SHORT

    return SignalSide.UNKNOWN


def is_directional_side(side: SignalSide) -> bool:
    return side in {SignalSide.LONG, SignalSide.SHORT}


def cross_exchange_leg_metadata(value: Any) -> dict[str, Any]:
    payload = normalize_opportunity_payload(value)

    direction = cross_exchange_direction(payload)
    symbol = extract_symbol(payload)
    instrument_type = extract_instrument_type(payload)

    buy_exchange = extract_buy_exchange(payload) or extract_exchange_a(payload)
    sell_exchange = extract_sell_exchange(payload) or extract_exchange_b(payload)

    buy_market_type = extract_buy_market_type(payload) or extract_market_type_a(payload)
    sell_market_type = extract_sell_market_type(payload) or extract_market_type_b(payload)

    if direction == "LONG_A_SHORT_B":
        return {
            "spread_direction": direction,
            "long_leg": {
                "exchange": buy_exchange,
                "market_type": buy_market_type,
                "symbol": symbol,
                "instrument_type": normalize_label(instrument_type),
            },
            "short_leg": {
                "exchange": sell_exchange,
                "market_type": sell_market_type,
                "symbol": symbol,
                "instrument_type": normalize_label(instrument_type),
            },
        }

    if direction == "SHORT_A_LONG_B":
        return {
            "spread_direction": direction,
            "long_leg": {
                "exchange": sell_exchange,
                "market_type": sell_market_type,
                "symbol": symbol,
                "instrument_type": normalize_label(instrument_type),
            },
            "short_leg": {
                "exchange": buy_exchange,
                "market_type": buy_market_type,
                "symbol": symbol,
                "instrument_type": normalize_label(instrument_type),
            },
        }

    return {
        "spread_direction": "UNKNOWN",
        "long_leg": None,
        "short_leg": None,
    }


# =============================================================================
# Spread signal / confirmation helpers
# =============================================================================


def is_mean_reversion_signal(value: Any) -> bool:
    signal_type = extract_signal_type(value)
    label = normalize_label(signal_type)
    return label in {
        "mean_reversion",
        "basis_mean_reversion",
        "spread_mean_reversion",
        "reversion",
    }


def is_regime_shift_signal(value: Any) -> bool:
    signal_type = extract_signal_type(value)
    label = normalize_label(signal_type)
    return label in {
        "regime_shift",
        "spread_regime_shift",
        "basis_regime_shift",
    }


def is_anomaly_signal(value: Any) -> bool:
    signal_type = extract_signal_type(value)
    label = normalize_label(signal_type)
    return label in {
        "anomaly",
        "spread_anomaly",
        "basis_anomaly",
        "dislocation",
    }


def is_widening_signal(value: Any) -> bool:
    signal_type = extract_signal_type(value)
    label = normalize_label(signal_type)

    if label in {"widening", "spread_widening", "basis_widening"}:
        return True

    direction = extract_direction(value)
    return normalize_label(direction) == "widening"


def is_data_quality_signal(value: Any) -> bool:
    signal_type = extract_signal_type(value)
    label = normalize_label(signal_type)
    return label in {
        "stale_data",
        "invalid_data",
        "data_quality",
        "quote_invalid",
    }


# =============================================================================
# Quality filters
# =============================================================================


def quote_is_valid(value: Any) -> bool:
    validity = extract_quote_validity(value)

    if validity is None:
        return True

    label = normalize_label(validity)
    return label in {"valid", "ok", "healthy"}


def has_tradeable_edge(value: Any) -> bool:
    explicit = extract_has_edge(value)
    if explicit is not None:
        return explicit

    for candidate in (
        extract_net_edge(value),
        extract_net_edge_bps(value),
        extract_funding_adjusted_spread(value),
        extract_basis(value),
        extract_spread_bps(value),
    ):
        if candidate is not None and candidate != DECIMAL_ZERO:
            return True

    return False


def spread_quality_filter_reason(
    value: Any,
    *,
    min_score: float = 0.0,
    min_confidence: float = 0.0,
    require_valid_quote: bool = False,
    require_edge: bool = False,
    allowed_regimes: set[str] | frozenset[str] | None = None,
    stale_after_seconds: float | None = None,
    now: datetime | None = None,
) -> str | None:
    if value is None:
        return "missing_spread_context"

    confidence = extract_confidence(value)
    if confidence < min_confidence:
        return "spread_confidence_below_threshold"

    score = max(confidence, unit_score(abs(to_float(extract_zscore(value), 0.0) or 0.0) / 5.0))
    if score < min_score:
        return "spread_score_below_threshold"

    if require_valid_quote and not quote_is_valid(value):
        return "spread_quote_invalid"

    if require_edge and not has_tradeable_edge(value):
        return "spread_edge_missing"

    if allowed_regimes:
        regime = extract_regime(value)
        if normalize_label(regime) not in {item.lower() for item in allowed_regimes}:
            return "spread_regime_not_allowed"

    event_time = extract_timestamp(value)
    if is_stale(
        event_time=event_time,
        now=now,
        stale_after_seconds=stale_after_seconds,
    ):
        return "spread_context_stale"

    return None


def spot_futures_contract_error(value: Any) -> str | None:
    spread_type = extract_spread_type(value)
    if spread_type is not None and spread_type is not SpreadType.SPOT_FUTURES:
        return "spread_type_not_spot_futures"

    leg_a_type = extract_leg_a_instrument_type(value)
    leg_b_type = extract_leg_b_instrument_type(value)

    if leg_a_type is not None and leg_a_type is not InstrumentType.SPOT:
        return "leg_a_not_spot"

    if leg_b_type is not None and leg_b_type is InstrumentType.SPOT:
        return "leg_b_not_futures"

    if not extract_symbol(value):
        return "symbol_missing"

    if not extract_exchange_a(value):
        return "spot_exchange_missing"

    if not extract_exchange_b(value):
        return "futures_exchange_missing"

    return None


def cross_exchange_contract_error(value: Any) -> str | None:
    spread_type = extract_spread_type(value)
    if spread_type is not None and spread_type is not SpreadType.CROSS_EXCHANGE:
        return "spread_type_not_cross_exchange"

    if not extract_symbol(value):
        return "symbol_missing"

    exchange_a = extract_exchange_a(value) or extract_buy_exchange(value)
    exchange_b = extract_exchange_b(value) or extract_sell_exchange(value)

    if not exchange_a:
        return "exchange_a_missing"

    if not exchange_b:
        return "exchange_b_missing"

    if exchange_a == exchange_b:
        return "same_exchange"

    return None


# =============================================================================
# Scoring helpers
# =============================================================================


@dataclass(slots=True)
class ScoreBreakdown:
    """
    Reusable score DTO for concrete spread strategies.
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


def zscore_component(
    value: Any,
    *,
    entry_zscore: Decimal,
    stop_zscore: Decimal | None = None,
) -> float:
    zscore = extract_zscore(value)
    if zscore is None:
        return 0.0

    abs_z = abs(zscore)
    if stop_zscore is not None and stop_zscore > entry_zscore:
        return unit_score((abs_z - entry_zscore) / (stop_zscore - entry_zscore))

    return unit_score(abs_z / max(entry_zscore, DECIMAL_ONE))


def edge_component(
    value: Any,
    *,
    min_edge: Decimal = DECIMAL_ZERO,
    scale: Decimal | None = None,
) -> float:
    edge = first_non_empty(
        extract_funding_adjusted_spread(value),
        extract_net_edge(value),
        extract_net_edge_bps(value),
        extract_basis(value),
        extract_spread_bps(value),
    )
    edge_decimal = to_decimal(edge)
    if edge_decimal is None:
        return 0.0

    denominator = scale or max(abs(min_edge), DECIMAL_ONE)
    return decimal_unit_score(abs(edge_decimal), scale=denominator)


def quote_component(value: Any) -> float:
    return 1.0 if quote_is_valid(value) else 0.0


def regime_component(value: Any) -> float:
    regime = extract_regime(value)
    label = normalize_label(regime)

    if label == "dislocated":
        return 1.0

    if label == "extreme":
        return 0.9

    if label == "elevated":
        return 0.75

    if label == "normal":
        return 0.35

    if label == "compressed":
        return 0.25

    return 0.0


def opportunity_status_component(value: Any) -> float:
    if opportunity_is_tradeable(value):
        return 1.0

    if opportunity_is_active(value):
        return 0.7

    return 0.0


# =============================================================================
# Source-feature helpers
# =============================================================================


def source_features_from_paths(*paths: str) -> list[str]:
    result: list[str] = []

    for path in paths:
        if not isinstance(path, str) or not path.strip():
            continue

        normalized = path.strip()

        if normalized.startswith("spreads."):
            result.append(normalized)
        elif normalized.startswith("analytics.spreads."):
            result.append(normalized.replace("analytics.", "", 1))
        else:
            result.append(f"spreads.{normalized}")

    return list(dict.fromkeys(result))


def base_spreads_source_features() -> list[str]:
    return source_features_from_paths(
        "snapshot",
        "signal",
        "opportunity",
        "type",
        "symbol",
        "exchange_a",
        "exchange_b",
        "market_type_a",
        "market_type_b",
        "spread_bps",
        "basis",
        "funding_adjusted_spread",
        "net_edge",
        "net_edge_bps",
        "zscore",
        "regime",
        "direction",
        "quote_validity",
        "has_edge",
        "confidence",
    )


def spot_futures_source_features() -> list[str]:
    return source_features_from_paths(
        "snapshot",
        "signal",
        "type",
        "symbol",
        "exchange_a",
        "exchange_b",
        "market_type_a",
        "market_type_b",
        "spread_bps",
        "basis",
        "funding_adjusted_spread",
        "zscore",
        "regime",
        "direction",
        "quote_validity",
        "has_edge",
        "confidence",
        "leg_a.instrument_type",
        "leg_b.instrument_type",
    )


def cross_exchange_source_features() -> list[str]:
    return source_features_from_paths(
        "snapshot",
        "signal",
        "opportunity",
        "type",
        "symbol",
        "exchange_a",
        "exchange_b",
        "market_type_a",
        "market_type_b",
        "spread_bps",
        "net_edge",
        "net_edge_bps",
        "direction",
        "quote_validity",
        "has_edge",
        "confidence",
        "instrument_type",
    )


def arbitrage_opportunity_source_features() -> list[str]:
    return source_features_from_paths(
        "opportunity",
        "opportunity_key",
        "symbol",
        "buy_exchange",
        "sell_exchange",
        "buy_market_type",
        "sell_market_type",
        "instrument_type",
        "net_edge",
        "net_edge_bps",
        "status",
        "confidence",
        "persistence_ms",
    )


def funding_adjusted_source_features() -> list[str]:
    return source_features_from_paths(
        "snapshot",
        "signal",
        "basis",
        "funding_adjusted_spread",
        "funding_adjusted_basis",
        "funding_rate",
        "zscore",
        "regime",
        "confidence",
    )


def spread_mean_reversion_source_features() -> list[str]:
    return source_features_from_paths(
        "snapshot",
        "signal",
        "spread_bps",
        "basis",
        "zscore",
        "regime",
        "direction",
        "quote_validity",
        "has_edge",
        "confidence",
    )


def spread_momentum_source_features() -> list[str]:
    return source_features_from_paths(
        "snapshot",
        "signal",
        "spread_bps",
        "direction",
        "zscore",
        "regime",
        "quote_validity",
        "has_edge",
        "confidence",
    )