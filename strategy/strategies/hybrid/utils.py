# trading_system/strategy/strategies/hybrid/utils.py

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


HYBRID_CONTEXT_VERSION = "2.0.0"

DEFAULT_HYBRID_STALE_AFTER_SECONDS = 120.0

UNKNOWN_SIDE_LABELS: frozenset[str] = frozenset(
    {
        "",
        "none",
        "null",
        "unknown",
        "neutral",
        "mixed",
        "flat",
        "sideways",
        "no_signal",
        "no_side",
    }
)

LONG_SIDE_LABELS: frozenset[str] = frozenset(
    {
        "long",
        "buy",
        "bid",
        "bull",
        "bullish",
        "up",
        "upside",
        "trend_up",
        "breakout_up",
        "reversal_up",
        "continuation_up",
        "support",
        "demand",
        "accumulation",
        "short_squeeze",
        "sell_liquidation_reversal",
        "sell_liquidations",
        "sell_exhaustion",
        "buy_absorption",
        "positive",
    }
)

SHORT_SIDE_LABELS: frozenset[str] = frozenset(
    {
        "short",
        "sell",
        "ask",
        "bear",
        "bearish",
        "down",
        "downside",
        "trend_down",
        "breakdown",
        "breakout_down",
        "reversal_down",
        "continuation_down",
        "resistance",
        "supply",
        "distribution",
        "long_squeeze",
        "buy_liquidation_reversal",
        "buy_liquidations",
        "buy_exhaustion",
        "sell_absorption",
        "negative",
    }
)


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

    if isinstance(value, (list, tuple, set, frozenset)):
        return [serialize_for_metadata(item) for item in value]

    return value


# =============================================================================
# Generic mapping / path helpers
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


def unwrap_payload(value: Any) -> dict[str, Any]:
    """
    Backward-compatible unwrap for domain payload envelopes.

    Supports:
    - direct dict/model payload;
    - {"payload": {...}};
    - {"payload": {"result": {...}}};
    - {"payload": {"signal": {...}}};
    - {"payload": {"snapshot": {...}}};
    - {"payload": {"event": {...}}}.
    """
    mapping = as_mapping(value)
    if mapping is None:
        return {}

    raw = dict(mapping)
    inner = raw.get("payload")

    if isinstance(inner, Mapping):
        inner_dict = dict(inner)

        for key in (
            "result",
            "signal",
            "snapshot",
            "event",
            "analysis",
            "context",
            "setup",
            "opportunity",
            "features",
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
            "enabled",
            "active",
            "confirmed",
            "valid",
            "passed",
        }:
            return True

        if normalized in {
            "0",
            "false",
            "no",
            "n",
            "off",
            "disabled",
            "inactive",
            "invalid",
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


# =============================================================================
# FeatureSource / domain helpers
# =============================================================================


DOMAIN_FEATURE_SOURCES: tuple[FeatureSource, ...] = (
    FeatureSource.ORDERFLOW,
    FeatureSource.LIQUIDITY,
    FeatureSource.LIQUIDATIONS,
    FeatureSource.WHALES,
    FeatureSource.OPEN_INTEREST,
    FeatureSource.FUNDING,
    FeatureSource.PRICE_ACTION,
    FeatureSource.SPOOFING,
    FeatureSource.SPREADS,
)


def feature_source_value(source: FeatureSource | str) -> str:
    if isinstance(source, FeatureSource):
        return source.value
    return str(source).strip().lower()


def feature_source_label(source: FeatureSource | str) -> str:
    return feature_source_value(source).replace(".", "_")


def domain_dict(
    context: StrategyContext,
    source: FeatureSource,
) -> dict[str, Any]:
    return dict(context.domain_dict(source))


def domain_path(
    context: StrategyContext,
    source: FeatureSource,
    path: str,
    default: Any = None,
) -> Any:
    if not isinstance(path, str) or not path.strip():
        return default

    source_label = feature_source_value(source)
    source_alias = feature_source_label(source)

    normalized = path.strip()

    candidates = [
        normalized,
        f"{source_label}.{normalized}",
        f"{source_alias}.{normalized}",
        f"analytics.{source_label}.{normalized}",
        f"analytics.{source_alias}.{normalized}",
    ]

    for feature_name in candidates:
        if context.has_feature(feature_name):
            return context.get_feature(feature_name)

    domain = domain_dict(context, source)

    if normalized.startswith(f"{source_label}."):
        normalized = normalized.removeprefix(f"{source_label}.")
    elif normalized.startswith(f"{source_alias}."):
        normalized = normalized.removeprefix(f"{source_alias}.")
    elif normalized.startswith(f"analytics.{source_label}."):
        normalized = normalized.removeprefix(f"analytics.{source_label}.")
    elif normalized.startswith(f"analytics.{source_alias}."):
        normalized = normalized.removeprefix(f"analytics.{source_alias}.")

    return get_path(domain, normalized, default)


def domain_available(
    context: StrategyContext,
    source: FeatureSource,
    *,
    required_paths: Sequence[str] = (),
) -> bool:
    domain = domain_dict(context, source)
    if domain:
        if not required_paths:
            return True

        return any(
            domain_path(context, source, path, None) is not None
            for path in required_paths
        )

    source_label = feature_source_value(source)
    source_alias = feature_source_label(source)

    if not required_paths:
        prefixes = (
            f"{source_label}.",
            f"{source_alias}.",
            f"analytics.{source_label}.",
            f"analytics.{source_alias}.",
        )
        features = getattr(context, "features", {})
        if isinstance(features, Mapping):
            return any(
                isinstance(name, str) and name.startswith(prefixes)
                for name in features.keys()
            )
        return False

    return any(
        domain_path(context, source, path, None) is not None
        for path in required_paths
    )


def required_domains_available(
    context: StrategyContext,
    sources: Sequence[FeatureSource],
    *,
    allow_missing: int = 0,
) -> bool:
    missing = 0

    for source in sources:
        if not domain_available(context, source):
            missing += 1

    return missing <= max(0, allow_missing)


def available_domain_sources(
    context: StrategyContext,
    sources: Sequence[FeatureSource] = DOMAIN_FEATURE_SOURCES,
) -> list[FeatureSource]:
    return [
        source
        for source in sources
        if domain_available(context, source)
    ]


def missing_domain_sources(
    context: StrategyContext,
    sources: Sequence[FeatureSource],
) -> list[FeatureSource]:
    return [
        source
        for source in sources
        if not domain_available(context, source)
    ]


# =============================================================================
# Common domain field extraction
# =============================================================================


SIDE_PATHS: tuple[str, ...] = (
    "side",
    "signal_side",
    "direction",
    "bias",
    "trend_direction",
    "dominant_side",
    "setup_side",
    "expected_side",
    "reversal_side",
    "breakout_side",
    "continuation_side",
    "whale_side",
    "liquidation_side",
    "cluster_side",
    "exhausted_side",
    "sweep_side",
    "absorption_side",
    "pressure_side",
    "metadata.side",
    "metadata.signal_side",
    "metadata.direction",
    "metadata.bias",
)

SCORE_PATHS: tuple[str, ...] = (
    "score",
    "signal_score",
    "setup_score",
    "strength",
    "context_strength",
    "quality_score",
    "confidence_score",
    "metadata.score",
    "metadata.signal_score",
    "metadata.setup_score",
    "metadata.strength",
)

CONFIDENCE_PATHS: tuple[str, ...] = (
    "confidence",
    "signal_confidence",
    "setup_confidence",
    "probability",
    "continuation_probability",
    "reversal_probability",
    "breakout_probability",
    "metadata.confidence",
    "metadata.signal_confidence",
    "metadata.probability",
)

TIMESTAMP_PATHS: tuple[str, ...] = (
    "timestamp",
    "timestamp_ms",
    "event_time",
    "detected_at",
    "created_at",
    "updated_at",
    "time",
    "metadata.timestamp",
    "metadata.event_time",
    "metadata.detected_at",
)


def side_label(value: Any) -> str:
    label = normalize_label(value)

    if label in UNKNOWN_SIDE_LABELS:
        return "unknown"

    if label in LONG_SIDE_LABELS:
        return "long"

    if label in SHORT_SIDE_LABELS:
        return "short"

    if "bull" in label or "buy" in label or "long" in label or "up" in label:
        return "long"

    if "bear" in label or "sell" in label or "short" in label or "down" in label:
        return "short"

    return label or "unknown"


def side_to_signal_side(value: Any) -> SignalSide:
    label = side_label(value)

    if label == "long":
        return SignalSide.LONG

    if label == "short":
        return SignalSide.SHORT

    return SignalSide.UNKNOWN


def signal_side_to_label(side: SignalSide | str) -> str:
    if isinstance(side, SignalSide):
        if side is SignalSide.LONG:
            return "long"
        if side is SignalSide.SHORT:
            return "short"
        return "unknown"

    return side_label(side)


def opposite_side(value: Any) -> str:
    label = side_label(value)

    if label == "long":
        return "short"

    if label == "short":
        return "long"

    return "unknown"


def opposite_signal_side(side: SignalSide) -> SignalSide:
    if side is SignalSide.LONG:
        return SignalSide.SHORT

    if side is SignalSide.SHORT:
        return SignalSide.LONG

    return SignalSide.UNKNOWN


def is_directional_side(side: SignalSide) -> bool:
    return side in {SignalSide.LONG, SignalSide.SHORT}


def sides_align(
    left: SignalSide | str,
    right: SignalSide | str,
    *,
    unknown_aligns: bool = False,
) -> bool:
    left_label = signal_side_to_label(left)
    right_label = signal_side_to_label(right)

    if "unknown" in {left_label, right_label}:
        return unknown_aligns

    return left_label == right_label


def sides_conflict(
    left: SignalSide | str,
    right: SignalSide | str,
) -> bool:
    left_label = signal_side_to_label(left)
    right_label = signal_side_to_label(right)

    if "unknown" in {left_label, right_label}:
        return False

    return left_label != right_label


def extract_domain_side(
    payload: Any,
    *,
    default: SignalSide = SignalSide.UNKNOWN,
) -> SignalSide:
    raw_payload = unwrap_payload(payload)

    for path in SIDE_PATHS:
        value = get_path(raw_payload, path, None)
        side = side_to_signal_side(value)
        if is_directional_side(side):
            return side

    return default


def extract_domain_score(
    payload: Any,
    *,
    default: float = 0.0,
) -> float:
    raw_payload = unwrap_payload(payload)

    for path in SCORE_PATHS:
        value = get_path(raw_payload, path, None)
        if value is not None:
            return unit_score(value, default)

    return unit_score(default)


def extract_domain_confidence(
    payload: Any,
    *,
    default: float = 0.0,
) -> float:
    raw_payload = unwrap_payload(payload)

    for path in CONFIDENCE_PATHS:
        value = get_path(raw_payload, path, None)
        if value is not None:
            return unit_score(value, default)

    return unit_score(default)


def extract_domain_timestamp(payload: Any) -> datetime | None:
    raw_payload = unwrap_payload(payload)

    for path in TIMESTAMP_PATHS:
        value = get_path(raw_payload, path, None)
        parsed = parse_datetime(value)
        if parsed is not None:
            return parsed

    return None


def extract_domain_symbol(payload: Any) -> str:
    raw_payload = unwrap_payload(payload)
    return normalize_symbol(
        first_non_empty(
            raw_payload.get("symbol"),
            raw_payload.get("exchange_symbol"),
            get_path(raw_payload, "scope.symbol"),
            get_path(raw_payload, "metadata.symbol"),
        )
    )


def extract_domain_exchange(payload: Any) -> str:
    raw_payload = unwrap_payload(payload)
    return normalize_exchange(
        first_non_empty(
            raw_payload.get("exchange"),
            raw_payload.get("venue"),
            get_path(raw_payload, "scope.exchange"),
            get_path(raw_payload, "metadata.exchange"),
        )
    )


def extract_domain_market_type(payload: Any) -> str:
    raw_payload = unwrap_payload(payload)
    return normalize_market_type(
        first_non_empty(
            raw_payload.get("market_type"),
            raw_payload.get("instrument_type"),
            raw_payload.get("contract_type"),
            get_path(raw_payload, "scope.market_type"),
            get_path(raw_payload, "metadata.market_type"),
        )
    )


def extract_domain_timeframe(payload: Any, default: str = "1m") -> str:
    raw_payload = unwrap_payload(payload)
    return (
        to_str(
            first_non_empty(
                raw_payload.get("timeframe"),
                get_path(raw_payload, "scope.timeframe"),
                get_path(raw_payload, "metadata.timeframe"),
            ),
            default,
        )
        or default
    )


# =============================================================================
# Domain snapshot extraction from StrategyContext
# =============================================================================


DOMAIN_PRIMARY_KEYS: dict[FeatureSource, tuple[str, ...]] = {
    FeatureSource.ORDERFLOW: (
        "orderflow",
        "signal",
        "snapshot",
        "cvd",
        "delta",
        "pressure",
    ),
    FeatureSource.LIQUIDITY: (
        "liquidity",
        "sweep",
        "stop_hunt",
        "equal_high_low",
        "signal",
        "snapshot",
    ),
    FeatureSource.LIQUIDATIONS: (
        "liquidations",
        "liquidation",
        "cascade",
        "squeeze",
        "signal",
        "snapshot",
    ),
    FeatureSource.WHALES: (
        "whales",
        "pressure",
        "activity",
        "large_trade",
        "cluster",
        "liquidation_context",
        "signal",
        "snapshot",
    ),
    FeatureSource.OPEN_INTEREST: (
        "open_interest",
        "oi",
        "analysis",
        "regime",
        "divergence",
        "anomaly",
        "signal",
        "snapshot",
    ),
    FeatureSource.FUNDING: (
        "funding",
        "extreme",
        "divergence",
        "pressure",
        "signal",
        "snapshot",
    ),
    FeatureSource.PRICE_ACTION: (
        "price_action",
        "market_structure",
        "fvg",
        "trend",
        "support_resistance",
        "signal",
        "snapshot",
    ),
    FeatureSource.SPOOFING: (
        "spoofing",
        "pattern",
        "trap",
        "layering",
        "signal",
        "snapshot",
    ),
    FeatureSource.SPREADS: (
        "spreads",
        "snapshot",
        "signal",
        "opportunity",
        "basis",
        "arbitrage",
    ),
}


def extract_domain_payload(
    context: StrategyContext,
    source: FeatureSource,
) -> dict[str, Any]:
    domain = domain_dict(context, source)
    if not domain:
        return {}

    for key in DOMAIN_PRIMARY_KEYS.get(source, ()):
        nested = get_path(domain, key, None)
        if isinstance(nested, Mapping):
            result = dict(nested)
            result.setdefault("_domain", domain)
            result.setdefault("_source", feature_source_value(source))
            return result

    domain.setdefault("_source", feature_source_value(source))
    return domain


def domain_freshness_score(
    payload: Any,
    *,
    now: datetime | None = None,
    stale_after_seconds: float | None = DEFAULT_HYBRID_STALE_AFTER_SECONDS,
) -> float:
    return freshness_score(
        event_time=extract_domain_timestamp(payload),
        now=now,
        stale_after_seconds=stale_after_seconds,
    )


def domain_is_stale(
    payload: Any,
    *,
    now: datetime | None = None,
    stale_after_seconds: float | None = DEFAULT_HYBRID_STALE_AFTER_SECONDS,
) -> bool:
    return is_stale(
        event_time=extract_domain_timestamp(payload),
        now=now,
        stale_after_seconds=stale_after_seconds,
    )


def extract_context_domain_payloads(
    context: StrategyContext,
    sources: Sequence[FeatureSource] = DOMAIN_FEATURE_SOURCES,
) -> dict[FeatureSource, dict[str, Any]]:
    return {
        source: extract_domain_payload(context, source)
        for source in sources
        if domain_available(context, source)
    }


# =============================================================================
# Direction votes
# =============================================================================


@dataclass(slots=True)
class DirectionVote:
    _logger = logging.getLogger(__name__ + ".DirectionVote")
    source: FeatureSource
    side: SignalSide = SignalSide.UNKNOWN
    confidence: float = 0.0
    score: float = 0.0
    weight: float = 1.0
    reason: str = ""
    timestamp: datetime | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering DirectionVote.__post_init__")
        self.confidence = unit_score(self.confidence)
        self.score = unit_score(self.score)
        self.weight = max(0.0, float(self.weight))
        self.timestamp = parse_datetime(self.timestamp)
        self.payload = as_dict(self.payload)
        self.metadata = as_dict(self.metadata)

    @property
    def directional(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering DirectionVote.directional")
        return is_directional_side(self.side)

    @property
    def weighted_strength(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering DirectionVote.weighted_strength")
        if not self.directional:
            return 0.0
        return unit_score(self.confidence * 0.5 + self.score * 0.5) * self.weight

    @property
    def side_label(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering DirectionVote.side_label")
        return signal_side_to_label(self.side)

    def to_dict(self) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering DirectionVote.to_dict")
        return {
            "source": self.source.value,
            "side": self.side.value,
            "side_label": self.side_label,
            "confidence": self.confidence,
            "score": self.score,
            "weight": self.weight,
            "weighted_strength": self.weighted_strength,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "timestamp_ms": timestamp_ms(self.timestamp),
            "metadata": serialize_for_metadata(self.metadata),
            "payload": serialize_for_metadata(self.payload),
        }


def build_direction_vote(
    *,
    source: FeatureSource,
    payload: Mapping[str, Any],
    weight: float = 1.0,
    reason: str | None = None,
) -> DirectionVote:
    side = extract_domain_side(payload)
    score = extract_domain_score(payload)
    confidence = extract_domain_confidence(payload)
    timestamp = extract_domain_timestamp(payload)

    return DirectionVote(
        source=source,
        side=side,
        confidence=confidence,
        score=score,
        weight=weight,
        reason=reason or f"{source.value}_direction_vote",
        timestamp=timestamp,
        payload=dict(payload),
        metadata={
            "source": source.value,
            "domain_score": score,
            "domain_confidence": confidence,
        },
    )


def build_direction_votes(
    context: StrategyContext,
    *,
    sources: Sequence[FeatureSource],
    weights: Mapping[FeatureSource, float] | None = None,
) -> list[DirectionVote]:
    result: list[DirectionVote] = []
    weights = weights or {}

    for source in sources:
        payload = extract_domain_payload(context, source)
        if not payload:
            continue

        vote = build_direction_vote(
            source=source,
            payload=payload,
            weight=weights.get(source, 1.0),
        )

        if vote.directional:
            result.append(vote)

    return result


def vote_strength_for_side(
    votes: Sequence[DirectionVote],
    side: SignalSide,
) -> float:
    return sum(
        vote.weighted_strength
        for vote in votes
        if vote.side is side
    )


def dominant_side_from_votes(
    votes: Sequence[DirectionVote],
    *,
    min_strength: float = 0.0,
) -> SignalSide:
    long_strength = vote_strength_for_side(votes, SignalSide.LONG)
    short_strength = vote_strength_for_side(votes, SignalSide.SHORT)

    if long_strength <= 0.0 and short_strength <= 0.0:
        return SignalSide.UNKNOWN

    if abs(long_strength - short_strength) < min_strength:
        return SignalSide.UNKNOWN

    return SignalSide.LONG if long_strength > short_strength else SignalSide.SHORT


def votes_for_side(
    votes: Sequence[DirectionVote],
    side: SignalSide,
) -> list[DirectionVote]:
    return [vote for vote in votes if vote.side is side]


def votes_against_side(
    votes: Sequence[DirectionVote],
    side: SignalSide,
) -> list[DirectionVote]:
    opposite = opposite_signal_side(side)
    return [vote for vote in votes if vote.side is opposite]


def strong_votes(
    votes: Sequence[DirectionVote],
    *,
    min_score: float = 0.55,
    min_confidence: float = 0.55,
) -> list[DirectionVote]:
    return [
        vote
        for vote in votes
        if vote.directional
        and vote.score >= min_score
        and vote.confidence >= min_confidence
    ]


# =============================================================================
# Alignment / conflict / confluence scoring
# =============================================================================


def alignment_score(
    votes: Sequence[DirectionVote],
    *,
    side: SignalSide | None = None,
) -> float:
    directional_votes = [vote for vote in votes if vote.directional]
    if not directional_votes:
        return 0.0

    target_side = side or dominant_side_from_votes(directional_votes)
    if not is_directional_side(target_side):
        return 0.0

    total_strength = sum(vote.weighted_strength for vote in directional_votes)
    if total_strength <= 0.0:
        return 0.0

    aligned_strength = vote_strength_for_side(directional_votes, target_side)
    return unit_score(aligned_strength / total_strength)


def conflict_score(
    votes: Sequence[DirectionVote],
    *,
    side: SignalSide | None = None,
) -> float:
    directional_votes = [vote for vote in votes if vote.directional]
    if not directional_votes:
        return 0.0

    target_side = side or dominant_side_from_votes(directional_votes)
    if not is_directional_side(target_side):
        return 1.0 if len(directional_votes) > 1 else 0.0

    total_strength = sum(vote.weighted_strength for vote in directional_votes)
    if total_strength <= 0.0:
        return 0.0

    conflict_strength = sum(
        vote.weighted_strength
        for vote in directional_votes
        if vote.side is opposite_signal_side(target_side)
    )
    return unit_score(conflict_strength / total_strength)


def conflict_penalty(
    votes: Sequence[DirectionVote],
    *,
    side: SignalSide | None = None,
    max_penalty: float = 0.35,
) -> float:
    return unit_score(conflict_score(votes, side=side) * max(0.0, max_penalty))


def domain_confluence_score(
    votes: Sequence[DirectionVote],
    *,
    side: SignalSide | None = None,
    min_domains: int = 2,
) -> float:
    directional_votes = [vote for vote in votes if vote.directional]
    if not directional_votes:
        return 0.0

    target_side = side or dominant_side_from_votes(directional_votes)
    if not is_directional_side(target_side):
        return 0.0

    aligned_votes = votes_for_side(directional_votes, target_side)
    domain_count_score = unit_score(len(aligned_votes) / max(min_domains, 1))
    alignment = alignment_score(directional_votes, side=target_side)

    avg_vote_strength = average_score(
        *[vote.weighted_strength for vote in aligned_votes],
        default=0.0,
    )

    return weighted_score(
        {
            "domain_count": domain_count_score,
            "alignment": alignment,
            "vote_strength": avg_vote_strength,
        },
        {
            "domain_count": 0.30,
            "alignment": 0.40,
            "vote_strength": 0.30,
        },
    )


def weighted_direction_vote(
    votes: Sequence[DirectionVote],
    *,
    side: SignalSide | None = None,
    min_domains: int = 2,
    conflict_penalty_weight: float = 0.25,
) -> float:
    target_side = side or dominant_side_from_votes(votes)
    if not is_directional_side(target_side):
        return 0.0

    base_score = domain_confluence_score(
        votes,
        side=target_side,
        min_domains=min_domains,
    )
    penalty = conflict_penalty(
        votes,
        side=target_side,
        max_penalty=conflict_penalty_weight,
    )

    return unit_score(base_score - penalty)


def aligned_source_names(
    votes: Sequence[DirectionVote],
    side: SignalSide,
) -> list[str]:
    return [
        vote.source.value
        for vote in votes
        if vote.side is side
    ]


def conflicting_source_names(
    votes: Sequence[DirectionVote],
    side: SignalSide,
) -> list[str]:
    opposite = opposite_signal_side(side)
    return [
        vote.source.value
        for vote in votes
        if vote.side is opposite
    ]


# =============================================================================
# Freshness helpers
# =============================================================================


def freshness_score(
    *,
    event_time: datetime | None,
    now: datetime | None = None,
    stale_after_seconds: float | None = DEFAULT_HYBRID_STALE_AFTER_SECONDS,
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
    stale_after_seconds: float | None = DEFAULT_HYBRID_STALE_AFTER_SECONDS,
) -> bool:
    if event_time is None or stale_after_seconds is None:
        return False

    current = ensure_aware_utc(now or utc_now())
    event_ts = ensure_aware_utc(event_time)

    age = max(0.0, (current - event_ts).total_seconds())
    return age > stale_after_seconds


def latest_timestamp_from_payloads(
    payloads: Mapping[FeatureSource, Mapping[str, Any]],
    *,
    fallback: datetime | None = None,
) -> datetime | None:
    timestamps: list[datetime] = []

    for payload in payloads.values():
        event_time = extract_domain_timestamp(payload)
        if event_time is not None:
            timestamps.append(event_time)

    if timestamps:
        return max(timestamps)

    return parse_datetime(fallback)


def hybrid_freshness_score(
    payloads: Mapping[FeatureSource, Mapping[str, Any]],
    *,
    now: datetime | None = None,
    stale_after_seconds: float | None = DEFAULT_HYBRID_STALE_AFTER_SECONDS,
) -> float:
    if not payloads:
        return 0.0

    scores = [
        domain_freshness_score(
            payload,
            now=now,
            stale_after_seconds=stale_after_seconds,
        )
        for payload in payloads.values()
    ]

    return average_score(*scores, default=1.0)


# =============================================================================
# Hybrid score DTO
# =============================================================================


@dataclass(slots=True)
class HybridScoreBreakdown:
    _logger = logging.getLogger(__name__ + ".HybridScoreBreakdown")
    score: float = 0.0
    confidence: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    votes: list[DirectionVote] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    confirmations: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    def normalize(self) -> HybridScoreBreakdown:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridScoreBreakdown.normalize")
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
        self.conflicts = list(
            dict.fromkeys(
                str(item).strip()
                for item in self.conflicts
                if str(item).strip()
            )
        )
        return self

    def to_dict(self) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering HybridScoreBreakdown.to_dict")
        self.normalize()
        return {
            "score": self.score,
            "confidence": self.confidence,
            "components": dict(self.components),
            "weights": dict(self.weights),
            "votes": [vote.to_dict() for vote in self.votes],
            "reasons": list(self.reasons),
            "confirmations": list(self.confirmations),
            "conflicts": list(self.conflicts),
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


def build_hybrid_score_breakdown(
    *,
    votes: Sequence[DirectionVote],
    side: SignalSide,
    payloads: Mapping[FeatureSource, Mapping[str, Any]],
    now: datetime | None = None,
    stale_after_seconds: float | None = DEFAULT_HYBRID_STALE_AFTER_SECONDS,
    min_domains: int = 2,
    weights: Mapping[str, float] | None = None,
    reasons: list[str] | None = None,
    confirmations: list[str] | None = None,
) -> HybridScoreBreakdown:
    weights = dict(
        weights
        or {
            "confluence": 0.34,
            "alignment": 0.26,
            "vote_strength": 0.20,
            "freshness": 0.10,
            "conflict_inverse": 0.10,
        }
    )

    aligned_votes = votes_for_side(votes, side)
    conflicts = conflicting_source_names(votes, side)

    confluence = domain_confluence_score(
        votes,
        side=side,
        min_domains=min_domains,
    )
    alignment = alignment_score(votes, side=side)
    vote_strength = average_score(
        *[vote.weighted_strength for vote in aligned_votes],
        default=0.0,
    )
    freshness = hybrid_freshness_score(
        payloads,
        now=now,
        stale_after_seconds=stale_after_seconds,
    )
    conflict_inverse = 1.0 - conflict_score(votes, side=side)

    components = {
        "confluence": confluence,
        "alignment": alignment,
        "vote_strength": vote_strength,
        "freshness": freshness,
        "conflict_inverse": conflict_inverse,
    }

    score = weighted_score(components, weights, default=confluence)
    confidence = confidence_from_components(
        primary=average_score(confluence, vote_strength),
        context=alignment,
        confirmation=conflict_inverse,
        freshness=freshness,
    )

    return HybridScoreBreakdown(
        score=score,
        confidence=confidence,
        components=components,
        weights=dict(weights),
        votes=list(votes),
        reasons=list(reasons or []),
        confirmations=list(confirmations or []),
        conflicts=conflicts,
    ).normalize()


# =============================================================================
# Domain-specific source groups
# =============================================================================


TREND_STACK_SOURCES: tuple[FeatureSource, ...] = (
    FeatureSource.PRICE_ACTION,
    FeatureSource.ORDERFLOW,
    FeatureSource.OPEN_INTEREST,
    FeatureSource.WHALES,
    FeatureSource.FUNDING,
)

MEAN_REVERSION_STACK_SOURCES: tuple[FeatureSource, ...] = (
    FeatureSource.LIQUIDITY,
    FeatureSource.ORDERFLOW,
    FeatureSource.PRICE_ACTION,
    FeatureSource.LIQUIDATIONS,
    FeatureSource.WHALES,
)

LIQUIDATION_WHALE_SOURCES: tuple[FeatureSource, ...] = (
    FeatureSource.LIQUIDATIONS,
    FeatureSource.WHALES,
)

LIQUIDITY_ORDERFLOW_REVERSAL_SOURCES: tuple[FeatureSource, ...] = (
    FeatureSource.LIQUIDITY,
    FeatureSource.ORDERFLOW,
    FeatureSource.PRICE_ACTION,
)

OI_FUNDING_SQUEEZE_SOURCES: tuple[FeatureSource, ...] = (
    FeatureSource.OPEN_INTEREST,
    FeatureSource.FUNDING,
    FeatureSource.PRICE_ACTION,
)

WHALE_ORDERFLOW_BREAKOUT_SOURCES: tuple[FeatureSource, ...] = (
    FeatureSource.WHALES,
    FeatureSource.ORDERFLOW,
    FeatureSource.PRICE_ACTION,
)


# =============================================================================
# Hybrid source-feature helpers
# =============================================================================


def source_features_from_paths(*paths: str) -> list[str]:
    result: list[str] = []

    for path in paths:
        if not isinstance(path, str) or not path.strip():
            continue

        normalized = path.strip()

        if normalized.startswith("hybrid."):
            result.append(normalized)
        else:
            result.append(f"hybrid.{normalized}")

    return list(dict.fromkeys(result))


def hybrid_base_source_features() -> list[str]:
    return source_features_from_paths(
        "orderflow",
        "liquidity",
        "liquidations",
        "whales",
        "open_interest",
        "funding",
        "price_action",
        "spoofing",
        "spreads",
        "dominant_side",
        "alignment_score",
        "conflict_score",
        "confluence_score",
        "confidence",
        "votes",
    )


def confluence_source_features() -> list[str]:
    return list(
        dict.fromkeys(
            [
                *hybrid_base_source_features(),
                "orderflow.side",
                "liquidity.side",
                "liquidations.side",
                "whales.side",
                "open_interest.side",
                "funding.side",
                "price_action.side",
            ]
        )
    )


def trend_stack_source_features() -> list[str]:
    return source_features_from_paths(
        "price_action.trend",
        "price_action.side",
        "orderflow.side",
        "orderflow.continuation",
        "open_interest.side",
        "open_interest.expansion",
        "funding.extreme",
        "whales.side",
        "alignment_score",
        "confluence_score",
    )


def mean_reversion_stack_source_features() -> list[str]:
    return source_features_from_paths(
        "liquidity.sweep",
        "liquidity.side",
        "orderflow.exhaustion",
        "orderflow.side",
        "price_action.rejection",
        "liquidations.side",
        "whales.absorption",
        "alignment_score",
        "confluence_score",
    )


def liquidation_whale_source_features() -> list[str]:
    return source_features_from_paths(
        "liquidations",
        "liquidations.side",
        "liquidations.cascade",
        "whales",
        "whales.whale_side",
        "whales.liquidation_context",
        "whales.exhaustion",
        "alignment_score",
        "confluence_score",
    )


def liquidity_orderflow_reversal_source_features() -> list[str]:
    return source_features_from_paths(
        "liquidity.sweep",
        "liquidity.stop_hunt",
        "orderflow.exhaustion",
        "orderflow.absorption",
        "price_action.rejection",
        "alignment_score",
        "confluence_score",
    )


def oi_funding_squeeze_source_features() -> list[str]:
    return source_features_from_paths(
        "open_interest",
        "open_interest.regime",
        "open_interest.divergence",
        "funding",
        "funding.extreme",
        "funding.divergence",
        "price_action.confirmation",
        "alignment_score",
        "confluence_score",
    )


def whale_orderflow_breakout_source_features() -> list[str]:
    return source_features_from_paths(
        "whales.activity",
        "whales.pressure",
        "whales.large_trade",
        "orderflow.continuation",
        "orderflow.delta",
        "price_action.breakout",
        "alignment_score",
        "confluence_score",
    )