# trading_system/strategy/strategies/liquidity/base.py

from __future__ import annotations
import logging

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from math import isfinite
from typing import Any

from analytics.liquidity.enums import (
    LiquiditySide,
)
from analytics.liquidity.models import (
    LiquidityLevel,
    LiquidityMapSnapshot,
    LiquidityZone,
    StopCluster,
)
from core.event_bus import EventBus
from core.scheduler import Scheduler
from ..base_strategy import TradingStrategy
from ...config import StrategyConfig, StrategyDefinitionConfig
from ...enums import (
    EntryType,
    ExitType,
    FeatureSource,
    FilterDecision,
    MarketRegime,
    SetupType,
    SignalPriority,
    SignalSide,
    StrategyCategory,
    StrategyMarginMode,
    StrategyMarketType,
    StrategyOrderIntent,
    StrategyTradeTier,
    Timeframe,
)
from ...exceptions import StrategyConfigError, StrategyEvaluationError
from ...models import (
    EntryPlan,
    ExitPlan,
    FeatureSnapshot,
    FilterResult,
    InvalidationPlan,
    StrategyContext,
    StrategySignal,
    TargetPlan,
    clamp,
    ensure_aware_utc,
    utcnow,
)

DECIMAL_ZERO = Decimal("0")


# =============================================================================
# Generic helpers
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
    Serialize small nested values for StrategySignal.metadata.

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


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return converted

    return None


def _get_attr_or_key(value: Any, key: str, default: Any = None) -> Any:
    mapping = _as_mapping(value)
    if mapping is not None:
        return mapping.get(key, default)

    return getattr(value, key, default)


def _get_path(value: Any, path: str, default: Any = None) -> Any:
    """
    Read dotted path from dict-like/object-like data.

    Important compatibility detail: StrategyContext domain data often contains
    literal dotted feature-map keys such as ``liquidity.snapshot`` or
    ``liquidity.map.snapshot`` alongside nested objects.  Older liquidity
    strategy helpers only walked nested paths, so those literal keys were
    invisible and full-snapshot strategies were routed but then failed with
    ``missing_liquidity_snapshot_contract``.
    """
    if not isinstance(path, str) or not path.strip():
        return default

    normalized = path.strip()
    mapping = _as_mapping(value)
    if mapping is not None and normalized in mapping:
        item = mapping.get(normalized)
        return default if item is None else item

    current = value

    for index, part in enumerate(normalized.split(".")):
        if current is None:
            return default

        part = part.strip()
        if not part:
            return default

        current_mapping = _as_mapping(current)
        if current_mapping is not None:
            # Prefer exact remainder for payloads that store literal dotted keys.
            remainder = ".".join(normalized.split(".")[index:])
            if remainder in current_mapping:
                item = current_mapping.get(remainder)
                return default if item is None else item

        current = _get_attr_or_key(current, part, default=None)

    return default if current is None else current


def _first_present(
    value: Any,
    paths: Sequence[str],
    *,
    default: Any = None,
) -> Any:
    for path in paths:
        item = _get_path(value, path, default=None)
        if item is not None:
            return item
    return default


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float, Decimal)):
        parsed = float(value)
        return parsed if isfinite(parsed) else default

    if isinstance(value, Enum):
        return _to_float(value.value, default)

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


def _to_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    if isinstance(value, (float, Decimal)):
        return int(value)

    if isinstance(value, Enum):
        return _to_int(value.value, default)

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


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, Enum):
        return _to_bool(value.value, default)

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"1", "true", "yes", "y", "on", "active", "valid"}:
            return True

        if normalized in {"0", "false", "no", "n", "off", "inactive", "invalid"}:
            return False

    if isinstance(value, (int, float, Decimal)):
        return bool(value)

    return default


def _normalize_label(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value

    if value is None:
        return ""

    return str(value).strip().lower()


def _value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def _clamp01(value: Any, default: float = 0.0) -> float:
    parsed = _to_float(value, default)
    return clamp(float(parsed if parsed is not None else default), 0.0, 1.0)


def _clamp_signed(value: Any, default: float = 0.0) -> float:
    parsed = _to_float(value, default)
    return clamp(float(parsed if parsed is not None else default), -1.0, 1.0)


# =============================================================================
# Feature contract
# =============================================================================


@dataclass(frozen=True, slots=True)
class LiquidityFeatureNames:
    """
    Stable feature names expected in StrategyContext.

    Generic SignalNormalizer can create these from analytics.liquidity.* payloads,
    or concrete strategies can read equivalent values from domain_data aliases.
    """
    _logger = logging.getLogger(__name__ + ".LiquidityFeatureNames")

    SNAPSHOT: str = "liquidity.snapshot"
    MAP_SNAPSHOT: str = "liquidity.map.snapshot"
    CURRENT_PRICE: str = "liquidity.current_price"

    ABOVE_LIQUIDITY_SCORE: str = "liquidity.above_liquidity_score"
    BELOW_LIQUIDITY_SCORE: str = "liquidity.below_liquidity_score"
    PRESSURE_SCORE: str = "liquidity.pressure_score"
    BIAS: str = "liquidity.bias"

    SWEEP_RISK_UP: str = "liquidity.sweep_risk.up"
    SWEEP_RISK_DOWN: str = "liquidity.sweep_risk.down"
    MAGNET_UP: str = "liquidity.magnet.up"
    MAGNET_DOWN: str = "liquidity.magnet.down"

    NEAREST_ABOVE_LEVEL: str = "liquidity.nearest_above_level"
    NEAREST_BELOW_LEVEL: str = "liquidity.nearest_below_level"
    STRONGEST_CLUSTER_ABOVE: str = "liquidity.strongest_cluster_above"
    STRONGEST_CLUSTER_BELOW: str = "liquidity.strongest_cluster_below"

    EQUAL_LEVELS: str = "liquidity.equal_levels"
    ACTIVE_LEVELS: str = "liquidity.active_levels"
    STOP_CLUSTERS: str = "liquidity.stop_clusters"
    ZONES: str = "liquidity.zones"

    @classmethod
    def all(cls) -> set[str]:
        _strategy_logger = getattr(cls, "_logger", None) or logging.getLogger(__name__ + ".LiquidityFeatureNames")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityFeatureNames.all")
        instance = cls()
        return {
            getattr(instance, item.name)
            for item in fields(cls)
            if isinstance(getattr(instance, item.name), str)
            and getattr(instance, item.name).strip()
        }


LIQUIDITY_FEATURES = LiquidityFeatureNames()


LIQUIDITY_DOMAIN_ALIASES: dict[str, tuple[str, ...]] = {
    "snapshot": (
        "snapshot",
        "liquidity_snapshot",
        "liquidity_map_snapshot",
        "liquidity.map.snapshot",
        "liquidity.snapshot",
        "map_snapshot",
        "map",
        "liquidity_map",
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
    "liquidity.map.updated",
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


# =============================================================================
# Scope / config
# =============================================================================


@dataclass(frozen=True, slots=True)
class LiquidityStrategyScope:
    _logger = logging.getLogger(__name__ + ".LiquidityStrategyScope")
    exchange: str
    market_type: str
    symbol: str
    timeframe: str
    exchange_symbol: str | None = None

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityStrategyScope.__post_init__")
        exchange = str(self.exchange or "unknown").strip().lower()
        market_type = str(
            self.market_type or StrategyMarketType.USDM_FUTURES.value
        ).strip()
        symbol = str(self.symbol or "").strip().upper()
        timeframe = str(self.timeframe or Timeframe.M1.value).strip().lower()
        exchange_symbol = str(self.exchange_symbol or symbol).strip().upper()

        if not symbol:
            raise StrategyEvaluationError("LiquidityStrategyScope.symbol cannot be empty")

        object.__setattr__(self, "exchange", exchange)
        object.__setattr__(self, "market_type", market_type)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timeframe", timeframe)
        object.__setattr__(self, "exchange_symbol", exchange_symbol)

    @property
    def key(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityStrategyScope.key")
        return f"{self.exchange}:{self.market_type}:{self.symbol}:{self.timeframe}"

    @property
    def legacy_key(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityStrategyScope.legacy_key")
        return f"{self.symbol}:{self.exchange}"

    def to_dict(self) -> dict[str, str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityStrategyScope.to_dict")
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "exchange_symbol": self.exchange_symbol or self.symbol,
            "key": self.key,
            "legacy_key": self.legacy_key,
        }


@dataclass(slots=True)
class LiquidityStrategyConfig:
    """
    Domain config shared by concrete liquidity strategies.

    Runtime enabled/symbol/timeframe/regime checks belong to StrategyConfig /
    StrategyDefinitionConfig. This config keeps liquidity-specific defaults and
    quality thresholds.
    """
    _logger = logging.getLogger(__name__ + ".LiquidityStrategyConfig")

    default_market_type: StrategyMarketType = StrategyMarketType.USDM_FUTURES
    default_margin_mode: StrategyMarginMode = StrategyMarginMode.ISOLATED
    default_order_intent: StrategyOrderIntent = StrategyOrderIntent.OPEN
    default_trade_tier: StrategyTradeTier = StrategyTradeTier.T2

    default_entry_type: EntryType = EntryType.MARKET
    default_exit_types: tuple[ExitType, ...] = (
        ExitType.TAKE_PROFIT,
        ExitType.STOP_LOSS,
        ExitType.INVALIDATION,
    )

    min_context_confidence: float = 0.0
    min_signal_confidence: float = 0.45
    min_signal_score: float = 0.30

    min_snapshot_confidence: float = 0.0
    min_liquidity_strength: float = 0.0
    max_snapshot_age_seconds: float | None = 300.0

    require_futures_market_type: bool = True

    requested_leverage: float | None = None
    max_slippage_bps: float | None = None
    entry_timeout_seconds: int | None = None
    max_holding_seconds: int | None = None

    attach_liquidity_context_metadata: bool = True
    attach_snapshot_metadata: bool = True
    attach_scope_metadata: bool = True
    attach_feature_values_metadata: bool = True

    tag_liquidity: str = "liquidity"
    tag_sweep: str = "liquidity_sweep"
    tag_bias: str = "liquidity_bias"
    tag_stop_hunt: str = "stop_hunt"
    tag_equal_levels: str = "equal_high_low"
    tag_reversal: str = "reversal"
    tag_continuation: str = "continuation"

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityStrategyConfig.validate")
        bounded = {
            "min_context_confidence": self.min_context_confidence,
            "min_signal_confidence": self.min_signal_confidence,
            "min_signal_score": self.min_signal_score,
            "min_snapshot_confidence": self.min_snapshot_confidence,
            "min_liquidity_strength": self.min_liquidity_strength,
        }

        for name, value in bounded.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{name} must be between 0.0 and 1.0")

        if self.max_snapshot_age_seconds is not None and self.max_snapshot_age_seconds <= 0:
            raise StrategyConfigError("max_snapshot_age_seconds must be > 0")

        if self.requested_leverage is not None and self.requested_leverage <= 0:
            raise StrategyConfigError("requested_leverage must be > 0")

        if self.max_slippage_bps is not None and self.max_slippage_bps < 0:
            raise StrategyConfigError("max_slippage_bps must be >= 0")

        if self.entry_timeout_seconds is not None and self.entry_timeout_seconds <= 0:
            raise StrategyConfigError("entry_timeout_seconds must be > 0")

        if self.max_holding_seconds is not None and self.max_holding_seconds <= 0:
            raise StrategyConfigError("max_holding_seconds must be > 0")

        for attr in (
            "tag_liquidity",
            "tag_sweep",
            "tag_bias",
            "tag_stop_hunt",
            "tag_equal_levels",
            "tag_reversal",
            "tag_continuation",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")


# =============================================================================
# Base strategy
# =============================================================================




# =============================================================================
# Snapshot normalization helpers
# =============================================================================


def _truthy_payload(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (list, tuple, set)):
        return bool(value)
    return True


def _merge_mapping_values(*values: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in values:
        mapping = _as_mapping(value)
        if mapping:
            merged.update(dict(mapping))
    return merged


def _scope_mapping(value: Any) -> Mapping[str, Any]:
    scope = _get_path(value, "scope", default=None)
    mapping = _as_mapping(scope)
    return mapping or {}


def _candidate_with_feature_map(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """
    Merge common normalized envelopes into one mapping.

    Analytics liquidity events often carry useful fields in ``feature_map`` and
    ``payload`` while StrategyContext domain_data keeps top-level aliases.  The
    strategy-side resolver should see one canonical mapping instead of guessing
    which envelope variant was used.
    """
    result = dict(candidate)

    raw = candidate.get("raw")
    raw_mapping = _as_mapping(raw)
    if raw_mapping:
        result.update({k: v for k, v in raw_mapping.items() if k not in result or result[k] is None})

    feature_map = candidate.get("feature_map")
    feature_mapping = _as_mapping(feature_map)
    if feature_mapping:
        for key, value in feature_mapping.items():
            result.setdefault(str(key), value)

    payload = candidate.get("payload")
    payload_mapping = _as_mapping(payload)
    if payload_mapping:
        for key, value in payload_mapping.items():
            result.setdefault(str(key), value)

    signal = (
        candidate.get("signal")
        or candidate.get("liquidity_signal")
        or candidate.get("analytics_signal")
    )
    signal_mapping = _as_mapping(signal)
    if signal_mapping:
        for key in ("confidence", "score", "side", "bias", "strength"):
            if key in signal_mapping:
                result.setdefault(key, signal_mapping.get(key))

    return result


def _first_payload_value(value: Any, paths: Sequence[str], default: Any = None) -> Any:
    for path in paths:
        item = _get_path(value, path, default=None)
        if item is not None:
            return item
    return default


def _coerce_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if item is not None]
    return [value]


def _normalize_snapshot_mapping(candidate: Mapping[str, Any]) -> dict[str, Any]:
    data = _candidate_with_feature_map(candidate)
    scope = _scope_mapping(data)

    symbol = str(
        _first_payload_value(
            data,
            (
                "symbol",
                "scope.symbol",
                "liquidity.symbol",
                "hybrid.symbol",
            ),
            "",
        )
        or scope.get("symbol")
        or ""
    ).strip().upper()

    exchange = str(
        _first_payload_value(data, ("exchange", "scope.exchange", "hybrid.exchange"), "")
        or scope.get("exchange")
        or "binance"
    ).strip().lower()

    market_type = str(
        _first_payload_value(data, ("market_type", "scope.market_type", "hybrid.market_type"), "")
        or scope.get("market_type")
        or "usdm_futures"
    ).strip()

    timeframe = str(
        _first_payload_value(data, ("timeframe", "scope.timeframe", "hybrid.timeframe"), "")
        or scope.get("timeframe")
        or "1m"
    ).strip().lower()

    exchange_symbol = str(
        _first_payload_value(data, ("exchange_symbol", "scope.exchange_symbol"), "")
        or scope.get("exchange_symbol")
        or symbol
    ).strip().upper()

    current_price = _to_float(
        _first_payload_value(
            data,
            (
                "current_price",
                "reference_price",
                "entry_reference_price",
                "last_price",
                "price",
                "mark_price",
                "close",
                "liquidity.current_price",
            ),
        )
    )

    active_levels = _coerce_items(
        _first_payload_value(
            data,
            (
                "active_levels",
                "levels",
                "liquidity_levels",
                "source_levels",
                "liquidity.active_levels",
            ),
            [],
        )
    )

    equal_levels = _coerce_items(
        _first_payload_value(
            data,
            (
                "equal_levels",
                "equal_highs_lows",
                "liquidity.equal_levels",
            ),
            [],
        )
    )

    stop_clusters = _coerce_items(
        _first_payload_value(
            data,
            (
                "stop_clusters",
                "clusters",
                "liquidity_clusters",
                "liquidity.stop_clusters",
            ),
            [],
        )
    )

    zones = _coerce_items(
        _first_payload_value(
            data,
            (
                "zones",
                "liquidity_zones",
                "liquidity.zones",
            ),
            [],
        )
    )

    nearest_above = _first_payload_value(
        data,
        (
            "nearest_above_level",
            "nearest_buy_side_liquidity",
            "strongest_cluster_above",
            "liquidity.nearest_above_level",
        ),
    )
    nearest_below = _first_payload_value(
        data,
        (
            "nearest_below_level",
            "nearest_sell_side_liquidity",
            "strongest_cluster_below",
            "liquidity.nearest_below_level",
        ),
    )

    strongest_above = _first_payload_value(
        data,
        (
            "strongest_cluster_above",
            "nearest_buy_side_liquidity",
            "liquidity.strongest_cluster_above",
        ),
    )
    strongest_below = _first_payload_value(
        data,
        (
            "strongest_cluster_below",
            "nearest_sell_side_liquidity",
            "liquidity.strongest_cluster_below",
        ),
    )

    above_score = _to_float(
        _first_payload_value(
            data,
            (
                "above_liquidity_score",
                "magnet_score_up",
                "upside_sweep_risk",
                "sweep_risk_up",
                "liquidity.magnet.up",
                "liquidity.sweep_risk.up",
            ),
        ),
        0.0,
    )
    below_score = _to_float(
        _first_payload_value(
            data,
            (
                "below_liquidity_score",
                "magnet_score_down",
                "downside_sweep_risk",
                "sweep_risk_down",
                "liquidity.magnet.down",
                "liquidity.sweep_risk.down",
            ),
        ),
        0.0,
    )

    pressure = _to_float(
        _first_payload_value(
            data,
            (
                "liquidity_pressure_score",
                "pressure_score",
                "liquidity.pressure_score",
            ),
        ),
        None,
    )
    if pressure is None:
        up = _to_float(_first_payload_value(data, ("sweep_risk_up", "magnet_score_up")), 0.0) or 0.0
        down = _to_float(_first_payload_value(data, ("sweep_risk_down", "magnet_score_down")), 0.0) or 0.0
        pressure = up - down

    confidence = _to_float(
        _first_payload_value(data, ("confidence", "signal.confidence", "liquidity_signal.confidence")),
        0.0,
    )

    metadata = _merge_mapping_values(data.get("metadata"))
    metadata.update(
        {
            "source": "strategy_liquidity_resolver",
            "payload_contract_level": data.get("payload_contract_level"),
            "confidence": confidence,
            "raw_event_name": data.get("event_name") or data.get("topic"),
        }
    )

    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
        "exchange_symbol": exchange_symbol,
        "timestamp": parse_datetime(
            _first_payload_value(data, ("timestamp", "event_time", "updated_at", "created_at", "price_timestamp"))
        )
        or utc_now(),
        "current_price": current_price,
        "active_levels": active_levels,
        "equal_levels": equal_levels,
        "stop_clusters": stop_clusters,
        "zones": zones,
        "nearest_above_level": nearest_above,
        "nearest_below_level": nearest_below,
        "strongest_cluster_above": strongest_above,
        "strongest_cluster_below": strongest_below,
        "above_liquidity_score": _clamp01(above_score),
        "below_liquidity_score": _clamp01(below_score),
        "liquidity_pressure_score": _clamp_signed(pressure),
        "bias": _first_payload_value(data, ("bias", "liquidity.bias"), "neutral"),
        "signal": data.get("signal") or data.get("liquidity_signal") or data.get("analytics_signal"),
        "metadata": metadata,
    }

class LiquidityTradingStrategy(TradingStrategy):
    """
    Base class for concrete strategy/strategies/liquidity/* classes.

    Responsibilities:
    - read LiquidityMapSnapshot / liquidity domain data from StrategyContext;
    - provide reusable level/cluster/zone/target helpers;
    - build internal StrategySignal objects;
    - attach liquidity metadata for SignalProcessor.

    Forbidden:
    - no direct analytics detector calls;
    - no raw exchange/data cache reads;
    - no direct risk/execution calls;
    - no EventBus emit of signal.generated;
    - no evaluate_and_emit() in concrete strategies.
    """
    _logger = logging.getLogger(__name__ + ".LiquidityTradingStrategy")

    component_namespace = "strategy.liquidity"
    category: StrategyCategory = StrategyCategory.LIQUIDITY
    default_setup_type: SetupType = SetupType.UNKNOWN
    default_timeframe: Timeframe = Timeframe.M1

    feature_names = LIQUIDITY_FEATURES

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        liquidity_config: LiquidityStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.__init__")
        self.liquidity_config = liquidity_config or LiquidityStrategyConfig()
        self.liquidity_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            service_name=service_name,
        )

    def validate_config(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.validate_config")
        super().validate_config()
        self.liquidity_config.validate()

    # ------------------------------------------------------------------
    # Context / domain
    # ------------------------------------------------------------------

    def liquidity_domain(self, context: StrategyContext) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.liquidity_domain")
        self.validate_context(context)
        domain = context.domain_dict(FeatureSource.LIQUIDITY)
        return dict(domain)

    def liquidity_item(
        self,
        context: StrategyContext,
        key: str,
        default: Any = None,
    ) -> Any:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.liquidity_item")
        domain = self.liquidity_domain(context)

        if key in domain:
            return domain[key]

        for alias in LIQUIDITY_DOMAIN_ALIASES.get(key, ()):
            value = _get_path(domain, alias, default=None)
            if value is not None:
                return value

        return default

    def liquidity_path(
        self,
        context: StrategyContext,
        path: str,
        default: Any = None,
    ) -> Any:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.liquidity_path")
        self.validate_context(context)

        if not isinstance(path, str) or not path.strip():
            raise StrategyEvaluationError("liquidity path cannot be empty")

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

        domain = self.liquidity_domain(context)
        if normalized.startswith("liquidity."):
            normalized = normalized.removeprefix("liquidity.")

        return _get_path(domain, normalized, default)

    def liquidity_float(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float | None = None,
    ) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.liquidity_float")
        return _to_float(self.liquidity_path(context, path, default), default)

    def liquidity_score(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float = 0.0,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.liquidity_score")
        return _clamp01(self.liquidity_path(context, path, default), default)

    def liquidity_signed_score(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float = 0.0,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.liquidity_signed_score")
        return _clamp_signed(self.liquidity_path(context, path, default), default)

    def liquidity_snapshot(
        self,
        context: StrategyContext,
    ) -> LiquidityMapSnapshot | None:
        """
        Extract LiquidityMapSnapshot from StrategyContext.

        Supports:
        - domain_data[FeatureSource.LIQUIDITY]["snapshot" / "liquidity_map_snapshot"];
        - context.get_feature(...);
        - context.get_feature_snapshot(...);
        - wrappers: value/data/snapshot/payload;
        - legacy context.liquidity.snapshot-like objects.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.liquidity_snapshot")
        self.validate_context(context)

        domain = self.liquidity_domain(context)
        candidates: list[Any] = [domain]

        for key in LIQUIDITY_DOMAIN_ALIASES["snapshot"]:
            candidates.append(_get_path(domain, key, default=None))

        legacy_liquidity_context = getattr(context, "liquidity", None)
        if legacy_liquidity_context is not None:
            for key in LIQUIDITY_DOMAIN_ALIASES["snapshot"]:
                candidates.append(_get_path(legacy_liquidity_context, key, default=None))

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
            snapshot = self._unwrap_snapshot_candidate(candidate)
            if snapshot is not None:
                return snapshot

        return None

    def _unwrap_snapshot_candidate(
        self,
        candidate: Any,
    ) -> LiquidityMapSnapshot | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy._unwrap_snapshot_candidate")
        if isinstance(candidate, LiquidityMapSnapshot):
            return candidate

        if candidate is None:
            return None

        if isinstance(candidate, FeatureSnapshot):
            return self._unwrap_snapshot_candidate(candidate.value)

        if isinstance(candidate, Mapping):
            normalized = _normalize_snapshot_mapping(candidate)

            # Build a strategy-side LiquidityMapSnapshot when the payload has a
            # real symbol and usable price.  This supports both
            # analytics.liquidity.map.updated and enriched signal.updated events
            # where symbol lives under scope and snapshot fields are flattened.
            if normalized.get("current_price") is not None and normalized.get("symbol"):
                try:
                    return LiquidityMapSnapshot(**normalized)
                except Exception:
                    # Continue unwrapping nested candidates below.  Some older
                    # deployments may require analytics model instances for
                    # levels/clusters; nested snapshot objects should still win.
                    pass

            for key in ("snapshot", "liquidity_snapshot", "liquidity_map_snapshot", "map_snapshot", "map", "value", "data", "payload"):
                nested = candidate.get(key)
                snapshot = self._unwrap_snapshot_candidate(nested)
                if snapshot is not None:
                    return snapshot

        for attr in ("snapshot", "value", "data", "payload"):
            nested = getattr(candidate, attr, None)
            snapshot = self._unwrap_snapshot_candidate(nested)
            if snapshot is not None:
                return snapshot

        return None

    def current_price(
        self,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot | None = None,
    ) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.current_price")
        candidates = [
            getattr(context, "current_price", None),
            getattr(context, "price", None),
            getattr(context, "mark_price", None),
            _get_path(getattr(context, "market", None), "current_price"),
            _get_path(getattr(context, "market", None), "price"),
            self.liquidity_item(context, "current_price"),
            self.liquidity_path(context, "current_price", default=None),
        ]

        if snapshot is not None:
            candidates.extend(
                [
                    getattr(snapshot, "current_price", None),
                    getattr(snapshot, "price", None),
                    getattr(snapshot, "mark_price", None),
                    getattr(snapshot, "last_price", None),
                    _get_path(getattr(snapshot, "metadata", None), "current_price"),
                    _get_path(getattr(snapshot, "metadata", None), "price"),
                    _get_path(getattr(snapshot, "metadata", None), "mark_price"),
                ]
            )

        for candidate in candidates:
            value = _to_float(candidate)
            if value is not None and value > 0:
                return value

        return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def base_context_is_valid(
        self,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.base_context_is_valid")
        if not self._snapshot_matches_context(context=context, snapshot=snapshot):
            return False

        if self.liquidity_config.require_futures_market_type:
            if not self._is_futures_market_type(getattr(snapshot, "market_type", "")):
                return False

        if self._snapshot_is_stale(context, snapshot):
            return False

        snapshot_confidence = self._snapshot_confidence(snapshot)
        if snapshot_confidence < self.liquidity_config.min_snapshot_confidence:
            return False

        liquidity_strength = self._snapshot_liquidity_strength(snapshot)
        if liquidity_strength < self.liquidity_config.min_liquidity_strength:
            return False

        return True

    def _snapshot_matches_context(
        self,
        *,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy._snapshot_matches_context")
        context_symbol = str(context.symbol or "").strip().upper()
        snapshot_symbol = str(getattr(snapshot, "symbol", "") or "").strip().upper()

        if context_symbol and snapshot_symbol and context_symbol != snapshot_symbol:
            return False

        context_timeframe = self._timeframe_value(context.timeframe)
        snapshot_timeframe = str(getattr(snapshot, "timeframe", "") or "").strip().lower()

        if context_timeframe and snapshot_timeframe and context_timeframe != snapshot_timeframe:
            return False

        metadata = dict(context.metadata or {})
        context_exchange = str(metadata.get("exchange") or "").strip().lower()
        snapshot_exchange = str(getattr(snapshot, "exchange", "") or "").strip().lower()

        if context_exchange and snapshot_exchange and context_exchange != snapshot_exchange:
            return False

        context_market_type = str(metadata.get("market_type") or "").strip().lower()
        snapshot_market_type = str(getattr(snapshot, "market_type", "") or "").strip().lower()

        if context_market_type and snapshot_market_type and context_market_type != snapshot_market_type:
            return False

        return True

    def _snapshot_is_stale(
        self,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy._snapshot_is_stale")
        threshold = self.liquidity_config.max_snapshot_age_seconds
        if threshold is None:
            return False

        timestamp = parse_datetime(getattr(snapshot, "timestamp", None))
        if timestamp is None:
            return False

        context_ts = ensure_aware_utc(context.timestamp or utcnow())
        age = max(0.0, (context_ts - timestamp).total_seconds())
        return age > threshold

    @staticmethod
    def _is_futures_market_type(value: Any) -> bool:
        _strategy_logger = logging.getLogger(__name__ + ".LiquidityTradingStrategy._is_futures_market_type")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy._is_futures_market_type")
        return _normalize_label(value) in FUTURES_MARKET_TYPES

    @staticmethod
    def _timeframe_value(value: Any) -> str:
        _strategy_logger = logging.getLogger(__name__ + ".LiquidityTradingStrategy._timeframe_value")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy._timeframe_value")
        if isinstance(value, Enum):
            value = value.value
        return str(value or "").strip().lower()

    def _snapshot_confidence(self, snapshot: LiquidityMapSnapshot) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy._snapshot_confidence")
        signal = getattr(snapshot, "signal", None)
        if signal is not None:
            return _clamp01(getattr(signal, "confidence", 0.0))

        metadata = getattr(snapshot, "metadata", None)
        return _clamp01(_get_path(metadata, "confidence", default=0.0))

    def _snapshot_liquidity_strength(self, snapshot: LiquidityMapSnapshot) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy._snapshot_liquidity_strength")
        above = _clamp01(getattr(snapshot, "above_liquidity_score", 0.0))
        below = _clamp01(getattr(snapshot, "below_liquidity_score", 0.0))
        pressure = abs(_clamp_signed(getattr(snapshot, "liquidity_pressure_score", 0.0)))
        return _clamp01(max(above, below, pressure))

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def run_common_pre_filters(
        self,
        *,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot,
        current_price: float | None,
    ) -> list[FilterResult]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.run_common_pre_filters")
        results: list[FilterResult] = []

        if self._snapshot_matches_context(context=context, snapshot=snapshot):
            results.append(
                FilterResult(
                    name="liquidity_scope_filter",
                    decision=FilterDecision.PASS,
                    reason="Liquidity snapshot scope matches StrategyContext",
                )
            )
        else:
            results.append(
                FilterResult(
                    name="liquidity_scope_filter",
                    decision=FilterDecision.BLOCK,
                    reason="Liquidity snapshot scope does not match StrategyContext",
                )
            )

        if not self.liquidity_config.require_futures_market_type or self._is_futures_market_type(
            getattr(snapshot, "market_type", "")
        ):
            results.append(
                FilterResult(
                    name="futures_market_filter",
                    decision=FilterDecision.PASS,
                    reason=f"Accepted market_type: {getattr(snapshot, 'market_type', None)}",
                )
            )
        else:
            results.append(
                FilterResult(
                    name="futures_market_filter",
                    decision=FilterDecision.BLOCK,
                    reason=f"Non-futures market_type rejected: {getattr(snapshot, 'market_type', None)}",
                )
            )

        if current_price is not None and current_price > 0:
            results.append(
                FilterResult(
                    name="price_validation_filter",
                    decision=FilterDecision.PASS,
                    reason=f"Current price accepted: {current_price}",
                )
            )
        else:
            results.append(
                FilterResult(
                    name="price_validation_filter",
                    decision=FilterDecision.BLOCK,
                    reason="Current price is missing or invalid",
                )
            )

        if self._snapshot_is_stale(context, snapshot):
            results.append(
                FilterResult(
                    name="liquidity_snapshot_freshness",
                    decision=FilterDecision.BLOCK,
                    reason="Liquidity snapshot is stale",
                )
            )
        else:
            results.append(
                FilterResult(
                    name="liquidity_snapshot_freshness",
                    decision=FilterDecision.PASS,
                    reason="Liquidity snapshot is fresh",
                )
            )

        for item in results:
            item.validate()

        return results

    # ------------------------------------------------------------------
    # Scope / metadata
    # ------------------------------------------------------------------

    def liquidity_scope(
        self,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot | None = None,
    ) -> LiquidityStrategyScope:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.liquidity_scope")
        self.validate_context(context)

        metadata = dict(context.metadata or {})
        domain = self.liquidity_domain(context)

        exchange = (
            metadata.get("exchange")
            or domain.get("exchange")
            or getattr(snapshot, "exchange", None)
            or "unknown"
        )
        market_type = (
            metadata.get("market_type")
            or domain.get("market_type")
            or getattr(snapshot, "market_type", None)
            or self.liquidity_config.default_market_type.value
        )
        timeframe = (
            metadata.get("timeframe")
            or domain.get("timeframe")
            or getattr(snapshot, "timeframe", None)
            or context.timeframe.value
        )
        exchange_symbol = (
            metadata.get("exchange_symbol")
            or domain.get("exchange_symbol")
            or getattr(snapshot, "exchange_symbol", None)
            or context.symbol
        )

        return LiquidityStrategyScope(
            exchange=str(_value(exchange)),
            market_type=str(_value(market_type)),
            symbol=context.symbol,
            timeframe=str(_value(timeframe)),
            exchange_symbol=str(_value(exchange_symbol)),
        )

    def liquidity_context_metadata(
        self,
        *,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot | None = None,
        current_price: float | None = None,
        source_features: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.liquidity_context_metadata")
        metadata: dict[str, Any] = {
            "feature_source": FeatureSource.LIQUIDITY.value,
            "strategy_category": StrategyCategory.LIQUIDITY.value,
        }

        if self.liquidity_config.attach_scope_metadata:
            metadata["liquidity_scope"] = self.liquidity_scope(context, snapshot).to_dict()

        if current_price is not None:
            metadata["current_price"] = current_price

        if self.liquidity_config.attach_snapshot_metadata and snapshot is not None:
            metadata["liquidity_snapshot"] = self._snapshot_metadata(
                snapshot=snapshot,
                current_price=current_price,
            )

        if self.liquidity_config.attach_feature_values_metadata:
            metadata["liquidity_features"] = self._selected_feature_values(
                context=context,
                source_features=source_features or [],
            )

        if self.liquidity_config.attach_liquidity_context_metadata:
            metadata["liquidity_context_keys"] = sorted(self.liquidity_domain(context).keys())

        metadata.update(dict(self.liquidity_config.metadata))

        if extra:
            metadata.update(extra)

        return metadata

    def _selected_feature_values(
        self,
        *,
        context: StrategyContext,
        source_features: list[str],
    ) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy._selected_feature_values")
        result: dict[str, Any] = {}

        for feature in source_features:
            if not isinstance(feature, str) or not feature.strip():
                continue

            if context.has_feature(feature):
                result[feature] = serialize_for_metadata(context.get_feature(feature))
                continue

            value = self.liquidity_path(context, feature, default=None)
            if value is not None:
                result[feature] = serialize_for_metadata(value)

        return result

    def _snapshot_metadata(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        current_price: float | None = None,
    ) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy._snapshot_metadata")
        metadata: dict[str, Any] = {
            "exchange": getattr(snapshot, "exchange", None),
            "market_type": getattr(snapshot, "market_type", None),
            "symbol": getattr(snapshot, "symbol", None),
            "timeframe": getattr(snapshot, "timeframe", None),
            "timestamp": serialize_for_metadata(getattr(snapshot, "timestamp", None)),
            "bias": serialize_for_metadata(getattr(snapshot, "bias", None)),
            "above_liquidity_score": _clamp01(getattr(snapshot, "above_liquidity_score", 0.0)),
            "below_liquidity_score": _clamp01(getattr(snapshot, "below_liquidity_score", 0.0)),
            "liquidity_pressure_score": _clamp_signed(getattr(snapshot, "liquidity_pressure_score", 0.0)),
            "active_levels_count": len(list(getattr(snapshot, "active_levels", []) or [])),
            "equal_levels_count": len(list(getattr(snapshot, "equal_levels", []) or [])),
            "stop_clusters_count": len(list(getattr(snapshot, "stop_clusters", []) or [])),
            "zones_count": len(list(getattr(snapshot, "zones", []) or [])),
        }

        if current_price is not None:
            metadata["current_price"] = current_price

        return metadata

    # ------------------------------------------------------------------
    # Price / distance / target helpers
    # ------------------------------------------------------------------

    @staticmethod
    def reference_price(item: Any) -> float:
        _strategy_logger = logging.getLogger(__name__ + ".LiquidityTradingStrategy.reference_price")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.reference_price")
        for attr in (
            "price",
            "center_price",
            "level_price",
            "reference_price",
            "mid_price",
        ):
            value = _to_float(getattr(item, attr, None))
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
                value = _to_float(item.get(key))
                if value is not None and value > 0:
                    return value

        return 0.0

    @staticmethod
    def distance_pct(price: float, current_price: float) -> float:
        _strategy_logger = logging.getLogger(__name__ + ".LiquidityTradingStrategy.distance_pct")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.distance_pct")
        if price <= 0 or current_price <= 0:
            return float("inf")
        return abs(price - current_price) / current_price

    def distance_score(
        self,
        *,
        price: float,
        current_price: float,
        max_distance_pct: float,
        min_distance_pct: float = 0.0,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.distance_score")
        if current_price <= 0 or price <= 0 or max_distance_pct <= 0:
            return 0.0

        distance = self.distance_pct(price, current_price)

        if distance < min_distance_pct:
            return 0.0

        if distance > max_distance_pct:
            return 0.0

        return _clamp01(1.0 - distance / max_distance_pct)

    def is_above_price(self, item: Any, current_price: float) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.is_above_price")
        return self.reference_price(item) > current_price

    def is_below_price(self, item: Any, current_price: float) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.is_below_price")
        return self.reference_price(item) < current_price

    def dedupe_liquidity_items(
        self,
        items: Sequence[LiquidityLevel | StopCluster | LiquidityZone],
    ) -> list[LiquidityLevel | StopCluster | LiquidityZone]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.dedupe_liquidity_items")
        result: dict[str, LiquidityLevel | StopCluster | LiquidityZone] = {}

        for item in items:
            key = str(
                getattr(item, "key", None)
                or getattr(item, "id", None)
                or f"{item.__class__.__name__}:{self.reference_price(item):.12f}"
            )
            existing = result.get(key)
            if existing is None:
                result[key] = item
                continue

            if self.item_strength(item) > self.item_strength(existing):
                result[key] = item

        return list(result.values())

    def item_strength(self, item: Any) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.item_strength")
        candidates = [
            getattr(item, "score", None),
            getattr(item, "strength", None),
            getattr(item, "confidence", None),
            getattr(item, "volume_score", None),
            getattr(item, "notional_score", None),
        ]

        for candidate in candidates:
            value = _to_float(candidate)
            if value is not None:
                return _clamp01(value)

        notional = safe_decimal(getattr(item, "total_notional", None))
        if notional <= 0:
            notional = safe_decimal(getattr(item, "notional", None))
        if notional <= 0:
            notional = safe_decimal(getattr(item, "total_notional_usd", None))

        if notional > 0:
            return _clamp01(float(min(notional / Decimal("1000000"), Decimal("1"))))

        return 0.0

    def is_terminal_item(self, item: Any) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.is_terminal_item")
        status = _normalize_label(getattr(item, "sweep_status", None))
        if status in {"swept", "partially_swept", "invalidated", "expired"}:
            return True

        for method_name in ("is_swept", "is_partially_swept", "is_invalidated", "is_expired"):
            method = getattr(item, method_name, None)
            if callable(method):
                try:
                    if bool(method()):
                        return True
                except Exception:
                    continue

        return False

    def is_active_item(self, item: Any) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.is_active_item")
        if self.is_terminal_item(item):
            return False

        method = getattr(item, "is_active", None)
        if callable(method):
            try:
                return bool(method())
            except Exception:
                return True

        return True

    # ------------------------------------------------------------------
    # Level / cluster / zone selection
    # ------------------------------------------------------------------

    def active_levels(self, snapshot: LiquidityMapSnapshot) -> list[LiquidityLevel]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.active_levels")
        levels = list(getattr(snapshot, "active_levels", []) or [])
        return [level for level in levels if self.is_active_item(level)]

    def equal_levels(self, snapshot: LiquidityMapSnapshot) -> list[LiquidityLevel]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.equal_levels")
        levels = list(getattr(snapshot, "equal_levels", []) or [])
        return [level for level in levels if self.is_active_item(level)]

    def stop_clusters(self, snapshot: LiquidityMapSnapshot) -> list[StopCluster]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.stop_clusters")
        return list(getattr(snapshot, "stop_clusters", []) or [])

    def zones(self, snapshot: LiquidityMapSnapshot) -> list[LiquidityZone]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.zones")
        return list(getattr(snapshot, "zones", []) or [])

    def directional_levels(
        self,
        snapshot: LiquidityMapSnapshot,
        side: LiquiditySide,
    ) -> list[LiquidityLevel]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.directional_levels")
        return [
            level
            for level in [*self.active_levels(snapshot), *self.equal_levels(snapshot)]
            if getattr(level, "side", None) == side
        ]

    def directional_clusters(
        self,
        snapshot: LiquidityMapSnapshot,
        side: LiquiditySide,
    ) -> list[StopCluster]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.directional_clusters")
        return [
            cluster
            for cluster in self.stop_clusters(snapshot)
            if getattr(cluster, "side", None) == side
        ]

    def directional_zones(
        self,
        snapshot: LiquidityMapSnapshot,
        side: LiquiditySide,
    ) -> list[LiquidityZone]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.directional_zones")
        return [
            zone
            for zone in self.zones(snapshot)
            if getattr(zone, "side", None) == side
        ]

    def collect_targets_above(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> list[LiquidityLevel | StopCluster]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.collect_targets_above")
        candidates: list[LiquidityLevel | StopCluster] = []

        nearest = getattr(snapshot, "nearest_above_level", None)
        strongest = getattr(snapshot, "strongest_cluster_above", None)

        for item in (nearest, strongest):
            if item is not None and self.reference_price(item) > current_price:
                candidates.append(item)

        candidates.extend(
            level
            for level in self.active_levels(snapshot)
            if self.reference_price(level) > current_price
        )
        candidates.extend(
            cluster
            for cluster in self.stop_clusters(snapshot)
            if self.reference_price(cluster) > current_price
        )

        deduped = self.dedupe_liquidity_items(candidates)
        return sorted(deduped, key=self.reference_price)

    def collect_targets_below(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> list[LiquidityLevel | StopCluster]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.collect_targets_below")
        candidates: list[LiquidityLevel | StopCluster] = []

        nearest = getattr(snapshot, "nearest_below_level", None)
        strongest = getattr(snapshot, "strongest_cluster_below", None)

        for item in (nearest, strongest):
            if item is not None and self.reference_price(item) < current_price:
                candidates.append(item)

        candidates.extend(
            level
            for level in self.active_levels(snapshot)
            if self.reference_price(level) < current_price
        )
        candidates.extend(
            cluster
            for cluster in self.stop_clusters(snapshot)
            if self.reference_price(cluster) < current_price
        )

        deduped = self.dedupe_liquidity_items(candidates)
        return sorted(deduped, key=self.reference_price, reverse=True)

    def best_zone_for_side(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        side: LiquiditySide,
        current_price: float,
    ) -> LiquidityZone | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.best_zone_for_side")
        candidates = [
            zone
            for zone in self.directional_zones(snapshot, side)
            if self.reference_price(zone) > 0
        ]

        if not candidates:
            return None

        def rank(zone: LiquidityZone) -> tuple[float, float]:
            distance = self.distance_pct(self.reference_price(zone), current_price)
            return (_clamp01(getattr(zone, "score", 0.0)), -distance)

        return max(candidates, key=rank)

    # ------------------------------------------------------------------
    # Liquidity map scores
    # ------------------------------------------------------------------

    def magnet_score_up(self, snapshot: LiquidityMapSnapshot) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.magnet_score_up")
        candidates = [
            getattr(snapshot, "upside_magnet_score", None),
            _get_path(getattr(snapshot, "metadata", None), "upside_magnet_score"),
            getattr(snapshot, "above_liquidity_score", None),
        ]
        return max(_clamp01(item) for item in candidates if item is not None) if any(
            item is not None for item in candidates
        ) else 0.0

    def magnet_score_down(self, snapshot: LiquidityMapSnapshot) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.magnet_score_down")
        candidates = [
            getattr(snapshot, "downside_magnet_score", None),
            _get_path(getattr(snapshot, "metadata", None), "downside_magnet_score"),
            getattr(snapshot, "below_liquidity_score", None),
        ]
        return max(_clamp01(item) for item in candidates if item is not None) if any(
            item is not None for item in candidates
        ) else 0.0

    def sweep_risk_up(self, snapshot: LiquidityMapSnapshot) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.sweep_risk_up")
        candidates = [
            getattr(snapshot, "upside_sweep_risk", None),
            getattr(snapshot, "sweep_risk_up", None),
            _get_path(getattr(snapshot, "metadata", None), "upside_sweep_risk"),
        ]
        return max(_clamp01(item) for item in candidates if item is not None) if any(
            item is not None for item in candidates
        ) else 0.0

    def sweep_risk_down(self, snapshot: LiquidityMapSnapshot) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.sweep_risk_down")
        candidates = [
            getattr(snapshot, "downside_sweep_risk", None),
            getattr(snapshot, "sweep_risk_down", None),
            _get_path(getattr(snapshot, "metadata", None), "downside_sweep_risk"),
        ]
        return max(_clamp01(item) for item in candidates if item is not None) if any(
            item is not None for item in candidates
        ) else 0.0

    def zone_score(
        self,
        snapshot: LiquidityMapSnapshot,
        side: LiquiditySide,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.zone_score")
        zones = self.directional_zones(snapshot, side)
        if not zones:
            return 0.0
        return max(_clamp01(getattr(zone, "score", 0.0)) for zone in zones)

    # ------------------------------------------------------------------
    # Plans / signal building
    # ------------------------------------------------------------------

    def build_basic_liquidity_plans(
        self,
        *,
        context: StrategyContext,
        side: SignalSide,
        entry_price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        target_levels: list[TargetPlan] | None = None,
        invalidation_reason: str | None = None,
    ) -> tuple[EntryPlan, ExitPlan, InvalidationPlan]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.build_basic_liquidity_plans")
        self.validate_context(context)

        if side not in {SignalSide.LONG, SignalSide.SHORT}:
            raise StrategyEvaluationError("liquidity trade plans require directional side")

        entry = EntryPlan(
            entry_type=self.liquidity_config.default_entry_type,
            price=entry_price,
            timeout_seconds=self.liquidity_config.entry_timeout_seconds,
            max_slippage_bps=self.liquidity_config.max_slippage_bps,
            confirmation_required=False,
            notes=["liquidity_strategy_entry_draft"],
            metadata={
                "source": self.strategy_name,
                "feature_source": FeatureSource.LIQUIDITY.value,
            },
        )

        targets = list(target_levels or [])
        if take_profit is not None:
            targets.append(
                TargetPlan(
                    price=take_profit,
                    size_fraction=1.0,
                    label="liquidity_take_profit",
                )
            )

        exit_plan = ExitPlan(
            exit_types=list(self.liquidity_config.default_exit_types),
            stop_loss=stop_loss,
            take_profit_levels=targets,
            max_holding_seconds=self.liquidity_config.max_holding_seconds,
            metadata={
                "source": self.strategy_name,
                "feature_source": FeatureSource.LIQUIDITY.value,
            },
        )

        invalidation = InvalidationPlan(
            price=stop_loss,
            reason=invalidation_reason or "liquidity_context_invalidated",
            timeout_seconds=self.liquidity_config.max_holding_seconds,
            conditions=[
                "liquidity_bias_flipped",
                "target_liquidity_removed",
                "market_structure_invalidated",
            ],
            metadata={
                "source": self.strategy_name,
                "feature_source": FeatureSource.LIQUIDITY.value,
            },
        )

        entry.validate()
        exit_plan.validate()
        invalidation.validate()
        return entry, exit_plan, invalidation

    def build_liquidity_trade_metadata(
        self,
        *,
        context: StrategyContext,
        side: SignalSide,
        setup_quality: float,
        snapshot: LiquidityMapSnapshot | None = None,
        current_price: float | None = None,
        confluence_score: float = 0.0,
        liquidity_score: float = 0.5,
        risk_reward_score: float = 0.0,
        execution_quality_score: float = 0.5,
        regime_alignment_score: float = 0.0,
        freshness_score: float = 1.0,
        source_features: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.build_liquidity_trade_metadata")
        priority = self.build_priority_metadata(
            setup_quality=setup_quality,
            confluence_score=confluence_score,
            liquidity_score=liquidity_score,
            risk_reward_score=risk_reward_score,
            execution_quality_score=execution_quality_score,
            regime_alignment_score=regime_alignment_score,
            freshness_score=freshness_score,
        )

        scope = self.liquidity_scope(context, snapshot)

        metadata = self.build_trade_metadata(
            tier=self.liquidity_config.default_trade_tier,
            order_intent=self.liquidity_config.default_order_intent,
            margin_mode=self.liquidity_config.default_margin_mode,
            market_type=self.liquidity_config.default_market_type,
            requested_leverage=self.liquidity_config.requested_leverage,
            exchange=scope.exchange,
            extra={
                **priority,
                "liquidity_side": side.value,
                "liquidity_score": _clamp01(liquidity_score),
                "execution_quality_score": _clamp01(execution_quality_score),
                "regime_alignment_score": _clamp01(regime_alignment_score),
                "freshness_score": _clamp01(freshness_score),
                **self.liquidity_context_metadata(
                    context=context,
                    snapshot=snapshot,
                    current_price=current_price,
                    source_features=source_features,
                ),
            },
        )

        if extra:
            metadata.update(extra)

        return metadata

    def build_liquidity_signal(
        self,
        *,
        context: StrategyContext,
        side: SignalSide,
        confidence: float,
        score: float,
        setup_type: SetupType | None = None,
        reasons: list[str] | None = None,
        confirmations: list[str] | None = None,
        source_features: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        priority: SignalPriority = SignalPriority.MEDIUM,
        snapshot: LiquidityMapSnapshot | None = None,
        current_price: float | None = None,
        entry_price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> StrategySignal:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.build_liquidity_signal")
        if side not in {SignalSide.LONG, SignalSide.SHORT}:
            raise StrategyEvaluationError(
                f"{self.strategy_name}: liquidity signal side must be LONG or SHORT"
            )

        final_source_features = list(source_features or [])
        signal_metadata = self.build_liquidity_trade_metadata(
            context=context,
            side=side,
            setup_quality=score,
            snapshot=snapshot,
            current_price=current_price,
            confluence_score=confidence,
            risk_reward_score=0.0,
            source_features=final_source_features,
            extra=metadata,
        )

        signal = self.build_directional_signal(
            context=context,
            side=side,
            confidence=confidence,
            score=score,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            setup_type=setup_type or self.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=final_source_features,
            metadata=signal_metadata,
            priority=priority,
            tier=self.liquidity_config.default_trade_tier,
            order_intent=self.liquidity_config.default_order_intent,
            requested_leverage=self.liquidity_config.requested_leverage,
            margin_mode=self.liquidity_config.default_margin_mode,
            liquidity_class=None,
            execution_quality=None,
            market_type=self.liquidity_config.default_market_type,
        )

        signal.metadata.setdefault("feature_source", FeatureSource.LIQUIDITY.value)
        signal.metadata.setdefault("liquidity_strategy_base", self.__class__.__name__)
        signal.validate()
        return signal

    # ------------------------------------------------------------------
    # Applicability
    # ------------------------------------------------------------------

    def validate_context_requirements(self, context: StrategyContext) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.validate_context_requirements")
        super().validate_context_requirements(context)

        snapshot = self.liquidity_snapshot(context)
        domain = self.liquidity_domain(context)

        has_liquidity_feature = any(
            context.has_feature(feature)
            for feature in LiquidityFeatureNames.all()
        )

        if snapshot is None and not domain and not has_liquidity_feature:
            raise StrategyEvaluationError(
                f"{self.strategy_name}: missing liquidity domain data for {context.symbol}"
            )

    def supports_regime(self, regime: MarketRegime) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidityTradingStrategy.supports_regime")
        return super().supports_regime(regime)


# Temporary migration aliases.
# Prefer importing LiquidityTradingStrategy in new code.
BaseLiquidityStrategy = LiquidityTradingStrategy


__all__ = [
    "FUTURES_MARKET_TYPES",
    "LIQUIDITY_DOMAIN_ALIASES",
    "LIQUIDITY_FEATURES",
    "SNAPSHOT_FEATURE_KEYS",
    "BaseLiquidityStrategy",
    "LiquidityFeatureNames",
    "LiquidityStrategyConfig",
    "LiquidityStrategyScope",
    "LiquidityTradingStrategy",
    "ensure_utc",
    "parse_datetime",
    "safe_decimal",
    "serialize_for_metadata",
    "utc_now",
]