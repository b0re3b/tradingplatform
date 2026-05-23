# trading_system/strategy/strategies/liquidations/base.py

from __future__ import annotations
import logging

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from core.event_bus import EventBus
from core.scheduler import Scheduler

from ...config import StrategyConfig, StrategyDefinitionConfig
from ...enums import (
    EntryType,
    ExitType,
    FeatureSource,
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
    InvalidationPlan,
    StrategyContext,
    StrategySignal,
    TargetPlan,
    clamp,
    ensure_aware_utc,
    utcnow,
)
from ..base_strategy import TradingStrategy


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
    """
    Parse timestamps used inside normalized liquidation domain payloads.

    Concrete strategies should normally receive StrategyContext.timestamp, but
    nested analytics payloads may still contain detected_at / event_time fields.
    """
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
    Serialize nested analytics values for StrategySignal.metadata.

    This is not a RiskReadySignalPayload builder. SignalProcessor / SignalBuilder
    owns final payload conversion.
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
    Read dotted path from dict-like or object-like nested data.

    Examples:
        cascade.confidence
        exhaustion.exhaustion_bias
        squeeze.status
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

        current = _get_attr_or_key(current, part, default=None)

    return default if current is None else current


def _first_present(
    value: Any,
    paths: tuple[str, ...],
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
        return float(value)

    if isinstance(value, Enum):
        return _to_float(value.value, default)

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return default

        try:
            return float(raw)
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

        if normalized in {"1", "true", "yes", "y", "on", "confirmed"}:
            return True

        if normalized in {"0", "false", "no", "n", "off", "rejected"}:
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


def _unit_score(value: Any, default: float = 0.0) -> float:
    parsed = _to_float(value, default)
    return clamp(float(parsed if parsed is not None else default), 0.0, 1.0)


def _signed_score(value: Any, default: float = 0.0) -> float:
    parsed = _to_float(value, default)
    return clamp(float(parsed if parsed is not None else default), -1.0, 1.0)


def _weighted_score(
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
        total_value += _unit_score(values.get(key, default), default) * weight_f

    if total_weight <= 0.0:
        return _unit_score(default)

    return _unit_score(total_value / total_weight)


# =============================================================================
# Liquidations feature contract
# =============================================================================


@dataclass(frozen=True, slots=True)
class LiquidationsFeatureNames:
    """
    Stable feature names expected in StrategyContext.

    Generic SignalNormalizer may create these names from analytics.liquidations.*
    payloads, or strategies can read equivalent values from domain_data aliases.
    """
    _logger = logging.getLogger(__name__ + ".LiquidationsFeatureNames")

    CASCADE: str = "liquidations.cascade"
    CASCADE_CONFIDENCE: str = "liquidations.cascade.confidence"
    CASCADE_INTENSITY: str = "liquidations.cascade.intensity_score"
    CASCADE_DIRECTION: str = "liquidations.cascade.direction"
    CASCADE_SEVERITY: str = "liquidations.cascade.severity"
    CASCADE_CONTINUATION_BIAS: str = "liquidations.cascade.continuation_bias"
    CASCADE_EXHAUSTION_BIAS: str = "liquidations.cascade.exhaustion_bias"
    CASCADE_NOTIONAL_USD: str = "liquidations.cascade.total_notional_usd"
    CASCADE_EVENT_COUNT: str = "liquidations.cascade.event_count"

    EXHAUSTION: str = "liquidations.exhaustion"
    EXHAUSTION_CONFIDENCE: str = "liquidations.exhaustion.confidence"
    EXHAUSTION_BIAS: str = "liquidations.exhaustion.exhaustion_bias"
    EXHAUSTION_BIAS_DELTA: str = "liquidations.exhaustion.bias_delta"
    EXHAUSTION_CONFIRMED: str = "liquidations.exhaustion.confirmed"

    SQUEEZE: str = "liquidations.squeeze"
    SQUEEZE_CONFIRMED: str = "liquidations.squeeze.confirmed"
    SQUEEZE_SCORE: str = "liquidations.squeeze.score"
    SQUEEZE_DIRECTION: str = "liquidations.squeeze.direction"

    CLUSTER: str = "liquidations.cluster"
    CLUSTER_DURATION_SECONDS: str = "liquidations.cluster.duration_seconds"
    CLUSTER_AVG_NOTIONAL_PER_EVENT: str = "liquidations.cluster.avg_notional_per_event"
    CLUSTER_SIDE_IMBALANCE_RATIO: str = "liquidations.cluster.side_imbalance_ratio"
    CLUSTER_EVENT_IMBALANCE_RATIO: str = "liquidations.cluster.event_imbalance_ratio"
    CLUSTER_ACCELERATION_RATIO: str = "liquidations.cluster.acceleration_ratio"

    @classmethod
    def all(cls) -> set[str]:
        _strategy_logger = getattr(cls, "_logger", None) or logging.getLogger(__name__ + ".LiquidationsFeatureNames")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsFeatureNames.all")
        instance = cls()
        return {
            getattr(instance, item.name)
            for item in fields(cls)
            if isinstance(getattr(instance, item.name), str)
            and getattr(instance, item.name).strip()
        }


LIQUIDATIONS_FEATURES = LiquidationsFeatureNames()


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


# =============================================================================
# Scope
# =============================================================================


@dataclass(frozen=True, slots=True)
class LiquidationsStrategyScope:
    """
    Futures liquidation scope used only for metadata and normalization.

    Concrete strategies still make decisions from StrategyContext.
    """
    _logger = logging.getLogger(__name__ + ".LiquidationsStrategyScope")

    exchange: str
    market_type: str
    symbol: str
    timeframe: str
    exchange_symbol: str | None = None

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsStrategyScope.__post_init__")
        exchange = str(self.exchange or "unknown").strip().lower()
        market_type = str(
            self.market_type or StrategyMarketType.USDM_FUTURES.value
        ).strip()
        symbol = str(self.symbol or "").strip().upper()
        timeframe = str(self.timeframe or Timeframe.M1.value).strip().lower()
        exchange_symbol = str(self.exchange_symbol or symbol).strip().upper()

        if not symbol:
            raise StrategyEvaluationError(
                "LiquidationsStrategyScope.symbol cannot be empty"
            )

        object.__setattr__(self, "exchange", exchange)
        object.__setattr__(self, "market_type", market_type)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timeframe", timeframe)
        object.__setattr__(self, "exchange_symbol", exchange_symbol)

    @property
    def key(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsStrategyScope.key")
        return f"{self.exchange}:{self.market_type}:{self.symbol}:{self.timeframe}"

    @property
    def legacy_key(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsStrategyScope.legacy_key")
        return f"{self.symbol}:{self.exchange}"

    def to_dict(self) -> dict[str, str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsStrategyScope.to_dict")
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "exchange_symbol": self.exchange_symbol or self.symbol,
            "key": self.key,
            "legacy_key": self.legacy_key,
        }


# =============================================================================
# Config
# =============================================================================


@dataclass(slots=True)
class LiquidationsStrategyConfig:
    """
    Domain config shared by concrete liquidation strategies.

    Runtime enabled/symbol/timeframe/regime checks belong to StrategyConfig /
    StrategyDefinitionConfig. This config keeps liquidation-specific defaults and
    quality thresholds.
    """
    _logger = logging.getLogger(__name__ + ".LiquidationsStrategyConfig")

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
    min_signal_confidence: float = 0.50
    min_signal_score: float = 0.35

    min_intensity_score: float = 0.0
    min_total_notional_usd: Decimal = Decimal("0")
    min_event_count: int = 0
    max_price_range_pct: float | None = None

    requested_leverage: float | None = None
    max_slippage_bps: float | None = None
    entry_timeout_seconds: int | None = None
    max_holding_seconds: int | None = None

    stale_feature_max_age_seconds: float | None = None

    attach_liquidations_context_metadata: bool = True
    attach_scope_metadata: bool = True
    attach_feature_values_metadata: bool = True

    tag_liquidations: str = "liquidations"
    tag_cascade: str = "liquidation_cascade"
    tag_exhaustion: str = "liquidation_exhaustion"
    tag_squeeze: str = "liquidation_squeeze"
    tag_reversal: str = "reversal"
    tag_continuation: str = "continuation"

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsStrategyConfig.validate")
        bounded = {
            "min_context_confidence": self.min_context_confidence,
            "min_signal_confidence": self.min_signal_confidence,
            "min_signal_score": self.min_signal_score,
            "min_intensity_score": self.min_intensity_score,
        }

        for name, value in bounded.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{name} must be between 0.0 and 1.0")

        if self.min_total_notional_usd < 0:
            raise StrategyConfigError("min_total_notional_usd must be >= 0")

        if self.min_event_count < 0:
            raise StrategyConfigError("min_event_count must be >= 0")

        if self.max_price_range_pct is not None and self.max_price_range_pct < 0:
            raise StrategyConfigError("max_price_range_pct must be >= 0")

        if self.requested_leverage is not None and self.requested_leverage <= 0:
            raise StrategyConfigError("requested_leverage must be > 0")

        if self.max_slippage_bps is not None and self.max_slippage_bps < 0:
            raise StrategyConfigError("max_slippage_bps must be >= 0")

        if self.entry_timeout_seconds is not None and self.entry_timeout_seconds <= 0:
            raise StrategyConfigError("entry_timeout_seconds must be > 0")

        if self.max_holding_seconds is not None and self.max_holding_seconds <= 0:
            raise StrategyConfigError("max_holding_seconds must be > 0")

        if (
            self.stale_feature_max_age_seconds is not None
            and self.stale_feature_max_age_seconds <= 0
        ):
            raise StrategyConfigError("stale_feature_max_age_seconds must be > 0")

        for attr in (
            "tag_liquidations",
            "tag_cascade",
            "tag_exhaustion",
            "tag_squeeze",
            "tag_reversal",
            "tag_continuation",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")


# =============================================================================
# Base liquidation strategy
# =============================================================================


class LiquidationsTradingStrategy(TradingStrategy):
    """
    Base class for concrete strategy/strategies/liquidations/* classes.

    Responsibilities:
    - read liquidation analytics data from StrategyContext only;
    - provide helper methods for liquidations domain extraction and scoring;
    - build internal StrategySignal objects through TradingStrategy helpers;
    - attach futures/liquidations metadata for SignalProcessor.

    Forbidden:
    - no direct analytics.liquidations.* EventBus subscriptions;
    - no local signal/rejection state machine;
    - no diagnostics scheduler jobs;
    - no EventBus emit of signal.generated;
    - no RiskManager / Execution calls;
    - no raw market data reads.
    """
    _logger = logging.getLogger(__name__ + ".LiquidationsTradingStrategy")

    component_namespace = "strategy.liquidations"
    category: StrategyCategory = StrategyCategory.LIQUIDATIONS
    default_setup_type: SetupType = SetupType.UNKNOWN
    default_timeframe: Timeframe = Timeframe.M1

    feature_names = LIQUIDATIONS_FEATURES

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        liquidations_config: LiquidationsStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.__init__")
        self.liquidations_config = liquidations_config or LiquidationsStrategyConfig()
        self.liquidations_config.validate()

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
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.validate_config")
        super().validate_config()
        self.liquidations_config.validate()

    # ------------------------------------------------------------------
    # Context / domain access
    # ------------------------------------------------------------------

    def liquidations_domain(self, context: StrategyContext) -> dict[str, Any]:
        """
        Return liquidation domain data from StrategyContext.

        Generic SignalNormalizer / StrategyContextBuilder should populate this
        from analytics.liquidations.* payloads.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.liquidations_domain")
        self.validate_context(context)
        domain = context.domain_dict(FeatureSource.LIQUIDATIONS)
        return dict(domain)

    def liquidations_item(
        self,
        context: StrategyContext,
        key: str,
        default: Any = None,
    ) -> Any:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.liquidations_item")
        domain = self.liquidations_domain(context)

        if key in domain:
            return domain[key]

        for alias in LIQUIDATIONS_DOMAIN_ALIASES.get(key, ()):
            value = _get_path(domain, alias, default=None)
            if value is not None:
                return value

        return default

    def liquidations_path(
        self,
        context: StrategyContext,
        path: str,
        default: Any = None,
    ) -> Any:
        """
        Read liquidation value by dotted path.

        Priority:
        1. exact StrategyContext feature name;
        2. liquidations-prefixed feature name;
        3. liquidation domain dotted path.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.liquidations_path")
        self.validate_context(context)

        if not isinstance(path, str) or not path.strip():
            raise StrategyEvaluationError("liquidations path cannot be empty")

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

        domain = self.liquidations_domain(context)

        if normalized.startswith("liquidations."):
            normalized = normalized.removeprefix("liquidations.")

        return _get_path(domain, normalized, default)

    def liquidations_float(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float | None = None,
    ) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.liquidations_float")
        return _to_float(self.liquidations_path(context, path, default), default)

    def liquidations_score(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float = 0.0,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.liquidations_score")
        value = self.liquidations_float(context, path, default=default)
        return clamp(float(value if value is not None else default), 0.0, 1.0)

    def liquidations_signed_score(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float = 0.0,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.liquidations_signed_score")
        value = self.liquidations_float(context, path, default=default)
        return clamp(float(value if value is not None else default), -1.0, 1.0)

    def liquidations_bool(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: bool = False,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.liquidations_bool")
        return _to_bool(self.liquidations_path(context, path, default), default)

    def liquidations_str(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: str | None = None,
    ) -> str | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.liquidations_str")
        value = self.liquidations_path(context, path, default)

        if value is None:
            return default

        if isinstance(value, Enum):
            return str(value.value)

        text = str(value).strip()
        return text if text else default

    def liquidations_datetime(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: datetime | None = None,
    ) -> datetime | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.liquidations_datetime")
        return parse_datetime(self.liquidations_path(context, path, default))

    def liquidations_feature_snapshot(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> FeatureSnapshot | None:
        """
        Return full FeatureSnapshot if StrategyContext stores one.

        Best-effort helper because StrategyContext may store raw values or
        FeatureSnapshot objects depending on normalization.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.liquidations_feature_snapshot")
        self.validate_context(context)

        if not isinstance(feature_name, str) or not feature_name.strip():
            raise StrategyEvaluationError("feature_name cannot be empty")

        features_map = getattr(context, "features", None)
        if isinstance(features_map, Mapping):
            raw = features_map.get(feature_name)
            if isinstance(raw, FeatureSnapshot):
                return raw

        return None

    def liquidations_feature_age_seconds(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.liquidations_feature_age_seconds")
        snapshot = self.liquidations_feature_snapshot(context, feature_name)
        if snapshot is None:
            return None
        return snapshot.age_seconds(context.timestamp)

    def liquidations_feature_is_fresh(
        self,
        context: StrategyContext,
        feature_name: str,
        *,
        max_age_seconds: float | None = None,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.liquidations_feature_is_fresh")
        age = self.liquidations_feature_age_seconds(context, feature_name)
        if age is None:
            return True

        threshold = (
            max_age_seconds
            if max_age_seconds is not None
            else self.liquidations_config.stale_feature_max_age_seconds
        )

        if threshold is None:
            return True

        return age <= threshold

    # ------------------------------------------------------------------
    # Scope / metadata
    # ------------------------------------------------------------------

    def liquidations_scope(self, context: StrategyContext) -> LiquidationsStrategyScope:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.liquidations_scope")
        self.validate_context(context)

        metadata = dict(context.metadata or {})
        domain = self.liquidations_domain(context)

        exchange = (
            metadata.get("exchange")
            or domain.get("exchange")
            or self.liquidations_path(context, "cascade.exchange")
            or self.liquidations_path(context, "exhaustion.exchange")
            or self.liquidations_path(context, "squeeze.exchange")
            or "unknown"
        )
        market_type = (
            metadata.get("market_type")
            or domain.get("market_type")
            or self.liquidations_path(context, "cascade.market_type")
            or self.liquidations_path(context, "exhaustion.market_type")
            or self.liquidations_path(context, "squeeze.market_type")
            or self.liquidations_config.default_market_type.value
        )
        timeframe = (
            metadata.get("timeframe")
            or domain.get("timeframe")
            or self.liquidations_path(context, "cascade.timeframe")
            or self.liquidations_path(context, "exhaustion.timeframe")
            or self.liquidations_path(context, "squeeze.timeframe")
            or context.timeframe.value
        )
        exchange_symbol = (
            metadata.get("exchange_symbol")
            or domain.get("exchange_symbol")
            or self.liquidations_path(context, "cascade.exchange_symbol")
            or self.liquidations_path(context, "exhaustion.exchange_symbol")
            or self.liquidations_path(context, "squeeze.exchange_symbol")
            or context.symbol
        )

        return LiquidationsStrategyScope(
            exchange=str(exchange),
            market_type=str(
                market_type.value
                if isinstance(market_type, Enum)
                else market_type
            ),
            symbol=context.symbol,
            timeframe=str(
                timeframe.value
                if isinstance(timeframe, Enum)
                else timeframe
            ),
            exchange_symbol=str(exchange_symbol),
        )

    def liquidations_context_metadata(
        self,
        context: StrategyContext,
        *,
        source_features: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.liquidations_context_metadata")
        metadata: dict[str, Any] = {
            "feature_source": FeatureSource.LIQUIDATIONS.value,
            "strategy_category": StrategyCategory.LIQUIDATIONS.value,
        }

        if self.liquidations_config.attach_scope_metadata:
            metadata["liquidations_scope"] = self.liquidations_scope(context).to_dict()

        if self.liquidations_config.attach_feature_values_metadata:
            metadata["liquidations_features"] = self._selected_feature_values(
                context=context,
                source_features=source_features or [],
            )

        if self.liquidations_config.attach_liquidations_context_metadata:
            domain = self.liquidations_domain(context)
            metadata["liquidations_context_keys"] = sorted(domain.keys())

            for key in (
                "cascade",
                "exhaustion",
                "squeeze",
                "cluster",
                "signal",
            ):
                value = self.liquidations_item(context, key)
                if value is not None:
                    metadata[f"liquidations_{key}"] = serialize_for_metadata(value)

        metadata.update(dict(self.liquidations_config.metadata))

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
            _strategy_logger.debug("Entering LiquidationsTradingStrategy._selected_feature_values")
        result: dict[str, Any] = {}

        for feature in source_features:
            if not isinstance(feature, str) or not feature.strip():
                continue

            if context.has_feature(feature):
                result[feature] = serialize_for_metadata(context.get_feature(feature))
                continue

            value = self.liquidations_path(context, feature, default=None)
            if value is not None:
                result[feature] = serialize_for_metadata(value)

        return result

    # ------------------------------------------------------------------
    # Direction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def side_from_direction(value: Any) -> SignalSide:
        """
        Continuation side from liquidation cascade direction.

        DOWN cascade -> SHORT continuation.
        UP cascade   -> LONG continuation.
        """
        _strategy_logger = logging.getLogger(__name__ + ".LiquidationsTradingStrategy.side_from_direction")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.side_from_direction")
        label = _normalize_label(value)

        if not label:
            return SignalSide.UNKNOWN

        long_values = {
            "up",
            "long",
            "bullish",
            "buy",
            "upward",
            "cascade_up",
            "liquidations_up",
        }
        short_values = {
            "down",
            "short",
            "bearish",
            "sell",
            "downward",
            "cascade_down",
            "liquidations_down",
        }

        if label in long_values or "up" in label:
            return SignalSide.LONG

        if label in short_values or "down" in label:
            return SignalSide.SHORT

        return SignalSide.UNKNOWN

    @staticmethod
    def reversal_side_from_direction(value: Any) -> SignalSide:
        """
        Reversal side from liquidation cascade/exhaustion direction.

        DOWN exhaustion -> LONG reversal.
        UP exhaustion   -> SHORT reversal.
        """
        _strategy_logger = logging.getLogger(__name__ + ".LiquidationsTradingStrategy.reversal_side_from_direction")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.reversal_side_from_direction")
        continuation = LiquidationsTradingStrategy.side_from_direction(value)

        if continuation is SignalSide.LONG:
            return SignalSide.SHORT

        if continuation is SignalSide.SHORT:
            return SignalSide.LONG

        return SignalSide.UNKNOWN

    @staticmethod
    def opposite_side(side: SignalSide) -> SignalSide:
        _strategy_logger = logging.getLogger(__name__ + ".LiquidationsTradingStrategy.opposite_side")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.opposite_side")
        if side is SignalSide.LONG:
            return SignalSide.SHORT
        if side is SignalSide.SHORT:
            return SignalSide.LONG
        return SignalSide.UNKNOWN

    @staticmethod
    def is_directional_side(side: SignalSide) -> bool:
        _strategy_logger = logging.getLogger(__name__ + ".LiquidationsTradingStrategy.is_directional_side")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.is_directional_side")
        return side in {SignalSide.LONG, SignalSide.SHORT}

    def side_from_signed_value(
        self,
        value: Any,
        *,
        positive_side: SignalSide = SignalSide.LONG,
        negative_side: SignalSide = SignalSide.SHORT,
        dead_zone: float = 0.0,
    ) -> SignalSide:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.side_from_signed_value")
        parsed = _to_float(value)
        if parsed is None:
            return SignalSide.UNKNOWN

        if parsed > dead_zone:
            return positive_side

        if parsed < -dead_zone:
            return negative_side

        return SignalSide.UNKNOWN

    # ------------------------------------------------------------------
    # Analytics extraction helpers
    # ------------------------------------------------------------------

    def extract_confidence(self, value: Any, *, default: float = 0.0) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.extract_confidence")
        return _unit_score(
            _first_present(
                value,
                (
                    "confidence",
                    "probability",
                    "score_confidence",
                    "metadata.confidence",
                ),
                default=default,
            ),
            default,
        )

    def extract_score(self, value: Any, *, default: float = 0.0) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.extract_score")
        return _unit_score(
            _first_present(
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

    def extract_event_time(self, value: Any) -> datetime | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.extract_event_time")
        return parse_datetime(
            _first_present(
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

    def extract_notional_usd(self, value: Any) -> Decimal:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.extract_notional_usd")
        raw = _first_present(
            value,
            (
                "total_notional_usd",
                "notional_usd",
                "notional",
                "cluster_total_notional_usd",
                "metadata.total_notional_usd",
            ),
            default=Decimal("0"),
        )

        if isinstance(raw, Decimal):
            return raw

        try:
            return Decimal(str(raw))
        except Exception:
            return Decimal("0")

    def extract_event_count(self, value: Any) -> int:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.extract_event_count")
        raw = _first_present(
            value,
            (
                "event_count",
                "count",
                "liquidation_count",
                "metadata.event_count",
            ),
            default=0,
        )

        try:
            return max(0, int(raw))
        except Exception:
            return 0

    def extract_direction(self, value: Any) -> Any:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.extract_direction")
        return _first_present(
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

    def extract_severity_label(self, value: Any, *, default: str = "unknown") -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.extract_severity_label")
        raw = _first_present(
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
        label = _normalize_label(raw)
        return label or default

    def severity_score(self, value: Any, *, default: float = 0.0) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.severity_score")
        label = self.extract_severity_label(value, default="unknown")

        mapping = {
            "low": 0.25,
            "medium": 0.50,
            "moderate": 0.50,
            "high": 0.75,
            "extreme": 1.00,
            "critical": 1.00,
        }

        return mapping.get(label, default)

    def freshness_score(
        self,
        *,
        event_time: datetime | None,
        now: datetime | None = None,
        stale_after_seconds: float | None = None,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.freshness_score")
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
        self,
        *,
        event_time: datetime | None,
        now: datetime | None = None,
        stale_after_seconds: float | None = None,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.is_stale")
        if event_time is None or stale_after_seconds is None:
            return False

        if stale_after_seconds <= 0:
            return True

        current = ensure_aware_utc(now or utcnow())
        normalized_event_time = ensure_aware_utc(event_time)
        age = max(0.0, (current - normalized_event_time).total_seconds())
        return age > stale_after_seconds

    def weighted_score(
        self,
        values: Mapping[str, float],
        weights: Mapping[str, float],
        *,
        default: float = 0.0,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.weighted_score")
        return _weighted_score(values, weights, default=default)

    # ------------------------------------------------------------------
    # Plan / signal builders
    # ------------------------------------------------------------------

    def build_liquidations_trade_metadata(
        self,
        *,
        context: StrategyContext,
        side: SignalSide,
        setup_quality: float,
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
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.build_liquidations_trade_metadata")
        priority = self.build_priority_metadata(
            setup_quality=setup_quality,
            confluence_score=confluence_score,
            liquidity_score=liquidity_score,
            risk_reward_score=risk_reward_score,
            execution_quality_score=execution_quality_score,
            regime_alignment_score=regime_alignment_score,
            freshness_score=freshness_score,
        )

        scope = self.liquidations_scope(context)

        metadata = self.build_trade_metadata(
            tier=self.liquidations_config.default_trade_tier,
            order_intent=self.liquidations_config.default_order_intent,
            margin_mode=self.liquidations_config.default_margin_mode,
            market_type=self.liquidations_config.default_market_type,
            requested_leverage=self.liquidations_config.requested_leverage,
            exchange=scope.exchange,
            extra={
                **priority,
                "liquidations_side": side.value,
                "liquidity_score": clamp(float(liquidity_score), 0.0, 1.0),
                "execution_quality_score": clamp(float(execution_quality_score), 0.0, 1.0),
                "regime_alignment_score": clamp(float(regime_alignment_score), 0.0, 1.0),
                "freshness_score": clamp(float(freshness_score), 0.0, 1.0),
                **self.liquidations_context_metadata(
                    context,
                    source_features=source_features,
                ),
            },
        )

        if extra:
            metadata.update(extra)

        return metadata

    def build_basic_liquidations_plans(
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
        """
        Build lightweight draft plans.

        SignalBuilder may later enrich or replace these before converting the
        signal into RiskReadySignalPayload.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.build_basic_liquidations_plans")
        self.validate_context(context)

        if side not in {SignalSide.LONG, SignalSide.SHORT}:
            raise StrategyEvaluationError(
                "liquidations trade plans require directional side"
            )

        entry = EntryPlan(
            entry_type=self.liquidations_config.default_entry_type,
            price=entry_price,
            timeout_seconds=self.liquidations_config.entry_timeout_seconds,
            max_slippage_bps=self.liquidations_config.max_slippage_bps,
            confirmation_required=False,
            notes=["liquidations_strategy_entry_draft"],
            metadata={
                "source": self.strategy_name,
                "feature_source": FeatureSource.LIQUIDATIONS.value,
            },
        )

        targets: list[TargetPlan] = list(target_levels or [])
        if take_profit is not None:
            targets.append(
                TargetPlan(
                    price=take_profit,
                    size_fraction=1.0,
                    label="liquidations_take_profit",
                )
            )

        exit_plan = ExitPlan(
            exit_types=list(self.liquidations_config.default_exit_types),
            stop_loss=stop_loss,
            take_profit_levels=targets,
            max_holding_seconds=self.liquidations_config.max_holding_seconds,
            metadata={
                "source": self.strategy_name,
                "feature_source": FeatureSource.LIQUIDATIONS.value,
            },
        )

        invalidation = InvalidationPlan(
            price=stop_loss,
            reason=invalidation_reason or "liquidations_context_invalidated",
            timeout_seconds=self.liquidations_config.max_holding_seconds,
            conditions=[
                "liquidation_flow_reversed",
                "liquidation_intensity_faded",
                "market_structure_invalidated",
            ],
            metadata={
                "source": self.strategy_name,
                "feature_source": FeatureSource.LIQUIDATIONS.value,
            },
        )

        entry.validate()
        exit_plan.validate()
        invalidation.validate()
        return entry, exit_plan, invalidation

    def build_liquidations_signal(
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
        priority: SignalPriority = SignalPriority.HIGH,
        entry_price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> StrategySignal:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.build_liquidations_signal")
        if side not in {SignalSide.LONG, SignalSide.SHORT}:
            raise StrategyEvaluationError(
                f"{self.strategy_name}: liquidation signal side must be LONG or SHORT"
            )

        final_source_features = list(source_features or [])

        signal_metadata = self.build_liquidations_trade_metadata(
            context=context,
            side=side,
            setup_quality=score,
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
            tier=self.liquidations_config.default_trade_tier,
            order_intent=self.liquidations_config.default_order_intent,
            requested_leverage=self.liquidations_config.requested_leverage,
            margin_mode=self.liquidations_config.default_margin_mode,
            liquidity_class=None,
            execution_quality=None,
            market_type=self.liquidations_config.default_market_type,
        )

        signal.metadata.setdefault("feature_source", FeatureSource.LIQUIDATIONS.value)
        signal.metadata.setdefault("liquidations_strategy_base", self.__class__.__name__)
        signal.validate()
        return signal

    # ------------------------------------------------------------------
    # Applicability
    # ------------------------------------------------------------------

    def validate_context_requirements(self, context: StrategyContext) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.validate_context_requirements")
        super().validate_context_requirements(context)

        domain = self.liquidations_domain(context)
        has_liquidations_feature = any(
            context.has_feature(feature)
            for feature in LiquidationsFeatureNames.all()
        )

        if not domain and not has_liquidations_feature:
            raise StrategyEvaluationError(
                f"{self.strategy_name}: missing liquidations domain data for {context.symbol}"
            )

    def supports_regime(self, regime: MarketRegime) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering LiquidationsTradingStrategy.supports_regime")
        return super().supports_regime(regime)


__all__ = [
    "LIQUIDATIONS_FEATURES",
    "LIQUIDATIONS_DOMAIN_ALIASES",
    "LiquidationsFeatureNames",
    "LiquidationsStrategyConfig",
    "LiquidationsStrategyScope",
    "LiquidationsTradingStrategy",
    "ensure_utc",
    "parse_datetime",
    "serialize_for_metadata",
    "utc_now",
]