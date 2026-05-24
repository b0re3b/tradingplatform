# trading_system/strategy/strategies/whales/base.py

from __future__ import annotations
import logging

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any

from core.event_bus import EventBus
from core.scheduler import Scheduler
from .utils import (
    DEFAULT_WHALE_FEATURE_MAX_AGE_SECONDS,
    FUTURES_MARKET_TYPES,
    as_dict,
    base_whales_source_features,
    extract_cluster_side,
    extract_context_strength,
    extract_dominant_side,
    extract_event_time,
    extract_exchange,
    extract_exchange_symbol,
    extract_exhausted_side,
    extract_large_trade_notional,
    extract_large_trade_payload,
    extract_large_trade_zscore,
    extract_liquidation_notional,
    extract_liquidation_side,
    extract_market_type,
    extract_metadata,
    extract_notional,
    extract_reference_price,
    extract_symbol,
    extract_timeframe,
    extract_total_notional,
    extract_trade_count,
    extract_whale_activity_payload,
    extract_whale_cluster_exhaustion_payload,
    extract_whale_cluster_payload,
    extract_whale_cluster_update_payload,
    extract_whale_liquidation_context_payload,
    extract_whale_pressure_payload,
    extract_whale_side,
    extract_imbalance_ratio,
    extract_pressure_score,
    freshness_score,
    is_directional_side,
    normalize_exchange,
    normalize_market_type,
    normalize_symbol,
    parse_datetime,
    resolve_cluster_score,
    resolve_cluster_side,
    resolve_continuation_probability,
    resolve_exhausted_side,
    resolve_exhaustion_probability,
    serialize_for_metadata,
    side_label_to_signal_side,
    timestamp_ms,
    to_bool,
    to_float,
    to_int,
    to_str,
    unit_score,
    whale_payload_validation_reason,
    whale_quality_filter_reason,
    whales_domain,
    whales_item,
    whales_path,
)
from ..base_strategy import TradingStrategy
from ...config import StrategyConfig, StrategyDefinitionConfig
from ...enums import (
    EntryType,
    ExitType,
    FeatureSource,
    SetupType,
    SignalOrigin,
    SignalPriority,
    SignalSide,
    SignalStatus,
    StrategyCategory,
    StrategyMarginMode,
    StrategyMarketType,
    StrategyOrderIntent,
    StrategyTradeTier,
    Timeframe,
    TriggerType,
)
from ...exceptions import StrategyConfigError, StrategyEvaluationError
from ...models import FeatureSnapshot, StrategyContext, StrategySignal


# =============================================================================
# Feature contract
# =============================================================================


@dataclass(frozen=True, slots=True)
class WhalesFeatureNames:
    """
    Stable whales feature names expected in StrategyContext.

    StrategyContextBuilder / SignalNormalizer should populate these from
    analytics.whales.* payloads. Concrete strategies may also read equivalent
    values from FeatureSource.WHALES domain_data aliases.
    """
    _logger = logging.getLogger(__name__ + ".WhalesFeatureNames")

    PRESSURE: str = "whales.pressure"
    ACTIVITY: str = "whales.activity"
    LARGE_TRADE: str = "whales.large_trade"

    CLUSTER: str = "whales.cluster"
    CLUSTER_UPDATE: str = "whales.cluster_update"
    CLUSTER_EXHAUSTION: str = "whales.cluster_exhaustion"

    LIQUIDATION_CONTEXT: str = "whales.liquidation_context"

    SYMBOL: str = "whales.symbol"
    EXCHANGE: str = "whales.exchange"
    MARKET_TYPE: str = "whales.market_type"
    TIMEFRAME: str = "whales.timeframe"
    EXCHANGE_SYMBOL: str = "whales.exchange_symbol"

    DOMINANT_SIDE: str = "whales.dominant_side"
    WHALE_SIDE: str = "whales.whale_side"
    LIQUIDATION_SIDE: str = "whales.liquidation_side"
    EXHAUSTED_SIDE: str = "whales.exhausted_side"
    CLUSTER_SIDE: str = "whales.cluster_side"

    IMBALANCE_RATIO: str = "whales.imbalance_ratio"
    PRESSURE_SCORE: str = "whales.pressure_score"
    CONTEXT_STRENGTH: str = "whales.context_strength"
    CLUSTER_SCORE: str = "whales.cluster_score"
    CONTINUATION_PROBABILITY: str = "whales.continuation_probability"
    EXHAUSTION_PROBABILITY: str = "whales.exhaustion_probability"

    TOTAL_NOTIONAL: str = "whales.total_notional"
    LIQUIDATION_NOTIONAL: str = "whales.liquidation_notional"
    TRADE_COUNT: str = "whales.trade_count"
    LARGE_TRADE_NOTIONAL: str = "whales.large_trade_notional"
    LARGE_TRADE_ZSCORE: str = "whales.large_trade_zscore"

    REFERENCE_PRICE: str = "whales.reference_price"
    CONFIDENCE: str = "whales.confidence"
    TIMESTAMP: str = "whales.timestamp"
    METADATA: str = "whales.metadata"

    @classmethod
    def all(cls) -> set[str]:
        _strategy_logger = getattr(cls, "_logger", None) or logging.getLogger(__name__ + ".WhalesFeatureNames")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesFeatureNames.all")
        instance = cls()
        return {
            getattr(instance, item.name)
            for item in fields(cls)
            if isinstance(getattr(instance, item.name), str)
            and getattr(instance, item.name).strip()
        }


WHALES_FEATURES = WhalesFeatureNames()


# =============================================================================
# Lightweight validation / feature DTOs
# =============================================================================


@dataclass(frozen=True, slots=True)
class WhalePayloadValidation:
    _logger = logging.getLogger(__name__ + ".WhalePayloadValidation")
    valid: bool
    reason: str | None = None
    payload_age_seconds: float | None = None
    source: str = "unknown"

    @classmethod
    def ok(
        cls,
        *,
        payload_age_seconds: float | None = None,
        source: str = "unknown",
    ) -> WhalePayloadValidation:
        _strategy_logger = getattr(cls, "_logger", None) or logging.getLogger(__name__ + ".WhalePayloadValidation")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalePayloadValidation.ok")
        return cls(
            valid=True,
            reason=None,
            payload_age_seconds=payload_age_seconds,
            source=source,
        )

    @classmethod
    def failed(
        cls,
        reason: str,
        *,
        payload_age_seconds: float | None = None,
        source: str = "unknown",
    ) -> WhalePayloadValidation:
        _strategy_logger = getattr(cls, "_logger", None) or logging.getLogger(__name__ + ".WhalePayloadValidation")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalePayloadValidation.failed")
        return cls(
            valid=False,
            reason=reason,
            payload_age_seconds=payload_age_seconds,
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalePayloadValidation.to_dict")
        return {
            "valid": self.valid,
            "reason": self.reason,
            "payload_age_seconds": self.payload_age_seconds,
            "source": self.source,
        }


@dataclass(slots=True)
class WhaleFeaturePayload:
    _logger = logging.getLogger(__name__ + ".WhaleFeaturePayload")
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    event_time: datetime | None = None
    validation: WhalePayloadValidation = field(
        default_factory=WhalePayloadValidation.ok
    )

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleFeaturePayload.__post_init__")
        self.payload = as_dict(self.payload)
        self.event_time = parse_datetime(self.event_time)

    @property
    def available(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleFeaturePayload.available")
        return bool(self.payload) and self.validation.valid

    def age_seconds(self, now: datetime | None = None) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleFeaturePayload.age_seconds")
        if self.event_time is None:
            return None
        current = parse_datetime(now) or parse_datetime(datetime.utcnow())
        if current is None:
            return None
        return max(0.0, (current - self.event_time).total_seconds())

    def to_dict(self) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleFeaturePayload.to_dict")
        return {
            "name": self.name,
            "available": self.available,
            "event_time": self.event_time.isoformat() if self.event_time else None,
            "timestamp_ms": timestamp_ms(self.event_time),
            "validation": self.validation.to_dict(),
            "payload": serialize_for_metadata(self.payload),
        }


# =============================================================================
# Scope
# =============================================================================


@dataclass(frozen=True, slots=True)
class WhalesStrategyScope:
    _logger = logging.getLogger(__name__ + ".WhalesStrategyScope")
    exchange: str
    market_type: str
    symbol: str
    timeframe: str = Timeframe.M1.value
    exchange_symbol: str | None = None

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesStrategyScope.__post_init__")
        exchange = normalize_exchange(self.exchange) or "unknown"
        market_type = normalize_market_type(self.market_type) or "unknown"
        symbol = normalize_symbol(self.symbol)
        timeframe = str(self.timeframe or Timeframe.M1.value).strip().lower()
        exchange_symbol = to_str(self.exchange_symbol) or symbol

        if not symbol:
            raise StrategyEvaluationError("WhalesStrategyScope.symbol cannot be empty")

        object.__setattr__(self, "exchange", exchange)
        object.__setattr__(self, "market_type", market_type)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timeframe", timeframe)
        object.__setattr__(self, "exchange_symbol", exchange_symbol)

    @property
    def key(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesStrategyScope.key")
        return (
            f"{self.exchange}:"
            f"{self.market_type}:"
            f"{self.symbol}:"
            f"{self.timeframe}"
        )

    @property
    def legacy_key(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesStrategyScope.legacy_key")
        return f"{self.exchange}:{self.symbol}"

    @property
    def is_futures(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesStrategyScope.is_futures")
        if not self.market_type or self.market_type == "unknown":
            return True
        return self.market_type in FUTURES_MARKET_TYPES

    def to_dict(self) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesStrategyScope.to_dict")
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "exchange_symbol": self.exchange_symbol,
            "key": self.key,
            "legacy_key": self.legacy_key,
            "is_futures": self.is_futures,
        }


# =============================================================================
# Config
# =============================================================================


@dataclass(slots=True)
class WhalesStrategyConfig:
    """
    Stateless domain config shared by concrete whale strategies.

    No EventBus topics, no context subscription, no local evaluator lifecycle,
    no StrategyEvaluation publishing. Concrete strategies consume StrategyContext
    and return StrategySignal only.
    """
    _logger = logging.getLogger(__name__ + ".WhalesStrategyConfig")

    min_score: float = 0.60
    min_confidence: float = 0.55
    stale_feature_max_age_seconds: float | None = DEFAULT_WHALE_FEATURE_MAX_AGE_SECONDS

    require_futures_market_type: bool = True
    allowed_market_types: set[str] = field(
        default_factory=lambda: set(FUTURES_MARKET_TYPES)
    )
    validate_scope: bool = True
    strict_timeframe_scope: bool = False

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

    requested_leverage: float | None = None
    max_slippage_bps: float | None = None
    entry_timeout_seconds: int | None = None
    max_holding_seconds: int | None = None

    attach_whale_context_metadata: bool = True
    attach_scope_metadata: bool = True
    attach_raw_payload_metadata: bool = True
    attach_feature_validation_metadata: bool = True

    tag_whales: str = "whales"
    tag_absorption: str = "whale_absorption"
    tag_breakout: str = "whale_breakout"
    tag_accumulation: str = "whale_accumulation"
    tag_distribution: str = "whale_distribution"
    tag_liquidation_reversal: str = "whale_liquidation_reversal"

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesStrategyConfig.validate")
        bounded = {
            "min_score": self.min_score,
            "min_confidence": self.min_confidence,
        }
        for name, value in bounded.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{name} must be between 0.0 and 1.0")

        if (
            self.stale_feature_max_age_seconds is not None
            and self.stale_feature_max_age_seconds <= 0
        ):
            raise StrategyConfigError("stale_feature_max_age_seconds must be > 0")

        if self.requested_leverage is not None and self.requested_leverage <= 0:
            raise StrategyConfigError("requested_leverage must be > 0")

        if self.max_slippage_bps is not None and self.max_slippage_bps < 0:
            raise StrategyConfigError("max_slippage_bps must be >= 0")

        if self.entry_timeout_seconds is not None and self.entry_timeout_seconds <= 0:
            raise StrategyConfigError("entry_timeout_seconds must be > 0")

        if self.max_holding_seconds is not None and self.max_holding_seconds <= 0:
            raise StrategyConfigError("max_holding_seconds must be > 0")

        if not self.allowed_market_types:
            raise StrategyConfigError("allowed_market_types cannot be empty")

        for market_type in self.allowed_market_types:
            if not isinstance(market_type, str) or not market_type.strip():
                raise StrategyConfigError("allowed_market_types must contain strings only")

        for attr in (
            "tag_whales",
            "tag_absorption",
            "tag_breakout",
            "tag_accumulation",
            "tag_distribution",
            "tag_liquidation_reversal",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")


# =============================================================================
# Composite snapshot
# =============================================================================


@dataclass(slots=True)
class WhaleCompositeSnapshot:
    """
    Strategy-level normalized projection over analytics.whales payloads.

    This is not an analytics model. It is a stable strategy-side DTO that lets
    concrete whale strategies consume pressure/activity/cluster/liquidation
    payloads through one contract.
    """
    _logger = logging.getLogger(__name__ + ".WhaleCompositeSnapshot")

    symbol: str
    exchange: str = "unknown"
    market_type: str = "unknown"
    timeframe: str = Timeframe.M1.value
    exchange_symbol: str | None = None

    pressure: dict[str, Any] = field(default_factory=dict)
    activity: dict[str, Any] = field(default_factory=dict)
    large_trade: dict[str, Any] = field(default_factory=dict)
    cluster: dict[str, Any] = field(default_factory=dict)
    cluster_update: dict[str, Any] = field(default_factory=dict)
    cluster_exhaustion: dict[str, Any] = field(default_factory=dict)
    liquidation_context: dict[str, Any] = field(default_factory=dict)

    dominant_side: str = "unknown"
    whale_side: str = "unknown"
    liquidation_side: str = "unknown"
    exhausted_side: str = "unknown"
    cluster_side: str = "unknown"

    imbalance_ratio: float = 0.0
    pressure_score: float = 0.0
    context_strength: float = 0.0
    cluster_score: float | None = None
    continuation_probability: float | None = None
    exhaustion_probability: float | None = None

    total_notional: float = 0.0
    liquidation_notional: float = 0.0
    trade_count: int = 0
    large_trade_notional: float = 0.0
    large_trade_zscore: float | None = None

    reference_price: float | None = None
    confidence: float = 0.0
    timestamp: datetime | None = None
    source: str = "unknown"

    feature_validations: dict[str, WhalePayloadValidation] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleCompositeSnapshot.__post_init__")
        self.symbol = normalize_symbol(self.symbol)
        self.exchange = normalize_exchange(self.exchange) or "unknown"
        self.market_type = normalize_market_type(self.market_type) or "unknown"
        self.timeframe = str(self.timeframe or Timeframe.M1.value).strip().lower()
        self.exchange_symbol = to_str(self.exchange_symbol) or self.symbol

        if not self.symbol:
            raise StrategyEvaluationError("WhaleCompositeSnapshot.symbol cannot be empty")

        self.pressure = as_dict(self.pressure)
        self.activity = as_dict(self.activity)
        self.large_trade = as_dict(self.large_trade)
        self.cluster = as_dict(self.cluster)
        self.cluster_update = as_dict(self.cluster_update)
        self.cluster_exhaustion = as_dict(self.cluster_exhaustion)
        self.liquidation_context = as_dict(self.liquidation_context)

        self.imbalance_ratio = unit_score(self.imbalance_ratio)
        self.pressure_score = unit_score(self.pressure_score)
        self.context_strength = unit_score(self.context_strength)
        self.cluster_score = (
            unit_score(self.cluster_score)
            if self.cluster_score is not None
            else None
        )
        self.continuation_probability = (
            unit_score(self.continuation_probability)
            if self.continuation_probability is not None
            else None
        )
        self.exhaustion_probability = (
            unit_score(self.exhaustion_probability)
            if self.exhaustion_probability is not None
            else None
        )

        self.total_notional = max(0.0, float(self.total_notional))
        self.liquidation_notional = max(0.0, float(self.liquidation_notional))
        self.trade_count = max(0, int(self.trade_count))
        self.large_trade_notional = max(0.0, float(self.large_trade_notional))
        self.large_trade_zscore = (
            max(0.0, float(self.large_trade_zscore))
            if self.large_trade_zscore is not None
            else None
        )
        self.reference_price = (
            max(0.0, float(self.reference_price))
            if self.reference_price is not None
            else None
        )
        self.confidence = unit_score(self.confidence)
        self.timestamp = parse_datetime(self.timestamp)
        self.metadata = as_dict(self.metadata)

    @property
    def scope(self) -> WhalesStrategyScope:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleCompositeSnapshot.scope")
        return WhalesStrategyScope(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
            exchange_symbol=self.exchange_symbol,
        )

    @property
    def scope_key(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleCompositeSnapshot.scope_key")
        return self.scope.key

    @property
    def is_futures(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleCompositeSnapshot.is_futures")
        return self.scope.is_futures

    @property
    def whale_signal_side(self) -> SignalSide:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleCompositeSnapshot.whale_signal_side")
        for candidate in (
            self.dominant_side,
            self.whale_side,
            self.cluster_side,
            self.exhausted_side,
        ):
            side = side_label_to_signal_side(candidate)
            if is_directional_side(side):
                return side
        return SignalSide.UNKNOWN

    @property
    def has_pressure(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleCompositeSnapshot.has_pressure")
        return bool(self.pressure)

    @property
    def has_activity(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleCompositeSnapshot.has_activity")
        return bool(self.activity)

    @property
    def has_large_trade(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleCompositeSnapshot.has_large_trade")
        return bool(self.large_trade)

    @property
    def has_cluster(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleCompositeSnapshot.has_cluster")
        return bool(self.cluster or self.cluster_update or self.cluster_exhaustion)

    @property
    def has_liquidation_context(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleCompositeSnapshot.has_liquidation_context")
        return bool(self.liquidation_context)

    def has_minimum_data(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleCompositeSnapshot.has_minimum_data")
        return any(
            (
                self.has_pressure,
                self.has_activity,
                self.has_large_trade,
                self.has_cluster,
                self.has_liquidation_context,
                self.imbalance_ratio > 0.0,
                self.context_strength > 0.0,
                self.total_notional > 0.0,
                self.large_trade_notional > 0.0,
                self.confidence > 0.0,
            )
        )

    def inputs(self) -> dict[str, dict[str, Any]]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleCompositeSnapshot.inputs")
        return {
            "pressure": self.pressure,
            "activity": self.activity,
            "large_trade": self.large_trade,
            "cluster": self.cluster,
            "cluster_update": self.cluster_update,
            "cluster_exhaustion": self.cluster_exhaustion,
            "liquidation_context": self.liquidation_context,
        }

    def to_signal_payload(self) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleCompositeSnapshot.to_signal_payload")
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "market_type": self.market_type,
            "timeframe": self.timeframe,
            "exchange_symbol": self.exchange_symbol,
            "dominant_side": self.dominant_side,
            "whale_side": self.whale_side,
            "liquidation_side": self.liquidation_side,
            "exhausted_side": self.exhausted_side,
            "cluster_side": self.cluster_side,
            "imbalance_ratio": self.imbalance_ratio,
            "pressure_score": self.pressure_score,
            "context_strength": self.context_strength,
            "cluster_score": self.cluster_score,
            "continuation_probability": self.continuation_probability,
            "exhaustion_probability": self.exhaustion_probability,
            "total_notional": self.total_notional,
            "liquidation_notional": self.liquidation_notional,
            "trade_count": self.trade_count,
            "large_trade_notional": self.large_trade_notional,
            "large_trade_zscore": self.large_trade_zscore,
            "reference_price": self.reference_price,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleCompositeSnapshot.to_dict")
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "market_type": self.market_type,
            "timeframe": self.timeframe,
            "exchange_symbol": self.exchange_symbol,
            "scope": self.scope.to_dict(),
            "scope_key": self.scope_key,
            "is_futures": self.is_futures,
            "dominant_side": self.dominant_side,
            "whale_side": self.whale_side,
            "liquidation_side": self.liquidation_side,
            "exhausted_side": self.exhausted_side,
            "cluster_side": self.cluster_side,
            "whale_signal_side": self.whale_signal_side.value,
            "imbalance_ratio": self.imbalance_ratio,
            "pressure_score": self.pressure_score,
            "context_strength": self.context_strength,
            "cluster_score": self.cluster_score,
            "continuation_probability": self.continuation_probability,
            "exhaustion_probability": self.exhaustion_probability,
            "total_notional": self.total_notional,
            "liquidation_notional": self.liquidation_notional,
            "trade_count": self.trade_count,
            "large_trade_notional": self.large_trade_notional,
            "large_trade_zscore": self.large_trade_zscore,
            "reference_price": self.reference_price,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "timestamp_ms": timestamp_ms(self.timestamp),
            "source": self.source,
            "available_inputs": {
                "pressure": self.has_pressure,
                "activity": self.has_activity,
                "large_trade": self.has_large_trade,
                "cluster": self.has_cluster,
                "liquidation_context": self.has_liquidation_context,
            },
            "feature_validations": {
                name: validation.to_dict()
                for name, validation in self.feature_validations.items()
            },
            "raw": {
                "pressure": serialize_for_metadata(self.pressure),
                "activity": serialize_for_metadata(self.activity),
                "large_trade": serialize_for_metadata(self.large_trade),
                "cluster": serialize_for_metadata(self.cluster),
                "cluster_update": serialize_for_metadata(self.cluster_update),
                "cluster_exhaustion": serialize_for_metadata(self.cluster_exhaustion),
                "liquidation_context": serialize_for_metadata(self.liquidation_context),
            },
            "metadata": serialize_for_metadata(self.metadata),
        }


# =============================================================================
# Base whales strategy
# =============================================================================


class WhalesTradingStrategy(TradingStrategy):
    """
    Base class for concrete strategy/strategies/whales/* classes.

    Responsibilities:
    - read whale analytics data from StrategyContext only;
    - provide helpers for pressure/activity/cluster/liquidation payloads;
    - validate scope/freshness/futures-only guards;
    - build internal StrategySignal objects;
    - attach whale metadata for SignalProcessor.

    Forbidden:
    - no direct analytics.whales.* EventBus subscriptions;
    - no local evaluator lifecycle;
    - no direct signal.generated / signal.rejected publishing;
    - no StrategyEvaluation as concrete strategy output;
    - no RiskManager / Execution calls;
    - no raw market data reads.
    """
    _logger = logging.getLogger(__name__ + ".WhalesTradingStrategy")

    component_namespace = "strategy.whales"
    category: StrategyCategory = StrategyCategory.WHALES
    default_setup_type: SetupType = SetupType.UNKNOWN
    default_timeframe: Timeframe = Timeframe.M1

    feature_names = WHALES_FEATURES

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        whales_config: WhalesStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy.__init__")
        self.whales_config = whales_config or WhalesStrategyConfig()
        self.whales_config.validate()

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
            _strategy_logger.debug("Entering WhalesTradingStrategy.validate_config")
        super().validate_config()
        self.whales_config.validate()

    # ------------------------------------------------------------------
    # Context/domain access
    # ------------------------------------------------------------------

    def whales_domain(self, context: StrategyContext) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy.whales_domain")
        self.validate_context(context)
        return whales_domain(context)

    def whales_item(
        self,
        context: StrategyContext,
        key: str,
        default: Any = None,
    ) -> Any:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy.whales_item")
        self.validate_context(context)
        return whales_item(context, key, default)

    def whales_path(
        self,
        context: StrategyContext,
        path: str,
        default: Any = None,
    ) -> Any:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy.whales_path")
        self.validate_context(context)

        if not isinstance(path, str) or not path.strip():
            raise StrategyEvaluationError("whales path cannot be empty")

        return whales_path(context, path, default)

    def whales_float(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float | None = None,
    ) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy.whales_float")
        return to_float(self.whales_path(context, path, default), default)

    def whales_int(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: int | None = None,
    ) -> int | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy.whales_int")
        return to_int(self.whales_path(context, path, default), default)

    def whales_score(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float = 0.0,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy.whales_score")
        return unit_score(self.whales_path(context, path, default), default)

    def whales_bool(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: bool = False,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy.whales_bool")
        return to_bool(self.whales_path(context, path, default), default)

    def whales_str(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: str | None = None,
    ) -> str | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy.whales_str")
        return to_str(self.whales_path(context, path, default), default)

    def whales_datetime(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: datetime | None = None,
    ) -> datetime | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy.whales_datetime")
        return parse_datetime(self.whales_path(context, path, default))

    def whales_feature_snapshot(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> FeatureSnapshot | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy.whales_feature_snapshot")
        self.validate_context(context)

        if not isinstance(feature_name, str) or not feature_name.strip():
            raise StrategyEvaluationError("feature_name cannot be empty")

        features_map = getattr(context, "features", None)
        if isinstance(features_map, Mapping):
            raw = features_map.get(feature_name)
            if isinstance(raw, FeatureSnapshot):
                return raw

        return None

    def whales_feature_age_seconds(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy.whales_feature_age_seconds")
        snapshot = self.whales_feature_snapshot(context, feature_name)
        if snapshot is None:
            return None
        return snapshot.age_seconds(context.timestamp)

    def whales_feature_is_stale(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy.whales_feature_is_stale")
        max_age = self.whales_config.stale_feature_max_age_seconds
        if max_age is None:
            return False

        age = self.whales_feature_age_seconds(context, feature_name)
        if age is None:
            return False

        return age > max_age

    def has_any_whales_data(
        self,
        context: StrategyContext,
        feature_names: tuple[str, ...] = (),
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy.has_any_whales_data")
        self.validate_context(context)

        if self.whales_domain(context):
            return True

        return any(context.has_feature(name) for name in feature_names)

    def has_stale_whales_features(
        self,
        context: StrategyContext,
        feature_names: tuple[str, ...] | None = None,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy.has_stale_whales_features")
        names = feature_names or tuple(self.required_features())

        return any(
            self.whales_feature_is_stale(context, feature_name)
            for feature_name in names
        )

    # ------------------------------------------------------------------
    # Scope
    # ------------------------------------------------------------------

    def whales_scope(self, context: StrategyContext) -> WhalesStrategyScope:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy.whales_scope")
        domain = self.whales_domain(context)

        candidate = (
            self.whales_item(context, "pressure")
            or self.whales_item(context, "activity")
            or self.whales_item(context, "large_trade")
            or self.whales_item(context, "cluster")
            or self.whales_item(context, "liquidation_context")
            or domain
        )

        symbol = (
            extract_symbol(candidate)
            or normalize_symbol(context.symbol)
        )
        exchange = (
            extract_exchange(candidate)
            or normalize_exchange(context.metadata.get("exchange"))
            or "unknown"
        )
        market_type = (
            extract_market_type(candidate)
            or normalize_market_type(context.metadata.get("market_type"))
            or self.whales_config.default_market_type.value
        )
        timeframe_value = (
            extract_timeframe(candidate, "")
            or str(getattr(context.timeframe, "value", context.timeframe) or Timeframe.M1.value)
        )

        return WhalesStrategyScope(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe_value,
            exchange_symbol=extract_exchange_symbol(candidate) or symbol,
        )

    # ------------------------------------------------------------------
    # Payload resolution
    # ------------------------------------------------------------------

    def resolve_whale_snapshot(
        self,
        context: StrategyContext,
    ) -> WhaleCompositeSnapshot | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy.resolve_whale_snapshot")
        self.validate_context(context)

        scope = self.whales_scope(context)

        pressure = self._resolve_feature_payload(
            context,
            key="pressure",
            feature_name=WHALES_FEATURES.PRESSURE,
            extractor=extract_whale_pressure_payload,
        )
        activity = self._resolve_feature_payload(
            context,
            key="activity",
            feature_name=WHALES_FEATURES.ACTIVITY,
            extractor=extract_whale_activity_payload,
        )
        large_trade = self._resolve_feature_payload(
            context,
            key="large_trade",
            feature_name=WHALES_FEATURES.LARGE_TRADE,
            extractor=extract_large_trade_payload,
        )
        cluster = self._resolve_feature_payload(
            context,
            key="cluster",
            feature_name=WHALES_FEATURES.CLUSTER,
            extractor=extract_whale_cluster_payload,
        )
        cluster_update = self._resolve_feature_payload(
            context,
            key="cluster_update",
            feature_name=WHALES_FEATURES.CLUSTER_UPDATE,
            extractor=extract_whale_cluster_update_payload,
        )
        cluster_exhaustion = self._resolve_feature_payload(
            context,
            key="cluster_exhaustion",
            feature_name=WHALES_FEATURES.CLUSTER_EXHAUSTION,
            extractor=extract_whale_cluster_exhaustion_payload,
        )
        liquidation_context = self._resolve_feature_payload(
            context,
            key="liquidation_context",
            feature_name=WHALES_FEATURES.LIQUIDATION_CONTEXT,
            extractor=extract_whale_liquidation_context_payload,
        )

        payloads = {
            "pressure": pressure.payload,
            "activity": activity.payload,
            "large_trade": large_trade.payload,
            "cluster": cluster.payload,
            "cluster_update": cluster_update.payload,
            "cluster_exhaustion": cluster_exhaustion.payload,
            "liquidation_context": liquidation_context.payload,
        }

        if not any(payloads.values()):
            feature_snapshot = self._build_snapshot_from_features(
                context=context,
                scope=scope,
            )
            if feature_snapshot is not None and feature_snapshot.has_minimum_data():
                return feature_snapshot
            return None

        metadata: dict[str, Any] = {}
        for payload in payloads.values():
            metadata.update(extract_metadata(payload))

        timestamp = self._resolve_snapshot_timestamp(
            context=context,
            payloads=payloads,
            fallback=context.timestamp,
        )

        snapshot = WhaleCompositeSnapshot(
            symbol=self._first_non_empty_symbol(payloads, scope.symbol),
            exchange=self._first_non_empty_exchange(payloads, scope.exchange),
            market_type=self._first_non_empty_market_type(payloads, scope.market_type),
            timeframe=self._first_non_empty_timeframe(payloads, scope.timeframe),
            exchange_symbol=self._first_non_empty_exchange_symbol(
                payloads,
                scope.exchange_symbol,
            ),
            pressure=pressure.payload,
            activity=activity.payload,
            large_trade=large_trade.payload,
            cluster=cluster.payload,
            cluster_update=cluster_update.payload,
            cluster_exhaustion=cluster_exhaustion.payload,
            liquidation_context=liquidation_context.payload,
            dominant_side=self._resolve_dominant_side(payloads),
            whale_side=self._resolve_whale_side(payloads),
            liquidation_side=self._resolve_liquidation_side(payloads),
            exhausted_side=resolve_exhausted_side(payloads),
            cluster_side=resolve_cluster_side(payloads),
            imbalance_ratio=extract_imbalance_ratio(pressure.payload),
            pressure_score=extract_pressure_score(pressure.payload),
            context_strength=extract_context_strength(liquidation_context.payload),
            cluster_score=resolve_cluster_score(payloads),
            continuation_probability=resolve_continuation_probability(payloads),
            exhaustion_probability=resolve_exhaustion_probability(payloads),
            total_notional=self._resolve_total_notional(payloads),
            liquidation_notional=extract_liquidation_notional(
                liquidation_context.payload
            ),
            trade_count=self._resolve_trade_count(payloads),
            large_trade_notional=extract_large_trade_notional(large_trade.payload),
            large_trade_zscore=extract_large_trade_zscore(large_trade.payload),
            reference_price=self._resolve_reference_price(payloads),
            confidence=self._resolve_confidence(payloads),
            timestamp=timestamp,
            source="context.domain",
            feature_validations={
                "pressure": pressure.validation,
                "activity": activity.validation,
                "large_trade": large_trade.validation,
                "cluster": cluster.validation,
                "cluster_update": cluster_update.validation,
                "cluster_exhaustion": cluster_exhaustion.validation,
                "liquidation_context": liquidation_context.validation,
            },
            metadata={
                **metadata,
                "scope": scope.to_dict(),
                "source": "context.domain",
            },
        )

        if snapshot.has_minimum_data():
            return snapshot

        return None

    def accepts_whale_snapshot(
        self,
        snapshot: WhaleCompositeSnapshot,
        *,
        require_futures_market_type: bool | None = None,
        min_confidence: float | None = None,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy.accepts_whale_snapshot")
        if not snapshot.has_minimum_data():
            return False

        require_futures = (
            self.whales_config.require_futures_market_type
            if require_futures_market_type is None
            else require_futures_market_type
        )
        if require_futures and not snapshot.is_futures:
            return False

        if snapshot.market_type != "unknown":
            if snapshot.market_type not in {
                normalize_market_type(item)
                for item in self.whales_config.allowed_market_types
            }:
                return False

        confidence_threshold = (
            self.whales_config.min_confidence
            if min_confidence is None
            else min_confidence
        )
        if snapshot.confidence < confidence_threshold:
            return False

        return True

    def _resolve_feature_payload(
        self,
        context: StrategyContext,
        *,
        key: str,
        feature_name: str,
        extractor: Any,
    ) -> WhaleFeaturePayload:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy._resolve_feature_payload")
        domain = self.whales_domain(context)
        event_name = str(domain.get("event_name") or domain.get("topic") or domain.get("source_topic") or "")

        candidate = (
            self.whales_item(context, key)
            or self.whales_path(context, key, None)
            or self._feature_value(context, feature_name)
        )

        # Some normalized whale analytics events are themselves the canonical
        # section payload, e.g. analytics.whales.large_trade has side/notional/
        # zscore at the domain top-level rather than under domain["large_trade"].
        # Use the full domain only for section-compatible event types. Do not
        # synthesize liquidation_context/cluster from plain large-trade events.
        if candidate is None:
            normalized_event = event_name.lower()
            if key == "large_trade" and "large_trade" in normalized_event:
                candidate = domain
            elif key == "pressure" and "pressure" in normalized_event:
                candidate = domain
            elif key == "activity" and "activity" in normalized_event:
                candidate = domain

        payload = extractor(candidate) if candidate is not None else {}

        if not payload and isinstance(candidate, Mapping):
            payload = dict(candidate)

        event_time = extract_event_time(payload)

        validation_reason = whale_payload_validation_reason(
            payload,
            context=context,
            require_futures_market_type=self.whales_config.require_futures_market_type,
            validate_scope=self.whales_config.validate_scope,
            strict_timeframe=self.whales_config.strict_timeframe_scope,
            stale_after_seconds=self.whales_config.stale_feature_max_age_seconds,
            now=context.timestamp,
        )

        validation = (
            WhalePayloadValidation.ok(
                payload_age_seconds=self._payload_age_seconds(
                    event_time=event_time,
                    now=context.timestamp,
                ),
                source=feature_name,
            )
            if validation_reason is None
            else WhalePayloadValidation.failed(
                validation_reason,
                payload_age_seconds=self._payload_age_seconds(
                    event_time=event_time,
                    now=context.timestamp,
                ),
                source=feature_name,
            )
        )

        if validation_reason is not None:
            return WhaleFeaturePayload(
                name=feature_name,
                payload={},
                event_time=event_time,
                validation=validation,
            )

        return WhaleFeaturePayload(
            name=feature_name,
            payload=payload,
            event_time=event_time,
            validation=validation,
        )

    def _build_snapshot_from_features(
        self,
        *,
        context: StrategyContext,
        scope: WhalesStrategyScope,
    ) -> WhaleCompositeSnapshot | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy._build_snapshot_from_features")
        payload = {
            "symbol": self._feature_value(context, WHALES_FEATURES.SYMBOL),
            "exchange": self._feature_value(context, WHALES_FEATURES.EXCHANGE),
            "market_type": self._feature_value(context, WHALES_FEATURES.MARKET_TYPE),
            "timeframe": self._feature_value(context, WHALES_FEATURES.TIMEFRAME),
            "exchange_symbol": self._feature_value(
                context,
                WHALES_FEATURES.EXCHANGE_SYMBOL,
            ),
            "dominant_side": self._feature_value(
                context,
                WHALES_FEATURES.DOMINANT_SIDE,
            ),
            "whale_side": self._feature_value(context, WHALES_FEATURES.WHALE_SIDE),
            "liquidation_side": self._feature_value(
                context,
                WHALES_FEATURES.LIQUIDATION_SIDE,
            ),
            "exhausted_side": self._feature_value(
                context,
                WHALES_FEATURES.EXHAUSTED_SIDE,
            ),
            "cluster_side": self._feature_value(
                context,
                WHALES_FEATURES.CLUSTER_SIDE,
            ),
            "imbalance_ratio": self._feature_value(
                context,
                WHALES_FEATURES.IMBALANCE_RATIO,
            ),
            "pressure_score": self._feature_value(
                context,
                WHALES_FEATURES.PRESSURE_SCORE,
            ),
            "context_strength": self._feature_value(
                context,
                WHALES_FEATURES.CONTEXT_STRENGTH,
            ),
            "cluster_score": self._feature_value(
                context,
                WHALES_FEATURES.CLUSTER_SCORE,
            ),
            "continuation_probability": self._feature_value(
                context,
                WHALES_FEATURES.CONTINUATION_PROBABILITY,
            ),
            "exhaustion_probability": self._feature_value(
                context,
                WHALES_FEATURES.EXHAUSTION_PROBABILITY,
            ),
            "total_notional": self._feature_value(
                context,
                WHALES_FEATURES.TOTAL_NOTIONAL,
            ),
            "liquidation_notional": self._feature_value(
                context,
                WHALES_FEATURES.LIQUIDATION_NOTIONAL,
            ),
            "trade_count": self._feature_value(context, WHALES_FEATURES.TRADE_COUNT),
            "large_trade_notional": self._feature_value(
                context,
                WHALES_FEATURES.LARGE_TRADE_NOTIONAL,
            ),
            "large_trade_zscore": self._feature_value(
                context,
                WHALES_FEATURES.LARGE_TRADE_ZSCORE,
            ),
            "reference_price": self._feature_value(
                context,
                WHALES_FEATURES.REFERENCE_PRICE,
            ),
            "confidence": self._feature_value(context, WHALES_FEATURES.CONFIDENCE),
            "timestamp": self._feature_value(context, WHALES_FEATURES.TIMESTAMP),
            "metadata": self._feature_value(context, WHALES_FEATURES.METADATA),
        }

        if not self._has_any_value(payload):
            return None

        snapshot = WhaleCompositeSnapshot(
            symbol=extract_symbol(payload) or scope.symbol,
            exchange=extract_exchange(payload) or scope.exchange,
            market_type=extract_market_type(payload) or scope.market_type,
            timeframe=extract_timeframe(payload, scope.timeframe),
            exchange_symbol=extract_exchange_symbol(payload) or scope.exchange_symbol,
            pressure={},
            activity={},
            large_trade={},
            cluster={},
            cluster_update={},
            cluster_exhaustion={},
            liquidation_context={},
            dominant_side=extract_dominant_side(payload),
            whale_side=extract_whale_side(payload),
            liquidation_side=extract_liquidation_side(payload),
            exhausted_side=extract_exhausted_side(payload),
            cluster_side=extract_cluster_side(payload),
            imbalance_ratio=to_float(payload.get("imbalance_ratio"), 0.0) or 0.0,
            pressure_score=to_float(payload.get("pressure_score"), 0.0) or 0.0,
            context_strength=to_float(payload.get("context_strength"), 0.0) or 0.0,
            cluster_score=to_float(payload.get("cluster_score"), None),
            continuation_probability=to_float(
                payload.get("continuation_probability"),
                None,
            ),
            exhaustion_probability=to_float(
                payload.get("exhaustion_probability"),
                None,
            ),
            total_notional=to_float(payload.get("total_notional"), 0.0) or 0.0,
            liquidation_notional=to_float(
                payload.get("liquidation_notional"),
                0.0,
            ) or 0.0,
            trade_count=to_int(payload.get("trade_count"), 0) or 0,
            large_trade_notional=to_float(
                payload.get("large_trade_notional"),
                0.0,
            ) or 0.0,
            large_trade_zscore=to_float(payload.get("large_trade_zscore"), None),
            reference_price=to_float(payload.get("reference_price"), None),
            confidence=to_float(payload.get("confidence"), 0.0) or 0.0,
            timestamp=parse_datetime(payload.get("timestamp")) or context.timestamp,
            source="context.features",
            metadata={
                **as_dict(payload.get("metadata")),
                "scope": scope.to_dict(),
                "source": "context.features",
            },
        )

        if snapshot.has_minimum_data():
            return snapshot

        return None

    # ------------------------------------------------------------------
    # Signal builder
    # ------------------------------------------------------------------

    def build_whale_signal(
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
        trigger_type: TriggerType = TriggerType.PRIMARY,
        origin: SignalOrigin = SignalOrigin.SINGLE_STRATEGY,
        status: SignalStatus = SignalStatus.NEW,
    ) -> StrategySignal:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy.build_whale_signal")
        if not is_directional_side(side):
            raise StrategyEvaluationError(
                f"{self.strategy_name}: whale signal side must be LONG or SHORT"
            )

        scope = self.whales_scope(context)

        signal_metadata = dict(metadata or {})
        signal_metadata.setdefault("domain", FeatureSource.WHALES.value)
        signal_metadata.setdefault("whales_strategy_version", "2.0.0")
        signal_metadata.setdefault(
            "order_intent",
            self.whales_config.default_order_intent.value,
        )
        signal_metadata.setdefault(
            "margin_mode",
            self.whales_config.default_margin_mode.value,
        )
        signal_metadata.setdefault(
            "market_type",
            self.whales_config.default_market_type.value,
        )
        signal_metadata.setdefault(
            "tier",
            self.whales_config.default_trade_tier.value,
        )

        if self.whales_config.requested_leverage is not None:
            signal_metadata.setdefault(
                "requested_leverage",
                float(self.whales_config.requested_leverage),
            )

        if self.whales_config.max_slippage_bps is not None:
            signal_metadata.setdefault(
                "max_slippage_bps",
                float(self.whales_config.max_slippage_bps),
            )

        if self.whales_config.entry_timeout_seconds is not None:
            signal_metadata.setdefault(
                "entry_timeout_seconds",
                int(self.whales_config.entry_timeout_seconds),
            )

        if self.whales_config.max_holding_seconds is not None:
            signal_metadata.setdefault(
                "max_holding_seconds",
                int(self.whales_config.max_holding_seconds),
            )

        if self.whales_config.attach_scope_metadata:
            signal_metadata.setdefault("scope", scope.to_dict())

        if self.whales_config.attach_whale_context_metadata:
            signal_metadata.setdefault(
                "whale_context",
                self.whale_context_metadata(context),
            )

        if self.whales_config.metadata:
            signal_metadata.setdefault(
                "whales_config_metadata",
                serialize_for_metadata(self.whales_config.metadata),
            )

        final_reasons = list(
            dict.fromkeys(
                [
                    "whales_strategy_signal",
                    *(reasons or []),
                ]
            )
        )
        final_confirmations = list(dict.fromkeys(confirmations or []))
        final_features = list(
            dict.fromkeys(source_features or base_whales_source_features())
        )

        signal = self.build_signal(
            context=context,
            side=side,
            confidence=confidence,
            score=score,
            setup_type=setup_type or self.default_setup_type,
            reasons=final_reasons,
            confirmations=final_confirmations,
            source_features=final_features,
            metadata=serialize_for_metadata(signal_metadata),
            trigger_type=trigger_type,
            origin=origin,
            priority=priority,
            status=status,
        )

        signal.validate()
        return signal

    def whale_context_metadata(
        self,
        context: StrategyContext,
    ) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy.whale_context_metadata")
        metadata: dict[str, Any] = {}

        snapshot = self.resolve_whale_snapshot(context)
        if snapshot is not None:
            metadata["snapshot"] = snapshot.to_dict()

            if self.whales_config.attach_raw_payload_metadata:
                metadata["raw"] = {
                    "pressure": snapshot.pressure,
                    "activity": snapshot.activity,
                    "large_trade": snapshot.large_trade,
                    "cluster": snapshot.cluster,
                    "cluster_update": snapshot.cluster_update,
                    "cluster_exhaustion": snapshot.cluster_exhaustion,
                    "liquidation_context": snapshot.liquidation_context,
                }

            if self.whales_config.attach_feature_validation_metadata:
                metadata["feature_validations"] = {
                    name: validation.to_dict()
                    for name, validation in snapshot.feature_validations.items()
                }

        metadata["feature_values"] = {
            "dominant_side": self.whales_path(context, "dominant_side", None),
            "whale_side": self.whales_path(context, "whale_side", None),
            "liquidation_side": self.whales_path(context, "liquidation_side", None),
            "exhausted_side": self.whales_path(context, "exhausted_side", None),
            "cluster_side": self.whales_path(context, "cluster_side", None),
            "imbalance_ratio": self.whales_path(context, "imbalance_ratio", None),
            "pressure_score": self.whales_path(context, "pressure_score", None),
            "context_strength": self.whales_path(context, "context_strength", None),
            "cluster_score": self.whales_path(context, "cluster_score", None),
            "continuation_probability": self.whales_path(
                context,
                "continuation_probability",
                None,
            ),
            "exhaustion_probability": self.whales_path(
                context,
                "exhaustion_probability",
                None,
            ),
            "total_notional": self.whales_path(context, "total_notional", None),
            "liquidation_notional": self.whales_path(
                context,
                "liquidation_notional",
                None,
            ),
            "large_trade_notional": self.whales_path(
                context,
                "large_trade_notional",
                None,
            ),
            "large_trade_zscore": self.whales_path(
                context,
                "large_trade_zscore",
                None,
            ),
        }

        return serialize_for_metadata(metadata)

    # ------------------------------------------------------------------
    # Shared quality helpers for concrete strategies
    # ------------------------------------------------------------------

    def whale_quality_rejection_reason(
        self,
        *,
        snapshot: WhaleCompositeSnapshot,
        score: float,
        confidence: float,
        required_inputs: tuple[str, ...] = (),
    ) -> str | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy.whale_quality_rejection_reason")
        return whale_quality_filter_reason(
            inputs=snapshot.inputs(),
            min_score=self.whales_config.min_score,
            min_confidence=self.whales_config.min_confidence,
            score=score,
            confidence=confidence,
            required_keys=required_inputs,
        )

    def snapshot_freshness_score(
        self,
        snapshot: WhaleCompositeSnapshot,
        *,
        now: datetime | None = None,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy.snapshot_freshness_score")
        return freshness_score(
            event_time=snapshot.timestamp,
            now=now,
            stale_after_seconds=self.whales_config.stale_feature_max_age_seconds,
        )

    # ------------------------------------------------------------------
    # Internal extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_value(context: StrategyContext, feature_name: str) -> Any:
        _strategy_logger = logging.getLogger(__name__ + ".WhalesTradingStrategy._feature_value")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy._feature_value")
        if not isinstance(feature_name, str) or not feature_name.strip():
            return None

        if not context.has_feature(feature_name):
            return None

        value = context.get_feature(feature_name)

        if isinstance(value, FeatureSnapshot):
            return value.value

        return value

    @staticmethod
    def _has_any_value(value: Any) -> bool:
        _strategy_logger = logging.getLogger(__name__ + ".WhalesTradingStrategy._has_any_value")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy._has_any_value")
        if value is None:
            return False

        if isinstance(value, Mapping):
            return any(
                WhalesTradingStrategy._has_any_value(item)
                for item in value.values()
            )

        if isinstance(value, (list, tuple, set)):
            return any(
                WhalesTradingStrategy._has_any_value(item)
                for item in value
            )

        return value is not None

    @staticmethod
    def _payload_age_seconds(
        *,
        event_time: datetime | None,
        now: datetime | None,
    ) -> float | None:
        _strategy_logger = logging.getLogger(__name__ + ".WhalesTradingStrategy._payload_age_seconds")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy._payload_age_seconds")
        event_ts = parse_datetime(event_time)
        now_ts = parse_datetime(now)
        if event_ts is None or now_ts is None:
            return None
        return max(0.0, (now_ts - event_ts).total_seconds())

    @staticmethod
    def _resolve_snapshot_timestamp(
        *,
        context: StrategyContext,
        payloads: Mapping[str, Mapping[str, Any]],
        fallback: datetime | None,
    ) -> datetime | None:
        _strategy_logger = logging.getLogger(__name__ + ".WhalesTradingStrategy._resolve_snapshot_timestamp")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy._resolve_snapshot_timestamp")
        timestamps: list[datetime] = []

        for payload in payloads.values():
            event_time = extract_event_time(payload)
            if event_time is not None:
                timestamps.append(event_time)

        if timestamps:
            return max(timestamps)

        return parse_datetime(fallback) or context.timestamp

    @staticmethod
    def _first_non_empty_symbol(
        payloads: Mapping[str, Mapping[str, Any]],
        fallback: str,
    ) -> str:
        _strategy_logger = logging.getLogger(__name__ + ".WhalesTradingStrategy._first_non_empty_symbol")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy._first_non_empty_symbol")
        for payload in payloads.values():
            symbol = extract_symbol(payload)
            if symbol:
                return symbol
        return normalize_symbol(fallback)

    @staticmethod
    def _first_non_empty_exchange(
        payloads: Mapping[str, Mapping[str, Any]],
        fallback: str,
    ) -> str:
        _strategy_logger = logging.getLogger(__name__ + ".WhalesTradingStrategy._first_non_empty_exchange")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy._first_non_empty_exchange")
        for payload in payloads.values():
            exchange = extract_exchange(payload)
            if exchange:
                return exchange
        return normalize_exchange(fallback) or "unknown"

    @staticmethod
    def _first_non_empty_market_type(
        payloads: Mapping[str, Mapping[str, Any]],
        fallback: str,
    ) -> str:
        _strategy_logger = logging.getLogger(__name__ + ".WhalesTradingStrategy._first_non_empty_market_type")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy._first_non_empty_market_type")
        for payload in payloads.values():
            market_type = extract_market_type(payload)
            if market_type:
                return market_type
        return normalize_market_type(fallback) or "unknown"

    @staticmethod
    def _first_non_empty_timeframe(
        payloads: Mapping[str, Mapping[str, Any]],
        fallback: str,
    ) -> str:
        _strategy_logger = logging.getLogger(__name__ + ".WhalesTradingStrategy._first_non_empty_timeframe")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy._first_non_empty_timeframe")
        for payload in payloads.values():
            timeframe = extract_timeframe(payload, "")
            if timeframe:
                return timeframe
        return str(fallback or Timeframe.M1.value).lower()

    @staticmethod
    def _first_non_empty_exchange_symbol(
        payloads: Mapping[str, Mapping[str, Any]],
        fallback: str | None,
    ) -> str | None:
        _strategy_logger = logging.getLogger(__name__ + ".WhalesTradingStrategy._first_non_empty_exchange_symbol")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy._first_non_empty_exchange_symbol")
        for payload in payloads.values():
            exchange_symbol = extract_exchange_symbol(payload)
            if exchange_symbol:
                return exchange_symbol
        return fallback

    @staticmethod
    def _resolve_dominant_side(
        payloads: Mapping[str, Mapping[str, Any]],
    ) -> str:
        _strategy_logger = logging.getLogger(__name__ + ".WhalesTradingStrategy._resolve_dominant_side")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy._resolve_dominant_side")
        for key in (
            "pressure",
            "activity",
            "large_trade",
            "cluster_update",
            "cluster",
        ):
            side = extract_dominant_side(payloads.get(key) or {})
            if side not in {"unknown", ""}:
                return side
        return "unknown"

    @staticmethod
    def _resolve_whale_side(
        payloads: Mapping[str, Mapping[str, Any]],
    ) -> str:
        _strategy_logger = logging.getLogger(__name__ + ".WhalesTradingStrategy._resolve_whale_side")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy._resolve_whale_side")
        for key in (
            "liquidation_context",
            "pressure",
            "activity",
            "large_trade",
            "cluster_update",
            "cluster",
        ):
            side = extract_whale_side(payloads.get(key) or {})
            if side not in {"unknown", ""}:
                return side
        return "unknown"

    @staticmethod
    def _resolve_liquidation_side(
        payloads: Mapping[str, Mapping[str, Any]],
    ) -> str:
        _strategy_logger = logging.getLogger(__name__ + ".WhalesTradingStrategy._resolve_liquidation_side")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy._resolve_liquidation_side")
        for key in ("liquidation_context", "cluster_exhaustion", "cluster_update"):
            side = extract_liquidation_side(payloads.get(key) or {})
            if side not in {"unknown", ""}:
                return side
        return "unknown"

    @staticmethod
    def _resolve_total_notional(
        payloads: Mapping[str, Mapping[str, Any]],
    ) -> float:
        _strategy_logger = logging.getLogger(__name__ + ".WhalesTradingStrategy._resolve_total_notional")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy._resolve_total_notional")
        values = [
            extract_total_notional(payloads.get("activity") or {}),
            extract_notional(payloads.get("pressure") or {}),
            extract_notional(payloads.get("cluster") or {}),
            extract_notional(payloads.get("cluster_update") or {}),
            extract_notional(payloads.get("cluster_exhaustion") or {}),
            extract_liquidation_notional(payloads.get("liquidation_context") or {}),
            extract_large_trade_notional(payloads.get("large_trade") or {}),
        ]
        return max(values or [0.0])

    @staticmethod
    def _resolve_trade_count(
        payloads: Mapping[str, Mapping[str, Any]],
    ) -> int:
        _strategy_logger = logging.getLogger(__name__ + ".WhalesTradingStrategy._resolve_trade_count")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy._resolve_trade_count")
        values = [
            extract_trade_count(payloads.get("activity") or {}),
            extract_trade_count(payloads.get("pressure") or {}),
            extract_trade_count(payloads.get("large_trade") or {}),
            extract_trade_count(payloads.get("cluster") or {}),
            extract_trade_count(payloads.get("cluster_update") or {}),
        ]
        return max(values or [0])

    @staticmethod
    def _resolve_reference_price(
        payloads: Mapping[str, Mapping[str, Any]],
    ) -> float | None:
        _strategy_logger = logging.getLogger(__name__ + ".WhalesTradingStrategy._resolve_reference_price")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy._resolve_reference_price")
        for key in (
            "large_trade",
            "activity",
            "pressure",
            "liquidation_context",
            "cluster",
            "cluster_update",
            "cluster_exhaustion",
        ):
            price = extract_reference_price(payloads.get(key) or {})
            if price is not None and price > 0:
                return price
        return None

    @staticmethod
    def _resolve_confidence(
        payloads: Mapping[str, Mapping[str, Any]],
    ) -> float:
        _strategy_logger = logging.getLogger(__name__ + ".WhalesTradingStrategy._resolve_confidence")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhalesTradingStrategy._resolve_confidence")
        candidates = [
            extract_pressure_score(payloads.get("pressure") or {}),
            extract_context_strength(payloads.get("liquidation_context") or {}),
            resolve_cluster_score(payloads),
            resolve_continuation_probability(payloads),
            resolve_exhaustion_probability(payloads),
        ]

        valid = [
            unit_score(value)
            for value in candidates
            if value is not None
        ]

        if not valid:
            return 0.0

        return sum(valid) / len(valid)


# Backward-compatible aliases while concrete whale strategies are migrated.
WhaleStrategyBase = WhalesTradingStrategy
WhaleStrategyConfig = WhalesStrategyConfig