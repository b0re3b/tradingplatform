# trading_system/strategy/strategies/spoofing/base.py

from __future__ import annotations
import logging

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any

from analytics.spoofing.enums import (
    SpoofingComponent,
    SpoofingPattern,
    SpoofingSeverity,
    SpoofingSide,
    SpoofingStatus,
    SpoofingType,
)
from core.event_bus import EventBus
from core.scheduler import Scheduler
from .utils import (
    as_dict,
    detector_agreement_ratio,
    detector_average_confidence,
    detector_count,
    detector_payload,
    detector_passed,
    detector_score,
    extract_analytics_metadata,
    extract_cancel_to_fill_ratio,
    extract_confidence,
    extract_detector_results,
    extract_distance_from_mid_bps,
    extract_event_time,
    extract_features,
    extract_fill_ratio,
    extract_layer_count,
    extract_layer_price_span_bps,
    extract_lifetime_ms,
    extract_price_level,
    extract_pressure_flip_strength,
    extract_price_reaction_bps,
    extract_pull_ratio,
    extract_pulled_notional,
    extract_score,
    extract_score_breakdown,
    extract_signal_payload,
    extract_spoofing_pattern,
    extract_spoofing_severity,
    extract_spoofing_side,
    extract_spoofing_status,
    extract_spoofing_type,
    extract_wall_id,
    extract_wall_notional,
    first_non_empty,
    get_path,
    is_directional_side,
    normalize_signal_payload,
    parse_datetime,
    serialize_for_metadata,
    spoofing_domain,
    spoofing_item,
    spoofing_path,
    spoofing_side_to_signal_side,
    to_bool,
    to_float,
    to_int,
    to_str,
    unit_score,
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
class SpoofingFeatureNames:
    """
    Stable spoofing feature names expected in StrategyContext.

    StrategyContextBuilder / SignalNormalizer should populate these from
    analytics.spoofing.* payloads. Concrete strategies may also read equivalent
    values from FeatureSource.SPOOFING domain_data aliases.
    """
    _logger = logging.getLogger(__name__ + ".SpoofingFeatureNames")

    COMPOSITE: str = "spoofing.composite"
    SIGNAL: str = "spoofing.signal"

    FEATURES: str = "spoofing.features"
    DETECTOR_RESULTS: str = "spoofing.detector_results"
    SCORE_BREAKDOWN: str = "spoofing.score_breakdown"
    ANALYTICS_METADATA: str = "spoofing.analytics_metadata"

    SPOOFING_TYPE: str = "spoofing.type"
    PATTERN: str = "spoofing.pattern"
    SIDE: str = "spoofing.side"
    SEVERITY: str = "spoofing.severity"
    STATUS: str = "spoofing.status"

    SCORE: str = "spoofing.score"
    CONFIDENCE: str = "spoofing.confidence"

    PRICE_LEVEL: str = "spoofing.price_level"
    WALL_ID: str = "spoofing.wall_id"
    EVENT_TIME: str = "spoofing.event_time"

    PULL_RATIO: str = "spoofing.features.pull_ratio"
    FILL_RATIO: str = "spoofing.features.fill_ratio"
    PRICE_REACTION_BPS: str = "spoofing.features.price_reaction_bps"
    SIGNED_PRICE_REACTION_BPS: str = (
        "spoofing.features.signed_price_reaction_bps"
    )
    LIFETIME_MS: str = "spoofing.features.lifetime_ms"
    WALL_NOTIONAL: str = "spoofing.features.wall_notional"
    PULLED_NOTIONAL: str = "spoofing.features.pulled_notional"
    CANCEL_TO_FILL_RATIO: str = "spoofing.features.cancel_to_fill_ratio"
    DISTANCE_FROM_MID_BPS: str = "spoofing.features.distance_from_mid_bps"

    LAYER_COUNT: str = "spoofing.features.layer_count"
    LAYER_PRICE_SPAN_BPS: str = "spoofing.features.layer_price_span_bps"
    PRESSURE_FLIP_STRENGTH: str = (
        "spoofing.features.pressure_flip_strength"
    )

    @classmethod
    def all(cls) -> set[str]:
        _strategy_logger = getattr(cls, "_logger", None) or logging.getLogger(__name__ + ".SpoofingFeatureNames")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingFeatureNames.all")
        instance = cls()
        return {
            getattr(instance, item.name)
            for item in fields(cls)
            if isinstance(getattr(instance, item.name), str)
            and getattr(instance, item.name).strip()
        }


SPOOFING_FEATURES = SpoofingFeatureNames()


# =============================================================================
# Scope
# =============================================================================


@dataclass(frozen=True, slots=True)
class SpoofingStrategyScope:
    """
    Futures spoofing scope used only for metadata and normalization.

    Concrete strategies still make decisions from StrategyContext.
    """
    _logger = logging.getLogger(__name__ + ".SpoofingStrategyScope")

    exchange: str
    market_type: str
    symbol: str
    timeframe: str
    exchange_symbol: str | None = None

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingStrategyScope.__post_init__")
        exchange = str(self.exchange or "unknown").strip().lower()
        market_type = str(
            self.market_type or StrategyMarketType.USDM_FUTURES.value
        ).strip()
        symbol = str(self.symbol or "").strip().upper()
        timeframe = str(self.timeframe or Timeframe.M1.value).strip().lower()
        exchange_symbol = str(self.exchange_symbol or symbol).strip().upper()

        if not symbol:
            raise StrategyEvaluationError(
                "SpoofingStrategyScope.symbol cannot be empty"
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
            _strategy_logger.debug("Entering SpoofingStrategyScope.key")
        return f"{self.exchange}:{self.market_type}:{self.symbol}:{self.timeframe}"

    @property
    def legacy_key(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingStrategyScope.legacy_key")
        return f"{self.symbol}:{self.exchange}"

    def to_dict(self) -> dict[str, str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingStrategyScope.to_dict")
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
class SpoofingStrategyConfig:
    """
    Domain config shared by concrete spoofing strategies.

    This config is intentionally stateless. It does not contain EventBus topics,
    setup TTLs, active setup indexes, cleanup loops or trade confirmation logic.
    Those concerns must stay outside concrete strategy classes.
    """
    _logger = logging.getLogger(__name__ + ".SpoofingStrategyConfig")

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

    min_score: float = 0.65
    min_confidence: float = 0.55
    allowed_severities: tuple[SpoofingSeverity, ...] = field(default_factory=tuple)

    require_score_passed: bool = False
    min_detector_count: int = 1
    min_agreement_ratio: float = 0.0
    min_average_confidence: float = 0.0

    stale_feature_max_age_seconds: float | None = None

    requested_leverage: float | None = None
    max_slippage_bps: float | None = None
    entry_timeout_seconds: int | None = None
    max_holding_seconds: int | None = None

    attach_spoofing_context_metadata: bool = True
    attach_scope_metadata: bool = True
    attach_feature_values_metadata: bool = True
    attach_detector_metadata: bool = True

    tag_spoofing: str = "spoofing"
    tag_reversal: str = "reversal"
    tag_continuation: str = "continuation"
    tag_fake_liquidity: str = "fake_liquidity"
    tag_liquidity_trap: str = "liquidity_trap"
    tag_order_pull: str = "order_pull"
    tag_pressure_bluff: str = "pressure_bluff"
    tag_layering: str = "layering"
    tag_composite: str = "composite"
    tag_absorption: str = "absorption"

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingStrategyConfig.validate")
        bounded = {
            "min_score": self.min_score,
            "min_confidence": self.min_confidence,
            "min_agreement_ratio": self.min_agreement_ratio,
            "min_average_confidence": self.min_average_confidence,
        }
        for name, value in bounded.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{name} must be between 0.0 and 1.0")

        if self.min_detector_count < 0:
            raise StrategyConfigError("min_detector_count must be >= 0")

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

        for severity in self.allowed_severities:
            if not isinstance(severity, SpoofingSeverity):
                raise StrategyConfigError(
                    "allowed_severities must contain SpoofingSeverity values"
                )

        for attr in (
            "tag_spoofing",
            "tag_reversal",
            "tag_continuation",
            "tag_fake_liquidity",
            "tag_liquidity_trap",
            "tag_order_pull",
            "tag_pressure_bluff",
            "tag_layering",
            "tag_composite",
            "tag_absorption",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")


# =============================================================================
# Composite snapshot
# =============================================================================


@dataclass(slots=True)
class SpoofingCompositeSnapshot:
    """
    Strategy-level normalized view over analytics.spoofing signal.

    This is intentionally not an analytics model. It is a strategy-side
    projection that lets concrete spoofing strategies consume spoofing type,
    pattern, side, features, detector results and score breakdown through one
    stable contract.
    """
    _logger = logging.getLogger(__name__ + ".SpoofingCompositeSnapshot")

    exchange: str
    market_type: str
    symbol: str
    timeframe: str

    exchange_symbol: str | None = None
    timestamp: datetime | None = None
    source: str = "unknown"

    spoofing_type: SpoofingType | None = None
    pattern: SpoofingPattern | None = None
    side: SpoofingSide | None = None
    severity: SpoofingSeverity | None = None
    status: SpoofingStatus | None = None

    score: float = 0.0
    confidence: float = 0.0

    price_level: float | None = None
    wall_id: str | None = None

    pull_ratio: float = 0.0
    fill_ratio: float = 0.0
    price_reaction_bps: float = 0.0
    signed_price_reaction_bps: float = 0.0
    lifetime_ms: float = 0.0
    wall_notional: float = 0.0
    pulled_notional: float = 0.0
    cancel_to_fill_ratio: float = 0.0
    distance_from_mid_bps: float = 0.0

    layer_count: int = 0
    layer_price_span_bps: float = 0.0
    pressure_flip_strength: float = 0.0

    features: dict[str, Any] = field(default_factory=dict)
    detector_results: dict[str, Any] = field(default_factory=dict)
    score_breakdown: dict[str, Any] = field(default_factory=dict)
    analytics_metadata: dict[str, Any] = field(default_factory=dict)
    raw_signal: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingCompositeSnapshot.__post_init__")
        exchange = str(self.exchange or "unknown").strip().lower()
        market_type = str(
            self.market_type or StrategyMarketType.USDM_FUTURES.value
        ).strip()
        symbol = str(self.symbol or "").strip().upper()
        timeframe = str(self.timeframe or Timeframe.M1.value).strip().lower()
        exchange_symbol = str(self.exchange_symbol or symbol).strip().upper()

        if not symbol:
            raise StrategyEvaluationError(
                "SpoofingCompositeSnapshot.symbol cannot be empty"
            )

        self.exchange = exchange
        self.market_type = market_type
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange_symbol = exchange_symbol

        self.timestamp = parse_datetime(self.timestamp)
        self.score = unit_score(self.score)
        self.confidence = unit_score(self.confidence)

        self.price_level = to_float(self.price_level)
        self.wall_id = to_str(self.wall_id)

        self.pull_ratio = unit_score(self.pull_ratio)
        self.fill_ratio = unit_score(self.fill_ratio)
        self.cancel_to_fill_ratio = unit_score(self.cancel_to_fill_ratio)
        self.pressure_flip_strength = unit_score(self.pressure_flip_strength)

        self.price_reaction_bps = max(
            0.0,
            float(to_float(self.price_reaction_bps, 0.0) or 0.0),
        )
        self.signed_price_reaction_bps = float(
            to_float(self.signed_price_reaction_bps, 0.0) or 0.0
        )
        self.lifetime_ms = max(float(to_float(self.lifetime_ms, 0.0) or 0.0), 0.0)
        self.wall_notional = max(float(to_float(self.wall_notional, 0.0) or 0.0), 0.0)
        self.pulled_notional = max(
            float(to_float(self.pulled_notional, 0.0) or 0.0),
            0.0,
        )
        self.distance_from_mid_bps = max(
            float(to_float(self.distance_from_mid_bps, 0.0) or 0.0),
            0.0,
        )
        self.layer_count = max(int(to_int(self.layer_count, 0) or 0), 0)
        self.layer_price_span_bps = max(
            float(to_float(self.layer_price_span_bps, 0.0) or 0.0),
            0.0,
        )

        self.features = as_dict(self.features)
        self.detector_results = as_dict(self.detector_results)
        self.score_breakdown = as_dict(self.score_breakdown)
        self.analytics_metadata = as_dict(self.analytics_metadata)
        self.raw_signal = as_dict(self.raw_signal)
        self.metadata = as_dict(self.metadata)

    @property
    def scope(self) -> SpoofingStrategyScope:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingCompositeSnapshot.scope")
        return SpoofingStrategyScope(
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
            _strategy_logger.debug("Entering SpoofingCompositeSnapshot.scope_key")
        return self.scope.key

    @property
    def signal_side(self) -> SignalSide:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingCompositeSnapshot.signal_side")
        return spoofing_side_to_signal_side(self.side)

    @property
    def detector_count(self) -> int:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingCompositeSnapshot.detector_count")
        return detector_count(self.raw_signal or self.to_signal_payload())

    @property
    def detector_agreement_ratio(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingCompositeSnapshot.detector_agreement_ratio")
        return detector_agreement_ratio(self.raw_signal or self.to_signal_payload())

    @property
    def detector_average_confidence(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingCompositeSnapshot.detector_average_confidence")
        return detector_average_confidence(self.raw_signal or self.to_signal_payload())

    def detector_payload(self, component: SpoofingComponent | str) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingCompositeSnapshot.detector_payload")
        return detector_payload(self.raw_signal or self.to_signal_payload(), component)

    def detector_score(self, component: SpoofingComponent | str) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingCompositeSnapshot.detector_score")
        return detector_score(self.raw_signal or self.to_signal_payload(), component)

    def detector_passed(self, component: SpoofingComponent | str) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingCompositeSnapshot.detector_passed")
        return detector_passed(self.raw_signal or self.to_signal_payload(), component)

    def has_detector(self, component: SpoofingComponent | str) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingCompositeSnapshot.has_detector")
        return bool(self.detector_payload(component))

    def to_signal_payload(self) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingCompositeSnapshot.to_signal_payload")
        return {
            "spoofing_type": self.spoofing_type,
            "pattern": self.pattern,
            "side": self.side,
            "severity": self.severity,
            "status": self.status,
            "score": self.score,
            "confidence": self.confidence,
            "price_level": self.price_level,
            "wall_id": self.wall_id,
            "features": self.features,
            "detector_results": self.detector_results,
            "score_breakdown": self.score_breakdown,
            "analytics_metadata": self.analytics_metadata,
            "metadata": self.analytics_metadata,
        }

    def has_minimum_data(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingCompositeSnapshot.has_minimum_data")
        return (
            self.spoofing_type is not None
            or self.pattern is not None
            or self.side is not None
            or self.score > 0.0
            or self.confidence > 0.0
            or bool(self.features)
            or bool(self.detector_results)
        )

    def to_dict(self) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingCompositeSnapshot.to_dict")
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "exchange_symbol": self.exchange_symbol,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "source": self.source,
            "spoofing_type": (
                self.spoofing_type.value if self.spoofing_type else None
            ),
            "pattern": self.pattern.value if self.pattern else None,
            "side": self.side.value if self.side else None,
            "severity": self.severity.value if self.severity else None,
            "status": self.status.value if self.status else None,
            "score": self.score,
            "confidence": self.confidence,
            "price_level": self.price_level,
            "wall_id": self.wall_id,
            "pull_ratio": self.pull_ratio,
            "fill_ratio": self.fill_ratio,
            "price_reaction_bps": self.price_reaction_bps,
            "signed_price_reaction_bps": self.signed_price_reaction_bps,
            "lifetime_ms": self.lifetime_ms,
            "wall_notional": self.wall_notional,
            "pulled_notional": self.pulled_notional,
            "cancel_to_fill_ratio": self.cancel_to_fill_ratio,
            "distance_from_mid_bps": self.distance_from_mid_bps,
            "layer_count": self.layer_count,
            "layer_price_span_bps": self.layer_price_span_bps,
            "pressure_flip_strength": self.pressure_flip_strength,
            "features": serialize_for_metadata(self.features),
            "detector_results": serialize_for_metadata(self.detector_results),
            "score_breakdown": serialize_for_metadata(self.score_breakdown),
            "analytics_metadata": serialize_for_metadata(self.analytics_metadata),
            "raw_signal": serialize_for_metadata(self.raw_signal),
            "metadata": serialize_for_metadata(self.metadata),
            "scope": self.scope.to_dict(),
            "scope_key": self.scope_key,
            "signal_side": self.signal_side.value,
            "detector_count": self.detector_count,
            "detector_agreement_ratio": self.detector_agreement_ratio,
            "detector_average_confidence": self.detector_average_confidence,
        }


# =============================================================================
# Base spoofing strategy
# =============================================================================


class SpoofingTradingStrategy(TradingStrategy):
    """
    Base class for concrete strategy/strategies/spoofing/* classes.

    Responsibilities:
    - read spoofing analytics data from StrategyContext only;
    - provide helpers for spoofing signal/features/detectors/scoring;
    - build internal StrategySignal objects through TradingStrategy helpers;
    - attach spoofing/futures metadata for SignalProcessor.

    Forbidden:
    - no direct analytics.spoofing.* EventBus subscriptions;
    - no market.trades.updated subscriptions;
    - no setup lifecycle state;
    - no pending/confirmed/expired setup indexes;
    - no direct signal.generated emission;
    - no RiskManager / Execution calls;
    - no raw market data reads.
    """
    _logger = logging.getLogger(__name__ + ".SpoofingTradingStrategy")

    component_namespace = "strategy.spoofing"
    category: StrategyCategory = StrategyCategory.SPOOFING
    default_setup_type: SetupType = SetupType.UNKNOWN
    default_timeframe: Timeframe = Timeframe.M1

    feature_names = SPOOFING_FEATURES

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        spoofing_config: SpoofingStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy.__init__")
        self.spoofing_config = spoofing_config or SpoofingStrategyConfig()
        self.spoofing_config.validate()

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
            _strategy_logger.debug("Entering SpoofingTradingStrategy.validate_config")
        super().validate_config()
        self.spoofing_config.validate()

    # ------------------------------------------------------------------
    # Context / domain access
    # ------------------------------------------------------------------

    def spoofing_domain(self, context: StrategyContext) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy.spoofing_domain")
        self.validate_context(context)
        return spoofing_domain(context)

    def spoofing_item(
        self,
        context: StrategyContext,
        key: str,
        default: Any = None,
    ) -> Any:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy.spoofing_item")
        self.validate_context(context)
        return spoofing_item(context, key, default)

    def spoofing_path(
        self,
        context: StrategyContext,
        path: str,
        default: Any = None,
    ) -> Any:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy.spoofing_path")
        self.validate_context(context)

        if not isinstance(path, str) or not path.strip():
            raise StrategyEvaluationError("spoofing path cannot be empty")

        return spoofing_path(context, path, default)

    def spoofing_float(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float | None = None,
    ) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy.spoofing_float")
        return to_float(self.spoofing_path(context, path, default), default)

    def spoofing_int(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: int | None = None,
    ) -> int | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy.spoofing_int")
        return to_int(self.spoofing_path(context, path, default), default)

    def spoofing_score(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float = 0.0,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy.spoofing_score")
        return unit_score(self.spoofing_path(context, path, default), default)

    def spoofing_bool(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: bool = False,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy.spoofing_bool")
        return to_bool(self.spoofing_path(context, path, default), default)

    def spoofing_str(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: str | None = None,
    ) -> str | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy.spoofing_str")
        return to_str(self.spoofing_path(context, path, default), default)

    def spoofing_datetime(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: datetime | None = None,
    ) -> datetime | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy.spoofing_datetime")
        return parse_datetime(self.spoofing_path(context, path, default))

    def spoofing_feature_snapshot(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> FeatureSnapshot | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy.spoofing_feature_snapshot")
        self.validate_context(context)

        if not isinstance(feature_name, str) or not feature_name.strip():
            raise StrategyEvaluationError("feature_name cannot be empty")

        features_map = getattr(context, "features", None)
        if isinstance(features_map, Mapping):
            raw = features_map.get(feature_name)
            if isinstance(raw, FeatureSnapshot):
                return raw

        return None

    def spoofing_feature_age_seconds(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy.spoofing_feature_age_seconds")
        snapshot = self.spoofing_feature_snapshot(context, feature_name)
        if snapshot is None:
            return None
        return snapshot.age_seconds(context.timestamp)

    def spoofing_feature_is_stale(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy.spoofing_feature_is_stale")
        max_age = self.spoofing_config.stale_feature_max_age_seconds
        if max_age is None:
            return False

        age = self.spoofing_feature_age_seconds(context, feature_name)
        if age is None:
            return False

        return age > max_age

    def has_any_spoofing_data(
        self,
        context: StrategyContext,
        feature_names: tuple[str, ...] = (),
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy.has_any_spoofing_data")
        self.validate_context(context)

        if self.spoofing_domain(context):
            return True

        return any(context.has_feature(name) for name in feature_names)

    def has_stale_spoofing_features(
        self,
        context: StrategyContext,
        feature_names: tuple[str, ...] | None = None,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy.has_stale_spoofing_features")
        names = feature_names or tuple(self.required_features())

        return any(
            self.spoofing_feature_is_stale(context, feature_name)
            for feature_name in names
        )

    # ------------------------------------------------------------------
    # Scope
    # ------------------------------------------------------------------

    def spoofing_scope(self, context: StrategyContext) -> SpoofingStrategyScope:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy.spoofing_scope")
        domain = self.spoofing_domain(context)

        exchange = (
            to_str(domain.get("exchange"))
            or to_str(get_path(domain, "scope.exchange"))
            or to_str(context.metadata.get("exchange"))
            or "unknown"
        )
        market_type = (
            to_str(domain.get("market_type"))
            or to_str(get_path(domain, "scope.market_type"))
            or to_str(context.metadata.get("market_type"))
            or self.spoofing_config.default_market_type.value
        )
        exchange_symbol = (
            to_str(domain.get("exchange_symbol"))
            or to_str(get_path(domain, "scope.exchange_symbol"))
            or to_str(context.metadata.get("exchange_symbol"))
            or context.symbol
        )

        timeframe_value = getattr(context.timeframe, "value", context.timeframe)

        return SpoofingStrategyScope(
            exchange=exchange,
            market_type=market_type,
            symbol=context.symbol,
            timeframe=str(timeframe_value or Timeframe.M1.value),
            exchange_symbol=exchange_symbol,
        )

    # ------------------------------------------------------------------
    # Snapshot resolution
    # ------------------------------------------------------------------

    def resolve_spoofing_snapshot(
        self,
        context: StrategyContext,
    ) -> SpoofingCompositeSnapshot | None:
        """
        Resolve normalized spoofing snapshot from StrategyContext only.

        Preferred:
            FeatureSource.SPOOFING domain["signal"] / domain["composite"]

        Fallback:
            individual spoofing.* features.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy.resolve_spoofing_snapshot")
        self.validate_context(context)

        scope = self.spoofing_scope(context)

        for candidate in (
            self.spoofing_item(context, "signal"),
            self.spoofing_item(context, "composite"),
            self.spoofing_path(context, "signal", None),
            self.spoofing_path(context, "composite", None),
        ):
            snapshot = self._coerce_spoofing_snapshot(
                candidate,
                context=context,
                scope=scope,
                source="context.domain",
            )
            if snapshot is not None and snapshot.has_minimum_data():
                return snapshot

        domain_snapshot = self._build_snapshot_from_domain(
            context=context,
            scope=scope,
        )
        if domain_snapshot is not None and domain_snapshot.has_minimum_data():
            return domain_snapshot

        feature_snapshot = self._build_snapshot_from_features(
            context=context,
            scope=scope,
        )
        if feature_snapshot is not None and feature_snapshot.has_minimum_data():
            return feature_snapshot

        return domain_snapshot or feature_snapshot

    def accepts_spoofing_snapshot(
        self,
        snapshot: SpoofingCompositeSnapshot,
    ) -> bool:
        """
        Shared base acceptance gate.

        Concrete strategies should add their own pattern-specific checks.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy.accepts_spoofing_snapshot")
        if not snapshot.has_minimum_data():
            return False

        if snapshot.score < self.spoofing_config.min_score:
            return False

        if snapshot.confidence < self.spoofing_config.min_confidence:
            return False

        if self.spoofing_config.allowed_severities:
            if snapshot.severity not in self.spoofing_config.allowed_severities:
                return False

        if snapshot.detector_count < self.spoofing_config.min_detector_count:
            return False

        if snapshot.detector_agreement_ratio < self.spoofing_config.min_agreement_ratio:
            return False

        if (
            snapshot.detector_average_confidence
            < self.spoofing_config.min_average_confidence
        ):
            return False

        if self.spoofing_config.require_score_passed:
            if not to_bool(
                snapshot.score_breakdown.get("passed")
                or snapshot.score_breakdown.get("score_passed")
                or snapshot.analytics_metadata.get("score_passed"),
                default=True,
            ):
                return False

        return True

    def _coerce_spoofing_snapshot(
        self,
        value: Any,
        *,
        context: StrategyContext,
        scope: SpoofingStrategyScope,
        source: str,
    ) -> SpoofingCompositeSnapshot | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy._coerce_spoofing_snapshot")
        if value is None:
            return None

        if isinstance(value, SpoofingCompositeSnapshot):
            return value

        payload = normalize_signal_payload(value)
        if not payload:
            return None

        return self._snapshot_from_payload(
            payload,
            context=context,
            scope=scope,
            source=source,
        )

    def _build_snapshot_from_domain(
        self,
        *,
        context: StrategyContext,
        scope: SpoofingStrategyScope,
    ) -> SpoofingCompositeSnapshot | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy._build_snapshot_from_domain")
        domain = self.spoofing_domain(context)
        if not domain:
            return None

        signal = (
            self.spoofing_item(context, "signal")
            or self.spoofing_item(context, "composite")
            or domain
        )
        payload = normalize_signal_payload(signal)
        if not payload:
            payload = dict(domain)

        return self._snapshot_from_payload(
            payload,
            context=context,
            scope=scope,
            source="context.domain",
        )

    def _build_snapshot_from_features(
        self,
        *,
        context: StrategyContext,
        scope: SpoofingStrategyScope,
    ) -> SpoofingCompositeSnapshot | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy._build_snapshot_from_features")
        payload = {
            "spoofing_type": self._feature_value(
                context,
                SPOOFING_FEATURES.SPOOFING_TYPE,
            ),
            "pattern": self._feature_value(
                context,
                SPOOFING_FEATURES.PATTERN,
            ),
            "side": self._feature_value(
                context,
                SPOOFING_FEATURES.SIDE,
            ),
            "severity": self._feature_value(
                context,
                SPOOFING_FEATURES.SEVERITY,
            ),
            "status": self._feature_value(
                context,
                SPOOFING_FEATURES.STATUS,
            ),
            "score": self._feature_value(
                context,
                SPOOFING_FEATURES.SCORE,
            ),
            "confidence": self._feature_value(
                context,
                SPOOFING_FEATURES.CONFIDENCE,
            ),
            "price_level": self._feature_value(
                context,
                SPOOFING_FEATURES.PRICE_LEVEL,
            ),
            "wall_id": self._feature_value(
                context,
                SPOOFING_FEATURES.WALL_ID,
            ),
            "event_time": self._feature_value(
                context,
                SPOOFING_FEATURES.EVENT_TIME,
            ),
            "features": {
                "pull_ratio": self._feature_value(
                    context,
                    SPOOFING_FEATURES.PULL_RATIO,
                ),
                "fill_ratio": self._feature_value(
                    context,
                    SPOOFING_FEATURES.FILL_RATIO,
                ),
                "price_reaction_bps": self._feature_value(
                    context,
                    SPOOFING_FEATURES.PRICE_REACTION_BPS,
                ),
                "signed_price_reaction_bps": self._feature_value(
                    context,
                    SPOOFING_FEATURES.SIGNED_PRICE_REACTION_BPS,
                ),
                "lifetime_ms": self._feature_value(
                    context,
                    SPOOFING_FEATURES.LIFETIME_MS,
                ),
                "wall_notional": self._feature_value(
                    context,
                    SPOOFING_FEATURES.WALL_NOTIONAL,
                ),
                "pulled_notional": self._feature_value(
                    context,
                    SPOOFING_FEATURES.PULLED_NOTIONAL,
                ),
                "cancel_to_fill_ratio": self._feature_value(
                    context,
                    SPOOFING_FEATURES.CANCEL_TO_FILL_RATIO,
                ),
                "distance_from_mid_bps": self._feature_value(
                    context,
                    SPOOFING_FEATURES.DISTANCE_FROM_MID_BPS,
                ),
                "layer_count": self._feature_value(
                    context,
                    SPOOFING_FEATURES.LAYER_COUNT,
                ),
                "layer_price_span_bps": self._feature_value(
                    context,
                    SPOOFING_FEATURES.LAYER_PRICE_SPAN_BPS,
                ),
                "pressure_flip_strength": self._feature_value(
                    context,
                    SPOOFING_FEATURES.PRESSURE_FLIP_STRENGTH,
                ),
            },
            "detector_results": self._feature_value(
                context,
                SPOOFING_FEATURES.DETECTOR_RESULTS,
            ),
            "score_breakdown": self._feature_value(
                context,
                SPOOFING_FEATURES.SCORE_BREAKDOWN,
            ),
            "analytics_metadata": self._feature_value(
                context,
                SPOOFING_FEATURES.ANALYTICS_METADATA,
            ),
        }

        if not self._has_any_value(payload):
            return None

        return self._snapshot_from_payload(
            payload,
            context=context,
            scope=scope,
            source="context.features",
        )

    def _snapshot_from_payload(
        self,
        payload: Mapping[str, Any],
        *,
        context: StrategyContext,
        scope: SpoofingStrategyScope,
        source: str,
    ) -> SpoofingCompositeSnapshot:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy._snapshot_from_payload")
        signal_payload = extract_signal_payload(payload) or dict(payload)

        features = extract_features(signal_payload)
        detector_results = extract_detector_results(signal_payload)
        score_breakdown = extract_score_breakdown(signal_payload)
        analytics_metadata = extract_analytics_metadata(signal_payload)

        timestamp = (
            extract_event_time(signal_payload)
            or parse_datetime(signal_payload.get("event_time"))
            or context.timestamp
        )

        return SpoofingCompositeSnapshot(
            exchange=scope.exchange,
            market_type=scope.market_type,
            symbol=scope.symbol,
            timeframe=scope.timeframe,
            exchange_symbol=scope.exchange_symbol,
            timestamp=timestamp,
            source=source,
            spoofing_type=extract_spoofing_type(signal_payload),
            pattern=extract_spoofing_pattern(signal_payload),
            side=extract_spoofing_side(signal_payload),
            severity=extract_spoofing_severity(signal_payload),
            status=extract_spoofing_status(signal_payload),
            score=extract_score(signal_payload),
            confidence=extract_confidence(signal_payload),
            price_level=extract_price_level(signal_payload),
            wall_id=extract_wall_id(signal_payload),
            pull_ratio=extract_pull_ratio(signal_payload),
            fill_ratio=extract_fill_ratio(signal_payload),
            price_reaction_bps=extract_price_reaction_bps(signal_payload),
            signed_price_reaction_bps=to_float(
                first_non_empty(
                    get_path(features, "signed_price_reaction_bps"),
                    get_path(features, "price_reaction_bps"),
                    get_path(analytics_metadata, "signed_price_reaction_bps"),
                    get_path(analytics_metadata, "price_reaction_bps"),
                ),
                0.0,
            )
            or 0.0,
            lifetime_ms=extract_lifetime_ms(signal_payload),
            wall_notional=extract_wall_notional(signal_payload),
            pulled_notional=extract_pulled_notional(signal_payload),
            cancel_to_fill_ratio=extract_cancel_to_fill_ratio(signal_payload),
            distance_from_mid_bps=extract_distance_from_mid_bps(signal_payload),
            layer_count=extract_layer_count(signal_payload),
            layer_price_span_bps=extract_layer_price_span_bps(signal_payload),
            pressure_flip_strength=extract_pressure_flip_strength(signal_payload),
            features=features,
            detector_results=detector_results,
            score_breakdown=score_breakdown,
            analytics_metadata=analytics_metadata,
            raw_signal=dict(signal_payload),
            metadata={
                "scope": scope.to_dict(),
                "source": source,
            },
        )

    # ------------------------------------------------------------------
    # Detector helpers
    # ------------------------------------------------------------------

    def detector_payload(
        self,
        snapshot: SpoofingCompositeSnapshot,
        component: SpoofingComponent | str,
    ) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy.detector_payload")
        return snapshot.detector_payload(component)

    def detector_score(
        self,
        snapshot: SpoofingCompositeSnapshot,
        component: SpoofingComponent | str,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy.detector_score")
        return snapshot.detector_score(component)

    def detector_passed(
        self,
        snapshot: SpoofingCompositeSnapshot,
        component: SpoofingComponent | str,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy.detector_passed")
        return snapshot.detector_passed(component)

    # ------------------------------------------------------------------
    # Signal builder
    # ------------------------------------------------------------------

    def build_spoofing_signal(
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
        """
        Build internal StrategySignal with spoofing/futures metadata.

        Final risk-ready payload conversion belongs to SignalProcessor /
        SignalBuilder, not to this domain strategy.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy.build_spoofing_signal")
        if not is_directional_side(side):
            raise StrategyEvaluationError(
                f"{self.strategy_name}: spoofing signal side must be LONG or SHORT"
            )

        scope = self.spoofing_scope(context)

        signal_metadata = dict(metadata or {})
        signal_metadata.setdefault("domain", FeatureSource.SPOOFING.value)
        signal_metadata.setdefault("spoofing_strategy_version", "2.0.0")
        signal_metadata.setdefault(
            "order_intent",
            self.spoofing_config.default_order_intent.value,
        )
        signal_metadata.setdefault(
            "margin_mode",
            self.spoofing_config.default_margin_mode.value,
        )
        signal_metadata.setdefault(
            "market_type",
            self.spoofing_config.default_market_type.value,
        )
        signal_metadata.setdefault(
            "tier",
            self.spoofing_config.default_trade_tier.value,
        )

        if self.spoofing_config.requested_leverage is not None:
            signal_metadata.setdefault(
                "requested_leverage",
                float(self.spoofing_config.requested_leverage),
            )

        if self.spoofing_config.max_slippage_bps is not None:
            signal_metadata.setdefault(
                "max_slippage_bps",
                float(self.spoofing_config.max_slippage_bps),
            )

        if self.spoofing_config.entry_timeout_seconds is not None:
            signal_metadata.setdefault(
                "entry_timeout_seconds",
                int(self.spoofing_config.entry_timeout_seconds),
            )

        if self.spoofing_config.max_holding_seconds is not None:
            signal_metadata.setdefault(
                "max_holding_seconds",
                int(self.spoofing_config.max_holding_seconds),
            )

        if self.spoofing_config.attach_scope_metadata:
            signal_metadata.setdefault("scope", scope.to_dict())

        if self.spoofing_config.attach_spoofing_context_metadata:
            signal_metadata.setdefault(
                "spoofing_context",
                self.spoofing_context_metadata(context),
            )

        if self.spoofing_config.metadata:
            signal_metadata.setdefault(
                "spoofing_config_metadata",
                serialize_for_metadata(self.spoofing_config.metadata),
            )

        final_reasons = list(
            dict.fromkeys(
                [
                    "spoofing_strategy_signal",
                    *(reasons or []),
                ]
            )
        )
        final_confirmations = list(dict.fromkeys(confirmations or []))
        final_features = list(dict.fromkeys(source_features or []))

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

    def spoofing_context_metadata(
        self,
        context: StrategyContext,
    ) -> dict[str, Any]:
        """
        Compact serialized spoofing context for StrategySignal.metadata.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy.spoofing_context_metadata")
        metadata: dict[str, Any] = {}

        snapshot = self.resolve_spoofing_snapshot(context)
        if snapshot is not None:
            metadata["snapshot"] = snapshot.to_dict()

        if self.spoofing_config.attach_feature_values_metadata:
            metadata["feature_values"] = {
                "spoofing_type": self.spoofing_path(context, "type", None),
                "pattern": self.spoofing_path(context, "pattern", None),
                "side": self.spoofing_path(context, "side", None),
                "score": self.spoofing_path(context, "score", None),
                "confidence": self.spoofing_path(context, "confidence", None),
                "pull_ratio": self.spoofing_path(
                    context,
                    "features.pull_ratio",
                    None,
                ),
                "fill_ratio": self.spoofing_path(
                    context,
                    "features.fill_ratio",
                    None,
                ),
                "price_reaction_bps": self.spoofing_path(
                    context,
                    "features.price_reaction_bps",
                    None,
                ),
                "wall_notional": self.spoofing_path(
                    context,
                    "features.wall_notional",
                    None,
                ),
                "pulled_notional": self.spoofing_path(
                    context,
                    "features.pulled_notional",
                    None,
                ),
            }

        if self.spoofing_config.attach_detector_metadata and snapshot is not None:
            metadata["detectors"] = {
                "detector_count": snapshot.detector_count,
                "agreement_ratio": snapshot.detector_agreement_ratio,
                "average_confidence": snapshot.detector_average_confidence,
                "detector_results": snapshot.detector_results,
            }

        return serialize_for_metadata(metadata)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_value(context: StrategyContext, feature_name: str) -> Any:
        _strategy_logger = logging.getLogger(__name__ + ".SpoofingTradingStrategy._feature_value")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy._feature_value")
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
        _strategy_logger = logging.getLogger(__name__ + ".SpoofingTradingStrategy._has_any_value")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpoofingTradingStrategy._has_any_value")
        if value is None:
            return False

        if isinstance(value, Mapping):
            return any(
                SpoofingTradingStrategy._has_any_value(item)
                for item in value.values()
            )

        if isinstance(value, (list, tuple, set)):
            return any(
                SpoofingTradingStrategy._has_any_value(item)
                for item in value
            )

        return value is not None


# Backward-compatible aliases while concrete spoofing strategies are migrated.
BaseSpoofingStrategy = SpoofingTradingStrategy
BaseSpoofingStrategyConfig = SpoofingStrategyConfig