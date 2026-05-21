# trading_system/strategy/strategies/open_interest/base.py

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from analytics.open_interest.enums import (
    OIAnomalyType,
    OIDivergenceType,
    OIRegime,
)
from analytics.open_interest.models import (
    OIAnalysisResult,
    OIAnomalyResult,
    OIDivergenceResult,
    OIFeatures,
    OIMarketContext,
    OIRegimeResult,
    OISnapshot,
)
from core.event_bus import EventBus
from core.scheduler import Scheduler

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
from ...models import (
    FeatureSnapshot,
    StrategyContext,
    StrategySignal,
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
    Parse timestamps used inside normalized open-interest domain payloads.

    Concrete strategies normally receive StrategyContext.timestamp, but nested
    analytics payloads may still contain detected_at / event_time / timestamp.
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
    owns final risk-ready payload conversion.
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


def _as_dict(value: Any) -> dict[str, Any]:
    mapping = _as_mapping(value)
    return dict(mapping) if mapping is not None else {}


def _get_attr_or_key(value: Any, key: str, default: Any = None) -> Any:
    mapping = _as_mapping(value)
    if mapping is not None:
        return mapping.get(key, default)

    return getattr(value, key, default)


def _get_path(value: Any, path: str, default: Any = None) -> Any:
    """
    Read dotted path from dict-like or object-like nested data.

    Examples:
        regime.confidence
        divergence.divergence_type
        anomaly.anomaly_type
        features.oi_delta_pct
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


def _to_str(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default

    if isinstance(value, Enum):
        return str(value.value)

    text = str(value).strip()
    return text if text else default


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


# =============================================================================
# Open-interest feature contract
# =============================================================================


@dataclass(frozen=True, slots=True)
class OpenInterestFeatureNames:
    """
    Stable feature names expected in StrategyContext.

    Generic SignalNormalizer may create these names from analytics.open_interest.*
    payloads, or strategies can read equivalent values from domain_data aliases.
    """

    ANALYSIS: str = "open_interest.analysis"

    SNAPSHOT: str = "open_interest.snapshot"
    MARKET_CONTEXT: str = "open_interest.context"
    FEATURES: str = "open_interest.features"

    REGIME: str = "open_interest.regime"
    REGIME_TYPE: str = "open_interest.regime.type"
    REGIME_CONFIDENCE: str = "open_interest.regime.confidence"
    REGIME_SCORE: str = "open_interest.regime.score"

    DIVERGENCE: str = "open_interest.divergence"
    DIVERGENCE_DETECTED: str = "open_interest.divergence.detected"
    DIVERGENCE_TYPE: str = "open_interest.divergence.type"
    DIVERGENCE_CONFIDENCE: str = "open_interest.divergence.confidence"
    DIVERGENCE_SCORE: str = "open_interest.divergence.score"
    DIVERGENCE_WINDOW_SIZE: str = "open_interest.divergence.window_size"

    ANOMALY: str = "open_interest.anomaly"
    ANOMALY_DETECTED: str = "open_interest.anomaly.detected"
    ANOMALY_TYPE: str = "open_interest.anomaly.type"
    ANOMALY_CONFIDENCE: str = "open_interest.anomaly.confidence"
    ANOMALY_SCORE: str = "open_interest.anomaly.score"
    ANOMALY_STRENGTH: str = "open_interest.anomaly.strength"

    OI_DELTA_PCT: str = "open_interest.features.oi_delta_pct"
    PRICE_DELTA_PCT: str = "open_interest.features.price_delta_pct"
    OI_PRESSURE_SCORE: str = "open_interest.features.oi_pressure_score"
    AGGRESSIVE_FLOW_IMBALANCE: str = (
        "open_interest.features.aggressive_flow_imbalance"
    )
    FUNDING_RATE: str = "open_interest.features.funding_rate"
    LIQUIDATION_PRESSURE: str = "open_interest.features.liquidation_pressure"

    @classmethod
    def all(cls) -> set[str]:
        instance = cls()
        return {
            getattr(instance, item.name)
            for item in fields(cls)
            if isinstance(getattr(instance, item.name), str)
            and getattr(instance, item.name).strip()
        }


OPEN_INTEREST_FEATURES = OpenInterestFeatureNames()


OPEN_INTEREST_DOMAIN_ALIASES: dict[str, tuple[str, ...]] = {
    "analysis": (
        "analysis",
        "oi_analysis",
        "open_interest_analysis",
        "result",
    ),
    "snapshot": (
        "snapshot",
        "oi_snapshot",
        "open_interest_snapshot",
        "analysis.snapshot",
    ),
    "market_context": (
        "context",
        "market_context",
        "oi_context",
        "open_interest_context",
        "analysis.context",
    ),
    "features": (
        "features",
        "oi_features",
        "open_interest_features",
        "analysis.features",
    ),
    "regime": (
        "regime",
        "regime_result",
        "oi_regime",
        "open_interest_regime",
        "new_regime",
        "analysis.regime",
    ),
    "divergence": (
        "divergence",
        "divergence_result",
        "oi_divergence",
        "open_interest_divergence",
        "analysis.divergence",
    ),
    "anomaly": (
        "anomaly",
        "anomaly_result",
        "oi_anomaly",
        "open_interest_anomaly",
        "analysis.anomaly",
    ),
    "signal": (
        "signal",
        "oi_signal",
        "open_interest_signal",
        "analytics_signal",
    ),
}


# =============================================================================
# Scope
# =============================================================================


@dataclass(frozen=True, slots=True)
class OpenInterestStrategyScope:
    """
    Futures open-interest scope used only for metadata and normalization.

    Concrete strategies still make decisions from StrategyContext.
    """

    exchange: str
    market_type: str
    symbol: str
    timeframe: str
    exchange_symbol: str | None = None

    def __post_init__(self) -> None:
        exchange = str(self.exchange or "unknown").strip().lower()
        market_type = str(
            self.market_type or StrategyMarketType.USDM_FUTURES.value
        ).strip()
        symbol = str(self.symbol or "").strip().upper()
        timeframe = str(self.timeframe or Timeframe.M1.value).strip().lower()
        exchange_symbol = str(self.exchange_symbol or symbol).strip().upper()

        if not symbol:
            raise StrategyEvaluationError(
                "OpenInterestStrategyScope.symbol cannot be empty"
            )

        object.__setattr__(self, "exchange", exchange)
        object.__setattr__(self, "market_type", market_type)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timeframe", timeframe)
        object.__setattr__(self, "exchange_symbol", exchange_symbol)

    @property
    def key(self) -> str:
        return f"{self.exchange}:{self.market_type}:{self.symbol}:{self.timeframe}"

    @property
    def legacy_key(self) -> str:
        return f"{self.symbol}:{self.exchange}"

    def to_dict(self) -> dict[str, str]:
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
class OpenInterestStrategyConfig:
    """
    Domain config shared by concrete open-interest strategies.

    Runtime enabled/symbol/timeframe/regime checks belong to StrategyConfig /
    StrategyDefinitionConfig. This config keeps OI-specific defaults and
    quality thresholds.
    """

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

    min_regime_confidence: float = 0.0
    min_divergence_confidence: float = 0.0
    min_anomaly_confidence: float = 0.0

    requested_leverage: float | None = None
    max_slippage_bps: float | None = None
    entry_timeout_seconds: int | None = None
    max_holding_seconds: int | None = None

    stale_feature_max_age_seconds: float | None = None

    attach_open_interest_context_metadata: bool = True
    attach_scope_metadata: bool = True
    attach_feature_values_metadata: bool = True

    tag_open_interest: str = "open_interest"
    tag_regime: str = "oi_regime"
    tag_divergence: str = "oi_divergence"
    tag_anomaly: str = "oi_anomaly"
    tag_capitulation: str = "oi_capitulation"
    tag_breakout: str = "oi_breakout"
    tag_reversal: str = "reversal"
    tag_continuation: str = "continuation"

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        bounded = {
            "min_context_confidence": self.min_context_confidence,
            "min_signal_confidence": self.min_signal_confidence,
            "min_signal_score": self.min_signal_score,
            "min_regime_confidence": self.min_regime_confidence,
            "min_divergence_confidence": self.min_divergence_confidence,
            "min_anomaly_confidence": self.min_anomaly_confidence,
        }

        for name, value in bounded.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{name} must be between 0.0 and 1.0")

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
            "tag_open_interest",
            "tag_regime",
            "tag_divergence",
            "tag_anomaly",
            "tag_capitulation",
            "tag_breakout",
            "tag_reversal",
            "tag_continuation",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")


# =============================================================================
# Base open-interest strategy
# =============================================================================


class OpenInterestTradingStrategy(TradingStrategy):
    """
    Base class for concrete strategy/strategies/open_interest/* classes.

    Responsibilities:
    - read open-interest analytics data from StrategyContext only;
    - provide helper methods for OI domain extraction and scoring;
    - build internal StrategySignal objects through TradingStrategy helpers;
    - attach futures/open-interest metadata for SignalProcessor.

    Forbidden:
    - no direct analytics.open_interest.* EventBus subscriptions;
    - no local signal/rejection state machine;
    - no diagnostics scheduler jobs;
    - no EventBus emit of signal.generated;
    - no RiskManager / Execution calls;
    - no raw market data reads.
    """

    component_namespace = "strategy.open_interest"
    category: StrategyCategory = StrategyCategory.OPEN_INTEREST
    default_setup_type: SetupType = SetupType.UNKNOWN
    default_timeframe: Timeframe = Timeframe.M1

    feature_names = OPEN_INTEREST_FEATURES

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        open_interest_config: OpenInterestStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        self.open_interest_config = (
            open_interest_config or OpenInterestStrategyConfig()
        )
        self.open_interest_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            service_name=service_name,
        )

    def validate_config(self) -> None:
        super().validate_config()
        self.open_interest_config.validate()

    # ------------------------------------------------------------------
    # No-signal diagnostics
    # ------------------------------------------------------------------

    def remember_no_signal(self, reason: str, **metadata: Any) -> None:
        """
        Store an exact no-signal reason for BaseStrategy.evaluate().

        This keeps concrete strategies in the StrategySignal | None contract,
        while allowing direct-batch / E2E diagnostics to show the exact gate
        that rejected the context instead of a generic no_signal_generated.
        """
        parent = getattr(super(), "remember_no_signal", None)
        if callable(parent):
            parent(reason, **metadata)
            return

        normalized = str(reason or "").strip() or "no_signal_generated"
        self._last_no_signal_reason = normalized
        self._last_no_signal_metadata = dict(metadata)

    def clear_no_signal_reason(self) -> None:
        parent = getattr(super(), "clear_no_signal_reason", None)
        if callable(parent):
            parent()
            return

        self._last_no_signal_reason = None
        self._last_no_signal_metadata = {}

    def consume_no_signal_reason(self) -> tuple[list[str], dict[str, Any]]:
        parent = getattr(super(), "consume_no_signal_reason", None)
        if callable(parent):
            return parent()

        reason = getattr(self, "_last_no_signal_reason", None) or "no_signal_generated"
        metadata = dict(getattr(self, "_last_no_signal_metadata", {}) or {})
        self.clear_no_signal_reason()
        return [reason], metadata

    # ------------------------------------------------------------------
    # Context / domain access
    # ------------------------------------------------------------------

    def open_interest_domain(self, context: StrategyContext) -> dict[str, Any]:
        """
        Return open-interest domain data from StrategyContext.

        Generic SignalNormalizer / StrategyContextBuilder should populate this
        from analytics.open_interest.* payloads.
        """
        self.validate_context(context)
        domain = context.domain_dict(FeatureSource.OPEN_INTEREST)
        return dict(domain)

    def open_interest_item(
        self,
        context: StrategyContext,
        key: str,
        default: Any = None,
    ) -> Any:
        domain = self.open_interest_domain(context)

        if key in domain:
            return domain[key]

        for alias in OPEN_INTEREST_DOMAIN_ALIASES.get(key, ()):
            value = _get_path(domain, alias, default=None)
            if value is not None:
                return value

        return default

    def open_interest_path(
        self,
        context: StrategyContext,
        path: str,
        default: Any = None,
    ) -> Any:
        """
        Read open-interest value by dotted path.

        Priority:
        1. exact StrategyContext feature name;
        2. open_interest-prefixed feature name;
        3. oi-prefixed legacy feature name;
        4. open-interest domain dotted path.
        """
        self.validate_context(context)

        if not isinstance(path, str) or not path.strip():
            raise StrategyEvaluationError("open_interest path cannot be empty")

        normalized = path.strip()
        open_interest_feature_name = (
            normalized
            if normalized.startswith("open_interest.")
            else f"open_interest.{normalized}"
        )
        legacy_feature_name = (
            normalized
            if normalized.startswith("oi.")
            else f"oi.{normalized}"
        )

        if context.has_feature(normalized):
            return self._feature_value(context.get_feature(normalized), default)

        if context.has_feature(open_interest_feature_name):
            return self._feature_value(
                context.get_feature(open_interest_feature_name),
                default,
            )

        if context.has_feature(legacy_feature_name):
            return self._feature_value(
                context.get_feature(legacy_feature_name),
                default,
            )

        domain = self.open_interest_domain(context)

        if normalized.startswith("open_interest."):
            normalized = normalized.removeprefix("open_interest.")
        elif normalized.startswith("oi."):
            normalized = normalized.removeprefix("oi.")

        return _get_path(domain, normalized, default)

    def open_interest_float(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float | None = None,
    ) -> float | None:
        return _to_float(self.open_interest_path(context, path, default), default)

    def open_interest_score(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float = 0.0,
    ) -> float:
        value = self.open_interest_float(context, path, default=default)
        return clamp(float(value if value is not None else default), 0.0, 1.0)

    def open_interest_signed_score(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float = 0.0,
    ) -> float:
        value = self.open_interest_float(context, path, default=default)
        return clamp(float(value if value is not None else default), -1.0, 1.0)

    def open_interest_bool(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: bool = False,
    ) -> bool:
        return _to_bool(self.open_interest_path(context, path, default), default)

    def open_interest_str(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: str | None = None,
    ) -> str | None:
        return _to_str(self.open_interest_path(context, path, default), default)

    def open_interest_datetime(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: datetime | None = None,
    ) -> datetime | None:
        return parse_datetime(self.open_interest_path(context, path, default))

    def open_interest_feature_snapshot(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> FeatureSnapshot | None:
        """
        Return full FeatureSnapshot if StrategyContext stores one.

        Best-effort helper because StrategyContext may store raw values or
        FeatureSnapshot objects depending on normalization.
        """
        self.validate_context(context)

        if not isinstance(feature_name, str) or not feature_name.strip():
            raise StrategyEvaluationError("feature_name cannot be empty")

        features_map = getattr(context, "features", None)
        if isinstance(features_map, Mapping):
            raw = features_map.get(feature_name)
            if isinstance(raw, FeatureSnapshot):
                return raw

        return None

    def open_interest_feature_age_seconds(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> float | None:
        snapshot = self.open_interest_feature_snapshot(context, feature_name)
        if snapshot is None:
            return None
        return snapshot.age_seconds(context.timestamp)

    def open_interest_feature_is_stale(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> bool:
        max_age = self.open_interest_config.stale_feature_max_age_seconds
        if max_age is None:
            return False

        age = self.open_interest_feature_age_seconds(context, feature_name)
        if age is None:
            return False

        return age > max_age

    def has_any_open_interest_data(
        self,
        context: StrategyContext,
        feature_names: tuple[str, ...] = (),
    ) -> bool:
        self.validate_context(context)

        if self.open_interest_domain(context):
            return True

        return any(context.has_feature(name) for name in feature_names)

    def has_stale_open_interest_features(
        self,
        context: StrategyContext,
        feature_names: tuple[str, ...] | None = None,
    ) -> bool:
        names = feature_names or tuple(self.required_features())

        return any(
            self.open_interest_feature_is_stale(context, feature_name)
            for feature_name in names
        )

    # ------------------------------------------------------------------
    # Scope
    # ------------------------------------------------------------------

    def open_interest_scope(self, context: StrategyContext) -> OpenInterestStrategyScope:
        domain = self.open_interest_domain(context)

        exchange = (
            _to_str(domain.get("exchange"))
            or _to_str(_get_path(domain, "scope.exchange"))
            or _to_str(context.metadata.get("exchange"))
            or "unknown"
        )
        market_type = (
            _to_str(domain.get("market_type"))
            or _to_str(_get_path(domain, "scope.market_type"))
            or _to_str(context.metadata.get("market_type"))
            or self.open_interest_config.default_market_type.value
        )
        exchange_symbol = (
            _to_str(domain.get("exchange_symbol"))
            or _to_str(_get_path(domain, "scope.exchange_symbol"))
            or _to_str(context.metadata.get("exchange_symbol"))
            or context.symbol
        )

        return OpenInterestStrategyScope(
            exchange=exchange,
            market_type=market_type,
            symbol=context.symbol,
            timeframe=context.timeframe.value,
            exchange_symbol=exchange_symbol,
        )

    # ------------------------------------------------------------------
    # Analytics model extraction
    # ------------------------------------------------------------------

    def extract_oi_analysis(
        self,
        context: StrategyContext,
    ) -> OIAnalysisResult | None:
        """
        Extract full OIAnalysisResult from StrategyContext.

        Preferred domain shape:
            context.domain_dict(FeatureSource.OPEN_INTEREST)["analysis"]

        Supported fallback:
            domain itself can be OIAnalysisResult.to_dict().
        """
        raw_analysis = self.open_interest_item(context, "analysis")

        if isinstance(raw_analysis, OIAnalysisResult):
            return raw_analysis

        if isinstance(raw_analysis, Mapping):
            parsed = self._parse_oi_analysis(raw_analysis)
            if parsed is not None:
                return parsed

        domain = self.open_interest_domain(context)
        if self._looks_like_oi_analysis(domain):
            return self._parse_oi_analysis(domain)

        return None

    def extract_oi_snapshot(self, context: StrategyContext) -> OISnapshot | None:
        analysis = self.extract_oi_analysis(context)
        if analysis is not None:
            return analysis.snapshot

        raw = self.open_interest_item(context, "snapshot")
        if isinstance(raw, OISnapshot):
            return raw

        if isinstance(raw, Mapping):
            try:
                return OISnapshot.from_dict(dict(raw))
            except Exception as exc:
                self.log_debug(
                    "Failed to parse OISnapshot",
                    symbol=context.symbol,
                    error=repr(exc),
                )

        return None

    def extract_oi_market_context(
        self,
        context: StrategyContext,
    ) -> OIMarketContext | None:
        analysis = self.extract_oi_analysis(context)
        if analysis is not None:
            return analysis.context

        raw = self.open_interest_item(context, "market_context")
        if isinstance(raw, OIMarketContext):
            return raw

        if isinstance(raw, Mapping):
            try:
                return OIMarketContext.from_dict(dict(raw))
            except Exception as exc:
                self.log_debug(
                    "Failed to parse OIMarketContext",
                    symbol=context.symbol,
                    error=repr(exc),
                )

        return None

    def extract_oi_features(self, context: StrategyContext) -> OIFeatures | None:
        """
        Extract OIFeatures from:
        1. full OIAnalysisResult;
        2. FeatureSource.OPEN_INTEREST domain["features"];
        3. open_interest.* / oi.* feature-map values.
        """
        analysis = self.extract_oi_analysis(context)
        if analysis is not None:
            return analysis.features

        raw = self.open_interest_item(context, "features")
        if isinstance(raw, OIFeatures):
            return raw

        if isinstance(raw, Mapping):
            parsed = self._parse_oi_features(raw, context=context)
            if parsed is not None:
                return parsed

        legacy = self._extract_feature_payload(context)
        if legacy:
            parsed = self._parse_oi_features(legacy, context=context)
            if parsed is not None:
                return parsed

        return None

    def extract_oi_regime_result(
        self,
        context: StrategyContext,
    ) -> OIRegimeResult | None:
        """
        Extract OIRegimeResult from:
        - analysis.regime;
        - domain["regime"];
        - specialized regime_changed payload shape;
        - open_interest.* / oi.* feature names.
        """
        analysis = self.extract_oi_analysis(context)
        if analysis is not None:
            return analysis.regime

        raw = self.open_interest_item(context, "regime")

        if isinstance(raw, OIRegimeResult):
            return raw

        if isinstance(raw, Mapping):
            data = dict(raw)
            regime_value = (
                data.get("regime")
                or data.get("type")
                or data.get("regime_type")
                or data.get("new_regime")
            )
            if regime_value is not None:
                try:
                    return OIRegimeResult.from_dict(
                        {
                            "regime": regime_value,
                            "confidence": data.get("confidence", 0.0),
                            "score": data.get("score"),
                            "reasons": list(data.get("reasons") or []),
                        }
                    )
                except Exception:
                    pass

        domain = self.open_interest_domain(context)
        flat_regime = (
            domain.get("new_regime")
            or domain.get("regime_type")
            or domain.get("oi_regime_type")
            or domain.get("oi_regime")
        )
        if flat_regime is not None:
            try:
                return OIRegimeResult.from_dict(
                    {
                        "regime": flat_regime,
                        "confidence": (
                            domain.get("regime_confidence")
                            if domain.get("regime_confidence") is not None
                            else domain.get("confidence", 0.0)
                        ),
                        "score": domain.get("regime_score", domain.get("score")),
                        "reasons": self.normalize_reasons(
                            domain.get("reasons")
                            or domain.get("regime_reasons")
                            or []
                        ),
                    }
                )
            except Exception:
                pass

        legacy_regime = (
            self.open_interest_path(context, "regime.type", None)
            or self.open_interest_path(context, "regime.regime", None)
        )
        if legacy_regime is not None:
            try:
                return OIRegimeResult.from_dict(
                    {
                        "regime": legacy_regime,
                        "confidence": self.open_interest_path(
                            context,
                            "regime.confidence",
                            0.0,
                        ),
                        "score": self.open_interest_path(
                            context,
                            "regime.score",
                            None,
                        ),
                        "reasons": self.normalize_reasons(
                            self.open_interest_path(
                                context,
                                "regime.reasons",
                                [],
                            )
                        ),
                    }
                )
            except Exception:
                return None

        return None

    def extract_oi_divergence_result(
        self,
        context: StrategyContext,
    ) -> OIDivergenceResult | None:
        """
        Extract OIDivergenceResult from:
        - analysis.divergence;
        - domain["divergence"];
        - specialized divergence payload shape;
        - open_interest.* / oi.* feature names.
        """
        analysis = self.extract_oi_analysis(context)
        if analysis is not None:
            return analysis.divergence

        raw = self.open_interest_item(context, "divergence")

        if isinstance(raw, OIDivergenceResult):
            return raw

        if isinstance(raw, Mapping):
            data = self._normalize_divergence_payload(dict(raw))
            try:
                return OIDivergenceResult.from_dict(data)
            except Exception:
                pass

        domain = self.open_interest_domain(context)
        flat_type = (
            domain.get("divergence_type")
            or domain.get("oi_divergence_type")
            or domain.get("open_interest_divergence_type")
            or domain.get("type")
        )
        if flat_type is not None:
            data = self._normalize_divergence_payload(domain)
            try:
                return OIDivergenceResult.from_dict(data)
            except Exception:
                pass

        legacy_type = (
            self.open_interest_path(context, "divergence.type", None)
            or self.open_interest_path(context, "divergence.divergence_type", None)
        )
        if legacy_type is not None:
            try:
                return OIDivergenceResult.from_dict(
                    {
                        "detected": self.open_interest_bool(
                            context,
                            "divergence.detected",
                            default=legacy_type != OIDivergenceType.NONE.value,
                        ),
                        "divergence_type": legacy_type,
                        "confidence": self.open_interest_path(
                            context,
                            "divergence.confidence",
                            0.0,
                        ),
                        "score": self.open_interest_path(
                            context,
                            "divergence.score",
                            None,
                        ),
                        "window_size": self.open_interest_path(
                            context,
                            "divergence.window_size",
                            None,
                        ),
                        "reasons": self.normalize_reasons(
                            self.open_interest_path(
                                context,
                                "divergence.reasons",
                                [],
                            )
                        ),
                    }
                )
            except Exception:
                return None

        return None

    def extract_oi_anomaly_result(
        self,
        context: StrategyContext,
    ) -> OIAnomalyResult | None:
        """
        Extract OIAnomalyResult from:
        - analysis.anomaly;
        - domain["anomaly"];
        - anomaly/capitulation specialized payload shape;
        - open_interest.* / oi.* feature names.
        """
        analysis = self.extract_oi_analysis(context)
        if analysis is not None:
            return analysis.anomaly

        raw = self.open_interest_item(context, "anomaly")

        if isinstance(raw, OIAnomalyResult):
            return raw

        if isinstance(raw, Mapping):
            data = self._normalize_anomaly_payload(dict(raw))
            try:
                return OIAnomalyResult.from_dict(data)
            except Exception:
                pass

        domain = self.open_interest_domain(context)
        flat_type = (
            domain.get("anomaly_type")
            or domain.get("oi_anomaly_type")
            or domain.get("open_interest_anomaly_type")
        )
        if flat_type is not None:
            data = self._normalize_anomaly_payload(domain)
            try:
                return OIAnomalyResult.from_dict(data)
            except Exception:
                pass

        legacy_type = (
            self.open_interest_path(context, "anomaly.type", None)
            or self.open_interest_path(context, "anomaly.anomaly_type", None)
        )
        if legacy_type is not None:
            try:
                return OIAnomalyResult.from_dict(
                    {
                        "detected": self.open_interest_bool(
                            context,
                            "anomaly.detected",
                            default=legacy_type != OIAnomalyType.NONE.value,
                        ),
                        "anomaly_type": legacy_type,
                        "confidence": self.open_interest_path(
                            context,
                            "anomaly.confidence",
                            0.0,
                        ),
                        "score": self.open_interest_path(
                            context,
                            "anomaly.score",
                            None,
                        ),
                        "strength": self.open_interest_path(
                            context,
                            "anomaly.strength",
                            None,
                        ),
                        "reasons": self.normalize_reasons(
                            self.open_interest_path(
                                context,
                                "anomaly.reasons",
                                [],
                            )
                        ),
                    }
                )
            except Exception:
                return None

        return None

    def oi_analysis_confidence(self, context: StrategyContext) -> float:
        analysis = self.extract_oi_analysis(context)
        if analysis is not None:
            return _unit_score(getattr(analysis, "confidence", 0.0))

        value = (
            self.open_interest_path(context, "analysis.confidence", None)
            or self.open_interest_path(context, "confidence", None)
            or self.open_interest_path(context, "score", None)
        )
        return _unit_score(value, 0.0)

    # ------------------------------------------------------------------
    # Direction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parse_side(value: Any) -> SignalSide:
        if isinstance(value, SignalSide):
            return value

        if value is None:
            return SignalSide.UNKNOWN

        label = _normalize_label(value)

        if label in {
            "long",
            "buy",
            "bull",
            "bullish",
            "up",
            "upside",
            "positive",
        }:
            return SignalSide.LONG

        if label in {
            "short",
            "sell",
            "bear",
            "bearish",
            "down",
            "downside",
            "negative",
        }:
            return SignalSide.SHORT

        return SignalSide.UNKNOWN

    @staticmethod
    def opposite_side(side: SignalSide) -> SignalSide:
        if side is SignalSide.LONG:
            return SignalSide.SHORT
        if side is SignalSide.SHORT:
            return SignalSide.LONG
        return SignalSide.UNKNOWN

    @staticmethod
    def is_directional_side(side: SignalSide) -> bool:
        return side in {SignalSide.LONG, SignalSide.SHORT}

    def regime_to_side_hint(self, regime: OIRegime | Any) -> str:
        """
        Semantic side hint from OIRegime.

        Returns:
            bullish | bearish | contextual | neutral
        """
        label = _normalize_label(regime)

        if label in {
            "long_buildup",
            "short_covering",
            "bullish_oi_expansion",
        }:
            return "bullish"

        if label in {
            "short_buildup",
            "long_unwind",
            "bearish_oi_expansion",
        }:
            return "bearish"

        if label in {
            "trend_confirmation",
            "squeeze_setup",
            "capitulation",
            "trend_exhaustion",
            "overheated",
        }:
            return "contextual"

        return "neutral"

    def divergence_to_side_hint(self, divergence_type: OIDivergenceType | Any) -> str:
        """
        Semantic side hint from OIDivergenceType.

        The exact enum values may evolve in analytics.open_interest, so this
        method is label-based and tolerant.
        """
        label = _normalize_label(divergence_type)

        if not label or label in {"none", "unknown", "neutral"}:
            return "neutral"

        if "bull" in label or "positive" in label or "long" in label:
            return "bullish"

        if "bear" in label or "negative" in label or "short" in label:
            return "bearish"

        if "exhaustion" in label or "reversal" in label:
            return "contextual"

        return "neutral"

    def anomaly_to_setup_hint(self, anomaly_type: OIAnomalyType | Any) -> str:
        label = _normalize_label(anomaly_type)

        if label in {
            "oi_collapse",
            "liquidation_driven_oi_drop",
            "sudden_deleveraging",
            "overheated_buildup",
            "extreme_crowding",
            "funding_oi_imbalance",
            "oi_price_dislocation",
        }:
            return "reversal"

        if label in {
            "oi_spike",
            "oi_volume_dislocation",
        }:
            return "continuation"

        return "unknown"

    def side_from_features(
        self,
        features: OIFeatures | None,
        *,
        dead_zone: float = 0.0,
    ) -> SignalSide:
        if features is None:
            return SignalSide.UNKNOWN

        price_delta = _to_float(getattr(features, "price_delta_pct", None))
        oi_delta = _to_float(getattr(features, "oi_delta_pct", None))
        pressure = _to_float(getattr(features, "oi_pressure_score", None))
        flow = _to_float(getattr(features, "aggressive_flow_imbalance", None))

        if price_delta is not None and oi_delta is not None:
            if price_delta > dead_zone and oi_delta > dead_zone:
                if pressure is None or pressure >= -dead_zone:
                    if flow is None or flow >= -dead_zone:
                        return SignalSide.LONG

            if price_delta < -dead_zone and oi_delta > dead_zone:
                if pressure is None or pressure <= dead_zone:
                    if flow is None or flow <= dead_zone:
                        return SignalSide.SHORT

        if pressure is not None:
            if pressure > dead_zone:
                return SignalSide.LONG
            if pressure < -dead_zone:
                return SignalSide.SHORT

        if flow is not None:
            if flow > dead_zone:
                return SignalSide.LONG
            if flow < -dead_zone:
                return SignalSide.SHORT

        return SignalSide.UNKNOWN

    # ------------------------------------------------------------------
    # Signal builder
    # ------------------------------------------------------------------

    def build_open_interest_signal(
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
        Build internal StrategySignal with open-interest/futures metadata.

        Final risk-ready payload conversion belongs to SignalProcessor /
        SignalBuilder, not to this domain strategy.
        """
        if side not in {SignalSide.LONG, SignalSide.SHORT}:
            raise StrategyEvaluationError(
                f"{self.strategy_name}: OI signal side must be LONG or SHORT"
            )

        scope = self.open_interest_scope(context)

        signal_metadata = dict(metadata or {})
        signal_metadata.setdefault("domain", FeatureSource.OPEN_INTEREST.value)
        signal_metadata.setdefault("open_interest_strategy_version", "2.0.0")
        signal_metadata.setdefault("order_intent", self.open_interest_config.default_order_intent.value)
        signal_metadata.setdefault("margin_mode", self.open_interest_config.default_margin_mode.value)
        signal_metadata.setdefault("market_type", self.open_interest_config.default_market_type.value)
        signal_metadata.setdefault("tier", self.open_interest_config.default_trade_tier.value)

        if self.open_interest_config.requested_leverage is not None:
            signal_metadata.setdefault(
                "requested_leverage",
                float(self.open_interest_config.requested_leverage),
            )

        if self.open_interest_config.max_slippage_bps is not None:
            signal_metadata.setdefault(
                "max_slippage_bps",
                float(self.open_interest_config.max_slippage_bps),
            )

        if self.open_interest_config.entry_timeout_seconds is not None:
            signal_metadata.setdefault(
                "entry_timeout_seconds",
                int(self.open_interest_config.entry_timeout_seconds),
            )

        if self.open_interest_config.max_holding_seconds is not None:
            signal_metadata.setdefault(
                "max_holding_seconds",
                int(self.open_interest_config.max_holding_seconds),
            )

        if self.open_interest_config.attach_scope_metadata:
            signal_metadata.setdefault("scope", scope.to_dict())

        if self.open_interest_config.attach_open_interest_context_metadata:
            signal_metadata.setdefault(
                "open_interest_context",
                self.open_interest_context_metadata(context),
            )

        if self.open_interest_config.metadata:
            signal_metadata.setdefault(
                "open_interest_config_metadata",
                serialize_for_metadata(self.open_interest_config.metadata),
            )

        final_reasons = list(
            dict.fromkeys(
                [
                    "open_interest_strategy_signal",
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
            metadata=signal_metadata,
            trigger_type=trigger_type,
            origin=origin,
            priority=priority,
            status=status,
        )

        signal.validate()
        return signal

    def open_interest_context_metadata(
        self,
        context: StrategyContext,
    ) -> dict[str, Any]:
        """
        Compact serialized OI context for StrategySignal.metadata.
        """
        metadata: dict[str, Any] = {}

        analysis = self.extract_oi_analysis(context)
        if analysis is not None:
            metadata["analysis"] = serialize_for_metadata(analysis)

        regime = self.extract_oi_regime_result(context)
        if regime is not None:
            metadata["regime"] = serialize_for_metadata(regime)

        divergence = self.extract_oi_divergence_result(context)
        if divergence is not None:
            metadata["divergence"] = serialize_for_metadata(divergence)

        anomaly = self.extract_oi_anomaly_result(context)
        if anomaly is not None:
            metadata["anomaly"] = serialize_for_metadata(anomaly)

        features = self.extract_oi_features(context)
        if features is not None:
            metadata["features"] = serialize_for_metadata(features)

        if self.open_interest_config.attach_feature_values_metadata:
            metadata["feature_values"] = {
                "analysis_confidence": self.oi_analysis_confidence(context),
                "oi_delta_pct": self.open_interest_path(
                    context,
                    "features.oi_delta_pct",
                    None,
                ),
                "price_delta_pct": self.open_interest_path(
                    context,
                    "features.price_delta_pct",
                    None,
                ),
                "oi_pressure_score": self.open_interest_path(
                    context,
                    "features.oi_pressure_score",
                    None,
                ),
                "aggressive_flow_imbalance": self.open_interest_path(
                    context,
                    "features.aggressive_flow_imbalance",
                    None,
                ),
            }

        return serialize_for_metadata(metadata)

    # ------------------------------------------------------------------
    # Normalization internals
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_reasons(value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []

        if isinstance(value, (list, tuple, set)):
            result: list[str] = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    result.append(text)
            return list(dict.fromkeys(result))

        text = str(value).strip()
        return [text] if text else []

    @staticmethod
    def _feature_value(value: Any, default: Any = None) -> Any:
        if isinstance(value, FeatureSnapshot):
            return value.value
        return default if value is None else value

    @staticmethod
    def _looks_like_oi_analysis(value: Mapping[str, Any]) -> bool:
        required = {"symbol", "timestamp", "snapshot", "context", "features", "regime"}
        return required.issubset(set(value.keys()))

    def _parse_oi_analysis(
        self,
        value: Mapping[str, Any],
    ) -> OIAnalysisResult | None:
        try:
            return OIAnalysisResult.from_dict(dict(value))
        except Exception as exc:
            self.log_debug(
                "Failed to parse OIAnalysisResult",
                error=repr(exc),
            )
            return None

    def _parse_oi_features(
        self,
        value: Mapping[str, Any],
        *,
        context: StrategyContext,
    ) -> OIFeatures | None:
        data = dict(value)

        try:
            return OIFeatures.from_dict(data)
        except Exception:
            try:
                return OIFeatures(**data)
            except Exception as exc:
                self.log_debug(
                    "Failed to parse OIFeatures",
                    symbol=context.symbol,
                    error=repr(exc),
                )
                return None

    def _extract_feature_payload(self, context: StrategyContext) -> dict[str, Any]:
        """
        Best-effort legacy feature-map extraction.

        This keeps compatibility with old oi.* / open_interest.* feature names
        while new strategies should prefer FeatureSource.OPEN_INTEREST domain data.
        """
        candidates = {
            "oi_delta_pct": (
                "features.oi_delta_pct",
                "oi_delta_pct",
                "open_interest.features.oi_delta_pct",
                "oi.features.oi_delta_pct",
            ),
            "price_delta_pct": (
                "features.price_delta_pct",
                "price_delta_pct",
                "open_interest.features.price_delta_pct",
                "oi.features.price_delta_pct",
            ),
            "oi_pressure_score": (
                "features.oi_pressure_score",
                "oi_pressure_score",
                "open_interest.features.oi_pressure_score",
                "oi.features.oi_pressure_score",
            ),
            "aggressive_flow_imbalance": (
                "features.aggressive_flow_imbalance",
                "aggressive_flow_imbalance",
                "open_interest.features.aggressive_flow_imbalance",
                "oi.features.aggressive_flow_imbalance",
            ),
            "funding_rate": (
                "features.funding_rate",
                "funding_rate",
                "open_interest.features.funding_rate",
                "oi.features.funding_rate",
            ),
            "liquidation_pressure": (
                "features.liquidation_pressure",
                "liquidation_pressure",
                "open_interest.features.liquidation_pressure",
                "oi.features.liquidation_pressure",
            ),
        }

        payload: dict[str, Any] = {}

        for target_key, paths in candidates.items():
            for path in paths:
                try:
                    value = self.open_interest_path(context, path, None)
                except StrategyEvaluationError:
                    value = None

                if value is not None:
                    payload[target_key] = value
                    break

        return payload

    def _normalize_divergence_payload(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        data = dict(payload)

        divergence_type = (
            data.get("divergence_type")
            or data.get("type")
            or data.get("oi_divergence_type")
            or data.get("open_interest_divergence_type")
            or OIDivergenceType.NONE.value
        )

        detected_default = divergence_type != OIDivergenceType.NONE.value

        return {
            "detected": _to_bool(data.get("detected"), detected_default),
            "divergence_type": divergence_type,
            "confidence": _unit_score(data.get("confidence"), 0.0),
            "score": data.get("score"),
            "window_size": _to_int(data.get("window_size")),
            "reasons": self.normalize_reasons(data.get("reasons")),
        }

    def _normalize_anomaly_payload(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        data = dict(payload)

        anomaly_type = (
            data.get("anomaly_type")
            or data.get("type")
            or data.get("oi_anomaly_type")
            or data.get("open_interest_anomaly_type")
            or OIAnomalyType.NONE.value
        )

        detected_default = anomaly_type != OIAnomalyType.NONE.value

        normalized = {
            "detected": _to_bool(data.get("detected"), detected_default),
            "anomaly_type": anomaly_type,
            "confidence": _unit_score(data.get("confidence"), 0.0),
            "score": data.get("score"),
            "reasons": self.normalize_reasons(data.get("reasons")),
        }

        strength = data.get("strength")
        if strength is not None:
            normalized["strength"] = strength

        return normalized


# Backward-compatible alias while concrete OI strategies are migrated.
OpenInterestStrategyBase = OpenInterestTradingStrategy