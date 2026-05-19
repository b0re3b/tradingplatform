# trading_system/strategy/strategies/funding/base.py

from __future__ import annotations

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
    Parse timestamps used in analytics payloads / metadata.

    Concrete strategies should normally receive already-normalized timestamps
    through StrategyContext, but this helper is useful when funding domain data
    still contains raw nested analytics fields.
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        return ensure_aware_utc(value)

    if isinstance(value, (int, float)):
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
    Serialize small nested values for signal.metadata.

    This is metadata-focused only. Final RiskReadySignalPayload creation belongs
    to SignalProcessor.
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


def unwrap_analytics_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """
    Backward-compatible helper for funding analytics envelopes.

    New concrete strategies should not subscribe to analytics events directly.
    SignalNormalizer / StrategyContextBuilder should use this kind of logic when
    converting analytics.funding.* payloads into StrategyContext funding domain
    data and FeatureSnapshot objects.
    """
    raw = dict(payload)
    inner = raw.get("payload")

    if isinstance(inner, Mapping):
        inner_dict = dict(inner)

        for key in (
            "snapshot",
            "statistics",
            "regime_state",
            "pressure_state",
            "extreme_event",
            "divergence_event",
            "flip_event",
            "signal",
        ):
            nested_value = inner_dict.get(key)
            if isinstance(nested_value, Mapping):
                nested = dict(nested_value)
                nested.setdefault("_envelope", raw)
                return nested

        inner_dict.setdefault("_envelope", raw)
        return inner_dict

    return raw


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
    Read dotted paths from dict-like or object-like funding domain data.

    Examples:
        divergence.confidence
        pressure.score
        extreme.mean_reversion_probability
    """
    if not path.strip():
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


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float, Decimal)):
        return float(value)

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

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {
            "1",
            "true",
            "yes",
            "y",
            "on",
            "bullish",
            "long",
        }:
            return True

        if normalized in {
            "0",
            "false",
            "no",
            "n",
            "off",
            "bearish",
            "short",
        }:
            return False

    if isinstance(value, (int, float)):
        return bool(value)

    return default


# =============================================================================
# Funding feature contract
# =============================================================================


@dataclass(frozen=True, slots=True)
class FundingFeatureNames:
    """
    Stable feature names expected in StrategyContext.

    SignalNormalizer should create these FeatureSnapshot names from
    analytics.funding.* events. Concrete funding strategies can require a subset
    through StrategyDefinitionConfig.required_features.
    """

    SNAPSHOT: str = "funding.snapshot"
    STATISTICS: str = "funding.statistics"

    REGIME: str = "funding.regime"
    REGIME_CONFIDENCE: str = "funding.regime.confidence"

    PRESSURE: str = "funding.pressure"
    PRESSURE_SCORE: str = "funding.pressure.score"
    PRESSURE_LEVEL: str = "funding.pressure.level"
    PRESSURE_DIRECTION: str = "funding.pressure.direction"

    EXTREME: str = "funding.extreme"
    EXTREME_TYPE: str = "funding.extreme.type"
    EXTREME_SEVERITY: str = "funding.extreme.severity"
    EXTREME_MEAN_REVERSION_PROBABILITY: str = (
        "funding.extreme.mean_reversion_probability"
    )
    EXTREME_SQUEEZE_PROBABILITY: str = "funding.extreme.squeeze_probability"

    DIVERGENCE: str = "funding.divergence"
    DIVERGENCE_TYPE: str = "funding.divergence.type"
    DIVERGENCE_CONFIDENCE: str = "funding.divergence.confidence"
    DIVERGENCE_SCORE: str = "funding.divergence.score"

    FLIP: str = "funding.flip"
    FLIP_TYPE: str = "funding.flip.type"
    FLIP_CONFIDENCE: str = "funding.flip.confidence"

    SIGNAL: str = "funding.signal"
    SIGNAL_TYPE: str = "funding.signal.type"
    SIGNAL_SCORE: str = "funding.signal.score"
    SIGNAL_CONFIDENCE: str = "funding.signal.confidence"
    SIGNAL_BIAS: str = "funding.signal.bias"

    @classmethod
    def all(cls) -> set[str]:
        instance = cls()
        return {
            getattr(instance, item.name)
            for item in fields(cls)
            if isinstance(getattr(instance, item.name), str)
            and getattr(instance, item.name).strip()
        }


FUNDING_FEATURES = FundingFeatureNames()


# =============================================================================
# Scope
# =============================================================================


@dataclass(frozen=True, slots=True)
class FundingStrategyScope:
    """
    Futures funding scope used only for metadata / normalization.

    Concrete strategies should still make decisions from StrategyContext.
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
            raise StrategyEvaluationError("FundingStrategyScope.symbol cannot be empty")

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
class FundingStrategyConfig:
    """
    Domain config shared by concrete funding strategies.

    This is not the global StrategyConfig. It contains only funding-specific
    thresholds / metadata defaults. Runtime enabled/symbol/timeframe/regime
    checks still belong to StrategyDefinitionConfig / StrategyRuntimeConfig.
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
    min_signal_confidence: float = 0.45
    min_signal_score: float = 0.30

    min_regime_confidence: float = 0.10
    min_pressure_score: float = 0.30

    requested_leverage: float | None = None
    max_slippage_bps: float | None = None
    entry_timeout_seconds: int | None = None
    max_holding_seconds: int | None = None

    attach_funding_context_metadata: bool = True
    attach_scope_metadata: bool = True
    attach_feature_values_metadata: bool = True

    stale_feature_max_age_seconds: float | None = None

    tag_funding: str = "funding"
    tag_reversal: str = "reversal"
    tag_continuation: str = "continuation"
    tag_dislocation: str = "funding_dislocation"

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        bounded = {
            "min_context_confidence": self.min_context_confidence,
            "min_signal_confidence": self.min_signal_confidence,
            "min_signal_score": self.min_signal_score,
            "min_regime_confidence": self.min_regime_confidence,
            "min_pressure_score": self.min_pressure_score,
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
            "tag_funding",
            "tag_reversal",
            "tag_continuation",
            "tag_dislocation",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")


# =============================================================================
# Base funding strategy
# =============================================================================


class FundingTradingStrategy(TradingStrategy):
    """
    Base class for concrete strategy/strategies/funding/* classes.

    Responsibilities:
    - read funding data from StrategyContext only;
    - provide helper methods for funding domain extraction and scoring;
    - build internal StrategySignal objects through TradingStrategy helpers;
    - attach futures/funding metadata for SignalProcessor.

    Forbidden:
    - no direct analytics.funding.* subscriptions;
    - no local setup/confirmed/invalidated state machine;
    - no EventBus emit of signal.generated or strategy.funding.* lifecycle events;
    - no RiskManager / Execution calls;
    - no parquet/history persistence.
    """

    component_namespace = "strategy.funding"
    category: StrategyCategory = StrategyCategory.FUNDING
    default_setup_type: SetupType = SetupType.FUNDING_EXTREME
    default_timeframe: Timeframe = Timeframe.H1

    feature_names = FUNDING_FEATURES

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        funding_config: FundingStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        self.funding_config = funding_config or FundingStrategyConfig()
        self.funding_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            service_name=service_name,
        )

    def validate_config(self) -> None:
        super().validate_config()
        self.funding_config.validate()

    # ------------------------------------------------------------------
    # Context / domain access
    # ------------------------------------------------------------------

    def funding_domain(self, context: StrategyContext) -> dict[str, Any]:
        """
        Return funding domain data from StrategyContext.

        SignalNormalizer / StrategyContextBuilder should populate this from
        analytics.funding.* payloads.
        """
        self.validate_context(context)
        domain = context.domain_dict(FeatureSource.FUNDING)
        return dict(domain)

    def funding_item(
        self,
        context: StrategyContext,
        key: str,
        default: Any = None,
    ) -> Any:
        domain = self.funding_domain(context)

        if key in domain:
            return domain[key]

        aliases = {
            "regime": ("regime", "regime_state"),
            "pressure": ("pressure", "pressure_state"),
            "extreme": ("extreme", "extreme_event"),
            "divergence": ("divergence", "divergence_event"),
            "flip": ("flip", "flip_event"),
            "signal": ("signal", "funding_signal"),
            "snapshot": ("snapshot", "funding_snapshot"),
            "statistics": ("statistics", "stats", "funding_statistics"),
        }

        for alias in aliases.get(key, ()):
            if alias in domain:
                return domain[alias]

        return default

    def funding_path(
        self,
        context: StrategyContext,
        path: str,
        default: Any = None,
    ) -> Any:
        """
        Read a funding value by dotted path.

        Priority:
        1. StrategyContext feature with exact path;
        2. Feature name prefixed with 'funding.';
        3. Funding domain data dotted path.
        """
        self.validate_context(context)

        if not path.strip():
            raise StrategyEvaluationError("funding path cannot be empty")

        normalized = path.strip()
        feature_name = (
            normalized
            if normalized.startswith("funding.")
            else f"funding.{normalized}"
        )

        if context.has_feature(normalized):
            return context.get_feature(normalized)

        if context.has_feature(feature_name):
            return context.get_feature(feature_name)

        domain = self.funding_domain(context)
        if normalized.startswith("funding."):
            normalized = normalized.removeprefix("funding.")

        return _get_path(domain, normalized, default)

    def funding_float(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float | None = None,
    ) -> float | None:
        return _to_float(self.funding_path(context, path, default), default)

    def funding_score(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float = 0.0,
    ) -> float:
        value = self.funding_float(context, path, default=default)
        return clamp(float(value if value is not None else default), 0.0, 1.0)

    def funding_signed_score(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float = 0.0,
    ) -> float:
        value = self.funding_float(context, path, default=default)
        return clamp(float(value if value is not None else default), -1.0, 1.0)

    def funding_abs_score(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float = 0.0,
    ) -> float:
        return abs(self.funding_signed_score(context, path, default=default))

    def funding_bool(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: bool = False,
    ) -> bool:
        return _to_bool(self.funding_path(context, path, default), default)

    def funding_str(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: str | None = None,
    ) -> str | None:
        value = self.funding_path(context, path, default)
        if value is None:
            return default

        if isinstance(value, Enum):
            return str(value.value)

        return str(value)

    def funding_datetime(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: datetime | None = None,
    ) -> datetime | None:
        return parse_datetime(self.funding_path(context, path, default))

    def funding_feature_snapshot(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> FeatureSnapshot | None:
        """
        Return the full FeatureSnapshot if StrategyContext exposes one.

        This method is best-effort because StrategyContext may store either raw
        values or FeatureSnapshot objects depending on normalization.
        """
        self.validate_context(context)

        if not feature_name.strip():
            raise StrategyEvaluationError("feature_name cannot be empty")

        features_map = getattr(context, "features", None)
        if isinstance(features_map, Mapping):
            raw = features_map.get(feature_name)
            if isinstance(raw, FeatureSnapshot):
                return raw

        return None

    def funding_feature_age_seconds(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> float | None:
        snapshot = self.funding_feature_snapshot(context, feature_name)
        if snapshot is None:
            return None
        return snapshot.age_seconds(context.timestamp)

    def funding_feature_is_fresh(
        self,
        context: StrategyContext,
        feature_name: str,
        *,
        max_age_seconds: float | None = None,
    ) -> bool:
        age = self.funding_feature_age_seconds(context, feature_name)
        if age is None:
            return True

        threshold = (
            max_age_seconds
            if max_age_seconds is not None
            else self.funding_config.stale_feature_max_age_seconds
        )

        if threshold is None:
            return True

        return age <= threshold

    # ------------------------------------------------------------------
    # Scope / metadata
    # ------------------------------------------------------------------

    def funding_scope(self, context: StrategyContext) -> FundingStrategyScope:
        self.validate_context(context)

        metadata = dict(context.metadata or {})
        domain = self.funding_domain(context)

        exchange = (
            metadata.get("exchange")
            or domain.get("exchange")
            or self.funding_path(context, "snapshot.exchange")
            or "unknown"
        )
        market_type = (
            metadata.get("market_type")
            or domain.get("market_type")
            or self.funding_path(context, "snapshot.market_type")
            or self.funding_config.default_market_type.value
        )
        timeframe = (
            metadata.get("timeframe")
            or domain.get("timeframe")
            or context.timeframe.value
        )
        exchange_symbol = (
            metadata.get("exchange_symbol")
            or domain.get("exchange_symbol")
            or self.funding_path(context, "snapshot.exchange_symbol")
            or context.symbol
        )

        return FundingStrategyScope(
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

    def funding_context_metadata(
        self,
        context: StrategyContext,
        *,
        source_features: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "feature_source": FeatureSource.FUNDING.value,
            "strategy_category": StrategyCategory.FUNDING.value,
        }

        if self.funding_config.attach_scope_metadata:
            metadata["funding_scope"] = self.funding_scope(context).to_dict()

        if self.funding_config.attach_feature_values_metadata:
            metadata["funding_features"] = self._selected_feature_values(
                context=context,
                source_features=source_features or [],
            )

        if self.funding_config.attach_funding_context_metadata:
            domain = self.funding_domain(context)
            metadata["funding_context_keys"] = sorted(domain.keys())

            for key in (
                "snapshot",
                "statistics",
                "regime",
                "pressure",
                "extreme",
                "divergence",
                "flip",
                "signal",
            ):
                value = self.funding_item(context, key)
                if value is not None:
                    metadata[f"funding_{key}"] = serialize_for_metadata(value)

        metadata.update(dict(self.funding_config.metadata))

        if extra:
            metadata.update(extra)

        return metadata

    def _selected_feature_values(
        self,
        *,
        context: StrategyContext,
        source_features: list[str],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        for feature in source_features:
            if not isinstance(feature, str) or not feature.strip():
                continue

            if context.has_feature(feature):
                result[feature] = serialize_for_metadata(context.get_feature(feature))
                continue

            value = self.funding_path(context, feature, default=None)
            if value is not None:
                result[feature] = serialize_for_metadata(value)

        return result

    # ------------------------------------------------------------------
    # Direction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def side_from_bias(value: Any) -> SignalSide:
        """
        Convert normalized funding bias/direction into strategy-side signal side.
        """
        if isinstance(value, SignalSide):
            return value

        if isinstance(value, Enum):
            value = value.value

        if value is None:
            return SignalSide.UNKNOWN

        text = str(value).strip().lower()

        bullish_values = {
            "long",
            "bullish",
            "buy",
            "up",
            "positive_long",
            "negative_extreme_reversal",
            "bullish_divergence",
            "negative_funding_reversal",
        }
        bearish_values = {
            "short",
            "bearish",
            "sell",
            "down",
            "negative_short",
            "positive_extreme_reversal",
            "bearish_divergence",
            "positive_funding_reversal",
        }

        if text in bullish_values:
            return SignalSide.LONG

        if text in bearish_values:
            return SignalSide.SHORT

        return SignalSide.UNKNOWN

    @staticmethod
    def opposite_side(side: SignalSide) -> SignalSide:
        if side is SignalSide.LONG:
            return SignalSide.SHORT
        if side is SignalSide.SHORT:
            return SignalSide.LONG
        return SignalSide.UNKNOWN

    def side_from_signed_value(
        self,
        value: float | int | Decimal | None,
        *,
        positive_side: SignalSide = SignalSide.LONG,
        negative_side: SignalSide = SignalSide.SHORT,
        dead_zone: float = 0.0,
    ) -> SignalSide:
        parsed = _to_float(value)
        if parsed is None:
            return SignalSide.UNKNOWN

        if parsed > dead_zone:
            return positive_side

        if parsed < -dead_zone:
            return negative_side

        return SignalSide.UNKNOWN

    def contrarian_side_from_funding_bias(self, value: Any) -> SignalSide:
        return self.opposite_side(self.side_from_bias(value))

    # ------------------------------------------------------------------
    # Alignment / scoring helpers
    # ------------------------------------------------------------------

    def regime_alignment_score(
        self,
        context: StrategyContext,
        side: SignalSide,
        *,
        confidence_path: str = "regime.confidence",
        bias_path: str = "regime.bias",
        default: float = 0.0,
    ) -> float:
        if side not in {SignalSide.LONG, SignalSide.SHORT}:
            return 0.0

        confidence = self.funding_score(context, confidence_path, default=default)
        bias_side = self.side_from_bias(self.funding_path(context, bias_path))

        if bias_side is SignalSide.UNKNOWN:
            return confidence * 0.5

        return confidence if bias_side is side else 0.0

    def pressure_alignment_score(
        self,
        context: StrategyContext,
        side: SignalSide,
        *,
        score_path: str = "pressure.score",
        direction_path: str = "pressure.direction",
        default: float = 0.0,
    ) -> float:
        if side not in {SignalSide.LONG, SignalSide.SHORT}:
            return 0.0

        score = self.funding_score(context, score_path, default=default)
        pressure_side = self.side_from_bias(self.funding_path(context, direction_path))

        if pressure_side is SignalSide.UNKNOWN:
            return score * 0.5

        return score if pressure_side is side else 0.0

    def signal_alignment_score(
        self,
        context: StrategyContext,
        side: SignalSide,
        *,
        score_path: str = "signal.score",
        confidence_path: str = "signal.confidence",
        bias_path: str = "signal.bias",
        default: float = 0.0,
    ) -> float:
        if side not in {SignalSide.LONG, SignalSide.SHORT}:
            return 0.0

        raw_score = self.funding_abs_score(context, score_path, default=default)
        confidence = self.funding_score(context, confidence_path, default=default)
        signal_side = self.side_from_bias(self.funding_path(context, bias_path))

        if signal_side is SignalSide.UNKNOWN:
            return 0.5 * (raw_score + confidence)

        if signal_side is not side:
            return 0.0

        return clamp(0.5 * raw_score + 0.5 * confidence, 0.0, 1.0)

    @staticmethod
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
            if weight_f == 0.0:
                continue

            total_weight += weight_f
            total_value += clamp(float(values.get(key, default)), 0.0, 1.0) * weight_f

        if total_weight <= 0:
            return clamp(float(default), 0.0, 1.0)

        return clamp(total_value / total_weight, 0.0, 1.0)

    def confidence_from_components(
        self,
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
        return self.weighted_score(
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

    # ------------------------------------------------------------------
    # Plan / signal builders
    # ------------------------------------------------------------------

    def build_funding_trade_metadata(
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
        """
        Build first-pass funding metadata.

        Do not resolve liquidity_class/execution_quality here. SignalProcessor
        owns final risk-ready enrichment and will parse fallback values before
        RiskReadySignalPayload conversion.
        """
        priority = self.build_priority_metadata(
            setup_quality=setup_quality,
            confluence_score=confluence_score,
            liquidity_score=liquidity_score,
            risk_reward_score=risk_reward_score,
            execution_quality_score=execution_quality_score,
            regime_alignment_score=regime_alignment_score,
            freshness_score=freshness_score,
        )

        scope = self.funding_scope(context)

        metadata = self.build_trade_metadata(
            tier=self.funding_config.default_trade_tier,
            order_intent=self.funding_config.default_order_intent,
            margin_mode=self.funding_config.default_margin_mode,
            market_type=self.funding_config.default_market_type,
            requested_leverage=self.funding_config.requested_leverage,
            exchange=scope.exchange,
            extra={
                **priority,
                "funding_side": side.value,
                "liquidity_score": clamp(float(liquidity_score), 0.0, 1.0),
                "execution_quality_score": clamp(float(execution_quality_score), 0.0, 1.0),
                "regime_alignment_score": clamp(float(regime_alignment_score), 0.0, 1.0),
                "freshness_score": clamp(float(freshness_score), 0.0, 1.0),
                **self.funding_context_metadata(
                    context,
                    source_features=source_features,
                ),
            },
        )

        if extra:
            metadata.update(extra)

        return metadata

    def build_basic_funding_plans(
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

        SignalBuilder may later replace/enrich these plans before converting the
        signal into RiskReadySignalPayload.
        """
        self.validate_context(context)

        if side not in {SignalSide.LONG, SignalSide.SHORT}:
            raise StrategyEvaluationError("funding trade plans require directional side")

        entry = EntryPlan(
            entry_type=self.funding_config.default_entry_type,
            price=entry_price,
            timeout_seconds=self.funding_config.entry_timeout_seconds,
            max_slippage_bps=self.funding_config.max_slippage_bps,
            confirmation_required=False,
            notes=["funding_strategy_entry_draft"],
            metadata={
                "source": self.strategy_name,
                "feature_source": FeatureSource.FUNDING.value,
            },
        )

        targets: list[TargetPlan] = list(target_levels or [])
        if take_profit is not None:
            targets.append(
                TargetPlan(
                    price=take_profit,
                    size_fraction=1.0,
                    label="funding_take_profit",
                )
            )

        exit_plan = ExitPlan(
            exit_types=list(self.funding_config.default_exit_types),
            stop_loss=stop_loss,
            take_profit_levels=targets,
            max_holding_seconds=self.funding_config.max_holding_seconds,
            metadata={
                "source": self.strategy_name,
                "feature_source": FeatureSource.FUNDING.value,
            },
        )

        invalidation = InvalidationPlan(
            price=stop_loss,
            reason=invalidation_reason or "funding_context_invalidated",
            timeout_seconds=self.funding_config.max_holding_seconds,
            conditions=[
                "funding_signal_opposite",
                "funding_pressure_breakdown",
                "funding_regime_conflict",
            ],
            metadata={
                "source": self.strategy_name,
                "feature_source": FeatureSource.FUNDING.value,
            },
        )

        entry.validate()
        exit_plan.validate()
        invalidation.validate()
        return entry, exit_plan, invalidation

    def build_funding_signal(
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
        entry_price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> StrategySignal:
        if side not in {SignalSide.LONG, SignalSide.SHORT}:
            raise StrategyEvaluationError(
                f"{self.strategy_name}: funding signal side must be LONG or SHORT"
            )

        final_source_features = list(source_features or [])
        signal_metadata = self.build_funding_trade_metadata(
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
            tier=self.funding_config.default_trade_tier,
            order_intent=self.funding_config.default_order_intent,
            requested_leverage=self.funding_config.requested_leverage,
            margin_mode=self.funding_config.default_margin_mode,
            liquidity_class=None,
            execution_quality=None,
            market_type=self.funding_config.default_market_type,
        )

        signal.metadata.setdefault("feature_source", FeatureSource.FUNDING.value)
        signal.metadata.setdefault("funding_strategy_base", self.__class__.__name__)
        signal.validate()
        return signal

    # ------------------------------------------------------------------
    # Applicability
    # ------------------------------------------------------------------

    def validate_context_requirements(self, context: StrategyContext) -> None:
        super().validate_context_requirements(context)

        domain = self.funding_domain(context)
        has_funding_feature = any(
            context.has_feature(feature)
            for feature in FundingFeatureNames.all()
        )

        if not domain and not has_funding_feature:
            raise StrategyEvaluationError(
                f"{self.strategy_name}: missing funding domain data for {context.symbol}"
            )

    def supports_regime(self, regime: MarketRegime) -> bool:
        return super().supports_regime(regime)


__all__ = [
    "FUNDING_FEATURES",
    "FundingFeatureNames",
    "FundingStrategyConfig",
    "FundingStrategyScope",
    "FundingTradingStrategy",
    "ensure_utc",
    "parse_datetime",
    "serialize_for_metadata",
    "unwrap_analytics_payload",
    "utc_now",
]