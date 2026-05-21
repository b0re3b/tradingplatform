from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TypeVar

from core.event_bus import EventBus, EventPriority
from core.scheduler import Scheduler
from strategy.base import BaseStrategy, BaseStrategyComponent
from strategy.config import (
    BuilderConfig,
    ConfluenceConfig,
    PortfolioCoordinatorConfig,
    RoutingConfig,
    StrategyConfig,
)
from strategy.enums import (
    ConfidenceGrade,
    ConflictType,
    EntryType,
    ExitType,
    FeatureSource,
    FilterDecision,
    MarketRegime,
    SignalOrigin,
    SignalSide,
    SignalStrength,
    StrategyCategory,
    StrategyExecutionQuality,
    StrategyLiquidityClass,
    StrategyMarginMode,
    StrategyMarketType,
    StrategyOrderIntent,
    StrategyTradeTier,
    Timeframe,
    TriggerType,
)
from strategy.exceptions import (
    BuilderError,
    ConfluenceError,
    PortfolioCoordinationError,
    SignalNormalizationError,
    SignalRoutingError,
    StrategyEvaluationError,
)
from strategy.models import (
    ConflictRecord,
    ConfluenceResult,
    EntryPlan,
    ExecutionCostPayload,
    ExecutionPlanDraft,
    ExitPlan,
    FeatureSnapshot,
    FilterResult,
    PriceSnapshot,
    InvalidationPlan,
    RiskReadySignalPayload,
    StrategyContext,
    StrategyEvaluation,
    StrategySignal,
    TargetPlan,
    clamp,
    confidence_to_grade,
    confidence_to_strength,
    ensure_aware_utc,
    utcnow,
)
from strategy.registry import StrategyRegistry
from strategy.state import StrategyRuntimeState

EnumT = TypeVar("EnumT")


# =============================================================================
# Small typed helpers
# =============================================================================


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value)
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

    if isinstance(value, float):
        return int(value)

    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default

    return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False

    if isinstance(value, (int, float)):
        return bool(value)

    return default


def _parse_enum(enum_cls: type[EnumT], value: Any, default: EnumT) -> EnumT:
    if isinstance(value, enum_cls):
        return value

    if isinstance(value, str):
        try:
            return enum_cls(value)  # type: ignore[misc, call-arg]
        except ValueError:
            return default

    return default


# =============================================================================
# Pipeline DTOs
# =============================================================================


@dataclass(slots=True)
class NormalizedPayload:
    source: FeatureSource
    symbol: str
    timestamp: datetime
    timeframe: Timeframe = Timeframe.M1
    domain_data: dict[str, Any] = field(default_factory=dict)
    extra_domain_data: dict[FeatureSource, dict[str, Any]] = field(default_factory=dict)
    features: list[FeatureSnapshot] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamp = ensure_aware_utc(self.timestamp)


@dataclass(slots=True)
class RouteDecision:
    event_name: str
    symbol: str
    source: FeatureSource | None = None
    timestamp: datetime = field(default_factory=utcnow)
    selected: list[BaseStrategy] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    categories_used: list[StrategyCategory] = field(default_factory=list)
    matched_features: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamp = ensure_aware_utc(self.timestamp)

    @property
    def selected_names(self) -> list[str]:
        return [strategy.strategy_name for strategy in self.selected]

    @property
    def total_selected(self) -> int:
        return len(self.selected)

    @property
    def is_empty(self) -> bool:
        return not self.selected


@dataclass(slots=True)
class WeightedSignal:
    signal: StrategySignal
    category_weight: float
    regime_weight: float
    strategy_weight: float
    final_weight: float
    weighted_score: float
    weighted_confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.signal.validate()

        if self.category_weight < 0:
            raise ConfluenceError("WeightedSignal.category_weight must be >= 0")
        if self.regime_weight < 0:
            raise ConfluenceError("WeightedSignal.regime_weight must be >= 0")
        if self.strategy_weight < 0:
            raise ConfluenceError("WeightedSignal.strategy_weight must be >= 0")
        if self.final_weight < 0:
            raise ConfluenceError("WeightedSignal.final_weight must be >= 0")


@dataclass(slots=True)
class VoteSummary:
    total_votes: int = 0
    long_votes: int = 0
    short_votes: int = 0
    flat_votes: int = 0
    confirmation_count: int = 0
    primary_count: int = 0
    dominant_side: SignalSide = SignalSide.UNKNOWN
    accepted: bool = False
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ConflictSummary:
    accepted: bool = True
    total_penalty: float = 0.0
    conflicts: list[ConflictRecord] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def add_conflict(self, conflict: ConflictRecord) -> None:
        conflict.validate()
        self.conflicts.append(conflict)
        self.total_penalty += conflict.penalty


@dataclass(slots=True)
class ConfluenceEvaluation:
    symbol: str
    timestamp: datetime
    raw_signals: list[StrategySignal] = field(default_factory=list)
    eligible_signals: list[StrategySignal] = field(default_factory=list)
    accepted_signals: list[StrategySignal] = field(default_factory=list)
    rejected_signals: dict[str, str] = field(default_factory=dict)
    result: ConfluenceResult | None = None
    merged_signal: StrategySignal | None = None
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamp = ensure_aware_utc(self.timestamp)

    @property
    def accepted(self) -> bool:
        return self.result is not None and self.result.accepted

    @property
    def selected_strategy_names(self) -> list[str]:
        return [signal.strategy_name for signal in self.accepted_signals]


@dataclass(slots=True)
class FilterEvaluation:
    signal: StrategySignal
    context_symbol: str
    timestamp: datetime = field(default_factory=utcnow)
    results: list[FilterResult] = field(default_factory=list)
    accepted: bool = True
    blocking_filters: list[str] = field(default_factory=list)
    warning_filters: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamp = ensure_aware_utc(self.timestamp)

    def add_result(self, result: FilterResult) -> None:
        result.validate()
        self.results.append(result)

        if result.decision is FilterDecision.BLOCK:
            self.accepted = False
            self.blocking_filters.append(result.name)
            if result.reason:
                self.reasons.append(result.reason)

        elif result.decision is FilterDecision.WARN:
            self.warning_filters.append(result.name)
            if result.reason:
                self.reasons.append(result.reason)

    @property
    def has_warnings(self) -> bool:
        return bool(self.warning_filters)

    @property
    def has_blocks(self) -> bool:
        return bool(self.blocking_filters)


@dataclass(slots=True)
class BuildEvaluation:
    signal: StrategySignal
    context_symbol: str
    entry: EntryPlan | None = None
    invalidation: InvalidationPlan | None = None
    targets: list[TargetPlan] = field(default_factory=list)
    exit_plan: ExitPlan | None = None
    execution_plan: ExecutionPlanDraft | None = None
    accepted: bool = True
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.accepted = False
        if reason not in self.reasons:
            self.reasons.append(reason)


@dataclass(slots=True)
class CoordinationDecision:
    symbol: str
    timestamp: datetime
    raw_signals: list[StrategySignal] = field(default_factory=list)
    accepted_signals: list[StrategySignal] = field(default_factory=list)
    rejected_signals: dict[str, str] = field(default_factory=dict)
    merged_signals: list[StrategySignal] = field(default_factory=list)
    throttled_signals: dict[str, str] = field(default_factory=dict)
    suppressed_signals: dict[str, str] = field(default_factory=dict)
    accepted: bool = True
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamp = ensure_aware_utc(self.timestamp)

    @property
    def final_signals(self) -> list[StrategySignal]:
        return self.merged_signals if self.merged_signals else self.accepted_signals

    @property
    def selected_names(self) -> list[str]:
        return [signal.strategy_name for signal in self.final_signals]


@dataclass(slots=True)
class ProcessedSignalBatch:
    symbol: str
    timestamp: datetime
    normalized: NormalizedPayload | None = None
    context: StrategyContext | None = None
    route: RouteDecision | None = None
    evaluations: list[StrategyEvaluation] = field(default_factory=list)
    raw_signals: list[StrategySignal] = field(default_factory=list)
    filtered_signals: list[StrategySignal] = field(default_factory=list)
    confluence: ConfluenceEvaluation | None = None
    coordinated: CoordinationDecision | None = None
    final_signals: list[StrategySignal] = field(default_factory=list)
    risk_payloads: list[RiskReadySignalPayload] = field(default_factory=list)
    accepted: bool = False
    emitted: bool = False
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamp = ensure_aware_utc(self.timestamp)


# =============================================================================
# SignalNormalizer
# =============================================================================


class SignalNormalizer(BaseStrategyComponent):
    """
    Normalizes analytics payloads into StrategyContext domain data and features.

    Supported analytics payload shapes:
    - explicit payload["features"] as list[dict];
    - payload["feature_map"] as dict[str, value | dict];
    - nested analytics sections: stats, context, signal, analysis, result,
      metrics, snapshot, state, pressure, regime, divergence, anomaly;
    - top-level scalar fallback.

    The normalizer keeps domain_data intact for concrete strategies that read
    context.domain_dict(source), while also producing FeatureSnapshot entries
    for registry routing and required_features checks.
    """

    component_namespace = "strategy.processor.normalizer"

    _FEATURE_CONTAINER_KEYS: tuple[str, ...] = (
        "features",
        "feature_map",
        "stats",
        "context",
        "signal",
        "analysis",
        "result",
        "metrics",
        "snapshot",
        "state",
        "pressure",
        "regime",
        "divergence",
        "anomaly",
        "event",
        "setup",
    )

    _DOMAIN_EXCLUDED_KEYS: set[str] = {
        "symbol",
        "instrument",
        "market",
        "timestamp",
        "ts",
        "source",
        "features",
        "feature_map",
        "metadata",
    }

    _SCALAR_FEATURE_EXCLUDED_KEYS: set[str] = {
        "symbol",
        "instrument",
        "market",
        "timestamp",
        "ts",
        "source",
        "metadata",
        "scope",
        "key",
        "orderflow_key",
        "scope_key",
        "exchange_symbol",
    }

    _TIMEFRAME_ALIASES: dict[str, Timeframe] = {
        "1m": Timeframe.M1,
        "m1": Timeframe.M1,
        "3m": Timeframe.M3,
        "m3": Timeframe.M3,
        "5m": Timeframe.M5,
        "m5": Timeframe.M5,
        "15m": Timeframe.M15,
        "m15": Timeframe.M15,
        "30m": Timeframe.M30,
        "m30": Timeframe.M30,
        "1h": Timeframe.H1,
        "h1": Timeframe.H1,
        "4h": Timeframe.H4,
        "h4": Timeframe.H4,
        "1d": Timeframe.D1,
        "d1": Timeframe.D1,
    }

    def _augment_funding_domain_data(
            self,
            *,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        """
        Normalize analytics.funding.* payloads into the stable funding
        StrategyContext contract.

        OI-style contract adapter only:
        - expose analytics-provided sections under stable canonical aliases;
        - extract nested analysis/result sections;
        - enrich flat funding payloads into snapshot/statistics only when safe;
        - do NOT synthesize extreme/divergence/flip/signal unless analytics
          explicitly supplied a detected/actionable context.
        """
        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def mapping_for(*keys: str) -> dict[str, Any] | None:
            for key in keys:
                value = payload.get(key)
                if isinstance(value, dict):
                    return dict(value)

                value = feature_map.get(key)
                if isinstance(value, dict):
                    return dict(value)

            return None

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]
            return default

        topic = (
            str(
                payload.get("event_name")
                or payload.get("topic")
                or payload.get("source_topic")
                or ""
            )
            .strip()
            .lower()
        )

        def to_bool(value: Any, default: bool = False) -> bool:
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
                    "detected",
                    "active",
                    "confirmed",
                    "triggered",
                }:
                    return True
                if normalized in {
                    "0",
                    "false",
                    "no",
                    "n",
                    "off",
                    "none",
                    "not_detected",
                    "inactive",
                }:
                    return False

            if isinstance(value, (int, float)):
                return bool(value)

            return default

        def section_detected(section: dict[str, Any] | None) -> bool:
            if not section:
                return False

            detected = section.get(
                "detected",
                section.get(
                    "is_detected",
                    section.get(
                        "active",
                        section.get("confirmed", None),
                    ),
                ),
            )

            if detected is None:
                # Typed nested analytics section without explicit detected flag is
                # considered analytics-provided/actionable. Empty dicts are rejected.
                return True

            return to_bool(detected, default=False)

        def set_aliases(
                target: str,
                aliases: tuple[str, ...],
                value: dict[str, Any] | None,
                *,
                override: bool = True,
        ) -> None:
            if value is None:
                return

            # Canonical funding sections should override raw/flat fields already
            # copied into domain_data. This mirrors the OI contract behavior.
            if override:
                domain_data[target] = value
            else:
                domain_data.setdefault(target, value)

            for alias in aliases:
                domain_data.setdefault(alias, value)

        analysis = mapping_for(
            "analysis",
            "result",
            "funding_analysis",
            "funding_result",
        )
        snapshot = mapping_for(
            "snapshot",
            "funding_snapshot",
        )
        statistics = mapping_for(
            "statistics",
            "stats",
            "funding_statistics",
        )
        regime = mapping_for(
            "regime",
            "regime_state",
            "funding_regime",
            "funding_regime_state",
        )
        pressure = mapping_for(
            "pressure",
            "pressure_state",
            "funding_pressure",
            "funding_pressure_state",
        )
        extreme = mapping_for(
            "extreme",
            "extreme_event",
            "funding_extreme",
            "funding_extreme_event",
        )
        divergence = mapping_for(
            "divergence",
            "divergence_event",
            "funding_divergence",
            "funding_divergence_event",
        )
        flip = mapping_for(
            "flip",
            "flip_event",
            "funding_flip",
            "funding_flip_event",
        )
        signal = mapping_for(
            "signal",
            "funding_signal",
            "strategy_signal",
            "setup",
        )

        if analysis is not None:
            nested_snapshot = (
                    analysis.get("snapshot")
                    or analysis.get("funding_snapshot")
            )
            nested_statistics = (
                    analysis.get("statistics")
                    or analysis.get("stats")
                    or analysis.get("funding_statistics")
            )
            nested_regime = (
                    analysis.get("regime")
                    or analysis.get("regime_state")
                    or analysis.get("funding_regime")
            )
            nested_pressure = (
                    analysis.get("pressure")
                    or analysis.get("pressure_state")
                    or analysis.get("funding_pressure")
            )
            nested_extreme = (
                    analysis.get("extreme")
                    or analysis.get("extreme_event")
                    or analysis.get("funding_extreme")
            )
            nested_divergence = (
                    analysis.get("divergence")
                    or analysis.get("divergence_event")
                    or analysis.get("funding_divergence")
            )
            nested_flip = (
                    analysis.get("flip")
                    or analysis.get("flip_event")
                    or analysis.get("funding_flip")
            )
            nested_signal = (
                    analysis.get("signal")
                    or analysis.get("funding_signal")
                    or analysis.get("setup")
            )

            if isinstance(nested_snapshot, dict) and snapshot is None:
                snapshot = dict(nested_snapshot)
            if isinstance(nested_statistics, dict) and statistics is None:
                statistics = dict(nested_statistics)
            if isinstance(nested_regime, dict) and regime is None:
                regime = dict(nested_regime)
            if isinstance(nested_pressure, dict) and pressure is None:
                pressure = dict(nested_pressure)
            if isinstance(nested_extreme, dict) and extreme is None:
                extreme = dict(nested_extreme)
            if isinstance(nested_divergence, dict) and divergence is None:
                divergence = dict(nested_divergence)
            if isinstance(nested_flip, dict) and flip is None:
                flip = dict(nested_flip)
            if isinstance(nested_signal, dict) and signal is None:
                signal = dict(nested_signal)

        # Safe flat enrichment: snapshot/statistics are context sections, not
        # trade/setup sections.
        if snapshot is None:
            flat_snapshot: dict[str, Any] = {}
            for key in (
                    "funding_rate",
                    "current_rate",
                    "next_funding_rate",
                    "predicted_rate",
                    "annualized_rate",
                    "premium_index",
                    "mark_price",
                    "index_price",
                    "next_funding_time",
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "timestamp",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    flat_snapshot[key] = value

            if flat_snapshot:
                snapshot = flat_snapshot

        if statistics is None:
            flat_statistics: dict[str, Any] = {}
            for key in (
                    "mean",
                    "median",
                    "std",
                    "zscore",
                    "z_score",
                    "percentile",
                    "min",
                    "max",
                    "window",
                    "lookback",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    flat_statistics[key] = value

            if flat_statistics:
                statistics = flat_statistics

        # Flat signal/setup section only when analytics explicitly produced
        # signal/setup context.
        signal_like_topic = any(
            token in topic
            for token in (
                "signal",
                "setup",
                "confirmed",
                "generated",
            )
        )
        explicit_signal = bool(
            signal_like_topic
            or value_for("signal_type", "setup_type", "signal_bias", default=None)
            is not None
            or to_bool(value_for("signal_detected", "setup_detected", default=False))
        )

        if signal is None and explicit_signal:
            signal = {
                "type": value_for("signal_type", "setup_type", default=None),
                "bias": value_for("bias", "direction", "signal_bias", default=None),
                "score": value_for("signal_score", "score", default=0.0),
                "confidence": value_for(
                    "signal_confidence",
                    "confidence",
                    default=0.0,
                ),
                "origin": value_for("origin", "signal_origin", default="funding"),
                "detected": True,
            }

        # Event-specific flat sections only when explicitly indicated by analytics.
        if extreme is None and (
                "extreme" in topic
                or value_for("extreme_type", "extreme_severity", default=None) is not None
                or to_bool(value_for("extreme_detected", default=False))
        ):
            extreme = {
                "type": value_for("extreme_type", "type", default=None),
                "severity": value_for(
                    "extreme_severity",
                    "severity",
                    "score",
                    default=0.0,
                ),
                "mean_reversion_probability": value_for(
                    "mean_reversion_probability",
                    "reversion_probability",
                    default=0.0,
                ),
                "squeeze_probability": value_for(
                    "squeeze_probability",
                    default=0.0,
                ),
                "detected": True,
            }

        if divergence is None and (
                "divergence" in topic
                or value_for("divergence_type", "divergence_score", default=None) is not None
                or to_bool(value_for("divergence_detected", default=False))
        ):
            divergence = {
                "type": value_for("divergence_type", "type", default=None),
                "score": value_for("divergence_score", "score", default=0.0),
                "confidence": value_for(
                    "divergence_confidence",
                    "confidence",
                    default=0.0,
                ),
                "bias": value_for("bias", "direction", default=None),
                "detected": True,
            }

        if flip is None and (
                "flip" in topic
                or value_for("flip_type", "flip_confidence", default=None) is not None
                or to_bool(value_for("flip_detected", default=False))
        ):
            flip = {
                "type": value_for("flip_type", "type", default=None),
                "confidence": value_for(
                    "flip_confidence",
                    "confidence",
                    default=0.0,
                ),
                "bias": value_for("bias", "direction", default=None),
                "detected": True,
            }

        set_aliases("analysis", ("funding_analysis", "result"), analysis)
        set_aliases("snapshot", ("funding_snapshot",), snapshot)
        set_aliases("statistics", ("stats", "funding_statistics"), statistics)
        set_aliases("regime", ("regime_state", "funding_regime"), regime)
        set_aliases("pressure", ("pressure_state", "funding_pressure"), pressure)

        if section_detected(extreme):
            set_aliases(
                "extreme",
                ("extreme_event", "funding_extreme"),
                extreme,
            )

        if section_detected(divergence):
            set_aliases(
                "divergence",
                ("divergence_event", "funding_divergence"),
                divergence,
            )

        if section_detected(flip):
            set_aliases(
                "flip",
                ("flip_event", "funding_flip"),
                flip,
            )

        if section_detected(signal):
            set_aliases(
                "signal",
                ("funding_signal", "setup"),
                signal,
            )

    def _augment_orderflow_domain_data(
            self,
            *,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        """
        Normalize analytics.orderflow.* payloads into a stable orderflow contract.

        OI-style contract adapter:
        - exposes canonical sections: composite, cvd, volume_delta,
          aggressive_trades, orderbook_imbalance, signal;
        - preserves analytics-provided nested sections and typed-like objects;
        - enriches flat payload into sections only when meaningful;
        - does not synthesize trade setup sections.
        """
        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def as_mapping(value: Any) -> dict[str, Any] | None:
            if isinstance(value, dict):
                return value

            to_dict = getattr(value, "to_dict", None)
            if callable(to_dict):
                converted = to_dict()
                if isinstance(converted, dict):
                    return converted

            return None

        def get_item(value: Any, key: str, default: Any = None) -> Any:
            mapping = as_mapping(value)
            if mapping is not None:
                return mapping.get(key, default)
            return getattr(value, key, default)

        def mapping_or_object_for(*keys: str) -> Any | None:
            for key in keys:
                if key in payload and payload[key] is not None:
                    return payload[key]
                if key in feature_map and feature_map[key] is not None:
                    return feature_map[key]
            return None

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]

            for container in (
                    composite,
                    cvd,
                    volume_delta,
                    aggressive_trades,
                    orderbook_imbalance,
                    signal,
            ):
                for key in keys:
                    value = get_item(container, key, None)
                    if value is not None:
                        return value

            return default

        def set_aliases(
                target: str,
                aliases: tuple[str, ...],
                value: Any | None,
                *,
                override: bool = True,
        ) -> None:
            if value is None:
                return

            if override:
                domain_data[target] = value
            else:
                domain_data.setdefault(target, value)

            for alias in aliases:
                domain_data.setdefault(alias, value)

        def to_bool(value: Any, default: bool = False) -> bool:
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
                    "active",
                    "valid",
                    "confirmed",
                    "detected",
                    "triggered",
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
                    "none",
                    "not_detected",
                }:
                    return False

            if isinstance(value, (int, float)):
                return bool(value)

            return default

        def section_present(value: Any | None) -> bool:
            if value is None:
                return False

            mapping = as_mapping(value)
            if mapping is None:
                return True

            if not mapping:
                return False

            detected = (
                mapping.get("detected")
                if "detected" in mapping
                else mapping.get("active", mapping.get("valid", None))
            )

            if detected is None:
                return True

            return to_bool(detected, default=False)

        topic = (
            str(
                payload.get("event_name")
                or payload.get("topic")
                or payload.get("source_topic")
                or ""
            )
            .strip()
            .lower()
        )

        analysis = mapping_or_object_for(
            "analysis",
            "orderflow_analysis",
            "result",
        )
        composite = mapping_or_object_for(
            "composite",
            "snapshot",
            "orderflow_snapshot",
            "composite_snapshot",
            "orderflow_composite",
        )
        cvd = mapping_or_object_for(
            "cvd",
            "cvd_snapshot",
            "cvd_metrics",
            "cumulative_volume_delta",
        )
        volume_delta = mapping_or_object_for(
            "volume_delta",
            "volume_delta_snapshot",
            "delta",
            "delta_metrics",
        )
        aggressive_trades = mapping_or_object_for(
            "aggressive_trades",
            "aggressive",
            "aggressive_flow",
            "aggressive_trades_snapshot",
        )
        orderbook_imbalance = mapping_or_object_for(
            "orderbook_imbalance",
            "orderbook",
            "orderbook_snapshot",
            "imbalance",
        )
        signal = mapping_or_object_for(
            "signal",
            "orderflow_signal",
            "analytics_signal",
            "setup",
        )

        for container_key in (
                "result",
                "payload",
                "data",
                "analysis",
                "orderflow_analysis",
        ):
            container = mapping_or_object_for(container_key)
            container_map = as_mapping(container)
            if container_map is None:
                continue

            nested_composite = (
                    container_map.get("composite")
                    or container_map.get("snapshot")
                    or container_map.get("orderflow_snapshot")
                    or container_map.get("composite_snapshot")
            )
            nested_cvd = (
                    container_map.get("cvd")
                    or container_map.get("cvd_snapshot")
                    or container_map.get("cvd_metrics")
            )
            nested_volume_delta = (
                    container_map.get("volume_delta")
                    or container_map.get("delta")
                    or container_map.get("delta_metrics")
            )
            nested_aggressive = (
                    container_map.get("aggressive_trades")
                    or container_map.get("aggressive")
                    or container_map.get("aggressive_flow")
            )
            nested_orderbook = (
                    container_map.get("orderbook_imbalance")
                    or container_map.get("orderbook")
                    or container_map.get("imbalance")
            )
            nested_signal = (
                    container_map.get("signal")
                    or container_map.get("orderflow_signal")
                    or container_map.get("analytics_signal")
                    or container_map.get("setup")
            )

            if composite is None and nested_composite is not None:
                composite = nested_composite
            if cvd is None and nested_cvd is not None:
                cvd = nested_cvd
            if volume_delta is None and nested_volume_delta is not None:
                volume_delta = nested_volume_delta
            if aggressive_trades is None and nested_aggressive is not None:
                aggressive_trades = nested_aggressive
            if orderbook_imbalance is None and nested_orderbook is not None:
                orderbook_imbalance = nested_orderbook
            if signal is None and nested_signal is not None:
                signal = nested_signal

        if composite is not None:
            composite_map = as_mapping(composite)

            if cvd is None:
                nested = get_item(composite, "cvd", None)
                if nested is not None:
                    cvd = nested

            if volume_delta is None:
                nested = get_item(composite, "volume_delta", None)
                if nested is not None:
                    volume_delta = nested

            if aggressive_trades is None:
                nested = get_item(composite, "aggressive_trades", None)
                if nested is not None:
                    aggressive_trades = nested

            if orderbook_imbalance is None:
                nested = get_item(composite, "orderbook_imbalance", None)
                if nested is not None:
                    orderbook_imbalance = nested

            if composite_map is not None:
                metadata = composite_map.get("metadata")
                if isinstance(metadata, dict):
                    for key in (
                            "exchange",
                            "market_type",
                            "symbol",
                            "exchange_symbol",
                            "timeframe",
                    ):
                        if key in metadata:
                            domain_data.setdefault(key, metadata[key])

        if cvd is None:
            flat_cvd: dict[str, Any] = {}
            for key in (
                    "cvd_value",
                    "value",
                    "cvd_delta_ratio",
                    "delta_ratio",
                    "cvd_change_pct",
                    "cvd_slope",
                    "price_change_pct",
                    "cvd_buy_ratio",
                    "cvd_sell_ratio",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    flat_cvd[key] = value

            if flat_cvd:
                cvd = flat_cvd

        if volume_delta is None:
            flat_volume_delta: dict[str, Any] = {}
            for key in (
                    "volume_delta",
                    "notional_delta",
                    "volume_delta_ratio",
                    "delta_ratio",
                    "cumulative_volume_delta",
                    "cumulative_notional_delta",
                    "buy_volume",
                    "sell_volume",
                    "buy_notional",
                    "sell_notional",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    flat_volume_delta[key] = value

            if flat_volume_delta:
                volume_delta = flat_volume_delta

        if aggressive_trades is None:
            flat_aggressive: dict[str, Any] = {}
            for key in (
                    "aggressive_buy_ratio",
                    "aggressive_sell_ratio",
                    "aggressive_burst_score",
                    "aggressive_net_volume_delta",
                    "aggressive_net_notional_delta",
                    "large_buy_trades",
                    "large_sell_trades",
                    "aggressive_buy_count",
                    "aggressive_sell_count",
                    "aggressive_buy_volume",
                    "aggressive_sell_volume",
                    "aggressive_buy_notional",
                    "aggressive_sell_notional",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    flat_aggressive[key] = value

            if flat_aggressive:
                aggressive_trades = flat_aggressive

        if orderbook_imbalance is None:
            flat_orderbook: dict[str, Any] = {}
            for key in (
                    "orderbook_imbalance_ratio",
                    "orderbook_imbalance_diff",
                    "ratio",
                    "diff",
                    "bid_volume",
                    "ask_volume",
                    "best_bid",
                    "best_ask",
                    "spread",
                    "mid_price",
                    "depth_levels_used",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    flat_orderbook[key] = value

            if flat_orderbook:
                orderbook_imbalance = flat_orderbook

        # Composite is context, so it may be composed from the real sections.
        # This is not a strategy setup; it is the stable orderflow data container.
        if composite is None and any(
                section is not None
                for section in (cvd, volume_delta, aggressive_trades, orderbook_imbalance)
        ):
            composite = {
                "cvd": cvd or {},
                "volume_delta": volume_delta or {},
                "aggressive_trades": aggressive_trades or {},
                "orderbook_imbalance": orderbook_imbalance or {},
            }

            for key in (
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "timestamp",
                    "last_price",
                    "price",
                    "price_change",
                    "price_change_pct",
                    "window_seconds",
                    "trades_count",
                    "total_volume",
                    "total_notional",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    composite[key] = value

        signal_like_topic = any(
            token in topic
            for token in (
                "signal",
                "setup",
                "continuation",
                "reversal",
                "divergence",
                "confirmed",
                "generated",
            )
        )
        explicit_signal = bool(
            signal_like_topic
            or payload.get("signal_type") is not None
            or payload.get("setup_type") is not None
            or payload.get("signal_side") is not None
            or to_bool(payload.get("signal_detected", False))
            or to_bool(payload.get("setup_detected", False))
        )

        if signal is None and explicit_signal:
            signal = {
                "detected": True,
                "type": payload.get("signal_type") or payload.get("setup_type"),
                "side": (
                        payload.get("signal_side")
                        or payload.get("side")
                        or payload.get("direction")
                        or payload.get("bias")
                ),
                "score": payload.get("signal_score", payload.get("score", 0.0)),
                "confidence": payload.get(
                    "signal_confidence",
                    payload.get("confidence", 0.0),
                ),
                "origin": payload.get("origin", "orderflow"),
            }

        set_aliases(
            "analysis",
            ("orderflow_analysis", "result"),
            analysis,
            override=False,
        )

        if composite is not None:
            set_aliases(
                "composite",
                ("snapshot", "orderflow_snapshot", "composite_snapshot"),
                composite,
            )

        if cvd is not None:
            set_aliases(
                "cvd",
                ("cvd_snapshot", "cvd_metrics"),
                cvd,
            )

        if volume_delta is not None:
            set_aliases(
                "volume_delta",
                ("delta", "delta_metrics", "volume_delta_snapshot"),
                volume_delta,
            )

        if aggressive_trades is not None:
            set_aliases(
                "aggressive_trades",
                ("aggressive", "aggressive_flow", "aggressive_trades_snapshot"),
                aggressive_trades,
            )

        if orderbook_imbalance is not None:
            set_aliases(
                "orderbook_imbalance",
                ("orderbook", "imbalance", "orderbook_snapshot"),
                orderbook_imbalance,
            )

        if section_present(signal):
            set_aliases(
                "signal",
                ("orderflow_signal", "analytics_signal", "setup"),
                signal,
            )

        for key in (
                "exchange",
                "market_type",
                "symbol",
                "exchange_symbol",
                "timeframe",
                "timestamp",
                "last_price",
                "price_change_pct",
                "trades_count",
                "total_volume",
                "total_notional",
        ):
            value = value_for(key, default=None)
            if value is not None:
                domain_data.setdefault(key, value)

    def _augment_open_interest_domain_data(
            self,
            *,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        """
        Normalize analytics.oi/open_interest payloads into the stable
        open-interest StrategyContext contract.

        This is a contract adapter only:
        - it exposes analytics-provided sections under stable aliases;
        - it enriches flat OI payloads into the "features" section;
        - it does NOT synthesize anomaly/divergence sections unless analytics
          actually supplied a detected anomaly/divergence context.
        """
        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def mapping_for(*keys: str) -> dict[str, Any] | None:
            for key in keys:
                value = payload.get(key)
                if isinstance(value, dict):
                    return dict(value)

                value = feature_map.get(key)
                if isinstance(value, dict):
                    return dict(value)

            return None

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]
            return default

        topic = (
            str(
                payload.get("event_name")
                or payload.get("topic")
                or payload.get("source_topic")
                or ""
            )
            .strip()
            .lower()
        )

        def to_bool(value: Any, default: bool = False) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"1", "true", "yes", "y", "on", "detected", "active", "confirmed"}:
                    return True
                if normalized in {"0", "false", "no", "n", "off", "none", "not_detected"}:
                    return False
            if isinstance(value, (int, float)):
                return bool(value)
            return default

        def section_detected(section: dict[str, Any] | None) -> bool:
            if not section:
                return False

            detected = section.get("detected", section.get("is_detected", None))
            if detected is None:
                # A typed nested section without explicit detected flag is
                # considered present/actionable by the strategy-specific parser.
                return True

            return to_bool(detected, default=False)

        def set_aliases(target: str, aliases: tuple[str, ...], value: dict[str, Any] | None) -> None:
            if value is None:
                return

            # Canonical OI sections must override raw/flat fields already
            # copied from analytics payload. Otherwise concrete OI strategies
            # read lower-case / partial raw sections and fail payload resolution.
            domain_data[target] = value
            for alias in aliases:
                domain_data.setdefault(alias, value)

        analysis = mapping_for(
            "analysis",
            "oi_analysis",
            "open_interest_analysis",
            "result",
        )
        snapshot = mapping_for(
            "snapshot",
            "oi_snapshot",
            "open_interest_snapshot",
        )
        market_context = mapping_for(
            "context",
            "market_context",
            "oi_context",
            "open_interest_context",
        )
        features = mapping_for(
            "features",
            "oi_features",
            "open_interest_features",
        )
        regime = mapping_for(
            "regime",
            "regime_result",
            "oi_regime",
            "open_interest_regime",
            "new_regime",
        )
        divergence = mapping_for(
            "divergence",
            "divergence_result",
            "oi_divergence",
            "open_interest_divergence",
        )
        anomaly = mapping_for(
            "anomaly",
            "anomaly_result",
            "oi_anomaly",
            "open_interest_anomaly",
        )

        if analysis is not None:
            nested_snapshot = analysis.get("snapshot")
            nested_context = (
                analysis.get("context")
                or analysis.get("market_context")
                or analysis.get("oi_context")
            )
            nested_features = analysis.get("features")
            nested_regime = (
                analysis.get("regime")
                or analysis.get("regime_result")
                or analysis.get("oi_regime")
            )
            nested_divergence = (
                analysis.get("divergence")
                or analysis.get("divergence_result")
                or analysis.get("oi_divergence")
            )
            nested_anomaly = (
                analysis.get("anomaly")
                or analysis.get("anomaly_result")
                or analysis.get("oi_anomaly")
            )

            if isinstance(nested_snapshot, dict) and snapshot is None:
                snapshot = dict(nested_snapshot)
            if isinstance(nested_context, dict) and market_context is None:
                market_context = dict(nested_context)
            if isinstance(nested_features, dict) and features is None:
                features = dict(nested_features)
            if isinstance(nested_regime, dict) and regime is None:
                regime = dict(nested_regime)
            if isinstance(nested_divergence, dict) and divergence is None:
                divergence = dict(nested_divergence)
            if isinstance(nested_anomaly, dict) and anomaly is None:
                anomaly = dict(nested_anomaly)

        if features is None:
            has_feature_like_values = any(
                key in payload or key in feature_map
                for key in (
                    "oi",
                    "open_interest",
                    "open_interest_value",
                    "oi_delta",
                    "oi_delta_pct",
                    "oi_direction",
                    "oi_acceleration",
                    "price_delta_pct",
                    "volume_ratio",
                    "oi_zscore",
                    "oi_pressure_score",
                    "funding_rate",
                    "liquidation_pressure",
                    "liquidation_imbalance",
                    "aggressive_flow_imbalance",
                    "oi_price_efficiency",
                )
            )

            if has_feature_like_values:
                features = {
                    "oi": value_for("oi", "open_interest", default=None),
                    "open_interest": value_for("open_interest", "oi", default=None),
                    "open_interest_value": value_for("open_interest_value", default=None),
                    "oi_delta": value_for("oi_delta", default=None),
                    "oi_delta_pct": value_for("oi_delta_pct", default=None),
                    "oi_direction": value_for("oi_direction", default=None),
                    "oi_acceleration": value_for("oi_acceleration", default=None),
                    "price_delta_pct": value_for("price_delta_pct", default=None),
                    "volume_ratio": value_for("volume_ratio", default=None),
                    "oi_zscore": value_for("oi_zscore", default=None),
                    "oi_pressure_score": value_for("oi_pressure_score", default=None),
                    "funding_rate": value_for("funding_rate", default=None),
                    "liquidation_pressure": value_for(
                        "liquidation_pressure",
                        "liquidation_imbalance",
                        default=None,
                    ),
                    "liquidation_imbalance": value_for(
                        "liquidation_imbalance",
                        "liquidation_pressure",
                        default=None,
                    ),
                    "aggressive_flow_imbalance": value_for(
                        "aggressive_flow_imbalance",
                        default=None,
                    ),
                    "oi_price_efficiency": value_for(
                        "oi_price_efficiency",
                        default=None,
                    ),
                }
                features = {key: value for key, value in features.items() if value is not None}

        if regime is None:
            regime_value = value_for(
                "regime",
                "oi_regime",
                "market_regime",
                "new_regime",
                "regime_type",
                default=None,
            )
            if regime_value is not None:
                regime = {
                    "regime": regime_value,
                    "confidence": value_for("regime_confidence", "confidence", default=0.0),
                    "score": value_for("regime_score", "score", default=0.0),
                    "reasons": value_for("regime_reasons", "reasons", default=[]),
                }

        # Only synthesize divergence/anomaly from flat fields when analytics
        # clearly marked this event as the corresponding setup.
        if divergence is None:
            divergence_type = value_for(
                "divergence_type",
                "price_oi_divergence",
                default=None,
            )
            divergence_detected = value_for(
                "divergence_detected",
                "is_divergence",
                default=None,
            )
            is_divergence_topic = (
                ".divergence" in topic
                or topic.endswith("divergence")
                or topic.endswith("oi.divergence")
                or topic.endswith("open_interest.divergence")
            )
            if divergence_type is None and is_divergence_topic:
                raw_side = str(value_for("side", "direction", "bias", default="")).strip().lower()
                if raw_side in {"long", "bullish", "buy", "up"}:
                    divergence_type = "bullish"
                elif raw_side in {"short", "bearish", "sell", "down"}:
                    divergence_type = "bearish"
                else:
                    divergence_type = "bullish" if float(value_for("score", default=0.0) or 0.0) >= 0 else "bearish"
            if divergence_type is not None or to_bool(divergence_detected, default=False) or is_divergence_topic:
                divergence = {
                    "detected": to_bool(divergence_detected, default=True),
                    "divergence_type": divergence_type,
                    "price_oi_divergence": value_for(
                        "price_oi_divergence",
                        default=divergence_type,
                    ),
                    "side": value_for("side", "direction", "bias", default=None),
                    "direction": value_for("direction", "side", "bias", default=None),
                    "confidence": value_for("divergence_confidence", "confidence", default=0.0),
                    "score": value_for("divergence_score", "score", default=0.0),
                    "window_size": value_for("divergence_window_size", "window_size", default=None),
                    "reasons": value_for("divergence_reasons", "reasons", default=[]),
                }

        if anomaly is None:
            anomaly_type = value_for("anomaly_type", default=None)
            anomaly_detected = value_for(
                "anomaly_detected",
                "is_anomaly",
                default=None,
            )
            capitulation = value_for("capitulation", default=None)
            squeeze_setup = value_for("squeeze_setup", default=None)
            if (
                anomaly_type is not None
                or to_bool(anomaly_detected, default=False)
                or to_bool(capitulation, default=False)
                or to_bool(squeeze_setup, default=False)
            ):
                anomaly = {
                    "detected": to_bool(
                        anomaly_detected,
                        default=anomaly_type is not None
                        or to_bool(capitulation, default=False)
                        or to_bool(squeeze_setup, default=False),
                    ),
                    "anomaly_type": anomaly_type,
                    "confidence": value_for("anomaly_confidence", "confidence", default=0.0),
                    "score": value_for("anomaly_score", "score", default=0.0),
                    "strength": value_for("anomaly_strength", "strength", default=value_for("score", default=0.0)),
                    "capitulation": capitulation,
                    "capitulation_score": value_for("capitulation_score", default=None),
                    "squeeze_setup": squeeze_setup,
                    "squeeze_score": value_for("squeeze_score", default=None),
                    "liquidation_imbalance": value_for(
                        "liquidation_imbalance",
                        "liquidation_pressure",
                        default=None,
                    ),
                    "reasons": value_for("anomaly_reasons", "reasons", default=[]),
                }


        if isinstance(divergence, dict):
            divergence_type = (
                divergence.get("divergence_type")
                or divergence.get("type")
                or divergence.get("price_oi_divergence")
            )
            if divergence_type is not None:
                divergence.setdefault("divergence_type", divergence_type)
                divergence.setdefault("type", divergence_type)
            divergence.setdefault("detected", True)
            divergence.setdefault("is_detected", divergence.get("detected", True))
            divergence.setdefault("confidence", value_for("divergence_confidence", "confidence", default=0.0))
            divergence.setdefault("score", value_for("divergence_score", "score", default=divergence.get("confidence", 0.0)))
            divergence.setdefault("reasons", value_for("divergence_reasons", "reasons", default=[]))

        if isinstance(anomaly, dict):
            anomaly_type = (
                anomaly.get("anomaly_type")
                or anomaly.get("type")
            )
            if anomaly_type is not None:
                anomaly.setdefault("anomaly_type", anomaly_type)
                anomaly.setdefault("type", anomaly_type)
            anomaly.setdefault("is_detected", anomaly.get("detected", False))
            anomaly.setdefault("confidence", value_for("anomaly_confidence", "confidence", default=0.0))
            anomaly.setdefault("score", value_for("anomaly_score", "score", default=anomaly.get("confidence", 0.0)))
            anomaly.setdefault("reasons", value_for("anomaly_reasons", "reasons", default=[]))



        def enum_label(value: Any) -> str | None:
            if value is None:
                return None
            raw = getattr(value, "value", value)
            text = str(raw).strip()
            if not text:
                return None
            return text.upper().replace("-", "_").replace(" ", "_")

        def normalize_oi_direction(value: Any) -> str | None:
            text = enum_label(value)
            if text in {"LONG", "BUY", "BULL", "BULLISH", "UP", "UPTREND"}:
                return "UP"
            if text in {"SHORT", "SELL", "BEAR", "BEARISH", "DOWN", "DOWNTREND"}:
                return "DOWN"
            if text in {"FLAT", "SIDEWAYS", "NEUTRAL"}:
                return "FLAT"
            return text

        def normalize_divergence_type(value: Any, *, fallback_side: Any = None) -> str:
            text = enum_label(value)
            if text in {None, "", "NONE", "UNKNOWN"}:
                side_text = enum_label(fallback_side)
                if side_text in {"LONG", "BUY", "BULL", "BULLISH", "UP"}:
                    return "BULLISH"
                if side_text in {"SHORT", "SELL", "BEAR", "BEARISH", "DOWN"}:
                    return "BEARISH"
                return "NONE"
            if text in {"LONG", "BUY", "BULL", "UP"}:
                return "BULLISH"
            if text in {"SHORT", "SELL", "BEAR", "DOWN"}:
                return "BEARISH"
            return text

        def normalize_anomaly_type(value: Any) -> str:
            text = enum_label(value)
            if text in {None, "", "NONE", "UNKNOWN"}:
                return "NONE"
            if text in {"SPIKE", "OI_SPIKE", "OPEN_INTEREST_SPIKE"}:
                return "OI_SPIKE"
            if text in {"COLLAPSE", "OI_COLLAPSE", "OPEN_INTEREST_COLLAPSE"}:
                return "OI_COLLAPSE"
            return text

        def normalize_regime(value: Any) -> str:
            text = enum_label(value)
            if text in {None, "", "UNKNOWN"}:
                return "NEUTRAL"
            mapping = {
                "EXPANSION": "TREND_CONFIRMATION",
                "OI_EXPANSION": "TREND_CONFIRMATION",
                "TRENDING": "TREND_CONFIRMATION",
                "BUILDUP": "LONG_BUILDUP",
                "LONG": "LONG_BUILDUP",
                "SHORT": "SHORT_BUILDUP",
                "CONTRACTION": "NEUTRAL",
                "RANGE": "NEUTRAL",
                "RANGING": "NEUTRAL",
                "EXTREME_NEGATIVE": "CAPITULATION",
                "EXTREME_POSITIVE": "OVERHEATED",
            }
            return mapping.get(text, text)

        def normalize_oi_section(section: dict[str, Any] | None, kind: str) -> dict[str, Any] | None:
            if section is None:
                return None

            result = dict(section)
            if kind == "divergence":
                result["detected"] = to_bool(
                    result.get("detected", result.get("is_detected", True)),
                    default=True,
                )
                result["divergence_type"] = normalize_divergence_type(
                    result.get("divergence_type")
                    or result.get("type")
                    or result.get("price_oi_divergence"),
                    fallback_side=result.get("side") or result.get("direction") or payload.get("side") or payload.get("direction"),
                )
                result.setdefault("type", result["divergence_type"])
                result.setdefault("is_detected", result["detected"])
                result.setdefault("confidence", value_for("divergence_confidence", "confidence", default=0.0))
                result.setdefault("score", value_for("divergence_score", "score", default=result.get("confidence", 0.0)))
                result.setdefault("window_size", value_for("divergence_window_size", "window_size", default=None))
                result.setdefault("reasons", value_for("divergence_reasons", "reasons", default=[]))

            elif kind == "anomaly":
                result["detected"] = to_bool(
                    result.get("detected", result.get("is_detected", True)),
                    default=True,
                )
                result["anomaly_type"] = normalize_anomaly_type(
                    result.get("anomaly_type") or result.get("type")
                )
                result.setdefault("type", result["anomaly_type"])
                result.setdefault("is_detected", result["detected"])
                result.setdefault("strength", enum_label(result.get("strength")) or "MEDIUM")
                result.setdefault("confidence", value_for("anomaly_confidence", "confidence", default=0.0))
                result.setdefault("score", value_for("anomaly_score", "score", default=result.get("confidence", 0.0)))
                result.setdefault("reasons", value_for("anomaly_reasons", "reasons", default=[]))

            elif kind == "regime":
                result["regime"] = normalize_regime(
                    result.get("regime")
                    or result.get("regime_type")
                    or result.get("type")
                    or result.get("state")
                )
                result.setdefault("confidence", value_for("regime_confidence", "confidence", default=0.0))
                result.setdefault("score", value_for("regime_score", "score", default=result.get("confidence", 0.0)))
                result.setdefault("reasons", value_for("regime_reasons", "reasons", default=[]))

            elif kind == "features":
                result.setdefault("oi", value_for("oi", "open_interest", default=result.get("open_interest")))
                result.setdefault("open_interest", value_for("open_interest", "oi", default=result.get("oi")))
                result.setdefault("oi_direction", normalize_oi_direction(result.get("oi_direction") or payload.get("oi_direction") or payload.get("direction")))
                result.setdefault("price_direction", normalize_oi_direction(result.get("price_direction") or payload.get("price_direction")))

            return result

        divergence = normalize_oi_section(divergence, "divergence")
        anomaly = normalize_oi_section(anomaly, "anomaly")
        regime = normalize_oi_section(regime, "regime")
        features = normalize_oi_section(features, "features")

        set_aliases(
            "analysis",
            ("oi_analysis", "open_interest_analysis", "result"),
            analysis,
        )
        set_aliases(
            "snapshot",
            ("oi_snapshot", "open_interest_snapshot"),
            snapshot,
        )
        set_aliases(
            "market_context",
            ("context", "oi_context", "open_interest_context"),
            market_context,
        )
        set_aliases(
            "features",
            ("oi_features", "open_interest_features"),
            features if features else None,
        )
        set_aliases(
            "regime",
            ("regime_result", "oi_regime", "open_interest_regime", "new_regime"),
            regime,
        )

        if section_detected(divergence):
            set_aliases(
                "divergence",
                ("divergence_result", "oi_divergence", "open_interest_divergence"),
                divergence,
            )

        if section_detected(anomaly):
            set_aliases(
                "anomaly",
                ("anomaly_result", "oi_anomaly", "open_interest_anomaly"),
                anomaly,
            )

        domain_data.setdefault("raw", dict(payload))

    def _augment_liquidations_domain_data(
            self,
            *,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        """
        Normalize analytics.liquidations.* payloads into the stable liquidations
        StrategyContext contract.

        OI-style contract adapter only:
        - exposes analytics-provided sections under stable canonical aliases;
        - extracts nested analysis/result sections;
        - enriches flat cluster/snapshot-like context only when safe;
        - does NOT synthesize cascade/exhaustion/squeeze/signal sections unless
          analytics explicitly supplied a detected/actionable context.
        """
        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def mapping_for(*keys: str) -> dict[str, Any] | None:
            for key in keys:
                value = payload.get(key)
                if isinstance(value, dict):
                    return dict(value)

                value = feature_map.get(key)
                if isinstance(value, dict):
                    return dict(value)

            return None

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]
            return default

        topic = (
            str(
                payload.get("event_name")
                or payload.get("topic")
                or payload.get("source_topic")
                or ""
            )
            .strip()
            .lower()
        )

        def to_bool(value: Any, default: bool = False) -> bool:
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
                    "detected",
                    "active",
                    "confirmed",
                    "triggered",
                    "valid",
                }:
                    return True

                if normalized in {
                    "0",
                    "false",
                    "no",
                    "n",
                    "off",
                    "none",
                    "not_detected",
                    "inactive",
                    "rejected",
                    "invalid",
                    "expired",
                }:
                    return False

            if isinstance(value, (int, float)):
                return bool(value)

            return default

        def section_detected(section: dict[str, Any] | None) -> bool:
            if not section:
                return False

            detected = section.get(
                "detected",
                section.get(
                    "is_detected",
                    section.get(
                        "active",
                        section.get(
                            "confirmed",
                            section.get("valid", None),
                        ),
                    ),
                ),
            )

            if detected is None:
                # Typed nested analytics section without explicit detected flag is
                # considered analytics-provided/actionable. Empty dicts are rejected.
                return True

            return to_bool(detected, default=False)

        def set_aliases(
                target: str,
                aliases: tuple[str, ...],
                value: dict[str, Any] | None,
                *,
                override: bool = True,
        ) -> None:
            if value is None:
                return

            # Canonical sections override raw/flat fields already copied into
            # domain_data. This mirrors the OI contract behavior.
            if override:
                domain_data[target] = value
            else:
                domain_data.setdefault(target, value)

            for alias in aliases:
                domain_data.setdefault(alias, value)

        analysis = mapping_for(
            "analysis",
            "liquidations_analysis",
            "liquidation_analysis",
        )
        result = mapping_for(
            "result",
            "liquidations_result",
            "liquidation_result",
        )
        cascade = mapping_for(
            "cascade",
            "cascade_result",
            "cascade_detection",
            "cascade_detected",
            "liquidation_cascade",
        )
        exhaustion = mapping_for(
            "exhaustion",
            "exhaustion_result",
            "exhaustion_detection",
            "exhaustion_detected",
            "reversal_context",
            "liquidation_exhaustion",
        )
        squeeze = mapping_for(
            "squeeze",
            "squeeze_result",
            "squeeze_reversal",
            "squeeze_context",
            "pending_confirmation",
            "liquidation_squeeze",
        )
        cluster = mapping_for(
            "cluster",
            "liquidation_cluster",
            "cluster_stats",
            "liquidations_cluster",
        )
        signal = mapping_for(
            "signal",
            "liquidation_signal",
            "liquidations_signal",
            "analytics_signal",
            "setup",
        )

        # Some analytics modules publish the primary result under generic "result".
        # Resolve it by topic/shape, but do not blindly expose result as cascade.
        if result is not None:
            result_kind = (
                str(
                    result.get("kind")
                    or result.get("type")
                    or result.get("result_type")
                    or result.get("setup_type")
                    or ""
                )
                .strip()
                .lower()
            )

            if cascade is None and (
                    "cascade" in topic
                    or "cascade" in result_kind
                    or "continuation_bias" in result
                    or "intensity_score" in result
            ):
                cascade = dict(result)

            if exhaustion is None and (
                    "exhaustion" in topic
                    or "reversal" in topic
                    or "exhaustion" in result_kind
                    or "exhaustion_bias" in result
                    or "bias_delta" in result
            ):
                exhaustion = dict(result)

            if squeeze is None and (
                    "squeeze" in topic
                    or "squeeze" in result_kind
                    or "squeeze_score" in result
                    or "squeeze_direction" in result
            ):
                squeeze = dict(result)

        # Nested analysis/result extraction.
        for container in (analysis, result):
            if container is None:
                continue

            nested_cascade = (
                    container.get("cascade")
                    or container.get("cascade_result")
                    or container.get("cascade_detection")
                    or container.get("liquidation_cascade")
            )
            nested_exhaustion = (
                    container.get("exhaustion")
                    or container.get("exhaustion_result")
                    or container.get("exhaustion_detection")
                    or container.get("reversal_context")
            )
            nested_squeeze = (
                    container.get("squeeze")
                    or container.get("squeeze_result")
                    or container.get("squeeze_reversal")
                    or container.get("squeeze_context")
            )
            nested_cluster = (
                    container.get("cluster")
                    or container.get("liquidation_cluster")
                    or container.get("cluster_stats")
            )
            nested_signal = (
                    container.get("signal")
                    or container.get("liquidation_signal")
                    or container.get("analytics_signal")
                    or container.get("setup")
            )

            if isinstance(nested_cascade, dict) and cascade is None:
                cascade = dict(nested_cascade)
            if isinstance(nested_exhaustion, dict) and exhaustion is None:
                exhaustion = dict(nested_exhaustion)
            if isinstance(nested_squeeze, dict) and squeeze is None:
                squeeze = dict(nested_squeeze)
            if isinstance(nested_cluster, dict) and cluster is None:
                cluster = dict(nested_cluster)
            if isinstance(nested_signal, dict) and signal is None:
                signal = dict(nested_signal)

        # Safe flat enrichment for cluster only. Cluster is context, not a setup.
        if cluster is None:
            flat_cluster: dict[str, Any] = {}
            for key in (
                    "duration_seconds",
                    "avg_notional_per_event",
                    "side_imbalance_ratio",
                    "event_imbalance_ratio",
                    "acceleration_ratio",
                    "event_count",
                    "total_notional_usd",
                    "price_range_pct",
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "timestamp",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    flat_cluster[key] = value

            if flat_cluster:
                cluster = flat_cluster

        # Flat cascade only when analytics explicitly indicates cascade context.
        if cascade is None and (
                "cascade" in topic
                or value_for("cascade_detected", default=None) is not None
                or value_for("cascade_direction", "continuation_bias", default=None) is not None
        ):
            cascade = {
                "detected": to_bool(value_for("cascade_detected", default=True), True),
                "confirmed": to_bool(value_for("confirmed", default=True), True),
                "confidence": value_for("confidence", "cascade_confidence", default=0.0),
                "intensity_score": value_for(
                    "intensity_score",
                    "intensity",
                    default=0.0,
                ),
                "direction": value_for(
                    "direction",
                    "cascade_direction",
                    "side",
                    default=None,
                ),
                "severity": value_for(
                    "severity",
                    "severity_label",
                    default=None,
                ),
                "continuation_bias": value_for(
                    "continuation_bias",
                    default=0.0,
                ),
                "exhaustion_bias": value_for(
                    "exhaustion_bias",
                    default=0.0,
                ),
                "total_notional_usd": value_for(
                    "total_notional_usd",
                    "notional_usd",
                    default=0.0,
                ),
                "event_count": value_for("event_count", default=0),
                "price_range_pct": value_for("price_range_pct", default=None),
            }

        # Flat exhaustion only when analytics explicitly indicates exhaustion.
        if exhaustion is None and (
                "exhaustion" in topic
                or "reversal" in topic
                or value_for("exhaustion_detected", default=None) is not None
                or value_for("exhaustion_bias", "bias_delta", default=None) is not None
        ):
            exhaustion = {
                "detected": to_bool(value_for("exhaustion_detected", default=True), True),
                "confirmed": to_bool(
                    value_for("exhaustion_confirmed", "confirmed", default=True),
                    True,
                ),
                "confidence": value_for(
                    "confidence",
                    "exhaustion_confidence",
                    default=0.0,
                ),
                "intensity_score": value_for(
                    "intensity_score",
                    "intensity",
                    default=0.0,
                ),
                "direction": value_for(
                    "direction",
                    "cascade_direction",
                    "side",
                    default=None,
                ),
                "severity": value_for(
                    "severity",
                    "severity_label",
                    default=None,
                ),
                "exhaustion_bias": value_for("exhaustion_bias", default=0.0),
                "bias_delta": value_for("bias_delta", default=0.0),
                "continuation_bias": value_for("continuation_bias", default=0.0),
                "total_notional_usd": value_for(
                    "total_notional_usd",
                    "notional_usd",
                    default=0.0,
                ),
                "event_count": value_for("event_count", default=0),
                "price_range_pct": value_for("price_range_pct", default=None),
            }

        # Flat squeeze only when analytics explicitly indicates squeeze context.
        if squeeze is None and (
                "squeeze" in topic
                or value_for("squeeze_confirmed", "squeeze_score", default=None) is not None
        ):
            squeeze = {
                "detected": to_bool(value_for("squeeze_detected", default=True), True),
                "confirmed": to_bool(
                    value_for("squeeze_confirmed", "confirmed", default=True),
                    True,
                ),
                "score": value_for("squeeze_score", "score", default=0.0),
                "direction": value_for(
                    "squeeze_direction",
                    "direction",
                    "side",
                    default=None,
                ),
                "confidence": value_for(
                    "squeeze_confidence",
                    "confidence",
                    default=0.0,
                ),
            }

        signal_like_topic = any(
            token in topic
            for token in (
                "signal",
                "setup",
                "confirmed",
                "generated",
            )
        )
        explicit_signal = bool(
            signal_like_topic
            or value_for("signal_type", "setup_type", "signal_side", default=None)
            is not None
            or to_bool(value_for("signal_detected", "setup_detected", default=False))
        )

        if signal is None and explicit_signal:
            signal = {
                "detected": True,
                "type": value_for("signal_type", "setup_type", default=None),
                "side": value_for(
                    "signal_side",
                    "side",
                    "direction",
                    default=None,
                ),
                "score": value_for("signal_score", "score", default=0.0),
                "confidence": value_for(
                    "signal_confidence",
                    "confidence",
                    default=0.0,
                ),
                "origin": value_for(
                    "origin",
                    "signal_origin",
                    default="liquidations",
                ),
            }

        set_aliases("analysis", ("liquidations_analysis",), analysis)
        set_aliases("cluster", ("liquidation_cluster", "cluster_stats"), cluster)

        if section_detected(cascade):
            set_aliases(
                "cascade",
                (
                    "cascade_result",
                    "cascade_detection",
                    "cascade_detected",
                    "liquidation_cascade",
                    "result",
                ),
                cascade,
            )

        if section_detected(exhaustion):
            set_aliases(
                "exhaustion",
                (
                    "exhaustion_result",
                    "exhaustion_detection",
                    "exhaustion_detected",
                    "reversal_context",
                ),
                exhaustion,
            )

        if section_detected(squeeze):
            set_aliases(
                "squeeze",
                (
                    "squeeze_result",
                    "squeeze_reversal",
                    "squeeze_context",
                    "pending_confirmation",
                ),
                squeeze,
            )

        if section_detected(signal):
            set_aliases(
                "signal",
                (
                    "liquidation_signal",
                    "liquidations_signal",
                    "analytics_signal",
                    "setup",
                ),
                signal,
            )

    def _augment_liquidity_domain_data(
            self,
            *,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        """
        Normalize analytics.liquidity.* payloads into the stable liquidity
        StrategyContext contract.

        OI-style contract adapter only:
        - exposes analytics-provided liquidity map snapshot under stable aliases;
        - preserves typed LiquidityMapSnapshot objects if analytics provided them;
        - exposes levels / clusters / zones / signal under canonical sections;
        - enriches context scores only when they are present;
        - does NOT fabricate a liquidity snapshot from a generic flat payload.
        """
        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def as_mapping(value: Any) -> dict[str, Any] | None:
            if isinstance(value, dict):
                return value

            to_dict = getattr(value, "to_dict", None)
            if callable(to_dict):
                converted = to_dict()
                if isinstance(converted, dict):
                    return converted

            return None

        def get_item(value: Any, key: str, default: Any = None) -> Any:
            mapping = as_mapping(value)
            if mapping is not None:
                return mapping.get(key, default)
            return getattr(value, key, default)

        def mapping_or_object_for(*keys: str) -> Any | None:
            for key in keys:
                if key in payload and payload[key] is not None:
                    return payload[key]
                if key in feature_map and feature_map[key] is not None:
                    return feature_map[key]
            return None

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]

            snapshot = domain_data.get("snapshot")
            for key in keys:
                value = get_item(snapshot, key, default=None)
                if value is not None:
                    return value

            signal = domain_data.get("signal")
            for key in keys:
                value = get_item(signal, key, default=None)
                if value is not None:
                    return value

            return default

        def set_aliases(
                target: str,
                aliases: tuple[str, ...],
                value: Any | None,
                *,
                override: bool = True,
        ) -> None:
            if value is None:
                return

            if override:
                domain_data[target] = value
            else:
                domain_data.setdefault(target, value)

            for alias in aliases:
                domain_data.setdefault(alias, value)

        def to_bool(value: Any, default: bool = False) -> bool:
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
                    "active",
                    "valid",
                    "confirmed",
                    "detected",
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
                    "none",
                    "not_detected",
                }:
                    return False

            if isinstance(value, (int, float)):
                return bool(value)

            return default

        def section_present(value: Any | None) -> bool:
            if value is None:
                return False

            mapping = as_mapping(value)
            if mapping is None:
                return True

            if not mapping:
                return False

            detected = (
                mapping.get("detected")
                if "detected" in mapping
                else mapping.get("active", mapping.get("valid", None))
            )

            if detected is None:
                return True

            return to_bool(detected, default=False)

        topic = (
            str(
                payload.get("event_name")
                or payload.get("topic")
                or payload.get("source_topic")
                or ""
            )
            .strip()
            .lower()
        )

        analysis = mapping_or_object_for(
            "analysis",
            "liquidity_analysis",
            "result",
        )

        snapshot = mapping_or_object_for(
            "snapshot",
            "liquidity_map_snapshot",
            "map_snapshot",
            "last_snapshot",
            "liquidity_snapshot",
        )

        # Some analytics envelopes publish {"result": {"snapshot": ...}} or
        # {"liquidity": {"snapshot": ...}}. Preserve the nested typed object.
        for container_key in (
                "result",
                "liquidity",
                "data",
                "payload",
                "analysis",
                "liquidity_analysis",
        ):
            container = mapping_or_object_for(container_key)
            container_map = as_mapping(container)
            if container_map is None:
                continue

            nested_snapshot = (
                    container_map.get("snapshot")
                    or container_map.get("liquidity_map_snapshot")
                    or container_map.get("map_snapshot")
                    or container_map.get("last_snapshot")
            )
            if nested_snapshot is not None and snapshot is None:
                snapshot = nested_snapshot

        signal = mapping_or_object_for(
            "signal",
            "liquidity_signal",
            "analytics_signal",
            "setup",
        )

        levels = mapping_or_object_for(
            "levels",
            "active_levels",
            "liquidity_levels",
        )
        clusters = mapping_or_object_for(
            "clusters",
            "stop_clusters",
            "liquidity_clusters",
        )
        zones = mapping_or_object_for(
            "zones",
            "liquidity_zones",
        )
        equal_levels = mapping_or_object_for(
            "equal_levels",
            "equal_highs_lows",
        )

        if snapshot is not None:
            snapshot_map = as_mapping(snapshot)

            nested_levels = get_item(snapshot, "levels", None)
            nested_active_levels = get_item(snapshot, "active_levels", None)
            nested_equal_levels = get_item(snapshot, "equal_levels", None)
            nested_clusters = get_item(snapshot, "stop_clusters", None)
            nested_zones = get_item(snapshot, "zones", None)

            if levels is None and nested_levels is not None:
                levels = nested_levels
            if levels is None and nested_active_levels is not None:
                levels = nested_active_levels
            if equal_levels is None and nested_equal_levels is not None:
                equal_levels = nested_equal_levels
            if clusters is None and nested_clusters is not None:
                clusters = nested_clusters
            if zones is None and nested_zones is not None:
                zones = nested_zones

            # Preserve scope-like fields in liquidity domain for metadata/scope.
            for key in (
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "current_price",
                    "price",
                    "mark_price",
                    "last_price",
            ):
                value = get_item(snapshot, key, None)
                if value is not None:
                    domain_data.setdefault(key, value)

            if snapshot_map is not None:
                metadata = snapshot_map.get("metadata")
                if isinstance(metadata, dict):
                    for key in (
                            "exchange",
                            "market_type",
                            "symbol",
                            "exchange_symbol",
                            "timeframe",
                            "current_price",
                            "price",
                            "mark_price",
                    ):
                        if key in metadata:
                            domain_data.setdefault(key, metadata[key])

        # Only expose a signal/setup if analytics explicitly supplied one.
        signal_like_topic = any(
            token in topic
            for token in (
                "signal",
                "setup",
                "bias",
                "sweep",
                "stop_hunt",
                "equal",
            )
        )
        explicit_signal = bool(
            signal_like_topic
            or payload.get("signal_type") is not None
            or payload.get("setup_type") is not None
            or payload.get("signal_side") is not None
            or to_bool(payload.get("signal_detected", False))
            or to_bool(payload.get("setup_detected", False))
        )

        if signal is None and explicit_signal:
            signal = {
                "detected": True,
                "type": payload.get("signal_type") or payload.get("setup_type"),
                "side": (
                        payload.get("signal_side")
                        or payload.get("side")
                        or payload.get("direction")
                        or payload.get("bias")
                ),
                "score": payload.get("signal_score", payload.get("score", 0.0)),
                "confidence": payload.get(
                    "signal_confidence",
                    payload.get("confidence", 0.0),
                ),
                "origin": payload.get("origin", "liquidity"),
            }

        set_aliases(
            "analysis",
            ("liquidity_analysis", "result"),
            analysis,
            override=False,
        )

        # Critical: do not set snapshot to raw payload. Concrete liquidity strategies
        # require an unwrap-able LiquidityMapSnapshot / snapshot wrapper.
        if snapshot is not None:
            set_aliases(
                "snapshot",
                (
                    "liquidity_map_snapshot",
                    "map_snapshot",
                    "last_snapshot",
                    "liquidity_snapshot",
                ),
                snapshot,
            )

        if levels is not None:
            set_aliases(
                "levels",
                ("active_levels", "liquidity_levels"),
                levels,
            )

        if equal_levels is not None:
            set_aliases(
                "equal_levels",
                ("equal_highs_lows",),
                equal_levels,
            )

        if clusters is not None:
            set_aliases(
                "clusters",
                ("stop_clusters", "liquidity_clusters"),
                clusters,
            )

        if zones is not None:
            set_aliases(
                "zones",
                ("liquidity_zones",),
                zones,
            )

        if section_present(signal):
            set_aliases(
                "signal",
                ("liquidity_signal", "analytics_signal", "setup"),
                signal,
            )

        current_price = value_for(
            "current_price",
            "price",
            "mark_price",
            "last_price",
            "close",
            default=None,
        )
        if current_price is not None:
            domain_data.setdefault("current_price", current_price)

        for key, aliases in {
            "above_liquidity_score": ("above_score",),
            "below_liquidity_score": ("below_score",),
            "liquidity_pressure_score": ("pressure_score", "liquidity_pressure"),
            "bias": ("liquidity_bias", "direction"),
            "upside_sweep_risk": ("sweep_risk_up", "sweep_risk.up"),
            "downside_sweep_risk": ("sweep_risk_down", "sweep_risk.down"),
            "upside_magnet_score": ("magnet_up", "magnet.up"),
            "downside_magnet_score": ("magnet_down", "magnet.down"),
            "nearest_above_level": ("nearest_above",),
            "nearest_below_level": ("nearest_below",),
            "strongest_cluster_above": ("strongest_above",),
            "strongest_cluster_below": ("strongest_below",),
        }.items():
            value = value_for(key, *aliases, default=None)
            if value is not None:
                domain_data.setdefault(key, value)

    def _augment_price_action_domain_data(
            self,
            *,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        """
        Normalize analytics.price_action.* payloads into the stable
        price-action domain contract consumed by concrete strategies.

        Important:
        - module sections are populated only when the payload actually belongs
          to that module;
        - event-specific payloads are wrapped into the shape expected by
          concrete strategies, e.g. market_structure.last_break_event,
          trend.last_signal, fair_value_gap.last_event;
        - missing FVG/SR/Trend/MarketStructure sections are not synthesized,
          preventing registry from routing incompatible strategies.
        """
        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def mapping_for(*keys: str) -> dict[str, Any] | None:
            for key in keys:
                value = payload.get(key)
                if isinstance(value, dict):
                    return dict(value)

                value = feature_map.get(key)
                if isinstance(value, dict):
                    return dict(value)

            return None

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]
            return default

        def set_aliases(target: str, aliases: tuple[str, ...], value: dict[str, Any] | None) -> None:
            if value is None:
                return

            # Canonical module keys must override the raw flat payload copied
            # into domain_data earlier. Concrete strategies read these primary
            # keys first, so setdefault() leaves stale flat views in place.
            domain_data[target] = value
            for alias in aliases:
                domain_data.setdefault(alias, value)

        def score_from(mapping: dict[str, Any] | None, default: float = 0.0) -> float:
            raw = default
            if mapping is not None:
                raw = mapping.get("score", mapping.get("confidence", default))
            try:
                return max(0.0, min(1.0, float(raw)))
            except (TypeError, ValueError):
                return default

        def trend_direction_for_strategy(value: Any) -> str | None:
            raw = str(value or "").strip().lower()
            if raw in {"long", "buy", "up", "uptrend", "bull", "bullish"}:
                return "bullish"
            if raw in {"short", "sell", "down", "downtrend", "bear", "bearish"}:
                return "bearish"
            if raw in {"neutral", "flat", "range", "ranging"}:
                return "neutral"
            return raw or None

        def trend_regime_for_strategy(value: Any) -> str | None:
            raw = str(value or "").strip().lower()
            if raw in {"uptrend", "downtrend", "trend", "trending_up", "trending_down", "bullish", "bearish"}:
                return "trending"
            if raw in {"pullback", "retracement"}:
                return "pullback"
            if raw in {"consolidation", "consolidating", "range", "ranging"}:
                return "consolidating"
            if raw in {"reversal", "reversing"}:
                return "reversing"
            if raw in {"exhausted", "exhaustion"}:
                return "exhausted"
            return raw or None

        topic = (
            str(
                payload.get("event_name")
                or payload.get("topic")
                or payload.get("source_topic")
                or ""
            )
            .strip()
            .lower()
        )

        state = mapping_for("state", "composite", "price_action", "snapshot", "result")
        if state is not None:
            domain_data.setdefault("state", state)
            domain_data.setdefault("composite", state)
            domain_data.setdefault("price_action", state)
            domain_data.setdefault("snapshot", state)

        market_structure = mapping_for("market_structure", "structure", "ms")
        support_resistance = mapping_for("support_resistance", "sr", "levels")
        fair_value_gap = mapping_for("fair_value_gap", "fvg", "fair_value_gaps")
        trend = mapping_for("trend", "trend_state")
        liquidity_levels = mapping_for("liquidity_levels", "liquidity")

        if isinstance(state, dict):
            if market_structure is None:
                market_structure = mapping_for("state.market_structure")
                for key in ("market_structure", "structure", "ms"):
                    nested = state.get(key)
                    if isinstance(nested, dict):
                        market_structure = dict(nested)
                        break
            if support_resistance is None:
                for key in ("support_resistance", "sr", "levels"):
                    nested = state.get(key)
                    if isinstance(nested, dict):
                        support_resistance = dict(nested)
                        break
            if fair_value_gap is None:
                for key in ("fair_value_gap", "fvg", "fair_value_gaps"):
                    nested = state.get(key)
                    if isinstance(nested, dict):
                        fair_value_gap = dict(nested)
                        break
            if trend is None:
                for key in ("trend", "trend_state"):
                    nested = state.get(key)
                    if isinstance(nested, dict):
                        trend = dict(nested)
                        break
            if liquidity_levels is None:
                for key in ("liquidity_levels", "liquidity"):
                    nested = state.get(key)
                    if isinstance(nested, dict):
                        liquidity_levels = dict(nested)
                        break

        event_payload = dict(payload)
        event_payload.setdefault("timestamp", value_for("timestamp", "event_time", default=None))
        event_payload.setdefault("price", value_for("price", "current_price", "last_price", "close", default=None))
        event_payload.setdefault("confidence", value_for("confidence", default=0.0))
        event_payload.setdefault("score", value_for("score", default=event_payload.get("confidence", 0.0)))

        # Event-specific fallback: market structure events.
        is_market_structure_event = (
            "market_structure" in topic
            or ".structure" in topic
            or any(
                key in payload
                for key in (
                    "swing_type",
                    "break_distance_pct",
                    "market_bias",
                    "broken_side",
                    "mtf_alignment",
                )
            )
        )
        if market_structure is None and is_market_structure_event:
            event_type = value_for("event_type", "type", "kind", default=None)
            if event_type is None:
                if ".bos" in topic or topic.endswith("bos"):
                    event_type = "bos"
                elif ".choch" in topic:
                    event_type = "choch"
                elif ".mss" in topic:
                    event_type = "mss"
            side = value_for("side", "direction", "bias", default=None)
            bias = value_for("market_bias", "bias", "direction", default=side)
            event_payload.setdefault("event_type", event_type)
            event_payload.setdefault("side", side)
            event_payload.setdefault("direction", side)
            event_payload.setdefault("market_bias", bias)
            event_payload.setdefault("confirmed", True)
            market_structure = {
                "last_break_event": event_payload,
                "last_event": event_payload,
                "event": event_payload,
                "external": {
                    "last_break_event": event_payload,
                    "last_event": event_payload,
                    "bias": bias,
                    "market_bias": bias,
                    "confidence": event_payload.get("confidence", 0.0),
                    "score": event_payload.get("score", 0.0),
                    "strength": event_payload.get("score", event_payload.get("confidence", 0.0)),
                    "trend_strength": event_payload.get("score", event_payload.get("confidence", 0.0)),
                },
                "internal": {
                    "bias": bias,
                    "market_bias": bias,
                    "confidence": event_payload.get("confidence", 0.0),
                    "score": event_payload.get("score", 0.0),
                },
                "mtf_alignment": value_for("mtf_alignment", "alignment", default=event_payload.get("score", 0.0)),
                "trend_strength": event_payload.get("score", event_payload.get("confidence", 0.0)),
            }

        # Event-specific fallback: support/resistance reactions.
        is_sr_event = (
            "support_resistance" in topic
            or topic.endswith(".sr")
            or ".sr." in topic
            or any(
                key in payload
                for key in (
                    "level_type",
                    "level_status",
                    "level_price",
                    "touch_count",
                    "reaction_count",
                    "break_count",
                )
            )
        )
        if support_resistance is None and is_sr_event:
            event_payload.setdefault("event_type", value_for("event_type", "type", "kind", default="reaction"))
            event_payload.setdefault("confirmed", True)
            level = {
                **event_payload,
                "price": value_for("level_price", "level", "price", "current_price", "close", default=None),
                "level_type": value_for("level_type", "type", default=None),
                "strength": value_for("strength", "score", "confidence", default=0.0),
                "confidence": event_payload.get("confidence", 0.0),
                "score": event_payload.get("score", event_payload.get("confidence", 0.0)),
                "touch_count": value_for("touch_count", "touches", default=1),
                "is_active": True,
            }
            support_resistance = {
                "last_event": event_payload,
                "event": event_payload,
                "nearest_support": level if str(level.get("level_type", "")).lower() == "support" else None,
                "nearest_resistance": level if str(level.get("level_type", "")).lower() == "resistance" else None,
                "external": {"last_event": event_payload, "confidence": event_payload.get("confidence", 0.0), "score": event_payload.get("score", 0.0)},
                "internal": {"last_event": event_payload, "confidence": event_payload.get("confidence", 0.0), "score": event_payload.get("score", 0.0)},
            }

        # Event-specific fallback: FVG reactions.
        is_fvg_event = (
            "fair_value_gap" in topic
            or "fvg" in topic
            or any(
                key in payload
                for key in (
                    "fvg_direction",
                    "gap_size_pct",
                    "fill_pct",
                    "upper_price",
                    "lower_price",
                    "mid_price",
                )
            )
        )
        if fair_value_gap is None and is_fvg_event:
            event_payload.setdefault("event_type", value_for("event_type", "type", "kind", default="retest"))
            event_payload.setdefault("confirmed", True)
            gap = {
                **event_payload,
                "direction": value_for("direction", "fvg_direction", "side", default=None),
                "status": value_for("status", "fvg_status", default="active"),
                "upper_price": value_for("upper_price", "upper", "high", default=None),
                "lower_price": value_for("lower_price", "lower", "low", default=None),
                "mid_price": value_for("mid_price", "mid", default=None),
                "strength": value_for("strength", "score", "confidence", default=0.0),
                "confidence": event_payload.get("confidence", 0.0),
                "score": event_payload.get("score", event_payload.get("confidence", 0.0)),
                "is_valid": True,
            }
            direction_label = str(gap.get("direction") or "").lower()
            fair_value_gap = {
                "last_event": event_payload,
                "event": event_payload,
                "nearest_bullish_gap": gap if "bull" in direction_label or direction_label == "long" else None,
                "nearest_bearish_gap": gap if "bear" in direction_label or direction_label == "short" else None,
                "external": {"last_event": event_payload, "confidence": event_payload.get("confidence", 0.0), "score": event_payload.get("score", 0.0)},
                "internal": {"last_event": event_payload, "confidence": event_payload.get("confidence", 0.0), "score": event_payload.get("score", 0.0)},
            }

        # Event-specific fallback: trend continuation/state events.
        is_trend_event = (
            "trend" in topic
            or any(
                key in payload
                for key in (
                    "trend_direction",
                    "trend_regime",
                    "continuation_probability",
                    "reversal_risk",
                    "exhaustion_score",
                    "overall_trend_score",
                )
            )
        )
        if trend is None and is_trend_event:
            event_payload.setdefault("event_type", value_for("event_type", "type", "kind", default="trend_alignment"))
            raw_trend_direction = value_for("trend_direction", "direction", "side", default=None)
            raw_trend_regime = value_for("trend_regime", "regime", "state", default=None)
            normalized_trend_direction = trend_direction_for_strategy(raw_trend_direction)
            normalized_trend_regime = trend_regime_for_strategy(raw_trend_regime or normalized_trend_direction)
            event_payload["direction"] = normalized_trend_direction or event_payload.get("direction")
            event_payload["trend_direction"] = normalized_trend_direction or event_payload.get("trend_direction")
            event_payload["trend_regime"] = normalized_trend_regime or event_payload.get("trend_regime")
            event_payload["regime"] = normalized_trend_regime or event_payload.get("regime")
            event_payload.setdefault("continuation_probability", value_for("continuation_probability", "continuation_prob", "probability", default=event_payload.get("score", 0.0)))
            event_payload.setdefault("confirmed", True)
            layer = {
                **event_payload,
                "direction": trend_direction_for_strategy(event_payload.get("trend_direction") or event_payload.get("direction")),
                "trend_direction": trend_direction_for_strategy(event_payload.get("trend_direction") or event_payload.get("direction")),
                "regime": trend_regime_for_strategy(event_payload.get("trend_regime") or event_payload.get("regime")),
                "trend_regime": trend_regime_for_strategy(event_payload.get("trend_regime") or event_payload.get("regime")),
                "trend_strength": value_for("trend_strength", "strength", "score", default=event_payload.get("score", 0.0)),
                "momentum_score": value_for("momentum_score", "momentum", default=event_payload.get("score", 0.0)),
                "slope_score": value_for("slope_score", "slope", default=event_payload.get("score", 0.0)),
                "continuation_probability": event_payload.get("continuation_probability"),
                "confidence": event_payload.get("confidence", 0.0),
                "score": event_payload.get("score", event_payload.get("confidence", 0.0)),
            }
            trend = {
                "last_signal": event_payload,
                "last_event": event_payload,
                "event": event_payload,
                "external": layer,
                "internal": layer,
                "internal_external_alignment": value_for("internal_external_alignment", "alignment", default=event_payload.get("score", 0.0)),
                "higher_timeframe_alignment": value_for("higher_timeframe_alignment", "htf_alignment", default=event_payload.get("score", 0.0)),
                "overall_trend_score": value_for("overall_trend_score", "score", default=event_payload.get("score", 0.0)),
            }


        # Compatibility shape for concrete price_action strategies:
        # Direct analytics event payloads often provide a flat module dict, while
        # concrete strategies expect a module with external/internal layer views.
        # Keep this in SignalNormalizer, not in concrete strategies.
        if isinstance(market_structure, dict):
            market_structure = self._ensure_price_action_market_structure_view(
                module=market_structure,
                event_payload=event_payload,
                fallback_topic=topic,
            )
        if isinstance(trend, dict):
            trend = self._ensure_price_action_trend_view(
                module=trend,
                event_payload=event_payload,
                fallback_topic=topic,
            )

        set_aliases("market_structure", ("structure", "ms"), market_structure)
        set_aliases("support_resistance", ("sr", "levels"), support_resistance)
        set_aliases("fair_value_gap", ("fvg", "fair_value_gaps"), fair_value_gap)
        set_aliases("trend", ("trend_state",), trend)
        set_aliases("liquidity_levels", ("liquidity",), liquidity_levels)

        current_price = value_for("current_price", "price", "last_price", "close", default=None)
        if current_price is not None:
            domain_data.setdefault("current_price", current_price)
            domain_data.setdefault("last_price", current_price)
            domain_data.setdefault("price", current_price)

        timestamp_value = value_for("timestamp", "event_time", "updated_at", "created_at", default=None)
        if timestamp_value is not None:
            domain_data.setdefault("timestamp", timestamp_value)

        domain_data.setdefault("raw", dict(payload))


    @staticmethod
    def _direction_to_long_short(value: Any) -> str | None:
        text = str(value or "").strip().lower()
        if text in {"long", "buy", "bull", "bullish", "up", "uptrend"}:
            return "long"
        if text in {"short", "sell", "bear", "bearish", "down", "downtrend"}:
            return "short"
        return None

    def _ensure_price_action_market_structure_view(
            self,
            *,
            module: dict[str, Any],
            event_payload: dict[str, Any],
            fallback_topic: str,
    ) -> dict[str, Any]:
        """
        Convert a direct analytics.price_action.market_structure event into the
        module/layer shape consumed by MarketStructureStrategy._extract_view().

        The strategy expects select_primary_layer(module) to find an external or
        internal layer and then read last_break_event / last_event from it.
        Backtests often emit a flat event payload, so the adapter must wrap it.
        """
        result = dict(module)

        event = dict(event_payload)
        event.update({key: value for key, value in module.items() if key not in event})

        event_type = (
            event.get("event_type")
            or event.get("type")
            or event.get("kind")
        )
        if event_type is None:
            topic = fallback_topic.lower()
            if ".bos" in topic or topic.endswith("bos"):
                event_type = "bos"
            elif ".choch" in topic or topic.endswith("choch"):
                event_type = "choch"
            elif ".mss" in topic or topic.endswith("mss"):
                event_type = "mss"
        if event_type is not None:
            event.setdefault("event_type", event_type)
            result.setdefault("event_type", event_type)

        side = (
            self._direction_to_long_short(event.get("side"))
            or self._direction_to_long_short(event.get("direction"))
            or self._direction_to_long_short(event.get("market_bias"))
            or self._direction_to_long_short(event.get("bias"))
        )
        if side is not None:
            event.setdefault("side", side)
            event.setdefault("direction", side)
            result.setdefault("direction", side)

        market_bias = event.get("market_bias") or event.get("bias") or side
        if market_bias is not None:
            event.setdefault("market_bias", market_bias)
            event.setdefault("bias", market_bias)
            result.setdefault("market_bias", market_bias)
            result.setdefault("bias", market_bias)

        confidence = event.get("confidence", result.get("confidence", 0.0))
        score = event.get("score", result.get("score", confidence))
        event.setdefault("confidence", confidence)
        event.setdefault("score", score)
        event.setdefault("confirmed", True)

        layer = dict(result)
        layer.update(
            {
                "bias": market_bias,
                "market_bias": market_bias,
                "direction": side or market_bias,
                "confidence": confidence,
                "score": score,
                "strength": result.get("strength", score),
                "trend_strength": result.get("trend_strength", score),
                "alignment_score": result.get(
                    "alignment_score",
                    result.get("mtf_alignment_score", score),
                ),
                "last_break_event": result.get("last_break_event") or event,
                "last_event": result.get("last_event") or event,
            }
        )

        external_layer = dict(layer)
        external_layer.setdefault("layer", "external")
        external_layer.setdefault("structure_layer", "external")

        internal_layer = dict(layer)
        internal_layer.setdefault("layer", "internal")
        internal_layer.setdefault("structure_layer", "internal")

        result["external"] = external_layer
        result["internal"] = internal_layer
        result.setdefault("last_price", event.get("price") or event.get("last_price") or event.get("close"))
        result.setdefault("symbol", event.get("symbol"))
        result.setdefault("timeframe", event.get("timeframe"))
        result.setdefault("updated_at", event.get("timestamp") or event.get("updated_at"))
        result.setdefault("last_update", event.get("timestamp") or event.get("last_update"))
        result["primary_layer"] = external_layer
        result["secondary_layer"] = internal_layer
        result.setdefault("last_break_event", event)
        result.setdefault("last_event", event)
        result.setdefault("event", event)
        result.setdefault("mtf_alignment_score", result.get("mtf_alignment_score", score))
        result.setdefault("alignment_score", result.get("alignment_score", score))
        result.setdefault("swing_progression_score", result.get("swing_progression_score", score))
        result.setdefault("trend_strength", result.get("trend_strength", score))
        return result

    def _ensure_price_action_trend_view(
            self,
            *,
            module: dict[str, Any],
            event_payload: dict[str, Any],
            fallback_topic: str,
    ) -> dict[str, Any]:
        """
        Convert a direct analytics.price_action.trend event into the
        external/internal layer shape consumed by TrendContinuationStrategy.
        """
        result = dict(module)

        event = dict(event_payload)
        event.update({key: value for key, value in module.items() if key not in event})

        def normalize_trend_direction(value: Any) -> str | None:
            text = str(value or "").strip().lower()
            if text in {"long", "buy", "bull", "bullish", "up", "uptrend", "trending_up"}:
                return "bullish"
            if text in {"short", "sell", "bear", "bearish", "down", "downtrend", "trending_down"}:
                return "bearish"
            if text in {"neutral", "flat", "range", "ranging"}:
                return "neutral"
            return text or None

        def normalize_trend_regime(value: Any, *, fallback_direction: str | None = None) -> str | None:
            text = str(value or "").strip().lower()
            if text in {"uptrend", "downtrend", "trend", "trending", "trending_up", "trending_down", "bullish", "bearish"}:
                return "trending"
            if text in {"pullback", "retracement"}:
                return "pullback"
            if text in {"consolidation", "consolidating", "range", "ranging"}:
                return "consolidating"
            if text in {"reversal", "reversing"}:
                return "reversing"
            if text in {"exhausted", "exhaustion"}:
                return "exhausted"
            if not text and fallback_direction in {"bullish", "bearish"}:
                return "trending"
            return text or None

        direction = (
            normalize_trend_direction(event.get("trend_direction"))
            or normalize_trend_direction(event.get("direction"))
            or normalize_trend_direction(event.get("side"))
        )
        if direction is not None:
            # Use analytics.price_action TrendDirection enum values, not strategy sides.
            event["direction"] = direction
            event["trend_direction"] = direction
            result["direction"] = direction
            result["trend_direction"] = direction

        regime = normalize_trend_regime(
            event.get("trend_regime") or event.get("regime") or event.get("state"),
            fallback_direction=direction,
        )
        if regime is not None:
            # Use analytics.price_action TrendRegime enum values.
            event["trend_regime"] = regime
            event["regime"] = regime
            result["trend_regime"] = regime
            result["regime"] = regime

        confidence = event.get("confidence", result.get("confidence", 0.0))
        score = event.get("score", result.get("score", confidence))
        continuation_probability = event.get(
            "continuation_probability",
            result.get("continuation_probability", result.get("continuation_prob", score)),
        )
        event.setdefault("confidence", confidence)
        event.setdefault("score", score)
        event.setdefault("continuation_probability", continuation_probability)
        event.setdefault("confirmed", True)

        topic = fallback_topic.lower()
        event_type = event.get("event_type") or event.get("type") or event.get("kind")
        if event_type is None:
            if "alignment" in topic:
                event_type = "trend_alignment"
            elif "started" in topic:
                event_type = "trend_started"
            elif "reversal" in topic:
                event_type = "trend_reversal"
        if event_type is not None:
            event.setdefault("event_type", event_type)

        layer = dict(result)
        layer.update(
            {
                "direction": direction,
                "trend_direction": direction,
                "regime": regime,
                "trend_regime": regime,
                "confidence": confidence,
                "score": score,
                "trend_strength": result.get("trend_strength", result.get("strength", score)),
                "momentum_score": result.get(
                    "momentum_score",
                    result.get("momentum", result.get("directional_momentum", score)),
                ),
                "slope_score": result.get(
                    "slope_score",
                    result.get("slope", result.get("trend_slope", score)),
                ),
                "directional_momentum": result.get(
                    "directional_momentum",
                    result.get("momentum_score", result.get("momentum", score)),
                ),
                "trend_slope": result.get(
                    "trend_slope",
                    result.get("slope_score", result.get("slope", score)),
                ),
                "strength": result.get("strength", result.get("trend_strength", score)),
                "continuation_probability": continuation_probability,
                "reversal_risk": result.get("reversal_risk", 0.0),
                "exhaustion_score": result.get("exhaustion_score", result.get("exhaustion", 0.0)),
                "pullback_quality": result.get("pullback_quality", result.get("pullback_score", 0.0)),
                "structure_score": result.get("structure_score", result.get("market_structure_score", score)),
                "last_signal": result.get("last_signal") or event,
                "last_event": result.get("last_event") or event,
            }
        )

        external_layer = dict(layer)
        external_layer.setdefault("layer", "external")
        external_layer.setdefault("structure_layer", "external")

        internal_layer = dict(layer)
        internal_layer.setdefault("layer", "internal")
        internal_layer.setdefault("structure_layer", "internal")

        result["external"] = external_layer
        result["internal"] = internal_layer
        result.setdefault("last_price", event.get("price") or event.get("last_price") or event.get("close"))
        result.setdefault("symbol", event.get("symbol"))
        result.setdefault("timeframe", event.get("timeframe"))
        result.setdefault("updated_at", event.get("timestamp") or event.get("updated_at"))
        result.setdefault("last_update", event.get("timestamp") or event.get("last_update"))
        result["primary_layer"] = external_layer
        result["secondary_layer"] = internal_layer
        result.setdefault("last_signal", event)
        result.setdefault("last_event", event)
        result.setdefault("event", event)
        result.setdefault("internal_external_alignment", result.get("internal_external_alignment", score))
        result.setdefault("higher_timeframe_alignment", result.get("higher_timeframe_alignment", score))
        result.setdefault("overall_trend_score", result.get("overall_trend_score", score))
        return result


    def _augment_spoofing_domain_data(
            self,
            *,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def mapping_for(*keys: str) -> dict[str, Any] | None:
            for key in keys:
                value = payload.get(key)
                if isinstance(value, dict):
                    return value

                value = feature_map.get(key)
                if isinstance(value, dict):
                    return value

            return None

        composite = mapping_for(
            "composite",
            "spoofing",
            "snapshot",
            "result",
        )
        signal = mapping_for(
            "signal",
            "spoofing_signal",
            "analytics_signal",
            "event",
        )
        features = mapping_for(
            "features",
            "spoofing_features",
        )
        detector_results = mapping_for(
            "detector_results",
            "detectors",
            "detector_result",
        )
        score_breakdown = mapping_for(
            "score_breakdown",
            "scores",
            "score_components",
        )
        analytics_metadata = mapping_for(
            "analytics_metadata",
            "metadata",
        )

        if composite is not None:
            if signal is None:
                for key in (
                        "signal",
                        "spoofing_signal",
                        "analytics_signal",
                        "event",
                ):
                    value = composite.get(key)
                    if isinstance(value, dict):
                        signal = value
                        break

            if features is None:
                for key in (
                        "features",
                        "spoofing_features",
                ):
                    value = composite.get(key)
                    if isinstance(value, dict):
                        features = value
                        break

            if detector_results is None:
                for key in (
                        "detector_results",
                        "detectors",
                        "detector_result",
                ):
                    value = composite.get(key)
                    if isinstance(value, dict):
                        detector_results = value
                        break

            if score_breakdown is None:
                for key in (
                        "score_breakdown",
                        "scores",
                        "score_components",
                ):
                    value = composite.get(key)
                    if isinstance(value, dict):
                        score_breakdown = value
                        break

            if analytics_metadata is None:
                for key in (
                        "analytics_metadata",
                        "metadata",
                ):
                    value = composite.get(key)
                    if isinstance(value, dict):
                        analytics_metadata = value
                        break

        if composite is not None:
            domain_data.setdefault("composite", composite)
            domain_data.setdefault("spoofing", composite)
            domain_data.setdefault("snapshot", composite)
            domain_data.setdefault("result", composite)

        if signal is not None:
            domain_data.setdefault("signal", signal)
            domain_data.setdefault("spoofing_signal", signal)
            domain_data.setdefault("analytics_signal", signal)

        if features is not None:
            domain_data.setdefault("features", features)
            domain_data.setdefault("spoofing_features", features)

        if detector_results is not None:
            domain_data.setdefault("detector_results", detector_results)
            domain_data.setdefault("detectors", detector_results)
            domain_data.setdefault("detector_result", detector_results)

        if score_breakdown is not None:
            domain_data.setdefault("score_breakdown", score_breakdown)
            domain_data.setdefault("scores", score_breakdown)
            domain_data.setdefault("score_components", score_breakdown)

        if analytics_metadata is not None:
            domain_data.setdefault("analytics_metadata", analytics_metadata)
            domain_data.setdefault("metadata", analytics_metadata)

        if "signal" not in domain_data and (
                "spoofing_type" in payload
                or "type" in payload
                or "pattern" in payload
                or "side" in payload
                or "score" in payload
                or "confidence" in payload
                or "pull_ratio" in payload
                or "fill_ratio" in payload
        ):
            domain_data["signal"] = dict(payload)
            domain_data.setdefault("spoofing_signal", domain_data["signal"])

        if "features" not in domain_data and (
                "pull_ratio" in payload
                or "fill_ratio" in payload
                or "price_reaction_bps" in payload
                or "lifetime_ms" in payload
                or "wall_notional" in payload
                or "pulled_notional" in payload
                or "cancel_to_fill_ratio" in payload
                or "distance_from_mid_bps" in payload
                or "layer_count" in payload
                or "layer_price_span_bps" in payload
                or "pressure_flip_strength" in payload
        ):
            domain_data["features"] = {
                "pull_ratio": payload.get("pull_ratio"),
                "fill_ratio": payload.get("fill_ratio"),
                "price_reaction_bps": payload.get("price_reaction_bps"),
                "signed_price_reaction_bps": payload.get(
                    "signed_price_reaction_bps",
                    payload.get("price_reaction_bps"),
                ),
                "lifetime_ms": payload.get("lifetime_ms"),
                "wall_notional": payload.get("wall_notional"),
                "pulled_notional": payload.get("pulled_notional"),
                "cancel_to_fill_ratio": payload.get("cancel_to_fill_ratio"),
                "distance_from_mid_bps": payload.get("distance_from_mid_bps"),
                "layer_count": payload.get("layer_count"),
                "layer_price_span_bps": payload.get("layer_price_span_bps"),
                "pressure_flip_strength": payload.get("pressure_flip_strength"),
            }
            domain_data.setdefault("spoofing_features", domain_data["features"])

        domain_data.setdefault("raw", dict(payload))

    def _augment_spreads_domain_data(
            self,
            *,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def mapping_for(*keys: str) -> dict[str, Any] | None:
            for key in keys:
                value = payload.get(key)
                if isinstance(value, dict):
                    return value

                value = feature_map.get(key)
                if isinstance(value, dict):
                    return value

            return None

        snapshot = mapping_for(
            "snapshot",
            "spread_snapshot",
            "spot_futures",
            "spot_futures_snapshot",
            "cross_exchange",
            "cross_exchange_snapshot",
            "result",
        )
        signal = mapping_for(
            "signal",
            "spread_signal",
            "analytics_signal",
            "event",
        )
        opportunity = mapping_for(
            "opportunity",
            "arbitrage_opportunity",
            "arb_opportunity",
        )
        metadata = mapping_for(
            "metadata",
            "spread_metadata",
        )

        if snapshot is not None:
            domain_data.setdefault("snapshot", snapshot)
            domain_data.setdefault("spread_snapshot", snapshot)
            domain_data.setdefault("result", snapshot)

        if signal is not None:
            domain_data.setdefault("signal", signal)
            domain_data.setdefault("spread_signal", signal)
            domain_data.setdefault("analytics_signal", signal)

        if opportunity is not None:
            domain_data.setdefault("opportunity", opportunity)
            domain_data.setdefault("arbitrage_opportunity", opportunity)
            domain_data.setdefault("arb_opportunity", opportunity)

        if metadata is not None:
            domain_data.setdefault("metadata", metadata)
            domain_data.setdefault("spread_metadata", metadata)

        if "snapshot" not in domain_data and (
                "spread_type" in payload
                or "spread_bps" in payload
                or "basis" in payload
                or "funding_adjusted_spread" in payload
                or "zscore" in payload
                or "regime" in payload
                or "quote_validity" in payload
        ):
            domain_data["snapshot"] = dict(payload)
            domain_data.setdefault("spread_snapshot", domain_data["snapshot"])

        if "signal" not in domain_data and (
                "signal_type" in payload
                or "spread_signal_type" in payload
                or "direction" in payload
                or "spread_direction" in payload
                or "confidence" in payload
        ):
            domain_data["signal"] = dict(payload)
            domain_data.setdefault("spread_signal", domain_data["signal"])

        if "opportunity" not in domain_data and (
                "opportunity_key" in payload
                or "net_edge" in payload
                or "net_edge_bps" in payload
                or "buy_exchange" in payload
                or "sell_exchange" in payload
                or "opportunity_status" in payload
        ):
            domain_data["opportunity"] = dict(payload)
            domain_data.setdefault("arbitrage_opportunity", domain_data["opportunity"])

        domain_data.setdefault("raw", dict(payload))

    def _augment_whales_domain_data(
            self,
            *,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def mapping_for(*keys: str) -> dict[str, Any] | None:
            for key in keys:
                value = payload.get(key)
                if isinstance(value, dict):
                    return value

                value = feature_map.get(key)
                if isinstance(value, dict):
                    return value

            return None

        composite = mapping_for(
            "composite",
            "whales",
            "snapshot",
            "result",
        )
        activity = mapping_for(
            "activity",
            "whale_activity",
            "activity_context",
        )
        pressure = mapping_for(
            "pressure",
            "whale_pressure",
            "pressure_context",
        )
        large_trade = mapping_for(
            "large_trade",
            "large_trade_event",
            "large_trade_context",
        )
        cluster = mapping_for(
            "cluster",
            "whale_cluster",
            "cluster_context",
        )
        liquidation_context = mapping_for(
            "liquidation_context",
            "whale_liquidation_context",
            "liquidations",
            "liquidation",
        )
        exhaustion = mapping_for(
            "exhaustion",
            "cluster_exhaustion",
            "exhaustion_context",
        )
        signal = mapping_for(
            "signal",
            "whale_signal",
            "analytics_signal",
            "event",
        )

        if composite is not None:
            if activity is None:
                for key in (
                        "activity",
                        "whale_activity",
                        "activity_context",
                ):
                    value = composite.get(key)
                    if isinstance(value, dict):
                        activity = value
                        break

            if pressure is None:
                for key in (
                        "pressure",
                        "whale_pressure",
                        "pressure_context",
                ):
                    value = composite.get(key)
                    if isinstance(value, dict):
                        pressure = value
                        break

            if large_trade is None:
                for key in (
                        "large_trade",
                        "large_trade_event",
                        "large_trade_context",
                ):
                    value = composite.get(key)
                    if isinstance(value, dict):
                        large_trade = value
                        break

            if cluster is None:
                for key in (
                        "cluster",
                        "whale_cluster",
                        "cluster_context",
                ):
                    value = composite.get(key)
                    if isinstance(value, dict):
                        cluster = value
                        break

            if liquidation_context is None:
                for key in (
                        "liquidation_context",
                        "whale_liquidation_context",
                        "liquidations",
                        "liquidation",
                ):
                    value = composite.get(key)
                    if isinstance(value, dict):
                        liquidation_context = value
                        break

            if exhaustion is None:
                for key in (
                        "exhaustion",
                        "cluster_exhaustion",
                        "exhaustion_context",
                ):
                    value = composite.get(key)
                    if isinstance(value, dict):
                        exhaustion = value
                        break

            if signal is None:
                for key in (
                        "signal",
                        "whale_signal",
                        "analytics_signal",
                        "event",
                ):
                    value = composite.get(key)
                    if isinstance(value, dict):
                        signal = value
                        break

        if composite is not None:
            domain_data.setdefault("composite", composite)
            domain_data.setdefault("whales", composite)
            domain_data.setdefault("snapshot", composite)
            domain_data.setdefault("result", composite)

        if activity is not None:
            domain_data.setdefault("activity", activity)
            domain_data.setdefault("whale_activity", activity)
            domain_data.setdefault("activity_context", activity)

        if pressure is not None:
            domain_data.setdefault("pressure", pressure)
            domain_data.setdefault("whale_pressure", pressure)
            domain_data.setdefault("pressure_context", pressure)

        if large_trade is not None:
            domain_data.setdefault("large_trade", large_trade)
            domain_data.setdefault("large_trade_event", large_trade)
            domain_data.setdefault("large_trade_context", large_trade)

        if cluster is not None:
            domain_data.setdefault("cluster", cluster)
            domain_data.setdefault("whale_cluster", cluster)
            domain_data.setdefault("cluster_context", cluster)

        if liquidation_context is not None:
            domain_data.setdefault("liquidation_context", liquidation_context)
            domain_data.setdefault("whale_liquidation_context", liquidation_context)
            domain_data.setdefault("liquidations", liquidation_context)
            domain_data.setdefault("liquidation", liquidation_context)

        if exhaustion is not None:
            domain_data.setdefault("exhaustion", exhaustion)
            domain_data.setdefault("cluster_exhaustion", exhaustion)
            domain_data.setdefault("exhaustion_context", exhaustion)

        if signal is not None:
            domain_data.setdefault("signal", signal)
            domain_data.setdefault("whale_signal", signal)
            domain_data.setdefault("analytics_signal", signal)

        if "composite" not in domain_data and (
                "whale_side" in payload
                or "activity_notional" in payload
                or "pressure_score" in payload
                or "cluster_score" in payload
                or "liquidation_context_strength" in payload
                or "exhaustion_probability" in payload
        ):
            domain_data["composite"] = dict(payload)
            domain_data.setdefault("whales", domain_data["composite"])
            domain_data.setdefault("snapshot", domain_data["composite"])

        if "activity" not in domain_data and (
                "activity_notional" in payload
                or "activity_trade_count" in payload
                or "activity_side" in payload
                or "whale_activity" in payload
        ):
            domain_data["activity"] = dict(payload)
            domain_data.setdefault("whale_activity", domain_data["activity"])
            domain_data.setdefault("activity_context", domain_data["activity"])

        if "pressure" not in domain_data and (
                "pressure_score" in payload
                or "pressure_side" in payload
                or "pressure_imbalance_ratio" in payload
                or "whale_pressure" in payload
        ):
            domain_data["pressure"] = dict(payload)
            domain_data.setdefault("whale_pressure", domain_data["pressure"])
            domain_data.setdefault("pressure_context", domain_data["pressure"])

        if "large_trade" not in domain_data and (
                "large_trade_notional" in payload
                or "large_trade_zscore" in payload
                or "large_trade_side" in payload
        ):
            domain_data["large_trade"] = dict(payload)
            domain_data.setdefault("large_trade_event", domain_data["large_trade"])
            domain_data.setdefault("large_trade_context", domain_data["large_trade"])

        if "cluster" not in domain_data and (
                "cluster_score" in payload
                or "cluster_side" in payload
                or "continuation_probability" in payload
                or "exhaustion_probability" in payload
        ):
            domain_data["cluster"] = dict(payload)
            domain_data.setdefault("whale_cluster", domain_data["cluster"])
            domain_data.setdefault("cluster_context", domain_data["cluster"])

        if "liquidation_context" not in domain_data and (
                "liquidation_side" in payload
                or "liquidation_notional" in payload
                or "total_liquidation_notional" in payload
                or "liquidation_context_strength" in payload
        ):
            domain_data["liquidation_context"] = dict(payload)
            domain_data.setdefault(
                "whale_liquidation_context",
                domain_data["liquidation_context"],
            )
            domain_data.setdefault("liquidations", domain_data["liquidation_context"])
            domain_data.setdefault("liquidation", domain_data["liquidation_context"])

        if "exhaustion" not in domain_data and (
                "exhaustion_probability" in payload
                or "exhaustion_side" in payload
        ):
            domain_data["exhaustion"] = dict(payload)
            domain_data.setdefault("cluster_exhaustion", domain_data["exhaustion"])
            domain_data.setdefault("exhaustion_context", domain_data["exhaustion"])

        if "signal" not in domain_data and (
                "side" in payload
                or "whale_side" in payload
                or "score" in payload
                or "confidence" in payload
        ):
            domain_data["signal"] = dict(payload)
            domain_data.setdefault("whale_signal", domain_data["signal"])
            domain_data.setdefault("analytics_signal", domain_data["signal"])

        domain_data.setdefault("raw", dict(payload))

    def _augment_domain_data_contracts(
            self,
            *,
            source: FeatureSource,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> dict[str, Any]:
        if source is FeatureSource.OPEN_INTEREST:
            self._augment_open_interest_domain_data(
                payload=payload,
                domain_data=domain_data,
            )

        elif source is FeatureSource.ORDERFLOW:
            self._augment_orderflow_domain_data(
                payload=payload,
                domain_data=domain_data,
            )

        elif source is FeatureSource.FUNDING:
            self._augment_funding_domain_data(
                payload=payload,
                domain_data=domain_data,
            )
        elif source is FeatureSource.LIQUIDATIONS:
            self._augment_liquidations_domain_data(
                payload=payload,
                domain_data=domain_data,
            )
        elif source is FeatureSource.LIQUIDITY:
            self._augment_liquidity_domain_data(
                payload=payload,
                domain_data=domain_data,
            )
        elif source is FeatureSource.PRICE_ACTION:
            self._augment_price_action_domain_data(
                payload=payload,
                domain_data=domain_data,
            )
        elif source is FeatureSource.SPOOFING:
            self._augment_spoofing_domain_data(
                payload=payload,
                domain_data=domain_data,
            )
        elif source is FeatureSource.SPREADS:
            self._augment_spreads_domain_data(
                payload=payload,
                domain_data=domain_data,
            )
        elif source is FeatureSource.WHALES:
            self._augment_whales_domain_data(
                payload=payload,
                domain_data=domain_data,
            )

        self._ensure_common_domain_contract(
            source=source,
            payload=payload,
            domain_data=domain_data,
        )
        return domain_data

    def _ensure_common_domain_contract(
            self,
            *,
            source: FeatureSource,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        """
        Add stable fields every concrete strategy can rely on.

        Domain-specific adapters still own trading semantics. This method only
        guarantees basic contract metadata and the original analytics payload
        for diagnostics / fallback-free debugging.
        """
        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]
            return default

        contract = domain_data.setdefault("contract", {})
        if isinstance(contract, dict):
            contract.setdefault("version", "strategy-domain-v1")
            contract.setdefault("source", source.value)
            contract.setdefault("event_name", value_for("event_name", "topic", "source_topic"))

        scope = domain_data.setdefault("scope", {})
        if isinstance(scope, dict):
            scope.setdefault("symbol", value_for("symbol", "instrument", "market"))
            scope.setdefault("exchange", value_for("exchange", default="unknown"))
            scope.setdefault("market_type", value_for("market_type", default="usdm_futures"))
            scope.setdefault("timeframe", value_for("timeframe"))
            scope.setdefault("exchange_symbol", value_for("exchange_symbol", "symbol", "instrument"))

        domain_data.setdefault("raw", dict(payload))

    def _build_cross_domain_contracts(
            self,
            *,
            source: FeatureSource,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> dict[FeatureSource, dict[str, Any]]:
        """Build non-primary StrategyContext domain contracts.

        FeatureSource currently has no HYBRID member, so the canonical hybrid
        view is stored under FeatureSource.SYSTEM using the stable key
        ``hybrid``. Hybrid concrete strategies should read
        ``context.domain_dict(FeatureSource.SYSTEM)["hybrid"]`` and native
        source domains, never raw analytics payloads.
        """
        if source not in {
            FeatureSource.OPEN_INTEREST,
            FeatureSource.FUNDING,
            FeatureSource.ORDERFLOW,
            FeatureSource.LIQUIDITY,
            FeatureSource.LIQUIDATIONS,
            FeatureSource.WHALES,
            FeatureSource.PRICE_ACTION,
            FeatureSource.SPOOFING,
            FeatureSource.SPREADS,
        }:
            return {}

        hybrid = self._build_hybrid_domain_data(
            source=source,
            payload=payload,
            domain_data=domain_data,
        )
        return {
            FeatureSource.SYSTEM: {
                "hybrid": hybrid,
                "hybrid_contract": hybrid,
                "hybrid.summary": hybrid.get("summary", {}),
                "hybrid.votes": hybrid.get("votes", {}),
                "hybrid.domains": hybrid.get("domains", {}),
            }
        }

    def _build_hybrid_domain_data(
            self,
            *,
            source: FeatureSource,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> dict[str, Any]:
        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]
                if key in domain_data:
                    return domain_data[key]
            return default

        def flag_for(domain: str, expected: FeatureSource) -> bool:
            explicit = value_for(f"hybrid.{domain}", domain, default=None)
            if explicit is not None:
                return _to_bool(explicit, default=False)
            return source is expected

        dominant_side = value_for(
            "hybrid.dominant_side", "dominant_side", "side", "bias", "direction",
            default="unknown",
        )
        alignment_score = _to_float(
            value_for("hybrid.alignment_score", "alignment_score", "alignment", default=0.0),
            0.0,
        )
        conflict_score = _to_float(
            value_for("hybrid.conflict_score", "conflict_score", "conflict", default=0.0),
            0.0,
        )
        confluence_score = _to_float(
            value_for("hybrid.confluence_score", "confluence_score", "score", default=0.0),
            0.0,
        )
        confidence = _to_float(
            value_for("hybrid.confidence", "confidence", default=0.0),
            0.0,
        )

        votes = value_for("hybrid.votes", "votes", default=None)
        if not isinstance(votes, dict):
            votes = {
                "long": _to_int(value_for("long_votes", "bullish_votes", default=0), 0),
                "short": _to_int(value_for("short_votes", "bearish_votes", default=0), 0),
                "flat": _to_int(value_for("flat_votes", "neutral_votes", default=0), 0),
                "total": _to_int(value_for("total_votes", default=0), 0),
            }

        domains = {
            "orderflow": flag_for("orderflow", FeatureSource.ORDERFLOW),
            "liquidity": flag_for("liquidity", FeatureSource.LIQUIDITY),
            "liquidations": flag_for("liquidations", FeatureSource.LIQUIDATIONS),
            "whales": flag_for("whales", FeatureSource.WHALES),
            "open_interest": flag_for("open_interest", FeatureSource.OPEN_INTEREST),
            "funding": flag_for("funding", FeatureSource.FUNDING),
            "price_action": flag_for("price_action", FeatureSource.PRICE_ACTION),
            "spoofing": flag_for("spoofing", FeatureSource.SPOOFING),
            "spreads": flag_for("spreads", FeatureSource.SPREADS),
        }

        summary = {
            "dominant_side": dominant_side,
            "alignment_score": alignment_score,
            "conflict_score": conflict_score,
            "confluence_score": confluence_score,
            "confidence": confidence,
            "trigger_source": source.value,
        }

        return {
            "contract": {
                "version": "strategy-hybrid-v1",
                "storage_source": FeatureSource.SYSTEM.value,
                "trigger_source": source.value,
            },
            "summary": summary,
            "dominant_side": dominant_side,
            "alignment_score": alignment_score,
            "conflict_score": conflict_score,
            "confluence_score": confluence_score,
            "confidence": confidence,
            "votes": votes,
            "domains": domains,
            "source_domain": {source.value: dict(domain_data)},
            "raw": dict(payload),
        }

    def _build_hybrid_contract_features(
            self,
            *,
            source: FeatureSource,
            symbol: str,
            payload: dict[str, Any],
            timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        result: list[FeatureSnapshot] = []
        confidence = payload.get("confidence", 0.0)

        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]
            return default

        def add(name: str, value: Any) -> None:
            result.append(
                self._snapshot_from_raw_value(
                    source=source,  # важливо: НЕ FeatureSource.HYBRID
                    symbol=symbol,
                    name=name,
                    value=value,
                    timestamp=timestamp,
                    confidence=confidence,
                    metadata={
                        "origin": "contract_feature",
                        "contract": "hybrid",
                        "trigger_source": getattr(source, "value", str(source)),
                    },
                )
            )

        dominant_side = value_for(
            "hybrid.dominant_side",
            "dominant_side",
            "side",
            "bias",
            "direction",
            default="unknown",
        )
        alignment_score = value_for(
            "hybrid.alignment_score",
            "alignment_score",
            "alignment",
            default=0.0,
        )
        conflict_score = value_for(
            "hybrid.conflict_score",
            "conflict_score",
            "conflict",
            default=0.0,
        )
        confluence_score = value_for(
            "hybrid.confluence_score",
            "confluence_score",
            "confluence",
            "score",
            default=0.0,
        )
        hybrid_confidence = value_for(
            "hybrid.confidence",
            "hybrid_confidence",
            "confidence",
            default=confidence,
        )
        votes = value_for(
            "hybrid.votes",
            "votes",
            default=[],
        )

        add("hybrid.dominant_side", dominant_side)
        add("hybrid.alignment_score", alignment_score)
        add("hybrid.conflict_score", conflict_score)
        add("hybrid.confluence_score", confluence_score)
        add("hybrid.confidence", hybrid_confidence)
        add("hybrid.votes", votes)

        for domain_name in (
                "orderflow",
                "liquidity",
                "liquidations",
                "whales",
                "open_interest",
                "funding",
                "price_action",
                "spoofing",
                "spreads",
        ):
            value = value_for(
                f"hybrid.{domain_name}",
                domain_name,
                f"hybrid_{domain_name}",
                default=None,
            )
            if value is not None:
                add(f"hybrid.{domain_name}", value)

        add("hybrid.symbol", value_for("hybrid.symbol", "symbol", default=symbol))
        add("hybrid.exchange", value_for("hybrid.exchange", "exchange", default="unknown"))
        add("hybrid.market_type", value_for("hybrid.market_type", "market_type", default="usdm_futures"))
        add("hybrid.timeframe", value_for("hybrid.timeframe", "timeframe", default=None))
        add("hybrid.exchange_symbol", value_for("hybrid.exchange_symbol", "exchange_symbol", default=symbol))
        add("hybrid.timestamp", value_for("hybrid.timestamp", "timestamp", default=timestamp))

        return result

    def normalize_event(
        self,
        *,
        event_name: str,
        payload: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> NormalizedPayload:
        if not event_name.strip():
            raise SignalNormalizationError("event_name cannot be empty")
        if not isinstance(payload, dict):
            raise SignalNormalizationError("payload must be a dict")

        source = self._resolve_source(event_name, payload)
        symbol = self._extract_symbol(payload)
        ts = self._extract_timestamp(payload, timestamp)
        timeframe = self._extract_timeframe(payload)

        # Contract adapters need the EventBus topic to build the right
        # canonical domain view. Many analytics payloads do not carry
        # event_name/topic/source_topic inside payload, so inject it here
        # without mutating the caller-owned payload.
        payload_for_contract = dict(payload)
        payload_for_contract.setdefault("event_name", event_name)
        payload_for_contract.setdefault("topic", event_name)
        payload_for_contract.setdefault("source_topic", event_name)

        domain_data = self._extract_domain_data(payload_for_contract)
        domain_data = self._augment_domain_data_contracts(
            source=source,
            payload=payload_for_contract,
            domain_data=domain_data,
        )
        extra_domain_data = self._build_cross_domain_contracts(
            source=source,
            payload=payload_for_contract,
            domain_data=domain_data,
        )
        features = self._extract_features(
            source=source,
            symbol=symbol,
            payload=payload_for_contract,
            timestamp=ts,
        )

        normalized = NormalizedPayload(
            source=source,
            symbol=symbol,
            timestamp=ts,
            timeframe=timeframe,
            domain_data=domain_data,
            extra_domain_data=extra_domain_data,
            features=features,
            metadata={
                "event_name": event_name,
                "raw_payload_keys": sorted(payload.keys()),
                "features_count": len(features),
                "extra_domain_sources": sorted(source.value for source in extra_domain_data),
                "timeframe": timeframe.value,
            },
        )

        self.log_debug(
            "Analytics event normalized",
            event_name=event_name,
            source=source.value,
            symbol=symbol,
            timeframe=timeframe.value,
            features_count=len(features),
            feature_names=[feature.name for feature in features],
        )
        return normalized

    def apply_to_context(
        self,
        context: StrategyContext,
        normalized: NormalizedPayload,
    ) -> StrategyContext:
        context.validate()

        if context.symbol != normalized.symbol:
            raise SignalNormalizationError(
                f"context symbol '{context.symbol}' != normalized symbol '{normalized.symbol}'"
            )

        context.timestamp = normalized.timestamp
        context.timeframe = normalized.timeframe

        for key, value in normalized.domain_data.items():
            context.put_domain_feature(normalized.source, key, value)

        for extra_source, values in normalized.extra_domain_data.items():
            for key, value in values.items():
                context.put_domain_feature(extra_source, key, value)

        for snapshot in normalized.features:
            context.put_feature(snapshot)
            if snapshot.freshness_seconds is not None:
                context.freshness_map[snapshot.name] = snapshot.freshness_seconds

        price_value = (
            _to_float(normalized.domain_data.get("current_price"))
            or _to_float(normalized.domain_data.get("last_price"))
            or _to_float(normalized.domain_data.get("price"))
            or _to_float(normalized.domain_data.get("close"))
            or _to_float(normalized.metadata.get("price"))
        )
        if price_value is not None and price_value > 0:
            bid = _to_float(normalized.domain_data.get("bid"))
            ask = _to_float(normalized.domain_data.get("ask"))
            mark_price = _to_float(normalized.domain_data.get("mark_price"))
            index_price = _to_float(normalized.domain_data.get("index_price"))
            spread_bps = _to_float(normalized.domain_data.get("spread_bps"))
            try:
                price_snapshot = PriceSnapshot(
                    symbol=normalized.symbol,
                    last_price=price_value,
                    bid=bid,
                    ask=ask,
                    mark_price=mark_price,
                    index_price=index_price,
                    spread_bps=spread_bps,
                    timestamp=normalized.timestamp,
                    metadata={
                        "source": self.component_name,
                        "event_name": normalized.metadata.get("event_name"),
                    },
                )
                price_snapshot.validate()
                context.price = price_snapshot
            except Exception as exc:
                self.log_debug(
                    "PriceSnapshot skipped during normalization",
                    symbol=normalized.symbol,
                    price=price_value,
                    error=str(exc),
                )

        context.metadata.setdefault("updated_by", self.component_name)
        context.metadata["last_source"] = normalized.source.value
        context.metadata["last_event_name"] = normalized.metadata.get("event_name")
        context.metadata["last_timeframe"] = normalized.timeframe.value
        context.metadata["last_feature_count"] = len(normalized.features)

        context.validate()
        return context

    def normalize_and_apply(
        self,
        *,
        context: StrategyContext,
        event_name: str,
        payload: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> StrategyContext:
        normalized = self.normalize_event(
            event_name=event_name,
            payload=payload,
            timestamp=timestamp,
        )
        return self.apply_to_context(context, normalized)

    def _resolve_source(self, event_name: str, payload: dict[str, Any]) -> FeatureSource:
        explicit = payload.get("source")

        if isinstance(explicit, FeatureSource):
            return explicit

        if isinstance(explicit, str):
            try:
                return FeatureSource(explicit)
            except ValueError:
                pass

        resolved = self._resolve_source_from_text(event_name)
        if resolved is not None:
            return resolved

        resolved = self._resolve_source_from_payload(payload)
        if resolved is not None:
            return resolved

        raise SignalNormalizationError(
            f"unable to resolve FeatureSource for event '{event_name}'"
        )

    @staticmethod
    def _resolve_source_from_text(value: str) -> FeatureSource | None:
        text = value.lower()

        if "orderflow" in text or "cvd" in text or "imbalance" in text or "volume_delta" in text:
            return FeatureSource.ORDERFLOW
        if "liquidity" in text or "stop_cluster" in text or "equal_high" in text or "equal_low" in text:
            return FeatureSource.LIQUIDITY
        if "price_action" in text or "market_structure" in text or "fvg" in text or "trend" in text:
            return FeatureSource.PRICE_ACTION
        if "liquidation" in text or "squeeze" in text:
            return FeatureSource.LIQUIDATIONS
        if "whale" in text or "large_trade" in text:
            return FeatureSource.WHALES
        if "spoof" in text or "fake_liquidity" in text or "layering" in text:
            return FeatureSource.SPOOFING
        if "spread" in text or "basis" in text or "arb" in text:
            return FeatureSource.SPREADS
        if "funding" in text:
            return FeatureSource.FUNDING
        if "open_interest" in text or "oi_" in text or ".oi" in text:
            return FeatureSource.OPEN_INTEREST

        return None

    @classmethod
    def _resolve_source_from_payload(cls, payload: dict[str, Any]) -> FeatureSource | None:
        candidates = [
            payload.get("metric"),
            payload.get("category"),
            payload.get("domain"),
            payload.get("source_type"),
            payload.get("event_type"),
        ]

        for value in candidates:
            if isinstance(value, str):
                resolved = cls._resolve_source_from_text(value)
                if resolved is not None:
                    return resolved

        return None

    @staticmethod
    def _extract_symbol(payload: dict[str, Any]) -> str:
        raw = (
            payload.get("symbol")
            or payload.get("instrument")
            or payload.get("market")
            or payload.get("exchange_symbol")
        )

        if not isinstance(raw, str) or not raw.strip():
            scope = payload.get("scope")
            if isinstance(scope, dict):
                raw = scope.get("symbol")

        if not isinstance(raw, str) or not raw.strip():
            raise SignalNormalizationError("payload does not contain valid symbol")

        return raw.strip().upper()

    @staticmethod
    def _extract_timestamp(
        payload: dict[str, Any],
        fallback: datetime | None = None,
    ) -> datetime:
        raw = (
            payload.get("timestamp")
            or payload.get("ts")
            or payload.get("created_at")
            or payload.get("event_time")
            or fallback
        )

        if raw is None:
            return utcnow()

        if isinstance(raw, datetime):
            return ensure_aware_utc(raw)

        if isinstance(raw, (int, float)):
            if raw > 10_000_000_000:
                return datetime.fromtimestamp(raw / 1000.0, tz=timezone.utc)
            return datetime.fromtimestamp(raw, tz=timezone.utc)

        if isinstance(raw, str):
            try:
                return ensure_aware_utc(datetime.fromisoformat(raw.replace("Z", "+00:00")))
            except ValueError:
                raise SignalNormalizationError("unsupported timestamp string in payload")

        raise SignalNormalizationError("unsupported timestamp type in payload")

    @classmethod
    def _extract_timeframe(cls, payload: dict[str, Any]) -> Timeframe:
        raw = payload.get("timeframe")

        if raw is None:
            scope = payload.get("scope")
            if isinstance(scope, dict):
                raw = scope.get("timeframe")

        if isinstance(raw, Timeframe):
            return raw

        if isinstance(raw, str):
            normalized = raw.strip().lower()
            if normalized in cls._TIMEFRAME_ALIASES:
                return cls._TIMEFRAME_ALIASES[normalized]

            try:
                return Timeframe(normalized)
            except ValueError:
                return Timeframe.M1

        return Timeframe.M1

    @classmethod
    def _extract_domain_data(cls, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in payload.items()
            if key not in cls._DOMAIN_EXCLUDED_KEYS
        }

    def _build_contract_features(
            self,
            *,
            source: FeatureSource,
            symbol: str,
            payload: dict[str, Any],
            timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        result: list[FeatureSnapshot] = []

        if source is FeatureSource.OPEN_INTEREST:
            result.extend(
                self._build_open_interest_contract_features(
                    symbol=symbol,
                    payload=payload,
                    timestamp=timestamp,
                )
            )

        elif source is FeatureSource.FUNDING:
            result.extend(
                self._build_funding_contract_features(
                    symbol=symbol,
                    payload=payload,
                    timestamp=timestamp,
                )
            )

        elif source is FeatureSource.ORDERFLOW:
            result.extend(
                self._build_orderflow_contract_features(
                    symbol=symbol,
                    payload=payload,
                    timestamp=timestamp,
                )
            )

        elif source is FeatureSource.LIQUIDATIONS:
            result.extend(
                self._build_liquidations_contract_features(
                    symbol=symbol,
                    payload=payload,
                    timestamp=timestamp,
                )
            )
        elif source is FeatureSource.LIQUIDITY:
            result.extend(
                self._build_liquidity_contract_features(
                    symbol=symbol,
                    payload=payload,
                    timestamp=timestamp,
                )
            )
        elif source is FeatureSource.PRICE_ACTION:
            result.extend(
                self._build_price_action_contract_features(
                    symbol=symbol,
                    payload=payload,
                    timestamp=timestamp,
                )
            )
        elif source is FeatureSource.SPOOFING:
            result.extend(
                self._build_spoofing_contract_features(
                    symbol=symbol,
                    payload=payload,
                    timestamp=timestamp,
                )
            )
        elif source is FeatureSource.SPREADS:
            result.extend(
                self._build_spreads_contract_features(
                    symbol=symbol,
                    payload=payload,
                    timestamp=timestamp,
                )
            )
        elif source is FeatureSource.WHALES:
            result.extend(
                self._build_whales_contract_features(
                    symbol=symbol,
                    payload=payload,
                    timestamp=timestamp,
                )
            )

        if source in {
            FeatureSource.OPEN_INTEREST,
            FeatureSource.FUNDING,
            FeatureSource.ORDERFLOW,
            FeatureSource.LIQUIDITY,
            FeatureSource.LIQUIDATIONS,
            FeatureSource.WHALES,
            FeatureSource.PRICE_ACTION,
            FeatureSource.SPOOFING,
            FeatureSource.SPREADS,
        }:
            result.extend(
                self._build_hybrid_contract_features(
                    source=source,
                    symbol=symbol,
                    payload=payload,
                    timestamp=timestamp,
                )
            )

        return result

    def _build_whales_contract_features(
            self,
            *,
            symbol: str,
            payload: dict[str, Any],
            timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        result: list[FeatureSnapshot] = []
        confidence = payload.get("confidence", 0.0)

        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def mapping_for(*keys: str) -> dict[str, Any]:
            for key in keys:
                value = payload.get(key)
                if isinstance(value, dict):
                    return value

                value = feature_map.get(key)
                if isinstance(value, dict):
                    return value

            return {}

        composite = mapping_for(
            "composite",
            "whales",
            "snapshot",
            "result",
        )
        activity = mapping_for(
            "activity",
            "whale_activity",
            "activity_context",
        )
        pressure = mapping_for(
            "pressure",
            "whale_pressure",
            "pressure_context",
        )
        large_trade = mapping_for(
            "large_trade",
            "large_trade_event",
            "large_trade_context",
        )
        cluster = mapping_for(
            "cluster",
            "whale_cluster",
            "cluster_context",
        )
        liquidation_context = mapping_for(
            "liquidation_context",
            "whale_liquidation_context",
            "liquidations",
            "liquidation",
        )
        exhaustion = mapping_for(
            "exhaustion",
            "cluster_exhaustion",
            "exhaustion_context",
        )
        signal = mapping_for(
            "signal",
            "whale_signal",
            "analytics_signal",
            "event",
        )

        if composite:
            if not activity:
                for key in ("activity", "whale_activity", "activity_context"):
                    value = composite.get(key)
                    if isinstance(value, dict):
                        activity = value
                        break

            if not pressure:
                for key in ("pressure", "whale_pressure", "pressure_context"):
                    value = composite.get(key)
                    if isinstance(value, dict):
                        pressure = value
                        break

            if not large_trade:
                for key in ("large_trade", "large_trade_event", "large_trade_context"):
                    value = composite.get(key)
                    if isinstance(value, dict):
                        large_trade = value
                        break

            if not cluster:
                for key in ("cluster", "whale_cluster", "cluster_context"):
                    value = composite.get(key)
                    if isinstance(value, dict):
                        cluster = value
                        break

            if not liquidation_context:
                for key in (
                        "liquidation_context",
                        "whale_liquidation_context",
                        "liquidations",
                        "liquidation",
                ):
                    value = composite.get(key)
                    if isinstance(value, dict):
                        liquidation_context = value
                        break

            if not exhaustion:
                for key in ("exhaustion", "cluster_exhaustion", "exhaustion_context"):
                    value = composite.get(key)
                    if isinstance(value, dict):
                        exhaustion = value
                        break

            if not signal:
                for key in ("signal", "whale_signal", "analytics_signal", "event"):
                    value = composite.get(key)
                    if isinstance(value, dict):
                        signal = value
                        break

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]
                if key in signal:
                    return signal[key]
                if key in activity:
                    return activity[key]
                if key in pressure:
                    return pressure[key]
                if key in large_trade:
                    return large_trade[key]
                if key in cluster:
                    return cluster[key]
                if key in liquidation_context:
                    return liquidation_context[key]
                if key in exhaustion:
                    return exhaustion[key]
                if key in composite:
                    return composite[key]
            return default

        def nested_value(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in mapping:
                    return mapping[key]
            return default

        def add(name: str, value: Any) -> None:
            result.append(
                self._snapshot_from_raw_value(
                    source=FeatureSource.WHALES,
                    symbol=symbol,
                    name=name,
                    value=value,
                    timestamp=timestamp,
                    confidence=confidence,
                    metadata={
                        "origin": "contract_feature",
                        "contract": "whales",
                    },
                )
            )

        # Main sections
        add("whales.composite", composite or payload)
        add("whales.signal", signal)
        add("whales.activity", activity)
        add("whales.pressure", pressure)
        add("whales.large_trade", large_trade)
        add("whales.cluster", cluster)
        add("whales.liquidation_context", liquidation_context)
        add("whales.exhaustion", exhaustion)

        # Common identity / direction
        add(
            "whales.side",
            value_for(
                "side",
                "whale_side",
                "direction",
                "bias",
                default=None,
            ),
        )
        add(
            "whales.score",
            value_for(
                "score",
                "whale_score",
                default=0.0,
            ),
        )
        add(
            "whales.confidence",
            value_for(
                "confidence",
                "whale_confidence",
                default=0.0,
            ),
        )
        add(
            "whales.event_time",
            value_for(
                "event_time",
                "timestamp",
                "created_at",
                "time",
                default=timestamp,
            ),
        )

        # Activity
        add(
            "whales.activity.notional",
            nested_value(
                activity,
                "notional",
                "total_notional",
                "total_notional_usd",
                default=value_for(
                    "activity_notional",
                    "total_notional",
                    "total_notional_usd",
                    default=0.0,
                ),
            ),
        )
        add(
            "whales.activity.trade_count",
            nested_value(
                activity,
                "trade_count",
                "trades_count",
                "event_count",
                default=value_for("activity_trade_count", "trade_count", default=0),
            ),
        )
        add(
            "whales.activity.side",
            nested_value(
                activity,
                "side",
                "dominant_side",
                default=value_for("activity_side", "dominant_side", default=None),
            ),
        )
        add(
            "whales.activity.score",
            nested_value(
                activity,
                "score",
                default=value_for("activity_score", default=0.0),
            ),
        )

        # Pressure
        add(
            "whales.pressure.side",
            nested_value(
                pressure,
                "side",
                "dominant_side",
                default=value_for("pressure_side", "dominant_side", default=None),
            ),
        )
        add(
            "whales.pressure.score",
            nested_value(
                pressure,
                "score",
                "pressure_score",
                default=value_for("pressure_score", default=0.0),
            ),
        )
        add(
            "whales.pressure.imbalance_ratio",
            nested_value(
                pressure,
                "imbalance_ratio",
                "pressure_imbalance_ratio",
                default=value_for("pressure_imbalance_ratio", default=0.0),
            ),
        )

        # Large trade
        add(
            "whales.large_trade.notional",
            nested_value(
                large_trade,
                "notional",
                "notional_usd",
                default=value_for("large_trade_notional", "notional_usd", default=0.0),
            ),
        )
        add(
            "whales.large_trade.zscore",
            nested_value(
                large_trade,
                "zscore",
                "z_score",
                default=value_for("large_trade_zscore", "zscore", default=0.0),
            ),
        )
        add(
            "whales.large_trade.side",
            nested_value(
                large_trade,
                "side",
                "trade_side",
                default=value_for("large_trade_side", "trade_side", default=None),
            ),
        )

        # Cluster
        add(
            "whales.cluster.score",
            nested_value(
                cluster,
                "score",
                "cluster_score",
                default=value_for("cluster_score", default=0.0),
            ),
        )
        add(
            "whales.cluster.side",
            nested_value(
                cluster,
                "side",
                "cluster_side",
                "dominant_side",
                default=value_for("cluster_side", default=None),
            ),
        )
        add(
            "whales.cluster.continuation_probability",
            nested_value(
                cluster,
                "continuation_probability",
                "continuation_prob",
                default=value_for("continuation_probability", default=0.0),
            ),
        )
        add(
            "whales.cluster.exhaustion_probability",
            nested_value(
                cluster,
                "exhaustion_probability",
                "exhaustion_prob",
                default=value_for("exhaustion_probability", default=0.0),
            ),
        )

        # Liquidation context
        add(
            "whales.liquidation_context.side",
            nested_value(
                liquidation_context,
                "side",
                "liquidation_side",
                default=value_for("liquidation_side", default=None),
            ),
        )
        add(
            "whales.liquidation_context.notional",
            nested_value(
                liquidation_context,
                "notional",
                "notional_usd",
                "total_notional_usd",
                default=value_for(
                    "liquidation_notional",
                    "total_liquidation_notional",
                    default=0.0,
                ),
            ),
        )
        add(
            "whales.liquidation_context.strength",
            nested_value(
                liquidation_context,
                "strength",
                "context_strength",
                default=value_for("liquidation_context_strength", "context_strength", default=0.0),
            ),
        )

        # Exhaustion
        add(
            "whales.exhaustion.probability",
            nested_value(
                exhaustion,
                "probability",
                "exhaustion_probability",
                default=value_for("exhaustion_probability", default=0.0),
            ),
        )
        add(
            "whales.exhaustion.side",
            nested_value(
                exhaustion,
                "side",
                "exhaustion_side",
                default=value_for("exhaustion_side", default=None),
            ),
        )

        return result

    def _build_spreads_contract_features(
            self,
            *,
            symbol: str,
            payload: dict[str, Any],
            timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        result: list[FeatureSnapshot] = []
        confidence = payload.get("confidence", 0.0)

        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def mapping_for(*keys: str) -> dict[str, Any]:
            for key in keys:
                value = payload.get(key)
                if isinstance(value, dict):
                    return value

                value = feature_map.get(key)
                if isinstance(value, dict):
                    return value

            return {}

        snapshot = mapping_for(
            "snapshot",
            "spread_snapshot",
            "spot_futures",
            "spot_futures_snapshot",
            "cross_exchange",
            "cross_exchange_snapshot",
            "result",
        )
        signal = mapping_for(
            "signal",
            "spread_signal",
            "analytics_signal",
            "event",
        )
        opportunity = mapping_for(
            "opportunity",
            "arbitrage_opportunity",
            "arb_opportunity",
        )
        metadata = mapping_for(
            "metadata",
            "spread_metadata",
        )

        if snapshot:
            if not signal:
                for key in ("signal", "spread_signal", "analytics_signal", "event"):
                    value = snapshot.get(key)
                    if isinstance(value, dict):
                        signal = value
                        break

            if not opportunity:
                for key in ("opportunity", "arbitrage_opportunity", "arb_opportunity"):
                    value = snapshot.get(key)
                    if isinstance(value, dict):
                        opportunity = value
                        break

            if not metadata:
                for key in ("metadata", "spread_metadata"):
                    value = snapshot.get(key)
                    if isinstance(value, dict):
                        metadata = value
                        break

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]
                if key in signal:
                    return signal[key]
                if key in opportunity:
                    return opportunity[key]
                if key in snapshot:
                    return snapshot[key]
                if key in metadata:
                    return metadata[key]
            return default

        def nested_value(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in mapping:
                    return mapping[key]
            return default

        def add(name: str, value: Any) -> None:
            result.append(
                self._snapshot_from_raw_value(
                    source=FeatureSource.SPREADS,
                    symbol=symbol,
                    name=name,
                    value=value,
                    timestamp=timestamp,
                    confidence=confidence,
                    metadata={
                        "origin": "contract_feature",
                        "contract": "spreads",
                    },
                )
            )

        # Main sections
        add("spreads.snapshot", snapshot or payload)
        add("spreads.signal", signal)
        add("spreads.opportunity", opportunity)

        # Identity
        add(
            "spreads.type",
            value_for(
                "spread_type",
                "type",
                default=None,
            ),
        )
        add(
            "spreads.symbol",
            value_for(
                "symbol",
                default=symbol,
            ),
        )

        # Legs / venues
        add(
            "spreads.exchange_a",
            value_for(
                "exchange_a",
                "spot_exchange",
                "buy_exchange",
                default=None,
            ),
        )
        add(
            "spreads.exchange_b",
            value_for(
                "exchange_b",
                "futures_exchange",
                "sell_exchange",
                default=None,
            ),
        )
        add(
            "spreads.market_type_a",
            value_for(
                "market_type_a",
                "spot_market_type",
                "buy_market_type",
                default=None,
            ),
        )
        add(
            "spreads.market_type_b",
            value_for(
                "market_type_b",
                "futures_market_type",
                "sell_market_type",
                default=None,
            ),
        )
        add(
            "spreads.exchange_symbol_a",
            value_for(
                "exchange_symbol_a",
                "symbol_a",
                "buy_exchange_symbol",
                default=None,
            ),
        )
        add(
            "spreads.exchange_symbol_b",
            value_for(
                "exchange_symbol_b",
                "symbol_b",
                "sell_exchange_symbol",
                default=None,
            ),
        )

        # Spread metrics
        add(
            "spreads.spread_bps",
            value_for(
                "spread_bps",
                "basis_bps",
                default=0.0,
            ),
        )
        add(
            "spreads.basis",
            value_for(
                "basis",
                "basis_value",
                default=0.0,
            ),
        )
        add(
            "spreads.funding_adjusted_spread",
            value_for(
                "funding_adjusted_spread",
                "funding_adjusted_edge",
                default=0.0,
            ),
        )
        add(
            "spreads.net_edge",
            value_for(
                "net_edge",
                "edge",
                default=0.0,
            ),
        )
        add(
            "spreads.net_edge_bps",
            value_for(
                "net_edge_bps",
                "edge_bps",
                default=0.0,
            ),
        )
        add(
            "spreads.zscore",
            value_for(
                "zscore",
                "z_score",
                default=0.0,
            ),
        )

        # Signal / regime
        add(
            "spreads.regime",
            value_for(
                "regime",
                "spread_regime",
                default=None,
            ),
        )
        add(
            "spreads.direction",
            value_for(
                "direction",
                "spread_direction",
                "bias",
                default=None,
            ),
        )
        add(
            "spreads.signal_type",
            value_for(
                "signal_type",
                "spread_signal_type",
                "type",
                default=None,
            ),
        )
        add(
            "spreads.quote_validity",
            value_for(
                "quote_validity",
                "validity",
                default=None,
            ),
        )
        add(
            "spreads.has_edge",
            value_for(
                "has_edge",
                "tradeable_edge",
                default=False,
            ),
        )
        add(
            "spreads.confidence",
            value_for(
                "confidence",
                "signal_confidence",
                default=0.0,
            ),
        )

        # Opportunity
        add(
            "spreads.opportunity_key",
            value_for(
                "opportunity_key",
                "key",
                default=None,
            ),
        )
        add(
            "spreads.opportunity_status",
            value_for(
                "opportunity_status",
                "status",
                default=None,
            ),
        )
        add(
            "spreads.persistence_ms",
            value_for(
                "persistence_ms",
                "duration_ms",
                default=0,
            ),
        )

        # Arbitrage leg direction
        add(
            "spreads.buy_exchange",
            value_for(
                "buy_exchange",
                default=nested_value(opportunity, "buy_exchange"),
            ),
        )
        add(
            "spreads.sell_exchange",
            value_for(
                "sell_exchange",
                default=nested_value(opportunity, "sell_exchange"),
            ),
        )
        add(
            "spreads.buy_market_type",
            value_for(
                "buy_market_type",
                default=nested_value(opportunity, "buy_market_type"),
            ),
        )
        add(
            "spreads.sell_market_type",
            value_for(
                "sell_market_type",
                default=nested_value(opportunity, "sell_market_type"),
            ),
        )

        # Instrument type compatibility
        add(
            "spreads.leg_a.instrument_type",
            value_for(
                "leg_a_instrument_type",
                "instrument_type_a",
                default=nested_value(snapshot, "leg_a", default={}).get("instrument_type")
                if isinstance(nested_value(snapshot, "leg_a", default={}), dict)
                else None,
            ),
        )
        add(
            "spreads.leg_b.instrument_type",
            value_for(
                "leg_b_instrument_type",
                "instrument_type_b",
                default=nested_value(snapshot, "leg_b", default={}).get("instrument_type")
                if isinstance(nested_value(snapshot, "leg_b", default={}), dict)
                else None,
            ),
        )
        add(
            "spreads.instrument_type",
            value_for(
                "instrument_type",
                default=None,
            ),
        )

        add("spreads.metadata", metadata)

        return result

    def _build_spoofing_contract_features(
            self,
            *,
            symbol: str,
            payload: dict[str, Any],
            timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        result: list[FeatureSnapshot] = []
        confidence = payload.get("confidence", 0.0)

        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def mapping_for(*keys: str) -> dict[str, Any]:
            for key in keys:
                value = payload.get(key)
                if isinstance(value, dict):
                    return value

                value = feature_map.get(key)
                if isinstance(value, dict):
                    return value

            return {}

        composite = mapping_for(
            "composite",
            "spoofing",
            "snapshot",
            "result",
        )
        signal = mapping_for(
            "signal",
            "spoofing_signal",
            "analytics_signal",
            "event",
        )
        features = mapping_for(
            "features",
            "spoofing_features",
        )
        detector_results = mapping_for(
            "detector_results",
            "detectors",
            "detector_result",
        )
        score_breakdown = mapping_for(
            "score_breakdown",
            "scores",
            "score_components",
        )
        analytics_metadata = mapping_for(
            "analytics_metadata",
            "metadata",
        )

        if composite:
            if not signal:
                for key in ("signal", "spoofing_signal", "analytics_signal", "event"):
                    value = composite.get(key)
                    if isinstance(value, dict):
                        signal = value
                        break

            if not features:
                for key in ("features", "spoofing_features"):
                    value = composite.get(key)
                    if isinstance(value, dict):
                        features = value
                        break

            if not detector_results:
                for key in ("detector_results", "detectors", "detector_result"):
                    value = composite.get(key)
                    if isinstance(value, dict):
                        detector_results = value
                        break

            if not score_breakdown:
                for key in ("score_breakdown", "scores", "score_components"):
                    value = composite.get(key)
                    if isinstance(value, dict):
                        score_breakdown = value
                        break

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]
                if key in signal:
                    return signal[key]
                if key in features:
                    return features[key]
                if key in composite:
                    return composite[key]
            return default

        def nested_value(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in mapping:
                    return mapping[key]
            return default

        def add(name: str, value: Any) -> None:
            result.append(
                self._snapshot_from_raw_value(
                    source=FeatureSource.SPOOFING,
                    symbol=symbol,
                    name=name,
                    value=value,
                    timestamp=timestamp,
                    confidence=confidence,
                    metadata={
                        "origin": "contract_feature",
                        "contract": "spoofing",
                    },
                )
            )

        # Composite / signal
        add("spoofing.composite", composite or payload)
        add("spoofing.signal", signal or payload)

        # Nested contract sections
        add("spoofing.features", features)
        add("spoofing.detector_results", detector_results)
        add("spoofing.score_breakdown", score_breakdown)
        add("spoofing.analytics_metadata", analytics_metadata)

        # Core signal identity
        add(
            "spoofing.type",
            value_for(
                "spoofing_type",
                "type",
                "signal_type",
                default=None,
            ),
        )
        add(
            "spoofing.pattern",
            value_for(
                "pattern",
                "spoofing_pattern",
                default=None,
            ),
        )
        add(
            "spoofing.side",
            value_for(
                "side",
                "spoofing_side",
                "direction",
                "bias",
                default=None,
            ),
        )
        add(
            "spoofing.severity",
            value_for(
                "severity",
                "spoofing_severity",
                default=None,
            ),
        )
        add(
            "spoofing.status",
            value_for(
                "status",
                "spoofing_status",
                default=None,
            ),
        )

        # Scores
        add(
            "spoofing.score",
            value_for(
                "score",
                "spoofing_score",
                default=0.0,
            ),
        )
        add(
            "spoofing.confidence",
            value_for(
                "confidence",
                "spoofing_confidence",
                default=0.0,
            ),
        )

        # Position / wall identity
        add(
            "spoofing.price_level",
            value_for(
                "price_level",
                "level",
                "wall_price",
                default=None,
            ),
        )
        add(
            "spoofing.wall_id",
            value_for(
                "wall_id",
                "id",
                "order_id",
                default=None,
            ),
        )
        add(
            "spoofing.event_time",
            value_for(
                "event_time",
                "timestamp",
                "created_at",
                "time",
                default=timestamp,
            ),
        )

        # Feature metrics
        add(
            "spoofing.features.pull_ratio",
            value_for(
                "pull_ratio",
                default=0.0,
            ),
        )
        add(
            "spoofing.features.fill_ratio",
            value_for(
                "fill_ratio",
                default=0.0,
            ),
        )
        add(
            "spoofing.features.price_reaction_bps",
            value_for(
                "price_reaction_bps",
                default=0.0,
            ),
        )
        add(
            "spoofing.features.signed_price_reaction_bps",
            value_for(
                "signed_price_reaction_bps",
                "price_reaction_bps",
                default=0.0,
            ),
        )
        add(
            "spoofing.features.lifetime_ms",
            value_for(
                "lifetime_ms",
                "wall_lifetime_ms",
                default=0.0,
            ),
        )
        add(
            "spoofing.features.wall_notional",
            value_for(
                "wall_notional",
                "notional",
                default=0.0,
            ),
        )
        add(
            "spoofing.features.pulled_notional",
            value_for(
                "pulled_notional",
                default=0.0,
            ),
        )
        add(
            "spoofing.features.cancel_to_fill_ratio",
            value_for(
                "cancel_to_fill_ratio",
                default=0.0,
            ),
        )
        add(
            "spoofing.features.distance_from_mid_bps",
            value_for(
                "distance_from_mid_bps",
                default=0.0,
            ),
        )
        add(
            "spoofing.features.layer_count",
            value_for(
                "layer_count",
                "layers_count",
                default=0,
            ),
        )
        add(
            "spoofing.features.layer_price_span_bps",
            value_for(
                "layer_price_span_bps",
                default=0.0,
            ),
        )
        add(
            "spoofing.features.pressure_flip_strength",
            value_for(
                "pressure_flip_strength",
                "flip_strength",
                default=0.0,
            ),
        )

        return result

    def _build_price_action_contract_features(
            self,
            *,
            symbol: str,
            payload: dict[str, Any],
            timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        """
        Build stable price-action contract features.

        Unlike the old broad fallback, module-level features are emitted only
        when a compatible section/event exists. This avoids routing FVG/SR/Trend
        strategies from a pure market-structure event and then getting generic
        no_signal_generated responses.
        """
        result: list[FeatureSnapshot] = []
        confidence = payload.get("confidence", 0.0)

        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def mapping_for(*keys: str) -> dict[str, Any] | None:
            for key in keys:
                value = payload.get(key)
                if isinstance(value, dict):
                    return dict(value)

                value = feature_map.get(key)
                if isinstance(value, dict):
                    return dict(value)

            return None

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]
                if isinstance(state, dict) and key in state:
                    return state[key]
            return default

        def nested_value(mapping: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
            if not mapping:
                return default
            for key in keys:
                if key in mapping:
                    return mapping[key]
            return default

        def add(name: str, value: Any, *, section: str | None = None) -> None:
            if value is None:
                return
            result.append(
                self._snapshot_from_raw_value(
                    source=FeatureSource.PRICE_ACTION,
                    symbol=symbol,
                    name=name,
                    value=value,
                    timestamp=timestamp,
                    confidence=confidence,
                    metadata={
                        "origin": "contract_feature",
                        "contract": "price_action",
                        **({"section": section} if section else {}),
                    },
                )
            )

        topic = (
            str(
                payload.get("event_name")
                or payload.get("topic")
                or payload.get("source_topic")
                or ""
            )
            .strip()
            .lower()
        )

        state = mapping_for("state", "composite", "price_action", "snapshot", "result")
        market_structure = mapping_for("market_structure", "structure", "ms")
        support_resistance = mapping_for("support_resistance", "sr", "levels")
        fair_value_gap = mapping_for("fair_value_gap", "fvg", "fair_value_gaps")
        trend = mapping_for("trend", "trend_state")
        liquidity_levels = mapping_for("liquidity_levels", "liquidity")

        if isinstance(state, dict):
            for target_name, current, aliases in (
                ("market_structure", market_structure, ("market_structure", "structure", "ms")),
                ("support_resistance", support_resistance, ("support_resistance", "sr", "levels")),
                ("fair_value_gap", fair_value_gap, ("fair_value_gap", "fvg", "fair_value_gaps")),
                ("trend", trend, ("trend", "trend_state")),
                ("liquidity_levels", liquidity_levels, ("liquidity_levels", "liquidity")),
            ):
                if current is not None:
                    continue
                for alias in aliases:
                    nested = state.get(alias)
                    if isinstance(nested, dict):
                        if target_name == "market_structure":
                            market_structure = dict(nested)
                        elif target_name == "support_resistance":
                            support_resistance = dict(nested)
                        elif target_name == "fair_value_gap":
                            fair_value_gap = dict(nested)
                        elif target_name == "trend":
                            trend = dict(nested)
                        elif target_name == "liquidity_levels":
                            liquidity_levels = dict(nested)
                        break

        event_payload = dict(payload)
        current_price = value_for("current_price", "price", "last_price", "close", default=None)
        if current_price is not None:
            event_payload.setdefault("price", current_price)
            event_payload.setdefault("current_price", current_price)
        event_payload.setdefault("timestamp", value_for("timestamp", "event_time", default=timestamp))
        event_payload.setdefault("confidence", value_for("confidence", default=0.0))
        event_payload.setdefault("score", value_for("score", default=event_payload.get("confidence", 0.0)))

        is_market_structure_event = (
            market_structure is not None
            or "market_structure" in topic
            or ".structure" in topic
            or any(key in payload for key in ("swing_type", "break_distance_pct", "market_bias", "broken_side", "mtf_alignment"))
        )
        is_sr_event = (
            support_resistance is not None
            or "support_resistance" in topic
            or ".sr." in topic
            or any(key in payload for key in ("level_type", "level_status", "level_price", "touch_count", "reaction_count", "break_count"))
        )
        is_fvg_event = (
            fair_value_gap is not None
            or "fair_value_gap" in topic
            or "fvg" in topic
            or any(key in payload for key in ("fvg_direction", "gap_size_pct", "fill_pct", "upper_price", "lower_price", "mid_price"))
        )
        is_trend_event = (
            trend is not None
            or "trend" in topic
            or any(key in payload for key in ("trend_direction", "trend_regime", "continuation_probability", "reversal_risk", "exhaustion_score", "overall_trend_score"))
        )

        # Composite / global features are safe for all price-action payloads.
        if state is not None or any((is_market_structure_event, is_sr_event, is_fvg_event, is_trend_event, liquidity_levels is not None)):
            add("price_action.composite", state or payload, section="composite")
        add("price_action.current_price", current_price, section="price")
        add("price_action.last_price", value_for("last_price", "price", "close", default=current_price), section="price")
        add("price_action.timestamp", value_for("timestamp", "event_time", default=timestamp), section="time")

        if is_market_structure_event:
            if market_structure is None:
                event_type = value_for("event_type", "type", "kind", default=None)
                if event_type is None:
                    if ".bos" in topic or topic.endswith("bos"):
                        event_type = "bos"
                    elif ".choch" in topic:
                        event_type = "choch"
                    elif ".mss" in topic:
                        event_type = "mss"
                side = value_for("side", "direction", "bias", default=None)
                bias = value_for("market_bias", "bias", "direction", default=side)
                event_payload.setdefault("event_type", event_type)
                event_payload.setdefault("side", side)
                event_payload.setdefault("direction", side)
                event_payload.setdefault("market_bias", bias)
                event_payload.setdefault("confirmed", True)
                market_structure = {
                    "last_break_event": event_payload,
                    "last_event": event_payload,
                    "event": event_payload,
                    "external": {
                        "last_break_event": event_payload,
                        "last_event": event_payload,
                        "bias": bias,
                        "market_bias": bias,
                        "confidence": event_payload.get("confidence", 0.0),
                        "score": event_payload.get("score", 0.0),
                        "strength": event_payload.get("score", event_payload.get("confidence", 0.0)),
                        "trend_strength": event_payload.get("score", event_payload.get("confidence", 0.0)),
                    },
                    "internal": {"bias": bias, "market_bias": bias, "confidence": event_payload.get("confidence", 0.0), "score": event_payload.get("score", 0.0)},
                    "mtf_alignment": value_for("mtf_alignment", "alignment", default=event_payload.get("score", 0.0)),
                    "trend_strength": event_payload.get("score", event_payload.get("confidence", 0.0)),
                }

            add("price_action.market_structure", market_structure, section="market_structure")
            add("price_action.market_structure.internal", nested_value(market_structure, "internal", default={}), section="market_structure")
            add("price_action.market_structure.external", nested_value(market_structure, "external", default={}), section="market_structure")
            add("price_action.market_structure.last_break_event", nested_value(market_structure, "last_break_event", "last_event", "event", default=event_payload), section="market_structure")
            add("price_action.market_structure.mtf_alignment", nested_value(market_structure, "mtf_alignment", "multi_timeframe_alignment", default=value_for("mtf_alignment", default=0.0)), section="market_structure")

        if is_sr_event:
            add("price_action.support_resistance", support_resistance or event_payload, section="support_resistance")
            add("price_action.support_resistance.internal", nested_value(support_resistance, "internal", default={}), section="support_resistance")
            add("price_action.support_resistance.external", nested_value(support_resistance, "external", default={}), section="support_resistance")
            add("price_action.support_resistance.last_event", nested_value(support_resistance, "last_event", "event", default=event_payload), section="support_resistance")
            add("price_action.support_resistance.nearest_support", nested_value(support_resistance, "nearest_support", "support", default=value_for("nearest_support", default=None)), section="support_resistance")
            add("price_action.support_resistance.nearest_resistance", nested_value(support_resistance, "nearest_resistance", "resistance", default=value_for("nearest_resistance", default=None)), section="support_resistance")

        if is_fvg_event:
            add("price_action.fair_value_gap", fair_value_gap or event_payload, section="fair_value_gap")
            add("price_action.fvg", fair_value_gap or event_payload, section="fair_value_gap")
            add("price_action.fair_value_gap.internal", nested_value(fair_value_gap, "internal", default={}), section="fair_value_gap")
            add("price_action.fair_value_gap.external", nested_value(fair_value_gap, "external", default={}), section="fair_value_gap")
            add("price_action.fair_value_gap.last_event", nested_value(fair_value_gap, "last_event", "event", default=event_payload), section="fair_value_gap")
            add("price_action.fair_value_gap.nearest_bullish_gap", nested_value(fair_value_gap, "nearest_bullish_gap", "bullish_gap", default=value_for("nearest_bullish_gap", default=None)), section="fair_value_gap")
            add("price_action.fair_value_gap.nearest_bearish_gap", nested_value(fair_value_gap, "nearest_bearish_gap", "bearish_gap", default=value_for("nearest_bearish_gap", default=None)), section="fair_value_gap")

        if is_trend_event:
            if trend is None:
                event_payload.setdefault("event_type", value_for("event_type", "type", "kind", default="trend_alignment"))
                event_payload.setdefault("trend_direction", value_for("trend_direction", "direction", "side", default=None))
                event_payload.setdefault("trend_regime", value_for("trend_regime", "regime", "state", default=None))
                event_payload.setdefault("continuation_probability", value_for("continuation_probability", "continuation_prob", "probability", default=event_payload.get("score", 0.0)))
                layer = {
                    **event_payload,
                    "direction": event_payload.get("trend_direction") or event_payload.get("direction"),
                    "trend_direction": event_payload.get("trend_direction") or event_payload.get("direction"),
                    "regime": event_payload.get("trend_regime"),
                    "trend_regime": event_payload.get("trend_regime"),
                    "trend_strength": value_for("trend_strength", "strength", "score", default=event_payload.get("score", 0.0)),
                    "momentum_score": value_for("momentum_score", "momentum", default=event_payload.get("score", 0.0)),
                    "slope_score": value_for("slope_score", "slope", default=event_payload.get("score", 0.0)),
                    "continuation_probability": event_payload.get("continuation_probability"),
                    "confidence": event_payload.get("confidence", 0.0),
                    "score": event_payload.get("score", event_payload.get("confidence", 0.0)),
                }
                trend = {
                    "last_signal": event_payload,
                    "last_event": event_payload,
                    "event": event_payload,
                    "external": layer,
                    "internal": layer,
                    "internal_external_alignment": value_for("internal_external_alignment", "alignment", default=event_payload.get("score", 0.0)),
                    "higher_timeframe_alignment": value_for("higher_timeframe_alignment", "htf_alignment", default=event_payload.get("score", 0.0)),
                    "overall_trend_score": value_for("overall_trend_score", "score", default=event_payload.get("score", 0.0)),
                }
            add("price_action.trend", trend, section="trend")
            add("price_action.trend.internal", nested_value(trend, "internal", default={}), section="trend")
            add("price_action.trend.external", nested_value(trend, "external", default={}), section="trend")
            add("price_action.trend.last_signal", nested_value(trend, "last_signal", "last_event", "event", default=event_payload), section="trend")
            add("price_action.trend.internal_external_alignment", nested_value(trend, "internal_external_alignment", default=value_for("internal_external_alignment", default=0.0)), section="trend")
            add("price_action.trend.higher_timeframe_alignment", nested_value(trend, "higher_timeframe_alignment", "htf_alignment", default=value_for("higher_timeframe_alignment", "htf_alignment", default=0.0)), section="trend")
            add("price_action.trend.overall_trend_score", nested_value(trend, "overall_trend_score", "score", default=value_for("overall_trend_score", "score", default=0.0)), section="trend")

        if liquidity_levels is not None:
            add("price_action.liquidity_levels", liquidity_levels, section="liquidity_levels")

        return result

    def _build_liquidity_contract_features(
            self,
            *,
            symbol: str,
            payload: dict[str, Any],
            timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        result: list[FeatureSnapshot] = []

        confidence = payload.get("confidence", 0.0)
        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def as_mapping(value: Any) -> dict[str, Any] | None:
            if isinstance(value, dict):
                return value

            to_dict = getattr(value, "to_dict", None)
            if callable(to_dict):
                converted = to_dict()
                if isinstance(converted, dict):
                    return converted

            return None

        def get_item(value: Any, key: str, default: Any = None) -> Any:
            mapping = as_mapping(value)
            if mapping is not None:
                return mapping.get(key, default)
            return getattr(value, key, default)

        def mapping_or_object_for(*keys: str) -> Any | None:
            for key in keys:
                if key in payload and payload[key] is not None:
                    return payload[key]
                if key in feature_map and feature_map[key] is not None:
                    return feature_map[key]
            return None

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]

            for container in (snapshot, signal):
                for key in keys:
                    value = get_item(container, key, None)
                    if value is not None:
                        return value

            snapshot_metadata = get_item(snapshot, "metadata", None)
            if isinstance(snapshot_metadata, dict):
                for key in keys:
                    if key in snapshot_metadata:
                        return snapshot_metadata[key]

            return default

        def nested_value(value: Any, *keys: str, default: Any = None) -> Any:
            for key in keys:
                item = get_item(value, key, None)
                if item is not None:
                    return item
            return default

        def add(
                name: str,
                value: Any,
                *,
                section: str | None = None,
        ) -> None:
            if value is None:
                return

            result.append(
                self._snapshot_from_raw_value(
                    source=FeatureSource.LIQUIDITY,
                    symbol=symbol,
                    name=name,
                    value=value,
                    timestamp=timestamp,
                    confidence=confidence,
                    metadata={
                        "origin": "contract_feature",
                        "contract": "liquidity",
                        **({"section": section} if section else {}),
                    },
                )
            )

        snapshot = mapping_or_object_for(
            "snapshot",
            "liquidity_map_snapshot",
            "map_snapshot",
            "last_snapshot",
            "liquidity_snapshot",
        )

        for container_key in (
                "result",
                "liquidity",
                "data",
                "payload",
                "analysis",
                "liquidity_analysis",
        ):
            container = mapping_or_object_for(container_key)
            container_map = as_mapping(container)
            if container_map is None:
                continue

            nested_snapshot = (
                    container_map.get("snapshot")
                    or container_map.get("liquidity_map_snapshot")
                    or container_map.get("map_snapshot")
                    or container_map.get("last_snapshot")
            )
            if nested_snapshot is not None and snapshot is None:
                snapshot = nested_snapshot

        signal = mapping_or_object_for(
            "signal",
            "liquidity_signal",
            "analytics_signal",
            "setup",
        )

        levels = (
                mapping_or_object_for("levels", "active_levels", "liquidity_levels")
                or nested_value(snapshot, "levels", "active_levels", default=None)
        )
        equal_levels = (
                mapping_or_object_for("equal_levels", "equal_highs_lows")
                or nested_value(snapshot, "equal_levels", default=None)
        )
        clusters = (
                mapping_or_object_for("clusters", "stop_clusters", "liquidity_clusters")
                or nested_value(snapshot, "stop_clusters", "clusters", default=None)
        )
        zones = (
                mapping_or_object_for("zones", "liquidity_zones")
                or nested_value(snapshot, "zones", default=None)
        )

        # Critical: only add liquidity.snapshot when analytics supplied an
        # actual snapshot object/wrapper. Do not use raw payload fallback here.
        if snapshot is not None:
            add("liquidity.snapshot", snapshot, section="snapshot")
            add("liquidity.map.snapshot", snapshot, section="snapshot")

        current_price = value_for(
            "current_price",
            "price",
            "mark_price",
            "last_price",
            "close",
            default=None,
        )
        add("liquidity.current_price", current_price, section="snapshot")

        add(
            "liquidity.above_liquidity_score",
            value_for(
                "above_liquidity_score",
                "above_score",
                default=None,
            ),
            section="snapshot",
        )
        add(
            "liquidity.below_liquidity_score",
            value_for(
                "below_liquidity_score",
                "below_score",
                default=None,
            ),
            section="snapshot",
        )
        add(
            "liquidity.pressure_score",
            value_for(
                "liquidity_pressure_score",
                "pressure_score",
                "liquidity_pressure",
                default=None,
            ),
            section="snapshot",
        )
        add(
            "liquidity.bias",
            value_for(
                "bias",
                "liquidity_bias",
                "direction",
                default=None,
            ),
            section="snapshot",
        )

        sweep_risk = mapping_or_object_for("sweep_risk", "sweep_risks")
        magnet = mapping_or_object_for("magnet", "magnets", "liquidity_magnets")

        add(
            "liquidity.sweep_risk.up",
            nested_value(
                sweep_risk,
                "up",
                "upside",
                "above",
                default=value_for(
                    "sweep_risk_up",
                    "upside_sweep_risk",
                    default=None,
                ),
            ),
            section="sweep_risk",
        )
        add(
            "liquidity.sweep_risk.down",
            nested_value(
                sweep_risk,
                "down",
                "downside",
                "below",
                default=value_for(
                    "sweep_risk_down",
                    "downside_sweep_risk",
                    default=None,
                ),
            ),
            section="sweep_risk",
        )
        add(
            "liquidity.magnet.up",
            nested_value(
                magnet,
                "up",
                "upside",
                "above",
                default=value_for(
                    "magnet_up",
                    "upside_magnet",
                    "upside_magnet_score",
                    default=None,
                ),
            ),
            section="magnet",
        )
        add(
            "liquidity.magnet.down",
            nested_value(
                magnet,
                "down",
                "downside",
                "below",
                default=value_for(
                    "magnet_down",
                    "downside_magnet",
                    "downside_magnet_score",
                    default=None,
                ),
            ),
            section="magnet",
        )

        add(
            "liquidity.nearest_above_level",
            value_for(
                "nearest_above_level",
                "nearest_above",
                default=nested_value(snapshot, "nearest_above_level", default=None),
            ),
            section="levels",
        )
        add(
            "liquidity.nearest_below_level",
            value_for(
                "nearest_below_level",
                "nearest_below",
                default=nested_value(snapshot, "nearest_below_level", default=None),
            ),
            section="levels",
        )
        add(
            "liquidity.strongest_cluster_above",
            value_for(
                "strongest_cluster_above",
                "strongest_above",
                default=nested_value(snapshot, "strongest_cluster_above", default=None),
            ),
            section="clusters",
        )
        add(
            "liquidity.strongest_cluster_below",
            value_for(
                "strongest_cluster_below",
                "strongest_below",
                default=nested_value(snapshot, "strongest_cluster_below", default=None),
            ),
            section="clusters",
        )

        if equal_levels is not None:
            add("liquidity.equal_levels", equal_levels, section="equal_levels")

        if levels is not None:
            add("liquidity.active_levels", levels, section="levels")

        if clusters is not None:
            add("liquidity.stop_clusters", clusters, section="clusters")

        if zones is not None:
            add("liquidity.zones", zones, section="zones")

        if signal is not None:
            add("liquidity.signal", signal, section="signal")
            add(
                "liquidity.signal.type",
                nested_value(
                    signal,
                    "type",
                    "signal_type",
                    "setup_type",
                    default=value_for("signal_type", "setup_type", default=None),
                ),
                section="signal",
            )
            add(
                "liquidity.signal.score",
                nested_value(
                    signal,
                    "score",
                    default=value_for("signal_score", "score", default=None),
                ),
                section="signal",
            )
            add(
                "liquidity.signal.confidence",
                nested_value(
                    signal,
                    "confidence",
                    default=value_for(
                        "signal_confidence",
                        "confidence",
                        default=None,
                    ),
                ),
                section="signal",
            )
            add(
                "liquidity.signal.side",
                nested_value(
                    signal,
                    "side",
                    "direction",
                    "bias",
                    default=value_for("signal_side", "side", "direction", default=None),
                ),
                section="signal",
            )

        return result

    def _build_liquidations_contract_features(
            self,
            *,
            symbol: str,
            payload: dict[str, Any],
            timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        result: list[FeatureSnapshot] = []

        confidence = payload.get("confidence", 0.0)
        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def mapping_for(*keys: str) -> dict[str, Any] | None:
            for key in keys:
                value = payload.get(key)
                if isinstance(value, dict):
                    return dict(value)

                value = feature_map.get(key)
                if isinstance(value, dict):
                    return dict(value)

            return None

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]
            return default

        def nested_value(
                mapping: dict[str, Any] | None,
                *keys: str,
                default: Any = None,
        ) -> Any:
            if not isinstance(mapping, dict):
                return default

            for key in keys:
                if key in mapping:
                    return mapping[key]

            metadata = mapping.get("metadata")
            if isinstance(metadata, dict):
                for key in keys:
                    if key in metadata:
                        return metadata[key]

            return default

        def to_bool(value: Any, default: bool = False) -> bool:
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
                    "detected",
                    "active",
                    "confirmed",
                    "triggered",
                    "valid",
                }:
                    return True

                if normalized in {
                    "0",
                    "false",
                    "no",
                    "n",
                    "off",
                    "none",
                    "not_detected",
                    "inactive",
                    "rejected",
                    "invalid",
                    "expired",
                }:
                    return False

            if isinstance(value, (int, float)):
                return bool(value)

            return default

        def section_detected(section: dict[str, Any] | None) -> bool:
            if not section:
                return False

            detected = section.get(
                "detected",
                section.get(
                    "is_detected",
                    section.get(
                        "active",
                        section.get(
                            "confirmed",
                            section.get("valid", None),
                        ),
                    ),
                ),
            )

            if detected is None:
                return True

            return to_bool(detected, default=False)

        def add(name: str, value: Any, *, section: str | None = None) -> None:
            result.append(
                self._snapshot_from_raw_value(
                    source=FeatureSource.LIQUIDATIONS,
                    symbol=symbol,
                    name=name,
                    value=value,
                    timestamp=timestamp,
                    confidence=confidence,
                    metadata={
                        "origin": "contract_feature",
                        "contract": "liquidations",
                        **({"section": section} if section else {}),
                    },
                )
            )

        analysis = mapping_for(
            "analysis",
            "liquidations_analysis",
            "liquidation_analysis",
        )
        result_section = mapping_for(
            "result",
            "liquidations_result",
            "liquidation_result",
        )
        cascade = mapping_for(
            "cascade",
            "cascade_result",
            "cascade_detection",
            "cascade_detected",
            "liquidation_cascade",
        )
        exhaustion = mapping_for(
            "exhaustion",
            "exhaustion_result",
            "exhaustion_detection",
            "exhaustion_detected",
            "reversal_context",
            "liquidation_exhaustion",
        )
        squeeze = mapping_for(
            "squeeze",
            "squeeze_result",
            "squeeze_reversal",
            "squeeze_context",
            "pending_confirmation",
            "liquidation_squeeze",
        )
        cluster = mapping_for(
            "cluster",
            "liquidation_cluster",
            "cluster_stats",
            "liquidations_cluster",
        )
        signal = mapping_for(
            "signal",
            "liquidation_signal",
            "liquidations_signal",
            "analytics_signal",
            "setup",
        )

        for container in (analysis, result_section):
            if container is None:
                continue

            cascade = cascade or nested_value(
                container,
                "cascade",
                "cascade_result",
                "cascade_detection",
                "liquidation_cascade",
            )
            exhaustion = exhaustion or nested_value(
                container,
                "exhaustion",
                "exhaustion_result",
                "exhaustion_detection",
                "reversal_context",
            )
            squeeze = squeeze or nested_value(
                container,
                "squeeze",
                "squeeze_result",
                "squeeze_reversal",
                "squeeze_context",
            )
            cluster = cluster or nested_value(
                container,
                "cluster",
                "liquidation_cluster",
                "cluster_stats",
            )
            signal = signal or nested_value(
                container,
                "signal",
                "liquidation_signal",
                "analytics_signal",
                "setup",
            )

        if not isinstance(cluster, dict):
            cluster = {}
            for key in (
                    "duration_seconds",
                    "avg_notional_per_event",
                    "side_imbalance_ratio",
                    "event_imbalance_ratio",
                    "acceleration_ratio",
                    "event_count",
                    "total_notional_usd",
                    "price_range_pct",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    cluster[key] = value

        if cluster:
            add("liquidations.cluster", cluster, section="cluster")
            add(
                "liquidations.cluster.duration_seconds",
                nested_value(
                    cluster,
                    "duration_seconds",
                    default=value_for("duration_seconds", default=0.0),
                ),
                section="cluster",
            )
            add(
                "liquidations.cluster.avg_notional_per_event",
                nested_value(
                    cluster,
                    "avg_notional_per_event",
                    default=value_for("avg_notional_per_event", default=0.0),
                ),
                section="cluster",
            )
            add(
                "liquidations.cluster.side_imbalance_ratio",
                nested_value(
                    cluster,
                    "side_imbalance_ratio",
                    default=value_for("side_imbalance_ratio", default=0.0),
                ),
                section="cluster",
            )
            add(
                "liquidations.cluster.event_imbalance_ratio",
                nested_value(
                    cluster,
                    "event_imbalance_ratio",
                    default=value_for("event_imbalance_ratio", default=0.0),
                ),
                section="cluster",
            )
            add(
                "liquidations.cluster.acceleration_ratio",
                nested_value(
                    cluster,
                    "acceleration_ratio",
                    default=value_for("acceleration_ratio", default=0.0),
                ),
                section="cluster",
            )

        if section_detected(cascade):
            add("liquidations.cascade", cascade, section="cascade")
            add(
                "liquidations.cascade.confidence",
                nested_value(
                    cascade,
                    "confidence",
                    default=value_for("cascade_confidence", "confidence", default=0.0),
                ),
                section="cascade",
            )
            add(
                "liquidations.cascade.intensity_score",
                nested_value(
                    cascade,
                    "intensity_score",
                    "intensity",
                    default=value_for("intensity_score", default=0.0),
                ),
                section="cascade",
            )
            add(
                "liquidations.cascade.direction",
                nested_value(
                    cascade,
                    "direction",
                    "cascade_direction",
                    "side",
                    default=value_for(
                        "direction",
                        "cascade_direction",
                        "side",
                        default=None,
                    ),
                ),
                section="cascade",
            )
            add(
                "liquidations.cascade.severity",
                nested_value(
                    cascade,
                    "severity",
                    "severity_label",
                    default=value_for("severity", "severity_label", default=None),
                ),
                section="cascade",
            )
            add(
                "liquidations.cascade.continuation_bias",
                nested_value(
                    cascade,
                    "continuation_bias",
                    default=value_for("continuation_bias", default=0.0),
                ),
                section="cascade",
            )
            add(
                "liquidations.cascade.exhaustion_bias",
                nested_value(
                    cascade,
                    "exhaustion_bias",
                    default=value_for("exhaustion_bias", default=0.0),
                ),
                section="cascade",
            )
            add(
                "liquidations.cascade.total_notional_usd",
                nested_value(
                    cascade,
                    "total_notional_usd",
                    "notional_usd",
                    default=value_for(
                        "total_notional_usd",
                        "notional_usd",
                        default=0.0,
                    ),
                ),
                section="cascade",
            )
            add(
                "liquidations.cascade.event_count",
                nested_value(
                    cascade,
                    "event_count",
                    default=value_for("event_count", default=0),
                ),
                section="cascade",
            )

        if section_detected(exhaustion):
            add("liquidations.exhaustion", exhaustion, section="exhaustion")
            add(
                "liquidations.exhaustion.confidence",
                nested_value(
                    exhaustion,
                    "confidence",
                    default=value_for(
                        "exhaustion_confidence",
                        "confidence",
                        default=0.0,
                    ),
                ),
                section="exhaustion",
            )
            add(
                "liquidations.exhaustion.exhaustion_bias",
                nested_value(
                    exhaustion,
                    "exhaustion_bias",
                    default=value_for("exhaustion_bias", default=0.0),
                ),
                section="exhaustion",
            )
            add(
                "liquidations.exhaustion.bias_delta",
                nested_value(
                    exhaustion,
                    "bias_delta",
                    default=value_for("bias_delta", default=0.0),
                ),
                section="exhaustion",
            )
            add(
                "liquidations.exhaustion.confirmed",
                nested_value(
                    exhaustion,
                    "confirmed",
                    "is_confirmed",
                    "status",
                    default=value_for(
                        "exhaustion_confirmed",
                        "confirmed",
                        default=False,
                    ),
                ),
                section="exhaustion",
            )

        if section_detected(squeeze):
            add("liquidations.squeeze", squeeze, section="squeeze")
            add(
                "liquidations.squeeze.confirmed",
                nested_value(
                    squeeze,
                    "confirmed",
                    "is_confirmed",
                    "status",
                    default=value_for(
                        "squeeze_confirmed",
                        "confirmed",
                        default=False,
                    ),
                ),
                section="squeeze",
            )
            add(
                "liquidations.squeeze.score",
                nested_value(
                    squeeze,
                    "score",
                    "squeeze_score",
                    default=value_for("squeeze_score", "score", default=0.0),
                ),
                section="squeeze",
            )
            add(
                "liquidations.squeeze.direction",
                nested_value(
                    squeeze,
                    "direction",
                    "squeeze_direction",
                    "side",
                    default=value_for(
                        "squeeze_direction",
                        "direction",
                        "side",
                        default=None,
                    ),
                ),
                section="squeeze",
            )

        if section_detected(signal):
            add("liquidations.signal", signal, section="signal")
            add(
                "liquidations.signal.type",
                nested_value(
                    signal,
                    "type",
                    "signal_type",
                    "setup_type",
                    default=value_for("signal_type", "setup_type", default=None),
                ),
                section="signal",
            )
            add(
                "liquidations.signal.score",
                nested_value(
                    signal,
                    "score",
                    default=value_for("signal_score", "score", default=0.0),
                ),
                section="signal",
            )
            add(
                "liquidations.signal.confidence",
                nested_value(
                    signal,
                    "confidence",
                    default=value_for(
                        "signal_confidence",
                        "confidence",
                        default=0.0,
                    ),
                ),
                section="signal",
            )
            add(
                "liquidations.signal.side",
                nested_value(
                    signal,
                    "side",
                    "direction",
                    "bias",
                    default=value_for("signal_side", "side", "direction", default=None),
                ),
                section="signal",
            )

        return result

    def _build_orderflow_contract_features(
            self,
            *,
            symbol: str,
            payload: dict[str, Any],
            timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        result: list[FeatureSnapshot] = []

        confidence = payload.get("confidence", 0.0)
        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def as_mapping(value: Any) -> dict[str, Any] | None:
            if isinstance(value, dict):
                return value

            to_dict = getattr(value, "to_dict", None)
            if callable(to_dict):
                converted = to_dict()
                if isinstance(converted, dict):
                    return converted

            return None

        def get_item(value: Any, key: str, default: Any = None) -> Any:
            mapping = as_mapping(value)
            if mapping is not None:
                return mapping.get(key, default)
            return getattr(value, key, default)

        def mapping_or_object_for(*keys: str) -> Any | None:
            for key in keys:
                if key in payload and payload[key] is not None:
                    return payload[key]
                if key in feature_map and feature_map[key] is not None:
                    return feature_map[key]
            return None

        def nested_value(value: Any, *keys: str, default: Any = None) -> Any:
            for key in keys:
                item = get_item(value, key, None)
                if item is not None:
                    return item

            metadata = get_item(value, "metadata", None)
            if isinstance(metadata, dict):
                for key in keys:
                    if key in metadata:
                        return metadata[key]

            return default

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]

            for container in (
                    composite,
                    cvd,
                    volume_delta,
                    aggressive_trades,
                    orderbook_imbalance,
                    signal,
            ):
                value = nested_value(container, *keys, default=None)
                if value is not None:
                    return value

            return default

        def add(name: str, value: Any, *, section: str | None = None) -> None:
            if value is None:
                return

            result.append(
                self._snapshot_from_raw_value(
                    source=FeatureSource.ORDERFLOW,
                    symbol=symbol,
                    name=name,
                    value=value,
                    timestamp=timestamp,
                    confidence=confidence,
                    metadata={
                        "origin": "contract_feature",
                        "contract": "orderflow",
                        **({"section": section} if section else {}),
                    },
                )
            )

        composite = mapping_or_object_for(
            "composite",
            "snapshot",
            "orderflow_snapshot",
            "composite_snapshot",
            "orderflow_composite",
        )
        cvd = mapping_or_object_for(
            "cvd",
            "cvd_snapshot",
            "cvd_metrics",
            "cumulative_volume_delta",
        )
        volume_delta = mapping_or_object_for(
            "volume_delta",
            "volume_delta_snapshot",
            "delta",
            "delta_metrics",
        )
        aggressive_trades = mapping_or_object_for(
            "aggressive_trades",
            "aggressive",
            "aggressive_flow",
            "aggressive_trades_snapshot",
        )
        orderbook_imbalance = mapping_or_object_for(
            "orderbook_imbalance",
            "orderbook",
            "orderbook_snapshot",
            "imbalance",
        )
        signal = mapping_or_object_for(
            "signal",
            "orderflow_signal",
            "analytics_signal",
            "setup",
        )

        for container_key in (
                "result",
                "payload",
                "data",
                "analysis",
                "orderflow_analysis",
        ):
            container = mapping_or_object_for(container_key)
            container_map = as_mapping(container)
            if container_map is None:
                continue

            composite = (
                    composite
                    or container_map.get("composite")
                    or container_map.get("snapshot")
                    or container_map.get("orderflow_snapshot")
                    or container_map.get("composite_snapshot")
            )
            cvd = (
                    cvd
                    or container_map.get("cvd")
                    or container_map.get("cvd_snapshot")
                    or container_map.get("cvd_metrics")
            )
            volume_delta = (
                    volume_delta
                    or container_map.get("volume_delta")
                    or container_map.get("delta")
                    or container_map.get("delta_metrics")
            )
            aggressive_trades = (
                    aggressive_trades
                    or container_map.get("aggressive_trades")
                    or container_map.get("aggressive")
                    or container_map.get("aggressive_flow")
            )
            orderbook_imbalance = (
                    orderbook_imbalance
                    or container_map.get("orderbook_imbalance")
                    or container_map.get("orderbook")
                    or container_map.get("imbalance")
            )
            signal = (
                    signal
                    or container_map.get("signal")
                    or container_map.get("orderflow_signal")
                    or container_map.get("analytics_signal")
                    or container_map.get("setup")
            )

        if composite is not None:
            cvd = cvd or nested_value(composite, "cvd", default=None)
            volume_delta = volume_delta or nested_value(
                composite,
                "volume_delta",
                default=None,
            )
            aggressive_trades = aggressive_trades or nested_value(
                composite,
                "aggressive_trades",
                default=None,
            )
            orderbook_imbalance = orderbook_imbalance or nested_value(
                composite,
                "orderbook_imbalance",
                default=None,
            )

        if cvd is None:
            flat_cvd = {
                key: value_for(key, default=None)
                for key in (
                    "cvd_value",
                    "value",
                    "cvd_delta_ratio",
                    "delta_ratio",
                    "cvd_change_pct",
                    "cvd_slope",
                    "price_change_pct",
                    "cvd_buy_ratio",
                    "cvd_sell_ratio",
                )
                if value_for(key, default=None) is not None
            }
            if flat_cvd:
                cvd = flat_cvd

        if volume_delta is None:
            flat_volume_delta = {
                key: value_for(key, default=None)
                for key in (
                    "volume_delta",
                    "notional_delta",
                    "volume_delta_ratio",
                    "delta_ratio",
                    "cumulative_volume_delta",
                    "cumulative_notional_delta",
                    "buy_volume",
                    "sell_volume",
                    "buy_notional",
                    "sell_notional",
                )
                if value_for(key, default=None) is not None
            }
            if flat_volume_delta:
                volume_delta = flat_volume_delta

        if aggressive_trades is None:
            flat_aggressive = {
                key: value_for(key, default=None)
                for key in (
                    "aggressive_buy_ratio",
                    "aggressive_sell_ratio",
                    "aggressive_burst_score",
                    "aggressive_net_volume_delta",
                    "aggressive_net_notional_delta",
                    "large_buy_trades",
                    "large_sell_trades",
                    "aggressive_buy_count",
                    "aggressive_sell_count",
                )
                if value_for(key, default=None) is not None
            }
            if flat_aggressive:
                aggressive_trades = flat_aggressive

        if orderbook_imbalance is None:
            flat_orderbook = {
                key: value_for(key, default=None)
                for key in (
                    "orderbook_imbalance_ratio",
                    "orderbook_imbalance_diff",
                    "ratio",
                    "diff",
                    "bid_volume",
                    "ask_volume",
                    "best_bid",
                    "best_ask",
                    "spread",
                    "mid_price",
                    "depth_levels_used",
                )
                if value_for(key, default=None) is not None
            }
            if flat_orderbook:
                orderbook_imbalance = flat_orderbook

        if composite is None and any(
                item is not None
                for item in (cvd, volume_delta, aggressive_trades, orderbook_imbalance)
        ):
            composite = {
                "cvd": cvd or {},
                "volume_delta": volume_delta or {},
                "aggressive_trades": aggressive_trades or {},
                "orderbook_imbalance": orderbook_imbalance or {},
            }

            for key in (
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "timestamp",
                    "last_price",
                    "price",
                    "price_change",
                    "price_change_pct",
                    "window_seconds",
                    "trades_count",
                    "total_volume",
                    "total_notional",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    composite[key] = value

        if composite is not None:
            add("orderflow.composite", composite, section="composite")

        if cvd is not None:
            add("orderflow.cvd", cvd, section="cvd")
            add(
                "orderflow.cvd.value",
                nested_value(
                    cvd,
                    "cvd_value",
                    "value",
                    default=value_for("cvd_value", "value", default=None),
                ),
                section="cvd",
            )
            add(
                "orderflow.cvd.delta_ratio",
                nested_value(
                    cvd,
                    "cvd_delta_ratio",
                    "delta_ratio",
                    default=value_for("cvd_delta_ratio", "delta_ratio", default=None),
                ),
                section="cvd",
            )
            add(
                "orderflow.cvd.cvd_change_pct",
                nested_value(
                    cvd,
                    "cvd_change_pct",
                    default=value_for("cvd_change_pct", default=None),
                ),
                section="cvd",
            )
            add(
                "orderflow.cvd.cvd_slope",
                nested_value(
                    cvd,
                    "cvd_slope",
                    default=value_for("cvd_slope", default=None),
                ),
                section="cvd",
            )
            add(
                "orderflow.cvd.price_change_pct",
                nested_value(
                    cvd,
                    "price_change_pct",
                    default=value_for("price_change_pct", default=None),
                ),
                section="cvd",
            )

        if volume_delta is not None:
            add("orderflow.volume_delta", volume_delta, section="volume_delta")
            add(
                "orderflow.volume_delta.volume_delta",
                nested_value(
                    volume_delta,
                    "volume_delta",
                    default=value_for("volume_delta", default=None),
                ),
                section="volume_delta",
            )
            add(
                "orderflow.volume_delta.delta_ratio",
                nested_value(
                    volume_delta,
                    "volume_delta_ratio",
                    "delta_ratio",
                    default=value_for(
                        "volume_delta_ratio",
                        "delta_ratio",
                        default=None,
                    ),
                ),
                section="volume_delta",
            )
            add(
                "orderflow.volume_delta.cumulative_volume_delta",
                nested_value(
                    volume_delta,
                    "cumulative_volume_delta",
                    default=value_for("cumulative_volume_delta", default=None),
                ),
                section="volume_delta",
            )
            add(
                "orderflow.volume_delta.notional_delta",
                nested_value(
                    volume_delta,
                    "notional_delta",
                    default=value_for("notional_delta", default=None),
                ),
                section="volume_delta",
            )
            add(
                "orderflow.volume_delta.cumulative_notional_delta",
                nested_value(
                    volume_delta,
                    "cumulative_notional_delta",
                    default=value_for("cumulative_notional_delta", default=None),
                ),
                section="volume_delta",
            )

        if aggressive_trades is not None:
            add("orderflow.aggressive_trades", aggressive_trades, section="aggressive")
            add(
                "orderflow.aggressive_trades.buy_ratio",
                nested_value(
                    aggressive_trades,
                    "aggressive_buy_ratio",
                    "buy_ratio",
                    default=value_for("aggressive_buy_ratio", "buy_ratio", default=None),
                ),
                section="aggressive",
            )
            add(
                "orderflow.aggressive_trades.sell_ratio",
                nested_value(
                    aggressive_trades,
                    "aggressive_sell_ratio",
                    "sell_ratio",
                    default=value_for("aggressive_sell_ratio", "sell_ratio", default=None),
                ),
                section="aggressive",
            )
            add(
                "orderflow.aggressive_trades.burst_score",
                nested_value(
                    aggressive_trades,
                    "aggressive_burst_score",
                    "burst_score",
                    default=value_for("aggressive_burst_score", "burst_score", default=None),
                ),
                section="aggressive",
            )
            add(
                "orderflow.aggressive_trades.net_volume_delta",
                nested_value(
                    aggressive_trades,
                    "aggressive_net_volume_delta",
                    "net_volume_delta",
                    default=value_for(
                        "aggressive_net_volume_delta",
                        "net_volume_delta",
                        default=None,
                    ),
                ),
                section="aggressive",
            )
            add(
                "orderflow.aggressive_trades.net_notional_delta",
                nested_value(
                    aggressive_trades,
                    "aggressive_net_notional_delta",
                    "net_notional_delta",
                    default=value_for(
                        "aggressive_net_notional_delta",
                        "net_notional_delta",
                        default=None,
                    ),
                ),
                section="aggressive",
            )
            add(
                "orderflow.aggressive_trades.large_buy_trades",
                nested_value(
                    aggressive_trades,
                    "large_buy_trades",
                    default=value_for("large_buy_trades", default=None),
                ),
                section="aggressive",
            )
            add(
                "orderflow.aggressive_trades.large_sell_trades",
                nested_value(
                    aggressive_trades,
                    "large_sell_trades",
                    default=value_for("large_sell_trades", default=None),
                ),
                section="aggressive",
            )

        if orderbook_imbalance is not None:
            add(
                "orderflow.orderbook_imbalance",
                orderbook_imbalance,
                section="orderbook",
            )
            add(
                "orderflow.orderbook_imbalance.ratio",
                nested_value(
                    orderbook_imbalance,
                    "orderbook_imbalance_ratio",
                    "ratio",
                    default=value_for(
                        "orderbook_imbalance_ratio",
                        "ratio",
                        default=None,
                    ),
                ),
                section="orderbook",
            )
            add(
                "orderflow.orderbook_imbalance.diff",
                nested_value(
                    orderbook_imbalance,
                    "orderbook_imbalance_diff",
                    "diff",
                    default=value_for(
                        "orderbook_imbalance_diff",
                        "diff",
                        default=None,
                    ),
                ),
                section="orderbook",
            )

        add(
            "orderflow.trades_count",
            value_for("trades_count", default=None),
            section="composite",
        )
        add(
            "orderflow.total_volume",
            value_for("total_volume", default=None),
            section="composite",
        )
        add(
            "orderflow.total_notional",
            value_for("total_notional", default=None),
            section="composite",
        )
        add(
            "orderflow.last_price",
            value_for("last_price", "price", default=None),
            section="composite",
        )
        add(
            "orderflow.price_change_pct",
            value_for("price_change_pct", default=None),
            section="composite",
        )

        if signal is not None:
            add("orderflow.signal", signal, section="signal")
            add(
                "orderflow.signal.type",
                nested_value(
                    signal,
                    "type",
                    "signal_type",
                    "setup_type",
                    default=value_for("signal_type", "setup_type", default=None),
                ),
                section="signal",
            )
            add(
                "orderflow.signal.score",
                nested_value(
                    signal,
                    "score",
                    default=value_for("signal_score", "score", default=None),
                ),
                section="signal",
            )
            add(
                "orderflow.signal.confidence",
                nested_value(
                    signal,
                    "confidence",
                    default=value_for("signal_confidence", "confidence", default=None),
                ),
                section="signal",
            )
            add(
                "orderflow.signal.side",
                nested_value(
                    signal,
                    "side",
                    "direction",
                    "bias",
                    default=value_for("signal_side", "side", "direction", default=None),
                ),
                section="signal",
            )

        return result

    def _build_open_interest_contract_features(
            self,
            *,
            symbol: str,
            payload: dict[str, Any],
            timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        """
        Build stable open-interest contract features.

        Setup-specific features are emitted only when the corresponding section
        is actually present/detected. This prevents the registry from routing
        OI divergence/anomaly/capitulation strategies on generic OI updates.
        """
        result: list[FeatureSnapshot] = []
        confidence = payload.get("confidence", 0.0)

        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        topic = (
            str(
                payload.get("event_name")
                or payload.get("topic")
                or payload.get("source_topic")
                or ""
            )
            .strip()
            .lower()
        )

        def mapping_for(*keys: str) -> dict[str, Any] | None:
            for key in keys:
                value = payload.get(key)
                if isinstance(value, dict):
                    return dict(value)

                value = feature_map.get(key)
                if isinstance(value, dict):
                    return dict(value)

            return None

        analysis = mapping_for(
            "analysis",
            "oi_analysis",
            "open_interest_analysis",
            "result",
        )
        features = mapping_for(
            "features",
            "oi_features",
            "open_interest_features",
        )
        snapshot = mapping_for(
            "snapshot",
            "oi_snapshot",
            "open_interest_snapshot",
        )
        market_context = mapping_for(
            "context",
            "market_context",
            "oi_context",
            "open_interest_context",
        )
        regime = mapping_for(
            "regime",
            "regime_result",
            "oi_regime",
            "open_interest_regime",
            "new_regime",
        )
        anomaly = mapping_for(
            "anomaly",
            "anomaly_result",
            "oi_anomaly",
            "open_interest_anomaly",
        )
        divergence = mapping_for(
            "divergence",
            "divergence_result",
            "oi_divergence",
            "open_interest_divergence",
        )

        if analysis:
            if not features:
                value = analysis.get("features")
                if isinstance(value, dict):
                    features = dict(value)

            if not snapshot:
                value = analysis.get("snapshot")
                if isinstance(value, dict):
                    snapshot = dict(value)

            if not market_context:
                for key in ("context", "market_context", "oi_context"):
                    value = analysis.get(key)
                    if isinstance(value, dict):
                        market_context = dict(value)
                        break

            if not regime:
                for key in ("regime", "regime_result", "oi_regime"):
                    value = analysis.get(key)
                    if isinstance(value, dict):
                        regime = dict(value)
                        break

            if not anomaly:
                for key in ("anomaly", "anomaly_result", "oi_anomaly"):
                    value = analysis.get(key)
                    if isinstance(value, dict):
                        anomaly = dict(value)
                        break

            if not divergence:
                for key in ("divergence", "divergence_result", "oi_divergence"):
                    value = analysis.get(key)
                    if isinstance(value, dict):
                        divergence = dict(value)
                        break

        def to_bool(value: Any, default: bool = False) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"1", "true", "yes", "y", "on", "detected", "active", "confirmed"}:
                    return True
                if normalized in {"0", "false", "no", "n", "off", "none", "not_detected"}:
                    return False
            if isinstance(value, (int, float)):
                return bool(value)
            return default

        def section_detected(mapping: dict[str, Any] | None) -> bool:
            if not mapping:
                return False

            detected = mapping.get("detected", mapping.get("is_detected", None))
            if detected is None:
                return True

            return to_bool(detected, default=False)

        def has_any(*keys: str) -> bool:
            containers: tuple[dict[str, Any], ...] = tuple(
                item for item in (payload, feature_map, features or {}, regime or {}, analysis or {})
                if isinstance(item, dict)
            )
            return any(key in container for key in keys for container in containers)

        def value_for(*keys: str, default: Any = None) -> Any:
            containers: tuple[dict[str, Any], ...] = tuple(
                item for item in (
                    payload,
                    feature_map,
                    features or {},
                    regime or {},
                    anomaly or {},
                    divergence or {},
                    analysis or {},
                )
                if isinstance(item, dict)
            )
            for key in keys:
                for container in containers:
                    if key in container:
                        return container[key]
            return default

        def add(name: str, value: Any = True, *, section: str | None = None) -> None:
            snapshot_obj = self._snapshot_from_raw_value(
                source=FeatureSource.OPEN_INTEREST,
                symbol=symbol,
                name=name,
                value=value,
                timestamp=timestamp,
                confidence=confidence,
                metadata={
                    "origin": "contract_feature",
                    "contract": "open_interest",
                    **({"section": section} if section else {}),
                },
            )
            result.append(snapshot_obj)

        if analysis:
            add("open_interest.analysis", analysis, section="analysis")

        if snapshot:
            add("open_interest.snapshot", snapshot, section="snapshot")

        if market_context:
            add("open_interest.context", market_context, section="market_context")

        if features is None and has_any(
                "oi",
                "open_interest",
                "open_interest_value",
                "oi_delta",
                "oi_delta_pct",
                "oi_direction",
                "oi_acceleration",
                "price_delta_pct",
                "volume_ratio",
                "oi_zscore",
                "oi_pressure_score",
                "funding_rate",
                "liquidation_pressure",
                "liquidation_imbalance",
                "aggressive_flow_imbalance",
                "oi_price_efficiency",
        ):
            features = {
                "oi": value_for("oi", "open_interest", default=None),
                "open_interest": value_for("open_interest", "oi", default=None),
                "open_interest_value": value_for("open_interest_value", default=None),
                "oi_delta": value_for("oi_delta", default=None),
                "oi_delta_pct": value_for("oi_delta_pct", default=None),
                "oi_direction": value_for("oi_direction", default=None),
                "oi_acceleration": value_for("oi_acceleration", default=None),
                "price_delta_pct": value_for("price_delta_pct", default=None),
                "volume_ratio": value_for("volume_ratio", default=None),
                "oi_zscore": value_for("oi_zscore", default=None),
                "oi_pressure_score": value_for("oi_pressure_score", default=None),
                "funding_rate": value_for("funding_rate", default=None),
                "liquidation_pressure": value_for(
                    "liquidation_pressure",
                    "liquidation_imbalance",
                    default=None,
                ),
                "liquidation_imbalance": value_for(
                    "liquidation_imbalance",
                    "liquidation_pressure",
                    default=None,
                ),
                "aggressive_flow_imbalance": value_for(
                    "aggressive_flow_imbalance",
                    default=None,
                ),
                "oi_price_efficiency": value_for(
                    "oi_price_efficiency",
                    default=None,
                ),
            }
            features = {key: value for key, value in features.items() if value is not None}

        if features:
            add("open_interest.features", features, section="features")
            for feature_name, aliases in {
                "open_interest.features.oi_delta_pct": ("oi_delta_pct",),
                "open_interest.features.price_delta_pct": ("price_delta_pct",),
                "open_interest.features.oi_pressure_score": ("oi_pressure_score",),
                "open_interest.features.aggressive_flow_imbalance": ("aggressive_flow_imbalance",),
                "open_interest.features.funding_rate": ("funding_rate",),
                "open_interest.features.liquidation_pressure": ("liquidation_pressure", "liquidation_imbalance"),
            }.items():
                value = value_for(*aliases, default=None)
                if value is not None:
                    add(feature_name, value, section="features")

        if regime is None:
            regime_value = value_for(
                "regime",
                "oi_regime",
                "market_regime",
                "new_regime",
                "regime_type",
                default=None,
            )
            if regime_value is not None:
                regime = {
                    "regime": regime_value,
                    "confidence": value_for("regime_confidence", "confidence", default=0.0),
                    "score": value_for("regime_score", "score", default=0.0),
                    "reasons": value_for("regime_reasons", "reasons", default=[]),
                }

        if regime:
            add("open_interest.regime", regime, section="regime")
            add(
                "open_interest.regime.type",
                value_for("regime", "oi_regime", "market_regime", "new_regime", "regime_type", default=regime.get("regime") or regime.get("type")),
                section="regime",
            )
            add(
                "open_interest.regime.confidence",
                value_for("regime_confidence", "confidence", default=regime.get("confidence", 0.0)),
                section="regime",
            )
            add(
                "open_interest.regime.score",
                value_for("regime_score", "score", default=regime.get("score", 0.0)),
                section="regime",
            )

        if divergence is None:
            divergence_type = value_for(
                "divergence_type",
                "price_oi_divergence",
                default=None,
            )
            divergence_detected = value_for(
                "divergence_detected",
                "is_divergence",
                default=None,
            )
            is_divergence_topic = (
                ".divergence" in topic
                or topic.endswith("divergence")
                or topic.endswith("oi.divergence")
                or topic.endswith("open_interest.divergence")
            )
            if divergence_type is None and is_divergence_topic:
                raw_side = str(value_for("side", "direction", "bias", default="")).strip().lower()
                if raw_side in {"long", "bullish", "buy", "up"}:
                    divergence_type = "bullish"
                elif raw_side in {"short", "bearish", "sell", "down"}:
                    divergence_type = "bearish"
                else:
                    divergence_type = "bullish"
            if divergence_type is not None or to_bool(divergence_detected, default=False) or is_divergence_topic:
                divergence = {
                    "detected": to_bool(divergence_detected, default=True),
                    "divergence_type": divergence_type,
                    "price_oi_divergence": value_for("price_oi_divergence", default=divergence_type),
                    "side": value_for("side", "direction", "bias", default=None),
                    "direction": value_for("direction", "side", "bias", default=None),
                    "confidence": value_for("divergence_confidence", "confidence", default=0.0),
                    "score": value_for("divergence_score", "score", default=0.0),
                    "window_size": value_for("divergence_window_size", "window_size", default=None),
                    "reasons": value_for("divergence_reasons", "reasons", default=[]),
                }

        if section_detected(divergence):
            add("open_interest.divergence", divergence, section="divergence")
            add(
                "open_interest.divergence.detected",
                value_for("divergence_detected", "is_divergence", "detected", default=divergence.get("detected", True)),
                section="divergence",
            )
            add(
                "open_interest.divergence.type",
                value_for("divergence_type", "price_oi_divergence", default=divergence.get("divergence_type") or divergence.get("type")),
                section="divergence",
            )
            add(
                "open_interest.divergence.confidence",
                value_for("divergence_confidence", "confidence", default=divergence.get("confidence", 0.0)),
                section="divergence",
            )
            add(
                "open_interest.divergence.score",
                value_for("divergence_score", "score", default=divergence.get("score", 0.0)),
                section="divergence",
            )
            if value_for("divergence_window_size", "window_size", default=divergence.get("window_size")) is not None:
                add(
                    "open_interest.divergence.window_size",
                    value_for("divergence_window_size", "window_size", default=divergence.get("window_size")),
                    section="divergence",
                )

        if anomaly is None:
            anomaly_type = value_for("anomaly_type", default=None)
            anomaly_detected = value_for(
                "anomaly_detected",
                "is_anomaly",
                default=None,
            )
            capitulation = value_for("capitulation", default=None)
            squeeze_setup = value_for("squeeze_setup", default=None)
            if (
                anomaly_type is not None
                or to_bool(anomaly_detected, default=False)
                or to_bool(capitulation, default=False)
                or to_bool(squeeze_setup, default=False)
            ):
                anomaly = {
                    "detected": to_bool(
                        anomaly_detected,
                        default=anomaly_type is not None
                        or to_bool(capitulation, default=False)
                        or to_bool(squeeze_setup, default=False),
                    ),
                    "anomaly_type": anomaly_type,
                    "confidence": value_for("anomaly_confidence", "confidence", default=0.0),
                    "score": value_for("anomaly_score", "score", default=0.0),
                    "strength": value_for("anomaly_strength", "strength", default=value_for("score", default=0.0)),
                    "capitulation": capitulation,
                    "capitulation_score": value_for("capitulation_score", default=None),
                    "squeeze_setup": squeeze_setup,
                    "squeeze_score": value_for("squeeze_score", default=None),
                    "liquidation_imbalance": value_for("liquidation_imbalance", "liquidation_pressure", default=None),
                    "reasons": value_for("anomaly_reasons", "reasons", default=[]),
                }

        if section_detected(anomaly):
            add("open_interest.anomaly", anomaly, section="anomaly")
            add(
                "open_interest.anomaly.detected",
                value_for("anomaly_detected", "is_anomaly", "detected", default=anomaly.get("detected", True)),
                section="anomaly",
            )
            add(
                "open_interest.anomaly.type",
                value_for("anomaly_type", default=anomaly.get("anomaly_type") or anomaly.get("type")),
                section="anomaly",
            )
            add(
                "open_interest.anomaly.confidence",
                value_for("anomaly_confidence", "confidence", default=anomaly.get("confidence", 0.0)),
                section="anomaly",
            )
            add(
                "open_interest.anomaly.score",
                value_for("anomaly_score", "score", default=anomaly.get("score", 0.0)),
                section="anomaly",
            )
            add(
                "open_interest.anomaly.strength",
                value_for("anomaly_strength", "strength", default=anomaly.get("strength", anomaly.get("score", 0.0))),
                section="anomaly",
            )

        return result

    def _build_funding_contract_features(
            self,
            *,
            symbol: str,
            payload: dict[str, Any],
            timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        result: list[FeatureSnapshot] = []

        confidence = payload.get("confidence", 0.0)
        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def mapping_for(*keys: str) -> dict[str, Any] | None:
            for key in keys:
                value = payload.get(key)
                if isinstance(value, dict):
                    return dict(value)

                value = feature_map.get(key)
                if isinstance(value, dict):
                    return dict(value)

            return None

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]
            return default

        def nested_value(
                mapping: dict[str, Any] | None,
                *keys: str,
                default: Any = None,
        ) -> Any:
            if not isinstance(mapping, dict):
                return default

            for key in keys:
                if key in mapping:
                    return mapping[key]

            metadata = mapping.get("metadata")
            if isinstance(metadata, dict):
                for key in keys:
                    if key in metadata:
                        return metadata[key]

            return default

        def to_bool(value: Any, default: bool = False) -> bool:
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
                    "detected",
                    "active",
                    "confirmed",
                    "triggered",
                }:
                    return True
                if normalized in {
                    "0",
                    "false",
                    "no",
                    "n",
                    "off",
                    "none",
                    "not_detected",
                    "inactive",
                }:
                    return False

            if isinstance(value, (int, float)):
                return bool(value)

            return default

        def section_detected(section: dict[str, Any] | None) -> bool:
            if not section:
                return False

            detected = section.get(
                "detected",
                section.get(
                    "is_detected",
                    section.get(
                        "active",
                        section.get("confirmed", None),
                    ),
                ),
            )

            if detected is None:
                return True

            return to_bool(detected, default=False)

        def add(name: str, value: Any, *, section: str | None = None) -> None:
            result.append(
                self._snapshot_from_raw_value(
                    source=FeatureSource.FUNDING,
                    symbol=symbol,
                    name=name,
                    value=value,
                    timestamp=timestamp,
                    confidence=confidence,
                    metadata={
                        "origin": "contract_feature",
                        "contract": "funding",
                        **({"section": section} if section else {}),
                    },
                )
            )

        analysis = mapping_for(
            "analysis",
            "result",
            "funding_analysis",
            "funding_result",
        )

        snapshot = mapping_for("snapshot", "funding_snapshot")
        statistics = mapping_for("statistics", "stats", "funding_statistics")
        regime = mapping_for("regime", "regime_state", "funding_regime")
        pressure = mapping_for("pressure", "pressure_state", "funding_pressure")
        extreme = mapping_for("extreme", "extreme_event", "funding_extreme")
        divergence = mapping_for(
            "divergence",
            "divergence_event",
            "funding_divergence",
        )
        flip = mapping_for("flip", "flip_event", "funding_flip")
        signal = mapping_for("signal", "funding_signal", "setup")

        if analysis is not None:
            nested = nested_value

            snapshot = snapshot or nested(
                analysis,
                "snapshot",
                "funding_snapshot",
            )
            statistics = statistics or nested(
                analysis,
                "statistics",
                "stats",
                "funding_statistics",
            )
            regime = regime or nested(
                analysis,
                "regime",
                "regime_state",
                "funding_regime",
            )
            pressure = pressure or nested(
                analysis,
                "pressure",
                "pressure_state",
                "funding_pressure",
            )
            extreme = extreme or nested(
                analysis,
                "extreme",
                "extreme_event",
                "funding_extreme",
            )
            divergence = divergence or nested(
                analysis,
                "divergence",
                "divergence_event",
                "funding_divergence",
            )
            flip = flip or nested(
                analysis,
                "flip",
                "flip_event",
                "funding_flip",
            )
            signal = signal or nested(
                analysis,
                "signal",
                "funding_signal",
                "setup",
            )

        if not isinstance(snapshot, dict):
            snapshot = {}
            for key in (
                    "funding_rate",
                    "current_rate",
                    "next_funding_rate",
                    "predicted_rate",
                    "annualized_rate",
                    "premium_index",
                    "mark_price",
                    "index_price",
                    "next_funding_time",
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "timestamp",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    snapshot[key] = value

        if snapshot:
            add("funding.snapshot", snapshot, section="snapshot")

        if statistics:
            add("funding.statistics", statistics, section="statistics")

        if regime:
            add("funding.regime", regime, section="regime")
            add(
                "funding.regime.confidence",
                nested_value(
                    regime,
                    "confidence",
                    default=value_for("regime_confidence", default=0.0),
                ),
                section="regime",
            )

        if pressure:
            add("funding.pressure", pressure, section="pressure")
            add(
                "funding.pressure.score",
                nested_value(
                    pressure,
                    "score",
                    "pressure_score",
                    default=value_for("pressure_score", default=0.0),
                ),
                section="pressure",
            )
            add(
                "funding.pressure.level",
                nested_value(
                    pressure,
                    "level",
                    "pressure_level",
                    default=value_for("pressure_level", default=None),
                ),
                section="pressure",
            )
            add(
                "funding.pressure.direction",
                nested_value(
                    pressure,
                    "direction",
                    "bias",
                    default=value_for("pressure_direction", default=None),
                ),
                section="pressure",
            )

        if section_detected(extreme):
            add("funding.extreme", extreme, section="extreme")
            add(
                "funding.extreme.type",
                nested_value(
                    extreme,
                    "type",
                    "extreme_type",
                    default=value_for("extreme_type", default=None),
                ),
                section="extreme",
            )
            add(
                "funding.extreme.severity",
                nested_value(
                    extreme,
                    "severity",
                    "score",
                    default=value_for("extreme_severity", default=0.0),
                ),
                section="extreme",
            )
            add(
                "funding.extreme.mean_reversion_probability",
                nested_value(
                    extreme,
                    "mean_reversion_probability",
                    "reversion_probability",
                    default=value_for("mean_reversion_probability", default=0.0),
                ),
                section="extreme",
            )
            add(
                "funding.extreme.squeeze_probability",
                nested_value(
                    extreme,
                    "squeeze_probability",
                    default=value_for("squeeze_probability", default=0.0),
                ),
                section="extreme",
            )

        if section_detected(divergence):
            add("funding.divergence", divergence, section="divergence")
            add(
                "funding.divergence.type",
                nested_value(
                    divergence,
                    "type",
                    "divergence_type",
                    default=value_for("divergence_type", default=None),
                ),
                section="divergence",
            )
            add(
                "funding.divergence.confidence",
                nested_value(
                    divergence,
                    "confidence",
                    default=value_for("divergence_confidence", default=0.0),
                ),
                section="divergence",
            )
            add(
                "funding.divergence.score",
                nested_value(
                    divergence,
                    "score",
                    default=value_for("divergence_score", default=0.0),
                ),
                section="divergence",
            )

        if section_detected(flip):
            add("funding.flip", flip, section="flip")
            add(
                "funding.flip.type",
                nested_value(
                    flip,
                    "type",
                    "flip_type",
                    default=value_for("flip_type", default=None),
                ),
                section="flip",
            )
            add(
                "funding.flip.confidence",
                nested_value(
                    flip,
                    "confidence",
                    default=value_for("flip_confidence", default=0.0),
                ),
                section="flip",
            )

        if section_detected(signal):
            add("funding.signal", signal, section="signal")
            add(
                "funding.signal.type",
                nested_value(
                    signal,
                    "type",
                    "signal_type",
                    default=value_for("signal_type", default=None),
                ),
                section="signal",
            )
            add(
                "funding.signal.score",
                nested_value(
                    signal,
                    "score",
                    default=value_for("signal_score", "score", default=0.0),
                ),
                section="signal",
            )
            add(
                "funding.signal.confidence",
                nested_value(
                    signal,
                    "confidence",
                    default=value_for(
                        "signal_confidence",
                        "confidence",
                        default=0.0,
                    ),
                ),
                section="signal",
            )
            add(
                "funding.signal.bias",
                nested_value(
                    signal,
                    "bias",
                    "direction",
                    default=value_for("bias", "direction", default=None),
                ),
                section="signal",
            )

        return result

    def _extract_features(
            self,
            *,
            source: FeatureSource,
            symbol: str,
            payload: dict[str, Any],
            timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        result: dict[str, FeatureSnapshot] = {}

        for snapshot in self._extract_explicit_features(
                source=source,
                symbol=symbol,
                payload=payload,
                timestamp=timestamp,
        ):
            result[snapshot.name] = snapshot

        for snapshot in self._extract_feature_map(
                source=source,
                symbol=symbol,
                payload=payload,
                timestamp=timestamp,
        ):
            result.setdefault(snapshot.name, snapshot)

        for snapshot in self._extract_nested_features(
                source=source,
                symbol=symbol,
                payload=payload,
                timestamp=timestamp,
        ):
            result.setdefault(snapshot.name, snapshot)

        for snapshot in self._build_implicit_features(
                source=source,
                symbol=symbol,
                payload=payload,
                timestamp=timestamp,
        ):
            result.setdefault(snapshot.name, snapshot)

        for snapshot in self._build_contract_features(
                source=source,
                symbol=symbol,
                payload=payload,
                timestamp=timestamp,
        ):
            result.setdefault(snapshot.name, snapshot)

        return list(result.values())

    def _extract_explicit_features(
            self,
            *,
            source: FeatureSource,
            symbol: str,
            payload: dict[str, Any],
            timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        explicit = payload.get("features")
        if explicit is None:
            return []

        if isinstance(explicit, dict):
            explicit = [
                {"name": name, "value": value}
                for name, value in explicit.items()
            ]

        if not isinstance(explicit, list):
            raise SignalNormalizationError("payload['features'] must be a list or dict")

        result: list[FeatureSnapshot] = []

        for item in explicit:
            if isinstance(item, FeatureSnapshot):
                item.validate()
                result.append(item)
                continue

            if isinstance(item, str):
                feature_name = self._normalize_feature_name(item)

                value = payload.get(item)
                if value is None:
                    value = payload.get(feature_name)

                snapshot = self._snapshot_from_raw_value(
                    source=source,
                    symbol=symbol,
                    name=feature_name,
                    value=True if value is None else value,
                    timestamp=timestamp,
                    confidence=payload.get("confidence", 0.0),
                    metadata={
                        "origin": "features",
                        "feature_declared_only": value is None,
                    },
                )
                result.append(snapshot)
                continue

            if not isinstance(item, dict):
                raise SignalNormalizationError(
                    "each feature item must be a dict, str, or FeatureSnapshot"
                )

            snapshot = self._snapshot_from_feature_item(
                source=source,
                symbol=symbol,
                item=item,
                timestamp=timestamp,
                default_confidence=payload.get("confidence", 0.0),
                metadata={"origin": "features"},
            )
            result.append(snapshot)

        return result

    def _extract_feature_map(
        self,
        *,
        source: FeatureSource,
        symbol: str,
        payload: dict[str, Any],
        timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        feature_map = payload.get("feature_map")
        if feature_map is None:
            return []

        if not isinstance(feature_map, dict):
            raise SignalNormalizationError("payload['feature_map'] must be a dict")

        result: list[FeatureSnapshot] = []
        default_confidence = payload.get("confidence", 0.0)

        for name, value in feature_map.items():
            if not isinstance(name, str) or not name.strip():
                continue

            if isinstance(value, dict):
                item = {"name": name, **value}
            else:
                item = {"name": name, "value": value}

            snapshot = self._snapshot_from_feature_item(
                source=source,
                symbol=symbol,
                item=item,
                timestamp=timestamp,
                default_confidence=default_confidence,
                metadata={"origin": "feature_map"},
            )
            result.append(snapshot)

        return result

    def _extract_nested_features(
        self,
        *,
        source: FeatureSource,
        symbol: str,
        payload: dict[str, Any],
        timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        result: list[FeatureSnapshot] = []
        default_confidence = payload.get("confidence", payload.get("strength", 0.0))

        for container_name in self._FEATURE_CONTAINER_KEYS:
            container = payload.get(container_name)

            if container is None:
                continue
            if container_name in {"features", "feature_map"}:
                continue

            if not isinstance(container, dict):
                continue

            flattened = self._flatten_scalar_dict(container)

            for path, value in flattened.items():
                leaf_name = path.rsplit(".", 1)[-1]
                feature_names = self._feature_name_candidates(
                    container_name=container_name,
                    path=path,
                    leaf_name=leaf_name,
                    payload=payload,
                )

                for feature_name in feature_names:
                    snapshot = self._snapshot_from_raw_value(
                        source=source,
                        symbol=symbol,
                        name=feature_name,
                        value=value,
                        timestamp=timestamp,
                        confidence=default_confidence,
                        metadata={
                            "origin": container_name,
                            "path": path,
                        },
                    )
                    result.append(snapshot)

        return result

    def _build_implicit_features(
        self,
        *,
        source: FeatureSource,
        symbol: str,
        payload: dict[str, Any],
        timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        base_confidence = self._safe_confidence(payload.get("confidence", 0.0))
        result: list[FeatureSnapshot] = []

        for key, value in payload.items():
            if key in self._SCALAR_FEATURE_EXCLUDED_KEYS or key.startswith("_"):
                continue
            if key in self._FEATURE_CONTAINER_KEYS:
                continue

            if self._is_feature_scalar(value):
                snapshot = self._snapshot_from_raw_value(
                    source=source,
                    symbol=symbol,
                    name=key,
                    value=value,
                    timestamp=timestamp,
                    confidence=base_confidence,
                    metadata={"origin": "top_level"},
                )
                result.append(snapshot)

        return result

    def _snapshot_from_feature_item(
        self,
        *,
        source: FeatureSource,
        symbol: str,
        item: dict[str, Any],
        timestamp: datetime,
        default_confidence: Any,
        metadata: dict[str, Any] | None = None,
    ) -> FeatureSnapshot:
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise SignalNormalizationError("feature item must contain non-empty 'name'")

        confidence = self._safe_confidence(item.get("confidence", default_confidence))
        normalized_value = self._safe_normalized_value(
            item.get("normalized_value", item.get("normalized"))
        )

        snapshot = FeatureSnapshot(
            name=self._normalize_feature_name(name),
            value=item.get("value"),
            source=source,
            symbol=symbol,
            timestamp=timestamp,
            confidence=confidence,
            normalized_value=normalized_value,
            freshness_seconds=self._resolve_freshness_seconds(
                feature_name=self._normalize_feature_name(name),
                explicit=item.get("freshness_seconds"),
            ),
            metadata={
                **dict(metadata or {}),
                **dict(item.get("metadata") or {}),
            },
        )
        snapshot.validate()
        return snapshot

    def _snapshot_from_raw_value(
        self,
        *,
        source: FeatureSource,
        symbol: str,
        name: str,
        value: Any,
        timestamp: datetime,
        confidence: Any,
        metadata: dict[str, Any] | None = None,
    ) -> FeatureSnapshot:
        feature_name = self._normalize_feature_name(name)
        snapshot = FeatureSnapshot(
            name=feature_name,
            value=value,
            source=source,
            symbol=symbol,
            timestamp=timestamp,
            confidence=self._safe_confidence(confidence),
            normalized_value=self._infer_normalized_value(value),
            freshness_seconds=self._resolve_freshness_seconds(feature_name, None),
            metadata=dict(metadata or {}),
        )
        snapshot.validate()
        return snapshot

    @classmethod
    def _feature_name_candidates(
        cls,
        *,
        container_name: str,
        path: str,
        leaf_name: str,
        payload: dict[str, Any],
    ) -> list[str]:
        metric = payload.get("metric")
        names: list[str] = []

        # Main lookup should be leaf name: cvd, delta_ratio, imbalance_ratio, etc.
        names.append(leaf_name)

        # Also add metric-prefixed names for disambiguation:
        # cvd.delta_ratio, volume_delta.delta_ratio, etc.
        if isinstance(metric, str) and metric.strip():
            names.append(f"{metric.strip()}.{leaf_name}")

        # Also add container-prefixed names:
        # stats.delta_ratio, context.absorption_score, etc.
        names.append(f"{container_name}.{path}")

        return list(dict.fromkeys(cls._normalize_feature_name(name) for name in names))

    @classmethod
    def _flatten_scalar_dict(
        cls,
        value: dict[str, Any],
        *,
        prefix: str = "",
        max_depth: int = 4,
    ) -> dict[str, Any]:
        if max_depth <= 0:
            return {}

        result: dict[str, Any] = {}

        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                continue
            if key.startswith("_"):
                continue

            normalized_key = cls._normalize_feature_name(key)
            path = f"{prefix}.{normalized_key}" if prefix else normalized_key

            if cls._is_feature_scalar(item):
                result[path] = item
                continue

            if isinstance(item, dict):
                result.update(
                    cls._flatten_scalar_dict(
                        item,
                        prefix=path,
                        max_depth=max_depth - 1,
                    )
                )

        return result

    @staticmethod
    def _is_feature_scalar(value: Any) -> bool:
        return isinstance(value, (int, float, bool, str)) and value is not None

    @staticmethod
    def _normalize_feature_name(value: str) -> str:
        return (
            str(value)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

    def _resolve_freshness_seconds(
        self,
        feature_name: str,
        explicit: Any,
    ) -> float | None:
        if explicit is not None:
            explicit_f = _to_float(explicit)
            if explicit_f is None or explicit_f <= 0:
                raise SignalNormalizationError(
                    f"freshness_seconds must be positive for feature '{feature_name}'"
                )
            return explicit_f

        return float(self.config.get_feature_ttl(feature_name))

    @staticmethod
    def _safe_confidence(value: Any) -> float:
        parsed = _to_float(value, default=0.0)
        return clamp(parsed if parsed is not None else 0.0, 0.0, 1.0)

    @staticmethod
    def _safe_normalized_value(value: Any) -> float | None:
        parsed = _to_float(value)
        if parsed is None:
            return None
        return clamp(parsed, -1.0, 1.0)

    @staticmethod
    def _infer_normalized_value(value: Any) -> float | None:
        if isinstance(value, bool):
            return 1.0 if value else 0.0

        parsed = _to_float(value)
        if parsed is None:
            return None

        if -1.0 <= parsed <= 1.0:
            return parsed

        return None


# =============================================================================
# SignalRouter
# =============================================================================


class SignalRouter(BaseStrategyComponent):
    """
    Selects strategies for analytics events and emits final risk-ready signals.
    """

    component_namespace = "strategy.processor.router"

    def __init__(
        self,
        config: StrategyConfig,
        registry: StrategyRegistry,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, scheduler=scheduler)
        self.registry = registry
        self.routing_config: RoutingConfig = config.routing

    def route(
        self,
        *,
        event_name: str,
        context: StrategyContext,
        source: FeatureSource | None = None,
        changed_features: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RouteDecision:
        if not event_name.strip():
            raise SignalRoutingError("event_name cannot be empty")

        context.validate()

        categories = self._resolve_categories(
            event_name=event_name,
            source=source,
        )

        selected: list[BaseStrategy] = []
        skipped: dict[str, str] = {}

        if hasattr(self.registry, "select"):
            candidates = self.registry.select(
                context=context,
                categories=categories or None,
                changed_features=changed_features or None,
            )
        else:
            candidates = self.registry.list_all()

        for strategy in candidates:
            try:
                if not strategy.should_evaluate(context):
                    skipped[strategy.strategy_name] = "strategy_not_applicable"
                    continue
                selected.append(strategy)
            except (StrategyEvaluationError, ValueError, TypeError, AttributeError) as exc:
                skipped[strategy.strategy_name] = f"route_check_failed:{exc}"

        selected.sort(key=lambda item: (item.priority, item.strategy_name))

        return RouteDecision(
            event_name=event_name,
            symbol=context.symbol,
            source=source,
            timestamp=context.timestamp,
            selected=selected,
            skipped=skipped,
            categories_used=categories,
            matched_features=list(changed_features or []),
            metadata=dict(metadata or {}),
        )

    async def emit_signal_generated(
        self,
        *,
        payload: RiskReadySignalPayload,
    ) -> None:
        """
        Emit the final signal.generated event consumed by RiskManager.
        """
        payload.validate()

        await self.emit_event(
            "signal.generated",
            payload.to_dict(),
            priority=self._event_priority(payload),
            source=self.component_name,
        )

    async def emit_signal_rejected(
        self,
        *,
        signal: StrategySignal | None,
        symbol: str,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "signal_id": getattr(signal, "signal_id", None),
            "symbol": symbol,
            "strategy_name": getattr(signal, "strategy_name", None),
            "reason": reason,
            "metadata": dict(metadata or {}),
        }

        await self.emit_event(
            "signal.rejected",
            payload,
            priority=EventPriority.LOW,
            source=self.component_name,
        )

    @staticmethod
    def _event_priority(payload: RiskReadySignalPayload) -> EventPriority:
        if payload.priority_score >= 0.85:
            return EventPriority.HIGH
        return EventPriority.NORMAL

    def _resolve_categories(
            self,
            *,
            event_name: str,
            source: FeatureSource | None,
    ) -> list[StrategyCategory]:
        categories = self.routing_config.categories_for_event(event_name)

        if categories:
            return categories

        if source is None:
            return []

        mapped = self._map_source_to_category(source)
        if mapped is None:
            return []

        categories = [mapped]

        if (
                self.routing_config.route_hybrid_on_domain_signal
                and StrategyCategory.HYBRID not in categories
        ):
            categories.append(StrategyCategory.HYBRID)

        return categories

    @staticmethod
    def _map_source_to_category(source: FeatureSource) -> StrategyCategory | None:
        mapping = {
            FeatureSource.ORDERFLOW: StrategyCategory.ORDERFLOW,
            FeatureSource.LIQUIDITY: StrategyCategory.LIQUIDITY,
            FeatureSource.PRICE_ACTION: StrategyCategory.PRICE_ACTION,
            FeatureSource.LIQUIDATIONS: StrategyCategory.LIQUIDATIONS,
            FeatureSource.WHALES: StrategyCategory.WHALES,
            FeatureSource.SPOOFING: StrategyCategory.SPOOFING,
            FeatureSource.SPREADS: StrategyCategory.SPREADS,
            FeatureSource.FUNDING: StrategyCategory.FUNDING,
            FeatureSource.OPEN_INTEREST: StrategyCategory.OPEN_INTEREST,
        }
        return mapping.get(source)


# =============================================================================
# SignalScorer / ConfluenceEngine
# =============================================================================


class SignalScorer(BaseStrategyComponent):
    """
    Scores and enriches StrategySignal objects before confluence/filter/build.
    """

    component_namespace = "strategy.processor.scorer"

    def score_signal(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> StrategySignal:
        signal.validate()
        context.validate()

        priority_score = self._calculate_priority_score(signal, context)
        signal.metadata["priority_score"] = priority_score

        if signal.metadata.get("tier") is None:
            signal.metadata["tier"] = self._tier_from_priority_score(priority_score).value

        if signal.metadata.get("liquidity_class") is None:
            signal.metadata["liquidity_class"] = self._liquidity_class(context).value

        if signal.metadata.get("execution_quality") is None:
            signal.metadata["execution_quality"] = self._execution_quality(signal).value

        signal.score = clamp(max(signal.score, priority_score), 0.0, 1.0)
        signal.confidence = clamp(signal.confidence, 0.0, 1.0)
        signal.confidence_grade = self._confidence_grade(signal.confidence)
        signal.strength = self._confidence_strength(signal.confidence)

        signal.validate()
        return signal

    def score_many(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        return [self.score_signal(signal=signal, context=context) for signal in signals]

    def score_signals(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext | None = None,
    ) -> ConfluenceResult:
        if not signals:
            raise ConfluenceError("signals cannot be empty")

        for signal in signals:
            signal.validate()

        symbol = signals[0].symbol
        if any(signal.symbol != symbol for signal in signals):
            raise ConfluenceError("all signals must belong to the same symbol")

        weighted = self._apply_weights(signals=signals, context=context)
        vote_summary = self._summarize_votes(signals)
        conflict_summary = self._resolve_conflicts(
            signals=signals,
            dominant_side=vote_summary.dominant_side,
        )

        timestamp = context.timestamp if context is not None else max(signal.timestamp for signal in signals)

        result = self._to_confluence_result(
            symbol=symbol,
            timestamp=timestamp,
            weighted_signals=weighted,
            vote_summary=vote_summary,
            conflict_summary=conflict_summary,
        )
        result.validate()
        return result

    def _calculate_priority_score(
        self,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> float:
        components = signal.metadata.get("priority_components")

        if isinstance(components, dict):
            setup_quality = _to_float(components.get("setup_quality"), signal.score) or signal.score
            confluence_score = _to_float(components.get("confluence_score"), 0.0) or 0.0
            liquidity_score = _to_float(components.get("liquidity_score"), self._liquidity_score(context)) or 0.0
            risk_reward_score = _to_float(components.get("risk_reward_score"), self._risk_reward_score(signal)) or 0.0
            execution_quality_score = _to_float(
                components.get("execution_quality_score"),
                self._execution_quality_score(signal),
            ) or 0.0
            regime_alignment_score = _to_float(
                components.get("regime_alignment_score"),
                self._regime_alignment_score(signal, context),
            ) or 0.0
            freshness_score = _to_float(components.get("freshness_score"), self._freshness_score(context)) or 0.0

            return clamp(
                0.25 * setup_quality
                + 0.20 * confluence_score
                + 0.15 * liquidity_score
                + 0.15 * risk_reward_score
                + 0.10 * execution_quality_score
                + 0.10 * regime_alignment_score
                + 0.05 * freshness_score,
                0.0,
                1.0,
            )

        return clamp(
            0.35 * signal.score
            + 0.25 * signal.confidence
            + 0.15 * self._liquidity_score(context)
            + 0.15 * self._risk_reward_score(signal)
            + 0.10 * self._execution_quality_score(signal),
            0.0,
            1.0,
        )

    def _apply_weights(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext | None,
    ) -> list[WeightedSignal]:
        result: list[WeightedSignal] = []

        for signal in signals:
            regime = context.current_regime if context is not None else signal.regime

            category_weight = self.config.get_category_weight(signal.category)
            regime_weight = self.config.get_regime_adjustment(regime)
            strategy_weight = self.config.get_strategy_weight(signal.strategy_name, default=1.0)

            final_weight = category_weight * regime_weight * strategy_weight

            weighted = WeightedSignal(
                signal=signal,
                category_weight=category_weight,
                regime_weight=regime_weight,
                strategy_weight=strategy_weight,
                final_weight=final_weight,
                weighted_score=signal.score * final_weight,
                weighted_confidence=signal.confidence * final_weight,
                metadata={
                    "category": signal.category.value,
                    "regime": regime.value if isinstance(regime, MarketRegime) else str(regime),
                },
            )
            weighted.validate()
            result.append(weighted)

        return result

    @staticmethod
    def _summarize_votes(signals: list[StrategySignal]) -> VoteSummary:
        summary = VoteSummary(total_votes=len(signals))

        for signal in signals:
            if signal.side is SignalSide.LONG:
                summary.long_votes += 1
            elif signal.side is SignalSide.SHORT:
                summary.short_votes += 1
            else:
                summary.flat_votes += 1

            if signal.trigger_type is TriggerType.CONFIRMATION:
                summary.confirmation_count += 1
            if signal.trigger_type is TriggerType.PRIMARY:
                summary.primary_count += 1

        if summary.long_votes > summary.short_votes:
            summary.dominant_side = SignalSide.LONG
        elif summary.short_votes > summary.long_votes:
            summary.dominant_side = SignalSide.SHORT
        else:
            summary.dominant_side = SignalSide.UNKNOWN

        summary.accepted = summary.dominant_side in {SignalSide.LONG, SignalSide.SHORT}
        if not summary.accepted:
            summary.reasons.append("no_dominant_direction")

        return summary

    def _resolve_conflicts(
        self,
        *,
        signals: list[StrategySignal],
        dominant_side: SignalSide,
    ) -> ConflictSummary:
        summary = ConflictSummary()

        for signal in signals:
            if dominant_side.is_directional and signal.side != dominant_side:
                conflict = ConflictRecord(
                    conflict_type=ConflictType.SIDE_CONFLICT,
                    source=signal.strategy_name,
                    message=f"signal side {signal.side.value} conflicts with dominant {dominant_side.value}",
                    penalty=self.config.confluence.conflict_penalty,
                )
                summary.add_conflict(conflict)

        max_penalty = self.config.confluence.conflict_penalty * 3
        if summary.total_penalty >= max_penalty:
            summary.accepted = False
            summary.reasons.append("conflict_penalty_too_high")

        return summary

    def _to_confluence_result(
        self,
        *,
        symbol: str,
        timestamp: datetime,
        weighted_signals: list[WeightedSignal],
        vote_summary: VoteSummary,
        conflict_summary: ConflictSummary,
    ) -> ConfluenceResult:
        if not weighted_signals:
            raise ConfluenceError("weighted_signals cannot be empty")

        total_weight = sum(item.final_weight for item in weighted_signals)
        if total_weight <= 0:
            total_weight = float(len(weighted_signals))

        score = sum(item.weighted_score for item in weighted_signals) / total_weight
        confidence = sum(item.weighted_confidence for item in weighted_signals) / total_weight

        adjusted_score = clamp(score - conflict_summary.total_penalty, 0.0, 1.0)
        adjusted_confidence = clamp(confidence - conflict_summary.total_penalty, 0.0, 1.0)

        config = self.config.confluence
        accepted = (
            vote_summary.accepted
            and conflict_summary.accepted
            and vote_summary.total_votes >= config.min_agreement_count
            and adjusted_confidence >= config.min_confidence
            and adjusted_score >= config.min_score
        )

        reasons = []
        reasons.extend(vote_summary.reasons)
        reasons.extend(conflict_summary.reasons)

        if vote_summary.total_votes < config.min_agreement_count:
            reasons.append("insufficient_agreement_count")
        if adjusted_confidence < config.min_confidence:
            reasons.append("confluence_confidence_too_low")
        if adjusted_score < config.min_score:
            reasons.append("confluence_score_too_low")

        return ConfluenceResult(
            symbol=symbol,
            timestamp=timestamp,
            side=vote_summary.dominant_side,
            score=adjusted_score,
            confidence=adjusted_confidence,
            confidence_grade=self._confidence_grade(adjusted_confidence),
            strength=self._confidence_strength(adjusted_confidence),
            strategy_names=[item.signal.strategy_name for item in weighted_signals],
            reasons=reasons,
            confirmations=[
                reason
                for item in weighted_signals
                for reason in item.signal.reasons
            ],
            conflicts=list(conflict_summary.conflicts),
            accepted=accepted,
            metadata={
                "votes": {
                    "total": vote_summary.total_votes,
                    "long": vote_summary.long_votes,
                    "short": vote_summary.short_votes,
                    "flat": vote_summary.flat_votes,
                },
                "conflict_penalty": conflict_summary.total_penalty,
            },
        )

    def _confidence_grade(self, confidence: float) -> ConfidenceGrade:
        value = clamp(confidence, 0.0, 1.0)
        very_low, low, medium, high = self.config.confidence.grade_bounds()

        if value >= high:
            return ConfidenceGrade.VERY_HIGH
        if value >= medium:
            return ConfidenceGrade.HIGH
        if value >= low:
            return ConfidenceGrade.MEDIUM
        if value >= very_low:
            return ConfidenceGrade.LOW
        return ConfidenceGrade.VERY_LOW

    def _confidence_strength(self, confidence: float) -> SignalStrength:
        value = clamp(confidence, 0.0, 1.0)
        _, low, medium, high = self.config.confidence.grade_bounds()

        if value >= high:
            return SignalStrength.EXTREME
        if value >= medium:
            return SignalStrength.STRONG
        if value >= low:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK

    @staticmethod
    def _tier_from_priority_score(value: float) -> StrategyTradeTier:
        score = clamp(value, 0.0, 1.0)
        if score >= 0.88:
            return StrategyTradeTier.T4
        if score >= 0.74:
            return StrategyTradeTier.T3
        if score >= 0.58:
            return StrategyTradeTier.T2
        return StrategyTradeTier.T1

    def _liquidity_class(self, context: StrategyContext) -> StrategyLiquidityClass:
        score = self._liquidity_score(context)

        if score >= 0.90:
            return StrategyLiquidityClass.TOP
        if score >= 0.75:
            return StrategyLiquidityClass.HIGH
        if score >= 0.50:
            return StrategyLiquidityClass.NORMAL
        if score >= 0.30:
            return StrategyLiquidityClass.LOW
        if score >= 0.15:
            return StrategyLiquidityClass.ILLIQUID
        return StrategyLiquidityClass.SHITCOIN

    def _execution_quality(self, signal: StrategySignal) -> StrategyExecutionQuality:
        score = self._execution_quality_score(signal)

        if score >= 0.90:
            return StrategyExecutionQuality.EXCELLENT
        if score >= 0.75:
            return StrategyExecutionQuality.GOOD
        if score >= 0.50:
            return StrategyExecutionQuality.ACCEPTABLE
        if score >= 0.25:
            return StrategyExecutionQuality.POOR
        return StrategyExecutionQuality.BLOCKED

    @staticmethod
    def _liquidity_score(context: StrategyContext) -> float:
        for name in ("liquidity_score", "market_liquidity_score", "depth_score"):
            value = context.get_feature(name, None)
            parsed = _to_float(value)
            if parsed is not None:
                return clamp(parsed, 0.0, 1.0)

        return 0.5

    @staticmethod
    def _risk_reward_score(signal: StrategySignal) -> float:
        rr = _to_float(signal.metadata.get("rr"))

        if rr is None:
            entry = signal.primary_entry_price
            stop = signal.primary_stop_loss
            target = signal.primary_take_profit

            if entry is None or stop is None or target is None:
                return 0.5

            if signal.side is SignalSide.LONG:
                risk = entry - stop
                reward = target - entry
            elif signal.side is SignalSide.SHORT:
                risk = stop - entry
                reward = entry - target
            else:
                return 0.0

            if risk <= 0:
                return 0.0

            rr = reward / risk

        if rr >= 3.0:
            return 1.0
        if rr >= 2.0:
            return 0.8
        if rr >= 1.5:
            return 0.65
        if rr >= 1.0:
            return 0.45
        return 0.2

    @staticmethod
    def _execution_quality_score(signal: StrategySignal) -> float:
        raw_score = _to_float(signal.metadata.get("execution_quality_score"))
        if raw_score is not None:
            return clamp(raw_score, 0.0, 1.0)

        raw_quality = signal.metadata.get("execution_quality")
        quality = _parse_enum(
            StrategyExecutionQuality,
            raw_quality,
            StrategyExecutionQuality.ACCEPTABLE,
        )

        return {
            StrategyExecutionQuality.EXCELLENT: 1.0,
            StrategyExecutionQuality.GOOD: 0.8,
            StrategyExecutionQuality.ACCEPTABLE: 0.6,
            StrategyExecutionQuality.POOR: 0.3,
            StrategyExecutionQuality.BLOCKED: 0.0,
        }[quality]

    @staticmethod
    def _regime_alignment_score(signal: StrategySignal, context: StrategyContext) -> float:
        regime = context.current_regime

        if regime is MarketRegime.UNKNOWN:
            return 0.5

        if signal.side is SignalSide.LONG and regime in {
            MarketRegime.TRENDING_UP,
            MarketRegime.BREAKOUT,
        }:
            return 1.0

        if signal.side is SignalSide.SHORT and regime in {
            MarketRegime.TRENDING_DOWN,
            MarketRegime.BREAKOUT,
        }:
            return 1.0

        if regime in {MarketRegime.RANGING, MarketRegime.SQUEEZE}:
            return 0.7

        if regime.is_risky:
            return 0.35

        return 0.5

    @staticmethod
    def _freshness_score(context: StrategyContext) -> float:
        if not context.feature_map:
            return 0.5

        total = 0
        stale_count = 0

        for feature in context.feature_map.values():
            total += 1
            if feature.is_stale():
                stale_count += 1

        if total <= 0:
            return 0.5

        return clamp(1.0 - stale_count / total, 0.0, 1.0)


class ConfluenceEngine(BaseStrategyComponent):
    """
    Builds one confluence evaluation and optionally merges compatible signals.
    """

    component_namespace = "strategy.processor.confluence"

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, scheduler=scheduler)
        self.confluence_config: ConfluenceConfig = config.confluence
        self.scorer = SignalScorer(config=config, event_bus=event_bus, scheduler=scheduler)

    def evaluate(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext,
    ) -> ConfluenceEvaluation:
        evaluation = ConfluenceEvaluation(
            symbol=context.symbol,
            timestamp=context.timestamp,
            raw_signals=list(signals),
        )

        if not self.confluence_config.enabled:
            evaluation.eligible_signals = list(signals)
            evaluation.accepted_signals = list(signals)
            evaluation.reasons.append("confluence_disabled")
            return evaluation

        eligible = [
            signal
            for signal in signals
            if signal.side in {SignalSide.LONG, SignalSide.SHORT}
        ]
        evaluation.eligible_signals = eligible

        if not eligible:
            evaluation.rejected_signals = {
                signal.strategy_name: "non_directional_signal"
                for signal in signals
            }
            evaluation.reasons.append("no_directional_signals")
            return evaluation

        result = self.scorer.score_signals(signals=eligible, context=context)
        evaluation.result = result

        if not result.accepted:
            evaluation.rejected_signals = {
                signal.strategy_name: "confluence_rejected"
                for signal in eligible
            }
            evaluation.reasons.extend(result.reasons)
            return evaluation

        accepted = [
            signal
            for signal in eligible
            if signal.side == result.side
        ]
        evaluation.accepted_signals = accepted
        evaluation.merged_signal = self._merge(
            result=result,
            signals=accepted,
        )
        evaluation.reasons.extend(result.reasons)
        return evaluation

    @staticmethod
    def _merge(
        *,
        result: ConfluenceResult,
        signals: list[StrategySignal],
    ) -> StrategySignal | None:
        if not signals:
            return None

        if len(signals) == 1:
            signal = signals[0]
            signal.origin = SignalOrigin.CONFLUENCE
            signal.confirmations = list(dict.fromkeys(signal.confirmations + result.confirmations))
            signal.conflicts = list(dict.fromkeys(signal.conflicts + result.conflicts))
            signal.metadata["confluence_score"] = result.score
            signal.metadata["confluence_confidence"] = result.confidence
            signal.metadata["confluence_strategy_names"] = list(result.strategy_names)
            signal.validate()
            return signal

        best = sorted(
            signals,
            key=lambda item: (
                -float(item.metadata.get("priority_score", item.score)),
                -item.confidence,
                -item.score,
                item.strategy_name,
            ),
        )[0]

        best.origin = SignalOrigin.CONFLUENCE
        best.score = max(best.score, result.score)
        best.confidence = max(best.confidence, result.confidence)
        best.confidence_grade = confidence_to_grade(best.confidence)
        best.strength = confidence_to_strength(best.confidence)
        best.combined_from = list(dict.fromkeys(best.combined_from + result.strategy_names))
        best.confirmations = list(dict.fromkeys(best.confirmations + result.confirmations))
        best.metadata["confluence_score"] = result.score
        best.metadata["confluence_confidence"] = result.confidence
        best.metadata["confluence_strategy_names"] = list(result.strategy_names)
        best.metadata["priority_score"] = max(
            _to_float(best.metadata.get("priority_score"), best.score) or best.score,
            result.score,
        )

        best.validate()
        return best


# =============================================================================
# SignalFilterChain
# =============================================================================


class SignalFilterChain(BaseStrategyComponent):
    """
    Applies strategy-level signal filters before SignalBuilder.
    """

    component_namespace = "strategy.processor.filters"

    def apply(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        accepted: list[StrategySignal] = []

        for signal in signals:
            evaluation = self.evaluate_signal(signal=signal, context=context)

            for result in evaluation.results:
                signal.add_filter_result(result)

            if evaluation.accepted:
                accepted.append(signal)
            else:
                signal.to_rejected()
                for reason in evaluation.reasons:
                    signal.add_reason(reason)

        return accepted

    def evaluate_signal(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> FilterEvaluation:
        signal.validate()
        context.validate()

        evaluation = FilterEvaluation(
            signal=signal,
            context_symbol=context.symbol,
            timestamp=context.timestamp,
        )

        # Hard safety filters are always enforced. Optional filters are governed
        # by FilterConfig so preset-level settings are not silently ignored.
        self._filter_symbol_match(evaluation, context)
        self._filter_directional(evaluation)

        filters = self.config.filters
        if not filters.enabled:
            return evaluation

        self._filter_confidence(evaluation)
        self._filter_score(evaluation)
        self._filter_risk_reward(evaluation)
        self._filter_age(evaluation, context)

        if filters.enable_freshness_filter:
            self._filter_freshness(evaluation, context)

        if filters.enable_regime_filter:
            self._filter_regime(evaluation, context)

        if filters.enable_volatility_filter:
            self._filter_volatility(evaluation, context)

        if filters.enable_liquidity_filter:
            self._filter_liquidity(evaluation, context)

        if filters.enable_spread_filter:
            self._filter_spread(evaluation, context)

        if filters.enable_funding_filter:
            self._filter_funding(evaluation, context)

        self._filter_execution_quality(evaluation)

        return evaluation

    @staticmethod
    def _filter_symbol_match(
        evaluation: FilterEvaluation,
        context: StrategyContext,
    ) -> None:
        if evaluation.signal.symbol != context.symbol:
            evaluation.add_result(
                FilterResult(
                    name="symbol_match",
                    decision=FilterDecision.BLOCK,
                    reason="signal_symbol_does_not_match_context",
                )
            )

    @staticmethod
    def _filter_directional(evaluation: FilterEvaluation) -> None:
        if evaluation.signal.side not in {SignalSide.LONG, SignalSide.SHORT}:
            evaluation.add_result(
                FilterResult(
                    name="directional_signal",
                    decision=FilterDecision.BLOCK,
                    reason="signal_side_is_not_directional",
                )
            )

    def _filter_confidence(self, evaluation: FilterEvaluation) -> None:
        configured = self.config.filters.min_signal_confidence
        threshold = self.config.runtime.min_confidence if configured is None else configured

        if evaluation.signal.confidence < threshold:
            evaluation.add_result(
                FilterResult(
                    name="min_confidence",
                    decision=FilterDecision.BLOCK,
                    reason="confidence_below_runtime_threshold",
                    score_impact=threshold - evaluation.signal.confidence,
                )
            )

    def _filter_score(self, evaluation: FilterEvaluation) -> None:
        configured = self.config.filters.min_signal_score
        threshold = self.config.runtime.min_score if configured is None else configured

        if evaluation.signal.score < threshold:
            evaluation.add_result(
                FilterResult(
                    name="min_score",
                    decision=FilterDecision.BLOCK,
                    reason="score_below_runtime_threshold",
                    score_impact=threshold - evaluation.signal.score,
                )
            )

    def _filter_age(
        self,
        evaluation: FilterEvaluation,
        context: StrategyContext,
    ) -> None:
        max_age = self.config.runtime.max_signal_age_seconds
        reference_time = ensure_aware_utc(context.timestamp)
        signal_time = ensure_aware_utc(evaluation.signal.timestamp)
        age = (reference_time - signal_time).total_seconds()
        if age < 0:
            age = 0.0

        if age > max_age:
            evaluation.add_result(
                FilterResult(
                    name="signal_age",
                    decision=FilterDecision.BLOCK,
                    reason="signal_expired_before_routing",
                    metadata={"age_seconds": age, "max_age_seconds": max_age},
                )
            )

    def _filter_risk_reward(self, evaluation: FilterEvaluation) -> None:
        threshold = self.config.filters.min_risk_reward
        if threshold <= 0:
            return

        rr = _to_float(evaluation.signal.metadata.get("rr"))
        if rr is None:
            entry = evaluation.signal.primary_entry_price
            stop = evaluation.signal.primary_stop_loss
            target = evaluation.signal.primary_take_profit
            if entry is not None and stop is not None and target is not None:
                risk = abs(entry - stop)
                reward = abs(target - entry)
                if risk > 0:
                    rr = reward / risk

        if rr is not None and rr < threshold:
            evaluation.add_result(
                FilterResult(
                    name="min_risk_reward",
                    decision=FilterDecision.BLOCK,
                    reason="risk_reward_below_filter_threshold",
                    metadata={"rr": rr, "threshold": threshold},
                )
            )

    def _filter_regime(self, evaluation: FilterEvaluation, context: StrategyContext) -> None:
        allowed = self.config.runtime.allowed_regimes
        if not allowed or MarketRegime.UNKNOWN in allowed:
            return

        regime = context.current_regime
        if regime not in allowed:
            evaluation.add_result(
                FilterResult(
                    name="regime",
                    decision=FilterDecision.BLOCK,
                    reason="regime_not_allowed_by_runtime_config",
                    metadata={"regime": regime.value, "allowed": [item.value for item in allowed]},
                )
            )

    def _filter_volatility(self, evaluation: FilterEvaluation, context: StrategyContext) -> None:
        value = self._context_float(
            context,
            "volatility_zscore",
            "volatility_z_score",
            "volatility.zscore",
            "volatility.z_score",
        )
        if value is None:
            return

        threshold = self.config.filters.max_volatility_zscore
        if abs(value) > threshold:
            evaluation.add_result(
                FilterResult(
                    name="volatility",
                    decision=FilterDecision.BLOCK,
                    reason="volatility_zscore_above_filter_threshold",
                    metadata={"volatility_zscore": value, "threshold": threshold},
                )
            )

    def _filter_liquidity(self, evaluation: FilterEvaluation, context: StrategyContext) -> None:
        value = self._context_float(
            context,
            "liquidity_score",
            "market_liquidity_score",
            "depth_score",
            "liquidity.score",
        )
        if value is None:
            return

        threshold = self.config.filters.min_liquidity_score
        if value < threshold:
            evaluation.add_result(
                FilterResult(
                    name="liquidity",
                    decision=FilterDecision.BLOCK,
                    reason="liquidity_score_below_filter_threshold",
                    metadata={"liquidity_score": value, "threshold": threshold},
                )
            )

    def _filter_spread(self, evaluation: FilterEvaluation, context: StrategyContext) -> None:
        value = self._context_float(
            context,
            "spread_bps",
            "market_spread_bps",
            "bid_ask_spread_bps",
            "spreads.spread_bps",
        )
        if value is None:
            return

        threshold = self.config.filters.max_spread_bps
        if value > threshold:
            evaluation.add_result(
                FilterResult(
                    name="spread",
                    decision=FilterDecision.BLOCK,
                    reason="spread_bps_above_filter_threshold",
                    metadata={"spread_bps": value, "threshold": threshold},
                )
            )

    def _filter_funding(self, evaluation: FilterEvaluation, context: StrategyContext) -> None:
        value = self._context_float(
            context,
            "funding_alignment",
            "funding.bias_alignment",
            "funding.alignment",
        )
        if value is None:
            return

        threshold = self.config.filters.min_funding_alignment
        if value < threshold:
            evaluation.add_result(
                FilterResult(
                    name="funding",
                    decision=FilterDecision.BLOCK,
                    reason="funding_alignment_below_filter_threshold",
                    metadata={"funding_alignment": value, "threshold": threshold},
                )
            )

    @staticmethod
    def _context_float(context: StrategyContext, *names: str) -> float | None:
        for name in names:
            value = context.get_feature(name, None)
            parsed = _to_float(value)
            if parsed is not None:
                return parsed

            if "." in name:
                domain_name, key = name.split(".", 1)
                domain = getattr(context, domain_name, None)
                if isinstance(domain, dict):
                    parsed = _to_float(domain.get(key))
                    if parsed is not None:
                        return parsed

        return None

    @staticmethod
    def _filter_freshness(
        evaluation: FilterEvaluation,
        context: StrategyContext,
    ) -> None:
        stale_features = [
            name
            for name, snapshot in context.feature_map.items()
            if snapshot.is_stale()
        ]

        if stale_features:
            evaluation.add_result(
                FilterResult(
                    name="feature_freshness",
                    decision=FilterDecision.WARN,
                    reason="context_has_stale_features",
                    metadata={"stale_features": stale_features},
                )
            )

    @staticmethod
    def _filter_execution_quality(evaluation: FilterEvaluation) -> None:
        raw = evaluation.signal.metadata.get("execution_quality")
        if raw is None:
            return

        quality = _parse_enum(
            StrategyExecutionQuality,
            raw,
            StrategyExecutionQuality.ACCEPTABLE,
        )

        if quality is StrategyExecutionQuality.BLOCKED:
            evaluation.add_result(
                FilterResult(
                    name="execution_quality",
                    decision=FilterDecision.BLOCK,
                    reason="execution_quality_blocked",
                )
            )


# =============================================================================
# SignalBuilder
# =============================================================================


class SignalBuilder(BaseStrategyComponent):
    """
    Ensures every final signal has entry/exit/invalidation/execution plan.
    """

    component_namespace = "strategy.processor.builder"

    @property
    def builder_config(self) -> BuilderConfig:
        return self.config.builders

    def build(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> BuildEvaluation:
        signal.validate()
        context.validate()

        evaluation = BuildEvaluation(
            signal=signal,
            context_symbol=context.symbol,
        )

        try:
            entry = self._resolve_entry(signal, context)
            invalidation = self._resolve_invalidation(signal, entry=entry)
            targets = self._resolve_targets(signal, entry=entry, invalidation=invalidation)
            exit_plan = self._resolve_exit_plan(signal, invalidation=invalidation, targets=targets)
            execution_plan = self._resolve_execution_plan(
                signal=signal,
                entry=entry,
                exit_plan=exit_plan,
                invalidation=invalidation,
            )

            signal.entry_plan = entry
            signal.invalidation_plan = invalidation
            signal.exit_plan = exit_plan
            signal.execution_plan = execution_plan

            signal.metadata.setdefault("entry_price", entry.price)
            signal.metadata.setdefault("stop_loss", exit_plan.stop_loss or invalidation.price)
            if targets:
                signal.metadata.setdefault("take_profit", targets[0].price)

            signal.validate()

            evaluation.entry = entry
            evaluation.invalidation = invalidation
            evaluation.targets = targets
            evaluation.exit_plan = exit_plan
            evaluation.execution_plan = execution_plan
            evaluation.accepted = True

        except (BuilderError, ValueError, TypeError, AttributeError) as exc:
            evaluation.reject(f"build_failed:{exc}")
            signal.to_rejected()
            signal.add_reason(f"build_failed:{exc}")

        return evaluation

    def build_many(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext,
    ) -> tuple[list[StrategySignal], dict[str, str]]:
        accepted: list[StrategySignal] = []
        rejected: dict[str, str] = {}

        for signal in signals:
            evaluation = self.build(signal=signal, context=context)

            if evaluation.accepted:
                accepted.append(signal)
            else:
                rejected[signal.strategy_name] = ";".join(evaluation.reasons) or "build_failed"

        return accepted, rejected

    @staticmethod
    def assert_risk_ready(signal: StrategySignal) -> None:
        signal.validate()

        if signal.execution_plan is None:
            raise BuilderError("signal.execution_plan is required")

        entry_price = signal.primary_entry_price
        stop_loss = signal.primary_stop_loss

        if entry_price is None or entry_price <= 0:
            raise BuilderError("risk-ready signal requires entry_price > 0")

        if stop_loss is None or stop_loss <= 0:
            raise BuilderError("risk-ready signal requires stop_loss > 0")

        if signal.side is SignalSide.LONG and stop_loss >= entry_price:
            raise BuilderError("long signal stop_loss must be below entry_price")

        if signal.side is SignalSide.SHORT and stop_loss <= entry_price:
            raise BuilderError("short signal stop_loss must be above entry_price")

    def _resolve_entry(
        self,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> EntryPlan:
        if signal.entry_plan is not None:
            signal.entry_plan.validate()
            return signal.entry_plan

        price = (
            _to_float(signal.metadata.get("entry_price"))
            or _to_float(signal.metadata.get("price"))
            or _to_float(signal.metadata.get("last_price"))
            or _to_float(signal.metadata.get("current_price"))
            or _to_float(signal.metadata.get("close"))
        )

        # In production/backtest analytics events the StrategySignal may not
        # carry an explicit EntryPlan yet.  In that case the builder must be
        # able to use the canonical StrategyContext price snapshot produced by
        # SignalNormalizer.apply_to_context().  PriceSnapshot in strategy.models
        # uses last_price; older compatibility code checked only last/close/price,
        # so valid replay events could still fail with
        # "build_failed:unable to resolve entry price".
        if price is None and context.price is not None:
            price = (
                _to_float(getattr(context.price, "last_price", None))
                or _to_float(getattr(context.price, "last", None))
                or _to_float(getattr(context.price, "close", None))
                or _to_float(getattr(context.price, "mark_price", None))
                or _to_float(getattr(context.price, "index_price", None))
                or _to_float(getattr(context.price, "price", None))
            )

        # Final fallback: read the active source-domain contract and generic
        # feature snapshots.  This keeps concrete strategies clean while making
        # SignalBuilder robust for merged/confluence signals whose metadata was
        # stripped during merging.
        if price is None:
            source_domain = context.domain_dict(FeatureSource.from_strategy_category(signal.category))
            for key in ("entry_price", "current_price", "last_price", "price", "close", "mark_price"):
                price = _to_float(source_domain.get(key))
                if price is not None and price > 0:
                    break

        if price is None:
            for feature_name in (
                "entry_price",
                "price",
                "last_price",
                "current_price",
                f"{signal.category.value}.entry_price",
                f"{signal.category.value}.price",
                f"{signal.category.value}.last_price",
                f"{signal.category.value}.current_price",
            ):
                snapshot = context.get_feature(feature_name)
                if snapshot is None:
                    continue
                price = _to_float(getattr(snapshot, "value", None))
                if price is not None and price > 0:
                    break

        if price is None or price <= 0:
            raise BuilderError("unable to resolve entry price")

        entry = EntryPlan(
            entry_type=self._entry_type(signal),
            price=price,
            timeout_seconds=_to_int(signal.metadata.get("entry_timeout_seconds")),
            max_slippage_bps=_to_float(signal.metadata.get("max_slippage_bps")),
            confirmation_required=_to_bool(signal.metadata.get("entry_confirmation_required")),
            metadata={"built_by": self.component_name},
        )
        entry.validate()
        return entry

    def _resolve_invalidation(
        self,
        signal: StrategySignal,
        *,
        entry: EntryPlan,
    ) -> InvalidationPlan:
        if signal.invalidation_plan is not None:
            signal.invalidation_plan.validate()
            return signal.invalidation_plan

        stop = _to_float(signal.metadata.get("stop_loss"))

        if stop is None and signal.exit_plan is not None:
            stop = _to_float(signal.exit_plan.stop_loss)

        if stop is None:
            stop = self._infer_stop_loss(signal, entry_price=entry.price)

        if stop is None or stop <= 0:
            raise BuilderError("unable to resolve stop_loss/invalidation price")

        invalidation = InvalidationPlan(
            price=stop,
            reason=str(signal.metadata.get("invalidation_reason", "strategy_invalidation")),
            timeout_seconds=_to_int(signal.metadata.get("invalidation_timeout_seconds")),
            metadata={"built_by": self.component_name},
        )
        invalidation.validate()
        return invalidation

    def _resolve_targets(
        self,
        signal: StrategySignal,
        *,
        entry: EntryPlan,
        invalidation: InvalidationPlan,
    ) -> list[TargetPlan]:
        if signal.exit_plan is not None and signal.exit_plan.take_profit_levels:
            for target in signal.exit_plan.take_profit_levels:
                target.validate()
            return list(signal.exit_plan.take_profit_levels)

        raw_targets = signal.metadata.get("targets")
        targets: list[TargetPlan] = []

        if isinstance(raw_targets, list):
            for item in raw_targets:
                if isinstance(item, TargetPlan):
                    item.validate()
                    targets.append(item)
                    continue

                if isinstance(item, dict):
                    price = _to_float(item.get("price"))
                    if price is None or price <= 0:
                        continue

                    size_fraction = _to_float(item.get("size_fraction"), 1.0)
                    if size_fraction is None:
                        size_fraction = 1.0

                    target = TargetPlan(
                        price=price,
                        size_fraction=size_fraction,
                        rr=_to_float(item.get("rr")),
                        label=item.get("label"),
                        metadata=dict(item.get("metadata") or {}),
                    )
                    target.validate()
                    targets.append(target)

        if targets:
            return targets

        take_profit = _to_float(signal.metadata.get("take_profit"))
        if take_profit is None:
            take_profit = self._infer_take_profit(
                signal,
                entry_price=entry.price,
                stop_loss=invalidation.price,
            )

        if take_profit is not None and take_profit > 0:
            target = TargetPlan(
                price=take_profit,
                size_fraction=1.0,
                label="primary",
            )
            target.validate()
            targets.append(target)

        return targets

    def _resolve_exit_plan(
        self,
        signal: StrategySignal,
        *,
        invalidation: InvalidationPlan,
        targets: list[TargetPlan],
    ) -> ExitPlan:
        if signal.exit_plan is not None:
            if signal.exit_plan.stop_loss is None:
                signal.exit_plan.stop_loss = invalidation.price
            if not signal.exit_plan.take_profit_levels and targets:
                signal.exit_plan.take_profit_levels = list(targets)
            signal.exit_plan.validate()
            return signal.exit_plan

        exit_types = [ExitType.STOP_LOSS]
        if targets:
            exit_types.append(ExitType.TAKE_PROFIT)

        exit_plan = ExitPlan(
            exit_types=exit_types,
            stop_loss=invalidation.price,
            take_profit_levels=list(targets),
            trailing_distance=_to_float(signal.metadata.get("trailing_distance")),
            max_holding_seconds=_to_int(signal.metadata.get("max_holding_seconds")),
            partial_exit_enabled=_to_bool(signal.metadata.get("partial_exit_enabled")),
            metadata={"built_by": self.component_name},
        )
        exit_plan.validate()
        return exit_plan

    def _resolve_execution_plan(
        self,
        *,
        signal: StrategySignal,
        entry: EntryPlan,
        exit_plan: ExitPlan,
        invalidation: InvalidationPlan,
    ) -> ExecutionPlanDraft:
        if signal.execution_plan is not None:
            signal.execution_plan.validate()
            return signal.execution_plan

        execution_plan = ExecutionPlanDraft(
            symbol=signal.symbol,
            side=signal.side,
            entry=entry,
            exit=exit_plan,
            invalidation=invalidation,
            leverage=_to_float(signal.metadata.get("requested_leverage")),
            reduce_only=_to_bool(signal.metadata.get("reduce_only")),
            post_only=_to_bool(signal.metadata.get("post_only")),
            expected_holding_seconds=_to_int(signal.metadata.get("expected_holding_seconds")),
            metadata={
                "built_by": self.component_name,
                "order_intent": signal.metadata.get("order_intent", StrategyOrderIntent.OPEN.value),
                "margin_mode": signal.metadata.get("margin_mode", StrategyMarginMode.ISOLATED.value),
                "market_type": signal.metadata.get("market_type", StrategyMarketType.USDM_FUTURES.value),
            },
        )
        execution_plan.validate()
        return execution_plan

    def _infer_stop_loss(
        self,
        signal: StrategySignal,
        *,
        entry_price: float,
    ) -> float | None:
        """
        Infer a protective invalidation price when a concrete strategy produced
        a valid directional signal but did not attach an explicit ExitPlan.

        Concrete strategies should still provide their own invalidation when
        they have domain-specific structure levels.  This fallback exists for
        analytics-driven/backtest payloads where the strategy intentionally
        returns only the decision and leaves plan completion to SignalBuilder.
        """
        stop_distance = _to_float(signal.metadata.get("stop_distance"))

        if stop_distance is None:
            atr = _to_float(signal.metadata.get("atr"))
            multiplier = _to_float(signal.metadata.get("atr_stop_multiplier"), 1.5) or 1.5
            if atr is not None and atr > 0:
                stop_distance = atr * multiplier

        if stop_distance is None:
            stop_bps = (
                _to_float(signal.metadata.get("stop_loss_bps"))
                or _to_float(signal.metadata.get("stop_bps"))
                or _to_float(signal.metadata.get("invalidation_bps"))
                or _to_float(getattr(self.builder_config, "default_stop_loss_bps", None))
                or _to_float(getattr(self.builder_config, "default_stop_bps", None))
                or _to_float(getattr(self.builder_config, "default_invalidation_bps", None))
            )
            if stop_bps is not None and stop_bps > 0:
                stop_distance = entry_price * (stop_bps / 10_000.0)

        if stop_distance is None:
            stop_pct = (
                _to_float(signal.metadata.get("stop_loss_pct"))
                or _to_float(signal.metadata.get("stop_pct"))
                or _to_float(signal.metadata.get("invalidation_pct"))
                or _to_float(getattr(self.builder_config, "default_stop_loss_pct", None))
                or _to_float(getattr(self.builder_config, "default_stop_pct", None))
                or _to_float(getattr(self.builder_config, "default_invalidation_pct", None))
            )
            if stop_pct is not None and stop_pct > 0:
                # Accept either fraction form (0.003 = 0.3%) or percent form
                # (0.3 = 0.3%, 1.0 = 1%). Values above 1 are treated as
                # percentages too, e.g. 2.5 -> 2.5%.
                pct_fraction = stop_pct if stop_pct < 0.05 else stop_pct / 100.0
                stop_distance = entry_price * pct_fraction

        if stop_distance is None:
            # Last-resort deterministic fallback for backtesting/simple
            # analytics signals.  It is deliberately conservative and only
            # used when no strategy/config metadata provides a stop.  Without
            # this, a valid StrategySignal cannot become risk-ready although
            # the builder has enough information to create a basic protective
            # plan.
            fallback_bps = (
                _to_float(getattr(self.builder_config, "fallback_stop_loss_bps", None))
                or _to_float(getattr(self.builder_config, "fallback_stop_bps", None))
                or 30.0
            )
            stop_distance = entry_price * (fallback_bps / 10_000.0)
            signal.metadata.setdefault("invalidation_source", "builder_fallback_stop_bps")
            signal.metadata.setdefault("fallback_stop_loss_bps", fallback_bps)

        if stop_distance is None or stop_distance <= 0:
            return None

        if signal.side is SignalSide.LONG:
            stop = entry_price - stop_distance
            return stop if stop > 0 else None

        if signal.side is SignalSide.SHORT:
            return entry_price + stop_distance

        return None

    def _infer_take_profit(
        self,
        signal: StrategySignal,
        *,
        entry_price: float,
        stop_loss: float,
    ) -> float | None:
        rr = _to_float(signal.metadata.get("rr"))
        if rr is None:
            rr = _to_float(getattr(self.builder_config, "default_rr_ratio", None), 2.0) or 2.0

        if rr <= 0:
            return None

        risk_distance = abs(entry_price - stop_loss)
        if risk_distance <= 0:
            return None

        if signal.side is SignalSide.LONG:
            return entry_price + risk_distance * rr

        if signal.side is SignalSide.SHORT:
            return entry_price - risk_distance * rr

        return None

    @staticmethod
    def _entry_type(signal: StrategySignal) -> EntryType:
        raw = signal.metadata.get("entry_type")
        return _parse_enum(EntryType, raw, EntryType.LIMIT)


# =============================================================================
# PortfolioCoordinator
# =============================================================================


class PortfolioCoordinator(BaseStrategyComponent):
    """
    Pre-risk portfolio coordination.

    It does not approve risk. It only decides which already built signals are
    worth sending to RiskManager.
    """

    component_namespace = "strategy.processor.portfolio"

    def __init__(
        self,
        config: StrategyConfig,
        state: StrategyRuntimeState,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, scheduler=scheduler)
        self.state = state
        self.portfolio_config: PortfolioCoordinatorConfig = config.portfolio

    def coordinate(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext,
    ) -> CoordinationDecision:
        decision = CoordinationDecision(
            symbol=context.symbol,
            timestamp=context.timestamp,
            raw_signals=list(signals),
        )

        if not signals:
            decision.accepted = False
            decision.reasons.append("no_signals_to_coordinate")
            return decision

        try:
            accepted, rejected = self._suppress_repeating_signals(
                symbol=context.symbol,
                signals=signals,
                now=context.timestamp,
            )
            decision.rejected_signals.update(rejected)

            accepted, rejected = self._deduplicate_by_side(accepted)
            decision.rejected_signals.update(rejected)

            accepted, rejected = self._apply_symbol_limits(context.symbol, accepted)
            decision.rejected_signals.update(rejected)

            merged = self._merge_similar_signals(accepted)

            decision.accepted_signals = accepted
            decision.merged_signals = merged
            decision.accepted = bool(decision.final_signals)

            if not decision.accepted:
                decision.reasons.append("all_signals_suppressed_by_portfolio_coordinator")

            self._update_state_after_acceptance(
                symbol=context.symbol,
                signals=decision.final_signals,
            )

            return decision

        except (PortfolioCoordinationError, ValueError, TypeError, AttributeError) as exc:
            raise PortfolioCoordinationError(f"portfolio coordination failed: {exc}") from exc

    def _suppress_repeating_signals(
        self,
        *,
        symbol: str,
        signals: list[StrategySignal],
        now: datetime,
    ) -> tuple[list[StrategySignal], dict[str, str]]:
        window = self.portfolio_config.repeated_signal_suppression_seconds
        if window <= 0:
            return signals, {}

        accepted: list[StrategySignal] = []
        rejected: dict[str, str] = {}

        for signal in signals:
            previous = self.state.signals.get_last_for_symbol_side(symbol, signal.side)
            if previous is None:
                accepted.append(signal)
                continue

            previous_ts = ensure_aware_utc(previous.timestamp)
            current_ts = ensure_aware_utc(now)
            delta = (current_ts - previous_ts).total_seconds()

            # Replay/backtest safety: identical or non-monotonic timestamps should
            # not be treated as a newer repeated live signal. In the backtesting
            # harness the same historical analytics event may be processed more
            # than once (EventBus path + direct debug path, or overlapping
            # subscriptions). Suppressing equal-timestamp signals prevents any
            # signal.generated from being emitted even though the signal is valid.
            # Live repeated-signal suppression still applies only to strictly
            # newer signals within the configured window.
            if delta <= 0:
                accepted.append(signal)
                continue

            if delta <= window and (
                previous.strategy_name == signal.strategy_name
                or previous.setup_type == signal.setup_type
            ):
                rejected[signal.strategy_name] = "repeating_signal_suppressed"
                continue

            accepted.append(signal)

        return accepted, rejected

    def _deduplicate_by_side(
        self,
        signals: list[StrategySignal],
    ) -> tuple[list[StrategySignal], dict[str, str]]:
        if not self.portfolio_config.deduplicate_by_side:
            return signals, {}

        best: dict[SignalSide, StrategySignal] = {}
        rejected: dict[str, str] = {}

        for signal in signals:
            current = best.get(signal.side)
            if current is None:
                best[signal.side] = signal
                continue

            challenger_score = _to_float(signal.metadata.get("priority_score"), signal.score) or signal.score
            current_score = _to_float(current.metadata.get("priority_score"), current.score) or current.score

            challenger_wins = (
                challenger_score > current_score
                or signal.confidence > current.confidence
                or signal.score > current.score
            )

            if challenger_wins:
                rejected[current.strategy_name] = f"deduplicated_by_side:{signal.strategy_name}"
                best[signal.side] = signal
            else:
                rejected[signal.strategy_name] = f"deduplicated_by_side:{current.strategy_name}"

        accepted = list(best.values())
        accepted.sort(
            key=lambda item: (
                -(_to_float(item.metadata.get("priority_score"), item.score) or item.score),
                -item.confidence,
                -item.score,
                item.strategy_name,
            )
        )
        return accepted, rejected

    def _apply_symbol_limits(
        self,
        symbol: str,
        signals: list[StrategySignal],
    ) -> tuple[list[StrategySignal], dict[str, str]]:
        active_count = len(self.state.signals.get_active_for_symbol(symbol))
        limit = self.portfolio_config.max_signals_per_symbol

        if active_count >= limit:
            return [], {
                signal.strategy_name: "max_signals_per_symbol_reached"
                for signal in signals
            }

        available = max(0, limit - active_count)
        ordered = sorted(
            signals,
            key=lambda item: (
                -(_to_float(item.metadata.get("priority_score"), item.score) or item.score),
                -item.confidence,
                -item.score,
                item.strategy_name,
            ),
        )

        accepted = ordered[:available]
        rejected = {
            signal.strategy_name: "max_signals_per_symbol_reached"
            for signal in ordered[available:]
        }
        return accepted, rejected

    @staticmethod
    def _merge_similar_signals(signals: list[StrategySignal]) -> list[StrategySignal]:
        if len(signals) <= 1:
            return signals

        grouped: dict[SignalSide, list[StrategySignal]] = {}
        for signal in signals:
            grouped.setdefault(signal.side, []).append(signal)

        merged: list[StrategySignal] = []

        for group in grouped.values():
            if len(group) == 1:
                merged.append(group[0])
                continue

            best = sorted(
                group,
                key=lambda item: (
                    -(_to_float(item.metadata.get("priority_score"), item.score) or item.score),
                    -item.confidence,
                    -item.score,
                ),
            )[0]

            best.combined_from = list(
                dict.fromkeys(best.combined_from + [signal.strategy_name for signal in group])
            )
            best.confirmations = list(
                dict.fromkeys(
                    best.confirmations
                    + [reason for signal in group for reason in signal.reasons]
                )
            )
            best.origin = SignalOrigin.MULTI_STRATEGY
            merged.append(best)

        return merged

    def _update_state_after_acceptance(
        self,
        *,
        symbol: str,
        signals: list[StrategySignal],
    ) -> None:
        for signal in signals:
            signal.to_pending()
            self.state.update_signal(signal, active=True)

            if self.portfolio_config.side_cooldown_seconds > 0:
                self.state.cooldowns.add_side_cooldown(
                    symbol=symbol,
                    side=signal.side,
                    seconds=self.portfolio_config.side_cooldown_seconds,
                    reason="side_signal_accepted",
                )


# =============================================================================
# SignalProcessor
# =============================================================================


class SignalProcessor(BaseStrategyComponent):
    """
    Facade class for full strategy signal processing.
    """

    component_namespace = "strategy.processor"

    def __init__(
        self,
        config: StrategyConfig,
        registry: StrategyRegistry,
        state: StrategyRuntimeState,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, scheduler=scheduler)
        self.registry = registry
        self.state = state

        self.normalizer = SignalNormalizer(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
        )
        self.router = SignalRouter(
            config=config,
            registry=registry,
            event_bus=event_bus,
            scheduler=scheduler,
        )
        self.scorer = SignalScorer(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
        )
        self.confluence = ConfluenceEngine(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
        )
        self.filters = SignalFilterChain(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
        )
        self.builder = SignalBuilder(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
        )
        self.portfolio = PortfolioCoordinator(
            config=config,
            state=state,
            event_bus=event_bus,
            scheduler=scheduler,
        )

    async def process_event(
        self,
        *,
        event_name: str,
        payload: dict[str, Any],
        timestamp: datetime | None = None,
        emit: bool = True,
    ) -> ProcessedSignalBatch:
        normalized = self.normalizer.normalize_event(
            event_name=event_name,
            payload=payload,
            timestamp=timestamp,
        )

        context = self._resolve_context(normalized)
        self.normalizer.apply_to_context(context, normalized)
        self.state.update_context(context)

        route = self.router.route(
            event_name=event_name,
            context=context,
            source=normalized.source,
            changed_features=[feature.name for feature in normalized.features],
            metadata=normalized.metadata,
        )

        batch = ProcessedSignalBatch(
            symbol=normalized.symbol,
            timestamp=context.timestamp,
            normalized=normalized,
            context=context,
            route=route,
            metadata={
                "event_name": event_name,
                "source_topic": event_name,
                "source": normalized.source.value,
                "timeframe": normalized.timeframe.value,
                "normalized": dict(normalized.metadata),
            },
        )

        self._update_batch_debug(
            batch,
            failure_stage="routing",
            reason="initial",
            payload=payload,
        )

        if route.is_empty:
            batch.reasons.append("no_strategies_routed")
            self.state.metrics.record_applicability_skip()
            self._update_batch_debug(
                batch,
                failure_stage="routing",
                reason="no_strategies_routed",
                payload=payload,
            )
            if emit:
                await self._emit_rejected_batch(batch, reason="no_strategies_routed")
            return batch

        evaluations = await self.evaluate_strategies(
            strategies=route.selected,
            context=context,
        )
        batch.evaluations = evaluations

        raw_signals: list[StrategySignal] = []
        for evaluation in evaluations:
            if evaluation.signal is not None and evaluation.passed:
                raw_signals.append(evaluation.signal)

        batch.raw_signals = raw_signals

        if not raw_signals:
            batch.reasons.append("no_passed_strategy_signals")
            self._update_batch_debug(
                batch,
                failure_stage="strategy_evaluation",
                reason="no_passed_strategy_signals",
                payload=payload,
            )
            if emit:
                await self._emit_rejected_batch(batch, reason="no_passed_strategy_signals")
            return batch

        scored = self.scorer.score_many(signals=raw_signals, context=context)

        filtered = self.filters.apply(signals=scored, context=context)
        batch.filtered_signals = filtered

        if not filtered:
            batch.reasons.append("all_signals_filtered")
            self._update_batch_debug(
                batch,
                failure_stage="filters",
                reason="all_signals_filtered",
                payload=payload,
                extra={"raw_signals": self._signals_debug(raw_signals)},
            )
            if emit:
                await self._emit_rejected_batch(batch, reason="all_signals_filtered")
            return batch

        confluence = self.confluence.evaluate(signals=filtered, context=context)
        batch.confluence = confluence

        confluence_signals = self._signals_after_confluence(confluence, filtered)
        if not confluence_signals:
            batch.reasons.append("confluence_rejected")
            self._update_batch_debug(
                batch,
                failure_stage="confluence",
                reason="confluence_rejected",
                payload=payload,
                extra={
                    "raw_signals": self._signals_debug(raw_signals),
                    "filtered_signals": self._signals_debug(filtered),
                    "confluence": self._confluence_debug(confluence),
                },
            )
            if emit:
                await self._emit_rejected_batch(batch, reason="confluence_rejected")
            return batch

        built_signals, build_rejected = self.builder.build_many(
            signals=confluence_signals,
            context=context,
        )

        if build_rejected:
            batch.metadata["build_rejected"] = build_rejected

        if not built_signals:
            batch.reasons.append("all_signals_failed_builder")
            self._update_batch_debug(
                batch,
                failure_stage="signal_builder",
                reason="all_signals_failed_builder",
                payload=payload,
                extra={
                    "raw_signals": self._signals_debug(raw_signals),
                    "filtered_signals": self._signals_debug(filtered),
                    "confluence_signals": self._signals_debug(confluence_signals),
                    "build_rejected": build_rejected,
                },
            )
            if emit:
                await self._emit_rejected_batch(batch, reason="all_signals_failed_builder")
            return batch

        coordinated = self.portfolio.coordinate(
            signals=built_signals,
            context=context,
        )
        batch.coordinated = coordinated
        batch.final_signals = list(coordinated.final_signals)

        if not coordinated.accepted or not batch.final_signals:
            batch.metadata["coordination_rejected"] = {
                "reasons": list(coordinated.reasons),
                "rejected_signals": dict(coordinated.rejected_signals),
                "throttled_signals": dict(coordinated.throttled_signals),
                "suppressed_signals": dict(coordinated.suppressed_signals),
                "raw_signal_count": len(coordinated.raw_signals),
                "accepted_signal_count": len(coordinated.accepted_signals),
                "merged_signal_count": len(coordinated.merged_signals),
            }
            batch.reasons.append("portfolio_coordination_rejected")
            self._update_batch_debug(
                batch,
                failure_stage="portfolio_coordination",
                reason="portfolio_coordination_rejected",
                payload=payload,
                extra={
                    "built_signals": self._signals_debug(built_signals),
                    "coordination": self._coordination_debug(coordinated),
                },
            )
            if emit:
                await self._emit_rejected_batch(batch, reason="portfolio_coordination_rejected")
            return batch

        risk_payloads: list[RiskReadySignalPayload] = []

        try:
            for signal in batch.final_signals:
                self.builder.assert_risk_ready(signal)

                risk_payload = self.to_risk_payload(
                    signal=signal,
                    context=context,
                )
                risk_payloads.append(risk_payload)
        except (SignalRoutingError, BuilderError, ValueError, TypeError, AttributeError) as exc:
            batch.reasons.append("risk_payload_build_failed")
            batch.metadata["risk_payload_error"] = {
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
            self._update_batch_debug(
                batch,
                failure_stage="risk_payload_builder",
                reason="risk_payload_build_failed",
                payload=payload,
                extra={
                    "final_signals": self._signals_debug(batch.final_signals),
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            )
            if emit:
                await self._emit_rejected_batch(batch, reason="risk_payload_build_failed")
            return batch

        batch.risk_payloads = risk_payloads
        batch.accepted = bool(risk_payloads)

        if not batch.accepted:
            batch.reasons.append("no_risk_payloads_built")
            self._update_batch_debug(
                batch,
                failure_stage="risk_payload_builder",
                reason="no_risk_payloads_built",
                payload=payload,
                extra={"final_signals": self._signals_debug(batch.final_signals)},
            )
            if emit:
                await self._emit_rejected_batch(batch, reason="no_risk_payloads_built")
            return batch

        if emit:
            for risk_payload in risk_payloads:
                await self.router.emit_signal_generated(payload=risk_payload)
            batch.emitted = True

        self._record_final_signals(batch.final_signals)

        return batch

    async def evaluate_strategies(
        self,
        *,
        strategies: list[BaseStrategy],
        context: StrategyContext,
    ) -> list[StrategyEvaluation]:
        result: list[StrategyEvaluation] = []

        for strategy in strategies:
            try:
                evaluation_result = strategy.evaluate(context)
                if inspect.isawaitable(evaluation_result):
                    evaluation_result = await evaluation_result

                evaluation_result.validate()
                result.append(evaluation_result)
                self.state.update_evaluation(evaluation_result)

            except (
                StrategyEvaluationError,
                ValueError,
                TypeError,
                AttributeError,
                RuntimeError,
            ) as exc:
                strategy_name = getattr(strategy, "strategy_name", strategy.__class__.__name__)

                self.log_exception(
                    "Strategy evaluation failed",
                    strategy_name=strategy_name,
                    symbol=context.symbol,
                    error=str(exc),
                )

                failed = StrategyEvaluation(
                    strategy_name=strategy_name,
                    symbol=context.symbol,
                    timestamp=context.timestamp,
                    passed=False,
                    reasons=[f"strategy_evaluation_error:{exc}"],
                    metadata={
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    },
                )
                failed.validate()
                result.append(failed)
                self.state.metrics.record_error(strategy_name=strategy_name)

        return result

    def to_risk_payload(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> RiskReadySignalPayload:
        """
        Convert final built StrategySignal into signal.generated payload.
        """
        signal.validate()
        context.validate()
        self.builder.assert_risk_ready(signal)

        execution = signal.execution_plan
        if execution is None:
            raise SignalRoutingError("signal.execution_plan is required")

        entry_price = signal.primary_entry_price
        stop_loss = signal.primary_stop_loss
        take_profit = signal.primary_take_profit

        if entry_price is None or stop_loss is None:
            raise SignalRoutingError("risk payload requires entry_price and stop_loss")

        priority_score = _to_float(
            signal.metadata.get("priority_score"),
            max(signal.score, signal.confidence),
        )
        if priority_score is None:
            priority_score = max(signal.score, signal.confidence)

        tier = _parse_enum(
            StrategyTradeTier,
            signal.metadata.get("tier"),
            self._tier_from_priority_score(priority_score),
        )
        order_intent = _parse_enum(
            StrategyOrderIntent,
            signal.metadata.get("order_intent"),
            StrategyOrderIntent.OPEN,
        )
        liquidity_class = _parse_enum(
            StrategyLiquidityClass,
            signal.metadata.get("liquidity_class"),
            StrategyLiquidityClass.NORMAL,
        )
        execution_quality = _parse_enum(
            StrategyExecutionQuality,
            signal.metadata.get("execution_quality"),
            StrategyExecutionQuality.ACCEPTABLE,
        )
        margin_mode = _parse_enum(
            StrategyMarginMode,
            signal.metadata.get("margin_mode"),
            StrategyMarginMode.ISOLATED,
        )
        market_type = _parse_enum(
            StrategyMarketType,
            signal.metadata.get("market_type") or context.metadata.get("market_type"),
            StrategyMarketType.USDM_FUTURES,
        )

        execution_cost = self._execution_cost_from_metadata(signal)

        payload = RiskReadySignalPayload(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            side=signal.side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=signal.strategy_name,
            tier=tier,
            order_intent=order_intent,
            confidence=signal.confidence,
            edge_score=clamp(signal.score, 0.0, 1.0),
            priority_score=priority_score,
            liquidity_class=liquidity_class,
            execution_quality=execution_quality,
            expected_reward=_to_float(signal.metadata.get("expected_reward")),
            expected_loss=_to_float(signal.metadata.get("expected_loss")),
            expected_win_probability=_to_float(signal.metadata.get("expected_win_probability")),
            expected_cost=_to_float(signal.metadata.get("expected_cost")),
            execution_cost=execution_cost,
            requested_size=_to_float(signal.metadata.get("requested_size")),
            requested_margin=_to_float(signal.metadata.get("requested_margin")),
            requested_leverage=execution.leverage or _to_float(signal.metadata.get("requested_leverage")),
            reduce_only=bool(getattr(execution, "reduce_only", False)),
            margin_mode=margin_mode,
            exchange=signal.metadata.get("exchange") or context.metadata.get("exchange"),
            market_type=market_type,
            timeframe=signal.timeframe,
            timestamp=signal.timestamp,
            metadata={
                "category": signal.category.value,
                "setup_type": signal.setup_type.value,
                "trigger_type": signal.trigger_type.value,
                "origin": signal.origin.value,
                "priority": signal.priority.value,
                "strength": signal.strength.value,
                "confidence_grade": signal.confidence_grade.value,
                "regime": signal.regime.value if isinstance(signal.regime, MarketRegime) else str(signal.regime),
                "reasons": list(signal.reasons),
                "confirmations": list(signal.confirmations),
                "source_features": list(signal.source_features),
                "combined_from": list(signal.combined_from),
                "processor": self.component_name,
                **dict(signal.metadata),
            },
        )
        payload.validate()
        return payload

    @staticmethod
    def _enum_value(value: Any) -> Any:
        if hasattr(value, "value"):
            return value.value
        return value

    @staticmethod
    def _safe_iso(value: Any) -> str | None:
        if isinstance(value, datetime):
            return ensure_aware_utc(value).isoformat()
        return None

    def _evaluation_debug(
        self,
        evaluations: list[StrategyEvaluation],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []

        for evaluation in evaluations:
            signal = getattr(evaluation, "signal", None)
            signal_side = getattr(signal, "side", None) if signal is not None else None

            result.append(
                {
                    "strategy_name": getattr(evaluation, "strategy_name", None),
                    "symbol": getattr(evaluation, "symbol", None),
                    "passed": getattr(evaluation, "passed", None),
                    "score": getattr(evaluation, "score", None),
                    "confidence": getattr(evaluation, "confidence", None),
                    "reasons": list(getattr(evaluation, "reasons", []) or []),
                    "metadata": dict(getattr(evaluation, "metadata", {}) or {}),
                    "signal_id": getattr(signal, "signal_id", None) if signal is not None else None,
                    "signal_strategy_name": getattr(signal, "strategy_name", None) if signal is not None else None,
                    "signal_side": self._enum_value(signal_side),
                    "signal_score": getattr(signal, "score", None) if signal is not None else None,
                    "signal_confidence": getattr(signal, "confidence", None) if signal is not None else None,
                }
            )

        return result

    def _route_debug(self, route: RouteDecision | None) -> dict[str, Any]:
        if route is None:
            return {}

        return {
            "event_name": route.event_name,
            "symbol": route.symbol,
            "source": route.source.value if route.source is not None else None,
            "selected_names": list(route.selected_names),
            "selected_count": route.total_selected,
            "categories_used": [
                item.value if hasattr(item, "value") else str(item)
                for item in route.categories_used
            ],
            "matched_features": list(route.matched_features),
            "skipped": dict(route.skipped),
            "metadata": dict(route.metadata),
        }

    def _context_debug(self, context: StrategyContext | None) -> dict[str, Any]:
        if context is None:
            return {}

        raw_features = list(getattr(context, "features", []) or [])
        domain_data = getattr(context, "domain_data", {}) or {}

        feature_names: list[str] = []
        for feature in raw_features[:200]:
            feature_names.append(str(getattr(feature, "name", feature)))

        domain_keys: dict[str, list[str]] = {}
        if isinstance(domain_data, dict):
            for source, value in domain_data.items():
                source_key = source.value if hasattr(source, "value") else str(source)
                if isinstance(value, dict):
                    domain_keys[source_key] = sorted(str(key) for key in value.keys())
                else:
                    domain_keys[source_key] = []

        return {
            "symbol": getattr(context, "symbol", None),
            "timeframe": (
                context.timeframe.value
                if hasattr(getattr(context, "timeframe", None), "value")
                else str(getattr(context, "timeframe", None))
            ),
            "timestamp": self._safe_iso(getattr(context, "timestamp", None)),
            "feature_count": len(raw_features),
            "feature_names": feature_names,
            "domain_sources": sorted(domain_keys.keys()),
            "domain_keys": domain_keys,
        }

    def _normalized_debug(self, normalized: NormalizedPayload | None) -> dict[str, Any]:
        if normalized is None:
            return {}

        return {
            "source": normalized.source.value,
            "symbol": normalized.symbol,
            "timeframe": normalized.timeframe.value,
            "timestamp": normalized.timestamp.isoformat(),
            "domain_keys": sorted(normalized.domain_data.keys()),
            "extra_domain_sources": sorted(
                source.value if hasattr(source, "value") else str(source)
                for source in normalized.extra_domain_data.keys()
            ),
            "feature_names": [feature.name for feature in normalized.features[:200]],
            "feature_count": len(normalized.features),
            "metadata": dict(normalized.metadata),
        }

    def _signals_debug(self, signals: list[StrategySignal]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []

        for signal in signals:
            result.append(
                {
                    "signal_id": getattr(signal, "signal_id", None),
                    "strategy_name": getattr(signal, "strategy_name", None),
                    "symbol": getattr(signal, "symbol", None),
                    "side": self._enum_value(getattr(signal, "side", None)),
                    "setup_type": self._enum_value(getattr(signal, "setup_type", None)),
                    "score": getattr(signal, "score", None),
                    "confidence": getattr(signal, "confidence", None),
                    "status": self._enum_value(getattr(signal, "status", None)),
                    "reasons": list(getattr(signal, "reasons", []) or []),
                    "confirmations": list(getattr(signal, "confirmations", []) or []),
                    "metadata": dict(getattr(signal, "metadata", {}) or {}),
                    "entry": getattr(signal, "primary_entry_price", None),
                    "stop_loss": getattr(signal, "primary_stop_loss", None),
                    "take_profit": getattr(signal, "primary_take_profit", None),
                }
            )

        return result

    def _confluence_debug(self, confluence: ConfluenceEvaluation | None) -> dict[str, Any]:
        if confluence is None:
            return {}

        return {
            "accepted": confluence.accepted,
            "selected_strategy_names": confluence.selected_strategy_names,
            "raw_signal_count": len(confluence.raw_signals),
            "eligible_signal_count": len(confluence.eligible_signals),
            "accepted_signal_count": len(confluence.accepted_signals),
            "rejected_signals": dict(confluence.rejected_signals),
            "reasons": list(confluence.reasons),
            "metadata": dict(confluence.metadata),
            "result": confluence.result.to_dict() if getattr(confluence.result, "to_dict", None) else None,
            "merged_signal": self._signals_debug([confluence.merged_signal]) if confluence.merged_signal else [],
        }

    def _coordination_debug(self, coordinated: CoordinationDecision | None) -> dict[str, Any]:
        if coordinated is None:
            return {}

        return {
            "accepted": coordinated.accepted,
            "selected_names": coordinated.selected_names,
            "raw_signal_count": len(coordinated.raw_signals),
            "accepted_signal_count": len(coordinated.accepted_signals),
            "merged_signal_count": len(coordinated.merged_signals),
            "final_signal_count": len(coordinated.final_signals),
            "rejected_signals": dict(coordinated.rejected_signals),
            "throttled_signals": dict(coordinated.throttled_signals),
            "suppressed_signals": dict(coordinated.suppressed_signals),
            "reasons": list(coordinated.reasons),
            "metadata": dict(coordinated.metadata),
        }

    def _update_batch_debug(
        self,
        batch: ProcessedSignalBatch,
        *,
        failure_stage: str,
        reason: str,
        payload: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        batch.debug.update(
            {
                "failure_stage": failure_stage,
                "reason": reason,
                "source_topic": batch.metadata.get("source_topic") or batch.metadata.get("event_name"),
                "event_name": batch.metadata.get("event_name"),
                "timeframe": (
                    batch.normalized.timeframe.value
                    if batch.normalized is not None
                    else None
                ),
                "symbol": batch.symbol,
                "timestamp": batch.timestamp.isoformat(),
                "timestamp_ms": int(batch.timestamp.timestamp() * 1000),
                "payload_keys": sorted((payload or {}).keys()),
                "normalized": self._normalized_debug(batch.normalized),
                "route": self._route_debug(batch.route),
                "context": self._context_debug(batch.context),
                "evaluations": self._evaluation_debug(batch.evaluations),
                "raw_signals": self._signals_debug(batch.raw_signals),
                "filtered_signals": self._signals_debug(batch.filtered_signals),
                "final_signals": self._signals_debug(batch.final_signals),
            }
        )

        if batch.confluence is not None:
            batch.debug["confluence"] = self._confluence_debug(batch.confluence)

        if batch.coordinated is not None:
            batch.debug["coordination"] = self._coordination_debug(batch.coordinated)

        if extra:
            batch.debug.update(extra)

    def _resolve_context(self, normalized: NormalizedPayload) -> StrategyContext:
        existing = self.state.contexts.get_context(normalized.symbol)
        if existing is not None:
            existing.timestamp = normalized.timestamp
            existing.timeframe = normalized.timeframe
            existing.validate()
            return existing

        return self.state.build_context(
            normalized.symbol,
            timestamp=normalized.timestamp,
            timeframe=normalized.timeframe,
            include_regime=True,
            include_portfolio=True,
        )

    @staticmethod
    def _signals_after_confluence(
        confluence: ConfluenceEvaluation,
        fallback: list[StrategySignal],
    ) -> list[StrategySignal]:
        if confluence.result is None:
            return list(confluence.accepted_signals or fallback)

        if not confluence.result.accepted:
            return []

        if confluence.merged_signal is not None:
            return [confluence.merged_signal]

        return list(confluence.accepted_signals)

    async def _emit_rejected_batch(
        self,
        batch: ProcessedSignalBatch,
        *,
        reason: str,
    ) -> None:
        if self.event_bus is None:
            return

        normalized = batch.normalized
        route = batch.route

        payload = {
            "signal_id": None,
            "symbol": batch.symbol,
            "strategy_name": None,
            "reason": reason,
            "reasons": list(batch.reasons),
            "timestamp": batch.timestamp.isoformat(),
            "timestamp_ms": int(batch.timestamp.timestamp() * 1000),
            "received_at_ms": int(utcnow().timestamp() * 1000),
            "timeframe": normalized.timeframe.value if normalized is not None else None,
            "source": normalized.source.value if normalized is not None else None,
            "source_topic": batch.metadata.get("source_topic") or batch.metadata.get("event_name"),
            "route": route.selected_names if route is not None else [],
            "selected_strategies": route.selected_names if route is not None else [],
            "route_skipped": dict(route.skipped) if route is not None else {},
            "evaluation_reasons": [
                {
                    "strategy_name": getattr(evaluation, "strategy_name", None),
                    "passed": getattr(evaluation, "passed", None),
                    "score": getattr(evaluation, "score", None),
                    "confidence": getattr(evaluation, "confidence", None),
                    "reasons": list(getattr(evaluation, "reasons", []) or []),
                    "metadata": dict(getattr(evaluation, "metadata", {}) or {}),
                    "signal_id": getattr(getattr(evaluation, "signal", None), "signal_id", None),
                }
                for evaluation in batch.evaluations
            ],
            "debug": dict(batch.debug),
            "metadata": dict(batch.metadata),
        }

        await self.emit_event(
            "signal.rejected",
            payload,
            priority=EventPriority.LOW,
            source=self.component_name,
        )

    def _record_final_signals(self, signals: list[StrategySignal]) -> None:
        for signal in signals:
            signal.to_pending()
            self.state.update_signal(signal, active=True)
            self.state.metrics.record_signal(signal)

    @staticmethod
    def _execution_cost_from_metadata(
        signal: StrategySignal,
    ) -> ExecutionCostPayload | None:
        raw = signal.metadata.get("execution_cost")

        if raw is None:
            return None

        if isinstance(raw, ExecutionCostPayload):
            raw.validate()
            return raw

        if isinstance(raw, dict):
            quality = _parse_enum(
                StrategyExecutionQuality,
                raw.get("quality"),
                StrategyExecutionQuality.ACCEPTABLE,
            )

            payload = ExecutionCostPayload(
                spread_cost=_to_float(raw.get("spread_cost"), 0.0) or 0.0,
                slippage_cost=_to_float(raw.get("slippage_cost"), 0.0) or 0.0,
                fee_cost=_to_float(raw.get("fee_cost"), 0.0) or 0.0,
                funding_cost=_to_float(raw.get("funding_cost"), 0.0) or 0.0,
                other_cost=_to_float(raw.get("other_cost"), 0.0) or 0.0,
                spread_pct=_to_float(raw.get("spread_pct")),
                slippage_pct=_to_float(raw.get("slippage_pct")),
                quality=quality,
                metadata=dict(raw.get("metadata") or {}),
            )
            payload.validate()
            return payload

        raise SignalRoutingError("execution_cost metadata must be dict or ExecutionCostPayload")

    def _confidence_grade(self, confidence: float) -> ConfidenceGrade:
        value = clamp(confidence, 0.0, 1.0)
        very_low, low, medium, high = self.config.confidence.grade_bounds()

        if value >= high:
            return ConfidenceGrade.VERY_HIGH
        if value >= medium:
            return ConfidenceGrade.HIGH
        if value >= low:
            return ConfidenceGrade.MEDIUM
        if value >= very_low:
            return ConfidenceGrade.LOW
        return ConfidenceGrade.VERY_LOW

    def _confidence_strength(self, confidence: float) -> SignalStrength:
        value = clamp(confidence, 0.0, 1.0)
        _, low, medium, high = self.config.confidence.grade_bounds()

        if value >= high:
            return SignalStrength.EXTREME
        if value >= medium:
            return SignalStrength.STRONG
        if value >= low:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK

    @staticmethod
    def _tier_from_priority_score(value: float) -> StrategyTradeTier:
        score = clamp(value, 0.0, 1.0)

        if score >= 0.88:
            return StrategyTradeTier.T4
        if score >= 0.74:
            return StrategyTradeTier.T3
        if score >= 0.58:
            return StrategyTradeTier.T2
        return StrategyTradeTier.T1


__all__ = [
    "NormalizedPayload",
    "RouteDecision",
    "WeightedSignal",
    "VoteSummary",
    "ConflictSummary",
    "ConfluenceEvaluation",
    "FilterEvaluation",
    "BuildEvaluation",
    "CoordinationDecision",
    "ProcessedSignalBatch",
    "SignalNormalizer",
    "SignalRouter",
    "SignalScorer",
    "ConfluenceEngine",
    "SignalFilterChain",
    "SignalBuilder",
    "PortfolioCoordinator",
    "SignalProcessor",
]