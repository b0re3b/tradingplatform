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
    ConflictType,
    EntryType,
    ExitType,
    FeatureSource,
    FilterDecision,
    MarketRegime,
    SignalOrigin,
    SignalSide,
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

        for target, aliases in {
            "snapshot": ("snapshot", "funding_snapshot"),
            "statistics": ("statistics", "stats", "funding_statistics"),
            "regime": ("regime", "regime_state", "funding_regime"),
            "pressure": ("pressure", "pressure_state", "funding_pressure"),
            "extreme": ("extreme", "extreme_event", "funding_extreme"),
            "divergence": ("divergence", "divergence_event", "funding_divergence"),
            "flip": ("flip", "flip_event", "funding_flip"),
            "signal": ("signal", "funding_signal"),
        }.items():
            value = mapping_for(*aliases)
            if value is not None:
                domain_data.setdefault(target, value)
                domain_data.setdefault(aliases[0], value)

        if "signal" not in domain_data and (
                "signal_type" in payload
                or "bias" in payload
                or "score" in payload
                or "confidence" in payload
        ):
            domain_data["signal"] = dict(payload)

        if "snapshot" not in domain_data:
            domain_data["snapshot"] = dict(payload)

    def _augment_orderflow_domain_data(
            self,
            *,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        composite = payload.get("composite") or payload.get("snapshot") or payload.get("orderflow")
        cvd = payload.get("cvd") or payload.get("cvd_stats")
        volume_delta = payload.get("volume_delta") or payload.get("volume_delta_stats")
        aggressive = payload.get("aggressive_trades") or payload.get("aggressive")
        imbalance = payload.get("orderbook_imbalance") or payload.get("imbalance")

        if isinstance(composite, dict):
            domain_data.setdefault("composite", composite)
            domain_data.setdefault("snapshot", composite)

        if isinstance(cvd, dict):
            domain_data.setdefault("cvd", cvd)

        if isinstance(volume_delta, dict):
            domain_data.setdefault("volume_delta", volume_delta)

        if isinstance(aggressive, dict):
            domain_data.setdefault("aggressive_trades", aggressive)

        if isinstance(imbalance, dict):
            domain_data.setdefault("orderbook_imbalance", imbalance)

        # Якщо analytics дає flat aggregate payload, нехай він теж буде composite.
        if "composite" not in domain_data:
            domain_data["composite"] = dict(payload)

    def _augment_open_interest_domain_data(
            self,
            *,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        """
        Add stable open-interest domain aliases expected by
        strategy/strategies/open_interest/*.

        This method does not detect OI regimes/anomalies/divergences by itself.
        It only adapts analytics.oi.* payload shapes into the StrategyContext
        contract consumed by OpenInterestTradingStrategy:

            context.domain_dict(FeatureSource.OPEN_INTEREST)["analysis"]
            context.domain_dict(FeatureSource.OPEN_INTEREST)["features"]
            context.domain_dict(FeatureSource.OPEN_INTEREST)["regime"]
            context.domain_dict(FeatureSource.OPEN_INTEREST)["divergence"]
            context.domain_dict(FeatureSource.OPEN_INTEREST)["anomaly"]
            context.domain_dict(FeatureSource.OPEN_INTEREST)["snapshot"]
            context.domain_dict(FeatureSource.OPEN_INTEREST)["market_context"]

        Supported input shapes:
            - full analytics payload with analysis/features/regime/divergence/anomaly
            - event-specific payload with regime_result/divergence_result/anomaly_result
            - flat payload with oi_delta_pct, anomaly_type, divergence_type, etc.
            - payload["feature_map"] carrying the same flat/nested data
        """
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

            domain_data.setdefault(target, value)
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

        # If a full analysis object is present, expose its nested pieces too.
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
                snapshot = nested_snapshot
            if isinstance(nested_context, dict) and market_context is None:
                market_context = nested_context
            if isinstance(nested_features, dict) and features is None:
                features = nested_features
            if isinstance(nested_regime, dict) and regime is None:
                regime = nested_regime
            if isinstance(nested_divergence, dict) and divergence is None:
                divergence = nested_divergence
            if isinstance(nested_anomaly, dict) and anomaly is None:
                anomaly = nested_anomaly

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
            features,
        )
        set_aliases(
            "regime",
            ("regime_result", "oi_regime", "open_interest_regime", "new_regime"),
            regime,
        )
        set_aliases(
            "divergence",
            ("divergence_result", "oi_divergence", "open_interest_divergence"),
            divergence,
        )
        set_aliases(
            "anomaly",
            ("anomaly_result", "oi_anomaly", "open_interest_anomaly"),
            anomaly,
        )

        # Build strategy-friendly fallback features from flat analytics payload.
        if "features" not in domain_data:
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
                domain_data["features"] = {
                    "oi": value_for("oi", "open_interest"),
                    "open_interest": value_for("open_interest", "oi"),
                    "open_interest_value": value_for("open_interest_value"),
                    "oi_delta": value_for("oi_delta"),
                    "oi_delta_pct": value_for("oi_delta_pct"),
                    "oi_direction": value_for("oi_direction"),
                    "oi_acceleration": value_for("oi_acceleration"),
                    "price_delta_pct": value_for("price_delta_pct"),
                    "volume_ratio": value_for("volume_ratio"),
                    "oi_zscore": value_for("oi_zscore"),
                    "oi_pressure_score": value_for("oi_pressure_score"),
                    "funding_rate": value_for("funding_rate"),
                    "liquidation_pressure": value_for(
                        "liquidation_pressure",
                        "liquidation_imbalance",
                    ),
                    "liquidation_imbalance": value_for(
                        "liquidation_imbalance",
                        "liquidation_pressure",
                    ),
                    "aggressive_flow_imbalance": value_for(
                        "aggressive_flow_imbalance",
                    ),
                    "oi_price_efficiency": value_for("oi_price_efficiency"),
                }

        if "regime" not in domain_data:
            regime_value = value_for(
                "regime",
                "oi_regime",
                "market_regime",
                "new_regime",
            )
            if regime_value is not None:
                domain_data["regime"] = {
                    "regime": regime_value,
                    "confidence": value_for(
                        "regime_confidence",
                        "confidence",
                        default=0.0,
                    ),
                    "score": value_for("regime_score", "score", default=0.0),
                    "reasons": value_for("regime_reasons", "reasons", default=[]),
                }

        if "divergence" not in domain_data:
            divergence_type = value_for(
                "divergence_type",
                "price_oi_divergence",
            )
            divergence_detected = value_for(
                "divergence_detected",
                "is_divergence",
                "divergence",
                default=None,
            )

            if divergence_type is not None or divergence_detected is not None:
                domain_data["divergence"] = {
                    "detected": bool(
                        True if divergence_detected is None else divergence_detected
                    ),
                    "divergence_type": divergence_type,
                    "confidence": value_for(
                        "divergence_confidence",
                        "confidence",
                        default=0.0,
                    ),
                    "score": value_for("divergence_score", "score", default=0.0),
                    "window_size": value_for("divergence_window_size", "window_size"),
                    "reasons": value_for(
                        "divergence_reasons",
                        "reasons",
                        default=[],
                    ),
                }

        if "anomaly" not in domain_data:
            anomaly_type = value_for("anomaly_type")
            anomaly_detected = value_for(
                "anomaly_detected",
                "is_anomaly",
                "anomaly",
                "capitulation",
                "squeeze_setup",
                default=None,
            )

            if anomaly_type is not None or anomaly_detected is not None:
                domain_data["anomaly"] = {
                    "detected": bool(
                        True if anomaly_detected is None else anomaly_detected
                    ),
                    "anomaly_type": anomaly_type,
                    "confidence": value_for(
                        "anomaly_confidence",
                        "confidence",
                        default=0.0,
                    ),
                    "score": value_for("anomaly_score", "score", default=0.0),
                    "strength": value_for(
                        "anomaly_strength",
                        "strength",
                        default=value_for("score", default=0.0),
                    ),
                    "capitulation": bool(value_for("capitulation", default=False)),
                    "capitulation_score": value_for(
                        "capitulation_score",
                        default=value_for("score", default=0.0),
                    ),
                    "squeeze_setup": bool(value_for("squeeze_setup", default=False)),
                    "squeeze_score": value_for(
                        "squeeze_score",
                        default=value_for("score", default=0.0),
                    ),
                    "liquidation_imbalance": value_for(
                        "liquidation_imbalance",
                        "liquidation_pressure",
                    ),
                    "reasons": value_for("anomaly_reasons", "reasons", default=[]),
                }

        # Keep a full raw fallback for debugging / metadata, but do not overwrite
        # canonical nested contracts above.
        domain_data.setdefault("raw", dict(payload))

    def _augment_liquidations_domain_data(
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

        for target, aliases in {
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
            ),
            "signal": (
                    "signal",
                    "liquidation_signal",
                    "analytics_signal",
            ),
        }.items():
            value = mapping_for(*aliases)
            if value is not None:
                domain_data.setdefault(target, value)
                for alias in aliases:
                    domain_data.setdefault(alias, value)

        # analytics.liquidations.cascade_detected often arrives as flat payload.
        if "cascade" not in domain_data and (
                "direction" in payload
                or "cascade_direction" in payload
                or "intensity_score" in payload
                or "total_notional_usd" in payload
                or "event_count" in payload
        ):
            domain_data["cascade"] = dict(payload)
            domain_data.setdefault("result", domain_data["cascade"])

        # analytics.liquidations.exhaustion_detected may also be flat.
        if "exhaustion" not in domain_data and (
                "exhaustion_bias" in payload
                or "bias_delta" in payload
                or "reversal_side" in payload
        ):
            domain_data["exhaustion"] = dict(payload)
            domain_data.setdefault("exhaustion_result", domain_data["exhaustion"])

        # Squeeze reversal context may be flat.
        if "squeeze" not in domain_data and (
                "squeeze_score" in payload
                or "squeeze_confirmed" in payload
                or "squeeze_direction" in payload
        ):
            domain_data["squeeze"] = dict(payload)
            domain_data.setdefault("squeeze_result", domain_data["squeeze"])

        domain_data.setdefault("raw", dict(payload))

    def _augment_liquidity_domain_data(
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

        for target, aliases in {
            "snapshot": (
                    "snapshot",
                    "liquidity_map_snapshot",
                    "map_snapshot",
                    "last_snapshot",
                    "liquidity",
                    "result",
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
            "sweep_risk": (
                    "sweep_risk",
                    "sweep_risks",
            ),
            "magnet": (
                    "magnet",
                    "magnets",
                    "liquidity_magnets",
            ),
        }.items():
            value = mapping_for(*aliases)
            if value is not None:
                domain_data.setdefault(target, value)
                for alias in aliases:
                    domain_data.setdefault(alias, value)

        # Flat liquidity map update fallback.
        if "snapshot" not in domain_data and (
                "above_liquidity_score" in payload
                or "below_liquidity_score" in payload
                or "liquidity_pressure_score" in payload
                or "bias" in payload
                or "active_levels" in payload
                or "stop_clusters" in payload
                or "zones" in payload
        ):
            domain_data["snapshot"] = dict(payload)
            domain_data.setdefault("map_snapshot", domain_data["snapshot"])
            domain_data.setdefault("liquidity_map_snapshot", domain_data["snapshot"])

        if "signal" not in domain_data and (
                "bias" in payload
                or "score" in payload
                or "confidence" in payload
                or "above_liquidity_score" in payload
                or "below_liquidity_score" in payload
        ):
            domain_data["signal"] = dict(payload)

        current_price = None
        for key in ("current_price", "price", "mark_price", "last_price", "close"):
            if key in payload:
                current_price = payload[key]
                break
            if key in feature_map:
                current_price = feature_map[key]
                break

        if current_price is not None:
            domain_data.setdefault("current_price", current_price)
            domain_data.setdefault("price", current_price)

        domain_data.setdefault("raw", dict(payload))

    def _augment_price_action_domain_data(
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

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]
            return default

        state = mapping_for(
            "state",
            "composite",
            "price_action",
            "snapshot",
            "result",
        )

        if state is not None:
            domain_data.setdefault("state", state)
            domain_data.setdefault("composite", state)
            domain_data.setdefault("price_action", state)
            domain_data.setdefault("snapshot", state)

        for target, aliases in {
            "market_structure": (
                    "market_structure",
                    "structure",
                    "ms",
            ),
            "support_resistance": (
                    "support_resistance",
                    "sr",
                    "levels",
            ),
            "fair_value_gap": (
                    "fair_value_gap",
                    "fvg",
                    "fair_value_gaps",
            ),
            "trend": (
                    "trend",
                    "trend_state",
            ),
            "liquidity_levels": (
                    "liquidity_levels",
                    "liquidity",
            ),
        }.items():
            value = mapping_for(*aliases)

            if value is None and isinstance(state, dict):
                for alias in aliases:
                    nested = state.get(alias)
                    if isinstance(nested, dict):
                        value = nested
                        break

            if value is not None:
                domain_data.setdefault(target, value)
                for alias in aliases:
                    domain_data.setdefault(alias, value)

        # Event-specific fallbacks for analytics.price_action.<module>.<event>
        event_payload = dict(payload)

        topic = (
            str(payload.get("event_name") or payload.get("topic") or "")
            .strip()
            .lower()
        )

        if "market_structure" not in domain_data and (
                "market_structure" in topic
                or any(
            key in payload
            for key in (
                    "swing_type",
                    "event_type",
                    "break_distance_pct",
                    "market_bias",
                    "bias",
                    "mtf_alignment",
            )
        )
        ):
            domain_data["market_structure"] = event_payload
            domain_data.setdefault("structure", event_payload)

        if "support_resistance" not in domain_data and (
                "support_resistance" in topic
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
        ):
            domain_data["support_resistance"] = event_payload
            domain_data.setdefault("sr", event_payload)

        if "fair_value_gap" not in domain_data and (
                "fair_value_gap" in topic
                or ".fvg_" in topic
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
        ):
            domain_data["fair_value_gap"] = event_payload
            domain_data.setdefault("fvg", event_payload)

        if "trend" not in domain_data and (
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
        ):
            domain_data["trend"] = event_payload

        if "liquidity_levels" not in domain_data and (
                "liquidity_levels" in topic
                or any(
            key in payload
            for key in (
                    "liquidity_touched",
                    "liquidity_swept",
                    "liquidity_reclaimed",
                    "stop_run",
                    "failed_breakout",
            )
        )
        ):
            domain_data["liquidity_levels"] = event_payload

        current_price = value_for(
            "current_price",
            "price",
            "last_price",
            "close",
            default=None,
        )
        if current_price is not None:
            domain_data.setdefault("current_price", current_price)
            domain_data.setdefault("last_price", current_price)
            domain_data.setdefault("price", current_price)

        timestamp_value = value_for(
            "timestamp",
            "event_time",
            "updated_at",
            "created_at",
            default=None,
        )
        if timestamp_value is not None:
            domain_data.setdefault("timestamp", timestamp_value)

        domain_data.setdefault("raw", dict(payload))

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

        return domain_data

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

        domain_data = self._extract_domain_data(payload)
        domain_data = self._augment_domain_data_contracts(
            source=source,
            payload=payload,
            domain_data=domain_data,
        )
        features = self._extract_features(
            source=source,
            symbol=symbol,
            payload=payload,
            timestamp=ts,
        )

        normalized = NormalizedPayload(
            source=source,
            symbol=symbol,
            timestamp=ts,
            timeframe=timeframe,
            domain_data=domain_data,
            features=features,
            metadata={
                "event_name": event_name,
                "raw_payload_keys": sorted(payload.keys()),
                "features_count": len(features),
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

        for snapshot in normalized.features:
            context.put_feature(snapshot)
            if snapshot.freshness_seconds is not None:
                context.freshness_map[snapshot.name] = snapshot.freshness_seconds

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

        state = mapping_for(
            "state",
            "composite",
            "price_action",
            "snapshot",
            "result",
        )
        market_structure = mapping_for(
            "market_structure",
            "structure",
            "ms",
        )
        support_resistance = mapping_for(
            "support_resistance",
            "sr",
            "levels",
        )
        fair_value_gap = mapping_for(
            "fair_value_gap",
            "fvg",
            "fair_value_gaps",
        )
        trend = mapping_for(
            "trend",
            "trend_state",
        )
        liquidity_levels = mapping_for(
            "liquidity_levels",
            "liquidity",
        )

        if state:
            if not market_structure:
                for key in ("market_structure", "structure", "ms"):
                    value = state.get(key)
                    if isinstance(value, dict):
                        market_structure = value
                        break

            if not support_resistance:
                for key in ("support_resistance", "sr", "levels"):
                    value = state.get(key)
                    if isinstance(value, dict):
                        support_resistance = value
                        break

            if not fair_value_gap:
                for key in ("fair_value_gap", "fvg", "fair_value_gaps"):
                    value = state.get(key)
                    if isinstance(value, dict):
                        fair_value_gap = value
                        break

            if not trend:
                for key in ("trend", "trend_state"):
                    value = state.get(key)
                    if isinstance(value, dict):
                        trend = value
                        break

            if not liquidity_levels:
                for key in ("liquidity_levels", "liquidity"):
                    value = state.get(key)
                    if isinstance(value, dict):
                        liquidity_levels = value
                        break

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]
                if key in state:
                    return state[key]
            return default

        def nested_value(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in mapping:
                    return mapping[key]
            return default

        def add(name: str, value: Any) -> None:
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
                    },
                )
            )

        current_price = value_for(
            "current_price",
            "price",
            "last_price",
            "close",
            default=None,
        )

        # Composite / global
        add("price_action.composite", state or payload)
        add("price_action.current_price", current_price)
        add(
            "price_action.last_price",
            value_for(
                "last_price",
                "price",
                "close",
                default=current_price,
            ),
        )
        add(
            "price_action.timestamp",
            value_for(
                "timestamp",
                "event_time",
                default=timestamp,
            ),
        )

        # Market structure
        add("price_action.market_structure", market_structure or payload)
        add(
            "price_action.market_structure.internal",
            nested_value(
                market_structure,
                "internal",
                default={},
            ),
        )
        add(
            "price_action.market_structure.external",
            nested_value(
                market_structure,
                "external",
                default={},
            ),
        )
        add(
            "price_action.market_structure.last_break_event",
            nested_value(
                market_structure,
                "last_break_event",
                "last_event",
                "event",
                default=value_for("last_break_event", "event", default=None),
            ),
        )
        add(
            "price_action.market_structure.mtf_alignment",
            nested_value(
                market_structure,
                "mtf_alignment",
                "multi_timeframe_alignment",
                default=value_for("mtf_alignment", default=0.0),
            ),
        )

        # Support / resistance
        add("price_action.support_resistance", support_resistance or payload)
        add(
            "price_action.support_resistance.internal",
            nested_value(
                support_resistance,
                "internal",
                default={},
            ),
        )
        add(
            "price_action.support_resistance.external",
            nested_value(
                support_resistance,
                "external",
                default={},
            ),
        )
        add(
            "price_action.support_resistance.last_event",
            nested_value(
                support_resistance,
                "last_event",
                "event",
                default=value_for("last_event", "event", default=None),
            ),
        )
        add(
            "price_action.support_resistance.nearest_support",
            nested_value(
                support_resistance,
                "nearest_support",
                "support",
                default=value_for("nearest_support", default=None),
            ),
        )
        add(
            "price_action.support_resistance.nearest_resistance",
            nested_value(
                support_resistance,
                "nearest_resistance",
                "resistance",
                default=value_for("nearest_resistance", default=None),
            ),
        )

        # Fair value gap / FVG
        add("price_action.fair_value_gap", fair_value_gap or payload)
        add("price_action.fvg", fair_value_gap or payload)
        add(
            "price_action.fair_value_gap.internal",
            nested_value(
                fair_value_gap,
                "internal",
                default={},
            ),
        )
        add(
            "price_action.fair_value_gap.external",
            nested_value(
                fair_value_gap,
                "external",
                default={},
            ),
        )
        add(
            "price_action.fair_value_gap.last_event",
            nested_value(
                fair_value_gap,
                "last_event",
                "event",
                default=value_for("last_event", "event", default=None),
            ),
        )
        add(
            "price_action.fair_value_gap.nearest_bullish_gap",
            nested_value(
                fair_value_gap,
                "nearest_bullish_gap",
                "bullish_gap",
                default=value_for("nearest_bullish_gap", default=None),
            ),
        )
        add(
            "price_action.fair_value_gap.nearest_bearish_gap",
            nested_value(
                fair_value_gap,
                "nearest_bearish_gap",
                "bearish_gap",
                default=value_for("nearest_bearish_gap", default=None),
            ),
        )

        # Trend
        add("price_action.trend", trend or payload)
        add(
            "price_action.trend.internal",
            nested_value(
                trend,
                "internal",
                default={},
            ),
        )
        add(
            "price_action.trend.external",
            nested_value(
                trend,
                "external",
                default={},
            ),
        )
        add(
            "price_action.trend.last_signal",
            nested_value(
                trend,
                "last_signal",
                "last_event",
                "event",
                default=value_for("last_signal", "last_event", default=None),
            ),
        )
        add(
            "price_action.trend.internal_external_alignment",
            nested_value(
                trend,
                "internal_external_alignment",
                default=value_for("internal_external_alignment", default=0.0),
            ),
        )
        add(
            "price_action.trend.higher_timeframe_alignment",
            nested_value(
                trend,
                "higher_timeframe_alignment",
                "htf_alignment",
                default=value_for(
                    "higher_timeframe_alignment",
                    "htf_alignment",
                    default=0.0,
                ),
            ),
        )
        add(
            "price_action.trend.overall_trend_score",
            nested_value(
                trend,
                "overall_trend_score",
                "score",
                default=value_for("overall_trend_score", "score", default=0.0),
            ),
        )

        # Price-action liquidity levels
        add("price_action.liquidity_levels", liquidity_levels)

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
            "liquidity_map_snapshot",
            "map_snapshot",
            "last_snapshot",
            "liquidity",
            "result",
        )
        signal = mapping_for(
            "signal",
            "liquidity_signal",
            "analytics_signal",
        )

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]
                if key in snapshot:
                    return snapshot[key]
                if key in signal:
                    return signal[key]
            return default

        def nested_value(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in mapping:
                    return mapping[key]
            return default

        def add(name: str, value: Any) -> None:
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
                    },
                )
            )

        current_price = value_for(
            "current_price",
            "price",
            "mark_price",
            "last_price",
            "close",
        )

        add("liquidity.snapshot", snapshot or payload)
        add("liquidity.map.snapshot", snapshot or payload)
        add("liquidity.current_price", current_price)

        add(
            "liquidity.above_liquidity_score",
            value_for("above_liquidity_score", "above_score", default=0.0),
        )
        add(
            "liquidity.below_liquidity_score",
            value_for("below_liquidity_score", "below_score", default=0.0),
        )
        add(
            "liquidity.pressure_score",
            value_for(
                "liquidity_pressure_score",
                "pressure_score",
                "liquidity_pressure",
                default=0.0,
            ),
        )
        add(
            "liquidity.bias",
            value_for("bias", "liquidity_bias", "direction", default=None),
        )

        sweep_risk = mapping_for("sweep_risk", "sweep_risks")
        magnet = mapping_for("magnet", "magnets", "liquidity_magnets")

        add(
            "liquidity.sweep_risk.up",
            nested_value(
                sweep_risk,
                "up",
                "upside",
                "above",
                default=value_for("sweep_risk_up", "upside_sweep_risk", default=0.0),
            ),
        )
        add(
            "liquidity.sweep_risk.down",
            nested_value(
                sweep_risk,
                "down",
                "downside",
                "below",
                default=value_for("sweep_risk_down", "downside_sweep_risk", default=0.0),
            ),
        )
        add(
            "liquidity.magnet.up",
            nested_value(
                magnet,
                "up",
                "upside",
                "above",
                default=value_for("magnet_up", "upside_magnet", default=0.0),
            ),
        )
        add(
            "liquidity.magnet.down",
            nested_value(
                magnet,
                "down",
                "downside",
                "below",
                default=value_for("magnet_down", "downside_magnet", default=0.0),
            ),
        )

        add(
            "liquidity.nearest_above_level",
            value_for("nearest_above_level", "nearest_liquidity_above"),
        )
        add(
            "liquidity.nearest_below_level",
            value_for("nearest_below_level", "nearest_liquidity_below"),
        )
        add(
            "liquidity.strongest_cluster_above",
            value_for("strongest_cluster_above"),
        )
        add(
            "liquidity.strongest_cluster_below",
            value_for("strongest_cluster_below"),
        )

        add(
            "liquidity.equal_levels",
            value_for("equal_levels", default=[]),
        )
        add(
            "liquidity.active_levels",
            value_for("active_levels", "levels", "liquidity_levels", default=[]),
        )
        add(
            "liquidity.stop_clusters",
            value_for("stop_clusters", "clusters", "liquidity_clusters", default=[]),
        )
        add(
            "liquidity.zones",
            value_for("zones", "liquidity_zones", default=[]),
        )

        if signal:
            add("liquidity.signal", signal)

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

        def mapping_for(*keys: str) -> dict[str, Any]:
            for key in keys:
                value = payload.get(key)
                if isinstance(value, dict):
                    return value

                value = feature_map.get(key)
                if isinstance(value, dict):
                    return value

            return {}

        cascade = mapping_for(
            "cascade",
            "cascade_result",
            "cascade_detection",
            "cascade_detected",
            "result",
        )
        exhaustion = mapping_for(
            "exhaustion",
            "exhaustion_result",
            "exhaustion_detection",
            "exhaustion_detected",
            "reversal_context",
        )
        squeeze = mapping_for(
            "squeeze",
            "squeeze_result",
            "squeeze_reversal",
            "squeeze_context",
            "pending_confirmation",
        )
        cluster = mapping_for(
            "cluster",
            "liquidation_cluster",
            "cluster_stats",
        )
        signal = mapping_for(
            "signal",
            "liquidation_signal",
            "analytics_signal",
        )

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]
            return default

        def nested_value(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in mapping:
                    return mapping[key]
            return default

        def add(name: str, value: Any) -> None:
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
                    },
                )
            )

        # Cascade
        add("liquidations.cascade", cascade or payload)
        add(
            "liquidations.cascade.confidence",
            nested_value(cascade, "confidence", default=value_for("confidence", default=0.0)),
        )
        add(
            "liquidations.cascade.intensity_score",
            nested_value(
                cascade,
                "intensity_score",
                "intensity",
                default=value_for("intensity_score", default=0.0),
            ),
        )
        add(
            "liquidations.cascade.direction",
            nested_value(
                cascade,
                "direction",
                "cascade_direction",
                "side",
                default=value_for("direction", "cascade_direction", "side"),
            ),
        )
        add(
            "liquidations.cascade.severity",
            nested_value(
                cascade,
                "severity",
                "severity_label",
                default=value_for("severity", "severity_label"),
            ),
        )
        add(
            "liquidations.cascade.continuation_bias",
            nested_value(
                cascade,
                "continuation_bias",
                default=value_for("continuation_bias", default=0.0),
            ),
        )
        add(
            "liquidations.cascade.exhaustion_bias",
            nested_value(
                cascade,
                "exhaustion_bias",
                default=value_for("exhaustion_bias", default=0.0),
            ),
        )
        add(
            "liquidations.cascade.total_notional_usd",
            nested_value(
                cascade,
                "total_notional_usd",
                "notional_usd",
                default=value_for("total_notional_usd", "notional_usd", default=0.0),
            ),
        )
        add(
            "liquidations.cascade.event_count",
            nested_value(
                cascade,
                "event_count",
                "events_count",
                default=value_for("event_count", "events_count", default=0),
            ),
        )

        # Exhaustion
        if exhaustion:
            add("liquidations.exhaustion", exhaustion)
            add(
                "liquidations.exhaustion.confidence",
                nested_value(exhaustion, "confidence", default=value_for("confidence", default=0.0)),
            )
            add(
                "liquidations.exhaustion.exhaustion_bias",
                nested_value(
                    exhaustion,
                    "exhaustion_bias",
                    default=value_for("exhaustion_bias", default=0.0),
                ),
            )
            add(
                "liquidations.exhaustion.bias_delta",
                nested_value(
                    exhaustion,
                    "bias_delta",
                    default=value_for("bias_delta", default=0.0),
                ),
            )
            add(
                "liquidations.exhaustion.confirmed",
                nested_value(
                    exhaustion,
                    "confirmed",
                    "is_confirmed",
                    default=value_for("confirmed", "is_confirmed", default=False),
                ),
            )

        # Squeeze
        if squeeze:
            add("liquidations.squeeze", squeeze)
            add(
                "liquidations.squeeze.confirmed",
                nested_value(
                    squeeze,
                    "confirmed",
                    "is_confirmed",
                    default=value_for("squeeze_confirmed", default=False),
                ),
            )
            add(
                "liquidations.squeeze.score",
                nested_value(
                    squeeze,
                    "score",
                    default=value_for("squeeze_score", "score", default=0.0),
                ),
            )
            add(
                "liquidations.squeeze.direction",
                nested_value(
                    squeeze,
                    "direction",
                    "reversal_side",
                    "side",
                    default=value_for("squeeze_direction", "reversal_side", "side"),
                ),
            )

        # Cluster
        if cluster:
            add("liquidations.cluster", cluster)
            add(
                "liquidations.cluster.duration_seconds",
                nested_value(cluster, "duration_seconds", default=value_for("duration_seconds")),
            )
            add(
                "liquidations.cluster.avg_notional_per_event",
                nested_value(
                    cluster,
                    "avg_notional_per_event",
                    default=value_for("avg_notional_per_event"),
                ),
            )
            add(
                "liquidations.cluster.side_imbalance_ratio",
                nested_value(
                    cluster,
                    "side_imbalance_ratio",
                    default=value_for("side_imbalance_ratio"),
                ),
            )
            add(
                "liquidations.cluster.event_imbalance_ratio",
                nested_value(
                    cluster,
                    "event_imbalance_ratio",
                    default=value_for("event_imbalance_ratio"),
                ),
            )
            add(
                "liquidations.cluster.acceleration_ratio",
                nested_value(
                    cluster,
                    "acceleration_ratio",
                    default=value_for("acceleration_ratio"),
                ),
            )

        if signal:
            add("liquidations.signal", signal)

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

        composite = payload.get("composite") or payload.get("snapshot") or payload.get("orderflow")
        if not isinstance(composite, dict):
            composite = {}

        cvd = payload.get("cvd") or payload.get("cvd_stats") or composite.get("cvd")
        volume_delta = (
                payload.get("volume_delta")
                or payload.get("volume_delta_stats")
                or composite.get("volume_delta")
        )
        aggressive = (
                payload.get("aggressive_trades")
                or payload.get("aggressive")
                or composite.get("aggressive_trades")
        )
        imbalance = (
                payload.get("orderbook_imbalance")
                or payload.get("imbalance")
                or composite.get("orderbook_imbalance")
        )

        cvd = cvd if isinstance(cvd, dict) else {}
        volume_delta = volume_delta if isinstance(volume_delta, dict) else {}
        aggressive = aggressive if isinstance(aggressive, dict) else {}
        imbalance = imbalance if isinstance(imbalance, dict) else {}

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]
                if key in composite:
                    return composite[key]
            return default

        def nested_value(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in mapping:
                    return mapping[key]
            return default

        def add(name: str, value: Any) -> None:
            snapshot = self._snapshot_from_raw_value(
                source=FeatureSource.ORDERFLOW,
                symbol=symbol,
                name=name,
                value=value,
                timestamp=timestamp,
                confidence=confidence,
                metadata={
                    "origin": "contract_feature",
                    "contract": "orderflow",
                },
            )
            result.append(snapshot)

        cvd_delta_ratio = value_for(
            "cvd_delta_ratio",
            "delta_ratio",
            default=nested_value(cvd, "delta_ratio", "cvd_delta_ratio", default=0.0),
        )
        cvd_change_pct = value_for(
            "cvd_change_pct",
            default=nested_value(cvd, "cvd_change_pct", "change_pct", default=0.0),
        )
        cvd_slope = value_for(
            "cvd_slope",
            default=nested_value(cvd, "cvd_slope", "slope", default=0.0),
        )
        price_change_pct = value_for(
            "price_change_pct",
            default=nested_value(cvd, "price_change_pct", default=0.0),
        )

        volume_delta_ratio = value_for(
            "volume_delta_ratio",
            default=nested_value(volume_delta, "delta_ratio", "volume_delta_ratio", default=0.0),
        )
        volume_delta_value = value_for(
            "volume_delta",
            default=nested_value(volume_delta, "volume_delta", "delta", default=0.0),
        )

        aggressive_buy_ratio = value_for(
            "aggressive_buy_ratio",
            "buy_ratio",
            default=nested_value(aggressive, "buy_ratio", "aggressive_buy_ratio", default=0.0),
        )
        aggressive_sell_ratio = value_for(
            "aggressive_sell_ratio",
            "sell_ratio",
            default=nested_value(aggressive, "sell_ratio", "aggressive_sell_ratio", default=0.0),
        )

        imbalance_ratio = value_for(
            "orderbook_imbalance_ratio",
            "imbalance_ratio",
            default=nested_value(imbalance, "ratio", "imbalance_ratio", default=0.0),
        )
        imbalance_diff = value_for(
            "orderbook_imbalance_diff",
            "imbalance_diff",
            default=nested_value(imbalance, "diff", "imbalance_diff", default=0.0),
        )

        add("orderflow.composite", composite or payload)
        add("orderflow.cvd", cvd or {
            "delta_ratio": cvd_delta_ratio,
            "cvd_change_pct": cvd_change_pct,
            "cvd_slope": cvd_slope,
            "price_change_pct": price_change_pct,
        })
        add("orderflow.cvd.delta_ratio", cvd_delta_ratio)
        add("orderflow.cvd.cvd_change_pct", cvd_change_pct)
        add("orderflow.cvd.cvd_slope", cvd_slope)
        add("orderflow.cvd.price_change_pct", price_change_pct)

        add("orderflow.volume_delta", volume_delta or {
            "delta_ratio": volume_delta_ratio,
            "volume_delta": volume_delta_value,
        })
        add("orderflow.volume_delta.delta_ratio", volume_delta_ratio)
        add("orderflow.volume_delta.volume_delta", volume_delta_value)

        add("orderflow.aggressive_trades", aggressive or {
            "buy_ratio": aggressive_buy_ratio,
            "sell_ratio": aggressive_sell_ratio,
        })
        add("orderflow.aggressive_trades.buy_ratio", aggressive_buy_ratio)
        add("orderflow.aggressive_trades.sell_ratio", aggressive_sell_ratio)

        add("orderflow.orderbook_imbalance", imbalance or {
            "ratio": imbalance_ratio,
            "diff": imbalance_diff,
        })
        add("orderflow.orderbook_imbalance.ratio", imbalance_ratio)
        add("orderflow.orderbook_imbalance.diff", imbalance_diff)

        add("orderflow.trades_count", value_for("trades_count", default=0))
        add("orderflow.total_volume", value_for("total_volume", default=0.0))
        add("orderflow.total_notional", value_for("total_notional", default=0.0))
        add("orderflow.last_price", value_for("last_price", "price", default=None))
        add("orderflow.price_change_pct", price_change_pct)

        return result

    def _build_open_interest_contract_features(
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
                    features = value

            if not snapshot:
                value = analysis.get("snapshot")
                if isinstance(value, dict):
                    snapshot = value

            if not market_context:
                for key in ("context", "market_context", "oi_context"):
                    value = analysis.get(key)
                    if isinstance(value, dict):
                        market_context = value
                        break

            if not regime:
                for key in ("regime", "regime_result", "oi_regime"):
                    value = analysis.get(key)
                    if isinstance(value, dict):
                        regime = value
                        break

            if not anomaly:
                for key in ("anomaly", "anomaly_result", "oi_anomaly"):
                    value = analysis.get(key)
                    if isinstance(value, dict):
                        anomaly = value
                        break

            if not divergence:
                for key in ("divergence", "divergence_result", "oi_divergence"):
                    value = analysis.get(key)
                    if isinstance(value, dict):
                        divergence = value
                        break

        def has_any(*keys: str) -> bool:
            for key in keys:
                if key in payload:
                    return True
                if key in feature_map:
                    return True
                if key in features:
                    return True
                if key in regime:
                    return True
                if key in anomaly:
                    return True
                if key in divergence:
                    return True
                if key in analysis:
                    return True

            return False

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]
                if key in features:
                    return features[key]
                if key in regime:
                    return regime[key]
                if key in anomaly:
                    return anomaly[key]
                if key in divergence:
                    return divergence[key]
                if key in analysis:
                    return analysis[key]

            return default

        def add(name: str, value: Any = True) -> None:
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
                },
            )
            result.append(snapshot_obj)

        # Main OI contract sections
        if analysis:
            add("open_interest.analysis", analysis)

        if snapshot:
            add("open_interest.snapshot", snapshot)

        if market_context:
            add("open_interest.market_context", market_context)

        if features or has_any(
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
            add(
                "open_interest.features",
                features or {
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
                },
            )

        if regime or has_any(
                "regime",
                "oi_regime",
                "market_regime",
                "new_regime",
        ):
            add(
                "open_interest.regime",
                regime or {
                    "regime": value_for(
                        "regime",
                        "oi_regime",
                        "market_regime",
                        "new_regime",
                        default=None,
                    ),
                    "confidence": value_for(
                        "regime_confidence",
                        "confidence",
                        default=0.0,
                    ),
                    "score": value_for(
                        "regime_score",
                        "score",
                        default=0.0,
                    ),
                    "reasons": value_for(
                        "regime_reasons",
                        "reasons",
                        default=[],
                    ),
                },
            )

        if anomaly or has_any(
                "anomaly",
                "anomaly_type",
                "capitulation",
                "capitulation_score",
                "squeeze_setup",
                "squeeze_score",
                "liquidation_imbalance",
                "anomaly_detected",
                "is_anomaly",
        ):
            add(
                "open_interest.anomaly",
                anomaly or {
                    "detected": value_for(
                        "anomaly_detected",
                        "is_anomaly",
                        "anomaly",
                        "capitulation",
                        "squeeze_setup",
                        default=True,
                    ),
                    "anomaly_type": value_for("anomaly_type", default=None),
                    "confidence": value_for(
                        "anomaly_confidence",
                        "confidence",
                        default=0.0,
                    ),
                    "score": value_for(
                        "anomaly_score",
                        "score",
                        default=0.0,
                    ),
                    "strength": value_for(
                        "anomaly_strength",
                        "strength",
                        default=value_for("score", default=0.0),
                    ),
                    "capitulation": value_for("capitulation", default=None),
                    "capitulation_score": value_for(
                        "capitulation_score",
                        default=None,
                    ),
                    "squeeze_setup": value_for("squeeze_setup", default=None),
                    "squeeze_score": value_for(
                        "squeeze_score",
                        default=None,
                    ),
                    "liquidation_imbalance": value_for(
                        "liquidation_imbalance",
                        "liquidation_pressure",
                        default=None,
                    ),
                    "reasons": value_for(
                        "anomaly_reasons",
                        "reasons",
                        default=[],
                    ),
                },
            )

        if divergence or has_any(
                "divergence",
                "divergence_type",
                "price_oi_divergence",
                "cvd_delta",
                "funding_rate",
                "divergence_detected",
                "is_divergence",
        ):
            add(
                "open_interest.divergence",
                divergence or {
                    "detected": value_for(
                        "divergence_detected",
                        "is_divergence",
                        "divergence",
                        default=True,
                    ),
                    "divergence_type": value_for(
                        "divergence_type",
                        "price_oi_divergence",
                        default=None,
                    ),
                    "price_oi_divergence": value_for(
                        "price_oi_divergence",
                        "divergence_type",
                        default=None,
                    ),
                    "confidence": value_for(
                        "divergence_confidence",
                        "confidence",
                        default=0.0,
                    ),
                    "score": value_for(
                        "divergence_score",
                        "score",
                        default=0.0,
                    ),
                    "window_size": value_for(
                        "divergence_window_size",
                        "window_size",
                        default=None,
                    ),
                    "cvd_delta": value_for("cvd_delta", default=None),
                    "funding_rate": value_for("funding_rate", default=None),
                    "reasons": value_for(
                        "divergence_reasons",
                        "reasons",
                        default=[],
                    ),
                },
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

        def mapping_for(*keys: str) -> dict[str, Any]:
            for key in keys:
                value = payload.get(key)
                if isinstance(value, dict):
                    return value

                value = feature_map.get(key)
                if isinstance(value, dict):
                    return value

            return {}

        snapshot = mapping_for("snapshot", "funding_snapshot")
        statistics = mapping_for("statistics", "stats", "funding_statistics")
        regime = mapping_for("regime", "regime_state", "funding_regime")
        pressure = mapping_for("pressure", "pressure_state", "funding_pressure")
        extreme = mapping_for("extreme", "extreme_event", "funding_extreme")
        divergence = mapping_for("divergence", "divergence_event", "funding_divergence")
        flip = mapping_for("flip", "flip_event", "funding_flip")
        signal = mapping_for("signal", "funding_signal")

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]
            return default

        def nested_value(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in mapping:
                    return mapping[key]
            return default

        def add(name: str, value: Any) -> None:
            snapshot_obj = self._snapshot_from_raw_value(
                source=FeatureSource.FUNDING,
                symbol=symbol,
                name=name,
                value=value,
                timestamp=timestamp,
                confidence=confidence,
                metadata={
                    "origin": "contract_feature",
                    "contract": "funding",
                },
            )
            result.append(snapshot_obj)

        add("funding.snapshot", snapshot or payload)
        add("funding.statistics", statistics)

        add("funding.regime", regime)
        add(
            "funding.regime.confidence",
            nested_value(regime, "confidence", default=value_for("regime_confidence", default=0.0)),
        )

        add("funding.pressure", pressure)
        add(
            "funding.pressure.score",
            nested_value(pressure, "score", "pressure_score", default=value_for("pressure_score", default=0.0)),
        )
        add(
            "funding.pressure.level",
            nested_value(pressure, "level", "pressure_level", default=value_for("pressure_level", default=None)),
        )
        add(
            "funding.pressure.direction",
            nested_value(pressure, "direction", "bias", default=value_for("pressure_direction", default=None)),
        )

        add("funding.extreme", extreme)
        add(
            "funding.extreme.type",
            nested_value(extreme, "type", "extreme_type", default=value_for("extreme_type", default=None)),
        )
        add(
            "funding.extreme.severity",
            nested_value(extreme, "severity", "score", default=value_for("extreme_severity", default=0.0)),
        )
        add(
            "funding.extreme.mean_reversion_probability",
            nested_value(
                extreme,
                "mean_reversion_probability",
                "reversion_probability",
                default=value_for("mean_reversion_probability", default=0.0),
            ),
        )
        add(
            "funding.extreme.squeeze_probability",
            nested_value(extreme, "squeeze_probability", default=value_for("squeeze_probability", default=0.0)),
        )

        add("funding.divergence", divergence)
        add(
            "funding.divergence.type",
            nested_value(divergence, "type", "divergence_type", default=value_for("divergence_type", default=None)),
        )
        add(
            "funding.divergence.confidence",
            nested_value(divergence, "confidence", default=value_for("divergence_confidence", default=0.0)),
        )
        add(
            "funding.divergence.score",
            nested_value(divergence, "score", default=value_for("divergence_score", default=0.0)),
        )

        add("funding.flip", flip)
        add(
            "funding.flip.type",
            nested_value(flip, "type", "flip_type", default=value_for("flip_type", default=None)),
        )
        add(
            "funding.flip.confidence",
            nested_value(flip, "confidence", default=value_for("flip_confidence", default=0.0)),
        )

        add("funding.signal", signal)
        add(
            "funding.signal.type",
            nested_value(signal, "type", "signal_type", default=value_for("signal_type", default=None)),
        )
        add(
            "funding.signal.score",
            nested_value(signal, "score", default=value_for("signal_score", "score", default=0.0)),
        )
        add(
            "funding.signal.confidence",
            nested_value(signal, "confidence", default=value_for("signal_confidence", "confidence", default=0.0)),
        )
        add(
            "funding.signal.bias",
            nested_value(signal, "bias", "direction", default=value_for("bias", "direction", default=None)),
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
        signal.confidence_grade = confidence_to_grade(signal.confidence)
        signal.strength = confidence_to_strength(signal.confidence)

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
            confidence_grade=confidence_to_grade(adjusted_confidence),
            strength=confidence_to_strength(adjusted_confidence),
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

        self._filter_symbol_match(evaluation, context)
        self._filter_directional(evaluation)
        self._filter_confidence(evaluation)
        self._filter_score(evaluation)
        self._filter_age(evaluation)
        self._filter_freshness(evaluation, context)
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
        threshold = self.config.runtime.min_confidence

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
        threshold = self.config.runtime.min_score

        if evaluation.signal.score < threshold:
            evaluation.add_result(
                FilterResult(
                    name="min_score",
                    decision=FilterDecision.BLOCK,
                    reason="score_below_runtime_threshold",
                    score_impact=threshold - evaluation.signal.score,
                )
            )

    def _filter_age(self, evaluation: FilterEvaluation) -> None:
        max_age = self.config.runtime.max_signal_age_seconds
        age = (utcnow() - evaluation.signal.timestamp).total_seconds()

        if age > max_age:
            evaluation.add_result(
                FilterResult(
                    name="signal_age",
                    decision=FilterDecision.BLOCK,
                    reason="signal_expired_before_routing",
                    metadata={"age_seconds": age, "max_age_seconds": max_age},
                )
            )

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

        price = _to_float(signal.metadata.get("entry_price"))

        if price is None and context.price is not None:
            price = (
                _to_float(getattr(context.price, "last", None))
                or _to_float(getattr(context.price, "close", None))
                or _to_float(getattr(context.price, "mark_price", None))
                or _to_float(getattr(context.price, "price", None))
            )

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

    @staticmethod
    def _infer_stop_loss(
        signal: StrategySignal,
        *,
        entry_price: float,
    ) -> float | None:
        stop_distance = _to_float(signal.metadata.get("stop_distance"))

        if stop_distance is None:
            atr = _to_float(signal.metadata.get("atr"))
            multiplier = _to_float(signal.metadata.get("atr_stop_multiplier"), 1.5) or 1.5
            if atr is not None:
                stop_distance = atr * multiplier

        if stop_distance is None or stop_distance <= 0:
            return None

        if signal.side is SignalSide.LONG:
            return entry_price - stop_distance

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
            rr = _to_float(getattr(self.builder_config, "default_rr", None), 2.0) or 2.0

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

            delta = (ensure_aware_utc(now) - previous.timestamp).total_seconds()
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
        )

        if route.is_empty:
            batch.reasons.append("no_strategies_routed")
            self.state.metrics.record_applicability_skip()
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
            await self._emit_rejected_batch(batch, reason="no_passed_strategy_signals")
            return batch

        scored = self.scorer.score_many(signals=raw_signals, context=context)

        filtered = self.filters.apply(signals=scored, context=context)
        batch.filtered_signals = filtered

        if not filtered:
            batch.reasons.append("all_signals_filtered")
            await self._emit_rejected_batch(batch, reason="all_signals_filtered")
            return batch

        confluence = self.confluence.evaluate(signals=filtered, context=context)
        batch.confluence = confluence

        confluence_signals = self._signals_after_confluence(confluence, filtered)
        if not confluence_signals:
            batch.reasons.append("confluence_rejected")
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
            await self._emit_rejected_batch(batch, reason="all_signals_failed_builder")
            return batch

        coordinated = self.portfolio.coordinate(
            signals=built_signals,
            context=context,
        )
        batch.coordinated = coordinated
        batch.final_signals = list(coordinated.final_signals)

        if not coordinated.accepted or not batch.final_signals:
            batch.reasons.append("portfolio_coordination_rejected")
            await self._emit_rejected_batch(batch, reason="portfolio_coordination_rejected")
            return batch

        risk_payloads: list[RiskReadySignalPayload] = []

        for signal in batch.final_signals:
            self.builder.assert_risk_ready(signal)

            risk_payload = self.to_risk_payload(
                signal=signal,
                context=context,
            )
            risk_payloads.append(risk_payload)

        batch.risk_payloads = risk_payloads
        batch.accepted = bool(risk_payloads)

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

        await self.emit_event(
            "signal.rejected",
            {
                "symbol": batch.symbol,
                "reason": reason,
                "route": batch.route.selected_names if batch.route else [],
                "reasons": list(batch.reasons),
                "metadata": dict(batch.metadata),
            },
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