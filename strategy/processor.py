# trading_system/strategy/processor.py

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

        return domain_data
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
        if source is FeatureSource.OPEN_INTEREST:
            return self._build_open_interest_contract_features(
                symbol=symbol,
                payload=payload,
                timestamp=timestamp,
            )
        if source is FeatureSource.FUNDING:
            return self._build_funding_contract_features(
                symbol=symbol,
                payload=payload,
                timestamp=timestamp,
            )

        return []

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

        def has_any(*keys: str) -> bool:
            return any(key in payload or key in feature_map for key in keys)

        def value_for(*keys: str, default: Any = True) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]
            return default

        def add(name: str, value: Any = True) -> None:
            snapshot = self._snapshot_from_raw_value(
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
            result.append(snapshot)

        if has_any(
                "oi",
                "open_interest",
                "open_interest_value",
                "oi_delta",
                "oi_delta_pct",
                "oi_direction",
                "oi_acceleration",
        ):
            add(
                "open_interest.features",
                {
                    "oi": value_for("oi", "open_interest", default=None),
                    "open_interest_value": value_for("open_interest_value", default=None),
                    "oi_delta": value_for("oi_delta", default=None),
                    "oi_delta_pct": value_for("oi_delta_pct", default=None),
                    "oi_direction": value_for("oi_direction", default=None),
                    "oi_acceleration": value_for("oi_acceleration", default=None),
                },
            )

        if has_any("regime", "oi_regime", "market_regime"):
            add(
                "open_interest.regime",
                value_for("regime", "oi_regime", "market_regime"),
            )

        if has_any(
                "anomaly",
                "anomaly_type",
                "capitulation",
                "capitulation_score",
                "squeeze_setup",
                "squeeze_score",
                "liquidation_imbalance",
        ):
            add(
                "open_interest.anomaly",
                {
                    "anomaly": value_for("anomaly", default=None),
                    "anomaly_type": value_for("anomaly_type", default=None),
                    "capitulation": value_for("capitulation", default=None),
                    "capitulation_score": value_for("capitulation_score", default=None),
                    "squeeze_setup": value_for("squeeze_setup", default=None),
                    "squeeze_score": value_for("squeeze_score", default=None),
                    "liquidation_imbalance": value_for("liquidation_imbalance", default=None),
                },
            )

        if has_any(
                "divergence",
                "divergence_type",
                "price_oi_divergence",
                "cvd_delta",
                "funding_rate",
        ):
            add(
                "open_interest.divergence",
                {
                    "divergence": value_for("divergence", default=None),
                    "divergence_type": value_for("divergence_type", default=None),
                    "price_oi_divergence": value_for("price_oi_divergence", default=None),
                    "cvd_delta": value_for("cvd_delta", default=None),
                    "funding_rate": value_for("funding_rate", default=None),
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