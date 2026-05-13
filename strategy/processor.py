# trading_system/strategy/processor.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import datetime

from core.event_bus import EventBus, EventPriority
from core.scheduler import Scheduler

from .base import BaseStrategy, BaseStrategyComponent, ContextAwareStrategyComponent
from .config import (
    BuilderConfig,
    ConfluenceConfig,
    FilterConfig,
    PortfolioCoordinatorConfig,
    RoutingConfig,
    StrategyConfig,
)
from .enums import (
    ConfidenceGrade,
    ConflictType,
    EntryType,
    ExitType,
    FeatureSource,
    FilterDecision,
    MarketRegime,
    SignalOrigin,
    SignalPriority,
    SignalSide,
    SignalStatus,
    SignalStrength,
    StrategyCategory,
    TriggerType,
)
from .exceptions import (
    BuilderError,
    ConfluenceError,
    FilterExecutionError,
    PortfolioCoordinationError,
    SignalNormalizationError,
    SignalRoutingError,
    StrategyEvaluationError,
)
from .models import (
    ConflictRecord,
    ConfluenceResult,
    EntryPlan,
    ExecutionPlanDraft,
    ExitPlan,
    FeatureSnapshot,
    FilterResult,
    InvalidationPlan,
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
from .registry import StrategyRegistry
from .state import StrategyRuntimeState


@dataclass(slots=True)
class NormalizedPayload:
    source: FeatureSource
    symbol: str
    timestamp: datetime
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

        if result.decision == FilterDecision.BLOCK:
            self.accepted = False
            self.blocking_filters.append(result.name)
            if result.reason:
                self.reasons.append(result.reason)

        elif result.decision == FilterDecision.WARN:
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
    accepted: bool = False
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamp = ensure_aware_utc(self.timestamp)


class SignalNormalizer(BaseStrategyComponent):
    """
    Normalizes analytics payloads into:
    - domain data inside StrategyContext;
    - FeatureSnapshot entries for StrategyContextStore.
    """

    component_namespace = "strategy.processor.normalizer"

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

        domain_data = self._extract_domain_data(payload)
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
            domain_data=domain_data,
            features=features,
            metadata={"event_name": event_name},
        )

        self.log_debug(
            "Analytics event normalized",
            event_name=event_name,
            source=str(source),
            symbol=symbol,
            features_count=len(features),
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

        for key, value in normalized.domain_data.items():
            context.put_domain_feature(normalized.source, key, value)

        for snapshot in normalized.features:
            context.put_feature(snapshot)
            if snapshot.freshness_seconds is not None:
                context.freshness_map[snapshot.name] = snapshot.freshness_seconds

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

        raise SignalNormalizationError(
            f"unable to resolve FeatureSource for event '{event_name}'"
        )

    def _resolve_source_from_text(self, value: str) -> FeatureSource | None:
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

    def _extract_symbol(self, payload: dict[str, Any]) -> str:
        raw = payload.get("symbol") or payload.get("instrument") or payload.get("market")
        if not isinstance(raw, str) or not raw.strip():
            raise SignalNormalizationError("payload does not contain valid symbol")
        return raw.strip()

    def _extract_timestamp(
        self,
        payload: dict[str, Any],
        fallback: datetime | None = None,
    ) -> datetime:
        raw = payload.get("timestamp") or payload.get("ts") or fallback

        if raw is None:
            return utcnow()

        if isinstance(raw, datetime):
            return ensure_aware_utc(raw)

        if isinstance(raw, (int, float)):
            if raw > 10_000_000_000:
                from datetime import datetime, timezone
                return datetime.fromtimestamp(raw / 1000.0, tz=timezone.utc)
            from datetime import datetime, timezone
            return datetime.fromtimestamp(raw, tz=timezone.utc)

        raise SignalNormalizationError("unsupported timestamp type in payload")

    def _extract_domain_data(self, payload: dict[str, Any]) -> dict[str, Any]:
        excluded = {
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
        return {key: value for key, value in payload.items() if key not in excluded}

    def _extract_features(
        self,
        *,
        source: FeatureSource,
        symbol: str,
        payload: dict[str, Any],
        timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        explicit = payload.get("features")

        if explicit is None:
            return self._build_implicit_features(
                source=source,
                symbol=symbol,
                payload=payload,
                timestamp=timestamp,
            )

        if not isinstance(explicit, list):
            raise SignalNormalizationError("payload['features'] must be a list")

        result: list[FeatureSnapshot] = []

        for item in explicit:
            if not isinstance(item, dict):
                raise SignalNormalizationError("each feature item must be a dict")

            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                raise SignalNormalizationError("feature item must contain non-empty 'name'")

            snapshot = FeatureSnapshot(
                name=name.strip(),
                value=item.get("value"),
                source=source,
                symbol=symbol,
                timestamp=timestamp,
                confidence=self._safe_confidence(
                    item.get("confidence", payload.get("confidence", 0.0))
                ),
                normalized_value=self._safe_normalized_value(
                    item.get("normalized_value")
                ),
                freshness_seconds=self._resolve_freshness_seconds(
                    feature_name=name.strip(),
                    explicit=item.get("freshness_seconds"),
                ),
                metadata=dict(item.get("metadata") or {}),
            )
            snapshot.validate()
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
        excluded = {
            "symbol",
            "instrument",
            "market",
            "timestamp",
            "ts",
            "source",
            "metadata",
        }

        base_confidence = self._safe_confidence(payload.get("confidence", 0.0))
        result: list[FeatureSnapshot] = []

        for key, value in payload.items():
            if key in excluded or key.startswith("_"):
                continue

            if isinstance(value, (int, float, bool, str)):
                snapshot = FeatureSnapshot(
                    name=key,
                    value=value,
                    source=source,
                    symbol=symbol,
                    timestamp=timestamp,
                    confidence=base_confidence,
                    normalized_value=self._infer_normalized_value(value),
                    freshness_seconds=self._resolve_freshness_seconds(key, None),
                    metadata={},
                )
                snapshot.validate()
                result.append(snapshot)

        return result

    def _resolve_freshness_seconds(
        self,
        feature_name: str,
        explicit: Any,
    ) -> float | None:
        if explicit is not None:
            if not isinstance(explicit, (int, float)) or explicit <= 0:
                raise SignalNormalizationError(
                    f"freshness_seconds must be positive for feature '{feature_name}'"
                )
            return float(explicit)

        return float(self.config.freshness.get_ttl(feature_name))

    def _safe_confidence(self, value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return clamp(float(value), 0.0, 1.0)
        raise SignalNormalizationError(f"unsupported confidence value: {value!r}")

    def _safe_normalized_value(self, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return clamp(float(value), -1.0, 1.0)
        raise SignalNormalizationError(f"unsupported normalized_value: {value!r}")

    def _infer_normalized_value(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            numeric = float(value)
            if -1.0 <= numeric <= 1.0:
                return numeric
        return None


class SignalRouter(BaseStrategyComponent):
    """
    Selects strategies for an incoming normalized analytics event.
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

    @property
    def routing_config(self) -> RoutingConfig:
        return self.config.routing

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

        resolved_source = source or self._resolve_source_from_event_name(event_name)
        changed_features = changed_features or []

        decision = RouteDecision(
            event_name=event_name,
            symbol=context.symbol,
            source=resolved_source,
            timestamp=context.timestamp,
            metadata=metadata or {},
        )

        candidates = self._collect_candidates(
            event_name=event_name,
            source=resolved_source,
            changed_features=changed_features,
            decision=decision,
        )

        decision.selected = self._filter_applicable(
            candidates=candidates,
            context=context,
            changed_features=changed_features,
            decision=decision,
        )

        self.log_debug(
            "Routing completed",
            event_name=event_name,
            symbol=context.symbol,
            selected=decision.selected_names,
            skipped=decision.skipped,
        )
        return decision

    def route_from_features(
        self,
        *,
        context: StrategyContext,
        features: list[FeatureSnapshot],
        event_name: str = "strategy.feature_update",
        metadata: dict[str, Any] | None = None,
    ) -> RouteDecision:
        return self.route(
            event_name=event_name,
            context=context,
            source=features[0].source if features else None,
            changed_features=[feature.name for feature in features],
            metadata=metadata,
        )

    def _collect_candidates(
        self,
        *,
        event_name: str,
        source: FeatureSource | None,
        changed_features: list[str],
        decision: RouteDecision,
    ) -> list[BaseStrategy]:
        candidates: dict[str, BaseStrategy] = {}

        categories = self._resolve_categories_for_event(event_name, source)
        if categories:
            decision.categories_used.extend(categories)
            for strategy in self.registry.find_by_categories(categories):
                candidates[strategy.strategy_name] = strategy

        for feature_name in changed_features:
            matched = self.registry.find_by_required_feature(feature_name)
            if matched:
                decision.matched_features.append(feature_name)

            for strategy in matched:
                candidates[strategy.strategy_name] = strategy

        if self.routing_config.reevaluate_on_any_update and not candidates:
            for strategy in self.registry.list_enabled():
                candidates[strategy.strategy_name] = strategy

        if (
            self.routing_config.route_hybrid_on_domain_signal
            and source is not None
            and source != FeatureSource.SYSTEM
        ):
            for strategy in self.registry.list_by_category(StrategyCategory.HYBRID, only_enabled=True):
                candidates[strategy.strategy_name] = strategy
                if StrategyCategory.HYBRID not in decision.categories_used:
                    decision.categories_used.append(StrategyCategory.HYBRID)

        return sorted(
            candidates.values(),
            key=lambda strategy: (strategy.priority, strategy.strategy_name),
        )

    def _filter_applicable(
        self,
        *,
        candidates: list[BaseStrategy],
        context: StrategyContext,
        changed_features: list[str],
        decision: RouteDecision,
    ) -> list[BaseStrategy]:
        applicable: list[BaseStrategy] = []

        for strategy in candidates:
            reason = self._skip_reason(
                strategy=strategy,
                context=context,
                changed_features=changed_features,
            )
            if reason is not None:
                decision.skipped[strategy.strategy_name] = reason
                continue

            applicable.append(strategy)

        return sorted(
            applicable,
            key=lambda strategy: (strategy.priority, strategy.strategy_name),
        )

    def _skip_reason(
        self,
        *,
        strategy: BaseStrategy,
        context: StrategyContext,
        changed_features: list[str],
    ) -> str | None:
        if not strategy.is_enabled():
            return "strategy_disabled"

        runtime = self.config.get_strategy_runtime(strategy.strategy_name)

        if not runtime.allows_symbol(context.symbol):
            return "symbol_not_allowed"

        if not runtime.allows_timeframe(context.timeframe):
            return "timeframe_not_supported"

        if not runtime.allows_regime(context.current_regime):
            return "regime_not_supported"

        required = strategy.required_features()
        missing = [name for name in required if not context.has_feature(name)]
        if missing and not self.routing_config.allow_partial_context:
            return f"missing_required_features:{','.join(sorted(missing))}"

        stale = [
            name
            for name in required
            if context.has_feature(name) and self._feature_is_stale(context, name)
        ]
        if stale:
            return f"stale_required_features:{','.join(sorted(stale))}"

        if changed_features and not self._strategy_relevant_for_feature_change(
            strategy,
            changed_features,
        ):
            return "no_relevant_feature_change"

        try:
            if not strategy.should_evaluate(context):
                return "strategy_should_evaluate_false"
        except Exception as exc:
            return f"strategy_should_evaluate_error:{exc}"

        return None

    def _feature_is_stale(self, context: StrategyContext, feature_name: str) -> bool:
        snapshot = context.get_feature_snapshot(feature_name)
        if snapshot is None:
            return True

        ttl = context.freshness_map.get(
            feature_name,
            self.routing_config.stale_feature_threshold_seconds,
        )
        return snapshot.age_seconds(context.timestamp) > ttl

    def _strategy_relevant_for_feature_change(
        self,
        strategy: BaseStrategy,
        changed_features: list[str],
    ) -> bool:
        required = strategy.required_features()
        if not required:
            return True
        return bool(required.intersection(changed_features))

    def _resolve_categories_for_event(
        self,
        event_name: str,
        source: FeatureSource | None,
    ) -> list[StrategyCategory]:
        configured = self.routing_config.event_to_categories.get(event_name)
        if configured:
            return list(dict.fromkeys(configured))

        event_lower = event_name.lower()
        categories: list[StrategyCategory] = []

        for configured_event, configured_categories in self.routing_config.event_to_categories.items():
            if configured_event and configured_event.lower() in event_lower:
                categories.extend(configured_categories)

        if categories:
            return list(dict.fromkeys(categories))

        if source is not None:
            mapped = self._map_source_to_category(source)
            if mapped is not None:
                categories.append(mapped)

        return list(dict.fromkeys(categories))

    def _resolve_source_from_event_name(self, event_name: str) -> FeatureSource | None:
        return SignalNormalizer(self.config)._resolve_source_from_text(event_name)

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


class SignalScorer(BaseStrategyComponent):
    """
    Scoring facade:
    - weights;
    - voting;
    - conflicts;
    - confidence aggregation.
    """

    component_namespace = "strategy.processor.scorer"

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
            context=context,
        )

        result = self._to_confluence_result(
            symbol=symbol,
            timestamp=context.timestamp if context is not None else max(s.timestamp for s in signals),
            weighted_signals=weighted,
            vote_summary=vote_summary,
            conflict_summary=conflict_summary,
        )
        result.validate()
        return result

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
                    "category": str(signal.category),
                    "regime": str(regime),
                },
            )
            weighted.validate()
            result.append(weighted)

        return result

    def _summarize_votes(self, signals: list[StrategySignal]) -> VoteSummary:
        summary = VoteSummary(total_votes=len(signals))

        for signal in signals:
            if signal.side == SignalSide.LONG:
                summary.long_votes += 1
            elif signal.side == SignalSide.SHORT:
                summary.short_votes += 1
            elif signal.side == SignalSide.FLAT:
                summary.flat_votes += 1

            if signal.trigger_type == TriggerType.CONFIRMATION:
                summary.confirmation_count += 1
            if signal.trigger_type == TriggerType.PRIMARY:
                summary.primary_count += 1

        if summary.long_votes > summary.short_votes and summary.long_votes > 0:
            summary.dominant_side = SignalSide.LONG
        elif summary.short_votes > summary.long_votes and summary.short_votes > 0:
            summary.dominant_side = SignalSide.SHORT
        elif summary.flat_votes > 0 and summary.long_votes == 0 and summary.short_votes == 0:
            summary.dominant_side = SignalSide.FLAT
        else:
            summary.dominant_side = SignalSide.UNKNOWN

        reasons: list[str] = []

        if summary.total_votes < self.config.voting.min_total_votes:
            reasons.append("not_enough_total_votes")

        if summary.confirmation_count < self.config.voting.min_confirmations:
            reasons.append("not_enough_confirmations")

        if self.config.voting.require_primary_trigger and summary.primary_count < 1:
            reasons.append("primary_trigger_required")

        if summary.dominant_side == SignalSide.UNKNOWN:
            reasons.append("no_dominant_side")

        summary.reasons = reasons
        summary.accepted = not reasons
        return summary

    def _resolve_conflicts(
        self,
        *,
        signals: list[StrategySignal],
        dominant_side: SignalSide,
        context: StrategyContext | None,
    ) -> ConflictSummary:
        summary = ConflictSummary()

        if dominant_side in {SignalSide.LONG, SignalSide.SHORT}:
            opposite = SignalSide.SHORT if dominant_side == SignalSide.LONG else SignalSide.LONG

            for signal in signals:
                if signal.side == opposite:
                    summary.add_conflict(
                        ConflictRecord(
                            conflict_type=ConflictType.SIDE_CONFLICT,
                            source=signal.strategy_name,
                            message=f"signal side {signal.side.value} conflicts with dominant side {dominant_side.value}",
                            penalty=self.config.confluence.conflict_penalty,
                        )
                    )

        if context is not None:
            for signal in signals:
                if (
                    signal.regime != MarketRegime.UNKNOWN
                    and context.current_regime != MarketRegime.UNKNOWN
                    and signal.regime != context.current_regime
                ):
                    summary.add_conflict(
                        ConflictRecord(
                            conflict_type=ConflictType.REGIME_CONFLICT,
                            source=signal.strategy_name,
                            message=f"signal regime {signal.regime.value} conflicts with context regime {context.current_regime.value}",
                            penalty=self.config.confluence.conflict_penalty,
                        )
                    )

        if self.config.conflict.reject_on_side_conflict and any(
            c.conflict_type == ConflictType.SIDE_CONFLICT for c in summary.conflicts
        ):
            summary.reasons.append("rejected_on_side_conflict")

        if self.config.conflict.reject_on_regime_conflict and any(
            c.conflict_type == ConflictType.REGIME_CONFLICT for c in summary.conflicts
        ):
            summary.reasons.append("rejected_on_regime_conflict")

        if summary.total_penalty > self.config.conflict.max_total_penalty:
            summary.reasons.append("conflict_penalty_too_high")

        summary.accepted = not summary.reasons
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
            return ConfluenceResult(
                symbol=symbol,
                timestamp=timestamp,
                accepted=False,
                reasons=["no_weighted_signals"],
            )

        dominant_side = vote_summary.dominant_side
        side_signals = [
            item
            for item in weighted_signals
            if item.signal.side == dominant_side
        ]

        if not side_signals:
            return ConfluenceResult(
                symbol=symbol,
                timestamp=timestamp,
                side=dominant_side,
                accepted=False,
                reasons=["no_signals_for_dominant_side"],
            )

        total_weight = sum(item.final_weight for item in side_signals)
        if total_weight <= 0:
            confidence = 0.0
            score = 0.0
        else:
            score = sum(item.weighted_score for item in side_signals) / total_weight
            confidence = sum(item.weighted_confidence for item in side_signals) / total_weight

        confidence = clamp(confidence - conflict_summary.total_penalty, 0.0, 1.0)
        score = max(0.0, score - conflict_summary.total_penalty)

        reasons: list[str] = []
        confirmations: list[str] = []

        for item in side_signals:
            for reason in item.signal.reasons:
                if reason not in reasons:
                    reasons.append(reason)

            for confirmation in item.signal.confirmations:
                if confirmation not in confirmations:
                    confirmations.append(confirmation)

        for reason in vote_summary.reasons:
            if reason not in reasons:
                reasons.append(f"vote:{reason}")

        for reason in conflict_summary.reasons:
            if reason not in reasons:
                reasons.append(f"conflict:{reason}")

        accepted = (
            vote_summary.accepted
            and conflict_summary.accepted
            and confidence >= self.config.confluence.min_confidence
            and score >= self.config.confluence.min_score
        )

        return ConfluenceResult(
            symbol=symbol,
            timestamp=timestamp,
            side=dominant_side,
            score=score,
            confidence=confidence,
            confidence_grade=confidence_to_grade(confidence),
            strength=confidence_to_strength(confidence),
            strategy_names=[item.signal.strategy_name for item in side_signals],
            reasons=reasons,
            confirmations=confirmations,
            conflicts=conflict_summary.conflicts,
            accepted=accepted,
            metadata={
                "total_weight": total_weight,
                "vote_summary": {
                    "total_votes": vote_summary.total_votes,
                    "long_votes": vote_summary.long_votes,
                    "short_votes": vote_summary.short_votes,
                    "dominant_side": str(vote_summary.dominant_side),
                    "accepted": vote_summary.accepted,
                    "reasons": vote_summary.reasons,
                },
                "conflict_summary": {
                    "accepted": conflict_summary.accepted,
                    "total_penalty": conflict_summary.total_penalty,
                    "reasons": conflict_summary.reasons,
                },
            },
        )


class ConfluenceEngine(BaseStrategyComponent):
    """
    Merges multiple strategy signals into consensus.
    """

    component_namespace = "strategy.processor.confluence"

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, scheduler=scheduler)
        self.scorer = SignalScorer(config=config, event_bus=event_bus, scheduler=scheduler)

    @property
    def confluence_config(self) -> ConfluenceConfig:
        return self.config.confluence

    def evaluate(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext | None = None,
        merge_signal: bool = True,
    ) -> ConfluenceEvaluation:
        if not signals:
            raise ConfluenceError("signals cannot be empty")

        for signal in signals:
            signal.validate()

        symbol = self._ensure_same_symbol(signals)
        timestamp = context.timestamp if context is not None else max(s.timestamp for s in signals)

        evaluation = ConfluenceEvaluation(
            symbol=symbol,
            timestamp=timestamp,
            raw_signals=list(signals),
        )

        eligible, rejected = self._pre_filter_signals(signals=signals, context=context)
        evaluation.eligible_signals = eligible
        evaluation.rejected_signals = rejected

        if not eligible:
            evaluation.result = ConfluenceResult(
                symbol=symbol,
                timestamp=timestamp,
                accepted=False,
                reasons=["no_eligible_signals"],
            )
            evaluation.reasons.append("no_eligible_signals")
            return evaluation

        result = self.scorer.score_signals(signals=eligible, context=context)
        evaluation.result = result

        accepted, rejected_by_side = self._select_consensus_signals(
            signals=eligible,
            dominant_side=result.side,
        )
        evaluation.accepted_signals = accepted
        evaluation.rejected_signals.update(rejected_by_side)

        acceptance_reasons = self._evaluate_acceptance(
            result=result,
            accepted_signals=accepted,
        )
        if acceptance_reasons:
            result.accepted = False
            for reason in acceptance_reasons:
                if reason not in result.reasons:
                    result.reasons.append(reason)
                if reason not in evaluation.reasons:
                    evaluation.reasons.append(reason)

        if merge_signal and result.accepted and accepted:
            evaluation.merged_signal = self._merge_signals(
                signals=accepted,
                result=result,
                context=context,
            )

        return evaluation

    def merge_only(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext | None = None,
    ) -> StrategySignal:
        evaluation = self.evaluate(signals=signals, context=context, merge_signal=True)
        if evaluation.merged_signal is None:
            raise ConfluenceError("unable to build merged signal from provided signals")
        return evaluation.merged_signal

    def _ensure_same_symbol(self, signals: list[StrategySignal]) -> str:
        symbol = signals[0].symbol
        if any(signal.symbol != symbol for signal in signals):
            raise ConfluenceError("all signals must belong to the same symbol")
        return symbol

    def _pre_filter_signals(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext | None,
    ) -> tuple[list[StrategySignal], dict[str, str]]:
        eligible: list[StrategySignal] = []
        rejected: dict[str, str] = {}

        for signal in signals:
            reason = self._pre_filter_reason(signal=signal, context=context)
            if reason is not None:
                rejected[signal.strategy_name] = reason
                continue
            eligible.append(signal)

        return eligible, rejected

    def _pre_filter_reason(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext | None,
    ) -> str | None:
        if signal.status in {
            SignalStatus.REJECTED,
            SignalStatus.CANCELLED,
            SignalStatus.EXPIRED,
            SignalStatus.FAILED,
        }:
            return f"signal_status:{signal.status.value}"

        if not signal.is_directional:
            return "non_directional_signal"

        if signal.confidence < 0:
            return "negative_confidence"

        if signal.score < 0:
            return "negative_score"

        if context is not None and signal.symbol != context.symbol:
            return "symbol_mismatch"

        return None

    def _select_consensus_signals(
        self,
        *,
        signals: list[StrategySignal],
        dominant_side: SignalSide,
    ) -> tuple[list[StrategySignal], dict[str, str]]:
        if dominant_side not in {SignalSide.LONG, SignalSide.SHORT}:
            return [], {
                signal.strategy_name: "no_dominant_side"
                for signal in signals
            }

        accepted: list[StrategySignal] = []
        rejected: dict[str, str] = {}

        for signal in signals:
            if signal.side == dominant_side:
                accepted.append(signal)
            else:
                rejected[signal.strategy_name] = "not_in_consensus_side"

        return accepted, rejected

    def _evaluate_acceptance(
        self,
        *,
        result: ConfluenceResult,
        accepted_signals: list[StrategySignal],
    ) -> list[str]:
        reasons: list[str] = []

        if not result.accepted:
            return reasons

        if len(accepted_signals) < self.confluence_config.min_agreement_count:
            reasons.append("not_enough_agreeing_strategies")

        if result.confidence < self.confluence_config.min_confidence:
            reasons.append("confluence_confidence_too_low")

        if result.score < self.confluence_config.min_score:
            reasons.append("confluence_score_too_low")

        return reasons

    def _merge_signals(
        self,
        *,
        signals: list[StrategySignal],
        result: ConfluenceResult,
        context: StrategyContext | None,
    ) -> StrategySignal:
        first = signals[0]
        timestamp = context.timestamp if context is not None else result.timestamp

        reasons: list[str] = list(result.reasons)
        confirmations: list[str] = list(result.confirmations)
        source_features: list[str] = []
        combined_from: list[str] = []

        for signal in signals:
            combined_from.append(signal.strategy_name)

            for feature_name in signal.source_features:
                if feature_name not in source_features:
                    source_features.append(feature_name)

        return StrategySignal(
            symbol=first.symbol,
            side=result.side,
            strategy_name="ConfluenceEngine",
            category=StrategyCategory.HYBRID,
            timeframe=first.timeframe,
            setup_type=first.setup_type,
            timestamp=timestamp,
            confidence=result.confidence,
            score=result.score,
            strength=result.strength,
            confidence_grade=result.confidence_grade,
            status=SignalStatus.NEW,
            trigger_type=TriggerType.CONFLUENCE,
            origin=SignalOrigin.CONFLUENCE,
            priority=SignalPriority.HIGH,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            combined_from=combined_from,
            conflicts=list(result.conflicts),
            regime=context.current_regime if context is not None else first.regime,
            metadata={
                "confluence": result.metadata,
                "source_signal_count": len(signals),
            },
        )


class SignalFilterChain(ContextAwareStrategyComponent):
    """
    Applies common strategy filters.
    """

    component_namespace = "strategy.processor.filters"

    @property
    def filters_config(self) -> FilterConfig:
        return self.config.filters

    def evaluate(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> FilterEvaluation:
        self.validate_context(context)
        signal.validate()

        if signal.symbol != context.symbol:
            raise FilterExecutionError("signal symbol must match context symbol")

        evaluation = FilterEvaluation(
            signal=signal,
            context_symbol=context.symbol,
            timestamp=context.timestamp,
        )

        for result in (
            self._regime_filter(signal, context),
            self._volatility_filter(signal, context),
            self._liquidity_filter(signal, context),
            self._spread_filter(signal, context),
            self._funding_filter(signal, context),
        ):
            evaluation.add_result(result)

        signal.filter_results.extend(evaluation.results)

        for result in evaluation.results:
            if result.score_impact:
                signal.score = max(0.0, signal.score + result.score_impact)

        signal.validate()
        return evaluation

    def apply(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        accepted: list[StrategySignal] = []

        for signal in signals:
            evaluation = self.evaluate(signal=signal, context=context)
            if evaluation.accepted:
                accepted.append(signal)
            else:
                signal.to_rejected()
                signal.add_reason("blocked_by_filter_chain")

        return accepted

    def _pass(self, name: str, reason: str, **metadata: Any) -> FilterResult:
        return FilterResult(
            name=name,
            decision=FilterDecision.PASS,
            reason=reason,
            metadata=metadata,
        )

    def _warn(
        self,
        name: str,
        reason: str,
        *,
        score_impact: float = 0.0,
        **metadata: Any,
    ) -> FilterResult:
        return FilterResult(
            name=name,
            decision=FilterDecision.WARN,
            reason=reason,
            score_impact=score_impact,
            metadata=metadata,
        )

    def _block(
        self,
        name: str,
        reason: str,
        *,
        score_impact: float = 0.0,
        **metadata: Any,
    ) -> FilterResult:
        return FilterResult(
            name=name,
            decision=FilterDecision.BLOCK,
            reason=reason,
            score_impact=score_impact,
            metadata=metadata,
        )

    def _skip(self, name: str, reason: str) -> FilterResult:
        return FilterResult(
            name=name,
            decision=FilterDecision.SKIP,
            reason=reason,
        )

    def _regime_filter(
        self,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> FilterResult:
        name = "regime_filter"

        if not self.filters_config.enable_regime_filter:
            return self._skip(name, "regime_filter_disabled")

        regime = context.current_regime

        if regime == MarketRegime.UNKNOWN:
            return self._warn(name, "unknown_market_regime", score_impact=-0.05)

        if regime == MarketRegime.ILLIQUID:
            return self._block(name, "illiquid_market_regime", score_impact=-0.25)

        if regime == MarketRegime.NEWS_DRIVEN:
            return self._warn(name, "news_driven_market_regime", score_impact=-0.15)

        if regime == MarketRegime.RISK_OFF:
            return self._warn(name, "risk_off_market_regime", score_impact=-0.10)

        return self._pass(name, "regime_ok", regime=str(regime))

    def _volatility_filter(
        self,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> FilterResult:
        name = "volatility_filter"

        if not self.filters_config.enable_volatility_filter:
            return self._skip(name, "volatility_filter_disabled")

        value = context.get_feature("volatility_zscore")
        if value is None:
            value = context.get_feature("realized_volatility_zscore")

        if value is None:
            return self._warn(name, "volatility_data_missing", score_impact=-0.03)

        if not isinstance(value, (int, float)):
            return self._warn(name, "invalid_volatility_value", score_impact=-0.05)

        threshold = self.filters_config.max_volatility_zscore
        value = float(value)

        if value > threshold * 1.5:
            return self._block(
                name,
                "volatility_too_high",
                score_impact=-0.25,
                volatility_zscore=value,
                threshold=threshold,
            )

        if value > threshold:
            return self._warn(
                name,
                "elevated_volatility",
                score_impact=-0.10,
                volatility_zscore=value,
                threshold=threshold,
            )

        return self._pass(name, "volatility_ok", volatility_zscore=value)

    def _liquidity_filter(
        self,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> FilterResult:
        name = "liquidity_filter"

        if not self.filters_config.enable_liquidity_filter:
            return self._skip(name, "liquidity_filter_disabled")

        value = context.get_feature("liquidity_score")
        if value is None:
            value = context.liquidity.get("liquidity_score")

        if value is None:
            return self._warn(name, "liquidity_data_missing", score_impact=-0.05)

        if not isinstance(value, (int, float)):
            return self._warn(name, "invalid_liquidity_score", score_impact=-0.05)

        value = float(value)
        threshold = self.filters_config.min_liquidity_score

        if value < threshold * 0.5:
            return self._block(
                name,
                "liquidity_too_low",
                score_impact=-0.25,
                liquidity_score=value,
            )

        if value < threshold:
            return self._warn(
                name,
                "suboptimal_liquidity",
                score_impact=-0.10,
                liquidity_score=value,
            )

        return self._pass(name, "liquidity_ok", liquidity_score=value)

    def _spread_filter(
        self,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> FilterResult:
        name = "spread_filter"

        if not self.filters_config.enable_spread_filter:
            return self._skip(name, "spread_filter_disabled")

        value = context.price.spread_bps if context.price is not None else None
        if value is None:
            value = context.get_feature("spread_bps")

        if value is None:
            return self._warn(name, "spread_data_missing", score_impact=-0.03)

        if not isinstance(value, (int, float)):
            return self._warn(name, "invalid_spread_value", score_impact=-0.05)

        value = float(value)
        threshold = self.filters_config.max_spread_bps

        if value > threshold * 2:
            return self._block(
                name,
                "spread_too_wide",
                score_impact=-0.25,
                spread_bps=value,
            )

        if value > threshold:
            return self._warn(
                name,
                "spread_elevated",
                score_impact=-0.10,
                spread_bps=value,
            )

        return self._pass(name, "spread_ok", spread_bps=value)

    def _funding_filter(
        self,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> FilterResult:
        name = "funding_filter"

        if not self.filters_config.enable_funding_filter:
            return self._skip(name, "funding_filter_disabled")

        alignment = context.get_feature("funding_alignment")
        if alignment is None:
            alignment = context.funding.get("funding_alignment")

        if alignment is None:
            return self._warn(name, "funding_data_missing", score_impact=-0.02)

        if not isinstance(alignment, (int, float)):
            return self._warn(name, "invalid_funding_alignment", score_impact=-0.05)

        alignment = float(alignment)
        threshold = self.filters_config.min_funding_alignment

        if alignment < threshold:
            return self._warn(
                name,
                "funding_alignment_weak",
                score_impact=-0.05,
                funding_alignment=alignment,
            )

        return self._pass(name, "funding_ok", funding_alignment=alignment)


class SignalBuilder(ContextAwareStrategyComponent):
    """
    Builds entry / invalidation / targets / exit / execution plan.
    """

    component_namespace = "strategy.processor.builder"

    @property
    def builders_config(self) -> BuilderConfig:
        return self.config.builders

    def build(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> BuildEvaluation:
        self.validate_context(context)
        signal.validate()

        if signal.symbol != context.symbol:
            raise BuilderError("signal symbol must match context symbol")

        evaluation = BuildEvaluation(
            signal=signal,
            context_symbol=context.symbol,
        )

        try:
            entry = self._build_entry(signal=signal, context=context)
            invalidation = self._build_invalidation(
                signal=signal,
                context=context,
                entry=entry,
            )
            targets = self._build_targets(
                signal=signal,
                context=context,
                entry=entry,
                invalidation=invalidation,
            )
            exit_plan = self._build_exit(
                signal=signal,
                context=context,
                invalidation=invalidation,
                targets=targets,
            )
            execution_plan = ExecutionPlanDraft(
                symbol=signal.symbol,
                side=signal.side,
                entry=entry,
                exit=exit_plan,
                invalidation=invalidation,
                leverage=self._feature_float(context, "leverage"),
                reduce_only=bool(context.get_feature("reduce_only", False)),
                post_only=bool(context.get_feature("post_only", False)),
                expected_holding_seconds=self._feature_int(
                    context,
                    "expected_holding_seconds",
                ),
                notes=list(signal.reasons[:5]),
                metadata={"builder": self.component_name},
            )
            execution_plan.validate()

        except Exception as exc:
            evaluation.reject(str(exc))
            return evaluation

        signal.entry_plan = entry
        signal.invalidation_plan = invalidation
        signal.exit_plan = exit_plan
        signal.execution_plan = execution_plan
        signal.validate()

        evaluation.entry = entry
        evaluation.invalidation = invalidation
        evaluation.targets = targets
        evaluation.exit_plan = exit_plan
        evaluation.execution_plan = execution_plan
        return evaluation

    def _build_entry(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> EntryPlan:
        if signal.entry_plan is not None:
            signal.entry_plan.validate()
            return signal.entry_plan

        price = self._reference_price(signal=signal, context=context)
        plan = EntryPlan(
            entry_type=self._entry_type(context),
            price=price,
            timeout_seconds=self._feature_int(context, "entry_timeout_seconds"),
            max_slippage_bps=self._feature_float(context, "max_slippage_bps"),
            confirmation_required=bool(context.get_feature("entry_confirmation_required", False)),
            notes=[f"entry_type:{self._entry_type(context).value}"],
            metadata={"builder": "SignalBuilder"},
        )
        plan.validate()
        return plan

    def _build_invalidation(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
        entry: EntryPlan,
    ) -> InvalidationPlan:
        if signal.invalidation_plan is not None:
            signal.invalidation_plan.validate()
            return signal.invalidation_plan

        if not self.builders_config.require_invalidation:
            return InvalidationPlan(reason="invalidation_not_required")

        if entry.price is None:
            raise BuilderError("entry price is required to build invalidation")

        explicit = self._feature_float(context, "invalidation_price")
        if explicit is None:
            explicit = self._feature_float(context, "stop_loss")

        if explicit is None:
            fallback_pct = self._feature_float(context, "fallback_stop_pct", 0.003)
            if signal.is_long:
                explicit = entry.price * (1.0 - fallback_pct)
            elif signal.is_short:
                explicit = entry.price * (1.0 + fallback_pct)

        plan = InvalidationPlan(
            price=explicit,
            reason=str(context.get_feature("invalidation_reason", "setup_invalidated")),
            timeout_seconds=self._feature_int(context, "invalidation_timeout_seconds"),
            conditions=self._list_feature(context, "invalidation_conditions"),
            metadata={"builder": "SignalBuilder"},
        )
        plan.validate()
        return plan

    def _build_targets(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
        entry: EntryPlan,
        invalidation: InvalidationPlan,
    ) -> list[TargetPlan]:
        explicit = context.get_feature("targets")
        if isinstance(explicit, list):
            targets: list[TargetPlan] = []
            for item in explicit:
                if isinstance(item, TargetPlan):
                    item.validate()
                    targets.append(item)
                elif isinstance(item, dict):
                    target = TargetPlan(
                        price=float(item["price"]),
                        size_fraction=float(item.get("size_fraction", 1.0)),
                        rr=float(item["rr"]) if item.get("rr") is not None else None,
                        label=item.get("label"),
                        metadata=dict(item.get("metadata") or {}),
                    )
                    target.validate()
                    targets.append(target)
            if targets:
                return targets

        if entry.price is None or invalidation.price is None:
            raise BuilderError("entry and invalidation prices are required to build targets")

        risk = abs(entry.price - invalidation.price)
        if risk <= 0:
            raise BuilderError("risk distance must be > 0")

        rr = self._feature_float(context, "rr_ratio", self.builders_config.default_rr_ratio)
        levels = self.builders_config.default_partial_tp_levels

        targets: list[TargetPlan] = []
        for index, fraction in enumerate(levels, start=1):
            if signal.is_long:
                price = entry.price + risk * rr * index
            else:
                price = entry.price - risk * rr * index

            target = TargetPlan(
                price=price,
                size_fraction=fraction,
                rr=rr * index,
                label=f"tp_{index}",
                metadata={"builder": "SignalBuilder"},
            )
            target.validate()
            targets.append(target)

        return targets

    def _build_exit(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
        invalidation: InvalidationPlan,
        targets: list[TargetPlan],
    ) -> ExitPlan:
        if signal.exit_plan is not None:
            signal.exit_plan.validate()
            return signal.exit_plan

        exit_types = [ExitType.TAKE_PROFIT, ExitType.STOP_LOSS]
        max_holding = self._feature_int(context, "max_holding_seconds")

        plan = ExitPlan(
            exit_types=exit_types,
            stop_loss=invalidation.price,
            take_profit_levels=targets,
            trailing_distance=self._feature_float(context, "trailing_distance"),
            max_holding_seconds=max_holding,
            partial_exit_enabled=self.builders_config.enable_partial_take_profit,
            metadata={"builder": "SignalBuilder"},
        )
        plan.validate()
        return plan

    def _reference_price(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> float:
        if signal.entry_plan is not None and signal.entry_plan.price is not None:
            return float(signal.entry_plan.price)

        if context.mid_price is not None:
            return float(context.mid_price)

        entry_price = self._feature_float(context, "entry_price")
        if entry_price is not None and entry_price > 0:
            return entry_price

        raise BuilderError("unable to resolve reference price")

    def _entry_type(self, context: StrategyContext) -> EntryType:
        raw = context.get_feature("entry_type")
        if isinstance(raw, EntryType):
            return raw
        if isinstance(raw, str):
            try:
                return EntryType(raw)
            except ValueError:
                pass
        return self.builders_config.default_entry_type

    def _feature_float(
        self,
        context: StrategyContext,
        name: str,
        default: float | None = None,
    ) -> float | None:
        value = context.get_feature(name)
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        return default

    def _feature_int(
        self,
        context: StrategyContext,
        name: str,
        default: int | None = None,
    ) -> int | None:
        value = context.get_feature(name)
        if value is None:
            return default
        if isinstance(value, int):
            return value
        return default

    def _list_feature(self, context: StrategyContext, name: str) -> list[str]:
        value = context.get_feature(name)
        if isinstance(value, list):
            return [str(item) for item in value]
        return []


class PortfolioCoordinator(BaseStrategyComponent):
    """
    Final portfolio-level signal coordination.
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

    @property
    def portfolio_config(self) -> PortfolioCoordinatorConfig:
        return self.config.portfolio

    def coordinate(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext | None = None,
    ) -> CoordinationDecision:
        if not signals:
            raise PortfolioCoordinationError("signals cannot be empty")

        for signal in signals:
            signal.validate()

        symbol = self._ensure_same_symbol(signals)
        now = context.timestamp if context is not None else utcnow()

        decision = CoordinationDecision(
            symbol=symbol,
            timestamp=now,
            raw_signals=list(signals),
        )

        accepted, rejected = self._apply_prechecks(signals=signals, context=context)
        decision.accepted_signals = accepted
        decision.rejected_signals.update(rejected)

        if not accepted:
            decision.accepted = False
            decision.reasons.append("no_signals_after_prechecks")
            return decision

        accepted, suppressed = self._suppress_repeating_signals(
            symbol=symbol,
            signals=accepted,
            now=now,
        )
        decision.accepted_signals = accepted
        decision.suppressed_signals.update(suppressed)
        decision.rejected_signals.update(suppressed)

        accepted, rejected_dedup = self._deduplicate_by_side(accepted)
        decision.accepted_signals = accepted
        decision.rejected_signals.update(rejected_dedup)

        accepted, rejected_limits = self._apply_symbol_limits(symbol, accepted)
        decision.accepted_signals = accepted
        decision.rejected_signals.update(rejected_limits)

        if not accepted:
            decision.accepted = False
            decision.reasons.append("no_signals_after_symbol_limits")
            return decision

        merged = self._merge_similar_signals(accepted) if self.portfolio_config.merge_similar_signals else accepted
        decision.merged_signals = merged

        if not merged:
            decision.accepted = False
            decision.reasons.append("no_signals_after_merge")
            return decision

        self._update_state_after_acceptance(symbol=symbol, signals=merged)

        return decision

    def coordinate_one(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext | None = None,
    ) -> CoordinationDecision:
        return self.coordinate(signals=[signal], context=context)

    def _ensure_same_symbol(self, signals: list[StrategySignal]) -> str:
        symbol = signals[0].symbol
        if any(signal.symbol != symbol for signal in signals):
            raise PortfolioCoordinationError("all signals must belong to the same symbol")
        return symbol

    def _apply_prechecks(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext | None,
    ) -> tuple[list[StrategySignal], dict[str, str]]:
        symbol = signals[0].symbol
        accepted: list[StrategySignal] = []
        rejected: dict[str, str] = {}

        if self.state.is_symbol_blocked(symbol):
            return [], {signal.strategy_name: "symbol_blocked" for signal in signals}

        for signal in signals:
            if self.state.is_blocked_by_cooldown(
                symbol=symbol,
                strategy_name=signal.strategy_name,
                side=signal.side,
                now=context.timestamp if context is not None else None,
            ):
                rejected[signal.strategy_name] = "cooldown_blocked"
                continue

            if signal.status in {
                SignalStatus.REJECTED,
                SignalStatus.CANCELLED,
                SignalStatus.EXPIRED,
                SignalStatus.FAILED,
            }:
                rejected[signal.strategy_name] = f"invalid_signal_status:{signal.status.value}"
                continue

            if not signal.is_directional:
                rejected[signal.strategy_name] = "non_directional_signal"
                continue

            accepted.append(signal)

        return accepted, rejected

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

            challenger_wins = (
                signal.priority.value > current.priority.value
                or signal.confidence > current.confidence
                or signal.score > current.score
            )

            if challenger_wins:
                rejected[current.strategy_name] = f"deduplicated_by_side:{signal.strategy_name}"
                best[signal.side] = signal
            else:
                rejected[signal.strategy_name] = f"deduplicated_by_side:{current.strategy_name}"

        accepted = list(best.values())
        accepted.sort(key=lambda item: (-item.confidence, -item.score, item.strategy_name))
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
        ordered = sorted(signals, key=lambda item: (-item.confidence, -item.score, item.strategy_name))

        accepted = ordered[:available]
        rejected = {
            signal.strategy_name: "max_signals_per_symbol_reached"
            for signal in ordered[available:]
        }
        return accepted, rejected

    def _merge_similar_signals(self, signals: list[StrategySignal]) -> list[StrategySignal]:
        if len(signals) <= 1:
            return signals

        grouped: dict[SignalSide, list[StrategySignal]] = {}
        for signal in signals:
            grouped.setdefault(signal.side, []).append(signal)

        merged: list[StrategySignal] = []
        for side, group in grouped.items():
            if len(group) == 1:
                merged.append(group[0])
                continue

            best = sorted(group, key=lambda item: (-item.confidence, -item.score))[0]
            best.combined_from = list(dict.fromkeys(best.combined_from + [s.strategy_name for s in group]))
            best.confirmations = list(dict.fromkeys(best.confirmations + [r for s in group for r in s.reasons]))
            merged.append(best)

        return merged

    def _update_state_after_acceptance(
        self,
        *,
        symbol: str,
        signals: list[StrategySignal],
    ) -> None:
        for signal in signals:
            self.state.update_signal(signal, active=True)

            cooldown = self.config.get_strategy_runtime(signal.strategy_name).cooldown_seconds
            if cooldown > 0:
                self.state.add_strategy_cooldown(
                    symbol=symbol,
                    strategy_name=signal.strategy_name,
                    seconds=cooldown,
                    reason="signal_accepted",
                )

            if self.portfolio_config.side_cooldown_seconds > 0:
                self.state.add_side_cooldown(
                    symbol=symbol,
                    side=signal.side,
                    seconds=self.portfolio_config.side_cooldown_seconds,
                    reason="side_signal_accepted",
                )


class SignalProcessor(BaseStrategyComponent):
    """
    Facade class for full strategy signal processing.

    Pipeline:
    1. normalize analytics event;
    2. update/build StrategyContext;
    3. route to strategies;
    4. evaluate strategies;
    5. filter signals;
    6. confluence;
    7. build execution plans;
    8. portfolio coordination;
    9. emit signal.generated / signal.rejected events.
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

        raw_signals = [
            evaluation.signal
            for evaluation in evaluations
            if evaluation.signal is not None and evaluation.passed
        ]
        batch.raw_signals = raw_signals

        if not raw_signals:
            batch.reasons.append("no_passed_strategy_signals")
            await self._emit_rejected_batch(batch, reason="no_passed_strategy_signals")
            return batch

        filtered = self.filters.apply(signals=raw_signals, context=context)
        batch.filtered_signals = filtered

        if not filtered:
            batch.reasons.append("all_signals_filtered")
            await self._emit_rejected_batch(batch, reason="all_signals_filtered")
            return batch

        confluence_eval = self.confluence.evaluate(
            signals=filtered,
            context=context,
            merge_signal=True,
        )
        batch.confluence = confluence_eval

        confluence_signals = (
            [confluence_eval.merged_signal]
            if confluence_eval.merged_signal is not None
            else confluence_eval.accepted_signals
        )

        if not confluence_signals:
            batch.reasons.append("no_confluence_signal")
            await self._emit_rejected_batch(batch, reason="no_confluence_signal")
            return batch

        built_signals: list[StrategySignal] = []
        for signal in confluence_signals:
            build_result = self.builder.build(signal=signal, context=context)
            if build_result.accepted:
                built_signals.append(signal)
            else:
                signal.to_rejected()
                signal.reasons.extend(build_result.reasons)

        if not built_signals:
            batch.reasons.append("no_signals_after_building")
            await self._emit_rejected_batch(batch, reason="no_signals_after_building")
            return batch

        coordination = self.portfolio.coordinate(
            signals=built_signals,
            context=context,
        )
        batch.coordinated = coordination
        batch.final_signals = coordination.final_signals
        batch.accepted = coordination.accepted and bool(coordination.final_signals)

        if not batch.accepted:
            batch.reasons.extend(coordination.reasons)
            await self._emit_rejected_batch(batch, reason="portfolio_coordination_rejected")
            return batch

        for signal in batch.final_signals:
            self.state.update_signal(signal, active=True)
            await self.emit_event(
                "signal.generated",
                {
                    "symbol": signal.symbol,
                    "strategy_name": signal.strategy_name,
                    "side": signal.side.value,
                    "confidence": signal.confidence,
                    "score": signal.score,
                    "timestamp": signal.timestamp.isoformat(),
                    "signal": signal,
                },
                priority=EventPriority.HIGH,
                source=self.component_name,
            )

        return batch

    async def evaluate_strategies(
        self,
        *,
        strategies: list[BaseStrategy],
        context: StrategyContext,
    ) -> list[StrategyEvaluation]:
        evaluations: list[StrategyEvaluation] = []

        for strategy in strategies:
            try:
                evaluation = await strategy.evaluate(context)
                evaluations.append(evaluation)
                self.state.update_evaluation(evaluation)
            except Exception as exc:
                self.state.metrics.record_error(strategy_name=strategy.strategy_name)
                self.log_warning(
                    "Strategy evaluation failed",
                    strategy_name=strategy.strategy_name,
                    symbol=context.symbol,
                    error=str(exc),
                )

        return evaluations

    def _resolve_context(self, normalized: NormalizedPayload) -> StrategyContext:
        existing = self.state.contexts.get_context(normalized.symbol)
        if existing is not None:
            existing.timestamp = normalized.timestamp
            return existing

        return self.state.build_context(
            normalized.symbol,
            timestamp=normalized.timestamp,
        )

    async def _emit_rejected_batch(
        self,
        batch: ProcessedSignalBatch,
        *,
        reason: str,
    ) -> None:
        await self.emit_event(
            "signal.rejected",
            {
                "symbol": batch.symbol,
                "reason": reason,
                "reasons": batch.reasons,
                "timestamp": batch.timestamp.isoformat(),
            },
            priority=EventPriority.NORMAL,
            source=self.component_name,
        )