from __future__ import annotations
import logging

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
    _logger = logging.getLogger(__name__ + ".NormalizedPayload")
    source: FeatureSource
    symbol: str
    timestamp: datetime
    timeframe: Timeframe = Timeframe.M1
    domain_data: dict[str, Any] = field(default_factory=dict)
    extra_domain_data: dict[FeatureSource, dict[str, Any]] = field(default_factory=dict)
    features: list[FeatureSnapshot] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering NormalizedPayload.__post_init__")
        self.timestamp = ensure_aware_utc(self.timestamp)


@dataclass(slots=True)
class RouteDecision:
    _logger = logging.getLogger(__name__ + ".RouteDecision")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering RouteDecision.__post_init__")
        self.timestamp = ensure_aware_utc(self.timestamp)

    @property
    def selected_names(self) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering RouteDecision.selected_names")
        return [strategy.strategy_name for strategy in self.selected]

    @property
    def total_selected(self) -> int:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering RouteDecision.total_selected")
        return len(self.selected)

    @property
    def is_empty(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering RouteDecision.is_empty")
        return not self.selected


@dataclass(slots=True)
class WeightedSignal:
    _logger = logging.getLogger(__name__ + ".WeightedSignal")
    signal: StrategySignal
    category_weight: float
    regime_weight: float
    strategy_weight: float
    final_weight: float
    weighted_score: float
    weighted_confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WeightedSignal.validate")
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
    _logger = logging.getLogger(__name__ + ".VoteSummary")
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
    _logger = logging.getLogger(__name__ + ".ConflictSummary")
    accepted: bool = True
    total_penalty: float = 0.0
    conflicts: list[ConflictRecord] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def add_conflict(self, conflict: ConflictRecord) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConflictSummary.add_conflict")
        conflict.validate()
        self.conflicts.append(conflict)
        self.total_penalty += conflict.penalty


@dataclass(slots=True)
class ConfluenceEvaluation:
    _logger = logging.getLogger(__name__ + ".ConfluenceEvaluation")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceEvaluation.__post_init__")
        self.timestamp = ensure_aware_utc(self.timestamp)

    @property
    def accepted(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceEvaluation.accepted")
        return self.result is not None and self.result.accepted

    @property
    def selected_strategy_names(self) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceEvaluation.selected_strategy_names")
        return [signal.strategy_name for signal in self.accepted_signals]


@dataclass(slots=True)
class FilterEvaluation:
    _logger = logging.getLogger(__name__ + ".FilterEvaluation")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering FilterEvaluation.__post_init__")
        self.timestamp = ensure_aware_utc(self.timestamp)

    def add_result(self, result: FilterResult) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering FilterEvaluation.add_result")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering FilterEvaluation.has_warnings")
        return bool(self.warning_filters)

    @property
    def has_blocks(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering FilterEvaluation.has_blocks")
        return bool(self.blocking_filters)


@dataclass(slots=True)
class BuildEvaluation:
    _logger = logging.getLogger(__name__ + ".BuildEvaluation")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering BuildEvaluation.reject")
        self.accepted = False
        if reason not in self.reasons:
            self.reasons.append(reason)


@dataclass(slots=True)
class CoordinationDecision:
    _logger = logging.getLogger(__name__ + ".CoordinationDecision")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CoordinationDecision.__post_init__")
        self.timestamp = ensure_aware_utc(self.timestamp)

    @property
    def final_signals(self) -> list[StrategySignal]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CoordinationDecision.final_signals")
        return self.merged_signals if self.merged_signals else self.accepted_signals

    @property
    def selected_names(self) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CoordinationDecision.selected_names")
        return [signal.strategy_name for signal in self.final_signals]


@dataclass(slots=True)
class ProcessedSignalBatch:
    _logger = logging.getLogger(__name__ + ".ProcessedSignalBatch")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ProcessedSignalBatch.__post_init__")
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
    _logger = logging.getLogger(__name__ + ".SignalNormalizer")

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


    # ------------------------------------------------------------------
    # Topic-aware contract helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_mapping_or_none(value: Any) -> dict[str, Any] | None:
        _strategy_logger = logging.getLogger(__name__ + ".SignalNormalizer._as_mapping_or_none")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._as_mapping_or_none")
        if isinstance(value, dict):
            return value

        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            converted = to_dict()
            if isinstance(converted, dict):
                return converted

        return None

    @staticmethod
    def _topic_from_payload(payload: dict[str, Any]) -> str:
        _strategy_logger = logging.getLogger(__name__ + ".SignalNormalizer._topic_from_payload")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._topic_from_payload")
        return (
            str(
                payload.get("event_name")
                or payload.get("topic")
                or payload.get("source_topic")
                or payload.get("event_type")
                or ""
            )
            .strip()
            .lower()
        )

    def _direct_payload_value(
            self,
            payload: dict[str, Any],
            *,
            feature_map: dict[str, Any] | None = None,
    ) -> Any | None:
        """
        Return the real direct-event object from analytics payloads.

        Many analytics events use the same shape regardless of domain:
        {"state": ...}, {"snapshot": ...}, {"result": ...}, {"event": ...},
        {"signal": ...}.  The old adapters often treated payload["state"] as a
        composite object, which broke direct module events such as
        analytics.price_action.market_structure.updated.  This helper lets the
        topic decide the canonical section while preserving typed objects.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._direct_payload_value")
        feature_map = feature_map if isinstance(feature_map, dict) else {}
        for key in (
                "state",
                "snapshot",
                "result",
                "event",
                "setup",
                "signal",
                "data",
                "payload",
        ):
            if payload.get(key) is not None:
                return payload[key]
            if feature_map.get(key) is not None:
                return feature_map[key]
        return None


    # ------------------------------------------------------------------
    # Strategy contract safety-net adapters
    # ------------------------------------------------------------------

    def _analytics_effective_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Return a flattened analytics payload view for contract adapters.

        Some analytics services publish direct payloads, while others wrap the
        real payload under payload/data/result.  Strategy contracts should not
        depend on this transport envelope.  Top-level metadata stays
        authoritative, while nested analytics sections are made visible to the
        domain adapters.
        """
        result: dict[str, Any] = {}

        def merge_mapping(value: Any) -> None:
            mapping = self._as_mapping_or_none(value)
            if isinstance(mapping, dict):
                result.update(mapping)

        for key in ("payload", "data", "result", "analysis"):
            merge_mapping(payload.get(key))

        # Top-level fields override envelope fields such as event_name/source.
        result.update(payload)
        return result

    @staticmethod
    def _contract_get_path(value: Any, path: str, default: Any = None) -> Any:
        if value is None or not isinstance(path, str) or not path.strip():
            return default


        if isinstance(value, dict) and path in value:
            current = value.get(path)
            return default if current is None else current

        current = value
        for part in path.split("."):
            if current is None:
                return default
            part = part.strip()
            if not part:
                return default
            if isinstance(current, dict):
                if part in current:
                    current = current.get(part)
                else:
                    return default
            else:
                current = getattr(current, part, None)
        return default if current is None else current

    @classmethod
    def _contract_set_path(cls, target: dict[str, Any], path: str, value: Any) -> None:
        if value is None or not isinstance(path, str) or not path.strip():
            return
        parts = [part for part in path.split(".") if part]
        if not parts:
            return
        current = target
        for part in parts[:-1]:
            item = current.get(part)
            if not isinstance(item, dict):
                item = {}
                current[part] = item
            current = item
        current.setdefault(parts[-1], value)

    def _contract_first_value(
            self,
            *paths: str,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
            default: Any = None,
    ) -> Any:
        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        containers: tuple[Any, ...] = (
            domain_data,
            payload,
            feature_map,
            payload.get("context"),
            payload.get("stats"),
            payload.get("signal"),
            payload.get("snapshot"),
            payload.get("state"),
            payload.get("event"),
            payload.get("setup"),
        )
        for path in paths:
            for container in containers:
                value = self._contract_get_path(container, path, None)
                if value is not None:
                    return value
        return default

    def _contract_first_mapping(
            self,
            *paths: str,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        for path in paths:
            value = self._contract_first_value(
                path,
                payload=payload,
                domain_data=domain_data,
                default=None,
            )
            mapping = self._as_mapping_or_none(value)
            if isinstance(mapping, dict) and mapping:
                return dict(mapping)
        return None

    @staticmethod
    def _contract_section_present(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, dict):
            return bool(value)
        return True

    @staticmethod
    def _contract_side(value: Any) -> str | None:
        if value is None:
            return None
        text = str(getattr(value, "value", value)).strip().lower()
        if not text:
            return None
        if text in {"buy", "long", "bull", "bullish", "up", "bid"}:
            return "buy"
        if text in {"sell", "short", "bear", "bearish", "down", "ask"}:
            return "sell"
        return text

    def _ensure_strategy_domain_contracts(
            self,
            *,
            source: FeatureSource,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        """
        Final analytics -> strategy contract adapter.

        Domain-specific adapters above preserve rich analytics structures.  This
        safety net closes transport/shape gaps that commonly made strategies
        appear silent: wrapped payloads, CVD delta-only events, funding events
        under payload.*, opportunity-like spread signals, single whale trade
        events, and detector-style spoofing events.
        """
        effective_payload = self._analytics_effective_payload(payload)

        if source is FeatureSource.ORDERFLOW:
            self._ensure_orderflow_strategy_contract(
                payload=effective_payload,
                domain_data=domain_data,
            )
        elif source is FeatureSource.FUNDING:
            self._ensure_funding_strategy_contract(
                payload=effective_payload,
                domain_data=domain_data,
            )
        elif source is FeatureSource.SPREADS:
            self._ensure_spreads_strategy_contract(
                payload=effective_payload,
                domain_data=domain_data,
            )
        elif source is FeatureSource.WHALES:
            self._ensure_whales_strategy_contract(
                payload=effective_payload,
                domain_data=domain_data,
            )
        elif source is FeatureSource.SPOOFING:
            self._ensure_spoofing_strategy_contract(
                payload=effective_payload,
                domain_data=domain_data,
            )
        elif source is FeatureSource.LIQUIDATIONS:
            self._ensure_liquidations_strategy_contract(
                payload=effective_payload,
                domain_data=domain_data,
            )
        elif source is FeatureSource.LIQUIDITY:
            self._ensure_liquidity_strategy_contract(
                payload=effective_payload,
                domain_data=domain_data,
            )
        elif source is FeatureSource.OPEN_INTEREST:
            self._ensure_open_interest_strategy_contract(
                payload=effective_payload,
                domain_data=domain_data,
            )
        elif source is FeatureSource.PRICE_ACTION:
            self._ensure_price_action_strategy_contract(
                payload=effective_payload,
                domain_data=domain_data,
            )

    def _ensure_orderflow_strategy_contract(
            self,
            *,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        """
        Final ORDERFLOW contract adapter used by StrategyContext.

        analytics.orderflow analyzers publish metric-specific values mostly under
        payload["stats"].  This method lifts those flat stats into the canonical
        sections that concrete orderflow strategies read via
        FeatureSource.ORDERFLOW:

            composite, cvd, volume_delta, aggressive_trades,
            orderbook_imbalance, signal
        """
        get = lambda *paths, default=None: self._contract_first_value(
            *paths,
            payload=payload,
            domain_data=domain_data,
            default=default,
        )

        def mapping(*paths: str) -> dict[str, Any]:
            return self._contract_first_mapping(*paths, payload=payload, domain_data=domain_data) or {}

        def put(target: dict[str, Any], key: str, value: Any) -> None:
            if value is not None:
                target.setdefault(key, value)

        cvd = mapping(
            "cvd",
            "cvd_snapshot",
            "cvd_metrics",
            "cumulative_delta",
            "cumulative_volume_delta",
            "stats.cvd",
            "context.cvd",
            "context.stats.cvd",
            "orderflow.cvd",
        )
        volume_delta = mapping(
            "volume_delta",
            "volume_delta_snapshot",
            "delta",
            "delta_metrics",
            "stats.volume_delta_section",
            "context.volume_delta",
            "context.stats.volume_delta_section",
            "orderflow.volume_delta",
        )
        aggressive = mapping(
            "aggressive_trades",
            "aggressive",
            "aggression",
            "aggressive_flow",
            "stats.aggressive_trades",
            "context.aggressive_trades",
            "context.stats.aggressive_trades",
            "orderflow.aggressive_trades",
        )
        orderbook = mapping(
            "orderbook_imbalance",
            "orderbook",
            "book_imbalance",
            "imbalance",
            "stats.orderbook_imbalance",
            "context.orderbook_imbalance",
            "context.stats.orderbook_imbalance",
            "orderflow.orderbook_imbalance",
        )
        composite = mapping("composite", "snapshot", "orderflow_snapshot", "composite_snapshot", "orderflow.composite")

        cvd_value = get(
            "orderflow.cvd.value",
            "cvd.value",
            "cvd.cvd_value",
            "cvd_value",
            "stats.cvd_value",
            "context.stats.cvd_value",
            "cvd_close",
            "stats.cvd_close",
            "cumulative_volume_delta",
        )
        delta_value = get(
            "orderflow.volume_delta.volume_delta",
            "volume_delta.volume_delta",
            "volume_delta",
            "delta",
            "stats.volume_delta",
            "context.stats.volume_delta",
        )
        delta_ratio = get(
            "orderflow.cvd.delta_ratio",
            "orderflow.volume_delta.delta_ratio",
            "cvd.delta_ratio",
            "volume_delta.delta_ratio",
            "delta_ratio",
            "volume_delta_ratio",
            "cvd_delta_ratio",
            "stats.delta_ratio",
            "context.stats.delta_ratio",
        )

        buy_volume = _to_float(get("buy_volume", "stats.buy_volume", "context.stats.buy_volume", "aggressive_buy_volume", "stats.aggressive_buy_volume"), None)
        sell_volume = _to_float(get("sell_volume", "stats.sell_volume", "context.stats.sell_volume", "aggressive_sell_volume", "stats.aggressive_sell_volume"), None)
        buy_notional = _to_float(get("buy_notional", "stats.buy_notional", "context.stats.buy_notional", "aggressive_buy_notional", "stats.aggressive_buy_notional"), None)
        sell_notional = _to_float(get("sell_notional", "stats.sell_notional", "context.stats.sell_notional", "aggressive_sell_notional", "stats.aggressive_sell_notional"), None)
        total_volume = _to_float(get("total_volume", "volume", "stats.total_volume", "context.stats.total_volume"), None)
        if total_volume is None and buy_volume is not None and sell_volume is not None:
            total_volume = buy_volume + sell_volume
        total_notional = _to_float(get("total_notional", "notional", "quote_volume", "stats.total_notional", "context.stats.total_notional"), None)
        if total_notional is None and buy_notional is not None and sell_notional is not None:
            total_notional = buy_notional + sell_notional

        buy_ratio = _to_float(get(
            "orderflow.aggressive_trades.buy_ratio",
            "aggressive_trades.buy_ratio",
            "buy_ratio",
            "cvd_buy_ratio",
            "aggressive_buy_ratio",
            "stats.buy_ratio",
            "stats.aggressive_buy_ratio",
            "context.stats.buy_ratio",
        ), None)
        sell_ratio = _to_float(get(
            "orderflow.aggressive_trades.sell_ratio",
            "aggressive_trades.sell_ratio",
            "sell_ratio",
            "cvd_sell_ratio",
            "aggressive_sell_ratio",
            "stats.sell_ratio",
            "stats.aggressive_sell_ratio",
            "context.stats.sell_ratio",
        ), None)
        if total_volume and total_volume > 0:
            if buy_ratio is None and buy_volume is not None:
                buy_ratio = buy_volume / total_volume
            if sell_ratio is None and sell_volume is not None:
                sell_ratio = sell_volume / total_volume
        if buy_ratio is None and sell_ratio is not None:
            buy_ratio = max(0.0, 1.0 - sell_ratio)
        if sell_ratio is None and buy_ratio is not None:
            sell_ratio = max(0.0, 1.0 - buy_ratio)

        for key, value in {
            "value": cvd_value,
            "cvd_value": cvd_value,
            "cvd_open": get("cvd.cvd_open", "cvd_open", "stats.cvd_open"),
            "cvd_high": get("cvd.cvd_high", "cvd_high", "stats.cvd_high"),
            "cvd_low": get("cvd.cvd_low", "cvd_low", "stats.cvd_low"),
            "cvd_close": get("cvd.cvd_close", "cvd_close", "stats.cvd_close"),
            "cvd_change": get("cvd.cvd_change", "cvd_change", "stats.cvd_change"),
            "cvd_change_pct": get("orderflow.cvd.cvd_change_pct", "cvd.cvd_change_pct", "cvd_change_pct", "change_pct", "stats.cvd_change_pct"),
            "cvd_slope": get("orderflow.cvd.cvd_slope", "cvd.cvd_slope", "cvd_slope", "slope", "stats.cvd_slope"),
            "delta_ratio": delta_ratio,
            "price_change_pct": get("orderflow.cvd.price_change_pct", "cvd.price_change_pct", "price_change_pct", "price_delta_pct", "stats.price_change_pct"),
            "buy_ratio": buy_ratio,
            "sell_ratio": sell_ratio,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "buy_notional": buy_notional,
            "sell_notional": sell_notional,
            "trades_count": get("trades_count", "trades", "trade_count", "stats.trades_count"),
            "total_volume": total_volume,
            "total_notional": total_notional,
            "last_price": get("last_price", "price", "close", "mark_price", "stats.last_price"),
            "window_seconds": get("window_seconds", "stats.window_seconds"),
        }.items():
            put(cvd, key, value)

        for key, value in {
            "volume_delta": delta_value,
            "delta_ratio": delta_ratio,
            "cumulative_volume_delta": get("volume_delta.cumulative_volume_delta", "cumulative_volume_delta", "stats.cumulative_volume_delta", "cvd_value", "stats.cvd_value"),
            "notional_delta": get("volume_delta.notional_delta", "notional_delta", "stats.notional_delta"),
            "cumulative_notional_delta": get("volume_delta.cumulative_notional_delta", "cumulative_notional_delta", "stats.cumulative_notional_delta"),
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "buy_notional": buy_notional,
            "sell_notional": sell_notional,
            "buy_ratio": buy_ratio,
            "sell_ratio": sell_ratio,
            "trades_count": get("trades_count", "trades", "trade_count", "stats.trades_count"),
            "total_volume": total_volume,
            "total_notional": total_notional,
        }.items():
            put(volume_delta, key, value)

        for key, value in {
            "buy_ratio": buy_ratio,
            "sell_ratio": sell_ratio,
            "net_volume_delta": get("orderflow.aggressive_trades.net_volume_delta", "aggressive_trades.net_volume_delta", "net_volume_delta", "stats.net_volume_delta", "volume_delta", "stats.volume_delta", "delta"),
            "net_notional_delta": get("orderflow.aggressive_trades.net_notional_delta", "aggressive_trades.net_notional_delta", "net_notional_delta", "stats.net_notional_delta", "notional_delta", "stats.notional_delta"),
            "burst_score": get("aggressive_trades.burst_score", "burst_score", "aggressive_burst_score", "aggression_score", "stats.burst_score"),
            "large_buy_trades": get("aggressive_trades.large_buy_trades", "large_buy_trades", "stats.large_buy_trades"),
            "large_sell_trades": get("aggressive_trades.large_sell_trades", "large_sell_trades", "stats.large_sell_trades"),
            "aggressive_buy_count": get("aggressive_buy_count", "stats.aggressive_buy_count"),
            "aggressive_sell_count": get("aggressive_sell_count", "stats.aggressive_sell_count"),
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "buy_notional": buy_notional,
            "sell_notional": sell_notional,
            "trades_count": get("trades_count", "trades", "trade_count", "stats.trades_count"),
        }.items():
            put(aggressive, key, value)

        for key, value in {
            "ratio": get("orderbook_imbalance.ratio", "orderbook_imbalance.imbalance_ratio", "imbalance_ratio", "orderbook_imbalance_ratio", "stats.imbalance_ratio"),
            "diff": get("orderbook_imbalance.diff", "orderbook_imbalance.imbalance_diff", "imbalance_diff", "orderbook_imbalance_diff", "stats.imbalance_diff"),
            "imbalance_ratio": get("orderbook_imbalance.imbalance_ratio", "imbalance_ratio", "orderbook_imbalance_ratio", "stats.imbalance_ratio"),
            "imbalance_diff": get("orderbook_imbalance.imbalance_diff", "imbalance_diff", "orderbook_imbalance_diff", "stats.imbalance_diff"),
            "bid_volume": get("bid_volume", "stats.bid_volume"),
            "ask_volume": get("ask_volume", "stats.ask_volume"),
            "best_bid": get("best_bid", "stats.best_bid"),
            "best_ask": get("best_ask", "stats.best_ask"),
            "spread": get("spread", "stats.spread"),
            "mid_price": get("mid_price", "stats.mid_price"),
            "depth_levels_used": get("depth_levels_used", "stats.depth_levels_used"),
        }.items():
            put(orderbook, key, value)

        for key, section in (
                ("cvd", cvd),
                ("volume_delta", volume_delta),
                ("aggressive_trades", aggressive),
                ("orderbook_imbalance", orderbook),
        ):
            if section:
                domain_data[key] = section
                composite.setdefault(key, section)

        # Stable aliases expected by strategy/strategies/orderflow/utils.py.
        if cvd:
            domain_data.setdefault("cvd_metrics", cvd)
            domain_data.setdefault("cvd_snapshot", cvd)
            domain_data.setdefault("cumulative_delta", cvd)
        if volume_delta:
            domain_data.setdefault("delta", volume_delta)
            domain_data.setdefault("delta_metrics", volume_delta)
            domain_data.setdefault("volume_delta_snapshot", volume_delta)
        if aggressive:
            domain_data.setdefault("aggressive", aggressive)
            domain_data.setdefault("aggressive_flow", aggressive)
            domain_data.setdefault("aggressive_trades_snapshot", aggressive)
        if orderbook:
            domain_data.setdefault("orderbook", orderbook)
            domain_data.setdefault("imbalance", orderbook)
            domain_data.setdefault("orderbook_snapshot", orderbook)

        signal = mapping("signal", "setup", "orderflow_signal", "analytics_signal")
        side = self._contract_side(get("side", "direction", "bias", "signal.side", "setup.side"))
        signal_type = get("signal_type", "setup_type", "type", "signal.type", "setup.type")
        score = get("score", "signal_score", "signal.score", "setup.score", "strength")
        confidence = get("confidence", "signal_confidence", "signal.confidence", "setup.confidence", "strength")
        if signal or side is not None or signal_type is not None:
            signal.setdefault("detected", True)
            signal.setdefault("type", signal_type or "orderflow_signal")
            if side is not None:
                signal.setdefault("side", side)
            if score is not None:
                signal.setdefault("score", score)
            if confidence is not None:
                signal.setdefault("confidence", confidence)
            signal.setdefault("origin", "orderflow")
            domain_data.setdefault("signal", signal)
            domain_data.setdefault("setup", signal)
            domain_data.setdefault("orderflow_signal", signal)
            domain_data.setdefault("analytics_signal", signal)

        for key, value in {
            "exchange": get("exchange", "stats.exchange", "scope.exchange"),
            "market_type": get("market_type", "stats.market_type", "scope.market_type"),
            "symbol": get("symbol", "stats.symbol", "scope.symbol"),
            "exchange_symbol": get("exchange_symbol", "stats.exchange_symbol", "scope.exchange_symbol"),
            "timeframe": get("timeframe", "stats.timeframe", "scope.timeframe"),
            "timestamp": get("timestamp", "stats.timestamp", "event_time"),
            "last_price": get("last_price", "price", "close", "mark_price", "mid_price", "stats.last_price", "stats.mid_price"),
            "price": get("price", "last_price", "close", "mark_price", "mid_price", "stats.last_price", "stats.mid_price"),
            "price_change": get("price_change", "stats.price_change"),
            "price_change_pct": get("price_change_pct", "price_delta_pct", "stats.price_change_pct"),
            "window_seconds": get("window_seconds", "stats.window_seconds"),
            "trades_count": get("trades_count", "trades", "trade_count", "stats.trades_count"),
            "total_volume": total_volume,
            "total_notional": total_notional,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "buy_notional": buy_notional,
            "sell_notional": sell_notional,
            "delta_ratio": delta_ratio,
            "side": side,
            "score": score,
            "confidence": confidence,
            "metric": get("metric", "stats.metric"),
            "source_type": get("source_type", "stats.source_type"),
        }.items():
            if value is not None:
                domain_data.setdefault(key, value)
                composite.setdefault(key, value)

        if composite:
            domain_data["composite"] = composite
            domain_data.setdefault("snapshot", composite)
            domain_data.setdefault("orderflow_snapshot", composite)
            domain_data.setdefault("composite_snapshot", composite)

        # Dotted aliases make context.domain_dict(ORDERFLOW) usable even when
        # helper code asks for full contract paths directly.
        for path, value in {
            "orderflow.composite": composite or None,
            "orderflow.cvd": cvd or None,
            "orderflow.cvd.value": cvd.get("value"),
            "orderflow.cvd.delta_ratio": cvd.get("delta_ratio"),
            "orderflow.cvd.cvd_change_pct": cvd.get("cvd_change_pct"),
            "orderflow.cvd.cvd_slope": cvd.get("cvd_slope"),
            "orderflow.cvd.price_change_pct": cvd.get("price_change_pct"),
            "orderflow.volume_delta": volume_delta or None,
            "orderflow.volume_delta.volume_delta": volume_delta.get("volume_delta"),
            "orderflow.volume_delta.delta_ratio": volume_delta.get("delta_ratio"),
            "orderflow.volume_delta.cumulative_volume_delta": volume_delta.get("cumulative_volume_delta"),
            "orderflow.volume_delta.notional_delta": volume_delta.get("notional_delta"),
            "orderflow.volume_delta.cumulative_notional_delta": volume_delta.get("cumulative_notional_delta"),
            "orderflow.aggressive_trades": aggressive or None,
            "orderflow.aggressive_trades.buy_ratio": aggressive.get("buy_ratio"),
            "orderflow.aggressive_trades.sell_ratio": aggressive.get("sell_ratio"),
            "orderflow.aggressive_trades.burst_score": aggressive.get("burst_score"),
            "orderflow.aggressive_trades.net_volume_delta": aggressive.get("net_volume_delta"),
            "orderflow.aggressive_trades.net_notional_delta": aggressive.get("net_notional_delta"),
            "orderflow.aggressive_trades.large_buy_trades": aggressive.get("large_buy_trades"),
            "orderflow.aggressive_trades.large_sell_trades": aggressive.get("large_sell_trades"),
            "orderflow.orderbook_imbalance": orderbook or None,
            "orderflow.orderbook_imbalance.ratio": orderbook.get("ratio") or orderbook.get("imbalance_ratio"),
            "orderflow.orderbook_imbalance.diff": orderbook.get("diff") or orderbook.get("imbalance_diff"),
            "orderflow.trades_count": domain_data.get("trades_count"),
            "orderflow.total_volume": domain_data.get("total_volume"),
            "orderflow.total_notional": domain_data.get("total_notional"),
            "orderflow.last_price": domain_data.get("last_price"),
            "orderflow.price_change_pct": domain_data.get("price_change_pct"),
            "orderflow.signal": signal or None,
            "orderflow.signal.side": signal.get("side") if signal else None,
            "orderflow.signal.score": signal.get("score") if signal else None,
            "orderflow.signal.confidence": signal.get("confidence") if signal else None,
        }.items():
            if value is not None:
                domain_data.setdefault(path, value)

    def _ensure_funding_strategy_contract(
            self,
            *,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        def mapping(*paths: str) -> dict[str, Any] | None:
            return self._contract_first_mapping(*paths, payload=payload, domain_data=domain_data)

        def get(*paths: str, default: Any = None) -> Any:
            return self._contract_first_value(*paths, payload=payload, domain_data=domain_data, default=default)

        def section_get(section: dict[str, Any] | None, *paths: str, default: Any = None) -> Any:
            if not isinstance(section, dict):
                return default
            for path in paths:
                if path in section and section[path] is not None:
                    return section[path]
                cur: Any = section
                for part in path.split("."):
                    if not isinstance(cur, dict) or part not in cur:
                        cur = None
                        break
                    cur = cur.get(part)
                if cur is not None:
                    return cur
            return default

        def truthy(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y", "on", "active", "detected", "confirmed"}
            if isinstance(value, (int, float)):
                return bool(value)
            return False

        def put_alias(target: str, value: dict[str, Any], *aliases: str) -> None:
            domain_data[target] = value
            for alias in aliases:
                domain_data.setdefault(alias, value)

        def normalize(name: str, raw: dict[str, Any] | None) -> dict[str, Any] | None:
            if not isinstance(raw, dict) or not raw:
                return None
            section = dict(raw)
            for _runtime_key in ("feature_map", "features", "strategy_contract", "strategy_contract_version"):
                section.pop(_runtime_key, None)

            def fill(canonical: str, *paths: str, default: Any = None) -> None:
                value = section_get(section, *paths, default=None)
                if value is None:
                    for path in paths:
                        value = get(f"funding.{name}.{path}", f"{name}_{path}", path, default=None)
                        if value is not None:
                            break
                if value is None:
                    value = default
                if value is not None:
                    section.setdefault(canonical, value)

            if name == "snapshot":
                fill("funding_rate", "funding_rate", "current_rate", "rate")
                fill("current_rate", "current_rate", "funding_rate", "rate")
                fill("predicted_funding_rate", "predicted_funding_rate", "predicted_rate", "next_funding_rate")
                fill("predicted_rate", "predicted_rate", "predicted_funding_rate", "next_funding_rate")
                fill("mark_price", "mark_price", "current_price", "reference_price")
                fill("event_time", "event_time", "timestamp", "updated_at", "received_at")
            elif name == "statistics":
                fill("current_rate", "current_rate", "funding_rate", "rate")
                fill("mean_rate", "mean_rate", "mean")
                fill("median_rate", "median_rate", "median")
                fill("std_rate", "std_rate", "std", "stdev")
                fill("zscore", "zscore", "z_score")
                fill("sample_size", "sample_size", "samples", "count")
            elif name == "regime":
                fill("type", "type", "regime", "name", "state")
                fill("regime", "regime", "type", "name")
                fill("bias", "bias", "direction", "side")
                fill("score", "score", "confidence")
                fill("confidence", "confidence", "score")
                fill("event_time", "event_time", "timestamp", "updated_at")
            elif name == "pressure":
                fill("score", "score", "pressure_score", "strength", "normalized_score")
                fill("pressure_score", "pressure_score", "score", "strength", "normalized_score")
                fill("level", "level", "pressure_level", "type")
                fill("direction", "direction", "pressure_direction", "bias", "side")
                fill("bias", "bias", "direction", "pressure_direction")
                fill("mean_reversion_probability", "mean_reversion_probability", "reversion_probability", "reversal_probability")
                fill("squeeze_probability", "squeeze_probability", "squeeze_risk")
                fill("event_time", "event_time", "timestamp", "updated_at")
            elif name == "extreme":
                fill("type", "type", "extreme_type", "kind")
                fill("extreme_type", "extreme_type", "type", "kind")
                fill("score", "score", "severity", "strength", "normalized_score")
                fill("severity", "severity", "score", "strength", "normalized_score")
                fill("confidence", "confidence", "severity", "mean_reversion_probability")
                fill("funding_rate", "funding_rate", "current_rate", "rate")
                fill("reversal_risk", "reversal_risk", "is_reversal_risk", "has_reversal_risk", "mean_reversion_risk")
                fill("squeeze_risk", "squeeze_risk", "is_squeeze_risk", "has_squeeze_risk")
                fill("mean_reversion_probability", "mean_reversion_probability", "reversion_probability", "reversal_probability")
                fill("squeeze_probability", "squeeze_probability", "short_squeeze_probability", "long_squeeze_probability", "squeeze_risk")
                fill("event_time", "event_time", "timestamp", "updated_at")
                severity = section_get(section, "severity", "score", default=0.0)
                if section_get(section, "is_reversal_risk") is not None:
                    section.setdefault("reversal_risk", section_get(section, "is_reversal_risk"))
                if section_get(section, "is_squeeze_risk") is not None:
                    section.setdefault("squeeze_risk", section_get(section, "is_squeeze_risk"))
                if truthy(section.get("reversal_risk")) and section.get("mean_reversion_probability") is None:
                    section["mean_reversion_probability"] = severity
                if truthy(section.get("squeeze_risk")) and section.get("squeeze_probability") is None:
                    section["squeeze_probability"] = severity
                section.setdefault("detected", True)
            elif name == "divergence":
                fill("type", "type", "divergence_type", "kind")
                fill("divergence_type", "divergence_type", "type", "kind")
                fill("score", "score", "confidence", "strength", "signed_score")
                fill("confidence", "confidence", "score")
                fill("bias", "bias", "direction", "side", "expected_side")
                fill("side", "side", "signal_side", "expected_side", "target_side", "bias", "direction")
                fill("event_time", "event_time", "timestamp", "updated_at")
                section.setdefault("detected", True)
            elif name == "flip":
                fill("type", "type", "flip_type", "kind")
                fill("flip_type", "flip_type", "type", "kind")
                fill("score", "score", "confidence", "flip_magnitude")
                fill("confidence", "confidence", "score")
                fill("magnitude", "magnitude", "flip_magnitude")
                fill("event_time", "event_time", "timestamp", "updated_at")
                section.setdefault("detected", True)
            elif name == "signal":
                fill("type", "type", "signal_type", "setup_type")
                fill("signal_type", "signal_type", "type", "setup_type")
                fill("score", "score", "signed_score", "strength")
                fill("confidence", "confidence", "score_confidence")
                fill("bias", "bias", "direction", "side")
                fill("origin", "origin", "signal_origin")
                fill("event_time", "event_time", "timestamp", "updated_at")
                section.setdefault("detected", True)
            return section

        sections = {
            "snapshot": mapping("snapshot", "funding_snapshot"),
            "statistics": mapping("statistics", "stats", "funding_statistics"),
            "regime": mapping("regime", "regime_state", "funding_regime", "funding_regime_state"),
            "pressure": mapping("pressure", "pressure_state", "funding_pressure", "funding_pressure_state"),
            "extreme": mapping("extreme", "extreme_event", "funding_extreme", "funding_extreme_event"),
            "divergence": mapping("divergence", "divergence_event", "funding_divergence", "funding_divergence_event"),
            "flip": mapping("flip", "flip_event", "funding_flip", "funding_flip_event"),
            "signal": mapping("signal", "funding_signal", "analytics_signal", "setup", "strategy_signal"),
        }

        if sections["snapshot"] is None:
            flat_snapshot = {
                key: get(key, f"funding.{key}")
                for key in (
                    "funding_rate", "current_rate", "next_funding_rate", "predicted_rate",
                    "predicted_funding_rate", "annualized_rate", "premium_index",
                    "mark_price", "index_price", "open_interest", "volume_24h",
                    "next_funding_time", "exchange", "market_type", "symbol",
                    "exchange_symbol", "timeframe", "timestamp", "event_time",
                )
                if get(key, f"funding.{key}") is not None
            }
            if flat_snapshot:
                sections["snapshot"] = flat_snapshot

        if sections["statistics"] is None:
            flat_stats = {
                key: get(key, f"funding.statistics.{key}")
                for key in (
                    "current_rate", "mean_rate", "median_rate", "std_rate",
                    "zscore", "z_score", "percentile", "min_rate", "max_rate",
                    "sample_size", "samples", "window_start", "window_end", "updated_at",
                )
                if get(key, f"funding.statistics.{key}") is not None
            }
            if flat_stats:
                sections["statistics"] = flat_stats

        topic = str(get("event_name", "topic", "source_topic", default="")).lower()
        event_type = str(get("event_type", "type", default="")).lower()
        if sections["pressure"] is None and ("pressure" in topic or event_type == "pressure"):
            sections["pressure"] = dict(payload)
        if sections["regime"] is None and ("regime" in topic or event_type == "regime"):
            sections["regime"] = dict(payload)
        if sections["extreme"] is None and ("extreme" in topic or event_type == "extreme"):
            sections["extreme"] = dict(payload)
        if sections["divergence"] is None and ("divergence" in topic or event_type == "divergence"):
            sections["divergence"] = dict(payload)
        if sections["flip"] is None and ("flip" in topic or event_type == "flip"):
            sections["flip"] = dict(payload)
        if sections["signal"] is None and ("signal" in topic or event_type == "signal"):
            sections["signal"] = dict(payload)

        alias_map = {
            "snapshot": ("funding_snapshot",),
            "statistics": ("stats", "funding_statistics"),
            "regime": ("regime_state", "funding_regime", "funding_regime_state"),
            "pressure": ("pressure_state", "funding_pressure", "funding_pressure_state"),
            "extreme": ("extreme_event", "funding_extreme", "funding_extreme_event"),
            "divergence": ("divergence_event", "funding_divergence", "funding_divergence_event"),
            "flip": ("flip_event", "funding_flip", "funding_flip_event"),
            "signal": ("funding_signal", "analytics_signal", "setup"),
        }

        for name, section in list(sections.items()):
            normalized = normalize(name, section)
            if normalized:
                put_alias(name, normalized, *alias_map[name])

        feature_aliases = {
            "funding.snapshot": domain_data.get("snapshot"),
            "funding.statistics": domain_data.get("statistics"),
            "funding.regime": domain_data.get("regime"),
            "funding.pressure": domain_data.get("pressure"),
            "funding.extreme": domain_data.get("extreme"),
            "funding.divergence": domain_data.get("divergence"),
            "funding.flip": domain_data.get("flip"),
            "funding.signal": domain_data.get("signal"),
            "funding.regime.confidence": section_get(domain_data.get("regime"), "confidence", "score"),
            "funding.pressure.score": section_get(domain_data.get("pressure"), "score", "pressure_score"),
            "funding.pressure.level": section_get(domain_data.get("pressure"), "level", "pressure_level"),
            "funding.pressure.direction": section_get(domain_data.get("pressure"), "direction", "pressure_direction", "bias"),
            "funding.extreme.type": section_get(domain_data.get("extreme"), "type", "extreme_type"),
            "funding.extreme.severity": section_get(domain_data.get("extreme"), "severity", "score"),
            "funding.extreme.mean_reversion_probability": section_get(domain_data.get("extreme"), "mean_reversion_probability", "reversion_probability"),
            "funding.extreme.squeeze_probability": section_get(domain_data.get("extreme"), "squeeze_probability"),
            "funding.divergence.type": section_get(domain_data.get("divergence"), "type", "divergence_type"),
            "funding.divergence.confidence": section_get(domain_data.get("divergence"), "confidence", "score"),
            "funding.divergence.score": section_get(domain_data.get("divergence"), "score", "confidence", "signed_score"),
            "funding.flip.type": section_get(domain_data.get("flip"), "type", "flip_type"),
            "funding.flip.confidence": section_get(domain_data.get("flip"), "confidence", "score"),
            "funding.signal.type": section_get(domain_data.get("signal"), "type", "signal_type"),
            "funding.signal.score": section_get(domain_data.get("signal"), "score", "signed_score"),
            "funding.signal.confidence": section_get(domain_data.get("signal"), "confidence"),
            "funding.signal.bias": section_get(domain_data.get("signal"), "bias", "direction", "side"),
        }
        for path, value in feature_aliases.items():
            if value is not None:
                domain_data.setdefault(path, value)


    def _ensure_spreads_strategy_contract(
            self,
            *,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        get = lambda *paths, default=None: self._contract_first_value(*paths, payload=payload, domain_data=domain_data, default=default)
        mapping = lambda *paths: self._contract_first_mapping(*paths, payload=payload, domain_data=domain_data)

        snapshot = mapping("snapshot", "spread_snapshot", "basis", "spread")
        signal = mapping("signal", "spread_signal", "setup")
        opportunity = mapping("opportunity", "arb_opportunity", "cross_exchange_opportunity")

        flat = {
            key: get(key)
            for key in (
                "spread_type",
                "type",
                "symbol",
                "exchange_a",
                "exchange_b",
                "market_type_a",
                "market_type_b",
                "exchange_symbol_a",
                "exchange_symbol_b",
                "spread_bps",
                "basis",
                "funding_adjusted_spread",
                "net_edge",
                "net_edge_bps",
                "zscore",
                "z_score",
                "regime",
                "direction",
                "signal_type",
                "quote_validity",
                "has_edge",
                "confidence",
                "opportunity_key",
                "opportunity_status",
                "persistence_ms",
                "buy_exchange",
                "sell_exchange",
                "buy_market_type",
                "sell_market_type",
                "instrument_type",
            )
            if get(key) is not None
        }

        if snapshot is None and flat:
            snapshot = dict(flat)
        if signal is None and any(key in flat for key in ("signal_type", "direction", "has_edge", "confidence")):
            signal = dict(flat)
        if opportunity is None and any(key in flat for key in ("buy_exchange", "sell_exchange", "net_edge", "net_edge_bps", "has_edge")):
            opportunity = dict(flat)
            opportunity.setdefault("detected", bool(flat.get("has_edge", True)))

        for name, section in (("snapshot", snapshot), ("signal", signal), ("opportunity", opportunity)):
            if isinstance(section, dict) and section:
                domain_data[name] = section

        for key, value in flat.items():
            if value is not None:
                normalized_key = "zscore" if key == "z_score" else key
                domain_data.setdefault(normalized_key, value)

        for path, value in {
            "spreads.snapshot": domain_data.get("snapshot"),
            "spreads.signal": domain_data.get("signal"),
            "spreads.opportunity": domain_data.get("opportunity"),
        }.items():
            if value is not None:
                domain_data.setdefault(path, value)

    def _ensure_whales_strategy_contract(
            self,
            *,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        get = lambda *paths, default=None: self._contract_first_value(*paths, payload=payload, domain_data=domain_data, default=default)
        mapping = lambda *paths: self._contract_first_mapping(*paths, payload=payload, domain_data=domain_data)

        pressure = mapping("pressure", "whale_pressure", "whale_pressure_signal")
        activity = mapping("activity", "whale_activity", "whale_activity_signal")
        large_trade = mapping("large_trade", "large_trade_signal", "whale_large_trade")
        cluster = mapping("cluster", "whale_cluster", "whale_cluster_signal")
        cluster_update = mapping("cluster_update", "whale_cluster_update")
        cluster_exhaustion = mapping("cluster_exhaustion", "whale_cluster_exhaustion")
        liquidation_context = mapping("liquidation_context", "whale_liquidation_context")

        flat = {
            key: get(key)
            for key in (
                "dominant_side",
                "whale_side",
                "side",
                "liquidation_side",
                "exhausted_side",
                "cluster_side",
                "imbalance_ratio",
                "pressure_score",
                "context_strength",
                "cluster_score",
                "continuation_probability",
                "exhaustion_probability",
                "total_notional",
                "notional",
                "liquidation_notional",
                "trade_count",
                "large_trade_notional",
                "large_notional",
                "large_trade_zscore",
                "zscore",
                "reference_price",
                "price",
                "confidence",
                "timestamp",
            )
            if get(key) is not None
        }
        if "side" in flat and "dominant_side" not in flat:
            flat["dominant_side"] = self._contract_side(flat.get("side"))
        if "notional" in flat and "total_notional" not in flat:
            flat["total_notional"] = flat["notional"]
        if "large_notional" in flat and "large_trade_notional" not in flat:
            flat["large_trade_notional"] = flat["large_notional"]
        if "price" in flat and "reference_price" not in flat:
            flat["reference_price"] = flat["price"]

        topic = self._topic_from_payload(payload)
        is_large_trade_event = "large_trade" in topic
        if large_trade is None and any(key in flat for key in ("large_trade_notional", "large_trade_zscore", "total_notional")):
            large_trade = dict(flat)
        # Do not synthesize pressure/liquidation_context from a plain large-trade
        # event.  Those sections are distinct whale setups and routing should not
        # make absorption/liquidation strategies think their required contracts exist.
        if activity is None and (large_trade or cluster or cluster_update):
            activity = dict(flat)
            if large_trade:
                activity.setdefault("large_trade", large_trade)
        if (not is_large_trade_event) and pressure is None and any(key in flat for key in ("pressure_score", "imbalance_ratio")):
            pressure = dict(flat)
        if (not is_large_trade_event) and liquidation_context is None and any(key in flat for key in ("liquidation_side", "liquidation_notional", "exhaustion_probability")):
            liquidation_context = dict(flat)

        for name, section in (
            ("pressure", pressure),
            ("activity", activity),
            ("large_trade", large_trade),
            ("cluster", cluster),
            ("cluster_update", cluster_update),
            ("cluster_exhaustion", cluster_exhaustion),
            ("liquidation_context", liquidation_context),
        ):
            if self._contract_section_present(section):
                domain_data[name] = dict(section) if isinstance(section, dict) else section

        for key, value in flat.items():
            if value is not None:
                normalized_key = {
                    "side": "dominant_side",
                    "notional": "total_notional",
                    "large_notional": "large_trade_notional",
                    "zscore": "large_trade_zscore",
                    "price": "reference_price",
                }.get(key, key)
                domain_data.setdefault(normalized_key, value)

    def _ensure_spoofing_strategy_contract(
            self,
            *,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        get = lambda *paths, default=None: self._contract_first_value(*paths, payload=payload, domain_data=domain_data, default=default)
        mapping = lambda *paths: self._contract_first_mapping(*paths, payload=payload, domain_data=domain_data)

        signal = mapping("signal", "spoofing_signal", "setup")
        features = mapping("features", "spoofing_features") or {}
        detector_results = mapping("detector_results", "detectors", "results")

        for key in (
            "pull_ratio",
            "fill_ratio",
            "price_reaction_bps",
            "signed_price_reaction_bps",
            "lifetime_ms",
            "wall_notional",
            "pulled_notional",
            "cancel_to_fill_ratio",
            "distance_from_mid_bps",
            "layer_count",
            "layer_price_span_bps",
            "pressure_flip_strength",
        ):
            value = get(key, f"features.{key}")
            if value is not None:
                features.setdefault(key, value)

        signal_flat = {
            key: get(key)
            for key in (
                "type",
                "spoofing_type",
                "pattern",
                "side",
                "severity",
                "status",
                "score",
                "confidence",
                "price_level",
                "wall_id",
                "event_time",
            )
            if get(key) is not None
        }
        if signal is None and signal_flat:
            signal = dict(signal_flat)
            signal.setdefault("detected", True)

        if signal:
            domain_data["signal"] = signal
        if features:
            domain_data["features"] = features
        if detector_results:
            domain_data["detector_results"] = detector_results

    def _ensure_liquidations_strategy_contract(
            self,
            *,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        get = lambda *paths, default=None: self._contract_first_value(*paths, payload=payload, domain_data=domain_data, default=default)
        mapping = lambda *paths: self._contract_first_mapping(*paths, payload=payload, domain_data=domain_data)
        for name, aliases in {
            "cascade": ("cascade", "cascade_result", "liquidation_cascade"),
            "exhaustion": ("exhaustion", "exhaustion_result"),
            "squeeze": ("squeeze", "squeeze_result"),
            "cluster": ("cluster", "liquidation_cluster"),
        }.items():
            section = mapping(*aliases)
            if section is None:
                flat = {
                    key: get(key)
                    for key in (
                        "confidence", "intensity_score", "direction", "severity",
                        "continuation_bias", "exhaustion_bias", "total_notional_usd",
                        "event_count", "confirmed", "score", "duration_seconds",
                        "avg_notional_per_event", "side_imbalance_ratio",
                        "event_imbalance_ratio", "acceleration_ratio",
                    )
                    if get(key) is not None
                }
                if name == "cascade" and any(k in flat for k in ("intensity_score", "direction", "severity")):
                    section = flat
                elif name == "exhaustion" and any(k in flat for k in ("exhaustion_bias", "confirmed")):
                    section = flat
                elif name == "squeeze" and any(k in flat for k in ("score", "direction", "confirmed")):
                    section = flat
                elif name == "cluster" and any(k in flat for k in ("duration_seconds", "side_imbalance_ratio", "event_count")):
                    section = flat
            if section:
                domain_data[name] = section

    def _ensure_liquidity_strategy_contract(
            self,
            *,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        get = lambda *paths, default=None: self._contract_first_value(*paths, payload=payload, domain_data=domain_data, default=default)
        mapping = lambda *paths: self._contract_first_mapping(*paths, payload=payload, domain_data=domain_data)
        snapshot = mapping("snapshot", "map", "liquidity_map", "map_snapshot")
        active_levels = get("active_levels", "levels", "level")
        stop_clusters = get("stop_clusters", "clusters", "stop_cluster")
        if snapshot is None:
            flat = {
                key: get(key)
                for key in (
                    "current_price", "last_price", "reference_price", "price",
                    "above_liquidity_score", "below_liquidity_score",
                    "pressure_score", "liquidity_pressure_score", "bias",
                    "nearest_above_level", "nearest_below_level",
                    "strongest_cluster_above", "strongest_cluster_below",
                    "equal_levels", "zones",
                )
                if get(key) is not None
            }
            if active_levels is not None:
                flat.setdefault("active_levels", active_levels)
            if stop_clusters is not None:
                flat.setdefault("stop_clusters", stop_clusters)
            price = flat.get("current_price") or flat.get("last_price") or flat.get("reference_price") or flat.get("price")
            has_structure = any(flat.get(key) not in (None, [], {}) for key in (
                "active_levels", "stop_clusters", "equal_levels", "zones",
                "nearest_above_level", "nearest_below_level",
                "strongest_cluster_above", "strongest_cluster_below",
            ))
            if price is not None and has_structure:
                flat.setdefault("current_price", price)
                snapshot = flat
        if self._contract_section_present(snapshot):
            domain_data["snapshot"] = snapshot
            domain_data.setdefault("map", snapshot)
            domain_data.setdefault("map_snapshot", snapshot)
            domain_data.setdefault("liquidity_map", snapshot)
        elif "signal" in self._topic_from_payload(payload):
            # Signal-only liquidity events are useful for diagnostics/hybrid votes,
            # but they are not a full LiquidityMapSnapshot contract.
            domain_data.setdefault("payload_contract_level", "signal_only")
        if active_levels is not None:
            domain_data.setdefault("active_levels", active_levels)
        if stop_clusters is not None:
            domain_data.setdefault("stop_clusters", stop_clusters)

    def _ensure_open_interest_strategy_contract(
            self,
            *,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        get = lambda *paths, default=None: self._contract_first_value(*paths, payload=payload, domain_data=domain_data, default=default)
        for name, aliases in {
            "features": ("features", "oi_features", "open_interest_features"),
            "regime": ("regime", "regime_result", "oi_regime"),
            "divergence": ("divergence", "divergence_result", "oi_divergence"),
            "anomaly": ("anomaly", "anomaly_result", "oi_anomaly"),
            "snapshot": ("snapshot", "oi_snapshot", "open_interest_snapshot"),
        }.items():
            section = self._contract_first_mapping(*aliases, payload=payload, domain_data=domain_data)
            if section:
                domain_data.setdefault(name, section)
        if "anomaly" not in domain_data and get("anomaly_type", "type") is not None and "anomaly" in self._topic_from_payload(payload):
            domain_data["anomaly"] = {
                "detected": True,
                "type": get("anomaly_type", "type"),
                "anomaly_type": get("anomaly_type", "type"),
                "confidence": get("anomaly_confidence", "confidence", default=0.0),
                "score": get("anomaly_score", "score", default=0.0),
            }
        if "divergence" not in domain_data and get("divergence_type", "type") is not None and "divergence" in self._topic_from_payload(payload):
            domain_data["divergence"] = {
                "detected": True,
                "type": get("divergence_type", "type"),
                "divergence_type": get("divergence_type", "type"),
                "confidence": get("divergence_confidence", "confidence", default=0.0),
                "score": get("divergence_score", "score", default=0.0),
            }

    def _ensure_price_action_strategy_contract(
            self,
            *,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        section = self._direct_topic_section(FeatureSource.PRICE_ACTION, payload)
        direct = self._direct_payload_value(payload)
        direct_map = self._as_mapping_or_none(direct)
        if section and direct_map and section not in domain_data:
            domain_data[section] = direct_map
        if "fair_value_gap" in domain_data:
            domain_data.setdefault("fvg", domain_data["fair_value_gap"])

    def _build_strategy_contract_feature_snapshots(
            self,
            *,
            source: FeatureSource,
            symbol: str,
            domain_data: dict[str, Any],
            timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        """Create FeatureSnapshot entries from canonical domain sections."""
        if not isinstance(domain_data, dict):
            return []

        contract_paths: dict[FeatureSource, tuple[tuple[str, str], ...]] = {
            FeatureSource.ORDERFLOW: (
                ("orderflow.cvd", "cvd"),
                ("orderflow.cvd.value", "cvd.value"),
                ("orderflow.cvd.delta_ratio", "cvd.delta_ratio"),
                ("orderflow.cvd.cvd_change_pct", "cvd.cvd_change_pct"),
                ("orderflow.cvd.cvd_slope", "cvd.cvd_slope"),
                ("orderflow.cvd.price_change_pct", "cvd.price_change_pct"),
                ("orderflow.volume_delta", "volume_delta"),
                ("orderflow.volume_delta.volume_delta", "volume_delta.volume_delta"),
                ("orderflow.volume_delta.delta_ratio", "volume_delta.delta_ratio"),
                ("orderflow.aggressive_trades", "aggressive_trades"),
                ("orderflow.aggressive_trades.buy_ratio", "aggressive_trades.buy_ratio"),
                ("orderflow.aggressive_trades.sell_ratio", "aggressive_trades.sell_ratio"),
                ("orderflow.aggressive_trades.net_volume_delta", "aggressive_trades.net_volume_delta"),
                ("orderflow.aggressive_trades.net_notional_delta", "aggressive_trades.net_notional_delta"),
                ("orderflow.orderbook_imbalance", "orderbook_imbalance"),
                ("orderflow.orderbook_imbalance.ratio", "orderbook_imbalance.ratio"),
                ("orderflow.orderbook_imbalance.diff", "orderbook_imbalance.diff"),
                ("orderflow.trades_count", "trades_count"),
                ("orderflow.total_volume", "total_volume"),
                ("orderflow.total_notional", "total_notional"),
                ("orderflow.last_price", "last_price"),
                ("orderflow.price_change_pct", "price_change_pct"),
                ("orderflow.signal", "signal"),
                ("orderflow.signal.side", "signal.side"),
                ("orderflow.signal.score", "signal.score"),
                ("orderflow.signal.confidence", "signal.confidence"),
            ),
            FeatureSource.FUNDING: (
                ("funding.snapshot", "snapshot"),
                ("funding.statistics", "statistics"),
                ("funding.regime", "regime"),
                ("funding.regime.confidence", "regime.confidence"),
                ("funding.pressure", "pressure"),
                ("funding.pressure.score", "pressure.score"),
                ("funding.pressure.level", "pressure.level"),
                ("funding.pressure.direction", "pressure.direction"),
                ("funding.extreme", "extreme"),
                ("funding.extreme.type", "extreme.type"),
                ("funding.extreme.severity", "extreme.severity"),
                ("funding.extreme.mean_reversion_probability", "extreme.mean_reversion_probability"),
                ("funding.extreme.squeeze_probability", "extreme.squeeze_probability"),
                ("funding.divergence", "divergence"),
                ("funding.divergence.type", "divergence.type"),
                ("funding.divergence.confidence", "divergence.confidence"),
                ("funding.divergence.score", "divergence.score"),
                ("funding.flip", "flip"),
                ("funding.flip.type", "flip.type"),
                ("funding.flip.confidence", "flip.confidence"),
                ("funding.signal", "signal"),
                ("funding.signal.type", "signal.type"),
                ("funding.signal.score", "signal.score"),
                ("funding.signal.confidence", "signal.confidence"),
                ("funding.signal.bias", "signal.bias"),
            ),
            FeatureSource.SPREADS: (
                ("spreads.snapshot", "snapshot"),
                ("spreads.signal", "signal"),
                ("spreads.opportunity", "opportunity"),
                ("spreads.spread_bps", "spread_bps"),
                ("spreads.basis", "basis"),
                ("spreads.funding_adjusted_spread", "funding_adjusted_spread"),
                ("spreads.net_edge", "net_edge"),
                ("spreads.net_edge_bps", "net_edge_bps"),
                ("spreads.zscore", "zscore"),
                ("spreads.regime", "regime"),
                ("spreads.direction", "direction"),
                ("spreads.has_edge", "has_edge"),
                ("spreads.confidence", "confidence"),
                ("spreads.buy_exchange", "buy_exchange"),
                ("spreads.sell_exchange", "sell_exchange"),
            ),
            FeatureSource.WHALES: (
                ("whales.pressure", "pressure"),
                ("whales.activity", "activity"),
                ("whales.large_trade", "large_trade"),
                ("whales.cluster", "cluster"),
                ("whales.cluster_update", "cluster_update"),
                ("whales.cluster_exhaustion", "cluster_exhaustion"),
                ("whales.liquidation_context", "liquidation_context"),
                ("whales.dominant_side", "dominant_side"),
                ("whales.whale_side", "whale_side"),
                ("whales.liquidation_side", "liquidation_side"),
                ("whales.pressure_score", "pressure_score"),
                ("whales.context_strength", "context_strength"),
                ("whales.cluster_score", "cluster_score"),
                ("whales.continuation_probability", "continuation_probability"),
                ("whales.exhaustion_probability", "exhaustion_probability"),
                ("whales.total_notional", "total_notional"),
                ("whales.trade_count", "trade_count"),
                ("whales.large_trade_notional", "large_trade_notional"),
                ("whales.large_trade_zscore", "large_trade_zscore"),
                ("whales.reference_price", "reference_price"),
                ("whales.confidence", "confidence"),
            ),
            FeatureSource.SPOOFING: (
                ("spoofing.signal", "signal"),
                ("spoofing.features", "features"),
                ("spoofing.detector_results", "detector_results"),
                ("spoofing.type", "signal.type"),
                ("spoofing.pattern", "signal.pattern"),
                ("spoofing.side", "signal.side"),
                ("spoofing.severity", "signal.severity"),
                ("spoofing.score", "signal.score"),
                ("spoofing.confidence", "signal.confidence"),
                ("spoofing.features.pull_ratio", "features.pull_ratio"),
                ("spoofing.features.fill_ratio", "features.fill_ratio"),
                ("spoofing.features.cancel_to_fill_ratio", "features.cancel_to_fill_ratio"),
                ("spoofing.features.distance_from_mid_bps", "features.distance_from_mid_bps"),
                ("spoofing.features.layer_count", "features.layer_count"),
            ),
            FeatureSource.LIQUIDATIONS: (
                ("liquidations.cascade", "cascade"),
                ("liquidations.cascade.confidence", "cascade.confidence"),
                ("liquidations.cascade.intensity_score", "cascade.intensity_score"),
                ("liquidations.cascade.direction", "cascade.direction"),
                ("liquidations.cascade.severity", "cascade.severity"),
                ("liquidations.cascade.continuation_bias", "cascade.continuation_bias"),
                ("liquidations.cascade.exhaustion_bias", "cascade.exhaustion_bias"),
                ("liquidations.exhaustion", "exhaustion"),
                ("liquidations.exhaustion.confidence", "exhaustion.confidence"),
                ("liquidations.exhaustion.exhaustion_bias", "exhaustion.exhaustion_bias"),
                ("liquidations.exhaustion.confirmed", "exhaustion.confirmed"),
                ("liquidations.squeeze", "squeeze"),
                ("liquidations.squeeze.confirmed", "squeeze.confirmed"),
                ("liquidations.squeeze.score", "squeeze.score"),
                ("liquidations.squeeze.direction", "squeeze.direction"),
                ("liquidations.cluster", "cluster"),
            ),
            FeatureSource.LIQUIDITY: (
                ("liquidity.snapshot", "snapshot"),
                ("liquidity.map.snapshot", "snapshot"),
                ("liquidity.current_price", "snapshot.current_price"),
                ("liquidity.above_liquidity_score", "snapshot.above_liquidity_score"),
                ("liquidity.below_liquidity_score", "snapshot.below_liquidity_score"),
                ("liquidity.pressure_score", "snapshot.pressure_score"),
                ("liquidity.bias", "snapshot.bias"),
                ("liquidity.equal_levels", "snapshot.equal_levels"),
                ("liquidity.active_levels", "active_levels"),
                ("liquidity.stop_clusters", "stop_clusters"),
                ("liquidity.zones", "snapshot.zones"),
            ),
            FeatureSource.OPEN_INTEREST: (
                ("open_interest.analysis", "analysis"),
                ("open_interest.snapshot", "snapshot"),
                ("open_interest.context", "market_context"),
                ("open_interest.features", "features"),
                ("open_interest.regime", "regime"),
                ("open_interest.regime.type", "regime.type"),
                ("open_interest.regime.confidence", "regime.confidence"),
                ("open_interest.divergence", "divergence"),
                ("open_interest.divergence.detected", "divergence.detected"),
                ("open_interest.divergence.type", "divergence.type"),
                ("open_interest.divergence.confidence", "divergence.confidence"),
                ("open_interest.anomaly", "anomaly"),
                ("open_interest.anomaly.detected", "anomaly.detected"),
                ("open_interest.anomaly.type", "anomaly.type"),
                ("open_interest.anomaly.confidence", "anomaly.confidence"),
                ("open_interest.features.oi_delta_pct", "features.oi_delta_pct"),
                ("open_interest.features.price_delta_pct", "features.price_delta_pct"),
                ("open_interest.features.oi_pressure_score", "features.oi_pressure_score"),
            ),
            FeatureSource.PRICE_ACTION: (
                ("price_action.market_structure", "market_structure"),
                ("price_action.support_resistance", "support_resistance"),
                ("price_action.fair_value_gap", "fair_value_gap"),
                ("price_action.fvg", "fvg"),
                ("price_action.trend", "trend"),
                ("price_action.current_price", "current_price"),
                ("price_action.last_price", "last_price"),
            ),
        }

        result: list[FeatureSnapshot] = []
        for name, path in contract_paths.get(source, ()):  # type: ignore[arg-type]
            value = self._contract_get_path(domain_data, path, None)
            if value is None and name in domain_data:
                value = domain_data.get(name)
            if value is None or (isinstance(value, (dict, list, tuple, set)) and not value):
                continue
            confidence = self._contract_get_path(domain_data, "confidence", None)
            if confidence is None:
                root = path.split(".")[0]
                confidence = self._contract_get_path(domain_data, f"{root}.confidence", 0.0)
            try:
                result.append(
                    self._snapshot_from_raw_value(
                        source=source,
                        symbol=symbol,
                        name=name,
                        value=value,
                        timestamp=timestamp,
                        confidence=confidence if confidence is not None else 0.0,
                        metadata={"origin": "strategy_contract_adapter", "path": path},
                    )
                )
            except Exception as exc:
                self.log_debug(
                    "Strategy contract feature skipped",
                    source=source.value,
                    symbol=symbol,
                    feature=name,
                    error=str(exc),
                )
        return result

    def _direct_topic_section(
            self,
            source: FeatureSource,
            payload: dict[str, Any],
    ) -> str | None:
        """Infer canonical domain section from the analytics topic."""
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._direct_topic_section")
        topic = self._topic_from_payload(payload)
        if not topic:
            return None

        # Price action modules.
        if source is FeatureSource.PRICE_ACTION:
            if "market_structure" in topic:
                return "market_structure"
            if "support_resistance" in topic or ".sr" in topic:
                return "support_resistance"
            if "fair_value_gap" in topic or ".fvg" in topic:
                return "fair_value_gap"
            if "liquidity_levels" in topic:
                return "liquidity_levels"
            if ".trend" in topic or topic.endswith("trend.updated"):
                return "trend"
            if topic.endswith("price_action.updated"):
                return "composite"
            return None

        if source is FeatureSource.ORDERFLOW:
            if ".cvd" in topic or "cumulative_delta" in topic:
                return "cvd"
            if "volume_delta" in topic or "delta" in topic:
                return "volume_delta"
            if "aggressive_trades" in topic or "aggression" in topic:
                return "aggressive_trades"
            if "orderbook_imbalance" in topic or "book_imbalance" in topic:
                return "orderbook_imbalance"
            if "signal" in topic or "setup" in topic:
                return "signal"
            if topic.endswith("orderflow.updated"):
                return "composite"
            return None

        if source is FeatureSource.OPEN_INTEREST:
            if "regime" in topic:
                return "regime"
            if "divergence" in topic:
                return "divergence"
            if "anomaly" in topic:
                return "anomaly"
            if "features" in topic:
                return "features"
            if "context" in topic:
                return "context"
            if topic.endswith("open_interest.updated") or topic.endswith("oi.updated"):
                return "snapshot"
            return None

        if source is FeatureSource.FUNDING:
            if "statistics" in topic or "stats" in topic:
                return "statistics"
            if "regime" in topic:
                return "regime"
            if "pressure" in topic:
                return "pressure"
            if "extreme" in topic:
                return "extreme"
            if "divergence" in topic:
                return "divergence"
            if "flip" in topic:
                return "flip"
            if "signal" in topic or "setup" in topic:
                return "signal"
            if topic.endswith("funding.updated") or topic.endswith("market.funding.updated"):
                return "snapshot"
            return None

        if source is FeatureSource.LIQUIDATIONS:
            if "cascade" in topic:
                return "cascade"
            if "exhaustion" in topic:
                return "exhaustion"
            if "squeeze" in topic:
                return "squeeze"
            if "cluster" in topic:
                return "cluster"
            if "signal" in topic or "setup" in topic:
                return "signal"
            if "liquidation" in topic:
                return "analysis"
            return None

        if source is FeatureSource.LIQUIDITY:
            if "map" in topic or "snapshot" in topic or topic.endswith("liquidity.updated"):
                return "snapshot"
            if "level" in topic:
                return "active_levels"
            if "cluster" in topic:
                return "stop_clusters"
            if "zone" in topic:
                return "liquidity_zones"
            if "signal" in topic or "setup" in topic:
                return "signal"
            return None

        if source is FeatureSource.SPOOFING:
            if "layering" in topic:
                return "layering"
            if "fake_liquidity" in topic or "fake" in topic:
                return "fake_liquidity"
            if "order_pull" in topic or "pull" in topic:
                return "order_pull"
            if "absorption" in topic:
                return "absorption"
            if "signal" in topic or "setup" in topic:
                return "signal"
            if "spoof" in topic:
                return "analysis"
            return None

        if source is FeatureSource.SPREADS:
            if "basis" in topic and "funding" in topic:
                return "funding_adjusted_spread"
            if "basis" in topic:
                return "basis"
            if "cross_exchange" in topic or "cross" in topic:
                return "cross_exchange"
            if "mean_reversion" in topic:
                return "mean_reversion"
            if "momentum" in topic:
                return "momentum"
            if "signal" in topic or "setup" in topic:
                return "signal"
            if "spread" in topic:
                return "analysis"
            return None

        if source is FeatureSource.WHALES:
            if "absorption" in topic:
                return "absorption"
            if "breakout" in topic:
                return "breakout"
            if "accumulation" in topic:
                return "accumulation"
            if "distribution" in topic:
                return "distribution"
            if "liquidation" in topic:
                return "liquidation_context"
            if "cluster" in topic:
                return "cluster"
            if "signal" in topic or "setup" in topic:
                return "signal"
            if "whale" in topic:
                return "analysis"
            return None

        return None

    def _apply_direct_topic_section_alias(
            self,
            *,
            source: FeatureSource,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        """Expose direct analytics event payload under its canonical section."""
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._apply_direct_topic_section_alias")
        section = self._direct_topic_section(source, payload)
        if not section:
            return

        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        value = self._direct_payload_value(payload, feature_map=feature_map)
        if value is None:
            return

        domain_data.setdefault(section, value)

        aliases_by_source: dict[FeatureSource, dict[str, tuple[str, ...]]] = {
            FeatureSource.PRICE_ACTION: {
                "market_structure": ("structure", "market_structure_state"),
                "support_resistance": ("sr", "support_resistance_state"),
                "fair_value_gap": ("fvg", "fair_value_gap_state", "fvg_state"),
                "liquidity_levels": ("liquidity", "levels"),
                "trend": ("trend_state",),
                "composite": ("state", "snapshot", "price_action", "price_action_state"),
            },
            FeatureSource.ORDERFLOW: {
                "composite": ("snapshot", "orderflow_snapshot", "composite_snapshot"),
                "cvd": ("cumulative_delta", "cvd_state"),
                "volume_delta": ("delta", "volume_delta_state"),
                "aggressive_trades": ("aggression", "aggressive_trades_state"),
                "orderbook_imbalance": ("orderbook", "order_book_imbalance"),
                "signal": ("orderflow_signal", "analytics_signal", "setup"),
            },
            FeatureSource.OPEN_INTEREST: {
                "snapshot": ("oi_snapshot", "open_interest_snapshot"),
                "context": ("oi_context", "open_interest_context"),
                "features": ("oi_features", "open_interest_features"),
                "regime": ("regime_result", "oi_regime", "open_interest_regime"),
                "divergence": ("divergence_result", "oi_divergence", "open_interest_divergence"),
                "anomaly": ("anomaly_result", "oi_anomaly", "open_interest_anomaly"),
            },
            FeatureSource.FUNDING: {
                "snapshot": ("funding_snapshot",),
                "statistics": ("stats", "funding_statistics"),
                "regime": ("regime_state", "funding_regime"),
                "pressure": ("pressure_state", "funding_pressure"),
                "extreme": ("extreme_event", "funding_extreme"),
                "divergence": ("divergence_event", "funding_divergence"),
                "flip": ("flip_event", "funding_flip"),
                "signal": ("funding_signal", "analytics_signal", "setup"),
            },
            FeatureSource.LIQUIDATIONS: {
                "analysis": ("liquidations_analysis", "liquidation_analysis", "result"),
                "cascade": ("liquidation_cascade",),
                "exhaustion": ("liquidation_exhaustion",),
                "squeeze": ("liquidation_squeeze",),
                "cluster": ("liquidation_cluster", "cluster_stats"),
                "signal": ("liquidation_signal", "liquidations_signal", "setup"),
            },
            FeatureSource.LIQUIDITY: {
                "snapshot": ("liquidity_map_snapshot", "liquidity_snapshot"),
                "active_levels": ("liquidity_levels",),
                "stop_clusters": ("liquidity_clusters",),
                "liquidity_zones": ("zones",),
                "signal": ("liquidity_signal", "analytics_signal", "setup"),
            },
            FeatureSource.SPOOFING: {
                "analysis": ("spoofing_analysis", "result"),
                "layering": ("layering_signal",),
                "fake_liquidity": ("fake_liquidity_signal",),
                "order_pull": ("order_pull_signal",),
                "absorption": ("absorption_signal",),
                "signal": ("spoofing_signal", "analytics_signal", "setup"),
            },
            FeatureSource.SPREADS: {
                "analysis": ("spread_analysis", "spreads_analysis", "result"),
                "basis": ("spot_futures_basis", "basis_signal"),
                "funding_adjusted_spread": ("funding_adjusted_basis", "funding_edge"),
                "cross_exchange": ("cross_exchange_spread", "cross_exchange_arb"),
                "mean_reversion": ("spread_mean_reversion",),
                "momentum": ("spread_momentum",),
                "signal": ("spread_signal", "analytics_signal", "setup"),
            },
            FeatureSource.WHALES: {
                "analysis": ("whale_analysis", "whales_analysis", "result"),
                "absorption": ("whale_absorption",),
                "breakout": ("whale_breakout",),
                "accumulation": ("whale_accumulation",),
                "distribution": ("whale_distribution",),
                "liquidation_context": ("whale_liquidation_context",),
                "cluster": ("whale_cluster",),
                "signal": ("whale_signal", "analytics_signal", "setup"),
            },
        }

        for alias in aliases_by_source.get(source, {}).get(section, ()):
            domain_data.setdefault(alias, value)

    def _wrap_price_action_module_view(
            self,
            *,
            module: Any,
            event_payload: dict[str, Any],
            fallback_topic: str,
            section: str,
    ) -> Any:
        """Wrap flat price-action module events into internal/external layer views."""
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._wrap_price_action_module_view")
        module_map = self._as_mapping_or_none(module)
        if module_map is None:
            return module

        if any(key in module_map for key in ("internal", "external", "last_event", "last_signal")):
            return module

        event = dict(event_payload)
        event.update({key: value for key, value in module_map.items() if key not in event})
        event.setdefault("event_type", event.get("type") or event.get("kind") or fallback_topic.split(".")[-1])
        event.setdefault("source_topic", fallback_topic)

        confidence = (
            event.get("confidence")
            or event.get("score")
            or module_map.get("confidence")
            or module_map.get("score")
            or 0.0
        )
        strength = (
            event.get("strength")
            or event.get("score")
            or module_map.get("strength")
            or module_map.get("score")
            or confidence
        )

        layer = dict(module_map)
        layer.setdefault("last_event", event)
        if section == "trend":
            layer.setdefault("last_signal", event)
        layer.setdefault("confidence", confidence)
        layer.setdefault("strength", strength)
        layer.setdefault("score", module_map.get("score", confidence))
        layer.setdefault("updated_at", event.get("timestamp") or event.get("time"))

        result = dict(module_map)
        result.setdefault("external", layer)
        result.setdefault("internal", layer)
        result.setdefault("last_event", event)
        if section == "trend":
            result.setdefault("last_signal", event)
        result.setdefault("confidence", confidence)
        result.setdefault("strength", strength)
        result.setdefault("updated_at", event.get("timestamp") or event.get("time"))
        return result

    def _augment_funding_domain_data(
            self,
            *,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        """
        Normalize analytics.funding.* payloads into the stable funding
        StrategyContext contract.

        Contract sections produced for concrete funding strategies:
            snapshot, statistics, regime, pressure, extreme, divergence, flip, signal

        Important: snapshot/statistics can be enriched from flat funding payloads,
        but actionable sections (extreme/divergence/flip/signal) are only created
        when analytics explicitly emitted that context/event.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._augment_funding_domain_data")

        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def nested_get(value: Any, path: str, default: Any = None) -> Any:
            if not isinstance(path, str) or not path:
                return default
            if isinstance(value, dict) and path in value:
                item = value.get(path)
                return default if item is None else item
            current = value
            for part in path.split("."):
                if current is None:
                    return default
                if isinstance(current, dict):
                    current = current.get(part)
                else:
                    current = getattr(current, part, None)
            return default if current is None else current

        def mapping_for(*keys: str) -> dict[str, Any] | None:
            for key in keys:
                for root in (payload, domain_data, feature_map):
                    value = nested_get(root, key)
                    if isinstance(value, dict) and value:
                        return dict(value)
                # feature_map commonly stores canonical dotted feature names.
                value = nested_get(feature_map, f"funding.{key}")
                if isinstance(value, dict) and value:
                    return dict(value)
            return None

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                for root in (payload, domain_data, feature_map):
                    value = nested_get(root, key)
                    if value is not None:
                        return value
                value = nested_get(feature_map, f"funding.{key}")
                if value is not None:
                    return value
            return default

        topic = str(value_for("event_name", "topic", "source_topic", default="")).strip().lower()
        event_type = str(value_for("event_type", "type", default="")).strip().lower()

        def to_bool(value: Any, default: bool = False) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"1", "true", "yes", "y", "on", "detected", "active", "confirmed", "triggered"}:
                    return True
                if normalized in {"0", "false", "no", "n", "off", "none", "not_detected", "inactive"}:
                    return False
            if isinstance(value, (int, float)):
                return bool(value)
            return default

        def section_detected(section: dict[str, Any] | None) -> bool:
            if not section:
                return False
            detected = section.get("detected", section.get("is_detected", section.get("active", section.get("confirmed"))))
            if detected is None:
                return True
            return to_bool(detected, default=False)

        def section_value(section: dict[str, Any] | None, *paths: str, default: Any = None) -> Any:
            if not isinstance(section, dict):
                return default
            for path in paths:
                value = nested_get(section, path)
                if value is not None:
                    return value
            return default

        def normalize_section(name: str, raw: dict[str, Any] | None) -> dict[str, Any] | None:
            if not isinstance(raw, dict) or not raw:
                return None
            section = dict(raw)
            for _runtime_key in ("feature_map", "features", "strategy_contract", "strategy_contract_version"):
                section.pop(_runtime_key, None)

            def fill(canonical: str, *paths: str, default: Any = None) -> None:
                value = section_value(section, *paths, default=None)
                if value is None:
                    for path in paths:
                        value = value_for(f"funding.{name}.{path}", f"{name}_{path}", path, default=None)
                        if value is not None:
                            break
                if value is None:
                    value = default
                if value is not None:
                    section.setdefault(canonical, value)

            if name == "snapshot":
                fill("funding_rate", "funding_rate", "current_rate", "rate")
                fill("current_rate", "current_rate", "funding_rate", "rate")
                fill("predicted_funding_rate", "predicted_funding_rate", "predicted_rate", "next_funding_rate")
                fill("predicted_rate", "predicted_rate", "predicted_funding_rate", "next_funding_rate")
                fill("mark_price", "mark_price", "current_price", "reference_price")
                fill("event_time", "event_time", "timestamp", "updated_at", "received_at")
            elif name == "statistics":
                fill("current_rate", "current_rate", "funding_rate", "rate")
                fill("mean_rate", "mean_rate", "mean")
                fill("median_rate", "median_rate", "median")
                fill("std_rate", "std_rate", "std", "stdev")
                fill("zscore", "zscore", "z_score")
                fill("sample_size", "sample_size", "samples", "count")
            elif name == "regime":
                fill("type", "type", "regime", "name", "state")
                fill("regime", "regime", "type", "name")
                fill("bias", "bias", "direction", "side")
                fill("score", "score", "confidence")
                fill("confidence", "confidence", "score")
                fill("event_time", "event_time", "timestamp", "updated_at")
            elif name == "pressure":
                fill("score", "score", "pressure_score", "strength", "normalized_score")
                fill("pressure_score", "pressure_score", "score", "strength", "normalized_score")
                fill("level", "level", "pressure_level", "type")
                fill("direction", "direction", "pressure_direction", "bias", "side")
                fill("bias", "bias", "direction", "pressure_direction")
                fill("squeeze_probability", "squeeze_probability", "squeeze_risk")
                fill("mean_reversion_probability", "mean_reversion_probability", "reversion_probability", "reversal_probability")
                fill("event_time", "event_time", "timestamp", "updated_at")
            elif name == "extreme":
                fill("type", "type", "extreme_type", "kind")
                fill("extreme_type", "extreme_type", "type", "kind")
                fill("score", "score", "severity", "strength", "normalized_score")
                fill("severity", "severity", "score", "strength", "normalized_score")
                fill("confidence", "confidence", "severity", "mean_reversion_probability")
                fill("funding_rate", "funding_rate", "current_rate", "rate")
                fill("reversal_risk", "reversal_risk", "is_reversal_risk", "has_reversal_risk", "mean_reversion_risk")
                fill("squeeze_risk", "squeeze_risk", "is_squeeze_risk", "has_squeeze_risk")
                fill("mean_reversion_probability", "mean_reversion_probability", "reversion_probability", "reversal_probability")
                fill("squeeze_probability", "squeeze_probability", "short_squeeze_probability", "long_squeeze_probability", "squeeze_risk")
                fill("event_time", "event_time", "timestamp", "updated_at")
                severity = section_value(section, "severity", "score", default=0.0)
                if section_value(section, "is_reversal_risk") is not None:
                    section.setdefault("reversal_risk", section_value(section, "is_reversal_risk"))
                if section_value(section, "is_squeeze_risk") is not None:
                    section.setdefault("squeeze_risk", section_value(section, "is_squeeze_risk"))
                if to_bool(section.get("reversal_risk")) and section.get("mean_reversion_probability") is None:
                    section["mean_reversion_probability"] = severity
                if to_bool(section.get("squeeze_risk")) and section.get("squeeze_probability") is None:
                    section["squeeze_probability"] = severity
                section.setdefault("detected", True)
            elif name == "divergence":
                fill("type", "type", "divergence_type", "kind")
                fill("divergence_type", "divergence_type", "type", "kind")
                fill("score", "score", "confidence", "strength", "signed_score")
                fill("confidence", "confidence", "score")
                fill("bias", "bias", "direction", "side", "expected_side")
                fill("side", "side", "signal_side", "expected_side", "target_side", "bias", "direction")
                fill("event_time", "event_time", "timestamp", "updated_at")
                section.setdefault("detected", True)
            elif name == "flip":
                fill("type", "type", "flip_type", "kind")
                fill("flip_type", "flip_type", "type", "kind")
                fill("score", "score", "confidence", "flip_magnitude")
                fill("confidence", "confidence", "score")
                fill("magnitude", "magnitude", "flip_magnitude")
                fill("event_time", "event_time", "timestamp", "updated_at")
                section.setdefault("detected", True)
            elif name == "signal":
                fill("type", "type", "signal_type", "setup_type")
                fill("signal_type", "signal_type", "type", "setup_type")
                fill("score", "score", "signed_score", "strength")
                fill("confidence", "confidence", "score_confidence")
                fill("bias", "bias", "direction", "side")
                fill("origin", "origin", "signal_origin")
                fill("event_time", "event_time", "timestamp", "updated_at")
                section.setdefault("detected", True)
            return section

        def set_aliases(target: str, aliases: tuple[str, ...], value: dict[str, Any] | None, *, require_detected: bool = False) -> None:
            normalized = normalize_section(target, value)
            if normalized is None:
                return
            if require_detected and not section_detected(normalized):
                return
            domain_data[target] = normalized
            for alias in aliases:
                domain_data.setdefault(alias, normalized)

        analysis = mapping_for("analysis", "result", "funding_analysis", "funding_result", "payload")
        snapshot = mapping_for("snapshot", "funding_snapshot", "payload.snapshot", "payload.funding_snapshot")
        statistics = mapping_for("statistics", "stats", "funding_statistics", "payload.statistics", "payload.stats")
        regime = mapping_for("regime", "regime_state", "funding_regime", "funding_regime_state", "payload.regime_state")
        pressure = mapping_for("pressure", "pressure_state", "funding_pressure", "funding_pressure_state", "payload.pressure_state")
        extreme = mapping_for("extreme", "extreme_event", "funding_extreme", "funding_extreme_event", "payload.extreme_event")
        divergence = mapping_for("divergence", "divergence_event", "funding_divergence", "funding_divergence_event", "payload.divergence_event")
        flip = mapping_for("flip", "flip_event", "funding_flip", "funding_flip_event", "payload.flip_event")
        signal = mapping_for("signal", "funding_signal", "analytics_signal", "strategy_signal", "setup", "payload.signal")

        if analysis is not None:
            snapshot = snapshot or mapping_for("analysis.snapshot", "analysis.funding_snapshot", "result.snapshot") or (analysis.get("snapshot") if isinstance(analysis.get("snapshot"), dict) else None)
            statistics = statistics or (analysis.get("statistics") if isinstance(analysis.get("statistics"), dict) else None) or (analysis.get("stats") if isinstance(analysis.get("stats"), dict) else None)
            regime = regime or (analysis.get("regime") if isinstance(analysis.get("regime"), dict) else None) or (analysis.get("regime_state") if isinstance(analysis.get("regime_state"), dict) else None)
            pressure = pressure or (analysis.get("pressure") if isinstance(analysis.get("pressure"), dict) else None) or (analysis.get("pressure_state") if isinstance(analysis.get("pressure_state"), dict) else None)
            extreme = extreme or (analysis.get("extreme") if isinstance(analysis.get("extreme"), dict) else None) or (analysis.get("extreme_event") if isinstance(analysis.get("extreme_event"), dict) else None)
            divergence = divergence or (analysis.get("divergence") if isinstance(analysis.get("divergence"), dict) else None) or (analysis.get("divergence_event") if isinstance(analysis.get("divergence_event"), dict) else None)
            flip = flip or (analysis.get("flip") if isinstance(analysis.get("flip"), dict) else None) or (analysis.get("flip_event") if isinstance(analysis.get("flip_event"), dict) else None)
            signal = signal or (analysis.get("signal") if isinstance(analysis.get("signal"), dict) else None) or (analysis.get("funding_signal") if isinstance(analysis.get("funding_signal"), dict) else None)

        if snapshot is None:
            flat_snapshot: dict[str, Any] = {}
            for key in (
                "funding_rate", "current_rate", "next_funding_rate", "predicted_rate",
                "predicted_funding_rate", "annualized_rate", "premium_index",
                "mark_price", "index_price", "open_interest", "volume_24h",
                "next_funding_time", "exchange", "market_type", "symbol",
                "exchange_symbol", "timeframe", "timestamp", "event_time",
            ):
                value = value_for(key, f"funding.{key}", default=None)
                if value is not None:
                    flat_snapshot[key] = value
            if flat_snapshot:
                snapshot = flat_snapshot

        if statistics is None:
            flat_statistics: dict[str, Any] = {}
            for key in (
                "current_rate", "mean_rate", "median_rate", "std_rate", "zscore",
                "z_score", "percentile", "min_rate", "max_rate", "sample_size",
                "samples", "window_start", "window_end", "updated_at",
            ):
                value = value_for(key, f"funding.statistics.{key}", default=None)
                if value is not None:
                    flat_statistics[key] = value
            if flat_statistics:
                statistics = flat_statistics

        if pressure is None and ("pressure" in topic or event_type == "pressure"):
            pressure = dict(payload)
        if regime is None and ("regime" in topic or event_type == "regime"):
            regime = dict(payload)
        if extreme is None and ("extreme" in topic or event_type == "extreme" or value_for("extreme_type", "funding.extreme.type", default=None) is not None):
            extreme = dict(payload)
        if divergence is None and ("divergence" in topic or event_type == "divergence" or value_for("divergence_type", "funding.divergence.type", default=None) is not None):
            divergence = dict(payload)
        if flip is None and ("flip" in topic or event_type == "flip" or value_for("flip_type", "funding.flip.type", default=None) is not None):
            flip = dict(payload)
        signal_like_topic = any(token in topic for token in ("signal", "setup", "confirmed", "generated"))
        if signal is None and (signal_like_topic or event_type == "signal" or value_for("signal_type", "setup_type", default=None) is not None):
            signal = dict(payload)

        set_aliases("analysis", ("funding_analysis", "result"), analysis)
        set_aliases("snapshot", ("funding_snapshot",), snapshot)
        set_aliases("statistics", ("stats", "funding_statistics"), statistics)
        set_aliases("regime", ("regime_state", "funding_regime", "funding_regime_state"), regime)
        set_aliases("pressure", ("pressure_state", "funding_pressure", "funding_pressure_state"), pressure)
        set_aliases("extreme", ("extreme_event", "funding_extreme", "funding_extreme_event"), extreme, require_detected=True)
        set_aliases("divergence", ("divergence_event", "funding_divergence", "funding_divergence_event"), divergence, require_detected=True)
        set_aliases("flip", ("flip_event", "funding_flip", "funding_flip_event"), flip, require_detected=True)
        set_aliases("signal", ("funding_signal", "analytics_signal", "setup"), signal, require_detected=True)

        # Dotted domain aliases make funding_path()/context.get_feature() stable.
        for name in ("snapshot", "statistics", "regime", "pressure", "extreme", "divergence", "flip", "signal"):
            section = domain_data.get(name)
            if section is not None:
                domain_data.setdefault(f"funding.{name}", section)

        def put(path: str, value: Any) -> None:
            if value is not None:
                domain_data.setdefault(path, value)

        put("funding.regime.confidence", section_value(domain_data.get("regime"), "confidence", "score"))
        put("funding.pressure.score", section_value(domain_data.get("pressure"), "score", "pressure_score"))
        put("funding.pressure.level", section_value(domain_data.get("pressure"), "level", "pressure_level"))
        put("funding.pressure.direction", section_value(domain_data.get("pressure"), "direction", "pressure_direction", "bias"))
        put("funding.extreme.type", section_value(domain_data.get("extreme"), "type", "extreme_type"))
        put("funding.extreme.severity", section_value(domain_data.get("extreme"), "severity", "score"))
        put("funding.extreme.mean_reversion_probability", section_value(domain_data.get("extreme"), "mean_reversion_probability", "reversion_probability"))
        put("funding.extreme.squeeze_probability", section_value(domain_data.get("extreme"), "squeeze_probability"))
        put("funding.divergence.type", section_value(domain_data.get("divergence"), "type", "divergence_type"))
        put("funding.divergence.confidence", section_value(domain_data.get("divergence"), "confidence", "score"))
        put("funding.divergence.score", section_value(domain_data.get("divergence"), "score", "confidence", "signed_score"))
        put("funding.flip.type", section_value(domain_data.get("flip"), "type", "flip_type"))
        put("funding.flip.confidence", section_value(domain_data.get("flip"), "confidence", "score"))
        put("funding.signal.type", section_value(domain_data.get("signal"), "type", "signal_type"))
        put("funding.signal.score", section_value(domain_data.get("signal"), "score", "signed_score"))
        put("funding.signal.confidence", section_value(domain_data.get("signal"), "confidence"))
        put("funding.signal.bias", section_value(domain_data.get("signal"), "bias", "direction", "side"))


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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._augment_orderflow_domain_data")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._augment_open_interest_domain_data")
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
        _strategy_logger = (
                getattr(self, "logger", None)
                or getattr(self, "_logger", None)
                or logging.getLogger(__name__ + "." + self.__class__.__name__)
        )
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug(
                "Entering SignalNormalizer._augment_liquidations_domain_data"
            )

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
            "liquidations.cascade",
        )
        exhaustion = mapping_for(
            "exhaustion",
            "exhaustion_result",
            "exhaustion_detection",
            "exhaustion_detected",
            "reversal_context",
            "liquidation_exhaustion",
            "liquidations.exhaustion",
        )
        squeeze = mapping_for(
            "squeeze",
            "squeeze_result",
            "squeeze_reversal",
            "squeeze_context",
            "pending_confirmation",
            "liquidation_squeeze",
            "liquidations.squeeze",
        )
        cluster = mapping_for(
            "cluster",
            "liquidation_cluster",
            "cluster_stats",
            "liquidations_cluster",
            "liquidations.cluster",
        )
        signal = mapping_for(
            "signal",
            "liquidation_signal",
            "liquidations_signal",
            "analytics_signal",
            "setup",
            "liquidations.signal",
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
                    or container.get("liquidations.cascade")
            )
            nested_exhaustion = (
                    container.get("exhaustion")
                    or container.get("exhaustion_result")
                    or container.get("exhaustion_detection")
                    or container.get("reversal_context")
                    or container.get("liquidations.exhaustion")
            )
            nested_squeeze = (
                    container.get("squeeze")
                    or container.get("squeeze_result")
                    or container.get("squeeze_reversal")
                    or container.get("squeeze_context")
                    or container.get("liquidations.squeeze")
            )
            nested_cluster = (
                    container.get("cluster")
                    or container.get("liquidation_cluster")
                    or container.get("cluster_stats")
                    or container.get("liquidations.cluster")
            )
            nested_signal = (
                    container.get("signal")
                    or container.get("liquidation_signal")
                    or container.get("analytics_signal")
                    or container.get("setup")
                    or container.get("liquidations.signal")
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
        # Do not synthesize cluster for plain raw/large liquidation events; only expose
        # flat cluster context when analytics explicitly published cluster/cascade-style
        # context.
        is_cluster_context_event = (
                "cluster" in topic
                or "cascade" in topic
                or "exhaustion" in topic
                or value_for("cluster_detected", default=None) is not None
                or value_for("cluster_id", default=None) is not None
                or value_for("event_count", default=None) is not None
                or value_for("duration_seconds", default=None) is not None
        )

        if cluster is None and is_cluster_context_event:
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
                or value_for("cascade_direction", "continuation_bias", default=None)
                is not None
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
                or value_for("squeeze_confirmed", "squeeze_score", default=None)
                is not None
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

        # Common flat aliases for diagnostics / fallback strategy helpers.
        for key, aliases in {
            "exchange": ("exchange",),
            "market_type": ("market_type",),
            "symbol": ("symbol",),
            "exchange_symbol": ("exchange_symbol",),
            "timeframe": ("timeframe",),
            "side": ("side", "liquidation_side", "direction"),
            "price": ("price", "avg_price", "average_price", "limit_price"),
            "quantity": ("quantity", "qty", "executed_qty"),
            "notional_usd": ("notional_usd", "notional", "total_notional_usd"),
            "timestamp": ("timestamp", "event_time", "trade_time"),
        }.items():
            if key in domain_data and domain_data[key] is not None:
                continue

            value = value_for(*aliases, default=None)
            if value is not None:
                domain_data[key] = value

        # Stable feature aliases. Only set actionable setup features when their
        # sections are truly present/detected.
        if cascade and section_detected(cascade):
            domain_data.setdefault("liquidations.cascade", cascade)
            if "confidence" in cascade:
                domain_data.setdefault(
                    "liquidations.cascade.confidence",
                    cascade.get("confidence"),
                )
            if "intensity_score" in cascade:
                domain_data.setdefault(
                    "liquidations.cascade.intensity_score",
                    cascade.get("intensity_score"),
                )
            if "direction" in cascade:
                domain_data.setdefault(
                    "liquidations.cascade.direction",
                    cascade.get("direction"),
                )

        if exhaustion and section_detected(exhaustion):
            domain_data.setdefault("liquidations.exhaustion", exhaustion)
            if "confidence" in exhaustion:
                domain_data.setdefault(
                    "liquidations.exhaustion.confidence",
                    exhaustion.get("confidence"),
                )
            if "exhaustion_bias" in exhaustion:
                domain_data.setdefault(
                    "liquidations.exhaustion.exhaustion_bias",
                    exhaustion.get("exhaustion_bias"),
                )
            if "bias_delta" in exhaustion:
                domain_data.setdefault(
                    "liquidations.exhaustion.bias_delta",
                    exhaustion.get("bias_delta"),
                )

        if signal and section_detected(signal):
            domain_data.setdefault("liquidations.signal", signal)
            if "confidence" in signal:
                domain_data.setdefault(
                    "liquidations.signal.confidence",
                    signal.get("confidence"),
                )
            if "score" in signal:
                domain_data.setdefault(
                    "liquidations.signal.score",
                    signal.get("score"),
                )
            if "side" in signal:
                domain_data.setdefault(
                    "liquidations.signal.side",
                    signal.get("side"),
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._augment_liquidity_domain_data")
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
        Normalize analytics.price_action.* payloads into a stable price-action
        StrategyContext contract.

        OI-style contract adapter:
        - exposes canonical sections:
          composite, market_structure, support_resistance, fair_value_gap, trend,
          liquidity_levels, signal;
        - preserves analytics-provided nested sections and typed-like objects;
        - enriches composite/state only from real price-action sections;
        - does not synthesize strategy setup sections without explicit signal/setup.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._augment_price_action_domain_data")
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
                    market_structure,
                    support_resistance,
                    fair_value_gap,
                    trend,
                    liquidity_levels,
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
                    "respected",
                    "retested",
                    "aligned",
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
                    "rejected",
                    "misaligned",
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
                else mapping.get(
                    "active",
                    mapping.get(
                        "valid",
                        mapping.get("confirmed", None),
                    ),
                )
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
            "price_action_analysis",
            "result",
        )
        composite = mapping_or_object_for(
            "composite",
            "state",
            "snapshot",
            "price_action",
            "price_action_state",
        )
        market_structure = mapping_or_object_for(
            "market_structure",
            "structure",
            "market_structure_state",
        )
        support_resistance = mapping_or_object_for(
            "support_resistance",
            "sr",
            "support_resistance_state",
        )
        fair_value_gap = mapping_or_object_for(
            "fair_value_gap",
            "fvg",
            "fair_value_gap_state",
            "fvg_state",
        )
        trend = mapping_or_object_for(
            "trend",
            "trend_state",
        )
        liquidity_levels = mapping_or_object_for(
            "liquidity_levels",
            "liquidity",
            "levels",
        )
        signal = mapping_or_object_for(
            "signal",
            "price_action_signal",
            "analytics_signal",
            "setup",
        )

        # Direct price-action module events usually arrive as {"state": ...}
        # and the topic determines whether that state is market_structure,
        # trend, fvg, support/resistance, liquidity, or a composite snapshot.
        direct_section = self._direct_topic_section(FeatureSource.PRICE_ACTION, payload)
        direct_value = self._direct_payload_value(payload, feature_map=feature_map)
        if direct_section and direct_value is not None:
            if direct_section == "market_structure" and market_structure is None:
                market_structure = direct_value
                if composite is direct_value:
                    composite = None
            elif direct_section == "support_resistance" and support_resistance is None:
                support_resistance = direct_value
                if composite is direct_value:
                    composite = None
            elif direct_section == "fair_value_gap" and fair_value_gap is None:
                fair_value_gap = direct_value
                if composite is direct_value:
                    composite = None
            elif direct_section == "trend" and trend is None:
                trend = direct_value
                if composite is direct_value:
                    composite = None
            elif direct_section == "liquidity_levels" and liquidity_levels is None:
                liquidity_levels = direct_value
                if composite is direct_value:
                    composite = None
            elif direct_section == "composite" and composite is None:
                composite = direct_value

        for container_key in (
                "result",
                "payload",
                "data",
                "analysis",
                "price_action_analysis",
                "state",
                "price_action",
                "composite",
        ):
            container = mapping_or_object_for(container_key)
            container_map = as_mapping(container)
            if container_map is None:
                continue

            nested_market_structure = (
                    container_map.get("market_structure")
                    or container_map.get("structure")
                    or container_map.get("market_structure_state")
            )
            nested_support_resistance = (
                    container_map.get("support_resistance")
                    or container_map.get("sr")
                    or container_map.get("support_resistance_state")
            )
            nested_fvg = (
                    container_map.get("fair_value_gap")
                    or container_map.get("fvg")
                    or container_map.get("fair_value_gap_state")
                    or container_map.get("fvg_state")
            )
            nested_trend = (
                    container_map.get("trend")
                    or container_map.get("trend_state")
            )
            nested_liquidity_levels = (
                    container_map.get("liquidity_levels")
                    or container_map.get("liquidity")
                    or container_map.get("levels")
            )
            nested_signal = (
                    container_map.get("signal")
                    or container_map.get("price_action_signal")
                    or container_map.get("analytics_signal")
                    or container_map.get("setup")
            )

            if market_structure is None and nested_market_structure is not None:
                market_structure = nested_market_structure
            if support_resistance is None and nested_support_resistance is not None:
                support_resistance = nested_support_resistance
            if fair_value_gap is None and nested_fvg is not None:
                fair_value_gap = nested_fvg
            if trend is None and nested_trend is not None:
                trend = nested_trend
            if liquidity_levels is None and nested_liquidity_levels is not None:
                liquidity_levels = nested_liquidity_levels
            if signal is None and nested_signal is not None:
                signal = nested_signal

        if composite is not None:
            if market_structure is None:
                nested = get_item(composite, "market_structure", None) or get_item(
                    composite,
                    "structure",
                    None,
                )
                if nested is not None:
                    market_structure = nested

            if support_resistance is None:
                nested = get_item(composite, "support_resistance", None) or get_item(
                    composite,
                    "sr",
                    None,
                )
                if nested is not None:
                    support_resistance = nested

            if fair_value_gap is None:
                nested = get_item(composite, "fair_value_gap", None) or get_item(
                    composite,
                    "fvg",
                    None,
                )
                if nested is not None:
                    fair_value_gap = nested

            if trend is None:
                nested = get_item(composite, "trend", None)
                if nested is not None:
                    trend = nested

            if liquidity_levels is None:
                nested = get_item(composite, "liquidity_levels", None) or get_item(
                    composite,
                    "liquidity",
                    None,
                )
                if nested is not None:
                    liquidity_levels = nested

            composite_map = as_mapping(composite)
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

        # Composite is context, not a setup. It is safe to compose it from real
        # module sections, but not from arbitrary raw payload.
        if composite is None and any(
                section is not None
                for section in (
                        market_structure,
                        support_resistance,
                        fair_value_gap,
                        trend,
                        liquidity_levels,
                )
        ):
            composite = {
                "market_structure": market_structure or {},
                "support_resistance": support_resistance or {},
                "fair_value_gap": fair_value_gap or {},
                "trend": trend or {},
                "liquidity_levels": liquidity_levels or {},
            }

            for key in (
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "timestamp",
                    "current_price",
                    "last_price",
                    "price",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    composite[key] = value

        signal_like_topic = any(
            token in topic
            for token in (
                "signal",
                "setup",
                "break",
                "bos",
                "choch",
                "mss",
                "fvg",
                "support",
                "resistance",
                "trend",
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
                "origin": payload.get("origin", "price_action"),
            }

        if market_structure is not None:
            market_structure = self._ensure_price_action_market_structure_view(
                module=self._as_mapping_or_none(market_structure) or {"value": market_structure},
                event_payload=payload,
                fallback_topic=topic,
            )
        if trend is not None:
            trend = self._ensure_price_action_trend_view(
                module=self._as_mapping_or_none(trend) or {"value": trend},
                event_payload=payload,
                fallback_topic=topic,
            )
        if support_resistance is not None:
            support_resistance = self._wrap_price_action_module_view(
                module=support_resistance,
                event_payload=payload,
                fallback_topic=topic,
                section="support_resistance",
            )
        if fair_value_gap is not None:
            fair_value_gap = self._wrap_price_action_module_view(
                module=fair_value_gap,
                event_payload=payload,
                fallback_topic=topic,
                section="fair_value_gap",
            )
        if liquidity_levels is not None:
            liquidity_levels = self._wrap_price_action_module_view(
                module=liquidity_levels,
                event_payload=payload,
                fallback_topic=topic,
                section="liquidity_levels",
            )

        set_aliases(
            "analysis",
            ("price_action_analysis", "result"),
            analysis,
            override=False,
        )

        if composite is not None:
            set_aliases(
                "composite",
                ("state", "snapshot", "price_action", "price_action_state"),
                composite,
            )

        if market_structure is not None:
            set_aliases(
                "market_structure",
                ("structure", "market_structure_state"),
                market_structure,
            )

        if support_resistance is not None:
            set_aliases(
                "support_resistance",
                ("sr", "support_resistance_state"),
                support_resistance,
            )

        if fair_value_gap is not None:
            set_aliases(
                "fair_value_gap",
                ("fvg", "fair_value_gap_state", "fvg_state"),
                fair_value_gap,
            )

        if trend is not None:
            set_aliases(
                "trend",
                ("trend_state",),
                trend,
            )

        if liquidity_levels is not None:
            set_aliases(
                "liquidity_levels",
                ("liquidity", "levels"),
                liquidity_levels,
            )

        if section_present(signal):
            set_aliases(
                "signal",
                ("price_action_signal", "analytics_signal", "setup"),
                signal,
            )

        for key in (
                "exchange",
                "market_type",
                "symbol",
                "exchange_symbol",
                "timeframe",
                "timestamp",
                "current_price",
                "last_price",
                "price",
        ):
            value = value_for(key, default=None)
            if value is not None:
                domain_data.setdefault(key, value)


    @staticmethod
    def _direction_to_long_short(value: Any) -> str | None:
        _strategy_logger = logging.getLogger(__name__ + ".SignalNormalizer._direction_to_long_short")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._direction_to_long_short")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._ensure_price_action_market_structure_view")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._ensure_price_action_trend_view")
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
        """
        Normalize analytics.spoofing.* payloads into the stable spoofing
        StrategyContext contract.

        OI-style contract adapter:
        - exposes canonical sections: composite, signal, features,
          detector_results, score_breakdown, analytics_metadata;
        - preserves analytics-provided signal/composite objects;
        - does not synthesize spoofing.signal from a generic payload unless the
          event/payload explicitly represents a spoofing signal/setup;
        - keeps detector-specific details available for concrete spoofing
          strategies without duplicating detector logic in strategies.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._augment_spoofing_domain_data")
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
                    signal,
                    features,
                    detector_results,
                    score_breakdown,
                    analytics_metadata,
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
                    "confirmed",
                    "valid",
                    "active",
                    "detected",
                    "passed",
                    "triggered",
                    "pulled",
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
                    "failed",
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
                else mapping.get(
                    "active",
                    mapping.get(
                        "valid",
                        mapping.get(
                            "passed",
                            mapping.get("confirmed", None),
                        ),
                    ),
                )
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
            "spoofing_analysis",
            "result",
        )
        composite = mapping_or_object_for(
            "composite",
            "snapshot",
            "spoofing",
            "spoofing_snapshot",
            "composite_snapshot",
        )
        signal = mapping_or_object_for(
            "signal",
            "spoofing_signal",
            "analytics_signal",
            "setup",
        )
        features = mapping_or_object_for(
            "features",
            "spoofing_features",
        )
        detector_results = mapping_or_object_for(
            "detector_results",
            "detectors",
            "detector_state",
            "components",
        )
        score_breakdown = mapping_or_object_for(
            "score_breakdown",
            "scores",
            "scoring",
        )
        analytics_metadata = mapping_or_object_for(
            "analytics_metadata",
            "metadata",
            "meta",
        )

        for container_key in (
                "result",
                "payload",
                "data",
                "analysis",
                "spoofing_analysis",
                "event",
                "spoofing",
                "composite",
        ):
            container = mapping_or_object_for(container_key)
            container_map = as_mapping(container)
            if container_map is None:
                continue

            nested_composite = (
                    container_map.get("composite")
                    or container_map.get("snapshot")
                    or container_map.get("spoofing")
                    or container_map.get("spoofing_snapshot")
                    or container_map.get("composite_snapshot")
            )
            nested_signal = (
                    container_map.get("signal")
                    or container_map.get("spoofing_signal")
                    or container_map.get("analytics_signal")
                    or container_map.get("setup")
            )
            nested_features = (
                    container_map.get("features")
                    or container_map.get("spoofing_features")
            )
            nested_detectors = (
                    container_map.get("detector_results")
                    or container_map.get("detectors")
                    or container_map.get("detector_state")
                    or container_map.get("components")
            )
            nested_score_breakdown = (
                    container_map.get("score_breakdown")
                    or container_map.get("scores")
                    or container_map.get("scoring")
            )
            nested_metadata = (
                    container_map.get("analytics_metadata")
                    or container_map.get("metadata")
                    or container_map.get("meta")
            )

            if composite is None and nested_composite is not None:
                composite = nested_composite
            if signal is None and nested_signal is not None:
                signal = nested_signal
            if features is None and nested_features is not None:
                features = nested_features
            if detector_results is None and nested_detectors is not None:
                detector_results = nested_detectors
            if score_breakdown is None and nested_score_breakdown is not None:
                score_breakdown = nested_score_breakdown
            if analytics_metadata is None and nested_metadata is not None:
                analytics_metadata = nested_metadata

        if signal is not None:
            if features is None:
                nested = get_item(signal, "features", None)
                if nested is not None:
                    features = nested

            if detector_results is None:
                nested = (
                        get_item(signal, "detector_results", None)
                        or get_item(signal, "detectors", None)
                        or get_item(signal, "components", None)
                )
                if nested is not None:
                    detector_results = nested

            if score_breakdown is None:
                nested = (
                        get_item(signal, "score_breakdown", None)
                        or get_item(signal, "scores", None)
                )
                if nested is not None:
                    score_breakdown = nested

            if analytics_metadata is None:
                nested = get_item(signal, "metadata", None)
                if nested is not None:
                    analytics_metadata = nested

        # Safe flat enrichment: features are context, not a strategy setup.
        if features is None:
            flat_features: dict[str, Any] = {}
            for key in (
                    "pull_ratio",
                    "fill_ratio",
                    "price_reaction_bps",
                    "signed_price_reaction_bps",
                    "lifetime_ms",
                    "wall_notional",
                    "pulled_notional",
                    "cancel_to_fill_ratio",
                    "distance_from_mid_bps",
                    "layer_count",
                    "layer_price_span_bps",
                    "pressure_flip_strength",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    flat_features[key] = value

            if flat_features:
                features = flat_features

        # Spoofing signal is actionable setup context. Only synthesize it when the
        # event/topic or explicit fields indicate analytics produced a spoofing signal.
        signal_like_topic = any(
            token in topic
            for token in (
                "signal",
                "setup",
                "spoofing",
                "fake_liquidity",
                "order_pull",
                "pressure_bluff",
                "layering",
                "absorption",
                "composite",
                "confirmed",
                "generated",
            )
        )
        explicit_signal = bool(
            signal_like_topic
            or payload.get("spoofing_type") is not None
            or payload.get("spoofing_side") is not None
            or payload.get("pattern") is not None
            or payload.get("signal_type") is not None
            or payload.get("setup_type") is not None
            or payload.get("signal_side") is not None
            or to_bool(payload.get("signal_detected", False))
            or to_bool(payload.get("setup_detected", False))
            or to_bool(payload.get("detected", False))
        )

        if signal is None and explicit_signal:
            signal = {
                "detected": True,
                "spoofing_type": payload.get("spoofing_type")
                                 or payload.get("type")
                                 or payload.get("signal_type")
                                 or payload.get("setup_type"),
                "pattern": payload.get("pattern") or payload.get("spoofing_pattern"),
                "side": (
                        payload.get("spoofing_side")
                        or payload.get("signal_side")
                        or payload.get("side")
                        or payload.get("direction")
                ),
                "severity": payload.get("severity"),
                "status": payload.get("status"),
                "score": payload.get("score", payload.get("signal_score", 0.0)),
                "confidence": payload.get(
                    "confidence",
                    payload.get("signal_confidence", 0.0),
                ),
                "price_level": payload.get("price_level"),
                "wall_id": payload.get("wall_id"),
                "event_time": payload.get("event_time") or payload.get("timestamp"),
                "features": features or {},
                "detector_results": detector_results or {},
                "score_breakdown": score_breakdown or {},
                "metadata": analytics_metadata or {},
            }

        # Composite is a context container. It may be composed from real signal/features,
        # but not from arbitrary raw payload.
        if composite is None and any(
                section is not None
                for section in (
                        signal,
                        features,
                        detector_results,
                        score_breakdown,
                        analytics_metadata,
                )
        ):
            composite = {
                "signal": signal or {},
                "features": features or {},
                "detector_results": detector_results or {},
                "score_breakdown": score_breakdown or {},
                "analytics_metadata": analytics_metadata or {},
            }

            for key in (
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "timestamp",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    composite[key] = value

        set_aliases(
            "analysis",
            ("spoofing_analysis", "result"),
            analysis,
            override=False,
        )

        if composite is not None:
            set_aliases(
                "composite",
                ("snapshot", "spoofing", "spoofing_snapshot", "composite_snapshot"),
                composite,
            )

        if section_present(signal):
            set_aliases(
                "signal",
                ("spoofing_signal", "analytics_signal", "setup"),
                signal,
            )

        if features is not None:
            set_aliases(
                "features",
                ("spoofing_features",),
                features,
            )

        if detector_results is not None:
            set_aliases(
                "detector_results",
                ("detectors", "detector_state", "components"),
                detector_results,
            )

        if score_breakdown is not None:
            set_aliases(
                "score_breakdown",
                ("scores", "scoring"),
                score_breakdown,
            )

        if analytics_metadata is not None:
            set_aliases(
                "analytics_metadata",
                ("metadata", "meta"),
                analytics_metadata,
                override=False,
            )

        # Stable flat aliases used by SpoofingCompositeSnapshot and concrete strategies.
        for key, aliases in {
            "spoofing_type": ("type", "signal_type", "setup_type"),
            "pattern": ("spoofing_pattern",),
            "side": ("spoofing_side", "signal_side", "direction"),
            "severity": ("spoofing_severity",),
            "status": ("spoofing_status",),
            "score": ("signal_score",),
            "confidence": ("signal_confidence",),
            "price_level": ("level", "wall_price"),
            "wall_id": (),
            "event_time": ("timestamp", "time"),
            "exchange": (),
            "market_type": (),
            "symbol": (),
            "exchange_symbol": (),
            "timeframe": (),
        }.items():
            value = value_for(key, *aliases, default=None)
            if value is not None:
                domain_data.setdefault(key, value)

    def _augment_spreads_domain_data(
            self,
            *,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        """
        Normalize analytics.spreads.* payloads into the stable spreads
        StrategyContext contract.

        OI-style contract adapter:
        - exposes canonical sections: snapshot, signal, opportunity;
        - preserves spot/futures and cross-exchange semantics;
        - keeps basis/funding-adjusted/cross-exchange fields under stable aliases;
        - does not synthesize setup/signal/opportunity unless analytics supplied it;
        - does not collapse spot_futures/cross_exchange structure.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._augment_spreads_domain_data")
        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def as_mapping(value: Any) -> dict[str, Any] | None:
            if isinstance(value, dict):
                return value

            to_payload = getattr(value, "to_payload", None)
            if callable(to_payload):
                converted = to_payload()
                if isinstance(converted, dict):
                    return converted

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

            for container in (snapshot, signal, opportunity):
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
                    "valid",
                    "active",
                    "tradeable",
                    "passed",
                    "open",
                    "confirmed",
                    "has_edge",
                    "detected",
                }:
                    return True
                if normalized in {
                    "0",
                    "false",
                    "no",
                    "n",
                    "off",
                    "invalid",
                    "inactive",
                    "expired",
                    "closed",
                    "failed",
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
                else mapping.get(
                    "active",
                    mapping.get(
                        "valid",
                        mapping.get(
                            "tradeable",
                            mapping.get("confirmed", None),
                        ),
                    ),
                )
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
            "spreads_analysis",
            "spread_analysis",
            "result",
        )
        snapshot = mapping_or_object_for(
            "snapshot",
            "spread_snapshot",
            "spreads_snapshot",
            "spot_futures_snapshot",
            "cross_exchange_snapshot",
            "basis_snapshot",
        )
        signal = mapping_or_object_for(
            "signal",
            "spread_signal",
            "spreads_signal",
            "analytics_signal",
            "setup",
        )
        opportunity = mapping_or_object_for(
            "opportunity",
            "arbitrage_opportunity",
            "arb_opportunity",
            "cross_exchange_opportunity",
        )

        for container_key in (
                "result",
                "payload",
                "data",
                "analysis",
                "spreads_analysis",
                "spread_analysis",
                "event",
        ):
            container = mapping_or_object_for(container_key)
            container_map = as_mapping(container)
            if container_map is None:
                continue

            nested_snapshot = (
                    container_map.get("snapshot")
                    or container_map.get("spread_snapshot")
                    or container_map.get("spreads_snapshot")
                    or container_map.get("spot_futures_snapshot")
                    or container_map.get("cross_exchange_snapshot")
                    or container_map.get("basis_snapshot")
            )
            nested_signal = (
                    container_map.get("signal")
                    or container_map.get("spread_signal")
                    or container_map.get("spreads_signal")
                    or container_map.get("analytics_signal")
                    or container_map.get("setup")
            )
            nested_opportunity = (
                    container_map.get("opportunity")
                    or container_map.get("arbitrage_opportunity")
                    or container_map.get("arb_opportunity")
                    or container_map.get("cross_exchange_opportunity")
            )

            if snapshot is None and nested_snapshot is not None:
                snapshot = nested_snapshot
            if signal is None and nested_signal is not None:
                signal = nested_signal
            if opportunity is None and nested_opportunity is not None:
                opportunity = nested_opportunity

        # If a typed SpreadCompositeSnapshot-like object is already passed as
        # payload/result, keep it as snapshot instead of flattening it.
        if snapshot is None:
            maybe_spread_type = value_for("spread_type", "type", default=None)
            maybe_spread_bps = value_for("spread_bps", "basis", "net_edge", default=None)
            maybe_exchange_a = value_for("exchange_a", "spot_exchange", "buy_exchange", default=None)
            maybe_exchange_b = value_for("exchange_b", "futures_exchange", "sell_exchange", default=None)

            if (
                    maybe_spread_type is not None
                    or maybe_spread_bps is not None
                    or maybe_exchange_a is not None
                    or maybe_exchange_b is not None
            ):
                flat_snapshot: dict[str, Any] = {}
                for key in (
                        "spread_type",
                        "type",
                        "symbol",
                        "exchange_a",
                        "exchange_b",
                        "market_type_a",
                        "market_type_b",
                        "exchange_symbol_a",
                        "exchange_symbol_b",
                        "spot_exchange",
                        "futures_exchange",
                        "spot_symbol",
                        "futures_symbol",
                        "instrument_type",
                        "leg_a",
                        "leg_b",
                        "spread_bps",
                        "basis",
                        "funding_adjusted_spread",
                        "net_edge",
                        "net_edge_bps",
                        "zscore",
                        "regime",
                        "direction",
                        "signal_type",
                        "quote_validity",
                        "has_edge",
                        "confidence",
                        "timestamp",
                        "timeframe",
                        "metadata",
                ):
                    value = value_for(key, default=None)
                    if value is not None:
                        flat_snapshot[key] = value

                if flat_snapshot:
                    snapshot = flat_snapshot

        # Signal is only exposed when analytics supplied signal/setup context.
        signal_like_topic = any(
            token in topic
            for token in (
                "signal",
                "setup",
                "mean_reversion",
                "momentum",
                "widening",
                "compressing",
                "regime_shift",
                "anomaly",
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
                "signal_type": payload.get("signal_type") or payload.get("setup_type"),
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
                "origin": payload.get("origin", "spreads"),
            }

        # Opportunity is only exposed when analytics supplied arbitrage/cross-exchange
        # opportunity context. Do not create opportunity from generic snapshot.
        opportunity_like_topic = any(
            token in topic
            for token in (
                "opportunity",
                "arbitrage",
                "arb",
                "cross_exchange",
            )
        )
        explicit_opportunity = bool(
            opportunity_like_topic
            or payload.get("opportunity_key") is not None
            or payload.get("opportunity_status") is not None
            or payload.get("buy_exchange") is not None
            or payload.get("sell_exchange") is not None
            or payload.get("net_edge") is not None
            or payload.get("net_edge_bps") is not None
        )

        if opportunity is None and explicit_opportunity:
            opportunity = {
                "active": to_bool(payload.get("active", True), True),
                "tradeable": to_bool(payload.get("tradeable", True), True),
                "opportunity_key": payload.get("opportunity_key"),
                "status": payload.get("opportunity_status") or payload.get("status"),
                "buy_exchange": payload.get("buy_exchange"),
                "sell_exchange": payload.get("sell_exchange"),
                "buy_market_type": payload.get("buy_market_type"),
                "sell_market_type": payload.get("sell_market_type"),
                "net_edge": payload.get("net_edge"),
                "net_edge_bps": payload.get("net_edge_bps"),
                "persistence_ms": payload.get("persistence_ms"),
                "confidence": payload.get("confidence"),
            }

        set_aliases(
            "analysis",
            ("spreads_analysis", "spread_analysis", "result"),
            analysis,
            override=False,
        )

        if snapshot is not None:
            set_aliases(
                "snapshot",
                (
                    "spread_snapshot",
                    "spreads_snapshot",
                    "spot_futures_snapshot",
                    "cross_exchange_snapshot",
                    "basis_snapshot",
                ),
                snapshot,
            )

        if section_present(signal):
            set_aliases(
                "signal",
                (
                    "spread_signal",
                    "spreads_signal",
                    "analytics_signal",
                    "setup",
                ),
                signal,
            )

        if section_present(opportunity):
            set_aliases(
                "opportunity",
                (
                    "arbitrage_opportunity",
                    "arb_opportunity",
                    "cross_exchange_opportunity",
                ),
                opportunity,
            )

        # Stable flat aliases used by SpreadCompositeSnapshot and spread strategies.
        for key, aliases in {
            "spread_type": ("type",),
            "symbol": (),
            "exchange_a": ("spot_exchange", "buy_exchange"),
            "exchange_b": ("futures_exchange", "sell_exchange"),
            "market_type_a": ("spot_market_type", "buy_market_type"),
            "market_type_b": ("futures_market_type", "sell_market_type"),
            "exchange_symbol_a": ("spot_symbol", "buy_symbol"),
            "exchange_symbol_b": ("futures_symbol", "sell_symbol"),
            "spread_bps": ("basis_bps",),
            "basis": ("basis_value",),
            "funding_adjusted_spread": ("funding_adjusted_basis", "funding_edge"),
            "net_edge": ("edge",),
            "net_edge_bps": ("edge_bps",),
            "zscore": ("z_score",),
            "regime": ("spread_regime",),
            "direction": ("spread_direction",),
            "signal_type": ("spread_signal_type",),
            "quote_validity": ("quote_status",),
            "has_edge": ("tradeable_edge",),
            "confidence": ("score_confidence",),
            "opportunity_key": (),
            "opportunity_status": ("status",),
            "persistence_ms": (),
            "buy_exchange": (),
            "sell_exchange": (),
            "buy_market_type": (),
            "sell_market_type": (),
            "instrument_type": (),
            "timestamp": ("event_time", "time"),
            "timeframe": (),
        }.items():
            value = value_for(key, *aliases, default=None)
            if value is not None:
                domain_data.setdefault(key, value)

    def _augment_whales_domain_data(
            self,
            *,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> None:
        """
        Normalize analytics.whales.* payloads into the stable whales
        StrategyContext contract.

        OI-style contract adapter:
        - exposes canonical sections:
          pressure, activity, large_trade, cluster, cluster_update,
          cluster_exhaustion, liquidation_context, metadata;
        - preserves analytics-provided nested/typed sections;
        - enriches flat fields only into the matching context section;
        - does not create trade signals or risk-ready payloads;
        - concrete whale strategies remain StrategyContext decision modules.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._augment_whales_domain_data")
        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def as_mapping(value: Any) -> dict[str, Any] | None:
            if isinstance(value, dict):
                return value

            to_payload = getattr(value, "to_payload", None)
            if callable(to_payload):
                converted = to_payload()
                if isinstance(converted, dict):
                    return converted

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
                    pressure,
                    activity,
                    large_trade,
                    cluster,
                    cluster_update,
                    cluster_exhaustion,
                    liquidation_context,
                    metadata,
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
                    "valid",
                    "active",
                    "tradeable",
                    "passed",
                    "confirmed",
                    "enabled",
                    "detected",
                }:
                    return True
                if normalized in {
                    "0",
                    "false",
                    "no",
                    "n",
                    "off",
                    "invalid",
                    "inactive",
                    "failed",
                    "disabled",
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

            detected = mapping.get(
                "detected",
                mapping.get(
                    "active",
                    mapping.get(
                        "valid",
                        mapping.get("confirmed", None),
                    ),
                ),
            )

            if detected is None:
                return True

            return to_bool(detected, default=False)

        analysis = mapping_or_object_for(
            "analysis",
            "whales_analysis",
            "whale_analysis",
            "result",
        )
        pressure = mapping_or_object_for(
            "pressure",
            "whale_pressure",
            "whale_pressure_signal",
        )
        activity = mapping_or_object_for(
            "activity",
            "whale_activity",
            "whale_activity_signal",
        )
        large_trade = mapping_or_object_for(
            "large_trade",
            "large_trade_signal",
            "whale_large_trade",
        )
        cluster = mapping_or_object_for(
            "cluster",
            "whale_cluster",
            "whale_cluster_signal",
        )
        cluster_update = mapping_or_object_for(
            "cluster_update",
            "whale_cluster_update",
            "whale_cluster_update_signal",
        )
        cluster_exhaustion = mapping_or_object_for(
            "cluster_exhaustion",
            "whale_cluster_exhaustion",
            "whale_cluster_exhaustion_signal",
        )
        liquidation_context = mapping_or_object_for(
            "liquidation_context",
            "whale_liquidation_context",
            "whale_liquidation_context_signal",
        )
        metadata = mapping_or_object_for(
            "metadata",
            "analytics_metadata",
            "meta",
        )

        for container_key in (
                "result",
                "payload",
                "data",
                "analysis",
                "whales_analysis",
                "whale_analysis",
                "event",
        ):
            container = mapping_or_object_for(container_key)
            container_map = as_mapping(container)
            if container_map is None:
                continue

            nested_pressure = (
                    container_map.get("pressure")
                    or container_map.get("whale_pressure")
                    or container_map.get("whale_pressure_signal")
            )
            nested_activity = (
                    container_map.get("activity")
                    or container_map.get("whale_activity")
                    or container_map.get("whale_activity_signal")
            )
            nested_large_trade = (
                    container_map.get("large_trade")
                    or container_map.get("large_trade_signal")
                    or container_map.get("whale_large_trade")
            )
            nested_cluster = (
                    container_map.get("cluster")
                    or container_map.get("whale_cluster")
                    or container_map.get("whale_cluster_signal")
            )
            nested_cluster_update = (
                    container_map.get("cluster_update")
                    or container_map.get("whale_cluster_update")
                    or container_map.get("whale_cluster_update_signal")
            )
            nested_cluster_exhaustion = (
                    container_map.get("cluster_exhaustion")
                    or container_map.get("whale_cluster_exhaustion")
                    or container_map.get("whale_cluster_exhaustion_signal")
            )
            nested_liquidation_context = (
                    container_map.get("liquidation_context")
                    or container_map.get("whale_liquidation_context")
                    or container_map.get("whale_liquidation_context_signal")
            )
            nested_metadata = (
                    container_map.get("metadata")
                    or container_map.get("analytics_metadata")
                    or container_map.get("meta")
            )

            if pressure is None and nested_pressure is not None:
                pressure = nested_pressure
            if activity is None and nested_activity is not None:
                activity = nested_activity
            if large_trade is None and nested_large_trade is not None:
                large_trade = nested_large_trade
            if cluster is None and nested_cluster is not None:
                cluster = nested_cluster
            if cluster_update is None and nested_cluster_update is not None:
                cluster_update = nested_cluster_update
            if cluster_exhaustion is None and nested_cluster_exhaustion is not None:
                cluster_exhaustion = nested_cluster_exhaustion
            if liquidation_context is None and nested_liquidation_context is not None:
                liquidation_context = nested_liquidation_context
            if metadata is None and nested_metadata is not None:
                metadata = nested_metadata

        # Flat enrichment is section-specific. Do not copy the whole payload into
        # all sections; only create a section when the relevant fields exist.
        topic = self._topic_from_payload(payload).lower()
        is_large_trade_event = "large_trade" in topic
        is_cluster_event = "cluster" in topic
        is_liquidation_event = "liquidation" in topic

        def has_any_value(*keys: str) -> bool:
            return any(value_for(key, default=None) is not None for key in keys)

        has_cluster_context_fields = is_cluster_event or has_any_value(
            "cluster_side",
            "cluster_score",
            "continuation_probability",
            "exhaustion_probability",
        )
        has_liquidation_context_fields = is_liquidation_event or has_any_value(
            "liquidation_side",
            "liquidation_notional",
            "liquidated_notional",
            "total_liquidation_notional",
        )
        if pressure is None and not is_large_trade_event:
            flat_pressure: dict[str, Any] = {}
            for key in (
                    "dominant_side",
                    "whale_side",
                    "imbalance_ratio",
                    "pressure_score",
                    "total_notional",
                    "trade_count",
                    "reference_price",
                    "confidence",
                    "timestamp",
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "metadata",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    flat_pressure[key] = value

            if flat_pressure:
                pressure = flat_pressure

        if activity is None:
            flat_activity: dict[str, Any] = {}
            for key in (
                    "activity_side",
                    "whale_side",
                    "dominant_side",
                    "total_notional",
                    "notional",
                    "trade_count",
                    "reference_price",
                    "confidence",
                    "timestamp",
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "metadata",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    flat_activity[key] = value

            if flat_activity:
                activity = flat_activity

        if large_trade is None:
            flat_large_trade: dict[str, Any] = {}
            for key in (
                    "large_trade_notional",
                    "large_trade_zscore",
                    "notional",
                    "zscore",
                    "z_score",
                    "whale_side",
                    "side",
                    "reference_price",
                    "confidence",
                    "timestamp",
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "metadata",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    flat_large_trade[key] = value

            if flat_large_trade:
                large_trade = flat_large_trade

        if cluster is None and not is_large_trade_event and has_cluster_context_fields:
            flat_cluster: dict[str, Any] = {}
            for key in (
                    "cluster_side",
                    "whale_side",
                    "cluster_score",
                    "context_strength",
                    "continuation_probability",
                    "exhaustion_probability",
                    "total_notional",
                    "trade_count",
                    "reference_price",
                    "confidence",
                    "timestamp",
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "metadata",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    flat_cluster[key] = value

            if flat_cluster:
                cluster = flat_cluster

        if cluster_update is None and not is_large_trade_event and has_cluster_context_fields:
            flat_cluster_update: dict[str, Any] = {}
            for key in (
                    "cluster_update_side",
                    "cluster_side",
                    "cluster_score",
                    "context_strength",
                    "continuation_probability",
                    "exhaustion_probability",
                    "total_notional",
                    "trade_count",
                    "reference_price",
                    "confidence",
                    "timestamp",
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "metadata",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    flat_cluster_update[key] = value

            if flat_cluster_update:
                cluster_update = flat_cluster_update

        if cluster_exhaustion is None and not is_large_trade_event and has_cluster_context_fields:
            flat_cluster_exhaustion: dict[str, Any] = {}
            for key in (
                    "exhausted_side",
                    "cluster_side",
                    "exhaustion_probability",
                    "context_strength",
                    "cluster_score",
                    "total_notional",
                    "trade_count",
                    "reference_price",
                    "confidence",
                    "timestamp",
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "metadata",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    flat_cluster_exhaustion[key] = value

            if flat_cluster_exhaustion:
                cluster_exhaustion = flat_cluster_exhaustion

        if liquidation_context is None and not is_large_trade_event and has_liquidation_context_fields:
            flat_liquidation_context: dict[str, Any] = {}
            for key in (
                    "liquidation_side",
                    "liquidation_notional",
                    "context_strength",
                    "exhaustion_probability",
                    "continuation_probability",
                    "total_notional",
                    "reference_price",
                    "confidence",
                    "timestamp",
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "metadata",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    flat_liquidation_context[key] = value

            if flat_liquidation_context:
                liquidation_context = flat_liquidation_context

        set_aliases(
            "analysis",
            ("whales_analysis", "whale_analysis", "result"),
            analysis,
            override=False,
        )

        if section_present(pressure):
            set_aliases(
                "pressure",
                ("whale_pressure", "whale_pressure_signal"),
                pressure,
            )

        if section_present(activity):
            set_aliases(
                "activity",
                ("whale_activity", "whale_activity_signal"),
                activity,
            )

        if section_present(large_trade):
            set_aliases(
                "large_trade",
                ("large_trade_signal", "whale_large_trade"),
                large_trade,
            )

        if section_present(cluster):
            set_aliases(
                "cluster",
                ("whale_cluster", "whale_cluster_signal"),
                cluster,
            )

        if section_present(cluster_update):
            set_aliases(
                "cluster_update",
                ("whale_cluster_update", "whale_cluster_update_signal"),
                cluster_update,
            )

        if section_present(cluster_exhaustion):
            set_aliases(
                "cluster_exhaustion",
                ("whale_cluster_exhaustion", "whale_cluster_exhaustion_signal"),
                cluster_exhaustion,
            )

        if section_present(liquidation_context):
            set_aliases(
                "liquidation_context",
                (
                    "whale_liquidation_context",
                    "whale_liquidation_context_signal",
                ),
                liquidation_context,
            )

        if metadata is not None:
            set_aliases(
                "metadata",
                ("analytics_metadata", "meta"),
                metadata,
                override=False,
            )

        # Stable flat aliases used by WhaleCompositeSnapshot and whale strategies.
        for key, aliases in {
            "exchange": (),
            "market_type": (),
            "symbol": (),
            "exchange_symbol": (),
            "timeframe": (),
            "dominant_side": ("whale_side", "side"),
            "whale_side": ("dominant_side", "side"),
            "liquidation_side": (),
            "exhausted_side": (),
            "cluster_side": (),
            "imbalance_ratio": ("pressure_imbalance_ratio",),
            "pressure_score": (),
            "context_strength": (),
            "cluster_score": (),
            "continuation_probability": (),
            "exhaustion_probability": (),
            "total_notional": ("notional",),
            "liquidation_notional": (),
            "trade_count": (),
            "large_trade_notional": ("large_notional",),
            "large_trade_zscore": ("large_trade_z_score", "zscore", "z_score"),
            "reference_price": ("price", "mark_price"),
            "confidence": (),
            "timestamp": ("event_time", "time"),
        }.items():
            value = value_for(key, *aliases, default=None)
            if value is not None:
                domain_data.setdefault(key, value)

    def _augment_domain_data_contracts(
            self,
            *,
            source: FeatureSource,
            payload: dict[str, Any],
            domain_data: dict[str, Any],
    ) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._augment_domain_data_contracts")
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

        # Final topic-aware safety net for direct analytics events such as
        # analytics.<domain>.<section>.updated with payload["state"].
        # Domain-specific adapters above keep richer logic; this only ensures
        # canonical section aliases exist for routing/required_features.
        self._apply_direct_topic_section_alias(
            source=source,
            payload=payload,
            domain_data=domain_data,
        )

        self._ensure_common_domain_contract(
            source=source,
            payload=payload,
            domain_data=domain_data,
        )
        self._ensure_strategy_domain_contracts(
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._ensure_common_domain_contract")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._build_cross_domain_contracts")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._build_hybrid_domain_data")
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
        """
        Build hybrid summary feature snapshots from a normalized analytics-domain payload.

        Important:
        - source remains the trigger domain source, not FeatureSource.HYBRID.
        - This method must not pretend that a full multi-domain hybrid summary exists
          when only one analytics payload is available.
        - It always emits a domain-presence flag and, when possible, a domain vote.
        - It emits global hybrid.dominant_side/alignment/conflict/confluence only when
          the payload explicitly provides hybrid summary data or a multi-vote summary.
        - Full multi-domain aggregation should still be done by StrategyContextBuilder
          or a context-aware normalizer step that can see all current domain sections.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._build_hybrid_contract_features")
        result: list[FeatureSnapshot] = []

        confidence = payload.get("confidence", 0.0)

        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        source_value = getattr(source, "value", str(source))
        source_name = str(source_value).strip().lower()

        domain_name_by_source: dict[str, str] = {
            "orderflow": "orderflow",
            "liquidity": "liquidity",
            "liquidations": "liquidations",
            "whales": "whales",
            "open_interest": "open_interest",
            "funding": "funding",
            "price_action": "price_action",
            "spoofing": "spoofing",
            "spreads": "spreads",
        }
        trigger_domain = domain_name_by_source.get(source_name, source_name)

        def as_mapping(value: Any) -> dict[str, Any] | None:
            if isinstance(value, dict):
                return value

            to_payload = getattr(value, "to_payload", None)
            if callable(to_payload):
                converted = to_payload()
                if isinstance(converted, dict):
                    return converted

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

        def value_for(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
                if key in feature_map:
                    return feature_map[key]

            for container_key in (
                    "signal",
                    "setup",
                    "analysis",
                    "result",
                    "snapshot",
                    "features",
                    "contract",
                    trigger_domain,
                    f"hybrid.{trigger_domain}",
            ):
                container = payload.get(container_key)
                for key in keys:
                    value = get_item(container, key, None)
                    if value is not None:
                        return value

                container = feature_map.get(container_key)
                for key in keys:
                    value = get_item(container, key, None)
                    if value is not None:
                        return value

            return default

        def to_float(value: Any, default: float | None = None) -> float | None:
            if value is None:
                return default

            if isinstance(value, bool):
                return float(value)

            if isinstance(value, (int, float)):
                return float(value)

            raw = getattr(value, "value", value)

            if isinstance(raw, str):
                text = raw.strip()
                if not text:
                    return default
                try:
                    return float(text)
                except ValueError:
                    return default

            try:
                return float(raw)
            except (TypeError, ValueError):
                return default

        def clamp_unit(value: Any, default: float = 0.0) -> float:
            parsed = to_float(value, default)
            return max(0.0, min(1.0, float(parsed if parsed is not None else default)))

        def normalize_side(value: Any) -> str:
            raw = getattr(value, "value", value)

            if raw is None:
                return "unknown"

            text = str(raw).strip().lower()
            if text in {
                "long",
                "buy",
                "bid",
                "bull",
                "bullish",
                "up",
                "upside",
                "positive",
                "support",
                "demand",
                "accumulation",
                "short_squeeze",
                "sell_liquidations",
                "sell_exhaustion",
                "buy_absorption",
            }:
                return "long"

            if text in {
                "short",
                "sell",
                "ask",
                "bear",
                "bearish",
                "down",
                "downside",
                "negative",
                "resistance",
                "supply",
                "distribution",
                "long_squeeze",
                "buy_liquidations",
                "buy_exhaustion",
                "sell_absorption",
            }:
                return "short"

            return "unknown"

        def add(
                name: str,
                value: Any,
                *,
                section: str | None = None,
                confidence_override: Any = None,
        ) -> None:
            if value is None:
                return

            result.append(
                self._snapshot_from_raw_value(
                    source=source,
                    symbol=symbol,
                    name=name,
                    value=value,
                    timestamp=timestamp,
                    confidence=(
                        clamp_unit(confidence_override, clamp_unit(confidence, 0.0))
                        if confidence_override is not None
                        else clamp_unit(confidence, 0.0)
                    ),
                    metadata={
                        "origin": "contract_feature",
                        "contract": "hybrid",
                        "trigger_source": source_value,
                        **({"section": section} if section else {}),
                    },
                )
            )

        def add_if_present(
                name: str,
                *keys: str,
                section: str | None = None,
                default: Any = None,
        ) -> Any:
            value = value_for(*keys, default=default)
            if value is not None:
                add(name, value, section=section)
            return value

        # -------------------------------------------------------------------------
        # 1. Domain presence flag: safe to emit from every domain payload.
        # -------------------------------------------------------------------------
        if trigger_domain in {
            "orderflow",
            "liquidity",
            "liquidations",
            "whales",
            "open_interest",
            "funding",
            "price_action",
            "spoofing",
            "spreads",
        }:
            domain_payload = value_for(
                f"hybrid.{trigger_domain}",
                trigger_domain,
                f"hybrid_{trigger_domain}",
                "signal",
                "setup",
                "snapshot",
                "features",
                "result",
                "analysis",
                default=payload,
            )

            add(
                f"hybrid.{trigger_domain}",
                {
                    "available": True,
                    "source": source_value,
                    "payload": domain_payload,
                },
                section="domain",
            )

        # -------------------------------------------------------------------------
        # 2. Build a single domain vote from the trigger analytics signal/contract.
        #    This is safe. It does NOT mean global hybrid confluence exists yet.
        # -------------------------------------------------------------------------
        side = normalize_side(
            value_for(
                "hybrid.dominant_side",
                "dominant_side",
                "signal_side",
                "side",
                "bias",
                "direction",
                "whale_side",
                "liquidation_side",
                "funding_side",
                "oi_side",
                "orderflow_side",
                "trend_side",
                "sweep_side",
                "reversal_side",
                default=None,
            )
        )

        score = clamp_unit(
            value_for(
                "hybrid.confluence_score",
                "confluence_score",
                "signal_score",
                "score",
                "strength",
                "pressure_score",
                "confidence",
                default=0.0,
            )
        )
        vote_confidence = clamp_unit(
            value_for(
                "hybrid.confidence",
                "hybrid_confidence",
                "signal_confidence",
                "confidence",
                default=confidence,
            )
        )

        vote = None
        if side != "unknown":
            vote = {
                "source": trigger_domain,
                "feature_source": source_value,
                "side": side,
                "score": score,
                "confidence": vote_confidence,
                "timestamp": value_for(
                    "hybrid.timestamp",
                    "timestamp",
                    "event_time",
                    default=timestamp,
                ),
                "reason": value_for(
                    "reason",
                    "setup_type",
                    "signal_type",
                    "type",
                    default=f"{trigger_domain}_vote",
                ),
            }
            add("hybrid.votes", [vote], section="vote", confidence_override=vote_confidence)
            add(f"hybrid.{trigger_domain}.vote", vote, section="vote", confidence_override=vote_confidence)

        # -------------------------------------------------------------------------
        # 3. Only emit global hybrid summary if it is explicitly provided OR if the
        #    payload already contains a multi-domain vote summary.
        # -------------------------------------------------------------------------
        explicit_votes = value_for("hybrid.votes", "votes", default=None)
        explicit_hybrid_summary = any(
            key in payload or key in feature_map
            for key in (
                "hybrid.dominant_side",
                "dominant_side",
                "hybrid.alignment_score",
                "alignment_score",
                "hybrid.conflict_score",
                "conflict_score",
                "hybrid.confluence_score",
                "confluence_score",
                "hybrid.confidence",
                "hybrid_confidence",
            )
        )

        votes_list: list[Any] = []
        if isinstance(explicit_votes, list):
            votes_list = explicit_votes
        elif isinstance(explicit_votes, tuple):
            votes_list = list(explicit_votes)

        has_multi_vote_summary = len(votes_list) >= 2

        if explicit_hybrid_summary or has_multi_vote_summary:
            dominant_side = normalize_side(
                value_for(
                    "hybrid.dominant_side",
                    "dominant_side",
                    "side",
                    "bias",
                    "direction",
                    default=side,
                )
            )
            alignment_score = clamp_unit(
                value_for(
                    "hybrid.alignment_score",
                    "alignment_score",
                    "alignment",
                    default=None,
                ),
                default=1.0 if has_multi_vote_summary else 0.0,
            )
            conflict_score = clamp_unit(
                value_for(
                    "hybrid.conflict_score",
                    "conflict_score",
                    "conflict",
                    default=None,
                ),
                default=0.0,
            )
            confluence_score = clamp_unit(
                value_for(
                    "hybrid.confluence_score",
                    "confluence_score",
                    "confluence",
                    "score",
                    default=None,
                ),
                default=score,
            )
            hybrid_confidence = clamp_unit(
                value_for(
                    "hybrid.confidence",
                    "hybrid_confidence",
                    "confidence",
                    default=vote_confidence,
                )
            )

            if dominant_side != "unknown":
                add(
                    "hybrid.dominant_side",
                    dominant_side,
                    section="summary",
                    confidence_override=hybrid_confidence,
                )

            add(
                "hybrid.alignment_score",
                alignment_score,
                section="summary",
                confidence_override=hybrid_confidence,
            )
            add(
                "hybrid.conflict_score",
                conflict_score,
                section="summary",
                confidence_override=hybrid_confidence,
            )
            add(
                "hybrid.confluence_score",
                confluence_score,
                section="summary",
                confidence_override=hybrid_confidence,
            )
            add(
                "hybrid.confidence",
                hybrid_confidence,
                section="summary",
                confidence_override=hybrid_confidence,
            )

            if votes_list:
                add(
                    "hybrid.votes",
                    votes_list,
                    section="summary",
                    confidence_override=hybrid_confidence,
                )

        # -------------------------------------------------------------------------
        # 4. Scope features.
        # -------------------------------------------------------------------------
        add(
            "hybrid.symbol",
            value_for("hybrid.symbol", "symbol", default=symbol),
            section="scope",
        )
        add(
            "hybrid.exchange",
            value_for("hybrid.exchange", "exchange", default=None),
            section="scope",
        )
        add(
            "hybrid.market_type",
            value_for("hybrid.market_type", "market_type", default="usdm_futures"),
            section="scope",
        )
        add(
            "hybrid.timeframe",
            value_for("hybrid.timeframe", "timeframe", default=None),
            section="scope",
        )
        add(
            "hybrid.exchange_symbol",
            value_for("hybrid.exchange_symbol", "exchange_symbol", default=symbol),
            section="scope",
        )
        add(
            "hybrid.timestamp",
            value_for("hybrid.timestamp", "timestamp", "event_time", default=timestamp),
            section="scope",
        )

        return result

    def normalize_event(
        self,
        *,
        event_name: str,
        payload: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> NormalizedPayload:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer.normalize_event")
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

        # Build a second pass of contract features from the already-augmented
        # canonical domain contract. This makes routing robust when the raw
        # analytics event is a direct module event, e.g.
        # {topic=analytics.price_action.trend.updated, state=<TrendState>}.
        #
        # Explicit non-empty features stay authoritative, but empty/None
        # first-pass placeholders are replaced by canonical contract values.
        # This prevents feature_map dict transport bugs from blocking routing.
        feature_index = {snapshot.name: index for index, snapshot in enumerate(features)}
        augmented_contract_payload = {**payload_for_contract, **domain_data}
        for snapshot in self._build_contract_features(
                source=source,
                symbol=symbol,
                payload=augmented_contract_payload,
                timestamp=ts,
        ):
            index = feature_index.get(snapshot.name)
            if index is None:
                feature_index[snapshot.name] = len(features)
                features.append(snapshot)
            elif not self._contract_section_present(features[index].value):
                features[index] = snapshot

        for snapshot in self._build_strategy_contract_feature_snapshots(
                source=source,
                symbol=symbol,
                domain_data=domain_data,
                timestamp=ts,
        ):
            index = feature_index.get(snapshot.name)
            if index is None:
                feature_index[snapshot.name] = len(features)
                features.append(snapshot)
            elif not self._contract_section_present(features[index].value):
                features[index] = snapshot

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
                "payload_contract_level": domain_data.get("payload_contract_level", "snapshot"),
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer.apply_to_context")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer.normalize_and_apply")
        normalized = self.normalize_event(
            event_name=event_name,
            payload=payload,
            timestamp=timestamp,
        )
        return self.apply_to_context(context, normalized)

    def _resolve_source(self, event_name: str, payload: dict[str, Any]) -> FeatureSource:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._resolve_source")
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
        _strategy_logger = logging.getLogger(__name__ + ".SignalNormalizer._resolve_source_from_text")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._resolve_source_from_text")
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
        _strategy_logger = getattr(cls, "_logger", None) or logging.getLogger(__name__ + ".SignalNormalizer")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._resolve_source_from_payload")
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
        _strategy_logger = logging.getLogger(__name__ + ".SignalNormalizer._extract_symbol")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._extract_symbol")
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
        _strategy_logger = logging.getLogger(__name__ + ".SignalNormalizer._extract_timestamp")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._extract_timestamp")
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
        _strategy_logger = getattr(cls, "_logger", None) or logging.getLogger(__name__ + ".SignalNormalizer")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._extract_timeframe")
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
        _strategy_logger = getattr(cls, "_logger", None) or logging.getLogger(__name__ + ".SignalNormalizer")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._extract_domain_data")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._build_contract_features")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._build_whales_contract_features")
        result: list[FeatureSnapshot] = []

        confidence = payload.get("confidence", 0.0)
        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def as_mapping(value: Any) -> dict[str, Any] | None:
            if isinstance(value, dict):
                return value

            to_payload = getattr(value, "to_payload", None)
            if callable(to_payload):
                converted = to_payload()
                if isinstance(converted, dict):
                    return converted

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
                    pressure,
                    activity,
                    large_trade,
                    cluster,
                    cluster_update,
                    cluster_exhaustion,
                    liquidation_context,
                    metadata,
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
                    source=FeatureSource.WHALES,
                    symbol=symbol,
                    name=name,
                    value=value,
                    timestamp=timestamp,
                    confidence=confidence,
                    metadata={
                        "origin": "contract_feature",
                        "contract": "whales",
                        **({"section": section} if section else {}),
                    },
                )
            )

        analysis = mapping_or_object_for(
            "analysis",
            "whales_analysis",
            "whale_analysis",
            "result",
        )
        pressure = mapping_or_object_for(
            "pressure",
            "whale_pressure",
            "whale_pressure_signal",
        )
        activity = mapping_or_object_for(
            "activity",
            "whale_activity",
            "whale_activity_signal",
        )
        large_trade = mapping_or_object_for(
            "large_trade",
            "large_trade_signal",
            "whale_large_trade",
        )
        cluster = mapping_or_object_for(
            "cluster",
            "whale_cluster",
            "whale_cluster_signal",
        )
        cluster_update = mapping_or_object_for(
            "cluster_update",
            "whale_cluster_update",
            "whale_cluster_update_signal",
        )
        cluster_exhaustion = mapping_or_object_for(
            "cluster_exhaustion",
            "whale_cluster_exhaustion",
            "whale_cluster_exhaustion_signal",
        )
        liquidation_context = mapping_or_object_for(
            "liquidation_context",
            "whale_liquidation_context",
            "whale_liquidation_context_signal",
        )
        metadata = mapping_or_object_for(
            "metadata",
            "analytics_metadata",
            "meta",
        )

        for container_key in (
                "result",
                "payload",
                "data",
                "analysis",
                "whales_analysis",
                "whale_analysis",
                "event",
        ):
            container = mapping_or_object_for(container_key)
            container_map = as_mapping(container)
            if container_map is None:
                continue

            pressure = (
                    pressure
                    or container_map.get("pressure")
                    or container_map.get("whale_pressure")
                    or container_map.get("whale_pressure_signal")
            )
            activity = (
                    activity
                    or container_map.get("activity")
                    or container_map.get("whale_activity")
                    or container_map.get("whale_activity_signal")
            )
            large_trade = (
                    large_trade
                    or container_map.get("large_trade")
                    or container_map.get("large_trade_signal")
                    or container_map.get("whale_large_trade")
            )
            cluster = (
                    cluster
                    or container_map.get("cluster")
                    or container_map.get("whale_cluster")
                    or container_map.get("whale_cluster_signal")
            )
            cluster_update = (
                    cluster_update
                    or container_map.get("cluster_update")
                    or container_map.get("whale_cluster_update")
                    or container_map.get("whale_cluster_update_signal")
            )
            cluster_exhaustion = (
                    cluster_exhaustion
                    or container_map.get("cluster_exhaustion")
                    or container_map.get("whale_cluster_exhaustion")
                    or container_map.get("whale_cluster_exhaustion_signal")
            )
            liquidation_context = (
                    liquidation_context
                    or container_map.get("liquidation_context")
                    or container_map.get("whale_liquidation_context")
                    or container_map.get("whale_liquidation_context_signal")
            )
            metadata = (
                    metadata
                    or container_map.get("metadata")
                    or container_map.get("analytics_metadata")
                    or container_map.get("meta")
            )

        topic = self._topic_from_payload(payload)
        is_large_trade_event = "large_trade" in topic

        if pressure is None and not is_large_trade_event:
            flat_pressure = {
                key: value_for(key, default=None)
                for key in (
                    "dominant_side",
                    "whale_side",
                    "imbalance_ratio",
                    "pressure_score",
                    "total_notional",
                    "trade_count",
                    "reference_price",
                    "confidence",
                    "timestamp",
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "metadata",
                )
                if value_for(key, default=None) is not None
            }
            if flat_pressure:
                pressure = flat_pressure

        if activity is None:
            flat_activity = {
                key: value_for(key, default=None)
                for key in (
                    "activity_side",
                    "whale_side",
                    "dominant_side",
                    "total_notional",
                    "notional",
                    "trade_count",
                    "reference_price",
                    "confidence",
                    "timestamp",
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "metadata",
                )
                if value_for(key, default=None) is not None
            }
            if flat_activity:
                activity = flat_activity

        if large_trade is None:
            flat_large_trade = {
                key: value_for(key, default=None)
                for key in (
                    "large_trade_notional",
                    "large_trade_zscore",
                    "notional",
                    "zscore",
                    "z_score",
                    "whale_side",
                    "side",
                    "reference_price",
                    "confidence",
                    "timestamp",
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "metadata",
                )
                if value_for(key, default=None) is not None
            }
            if flat_large_trade:
                large_trade = flat_large_trade

        if cluster is None and not is_large_trade_event:
            flat_cluster = {
                key: value_for(key, default=None)
                for key in (
                    "cluster_side",
                    "whale_side",
                    "cluster_score",
                    "context_strength",
                    "continuation_probability",
                    "exhaustion_probability",
                    "total_notional",
                    "trade_count",
                    "reference_price",
                    "confidence",
                    "timestamp",
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "metadata",
                )
                if value_for(key, default=None) is not None
            }
            if flat_cluster:
                cluster = flat_cluster

        if cluster_update is None and not is_large_trade_event:
            flat_cluster_update = {
                key: value_for(key, default=None)
                for key in (
                    "cluster_update_side",
                    "cluster_side",
                    "cluster_score",
                    "context_strength",
                    "continuation_probability",
                    "exhaustion_probability",
                    "total_notional",
                    "trade_count",
                    "reference_price",
                    "confidence",
                    "timestamp",
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "metadata",
                )
                if value_for(key, default=None) is not None
            }
            if flat_cluster_update:
                cluster_update = flat_cluster_update

        if cluster_exhaustion is None and not is_large_trade_event:
            flat_cluster_exhaustion = {
                key: value_for(key, default=None)
                for key in (
                    "exhausted_side",
                    "cluster_side",
                    "exhaustion_probability",
                    "context_strength",
                    "cluster_score",
                    "total_notional",
                    "trade_count",
                    "reference_price",
                    "confidence",
                    "timestamp",
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "metadata",
                )
                if value_for(key, default=None) is not None
            }
            if flat_cluster_exhaustion:
                cluster_exhaustion = flat_cluster_exhaustion

        if liquidation_context is None and not is_large_trade_event:
            flat_liquidation_context = {
                key: value_for(key, default=None)
                for key in (
                    "liquidation_side",
                    "liquidation_notional",
                    "context_strength",
                    "exhaustion_probability",
                    "continuation_probability",
                    "total_notional",
                    "reference_price",
                    "confidence",
                    "timestamp",
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "metadata",
                )
                if value_for(key, default=None) is not None
            }
            if flat_liquidation_context:
                liquidation_context = flat_liquidation_context

        add("whales.analysis", analysis, section="analysis")

        if pressure is not None:
            add("whales.pressure", pressure, section="pressure")

        if activity is not None:
            add("whales.activity", activity, section="activity")

        if large_trade is not None:
            add("whales.large_trade", large_trade, section="large_trade")

        if cluster is not None:
            add("whales.cluster", cluster, section="cluster")

        if cluster_update is not None:
            add("whales.cluster_update", cluster_update, section="cluster_update")

        if cluster_exhaustion is not None:
            add(
                "whales.cluster_exhaustion",
                cluster_exhaustion,
                section="cluster_exhaustion",
            )

        if liquidation_context is not None:
            add(
                "whales.liquidation_context",
                liquidation_context,
                section="liquidation_context",
            )

        add(
            "whales.symbol",
            value_for("symbol", default=symbol),
            section="scope",
        )
        add(
            "whales.exchange",
            value_for("exchange", default=None),
            section="scope",
        )
        add(
            "whales.market_type",
            value_for("market_type", default=None),
            section="scope",
        )
        add(
            "whales.timeframe",
            value_for("timeframe", default=None),
            section="scope",
        )
        add(
            "whales.exchange_symbol",
            value_for("exchange_symbol", default=None),
            section="scope",
        )

        add(
            "whales.dominant_side",
            value_for("dominant_side", "whale_side", "side", default=None),
            section="side",
        )
        add(
            "whales.whale_side",
            value_for("whale_side", "dominant_side", "side", default=None),
            section="side",
        )
        add(
            "whales.liquidation_side",
            value_for("liquidation_side", default=None),
            section="side",
        )
        add(
            "whales.exhausted_side",
            value_for("exhausted_side", default=None),
            section="side",
        )
        add(
            "whales.cluster_side",
            value_for("cluster_side", default=None),
            section="side",
        )

        add(
            "whales.imbalance_ratio",
            value_for("imbalance_ratio", "pressure_imbalance_ratio", default=None),
            section="pressure",
        )
        add(
            "whales.pressure_score",
            value_for("pressure_score", default=None),
            section="pressure",
        )
        add(
            "whales.context_strength",
            value_for("context_strength", default=None),
            section="context",
        )
        add(
            "whales.cluster_score",
            value_for("cluster_score", default=None),
            section="cluster",
        )
        add(
            "whales.continuation_probability",
            value_for("continuation_probability", default=None),
            section="context",
        )
        add(
            "whales.exhaustion_probability",
            value_for("exhaustion_probability", default=None),
            section="context",
        )

        add(
            "whales.total_notional",
            value_for("total_notional", "notional", default=None),
            section="notional",
        )
        add(
            "whales.liquidation_notional",
            value_for("liquidation_notional", default=None),
            section="liquidation_context",
        )
        add(
            "whales.trade_count",
            value_for("trade_count", default=None),
            section="activity",
        )
        add(
            "whales.large_trade_notional",
            value_for("large_trade_notional", "large_notional", default=None),
            section="large_trade",
        )
        add(
            "whales.large_trade_zscore",
            value_for(
                "large_trade_zscore",
                "large_trade_z_score",
                "zscore",
                "z_score",
                default=None,
            ),
            section="large_trade",
        )

        add(
            "whales.reference_price",
            value_for("reference_price", "price", "mark_price", default=None),
            section="price",
        )
        add(
            "whales.confidence",
            value_for("confidence", default=None),
            section="confidence",
        )
        add(
            "whales.timestamp",
            value_for("timestamp", "event_time", "time", default=None),
            section="timestamp",
        )
        add(
            "whales.metadata",
            metadata,
            section="metadata",
        )

        return result

    def _build_spreads_contract_features(
            self,
            *,
            symbol: str,
            payload: dict[str, Any],
            timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._build_spreads_contract_features")
        result: list[FeatureSnapshot] = []

        confidence = payload.get("confidence", 0.0)
        feature_map = payload.get("feature_map")
        if not isinstance(feature_map, dict):
            feature_map = {}

        def as_mapping(value: Any) -> dict[str, Any] | None:
            if isinstance(value, dict):
                return value

            to_payload = getattr(value, "to_payload", None)
            if callable(to_payload):
                converted = to_payload()
                if isinstance(converted, dict):
                    return converted

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

            for container in (snapshot, signal, opportunity):
                value = nested_value(container, *keys, default=None)
                if value is not None:
                    return value

            return default

        def add(name: str, value: Any, *, section: str | None = None) -> None:
            if value is None:
                return

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
                        **({"section": section} if section else {}),
                    },
                )
            )

        snapshot = mapping_or_object_for(
            "snapshot",
            "spread_snapshot",
            "spreads_snapshot",
            "spot_futures_snapshot",
            "cross_exchange_snapshot",
            "basis_snapshot",
        )
        signal = mapping_or_object_for(
            "signal",
            "spread_signal",
            "spreads_signal",
            "analytics_signal",
            "setup",
        )
        opportunity = mapping_or_object_for(
            "opportunity",
            "arbitrage_opportunity",
            "arb_opportunity",
            "cross_exchange_opportunity",
        )

        for container_key in (
                "result",
                "payload",
                "data",
                "analysis",
                "spreads_analysis",
                "spread_analysis",
                "event",
        ):
            container = mapping_or_object_for(container_key)
            container_map = as_mapping(container)
            if container_map is None:
                continue

            snapshot = (
                    snapshot
                    or container_map.get("snapshot")
                    or container_map.get("spread_snapshot")
                    or container_map.get("spreads_snapshot")
                    or container_map.get("spot_futures_snapshot")
                    or container_map.get("cross_exchange_snapshot")
                    or container_map.get("basis_snapshot")
            )
            signal = (
                    signal
                    or container_map.get("signal")
                    or container_map.get("spread_signal")
                    or container_map.get("spreads_signal")
                    or container_map.get("analytics_signal")
                    or container_map.get("setup")
            )
            opportunity = (
                    opportunity
                    or container_map.get("opportunity")
                    or container_map.get("arbitrage_opportunity")
                    or container_map.get("arb_opportunity")
                    or container_map.get("cross_exchange_opportunity")
            )

        if snapshot is None:
            flat_snapshot: dict[str, Any] = {}
            for key in (
                    "spread_type",
                    "type",
                    "symbol",
                    "exchange_a",
                    "exchange_b",
                    "market_type_a",
                    "market_type_b",
                    "exchange_symbol_a",
                    "exchange_symbol_b",
                    "spot_exchange",
                    "futures_exchange",
                    "spot_symbol",
                    "futures_symbol",
                    "instrument_type",
                    "leg_a",
                    "leg_b",
                    "spread_bps",
                    "basis",
                    "funding_adjusted_spread",
                    "net_edge",
                    "net_edge_bps",
                    "zscore",
                    "regime",
                    "direction",
                    "signal_type",
                    "quote_validity",
                    "has_edge",
                    "confidence",
                    "timestamp",
                    "timeframe",
                    "metadata",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    flat_snapshot[key] = value

            if flat_snapshot:
                snapshot = flat_snapshot

        add("spreads.snapshot", snapshot, section="snapshot")

        add(
            "spreads.type",
            value_for("spread_type", "type", default=None),
            section="snapshot",
        )
        add(
            "spreads.symbol",
            value_for("symbol", default=symbol),
            section="snapshot",
        )
        add(
            "spreads.exchange_a",
            value_for("exchange_a", "spot_exchange", "buy_exchange", default=None),
            section="snapshot",
        )
        add(
            "spreads.exchange_b",
            value_for("exchange_b", "futures_exchange", "sell_exchange", default=None),
            section="snapshot",
        )
        add(
            "spreads.market_type_a",
            value_for("market_type_a", "spot_market_type", "buy_market_type", default=None),
            section="snapshot",
        )
        add(
            "spreads.market_type_b",
            value_for(
                "market_type_b",
                "futures_market_type",
                "sell_market_type",
                default=None,
            ),
            section="snapshot",
        )
        add(
            "spreads.exchange_symbol_a",
            value_for("exchange_symbol_a", "spot_symbol", "buy_symbol", default=None),
            section="snapshot",
        )
        add(
            "spreads.exchange_symbol_b",
            value_for("exchange_symbol_b", "futures_symbol", "sell_symbol", default=None),
            section="snapshot",
        )

        add(
            "spreads.spread_bps",
            value_for("spread_bps", "basis_bps", default=None),
            section="snapshot",
        )
        add(
            "spreads.basis",
            value_for("basis", "basis_value", default=None),
            section="snapshot",
        )
        add(
            "spreads.funding_adjusted_spread",
            value_for(
                "funding_adjusted_spread",
                "funding_adjusted_basis",
                "funding_edge",
                default=None,
            ),
            section="snapshot",
        )
        add(
            "spreads.net_edge",
            value_for("net_edge", "edge", default=None),
            section="snapshot",
        )
        add(
            "spreads.net_edge_bps",
            value_for("net_edge_bps", "edge_bps", default=None),
            section="snapshot",
        )
        add(
            "spreads.zscore",
            value_for("zscore", "z_score", default=None),
            section="snapshot",
        )

        add(
            "spreads.regime",
            value_for("regime", "spread_regime", default=None),
            section="snapshot",
        )
        add(
            "spreads.direction",
            value_for("direction", "spread_direction", default=None),
            section="snapshot",
        )
        add(
            "spreads.signal_type",
            value_for("signal_type", "spread_signal_type", default=None),
            section="snapshot",
        )
        add(
            "spreads.quote_validity",
            value_for("quote_validity", "quote_status", default=None),
            section="snapshot",
        )
        add(
            "spreads.has_edge",
            value_for("has_edge", "tradeable_edge", default=None),
            section="snapshot",
        )
        add(
            "spreads.confidence",
            value_for("confidence", "score_confidence", default=None),
            section="snapshot",
        )

        add(
            "spreads.leg_a.instrument_type",
            nested_value(
                value_for("leg_a", default=None),
                "instrument_type",
                default=value_for("instrument_type_a", default=None),
            ),
            section="snapshot",
        )
        add(
            "spreads.leg_b.instrument_type",
            nested_value(
                value_for("leg_b", default=None),
                "instrument_type",
                default=value_for("instrument_type_b", default=None),
            ),
            section="snapshot",
        )
        add(
            "spreads.instrument_type",
            value_for("instrument_type", default=None),
            section="snapshot",
        )

        if signal is not None:
            add("spreads.signal", signal, section="signal")
            add(
                "spreads.signal_type",
                nested_value(
                    signal,
                    "signal_type",
                    "type",
                    "setup_type",
                    default=value_for("signal_type", "setup_type", default=None),
                ),
                section="signal",
            )
            add(
                "spreads.direction",
                nested_value(
                    signal,
                    "direction",
                    "spread_direction",
                    "side",
                    default=value_for("direction", "side", default=None),
                ),
                section="signal",
            )
            add(
                "spreads.confidence",
                nested_value(
                    signal,
                    "confidence",
                    default=value_for("signal_confidence", "confidence", default=None),
                ),
                section="signal",
            )

        if opportunity is not None:
            add("spreads.opportunity", opportunity, section="opportunity")
            add(
                "spreads.opportunity_key",
                nested_value(
                    opportunity,
                    "opportunity_key",
                    "key",
                    default=value_for("opportunity_key", default=None),
                ),
                section="opportunity",
            )
            add(
                "spreads.opportunity_status",
                nested_value(
                    opportunity,
                    "status",
                    "opportunity_status",
                    default=value_for("opportunity_status", "status", default=None),
                ),
                section="opportunity",
            )
            add(
                "spreads.persistence_ms",
                nested_value(
                    opportunity,
                    "persistence_ms",
                    default=value_for("persistence_ms", default=None),
                ),
                section="opportunity",
            )
            add(
                "spreads.buy_exchange",
                nested_value(
                    opportunity,
                    "buy_exchange",
                    default=value_for("buy_exchange", default=None),
                ),
                section="opportunity",
            )
            add(
                "spreads.sell_exchange",
                nested_value(
                    opportunity,
                    "sell_exchange",
                    default=value_for("sell_exchange", default=None),
                ),
                section="opportunity",
            )
            add(
                "spreads.buy_market_type",
                nested_value(
                    opportunity,
                    "buy_market_type",
                    default=value_for("buy_market_type", default=None),
                ),
                section="opportunity",
            )
            add(
                "spreads.sell_market_type",
                nested_value(
                    opportunity,
                    "sell_market_type",
                    default=value_for("sell_market_type", default=None),
                ),
                section="opportunity",
            )
            add(
                "spreads.net_edge",
                nested_value(
                    opportunity,
                    "net_edge",
                    "edge",
                    default=value_for("net_edge", "edge", default=None),
                ),
                section="opportunity",
            )
            add(
                "spreads.net_edge_bps",
                nested_value(
                    opportunity,
                    "net_edge_bps",
                    "edge_bps",
                    default=value_for("net_edge_bps", "edge_bps", default=None),
                ),
                section="opportunity",
            )

        add(
            "spreads.metadata",
            value_for("metadata", default=None),
            section="metadata",
        )

        return result

    def _build_spoofing_contract_features(
            self,
            *,
            symbol: str,
            payload: dict[str, Any],
            timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._build_spoofing_contract_features")
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
                    signal,
                    features,
                    detector_results,
                    score_breakdown,
                    analytics_metadata,
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
                    source=FeatureSource.SPOOFING,
                    symbol=symbol,
                    name=name,
                    value=value,
                    timestamp=timestamp,
                    confidence=confidence,
                    metadata={
                        "origin": "contract_feature",
                        "contract": "spoofing",
                        **({"section": section} if section else {}),
                    },
                )
            )

        composite = mapping_or_object_for(
            "composite",
            "snapshot",
            "spoofing",
            "spoofing_snapshot",
            "composite_snapshot",
        )
        signal = mapping_or_object_for(
            "signal",
            "spoofing_signal",
            "analytics_signal",
            "setup",
        )
        features = mapping_or_object_for(
            "features",
            "spoofing_features",
        )
        detector_results = mapping_or_object_for(
            "detector_results",
            "detectors",
            "detector_state",
            "components",
        )
        score_breakdown = mapping_or_object_for(
            "score_breakdown",
            "scores",
            "scoring",
        )
        analytics_metadata = mapping_or_object_for(
            "analytics_metadata",
            "metadata",
            "meta",
        )

        for container_key in (
                "result",
                "payload",
                "data",
                "analysis",
                "spoofing_analysis",
                "event",
                "spoofing",
                "composite",
        ):
            container = mapping_or_object_for(container_key)
            container_map = as_mapping(container)
            if container_map is None:
                continue

            composite = (
                    composite
                    or container_map.get("composite")
                    or container_map.get("snapshot")
                    or container_map.get("spoofing")
                    or container_map.get("spoofing_snapshot")
                    or container_map.get("composite_snapshot")
            )
            signal = (
                    signal
                    or container_map.get("signal")
                    or container_map.get("spoofing_signal")
                    or container_map.get("analytics_signal")
                    or container_map.get("setup")
            )
            features = (
                    features
                    or container_map.get("features")
                    or container_map.get("spoofing_features")
            )
            detector_results = (
                    detector_results
                    or container_map.get("detector_results")
                    or container_map.get("detectors")
                    or container_map.get("detector_state")
                    or container_map.get("components")
            )
            score_breakdown = (
                    score_breakdown
                    or container_map.get("score_breakdown")
                    or container_map.get("scores")
                    or container_map.get("scoring")
            )
            analytics_metadata = (
                    analytics_metadata
                    or container_map.get("analytics_metadata")
                    or container_map.get("metadata")
                    or container_map.get("meta")
            )

        if signal is not None:
            features = features or nested_value(signal, "features", default=None)
            detector_results = detector_results or nested_value(
                signal,
                "detector_results",
                "detectors",
                "components",
                default=None,
            )
            score_breakdown = score_breakdown or nested_value(
                signal,
                "score_breakdown",
                "scores",
                default=None,
            )
            analytics_metadata = analytics_metadata or nested_value(
                signal,
                "metadata",
                "analytics_metadata",
                default=None,
            )

        if features is None:
            flat_features: dict[str, Any] = {}
            for key in (
                    "pull_ratio",
                    "fill_ratio",
                    "price_reaction_bps",
                    "signed_price_reaction_bps",
                    "lifetime_ms",
                    "wall_notional",
                    "pulled_notional",
                    "cancel_to_fill_ratio",
                    "distance_from_mid_bps",
                    "layer_count",
                    "layer_price_span_bps",
                    "pressure_flip_strength",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    flat_features[key] = value

            if flat_features:
                features = flat_features

        if composite is None and any(
                section is not None
                for section in (
                        signal,
                        features,
                        detector_results,
                        score_breakdown,
                        analytics_metadata,
                )
        ):
            composite = {
                "signal": signal or {},
                "features": features or {},
                "detector_results": detector_results or {},
                "score_breakdown": score_breakdown or {},
                "analytics_metadata": analytics_metadata or {},
            }

            for key in (
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "timestamp",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    composite[key] = value

        if composite is not None:
            add("spoofing.composite", composite, section="composite")

        if signal is not None:
            add("spoofing.signal", signal, section="signal")

        if features is not None:
            add("spoofing.features", features, section="features")

        if detector_results is not None:
            add("spoofing.detector_results", detector_results, section="detector_results")

        if score_breakdown is not None:
            add("spoofing.score_breakdown", score_breakdown, section="score_breakdown")

        if analytics_metadata is not None:
            add("spoofing.analytics_metadata", analytics_metadata, section="metadata")

        add(
            "spoofing.type",
            value_for("spoofing_type", "type", "signal_type", "setup_type", default=None),
            section="signal",
        )
        add(
            "spoofing.pattern",
            value_for("pattern", "spoofing_pattern", default=None),
            section="signal",
        )
        add(
            "spoofing.side",
            value_for("side", "spoofing_side", "signal_side", "direction", default=None),
            section="signal",
        )
        add(
            "spoofing.severity",
            value_for("severity", "spoofing_severity", default=None),
            section="signal",
        )
        add(
            "spoofing.status",
            value_for("status", "spoofing_status", default=None),
            section="signal",
        )
        add(
            "spoofing.score",
            value_for("score", "signal_score", default=None),
            section="signal",
        )
        add(
            "spoofing.confidence",
            value_for("confidence", "signal_confidence", default=None),
            section="signal",
        )
        add(
            "spoofing.price_level",
            value_for("price_level", "level", "wall_price", default=None),
            section="signal",
        )
        add(
            "spoofing.wall_id",
            value_for("wall_id", default=None),
            section="signal",
        )
        add(
            "spoofing.event_time",
            value_for("event_time", "timestamp", "time", default=None),
            section="signal",
        )

        add(
            "spoofing.features.pull_ratio",
            value_for("pull_ratio", default=None),
            section="features",
        )
        add(
            "spoofing.features.fill_ratio",
            value_for("fill_ratio", default=None),
            section="features",
        )
        add(
            "spoofing.features.price_reaction_bps",
            value_for("price_reaction_bps", default=None),
            section="features",
        )
        add(
            "spoofing.features.signed_price_reaction_bps",
            value_for("signed_price_reaction_bps", default=None),
            section="features",
        )
        add(
            "spoofing.features.lifetime_ms",
            value_for("lifetime_ms", default=None),
            section="features",
        )
        add(
            "spoofing.features.wall_notional",
            value_for("wall_notional", default=None),
            section="features",
        )
        add(
            "spoofing.features.pulled_notional",
            value_for("pulled_notional", default=None),
            section="features",
        )
        add(
            "spoofing.features.cancel_to_fill_ratio",
            value_for("cancel_to_fill_ratio", default=None),
            section="features",
        )
        add(
            "spoofing.features.distance_from_mid_bps",
            value_for("distance_from_mid_bps", default=None),
            section="features",
        )
        add(
            "spoofing.features.layer_count",
            value_for("layer_count", default=None),
            section="features",
        )
        add(
            "spoofing.features.layer_price_span_bps",
            value_for("layer_price_span_bps", default=None),
            section="features",
        )
        add(
            "spoofing.features.pressure_flip_strength",
            value_for("pressure_flip_strength", default=None),
            section="features",
        )

        return result

    def _build_price_action_contract_features(
            self,
            *,
            symbol: str,
            payload: dict[str, Any],
            timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._build_price_action_contract_features")
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
                    market_structure,
                    support_resistance,
                    fair_value_gap,
                    trend,
                    liquidity_levels,
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

        composite = mapping_or_object_for(
            "composite",
            "state",
            "snapshot",
            "price_action",
            "price_action_state",
        )
        market_structure = mapping_or_object_for(
            "market_structure",
            "structure",
            "market_structure_state",
        )
        support_resistance = mapping_or_object_for(
            "support_resistance",
            "sr",
            "support_resistance_state",
        )
        fair_value_gap = mapping_or_object_for(
            "fair_value_gap",
            "fvg",
            "fair_value_gap_state",
            "fvg_state",
        )
        trend = mapping_or_object_for(
            "trend",
            "trend_state",
        )
        liquidity_levels = mapping_or_object_for(
            "liquidity_levels",
            "liquidity",
            "levels",
        )
        signal = mapping_or_object_for(
            "signal",
            "price_action_signal",
            "analytics_signal",
            "setup",
        )

        direct_section = self._direct_topic_section(FeatureSource.PRICE_ACTION, payload)
        direct_value = self._direct_payload_value(payload, feature_map=feature_map)
        if direct_section and direct_value is not None:
            if direct_section == "market_structure" and market_structure is None:
                market_structure = direct_value
                if composite is direct_value:
                    composite = None
            elif direct_section == "support_resistance" and support_resistance is None:
                support_resistance = direct_value
                if composite is direct_value:
                    composite = None
            elif direct_section == "fair_value_gap" and fair_value_gap is None:
                fair_value_gap = direct_value
                if composite is direct_value:
                    composite = None
            elif direct_section == "trend" and trend is None:
                trend = direct_value
                if composite is direct_value:
                    composite = None
            elif direct_section == "liquidity_levels" and liquidity_levels is None:
                liquidity_levels = direct_value
                if composite is direct_value:
                    composite = None
            elif direct_section == "composite" and composite is None:
                composite = direct_value

        for container_key in (
                "result",
                "payload",
                "data",
                "analysis",
                "price_action_analysis",
                "state",
                "price_action",
                "composite",
        ):
            container = mapping_or_object_for(container_key)
            container_map = as_mapping(container)
            if container_map is None:
                continue

            composite = composite or container_map.get("composite") or container_map.get(
                "state"
            )

            market_structure = (
                    market_structure
                    or container_map.get("market_structure")
                    or container_map.get("structure")
                    or container_map.get("market_structure_state")
            )
            support_resistance = (
                    support_resistance
                    or container_map.get("support_resistance")
                    or container_map.get("sr")
                    or container_map.get("support_resistance_state")
            )
            fair_value_gap = (
                    fair_value_gap
                    or container_map.get("fair_value_gap")
                    or container_map.get("fvg")
                    or container_map.get("fair_value_gap_state")
                    or container_map.get("fvg_state")
            )
            trend = (
                    trend
                    or container_map.get("trend")
                    or container_map.get("trend_state")
            )
            liquidity_levels = (
                    liquidity_levels
                    or container_map.get("liquidity_levels")
                    or container_map.get("liquidity")
                    or container_map.get("levels")
            )
            signal = (
                    signal
                    or container_map.get("signal")
                    or container_map.get("price_action_signal")
                    or container_map.get("analytics_signal")
                    or container_map.get("setup")
            )

        if composite is not None:
            market_structure = market_structure or nested_value(
                composite,
                "market_structure",
                "structure",
                default=None,
            )
            support_resistance = support_resistance or nested_value(
                composite,
                "support_resistance",
                "sr",
                default=None,
            )
            fair_value_gap = fair_value_gap or nested_value(
                composite,
                "fair_value_gap",
                "fvg",
                default=None,
            )
            trend = trend or nested_value(
                composite,
                "trend",
                default=None,
            )
            liquidity_levels = liquidity_levels or nested_value(
                composite,
                "liquidity_levels",
                "liquidity",
                default=None,
            )

        if market_structure is not None:
            market_structure = self._ensure_price_action_market_structure_view(
                module=self._as_mapping_or_none(market_structure) or {"value": market_structure},
                event_payload=payload,
                fallback_topic=self._topic_from_payload(payload),
            )
        if trend is not None:
            trend = self._ensure_price_action_trend_view(
                module=self._as_mapping_or_none(trend) or {"value": trend},
                event_payload=payload,
                fallback_topic=self._topic_from_payload(payload),
            )
        if support_resistance is not None:
            support_resistance = self._wrap_price_action_module_view(
                module=support_resistance,
                event_payload=payload,
                fallback_topic=self._topic_from_payload(payload),
                section="support_resistance",
            )
        if fair_value_gap is not None:
            fair_value_gap = self._wrap_price_action_module_view(
                module=fair_value_gap,
                event_payload=payload,
                fallback_topic=self._topic_from_payload(payload),
                section="fair_value_gap",
            )
        if liquidity_levels is not None:
            liquidity_levels = self._wrap_price_action_module_view(
                module=liquidity_levels,
                event_payload=payload,
                fallback_topic=self._topic_from_payload(payload),
                section="liquidity_levels",
            )

        if composite is None and any(
                section is not None
                for section in (
                        market_structure,
                        support_resistance,
                        fair_value_gap,
                        trend,
                        liquidity_levels,
                )
        ):
            composite = {
                "market_structure": market_structure or {},
                "support_resistance": support_resistance or {},
                "fair_value_gap": fair_value_gap or {},
                "trend": trend or {},
                "liquidity_levels": liquidity_levels or {},
            }

            for key in (
                    "exchange",
                    "market_type",
                    "symbol",
                    "exchange_symbol",
                    "timeframe",
                    "timestamp",
                    "current_price",
                    "last_price",
                    "price",
            ):
                value = value_for(key, default=None)
                if value is not None:
                    composite[key] = value

        if composite is not None:
            add("price_action.composite", composite, section="composite")

        current_price = value_for(
            "current_price",
            "last_price",
            "price",
            default=None,
        )
        add("price_action.current_price", current_price, section="composite")
        add("price_action.last_price", value_for("last_price", "price", default=None), section="composite")
        add("price_action.timestamp", value_for("timestamp", default=None), section="composite")

        if market_structure is not None:
            add(
                "price_action.market_structure",
                market_structure,
                section="market_structure",
            )
            add(
                "price_action.market_structure.internal",
                nested_value(market_structure, "internal", default=None),
                section="market_structure",
            )
            add(
                "price_action.market_structure.external",
                nested_value(market_structure, "external", default=None),
                section="market_structure",
            )
            add(
                "price_action.market_structure.last_break_event",
                nested_value(
                    market_structure,
                    "last_break_event",
                    "last_event",
                    "break_event",
                    default=None,
                ),
                section="market_structure",
            )
            add(
                "price_action.market_structure.mtf_alignment",
                nested_value(
                    market_structure,
                    "mtf_alignment",
                    "mtf_alignment_score",
                    "alignment_score",
                    default=None,
                ),
                section="market_structure",
            )

        if support_resistance is not None:
            add(
                "price_action.support_resistance",
                support_resistance,
                section="support_resistance",
            )
            add(
                "price_action.support_resistance.internal",
                nested_value(support_resistance, "internal", default=None),
                section="support_resistance",
            )
            add(
                "price_action.support_resistance.external",
                nested_value(support_resistance, "external", default=None),
                section="support_resistance",
            )
            add(
                "price_action.support_resistance.last_event",
                nested_value(
                    support_resistance,
                    "last_event",
                    "event",
                    default=None,
                ),
                section="support_resistance",
            )
            add(
                "price_action.support_resistance.nearest_support",
                nested_value(
                    support_resistance,
                    "nearest_support",
                    "support",
                    default=None,
                ),
                section="support_resistance",
            )
            add(
                "price_action.support_resistance.nearest_resistance",
                nested_value(
                    support_resistance,
                    "nearest_resistance",
                    "resistance",
                    default=None,
                ),
                section="support_resistance",
            )

        if fair_value_gap is not None:
            add(
                "price_action.fair_value_gap",
                fair_value_gap,
                section="fair_value_gap",
            )
            add("price_action.fvg", fair_value_gap, section="fair_value_gap")
            add(
                "price_action.fair_value_gap.internal",
                nested_value(fair_value_gap, "internal", default=None),
                section="fair_value_gap",
            )
            add(
                "price_action.fair_value_gap.external",
                nested_value(fair_value_gap, "external", default=None),
                section="fair_value_gap",
            )
            add(
                "price_action.fair_value_gap.last_event",
                nested_value(
                    fair_value_gap,
                    "last_event",
                    "event",
                    default=None,
                ),
                section="fair_value_gap",
            )
            add(
                "price_action.fair_value_gap.nearest_bullish_gap",
                nested_value(
                    fair_value_gap,
                    "nearest_bullish_gap",
                    "bullish_gap",
                    default=None,
                ),
                section="fair_value_gap",
            )
            add(
                "price_action.fair_value_gap.nearest_bearish_gap",
                nested_value(
                    fair_value_gap,
                    "nearest_bearish_gap",
                    "bearish_gap",
                    default=None,
                ),
                section="fair_value_gap",
            )

        if trend is not None:
            add("price_action.trend", trend, section="trend")
            add(
                "price_action.trend.internal",
                nested_value(trend, "internal", default=None),
                section="trend",
            )
            add(
                "price_action.trend.external",
                nested_value(trend, "external", default=None),
                section="trend",
            )
            add(
                "price_action.trend.last_signal",
                nested_value(
                    trend,
                    "last_signal",
                    "last_event",
                    "signal",
                    default=None,
                ),
                section="trend",
            )
            add(
                "price_action.trend.internal_external_alignment",
                nested_value(
                    trend,
                    "internal_external_alignment",
                    "alignment",
                    "alignment_score",
                    default=None,
                ),
                section="trend",
            )
            add(
                "price_action.trend.higher_timeframe_alignment",
                nested_value(
                    trend,
                    "higher_timeframe_alignment",
                    "htf_alignment",
                    default=None,
                ),
                section="trend",
            )
            add(
                "price_action.trend.overall_trend_score",
                nested_value(
                    trend,
                    "overall_trend_score",
                    "overall_score",
                    "trend_score",
                    default=None,
                ),
                section="trend",
            )

        if liquidity_levels is not None:
            add(
                "price_action.liquidity_levels",
                liquidity_levels,
                section="liquidity_levels",
            )

        if signal is not None:
            add("price_action.signal", signal, section="signal")
            add(
                "price_action.signal.type",
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
                "price_action.signal.score",
                nested_value(
                    signal,
                    "score",
                    default=value_for("signal_score", "score", default=None),
                ),
                section="signal",
            )
            add(
                "price_action.signal.confidence",
                nested_value(
                    signal,
                    "confidence",
                    default=value_for("signal_confidence", "confidence", default=None),
                ),
                section="signal",
            )
            add(
                "price_action.signal.side",
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

    def _build_liquidity_contract_features(
            self,
            *,
            symbol: str,
            payload: dict[str, Any],
            timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._build_liquidity_contract_features")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._build_liquidations_contract_features")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._build_orderflow_contract_features")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._build_open_interest_contract_features")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._build_funding_contract_features")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._extract_features")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._extract_explicit_features")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._extract_feature_map")
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
                # Only treat mappings with explicit FeatureSnapshot fields as
                # snapshot descriptors.  Domain contract mappings are the value.
                descriptor_keys = {
                    "value",
                    "normalized",
                    "normalized_value",
                    "freshness_seconds",
                    "metadata",
                }
                if descriptor_keys & set(value):
                    item = {"name": name, **value}
                else:
                    item = {"name": name, "value": value}
                    if "confidence" in value:
                        item["confidence"] = value.get("confidence")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._extract_nested_features")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._build_implicit_features")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._snapshot_from_feature_item")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise SignalNormalizationError("feature item must contain non-empty 'name'")

        confidence = self._safe_confidence(item.get("confidence", default_confidence))
        normalized_value = self._safe_normalized_value(
            item.get("normalized_value", item.get("normalized"))
        )

        # A strategy contract feature may legitimately be a mapping, e.g.
        # feature_map["liquidity.snapshot"] = { ...snapshot fields... }.  The
        # old code interpreted that mapping as FeatureSnapshot metadata and lost
        # the actual value unless it contained an explicit "value" key.  That
        # made context.has_feature(...) true but registry truthiness false
        # because snapshot.value became None, which rejected liquidity/whales/
        # orderflow/spreads/spoofing/liquidations payloads before strategies
        # could evaluate them.  Preserve mapping payloads as the feature value.
        if "value" in item:
            value = item.get("value")
        else:
            reserved_keys = {
                "name",
                "confidence",
                "normalized",
                "normalized_value",
                "freshness_seconds",
                "metadata",
            }
            raw_value = {
                key: value
                for key, value in item.items()
                if key not in reserved_keys
            }
            value = raw_value if raw_value else None

        snapshot = FeatureSnapshot(
            name=self._normalize_feature_name(name),
            value=value,
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._snapshot_from_raw_value")
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
        _strategy_logger = getattr(cls, "_logger", None) or logging.getLogger(__name__ + ".SignalNormalizer")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._feature_name_candidates")
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
        _strategy_logger = getattr(cls, "_logger", None) or logging.getLogger(__name__ + ".SignalNormalizer")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._flatten_scalar_dict")
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
        _strategy_logger = logging.getLogger(__name__ + ".SignalNormalizer._is_feature_scalar")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._is_feature_scalar")
        return isinstance(value, (int, float, bool, str)) and value is not None

    @staticmethod
    def _normalize_feature_name(value: str) -> str:
        _strategy_logger = logging.getLogger(__name__ + ".SignalNormalizer._normalize_feature_name")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._normalize_feature_name")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._resolve_freshness_seconds")
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
        _strategy_logger = logging.getLogger(__name__ + ".SignalNormalizer._safe_confidence")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._safe_confidence")
        parsed = _to_float(value, default=0.0)
        return clamp(parsed if parsed is not None else 0.0, 0.0, 1.0)

    @staticmethod
    def _safe_normalized_value(value: Any) -> float | None:
        _strategy_logger = logging.getLogger(__name__ + ".SignalNormalizer._safe_normalized_value")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._safe_normalized_value")
        parsed = _to_float(value)
        if parsed is None:
            return None
        return clamp(parsed, -1.0, 1.0)

    @staticmethod
    def _infer_normalized_value(value: Any) -> float | None:
        _strategy_logger = logging.getLogger(__name__ + ".SignalNormalizer._infer_normalized_value")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalNormalizer._infer_normalized_value")
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
    _logger = logging.getLogger(__name__ + ".SignalRouter")

    component_namespace = "strategy.processor.router"

    def __init__(
        self,
        config: StrategyConfig,
        registry: StrategyRegistry,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalRouter.__init__")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalRouter.route")
        if not event_name.strip():
            raise SignalRoutingError("event_name cannot be empty")

        context.validate()

        categories = self._resolve_categories(
            event_name=event_name,
            source=source,
            context=context,
        )

        selected: list[BaseStrategy] = []
        skipped: dict[str, str] = {}

        routing_diagnostics: dict[str, Any] | None = None
        if hasattr(self.registry, "select_for_event"):
            candidates = self.registry.select_for_event(
                context=context,
                event_name=event_name,
                categories=categories or None,
                changed_features=changed_features or None,
                source=source,
            )
        elif hasattr(self.registry, "select"):
            candidates = self.registry.select(
                context=context,
                categories=categories or None,
                changed_features=changed_features or None,
                source=source,
            )
        else:
            candidates = self.registry.list_all()

        if not candidates and hasattr(self.registry, "explain_selection"):
            try:
                routing_diagnostics = self.registry.explain_selection(
                    context=context,
                    categories=categories or None,
                    changed_features=changed_features or None,
                    source=source,
                    event_name=event_name,
                )
            except Exception as exc:
                routing_diagnostics = {"error": str(exc)}

        for strategy in candidates:
            try:
                if not strategy.should_evaluate(context):
                    skipped[strategy.strategy_name] = "strategy_not_applicable"
                    continue
                selected.append(strategy)
            except (StrategyEvaluationError, ValueError, TypeError, AttributeError) as exc:
                skipped[strategy.strategy_name] = f"route_check_failed:{exc}"

        selected.sort(key=lambda item: (item.priority, item.strategy_name))

        route_metadata = dict(metadata or {})
        if routing_diagnostics is not None:
            route_metadata["routing_diagnostics"] = routing_diagnostics

        return RouteDecision(
            event_name=event_name,
            symbol=context.symbol,
            source=source,
            timestamp=context.timestamp,
            selected=selected,
            skipped=skipped,
            categories_used=categories,
            matched_features=list(changed_features or []),
            metadata=route_metadata,
        )

    async def emit_signal_generated(
        self,
        *,
        payload: RiskReadySignalPayload,
    ) -> None:
        """
        Emit the final signal.generated event consumed by RiskManager.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalRouter.emit_signal_generated")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalRouter.emit_signal_rejected")
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
        _strategy_logger = logging.getLogger(__name__ + ".SignalRouter._event_priority")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalRouter._event_priority")
        if payload.priority_score >= 0.85:
            return EventPriority.HIGH
        return EventPriority.NORMAL

    def _resolve_categories(
            self,
            *,
            event_name: str,
            source: FeatureSource | None,
            context: StrategyContext,
    ) -> list[StrategyCategory]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalRouter._resolve_categories")
        categories = self.routing_config.categories_for_event(event_name)

        if not categories and source is not None:
            mapped = self._map_source_to_category(source)
            if mapped is not None:
                categories = [mapped]

        if not categories:
            return []

        if (
                self.routing_config.route_hybrid_on_domain_signal
                and StrategyCategory.HYBRID not in categories
                and self._should_route_hybrid(context=context, trigger_source=source)
        ):
            categories.append(StrategyCategory.HYBRID)

        return categories

    def _should_route_hybrid(
            self,
            *,
            context: StrategyContext,
            trigger_source: FeatureSource | None,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalRouter._should_route_hybrid")
        if trigger_source not in self._hybrid_route_domain_sources():
            return False

        fresh_sources = self._fresh_hybrid_domain_sources(context)
        return len(fresh_sources) >= self.routing_config.min_domains_for_hybrid_route

    def _fresh_hybrid_domain_sources(self, context: StrategyContext) -> list[FeatureSource]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalRouter._fresh_hybrid_domain_sources")
        fresh: list[FeatureSource] = []

        for source in self._hybrid_route_domain_sources():
            domain = context.domain_dict(source)
            if not domain:
                continue

            if not self.routing_config.require_fresh_domains_for_hybrid_route:
                fresh.append(source)
                continue

            if not self._domain_is_stale_for_hybrid_route(context=context, domain=domain):
                fresh.append(source)

        return fresh

    def _domain_is_stale_for_hybrid_route(
            self,
            *,
            context: StrategyContext,
            domain: dict[str, Any],
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalRouter._domain_is_stale_for_hybrid_route")
        timestamp = self._extract_domain_timestamp(domain)
        parsed = self._coerce_domain_timestamp(timestamp)
        if parsed is None:
            return True

        age_seconds = max(0.0, (context.timestamp - parsed).total_seconds())
        return age_seconds > float(self.routing_config.hybrid_route_stale_seconds)

    @staticmethod
    def _coerce_domain_timestamp(value: Any) -> datetime | None:
        if value is None:
            return None

        if isinstance(value, datetime):
            return ensure_aware_utc(value)

        if isinstance(value, (int, float)):
            raw = float(value)
            # Support both seconds and milliseconds epoch values.
            if raw > 10_000_000_000:
                raw = raw / 1000.0
            try:
                return datetime.fromtimestamp(raw, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                return None

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None

            if text.isdigit():
                return SignalRouter._coerce_domain_timestamp(float(text))

            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                return None
            return ensure_aware_utc(parsed)

        return None

    @classmethod
    def _extract_domain_timestamp(cls, domain: dict[str, Any]) -> Any:
        for key in (
                "timestamp",
                "created_at",
                "updated_at",
                "event_time",
                "event_timestamp",
                "ts",
        ):
            value = domain.get(key)
            if value is not None:
                return value

        for nested_key in ("contract", "scope", "summary", "raw"):
            nested = domain.get(nested_key)
            if isinstance(nested, dict):
                value = cls._extract_domain_timestamp(nested)
                if value is not None:
                    return value

        return None

    @staticmethod
    def _hybrid_route_domain_sources() -> tuple[FeatureSource, ...]:
        return (
            FeatureSource.ORDERFLOW,
            FeatureSource.LIQUIDITY,
            FeatureSource.PRICE_ACTION,
            FeatureSource.LIQUIDATIONS,
            FeatureSource.WHALES,
            FeatureSource.SPOOFING,
            FeatureSource.SPREADS,
            FeatureSource.FUNDING,
            FeatureSource.OPEN_INTEREST,
        )

    @staticmethod
    def _map_source_to_category(source: FeatureSource) -> StrategyCategory | None:
        _strategy_logger = logging.getLogger(__name__ + ".SignalRouter._map_source_to_category")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalRouter._map_source_to_category")
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
    _logger = logging.getLogger(__name__ + ".SignalScorer")

    component_namespace = "strategy.processor.scorer"

    def score_signal(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> StrategySignal:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalScorer.score_signal")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalScorer.score_many")
        return [self.score_signal(signal=signal, context=context) for signal in signals]

    def score_signals(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext | None = None,
    ) -> ConfluenceResult:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalScorer.score_signals")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalScorer._calculate_priority_score")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalScorer._apply_weights")
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
        _strategy_logger = logging.getLogger(__name__ + ".SignalScorer._summarize_votes")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalScorer._summarize_votes")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalScorer._resolve_conflicts")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalScorer._to_confluence_result")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalScorer._confidence_grade")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalScorer._confidence_strength")
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
        _strategy_logger = logging.getLogger(__name__ + ".SignalScorer._tier_from_priority_score")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalScorer._tier_from_priority_score")
        score = clamp(value, 0.0, 1.0)
        if score >= 0.88:
            return StrategyTradeTier.T4
        if score >= 0.74:
            return StrategyTradeTier.T3
        if score >= 0.58:
            return StrategyTradeTier.T2
        return StrategyTradeTier.T1

    def _liquidity_class(self, context: StrategyContext) -> StrategyLiquidityClass:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalScorer._liquidity_class")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalScorer._execution_quality")
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
        _strategy_logger = logging.getLogger(__name__ + ".SignalScorer._liquidity_score")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalScorer._liquidity_score")
        for name in ("liquidity_score", "market_liquidity_score", "depth_score"):
            value = context.get_feature(name, None)
            parsed = _to_float(value)
            if parsed is not None:
                return clamp(parsed, 0.0, 1.0)

        return 0.5

    @staticmethod
    def _risk_reward_score(signal: StrategySignal) -> float:
        _strategy_logger = logging.getLogger(__name__ + ".SignalScorer._risk_reward_score")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalScorer._risk_reward_score")
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
        _strategy_logger = logging.getLogger(__name__ + ".SignalScorer._execution_quality_score")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalScorer._execution_quality_score")
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
        _strategy_logger = logging.getLogger(__name__ + ".SignalScorer._regime_alignment_score")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalScorer._regime_alignment_score")
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
        _strategy_logger = logging.getLogger(__name__ + ".SignalScorer._freshness_score")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalScorer._freshness_score")
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
    _logger = logging.getLogger(__name__ + ".ConfluenceEngine")

    component_namespace = "strategy.processor.confluence"

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceEngine.__init__")
        super().__init__(config=config, event_bus=event_bus, scheduler=scheduler)
        self.confluence_config: ConfluenceConfig = config.confluence
        self.scorer = SignalScorer(config=config, event_bus=event_bus, scheduler=scheduler)

    def evaluate(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext,
    ) -> ConfluenceEvaluation:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceEngine.evaluate")
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
        _strategy_logger = logging.getLogger(__name__ + ".ConfluenceEngine._merge")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceEngine._merge")
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
    _logger = logging.getLogger(__name__ + ".SignalFilterChain")

    component_namespace = "strategy.processor.filters"

    def apply(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalFilterChain.apply")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalFilterChain.evaluate_signal")
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
        _strategy_logger = logging.getLogger(__name__ + ".SignalFilterChain._filter_symbol_match")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalFilterChain._filter_symbol_match")
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
        _strategy_logger = logging.getLogger(__name__ + ".SignalFilterChain._filter_directional")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalFilterChain._filter_directional")
        if evaluation.signal.side not in {SignalSide.LONG, SignalSide.SHORT}:
            evaluation.add_result(
                FilterResult(
                    name="directional_signal",
                    decision=FilterDecision.BLOCK,
                    reason="signal_side_is_not_directional",
                )
            )

    def _filter_confidence(self, evaluation: FilterEvaluation) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalFilterChain._filter_confidence")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalFilterChain._filter_score")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalFilterChain._filter_age")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalFilterChain._filter_risk_reward")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalFilterChain._filter_regime")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalFilterChain._filter_volatility")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalFilterChain._filter_liquidity")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalFilterChain._filter_spread")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalFilterChain._filter_funding")
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
        _strategy_logger = logging.getLogger(__name__ + ".SignalFilterChain._context_float")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalFilterChain._context_float")
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
        _strategy_logger = logging.getLogger(__name__ + ".SignalFilterChain._filter_freshness")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalFilterChain._filter_freshness")
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
        _strategy_logger = logging.getLogger(__name__ + ".SignalFilterChain._filter_execution_quality")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalFilterChain._filter_execution_quality")
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
    _logger = logging.getLogger(__name__ + ".SignalBuilder")

    component_namespace = "strategy.processor.builder"

    @property
    def builder_config(self) -> BuilderConfig:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalBuilder.builder_config")
        return self.config.builders

    def build(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> BuildEvaluation:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalBuilder.build")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalBuilder.build_many")
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
        _strategy_logger = logging.getLogger(__name__ + ".SignalBuilder.assert_risk_ready")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalBuilder.assert_risk_ready")
        signal.validate()

        if signal.execution_plan is None:
            raise BuilderError("signal.execution_plan is required")

        entry_price = signal.primary_entry_price
        stop_loss = signal.primary_stop_loss
        take_profit = signal.primary_take_profit

        if entry_price is None or entry_price <= 0:
            raise BuilderError("risk-ready signal requires entry_price > 0")

        if stop_loss is None or stop_loss <= 0:
            raise BuilderError("risk-ready signal requires stop_loss > 0")

        if take_profit is None or take_profit <= 0:
            raise BuilderError("risk-ready signal requires take_profit > 0")

        if signal.side is SignalSide.LONG and stop_loss >= entry_price:
            raise BuilderError("long signal stop_loss must be below entry_price")

        if signal.side is SignalSide.SHORT and stop_loss <= entry_price:
            raise BuilderError("short signal stop_loss must be above entry_price")

        if signal.side is SignalSide.LONG and take_profit <= entry_price:
            raise BuilderError("long signal take_profit must be above entry_price")

        if signal.side is SignalSide.SHORT and take_profit >= entry_price:
            raise BuilderError("short signal take_profit must be below entry_price")

    def _resolve_entry(
        self,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> EntryPlan:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalBuilder._resolve_entry")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalBuilder._resolve_invalidation")
        if signal.invalidation_plan is not None:
            signal.invalidation_plan.validate()
            return signal.invalidation_plan

        stop = _to_float(signal.metadata.get("stop_loss"))

        if stop is None and signal.exit_plan is not None:
            stop = _to_float(signal.exit_plan.stop_loss)

        if stop is None:
            entry_price = entry.price

            if entry_price is None or entry_price <= 0:
                raise BuilderError("entry.price is required to infer stop_loss")

            stop = self._infer_stop_loss(signal, entry_price=entry_price)

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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalBuilder._resolve_targets")
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
            entry_price = entry.price
            stop_loss = invalidation.price

            if entry_price is None or entry_price <= 0:
                raise BuilderError("entry.price is required to infer take_profit")

            if stop_loss is None or stop_loss <= 0:
                raise BuilderError("invalidation.price is required to infer_take_profit")

            take_profit = self._infer_take_profit(
                signal,
                entry_price=entry_price,
                stop_loss=stop_loss,
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalBuilder._resolve_exit_plan")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalBuilder._resolve_execution_plan")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalBuilder._infer_stop_loss")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalBuilder._infer_take_profit")
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
        _strategy_logger = logging.getLogger(__name__ + ".SignalBuilder._entry_type")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalBuilder._entry_type")
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
    _logger = logging.getLogger(__name__ + ".PortfolioCoordinator")

    component_namespace = "strategy.processor.portfolio"

    def __init__(
        self,
        config: StrategyConfig,
        state: StrategyRuntimeState,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PortfolioCoordinator.__init__")
        super().__init__(config=config, event_bus=event_bus, scheduler=scheduler)
        self.state = state
        self.portfolio_config: PortfolioCoordinatorConfig = config.portfolio

    def coordinate(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext,
    ) -> CoordinationDecision:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PortfolioCoordinator.coordinate")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PortfolioCoordinator._suppress_repeating_signals")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PortfolioCoordinator._deduplicate_by_side")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PortfolioCoordinator._apply_symbol_limits")
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
        _strategy_logger = logging.getLogger(__name__ + ".PortfolioCoordinator._merge_similar_signals")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PortfolioCoordinator._merge_similar_signals")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PortfolioCoordinator._update_state_after_acceptance")
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
    _logger = logging.getLogger(__name__ + ".SignalProcessor")

    component_namespace = "strategy.processor"

    def __init__(
        self,
        config: StrategyConfig,
        registry: StrategyRegistry,
        state: StrategyRuntimeState,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalProcessor.__init__")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalProcessor.process_event")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalProcessor.evaluate_strategies")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalProcessor.to_risk_payload")
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
        _strategy_logger = logging.getLogger(__name__ + ".SignalProcessor._enum_value")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalProcessor._enum_value")
        if hasattr(value, "value"):
            return value.value
        return value

    @staticmethod
    def _safe_iso(value: Any) -> str | None:
        _strategy_logger = logging.getLogger(__name__ + ".SignalProcessor._safe_iso")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalProcessor._safe_iso")
        if isinstance(value, datetime):
            return ensure_aware_utc(value).isoformat()
        return None

    def _evaluation_debug(
        self,
        evaluations: list[StrategyEvaluation],
    ) -> list[dict[str, Any]]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalProcessor._evaluation_debug")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalProcessor._route_debug")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalProcessor._context_debug")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalProcessor._normalized_debug")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalProcessor._signals_debug")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalProcessor._confluence_debug")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalProcessor._coordination_debug")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalProcessor._update_batch_debug")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalProcessor._resolve_context")
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
        _strategy_logger = logging.getLogger(__name__ + ".SignalProcessor._signals_after_confluence")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalProcessor._signals_after_confluence")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalProcessor._emit_rejected_batch")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalProcessor._record_final_signals")
        for signal in signals:
            signal.to_pending()
            self.state.update_signal(signal, active=True)
            self.state.metrics.record_signal(signal)

    @staticmethod
    def _execution_cost_from_metadata(
        signal: StrategySignal,
    ) -> ExecutionCostPayload | None:
        _strategy_logger = logging.getLogger(__name__ + ".SignalProcessor._execution_cost_from_metadata")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalProcessor._execution_cost_from_metadata")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalProcessor._confidence_grade")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalProcessor._confidence_strength")
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
        _strategy_logger = logging.getLogger(__name__ + ".SignalProcessor._tier_from_priority_score")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalProcessor._tier_from_priority_score")
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