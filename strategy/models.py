from __future__ import annotations
import logging

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
from typing import Any
from uuid import uuid4

from strategy.enums import (
    ConfidenceGrade,
    ConflictType,
    EntryType,
    ExitType,
    FeatureSource,
    FilterDecision,
    FreshnessStatus,
    MarketRegime,
    SetupType,
    SignalOrigin,
    SignalPriority,
    SignalSide,
    SignalStatus,
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
from strategy.exceptions import ValidationError


def utcnow() -> datetime:
    """
    Return timezone-aware UTC datetime.

    Важливо для:
    - EventBus timestamps;
    - PostgreSQL;
    - dashboard;
    - backtesting;
    - коректного порівняння часу між модулями.
    """
    return datetime.now(timezone.utc)


def ensure_aware_utc(value: datetime) -> datetime:
    """
    Normalize datetime to timezone-aware UTC.

    Якщо datetime naive — трактуємо його як UTC.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


@dataclass(slots=True)
class FeatureSnapshot:
    _logger = logging.getLogger(__name__ + ".FeatureSnapshot")
    name: str
    value: Any
    source: FeatureSource
    symbol: str
    timestamp: datetime
    confidence: float = 0.0
    normalized_value: float | None = None
    freshness_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering FeatureSnapshot.__post_init__")
        self.timestamp = ensure_aware_utc(self.timestamp)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering FeatureSnapshot.validate")
        if not self.name.strip():
            raise ValidationError("FeatureSnapshot.name cannot be empty")
        if not self.symbol.strip():
            raise ValidationError("FeatureSnapshot.symbol cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError("FeatureSnapshot.confidence must be between 0.0 and 1.0")
        if self.normalized_value is not None and not -1.0 <= self.normalized_value <= 1.0:
            raise ValidationError("FeatureSnapshot.normalized_value must be between -1.0 and 1.0")
        if self.freshness_seconds is not None and self.freshness_seconds <= 0:
            raise ValidationError("FeatureSnapshot.freshness_seconds must be > 0")

    def age_seconds(self, now: datetime | None = None) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering FeatureSnapshot.age_seconds")
        current = ensure_aware_utc(now or utcnow())
        return max(0.0, (current - self.timestamp).total_seconds())

    def freshness_status(self, now: datetime | None = None) -> FreshnessStatus:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering FeatureSnapshot.freshness_status")
        if self.freshness_seconds is None:
            return FreshnessStatus.FRESH

        age = self.age_seconds(now)
        if age <= self.freshness_seconds * 0.5:
            return FreshnessStatus.FRESH
        if age <= self.freshness_seconds:
            return FreshnessStatus.AGING
        if age <= self.freshness_seconds * 2:
            return FreshnessStatus.STALE
        return FreshnessStatus.EXPIRED

    def is_stale(self, now: datetime | None = None) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering FeatureSnapshot.is_stale")
        return self.freshness_status(now) in {
            FreshnessStatus.STALE,
            FreshnessStatus.EXPIRED,
        }

    def is_expired(self, now: datetime | None = None) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering FeatureSnapshot.is_expired")
        return self.freshness_status(now) == FreshnessStatus.EXPIRED


@dataclass(slots=True)
class StrategyMetadata:
    _logger = logging.getLogger(__name__ + ".StrategyMetadata")
    strategy_name: str
    category: StrategyCategory
    timeframe: Timeframe
    tags: list[str] = field(default_factory=list)
    version: str = "1.0.0"
    author: str | None = None
    description: str | None = None
    required_features: set[str] = field(default_factory=set)
    supported_regimes: set[MarketRegime] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyMetadata.validate")
        if not self.strategy_name.strip():
            raise ValidationError("StrategyMetadata.strategy_name cannot be empty")
        if not self.version.strip():
            raise ValidationError("StrategyMetadata.version cannot be empty")


@dataclass(slots=True)
class EntryPlan:
    _logger = logging.getLogger(__name__ + ".EntryPlan")
    entry_type: EntryType
    price: float | None = None
    timeout_seconds: int | None = None
    max_slippage_bps: float | None = None
    confirmation_required: bool = False
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EntryPlan.validate")
        if self.price is not None and self.price <= 0:
            raise ValidationError("EntryPlan.price must be > 0")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValidationError("EntryPlan.timeout_seconds must be > 0")
        if self.max_slippage_bps is not None and self.max_slippage_bps < 0:
            raise ValidationError("EntryPlan.max_slippage_bps must be >= 0")


@dataclass(slots=True)
class TargetPlan:
    _logger = logging.getLogger(__name__ + ".TargetPlan")
    price: float
    size_fraction: float = 1.0
    rr: float | None = None
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering TargetPlan.validate")
        if self.price <= 0:
            raise ValidationError("TargetPlan.price must be > 0")
        if not 0 < self.size_fraction <= 1:
            raise ValidationError("TargetPlan.size_fraction must be in (0, 1]")
        if self.rr is not None and self.rr <= 0:
            raise ValidationError("TargetPlan.rr must be > 0")


@dataclass(slots=True)
class InvalidationPlan:
    _logger = logging.getLogger(__name__ + ".InvalidationPlan")
    price: float | None = None
    reason: str | None = None
    timeout_seconds: int | None = None
    conditions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering InvalidationPlan.validate")
        if self.price is not None and self.price <= 0:
            raise ValidationError("InvalidationPlan.price must be > 0")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValidationError("InvalidationPlan.timeout_seconds must be > 0")


@dataclass(slots=True)
class ExitPlan:
    _logger = logging.getLogger(__name__ + ".ExitPlan")
    exit_types: list[ExitType] = field(default_factory=list)
    stop_loss: float | None = None
    take_profit_levels: list[TargetPlan] = field(default_factory=list)
    trailing_distance: float | None = None
    max_holding_seconds: int | None = None
    partial_exit_enabled: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ExitPlan.validate")
        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValidationError("ExitPlan.stop_loss must be > 0")
        if self.trailing_distance is not None and self.trailing_distance <= 0:
            raise ValidationError("ExitPlan.trailing_distance must be > 0")
        if self.max_holding_seconds is not None and self.max_holding_seconds <= 0:
            raise ValidationError("ExitPlan.max_holding_seconds must be > 0")

        for target in self.take_profit_levels:
            target.validate()


@dataclass(slots=True)
class ExecutionPlanDraft:
    _logger = logging.getLogger(__name__ + ".ExecutionPlanDraft")
    symbol: str
    side: SignalSide
    entry: EntryPlan
    exit: ExitPlan
    invalidation: InvalidationPlan
    leverage: float | None = None
    reduce_only: bool = False
    post_only: bool = False
    expected_holding_seconds: int | None = None
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ExecutionPlanDraft.validate")
        if not self.symbol.strip():
            raise ValidationError("ExecutionPlanDraft.symbol cannot be empty")
        if self.side not in {SignalSide.LONG, SignalSide.SHORT}:
            raise ValidationError("ExecutionPlanDraft.side must be LONG or SHORT")
        if self.leverage is not None and self.leverage <= 0:
            raise ValidationError("ExecutionPlanDraft.leverage must be > 0")
        if self.expected_holding_seconds is not None and self.expected_holding_seconds <= 0:
            raise ValidationError("ExecutionPlanDraft.expected_holding_seconds must be > 0")

        self.entry.validate()
        self.exit.validate()
        self.invalidation.validate()


@dataclass(slots=True)
class FilterResult:
    _logger = logging.getLogger(__name__ + ".FilterResult")
    name: str
    decision: FilterDecision
    reason: str | None = None
    score_impact: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering FilterResult.validate")
        if not self.name.strip():
            raise ValidationError("FilterResult.name cannot be empty")

    @property
    def passed(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering FilterResult.passed")
        return self.decision in {FilterDecision.PASS, FilterDecision.WARN}

    @property
    def blocked(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering FilterResult.blocked")
        return self.decision == FilterDecision.BLOCK


@dataclass(slots=True)
class ConflictRecord:
    _logger = logging.getLogger(__name__ + ".ConflictRecord")
    conflict_type: ConflictType
    source: str
    message: str
    penalty: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConflictRecord.validate")
        if not self.source.strip():
            raise ValidationError("ConflictRecord.source cannot be empty")
        if not self.message.strip():
            raise ValidationError("ConflictRecord.message cannot be empty")
        if self.penalty < 0:
            raise ValidationError("ConflictRecord.penalty must be >= 0")


@dataclass(slots=True)
class StrategySignal:
    """
    Internal strategy-layer signal.

    Concrete strategies return StrategySignal. It is not sent to RiskManager
    directly. SignalProcessor/SignalRouter converts it into RiskReadySignalPayload
    and emits signal.generated.
    """
    _logger = logging.getLogger(__name__ + ".StrategySignal")

    symbol: str
    side: SignalSide
    strategy_name: str
    category: StrategyCategory
    timeframe: Timeframe
    setup_type: SetupType
    timestamp: datetime

    signal_id: str = field(default_factory=lambda: uuid4().hex)

    confidence: float = 0.0
    score: float = 0.0
    strength: SignalStrength = SignalStrength.WEAK
    confidence_grade: ConfidenceGrade = ConfidenceGrade.VERY_LOW
    status: SignalStatus = SignalStatus.NEW

    trigger_type: TriggerType = TriggerType.PRIMARY
    origin: SignalOrigin = SignalOrigin.SINGLE_STRATEGY
    priority: SignalPriority = SignalPriority.MEDIUM

    reasons: list[str] = field(default_factory=list)
    confirmations: list[str] = field(default_factory=list)
    source_features: list[str] = field(default_factory=list)
    combined_from: list[str] = field(default_factory=list)

    conflicts: list[ConflictRecord] = field(default_factory=list)
    filter_results: list[FilterResult] = field(default_factory=list)

    entry_plan: EntryPlan | None = None
    exit_plan: ExitPlan | None = None
    invalidation_plan: InvalidationPlan | None = None
    execution_plan: ExecutionPlanDraft | None = None

    regime: MarketRegime = MarketRegime.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.__post_init__")
        self.timestamp = ensure_aware_utc(self.timestamp)
        self.symbol = self.symbol.strip()
        self.strategy_name = self.strategy_name.strip()
        self.signal_id = str(self.signal_id or uuid4().hex)
        self.confidence = clamp(self.confidence, 0.0, 1.0)
        self.score = max(0.0, float(self.score))

        # Keep a stable copy inside metadata for old state/event handlers that
        # still look up metadata["signal_id"].
        self.metadata.setdefault("signal_id", self.signal_id)

        if self.strength == SignalStrength.WEAK:
            self.strength = confidence_to_strength(self.confidence)

        if self.confidence_grade == ConfidenceGrade.VERY_LOW:
            self.confidence_grade = confidence_to_grade(self.confidence)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.validate")
        if not self.signal_id.strip():
            raise ValidationError("StrategySignal.signal_id cannot be empty")
        if not self.symbol.strip():
            raise ValidationError("StrategySignal.symbol cannot be empty")
        if not self.strategy_name.strip():
            raise ValidationError("StrategySignal.strategy_name cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError("StrategySignal.confidence must be between 0.0 and 1.0")
        if self.score < 0:
            raise ValidationError("StrategySignal.score must be >= 0")

        if not self.is_directional and self.status in {SignalStatus.PENDING, SignalStatus.CONFIRMED}:
            raise ValidationError("non-directional StrategySignal cannot be pending/confirmed")

        for conflict in self.conflicts:
            conflict.validate()
        for result in self.filter_results:
            result.validate()

        if self.entry_plan is not None:
            self.entry_plan.validate()
        if self.exit_plan is not None:
            self.exit_plan.validate()
        if self.invalidation_plan is not None:
            self.invalidation_plan.validate()
        if self.execution_plan is not None:
            self.execution_plan.validate()

    @property
    def is_long(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.is_long")
        return self.side == SignalSide.LONG

    @property
    def is_short(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.is_short")
        return self.side == SignalSide.SHORT

    @property
    def is_flat(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.is_flat")
        return self.side == SignalSide.FLAT

    @property
    def is_directional(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.is_directional")
        return self.side in {SignalSide.LONG, SignalSide.SHORT}

    @property
    def is_active(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.is_active")
        return self.status in {
            SignalStatus.NEW,
            SignalStatus.PENDING,
            SignalStatus.CONFIRMED,
        }

    @property
    def passed_filters(self) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.passed_filters")
        return all(result.passed for result in self.filter_results)

    @property
    def total_conflict_penalty(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.total_conflict_penalty")
        return sum(conflict.penalty for conflict in self.conflicts)

    @property
    def primary_entry_price(self) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.primary_entry_price")
        if self.execution_plan is not None and self.execution_plan.entry.price is not None:
            return self.execution_plan.entry.price
        if self.entry_plan is not None:
            return self.entry_plan.price
        return None

    @property
    def primary_stop_loss(self) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.primary_stop_loss")
        if self.execution_plan is not None:
            if self.execution_plan.exit.stop_loss is not None:
                return self.execution_plan.exit.stop_loss
            if self.execution_plan.invalidation.price is not None:
                return self.execution_plan.invalidation.price
        if self.exit_plan is not None and self.exit_plan.stop_loss is not None:
            return self.exit_plan.stop_loss
        if self.invalidation_plan is not None:
            return self.invalidation_plan.price
        return None

    @property
    def primary_take_profit(self) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.primary_take_profit")
        targets: list[TargetPlan] = []
        if self.execution_plan is not None:
            targets = self.execution_plan.exit.take_profit_levels
        elif self.exit_plan is not None:
            targets = self.exit_plan.take_profit_levels
        return targets[0].price if targets else None

    def add_reason(self, reason: str) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.add_reason")
        if reason and reason not in self.reasons:
            self.reasons.append(reason)

    def add_confirmation(self, confirmation: str) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.add_confirmation")
        if confirmation and confirmation not in self.confirmations:
            self.confirmations.append(confirmation)

    def add_source_feature(self, feature_name: str) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.add_source_feature")
        if feature_name and feature_name not in self.source_features:
            self.source_features.append(feature_name)

    def add_filter_result(self, filter_result: FilterResult) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.add_filter_result")
        filter_result.validate()
        self.filter_results.append(filter_result)

    def add_conflict(self, conflict: ConflictRecord) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.add_conflict")
        conflict.validate()
        self.conflicts.append(conflict)

    def to_pending(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.to_pending")
        self.status = SignalStatus.PENDING

    def to_confirmed(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.to_confirmed")
        self.status = SignalStatus.CONFIRMED

    def to_rejected(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.to_rejected")
        self.status = SignalStatus.REJECTED

    def to_cancelled(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.to_cancelled")
        self.status = SignalStatus.CANCELLED

    def to_expired(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.to_expired")
        self.status = SignalStatus.EXPIRED

    def to_executed(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.to_executed")
        self.status = SignalStatus.EXECUTED

    def to_failed(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.to_failed")
        self.status = SignalStatus.FAILED

    def to_dict(self) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignal.to_dict")
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "strategy_name": self.strategy_name,
            "category": self.category.value,
            "timeframe": self.timeframe.value,
            "setup_type": self.setup_type.value,
            "timestamp": self.timestamp.timestamp(),
            "confidence": self.confidence,
            "score": self.score,
            "strength": self.strength.value,
            "confidence_grade": self.confidence_grade.value,
            "status": self.status.value,
            "trigger_type": self.trigger_type.value,
            "origin": self.origin.value,
            "priority": self.priority.value,
            "reasons": list(self.reasons),
            "confirmations": list(self.confirmations),
            "source_features": list(self.source_features),
            "combined_from": list(self.combined_from),
            "regime": self.regime.value,
            "entry_price": self.primary_entry_price,
            "stop_loss": self.primary_stop_loss,
            "take_profit": self.primary_take_profit,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class RawStrategySignal:
    _logger = logging.getLogger(__name__ + ".RawStrategySignal")
    signal: StrategySignal
    raw_inputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering RawStrategySignal.validate")
        self.signal.validate()


@dataclass(slots=True)
class ConfirmedSignal:
    _logger = logging.getLogger(__name__ + ".ConfirmedSignal")
    signal: StrategySignal
    accepted_by_confluence: bool = False
    accepted_by_filters: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfirmedSignal.validate")
        self.signal.validate()


@dataclass(slots=True)
class TradeIdea:
    _logger = logging.getLogger(__name__ + ".TradeIdea")
    signal: StrategySignal
    execution_plan: ExecutionPlanDraft
    created_at: datetime = field(default_factory=utcnow)
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering TradeIdea.__post_init__")
        self.created_at = ensure_aware_utc(self.created_at)
        if self.expires_at is not None:
            self.expires_at = ensure_aware_utc(self.expires_at)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering TradeIdea.validate")
        self.signal.validate()
        self.execution_plan.validate()

        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValidationError("TradeIdea.expires_at must be after created_at")

    def is_expired(self, now: datetime | None = None) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering TradeIdea.is_expired")
        if self.expires_at is None:
            return False
        current = ensure_aware_utc(now or utcnow())
        return current >= self.expires_at




def _is_finite_number(value: float | int | None) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _optional_finite(value: float | int | None, field_name: str) -> float | None:
    if value is None:
        return None
    if not _is_finite_number(value):
        raise ValidationError(f"{field_name} must be finite")
    return float(value)


def _required_positive(value: float | int | None, field_name: str) -> float:
    if value is None or not _is_finite_number(value):
        raise ValidationError(f"{field_name} must be a finite positive number")
    value_f = float(value)
    if value_f <= 0:
        raise ValidationError(f"{field_name} must be > 0")
    return value_f


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


@dataclass(slots=True)
class ExecutionCostPayload:
    """
    Strategy-side execution-cost estimate sent inside signal.generated.

    RiskManager converts this dictionary into risk.models.ExecutionCostEstimate.
    """
    _logger = logging.getLogger(__name__ + ".ExecutionCostPayload")

    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    fee_cost: float = 0.0
    funding_cost: float = 0.0
    other_cost: float = 0.0
    spread_pct: float | None = None
    slippage_pct: float | None = None
    quality: StrategyExecutionQuality = StrategyExecutionQuality.ACCEPTABLE
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ExecutionCostPayload.validate")
        for field_name, value in {
            "spread_cost": self.spread_cost,
            "slippage_cost": self.slippage_cost,
            "fee_cost": self.fee_cost,
            "funding_cost": self.funding_cost,
            "other_cost": self.other_cost,
            "spread_pct": self.spread_pct,
            "slippage_pct": self.slippage_pct,
        }.items():
            if value is None:
                continue
            if not _is_finite_number(value):
                raise ValidationError(f"ExecutionCostPayload.{field_name} must be finite")
            if float(value) < 0:
                raise ValidationError(f"ExecutionCostPayload.{field_name} must be >= 0")

    @property
    def total_cost(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ExecutionCostPayload.total_cost")
        return (
            self.spread_cost
            + self.slippage_cost
            + self.fee_cost
            + self.funding_cost
            + self.other_cost
        )

    def to_dict(self) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ExecutionCostPayload.to_dict")
        self.validate()
        return {
            "spread_cost": float(self.spread_cost),
            "slippage_cost": float(self.slippage_cost),
            "fee_cost": float(self.fee_cost),
            "funding_cost": float(self.funding_cost),
            "other_cost": float(self.other_cost),
            "spread_pct": self.spread_pct,
            "slippage_pct": self.slippage_pct,
            "quality": self.quality.value,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class RiskReadySignalPayload:
    """
    Stable EventBus contract emitted as signal.generated and consumed by RiskManager.

    This class is the adapter between strategy and risk. It mirrors the fields
    that RiskManager._request_from_payload(...) reads, without importing risk.enums
    or risk.models into strategy.
    """
    _logger = logging.getLogger(__name__ + ".RiskReadySignalPayload")

    signal_id: str
    symbol: str
    side: SignalSide

    entry_price: float
    stop_loss: float | None
    take_profit: float | None

    strategy_name: str
    tier: StrategyTradeTier = StrategyTradeTier.T2
    order_intent: StrategyOrderIntent = StrategyOrderIntent.OPEN

    confidence: float = 0.0
    edge_score: float = 0.0
    priority_score: float = 0.0
    volatility: float | None = None

    liquidity_class: StrategyLiquidityClass = StrategyLiquidityClass.NORMAL
    execution_quality: StrategyExecutionQuality = StrategyExecutionQuality.ACCEPTABLE

    expected_reward: float | None = None
    expected_loss: float | None = None
    expected_win_probability: float | None = None
    expected_cost: float | None = None
    execution_cost: ExecutionCostPayload | None = None

    requested_size: float | None = None
    requested_margin: float | None = None
    requested_leverage: float | None = None

    reduce_only: bool = False
    margin_mode: StrategyMarginMode = StrategyMarginMode.ISOLATED

    exchange: str | None = None
    market_type: StrategyMarketType = StrategyMarketType.USDM_FUTURES
    timeframe: Timeframe | None = None
    timestamp: datetime = field(default_factory=utcnow)

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering RiskReadySignalPayload.__post_init__")
        self.signal_id = str(self.signal_id or uuid4().hex)
        self.symbol = self.symbol.strip()
        self.strategy_name = self.strategy_name.strip()
        self.timestamp = ensure_aware_utc(self.timestamp)
        self.confidence = clamp(float(self.confidence), 0.0, 1.0)
        self.edge_score = clamp(float(self.edge_score), 0.0, 1.0)
        self.priority_score = clamp(float(self.priority_score), 0.0, 1.0)
        self.entry_price = _required_positive(self.entry_price, "entry_price")
        self.stop_loss = _optional_finite(self.stop_loss, "stop_loss")
        self.take_profit = _optional_finite(self.take_profit, "take_profit")
        self.volatility = _optional_finite(self.volatility, "volatility")
        self.expected_reward = _optional_finite(self.expected_reward, "expected_reward")
        self.expected_loss = _optional_finite(self.expected_loss, "expected_loss")
        self.expected_win_probability = _optional_finite(
            self.expected_win_probability,
            "expected_win_probability",
        )
        self.expected_cost = _optional_finite(self.expected_cost, "expected_cost")
        self.requested_size = _optional_finite(self.requested_size, "requested_size")
        self.requested_margin = _optional_finite(self.requested_margin, "requested_margin")
        self.requested_leverage = _optional_finite(self.requested_leverage, "requested_leverage")

        self.metadata.setdefault("signal_id", self.signal_id)
        self.metadata.setdefault("priority_score", self.priority_score)
        self.metadata.setdefault("exchange", self.exchange)
        self.metadata.setdefault("market_type", self.market_type.value)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering RiskReadySignalPayload.validate")
        if not self.signal_id.strip():
            raise ValidationError("RiskReadySignalPayload.signal_id cannot be empty")
        if not self.symbol.strip():
            raise ValidationError("RiskReadySignalPayload.symbol cannot be empty")
        if not self.strategy_name.strip():
            raise ValidationError("RiskReadySignalPayload.strategy_name cannot be empty")
        if self.side not in {SignalSide.LONG, SignalSide.SHORT}:
            raise ValidationError("RiskReadySignalPayload.side must be LONG or SHORT")
        if self.entry_price <= 0:
            raise ValidationError("RiskReadySignalPayload.entry_price must be > 0")
        if self.stop_loss is None or self.stop_loss <= 0:
            raise ValidationError("RiskReadySignalPayload.stop_loss is required and must be > 0")
        if self.take_profit is not None and self.take_profit <= 0:
            raise ValidationError("RiskReadySignalPayload.take_profit must be > 0 when provided")
        if self.requested_leverage is not None and self.requested_leverage <= 0:
            raise ValidationError("RiskReadySignalPayload.requested_leverage must be > 0")
        if self.requested_size is not None and self.requested_size <= 0:
            raise ValidationError("RiskReadySignalPayload.requested_size must be > 0")
        if self.requested_margin is not None and self.requested_margin <= 0:
            raise ValidationError("RiskReadySignalPayload.requested_margin must be > 0")
        if self.expected_win_probability is not None and not 0.0 <= self.expected_win_probability <= 1.0:
            raise ValidationError("RiskReadySignalPayload.expected_win_probability must be between 0.0 and 1.0")

        if self.side is SignalSide.LONG:
            if self.stop_loss >= self.entry_price:
                raise ValidationError("LONG signal requires stop_loss < entry_price")
            if self.take_profit is not None and self.take_profit <= self.entry_price:
                raise ValidationError("LONG signal requires take_profit > entry_price")

        if self.side is SignalSide.SHORT:
            if self.stop_loss <= self.entry_price:
                raise ValidationError("SHORT signal requires stop_loss > entry_price")
            if self.take_profit is not None and self.take_profit >= self.entry_price:
                raise ValidationError("SHORT signal requires take_profit < entry_price")

        if self.execution_cost is not None:
            self.execution_cost.validate()

    def to_dict(self) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering RiskReadySignalPayload.to_dict")
        self.validate()

        payload = {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "strategy_name": self.strategy_name,
            "tier": self.tier.value,
            "order_intent": self.order_intent.value,
            "confidence": self.confidence,
            "edge_score": self.edge_score,
            "priority_score": self.priority_score,
            "volatility": self.volatility,
            "liquidity_class": self.liquidity_class.value,
            "execution_quality": self.execution_quality.value,
            "expected_reward": self.expected_reward,
            "expected_loss": self.expected_loss,
            "expected_win_probability": self.expected_win_probability,
            "expected_cost": self.expected_cost,
            "requested_size": self.requested_size,
            "requested_margin": self.requested_margin,
            "requested_leverage": self.requested_leverage,
            "reduce_only": self.reduce_only,
            "margin_mode": self.margin_mode.value,
            "exchange": self.exchange,
            "market_type": self.market_type.value,
            "timeframe": self.timeframe.value if self.timeframe else None,
            "timestamp": self.timestamp.timestamp(),
            "metadata": dict(self.metadata),
        }

        if self.execution_cost is not None:
            payload["execution_cost"] = self.execution_cost.to_dict()

        return payload

    @classmethod
    def from_signal(
            cls,
            signal: StrategySignal,
            *,
            tier: StrategyTradeTier = StrategyTradeTier.T2,
            order_intent: StrategyOrderIntent = StrategyOrderIntent.OPEN,
            liquidity_class: StrategyLiquidityClass = StrategyLiquidityClass.NORMAL,
            execution_quality: StrategyExecutionQuality = StrategyExecutionQuality.ACCEPTABLE,
            priority_score: float | None = None,
            expected_reward: float | None = None,
            expected_loss: float | None = None,
            expected_win_probability: float | None = None,
            expected_cost: float | None = None,
            execution_cost: ExecutionCostPayload | None = None,
            exchange: str | None = None,
            market_type: StrategyMarketType = StrategyMarketType.USDM_FUTURES,
            margin_mode: StrategyMarginMode = StrategyMarginMode.ISOLATED,
    ) -> "RiskReadySignalPayload":
        _strategy_logger = getattr(cls, "_logger", None) or logging.getLogger(__name__ + ".RiskReadySignalPayload")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering RiskReadySignalPayload.from_signal")
        signal.validate()

        entry_price = signal.primary_entry_price
        stop_loss = signal.primary_stop_loss
        take_profit = signal.primary_take_profit

        if entry_price is None or entry_price <= 0:
            raise ValueError(
                f"{signal.strategy_name}: signal.primary_entry_price is required "
                "to build RiskReadySignalPayload"
            )

        if stop_loss is None or stop_loss <= 0:
            raise ValueError(
                f"{signal.strategy_name}: signal.primary_stop_loss is required "
                "to build RiskReadySignalPayload"
            )

        if take_profit is None or take_profit <= 0:
            raise ValueError(
                f"{signal.strategy_name}: signal.primary_take_profit is required "
                "to build RiskReadySignalPayload"
            )

        requested_leverage = None
        reduce_only = False

        if signal.execution_plan is not None:
            requested_leverage = signal.execution_plan.leverage
            reduce_only = signal.execution_plan.reduce_only

        metadata = {
            "category": signal.category.value,
            "setup_type": signal.setup_type.value,
            "trigger_type": signal.trigger_type.value,
            "origin": signal.origin.value,
            "priority": signal.priority.value,
            "strength": signal.strength.value,
            "confidence_grade": signal.confidence_grade.value,
            "regime": signal.regime.value,
            "reasons": list(signal.reasons),
            "confirmations": list(signal.confirmations),
            "source_features": list(signal.source_features),
            "combined_from": list(signal.combined_from),
            **dict(signal.metadata),
        }

        if exchange is None:
            raw_exchange = metadata.get("exchange")
            exchange = str(raw_exchange) if raw_exchange else None

        return cls(
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
            edge_score=signal.score,
            priority_score=signal.score if priority_score is None else priority_score,
            liquidity_class=liquidity_class,
            execution_quality=execution_quality,
            expected_reward=expected_reward,
            expected_loss=expected_loss,
            expected_win_probability=expected_win_probability,
            expected_cost=expected_cost,
            execution_cost=execution_cost,
            requested_leverage=requested_leverage,
            reduce_only=reduce_only,
            margin_mode=margin_mode,
            exchange=exchange,
            market_type=market_type,
            timeframe=signal.timeframe,
            timestamp=signal.timestamp,
            metadata=metadata,
        )

@dataclass(slots=True)
class StrategyEvaluation:
    _logger = logging.getLogger(__name__ + ".StrategyEvaluation")
    strategy_name: str
    symbol: str
    timestamp: datetime
    signal: StrategySignal | None = None
    passed: bool = False
    score: float = 0.0
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyEvaluation.__post_init__")
        self.timestamp = ensure_aware_utc(self.timestamp)
        self.confidence = clamp(self.confidence, 0.0, 1.0)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyEvaluation.validate")
        if not self.strategy_name.strip():
            raise ValidationError("StrategyEvaluation.strategy_name cannot be empty")
        if not self.symbol.strip():
            raise ValidationError("StrategyEvaluation.symbol cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError("StrategyEvaluation.confidence must be between 0.0 and 1.0")
        if self.signal is not None:
            self.signal.validate()


@dataclass(slots=True)
class ConfluenceResult:
    _logger = logging.getLogger(__name__ + ".ConfluenceResult")
    symbol: str
    timestamp: datetime
    side: SignalSide = SignalSide.UNKNOWN
    score: float = 0.0
    confidence: float = 0.0
    confidence_grade: ConfidenceGrade = ConfidenceGrade.VERY_LOW
    strength: SignalStrength = SignalStrength.WEAK
    strategy_names: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    confirmations: list[str] = field(default_factory=list)
    conflicts: list[ConflictRecord] = field(default_factory=list)
    accepted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceResult.__post_init__")
        self.timestamp = ensure_aware_utc(self.timestamp)
        self.confidence = clamp(self.confidence, 0.0, 1.0)

        if self.confidence_grade == ConfidenceGrade.VERY_LOW:
            self.confidence_grade = confidence_to_grade(self.confidence)

        if self.strength == SignalStrength.WEAK:
            self.strength = confidence_to_strength(self.confidence)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceResult.validate")
        if not self.symbol.strip():
            raise ValidationError("ConfluenceResult.symbol cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError("ConfluenceResult.confidence must be between 0.0 and 1.0")
        for conflict in self.conflicts:
            conflict.validate()

    @property
    def total_conflict_penalty(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceResult.total_conflict_penalty")
        return sum(conflict.penalty for conflict in self.conflicts)

    def add_reason(self, reason: str) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceResult.add_reason")
        if reason and reason not in self.reasons:
            self.reasons.append(reason)

    def add_confirmation(self, confirmation: str) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering ConfluenceResult.add_confirmation")
        if confirmation and confirmation not in self.confirmations:
            self.confirmations.append(confirmation)


@dataclass(slots=True)
class PortfolioSnapshot:
    _logger = logging.getLogger(__name__ + ".PortfolioSnapshot")
    open_positions: int = 0
    active_signals: int = 0
    symbol_exposure: dict[str, float] = field(default_factory=dict)
    strategy_exposure: dict[str, float] = field(default_factory=dict)
    correlation_groups: dict[str, list[str]] = field(default_factory=dict)
    blocked_symbols: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PortfolioSnapshot.validate")
        if self.open_positions < 0:
            raise ValidationError("PortfolioSnapshot.open_positions must be >= 0")
        if self.active_signals < 0:
            raise ValidationError("PortfolioSnapshot.active_signals must be >= 0")

        for symbol, exposure in self.symbol_exposure.items():
            if not symbol.strip():
                raise ValidationError("PortfolioSnapshot.symbol_exposure contains empty symbol")
            if exposure < 0:
                raise ValidationError(f"PortfolioSnapshot.symbol_exposure[{symbol}] must be >= 0")

        for strategy_name, exposure in self.strategy_exposure.items():
            if not strategy_name.strip():
                raise ValidationError("PortfolioSnapshot.strategy_exposure contains empty strategy name")
            if exposure < 0:
                raise ValidationError(
                    f"PortfolioSnapshot.strategy_exposure[{strategy_name}] must be >= 0"
                )


@dataclass(slots=True)
class RegimeSnapshot:
    _logger = logging.getLogger(__name__ + ".RegimeSnapshot")
    symbol: str
    regime: MarketRegime = MarketRegime.UNKNOWN
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=utcnow)
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering RegimeSnapshot.__post_init__")
        self.timestamp = ensure_aware_utc(self.timestamp)
        self.confidence = clamp(self.confidence, 0.0, 1.0)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering RegimeSnapshot.validate")
        if not self.symbol.strip():
            raise ValidationError("RegimeSnapshot.symbol cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError("RegimeSnapshot.confidence must be between 0.0 and 1.0")


@dataclass(slots=True)
class PriceSnapshot:
    _logger = logging.getLogger(__name__ + ".PriceSnapshot")
    symbol: str
    last_price: float | None = None
    bid: float | None = None
    ask: float | None = None
    mark_price: float | None = None
    index_price: float | None = None
    spread_bps: float | None = None
    timestamp: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceSnapshot.__post_init__")
        self.timestamp = ensure_aware_utc(self.timestamp)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceSnapshot.validate")
        if not self.symbol.strip():
            raise ValidationError("PriceSnapshot.symbol cannot be empty")

        for field_name, value in {
            "last_price": self.last_price,
            "bid": self.bid,
            "ask": self.ask,
            "mark_price": self.mark_price,
            "index_price": self.index_price,
        }.items():
            if value is not None and value <= 0:
                raise ValidationError(f"PriceSnapshot.{field_name} must be > 0")

        if self.bid is not None and self.ask is not None and self.bid > self.ask:
            raise ValidationError("PriceSnapshot.bid cannot be greater than ask")

        if self.spread_bps is not None and self.spread_bps < 0:
            raise ValidationError("PriceSnapshot.spread_bps must be >= 0")

    @property
    def mid_price(self) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceSnapshot.mid_price")
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.last_price


@dataclass(slots=True)
class SignalContext:
    """
    Read-only market/analytics context consumed by strategies.

    Strategy should read this context and return StrategyEvaluation.
    Strategy should not call analytics/risk/execution modules directly.
    """
    _logger = logging.getLogger(__name__ + ".SignalContext")

    symbol: str
    timestamp: datetime
    timeframe: Timeframe = Timeframe.M1
    price: PriceSnapshot | None = None
    regime: RegimeSnapshot | None = None
    portfolio: PortfolioSnapshot | None = None

    orderflow: dict[str, Any] = field(default_factory=dict)
    liquidity: dict[str, Any] = field(default_factory=dict)
    price_action: dict[str, Any] = field(default_factory=dict)
    liquidations: dict[str, Any] = field(default_factory=dict)
    whales: dict[str, Any] = field(default_factory=dict)
    spoofing: dict[str, Any] = field(default_factory=dict)
    spreads: dict[str, Any] = field(default_factory=dict)
    funding: dict[str, Any] = field(default_factory=dict)
    open_interest: dict[str, Any] = field(default_factory=dict)
    system: dict[str, Any] = field(default_factory=dict)

    feature_map: dict[str, FeatureSnapshot] = field(default_factory=dict)
    freshness_map: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalContext.__post_init__")
        self.timestamp = ensure_aware_utc(self.timestamp)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalContext.validate")
        if not self.symbol.strip():
            raise ValidationError("SignalContext.symbol cannot be empty")

        if self.price is not None:
            self.price.validate()
            if self.price.symbol != self.symbol:
                raise ValidationError("SignalContext.price.symbol must match context.symbol")

        if self.regime is not None:
            self.regime.validate()
            if self.regime.symbol != self.symbol:
                raise ValidationError("SignalContext.regime.symbol must match context.symbol")

        if self.portfolio is not None:
            self.portfolio.validate()

        for feature_name, feature in self.feature_map.items():
            if feature_name != feature.name:
                raise ValidationError(
                    f"SignalContext.feature_map key '{feature_name}' does not match feature.name '{feature.name}'"
                )
            if feature.symbol != self.symbol:
                raise ValidationError(
                    f"FeatureSnapshot.symbol must match context.symbol for feature '{feature_name}'"
                )
            feature.validate()

        for feature_name, ttl in self.freshness_map.items():
            if not feature_name.strip():
                raise ValidationError("SignalContext.freshness_map contains empty feature name")
            if ttl <= 0:
                raise ValidationError(
                    f"SignalContext.freshness_map[{feature_name}] must be > 0"
                )

    @property
    def current_regime(self) -> MarketRegime:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalContext.current_regime")
        if self.regime is None:
            return MarketRegime.UNKNOWN
        return self.regime.regime

    @property
    def mid_price(self) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalContext.mid_price")
        if self.price is None:
            return None
        return self.price.mid_price

    def has_feature(self, name: str) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalContext.has_feature")
        return name in self.feature_map

    def get_feature_snapshot(self, name: str) -> FeatureSnapshot | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalContext.get_feature_snapshot")
        return self.feature_map.get(name)

    def get_feature(self, name: str, default: Any = None) -> Any:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalContext.get_feature")
        snapshot = self.feature_map.get(name)
        if snapshot is None:
            return default
        return snapshot.value

    def get_normalized_feature(
        self,
        name: str,
        default: float | None = None,
    ) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalContext.get_normalized_feature")
        snapshot = self.feature_map.get(name)
        if snapshot is None:
            return default
        return (
            snapshot.normalized_value
            if snapshot.normalized_value is not None
            else default
        )

    def feature_age_seconds(self, name: str) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalContext.feature_age_seconds")
        snapshot = self.feature_map.get(name)
        if snapshot is None:
            return None
        return max(0.0, (self.timestamp - snapshot.timestamp).total_seconds())

    def feature_is_stale(self, name: str) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalContext.feature_is_stale")
        snapshot = self.feature_map.get(name)
        if snapshot is None:
            return True

        ttl = self.freshness_map.get(name)
        if ttl is not None:
            return snapshot.age_seconds(self.timestamp) > ttl

        return snapshot.is_stale(self.timestamp)

    def feature_is_expired(self, name: str) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalContext.feature_is_expired")
        snapshot = self.feature_map.get(name)
        if snapshot is None:
            return True
        return snapshot.is_expired(self.timestamp)

    def put_feature(self, snapshot: FeatureSnapshot) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalContext.put_feature")
        snapshot.validate()

        if snapshot.symbol != self.symbol:
            raise ValidationError(
                f"FeatureSnapshot.symbol '{snapshot.symbol}' does not match context.symbol '{self.symbol}'"
            )

        self.feature_map[snapshot.name] = snapshot

    def put_domain_feature(
        self,
        source: FeatureSource,
        key: str,
        value: Any,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalContext.put_domain_feature")
        if not key.strip():
            raise ValidationError("domain feature key cannot be empty")

        self.domain_dict(source)[key] = value

    def domain_dict(self, source: FeatureSource) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalContext.domain_dict")
        mapping = {
            FeatureSource.ORDERFLOW: self.orderflow,
            FeatureSource.LIQUIDITY: self.liquidity,
            FeatureSource.PRICE_ACTION: self.price_action,
            FeatureSource.LIQUIDATIONS: self.liquidations,
            FeatureSource.WHALES: self.whales,
            FeatureSource.SPOOFING: self.spoofing,
            FeatureSource.SPREADS: self.spreads,
            FeatureSource.FUNDING: self.funding,
            FeatureSource.OPEN_INTEREST: self.open_interest,
            FeatureSource.SYSTEM: self.system,
        }
        return mapping.get(source, {})


@dataclass(slots=True)
class SignalEnvelope:
    _logger = logging.getLogger(__name__ + ".SignalEnvelope")
    signal: StrategySignal
    emitted_at: datetime = field(default_factory=utcnow)
    correlation_id: str | None = None
    trace_id: str | None = None
    source_event: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalEnvelope.__post_init__")
        self.emitted_at = ensure_aware_utc(self.emitted_at)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalEnvelope.validate")
        self.signal.validate()


@dataclass(slots=True)
class CooldownState:
    _logger = logging.getLogger(__name__ + ".CooldownState")
    symbol: str
    strategy_name: str
    until: datetime
    reason: str | None = None

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CooldownState.__post_init__")
        self.until = ensure_aware_utc(self.until)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CooldownState.validate")
        if not self.symbol.strip():
            raise ValidationError("CooldownState.symbol cannot be empty")
        if not self.strategy_name.strip():
            raise ValidationError("CooldownState.strategy_name cannot be empty")

    def is_active(self, now: datetime | None = None) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CooldownState.is_active")
        current = ensure_aware_utc(now or utcnow())
        return current < self.until

    def remaining_seconds(self, now: datetime | None = None) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CooldownState.remaining_seconds")
        current = ensure_aware_utc(now or utcnow())
        return max(0.0, (self.until - current).total_seconds())


@dataclass(slots=True)
class SignalWindow:
    _logger = logging.getLogger(__name__ + ".SignalWindow")
    opened_at: datetime
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalWindow.__post_init__")
        self.opened_at = ensure_aware_utc(self.opened_at)
        if self.expires_at is not None:
            self.expires_at = ensure_aware_utc(self.expires_at)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalWindow.validate")
        if self.expires_at is not None and self.expires_at <= self.opened_at:
            raise ValidationError("SignalWindow.expires_at must be after opened_at")

    def is_expired(self, now: datetime | None = None) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalWindow.is_expired")
        if self.expires_at is None:
            return False
        current = ensure_aware_utc(now or utcnow())
        return current >= self.expires_at

    @classmethod
    def from_seconds(cls, ttl_seconds: int | None) -> SignalWindow:
        _strategy_logger = getattr(cls, "_logger", None) or logging.getLogger(__name__ + ".SignalWindow")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SignalWindow.from_seconds")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValidationError("ttl_seconds must be > 0")

        now = utcnow()
        expires_at = None if ttl_seconds is None else now + timedelta(seconds=ttl_seconds)
        return cls(opened_at=now, expires_at=expires_at)


def confidence_to_grade(confidence: float) -> ConfidenceGrade:
    confidence = clamp(confidence, 0.0, 1.0)
    if confidence >= 0.90:
        return ConfidenceGrade.VERY_HIGH
    if confidence >= 0.75:
        return ConfidenceGrade.HIGH
    if confidence >= 0.55:
        return ConfidenceGrade.MEDIUM
    if confidence >= 0.35:
        return ConfidenceGrade.LOW
    return ConfidenceGrade.VERY_LOW


def confidence_to_strength(confidence: float) -> SignalStrength:
    confidence = clamp(confidence, 0.0, 1.0)
    if confidence >= 0.90:
        return SignalStrength.EXTREME
    if confidence >= 0.75:
        return SignalStrength.STRONG
    if confidence >= 0.50:
        return SignalStrength.MODERATE
    return SignalStrength.WEAK



StrategyContext = SignalContext

__all__ = [
    "utcnow",
    "ensure_aware_utc",
    "clamp",
    "FeatureSnapshot",
    "StrategyMetadata",
    "EntryPlan",
    "TargetPlan",
    "InvalidationPlan",
    "ExitPlan",
    "ExecutionPlanDraft",
    "FilterResult",
    "ConflictRecord",
    "StrategySignal",
    "RawStrategySignal",
    "ConfirmedSignal",
    "TradeIdea",
    "ExecutionCostPayload",
    "RiskReadySignalPayload",
    "StrategyEvaluation",
    "ConfluenceResult",
    "PortfolioSnapshot",
    "RegimeSnapshot",
    "PriceSnapshot",
    "SignalContext",
    "StrategyContext",
    "SignalEnvelope",
    "CooldownState",
    "SignalWindow",
    "confidence_to_grade",
    "confidence_to_strength",
]