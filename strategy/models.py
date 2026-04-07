from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .enums import (
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
    Timeframe,
    TriggerType,
)
from .exceptions import ValidationError


def utcnow() -> datetime:
    return datetime.utcnow()


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


@dataclass(slots=True)
class FeatureSnapshot:
    name: str
    value: Any
    source: FeatureSource
    symbol: str
    timestamp: datetime
    confidence: float = 0.0
    normalized_value: float | None = None
    freshness_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name.strip():
            raise ValidationError("FeatureSnapshot.name cannot be empty")
        if not self.symbol.strip():
            raise ValidationError("FeatureSnapshot.symbol cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError("FeatureSnapshot.confidence must be between 0.0 and 1.0")
        if self.normalized_value is not None and not -1.0 <= self.normalized_value <= 1.0:
            raise ValidationError("FeatureSnapshot.normalized_value must be between -1.0 and 1.0")

    def age_seconds(self, now: datetime | None = None) -> float:
        current = now or utcnow()
        return (current - self.timestamp).total_seconds()

    def freshness_status(self, now: datetime | None = None) -> FreshnessStatus:
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
        return self.freshness_status(now) in {FreshnessStatus.STALE, FreshnessStatus.EXPIRED}

    def is_expired(self, now: datetime | None = None) -> bool:
        return self.freshness_status(now) == FreshnessStatus.EXPIRED


@dataclass(slots=True)
class StrategyMetadata:
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
        if not self.strategy_name.strip():
            raise ValidationError("StrategyMetadata.strategy_name cannot be empty")


@dataclass(slots=True)
class EntryPlan:
    entry_type: EntryType
    price: float | None = None
    timeout_seconds: int | None = None
    max_slippage_bps: float | None = None
    confirmation_required: bool = False
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.price is not None and self.price <= 0:
            raise ValidationError("EntryPlan.price must be > 0")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValidationError("EntryPlan.timeout_seconds must be > 0")
        if self.max_slippage_bps is not None and self.max_slippage_bps < 0:
            raise ValidationError("EntryPlan.max_slippage_bps must be >= 0")


@dataclass(slots=True)
class TargetPlan:
    price: float
    size_fraction: float = 1.0
    rr: float | None = None
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.price <= 0:
            raise ValidationError("TargetPlan.price must be > 0")
        if not 0 < self.size_fraction <= 1:
            raise ValidationError("TargetPlan.size_fraction must be in (0, 1]")
        if self.rr is not None and self.rr <= 0:
            raise ValidationError("TargetPlan.rr must be > 0")


@dataclass(slots=True)
class InvalidationPlan:
    price: float | None = None
    reason: str | None = None
    timeout_seconds: int | None = None
    conditions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.price is not None and self.price <= 0:
            raise ValidationError("InvalidationPlan.price must be > 0")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValidationError("InvalidationPlan.timeout_seconds must be > 0")


@dataclass(slots=True)
class ExitPlan:
    exit_types: list[ExitType] = field(default_factory=list)
    stop_loss: float | None = None
    take_profit_levels: list[TargetPlan] = field(default_factory=list)
    trailing_distance: float | None = None
    max_holding_seconds: int | None = None
    partial_exit_enabled: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
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
    name: str
    decision: FilterDecision
    reason: str | None = None
    score_impact: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name.strip():
            raise ValidationError("FilterResult.name cannot be empty")

    @property
    def passed(self) -> bool:
        return self.decision in {FilterDecision.PASS, FilterDecision.WARN}

    @property
    def blocked(self) -> bool:
        return self.decision == FilterDecision.BLOCK


@dataclass(slots=True)
class ConflictRecord:
    conflict_type: ConflictType
    source: str
    message: str
    penalty: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.source.strip():
            raise ValidationError("ConflictRecord.source cannot be empty")
        if not self.message.strip():
            raise ValidationError("ConflictRecord.message cannot be empty")
        if self.penalty < 0:
            raise ValidationError("ConflictRecord.penalty must be >= 0")


@dataclass(slots=True)
class StrategySignal:
    symbol: str
    side: SignalSide
    strategy_name: str
    category: StrategyCategory
    timeframe: Timeframe
    setup_type: SetupType
    timestamp: datetime

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

    def validate(self) -> None:
        if not self.symbol.strip():
            raise ValidationError("StrategySignal.symbol cannot be empty")
        if not self.strategy_name.strip():
            raise ValidationError("StrategySignal.strategy_name cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError("StrategySignal.confidence must be between 0.0 and 1.0")

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
        return self.side == SignalSide.LONG

    @property
    def is_short(self) -> bool:
        return self.side == SignalSide.SHORT

    @property
    def is_directional(self) -> bool:
        return self.side in {SignalSide.LONG, SignalSide.SHORT}

    @property
    def is_active(self) -> bool:
        return self.status in {
            SignalStatus.NEW,
            SignalStatus.PENDING,
            SignalStatus.CONFIRMED,
        }

    @property
    def passed_filters(self) -> bool:
        return all(result.passed for result in self.filter_results)

    @property
    def total_conflict_penalty(self) -> float:
        return sum(conflict.penalty for conflict in self.conflicts)

    def add_reason(self, reason: str) -> None:
        if reason and reason not in self.reasons:
            self.reasons.append(reason)

    def add_confirmation(self, confirmation: str) -> None:
        if confirmation and confirmation not in self.confirmations:
            self.confirmations.append(confirmation)

    def add_source_feature(self, feature_name: str) -> None:
        if feature_name and feature_name not in self.source_features:
            self.source_features.append(feature_name)

    def add_filter_result(self, filter_result: FilterResult) -> None:
        filter_result.validate()
        self.filter_results.append(filter_result)

    def add_conflict(self, conflict: ConflictRecord) -> None:
        conflict.validate()
        self.conflicts.append(conflict)

    def to_confirmed(self) -> None:
        self.status = SignalStatus.CONFIRMED

    def to_rejected(self) -> None:
        self.status = SignalStatus.REJECTED

    def to_cancelled(self) -> None:
        self.status = SignalStatus.CANCELLED

    def to_expired(self) -> None:
        self.status = SignalStatus.EXPIRED


@dataclass(slots=True)
class RawStrategySignal:
    signal: StrategySignal
    raw_inputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.signal.validate()


@dataclass(slots=True)
class ConfirmedSignal:
    signal: StrategySignal
    accepted_by_confluence: bool = False
    accepted_by_filters: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.signal.validate()


@dataclass(slots=True)
class TradeIdea:
    signal: StrategySignal
    execution_plan: ExecutionPlanDraft
    created_at: datetime = field(default_factory=utcnow)
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.signal.validate()
        self.execution_plan.validate()

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        current = now or utcnow()
        return current >= self.expires_at


@dataclass(slots=True)
class StrategyEvaluation:
    strategy_name: str
    symbol: str
    timestamp: datetime
    signal: StrategySignal | None = None
    passed: bool = False
    score: float = 0.0
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
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

    def validate(self) -> None:
        if not self.symbol.strip():
            raise ValidationError("ConfluenceResult.symbol cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError("ConfluenceResult.confidence must be between 0.0 and 1.0")
        for conflict in self.conflicts:
            conflict.validate()

    @property
    def total_conflict_penalty(self) -> float:
        return sum(conflict.penalty for conflict in self.conflicts)

    def add_reason(self, reason: str) -> None:
        if reason and reason not in self.reasons:
            self.reasons.append(reason)

    def add_confirmation(self, confirmation: str) -> None:
        if confirmation and confirmation not in self.confirmations:
            self.confirmations.append(confirmation)


@dataclass(slots=True)
class PortfolioSnapshot:
    open_positions: int = 0
    active_signals: int = 0
    symbol_exposure: dict[str, float] = field(default_factory=dict)
    strategy_exposure: dict[str, float] = field(default_factory=dict)
    correlation_groups: dict[str, list[str]] = field(default_factory=dict)
    blocked_symbols: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.open_positions < 0:
            raise ValidationError("PortfolioSnapshot.open_positions must be >= 0")
        if self.active_signals < 0:
            raise ValidationError("PortfolioSnapshot.active_signals must be >= 0")


@dataclass(slots=True)
class RegimeSnapshot:
    symbol: str
    regime: MarketRegime = MarketRegime.UNKNOWN
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=utcnow)
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.symbol.strip():
            raise ValidationError("RegimeSnapshot.symbol cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError("RegimeSnapshot.confidence must be between 0.0 and 1.0")


@dataclass(slots=True)
class PriceSnapshot:
    symbol: str
    last_price: float | None = None
    bid: float | None = None
    ask: float | None = None
    mark_price: float | None = None
    index_price: float | None = None
    spread_bps: float | None = None
    timestamp: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
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
        if self.spread_bps is not None and self.spread_bps < 0:
            raise ValidationError("PriceSnapshot.spread_bps must be >= 0")

    @property
    def mid_price(self) -> float | None:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.last_price


@dataclass(slots=True)
class SignalContext:
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

    feature_map: dict[str, FeatureSnapshot] = field(default_factory=dict)
    freshness_map: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.symbol.strip():
            raise ValidationError("SignalContext.symbol cannot be empty")
        if self.price is not None:
            self.price.validate()
        if self.regime is not None:
            self.regime.validate()
        if self.portfolio is not None:
            self.portfolio.validate()
        for feature in self.feature_map.values():
            feature.validate()

    def has_feature(self, name: str) -> bool:
        return name in self.feature_map

    def get_feature_snapshot(self, name: str) -> FeatureSnapshot | None:
        return self.feature_map.get(name)

    def get_feature(self, name: str, default: Any = None) -> Any:
        snapshot = self.feature_map.get(name)
        if snapshot is None:
            return default
        return snapshot.value

    def get_normalized_feature(self, name: str, default: float | None = None) -> float | None:
        snapshot = self.feature_map.get(name)
        if snapshot is None:
            return default
        return snapshot.normalized_value if snapshot.normalized_value is not None else default

    def feature_age_seconds(self, name: str) -> float | None:
        snapshot = self.feature_map.get(name)
        if snapshot is None:
            return None
        return (self.timestamp - snapshot.timestamp).total_seconds()

    def feature_is_stale(self, name: str) -> bool:
        snapshot = self.feature_map.get(name)
        if snapshot is None:
            return True
        return snapshot.is_stale(self.timestamp)

    def put_feature(self, snapshot: FeatureSnapshot) -> None:
        snapshot.validate()
        self.feature_map[snapshot.name] = snapshot

    def domain_dict(self, source: FeatureSource) -> dict[str, Any]:
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
        }
        return mapping.get(source, {})


@dataclass(slots=True)
class SignalEnvelope:
    signal: StrategySignal
    emitted_at: datetime = field(default_factory=utcnow)
    correlation_id: str | None = None
    trace_id: str | None = None
    source_event: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.signal.validate()


@dataclass(slots=True)
class CooldownState:
    symbol: str
    strategy_name: str
    until: datetime
    reason: str | None = None

    def validate(self) -> None:
        if not self.symbol.strip():
            raise ValidationError("CooldownState.symbol cannot be empty")
        if not self.strategy_name.strip():
            raise ValidationError("CooldownState.strategy_name cannot be empty")

    def is_active(self, now: datetime | None = None) -> bool:
        current = now or utcnow()
        return current < self.until

    def remaining_seconds(self, now: datetime | None = None) -> float:
        current = now or utcnow()
        return max(0.0, (self.until - current).total_seconds())


@dataclass(slots=True)
class SignalWindow:
    opened_at: datetime
    expires_at: datetime | None = None

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        current = now or utcnow()
        return current >= self.expires_at

    @classmethod
    def from_seconds(cls, ttl_seconds: int | None) -> "SignalWindow":
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