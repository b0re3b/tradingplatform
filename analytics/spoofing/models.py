from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .enums import (
    DetectorDecision,
    LiquidityEventType,
    OrderbookWallState,
    ScoreComponent,
    SpoofingComponent,
    SpoofingPattern,
    SpoofingSeverity,
    SpoofingSide,
    SpoofingStatus,
    SpoofingType,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class OrderbookLevelSnapshot:
    """
    Знімок конкретного рівня стакана.
    """
    symbol: str
    exchange: str
    side: SpoofingSide
    price: float
    size: float
    best_bid: float | None = None
    best_ask: float | None = None
    mid_price: float | None = None
    spread: float | None = None
    sequence_id: int | None = None
    timestamp: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def level_key(self) -> str:
        return f"{self.exchange}:{self.symbol}:{self.side.value}:{self.price:.12f}"


@dataclass(slots=True)
class TrackedWall:
    """
    Внутрішня модель життєвого циклу великої стінки в стакані.
    """
    wall_id: str
    symbol: str
    exchange: str
    side: SpoofingSide
    price: float

    first_seen_at: datetime
    last_seen_at: datetime

    initial_size: float
    current_size: float
    max_size: float
    min_size: float

    best_bid_at_creation: float | None = None
    best_ask_at_creation: float | None = None
    mid_price_at_creation: float | None = None

    total_added_size: float = 0.0
    total_removed_size: float = 0.0
    estimated_filled_size: float = 0.0
    estimated_pulled_size: float = 0.0

    updates_count: int = 0
    touch_count: int = 0
    near_touch_count: int = 0

    state: OrderbookWallState = OrderbookWallState.ACTIVE
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def lifetime_ms(self) -> float:
        return max(0.0, (self.last_seen_at - self.first_seen_at).total_seconds() * 1000.0)

    @property
    def fill_ratio(self) -> float:
        if self.max_size <= 0:
            return 0.0
        return max(0.0, min(1.0, self.estimated_filled_size / self.max_size))

    @property
    def pull_ratio(self) -> float:
        if self.max_size <= 0:
            return 0.0
        return max(0.0, min(1.0, self.estimated_pulled_size / self.max_size))

    @property
    def current_to_max_ratio(self) -> float:
        if self.max_size <= 0:
            return 0.0
        return max(0.0, min(1.0, self.current_size / self.max_size))

    @property
    def size_delta(self) -> float:
        return self.current_size - self.initial_size


@dataclass(slots=True)
class LiquidityLifecycleEvent:
    """
    Подія життєвого циклу стінки/ліквідності.
    """
    wall_id: str
    symbol: str
    exchange: str
    side: SpoofingSide
    event_type: LiquidityEventType
    price: float
    size_before: float
    size_after: float
    delta_size: float
    timestamp: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SpoofingFeatures:
    """
    Ознаки, з яких формується рішення й score.
    """
    symbol: str
    exchange: str
    side: SpoofingSide
    price: float

    wall_size: float = 0.0
    wall_size_ratio: float = 0.0
    distance_from_mid_bps: float = 0.0

    lifetime_ms: float = 0.0
    updates_count: int = 0
    repetition_count: int = 0

    fill_ratio: float = 0.0
    pull_ratio: float = 0.0
    cancel_to_fill_ratio: float = 0.0

    price_reaction_bps: float = 0.0
    pressure_flip_strength: float = 0.0
    layering_score: float = 0.0

    is_near_best_quote: bool = False
    is_fast_pull: bool = False
    is_fake_liquidity: bool = False
    is_layering: bool = False

    timestamp: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DetectorResult:
    """
    Уніфікований результат окремого детектора.
    """
    detector: SpoofingComponent
    decision: DetectorDecision
    score: float
    confidence: float
    reason: str
    features: SpoofingFeatures | None = None
    wall_id: str | None = None
    pattern: SpoofingPattern = SpoofingPattern.UNKNOWN
    timestamp: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_positive(self) -> bool:
        return self.decision == DetectorDecision.POSITIVE


@dataclass(slots=True)
class ScoreContribution:
    """
    Внесок окремої ознаки в загальний score.
    """
    component: ScoreComponent
    raw_value: float
    normalized_value: float
    weight: float
    contribution: float
    description: str = ""


@dataclass(slots=True)
class SpoofingScore:
    """
    Підсумковий score із деталізацією.
    """
    total_score: float
    confidence: float
    severity: SpoofingSeverity
    contributions: list[ScoreContribution] = field(default_factory=list)
    threshold: float = 0.0
    passed: bool = False
    timestamp: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SpoofingSignal:
    """
    Фінальний сигнал spoofing, який можна публікувати в EventBus.
    """
    signal_id: str
    symbol: str
    exchange: str
    side: SpoofingSide
    spoofing_type: SpoofingType
    pattern: SpoofingPattern
    status: SpoofingStatus

    price_level: float
    wall_id: str | None

    score: float
    confidence: float
    severity: SpoofingSeverity

    first_seen_at: datetime
    detected_at: datetime = field(default_factory=utc_now)

    features: SpoofingFeatures | None = None
    detector_results: list[DetectorResult] = field(default_factory=list)
    score_breakdown: SpoofingScore | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnalyzerOutput:
    """
    Результат роботи analyzer.py за один цикл обробки.
    """
    symbol: str
    exchange: str
    signal: SpoofingSignal | None
    detector_results: list[DetectorResult] = field(default_factory=list)
    tracked_walls: list[TrackedWall] = field(default_factory=list)
    timestamp: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)