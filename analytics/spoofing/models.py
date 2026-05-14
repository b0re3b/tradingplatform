from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .enums import (
    CandidateStatus,
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
    TradeSide,
)


def utc_now() -> datetime:
    """
    Єдина точка для UTC timestamps у spoofing-пакеті.
    """
    return datetime.now(timezone.utc)


# =============================================================================
# Raw / normalized market models
# =============================================================================


@dataclass(slots=True)
class OrderbookLevel:
    """
    Простий рівень стакана.

    Використовується як lightweight model для raw orderbook payload-ів,
    перш ніж рівень буде перетворено в OrderbookLevelSnapshot.
    """

    price: float
    size: float


# Backward-compatible alias for legacy naming from old spoofing_detector.py.
OrderBookLevel = OrderbookLevel


@dataclass(slots=True)
class TradeTick:
    """
    Нормалізована trade-flow подія.

    Замість raw string side використовує TradeSide, щоб detector-и не
    працювали з невалідованими значеннями "buy" / "sell".
    """

    symbol: str
    price: float
    qty: float
    side: TradeSide
    ts_ms: int
    exchange: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrderbookLevelSnapshot:
    """
    Нормалізований знімок конкретного рівня стакана.

    Це основна input-модель для OrderbookWallDetector.
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


# =============================================================================
# Stateful lifecycle models
# =============================================================================


@dataclass(slots=True)
class TrackedWall:
    """
    Внутрішня модель життєвого циклу великої стінки в стакані.

    Створюється та оновлюється PersistenceTracker.
    Detector-и працюють уже з цією stateful-моделлю, а не з raw orderbook.
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
        return max(
            0.0,
            (self.last_seen_at - self.first_seen_at).total_seconds() * 1000.0,
        )

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
class SpoofingCandidate:
    """
    Внутрішній legacy-compatible кандидат spoofing-події.

    У новій архітектурі основним state-класом є TrackedWall, а фінальним
    результатом — SpoofingSignal. Ця модель залишена для міграції старої
    логіки з spoofing_detector.py, якщо потрібно тимчасово підтримати
    candidate-based flow.
    """

    candidate_id: str
    symbol: str
    side: SpoofingSide
    price: float

    initial_size: float
    peak_size: float

    detected_ts_ms: int
    last_seen_ts_ms: int

    best_bid_at_detection: float
    best_ask_at_detection: float
    mid_at_detection: float

    avg_same_side_size_at_detection: float
    distance_bps_at_detection: float
    size_multiple_at_detection: float

    exchange: str | None = None
    status: CandidateStatus = CandidateStatus.ACTIVE

    removed_ts_ms: int | None = None
    remaining_size: float | None = None
    cancel_ratio: float | None = None

    confirmation_ts_ms: int | None = None
    confirmation_price_move_bps: float | None = None
    opposite_pressure_ratio: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """
        Backward-compatible accessor for old code that used candidate.id.
        """
        return self.candidate_id


@dataclass(slots=True)
class LiquidityLifecycleEvent:
    """
    Подія життєвого циклу стінки/ліквідності.

    Генерується PersistenceTracker під час створення, оновлення, pull,
    fill або expiry tracked wall.
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


# =============================================================================
# Detector features and results
# =============================================================================


@dataclass(slots=True)
class SpoofingFeatures:
    """
    Уніфікований набір ознак, з яких формується detector decision і score.
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
    Уніфікований результат окремого detector-а.
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


# =============================================================================
# Detector-local context models moved here from detector files
# =============================================================================


@dataclass(slots=True)
class WallCandidateContext:
    """
    Контекст оцінки конкретного orderbook level як потенційної стінки.
    """

    snapshot: OrderbookLevelSnapshot
    baseline_size: float
    size_ratio: float
    distance_from_mid_bps: float
    near_best_quote: bool
    notional: float
    confidence: float
    score: float
    reason: str


@dataclass(slots=True)
class PullCandidateContext:
    """
    Контекст оцінки tracked wall як pull-event кандидата.
    """

    wall: TrackedWall
    pulled_notional: float
    pulled_size_ratio: float
    fill_ratio: float
    pull_ratio: float
    lifetime_ms: float
    is_fast_pull: bool
    is_strong_pull: bool
    confidence: float
    score: float
    reason: str


@dataclass(slots=True)
class FakeLiquidityCandidateContext:
    """
    Контекст оцінки tracked wall як fake-liquidity кандидата.
    """

    wall: TrackedWall
    wall_notional: float
    pulled_notional: float
    lifetime_ms: float
    fill_ratio: float
    pull_ratio: float
    price_reaction_bps: float
    distance_from_mid_bps: float
    is_short_lived: bool
    is_low_fill: bool
    is_high_pull: bool
    has_market_reaction: bool
    confidence: float
    score: float
    reason: str


@dataclass(slots=True)
class FlipPressureCandidateContext:
    """
    Контекст оцінки tracked wall як pressure-flip / pressure-bluff кандидата.
    """

    wall: TrackedWall
    wall_notional: float
    pulled_notional: float
    lifetime_ms: float
    fill_ratio: float
    pull_ratio: float
    price_reaction_bps: float
    pressure_flip_strength: float
    distance_from_mid_bps: float
    is_pressure_removed: bool
    is_short_lived: bool
    is_low_fill: bool
    has_reversal: bool
    confidence: float
    score: float
    reason: str


@dataclass(slots=True)
class LayeringCluster:
    """
    Кластер потенційного multi-level layering патерну.
    """

    exchange: str
    symbol: str
    side: SpoofingSide
    walls: list[TrackedWall]
    total_notional: float
    average_pull_ratio: float
    average_fill_ratio: float
    average_lifetime_ms: float
    synchronized_pull_ratio: float
    price_span_bps: float
    layering_score: float
    cluster_price: float
    cluster_wall_id: str | None


@dataclass(slots=True)
class LayeringCandidateContext:
    """
    Контекст оцінки LayeringCluster як detector result.
    """

    cluster: LayeringCluster
    confidence: float
    score: float
    reason: str
    price_reaction_bps: float


# =============================================================================
# Scoring / signal models
# =============================================================================


@dataclass(slots=True)
class ScoreContribution:
    """
    Внесок окремої ознаки в загальний spoofing score.
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
class AggregationContext:
    """
    Агрегований контекст перед побудовою фінального score/signal.
    """

    symbol: str
    exchange: str
    price: float
    features: SpoofingFeatures
    detector_results: list[DetectorResult]
    agreement_ratio: float
    average_confidence: float
    primary_pattern: SpoofingPattern
    spoofing_type: SpoofingType
    wall_id: str | None


@dataclass(slots=True)
class SpoofingSignal:
    """
    Фінальний spoofing-сигнал, який analyzer може публікувати через EventBus.
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
    Результат роботи SpoofingAnalyzer за один цикл обробки.
    """

    symbol: str
    exchange: str
    signal: SpoofingSignal | None
    detector_results: list[DetectorResult] = field(default_factory=list)
    tracked_walls: list[TrackedWall] = field(default_factory=list)
    lifecycle_events: list[LiquidityLifecycleEvent] = field(default_factory=list)
    timestamp: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "utc_now",

    # raw / normalized market models
    "OrderbookLevel",
    "OrderBookLevel",
    "TradeTick",
    "OrderbookLevelSnapshot",

    # lifecycle models
    "TrackedWall",
    "SpoofingCandidate",
    "LiquidityLifecycleEvent",

    # detector features/results
    "SpoofingFeatures",
    "DetectorResult",

    # detector contexts
    "WallCandidateContext",
    "PullCandidateContext",
    "FakeLiquidityCandidateContext",
    "FlipPressureCandidateContext",
    "LayeringCluster",
    "LayeringCandidateContext",

    # scoring/signal models
    "ScoreContribution",
    "SpoofingScore",
    "AggregationContext",
    "SpoofingSignal",
    "AnalyzerOutput",
]