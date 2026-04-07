from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .enums import (
    ClusterStrength,
    LiquidityBias,
    LiquidityLevelType,
    LiquiditySide,
    LiquidityStatus,
    SweepStatus,
)


@dataclass(slots=True)
class LiquidityLevel:
    """
    Базова модель рівня ліквідності.
    """

    symbol: str
    timeframe: str
    level_type: LiquidityLevelType
    side: LiquiditySide
    price: float

    status: LiquidityStatus = LiquidityStatus.ACTIVE
    sweep_status: SweepStatus = SweepStatus.NOT_SWEPT
    confidence: float = 0.0

    touches_count: int = 0
    reaction_count: int = 0

    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None
    swept_at: datetime | None = None

    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_active(self) -> bool:
        return self.status == LiquidityStatus.ACTIVE

    def is_swept(self) -> bool:
        return self.sweep_status == SweepStatus.SWEPT

    def mark_swept(self, swept_at: datetime | None = None) -> None:
        self.sweep_status = SweepStatus.SWEPT
        self.status = LiquidityStatus.SWEPT
        self.swept_at = swept_at

    def mark_invalidated(self) -> None:
        self.status = LiquidityStatus.INVALIDATED


@dataclass
class EqualLevel(LiquidityLevel):
    """
    Рівень типу equal highs / equal lows.
    Наслідує всі службові поля з LiquidityLevel:
    status, sweep_status, confidence, swept_at тощо.
    """

    tolerance_pct: float = 0.0
    cluster_low: float | None = None
    cluster_high: float | None = None
    level_prices: list[float] = field(default_factory=list)
    pivot_indexes: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.cluster_low is None:
            self.cluster_low = self.price
        if self.cluster_high is None:
            self.cluster_high = self.price


@dataclass(slots=True)
class StopCluster:
    """
    Оцінений кластер стопів навколо очевидного liquidity level.
    """

    symbol: str
    timeframe: str
    side: LiquiditySide

    low_price: float
    high_price: float
    center_price: float

    confidence: float = 0.0
    estimated_stop_density: float = 0.0
    touches_count: int = 0
    source_level_type: LiquidityLevelType = LiquidityLevelType.STOP_CLUSTER
    strength: ClusterStrength = ClusterStrength.LOW

    created_at: datetime | None = None
    updated_at: datetime | None = None
    invalidated_at: datetime | None = None

    source_levels: list[LiquidityLevel] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def width(self) -> float:
        return max(0.0, self.high_price - self.low_price)

    def contains_price(self, price: float) -> bool:
        return self.low_price <= price <= self.high_price

    def overlaps(self, other: "StopCluster") -> bool:
        return not (self.high_price < other.low_price or other.high_price < self.low_price)


@dataclass(slots=True)
class LiquidityZone:
    """
    Узагальнена зона ліквідності.
    Може використовуватись для dashboard / strategies / AI-пояснення.
    """

    symbol: str
    timeframe: str
    side: LiquiditySide
    low_price: float
    high_price: float

    score: float = 0.0
    label: str | None = None
    source_types: list[LiquidityLevelType] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def center_price(self) -> float:
        return (self.low_price + self.high_price) / 2.0

    def contains_price(self, price: float) -> bool:
        return self.low_price <= price <= self.high_price


@dataclass(slots=True)
class LiquiditySignal:
    """
    Підсумковий сигнал для strategies engine.
    """

    symbol: str
    timeframe: str
    timestamp: datetime

    bias: LiquidityBias = LiquidityBias.NEUTRAL
    nearest_buy_side_liquidity: LiquidityLevel | StopCluster | None = None
    nearest_sell_side_liquidity: LiquidityLevel | StopCluster | None = None

    sweep_risk_up: float = 0.0
    sweep_risk_down: float = 0.0
    magnet_score_up: float = 0.0
    magnet_score_down: float = 0.0

    explanation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LiquidityMapSnapshot:
    """
    Повний snapshot карти ліквідності для symbol + timeframe.
    """

    symbol: str
    timeframe: str
    timestamp: datetime
    current_price: float

    active_levels: list[LiquidityLevel] = field(default_factory=list)
    equal_levels: list[EqualLevel] = field(default_factory=list)
    stop_clusters: list[StopCluster] = field(default_factory=list)
    zones: list[LiquidityZone] = field(default_factory=list)

    nearest_above_level: LiquidityLevel | StopCluster | None = None
    nearest_below_level: LiquidityLevel | StopCluster | None = None

    strongest_cluster_above: StopCluster | None = None
    strongest_cluster_below: StopCluster | None = None

    above_liquidity_score: float = 0.0
    below_liquidity_score: float = 0.0
    liquidity_pressure_score: float = 0.0

    bias: LiquidityBias = LiquidityBias.NEUTRAL
    signal: LiquiditySignal | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def has_levels(self) -> bool:
        return bool(self.active_levels or self.equal_levels or self.stop_clusters)

    def get_active_levels(self) -> list[LiquidityLevel]:
        return [level for level in self.active_levels if level.is_active()]