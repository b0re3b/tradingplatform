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
    Базова доменна модель liquidity-рівня.

    Використовується для:
    - equal highs / equal lows;
    - swing highs / swing lows;
    - range highs / range lows;
    - external liquidity levels;
    - downstream strategy/dashboard/AI payloads.

    Це чиста модель:
    - не має EventBus;
    - не має Scheduler;
    - не має logger;
    - не виконує IO;
    - містить тільки стан і прості domain-helper методи.
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
    invalidated_at: datetime | None = None
    expired_at: datetime | None = None

    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = self._clamp01(self.confidence)
        self.touches_count = max(0, self.touches_count)
        self.reaction_count = max(0, self.reaction_count)

    @property
    def key(self) -> str:
        """
        Стабільний lightweight key для deduplication/cache/state.
        """
        return (
            f"{self.symbol}:"
            f"{self.timeframe}:"
            f"{self.level_type.value}:"
            f"{self.side.value}:"
            f"{self.price:.12f}"
        )

    @property
    def is_buy_side(self) -> bool:
        return self.side == LiquiditySide.BUY_SIDE

    @property
    def is_sell_side(self) -> bool:
        return self.side == LiquiditySide.SELL_SIDE

    def is_active(self) -> bool:
        return self.status == LiquidityStatus.ACTIVE

    def is_swept(self) -> bool:
        return self.sweep_status == SweepStatus.SWEPT

    def is_partially_swept(self) -> bool:
        return self.sweep_status == SweepStatus.PARTIALLY_SWEPT

    def is_terminal(self) -> bool:
        return self.status in {
            LiquidityStatus.SWEPT,
            LiquidityStatus.INVALIDATED,
            LiquidityStatus.EXPIRED,
        }

    def mark_swept(self, swept_at: datetime | None = None) -> None:
        self.sweep_status = SweepStatus.SWEPT
        self.status = LiquidityStatus.SWEPT
        self.swept_at = swept_at

    def mark_partially_swept(self, swept_at: datetime | None = None) -> None:
        self.sweep_status = SweepStatus.PARTIALLY_SWEPT
        if self.status != LiquidityStatus.INVALIDATED:
            self.status = LiquidityStatus.ACTIVE
        self.swept_at = swept_at

    def mark_invalidated(self, invalidated_at: datetime | None = None) -> None:
        self.status = LiquidityStatus.INVALIDATED
        self.invalidated_at = invalidated_at

    def mark_expired(self, expired_at: datetime | None = None) -> None:
        self.status = LiquidityStatus.EXPIRED
        self.expired_at = expired_at

    def mark_weak(self) -> None:
        if not self.is_terminal():
            self.status = LiquidityStatus.WEAK

    def touch(self, seen_at: datetime | None = None) -> None:
        self.touches_count += 1
        self.last_seen_at = seen_at or self.last_seen_at

        if self.first_seen_at is None:
            self.first_seen_at = self.last_seen_at

    def register_reaction(self, seen_at: datetime | None = None) -> None:
        self.reaction_count += 1
        self.last_seen_at = seen_at or self.last_seen_at

        if self.first_seen_at is None:
            self.first_seen_at = self.last_seen_at

    def distance_pct(self, current_price: float) -> float:
        if current_price == 0:
            return 0.0
        return abs(self.price - current_price) / abs(current_price)

    def to_event_payload(self) -> dict[str, Any]:
        """
        Payload-safe представлення для EventBus events / dashboard / storage.
        """
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "level_type": self.level_type.value,
            "side": self.side.value,
            "price": self.price,
            "status": self.status.value,
            "sweep_status": self.sweep_status.value,
            "confidence": self.confidence,
            "touches_count": self.touches_count,
            "reaction_count": self.reaction_count,
            "first_seen_at": self.first_seen_at.isoformat() if self.first_seen_at else None,
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
            "swept_at": self.swept_at.isoformat() if self.swept_at else None,
            "invalidated_at": self.invalidated_at.isoformat() if self.invalidated_at else None,
            "expired_at": self.expired_at.isoformat() if self.expired_at else None,
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(float(value), 1.0))


@dataclass(slots=True)
class EqualLevel(LiquidityLevel):
    """
    Рівень типу equal highs / equal lows.

    Наслідує службові поля з LiquidityLevel:
    - status;
    - sweep_status;
    - confidence;
    - touches_count;
    - reaction_count;
    - timestamps;
    - metadata.
    """

    tolerance_pct: float = 0.0
    cluster_low: float | None = None
    cluster_high: float | None = None
    level_prices: list[float] = field(default_factory=list)
    pivot_indexes: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()

        self.tolerance_pct = max(0.0, float(self.tolerance_pct))

        if self.cluster_low is None:
            self.cluster_low = self.price

        if self.cluster_high is None:
            self.cluster_high = self.price

        if self.cluster_low > self.cluster_high:
            self.cluster_low, self.cluster_high = self.cluster_high, self.cluster_low

        if not self.level_prices:
            self.level_prices = [self.price]

        if self.touches_count <= 0:
            self.touches_count = len(self.level_prices)

    @property
    def cluster_width(self) -> float:
        if self.cluster_low is None or self.cluster_high is None:
            return 0.0
        return max(0.0, self.cluster_high - self.cluster_low)

    @property
    def cluster_midpoint(self) -> float:
        if self.cluster_low is None or self.cluster_high is None:
            return self.price
        return (self.cluster_low + self.cluster_high) / 2.0

    def contains_price(self, price: float) -> bool:
        if self.cluster_low is None or self.cluster_high is None:
            return self.price == price
        return self.cluster_low <= price <= self.cluster_high

    def to_event_payload(self) -> dict[str, Any]:
        payload = super().to_event_payload()
        payload.update(
            {
                "tolerance_pct": self.tolerance_pct,
                "cluster_low": self.cluster_low,
                "cluster_high": self.cluster_high,
                "cluster_width": self.cluster_width,
                "level_prices": list(self.level_prices),
                "pivot_indexes": list(self.pivot_indexes),
            }
        )
        return payload


@dataclass(slots=True)
class StopCluster:
    """
    Оцінений кластер стопів навколо очевидного liquidity-рівня.

    Зазвичай будується з:
    - equal highs / lows;
    - swing highs / lows;
    - range boundaries;
    - external levels, якщо вони передані в LiquidityMap.
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
    swept_at: datetime | None = None

    source_levels: list[LiquidityLevel] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.low_price > self.high_price:
            self.low_price, self.high_price = self.high_price, self.low_price

        self.center_price = self.center_price or self._midpoint(self.low_price, self.high_price)
        self.confidence = self._clamp01(self.confidence)
        self.estimated_stop_density = self._clamp01(self.estimated_stop_density)
        self.touches_count = max(0, self.touches_count)

    @property
    def key(self) -> str:
        return (
            f"{self.symbol}:"
            f"{self.timeframe}:"
            f"{self.side.value}:"
            f"{self.low_price:.12f}:"
            f"{self.high_price:.12f}"
        )

    @property
    def is_buy_side(self) -> bool:
        return self.side == LiquiditySide.BUY_SIDE

    @property
    def is_sell_side(self) -> bool:
        return self.side == LiquiditySide.SELL_SIDE

    def width(self) -> float:
        return max(0.0, self.high_price - self.low_price)

    def width_pct(self) -> float:
        if self.center_price == 0:
            return 0.0
        return self.width() / abs(self.center_price)

    def contains_price(self, price: float) -> bool:
        return self.low_price <= price <= self.high_price

    def overlaps(self, other: StopCluster) -> bool:
        return not (
            self.high_price < other.low_price
            or other.high_price < self.low_price
        )

    def distance_pct(self, current_price: float) -> float:
        if current_price == 0:
            return 0.0
        return abs(self.center_price - current_price) / abs(current_price)

    def mark_invalidated(self, invalidated_at: datetime | None = None) -> None:
        self.invalidated_at = invalidated_at

    def mark_swept(self, swept_at: datetime | None = None) -> None:
        self.swept_at = swept_at

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "side": self.side.value,
            "low_price": self.low_price,
            "high_price": self.high_price,
            "center_price": self.center_price,
            "width": self.width(),
            "width_pct": self.width_pct(),
            "confidence": self.confidence,
            "estimated_stop_density": self.estimated_stop_density,
            "touches_count": self.touches_count,
            "source_level_type": self.source_level_type.value,
            "strength": self.strength.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "invalidated_at": self.invalidated_at.isoformat() if self.invalidated_at else None,
            "swept_at": self.swept_at.isoformat() if self.swept_at else None,
            "source_levels_count": len(self.source_levels),
            "source_levels": [
                level.to_event_payload()
                for level in self.source_levels
            ],
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def _midpoint(low: float, high: float) -> float:
        return (low + high) / 2.0

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(float(value), 1.0))


@dataclass(slots=True)
class LiquidityZone:
    """
    Узагальнена liquidity-зона.

    Використовується для:
    - dashboard heatmap;
    - strategy context;
    - AI explanation;
    - compact snapshot payloads.
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

    def __post_init__(self) -> None:
        if self.low_price > self.high_price:
            self.low_price, self.high_price = self.high_price, self.low_price

        self.score = max(0.0, min(float(self.score), 1.0))

    @property
    def center_price(self) -> float:
        return (self.low_price + self.high_price) / 2.0

    def width(self) -> float:
        return max(0.0, self.high_price - self.low_price)

    def width_pct(self) -> float:
        if self.center_price == 0:
            return 0.0
        return self.width() / abs(self.center_price)

    def contains_price(self, price: float) -> bool:
        return self.low_price <= price <= self.high_price

    def distance_pct(self, current_price: float) -> float:
        if current_price == 0:
            return 0.0
        return abs(self.center_price - current_price) / abs(current_price)

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "side": self.side.value,
            "low_price": self.low_price,
            "high_price": self.high_price,
            "center_price": self.center_price,
            "width": self.width(),
            "width_pct": self.width_pct(),
            "score": self.score,
            "label": self.label,
            "source_types": [level_type.value for level_type in self.source_types],
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class LiquiditySignal:
    """
    Підсумковий liquidity-сигнал для strategy layer.

    Це не trade-сигнал сам по собі.
    Він описує liquidity context, який потім може бути використаний
    StrategyEngine / ConfluenceEngine для формування signal.generated.
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

    confidence: float = 0.0
    explanation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sweep_risk_up = self._clamp01(self.sweep_risk_up)
        self.sweep_risk_down = self._clamp01(self.sweep_risk_down)
        self.magnet_score_up = self._clamp01(self.magnet_score_up)
        self.magnet_score_down = self._clamp01(self.magnet_score_down)
        self.confidence = self._clamp01(self.confidence)

    @property
    def is_directional(self) -> bool:
        return self.bias in {LiquidityBias.UP, LiquidityBias.DOWN}

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat(),
            "bias": self.bias.value,
            "sweep_risk_up": self.sweep_risk_up,
            "sweep_risk_down": self.sweep_risk_down,
            "magnet_score_up": self.magnet_score_up,
            "magnet_score_down": self.magnet_score_down,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "nearest_buy_side_liquidity": self._liquidity_payload(
                self.nearest_buy_side_liquidity
            ),
            "nearest_sell_side_liquidity": self._liquidity_payload(
                self.nearest_sell_side_liquidity
            ),
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def _liquidity_payload(
        value: LiquidityLevel | StopCluster | None,
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        return value.to_event_payload()

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(float(value), 1.0))


@dataclass(slots=True)
class LiquidityMapSnapshot:
    """
    Повний snapshot liquidity-карти для symbol + timeframe.

    Створюється LiquidityMap, зберігається в LiquidityState,
    публікується LiquidityService через EventBus як
    analytics.liquidity.map.updated.
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

    def __post_init__(self) -> None:
        self.above_liquidity_score = self._clamp01(self.above_liquidity_score)
        self.below_liquidity_score = self._clamp01(self.below_liquidity_score)
        self.liquidity_pressure_score = self._clamp01(self.liquidity_pressure_score)

    def has_levels(self) -> bool:
        return bool(
            self.active_levels
            or self.equal_levels
            or self.stop_clusters
            or self.zones
        )

    def get_active_levels(self) -> list[LiquidityLevel]:
        return [level for level in self.active_levels if level.is_active()]

    def get_buy_side_levels(self) -> list[LiquidityLevel]:
        return [
            level
            for level in self.active_levels
            if level.side == LiquiditySide.BUY_SIDE
        ]

    def get_sell_side_levels(self) -> list[LiquidityLevel]:
        return [
            level
            for level in self.active_levels
            if level.side == LiquiditySide.SELL_SIDE
        ]

    def get_buy_side_clusters(self) -> list[StopCluster]:
        return [
            cluster
            for cluster in self.stop_clusters
            if cluster.side == LiquiditySide.BUY_SIDE
        ]

    def get_sell_side_clusters(self) -> list[StopCluster]:
        return [
            cluster
            for cluster in self.stop_clusters
            if cluster.side == LiquiditySide.SELL_SIDE
        ]

    def to_event_payload(self) -> dict[str, Any]:
        """
        Payload-safe snapshot для EventBus, dashboard, storage, AI layer.
        """
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat(),
            "current_price": self.current_price,
            "active_levels": [
                level.to_event_payload()
                for level in self.active_levels
            ],
            "equal_levels": [
                level.to_event_payload()
                for level in self.equal_levels
            ],
            "stop_clusters": [
                cluster.to_event_payload()
                for cluster in self.stop_clusters
            ],
            "zones": [
                zone.to_event_payload()
                for zone in self.zones
            ],
            "nearest_above_level": self._liquidity_payload(self.nearest_above_level),
            "nearest_below_level": self._liquidity_payload(self.nearest_below_level),
            "strongest_cluster_above": self._cluster_payload(
                self.strongest_cluster_above
            ),
            "strongest_cluster_below": self._cluster_payload(
                self.strongest_cluster_below
            ),
            "above_liquidity_score": self.above_liquidity_score,
            "below_liquidity_score": self.below_liquidity_score,
            "liquidity_pressure_score": self.liquidity_pressure_score,
            "bias": self.bias.value,
            "signal": self.signal.to_event_payload() if self.signal else None,
            "metadata": {
                **dict(self.metadata),
                "active_levels_count": len(self.active_levels),
                "equal_levels_count": len(self.equal_levels),
                "stop_clusters_count": len(self.stop_clusters),
                "zones_count": len(self.zones),
            },
        }

    @staticmethod
    def _liquidity_payload(
        value: LiquidityLevel | StopCluster | None,
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        return value.to_event_payload()

    @staticmethod
    def _cluster_payload(value: StopCluster | None) -> dict[str, Any] | None:
        if value is None:
            return None
        return value.to_event_payload()

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(float(value), 1.0))