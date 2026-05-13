from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .enums import (
    ClusterStrength,
    LiquidityBias,
    LiquidityLevelType,
    LiquiditySide,
    LiquidityStatus,
    SweepStatus,
)


def _utcnow() -> datetime:
    """
    Timezone-aware UTC timestamp for model state transitions.

    The models module intentionally stays free of core dependencies:
    no EventBus, no Scheduler, no logger, no IO.
    """
    return datetime.now(timezone.utc)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: Any) -> float:
    return max(0.0, min(_safe_float(value), 1.0))


def _clamp_signed(value: Any) -> float:
    """
    Clamp signed scores to [-1.0, 1.0].

    Used for liquidity_pressure_score where direction must be preserved:
    - positive: upside / buy-side pressure depending on LiquidityMap semantics;
    - negative: downside / sell-side pressure depending on LiquidityMap semantics.
    """
    return max(-1.0, min(_safe_float(value), 1.0))


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
        self.price = _safe_float(self.price)
        self.confidence = _clamp01(self.confidence)
        self.touches_count = max(0, int(self.touches_count))
        self.reaction_count = max(0, int(self.reaction_count))

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

    def is_weak(self) -> bool:
        return self.status == LiquidityStatus.WEAK

    def is_swept(self) -> bool:
        return self.sweep_status == SweepStatus.SWEPT

    def is_partially_swept(self) -> bool:
        return self.sweep_status == SweepStatus.PARTIALLY_SWEPT

    def is_invalidated(self) -> bool:
        return self.status == LiquidityStatus.INVALIDATED

    def is_expired(self) -> bool:
        return self.status == LiquidityStatus.EXPIRED

    def is_terminal(self) -> bool:
        return self.status in {
            LiquidityStatus.SWEPT,
            LiquidityStatus.INVALIDATED,
            LiquidityStatus.EXPIRED,
        }

    def mark_swept(self, swept_at: datetime | None = None) -> None:
        event_ts = swept_at or _utcnow()
        self.sweep_status = SweepStatus.SWEPT
        self.status = LiquidityStatus.SWEPT
        self.swept_at = event_ts
        self.last_seen_at = event_ts

        if self.first_seen_at is None:
            self.first_seen_at = event_ts

    def mark_partially_swept(self, swept_at: datetime | None = None) -> None:
        event_ts = swept_at or _utcnow()
        self.sweep_status = SweepStatus.PARTIALLY_SWEPT

        if self.status != LiquidityStatus.INVALIDATED:
            self.status = LiquidityStatus.ACTIVE

        self.swept_at = event_ts
        self.last_seen_at = event_ts

        if self.first_seen_at is None:
            self.first_seen_at = event_ts

    def mark_invalidated(self, invalidated_at: datetime | None = None) -> None:
        event_ts = invalidated_at or _utcnow()
        self.status = LiquidityStatus.INVALIDATED
        self.invalidated_at = event_ts
        self.last_seen_at = event_ts

    def mark_expired(self, expired_at: datetime | None = None) -> None:
        event_ts = expired_at or _utcnow()
        self.status = LiquidityStatus.EXPIRED
        self.expired_at = event_ts
        self.last_seen_at = event_ts

    def mark_weak(self) -> None:
        if not self.is_terminal():
            self.status = LiquidityStatus.WEAK

    def touch(self, seen_at: datetime | None = None) -> None:
        event_ts = seen_at or _utcnow()
        self.touches_count += 1
        self.last_seen_at = event_ts

        if self.first_seen_at is None:
            self.first_seen_at = event_ts

    def register_reaction(self, seen_at: datetime | None = None) -> None:
        event_ts = seen_at or _utcnow()
        self.reaction_count += 1
        self.last_seen_at = event_ts

        if self.first_seen_at is None:
            self.first_seen_at = event_ts

    def distance_pct(self, current_price: float) -> float:
        current_price = _safe_float(current_price)
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
            "is_active": self.is_active(),
            "is_terminal": self.is_terminal(),
            "metadata": dict(self.metadata),
        }


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
        LiquidityLevel.__post_init__(self)

        self.tolerance_pct = max(0.0, _safe_float(self.tolerance_pct))

        if self.cluster_low is None:
            self.cluster_low = self.price
        else:
            self.cluster_low = _safe_float(self.cluster_low)

        if self.cluster_high is None:
            self.cluster_high = self.price
        else:
            self.cluster_high = _safe_float(self.cluster_high)

        if self.cluster_low > self.cluster_high:
            self.cluster_low, self.cluster_high = self.cluster_high, self.cluster_low

        if not self.level_prices:
            self.level_prices = [self.price]
        else:
            self.level_prices = [_safe_float(price) for price in self.level_prices]

        if self.touches_count <= 0:
            self.touches_count = len(self.level_prices)

        self.pivot_indexes = [max(0, int(index)) for index in self.pivot_indexes]

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
        price = _safe_float(price)
        if self.cluster_low is None or self.cluster_high is None:
            return self.price == price
        return self.cluster_low <= price <= self.cluster_high

    def to_event_payload(self) -> dict[str, Any]:
        payload = LiquidityLevel.to_event_payload(self)

        payload.update(
            {
                "tolerance_pct": self.tolerance_pct,
                "cluster_low": self.cluster_low,
                "cluster_high": self.cluster_high,
                "cluster_width": self.cluster_width,
                "cluster_midpoint": self.cluster_midpoint,
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

    Це чиста domain-модель. Вона не має EventBus, Scheduler, logger або IO.
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
        self.low_price = _safe_float(self.low_price)
        self.high_price = _safe_float(self.high_price)
        self.center_price = _safe_float(self.center_price)

        if self.low_price > self.high_price:
            self.low_price, self.high_price = self.high_price, self.low_price

        if self.center_price <= 0:
            self.center_price = self._midpoint(self.low_price, self.high_price)

        self.confidence = _clamp01(self.confidence)
        self.estimated_stop_density = _clamp01(self.estimated_stop_density)
        self.touches_count = max(0, int(self.touches_count))

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

    def is_swept(self) -> bool:
        return self.swept_at is not None

    def is_invalidated(self) -> bool:
        return self.invalidated_at is not None

    def is_active(self) -> bool:
        return not self.is_invalidated()

    def is_terminal(self) -> bool:
        return self.is_invalidated()

    def width(self) -> float:
        return max(0.0, self.high_price - self.low_price)

    def width_pct(self) -> float:
        if self.center_price == 0:
            return 0.0
        return self.width() / abs(self.center_price)

    def contains_price(self, price: float) -> bool:
        price = _safe_float(price)
        return self.low_price <= price <= self.high_price

    def overlaps(self, other: StopCluster) -> bool:
        return not (
            self.high_price < other.low_price
            or other.high_price < self.low_price
        )

    def distance_pct(self, current_price: float) -> float:
        current_price = _safe_float(current_price)
        if current_price == 0:
            return 0.0
        return abs(self.center_price - current_price) / abs(current_price)

    def mark_invalidated(self, invalidated_at: datetime | None = None) -> None:
        event_ts = invalidated_at or _utcnow()
        self.invalidated_at = event_ts
        self.updated_at = event_ts

    def mark_swept(self, swept_at: datetime | None = None) -> None:
        event_ts = swept_at or _utcnow()
        self.swept_at = event_ts
        self.updated_at = event_ts

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
            "is_active": self.is_active(),
            "is_swept": self.is_swept(),
            "is_terminal": self.is_terminal(),
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
        self.low_price = _safe_float(self.low_price)
        self.high_price = _safe_float(self.high_price)

        if self.low_price > self.high_price:
            self.low_price, self.high_price = self.high_price, self.low_price

        self.score = _clamp01(self.score)

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
        price = _safe_float(price)
        return self.low_price <= price <= self.high_price

    def distance_pct(self, current_price: float) -> float:
        current_price = _safe_float(current_price)
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
        self.sweep_risk_up = _clamp01(self.sweep_risk_up)
        self.sweep_risk_down = _clamp01(self.sweep_risk_down)
        self.magnet_score_up = _clamp01(self.magnet_score_up)
        self.magnet_score_down = _clamp01(self.magnet_score_down)
        self.confidence = _clamp01(self.confidence)

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
        self.current_price = _safe_float(self.current_price)
        self.above_liquidity_score = _clamp01(self.above_liquidity_score)
        self.below_liquidity_score = _clamp01(self.below_liquidity_score)

        # Important: this score is intentionally signed.
        # Do NOT clamp it to [0, 1], otherwise short/downside pressure
        # is lost before strategy layer consumes the snapshot.
        self.liquidity_pressure_score = _clamp_signed(self.liquidity_pressure_score)

    def has_levels(self) -> bool:
        return bool(
            self.active_levels
            or self.equal_levels
            or self.stop_clusters
            or self.zones
        )

    def get_active_levels(self) -> list[LiquidityLevel]:
        return [level for level in self.active_levels if level.is_active()]

    def get_terminal_levels(self) -> list[LiquidityLevel]:
        return [level for level in self.active_levels if level.is_terminal()]

    def get_buy_side_levels(self, *, active_only: bool = True) -> list[LiquidityLevel]:
        levels = self.get_active_levels() if active_only else list(self.active_levels)
        return [level for level in levels if level.side == LiquiditySide.BUY_SIDE]

    def get_sell_side_levels(self, *, active_only: bool = True) -> list[LiquidityLevel]:
        levels = self.get_active_levels() if active_only else list(self.active_levels)
        return [level for level in levels if level.side == LiquiditySide.SELL_SIDE]

    def get_equal_levels(self, *, active_only: bool = False) -> list[EqualLevel]:
        if not active_only:
            return list(self.equal_levels)
        return [level for level in self.equal_levels if level.is_active()]

    def get_buy_side_equal_levels(self, *, active_only: bool = False) -> list[EqualLevel]:
        return [
            level
            for level in self.get_equal_levels(active_only=active_only)
            if level.side == LiquiditySide.BUY_SIDE
        ]

    def get_sell_side_equal_levels(self, *, active_only: bool = False) -> list[EqualLevel]:
        return [
            level
            for level in self.get_equal_levels(active_only=active_only)
            if level.side == LiquiditySide.SELL_SIDE
        ]

    def get_active_clusters(self) -> list[StopCluster]:
        return [cluster for cluster in self.stop_clusters if cluster.is_active()]

    def get_swept_clusters(self) -> list[StopCluster]:
        return [cluster for cluster in self.stop_clusters if cluster.is_swept()]

    def get_buy_side_clusters(self, *, active_only: bool = True) -> list[StopCluster]:
        clusters = self.get_active_clusters() if active_only else list(self.stop_clusters)
        return [cluster for cluster in clusters if cluster.side == LiquiditySide.BUY_SIDE]

    def get_sell_side_clusters(self, *, active_only: bool = True) -> list[StopCluster]:
        clusters = self.get_active_clusters() if active_only else list(self.stop_clusters)
        return [cluster for cluster in clusters if cluster.side == LiquiditySide.SELL_SIDE]

    def get_nearest_directional_liquidity(
        self,
        side: LiquiditySide,
    ) -> LiquidityLevel | StopCluster | None:
        if side == LiquiditySide.BUY_SIDE:
            return self.nearest_above_level
        if side == LiquiditySide.SELL_SIDE:
            return self.nearest_below_level
        return None

    def get_strongest_directional_cluster(self, side: LiquiditySide) -> StopCluster | None:
        if side == LiquiditySide.BUY_SIDE:
            return self.strongest_cluster_above
        if side == LiquiditySide.SELL_SIDE:
            return self.strongest_cluster_below
        return None

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
                "active_clusters_count": len(self.get_active_clusters()),
                "swept_clusters_count": len(self.get_swept_clusters()),
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
