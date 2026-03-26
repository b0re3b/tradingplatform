from __future__ import annotations

from dataclasses import dataclass

from .config import LiquidityConfig
from .enums import ClusterStrength, LiquidityBias, LiquidityLevelType, LiquiditySide
from .models import EqualLevel, LiquidityLevel, LiquidityMapSnapshot, StopCluster
from .utils import clamp, normalize_confidence, pct_distance


@dataclass(slots=True)
class LiquidityScoringWeights:
    """
    Ваги для scoring-моделі liquidity-модуля.
    """

    touches_weight: float = 0.30
    reaction_weight: float = 0.20
    compactness_weight: float = 0.20
    recency_weight: float = 0.10
    sweep_penalty_weight: float = 0.20

    cluster_density_weight: float = 0.40
    cluster_source_count_weight: float = 0.25
    cluster_width_weight: float = 0.15
    cluster_proximity_weight: float = 0.20


class LiquidityScorer:
    """
    Центральний scorer для liquidity-рівнів і stop-кластерів.
    """

    def __init__(
        self,
        config: LiquidityConfig,
        weights: LiquidityScoringWeights | None = None,
    ) -> None:
        self._config = config
        self._weights = weights or LiquidityScoringWeights()

    def score_equal_level(
        self,
        level: EqualLevel,
        current_price: float | None = None,
    ) -> float:
        """
        Оцінка сили рівня equal highs / equal lows.

        Фактори:
        - кількість торкань
        - кількість реакцій
        - компактність кластера
        - штраф за sweep
        - опціонально близькість до current price
        """
        touches_score = clamp(level.touches_count / 5.0, 0.0, 1.0)
        reaction_score = clamp(level.reaction_count / 4.0, 0.0, 1.0)

        cluster_width = 0.0
        if level.cluster_low is not None and level.cluster_high is not None:
            cluster_width = abs(level.cluster_high - level.cluster_low)

        compactness_score = 1.0
        if level.price > 0:
            cluster_width_pct = cluster_width / level.price
            max_width_pct = max(self._config.max_equal_cluster_width_pct, 1e-12)
            compactness_score = 1.0 - clamp(cluster_width_pct / max_width_pct, 0.0, 1.0)

        recency_score = 1.0
        if current_price is not None and current_price > 0:
            distance = pct_distance(level.price, current_price)
            recency_score = 1.0 - clamp(distance / 0.02, 0.0, 1.0)

        sweep_penalty = 0.0
        if level.is_swept():
            sweep_penalty = 1.0
        elif level.sweep_status.value == "partially_swept":
            sweep_penalty = 0.5

        score = (
            touches_score * self._weights.touches_weight
            + reaction_score * self._weights.reaction_weight
            + compactness_score * self._weights.compactness_weight
            + recency_score * self._weights.recency_weight
            - sweep_penalty * self._weights.sweep_penalty_weight
        )

        return normalize_confidence(score)

    def score_stop_cluster(
        self,
        cluster: StopCluster,
        current_price: float | None = None,
    ) -> float:
        """
        Оцінка сили stop-кластера.
        """
        density_score = clamp(cluster.estimated_stop_density, 0.0, 1.0)

        source_count_score = clamp(len(cluster.source_levels) / 5.0, 0.0, 1.0)

        width_score = 1.0
        if cluster.center_price > 0:
            width_pct = cluster.width() / cluster.center_price
            width_score = 1.0 - clamp(width_pct / 0.01, 0.0, 1.0)

        proximity_score = 0.5
        if current_price is not None and current_price > 0:
            distance = pct_distance(cluster.center_price, current_price)
            proximity_score = 1.0 - clamp(distance / 0.02, 0.0, 1.0)

        score = (
            density_score * self._weights.cluster_density_weight
            + source_count_score * self._weights.cluster_source_count_weight
            + width_score * self._weights.cluster_width_weight
            + proximity_score * self._weights.cluster_proximity_weight
        )

        return normalize_confidence(score)

    def classify_cluster_strength(self, score: float) -> ClusterStrength:
        if score >= 0.85:
            return ClusterStrength.EXTREME
        if score >= 0.65:
            return ClusterStrength.HIGH
        if score >= 0.40:
            return ClusterStrength.MEDIUM
        return ClusterStrength.LOW

    def estimate_stop_density(
        self,
        source_levels: list[LiquidityLevel],
    ) -> float:
        """
        Спрощена оцінка щільності стопів у кластері.
        """
        if not source_levels:
            return 0.0

        touches_sum = sum(max(level.touches_count, 1) for level in source_levels)
        confidence_avg = sum(level.confidence for level in source_levels) / len(source_levels)

        density = 0.5 * clamp(touches_sum / 10.0, 0.0, 1.0) + 0.5 * clamp(confidence_avg, 0.0, 1.0)
        return normalize_confidence(density)

    def score_liquidity_side_pressure(
        self,
        levels: list[LiquidityLevel],
        current_price: float,
        side: LiquiditySide,
    ) -> float:
        """
        Оцінка сили ліквідності на одній стороні від поточної ціни.
        """
        if current_price <= 0 or not levels:
            return 0.0

        relevant: list[LiquidityLevel] = []
        for level in levels:
            if side == LiquiditySide.BUY_SIDE and level.price > current_price:
                relevant.append(level)
            elif side == LiquiditySide.SELL_SIDE and level.price < current_price:
                relevant.append(level)

        if not relevant:
            return 0.0

        total = 0.0
        for level in relevant:
            distance_score = 1.0 - clamp(pct_distance(level.price, current_price) / 0.03, 0.0, 1.0)
            total += 0.65 * clamp(level.confidence, 0.0, 1.0) + 0.35 * distance_score

        return normalize_confidence(total / len(relevant))

    def infer_bias_from_snapshot(self, snapshot: LiquidityMapSnapshot) -> LiquidityBias:
        """
        Груба евристика bias на основі above/below liquidity.
        """
        delta = snapshot.above_liquidity_score - snapshot.below_liquidity_score

        if delta >= 0.15:
            return LiquidityBias.UP
        if delta <= -0.15:
            return LiquidityBias.DOWN
        return LiquidityBias.NEUTRAL

    def infer_side_for_level_type(self, level_type: LiquidityLevelType) -> LiquiditySide:
        if level_type in {
            LiquidityLevelType.EQUAL_HIGHS,
            LiquidityLevelType.SWING_HIGH,
            LiquidityLevelType.RANGE_HIGH,
        }:
            return LiquiditySide.BUY_SIDE

        if level_type in {
            LiquidityLevelType.EQUAL_LOWS,
            LiquidityLevelType.SWING_LOW,
            LiquidityLevelType.RANGE_LOW,
        }:
            return LiquiditySide.SELL_SIDE

        return LiquiditySide.UNKNOWN