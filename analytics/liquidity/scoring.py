from __future__ import annotations

from dataclasses import dataclass, fields
from .config import LiquidityConfig
from .enums import (
    ClusterStrength,
    LiquidityBias,
    LiquidityLevelType,
    LiquiditySide,
)
from .models import (
    EqualLevel,
    LiquidityLevel,
    LiquidityMapSnapshot,
    StopCluster,
)
from .utils import (
    clamp,
    normalize_confidence,
    pct_distance,
    safe_mean,
)


@dataclass(slots=True)
class LiquidityScoringWeights:
    """
    Ваги scoring-моделі liquidity-модуля.

    Використовуються для:
    - equal highs / equal lows scoring;
    - stop cluster scoring;
    - side pressure scoring;
    - liquidity bias inference.

    Сума ваг не зобов'язана дорівнювати 1.0, але для стабільності
    краще тримати кожну групу ваг близько до 1.0.
    """

    # Equal level scoring
    touches_weight: float = 0.30
    reaction_weight: float = 0.20
    compactness_weight: float = 0.20
    proximity_weight: float = 0.10
    sweep_penalty_weight: float = 0.20

    # Stop cluster scoring
    cluster_density_weight: float = 0.40
    cluster_source_count_weight: float = 0.25
    cluster_width_weight: float = 0.15
    cluster_proximity_weight: float = 0.20

    # Side pressure scoring
    side_confidence_weight: float = 0.50
    side_distance_weight: float = 0.30
    side_touch_weight: float = 0.20

    def validate(self) -> None:
        errors: list[str] = []

        for field_info in fields(self):
            field_name = field_info.name
            value = getattr(self, field_name)

            if value < 0:
                errors.append(f"{field_name} must be >= 0")

        if errors:
            raise ValueError("Invalid LiquidityScoringWeights: " + "; ".join(errors))


class LiquidityScorer:
    """
    Центральний scorer для analytics/liquidity.

    Це чистий domain-компонент:
    - не має EventBus;
    - не має Scheduler;
    - не має logger;
    - не публікує події;
    - не зберігає runtime state.

    Його задача — оцінювати:
    - силу equal highs / equal lows;
    - силу stop clusters;
    - density stop-кластерів;
    - side pressure;
    - aggregate liquidity bias.
    """

    def __init__(
        self,
        config: LiquidityConfig,
        weights: LiquidityScoringWeights | None = None,
    ) -> None:
        self._config = config
        self._config.validate()

        self._weights = weights or LiquidityScoringWeights()
        self._weights.validate()

    # ------------------------------------------------------------------
    # Equal levels
    # ------------------------------------------------------------------

    def score_equal_level(
        self,
        level: EqualLevel,
        current_price: float | None = None,
    ) -> float:
        """
        Оцінює силу equal highs / equal lows рівня.

        Основні фактори:
        - кількість торкань;
        - кількість реакцій;
        - компактність price cluster;
        - близькість до current price;
        - штраф за sweep / partial sweep.
        """
        touches_score = self._score_touches(level.touches_count)
        reaction_score = self._score_reactions(level.reaction_count)
        compactness_score = self._score_equal_level_compactness(level)
        proximity_score = self._score_proximity(
            target_price=level.price,
            current_price=current_price,
            max_distance_pct=0.02,
            default=1.0,
        )
        sweep_penalty = self._score_sweep_penalty(level)

        raw_score = (
            touches_score * self._weights.touches_weight
            + reaction_score * self._weights.reaction_weight
            + compactness_score * self._weights.compactness_weight
            + proximity_score * self._weights.proximity_weight
            - sweep_penalty * self._weights.sweep_penalty_weight
        )

        return normalize_confidence(raw_score)

    def _score_equal_level_compactness(self, level: EqualLevel) -> float:
        """
        Чим компактніший equal-level cluster, тим вища оцінка.
        """
        if level.price <= 0:
            return 0.0

        cluster_width = level.cluster_width
        cluster_width_pct = cluster_width / abs(level.price)
        max_width_pct = max(self._config.max_equal_cluster_width_pct, 1e-12)

        return 1.0 - clamp(cluster_width_pct / max_width_pct, 0.0, 1.0)

    def _score_sweep_penalty(self, level: LiquidityLevel) -> float:
        """
        Штраф за swept / partially swept liquidity.
        """
        if not self._config.use_partial_sweep_penalty:
            return 0.0

        if level.is_swept():
            return 1.0

        if level.is_partially_swept():
            return 0.5

        return 0.0

    # ------------------------------------------------------------------
    # Stop clusters
    # ------------------------------------------------------------------

    def score_stop_cluster(
        self,
        cluster: StopCluster,
        current_price: float | None = None,
    ) -> float:
        """
        Оцінює силу stop/liquidity cluster.

        Основні фактори:
        - estimated stop density;
        - кількість source levels;
        - компактність ширини кластера;
        - близькість до current price.
        """
        density_score = clamp(cluster.estimated_stop_density, 0.0, 1.0)
        source_count_score = self._score_source_count(cluster.source_levels)
        width_score = self._score_cluster_width(cluster)
        proximity_score = self._score_proximity(
            target_price=cluster.center_price,
            current_price=current_price,
            max_distance_pct=0.02,
            default=0.5,
        )

        raw_score = (
            density_score * self._weights.cluster_density_weight
            + source_count_score * self._weights.cluster_source_count_weight
            + width_score * self._weights.cluster_width_weight
            + proximity_score * self._weights.cluster_proximity_weight
        )

        return normalize_confidence(raw_score)

    def _score_cluster_width(self, cluster: StopCluster) -> float:
        """
        Компактніші кластери отримують вищий score.
        """
        if cluster.center_price <= 0:
            return 0.0

        width_pct = cluster.width() / abs(cluster.center_price)
        return 1.0 - clamp(width_pct / 0.01, 0.0, 1.0)

    def _score_source_count(
        self,
        source_levels: list[LiquidityLevel],
    ) -> float:
        """
        Оцінка кількості source levels у stop cluster.
        """
        return clamp(len(source_levels) / 5.0, 0.0, 1.0)

    def classify_cluster_strength(self, score: float) -> ClusterStrength:
        """
        Перетворює numeric score у якісну силу cluster.
        """
        score = normalize_confidence(score)

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
        Оцінює приблизну щільність стопів у кластері.

        Фактори:
        - сумарна кількість touches;
        - середній confidence source-рівнів;
        - кількість source-рівнів;
        - частка active-рівнів.
        """
        if not source_levels:
            return 0.0

        touches_sum = sum(max(level.touches_count, 1) for level in source_levels)
        confidence_avg = safe_mean([level.confidence for level in source_levels])
        source_count_score = self._score_source_count(source_levels)

        active_count = sum(1 for level in source_levels if level.is_active())
        active_ratio = active_count / len(source_levels)

        touches_score = clamp(touches_sum / 10.0, 0.0, 1.0)
        active_score = clamp(active_ratio, 0.0, 1.0)

        density = (
            0.35 * touches_score
            + 0.35 * confidence_avg
            + 0.20 * source_count_score
            + 0.10 * active_score
        )

        return normalize_confidence(density)

    # ------------------------------------------------------------------
    # Side pressure / liquidity landscape
    # ------------------------------------------------------------------

    def score_liquidity_side_pressure(
        self,
        levels: list[LiquidityLevel],
        current_price: float,
        side: LiquiditySide,
    ) -> float:
        """
        Оцінює силу ліквідності на конкретній стороні від current_price.

        BUY_SIDE:
            Беруться рівні вище current_price.

        SELL_SIDE:
            Беруться рівні нижче current_price.
        """
        if current_price <= 0 or not levels:
            return 0.0

        relevant = self._filter_levels_by_side_and_price(
            levels=levels,
            current_price=current_price,
            side=side,
        )

        if not relevant:
            return 0.0

        scores = [
            self._score_level_pressure(
                level=level,
                current_price=current_price,
            )
            for level in relevant
        ]

        return normalize_confidence(safe_mean(scores))

    def _filter_levels_by_side_and_price(
        self,
        levels: list[LiquidityLevel],
        current_price: float,
        side: LiquiditySide,
    ) -> list[LiquidityLevel]:
        if side == LiquiditySide.BUY_SIDE:
            return [
                level
                for level in levels
                if level.price > current_price
                and level.side in {LiquiditySide.BUY_SIDE, LiquiditySide.BOTH}
            ]

        if side == LiquiditySide.SELL_SIDE:
            return [
                level
                for level in levels
                if level.price < current_price
                and level.side in {LiquiditySide.SELL_SIDE, LiquiditySide.BOTH}
            ]

        return []

    def _score_level_pressure(
        self,
        level: LiquidityLevel,
        current_price: float,
    ) -> float:
        confidence_score = clamp(level.confidence, 0.0, 1.0)

        distance_score = self._score_proximity(
            target_price=level.price,
            current_price=current_price,
            max_distance_pct=0.03,
            default=0.0,
        )

        touch_score = self._score_touches(level.touches_count)

        raw_score = (
            confidence_score * self._weights.side_confidence_weight
            + distance_score * self._weights.side_distance_weight
            + touch_score * self._weights.side_touch_weight
        )

        if level.is_swept():
            raw_score *= 0.45
        elif level.is_partially_swept():
            raw_score *= 0.75

        if not level.is_active():
            raw_score *= 0.70

        return normalize_confidence(raw_score)

    def infer_bias_from_snapshot(
        self,
        snapshot: LiquidityMapSnapshot,
    ) -> LiquidityBias:
        """
        Виводить liquidity bias на основі above/below liquidity scores.

        UP:
            Більше магнітної/стопової ліквідності зверху.

        DOWN:
            Більше ліквідності знизу.

        NEUTRAL:
            Різниця недостатня.
        """
        delta = snapshot.above_liquidity_score - snapshot.below_liquidity_score

        if delta >= 0.15:
            return LiquidityBias.UP

        if delta <= -0.15:
            return LiquidityBias.DOWN

        return LiquidityBias.NEUTRAL

    def infer_bias_from_scores(
        self,
        above_liquidity_score: float,
        below_liquidity_score: float,
        threshold: float = 0.15,
    ) -> LiquidityBias:
        """
        Виводить bias напряму з двох side scores.
        """
        above = normalize_confidence(above_liquidity_score)
        below = normalize_confidence(below_liquidity_score)
        threshold = clamp(threshold, 0.0, 1.0)

        delta = above - below

        if delta >= threshold:
            return LiquidityBias.UP

        if delta <= -threshold:
            return LiquidityBias.DOWN

        return LiquidityBias.NEUTRAL

    # ------------------------------------------------------------------
    # Level type helpers
    # ------------------------------------------------------------------

    def infer_side_for_level_type(
        self,
        level_type: LiquidityLevelType,
    ) -> LiquiditySide:
        """
        Визначає сторону liquidity за типом рівня.

        Делегує логіку enum-методу, щоб не дублювати mapping.
        """
        return level_type.infer_side()

    # ------------------------------------------------------------------
    # Generic scoring helpers
    # ------------------------------------------------------------------

    def _score_touches(self, touches_count: int) -> float:
        return clamp(max(touches_count, 0) / 5.0, 0.0, 1.0)

    def _score_reactions(self, reaction_count: int) -> float:
        return clamp(max(reaction_count, 0) / 4.0, 0.0, 1.0)

    def _score_proximity(
        self,
        target_price: float,
        current_price: float | None,
        max_distance_pct: float,
        default: float,
    ) -> float:
        if current_price is None or current_price <= 0 or target_price <= 0:
            return clamp(default, 0.0, 1.0)

        max_distance_pct = max(max_distance_pct, 1e-12)
        distance = pct_distance(target_price, current_price)

        return 1.0 - clamp(distance / max_distance_pct, 0.0, 1.0)