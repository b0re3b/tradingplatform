from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from core.logger import get_logger

from .config import LiquidityConfig
from .enums import LiquidityBias, LiquidityLevelType, LiquiditySide
from .models import (
    EqualLevel,
    LiquidityLevel,
    LiquidityMapSnapshot,
    LiquiditySignal,
    LiquidityZone,
    StopCluster,
)
from .scoring import LiquidityScorer
from .equal_highs_lows import EqualHighsLowsDetector
from .stop_clusters import StopClustersDetector
from .utils import clamp, merge_price_ranges, midpoint, pct_distance


@dataclass(slots=True)
class LiquidityMapFeatures:
    """
    Внутрішня структура для проміжних обчислень карти ліквідності.
    """

    above_levels: list[LiquidityLevel]
    below_levels: list[LiquidityLevel]
    above_clusters: list[StopCluster]
    below_clusters: list[StopCluster]

    nearest_above_level: LiquidityLevel | StopCluster | None = None
    nearest_below_level: LiquidityLevel | StopCluster | None = None
    strongest_cluster_above: StopCluster | None = None
    strongest_cluster_below: StopCluster | None = None

    above_liquidity_score: float = 0.0
    below_liquidity_score: float = 0.0
    sweep_risk_up: float = 0.0
    sweep_risk_down: float = 0.0
    magnet_score_up: float = 0.0
    magnet_score_down: float = 0.0
    pressure_score: float = 0.0
    bias: LiquidityBias = LiquidityBias.NEUTRAL


class LiquidityMap:
    """
    Центральний агрегатор liquidity-домену.

    Відповідає за:
    - побудову equal highs / lows
    - побудову stop clusters
    - агрегування liquidity landscape
    - розрахунок signal/bias/magnet/sweep-risk
    - формування LiquidityMapSnapshot
    """

    def __init__(
        self,
        config: LiquidityConfig,
        equal_detector: EqualHighsLowsDetector | None = None,
        stop_detector: StopClustersDetector | None = None,
        scorer: LiquidityScorer | None = None,
        event_bus: Any | None = None,
    ) -> None:
        self._config = config
        self._config.validate()

        self._scorer = scorer or LiquidityScorer(config=config)
        self._equal_detector = equal_detector or EqualHighsLowsDetector(
            config=config,
            scorer=self._scorer,
            event_bus=event_bus,
        )
        self._stop_detector = stop_detector or StopClustersDetector(
            config=config,
            scorer=self._scorer,
            event_bus=event_bus,
        )
        self._event_bus = event_bus
        self._logger = get_logger(__name__, service_name="liquidity")

    def build_snapshot(
        self,
        symbol: str,
        timeframe: str,
        candles: Sequence[Any],
        current_price: float,
        orderbook: dict[str, Sequence[Any]] | None = None,
        extra_levels: Sequence[LiquidityLevel] | None = None,
        extra_clusters: Sequence[StopCluster] | None = None,
        timestamp: datetime | None = None,
    ) -> LiquidityMapSnapshot:
        """
        Повна побудова snapshot карти ліквідності.

        Parameters
        ----------
        symbol:
            Торговий символ.
        timeframe:
            Таймфрейм.
        candles:
            Нормалізовані свічки.
        current_price:
            Поточна ціна.
        orderbook:
            Опціональний orderbook для підсилення stop-cluster analysis.
        extra_levels:
            Додаткові liquidity levels від інших модулів
            (наприклад range highs/lows, orderbook walls, liquidation zones).
        extra_clusters:
            Додаткові stop/liquidity clusters від інших джерел.
        timestamp:
            Явний timestamp snapshot-а.
        """
        snapshot_ts = timestamp or self._resolve_snapshot_timestamp(candles)

        equal_levels = self._equal_detector.detect(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles,
            current_price=current_price,
        )

        active_levels = self._merge_levels(
            equal_levels=equal_levels,
            extra_levels=extra_levels,
        )

        stop_clusters = self._stop_detector.detect_from_levels(
            symbol=symbol,
            timeframe=timeframe,
            levels=active_levels,
            current_price=current_price,
            candles=candles,
            orderbook=orderbook,
        )

        if extra_clusters:
            stop_clusters = self._merge_clusters(stop_clusters, list(extra_clusters))

        zones = self._build_liquidity_zones(
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            levels=active_levels,
            clusters=stop_clusters,
        )

        features = self._extract_features(
            current_price=current_price,
            levels=active_levels,
            clusters=stop_clusters,
        )

        signal = self._build_signal(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=snapshot_ts,
            features=features,
        )

        snapshot = LiquidityMapSnapshot(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=snapshot_ts,
            current_price=current_price,
            active_levels=active_levels,
            equal_levels=equal_levels,
            stop_clusters=stop_clusters,
            zones=zones,
            nearest_above_level=features.nearest_above_level,
            nearest_below_level=features.nearest_below_level,
            strongest_cluster_above=features.strongest_cluster_above,
            strongest_cluster_below=features.strongest_cluster_below,
            above_liquidity_score=features.above_liquidity_score,
            below_liquidity_score=features.below_liquidity_score,
            liquidity_pressure_score=features.pressure_score,
            bias=features.bias,
            signal=signal,
            metadata={
                "levels_count": len(active_levels),
                "equal_levels_count": len(equal_levels),
                "stop_clusters_count": len(stop_clusters),
                "zones_count": len(zones),
                "sweep_risk_up": features.sweep_risk_up,
                "sweep_risk_down": features.sweep_risk_down,
                "magnet_score_up": features.magnet_score_up,
                "magnet_score_down": features.magnet_score_down,
            },
        )

        self._logger.info(
            "Liquidity map snapshot built",
            extra={
                "symbol": symbol,
                "timeframe": timeframe,
                "current_price": current_price,
                "levels_count": len(active_levels),
                "equal_levels_count": len(equal_levels),
                "clusters_count": len(stop_clusters),
                "zones_count": len(zones),
                "bias": snapshot.bias.value,
                "above_liquidity_score": snapshot.above_liquidity_score,
                "below_liquidity_score": snapshot.below_liquidity_score,
                "pressure_score": snapshot.liquidity_pressure_score,
            },
        )

        return snapshot

    def build_snapshot_from_components(
        self,
        symbol: str,
        timeframe: str,
        current_price: float,
        levels: Sequence[LiquidityLevel],
        clusters: Sequence[StopCluster],
        equal_levels: Sequence[EqualLevel] | None = None,
        timestamp: datetime | None = None,
    ) -> LiquidityMapSnapshot:
        """
        Альтернативний шлях: якщо рівні/кластери вже пораховані зовні.
        """
        snapshot_ts = timestamp or datetime.now(timezone.utc).replace(tzinfo=None)
        active_levels = list(levels)
        stop_clusters = self._merge_clusters(list(clusters), [])
        zones = self._build_liquidity_zones(
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            levels=active_levels,
            clusters=stop_clusters,
        )
        features = self._extract_features(
            current_price=current_price,
            levels=active_levels,
            clusters=stop_clusters,
        )
        signal = self._build_signal(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=snapshot_ts,
            features=features,
        )

        return LiquidityMapSnapshot(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=snapshot_ts,
            current_price=current_price,
            active_levels=active_levels,
            equal_levels=list(equal_levels or []),
            stop_clusters=stop_clusters,
            zones=zones,
            nearest_above_level=features.nearest_above_level,
            nearest_below_level=features.nearest_below_level,
            strongest_cluster_above=features.strongest_cluster_above,
            strongest_cluster_below=features.strongest_cluster_below,
            above_liquidity_score=features.above_liquidity_score,
            below_liquidity_score=features.below_liquidity_score,
            liquidity_pressure_score=features.pressure_score,
            bias=features.bias,
            signal=signal,
            metadata={
                "levels_count": len(active_levels),
                "equal_levels_count": len(equal_levels or []),
                "stop_clusters_count": len(stop_clusters),
                "zones_count": len(zones),
                "sweep_risk_up": features.sweep_risk_up,
                "sweep_risk_down": features.sweep_risk_down,
                "magnet_score_up": features.magnet_score_up,
                "magnet_score_down": features.magnet_score_down,
            },
        )

    def _merge_levels(
        self,
        equal_levels: Sequence[EqualLevel],
        extra_levels: Sequence[LiquidityLevel] | None,
    ) -> list[LiquidityLevel]:
        merged: list[LiquidityLevel] = list(equal_levels)
        if extra_levels:
            merged.extend(extra_levels)

        if not merged:
            return []

        merged.sort(key=lambda level: level.price)
        deduplicated: list[LiquidityLevel] = [merged[0]]

        for level in merged[1:]:
            prev = deduplicated[-1]

            same_type = level.level_type == prev.level_type
            same_side = level.side == prev.side
            near = pct_distance(level.price, prev.price) <= self._config.equal_level_tolerance_pct

            if same_type and same_side and near:
                if level.confidence > prev.confidence:
                    deduplicated[-1] = level
            else:
                deduplicated.append(level)

        return deduplicated

    def _merge_clusters(
        self,
        primary: list[StopCluster],
        extra: list[StopCluster],
    ) -> list[StopCluster]:
        all_clusters = [*primary, *extra]
        if not all_clusters:
            return []

        all_clusters.sort(key=lambda cluster: (cluster.side.value, cluster.center_price))
        merged: list[StopCluster] = [all_clusters[0]]

        for cluster in all_clusters[1:]:
            prev = merged[-1]

            if cluster.side != prev.side:
                merged.append(cluster)
                continue

            if self._clusters_are_close(prev, cluster):
                merged[-1] = self._merge_two_clusters(prev, cluster)
            else:
                merged.append(cluster)

        return merged

    def _clusters_are_close(
        self,
        left: StopCluster,
        right: StopCluster,
    ) -> bool:
        if left.overlaps(right):
            return True

        return pct_distance(left.center_price, right.center_price) <= self._config.cluster_merge_distance_pct

    def _merge_two_clusters(
        self,
        left: StopCluster,
        right: StopCluster,
    ) -> StopCluster:
        merged = StopCluster(
            symbol=left.symbol,
            timeframe=left.timeframe,
            side=left.side,
            low_price=min(left.low_price, right.low_price),
            high_price=max(left.high_price, right.high_price),
            center_price=midpoint(
                min(left.low_price, right.low_price),
                max(left.high_price, right.high_price),
            ),
            confidence=max(left.confidence, right.confidence),
            estimated_stop_density=max(left.estimated_stop_density, right.estimated_stop_density),
            touches_count=left.touches_count + right.touches_count,
            source_level_type=left.source_level_type,
            strength=left.strength if left.confidence >= right.confidence else right.strength,
            created_at=self._min_dt(left.created_at, right.created_at),
            updated_at=self._max_dt(left.updated_at, right.updated_at),
            source_levels=[*left.source_levels, *right.source_levels],
            metadata={
                "merged": True,
                "left_metadata": left.metadata,
                "right_metadata": right.metadata,
            },
        )

        merged.confidence = self._scorer.score_stop_cluster(merged)
        merged.strength = self._scorer.classify_cluster_strength(merged.confidence)
        return merged

    def _build_liquidity_zones(
        self,
        symbol: str,
        timeframe: str,
        current_price: float,
        levels: Sequence[LiquidityLevel],
        clusters: Sequence[StopCluster],
    ) -> list[LiquidityZone]:
        """
        Формує агреговані liquidity zones для strategy/dashboard/AI.
        """
        ranges: list[tuple[float, float]] = []

        for level in levels:
            low_price, high_price = self._level_to_zone_range(level)
            ranges.append((low_price, high_price))

        for cluster in clusters:
            ranges.append((cluster.low_price, cluster.high_price))

        merged_ranges = merge_price_ranges(
            ranges=ranges,
            merge_distance_pct=self._config.cluster_merge_distance_pct,
        )

        zones: list[LiquidityZone] = []
        for low_price, high_price in merged_ranges:
            zone_levels = [level for level in levels if low_price <= level.price <= high_price]
            zone_clusters = [
                cluster
                for cluster in clusters
                if not (cluster.high_price < low_price or cluster.low_price > high_price)
            ]

            side = self._infer_zone_side(zone_levels, zone_clusters, current_price, low_price, high_price)
            score = self._calculate_zone_score(zone_levels, zone_clusters, current_price, low_price, high_price)
            source_types = self._collect_zone_source_types(zone_levels, zone_clusters)
            label = self._build_zone_label(side=side, score=score, current_price=current_price, low_price=low_price, high_price=high_price)

            zones.append(
                LiquidityZone(
                    symbol=symbol,
                    timeframe=timeframe,
                    side=side,
                    low_price=low_price,
                    high_price=high_price,
                    score=score,
                    label=label,
                    source_types=source_types,
                    metadata={
                        "levels_count": len(zone_levels),
                        "clusters_count": len(zone_clusters),
                        "distance_from_price_pct": self._distance_from_zone_pct(current_price, low_price, high_price),
                    },
                )
            )

        zones.sort(key=lambda z: z.center_price)
        return zones

    def _extract_features(
        self,
        current_price: float,
        levels: Sequence[LiquidityLevel],
        clusters: Sequence[StopCluster],
    ) -> LiquidityMapFeatures:
        above_levels = [level for level in levels if level.price > current_price]
        below_levels = [level for level in levels if level.price < current_price]

        above_clusters = [cluster for cluster in clusters if cluster.center_price > current_price]
        below_clusters = [cluster for cluster in clusters if cluster.center_price < current_price]

        nearest_above_level = self._find_nearest_above(current_price, levels, clusters)
        nearest_below_level = self._find_nearest_below(current_price, levels, clusters)

        strongest_cluster_above = self._find_strongest_cluster_above(current_price, clusters)
        strongest_cluster_below = self._find_strongest_cluster_below(current_price, clusters)

        above_liquidity_score = self._calculate_side_liquidity_score(
            current_price=current_price,
            levels=above_levels,
            clusters=above_clusters,
            side=LiquiditySide.BUY_SIDE,
        )
        below_liquidity_score = self._calculate_side_liquidity_score(
            current_price=current_price,
            levels=below_levels,
            clusters=below_clusters,
            side=LiquiditySide.SELL_SIDE,
        )

        sweep_risk_up = self._calculate_sweep_risk_up(
            current_price=current_price,
            above_levels=above_levels,
            above_clusters=above_clusters,
        )
        sweep_risk_down = self._calculate_sweep_risk_down(
            current_price=current_price,
            below_levels=below_levels,
            below_clusters=below_clusters,
        )

        magnet_score_up = self._calculate_magnet_score_up(
            current_price=current_price,
            above_levels=above_levels,
            above_clusters=above_clusters,
        )
        magnet_score_down = self._calculate_magnet_score_down(
            current_price=current_price,
            below_levels=below_levels,
            below_clusters=below_clusters,
        )

        pressure_score = self._calculate_pressure_score(
            above_liquidity_score=above_liquidity_score,
            below_liquidity_score=below_liquidity_score,
            magnet_score_up=magnet_score_up,
            magnet_score_down=magnet_score_down,
            sweep_risk_up=sweep_risk_up,
            sweep_risk_down=sweep_risk_down,
        )

        temp_snapshot = LiquidityMapSnapshot(
            symbol="",
            timeframe="",
            timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
            current_price=current_price,
            active_levels=list(levels),
            equal_levels=[],
            stop_clusters=list(clusters),
            above_liquidity_score=above_liquidity_score,
            below_liquidity_score=below_liquidity_score,
            liquidity_pressure_score=pressure_score,
        )
        bias = self._scorer.infer_bias_from_snapshot(temp_snapshot)

        return LiquidityMapFeatures(
            above_levels=above_levels,
            below_levels=below_levels,
            above_clusters=above_clusters,
            below_clusters=below_clusters,
            nearest_above_level=nearest_above_level,
            nearest_below_level=nearest_below_level,
            strongest_cluster_above=strongest_cluster_above,
            strongest_cluster_below=strongest_cluster_below,
            above_liquidity_score=above_liquidity_score,
            below_liquidity_score=below_liquidity_score,
            sweep_risk_up=sweep_risk_up,
            sweep_risk_down=sweep_risk_down,
            magnet_score_up=magnet_score_up,
            magnet_score_down=magnet_score_down,
            pressure_score=pressure_score,
            bias=bias,
        )

    def _build_signal(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
        features: LiquidityMapFeatures,
    ) -> LiquiditySignal:
        explanation = self._build_signal_explanation(features)

        return LiquiditySignal(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            bias=features.bias,
            nearest_buy_side_liquidity=features.nearest_above_level,
            nearest_sell_side_liquidity=features.nearest_below_level,
            sweep_risk_up=features.sweep_risk_up,
            sweep_risk_down=features.sweep_risk_down,
            magnet_score_up=features.magnet_score_up,
            magnet_score_down=features.magnet_score_down,
            explanation=explanation,
            metadata={
                "above_liquidity_score": features.above_liquidity_score,
                "below_liquidity_score": features.below_liquidity_score,
                "pressure_score": features.pressure_score,
                "strongest_cluster_above_confidence": (
                    features.strongest_cluster_above.confidence
                    if features.strongest_cluster_above
                    else 0.0
                ),
                "strongest_cluster_below_confidence": (
                    features.strongest_cluster_below.confidence
                    if features.strongest_cluster_below
                    else 0.0
                ),
            },
        )

    def _build_signal_explanation(self, features: LiquidityMapFeatures) -> str:
        parts: list[str] = []

        if features.bias == LiquidityBias.UP:
            parts.append("buy-side liquidity above price looks stronger")
        elif features.bias == LiquidityBias.DOWN:
            parts.append("sell-side liquidity below price looks stronger")
        else:
            parts.append("liquidity is relatively balanced")

        if features.sweep_risk_up > 0.65:
            parts.append("elevated upside sweep risk")
        if features.sweep_risk_down > 0.65:
            parts.append("elevated downside sweep risk")

        if features.magnet_score_up > 0.70:
            parts.append("strong upside magnet")
        if features.magnet_score_down > 0.70:
            parts.append("strong downside magnet")

        if features.strongest_cluster_above and features.strongest_cluster_above.confidence > 0.75:
            parts.append("high-confidence stop cluster above")
        if features.strongest_cluster_below and features.strongest_cluster_below.confidence > 0.75:
            parts.append("high-confidence stop cluster below")

        return "; ".join(parts)

    def _find_nearest_above(
        self,
        current_price: float,
        levels: Sequence[LiquidityLevel],
        clusters: Sequence[StopCluster],
    ) -> LiquidityLevel | StopCluster | None:
        candidates: list[LiquidityLevel | StopCluster] = []

        for level in levels:
            if level.price > current_price:
                candidates.append(level)

        for cluster in clusters:
            if cluster.center_price > current_price:
                candidates.append(cluster)

        if not candidates:
            return None

        return min(candidates, key=lambda item: self._reference_price(item))

    def _find_nearest_below(
        self,
        current_price: float,
        levels: Sequence[LiquidityLevel],
        clusters: Sequence[StopCluster],
    ) -> LiquidityLevel | StopCluster | None:
        candidates: list[LiquidityLevel | StopCluster] = []

        for level in levels:
            if level.price < current_price:
                candidates.append(level)

        for cluster in clusters:
            if cluster.center_price < current_price:
                candidates.append(cluster)

        if not candidates:
            return None

        return max(candidates, key=lambda item: self._reference_price(item))

    def _find_strongest_cluster_above(
        self,
        current_price: float,
        clusters: Sequence[StopCluster],
    ) -> StopCluster | None:
        candidates = [cluster for cluster in clusters if cluster.center_price > current_price]
        if not candidates:
            return None
        return max(candidates, key=lambda cluster: cluster.confidence)

    def _find_strongest_cluster_below(
        self,
        current_price: float,
        clusters: Sequence[StopCluster],
    ) -> StopCluster | None:
        candidates = [cluster for cluster in clusters if cluster.center_price < current_price]
        if not candidates:
            return None
        return max(candidates, key=lambda cluster: cluster.confidence)

    def _calculate_side_liquidity_score(
        self,
        current_price: float,
        levels: Sequence[LiquidityLevel],
        clusters: Sequence[StopCluster],
        side: LiquiditySide,
    ) -> float:
        level_score = self._scorer.score_liquidity_side_pressure(
            levels=list(levels),
            current_price=current_price,
            side=side,
        )

        if not clusters:
            return level_score

        cluster_scores: list[float] = []
        for cluster in clusters:
            proximity = 1.0 - clamp(pct_distance(cluster.center_price, current_price) / 0.03, 0.0, 1.0)
            score = 0.7 * cluster.confidence + 0.3 * proximity
            cluster_scores.append(clamp(score, 0.0, 1.0))

        avg_cluster_score = sum(cluster_scores) / len(cluster_scores) if cluster_scores else 0.0
        return clamp(0.55 * level_score + 0.45 * avg_cluster_score, 0.0, 1.0)

    def _calculate_sweep_risk_up(
        self,
        current_price: float,
        above_levels: Sequence[LiquidityLevel],
        above_clusters: Sequence[StopCluster],
    ) -> float:
        if not above_levels and not above_clusters:
            return 0.0

        nearest_component = 0.0
        nearest = self._find_nearest_above(current_price, above_levels, above_clusters)
        if nearest is not None:
            nearest_price = self._reference_price(nearest)
            distance_score = 1.0 - clamp(pct_distance(nearest_price, current_price) / 0.015, 0.0, 1.0)
            nearest_component = distance_score

        cluster_component = 0.0
        if above_clusters:
            cluster_component = max(cluster.confidence for cluster in above_clusters)

        level_component = 0.0
        if above_levels:
            level_component = max(level.confidence for level in above_levels)

        return clamp(
            0.35 * nearest_component + 0.35 * cluster_component + 0.30 * level_component,
            0.0,
            1.0,
        )

    def _calculate_sweep_risk_down(
        self,
        current_price: float,
        below_levels: Sequence[LiquidityLevel],
        below_clusters: Sequence[StopCluster],
    ) -> float:
        if not below_levels and not below_clusters:
            return 0.0

        nearest_component = 0.0
        nearest = self._find_nearest_below(current_price, below_levels, below_clusters)
        if nearest is not None:
            nearest_price = self._reference_price(nearest)
            distance_score = 1.0 - clamp(pct_distance(nearest_price, current_price) / 0.015, 0.0, 1.0)
            nearest_component = distance_score

        cluster_component = 0.0
        if below_clusters:
            cluster_component = max(cluster.confidence for cluster in below_clusters)

        level_component = 0.0
        if below_levels:
            level_component = max(level.confidence for level in below_levels)

        return clamp(
            0.35 * nearest_component + 0.35 * cluster_component + 0.30 * level_component,
            0.0,
            1.0,
        )

    def _calculate_magnet_score_up(
        self,
        current_price: float,
        above_levels: Sequence[LiquidityLevel],
        above_clusters: Sequence[StopCluster],
    ) -> float:
        return self._calculate_magnet_score(
            current_price=current_price,
            levels=above_levels,
            clusters=above_clusters,
        )

    def _calculate_magnet_score_down(
        self,
        current_price: float,
        below_levels: Sequence[LiquidityLevel],
        below_clusters: Sequence[StopCluster],
    ) -> float:
        return self._calculate_magnet_score(
            current_price=current_price,
            levels=below_levels,
            clusters=below_clusters,
        )

    def _calculate_magnet_score(
        self,
        current_price: float,
        levels: Sequence[LiquidityLevel],
        clusters: Sequence[StopCluster],
    ) -> float:
        """
        Оцінка того, наскільки сильним "магнітом" є ліквідність з однієї сторони.
        """
        if not levels and not clusters:
            return 0.0

        components: list[float] = []

        for level in levels:
            distance_score = 1.0 - clamp(pct_distance(level.price, current_price) / 0.025, 0.0, 1.0)
            components.append(clamp(0.65 * level.confidence + 0.35 * distance_score, 0.0, 1.0))

        for cluster in clusters:
            distance_score = 1.0 - clamp(pct_distance(cluster.center_price, current_price) / 0.025, 0.0, 1.0)
            density_score = clamp(cluster.estimated_stop_density, 0.0, 1.0)
            components.append(clamp(0.5 * cluster.confidence + 0.25 * density_score + 0.25 * distance_score, 0.0, 1.0))

        if not components:
            return 0.0

        top_components = sorted(components, reverse=True)[:3]
        return clamp(sum(top_components) / len(top_components), 0.0, 1.0)

    def _calculate_pressure_score(
        self,
        above_liquidity_score: float,
        below_liquidity_score: float,
        magnet_score_up: float,
        magnet_score_down: float,
        sweep_risk_up: float,
        sweep_risk_down: float,
    ) -> float:
        """
        Підсумковий pressure score:
        > 0  — upward liquidity pressure
        < 0  — downward liquidity pressure
        """
        upward = 0.45 * above_liquidity_score + 0.30 * magnet_score_up + 0.25 * sweep_risk_up
        downward = 0.45 * below_liquidity_score + 0.30 * magnet_score_down + 0.25 * sweep_risk_down

        raw = upward - downward
        return clamp(raw, -1.0, 1.0)

    def _calculate_zone_score(
        self,
        zone_levels: Sequence[LiquidityLevel],
        zone_clusters: Sequence[StopCluster],
        current_price: float,
        low_price: float,
        high_price: float,
    ) -> float:
        center = midpoint(low_price, high_price)
        distance_score = 1.0 - clamp(pct_distance(center, current_price) / 0.04, 0.0, 1.0)

        level_component = 0.0
        if zone_levels:
            level_component = sum(level.confidence for level in zone_levels) / len(zone_levels)

        cluster_component = 0.0
        if zone_clusters:
            cluster_component = sum(cluster.confidence for cluster in zone_clusters) / len(zone_clusters)

        density_component = 0.0
        if zone_clusters:
            density_component = sum(cluster.estimated_stop_density for cluster in zone_clusters) / len(zone_clusters)

        return clamp(
            0.35 * level_component
            + 0.35 * cluster_component
            + 0.15 * density_component
            + 0.15 * distance_score,
            0.0,
            1.0,
        )

    def _collect_zone_source_types(
        self,
        zone_levels: Sequence[LiquidityLevel],
        zone_clusters: Sequence[StopCluster],
    ) -> list[LiquidityLevelType]:
        source_types: set[LiquidityLevelType] = set()

        for level in zone_levels:
            source_types.add(level.level_type)

        for cluster in zone_clusters:
            source_types.add(cluster.source_level_type)

        return sorted(source_types, key=lambda x: x.value)

    def _infer_zone_side(
        self,
        zone_levels: Sequence[LiquidityLevel],
        zone_clusters: Sequence[StopCluster],
        current_price: float,
        low_price: float,
        high_price: float,
    ) -> LiquiditySide:
        center = midpoint(low_price, high_price)

        buy_votes = 0
        sell_votes = 0

        for level in zone_levels:
            if level.side == LiquiditySide.BUY_SIDE:
                buy_votes += 1
            elif level.side == LiquiditySide.SELL_SIDE:
                sell_votes += 1

        for cluster in zone_clusters:
            if cluster.side == LiquiditySide.BUY_SIDE:
                buy_votes += 2
            elif cluster.side == LiquiditySide.SELL_SIDE:
                sell_votes += 2

        if buy_votes > sell_votes:
            return LiquiditySide.BUY_SIDE
        if sell_votes > buy_votes:
            return LiquiditySide.SELL_SIDE

        if center > current_price:
            return LiquiditySide.BUY_SIDE
        if center < current_price:
            return LiquiditySide.SELL_SIDE

        return LiquiditySide.UNKNOWN

    def _build_zone_label(
        self,
        side: LiquiditySide,
        score: float,
        current_price: float,
        low_price: float,
        high_price: float,
    ) -> str:
        position = "above" if midpoint(low_price, high_price) > current_price else "below"

        if side == LiquiditySide.BUY_SIDE:
            base = f"buy-side liquidity zone {position}"
        elif side == LiquiditySide.SELL_SIDE:
            base = f"sell-side liquidity zone {position}"
        else:
            base = f"mixed liquidity zone {position}"

        if score >= 0.80:
            return f"strong {base}"
        if score >= 0.55:
            return f"medium {base}"
        return f"weak {base}"

    def _distance_from_zone_pct(
        self,
        current_price: float,
        low_price: float,
        high_price: float,
    ) -> float:
        if low_price <= current_price <= high_price:
            return 0.0

        if current_price < low_price:
            return pct_distance(low_price, current_price)

        return pct_distance(high_price, current_price)

    def _level_to_zone_range(self, level: LiquidityLevel) -> tuple[float, float]:
        """
        Перетворення рівня в мінімальну зону.
        """
        tolerance = self._config.equal_level_tolerance_pct

        if hasattr(level, "cluster_low") and hasattr(level, "cluster_high"):
            cluster_low = getattr(level, "cluster_low", None)
            cluster_high = getattr(level, "cluster_high", None)
            if cluster_low is not None and cluster_high is not None:
                return float(cluster_low), float(cluster_high)

        low_price = level.price * (1.0 - tolerance)
        high_price = level.price * (1.0 + tolerance)
        return min(low_price, high_price), max(low_price, high_price)

    def _reference_price(self, item: LiquidityLevel | StopCluster) -> float:
        if isinstance(item, StopCluster):
            return item.center_price
        return item.price

    def _resolve_snapshot_timestamp(self, candles: Sequence[Any]) -> datetime:
        if candles:
            last = candles[-1]
            for field in ("close_time", "timestamp", "time", "open_time"):
                value = self._get_value(last, field)
                if value is None:
                    continue

                parsed = self._parse_datetime(value)
                if parsed is not None:
                    return parsed

        return datetime.now(timezone.utc).replace(tzinfo=None)

    def _parse_datetime(self, value: Any) -> datetime | None:
        if isinstance(value, datetime):
            if value.tzinfo is not None:
                return value.astimezone(timezone.utc).replace(tzinfo=None)
            return value

        if isinstance(value, (int, float)):
            try:
                return datetime.utcfromtimestamp(value / 1000 if value > 1e12 else value)
            except Exception:
                return None

        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
                if dt.tzinfo is not None:
                    return dt.astimezone(timezone.utc).replace(tzinfo=None)
                return dt
            except Exception:
                return None

        return None

    def _get_value(self, obj: Any, field: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(field, default)
        return getattr(obj, field, default)

    def _min_dt(self, left: datetime | None, right: datetime | None) -> datetime | None:
        if left is None:
            return right
        if right is None:
            return left
        return min(left, right)

    def _max_dt(self, left: datetime | None, right: datetime | None) -> datetime | None:
        if left is None:
            return right
        if right is None:
            return left
        return max(left, right)