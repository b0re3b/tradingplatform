from __future__ import annotations

from statistics import mean
from typing import Iterable

from core.event_bus import EventBus
from core.scheduler import Scheduler

from .base import BaseSpoofingDetector
from .config import SpoofingConfig
from .enums import (
    DetectorDecision,
    OrderbookWallState,
    SpoofingComponent,
    SpoofingPattern,
    SpoofingSide,
)
from .models import (
    DetectorResult,
    LayeringCandidateContext,
    LayeringCluster,
    SpoofingFeatures,
    TrackedWall,
)
from .persistence_tracker import PersistenceTracker


class LayeringDetector(BaseSpoofingDetector):
    """
    Detector multi-level layering.

    Основна ідея:
    - на одній стороні книги присутні кілька аномально великих рівнів;
    - рівні знаходяться близько по ціні;
    - сумарна ліквідність велика;
    - значна частина рівнів знімається/слабшає в близькому часовому вікні.

    Важливо:
    - працює поверх PersistenceTracker state;
    - не підписується на EventBus;
    - не публікує події;
    - не запускає Scheduler jobs;
    - повертає тільки DetectorResult або None.
    """

    component = SpoofingComponent.LAYERING_DETECTOR

    def __init__(
        self,
        *,
        event_bus: EventBus | None,
        scheduler: Scheduler | None,
        config: SpoofingConfig,
        persistence_tracker: PersistenceTracker,
    ) -> None:
        super().__init__(
            event_bus=event_bus,
            scheduler=scheduler,
            config=config,
        )
        self.persistence_tracker = persistence_tracker

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def analyze(
        self,
        wall: TrackedWall,
        *,
        current_mid_price: float | None = None,
    ) -> DetectorResult | None:
        """
        Аналіз одного wall не є природним для layering,
        але підтримується через пошук кластера навколо цього рівня.
        """
        cluster = self._find_cluster_around_wall(
            wall=wall,
            current_mid_price=current_mid_price,
        )
        if cluster is None:
            return None

        candidate = self._evaluate_cluster(
            cluster=cluster,
            current_mid_price=current_mid_price,
        )
        if candidate is None:
            return None

        return self._build_result(candidate)

    def analyze_many(
        self,
        walls: Iterable[TrackedWall],
        *,
        exchange: str | None = None,
        symbol: str | None = None,
        current_mid_price: float | None = None,
    ) -> list[DetectorResult]:
        """
        Аналізує набір tracked walls і повертає позитивні layering candidates.
        """
        if not self.config.enabled or not self.config.layering.enabled:
            return []

        filtered = self._filter_walls(
            walls=walls,
            exchange=exchange,
            symbol=symbol,
        )
        if not filtered:
            return []

        clusters = self._build_clusters(
            walls=filtered,
            current_mid_price=current_mid_price,
        )

        results: list[DetectorResult] = []
        seen_cluster_keys: set[str] = set()

        for cluster in clusters:
            cluster_key = self._cluster_key(cluster)
            if cluster_key in seen_cluster_keys:
                continue
            seen_cluster_keys.add(cluster_key)

            candidate = self._evaluate_cluster(
                cluster=cluster,
                current_mid_price=current_mid_price,
            )
            if candidate is None:
                continue

            result = self._build_result(candidate)
            if result.is_positive():
                results.append(result)

        results.sort(key=lambda item: (item.score, item.confidence), reverse=True)
        return results

    def analyze_symbol(
        self,
        *,
        exchange: str,
        symbol: str,
        current_mid_price: float | None = None,
    ) -> list[DetectorResult]:
        """
        Аналізує всі tracked walls одного символу.
        """
        walls = self.persistence_tracker.get_walls_for_symbol(
            exchange=exchange,
            symbol=symbol,
        )
        return self.analyze_many(
            walls=walls,
            exchange=exchange,
            symbol=symbol,
            current_mid_price=current_mid_price,
        )

    def is_layering_candidate(
        self,
        wall: TrackedWall,
        *,
        current_mid_price: float | None = None,
    ) -> bool:
        """
        Boolean helper для швидкої перевірки tracked wall.
        """
        return self.analyze(
            wall,
            current_mid_price=current_mid_price,
        ) is not None

    # -------------------------------------------------------------------------
    # Cluster construction
    # -------------------------------------------------------------------------

    def _build_clusters(
        self,
        *,
        walls: list[TrackedWall],
        current_mid_price: float | None = None,
    ) -> list[LayeringCluster]:
        """
        Будує потенційні layering-кластери для кожної сторони окремо.
        """
        if not self.config.enabled or not self.config.layering.enabled:
            return []

        relevant = [
            wall
            for wall in walls
            if self._is_relevant_wall(
                wall=wall,
                current_mid_price=current_mid_price,
            )
        ]
        if not relevant:
            return []

        clusters: list[LayeringCluster] = []
        groups = self._group_walls_by_market_side(relevant)

        for (exchange, symbol, side), side_walls in groups.items():
            ordered = self._sort_walls_for_side(side_walls, side)
            current_group: list[TrackedWall] = []

            for wall in ordered:
                if not current_group:
                    current_group.append(wall)
                    continue

                previous = current_group[-1]
                gap_bps = self.bps_distance(wall.price, previous.price)

                if gap_bps <= self.config.layering.max_price_gap_bps_between_layers:
                    current_group.append(wall)
                    continue

                cluster = self._make_cluster(
                    exchange=exchange,
                    symbol=symbol,
                    side=side,
                    walls=current_group,
                    current_mid_price=current_mid_price,
                )
                if cluster is not None:
                    clusters.append(cluster)

                current_group = [wall]

            if current_group:
                cluster = self._make_cluster(
                    exchange=exchange,
                    symbol=symbol,
                    side=side,
                    walls=current_group,
                    current_mid_price=current_mid_price,
                )
                if cluster is not None:
                    clusters.append(cluster)

        return clusters

    def _find_cluster_around_wall(
        self,
        *,
        wall: TrackedWall,
        current_mid_price: float | None = None,
    ) -> LayeringCluster | None:
        symbol_walls = self.persistence_tracker.get_walls_for_symbol(
            exchange=wall.exchange,
            symbol=wall.symbol,
            side=wall.side,
        )
        clusters = self._build_clusters(
            walls=symbol_walls,
            current_mid_price=current_mid_price,
        )

        for cluster in clusters:
            if any(item.wall_id == wall.wall_id for item in cluster.walls):
                return cluster

        return None

    def _make_cluster(
        self,
        *,
        exchange: str,
        symbol: str,
        side: SpoofingSide,
        walls: list[TrackedWall],
        current_mid_price: float | None = None,
    ) -> LayeringCluster | None:
        if len(walls) < self.config.layering.min_layers:
            return None

        total_notional = sum(wall.price * wall.max_size for wall in walls)
        if total_notional < self.config.layering.min_total_layer_notional:
            return None

        average_pull_ratio = self._mean_or_zero(wall.pull_ratio for wall in walls)
        average_fill_ratio = self._mean_or_zero(wall.fill_ratio for wall in walls)
        average_lifetime_ms = self._mean_or_zero(wall.lifetime_ms for wall in walls)
        synchronized_pull_ratio = self._estimate_synchronized_pull_ratio(walls)
        price_span_bps = self._estimate_price_span_bps(walls)

        layering_score = self._estimate_layering_score(
            walls=walls,
            total_notional=total_notional,
            synchronized_pull_ratio=synchronized_pull_ratio,
            price_span_bps=price_span_bps,
            current_mid_price=current_mid_price,
        )

        cluster_price = self._mean_or_zero(wall.price for wall in walls)
        cluster_wall_id = self._resolve_cluster_wall_id(walls)

        return LayeringCluster(
            exchange=exchange,
            symbol=symbol,
            side=side,
            walls=walls,
            total_notional=total_notional,
            average_pull_ratio=average_pull_ratio,
            average_fill_ratio=average_fill_ratio,
            average_lifetime_ms=average_lifetime_ms,
            synchronized_pull_ratio=synchronized_pull_ratio,
            price_span_bps=price_span_bps,
            layering_score=layering_score,
            cluster_price=cluster_price,
            cluster_wall_id=cluster_wall_id,
        )

    # -------------------------------------------------------------------------
    # Core detection logic
    # -------------------------------------------------------------------------

    def _evaluate_cluster(
        self,
        *,
        cluster: LayeringCluster,
        current_mid_price: float | None = None,
    ) -> LayeringCandidateContext | None:
        if not self._passes_cluster_filters(cluster):
            return None

        price_reaction_bps = self._estimate_cluster_price_reaction_bps(
            cluster=cluster,
            current_mid_price=current_mid_price,
        )

        score = self._compute_score(
            cluster=cluster,
            price_reaction_bps=price_reaction_bps,
        )
        confidence = self._compute_confidence(
            cluster=cluster,
            price_reaction_bps=price_reaction_bps,
        )
        reason = self._build_reason(
            cluster=cluster,
            price_reaction_bps=price_reaction_bps,
        )

        return LayeringCandidateContext(
            cluster=cluster,
            confidence=confidence,
            score=score,
            reason=reason,
            price_reaction_bps=price_reaction_bps,
        )

    def _passes_cluster_filters(self, cluster: LayeringCluster) -> bool:
        if len(cluster.walls) < self.config.layering.min_layers:
            return False

        if cluster.total_notional < self.config.layering.min_total_layer_notional:
            return False

        if cluster.synchronized_pull_ratio <= 0.0:
            return False

        if cluster.synchronized_pull_ratio < 0.5:
            return False

        max_allowed_fill = max(
            self.config.fake_liquidity.max_fill_ratio,
            self.config.pull_detection.max_fill_ratio_for_pull,
        )
        if cluster.average_fill_ratio > max_allowed_fill:
            return False

        if cluster.average_pull_ratio < self.config.pull_detection.min_pull_ratio:
            return False

        return True

    def _build_result(
        self,
        candidate: LayeringCandidateContext,
    ) -> DetectorResult:
        features = self._build_features(candidate)

        return DetectorResult(
            detector=self.component,
            decision=DetectorDecision.POSITIVE,
            score=candidate.score,
            confidence=candidate.confidence,
            reason=candidate.reason,
            features=features,
            wall_id=candidate.cluster.cluster_wall_id,
            pattern=SpoofingPattern.MULTI_LEVEL_LAYERING,
            metadata={
                "layers": len(candidate.cluster.walls),
                "total_notional": candidate.cluster.total_notional,
                "average_pull_ratio": candidate.cluster.average_pull_ratio,
                "average_fill_ratio": candidate.cluster.average_fill_ratio,
                "average_lifetime_ms": candidate.cluster.average_lifetime_ms,
                "synchronized_pull_ratio": candidate.cluster.synchronized_pull_ratio,
                "price_span_bps": candidate.cluster.price_span_bps,
                "price_reaction_bps": candidate.price_reaction_bps,
                "layering_score": candidate.cluster.layering_score,
            },
        )

    def _build_features(
        self,
        candidate: LayeringCandidateContext,
    ) -> SpoofingFeatures:
        cluster = candidate.cluster
        repetition_count = self._estimate_cluster_repetition_count(cluster)

        cancel_to_fill_ratio = self._compute_cancel_to_fill_ratio(
            pull_ratio=cluster.average_pull_ratio,
            fill_ratio=cluster.average_fill_ratio,
        )

        reference_mid = self._resolve_cluster_reference_mid(cluster)
        distance_from_mid_bps = (
            self.bps_distance(cluster.cluster_price, reference_mid)
            if reference_mid is not None and reference_mid > 0
            else 0.0
        )

        return SpoofingFeatures(
            symbol=cluster.symbol,
            exchange=cluster.exchange,
            side=cluster.side,
            price=cluster.cluster_price,
            wall_size=sum(wall.max_size for wall in cluster.walls),
            wall_size_ratio=self._estimate_cluster_wall_size_ratio(cluster),
            distance_from_mid_bps=distance_from_mid_bps,
            lifetime_ms=cluster.average_lifetime_ms,
            updates_count=sum(wall.updates_count for wall in cluster.walls),
            repetition_count=repetition_count,
            fill_ratio=cluster.average_fill_ratio,
            pull_ratio=cluster.average_pull_ratio,
            cancel_to_fill_ratio=cancel_to_fill_ratio,
            price_reaction_bps=candidate.price_reaction_bps,
            pressure_flip_strength=0.0,
            layering_score=cluster.layering_score,
            is_near_best_quote=self._cluster_near_best_quote(cluster),
            is_fast_pull=self._cluster_fast_pull(cluster),
            is_fake_liquidity=False,
            is_layering=True,
            metadata={
                "layers": len(cluster.walls),
                "total_notional": cluster.total_notional,
                "synchronized_pull_ratio": cluster.synchronized_pull_ratio,
                "price_span_bps": cluster.price_span_bps,
                "wall_ids": [wall.wall_id for wall in cluster.walls],
                "detector": self.component.value,
            },
        )

    # -------------------------------------------------------------------------
    # Scoring / confidence
    # -------------------------------------------------------------------------

    def _compute_score(
        self,
        *,
        cluster: LayeringCluster,
        price_reaction_bps: float,
    ) -> float:
        notional_component = self._normalize_total_notional(cluster.total_notional)
        pull_component = self.clamp(cluster.average_pull_ratio, 0.0, 1.0)
        fill_component = 1.0 - self.clamp(cluster.average_fill_ratio, 0.0, 1.0)
        sync_component = self.clamp(cluster.synchronized_pull_ratio, 0.0, 1.0)

        max_span = self._cluster_max_span_bps(cluster)
        compactness_component = 1.0 - self.clamp(
            cluster.price_span_bps / max(max_span, 1e-12),
            0.0,
            1.0,
        )

        layering_component = self.clamp(cluster.layering_score, 0.0, 1.0)
        reaction_component = self.clamp(
            price_reaction_bps
            / max(self.config.flip_pressure.min_price_reaction_bps * 3.0, 1e-12),
            0.0,
            1.0,
        )

        raw_score = (
            0.18 * notional_component
            + 0.17 * pull_component
            + 0.12 * fill_component
            + 0.20 * sync_component
            + 0.12 * compactness_component
            + 0.16 * layering_component
            + 0.05 * reaction_component
        )
        return self.clamp(raw_score, 0.0, 1.0)

    def _compute_confidence(
        self,
        *,
        cluster: LayeringCluster,
        price_reaction_bps: float,
    ) -> float:
        confidence = 0.42

        if len(cluster.walls) >= self.config.layering.min_layers + 1:
            confidence += 0.10

        if cluster.synchronized_pull_ratio >= 0.75:
            confidence += 0.14

        if cluster.average_pull_ratio >= self.config.pull_detection.strong_pull_ratio:
            confidence += 0.10

        if cluster.average_fill_ratio <= self.config.fake_liquidity.max_fill_ratio * 0.75:
            confidence += 0.08

        if cluster.total_notional >= self.config.layering.min_total_layer_notional * 2.0:
            confidence += 0.08

        if price_reaction_bps >= self.config.flip_pressure.min_price_reaction_bps:
            confidence += 0.04

        if cluster.layering_score >= 0.75:
            confidence += 0.03

        return self.clamp(confidence, 0.0, 0.99)

    # -------------------------------------------------------------------------
    # Cluster metrics
    # -------------------------------------------------------------------------

    def _estimate_synchronized_pull_ratio(
        self,
        walls: list[TrackedWall],
    ) -> float:
        """
        Частка рівнів, які ослабли/зникли в близькому часовому вікні.
        """
        if not walls:
            return 0.0

        relevant_states = self._removed_or_weakened_states()
        reference_times = [
            wall.last_seen_at
            for wall in walls
            if wall.state in relevant_states
        ]

        if len(reference_times) < self.config.layering.min_layers:
            return 0.0

        reference_times.sort()
        window_ms = self.config.layering.synchronized_pull_window_ms

        best_count = 0
        for base_time in reference_times:
            count = sum(
                1
                for ts in reference_times
                if abs((ts - base_time).total_seconds() * 1000.0) <= window_ms
            )
            best_count = max(best_count, count)

        return self.normalize_ratio(best_count, len(walls))

    def _estimate_price_span_bps(
        self,
        walls: list[TrackedWall],
    ) -> float:
        if len(walls) < 2:
            return 0.0

        prices = [wall.price for wall in walls if wall.price > 0]
        if len(prices) < 2:
            return 0.0

        return self.bps_distance(max(prices), min(prices))

    def _estimate_layering_score(
        self,
        *,
        walls: list[TrackedWall],
        total_notional: float,
        synchronized_pull_ratio: float,
        price_span_bps: float,
        current_mid_price: float | None = None,
    ) -> float:
        layers_component = self.clamp(
            len(walls) / max(self.config.layering.min_layers + 2, 1),
            0.0,
            1.0,
        )
        notional_component = self._normalize_total_notional(total_notional)
        sync_component = self.clamp(synchronized_pull_ratio, 0.0, 1.0)

        max_span = self.config.layering.max_price_gap_bps_between_layers * max(
            len(walls) - 1,
            1,
        )
        compactness_component = 1.0 - self.clamp(
            price_span_bps / max(max_span, 1e-12),
            0.0,
            1.0,
        )

        proximity_component = 0.5
        if current_mid_price is not None and current_mid_price > 0:
            cluster_price = self._mean_or_zero(wall.price for wall in walls)
            distance = self.bps_distance(cluster_price, current_mid_price)
            max_distance = max(
                self.config.wall_detection.max_distance_from_mid_bps,
                1e-12,
            )
            proximity_component = 1.0 - self.clamp(
                distance / max_distance,
                0.0,
                1.0,
            )

        value = (
            0.20 * layers_component
            + 0.25 * notional_component
            + 0.30 * sync_component
            + 0.15 * compactness_component
            + 0.10 * proximity_component
        )
        return self.clamp(value, 0.0, 1.0)

    def _estimate_cluster_price_reaction_bps(
        self,
        *,
        cluster: LayeringCluster,
        current_mid_price: float | None,
    ) -> float:
        if current_mid_price is None or current_mid_price <= 0:
            return 0.0

        reference_mid = self._resolve_cluster_reference_mid(cluster)
        if reference_mid is None or reference_mid <= 0:
            return 0.0

        signed_move = self.signed_bps_move(current_mid_price, reference_mid)

        if cluster.side == SpoofingSide.ASK:
            return max(0.0, signed_move)

        if cluster.side == SpoofingSide.BID:
            return max(0.0, -signed_move)

        return 0.0

    def _estimate_cluster_wall_size_ratio(
        self,
        cluster: LayeringCluster,
    ) -> float:
        """
        Груба оцінка cluster-vs-baseline ratio.
        """
        base_notional = max(self.config.wall_detection.min_wall_size_abs, 1e-12)
        return self.clamp(cluster.total_notional / base_notional, 0.0, 1000.0)

    # -------------------------------------------------------------------------
    # Filters / helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _filter_walls(
        *,
        walls: Iterable[TrackedWall],
        exchange: str | None,
        symbol: str | None,
    ) -> list[TrackedWall]:
        return [
            wall
            for wall in walls
            if (exchange is None or wall.exchange == exchange)
            and (symbol is None or wall.symbol == symbol)
        ]

    def _is_relevant_wall(
        self,
        *,
        wall: TrackedWall,
        current_mid_price: float | None = None,
    ) -> bool:
        if wall.max_size <= 0 or wall.price <= 0:
            return False

        if wall.price * wall.max_size < self.config.wall_detection.min_wall_size_abs:
            return False

        if wall.state not in self._candidate_wall_states():
            return False

        if current_mid_price is not None and current_mid_price > 0:
            distance = self.bps_distance(wall.price, current_mid_price)
            if distance > self.config.wall_detection.max_distance_from_mid_bps * 2.0:
                return False

        if wall.side not in {SpoofingSide.BID, SpoofingSide.ASK}:
            return False

        return True

    @staticmethod
    def _group_walls_by_market_side(
        walls: Iterable[TrackedWall],
    ) -> dict[tuple[str, str, SpoofingSide], list[TrackedWall]]:
        groups: dict[tuple[str, str, SpoofingSide], list[TrackedWall]] = {}

        for wall in walls:
            key = (wall.exchange, wall.symbol, wall.side)
            groups.setdefault(key, []).append(wall)

        return groups

    @staticmethod
    def _sort_walls_for_side(
        walls: list[TrackedWall],
        side: SpoofingSide,
    ) -> list[TrackedWall]:
        if side == SpoofingSide.BID:
            return sorted(walls, key=lambda item: item.price, reverse=True)
        return sorted(walls, key=lambda item: item.price)

    @staticmethod
    def _resolve_cluster_wall_id(
        walls: list[TrackedWall],
    ) -> str | None:
        if not walls:
            return None

        strongest = max(walls, key=lambda item: item.price * item.max_size)
        return strongest.wall_id

    @staticmethod
    def _resolve_cluster_reference_mid(
        cluster: LayeringCluster,
    ) -> float | None:
        mids = [
            wall.mid_price_at_creation
            for wall in cluster.walls
            if wall.mid_price_at_creation is not None and wall.mid_price_at_creation > 0
        ]
        if not mids:
            return None
        return mean(mids)

    def _normalize_total_notional(self, total_notional: float) -> float:
        base = max(self.config.layering.min_total_layer_notional, 1e-12)
        value = (total_notional - base) / max(base * 2.0, 1e-12)
        return self.clamp(value, 0.0, 1.0)

    def _cluster_near_best_quote(
        self,
        cluster: LayeringCluster,
    ) -> bool:
        return any(
            wall.touch_count > 0 or wall.near_touch_count > 0
            for wall in cluster.walls
        )

    def _cluster_fast_pull(
        self,
        cluster: LayeringCluster,
    ) -> bool:
        return cluster.average_lifetime_ms <= self.config.pull_detection.fast_pull_lifetime_ms

    def _estimate_cluster_repetition_count(
        self,
        cluster: LayeringCluster,
    ) -> int:
        if not cluster.walls:
            return 0

        counts = []
        for wall in cluster.walls:
            history = self.persistence_tracker.get_recent_history(
                exchange=wall.exchange,
                symbol=wall.symbol,
                side=wall.side,
                price=wall.price,
                limit=50,
            )
            counts.append(len(history))

        return int(mean(counts)) if counts else 0

    @staticmethod
    def _cluster_key(cluster: LayeringCluster) -> str:
        wall_ids = sorted(wall.wall_id for wall in cluster.walls)
        return f"{cluster.exchange}:{cluster.symbol}:{cluster.side.value}:{'|'.join(wall_ids)}"

    @staticmethod
    def _candidate_wall_states() -> set[OrderbookWallState]:
        return {
            OrderbookWallState.ACTIVE,
            OrderbookWallState.WEAKENING,
            OrderbookWallState.PULLED,
            OrderbookWallState.EXPIRED,
            OrderbookWallState.FILLED,
        }

    @staticmethod
    def _removed_or_weakened_states() -> set[OrderbookWallState]:
        return {
            OrderbookWallState.WEAKENING,
            OrderbookWallState.PULLED,
            OrderbookWallState.EXPIRED,
            OrderbookWallState.FILLED,
        }

    @staticmethod
    def _compute_cancel_to_fill_ratio(
        *,
        pull_ratio: float,
        fill_ratio: float,
    ) -> float:
        if fill_ratio > 0:
            return pull_ratio / fill_ratio
        if pull_ratio > 0:
            return pull_ratio
        return 0.0

    def _cluster_max_span_bps(self, cluster: LayeringCluster) -> float:
        return self.config.layering.max_price_gap_bps_between_layers * max(
            len(cluster.walls) - 1,
            1,
        )

    @staticmethod
    def _mean_or_zero(values: Iterable[float]) -> float:
        items = [max(value, 0.0) for value in values]
        return mean(items) if items else 0.0

    @staticmethod
    def _build_reason(
        *,
        cluster: LayeringCluster,
        price_reaction_bps: float,
    ) -> str:
        return (
            f"layering candidate detected for {cluster.side.value.upper()} side, "
            f"layers={len(cluster.walls)}, "
            f"total_notional={cluster.total_notional:.2f}, "
            f"avg_pull_ratio={cluster.average_pull_ratio:.4f}, "
            f"avg_fill_ratio={cluster.average_fill_ratio:.4f}, "
            f"sync_pull_ratio={cluster.synchronized_pull_ratio:.4f}, "
            f"price_span_bps={cluster.price_span_bps:.4f}, "
            f"layering_score={cluster.layering_score:.4f}, "
            f"price_reaction_bps={price_reaction_bps:.4f}"
        )


__all__ = ["LayeringDetector"]