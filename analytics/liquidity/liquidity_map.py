from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from core.logger import get_logger

from analytics.liquidity.config import LiquidityConfig
from analytics.liquidity.enums import LiquidityBias, LiquidityLevelType, LiquiditySide
from analytics.liquidity.equal_highs_lows import EqualHighsLowsDetector
from analytics.liquidity.models import (
    DEFAULT_EXCHANGE,
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    EqualLevel,
    LiquidityKey,
    LiquidityLevel,
    LiquidityMapSnapshot,
    LiquiditySignal,
    LiquidityZone,
    StopCluster,
    ensure_utc,
    liquidity_key_to_dict,
    liquidity_key_to_string,
    make_liquidity_key,
    normalize_exchange,
    normalize_market_type,
    normalize_symbol,
    normalize_timeframe,
    utc_now,
)
from .scoring import LiquidityScorer
from .stop_clusters import StopClustersDetector
from .utils import (
    clamp,
    get_first_value,
    merge_price_ranges,
    midpoint,
    pct_distance,
    safe_float,
    safe_mean,
)


@dataclass(slots=True)
class LiquidityMapFeatures:
    """
    Внутрішня структура проміжних ознак liquidity landscape.

    Не є event payload і не має залежностей від EventBus/Scheduler.

    Semantics:
    - pressure_score > 0 means upside / buy-side liquidity pressure is stronger;
    - pressure_score < 0 means downside / sell-side liquidity pressure is stronger;
    - pressure_score == 0 means balanced liquidity pressure.
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

    def __post_init__(self) -> None:
        self.above_liquidity_score = clamp(self.above_liquidity_score, 0.0, 1.0)
        self.below_liquidity_score = clamp(self.below_liquidity_score, 0.0, 1.0)
        self.sweep_risk_up = clamp(self.sweep_risk_up, 0.0, 1.0)
        self.sweep_risk_down = clamp(self.sweep_risk_down, 0.0, 1.0)
        self.magnet_score_up = clamp(self.magnet_score_up, 0.0, 1.0)
        self.magnet_score_down = clamp(self.magnet_score_down, 0.0, 1.0)
        self.pressure_score = clamp(self.pressure_score, -1.0, 1.0)


class LiquidityMap:
    """
    Production-ready агрегатор liquidity-домену.

    Відповідальність:
    - побудувати equal highs / equal lows;
    - побудувати stop clusters;
    - об'єднати додаткові liquidity levels/clusters;
    - побудувати liquidity zones;
    - розрахувати sweep risk, magnet score, signed pressure, bias;
    - сформувати LiquiditySignal;
    - повернути LiquidityMapSnapshot.

    Архітектурні правила:
    - не приймає EventBus;
    - не приймає Scheduler;
    - не публікує події;
    - не керує lifecycle;
    - використовується LiquidityService як чистий domain aggregator.

    Multi-exchange scope:
        exchange + market_type + symbol + timeframe
    """

    def __init__(
        self,
        config: LiquidityConfig,
        equal_detector: EqualHighsLowsDetector | None = None,
        stop_detector: StopClustersDetector | None = None,
        scorer: LiquidityScorer | None = None,
    ) -> None:
        self._config = config
        self._config.validate()

        self._scorer = scorer or LiquidityScorer(config=config)

        self._equal_detector = equal_detector or EqualHighsLowsDetector(
            config=config,
            scorer=self._scorer,
        )

        self._stop_detector = stop_detector or StopClustersDetector(
            config=config,
            scorer=self._scorer,
        )

        self._logger = get_logger(
            __name__,
            service_name="analytics_liquidity",
            event_type="liquidity_map",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
    ) -> LiquidityMapSnapshot:
        """
        Повна побудова liquidity map snapshot для:
            exchange + market_type + symbol + timeframe

        У production LiquidityService має передавати exchange/market_type явно.
        """
        if not self._config.enabled:
            raise RuntimeError("LiquidityMap is disabled by LiquidityConfig")

        key = self._make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        scope = liquidity_key_to_dict(key)

        exchange = scope["exchange"]
        market_type = scope["market_type"]
        symbol = scope["symbol"]
        timeframe = scope["timeframe"]

        self._validate_symbol_timeframe(symbol=symbol, timeframe=timeframe)

        current_price = safe_float(current_price)
        if current_price <= 0:
            raise ValueError("current_price must be > 0")

        candles_list = list(candles)
        snapshot_ts = self._normalize_timestamp(timestamp) or self._resolve_snapshot_timestamp(
            candles_list
        )

        equal_levels = self._equal_detector.detect(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_list,
            current_price=current_price,
        )
        self._assert_and_scope_levels(
            levels=equal_levels,
            key=key,
        )

        scoped_extra_levels = list(extra_levels or [])
        self._assert_and_scope_levels(
            levels=scoped_extra_levels,
            key=key,
        )

        merged_levels = self._merge_levels(
            equal_levels=equal_levels,
            extra_levels=scoped_extra_levels,
        )
        active_levels = self._limit_levels(self._filter_active_levels(merged_levels))

        stop_clusters = self._stop_detector.detect_from_levels(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            levels=active_levels,
            current_price=current_price,
            candles=candles_list,
            orderbook=orderbook,
        )
        self._assert_and_scope_clusters(
            clusters=stop_clusters,
            key=key,
        )

        scoped_extra_clusters = list(extra_clusters or [])
        self._assert_and_scope_clusters(
            clusters=scoped_extra_clusters,
            key=key,
        )

        if scoped_extra_clusters:
            stop_clusters = self._merge_clusters(
                primary=stop_clusters,
                extra=scoped_extra_clusters,
                current_price=current_price,
            )
            self._assert_and_scope_clusters(
                clusters=stop_clusters,
                key=key,
            )

        active_clusters = self._limit_clusters(self._filter_active_clusters(stop_clusters))

        zones = self._build_liquidity_zones(
            key=key,
            current_price=current_price,
            levels=active_levels,
            clusters=active_clusters,
        )

        features = self._extract_features(
            current_price=current_price,
            levels=active_levels,
            clusters=active_clusters,
        )

        signal = self._build_signal(
            key=key,
            timestamp=snapshot_ts,
            features=features,
        )

        snapshot = LiquidityMapSnapshot(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=snapshot_ts,
            current_price=current_price,
            active_levels=active_levels,
            equal_levels=list(equal_levels),
            stop_clusters=active_clusters,
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
                "builder": self.__class__.__name__,
                "scope": scope,
                "scope_key": liquidity_key_to_string(key),
                "exchange": exchange,
                "market_type": market_type,
                "levels_count": len(active_levels),
                "raw_merged_levels_count": len(merged_levels),
                "equal_levels_count": len(equal_levels),
                "stop_clusters_count": len(active_clusters),
                "zones_count": len(zones),
                "sweep_risk_up": features.sweep_risk_up,
                "sweep_risk_down": features.sweep_risk_down,
                "magnet_score_up": features.magnet_score_up,
                "magnet_score_down": features.magnet_score_down,
                "pressure_score_semantics": (
                    "positive=upside_buy_side, negative=downside_sell_side"
                ),
                "orderbook_present": orderbook is not None,
                "extra_levels_count": len(extra_levels or []),
                "extra_clusters_count": len(extra_clusters or []),
            },
        )

        self._logger.info(
            "Liquidity map snapshot built",
            extra={
                "scope": scope,
                "scope_key": liquidity_key_to_string(key),
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "current_price": current_price,
                "levels_count": len(active_levels),
                "equal_levels_count": len(equal_levels),
                "clusters_count": len(active_clusters),
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
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
    ) -> LiquidityMapSnapshot:
        """
        Альтернативний шлях побудови snapshot, якщо рівні/кластери вже пораховані.
        """
        key = self._make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        scope = liquidity_key_to_dict(key)

        exchange = scope["exchange"]
        market_type = scope["market_type"]
        symbol = scope["symbol"]
        timeframe = scope["timeframe"]

        self._validate_symbol_timeframe(symbol=symbol, timeframe=timeframe)

        current_price = safe_float(current_price)
        if current_price <= 0:
            raise ValueError("current_price must be > 0")

        snapshot_ts = self._normalize_timestamp(timestamp) or utc_now()

        scoped_levels = list(levels)
        self._assert_and_scope_levels(
            levels=scoped_levels,
            key=key,
        )

        scoped_equal_levels = list(equal_levels or [])
        self._assert_and_scope_levels(
            levels=scoped_equal_levels,
            key=key,
        )

        scoped_clusters = list(clusters)
        self._assert_and_scope_clusters(
            clusters=scoped_clusters,
            key=key,
        )

        active_levels = self._limit_levels(self._filter_active_levels(scoped_levels))

        stop_clusters = self._merge_clusters(
            primary=scoped_clusters,
            extra=[],
            current_price=current_price,
        )
        self._assert_and_scope_clusters(
            clusters=stop_clusters,
            key=key,
        )

        active_clusters = self._limit_clusters(self._filter_active_clusters(stop_clusters))

        zones = self._build_liquidity_zones(
            key=key,
            current_price=current_price,
            levels=active_levels,
            clusters=active_clusters,
        )

        features = self._extract_features(
            current_price=current_price,
            levels=active_levels,
            clusters=active_clusters,
        )

        signal = self._build_signal(
            key=key,
            timestamp=snapshot_ts,
            features=features,
        )

        return LiquidityMapSnapshot(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=snapshot_ts,
            current_price=current_price,
            active_levels=active_levels,
            equal_levels=scoped_equal_levels,
            stop_clusters=active_clusters,
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
                "builder": self.__class__.__name__,
                "from_components": True,
                "scope": scope,
                "scope_key": liquidity_key_to_string(key),
                "exchange": exchange,
                "market_type": market_type,
                "levels_count": len(active_levels),
                "equal_levels_count": len(scoped_equal_levels),
                "stop_clusters_count": len(active_clusters),
                "zones_count": len(zones),
                "sweep_risk_up": features.sweep_risk_up,
                "sweep_risk_down": features.sweep_risk_down,
                "magnet_score_up": features.magnet_score_up,
                "magnet_score_down": features.magnet_score_down,
                "pressure_score_semantics": (
                    "positive=upside_buy_side, negative=downside_sell_side"
                ),
            },
        )

    # ------------------------------------------------------------------
    # Merge / filtering
    # ------------------------------------------------------------------

    def _merge_levels(
        self,
        equal_levels: Sequence[EqualLevel],
        extra_levels: Sequence[LiquidityLevel] | None,
    ) -> list[LiquidityLevel]:
        merged: list[LiquidityLevel] = [
            level
            for level in [*equal_levels, *(extra_levels or [])]
            if level.price > 0
        ]

        if not merged:
            return []

        merged.sort(
            key=lambda level: (
                level.exchange,
                level.market_type,
                level.symbol,
                level.timeframe,
                level.side.value,
                level.level_type.value,
                level.price,
            )
        )

        deduplicated: list[LiquidityLevel] = []

        for level in merged:
            duplicate_index = self._find_duplicate_level_index(
                levels=deduplicated,
                candidate=level,
            )

            if duplicate_index is None:
                deduplicated.append(level)
                continue

            current = deduplicated[duplicate_index]
            if self._is_better_level(level, current):
                deduplicated[duplicate_index] = level

        deduplicated.sort(key=lambda level: level.price)
        return deduplicated

    def _filter_active_levels(
        self,
        levels: Sequence[LiquidityLevel],
    ) -> list[LiquidityLevel]:
        return [
            level
            for level in levels
            if level.price > 0 and level.is_active()
        ]

    def _filter_active_clusters(
        self,
        clusters: Sequence[StopCluster],
    ) -> list[StopCluster]:
        return [
            cluster
            for cluster in clusters
            if cluster.center_price > 0 and cluster.is_active()
        ]

    def _find_duplicate_level_index(
        self,
        levels: list[LiquidityLevel],
        candidate: LiquidityLevel,
    ) -> int | None:
        for index, level in enumerate(levels):
            same_scope = level.liquidity_key == candidate.liquidity_key
            same_type = level.level_type == candidate.level_type
            same_side = level.side == candidate.side
            near = (
                pct_distance(level.price, candidate.price)
                <= self._config.equal_level_tolerance_pct
            )

            if same_scope and same_type and same_side and near:
                return index

        return None

    def _is_better_level(
        self,
        candidate: LiquidityLevel,
        current: LiquidityLevel,
    ) -> bool:
        candidate_score = (
            1 if candidate.is_active() else 0,
            candidate.confidence,
            candidate.touches_count,
            candidate.reaction_count,
        )
        current_score = (
            1 if current.is_active() else 0,
            current.confidence,
            current.touches_count,
            current.reaction_count,
        )
        return candidate_score > current_score

    def _merge_clusters(
        self,
        primary: Sequence[StopCluster],
        extra: Sequence[StopCluster],
        current_price: float,
    ) -> list[StopCluster]:
        all_clusters = [
            cluster
            for cluster in [*primary, *extra]
            if cluster.center_price > 0
        ]

        if not all_clusters:
            return []

        all_clusters.sort(
            key=lambda cluster: (
                cluster.exchange,
                cluster.market_type,
                cluster.symbol,
                cluster.timeframe,
                cluster.side.value,
                cluster.center_price,
            )
        )

        merged: list[StopCluster] = [all_clusters[0]]

        for cluster in all_clusters[1:]:
            previous = merged[-1]

            same_scope = cluster.liquidity_key == previous.liquidity_key

            if not same_scope or cluster.side != previous.side:
                merged.append(cluster)
                continue

            if self._clusters_are_close(previous, cluster):
                merged[-1] = self._merge_two_clusters(
                    left=previous,
                    right=cluster,
                    current_price=current_price,
                )
            else:
                merged.append(cluster)

        return merged

    def _clusters_are_close(
        self,
        left: StopCluster,
        right: StopCluster,
    ) -> bool:
        if left.liquidity_key != right.liquidity_key:
            return False

        if left.overlaps(right):
            return True

        return (
            pct_distance(left.center_price, right.center_price)
            <= self._config.cluster_merge_distance_pct
        )

    def _merge_two_clusters(
        self,
        left: StopCluster,
        right: StopCluster,
        current_price: float,
    ) -> StopCluster:
        if left.liquidity_key != right.liquidity_key:
            raise ValueError(
                "Cannot merge stop clusters from different scopes: "
                f"left={left.scope}, right={right.scope}"
            )

        source_levels = self._deduplicate_source_levels(
            [*left.source_levels, *right.source_levels]
        )

        merged = StopCluster(
            exchange=left.exchange,
            market_type=left.market_type,
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
            estimated_stop_density=max(
                left.estimated_stop_density,
                right.estimated_stop_density,
            ),
            touches_count=left.touches_count + right.touches_count,
            source_level_type=(
                left.source_level_type
                if left.confidence >= right.confidence
                else right.source_level_type
            ),
            strength=(
                left.strength
                if left.confidence >= right.confidence
                else right.strength
            ),
            created_at=self._min_datetime(left.created_at, right.created_at),
            updated_at=self._max_datetime(left.updated_at, right.updated_at),
            invalidated_at=self._max_datetime(left.invalidated_at, right.invalidated_at),
            swept_at=self._max_datetime(left.swept_at, right.swept_at),
            source_levels=source_levels,
            metadata={
                "merged": True,
                "scope": left.scope,
                "scope_key": left.scope_key,
                "left_metadata": dict(left.metadata),
                "right_metadata": dict(right.metadata),
                "source_count": len(source_levels),
                "left_key": left.key,
                "right_key": right.key,
            },
        )

        merged.confidence = self._scorer.score_stop_cluster(
            cluster=merged,
            current_price=current_price,
        )
        merged.strength = self._scorer.classify_cluster_strength(merged.confidence)

        return merged

    def _deduplicate_source_levels(
        self,
        levels: Sequence[LiquidityLevel],
    ) -> list[LiquidityLevel]:
        by_key: dict[str, LiquidityLevel] = {}

        for level in levels:
            existing = by_key.get(level.key)
            if existing is None or level.confidence > existing.confidence:
                by_key[level.key] = level

        return list(by_key.values())

    def _limit_levels(
        self,
        levels: Sequence[LiquidityLevel],
    ) -> list[LiquidityLevel]:
        sorted_levels = sorted(
            levels,
            key=lambda level: (
                level.confidence,
                level.touches_count,
                level.reaction_count,
            ),
            reverse=True,
        )
        return sorted_levels[: self._config.max_active_levels]

    def _limit_clusters(
        self,
        clusters: Sequence[StopCluster],
    ) -> list[StopCluster]:
        sorted_clusters = sorted(
            clusters,
            key=lambda cluster: (
                cluster.confidence,
                cluster.estimated_stop_density,
                cluster.touches_count,
            ),
            reverse=True,
        )
        return sorted_clusters[: self._config.max_active_clusters]

    # ------------------------------------------------------------------
    # Zones
    # ------------------------------------------------------------------

    def _build_liquidity_zones(
        self,
        key: LiquidityKey,
        current_price: float,
        levels: Sequence[LiquidityLevel],
        clusters: Sequence[StopCluster],
    ) -> list[LiquidityZone]:
        scope = liquidity_key_to_dict(key)

        ranges: list[tuple[float, float]] = []

        for level in levels:
            ranges.append(self._level_to_zone_range(level))

        for cluster in clusters:
            ranges.append((cluster.low_price, cluster.high_price))

        merged_ranges = merge_price_ranges(
            ranges=ranges,
            merge_distance_pct=self._config.cluster_merge_distance_pct,
        )

        zones: list[LiquidityZone] = []

        for low_price, high_price in merged_ranges:
            zone_levels = [
                level
                for level in levels
                if low_price <= level.price <= high_price
            ]

            zone_clusters = [
                cluster
                for cluster in clusters
                if not (
                    cluster.high_price < low_price
                    or cluster.low_price > high_price
                )
            ]

            side = self._infer_zone_side(
                zone_levels=zone_levels,
                zone_clusters=zone_clusters,
                current_price=current_price,
                low_price=low_price,
                high_price=high_price,
            )

            score = self._calculate_zone_score(
                zone_levels=zone_levels,
                zone_clusters=zone_clusters,
                current_price=current_price,
                low_price=low_price,
                high_price=high_price,
            )

            source_types = self._collect_zone_source_types(
                zone_levels=zone_levels,
                zone_clusters=zone_clusters,
            )

            label = self._build_zone_label(
                side=side,
                score=score,
                current_price=current_price,
                low_price=low_price,
                high_price=high_price,
            )

            zones.append(
                LiquidityZone(
                    exchange=scope["exchange"],
                    market_type=scope["market_type"],
                    symbol=scope["symbol"],
                    timeframe=scope["timeframe"],
                    side=side,
                    low_price=low_price,
                    high_price=high_price,
                    score=score,
                    label=label,
                    source_types=source_types,
                    metadata={
                        "scope": scope,
                        "scope_key": liquidity_key_to_string(key),
                        "levels_count": len(zone_levels),
                        "clusters_count": len(zone_clusters),
                        "distance_from_price_pct": self._distance_from_zone_pct(
                            current_price=current_price,
                            low_price=low_price,
                            high_price=high_price,
                        ),
                    },
                )
            )

        zones.sort(key=lambda zone: zone.center_price)
        return zones

    def _level_to_zone_range(
        self,
        level: LiquidityLevel,
    ) -> tuple[float, float]:
        if isinstance(level, EqualLevel):
            return (
                safe_float(level.cluster_low, default=level.price),
                safe_float(level.cluster_high, default=level.price),
            )

        tolerance = self._config.equal_level_tolerance_pct

        low_price = level.price * (1.0 - tolerance)
        high_price = level.price * (1.0 + tolerance)

        return min(low_price, high_price), max(low_price, high_price)

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

    def _calculate_zone_score(
        self,
        zone_levels: Sequence[LiquidityLevel],
        zone_clusters: Sequence[StopCluster],
        current_price: float,
        low_price: float,
        high_price: float,
    ) -> float:
        center = midpoint(low_price, high_price)

        distance_score = 1.0 - clamp(
            pct_distance(center, current_price) / 0.04,
            0.0,
            1.0,
        )

        level_component = safe_mean(
            [level.confidence for level in zone_levels]
        )

        cluster_component = safe_mean(
            [cluster.confidence for cluster in zone_clusters]
        )

        density_component = safe_mean(
            [cluster.estimated_stop_density for cluster in zone_clusters]
        )

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

        return sorted(source_types, key=lambda level_type: level_type.value)

    def _build_zone_label(
        self,
        side: LiquiditySide,
        score: float,
        current_price: float,
        low_price: float,
        high_price: float,
    ) -> str:
        position = (
            "above"
            if midpoint(low_price, high_price) > current_price
            else "below"
        )

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

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(
        self,
        current_price: float,
        levels: Sequence[LiquidityLevel],
        clusters: Sequence[StopCluster],
    ) -> LiquidityMapFeatures:
        active_levels = self._filter_active_levels(levels)
        active_clusters = self._filter_active_clusters(clusters)

        above_levels = [
            level
            for level in active_levels
            if level.price > current_price
        ]

        below_levels = [
            level
            for level in active_levels
            if level.price < current_price
        ]

        above_clusters = [
            cluster
            for cluster in active_clusters
            if cluster.center_price > current_price
        ]

        below_clusters = [
            cluster
            for cluster in active_clusters
            if cluster.center_price < current_price
        ]

        nearest_above_level = self._find_nearest_above(
            current_price=current_price,
            levels=active_levels,
            clusters=active_clusters,
        )

        nearest_below_level = self._find_nearest_below(
            current_price=current_price,
            levels=active_levels,
            clusters=active_clusters,
        )

        strongest_cluster_above = self._find_strongest_cluster_above(
            current_price=current_price,
            clusters=active_clusters,
        )

        strongest_cluster_below = self._find_strongest_cluster_below(
            current_price=current_price,
            clusters=active_clusters,
        )

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

        magnet_score_up = self._calculate_magnet_score(
            current_price=current_price,
            levels=above_levels,
            clusters=above_clusters,
        )

        magnet_score_down = self._calculate_magnet_score(
            current_price=current_price,
            levels=below_levels,
            clusters=below_clusters,
        )

        pressure_score = self._calculate_pressure_score(
            above_liquidity_score=above_liquidity_score,
            below_liquidity_score=below_liquidity_score,
            magnet_score_up=magnet_score_up,
            magnet_score_down=magnet_score_down,
            sweep_risk_up=sweep_risk_up,
            sweep_risk_down=sweep_risk_down,
        )

        bias = self._infer_bias_from_pressure(
            pressure_score=pressure_score,
            above_liquidity_score=above_liquidity_score,
            below_liquidity_score=below_liquidity_score,
        )

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

    def _find_nearest_above(
        self,
        current_price: float,
        levels: Sequence[LiquidityLevel],
        clusters: Sequence[StopCluster],
    ) -> LiquidityLevel | StopCluster | None:
        candidates: list[LiquidityLevel | StopCluster] = [
            level
            for level in levels
            if level.price > current_price and level.is_active()
        ]

        candidates.extend(
            cluster
            for cluster in clusters
            if cluster.center_price > current_price and cluster.is_active()
        )

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda item: self._reference_price(item),
        )

    def _find_nearest_below(
        self,
        current_price: float,
        levels: Sequence[LiquidityLevel],
        clusters: Sequence[StopCluster],
    ) -> LiquidityLevel | StopCluster | None:
        candidates: list[LiquidityLevel | StopCluster] = [
            level
            for level in levels
            if level.price < current_price and level.is_active()
        ]

        candidates.extend(
            cluster
            for cluster in clusters
            if cluster.center_price < current_price and cluster.is_active()
        )

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda item: self._reference_price(item),
        )

    def _find_strongest_cluster_above(
        self,
        current_price: float,
        clusters: Sequence[StopCluster],
    ) -> StopCluster | None:
        candidates = [
            cluster
            for cluster in clusters
            if cluster.center_price > current_price and cluster.is_active()
        ]

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda cluster: (
                cluster.confidence,
                cluster.estimated_stop_density,
                cluster.touches_count,
            ),
        )

    def _find_strongest_cluster_below(
        self,
        current_price: float,
        clusters: Sequence[StopCluster],
    ) -> StopCluster | None:
        candidates = [
            cluster
            for cluster in clusters
            if cluster.center_price < current_price and cluster.is_active()
        ]

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda cluster: (
                cluster.confidence,
                cluster.estimated_stop_density,
                cluster.touches_count,
            ),
        )

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------

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

        cluster_scores: list[float] = []

        for cluster in clusters:
            proximity = 1.0 - clamp(
                pct_distance(cluster.center_price, current_price) / 0.03,
                0.0,
                1.0,
            )

            swept_penalty = 0.15 if cluster.is_swept() else 0.0

            cluster_scores.append(
                clamp(
                    0.65 * cluster.confidence
                    + 0.25 * cluster.estimated_stop_density
                    + 0.10 * proximity
                    - swept_penalty,
                    0.0,
                    1.0,
                )
            )

        avg_cluster_score = safe_mean(cluster_scores)

        if not clusters:
            return level_score

        if not levels:
            return avg_cluster_score

        return clamp(
            0.55 * level_score + 0.45 * avg_cluster_score,
            0.0,
            1.0,
        )

    def _calculate_sweep_risk_up(
        self,
        current_price: float,
        above_levels: Sequence[LiquidityLevel],
        above_clusters: Sequence[StopCluster],
    ) -> float:
        return self._calculate_sweep_risk(
            current_price=current_price,
            levels=above_levels,
            clusters=above_clusters,
            find_nearest=self._find_nearest_above,
        )

    def _calculate_sweep_risk_down(
        self,
        current_price: float,
        below_levels: Sequence[LiquidityLevel],
        below_clusters: Sequence[StopCluster],
    ) -> float:
        return self._calculate_sweep_risk(
            current_price=current_price,
            levels=below_levels,
            clusters=below_clusters,
            find_nearest=self._find_nearest_below,
        )

    def _calculate_sweep_risk(
        self,
        current_price: float,
        levels: Sequence[LiquidityLevel],
        clusters: Sequence[StopCluster],
        find_nearest: Any,
    ) -> float:
        if not levels and not clusters:
            return 0.0

        nearest_component = 0.0
        nearest = find_nearest(current_price, levels, clusters)

        if nearest is not None:
            nearest_price = self._reference_price(nearest)
            nearest_component = 1.0 - clamp(
                pct_distance(nearest_price, current_price) / 0.015,
                0.0,
                1.0,
            )

        cluster_component = max(
            [cluster.confidence for cluster in clusters if cluster.is_active()],
            default=0.0,
        )

        level_component = max(
            [level.confidence for level in levels if level.is_active()],
            default=0.0,
        )

        return clamp(
            0.35 * nearest_component
            + 0.35 * cluster_component
            + 0.30 * level_component,
            0.0,
            1.0,
        )

    def _calculate_magnet_score(
        self,
        current_price: float,
        levels: Sequence[LiquidityLevel],
        clusters: Sequence[StopCluster],
    ) -> float:
        if not levels and not clusters:
            return 0.0

        components: list[float] = []

        for level in levels:
            if not level.is_active():
                continue

            distance_score = 1.0 - clamp(
                pct_distance(level.price, current_price) / 0.025,
                0.0,
                1.0,
            )

            components.append(
                clamp(
                    0.65 * level.confidence + 0.35 * distance_score,
                    0.0,
                    1.0,
                )
            )

        for cluster in clusters:
            if not cluster.is_active():
                continue

            distance_score = 1.0 - clamp(
                pct_distance(cluster.center_price, current_price) / 0.025,
                0.0,
                1.0,
            )

            density_score = clamp(
                cluster.estimated_stop_density,
                0.0,
                1.0,
            )

            components.append(
                clamp(
                    0.50 * cluster.confidence
                    + 0.25 * density_score
                    + 0.25 * distance_score,
                    0.0,
                    1.0,
                )
            )

        top_components = sorted(components, reverse=True)[:3]
        return clamp(safe_mean(top_components), 0.0, 1.0)

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
        Signed aggregate pressure score.

        Positive value means upside / buy-side liquidity pull is stronger.
        Negative value means downside / sell-side liquidity pull is stronger.
        """
        upward = (
            0.45 * above_liquidity_score
            + 0.30 * magnet_score_up
            + 0.25 * sweep_risk_up
        )

        downward = (
            0.45 * below_liquidity_score
            + 0.30 * magnet_score_down
            + 0.25 * sweep_risk_down
        )

        return clamp(upward - downward, -1.0, 1.0)

    def _infer_bias_from_pressure(
        self,
        pressure_score: float,
        above_liquidity_score: float,
        below_liquidity_score: float,
    ) -> LiquidityBias:
        """
        Infer bias from signed pressure first, with score delta as fallback.
        """
        if pressure_score >= 0.08:
            return LiquidityBias.UP

        if pressure_score <= -0.08:
            return LiquidityBias.DOWN

        return self._scorer.infer_bias_from_scores(
            above_liquidity_score=above_liquidity_score,
            below_liquidity_score=below_liquidity_score,
        )

    # ------------------------------------------------------------------
    # Signal
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        key: LiquidityKey,
        timestamp: datetime,
        features: LiquidityMapFeatures,
    ) -> LiquiditySignal:
        scope = liquidity_key_to_dict(key)

        confidence = self._calculate_signal_confidence(features)
        explanation = self._build_signal_explanation(features)

        return LiquiditySignal(
            exchange=scope["exchange"],
            market_type=scope["market_type"],
            symbol=scope["symbol"],
            timeframe=scope["timeframe"],
            timestamp=timestamp,
            bias=features.bias,
            nearest_buy_side_liquidity=features.nearest_above_level,
            nearest_sell_side_liquidity=features.nearest_below_level,
            sweep_risk_up=features.sweep_risk_up,
            sweep_risk_down=features.sweep_risk_down,
            magnet_score_up=features.magnet_score_up,
            magnet_score_down=features.magnet_score_down,
            confidence=confidence,
            explanation=explanation,
            metadata={
                "scope": scope,
                "scope_key": liquidity_key_to_string(key),
                "exchange": scope["exchange"],
                "market_type": scope["market_type"],
                "above_liquidity_score": features.above_liquidity_score,
                "below_liquidity_score": features.below_liquidity_score,
                "pressure_score": features.pressure_score,
                "pressure_score_semantics": (
                    "positive=upside_buy_side, negative=downside_sell_side"
                ),
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
                "above_levels_count": len(features.above_levels),
                "below_levels_count": len(features.below_levels),
                "above_clusters_count": len(features.above_clusters),
                "below_clusters_count": len(features.below_clusters),
            },
        )

    def _calculate_signal_confidence(
        self,
        features: LiquidityMapFeatures,
    ) -> float:
        directional_strength = abs(features.pressure_score)

        side_score_delta = abs(
            features.above_liquidity_score - features.below_liquidity_score
        )

        sweep_strength = max(
            features.sweep_risk_up,
            features.sweep_risk_down,
        )

        magnet_strength = max(
            features.magnet_score_up,
            features.magnet_score_down,
        )

        return clamp(
            0.35 * directional_strength
            + 0.25 * side_score_delta
            + 0.20 * sweep_strength
            + 0.20 * magnet_strength,
            0.0,
            1.0,
        )

    def _build_signal_explanation(
        self,
        features: LiquidityMapFeatures,
    ) -> str:
        parts: list[str] = []

        if features.bias == LiquidityBias.UP:
            parts.append("buy-side liquidity above price looks stronger")
        elif features.bias == LiquidityBias.DOWN:
            parts.append("sell-side liquidity below price looks stronger")
        else:
            parts.append("liquidity is relatively balanced")

        if features.pressure_score > 0:
            parts.append(f"signed pressure is upside: {features.pressure_score:.3f}")
        elif features.pressure_score < 0:
            parts.append(f"signed pressure is downside: {features.pressure_score:.3f}")

        if features.sweep_risk_up > 0.65:
            parts.append("elevated upside sweep risk")

        if features.sweep_risk_down > 0.65:
            parts.append("elevated downside sweep risk")

        if features.magnet_score_up > 0.70:
            parts.append("strong upside magnet")

        if features.magnet_score_down > 0.70:
            parts.append("strong downside magnet")

        if (
            features.strongest_cluster_above is not None
            and features.strongest_cluster_above.confidence > 0.75
        ):
            parts.append("high-confidence stop cluster above")

        if (
            features.strongest_cluster_below is not None
            and features.strongest_cluster_below.confidence > 0.75
        ):
            parts.append("high-confidence stop cluster below")

        return "; ".join(parts)

    # ------------------------------------------------------------------
    # Scope helpers
    # ------------------------------------------------------------------

    def _assert_and_scope_levels(
        self,
        *,
        levels: Sequence[LiquidityLevel],
        key: LiquidityKey,
    ) -> None:
        scope = liquidity_key_to_dict(key)

        for level in levels:
            level.exchange = scope["exchange"]
            level.market_type = scope["market_type"]
            level.symbol = scope["symbol"]
            level.timeframe = scope["timeframe"]
            level.metadata.setdefault("scope", scope)
            level.metadata.setdefault("scope_key", liquidity_key_to_string(key))

            if level.liquidity_key != key:
                raise ValueError(
                    "LiquidityLevel scope mismatch after scoping: "
                    f"expected={scope}, got={level.scope}"
                )

    def _assert_and_scope_clusters(
        self,
        *,
        clusters: Sequence[StopCluster],
        key: LiquidityKey,
    ) -> None:
        scope = liquidity_key_to_dict(key)

        for cluster in clusters:
            cluster.exchange = scope["exchange"]
            cluster.market_type = scope["market_type"]
            cluster.symbol = scope["symbol"]
            cluster.timeframe = scope["timeframe"]
            cluster.metadata.setdefault("scope", scope)
            cluster.metadata.setdefault("scope_key", liquidity_key_to_string(key))

            for source_level in cluster.source_levels:
                source_level.exchange = scope["exchange"]
                source_level.market_type = scope["market_type"]
                source_level.symbol = scope["symbol"]
                source_level.timeframe = scope["timeframe"]
                source_level.metadata.setdefault("scope", scope)
                source_level.metadata.setdefault("scope_key", liquidity_key_to_string(key))

            if cluster.liquidity_key != key:
                raise ValueError(
                    "StopCluster scope mismatch after scoping: "
                    f"expected={scope}, got={cluster.scope}"
                )

    @staticmethod
    def _make_key(
        *,
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
        symbol: str,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> LiquidityKey:
        return make_liquidity_key(
            exchange=normalize_exchange(exchange),
            market_type=normalize_market_type(market_type),
            symbol=normalize_symbol(symbol),
            timeframe=normalize_timeframe(timeframe),
        )

    # ------------------------------------------------------------------
    # Timestamp / parsing helpers
    # ------------------------------------------------------------------

    def _resolve_snapshot_timestamp(
        self,
        candles: Sequence[Any],
    ) -> datetime:
        if candles:
            value = get_first_value(
                candles[-1],
                (
                    "close_time_ms",
                    "open_time_ms",
                    "timestamp_ms",
                    "received_at_ms",
                    "close_time",
                    "timestamp",
                    "time",
                    "open_time",
                    "closeTime",
                    "openTime",
                ),
            )

            parsed = self._parse_datetime(value)
            if parsed is not None:
                return parsed

        return utc_now()

    def _parse_datetime(
        self,
        value: Any,
    ) -> datetime | None:
        if value is None:
            return None

        if isinstance(value, datetime):
            return self._normalize_timestamp(value)

        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(
                    value / 1000 if value > 1e12 else value,
                    tz=utc_now().tzinfo,
                )
            except (OSError, OverflowError, ValueError):
                return None

        if isinstance(value, str):
            try:
                normalized = value.replace("Z", "+00:00")
                return self._normalize_timestamp(
                    datetime.fromisoformat(normalized)
                )
            except ValueError:
                return None

        return None

    @staticmethod
    def _normalize_timestamp(
        value: datetime | None,
    ) -> datetime | None:
        if value is None:
            return None
        return ensure_utc(value)

    def _min_datetime(
        self,
        left: datetime | None,
        right: datetime | None,
    ) -> datetime | None:
        left = self._normalize_timestamp(left)
        right = self._normalize_timestamp(right)

        if left is None:
            return right

        if right is None:
            return left

        return min(left, right)

    def _max_datetime(
        self,
        left: datetime | None,
        right: datetime | None,
    ) -> datetime | None:
        left = self._normalize_timestamp(left)
        right = self._normalize_timestamp(right)

        if left is None:
            return right

        if right is None:
            return left

        return max(left, right)

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reference_price(
        item: LiquidityLevel | StopCluster,
    ) -> float:
        if isinstance(item, StopCluster):
            return item.center_price

        return item.price

    @staticmethod
    def _validate_symbol_timeframe(
        *,
        symbol: str,
        timeframe: str,
    ) -> None:
        if not symbol:
            raise ValueError("symbol is required")

        if not timeframe:
            raise ValueError("timeframe is required")


__all__ = [
    "LiquidityMapFeatures",
    "LiquidityMap",
]