# tests/analytics/liquidity/test_liquidity_detectors.py

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from analytics.liquidity.enums import (
    LiquidityLevelType,
    LiquiditySide,
    LiquidityStatus,
    SweepStatus,
)
from analytics.liquidity.equal_highs_lows import EqualHighsLowsDetector
from analytics.liquidity.models import EqualLevel, LiquidityLevel, StopCluster
from analytics.liquidity.scoring import LiquidityScorer
from analytics.liquidity.stop_clusters import StopClustersDetector


# ---------------------------------------------------------------------
# Local assertions / helpers
# ---------------------------------------------------------------------


def _assert_score01(value: float) -> None:
    assert 0.0 <= value <= 1.0


def _levels_by_type(
    levels: list[LiquidityLevel],
    level_type: LiquidityLevelType,
) -> list[LiquidityLevel]:
    return [level for level in levels if level.level_type == level_type]


def _levels_by_side(
    levels: list[LiquidityLevel],
    side: LiquiditySide,
) -> list[LiquidityLevel]:
    return [level for level in levels if level.side == side]


def _clusters_by_side(
    clusters: list[StopCluster],
    side: LiquiditySide,
) -> list[StopCluster]:
    return [cluster for cluster in clusters if cluster.side == side]


def _assert_cluster_has_valid_range(cluster: StopCluster) -> None:
    assert cluster.low_price > 0
    assert cluster.high_price > 0
    assert cluster.low_price <= cluster.high_price
    assert cluster.center_price >= cluster.low_price
    assert cluster.center_price <= cluster.high_price


def _assert_level_is_payload_safe(level: LiquidityLevel) -> None:
    payload = level.to_event_payload()

    assert payload["symbol"] == level.symbol
    assert payload["timeframe"] == level.timeframe
    assert payload["level_type"] == level.level_type.value
    assert payload["side"] == level.side.value
    assert payload["price"] == level.price
    assert payload["status"] == level.status.value
    assert payload["sweep_status"] == level.sweep_status.value
    assert isinstance(payload["metadata"], dict)


# ---------------------------------------------------------------------
# Equal highs / lows detector
# ---------------------------------------------------------------------


class TestEqualHighsLowsDetector:
    def test_detects_equal_highs_from_repeated_pivot_highs(
        self,
        equal_detector: EqualHighsLowsDetector,
        candles_with_equal_highs: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = equal_detector.detect(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_with_equal_highs,
            current_price=100.0,
        )

        equal_highs = _levels_by_type(levels, LiquidityLevelType.EQUAL_HIGHS)

        assert equal_highs
        assert all(level.side == LiquiditySide.BUY_SIDE for level in equal_highs)

        strongest = max(equal_highs, key=lambda level: level.confidence)

        assert strongest.price == pytest.approx(105.0, rel=0.002)
        assert strongest.touches_count >= 2
        assert strongest.is_active()
        _assert_score01(strongest.confidence)
        _assert_level_is_payload_safe(strongest)

    def test_detects_equal_lows_from_repeated_pivot_lows(
        self,
        equal_detector: EqualHighsLowsDetector,
        candles_with_equal_lows: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = equal_detector.detect(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_with_equal_lows,
            current_price=100.0,
        )

        equal_lows = _levels_by_type(levels, LiquidityLevelType.EQUAL_LOWS)

        assert equal_lows
        assert all(level.side == LiquiditySide.SELL_SIDE for level in equal_lows)

        strongest = max(equal_lows, key=lambda level: level.confidence)

        assert strongest.price == pytest.approx(95.0, rel=0.002)
        assert strongest.touches_count >= 2
        assert strongest.is_active()
        _assert_score01(strongest.confidence)
        _assert_level_is_payload_safe(strongest)

    def test_detects_both_equal_highs_and_equal_lows_from_two_sided_structure(
        self,
        equal_detector: EqualHighsLowsDetector,
        candles_with_both_sides: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = equal_detector.detect(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_with_both_sides,
            current_price=100.0,
        )

        equal_highs = _levels_by_type(levels, LiquidityLevelType.EQUAL_HIGHS)
        equal_lows = _levels_by_type(levels, LiquidityLevelType.EQUAL_LOWS)

        assert equal_highs
        assert equal_lows

        assert any(level.price == pytest.approx(105.0, rel=0.002) for level in equal_highs)
        assert any(level.price == pytest.approx(95.0, rel=0.002) for level in equal_lows)

    def test_returns_empty_when_not_enough_candles(
        self,
        equal_detector: EqualHighsLowsDetector,
        too_few_candles: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = equal_detector.detect(
            symbol=symbol,
            timeframe=timeframe,
            candles=too_few_candles,
            current_price=100.0,
        )

        assert levels == []

    def test_returns_empty_when_no_clear_equal_levels_exist(
        self,
        equal_detector: EqualHighsLowsDetector,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = equal_detector.detect(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_without_clear_equal_levels,
            current_price=105.0,
        )

        assert levels == []

    def test_rejects_missing_symbol(
        self,
        equal_detector: EqualHighsLowsDetector,
        candles_with_equal_highs: list[dict[str, Any]],
        timeframe: str,
    ) -> None:
        with pytest.raises(ValueError, match="symbol and timeframe"):
            equal_detector.detect(
                symbol="",
                timeframe=timeframe,
                candles=candles_with_equal_highs,
                current_price=100.0,
            )

    def test_rejects_missing_timeframe(
        self,
        equal_detector: EqualHighsLowsDetector,
        candles_with_equal_highs: list[dict[str, Any]],
        symbol: str,
    ) -> None:
        with pytest.raises(ValueError, match="symbol and timeframe"):
            equal_detector.detect(
                symbol=symbol,
                timeframe="",
                candles=candles_with_equal_highs,
                current_price=100.0,
            )

    def test_detector_respects_disabled_config(
        self,
        disabled_liquidity_config,
        scorer: LiquidityScorer,
        candles_with_equal_highs: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        detector = EqualHighsLowsDetector(
            config=disabled_liquidity_config,
            scorer=scorer,
        )

        levels = detector.detect(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_with_equal_highs,
            current_price=100.0,
        )

        assert levels == []

    def test_min_equal_touches_filter_removes_under_touched_levels(
        self,
        liquidity_config,
        scorer: LiquidityScorer,
        candles_with_equal_highs: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        liquidity_config.min_equal_touches = 4

        detector = EqualHighsLowsDetector(
            config=liquidity_config,
            scorer=scorer,
        )

        levels = detector.detect(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_with_equal_highs,
            current_price=100.0,
        )

        assert _levels_by_type(levels, LiquidityLevelType.EQUAL_HIGHS) == []

    def test_equal_high_is_marked_swept_when_current_price_breaks_above(
        self,
        equal_detector: EqualHighsLowsDetector,
        candles_with_equal_highs: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = equal_detector.detect(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_with_equal_highs,
            current_price=106.0,
        )

        equal_highs = _levels_by_type(levels, LiquidityLevelType.EQUAL_HIGHS)

        assert equal_highs
        assert any(level.is_swept() for level in equal_highs)
        assert any(level.swept_at is not None for level in equal_highs)

    def test_equal_low_is_marked_swept_when_current_price_breaks_below(
        self,
        equal_detector: EqualHighsLowsDetector,
        candles_with_equal_lows: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = equal_detector.detect(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_with_equal_lows,
            current_price=94.0,
        )

        equal_lows = _levels_by_type(levels, LiquidityLevelType.EQUAL_LOWS)

        assert equal_lows
        assert any(level.is_swept() for level in equal_lows)
        assert any(level.swept_at is not None for level in equal_lows)

    def test_detected_levels_are_sorted_by_price_then_type(
        self,
        equal_detector: EqualHighsLowsDetector,
        candles_with_both_sides: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = equal_detector.detect(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_with_both_sides,
            current_price=100.0,
        )

        sort_keys = [(level.price, level.level_type.value) for level in levels]

        assert sort_keys == sorted(sort_keys)

    def test_all_detected_confidences_are_clamped(
        self,
        equal_detector: EqualHighsLowsDetector,
        candles_with_both_sides: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = equal_detector.detect(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_with_both_sides,
            current_price=100.0,
        )

        assert levels
        for level in levels:
            _assert_score01(level.confidence)


# ---------------------------------------------------------------------
# Stop clusters detector
# ---------------------------------------------------------------------


class TestStopClustersDetector:
    def test_builds_stop_cluster_from_close_buy_side_levels(
        self,
        stop_detector: StopClustersDetector,
        buy_side_levels: list[LiquidityLevel],
        candles_with_equal_highs: list[dict[str, Any]],
        orderbook_near_buy_side_cluster: dict[str, list[list[float]]],
        symbol: str,
        timeframe: str,
    ) -> None:
        clusters = stop_detector.detect_from_levels(
            symbol=symbol,
            timeframe=timeframe,
            levels=buy_side_levels,
            current_price=100.0,
            candles=candles_with_equal_highs,
            orderbook=orderbook_near_buy_side_cluster,
        )

        buy_clusters = _clusters_by_side(clusters, LiquiditySide.BUY_SIDE)

        assert buy_clusters
        assert len(buy_clusters) == 1

        cluster = buy_clusters[0]

        _assert_cluster_has_valid_range(cluster)
        assert cluster.center_price == pytest.approx(105.0, rel=0.003)
        assert cluster.low_price < 105.0
        assert cluster.high_price > 105.0
        assert cluster.source_levels
        assert len(cluster.source_levels) == len(buy_side_levels)
        _assert_score01(cluster.confidence)
        _assert_score01(cluster.estimated_stop_density)

    def test_builds_stop_cluster_from_close_sell_side_levels(
        self,
        stop_detector: StopClustersDetector,
        sell_side_levels: list[LiquidityLevel],
        candles_with_equal_lows: list[dict[str, Any]],
        orderbook_near_sell_side_cluster: dict[str, list[list[float]]],
        symbol: str,
        timeframe: str,
    ) -> None:
        clusters = stop_detector.detect_from_levels(
            symbol=symbol,
            timeframe=timeframe,
            levels=sell_side_levels,
            current_price=100.0,
            candles=candles_with_equal_lows,
            orderbook=orderbook_near_sell_side_cluster,
        )

        sell_clusters = _clusters_by_side(clusters, LiquiditySide.SELL_SIDE)

        assert sell_clusters
        assert len(sell_clusters) == 1

        cluster = sell_clusters[0]

        _assert_cluster_has_valid_range(cluster)
        assert cluster.center_price == pytest.approx(95.0, rel=0.003)
        assert cluster.low_price < 95.0
        assert cluster.high_price > 95.0
        assert cluster.source_levels
        assert len(cluster.source_levels) == len(sell_side_levels)
        _assert_score01(cluster.confidence)
        _assert_score01(cluster.estimated_stop_density)

    def test_merges_overlapping_candidates_into_single_cluster(
        self,
        stop_detector: StopClustersDetector,
        buy_side_levels: list[LiquidityLevel],
        candles_with_equal_highs: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        clusters = stop_detector.detect_from_levels(
            symbol=symbol,
            timeframe=timeframe,
            levels=buy_side_levels,
            current_price=100.0,
            candles=candles_with_equal_highs,
            orderbook=None,
        )

        assert len(_clusters_by_side(clusters, LiquiditySide.BUY_SIDE)) == 1

    def test_keeps_buy_and_sell_side_clusters_separate(
        self,
        stop_detector: StopClustersDetector,
        mixed_side_levels: list[LiquidityLevel],
        candles_with_both_sides: list[dict[str, Any]],
        balanced_orderbook: dict[str, list[list[float]]],
        symbol: str,
        timeframe: str,
    ) -> None:
        clusters = stop_detector.detect_from_levels(
            symbol=symbol,
            timeframe=timeframe,
            levels=mixed_side_levels,
            current_price=100.0,
            candles=candles_with_both_sides,
            orderbook=balanced_orderbook,
        )

        buy_clusters = _clusters_by_side(clusters, LiquiditySide.BUY_SIDE)
        sell_clusters = _clusters_by_side(clusters, LiquiditySide.SELL_SIDE)

        assert buy_clusters
        assert sell_clusters
        assert all(cluster.center_price > 100.0 for cluster in buy_clusters)
        assert all(cluster.center_price < 100.0 for cluster in sell_clusters)

    def test_returns_empty_for_empty_levels(
        self,
        stop_detector: StopClustersDetector,
        symbol: str,
        timeframe: str,
    ) -> None:
        clusters = stop_detector.detect_from_levels(
            symbol=symbol,
            timeframe=timeframe,
            levels=[],
            current_price=100.0,
            candles=[],
            orderbook=None,
        )

        assert clusters == []

    def test_rejects_invalid_current_price(
        self,
        stop_detector: StopClustersDetector,
        buy_side_levels: list[LiquidityLevel],
        symbol: str,
        timeframe: str,
    ) -> None:
        with pytest.raises(ValueError, match="current_price"):
            stop_detector.detect_from_levels(
                symbol=symbol,
                timeframe=timeframe,
                levels=buy_side_levels,
                current_price=0.0,
                candles=[],
                orderbook=None,
            )

    def test_rejects_missing_symbol(
        self,
        stop_detector: StopClustersDetector,
        buy_side_levels: list[LiquidityLevel],
        timeframe: str,
    ) -> None:
        with pytest.raises(ValueError, match="symbol and timeframe"):
            stop_detector.detect_from_levels(
                symbol="",
                timeframe=timeframe,
                levels=buy_side_levels,
                current_price=100.0,
                candles=[],
                orderbook=None,
            )

    def test_rejects_missing_timeframe(
        self,
        stop_detector: StopClustersDetector,
        buy_side_levels: list[LiquidityLevel],
        symbol: str,
    ) -> None:
        with pytest.raises(ValueError, match="symbol and timeframe"):
            stop_detector.detect_from_levels(
                symbol=symbol,
                timeframe="",
                levels=buy_side_levels,
                current_price=100.0,
                candles=[],
                orderbook=None,
            )

    def test_ignores_invalidated_and_expired_levels(
        self,
        stop_detector: StopClustersDetector,
        invalidated_buy_side_level: LiquidityLevel,
        expired_sell_side_level: LiquidityLevel,
        symbol: str,
        timeframe: str,
    ) -> None:
        clusters = stop_detector.detect_from_levels(
            symbol=symbol,
            timeframe=timeframe,
            levels=[invalidated_buy_side_level, expired_sell_side_level],
            current_price=100.0,
            candles=[],
            orderbook=None,
        )

        assert clusters == []

    def test_partially_swept_source_levels_are_preserved_in_cluster_context(
        self,
        stop_detector: StopClustersDetector,
        buy_side_levels: list[LiquidityLevel],
        partially_swept_buy_side_level: LiquidityLevel,
        candles_with_equal_highs: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = [
            partially_swept_buy_side_level,
            *buy_side_levels,
        ]

        clusters = stop_detector.detect_from_levels(
            symbol=symbol,
            timeframe=timeframe,
            levels=levels,
            current_price=100.0,
            candles=candles_with_equal_highs,
            orderbook=None,
        )

        assert clusters

        cluster = _clusters_by_side(clusters, LiquiditySide.BUY_SIDE)[0]

        assert any(level.is_partially_swept() for level in cluster.source_levels)
        assert cluster.swept_at is not None or any(
            level.swept_at is not None for level in cluster.source_levels
        )
        _assert_score01(cluster.confidence)
        _assert_score01(cluster.estimated_stop_density)

    def test_swept_source_levels_are_preserved_for_reversal_context(
        self,
        stop_detector: StopClustersDetector,
        buy_side_levels: list[LiquidityLevel],
        swept_buy_side_level: LiquidityLevel,
        candles_with_equal_highs: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = [
            swept_buy_side_level,
            *buy_side_levels,
        ]

        clusters = stop_detector.detect_from_levels(
            symbol=symbol,
            timeframe=timeframe,
            levels=levels,
            current_price=100.0,
            candles=candles_with_equal_highs,
            orderbook=None,
        )

        assert clusters

        cluster = _clusters_by_side(clusters, LiquiditySide.BUY_SIDE)[0]

        assert any(level.is_swept() for level in cluster.source_levels)
        assert cluster.swept_at is not None or any(
            level.swept_at is not None for level in cluster.source_levels
        )
        _assert_score01(cluster.confidence)
        _assert_score01(cluster.estimated_stop_density)

    def test_orderbook_near_cluster_increases_or_preserves_density(
        self,
        stop_detector: StopClustersDetector,
        buy_side_levels: list[LiquidityLevel],
        candles_with_equal_highs: list[dict[str, Any]],
        orderbook_near_buy_side_cluster: dict[str, list[list[float]]],
        symbol: str,
        timeframe: str,
    ) -> None:
        without_orderbook = stop_detector.detect_from_levels(
            symbol=symbol,
            timeframe=timeframe,
            levels=buy_side_levels,
            current_price=100.0,
            candles=candles_with_equal_highs,
            orderbook=None,
        )

        with_orderbook = stop_detector.detect_from_levels(
            symbol=symbol,
            timeframe=timeframe,
            levels=buy_side_levels,
            current_price=100.0,
            candles=candles_with_equal_highs,
            orderbook=orderbook_near_buy_side_cluster,
        )

        assert without_orderbook
        assert with_orderbook

        base_cluster = _clusters_by_side(without_orderbook, LiquiditySide.BUY_SIDE)[0]
        enhanced_cluster = _clusters_by_side(with_orderbook, LiquiditySide.BUY_SIDE)[0]

        assert enhanced_cluster.estimated_stop_density >= base_cluster.estimated_stop_density
        assert enhanced_cluster.confidence >= base_cluster.confidence

    def test_build_stop_zones_returns_merged_price_ranges(
        self,
        stop_detector: StopClustersDetector,
        buy_side_levels: list[LiquidityLevel],
    ) -> None:
        zones = stop_detector.build_stop_zones(buy_side_levels)

        assert zones
        assert len(zones) == 1

        low, high = zones[0]

        assert low < 105.0
        assert high > 105.0
        assert low <= high

    def test_build_stop_zones_ignores_invalidated_and_expired_levels(
        self,
        stop_detector: StopClustersDetector,
        invalidated_buy_side_level: LiquidityLevel,
        expired_sell_side_level: LiquidityLevel,
    ) -> None:
        zones = stop_detector.build_stop_zones(
            [invalidated_buy_side_level, expired_sell_side_level]
        )

        assert zones == []

    def test_detect_from_equal_levels_wrapper_matches_detect_from_levels(
        self,
        stop_detector: StopClustersDetector,
        equal_high_level: EqualLevel,
        candles_with_equal_highs: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = [
            deepcopy(equal_high_level),
            deepcopy(equal_high_level),
            deepcopy(equal_high_level),
        ]

        levels[0].price = 104.95
        levels[1].price = 105.00
        levels[2].price = 105.05

        from_levels = stop_detector.detect_from_levels(
            symbol=symbol,
            timeframe=timeframe,
            levels=levels,
            current_price=100.0,
            candles=candles_with_equal_highs,
            orderbook=None,
        )

        from_equal_levels = stop_detector.detect_from_equal_levels(
            symbol=symbol,
            timeframe=timeframe,
            equal_levels=levels,
            current_price=100.0,
            candles=candles_with_equal_highs,
            orderbook=None,
        )

        assert len(from_equal_levels) == len(from_levels)
        assert [cluster.center_price for cluster in from_equal_levels] == pytest.approx(
            [cluster.center_price for cluster in from_levels],
            rel=0.001,
        )

    def test_all_detected_cluster_scores_are_clamped(
        self,
        stop_detector: StopClustersDetector,
        mixed_side_levels: list[LiquidityLevel],
        candles_with_both_sides: list[dict[str, Any]],
        balanced_orderbook: dict[str, list[list[float]]],
        symbol: str,
        timeframe: str,
    ) -> None:
        clusters = stop_detector.detect_from_levels(
            symbol=symbol,
            timeframe=timeframe,
            levels=mixed_side_levels,
            current_price=100.0,
            candles=candles_with_both_sides,
            orderbook=balanced_orderbook,
        )

        assert clusters

        for cluster in clusters:
            _assert_score01(cluster.confidence)
            _assert_score01(cluster.estimated_stop_density)


# ---------------------------------------------------------------------
# Liquidity scorer
# ---------------------------------------------------------------------


class TestLiquidityScorer:
    def test_equal_level_score_is_clamped(
        self,
        scorer: LiquidityScorer,
        equal_high_level: EqualLevel,
    ) -> None:
        score = scorer.score_equal_level(
            level=equal_high_level,
            current_price=100.0,
        )

        _assert_score01(score)

    def test_equal_level_score_penalizes_swept_level(
        self,
        scorer: LiquidityScorer,
        equal_high_level: EqualLevel,
    ) -> None:
        active_level = deepcopy(equal_high_level)
        swept_level = deepcopy(equal_high_level)

        active_level.sweep_status = SweepStatus.NOT_SWEPT
        active_level.status = LiquidityStatus.ACTIVE
        active_level.swept_at = None

        swept_level.mark_swept()

        active_score = scorer.score_equal_level(
            level=active_level,
            current_price=100.0,
        )
        swept_score = scorer.score_equal_level(
            level=swept_level,
            current_price=100.0,
        )

        assert swept_score < active_score
        _assert_score01(active_score)
        _assert_score01(swept_score)

    def test_equal_level_score_penalizes_partially_swept_less_than_fully_swept(
        self,
        scorer: LiquidityScorer,
        equal_high_level: EqualLevel,
    ) -> None:
        active_level = deepcopy(equal_high_level)
        partially_swept_level = deepcopy(equal_high_level)
        swept_level = deepcopy(equal_high_level)

        active_level.sweep_status = SweepStatus.NOT_SWEPT
        active_level.status = LiquidityStatus.ACTIVE
        active_level.swept_at = None

        partially_swept_level.mark_partially_swept()
        swept_level.mark_swept()

        active_score = scorer.score_equal_level(
            level=active_level,
            current_price=100.0,
        )
        partially_swept_score = scorer.score_equal_level(
            level=partially_swept_level,
            current_price=100.0,
        )
        swept_score = scorer.score_equal_level(
            level=swept_level,
            current_price=100.0,
        )

        assert active_score > partially_swept_score > swept_score

        _assert_score01(active_score)
        _assert_score01(partially_swept_score)
        _assert_score01(swept_score)

    def test_equal_level_score_increases_with_more_touches_and_reactions(
        self,
        scorer: LiquidityScorer,
        equal_high_level: EqualLevel,
    ) -> None:
        weak_level = deepcopy(equal_high_level)
        strong_level = deepcopy(equal_high_level)

        weak_level.touches_count = 2
        weak_level.reaction_count = 0

        strong_level.touches_count = 6
        strong_level.reaction_count = 4

        weak_score = scorer.score_equal_level(
            level=weak_level,
            current_price=100.0,
        )
        strong_score = scorer.score_equal_level(
            level=strong_level,
            current_price=100.0,
        )

        assert strong_score >= weak_score
        _assert_score01(weak_score)
        _assert_score01(strong_score)

    def test_equal_level_score_prefers_more_compact_price_cluster(
        self,
        scorer: LiquidityScorer,
        equal_high_level: EqualLevel,
    ) -> None:
        compact_level = deepcopy(equal_high_level)
        wide_level = deepcopy(equal_high_level)

        compact_level.cluster_low = 104.98
        compact_level.cluster_high = 105.02

        wide_level.cluster_low = 104.50
        wide_level.cluster_high = 105.50

        compact_score = scorer.score_equal_level(
            level=compact_level,
            current_price=100.0,
        )
        wide_score = scorer.score_equal_level(
            level=wide_level,
            current_price=100.0,
        )

        assert compact_score >= wide_score
        _assert_score01(compact_score)
        _assert_score01(wide_score)

    def test_stop_cluster_score_is_clamped(
        self,
        scorer: LiquidityScorer,
        buy_side_stop_cluster: StopCluster,
    ) -> None:
        score = scorer.score_stop_cluster(
            cluster=buy_side_stop_cluster,
            current_price=100.0,
        )

        _assert_score01(score)

    def test_stop_cluster_score_prefers_dense_cluster(
        self,
        scorer: LiquidityScorer,
        buy_side_stop_cluster: StopCluster,
    ) -> None:
        weak_cluster = deepcopy(buy_side_stop_cluster)
        strong_cluster = deepcopy(buy_side_stop_cluster)

        weak_cluster.estimated_stop_density = 0.20
        weak_cluster.confidence = 0.20
        weak_cluster.touches_count = 1

        strong_cluster.estimated_stop_density = 0.90
        strong_cluster.confidence = 0.90
        strong_cluster.touches_count = 6

        weak_score = scorer.score_stop_cluster(
            cluster=weak_cluster,
            current_price=100.0,
        )
        strong_score = scorer.score_stop_cluster(
            cluster=strong_cluster,
            current_price=100.0,
        )

        assert strong_score >= weak_score
        _assert_score01(weak_score)
        _assert_score01(strong_score)

    def test_stop_cluster_score_prefers_compact_cluster(
        self,
        scorer: LiquidityScorer,
        buy_side_stop_cluster: StopCluster,
    ) -> None:
        compact_cluster = deepcopy(buy_side_stop_cluster)
        wide_cluster = deepcopy(buy_side_stop_cluster)

        compact_cluster.low_price = 104.98
        compact_cluster.high_price = 105.02
        compact_cluster.center_price = 105.00

        wide_cluster.low_price = 103.50
        wide_cluster.high_price = 106.50
        wide_cluster.center_price = 105.00

        compact_score = scorer.score_stop_cluster(
            cluster=compact_cluster,
            current_price=100.0,
        )
        wide_score = scorer.score_stop_cluster(
            cluster=wide_cluster,
            current_price=100.0,
        )

        assert compact_score >= wide_score
        _assert_score01(compact_score)
        _assert_score01(wide_score)

    def test_stop_cluster_score_prefers_nearer_cluster_when_other_factors_equal(
        self,
        scorer: LiquidityScorer,
        buy_side_stop_cluster: StopCluster,
    ) -> None:
        near_cluster = deepcopy(buy_side_stop_cluster)
        far_cluster = deepcopy(buy_side_stop_cluster)

        near_cluster.low_price = 100.90
        near_cluster.high_price = 101.10
        near_cluster.center_price = 101.00

        far_cluster.low_price = 119.90
        far_cluster.high_price = 120.10
        far_cluster.center_price = 120.00

        near_score = scorer.score_stop_cluster(
            cluster=near_cluster,
            current_price=100.0,
        )
        far_score = scorer.score_stop_cluster(
            cluster=far_cluster,
            current_price=100.0,
        )

        assert near_score >= far_score
        _assert_score01(near_score)
        _assert_score01(far_score)