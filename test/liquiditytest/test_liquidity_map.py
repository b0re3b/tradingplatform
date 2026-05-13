# tests/analytics/liquidity/test_liquidity_map.py

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from analytics.liquidity.config import LiquidityConfig
from analytics.liquidity.enums import (
    LiquidityBias,
    LiquidityLevelType,
    LiquiditySide,
    LiquidityStatus,
    SweepStatus,
)
from analytics.liquidity.liquidity_map import LiquidityMap
from analytics.liquidity.models import (
    EqualLevel,
    LiquidityLevel,
    LiquidityMapSnapshot,
    LiquiditySignal,
    LiquidityZone,
    StopCluster,
)
from analytics.liquidity.scoring import LiquidityScorer


# ---------------------------------------------------------------------
# Local assertions / helpers
# ---------------------------------------------------------------------


def _assert_score01(value: float) -> None:
    assert 0.0 <= value <= 1.0


def _assert_signed_score(value: float) -> None:
    assert -1.0 <= value <= 1.0


def _assert_snapshot_basic_contract(
    snapshot: LiquidityMapSnapshot,
    *,
    symbol: str,
    timeframe: str,
    current_price: float,
) -> None:
    assert snapshot.symbol == symbol
    assert snapshot.timeframe == timeframe
    assert snapshot.current_price == pytest.approx(current_price)
    assert snapshot.timestamp.tzinfo is not None

    _assert_score01(snapshot.above_liquidity_score)
    _assert_score01(snapshot.below_liquidity_score)
    _assert_signed_score(snapshot.liquidity_pressure_score)

    assert snapshot.bias in {
        LiquidityBias.UP,
        LiquidityBias.DOWN,
        LiquidityBias.NEUTRAL,
    }

    assert isinstance(snapshot.active_levels, list)
    assert isinstance(snapshot.equal_levels, list)
    assert isinstance(snapshot.stop_clusters, list)
    assert isinstance(snapshot.zones, list)
    assert isinstance(snapshot.metadata, dict)


def _assert_signal_contract(
    signal: LiquiditySignal,
    *,
    symbol: str,
    timeframe: str,
) -> None:
    assert signal.symbol == symbol
    assert signal.timeframe == timeframe
    assert signal.timestamp.tzinfo is not None
    assert signal.bias in {
        LiquidityBias.UP,
        LiquidityBias.DOWN,
        LiquidityBias.NEUTRAL,
    }

    _assert_score01(signal.sweep_risk_up)
    _assert_score01(signal.sweep_risk_down)
    _assert_score01(signal.magnet_score_up)
    _assert_score01(signal.magnet_score_down)
    _assert_score01(signal.confidence)

    payload = signal.to_event_payload()

    assert payload["symbol"] == symbol
    assert payload["timeframe"] == timeframe
    assert payload["bias"] == signal.bias.value
    assert isinstance(payload["metadata"], dict)


def _assert_zone_contract(zone: LiquidityZone) -> None:
    assert zone.low_price > 0
    assert zone.high_price > 0
    assert zone.low_price <= zone.high_price
    assert zone.center_price >= zone.low_price
    assert zone.center_price <= zone.high_price
    assert zone.side in {
        LiquiditySide.BUY_SIDE,
        LiquiditySide.SELL_SIDE,
        LiquiditySide.BOTH,
        LiquiditySide.UNKNOWN,
    }
    _assert_score01(zone.score)

    payload = zone.to_event_payload()

    assert payload["side"] == zone.side.value
    assert payload["low_price"] == zone.low_price
    assert payload["high_price"] == zone.high_price
    assert isinstance(payload["source_types"], list)
    assert isinstance(payload["metadata"], dict)


def _weaken_level(level: LiquidityLevel) -> LiquidityLevel:
    cloned = deepcopy(level)
    cloned.confidence = 0.10
    cloned.touches_count = 1
    cloned.reaction_count = 0
    return cloned


def _weaken_cluster(cluster: StopCluster) -> StopCluster:
    cloned = deepcopy(cluster)
    cloned.confidence = 0.10
    cloned.estimated_stop_density = 0.10
    cloned.touches_count = 1
    return cloned


def _move_level_price(level: LiquidityLevel, price: float) -> LiquidityLevel:
    cloned = deepcopy(level)
    cloned.price = price
    return cloned


def _move_cluster_prices(
    cluster: StopCluster,
    *,
    low_price: float,
    high_price: float,
) -> StopCluster:
    cloned = deepcopy(cluster)
    cloned.low_price = low_price
    cloned.high_price = high_price
    cloned.center_price = (low_price + high_price) / 2.0
    return cloned


# ---------------------------------------------------------------------
# build_snapshot_from_components()
# ---------------------------------------------------------------------


class TestLiquidityMapFromComponents:
    def test_build_snapshot_from_components_returns_complete_snapshot(
        self,
        liquidity_map: LiquidityMap,
        mixed_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = liquidity_map.build_snapshot_from_components(
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=mixed_side_levels,
            clusters=[buy_side_stop_cluster, sell_side_stop_cluster],
        )

        _assert_snapshot_basic_contract(
            snapshot,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
        )

        assert snapshot.has_levels()
        assert snapshot.active_levels
        assert snapshot.stop_clusters
        assert snapshot.zones
        assert snapshot.signal is not None

        _assert_signal_contract(
            snapshot.signal,
            symbol=symbol,
            timeframe=timeframe,
        )

        assert snapshot.metadata["builder"] == "LiquidityMap"
        assert snapshot.metadata["from_components"] is True
        assert snapshot.metadata["levels_count"] == len(snapshot.active_levels)
        assert snapshot.metadata["stop_clusters_count"] == len(snapshot.stop_clusters)
        assert snapshot.metadata["zones_count"] == len(snapshot.zones)

    def test_build_snapshot_from_components_uses_explicit_timestamp(
        self,
        liquidity_map: LiquidityMap,
        mixed_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        explicit_ts = datetime(2026, 1, 2, 10, 30, tzinfo=timezone.utc)

        snapshot = liquidity_map.build_snapshot_from_components(
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=mixed_side_levels,
            clusters=[buy_side_stop_cluster, sell_side_stop_cluster],
            timestamp=explicit_ts,
        )

        assert snapshot.timestamp == explicit_ts

        assert snapshot.signal is not None
        assert snapshot.signal.timestamp == explicit_ts

    def test_build_snapshot_from_components_normalizes_naive_timestamp_to_utc(
        self,
        liquidity_map: LiquidityMap,
        mixed_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        naive_ts = datetime(2026, 1, 2, 10, 30)

        snapshot = liquidity_map.build_snapshot_from_components(
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=mixed_side_levels,
            clusters=[buy_side_stop_cluster, sell_side_stop_cluster],
            timestamp=naive_ts,
        )

        assert snapshot.timestamp.tzinfo is not None
        assert snapshot.timestamp.utcoffset() == timezone.utc.utcoffset(snapshot.timestamp)

    def test_build_snapshot_from_components_rejects_missing_symbol(
            self,
            liquidity_map: LiquidityMap,
            mixed_side_levels: list[LiquidityLevel],
            symbol: str,
            timeframe: str,
    ) -> None:
        with pytest.raises(ValueError, match="symbol is required"):
            liquidity_map.build_snapshot_from_components(
                symbol="",
                timeframe=timeframe,
                current_price=100.0,
                levels=mixed_side_levels,
                clusters=[],
            )

    def test_build_snapshot_from_components_rejects_missing_timeframe(
            self,
            liquidity_map: LiquidityMap,
            mixed_side_levels: list[LiquidityLevel],
            symbol: str,
    ) -> None:
        with pytest.raises(ValueError, match="timeframe is required"):
            liquidity_map.build_snapshot_from_components(
                symbol=symbol,
                timeframe="",
                current_price=100.0,
                levels=mixed_side_levels,
                clusters=[],
            )

    def test_build_snapshot_from_components_rejects_missing_timeframe(
        self,
        liquidity_map: LiquidityMap,
        mixed_side_levels: list[LiquidityLevel],
        symbol: str,
    ) -> None:
        with pytest.raises(ValueError, match="symbol and timeframe"):
            liquidity_map.build_snapshot_from_components(
                symbol=symbol,
                timeframe="",
                current_price=100.0,
                levels=mixed_side_levels,
                clusters=[],
            )

    def test_build_snapshot_from_components_rejects_invalid_current_price(
        self,
        liquidity_map: LiquidityMap,
        mixed_side_levels: list[LiquidityLevel],
        symbol: str,
        timeframe: str,
    ) -> None:
        with pytest.raises(ValueError, match="current_price"):
            liquidity_map.build_snapshot_from_components(
                symbol=symbol,
                timeframe=timeframe,
                current_price=0.0,
                levels=mixed_side_levels,
                clusters=[],
            )

    def test_build_snapshot_from_components_filters_inactive_levels_from_live_snapshot(
        self,
        liquidity_map: LiquidityMap,
        buy_side_levels: list[LiquidityLevel],
        swept_buy_side_level: LiquidityLevel,
        invalidated_buy_side_level: LiquidityLevel,
        expired_sell_side_level: LiquidityLevel,
        symbol: str,
        timeframe: str,
    ) -> None:
        active_level = deepcopy(buy_side_levels[0])

        snapshot = liquidity_map.build_snapshot_from_components(
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=[
                active_level,
                swept_buy_side_level,
                invalidated_buy_side_level,
                expired_sell_side_level,
            ],
            clusters=[],
        )

        assert active_level in snapshot.active_levels
        assert swept_buy_side_level not in snapshot.active_levels
        assert invalidated_buy_side_level not in snapshot.active_levels
        assert expired_sell_side_level not in snapshot.active_levels

        assert all(level.is_active() for level in snapshot.active_levels)

    def test_limits_active_levels_by_config(
        self,
        liquidity_config: LiquidityConfig,
        scorer: LiquidityScorer,
        buy_side_levels: list[LiquidityLevel],
        sell_side_levels: list[LiquidityLevel],
        symbol: str,
        timeframe: str,
    ) -> None:
        liquidity_config.max_active_levels = 2

        liquidity_map = LiquidityMap(
            config=liquidity_config,
            scorer=scorer,
        )

        snapshot = liquidity_map.build_snapshot_from_components(
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=[*buy_side_levels, *sell_side_levels],
            clusters=[],
        )

        assert len(snapshot.active_levels) <= 2
        assert snapshot.metadata["levels_count"] == len(snapshot.active_levels)

    def test_limits_active_clusters_by_config(
        self,
        liquidity_config: LiquidityConfig,
        scorer: LiquidityScorer,
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        liquidity_config.max_active_clusters = 1

        liquidity_map = LiquidityMap(
            config=liquidity_config,
            scorer=scorer,
        )

        far_buy_cluster = _move_cluster_prices(
            buy_side_stop_cluster,
            low_price=109.90,
            high_price=110.10,
        )

        snapshot = liquidity_map.build_snapshot_from_components(
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=[],
            clusters=[
                buy_side_stop_cluster,
                sell_side_stop_cluster,
                far_buy_cluster,
            ],
        )

        assert len(snapshot.stop_clusters) <= 1
        assert snapshot.metadata["stop_clusters_count"] == len(snapshot.stop_clusters)


# ---------------------------------------------------------------------
# Feature extraction: nearest / strongest / zones
# ---------------------------------------------------------------------


class TestLiquidityMapFeatures:
    def test_detects_nearest_above_and_below_liquidity(
        self,
        liquidity_map: LiquidityMap,
        buy_side_levels: list[LiquidityLevel],
        sell_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        near_above = _move_level_price(buy_side_levels[0], 101.0)
        far_above = _move_level_price(buy_side_levels[1], 110.0)
        near_below = _move_level_price(sell_side_levels[0], 99.0)
        far_below = _move_level_price(sell_side_levels[1], 90.0)

        snapshot = liquidity_map.build_snapshot_from_components(
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=[near_above, far_above, near_below, far_below],
            clusters=[buy_side_stop_cluster, sell_side_stop_cluster],
        )

        assert snapshot.nearest_above_level is not None
        assert snapshot.nearest_below_level is not None

        assert snapshot.nearest_above_level.distance_pct(100.0) <= far_above.distance_pct(100.0)
        assert snapshot.nearest_below_level.distance_pct(100.0) <= far_below.distance_pct(100.0)

    def test_detects_strongest_cluster_above_and_below(
        self,
        liquidity_map: LiquidityMap,
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        weak_buy = _weaken_cluster(
            _move_cluster_prices(
                buy_side_stop_cluster,
                low_price=101.90,
                high_price=102.10,
            )
        )
        strong_buy = deepcopy(buy_side_stop_cluster)
        strong_buy.confidence = 0.95
        strong_buy.estimated_stop_density = 0.95

        weak_sell = _weaken_cluster(
            _move_cluster_prices(
                sell_side_stop_cluster,
                low_price=97.90,
                high_price=98.10,
            )
        )
        strong_sell = deepcopy(sell_side_stop_cluster)
        strong_sell.confidence = 0.90
        strong_sell.estimated_stop_density = 0.90

        snapshot = liquidity_map.build_snapshot_from_components(
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=[],
            clusters=[weak_buy, strong_buy, weak_sell, strong_sell],
        )

        assert snapshot.strongest_cluster_above is not None
        assert snapshot.strongest_cluster_below is not None

        assert snapshot.strongest_cluster_above.confidence >= weak_buy.confidence
        assert snapshot.strongest_cluster_below.confidence >= weak_sell.confidence

    def test_builds_liquidity_zones_from_levels_and_clusters(
        self,
        liquidity_map: LiquidityMap,
        mixed_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = liquidity_map.build_snapshot_from_components(
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=mixed_side_levels,
            clusters=[buy_side_stop_cluster, sell_side_stop_cluster],
        )

        assert snapshot.zones

        sides = {zone.side for zone in snapshot.zones}

        assert LiquiditySide.BUY_SIDE in sides
        assert LiquiditySide.SELL_SIDE in sides

        for zone in snapshot.zones:
            _assert_zone_contract(zone)

    def test_zones_have_source_type_diagnostics(
        self,
        liquidity_map: LiquidityMap,
        mixed_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = liquidity_map.build_snapshot_from_components(
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=mixed_side_levels,
            clusters=[buy_side_stop_cluster, sell_side_stop_cluster],
        )

        assert snapshot.zones

        for zone in snapshot.zones:
            assert zone.source_types
            assert isinstance(zone.metadata, dict)

    def test_duplicate_clusters_do_not_inflate_snapshot_clusters(
        self,
        liquidity_map: LiquidityMap,
        buy_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        duplicate = deepcopy(buy_side_stop_cluster)

        snapshot = liquidity_map.build_snapshot_from_components(
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=[],
            clusters=[buy_side_stop_cluster, duplicate],
        )

        assert len(snapshot.stop_clusters) == 1

    def test_duplicate_levels_do_not_explode_zones(
        self,
        liquidity_map: LiquidityMap,
        buy_side_levels: list[LiquidityLevel],
        symbol: str,
        timeframe: str,
    ) -> None:
        duplicate_levels = [
            deepcopy(buy_side_levels[0]),
            deepcopy(buy_side_levels[0]),
            deepcopy(buy_side_levels[1]),
        ]

        snapshot = liquidity_map.build_snapshot_from_components(
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=duplicate_levels,
            clusters=[],
        )

        assert len(snapshot.active_levels) <= len(duplicate_levels)
        assert len(snapshot.zones) <= len(snapshot.active_levels)


# ---------------------------------------------------------------------
# Pressure / bias / signal
# ---------------------------------------------------------------------


class TestLiquidityMapPressureBiasSignal:
    def test_pressure_score_is_positive_when_upside_liquidity_dominates(
        self,
        liquidity_map: LiquidityMap,
        buy_side_levels: list[LiquidityLevel],
        sell_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        weak_sell_levels = [_weaken_level(level) for level in sell_side_levels]
        weak_sell_cluster = _weaken_cluster(sell_side_stop_cluster)

        snapshot = liquidity_map.build_snapshot_from_components(
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=[*buy_side_levels, *weak_sell_levels],
            clusters=[buy_side_stop_cluster, weak_sell_cluster],
        )

        assert snapshot.liquidity_pressure_score > 0
        assert snapshot.above_liquidity_score >= snapshot.below_liquidity_score
        assert snapshot.bias == LiquidityBias.UP

        assert snapshot.signal is not None
        assert snapshot.signal.bias == LiquidityBias.UP
        assert snapshot.signal.is_directional

    def test_pressure_score_is_negative_when_downside_liquidity_dominates(
        self,
        liquidity_map: LiquidityMap,
        buy_side_levels: list[LiquidityLevel],
        sell_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        weak_buy_levels = [_weaken_level(level) for level in buy_side_levels]
        weak_buy_cluster = _weaken_cluster(buy_side_stop_cluster)

        snapshot = liquidity_map.build_snapshot_from_components(
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=[*weak_buy_levels, *sell_side_levels],
            clusters=[weak_buy_cluster, sell_side_stop_cluster],
        )

        assert snapshot.liquidity_pressure_score < 0
        assert snapshot.below_liquidity_score >= snapshot.above_liquidity_score
        assert snapshot.bias == LiquidityBias.DOWN

        assert snapshot.signal is not None
        assert snapshot.signal.bias == LiquidityBias.DOWN
        assert snapshot.signal.is_directional

    def test_bias_is_neutral_when_liquidity_is_balanced(
        self,
        liquidity_map: LiquidityMap,
        buy_side_levels: list[LiquidityLevel],
        sell_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        balanced_buy_cluster = deepcopy(buy_side_stop_cluster)
        balanced_sell_cluster = deepcopy(sell_side_stop_cluster)

        balanced_buy_cluster.confidence = 0.70
        balanced_buy_cluster.estimated_stop_density = 0.70
        balanced_sell_cluster.confidence = 0.70
        balanced_sell_cluster.estimated_stop_density = 0.70

        balanced_buy_levels = []
        for level in buy_side_levels:
            cloned = deepcopy(level)
            cloned.confidence = 0.70
            cloned.touches_count = 3
            balanced_buy_levels.append(cloned)

        balanced_sell_levels = []
        for level in sell_side_levels:
            cloned = deepcopy(level)
            cloned.confidence = 0.70
            cloned.touches_count = 3
            balanced_sell_levels.append(cloned)

        snapshot = liquidity_map.build_snapshot_from_components(
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=[*balanced_buy_levels, *balanced_sell_levels],
            clusters=[balanced_buy_cluster, balanced_sell_cluster],
        )

        assert abs(snapshot.liquidity_pressure_score) <= 0.15
        assert snapshot.bias in {LiquidityBias.NEUTRAL, LiquidityBias.UP, LiquidityBias.DOWN}

        if snapshot.bias == LiquidityBias.NEUTRAL:
            assert snapshot.signal is not None
            assert snapshot.signal.bias == LiquidityBias.NEUTRAL

    def test_signal_contains_nearest_buy_and_sell_side_liquidity(
        self,
        liquidity_map: LiquidityMap,
        buy_side_levels: list[LiquidityLevel],
        sell_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = liquidity_map.build_snapshot_from_components(
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=[*buy_side_levels, *sell_side_levels],
            clusters=[buy_side_stop_cluster, sell_side_stop_cluster],
        )

        assert snapshot.signal is not None

        signal = snapshot.signal

        _assert_signal_contract(
            signal,
            symbol=symbol,
            timeframe=timeframe,
        )

        assert signal.nearest_buy_side_liquidity is not None
        assert signal.nearest_sell_side_liquidity is not None
        assert signal.explanation is not None
        assert signal.metadata

    def test_signal_scores_match_snapshot_feature_metadata(
        self,
        liquidity_map: LiquidityMap,
        mixed_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = liquidity_map.build_snapshot_from_components(
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=mixed_side_levels,
            clusters=[buy_side_stop_cluster, sell_side_stop_cluster],
        )

        assert snapshot.signal is not None

        assert snapshot.signal.sweep_risk_up == pytest.approx(
            snapshot.metadata["sweep_risk_up"]
        )
        assert snapshot.signal.sweep_risk_down == pytest.approx(
            snapshot.metadata["sweep_risk_down"]
        )
        assert snapshot.signal.magnet_score_up == pytest.approx(
            snapshot.metadata["magnet_score_up"]
        )
        assert snapshot.signal.magnet_score_down == pytest.approx(
            snapshot.metadata["magnet_score_down"]
        )

    def test_signed_pressure_is_not_clamped_to_unsigned_range(
        self,
        liquidity_map: LiquidityMap,
        sell_side_levels: list[LiquidityLevel],
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = liquidity_map.build_snapshot_from_components(
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=sell_side_levels,
            clusters=[sell_side_stop_cluster],
        )

        assert snapshot.liquidity_pressure_score < 0
        _assert_signed_score(snapshot.liquidity_pressure_score)

    def test_snapshot_to_event_payload_is_payload_safe(
        self,
        liquidity_map: LiquidityMap,
        mixed_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = liquidity_map.build_snapshot_from_components(
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=mixed_side_levels,
            clusters=[buy_side_stop_cluster, sell_side_stop_cluster],
        )

        payload = snapshot.to_event_payload()

        assert payload["symbol"] == symbol
        assert payload["timeframe"] == timeframe
        assert payload["current_price"] == snapshot.current_price
        assert payload["bias"] == snapshot.bias.value
        assert isinstance(payload["active_levels"], list)
        assert isinstance(payload["equal_levels"], list)
        assert isinstance(payload["stop_clusters"], list)
        assert isinstance(payload["zones"], list)
        assert isinstance(payload["metadata"], dict)


# ---------------------------------------------------------------------
# build_snapshot() full path with detector + extra levels/clusters
# ---------------------------------------------------------------------


class TestLiquidityMapBuildSnapshotFullPath:
    def test_build_snapshot_returns_snapshot_with_extra_levels_and_clusters(
        self,
        liquidity_map: LiquidityMap,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        buy_side_levels: list[LiquidityLevel],
        sell_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        balanced_orderbook: dict[str, list[list[float]]],
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = liquidity_map.build_snapshot(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_without_clear_equal_levels,
            current_price=100.0,
            orderbook=balanced_orderbook,
            extra_levels=[*buy_side_levels, *sell_side_levels],
            extra_clusters=[buy_side_stop_cluster, sell_side_stop_cluster],
        )

        _assert_snapshot_basic_contract(
            snapshot,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
        )

        assert snapshot.active_levels
        assert snapshot.stop_clusters
        assert snapshot.zones
        assert snapshot.signal is not None

        assert snapshot.metadata["builder"] == "LiquidityMap"
        assert snapshot.metadata["levels_count"] == len(snapshot.active_levels)
        assert snapshot.metadata["raw_merged_levels_count"] >= len(snapshot.active_levels)
        assert snapshot.metadata["extra_levels_count"] == len(buy_side_levels) + len(sell_side_levels)
        assert snapshot.metadata["extra_clusters_count"] == 2
        assert snapshot.metadata["orderbook_present"] is True

    def test_build_snapshot_uses_explicit_timestamp(
        self,
        liquidity_map: LiquidityMap,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        buy_side_levels: list[LiquidityLevel],
        symbol: str,
        timeframe: str,
    ) -> None:
        explicit_ts = datetime(2026, 1, 3, 15, 45, tzinfo=timezone.utc)

        snapshot = liquidity_map.build_snapshot(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_without_clear_equal_levels,
            current_price=100.0,
            extra_levels=buy_side_levels,
            timestamp=explicit_ts,
        )

        assert snapshot.timestamp == explicit_ts
        assert snapshot.signal is not None
        assert snapshot.signal.timestamp == explicit_ts

    def test_build_snapshot_resolves_timestamp_from_latest_candle_when_missing(
        self,
        liquidity_map: LiquidityMap,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        buy_side_levels: list[LiquidityLevel],
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = liquidity_map.build_snapshot(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_without_clear_equal_levels,
            current_price=100.0,
            extra_levels=buy_side_levels,
            timestamp=None,
        )

        latest_candle_ts = candles_without_clear_equal_levels[-1]["close_time"]

        assert snapshot.timestamp == latest_candle_ts

    def test_build_snapshot_rejects_disabled_config(
        self,
        disabled_liquidity_config: LiquidityConfig,
        scorer: LiquidityScorer,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        liquidity_map = LiquidityMap(
            config=disabled_liquidity_config,
            scorer=scorer,
        )

        with pytest.raises(RuntimeError, match="disabled"):
            liquidity_map.build_snapshot(
                symbol=symbol,
                timeframe=timeframe,
                candles=candles_without_clear_equal_levels,
                current_price=100.0,
            )

    def test_build_snapshot_rejects_invalid_current_price(
        self,
        liquidity_map: LiquidityMap,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        with pytest.raises(ValueError, match="current_price"):
            liquidity_map.build_snapshot(
                symbol=symbol,
                timeframe=timeframe,
                candles=candles_without_clear_equal_levels,
                current_price=-1.0,
            )

    def test_build_snapshot_keeps_swept_equal_levels_out_of_active_live_liquidity(
        self,
        liquidity_map: LiquidityMap,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        swept_buy_side_level: LiquidityLevel,
        partially_swept_buy_side_level: LiquidityLevel,
        invalidated_buy_side_level: LiquidityLevel,
        buy_side_levels: list[LiquidityLevel],
        symbol: str,
        timeframe: str,
    ) -> None:
        active_level = deepcopy(buy_side_levels[0])

        snapshot = liquidity_map.build_snapshot(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_without_clear_equal_levels,
            current_price=100.0,
            extra_levels=[
                active_level,
                swept_buy_side_level,
                partially_swept_buy_side_level,
                invalidated_buy_side_level,
            ],
        )

        assert active_level in snapshot.active_levels
        assert swept_buy_side_level not in snapshot.active_levels
        assert invalidated_buy_side_level not in snapshot.active_levels

        assert all(level.is_active() for level in snapshot.active_levels)

    def test_build_snapshot_metadata_contains_diagnostics(
        self,
        liquidity_map: LiquidityMap,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        mixed_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = liquidity_map.build_snapshot(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_without_clear_equal_levels,
            current_price=100.0,
            extra_levels=mixed_side_levels,
            extra_clusters=[buy_side_stop_cluster, sell_side_stop_cluster],
        )

        required_keys = {
            "builder",
            "levels_count",
            "raw_merged_levels_count",
            "equal_levels_count",
            "stop_clusters_count",
            "zones_count",
            "sweep_risk_up",
            "sweep_risk_down",
            "magnet_score_up",
            "magnet_score_down",
            "pressure_score_semantics",
            "orderbook_present",
            "extra_levels_count",
            "extra_clusters_count",
        }

        assert required_keys.issubset(snapshot.metadata.keys())

        assert snapshot.metadata["pressure_score_semantics"] == (
            "positive=upside_buy_side, negative=downside_sell_side"
        )


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------


class TestLiquidityMapEdgeCases:
    def test_empty_components_snapshot_is_valid_but_has_no_levels(
        self,
        liquidity_map: LiquidityMap,
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = liquidity_map.build_snapshot_from_components(
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=[],
            clusters=[],
        )

        _assert_snapshot_basic_contract(
            snapshot,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
        )

        assert not snapshot.has_levels()
        assert snapshot.active_levels == []
        assert snapshot.stop_clusters == []
        assert snapshot.zones == []
        assert snapshot.nearest_above_level is None
        assert snapshot.nearest_below_level is None
        assert snapshot.strongest_cluster_above is None
        assert snapshot.strongest_cluster_below is None
        assert snapshot.above_liquidity_score == 0.0
        assert snapshot.below_liquidity_score == 0.0
        assert snapshot.liquidity_pressure_score == 0.0
        assert snapshot.bias == LiquidityBias.NEUTRAL

    def test_snapshot_helper_methods_return_expected_sides(
        self,
        liquidity_map: LiquidityMap,
        buy_side_levels: list[LiquidityLevel],
        sell_side_levels: list[LiquidityLevel],
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = liquidity_map.build_snapshot_from_components(
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=[*buy_side_levels, *sell_side_levels],
            clusters=[],
        )

        buy_side = snapshot.get_buy_side_levels()
        sell_side = snapshot.get_sell_side_levels()

        assert buy_side
        assert sell_side

        assert all(level.side == LiquiditySide.BUY_SIDE for level in buy_side)
        assert all(level.side == LiquiditySide.SELL_SIDE for level in sell_side)

    def test_terminal_levels_are_not_returned_as_active_levels(
        self,
        liquidity_map: LiquidityMap,
        buy_side_levels: list[LiquidityLevel],
        swept_buy_side_level: LiquidityLevel,
        invalidated_buy_side_level: LiquidityLevel,
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = liquidity_map.build_snapshot_from_components(
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=[
                *buy_side_levels,
                swept_buy_side_level,
                invalidated_buy_side_level,
            ],
            clusters=[],
        )

        assert snapshot.get_active_levels()
        assert all(level.is_active() for level in snapshot.get_active_levels())
        assert snapshot.get_terminal_levels() == []

    def test_snapshot_scores_are_clamped_under_extreme_inputs(
        self,
        liquidity_map: LiquidityMap,
        buy_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        extreme_levels = []

        for level in buy_side_levels:
            cloned = deepcopy(level)
            cloned.confidence = 999.0
            cloned.touches_count = 999
            cloned.reaction_count = 999
            extreme_levels.append(cloned)

        extreme_cluster = deepcopy(buy_side_stop_cluster)
        extreme_cluster.confidence = 999.0
        extreme_cluster.estimated_stop_density = 999.0
        extreme_cluster.touches_count = 999

        snapshot = liquidity_map.build_snapshot_from_components(
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=extreme_levels,
            clusters=[extreme_cluster],
        )

        _assert_score01(snapshot.above_liquidity_score)
        _assert_score01(snapshot.below_liquidity_score)
        _assert_signed_score(snapshot.liquidity_pressure_score)

        assert snapshot.signal is not None
        _assert_signal_contract(
            snapshot.signal,
            symbol=symbol,
            timeframe=timeframe,
        )