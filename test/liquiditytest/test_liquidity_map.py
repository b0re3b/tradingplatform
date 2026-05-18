# tests/analytics/liquidity/test_liquidity_map.py

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
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
    make_liquidity_key,
)
from analytics.liquidity.scoring import LiquidityScorer


# ---------------------------------------------------------------------
# Canonical futures scope
# ---------------------------------------------------------------------


TEST_EXCHANGE = "binance"
TEST_MARKET_TYPE = "usdm_futures"
ALT_EXCHANGE = "bybit"
ALT_MARKET_TYPE = "linear"


# ---------------------------------------------------------------------
# Local assertions / helpers
# ---------------------------------------------------------------------


def _assert_score01(value: float) -> None:
    assert 0.0 <= value <= 1.0


def _assert_signed_score(value: float) -> None:
    assert -1.0 <= value <= 1.0


def _expected_scope(
    *,
    symbol: str,
    timeframe: str,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
) -> dict[str, str]:
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
    }


def _expected_scope_key(
    *,
    symbol: str,
    timeframe: str,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
) -> str:
    return f"{exchange}:{market_type}:{symbol}:{timeframe}"


def _expected_key(
    *,
    symbol: str,
    timeframe: str,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
):
    return make_liquidity_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )


def _assert_scoped_payload(
    payload: dict[str, Any],
    *,
    symbol: str,
    timeframe: str,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
) -> None:
    assert payload["exchange"] == exchange
    assert payload["market_type"] == market_type
    assert payload["symbol"] == symbol
    assert payload["timeframe"] == timeframe
    assert payload["scope"] == _expected_scope(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )
    assert payload["scope_key"] == _expected_scope_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )
    assert payload["liquidity_key"] == _expected_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )


def _assert_model_scope(
    model: Any,
    *,
    symbol: str,
    timeframe: str,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
) -> None:
    assert model.exchange == exchange
    assert model.market_type == market_type
    assert model.symbol == symbol
    assert model.timeframe == timeframe
    assert model.scope == _expected_scope(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )
    assert model.scope_key == _expected_scope_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )
    assert model.liquidity_key == _expected_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )


def _assert_signal_contract(
    signal: LiquiditySignal,
    *,
    symbol: str,
    timeframe: str,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
) -> None:
    _assert_model_scope(
        signal,
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )

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

    _assert_scoped_payload(
        payload,
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )

    assert payload["bias"] == signal.bias.value
    assert isinstance(payload["metadata"], dict)


def _assert_level_contract(
    level: LiquidityLevel,
    *,
    symbol: str,
    timeframe: str,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
) -> None:
    _assert_model_scope(
        level,
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )

    assert level.price > 0
    assert level.level_type in set(LiquidityLevelType)
    assert level.side in set(LiquiditySide)
    assert level.status in set(LiquidityStatus)
    assert level.sweep_status in set(SweepStatus)
    _assert_score01(level.confidence)

    payload = level.to_event_payload()

    _assert_scoped_payload(
        payload,
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )

    assert payload["level_type"] == level.level_type.value
    assert payload["side"] == level.side.value
    assert payload["status"] == level.status.value
    assert payload["sweep_status"] == level.sweep_status.value
    assert isinstance(payload["metadata"], dict)


def _assert_cluster_contract(
    cluster: StopCluster,
    *,
    symbol: str,
    timeframe: str,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
) -> None:
    _assert_model_scope(
        cluster,
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )

    assert cluster.low_price > 0
    assert cluster.high_price > 0
    assert cluster.low_price <= cluster.high_price
    assert cluster.center_price >= cluster.low_price
    assert cluster.center_price <= cluster.high_price
    assert cluster.side in set(LiquiditySide)
    _assert_score01(cluster.confidence)
    _assert_score01(cluster.estimated_stop_density)

    payload = cluster.to_event_payload()

    _assert_scoped_payload(
        payload,
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )

    assert payload["side"] == cluster.side.value
    assert payload["low_price"] == cluster.low_price
    assert payload["high_price"] == cluster.high_price
    assert payload["center_price"] == cluster.center_price
    assert isinstance(payload["source_levels"], list)
    assert isinstance(payload["metadata"], dict)

    for source_level in cluster.source_levels:
        _assert_level_contract(
            source_level,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )


def _assert_zone_contract(
    zone: LiquidityZone,
    *,
    symbol: str,
    timeframe: str,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
) -> None:
    _assert_model_scope(
        zone,
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )

    assert zone.low_price > 0
    assert zone.high_price > 0
    assert zone.low_price <= zone.high_price
    assert zone.center_price >= zone.low_price
    assert zone.center_price <= zone.high_price
    assert zone.side in set(LiquiditySide)
    _assert_score01(zone.score)

    payload = zone.to_event_payload()

    _assert_scoped_payload(
        payload,
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )

    assert payload["side"] == zone.side.value
    assert payload["low_price"] == zone.low_price
    assert payload["high_price"] == zone.high_price
    assert isinstance(payload["source_types"], list)
    assert isinstance(payload["metadata"], dict)


def _assert_snapshot_contract(
    snapshot: LiquidityMapSnapshot,
    *,
    symbol: str,
    timeframe: str,
    current_price: float,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
) -> None:
    _assert_model_scope(
        snapshot,
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )

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

    assert snapshot.metadata["scope"] == _expected_scope(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )
    assert snapshot.metadata["scope_key"] == _expected_scope_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )
    assert snapshot.metadata["exchange"] == exchange
    assert snapshot.metadata["market_type"] == market_type

    for level in snapshot.active_levels:
        _assert_level_contract(
            level,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

    for level in snapshot.equal_levels:
        _assert_level_contract(
            level,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

    for cluster in snapshot.stop_clusters:
        _assert_cluster_contract(
            cluster,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

    for zone in snapshot.zones:
        _assert_zone_contract(
            zone,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

    if snapshot.nearest_above_level is not None:
        _assert_model_scope(
            snapshot.nearest_above_level,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

    if snapshot.nearest_below_level is not None:
        _assert_model_scope(
            snapshot.nearest_below_level,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

    if snapshot.strongest_cluster_above is not None:
        _assert_cluster_contract(
            snapshot.strongest_cluster_above,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

    if snapshot.strongest_cluster_below is not None:
        _assert_cluster_contract(
            snapshot.strongest_cluster_below,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

    assert snapshot.signal is not None
    _assert_signal_contract(
        snapshot.signal,
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )

    payload = snapshot.to_event_payload()

    _assert_scoped_payload(
        payload,
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )

    assert payload["current_price"] == snapshot.current_price
    assert payload["bias"] == snapshot.bias.value
    assert isinstance(payload["active_levels"], list)
    assert isinstance(payload["equal_levels"], list)
    assert isinstance(payload["stop_clusters"], list)
    assert isinstance(payload["zones"], list)
    assert isinstance(payload["metadata"], dict)
    assert payload["metadata"]["scope"] == snapshot.scope
    assert payload["metadata"]["active_levels_count"] == len(snapshot.active_levels)
    assert payload["metadata"]["equal_levels_count"] == len(snapshot.equal_levels)
    assert payload["metadata"]["stop_clusters_count"] == len(snapshot.stop_clusters)
    assert payload["metadata"]["zones_count"] == len(snapshot.zones)


def _clone_levels(
    levels: list[LiquidityLevel],
    *,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
    symbol: str,
    timeframe: str,
) -> list[LiquidityLevel]:
    cloned_levels: list[LiquidityLevel] = []

    for level in levels:
        cloned = deepcopy(level)
        cloned.exchange = exchange
        cloned.market_type = market_type
        cloned.symbol = symbol
        cloned.timeframe = timeframe
        cloned.metadata = dict(cloned.metadata or {})
        cloned.metadata["scope"] = _expected_scope(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        cloned.metadata["scope_key"] = _expected_scope_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        cloned_levels.append(cloned)

    return cloned_levels


def _clone_clusters(
    clusters: list[StopCluster],
    *,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
    symbol: str,
    timeframe: str,
) -> list[StopCluster]:
    cloned_clusters: list[StopCluster] = []

    for cluster in clusters:
        cloned = deepcopy(cluster)
        cloned.exchange = exchange
        cloned.market_type = market_type
        cloned.symbol = symbol
        cloned.timeframe = timeframe
        cloned.metadata = dict(cloned.metadata or {})
        cloned.metadata["scope"] = _expected_scope(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        cloned.metadata["scope_key"] = _expected_scope_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

        cloned.source_levels = _clone_levels(
            list(cloned.source_levels),
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

        cloned_clusters.append(cloned)

    return cloned_clusters


def _build_component_snapshot(
    liquidity_map: LiquidityMap,
    *,
    symbol: str,
    timeframe: str,
    current_price: float = 100.0,
    levels: list[LiquidityLevel] | None = None,
    clusters: list[StopCluster] | None = None,
    equal_levels: list[EqualLevel] | None = None,
    timestamp: datetime | None = None,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
) -> LiquidityMapSnapshot:
    scoped_levels = _clone_levels(
        list(levels or []),
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )

    scoped_clusters = _clone_clusters(
        list(clusters or []),
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )

    scoped_equal_levels = _clone_levels(
        list(equal_levels or []),
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )

    return liquidity_map.build_snapshot_from_components(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
        current_price=current_price,
        levels=scoped_levels,
        clusters=scoped_clusters,
        equal_levels=scoped_equal_levels,
        timestamp=timestamp,
    )


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
    def test_build_snapshot_from_components_returns_complete_scoped_snapshot(
        self,
        liquidity_map: LiquidityMap,
        mixed_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = _build_component_snapshot(
            liquidity_map,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=mixed_side_levels,
            clusters=[buy_side_stop_cluster, sell_side_stop_cluster],
        )

        _assert_snapshot_contract(
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

        assert snapshot.metadata["builder"] == "LiquidityMap"
        assert snapshot.metadata["from_components"] is True
        assert snapshot.metadata["levels_count"] == len(snapshot.active_levels)
        assert snapshot.metadata["stop_clusters_count"] == len(snapshot.stop_clusters)
        assert snapshot.metadata["zones_count"] == len(snapshot.zones)

    def test_build_snapshot_from_components_uses_explicit_aware_timestamp(
        self,
        liquidity_map: LiquidityMap,
        mixed_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        explicit_ts = datetime(2026, 1, 2, 10, 30, tzinfo=timezone.utc)

        snapshot = _build_component_snapshot(
            liquidity_map,
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

        snapshot = _build_component_snapshot(
            liquidity_map,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=mixed_side_levels,
            clusters=[buy_side_stop_cluster, sell_side_stop_cluster],
            timestamp=naive_ts,
        )

        assert snapshot.timestamp.tzinfo is not None
        assert snapshot.timestamp.utcoffset() == timezone.utc.utcoffset(
            snapshot.timestamp
        )

    def test_build_snapshot_from_components_rejects_missing_symbol(
        self,
        liquidity_map: LiquidityMap,
        mixed_side_levels: list[LiquidityLevel],
        timeframe: str,
    ) -> None:
        with pytest.raises(ValueError, match="symbol"):
            liquidity_map.build_snapshot_from_components(
                exchange=TEST_EXCHANGE,
                market_type=TEST_MARKET_TYPE,
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
        with pytest.raises(ValueError, match="timeframe"):
            liquidity_map.build_snapshot_from_components(
                exchange=TEST_EXCHANGE,
                market_type=TEST_MARKET_TYPE,
                symbol=symbol,
                timeframe="",
                current_price=100.0,
                levels=mixed_side_levels,
                clusters=[],
            )

    @pytest.mark.parametrize("bad_price", [0.0, -1.0, None, "not-a-price"])
    def test_build_snapshot_from_components_rejects_invalid_current_price(
        self,
        liquidity_map: LiquidityMap,
        mixed_side_levels: list[LiquidityLevel],
        symbol: str,
        timeframe: str,
        bad_price: Any,
    ) -> None:
        with pytest.raises(ValueError, match="current_price"):
            liquidity_map.build_snapshot_from_components(
                exchange=TEST_EXCHANGE,
                market_type=TEST_MARKET_TYPE,
                symbol=symbol,
                timeframe=timeframe,
                current_price=bad_price,
                levels=mixed_side_levels,
                clusters=[],
            )

    def test_build_snapshot_from_components_rescopes_stale_component_models(
        self,
        liquidity_map: LiquidityMap,
        mixed_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        stale_levels = _clone_levels(
            mixed_side_levels,
            exchange="unknown",
            market_type="perpetual",
            symbol=symbol,
            timeframe=timeframe,
        )
        stale_clusters = _clone_clusters(
            [buy_side_stop_cluster],
            exchange="unknown",
            market_type="perpetual",
            symbol=symbol,
            timeframe=timeframe,
        )

        snapshot = liquidity_map.build_snapshot_from_components(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=stale_levels,
            clusters=stale_clusters,
        )

        _assert_snapshot_contract(
            snapshot,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
        )

        assert all(level.exchange == TEST_EXCHANGE for level in snapshot.active_levels)
        assert all(
            level.market_type == TEST_MARKET_TYPE
            for level in snapshot.active_levels
        )
        assert all(
            cluster.exchange == TEST_EXCHANGE
            for cluster in snapshot.stop_clusters
        )
        assert all(
            cluster.market_type == TEST_MARKET_TYPE
            for cluster in snapshot.stop_clusters
        )

    def test_build_snapshot_from_components_filters_terminal_levels_from_live_snapshot(
        self,
        liquidity_map: LiquidityMap,
        buy_side_levels: list[LiquidityLevel],
        swept_buy_side_level: LiquidityLevel,
        invalidated_buy_side_level: LiquidityLevel,
        expired_sell_side_level: LiquidityLevel,
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = _build_component_snapshot(
            liquidity_map,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=[
                *buy_side_levels,
                swept_buy_side_level,
                invalidated_buy_side_level,
                expired_sell_side_level,
            ],
            clusters=[],
        )

        assert snapshot.active_levels
        assert all(level.is_active() for level in snapshot.active_levels)
        assert swept_buy_side_level.key not in {
            level.key for level in snapshot.active_levels
        }
        assert invalidated_buy_side_level.key not in {
            level.key for level in snapshot.active_levels
        }
        assert expired_sell_side_level.key not in {
            level.key for level in snapshot.active_levels
        }

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

        snapshot = _build_component_snapshot(
            liquidity_map,
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

        snapshot = _build_component_snapshot(
            liquidity_map,
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
# build_snapshot() from candles / detectors
# ---------------------------------------------------------------------


class TestLiquidityMapFromCandles:
    def test_build_snapshot_from_candles_detects_equal_levels_and_clusters(
        self,
        liquidity_map: LiquidityMap,
        candles_with_equal_highs: list[dict[str, Any]],
        orderbook_near_buy_side_cluster: dict[str, list[list[float]]],
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = liquidity_map.build_snapshot(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_with_equal_highs,
            current_price=100.0,
            orderbook=orderbook_near_buy_side_cluster,
        )

        _assert_snapshot_contract(
            snapshot,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
        )

        assert snapshot.equal_levels
        assert snapshot.active_levels
        assert snapshot.metadata["orderbook_present"] is True
        assert snapshot.metadata["equal_levels_count"] == len(snapshot.equal_levels)
        assert snapshot.metadata["stop_clusters_count"] == len(snapshot.stop_clusters)

    def test_build_snapshot_from_candles_accepts_explicit_extra_levels_and_clusters(
        self,
        liquidity_map: LiquidityMap,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        buy_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        extra_levels = _clone_levels(
            buy_side_levels,
            symbol=symbol,
            timeframe=timeframe,
        )
        extra_clusters = _clone_clusters(
            [buy_side_stop_cluster],
            symbol=symbol,
            timeframe=timeframe,
        )

        snapshot = liquidity_map.build_snapshot(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_without_clear_equal_levels,
            current_price=100.0,
            extra_levels=extra_levels,
            extra_clusters=extra_clusters,
        )

        _assert_snapshot_contract(
            snapshot,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
        )

        assert snapshot.active_levels
        assert snapshot.stop_clusters
        assert snapshot.metadata["extra_levels_count"] == len(extra_levels)
        assert snapshot.metadata["extra_clusters_count"] == len(extra_clusters)

    def test_build_snapshot_uses_last_candle_timestamp_when_explicit_timestamp_missing(
        self,
        liquidity_map: LiquidityMap,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = liquidity_map.build_snapshot(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_without_clear_equal_levels,
            current_price=candles_without_clear_equal_levels[-1]["close"],
        )

        assert snapshot.timestamp == candles_without_clear_equal_levels[-1]["close_time"]

    def test_build_snapshot_uses_explicit_timestamp_over_candle_timestamp(
        self,
        liquidity_map: LiquidityMap,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        explicit_ts = datetime(2026, 1, 2, 9, 15, tzinfo=timezone.utc)

        snapshot = liquidity_map.build_snapshot(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_without_clear_equal_levels,
            current_price=100.0,
            timestamp=explicit_ts,
        )

        assert snapshot.timestamp == explicit_ts

    def test_build_snapshot_rejects_disabled_config(
        self,
        disabled_liquidity_config: LiquidityConfig,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        liquidity_map = LiquidityMap(config=disabled_liquidity_config)

        with pytest.raises(RuntimeError, match="disabled"):
            liquidity_map.build_snapshot(
                exchange=TEST_EXCHANGE,
                market_type=TEST_MARKET_TYPE,
                symbol=symbol,
                timeframe=timeframe,
                candles=candles_without_clear_equal_levels,
                current_price=100.0,
            )

    @pytest.mark.parametrize("bad_price", [0.0, -100.0, None, "bad-price"])
    def test_build_snapshot_rejects_invalid_current_price(
        self,
        liquidity_map: LiquidityMap,
        candles_without_clear_equal_levels: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
        bad_price: Any,
    ) -> None:
        with pytest.raises(ValueError, match="current_price"):
            liquidity_map.build_snapshot(
                exchange=TEST_EXCHANGE,
                market_type=TEST_MARKET_TYPE,
                symbol=symbol,
                timeframe=timeframe,
                candles=candles_without_clear_equal_levels,
                current_price=bad_price,
            )


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

        snapshot = _build_component_snapshot(
            liquidity_map,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=[near_above, far_above, near_below, far_below],
            clusters=[buy_side_stop_cluster, sell_side_stop_cluster],
        )

        assert snapshot.nearest_above_level is not None
        assert snapshot.nearest_below_level is not None

        assert (
            snapshot.nearest_above_level.distance_pct(100.0)
            <= far_above.distance_pct(100.0)
        )
        assert (
            snapshot.nearest_below_level.distance_pct(100.0)
            <= far_below.distance_pct(100.0)
        )

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

        snapshot = _build_component_snapshot(
            liquidity_map,
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

    def test_builds_zones_from_levels_and_clusters(
        self,
        liquidity_map: LiquidityMap,
        mixed_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = _build_component_snapshot(
            liquidity_map,
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
            _assert_zone_contract(
                zone,
                symbol=symbol,
                timeframe=timeframe,
            )

    def test_zones_have_source_type_and_scope_diagnostics(
        self,
        liquidity_map: LiquidityMap,
        mixed_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = _build_component_snapshot(
            liquidity_map,
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
            assert zone.metadata["scope"] == snapshot.scope

    def test_duplicate_clusters_do_not_inflate_snapshot_clusters(
        self,
        liquidity_map: LiquidityMap,
        buy_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        duplicate = deepcopy(buy_side_stop_cluster)

        snapshot = _build_component_snapshot(
            liquidity_map,
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

        snapshot = _build_component_snapshot(
            liquidity_map,
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

        snapshot = _build_component_snapshot(
            liquidity_map,
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

        snapshot = _build_component_snapshot(
            liquidity_map,
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

    def test_pressure_score_is_neutral_when_sides_are_balanced(
        self,
        liquidity_map: LiquidityMap,
        buy_side_levels: list[LiquidityLevel],
        sell_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        buy_cluster = deepcopy(buy_side_stop_cluster)
        sell_cluster = deepcopy(sell_side_stop_cluster)

        buy_cluster.confidence = 0.70
        buy_cluster.estimated_stop_density = 0.70
        sell_cluster.confidence = 0.70
        sell_cluster.estimated_stop_density = 0.70

        balanced_buy_levels = []
        for level in buy_side_levels:
            cloned = deepcopy(level)
            cloned.confidence = 0.70
            cloned.touches_count = 3
            cloned.reaction_count = 2
            balanced_buy_levels.append(cloned)

        balanced_sell_levels = []
        for level in sell_side_levels:
            cloned = deepcopy(level)
            cloned.confidence = 0.70
            cloned.touches_count = 3
            cloned.reaction_count = 2
            balanced_sell_levels.append(cloned)

        snapshot = _build_component_snapshot(
            liquidity_map,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=[*balanced_buy_levels, *balanced_sell_levels],
            clusters=[buy_cluster, sell_cluster],
        )

        _assert_signed_score(snapshot.liquidity_pressure_score)
        assert snapshot.bias in {
            LiquidityBias.UP,
            LiquidityBias.DOWN,
            LiquidityBias.NEUTRAL,
        }

        assert snapshot.signal is not None
        _assert_signal_contract(
            snapshot.signal,
            symbol=symbol,
            timeframe=timeframe,
        )

    def test_signal_payload_contains_nearest_liquidity_diagnostics(
        self,
        liquidity_map: LiquidityMap,
        mixed_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = _build_component_snapshot(
            liquidity_map,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=mixed_side_levels,
            clusters=[buy_side_stop_cluster, sell_side_stop_cluster],
        )

        assert snapshot.signal is not None

        payload = snapshot.signal.to_event_payload()

        _assert_scoped_payload(
            payload,
            symbol=symbol,
            timeframe=timeframe,
        )

        assert "nearest_buy_side_liquidity" in payload
        assert "nearest_sell_side_liquidity" in payload
        assert isinstance(payload["metadata"], dict)
        assert payload["metadata"]["scope"] == snapshot.scope


# ---------------------------------------------------------------------
# Scope isolation / futures contract
# ---------------------------------------------------------------------


class TestLiquidityMapScopeContract:
    def test_same_symbol_timeframe_snapshots_are_isolated_by_exchange_and_market_type(
        self,
        liquidity_map: LiquidityMap,
        mixed_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        binance_snapshot = _build_component_snapshot(
            liquidity_map,
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=mixed_side_levels,
            clusters=[buy_side_stop_cluster, sell_side_stop_cluster],
        )

        bybit_snapshot = _build_component_snapshot(
            liquidity_map,
            exchange=ALT_EXCHANGE,
            market_type=ALT_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=mixed_side_levels,
            clusters=[buy_side_stop_cluster, sell_side_stop_cluster],
        )

        assert binance_snapshot.liquidity_key != bybit_snapshot.liquidity_key
        assert binance_snapshot.scope_key != bybit_snapshot.scope_key

        assert binance_snapshot.exchange == TEST_EXCHANGE
        assert binance_snapshot.market_type == TEST_MARKET_TYPE

        assert bybit_snapshot.exchange == ALT_EXCHANGE
        assert bybit_snapshot.market_type == ALT_MARKET_TYPE

        for level in binance_snapshot.active_levels:
            assert level.liquidity_key == binance_snapshot.liquidity_key

        for level in bybit_snapshot.active_levels:
            assert level.liquidity_key == bybit_snapshot.liquidity_key

    def test_snapshot_payload_preserves_futures_market_type_across_children(
        self,
        liquidity_map: LiquidityMap,
        mixed_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = _build_component_snapshot(
            liquidity_map,
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=mixed_side_levels,
            clusters=[buy_side_stop_cluster, sell_side_stop_cluster],
        )

        payload = snapshot.to_event_payload()

        assert payload["market_type"] == TEST_MARKET_TYPE
        assert payload["metadata"]["scope"]["market_type"] == TEST_MARKET_TYPE

        for level_payload in payload["active_levels"]:
            assert level_payload["market_type"] == TEST_MARKET_TYPE
            assert level_payload["scope"]["market_type"] == TEST_MARKET_TYPE

        for cluster_payload in payload["stop_clusters"]:
            assert cluster_payload["market_type"] == TEST_MARKET_TYPE
            assert cluster_payload["scope"]["market_type"] == TEST_MARKET_TYPE

        for zone_payload in payload["zones"]:
            assert zone_payload["market_type"] == TEST_MARKET_TYPE
            assert zone_payload["scope"]["market_type"] == TEST_MARKET_TYPE

        assert payload["signal"] is not None
        assert payload["signal"]["market_type"] == TEST_MARKET_TYPE
        assert payload["signal"]["scope"]["market_type"] == TEST_MARKET_TYPE

    def test_snapshot_children_have_exact_same_liquidity_key_as_parent(
        self,
        liquidity_map: LiquidityMap,
        mixed_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = _build_component_snapshot(
            liquidity_map,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=mixed_side_levels,
            clusters=[buy_side_stop_cluster, sell_side_stop_cluster],
        )

        parent_key = snapshot.liquidity_key

        assert all(level.liquidity_key == parent_key for level in snapshot.active_levels)
        assert all(level.liquidity_key == parent_key for level in snapshot.equal_levels)
        assert all(cluster.liquidity_key == parent_key for cluster in snapshot.stop_clusters)
        assert all(zone.liquidity_key == parent_key for zone in snapshot.zones)

        if snapshot.nearest_above_level is not None:
            assert snapshot.nearest_above_level.liquidity_key == parent_key

        if snapshot.nearest_below_level is not None:
            assert snapshot.nearest_below_level.liquidity_key == parent_key

        if snapshot.strongest_cluster_above is not None:
            assert snapshot.strongest_cluster_above.liquidity_key == parent_key

        if snapshot.strongest_cluster_below is not None:
            assert snapshot.strongest_cluster_below.liquidity_key == parent_key

        assert snapshot.signal is not None
        assert snapshot.signal.liquidity_key == parent_key


# ---------------------------------------------------------------------
# Snapshot helper methods / payload safety
# ---------------------------------------------------------------------


class TestLiquidityMapSnapshotHelpers:
    def test_snapshot_helper_methods_return_expected_sides(
        self,
        liquidity_map: LiquidityMap,
        buy_side_levels: list[LiquidityLevel],
        sell_side_levels: list[LiquidityLevel],
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = _build_component_snapshot(
            liquidity_map,
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

    def test_snapshot_cluster_helper_methods_return_expected_sides(
        self,
        liquidity_map: LiquidityMap,
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = _build_component_snapshot(
            liquidity_map,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=[],
            clusters=[buy_side_stop_cluster, sell_side_stop_cluster],
        )

        buy_clusters = snapshot.get_buy_side_clusters()
        sell_clusters = snapshot.get_sell_side_clusters()

        assert buy_clusters
        assert sell_clusters

        assert all(cluster.side == LiquiditySide.BUY_SIDE for cluster in buy_clusters)
        assert all(cluster.side == LiquiditySide.SELL_SIDE for cluster in sell_clusters)

    def test_terminal_levels_are_not_returned_as_active_levels(
        self,
        liquidity_map: LiquidityMap,
        buy_side_levels: list[LiquidityLevel],
        swept_buy_side_level: LiquidityLevel,
        invalidated_buy_side_level: LiquidityLevel,
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = _build_component_snapshot(
            liquidity_map,
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

    def test_get_nearest_directional_liquidity_and_strongest_cluster_helpers(
        self,
        liquidity_map: LiquidityMap,
        mixed_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = _build_component_snapshot(
            liquidity_map,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=mixed_side_levels,
            clusters=[buy_side_stop_cluster, sell_side_stop_cluster],
        )

        assert (
            snapshot.get_nearest_directional_liquidity(LiquiditySide.BUY_SIDE)
            is snapshot.nearest_above_level
        )
        assert (
            snapshot.get_nearest_directional_liquidity(LiquiditySide.SELL_SIDE)
            is snapshot.nearest_below_level
        )
        assert (
            snapshot.get_strongest_directional_cluster(LiquiditySide.BUY_SIDE)
            is snapshot.strongest_cluster_above
        )
        assert (
            snapshot.get_strongest_directional_cluster(LiquiditySide.SELL_SIDE)
            is snapshot.strongest_cluster_below
        )

    def test_snapshot_to_event_payload_is_consistent_with_runtime_lists(
        self,
        liquidity_map: LiquidityMap,
        mixed_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = _build_component_snapshot(
            liquidity_map,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=mixed_side_levels,
            clusters=[buy_side_stop_cluster, sell_side_stop_cluster],
        )

        payload = snapshot.to_event_payload()

        assert len(payload["active_levels"]) == len(snapshot.active_levels)
        assert len(payload["equal_levels"]) == len(snapshot.equal_levels)
        assert len(payload["stop_clusters"]) == len(snapshot.stop_clusters)
        assert len(payload["zones"]) == len(snapshot.zones)

        assert payload["metadata"]["active_levels_count"] == len(snapshot.active_levels)
        assert payload["metadata"]["equal_levels_count"] == len(snapshot.equal_levels)
        assert payload["metadata"]["stop_clusters_count"] == len(snapshot.stop_clusters)
        assert payload["metadata"]["zones_count"] == len(snapshot.zones)


# ---------------------------------------------------------------------
# Adversarial / invariant tests
# ---------------------------------------------------------------------


class TestLiquidityMapAdversarialInvariants:
    def test_scores_are_clamped_under_extreme_inputs(
        self,
        liquidity_map: LiquidityMap,
        buy_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        extreme_levels: list[LiquidityLevel] = []

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

        snapshot = _build_component_snapshot(
            liquidity_map,
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

    def test_negative_and_nan_like_scores_do_not_escape_payload_contract(
        self,
        liquidity_map: LiquidityMap,
        buy_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        hostile_levels: list[LiquidityLevel] = []

        for level in buy_side_levels:
            cloned = deepcopy(level)
            cloned.confidence = -999.0
            cloned.touches_count = -50
            cloned.reaction_count = -20
            hostile_levels.append(cloned)

        hostile_cluster = deepcopy(buy_side_stop_cluster)
        hostile_cluster.confidence = -999.0
        hostile_cluster.estimated_stop_density = -999.0
        hostile_cluster.touches_count = -99

        snapshot = _build_component_snapshot(
            liquidity_map,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=hostile_levels,
            clusters=[hostile_cluster],
        )

        _assert_snapshot_contract(
            snapshot,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
        )

    def test_component_inputs_are_mutated_only_into_requested_snapshot_scope(
        self,
        liquidity_map: LiquidityMap,
        buy_side_levels: list[LiquidityLevel],
        buy_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = _clone_levels(
            buy_side_levels,
            exchange="unknown",
            market_type="perpetual",
            symbol=symbol,
            timeframe=timeframe,
        )
        clusters = _clone_clusters(
            [buy_side_stop_cluster],
            exchange="unknown",
            market_type="perpetual",
            symbol=symbol,
            timeframe=timeframe,
        )

        snapshot = liquidity_map.build_snapshot_from_components(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=levels,
            clusters=clusters,
        )

        assert snapshot.liquidity_key == _expected_key(
            symbol=symbol,
            timeframe=timeframe,
        )

        for level in levels:
            assert level.liquidity_key == snapshot.liquidity_key

        for cluster in clusters:
            assert cluster.liquidity_key == snapshot.liquidity_key
            for source_level in cluster.source_levels:
                assert source_level.liquidity_key == snapshot.liquidity_key

    def test_far_apart_same_side_clusters_remain_separate(
        self,
        liquidity_map: LiquidityMap,
        buy_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        near_cluster = _move_cluster_prices(
            buy_side_stop_cluster,
            low_price=104.90,
            high_price=105.10,
        )
        far_cluster = _move_cluster_prices(
            buy_side_stop_cluster,
            low_price=119.90,
            high_price=120.10,
        )

        snapshot = _build_component_snapshot(
            liquidity_map,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=[],
            clusters=[near_cluster, far_cluster],
        )

        assert len(snapshot.stop_clusters) == 2

    def test_opposite_side_clusters_are_never_merged_even_when_price_ranges_overlap(
        self,
        liquidity_map: LiquidityMap,
        buy_side_stop_cluster: StopCluster,
        sell_side_stop_cluster: StopCluster,
        symbol: str,
        timeframe: str,
    ) -> None:
        buy_cluster = _move_cluster_prices(
            buy_side_stop_cluster,
            low_price=99.90,
            high_price=100.10,
        )
        sell_cluster = _move_cluster_prices(
            sell_side_stop_cluster,
            low_price=99.90,
            high_price=100.10,
        )

        snapshot = _build_component_snapshot(
            liquidity_map,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=[],
            clusters=[buy_cluster, sell_cluster],
        )

        sides = {cluster.side for cluster in snapshot.stop_clusters}

        assert LiquiditySide.BUY_SIDE in sides
        assert LiquiditySide.SELL_SIDE in sides
        assert len(snapshot.stop_clusters) == 2

    def test_empty_components_still_return_valid_neutral_snapshot(
        self,
        liquidity_map: LiquidityMap,
        symbol: str,
        timeframe: str,
    ) -> None:
        snapshot = _build_component_snapshot(
            liquidity_map,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=[],
            clusters=[],
        )

        _assert_snapshot_contract(
            snapshot,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
        )

        assert snapshot.active_levels == []
        assert snapshot.equal_levels == []
        assert snapshot.stop_clusters == []
        assert snapshot.zones == []
        assert snapshot.bias == LiquidityBias.NEUTRAL
        assert snapshot.liquidity_pressure_score == pytest.approx(0.0)

    def test_build_snapshot_from_components_does_not_require_event_bus_or_scheduler(
        self,
        liquidity_config: LiquidityConfig,
        scorer: LiquidityScorer,
        buy_side_levels: list[LiquidityLevel],
        symbol: str,
        timeframe: str,
    ) -> None:
        liquidity_map = LiquidityMap(
            config=liquidity_config,
            scorer=scorer,
        )

        snapshot = _build_component_snapshot(
            liquidity_map,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
            levels=buy_side_levels,
            clusters=[],
        )

        _assert_snapshot_contract(
            snapshot,
            symbol=symbol,
            timeframe=timeframe,
            current_price=100.0,
        )