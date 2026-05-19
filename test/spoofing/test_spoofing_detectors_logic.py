# tests/analytics/spoofing/test_spoofing_detectors_logic.py

from __future__ import annotations

from typing import Any

import pytest

from analytics.spoofing import (
    DetectorDecision,
    DetectorResult,
    FakeLiquidityDetector,
    FlipPressureDetector,
    LayeringDetector,
    OrderPullDetector,
    OrderbookWallDetector,
    OrderbookWallState,
    PersistenceTracker,
    SpoofingComponent,
    SpoofingPattern,
    SpoofingSide,
)


# =============================================================================
# Local helpers
# =============================================================================


def make_tracker(spoofing_config) -> PersistenceTracker:
    return PersistenceTracker(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
    )


def make_key(
    detector,
    *,
    exchange: str = "binance",
    market_type: str = "perpetual",
    symbol: str = "BTCUSDT",
    timeframe: str = "realtime",
):
    return detector.make_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )


def store_wall_directly(tracker: PersistenceTracker, wall) -> None:
    """
    Test helper для analyze_key() scenarios.

    Production path створює walls через PersistenceTracker.upsert_snapshot(),
    але detector unit tests інколи потребують точного synthetic wall state.
    """
    tracker._walls_by_id[wall.wall_id] = wall
    tracker._wall_ids_by_key[wall.key].add(wall.wall_id)


def assert_no_infra_side_effects(mock_event_bus, mock_scheduler) -> None:
    assert mock_event_bus.subscriptions == []
    assert mock_event_bus.emitted == []
    assert mock_scheduler.interval_jobs == []


def assert_result_scope(result: DetectorResult, *, exchange: str, market_type: str, symbol: str, timeframe: str) -> None:
    assert result.features.exchange == exchange
    assert result.features.market_type == market_type
    assert result.features.symbol == symbol
    assert result.features.timeframe == timeframe
    assert result.metadata["scope"] == {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
    }


# =============================================================================
# Infrastructure contract: detector-и мають бути pure evaluators
# =============================================================================


def test_detectors_register_is_noop_and_does_not_touch_eventbus_or_scheduler(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
) -> None:
    """
    Detector-и не мають самостійно підписуватись на EventBus,
    публікувати події або створювати Scheduler jobs.
    """

    tracker = PersistenceTracker(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    detectors = [
        OrderbookWallDetector(
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            config=spoofing_config,
            persistence_tracker=tracker,
        ),
        OrderPullDetector(
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            config=spoofing_config,
            persistence_tracker=tracker,
        ),
        FakeLiquidityDetector(
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            config=spoofing_config,
            persistence_tracker=tracker,
        ),
        FlipPressureDetector(
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            config=spoofing_config,
            persistence_tracker=tracker,
        ),
        LayeringDetector(
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
            config=spoofing_config,
            persistence_tracker=tracker,
        ),
    ]

    for detector in detectors:
        assert detector.register() is None

    assert_no_infra_side_effects(mock_event_bus, mock_scheduler)


def test_detector_analyze_calls_do_not_emit_events_or_create_jobs(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
    orderbook_snapshot_factory,
    tracked_wall_factory,
) -> None:
    """
    Навіть positive detection не має напряму публікувати EventBus events.
    Публікація — відповідальність SpoofingAnalyzer.
    """

    tracker = PersistenceTracker(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    wall_detector = OrderbookWallDetector(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
        persistence_tracker=tracker,
    )
    pull_detector = OrderPullDetector(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    snapshot = orderbook_snapshot_factory(
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
        price=99.9,
        size=2_000.0,
        mid_price=100.0,
    )
    wall = tracked_wall_factory(
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
        price=100.0,
        max_size=1_000.0,
        current_size=50.0,
        estimated_pulled_size=950.0,
        estimated_filled_size=10.0,
        lifetime_ms=400.0,
        state=OrderbookWallState.PULLED,
    )

    wall_detector.analyze(snapshot, baseline_size=1.0)
    pull_detector.analyze(wall, current_mid_price=100.0)

    assert_no_infra_side_effects(mock_event_bus, mock_scheduler)


# =============================================================================
# OrderbookWallDetector
# =============================================================================


def test_wall_detector_detects_single_large_near_mid_wall_without_being_poisoned_by_wall_outlier(
    spoofing_config,
    orderbook_snapshot_factory,
    assert_positive_detector_result,
) -> None:
    """
    Один величезний wall не має псувати median baseline так,
    щоб detector перестав бачити сам wall.
    """

    spoofing_config.wall_detection.min_wall_size_abs = 10_000.0
    spoofing_config.wall_detection.min_wall_size_ratio = 3.0
    spoofing_config.wall_detection.max_distance_from_mid_bps = 100.0
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    detector = OrderbookWallDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    levels = [
        orderbook_snapshot_factory(
            side=SpoofingSide.BID,
            price=99.95,
            size=1.0,
            mid_price=100.0,
        ),
        orderbook_snapshot_factory(
            side=SpoofingSide.BID,
            price=99.90,
            size=1.1,
            mid_price=100.0,
        ),
        orderbook_snapshot_factory(
            side=SpoofingSide.BID,
            price=99.85,
            size=2_000.0,
            mid_price=100.0,
        ),
        orderbook_snapshot_factory(
            side=SpoofingSide.ASK,
            price=100.05,
            size=1.2,
            mid_price=100.0,
        ),
        orderbook_snapshot_factory(
            side=SpoofingSide.ASK,
            price=100.10,
            size=1.3,
            mid_price=100.0,
        ),
    ]

    results = detector.analyze_many(levels)

    assert len(results) == 1

    result = assert_positive_detector_result(results[0])
    assert result.detector == SpoofingComponent.ORDERBOOK_WALL_DETECTOR
    assert result.pattern == SpoofingPattern.SINGLE_LEVEL_SPOOF
    assert result.features.side == SpoofingSide.BID
    assert result.features.wall_size == 2_000.0
    assert result.metadata["baseline_size"] < 10.0
    assert result.metadata["size_ratio"] > 100.0
    assert_result_scope(
        result,
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
    )


def test_wall_detector_rejects_large_notional_when_too_far_from_mid(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    """
    Великий рівень далеко від mid не має ставати wall-сигналом.
    """

    spoofing_config.wall_detection.min_wall_size_abs = 10_000.0
    spoofing_config.wall_detection.min_wall_size_ratio = 2.0
    spoofing_config.wall_detection.max_distance_from_mid_bps = 20.0
    spoofing_config.validate()

    detector = OrderbookWallDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
    )

    far_snapshot = orderbook_snapshot_factory(
        side=SpoofingSide.BID,
        price=90.0,
        size=10_000.0,
        best_bid=99.9,
        best_ask=100.1,
        mid_price=100.0,
    )

    assert detector.analyze(far_snapshot, baseline_size=1.0) is None


def test_wall_detector_rejects_unknown_side_zero_price_zero_size_and_nonfinite_values(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    """
    Невалідний normalized snapshot не має проходити в detector.
    """

    detector = OrderbookWallDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
    )

    poison_levels = [
        orderbook_snapshot_factory(
            side=SpoofingSide.UNKNOWN,
            price=100.0,
            size=10_000.0,
        ),
        orderbook_snapshot_factory(
            side=SpoofingSide.BID,
            price=0.0,
            size=10_000.0,
        ),
        orderbook_snapshot_factory(
            side=SpoofingSide.ASK,
            price=100.1,
            size=0.0,
        ),
        orderbook_snapshot_factory(
            side=SpoofingSide.BID,
            price=float("inf"),
            size=10_000.0,
        ),
        orderbook_snapshot_factory(
            side=SpoofingSide.BID,
            price=99.9,
            size=float("nan"),
        ),
    ]

    for snapshot in poison_levels:
        assert detector.analyze(snapshot, baseline_size=1.0) is None

    assert detector.analyze_many(poison_levels) == []


def test_wall_detector_analyze_many_filters_by_full_key_and_side(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    """
    Detector не має змішувати рівні різних exchange/market_type/symbol/timeframe.
    """

    spoofing_config.wall_detection.min_wall_size_abs = 10_000.0
    spoofing_config.wall_detection.min_wall_size_ratio = 2.0
    spoofing_config.wall_detection.max_distance_from_mid_bps = 100.0
    spoofing_config.validate()

    detector = OrderbookWallDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
    )
    key = make_key(detector)

    levels = [
        orderbook_snapshot_factory(
            symbol="BTCUSDT",
            exchange="binance",
            market_type="perpetual",
            timeframe="realtime",
            side=SpoofingSide.BID,
            price=99.9,
            size=2_000.0,
        ),
        orderbook_snapshot_factory(
            symbol="ETHUSDT",
            exchange="binance",
            market_type="perpetual",
            timeframe="realtime",
            side=SpoofingSide.BID,
            price=99.9,
            size=3_000.0,
        ),
        orderbook_snapshot_factory(
            symbol="BTCUSDT",
            exchange="bybit",
            market_type="perpetual",
            timeframe="realtime",
            side=SpoofingSide.BID,
            price=99.9,
            size=4_000.0,
        ),
        orderbook_snapshot_factory(
            symbol="BTCUSDT",
            exchange="binance",
            market_type="linear",
            timeframe="realtime",
            side=SpoofingSide.BID,
            price=99.9,
            size=5_000.0,
        ),
        orderbook_snapshot_factory(
            symbol="BTCUSDT",
            exchange="binance",
            market_type="perpetual",
            timeframe="1m",
            side=SpoofingSide.BID,
            price=99.9,
            size=6_000.0,
        ),
        orderbook_snapshot_factory(
            symbol="BTCUSDT",
            exchange="binance",
            market_type="perpetual",
            timeframe="realtime",
            side=SpoofingSide.ASK,
            price=100.1,
            size=7_000.0,
        ),
    ]

    results = detector.analyze_many(
        levels,
        key=key,
        side=SpoofingSide.BID,
    )

    assert len(results) == 1
    assert_result_scope(
        results[0],
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
    )
    assert results[0].features.side == SpoofingSide.BID


def test_wall_detector_select_top_candidates_sorts_by_strength_and_applies_limit(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    """
    Downstream analyzer очікує, що найсильніші candidates не загубляться
    через нестабільне сортування.
    """

    spoofing_config.wall_detection.min_wall_size_abs = 1_000.0
    spoofing_config.wall_detection.min_wall_size_ratio = 1.2
    spoofing_config.wall_detection.max_distance_from_mid_bps = 200.0
    spoofing_config.validate()

    detector = OrderbookWallDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
    )

    levels = [
        orderbook_snapshot_factory(price=99.95, size=10.0),
        orderbook_snapshot_factory(price=99.90, size=20.0),
        orderbook_snapshot_factory(price=99.85, size=100.0),
        orderbook_snapshot_factory(price=99.80, size=300.0),
        orderbook_snapshot_factory(price=99.75, size=900.0),
    ]

    results = detector.select_top_candidates(levels, limit=2)

    assert len(results) == 2
    assert results[0].score >= results[1].score
    assert results[0].confidence >= 0.0
    assert results[0].features.wall_size >= results[1].features.wall_size


def test_wall_detector_build_snapshot_levels_from_orderbook_skips_malformed_numeric_values(
    spoofing_config,
) -> None:
    """
    Raw orderbook values можуть містити строки/None/NaN/inf/негативи.
    Builder не має падати і не має створювати snapshots із price/size=0.
    """

    detector = OrderbookWallDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
    )

    levels = detector.build_snapshot_levels_from_orderbook(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="perpetual",
        timeframe="realtime",
        bids=[
            (99.9, 1.0),
            ("bad-price", 10.0),
            (99.8, "bad-size"),
            (-1.0, 100.0),
            (99.7, 0.0),
            (99.6, float("inf")),
            (float("nan"), 10.0),
        ],
        asks=[
            (100.1, 1.0),
            ("bad", "bad"),
            (100.2, None),
            (float("inf"), 1.0),
            (100.3, float("-inf")),
            {"price": 100.4, "size": 2.0},
            {"price": "bad", "size": 2.0},
            {"price": 100.5, "qty": 3.0},
        ],
        best_bid=99.9,
        best_ask=100.1,
        sequence_id=123,
        metadata={"source": "malformed-test"},
    )

    assert len(levels) == 4
    assert all(item.symbol == "BTCUSDT" for item in levels)
    assert all(item.exchange == "binance" for item in levels)
    assert all(item.market_type == "perpetual" for item in levels)
    assert all(item.timeframe == "realtime" for item in levels)
    assert all(item.price > 0.0 for item in levels)
    assert all(item.size > 0.0 for item in levels)
    assert {item.side for item in levels} == {SpoofingSide.BID, SpoofingSide.ASK}
    assert all(item.metadata["source"] == "manual_or_test_helper" for item in levels)


def test_wall_detector_disabled_returns_no_results(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    spoofing_config.wall_detection.enabled = False
    spoofing_config.validate()

    detector = OrderbookWallDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
    )

    snapshot = orderbook_snapshot_factory(
        price=99.9,
        size=1_000_000.0,
    )

    assert detector.analyze_many([snapshot]) == []


# =============================================================================
# OrderPullDetector
# =============================================================================


def test_pull_detector_detects_fast_strong_unfilled_pull(
    spoofing_config,
    tracked_wall_factory,
    assert_positive_detector_result,
) -> None:
    spoofing_config.pull_detection.max_pull_lifetime_ms = 2_500
    spoofing_config.pull_detection.min_pull_ratio = 0.60
    spoofing_config.pull_detection.max_fill_ratio_for_pull = 0.25
    spoofing_config.pull_detection.min_removed_notional = 10_000.0
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    detector = OrderPullDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    wall = tracked_wall_factory(
        price=100.0,
        max_size=1_000.0,
        current_size=50.0,
        estimated_pulled_size=950.0,
        estimated_filled_size=10.0,
        lifetime_ms=400.0,
        state=OrderbookWallState.PULLED,
    )

    result = detector.analyze(
        wall,
        current_mid_price=99.5,
        repetition_count=3,
    )

    result = assert_positive_detector_result(result)

    assert result.detector == SpoofingComponent.ORDER_PULL_DETECTOR
    assert result.pattern == SpoofingPattern.PULL_AND_REVERSAL
    assert result.features.pull_ratio == pytest.approx(0.95)
    assert result.features.fill_ratio == pytest.approx(0.01)
    assert result.features.is_fast_pull is True
    assert result.metadata["is_fast_pull"] is True
    assert result.metadata["is_strong_pull"] is True
    assert result.metadata["pulled_notional"] == pytest.approx(95_000.0)
    assert_result_scope(
        result,
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
    )


def test_pull_detector_rejects_old_wall_even_if_pull_ratio_is_high(
    spoofing_config,
    tracked_wall_factory,
) -> None:
    spoofing_config.pull_detection.max_pull_lifetime_ms = 2_500
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    detector = OrderPullDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    wall = tracked_wall_factory(
        price=100.0,
        max_size=1_000.0,
        current_size=10.0,
        estimated_pulled_size=990.0,
        estimated_filled_size=0.0,
        lifetime_ms=30_000.0,
        state=OrderbookWallState.PULLED,
    )

    assert detector.analyze(wall, current_mid_price=99.0) is None


def test_pull_detector_rejects_high_fill_ratio_because_it_looks_like_real_liquidity(
    spoofing_config,
    tracked_wall_factory,
) -> None:
    spoofing_config.pull_detection.max_fill_ratio_for_pull = 0.25
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    detector = OrderPullDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    wall = tracked_wall_factory(
        price=100.0,
        max_size=1_000.0,
        current_size=50.0,
        estimated_pulled_size=900.0,
        estimated_filled_size=700.0,
        lifetime_ms=400.0,
        state=OrderbookWallState.PULLED,
    )

    assert detector.analyze(wall, current_mid_price=100.0) is None


def test_pull_detector_rejects_low_removed_notional_even_with_high_pull_ratio(
    spoofing_config,
    tracked_wall_factory,
) -> None:
    spoofing_config.pull_detection.min_removed_notional = 50_000.0
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    detector = OrderPullDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    wall = tracked_wall_factory(
        price=10.0,
        max_size=1_000.0,
        current_size=10.0,
        estimated_pulled_size=990.0,
        estimated_filled_size=0.0,
        lifetime_ms=300.0,
        state=OrderbookWallState.PULLED,
    )

    assert detector.analyze(wall, current_mid_price=10.0) is None


def test_pull_detector_analyze_many_filters_by_full_key_and_sorts_results(
    spoofing_config,
    tracked_wall_factory,
) -> None:
    spoofing_config.pull_detection.min_removed_notional = 1_000.0
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    detector = OrderPullDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )
    key = make_key(detector)

    walls = [
        tracked_wall_factory(
            wall_id="btc-weak",
            symbol="BTCUSDT",
            exchange="binance",
            market_type="perpetual",
            timeframe="realtime",
            price=100.0,
            max_size=1_000.0,
            estimated_pulled_size=650.0,
            estimated_filled_size=10.0,
            lifetime_ms=1_000.0,
        ),
        tracked_wall_factory(
            wall_id="btc-strong",
            symbol="BTCUSDT",
            exchange="binance",
            market_type="perpetual",
            timeframe="realtime",
            price=100.0,
            max_size=1_000.0,
            estimated_pulled_size=950.0,
            estimated_filled_size=5.0,
            lifetime_ms=300.0,
        ),
        tracked_wall_factory(
            wall_id="eth-should-be-filtered",
            symbol="ETHUSDT",
            exchange="binance",
            market_type="perpetual",
            timeframe="realtime",
            price=100.0,
            max_size=1_000.0,
            estimated_pulled_size=990.0,
            estimated_filled_size=0.0,
            lifetime_ms=100.0,
        ),
        tracked_wall_factory(
            wall_id="bybit-should-be-filtered",
            symbol="BTCUSDT",
            exchange="bybit",
            market_type="perpetual",
            timeframe="realtime",
            price=100.0,
            max_size=1_000.0,
            estimated_pulled_size=990.0,
            estimated_filled_size=0.0,
            lifetime_ms=100.0,
        ),
        tracked_wall_factory(
            wall_id="linear-should-be-filtered",
            symbol="BTCUSDT",
            exchange="binance",
            market_type="linear",
            timeframe="realtime",
            price=100.0,
            max_size=1_000.0,
            estimated_pulled_size=990.0,
            estimated_filled_size=0.0,
            lifetime_ms=100.0,
        ),
    ]

    results = detector.analyze_many(
        walls,
        key=key,
        current_mid_price=100.0,
    )

    assert len(results) == 2
    assert {item.wall_id for item in results} == {"btc-weak", "btc-strong"}
    assert [item.score for item in results] == sorted(
        [item.score for item in results],
        reverse=True,
    )


def test_pull_detector_analyze_key_reads_only_tracker_key_scope(
    spoofing_config,
    tracked_wall_factory,
) -> None:
    tracker = make_tracker(spoofing_config)

    detector = OrderPullDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )
    key = make_key(detector)

    target = tracked_wall_factory(
        wall_id="target-wall",
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
        estimated_pulled_size=950.0,
        estimated_filled_size=5.0,
        lifetime_ms=300.0,
    )
    wrong_key = tracked_wall_factory(
        wall_id="wrong-key-wall",
        exchange="bybit",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
        estimated_pulled_size=950.0,
        estimated_filled_size=5.0,
        lifetime_ms=300.0,
    )

    store_wall_directly(tracker, target)
    store_wall_directly(tracker, wrong_key)

    results = detector.analyze_key(key=key, current_mid_price=100.0)

    assert {item.wall_id for item in results} == {"target-wall"}


def test_pull_detector_disabled_returns_empty_result_set(
    spoofing_config,
    tracked_wall_factory,
) -> None:
    spoofing_config.pull_detection.enabled = False
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    detector = OrderPullDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    walls = [tracked_wall_factory()]
    assert detector.analyze_many(walls) == []


# =============================================================================
# FakeLiquidityDetector
# =============================================================================


def test_fake_liquidity_detector_detects_ask_wall_removed_before_upward_reaction(
    spoofing_config,
    tracked_wall_factory,
    assert_positive_detector_result,
) -> None:
    """
    ASK fake liquidity: зверху була велика ask-стіна, її зняли,
    після чого ціна пішла вгору.
    """

    spoofing_config.fake_liquidity.max_fill_ratio = 0.20
    spoofing_config.fake_liquidity.min_pull_ratio = 0.70
    spoofing_config.fake_liquidity.max_lifetime_ms = 4_000
    spoofing_config.fake_liquidity.min_price_reaction_bps = 5.0
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    detector = FakeLiquidityDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    wall = tracked_wall_factory(
        side=SpoofingSide.ASK,
        price=100.0,
        max_size=1_000.0,
        current_size=50.0,
        estimated_pulled_size=930.0,
        estimated_filled_size=20.0,
        lifetime_ms=700.0,
        state=OrderbookWallState.PULLED,
    )

    result = detector.analyze(
        wall,
        current_mid_price=101.0,
        repetition_count=4,
    )

    result = assert_positive_detector_result(result)

    assert result.detector == SpoofingComponent.FAKE_LIQUIDITY_DETECTOR
    assert result.pattern == SpoofingPattern.FAKE_ABSORPTION
    assert result.features.side == SpoofingSide.ASK
    assert result.features.is_fake_liquidity is True
    assert result.features.price_reaction_bps > spoofing_config.fake_liquidity.min_price_reaction_bps
    assert result.metadata["is_short_lived"] is True
    assert result.metadata["is_low_fill"] is True
    assert result.metadata["is_high_pull"] is True
    assert result.metadata["has_market_reaction"] is True
    assert_result_scope(
        result,
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
    )


def test_fake_liquidity_detector_detects_bid_wall_removed_before_downward_reaction(
    spoofing_config,
    tracked_wall_factory,
    assert_positive_detector_result,
) -> None:
    """
    BID fake liquidity: знизу була bid-стіна, її зняли,
    після чого ціна пішла вниз.
    """

    spoofing_config.fake_liquidity.min_price_reaction_bps = 5.0
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    detector = FakeLiquidityDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    wall = tracked_wall_factory(
        side=SpoofingSide.BID,
        price=100.0,
        max_size=1_000.0,
        current_size=50.0,
        estimated_pulled_size=930.0,
        estimated_filled_size=20.0,
        lifetime_ms=700.0,
        state=OrderbookWallState.PULLED,
    )

    result = detector.analyze(
        wall,
        current_mid_price=99.0,
        repetition_count=2,
    )

    result = assert_positive_detector_result(result)
    assert result.features.side == SpoofingSide.BID
    assert result.features.is_fake_liquidity is True


def test_fake_liquidity_detector_rejects_wrong_reaction_direction(
    spoofing_config,
    tracked_wall_factory,
) -> None:
    """
    Для ASK wall релевантна реакція — upward після pull.
    """

    tracker = make_tracker(spoofing_config)

    detector = FakeLiquidityDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    ask_wall = tracked_wall_factory(
        side=SpoofingSide.ASK,
        price=100.0,
        max_size=1_000.0,
        current_size=50.0,
        estimated_pulled_size=930.0,
        estimated_filled_size=20.0,
        lifetime_ms=700.0,
        state=OrderbookWallState.PULLED,
    )

    assert detector.analyze(ask_wall, current_mid_price=99.0) is None


def test_fake_liquidity_detector_rejects_no_price_reaction_even_if_pull_is_strong(
    spoofing_config,
    tracked_wall_factory,
) -> None:
    spoofing_config.fake_liquidity.min_price_reaction_bps = 5.0
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    detector = FakeLiquidityDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    wall = tracked_wall_factory(
        side=SpoofingSide.ASK,
        price=100.0,
        max_size=1_000.0,
        current_size=50.0,
        estimated_pulled_size=950.0,
        estimated_filled_size=0.0,
        lifetime_ms=500.0,
    )

    assert detector.analyze(wall, current_mid_price=100.01) is None


def test_fake_liquidity_detector_rejects_high_fill_ratio(
    spoofing_config,
    tracked_wall_factory,
) -> None:
    tracker = make_tracker(spoofing_config)

    detector = FakeLiquidityDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    wall = tracked_wall_factory(
        side=SpoofingSide.ASK,
        price=100.0,
        max_size=1_000.0,
        current_size=100.0,
        estimated_pulled_size=900.0,
        estimated_filled_size=500.0,
        lifetime_ms=500.0,
    )

    assert detector.analyze(wall, current_mid_price=101.0) is None


def test_fake_liquidity_detector_analyze_many_filters_by_full_key_and_sorts_results(
    spoofing_config,
    tracked_wall_factory,
) -> None:
    tracker = make_tracker(spoofing_config)

    detector = FakeLiquidityDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )
    key = make_key(detector)

    walls = [
        tracked_wall_factory(
            wall_id="btc-medium",
            symbol="BTCUSDT",
            exchange="binance",
            market_type="perpetual",
            timeframe="realtime",
            side=SpoofingSide.ASK,
            price=100.0,
            estimated_pulled_size=750.0,
            estimated_filled_size=10.0,
            lifetime_ms=1_500.0,
        ),
        tracked_wall_factory(
            wall_id="btc-strong",
            symbol="BTCUSDT",
            exchange="binance",
            market_type="perpetual",
            timeframe="realtime",
            side=SpoofingSide.ASK,
            price=100.0,
            estimated_pulled_size=950.0,
            estimated_filled_size=5.0,
            lifetime_ms=500.0,
        ),
        tracked_wall_factory(
            wall_id="eth-filtered",
            symbol="ETHUSDT",
            exchange="binance",
            market_type="perpetual",
            timeframe="realtime",
            side=SpoofingSide.ASK,
            price=100.0,
            estimated_pulled_size=950.0,
            estimated_filled_size=0.0,
            lifetime_ms=300.0,
        ),
        tracked_wall_factory(
            wall_id="linear-filtered",
            symbol="BTCUSDT",
            exchange="binance",
            market_type="linear",
            timeframe="realtime",
            side=SpoofingSide.ASK,
            price=100.0,
            estimated_pulled_size=950.0,
            estimated_filled_size=0.0,
            lifetime_ms=300.0,
        ),
    ]

    results = detector.analyze_many(
        walls,
        key=key,
        current_mid_price=101.0,
    )

    assert len(results) == 2
    assert {item.wall_id for item in results} == {"btc-medium", "btc-strong"}
    assert [item.score for item in results] == sorted(
        [item.score for item in results],
        reverse=True,
    )


def test_fake_liquidity_detector_analyze_key_reads_only_tracker_key_scope(
    spoofing_config,
    tracked_wall_factory,
) -> None:
    tracker = make_tracker(spoofing_config)

    detector = FakeLiquidityDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )
    key = make_key(detector)

    target = tracked_wall_factory(
        wall_id="target-fake-liquidity",
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
        side=SpoofingSide.ASK,
        estimated_pulled_size=950.0,
        estimated_filled_size=5.0,
        lifetime_ms=300.0,
    )
    wrong_key = tracked_wall_factory(
        wall_id="wrong-key-fake-liquidity",
        exchange="bybit",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
        side=SpoofingSide.ASK,
        estimated_pulled_size=950.0,
        estimated_filled_size=5.0,
        lifetime_ms=300.0,
    )

    store_wall_directly(tracker, target)
    store_wall_directly(tracker, wrong_key)

    results = detector.analyze_key(key=key, current_mid_price=101.0)

    assert {item.wall_id for item in results} == {"target-fake-liquidity"}


# =============================================================================
# FlipPressureDetector
# =============================================================================


def test_flip_pressure_detector_requires_current_mid_price(
    spoofing_config,
    tracked_wall_factory,
) -> None:
    tracker = make_tracker(spoofing_config)

    detector = FlipPressureDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    wall = tracked_wall_factory(
        side=SpoofingSide.ASK,
        price=100.0,
        max_size=1_000.0,
        current_size=0.0,
        estimated_pulled_size=950.0,
        estimated_filled_size=10.0,
        lifetime_ms=500.0,
    )

    assert detector.analyze(wall, current_mid_price=None) is None


def test_flip_pressure_detector_detects_ask_pressure_removed_and_price_moves_up(
    spoofing_config,
    tracked_wall_factory,
    assert_positive_detector_result,
) -> None:
    spoofing_config.flip_pressure.min_price_reaction_bps = 5.0
    spoofing_config.flip_pressure.min_pressure_flip_strength = 0.40
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    detector = FlipPressureDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    wall = tracked_wall_factory(
        side=SpoofingSide.ASK,
        price=100.0,
        max_size=1_000.0,
        current_size=20.0,
        estimated_pulled_size=960.0,
        estimated_filled_size=10.0,
        lifetime_ms=500.0,
        state=OrderbookWallState.PULLED,
    )

    result = detector.analyze(
        wall,
        current_mid_price=101.0,
        repetition_count=3,
    )

    result = assert_positive_detector_result(result)

    assert result.detector == SpoofingComponent.FLIP_PRESSURE_DETECTOR
    assert result.pattern == SpoofingPattern.PRESSURE_BLUFF
    assert result.features.side == SpoofingSide.ASK
    assert result.features.pressure_flip_strength >= spoofing_config.flip_pressure.min_pressure_flip_strength
    assert result.metadata["has_reversal"] is True
    assert result.metadata["is_pressure_removed"] is True
    assert_result_scope(
        result,
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
    )


def test_flip_pressure_detector_detects_bid_pressure_removed_and_price_moves_down(
    spoofing_config,
    tracked_wall_factory,
    assert_positive_detector_result,
) -> None:
    spoofing_config.flip_pressure.min_price_reaction_bps = 5.0
    spoofing_config.flip_pressure.min_pressure_flip_strength = 0.40
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    detector = FlipPressureDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    wall = tracked_wall_factory(
        side=SpoofingSide.BID,
        price=100.0,
        max_size=1_000.0,
        current_size=20.0,
        estimated_pulled_size=960.0,
        estimated_filled_size=10.0,
        lifetime_ms=500.0,
        state=OrderbookWallState.PULLED,
    )

    result = detector.analyze(
        wall,
        current_mid_price=99.0,
        repetition_count=2,
    )

    result = assert_positive_detector_result(result)

    assert result.features.side == SpoofingSide.BID
    assert result.features.pressure_flip_strength >= spoofing_config.flip_pressure.min_pressure_flip_strength


def test_flip_pressure_detector_rejects_wrong_direction_reversal(
    spoofing_config,
    tracked_wall_factory,
) -> None:
    tracker = make_tracker(spoofing_config)

    detector = FlipPressureDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    ask_wall = tracked_wall_factory(
        side=SpoofingSide.ASK,
        price=100.0,
        max_size=1_000.0,
        current_size=20.0,
        estimated_pulled_size=960.0,
        estimated_filled_size=10.0,
        lifetime_ms=500.0,
        state=OrderbookWallState.PULLED,
    )

    assert detector.analyze(ask_wall, current_mid_price=99.0) is None


def test_flip_pressure_detector_rejects_wall_that_was_substantially_filled(
    spoofing_config,
    tracked_wall_factory,
) -> None:
    tracker = make_tracker(spoofing_config)

    detector = FlipPressureDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    wall = tracked_wall_factory(
        side=SpoofingSide.ASK,
        price=100.0,
        max_size=1_000.0,
        current_size=20.0,
        estimated_pulled_size=960.0,
        estimated_filled_size=600.0,
        lifetime_ms=500.0,
        state=OrderbookWallState.PULLED,
    )

    assert detector.analyze(wall, current_mid_price=101.0) is None


def test_flip_pressure_detector_analyze_many_filters_by_full_key_and_sorts(
    spoofing_config,
    tracked_wall_factory,
) -> None:
    spoofing_config.flip_pressure.min_price_reaction_bps = 5.0
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    detector = FlipPressureDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )
    key = make_key(detector)

    walls = [
        tracked_wall_factory(
            wall_id="btc-medium",
            symbol="BTCUSDT",
            exchange="binance",
            market_type="perpetual",
            timeframe="realtime",
            side=SpoofingSide.ASK,
            price=100.0,
            estimated_pulled_size=750.0,
            estimated_filled_size=10.0,
            lifetime_ms=1_000.0,
        ),
        tracked_wall_factory(
            wall_id="btc-strong",
            symbol="BTCUSDT",
            exchange="binance",
            market_type="perpetual",
            timeframe="realtime",
            side=SpoofingSide.ASK,
            price=100.0,
            estimated_pulled_size=960.0,
            estimated_filled_size=5.0,
            lifetime_ms=300.0,
        ),
        tracked_wall_factory(
            wall_id="eth-filtered",
            symbol="ETHUSDT",
            exchange="binance",
            market_type="perpetual",
            timeframe="realtime",
            side=SpoofingSide.ASK,
            price=100.0,
            estimated_pulled_size=960.0,
            estimated_filled_size=5.0,
            lifetime_ms=300.0,
        ),
        tracked_wall_factory(
            wall_id="timeframe-filtered",
            symbol="BTCUSDT",
            exchange="binance",
            market_type="perpetual",
            timeframe="1m",
            side=SpoofingSide.ASK,
            price=100.0,
            estimated_pulled_size=960.0,
            estimated_filled_size=5.0,
            lifetime_ms=300.0,
        ),
    ]

    results = detector.analyze_many(
        walls,
        key=key,
        current_mid_price=101.0,
    )

    assert len(results) == 2
    assert {item.wall_id for item in results} == {"btc-medium", "btc-strong"}
    assert [item.score for item in results] == sorted(
        [item.score for item in results],
        reverse=True,
    )


def test_flip_pressure_detector_analyze_key_reads_only_tracker_key_scope(
    spoofing_config,
    tracked_wall_factory,
) -> None:
    tracker = make_tracker(spoofing_config)

    detector = FlipPressureDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )
    key = make_key(detector)

    target = tracked_wall_factory(
        wall_id="target-flip",
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
        side=SpoofingSide.ASK,
        estimated_pulled_size=960.0,
        estimated_filled_size=5.0,
        lifetime_ms=300.0,
    )
    wrong_key = tracked_wall_factory(
        wall_id="wrong-key-flip",
        exchange="binance",
        market_type="linear",
        symbol="BTCUSDT",
        timeframe="realtime",
        side=SpoofingSide.ASK,
        estimated_pulled_size=960.0,
        estimated_filled_size=5.0,
        lifetime_ms=300.0,
    )

    store_wall_directly(tracker, target)
    store_wall_directly(tracker, wrong_key)

    results = detector.analyze_key(key=key, current_mid_price=101.0)

    assert {item.wall_id for item in results} == {"target-flip"}


# =============================================================================
# LayeringDetector
# =============================================================================


def test_layering_detector_detects_synchronized_multi_level_bid_cluster(
    spoofing_config,
    layering_walls_factory,
    assert_positive_detector_result,
) -> None:
    """
    Detector має бачити не один wall, а узгоджений cluster
    із кількох близьких рівнів, які синхронно зняті.
    """

    spoofing_config.layering.min_layers = 3
    spoofing_config.layering.max_price_gap_bps_between_layers = 20.0
    spoofing_config.layering.min_total_layer_notional = 10_000.0
    spoofing_config.pull_detection.min_pull_ratio = 0.60
    spoofing_config.pull_detection.max_fill_ratio_for_pull = 0.25
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    detector = LayeringDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    walls = layering_walls_factory(
        side=SpoofingSide.BID,
        base_price=99.90,
        count=3,
        step=0.03,
        size=1_000.0,
    )

    result = detector.analyze_many(
        walls,
        key=walls[0].key,
        current_mid_price=100.0,
    )

    assert len(result) == 1
    result = assert_positive_detector_result(result[0])
    assert result.detector == SpoofingComponent.LAYERING_DETECTOR
    assert result.pattern == SpoofingPattern.MULTI_LEVEL_LAYERING
    assert result.features.side == SpoofingSide.BID
    assert result.features.is_layering is True
    assert result.metadata["layers"] == 3
    assert_result_scope(
        result,
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
    )


def test_layering_detector_detects_synchronized_multi_level_ask_cluster(
    spoofing_config,
    layering_walls_factory,
    assert_positive_detector_result,
) -> None:
    spoofing_config.layering.min_layers = 3
    spoofing_config.layering.max_price_gap_bps_between_layers = 20.0
    spoofing_config.layering.min_total_layer_notional = 10_000.0
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    detector = LayeringDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    walls = layering_walls_factory(
        side=SpoofingSide.ASK,
        base_price=100.10,
        count=3,
        step=0.03,
        size=1_000.0,
    )

    results = detector.analyze_many(
        walls,
        key=walls[0].key,
        current_mid_price=100.0,
    )

    assert len(results) == 1
    result = assert_positive_detector_result(results[0])
    assert result.features.side == SpoofingSide.ASK
    assert result.features.is_layering is True


def test_layering_detector_rejects_too_few_layers(
    spoofing_config,
    layering_walls_factory,
) -> None:
    spoofing_config.layering.min_layers = 3
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    detector = LayeringDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    walls = layering_walls_factory(count=2)

    assert detector.analyze_many(walls, key=walls[0].key, current_mid_price=100.0) == []


def test_layering_detector_rejects_cluster_with_wide_price_gaps(
    spoofing_config,
    layering_walls_factory,
) -> None:
    spoofing_config.layering.min_layers = 3
    spoofing_config.layering.max_price_gap_bps_between_layers = 2.0
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    detector = LayeringDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    walls = layering_walls_factory(
        count=3,
        step=1.0,
        size=1_000.0,
    )

    assert detector.analyze_many(walls, key=walls[0].key, current_mid_price=100.0) == []


def test_layering_detector_rejects_low_total_notional(
    spoofing_config,
    layering_walls_factory,
) -> None:
    spoofing_config.layering.min_layers = 3
    spoofing_config.layering.min_total_layer_notional = 1_000_000.0
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    detector = LayeringDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    walls = layering_walls_factory(
        count=3,
        size=10.0,
    )

    assert detector.analyze_many(walls, key=walls[0].key, current_mid_price=100.0) == []


def test_layering_detector_filters_by_full_key_and_deduplicates_clusters(
    spoofing_config,
    layering_walls_factory,
) -> None:
    """
    Overlapping cluster scan може повернути дублікати.
    Detector має deduplicate cluster keys і не змішувати markets.
    """

    spoofing_config.layering.min_layers = 3
    spoofing_config.layering.max_price_gap_bps_between_layers = 20.0
    spoofing_config.layering.min_total_layer_notional = 10_000.0
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    detector = LayeringDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )
    key = make_key(detector)

    btc_binance = layering_walls_factory(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="perpetual",
        timeframe="realtime",
        side=SpoofingSide.BID,
        base_price=99.90,
        count=4,
        step=0.03,
        size=1_000.0,
    )
    eth_binance = layering_walls_factory(
        symbol="ETHUSDT",
        exchange="binance",
        market_type="perpetual",
        timeframe="realtime",
        side=SpoofingSide.BID,
        base_price=99.90,
        count=4,
        step=0.03,
        size=1_000.0,
    )
    btc_bybit = layering_walls_factory(
        symbol="BTCUSDT",
        exchange="bybit",
        market_type="perpetual",
        timeframe="realtime",
        side=SpoofingSide.BID,
        base_price=99.90,
        count=4,
        step=0.03,
        size=1_000.0,
    )
    btc_linear = layering_walls_factory(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="linear",
        timeframe="realtime",
        side=SpoofingSide.BID,
        base_price=99.90,
        count=4,
        step=0.03,
        size=1_000.0,
    )

    results = detector.analyze_many(
        [*btc_binance, *eth_binance, *btc_bybit, *btc_linear],
        key=key,
        current_mid_price=100.0,
    )

    assert len(results) == 1
    assert_result_scope(
        results[0],
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
    )
    assert results[0].metadata["layers"] == 4


def test_layering_detector_analyze_key_reads_only_tracker_key_scope(
    spoofing_config,
    layering_walls_factory,
) -> None:
    tracker = make_tracker(spoofing_config)

    detector = LayeringDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )
    key = make_key(detector)

    target_walls = layering_walls_factory(
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
        count=3,
        size=1_000.0,
    )
    wrong_key_walls = layering_walls_factory(
        exchange="bybit",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
        count=3,
        size=1_000.0,
    )

    for wall in [*target_walls, *wrong_key_walls]:
        store_wall_directly(tracker, wall)

    results = detector.analyze_key(key=key, current_mid_price=100.0)

    assert len(results) == 1
    assert_result_scope(
        results[0],
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
    )


def test_layering_detector_disabled_returns_empty_result_set(
    spoofing_config,
    layering_walls_factory,
) -> None:
    spoofing_config.layering.enabled = False
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    detector = LayeringDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    walls = layering_walls_factory()

    assert detector.analyze_many(walls) == []


# =============================================================================
# Cross-detector edge contracts
# =============================================================================


@pytest.mark.parametrize(
    "detector_cls, config_attr",
    [
        (OrderPullDetector, "pull_detection"),
        (FakeLiquidityDetector, "fake_liquidity"),
        (FlipPressureDetector, "flip_pressure"),
        (LayeringDetector, "layering"),
    ],
)
def test_stateful_detectors_return_empty_when_global_package_disabled(
    spoofing_config,
    tracked_wall_factory,
    layering_walls_factory,
    detector_cls,
    config_attr: str,
) -> None:
    """
    Якщо config.enabled=False, detector-и не мають генерувати позитиви
    навіть при дуже сильних synthetic inputs.
    """

    spoofing_config.enabled = False
    getattr(spoofing_config, config_attr).enabled = True

    tracker = make_tracker(spoofing_config)

    detector = detector_cls(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    if detector_cls is LayeringDetector:
        walls = layering_walls_factory()
        result = detector.analyze_many(
            walls,
            key=walls[0].key,
            current_mid_price=100.0,
        )
        assert result == []
    else:
        wall = tracked_wall_factory(
            side=SpoofingSide.ASK,
            price=100.0,
            max_size=1_000.0,
            current_size=0.0,
            estimated_pulled_size=990.0,
            estimated_filled_size=0.0,
            lifetime_ms=100.0,
        )
        result = detector.analyze_many(
            [wall],
            key=wall.key,
            current_mid_price=101.0,
        )
        assert result == []


@pytest.mark.parametrize(
    "detector_cls, config_attr",
    [
        (OrderPullDetector, "pull_detection"),
        (FakeLiquidityDetector, "fake_liquidity"),
        (FlipPressureDetector, "flip_pressure"),
        (LayeringDetector, "layering"),
    ],
)
def test_stateful_detectors_return_empty_when_component_disabled(
    spoofing_config,
    tracked_wall_factory,
    layering_walls_factory,
    detector_cls,
    config_attr: str,
) -> None:
    """
    Якщо конкретний detector config disabled, він не має генерувати позитиви.
    """

    getattr(spoofing_config, config_attr).enabled = False
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    detector = detector_cls(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    if detector_cls is LayeringDetector:
        walls = layering_walls_factory()
        assert detector.analyze_many(walls, key=walls[0].key, current_mid_price=100.0) == []
    else:
        wall = tracked_wall_factory(
            side=SpoofingSide.ASK,
            price=100.0,
            max_size=1_000.0,
            current_size=0.0,
            estimated_pulled_size=990.0,
            estimated_filled_size=0.0,
            lifetime_ms=100.0,
        )
        assert detector.analyze_many([wall], key=wall.key, current_mid_price=101.0) == []


def test_all_detector_results_have_serializable_metadata_and_required_feature_identity(
    spoofing_config,
    orderbook_snapshot_factory,
    tracked_wall_factory,
    layering_walls_factory,
    assert_positive_detector_result,
) -> None:
    """
    Cross-detector contract: кожен positive DetectorResult має:
    - detector;
    - pattern;
    - features із full futures scope;
    - metadata без non-serializable об'єктів верхнього рівня.
    """

    tracker = make_tracker(spoofing_config)

    wall_detector = OrderbookWallDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )
    pull_detector = OrderPullDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )
    fake_detector = FakeLiquidityDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )
    flip_detector = FlipPressureDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )
    layering_detector = LayeringDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    wall_snapshot = orderbook_snapshot_factory(
        side=SpoofingSide.BID,
        price=99.9,
        size=2_000.0,
        mid_price=100.0,
    )
    pulled_ask_wall = tracked_wall_factory(
        side=SpoofingSide.ASK,
        price=100.0,
        max_size=1_000.0,
        current_size=20.0,
        estimated_pulled_size=960.0,
        estimated_filled_size=10.0,
        lifetime_ms=400.0,
        state=OrderbookWallState.PULLED,
    )
    pulled_bid_wall = tracked_wall_factory(
        wall_id="pulled-bid-wall",
        side=SpoofingSide.BID,
        price=100.0,
        max_size=1_000.0,
        current_size=20.0,
        estimated_pulled_size=960.0,
        estimated_filled_size=10.0,
        lifetime_ms=400.0,
        state=OrderbookWallState.PULLED,
    )
    layering_walls = layering_walls_factory(
        side=SpoofingSide.BID,
        base_price=99.90,
        count=3,
        step=0.03,
        size=1_000.0,
    )

    results = [
        wall_detector.analyze(wall_snapshot, baseline_size=1.0),
        pull_detector.analyze(pulled_ask_wall, current_mid_price=101.0),
        fake_detector.analyze(pulled_ask_wall, current_mid_price=101.0),
        flip_detector.analyze(pulled_ask_wall, current_mid_price=101.0),
        layering_detector.analyze_many(layering_walls, key=layering_walls[0].key, current_mid_price=100.0)[0],
        fake_detector.analyze(pulled_bid_wall, current_mid_price=99.0),
        flip_detector.analyze(pulled_bid_wall, current_mid_price=99.0),
    ]

    positive_results = [assert_positive_detector_result(item) for item in results]

    for result in positive_results:
        assert result.detector is not None
        assert result.pattern is not None
        assert result.features.exchange == "binance"
        assert result.features.market_type == "perpetual"
        assert result.features.symbol == "BTCUSDT"
        assert result.features.timeframe == "realtime"
        assert result.features.side in {SpoofingSide.BID, SpoofingSide.ASK}
        assert result.features.price > 0.0
        assert isinstance(result.metadata, dict)

        for key, value in result.metadata.items():
            assert isinstance(key, str)
            assert not callable(value)


def test_key_first_and_legacy_filters_return_same_results_for_same_scope(
    spoofing_config,
    tracked_wall_factory,
) -> None:
    """
    Поки legacy filters існують для міграції, key-first і legacy filters
    мають давати однаковий результат для одного scope.
    """

    tracker = make_tracker(spoofing_config)

    detector = OrderPullDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )
    key = make_key(detector)

    walls = [
        tracked_wall_factory(
            wall_id="target",
            exchange="binance",
            market_type="perpetual",
            symbol="BTCUSDT",
            timeframe="realtime",
            estimated_pulled_size=950.0,
            estimated_filled_size=5.0,
            lifetime_ms=300.0,
        ),
        tracked_wall_factory(
            wall_id="wrong-timeframe",
            exchange="binance",
            market_type="perpetual",
            symbol="BTCUSDT",
            timeframe="1m",
            estimated_pulled_size=950.0,
            estimated_filled_size=5.0,
            lifetime_ms=300.0,
        ),
    ]

    key_results = detector.analyze_many(
        walls,
        key=key,
        current_mid_price=100.0,
    )
    legacy_results = detector.analyze_many(
        walls,
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
        current_mid_price=100.0,
    )

    assert [item.wall_id for item in key_results] == [item.wall_id for item in legacy_results]
    assert {item.wall_id for item in key_results} == {"target"}


def test_detectors_tolerate_empty_inputs_without_side_effects(
    spoofing_config,
    mock_event_bus,
    mock_scheduler,
) -> None:
    tracker = PersistenceTracker(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
    )

    wall_detector = OrderbookWallDetector(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
        persistence_tracker=tracker,
    )
    pull_detector = OrderPullDetector(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
        persistence_tracker=tracker,
    )
    fake_detector = FakeLiquidityDetector(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
        persistence_tracker=tracker,
    )
    flip_detector = FlipPressureDetector(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
        persistence_tracker=tracker,
    )
    layering_detector = LayeringDetector(
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    assert wall_detector.analyze_many([]) == []
    assert pull_detector.analyze_many([]) == []
    assert fake_detector.analyze_many([]) == []
    assert flip_detector.analyze_many([]) == []
    assert layering_detector.analyze_many([]) == []

    assert_no_infra_side_effects(mock_event_bus, mock_scheduler)