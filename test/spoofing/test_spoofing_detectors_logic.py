# tests/analytics/spoofing/test_spoofing_detectors_logic.py

from __future__ import annotations

from typing import Any

import pytest

from analytics.spoofing import (
    DetectorDecision,
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


def assert_no_infra_side_effects(mock_event_bus, mock_scheduler) -> None:
    assert mock_event_bus.subscriptions == []
    assert mock_event_bus.emitted == []
    assert mock_scheduler.interval_jobs == []


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

    Якщо цей тест падає — detector почав змішувати доменну логіку
    з integration responsibilities analyzer-а.
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
        price=99.9,
        size=2_000.0,
        mid_price=100.0,
    )
    wall = tracked_wall_factory(
        price=100.0,
        max_size=1_000.0,
        current_size=50.0,
        estimated_pulled_size=950.0,
        estimated_filled_size=10.0,
        lifetime_ms=400.0,
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
    Вразливість: один величезний wall не має псувати baseline так,
    щоб detector перестав бачити сам wall.

    Detector використовує median baseline, тому цей сценарій має пройти.
    """

    spoofing_config.wall_detection.min_wall_size_abs = 10_000.0
    spoofing_config.wall_detection.min_wall_size_ratio = 3.0
    spoofing_config.wall_detection.max_distance_from_mid_bps = 100.0
    spoofing_config.validate()

    detector = OrderbookWallDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
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
    assert result.wall_id is not None
    assert result.features.symbol == "BTCUSDT"
    assert result.features.exchange == "binance"
    assert result.features.side == SpoofingSide.BID
    assert result.features.wall_size == 2_000.0
    assert result.metadata["baseline_size"] < 10.0
    assert result.metadata["size_ratio"] > 100.0


def test_wall_detector_rejects_large_notional_when_too_far_from_mid(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    """
    Вразливість: великий рівень далеко від mid не має ставати wall-сигналом.
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


def test_wall_detector_rejects_unknown_side_zero_price_and_zero_size(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    """
    Вразливість: невалідний normalized snapshot не має проходити в detector.
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
    ]

    for snapshot in poison_levels:
        assert detector.analyze(snapshot, baseline_size=1.0) is None

    assert detector.analyze_many(poison_levels) == []


def test_wall_detector_analyze_many_filters_by_exchange_symbol_and_side(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    """
    Вразливість: detector не має змішувати рівні різних ринків.
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

    levels = [
        orderbook_snapshot_factory(
            symbol="BTCUSDT",
            exchange="binance",
            side=SpoofingSide.BID,
            price=99.9,
            size=2_000.0,
        ),
        orderbook_snapshot_factory(
            symbol="ETHUSDT",
            exchange="binance",
            side=SpoofingSide.BID,
            price=99.9,
            size=3_000.0,
        ),
        orderbook_snapshot_factory(
            symbol="BTCUSDT",
            exchange="bybit",
            side=SpoofingSide.BID,
            price=99.9,
            size=4_000.0,
        ),
        orderbook_snapshot_factory(
            symbol="BTCUSDT",
            exchange="binance",
            side=SpoofingSide.ASK,
            price=100.1,
            size=5_000.0,
        ),
    ]

    results = detector.analyze_many(
        levels,
        symbol="BTCUSDT",
        exchange="binance",
        side=SpoofingSide.BID,
    )

    assert len(results) == 1
    assert results[0].features.symbol == "BTCUSDT"
    assert results[0].features.exchange == "binance"
    assert results[0].features.side == SpoofingSide.BID


def test_wall_detector_select_top_candidates_sorts_by_strength_and_applies_limit(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    """
    Вразливість: downstream analyzer очікує, що найсильніші candidates
    не загубляться через нестабільне сортування.
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


def test_wall_detector_build_snapshot_levels_from_orderbook_survives_malformed_numeric_values(
    spoofing_config,
) -> None:
    """
    Вразливість: raw orderbook values можуть містити строки/None/негативи.
    Detector-level builder не має падати на safe_float parsing.
    """

    detector = OrderbookWallDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
    )

    levels = detector.build_snapshot_levels_from_orderbook(
        symbol="BTCUSDT",
        exchange="binance",
        bids=[
            (99.9, 1.0),
            ("bad-price", 10.0),
            (99.8, "bad-size"),
            (-1.0, 100.0),
        ],
        asks=[
            (100.1, 1.0),
            ("bad", "bad"),
            (100.2, None),
        ],
        best_bid=99.9,
        best_ask=100.1,
        sequence_id=123,
        metadata={"source": "malformed-test"},
    )

    assert len(levels) == 7
    assert all(item.symbol == "BTCUSDT" for item in levels)
    assert all(item.exchange == "binance" for item in levels)
    assert any(item.price == 0.0 for item in levels)
    assert any(item.size == 0.0 for item in levels)
    assert all(item.metadata["source"] == "malformed-test" for item in levels)


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


def test_pull_detector_rejects_old_wall_even_if_pull_ratio_is_high(
    spoofing_config,
    tracked_wall_factory,
) -> None:
    """
    Вразливість: старий wall із високим pull ratio не має автоматично
    ставати spoofing pull.
    """

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
    """
    Вразливість: якщо більшість wall була виконана, це не spoofing pull.
    """

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
    """
    Вразливість: дрібна ліквідність із високим pull ratio не має давати сигнал.
    """

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


def test_pull_detector_analyze_many_filters_market_and_sorts_results(
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

    walls = [
        tracked_wall_factory(
            wall_id="btc-weak",
            symbol="BTCUSDT",
            exchange="binance",
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
            price=100.0,
            max_size=1_000.0,
            estimated_pulled_size=990.0,
            estimated_filled_size=0.0,
            lifetime_ms=100.0,
        ),
    ]

    results = detector.analyze_many(
        walls,
        exchange="binance",
        symbol="BTCUSDT",
        current_mid_price=100.0,
    )

    assert len(results) == 2
    assert {item.wall_id for item in results} == {"btc-weak", "btc-strong"}

    scores = [item.score for item in results]
    assert scores == sorted(scores, reverse=True)


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
    Вразливість: detector не має приймати будь-який рух ціни.
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


def test_fake_liquidity_detector_analyze_many_filters_market_and_sorts_results(
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

    walls = [
        tracked_wall_factory(
            wall_id="btc-medium",
            symbol="BTCUSDT",
            exchange="binance",
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
            side=SpoofingSide.ASK,
            price=100.0,
            estimated_pulled_size=950.0,
            estimated_filled_size=0.0,
            lifetime_ms=300.0,
        ),
    ]

    results = detector.analyze_many(
        walls,
        symbol="BTCUSDT",
        exchange="binance",
        current_mid_price=101.0,
    )

    assert len(results) == 2
    assert {item.wall_id for item in results} == {"btc-medium", "btc-strong"}
    assert [item.score for item in results] == sorted(
        [item.score for item in results],
        reverse=True,
    )


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


def test_flip_pressure_detector_analyze_many_filters_market_and_sorts(
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

    walls = [
        tracked_wall_factory(
            wall_id="btc-medium",
            symbol="BTCUSDT",
            exchange="binance",
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
            side=SpoofingSide.ASK,
            price=100.0,
            estimated_pulled_size=960.0,
            estimated_filled_size=5.0,
            lifetime_ms=300.0,
        ),
    ]

    results = detector.analyze_many(
        walls,
        symbol="BTCUSDT",
        exchange="binance",
        current_mid_price=101.0,
    )

    assert len(results) == 2
    assert {item.wall_id for item in results} == {"btc-medium", "btc-strong"}
    assert [item.score for item in results] == sorted(
        [item.score for item in results],
        reverse=True,
    )


# =============================================================================
# LayeringDetector
# =============================================================================


def test_layering_detector_detects_synchronized_multi_level_bid_cluster(
    spoofing_config,
    layering_walls_factory,
    assert_positive_detector_result,
) -> None:
    """
    Вразливість: detector має бачити не один wall, а узгоджений cluster
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

    results = detector.analyze_many(
        walls,
        exchange="binance",
        symbol="BTCUSDT",
        current_mid_price=100.0,
    )

    assert len(results) == 1

    result = assert_positive_detector_result(results[0])
    assert result.detector == SpoofingComponent.LAYERING_DETECTOR
    assert result.pattern == SpoofingPattern.MULTI_LEVEL_LAYERING
    assert result.features.is_layering is True
    assert result.metadata["layers"] == 3
    assert result.metadata["synchronized_pull_ratio"] >= 0.5
    assert result.metadata["average_pull_ratio"] >= spoofing_config.pull_detection.min_pull_ratio


def test_layering_detector_rejects_cluster_with_too_few_layers(
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

    walls = layering_walls_factory(
        side=SpoofingSide.BID,
        count=2,
        step=0.03,
        size=1_000.0,
    )

    assert detector.analyze_many(
        walls,
        exchange="binance",
        symbol="BTCUSDT",
        current_mid_price=100.0,
    ) == []


def test_layering_detector_rejects_cluster_when_price_gap_breaks_layers_apart(
    spoofing_config,
    layering_walls_factory,
) -> None:
    """
    Вразливість: кілька великих рівнів далеко один від одного не мають
    перетворюватися на layering cluster.
    """

    spoofing_config.layering.min_layers = 3
    spoofing_config.layering.max_price_gap_bps_between_layers = 2.0
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
        side=SpoofingSide.BID,
        base_price=99.90,
        count=3,
        step=0.10,
        size=1_000.0,
    )

    assert detector.analyze_many(
        walls,
        exchange="binance",
        symbol="BTCUSDT",
        current_mid_price=100.0,
    ) == []


def test_layering_detector_rejects_high_fill_cluster(
    spoofing_config,
    layering_walls_factory,
) -> None:
    """
    Вразливість: якщо cluster реально виконувався, це не spoofing layering.
    """

    spoofing_config.layering.min_layers = 3
    spoofing_config.layering.max_price_gap_bps_between_layers = 20.0
    spoofing_config.layering.min_total_layer_notional = 10_000.0
    spoofing_config.fake_liquidity.max_fill_ratio = 0.20
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

    for wall in walls:
        wall.estimated_filled_size = 700.0
        wall.estimated_pulled_size = 200.0
        wall.current_size = 100.0

    assert detector.analyze_many(
        walls,
        exchange="binance",
        symbol="BTCUSDT",
        current_mid_price=100.0,
    ) == []


def test_layering_detector_rejects_cluster_without_synchronized_pull(
    spoofing_config,
    layering_walls_factory,
) -> None:
    """
    Вразливість: просто кілька великих рівнів поруч — ще не layering spoof.
    Має бути синхронне зняття/ослаблення.
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

    walls = layering_walls_factory(
        side=SpoofingSide.BID,
        base_price=99.90,
        count=3,
        step=0.03,
        size=1_000.0,
    )

    for wall in walls:
        wall.state = OrderbookWallState.ACTIVE
        wall.estimated_pulled_size = 0.0
        wall.current_size = wall.max_size

    assert detector.analyze_many(
        walls,
        exchange="binance",
        symbol="BTCUSDT",
        current_mid_price=100.0,
    ) == []


def test_layering_detector_analyze_uses_tracker_state_to_find_cluster_around_single_wall(
    spoofing_config,
    layering_walls_factory,
) -> None:
    """
    analyze(single wall) для layering є небезпечним API, бо layering —
    cluster-based detector. Цей тест перевіряє, що single-wall analyze
    не аналізує isolated wall, а бере cluster зі state tracker-а.
    """

    spoofing_config.layering.min_layers = 3
    spoofing_config.layering.max_price_gap_bps_between_layers = 20.0
    spoofing_config.layering.min_total_layer_notional = 10_000.0
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    walls = layering_walls_factory(
        side=SpoofingSide.BID,
        base_price=99.90,
        count=3,
        step=0.03,
        size=1_000.0,
    )

    for wall in walls:
        tracker._walls_by_id[wall.wall_id] = wall
        tracker._wall_ids_by_symbol[(wall.exchange, wall.symbol)].add(wall.wall_id)

    detector = LayeringDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    result = detector.analyze(
        walls[1],
        current_mid_price=100.0,
    )

    assert result is not None
    assert result.pattern == SpoofingPattern.MULTI_LEVEL_LAYERING
    assert result.metadata["layers"] == 3


def test_layering_detector_analyze_returns_none_for_isolated_wall_not_in_tracker_cluster(
    spoofing_config,
    tracked_wall_factory,
) -> None:
    tracker = make_tracker(spoofing_config)

    detector = LayeringDetector(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
        persistence_tracker=tracker,
    )

    isolated_wall = tracked_wall_factory(
        side=SpoofingSide.BID,
        price=99.9,
        max_size=1_000.0,
        estimated_pulled_size=900.0,
        estimated_filled_size=10.0,
    )

    assert detector.analyze(isolated_wall, current_mid_price=100.0) is None


def test_layering_detector_filters_exchange_symbol_and_deduplicates_clusters(
    spoofing_config,
    layering_walls_factory,
) -> None:
    """
    Вразливість: overlapping cluster scan може повернути дублікати.
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

    btc_binance = layering_walls_factory(
        symbol="BTCUSDT",
        exchange="binance",
        side=SpoofingSide.BID,
        base_price=99.90,
        count=4,
        step=0.03,
        size=1_000.0,
    )
    eth_binance = layering_walls_factory(
        symbol="ETHUSDT",
        exchange="binance",
        side=SpoofingSide.BID,
        base_price=99.90,
        count=4,
        step=0.03,
        size=1_000.0,
    )
    btc_bybit = layering_walls_factory(
        symbol="BTCUSDT",
        exchange="bybit",
        side=SpoofingSide.BID,
        base_price=99.90,
        count=4,
        step=0.03,
        size=1_000.0,
    )

    results = detector.analyze_many(
        [*btc_binance, *eth_binance, *btc_bybit],
        exchange="binance",
        symbol="BTCUSDT",
        current_mid_price=100.0,
    )

    assert len(results) == 1
    assert results[0].features.exchange == "binance"
    assert results[0].features.symbol == "BTCUSDT"
    assert results[0].metadata["layers"] == 4


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
    Вразливість: якщо config.enabled=False, detector-и не мають генерувати
    позитиви навіть при дуже сильних synthetic inputs.
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
        result = detector.analyze_many(
            layering_walls_factory(),
            exchange="binance",
            symbol="BTCUSDT",
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
            exchange="binance",
            symbol="BTCUSDT",
            current_mid_price=101.0,
        )
        assert result == []


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
    - wall_id;
    - features із symbol/exchange/side/price;
    - metadata без non-serializable об'єктів верхнього рівня.

    Це важливо, бо analyzer потім серіалізує detector results у EventBus payload.
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
        *layering_detector.analyze_many(
            layering_walls,
            exchange="binance",
            symbol="BTCUSDT",
            current_mid_price=100.0,
        ),
    ]

    positives = [assert_positive_detector_result(result) for result in results]

    for result in positives:
        assert result.detector is not None
        assert result.decision == DetectorDecision.POSITIVE
        assert result.pattern != SpoofingPattern.UNKNOWN
        assert result.wall_id is not None

        assert result.features.symbol == "BTCUSDT"
        assert result.features.exchange == "binance"
        assert result.features.side in {SpoofingSide.BID, SpoofingSide.ASK}
        assert result.features.price > 0.0

        for key, value in result.metadata.items():
            assert isinstance(key, str)
            assert isinstance(
                value,
                (str, int, float, bool, type(None)),
            ), f"Non-serializable metadata value: key={key!r}, value={value!r}"


def test_detector_threshold_boundaries_do_not_accidentally_accept_equal_or_below_threshold_noise(
    spoofing_config,
    tracked_wall_factory,
) -> None:
    """
    Вразливість: boundary умови часто створюють false positives.
    Цей тест ставить wall точно на межі або нижче ключових порогів.
    """

    spoofing_config.pull_detection.min_pull_ratio = 0.60
    spoofing_config.pull_detection.max_fill_ratio_for_pull = 0.25
    spoofing_config.pull_detection.min_removed_notional = 50_000.0
    spoofing_config.fake_liquidity.min_pull_ratio = 0.70
    spoofing_config.fake_liquidity.max_fill_ratio = 0.20
    spoofing_config.fake_liquidity.min_price_reaction_bps = 5.0
    spoofing_config.flip_pressure.min_price_reaction_bps = 5.0
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

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

    weak_boundary_wall = tracked_wall_factory(
        side=SpoofingSide.ASK,
        price=100.0,
        max_size=1_000.0,
        current_size=400.0,
        estimated_pulled_size=599.0,
        estimated_filled_size=250.0,
        lifetime_ms=2_500.0,
        state=OrderbookWallState.PULLED,
    )

    assert pull_detector.analyze(weak_boundary_wall, current_mid_price=100.0) is None
    assert fake_detector.analyze(weak_boundary_wall, current_mid_price=100.05) is None
    assert flip_detector.analyze(weak_boundary_wall, current_mid_price=100.05) is None