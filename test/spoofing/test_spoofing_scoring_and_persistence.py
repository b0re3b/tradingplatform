# tests/analytics/spoofing/test_spoofing_scoring_and_persistence.py

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import Any

import pytest

from analytics.spoofing import (
    DetectorDecision,
    DetectorResult,
    OrderbookWallState,
    PersistenceTracker,
    SpoofingComponent,
    SpoofingPattern,
    SpoofingScoreEngine,
    SpoofingSeverity,
    SpoofingSide,
    SpoofingStatus,
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


def make_score_engine(spoofing_config) -> SpoofingScoreEngine:
    return SpoofingScoreEngine(
        event_bus=None,
        scheduler=None,
        config=spoofing_config,
    )


def store_wall_directly(tracker: PersistenceTracker, wall) -> None:
    """
    Test helper для scenarios, де треба напряму підкласти synthetic wall
    у state tracker-а.

    У production це робить upsert_snapshot(), але для cleanup/limit/history
    tests інколи потрібен точний контроль timestamp/state.
    """

    tracker._walls_by_id[wall.wall_id] = wall
    tracker._wall_ids_by_symbol[(wall.exchange, wall.symbol)].add(wall.wall_id)


# =============================================================================
# PersistenceTracker: creation / update / lifecycle
# =============================================================================


def test_persistence_tracker_creates_wall_with_indexes_and_create_lifecycle_event(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    """
    Вразливість: wall може бути створений, але не потрапити в symbol index
    або не мати lifecycle history. Тоді detector-и не знайдуть його через
    analyze_symbol()/get_walls_for_symbol().
    """

    tracker = make_tracker(spoofing_config)

    snapshot = orderbook_snapshot_factory(
        exchange="binance",
        symbol="BTCUSDT",
        side=SpoofingSide.BID,
        price=99.9,
        size=1_000.0,
        mid_price=100.0,
    )

    wall, events = tracker.upsert_snapshot(snapshot)

    assert wall.wall_id
    assert wall.exchange == "binance"
    assert wall.symbol == "BTCUSDT"
    assert wall.side == SpoofingSide.BID
    assert wall.price == 99.9

    assert wall.initial_size == 1_000.0
    assert wall.current_size == 1_000.0
    assert wall.max_size == 1_000.0
    assert wall.min_size == 1_000.0
    assert wall.state == OrderbookWallState.ACTIVE

    assert events
    assert tracker.get_wall(wall.wall_id) is wall
    assert tracker.get_wall_by_level(
        exchange="binance",
        symbol="BTCUSDT",
        side=SpoofingSide.BID,
        price=99.9,
    ) is wall

    symbol_walls = tracker.get_walls_for_symbol(
        exchange="binance",
        symbol="BTCUSDT",
    )
    assert [item.wall_id for item in symbol_walls] == [wall.wall_id]

    history = tracker.get_recent_history(
        exchange="binance",
        symbol="BTCUSDT",
        side=SpoofingSide.BID,
        price=99.9,
    )
    assert history
    assert history[-1].wall_id == wall.wall_id


def test_persistence_tracker_updates_existing_wall_instead_of_creating_duplicate(
    spoofing_config,
    orderbook_snapshot_factory,
    fixed_now,
) -> None:
    """
    Вразливість: той самий price level може створити дублікати wall_id при
    repeated snapshots, що зламає pull/fill lifecycle і symbol limits.
    """

    tracker = make_tracker(spoofing_config)

    first = orderbook_snapshot_factory(
        price=99.9,
        size=1_000.0,
        timestamp=fixed_now,
    )
    second = orderbook_snapshot_factory(
        price=99.9,
        size=1_500.0,
        timestamp=fixed_now + timedelta(milliseconds=100),
    )
    third = orderbook_snapshot_factory(
        price=99.9,
        size=400.0,
        timestamp=fixed_now + timedelta(milliseconds=200),
    )

    wall_1, events_1 = tracker.upsert_snapshot(first)
    wall_2, events_2 = tracker.upsert_snapshot(second)
    wall_3, events_3 = tracker.upsert_snapshot(third)

    assert wall_1 is wall_2 is wall_3
    assert len(tracker.snapshot_state()) == 1

    assert wall_3.initial_size == 1_000.0
    assert wall_3.current_size == 400.0
    assert wall_3.max_size == 1_500.0
    assert wall_3.min_size == 400.0
    assert wall_3.updates_count >= 2
    assert wall_3.total_added_size > 0.0
    assert wall_3.total_removed_size > 0.0

    assert events_1
    assert events_2
    assert events_3

    history = tracker.get_recent_history(
        exchange=wall_3.exchange,
        symbol=wall_3.symbol,
        side=wall_3.side,
        price=wall_3.price,
        limit=100,
    )
    assert len(history) >= 3


def test_persistence_tracker_price_rounding_prevents_duplicate_wall_ids_from_float_noise(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    """
    Вразливість: float-noise у price може створювати різні wall_id для того
    самого рівня. price_rounding_decimals має це стабілізувати.
    """

    spoofing_config.persistence.price_rounding_decimals = 4
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    first = orderbook_snapshot_factory(price=100.123456, size=1_000.0)
    second = orderbook_snapshot_factory(price=100.123459, size=900.0)

    wall_1, _ = tracker.upsert_snapshot(first)
    wall_2, _ = tracker.upsert_snapshot(second)

    assert wall_1.wall_id == wall_2.wall_id
    assert len(tracker.snapshot_state()) == 1


def test_persistence_tracker_very_close_prices_are_separate_when_rounding_allows_it(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    """
    Контр-тест до попереднього: tracker не має агресивно merge-ити
    distinct price levels, якщо config дозволяє більшу точність.
    """

    spoofing_config.persistence.price_rounding_decimals = 8
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    first = orderbook_snapshot_factory(price=100.12345601, size=1_000.0)
    second = orderbook_snapshot_factory(price=100.12345609, size=900.0)

    wall_1, _ = tracker.upsert_snapshot(first)
    wall_2, _ = tracker.upsert_snapshot(second)

    assert wall_1.wall_id != wall_2.wall_id
    assert len(tracker.snapshot_state()) == 2


def test_persistence_tracker_mark_pulled_updates_state_ratios_and_history(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    """
    Вразливість: mark_pulled має синхронно оновити current_size,
    estimated_pulled_size, total_removed_size, state і lifecycle history.
    """

    tracker = make_tracker(spoofing_config)

    snapshot = orderbook_snapshot_factory(
        side=SpoofingSide.ASK,
        price=100.1,
        size=1_000.0,
    )
    wall, _ = tracker.upsert_snapshot(snapshot)

    pulled_wall, event = tracker.mark_pulled(
        exchange=wall.exchange,
        symbol=wall.symbol,
        side=wall.side,
        price=wall.price,
        removed_size=850.0,
        metadata={"reason": "test-pull"},
    )

    assert pulled_wall is wall
    assert event is not None

    assert wall.current_size == pytest.approx(150.0)
    assert wall.estimated_pulled_size == pytest.approx(850.0)
    assert wall.total_removed_size >= 850.0
    assert wall.pull_ratio == pytest.approx(0.85)
    assert wall.state == OrderbookWallState.PULLED

    history = tracker.get_recent_history(
        exchange=wall.exchange,
        symbol=wall.symbol,
        side=wall.side,
        price=wall.price,
        limit=20,
    )
    assert history[-1].wall_id == wall.wall_id
    assert history[-1].metadata["reason"] == "test-pull"


def test_persistence_tracker_mark_pulled_unknown_level_is_noop(
    spoofing_config,
) -> None:
    """
    Unknown pull event не має створювати phantom wall.
    """

    tracker = make_tracker(spoofing_config)

    wall, event = tracker.mark_pulled(
        exchange="binance",
        symbol="BTCUSDT",
        side=SpoofingSide.BID,
        price=99.9,
        removed_size=1_000.0,
    )

    assert wall is None
    assert event is None
    assert tracker.snapshot_state() == []


def test_persistence_tracker_mark_pulled_clamps_removed_size_and_never_goes_negative(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    """
    Вразливість: removed_size більший за current_size не має робити
    current_size від'ємним.
    """

    tracker = make_tracker(spoofing_config)

    wall, _ = tracker.upsert_snapshot(
        orderbook_snapshot_factory(
            price=99.9,
            size=100.0,
        )
    )

    pulled_wall, _ = tracker.mark_pulled(
        exchange=wall.exchange,
        symbol=wall.symbol,
        side=wall.side,
        price=wall.price,
        removed_size=10_000.0,
    )

    assert pulled_wall is wall
    assert wall.current_size == 0.0
    assert wall.estimated_pulled_size == pytest.approx(100.0)
    assert wall.pull_ratio == pytest.approx(1.0)


def test_persistence_tracker_mark_pulled_negative_removed_size_is_safe_noop_delta(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    """
    Вразливість: negative removed_size не має збільшувати liquidity
    або ламати state.
    """

    tracker = make_tracker(spoofing_config)

    wall, _ = tracker.upsert_snapshot(
        orderbook_snapshot_factory(
            price=99.9,
            size=100.0,
        )
    )

    pulled_wall, event = tracker.mark_pulled(
        exchange=wall.exchange,
        symbol=wall.symbol,
        side=wall.side,
        price=wall.price,
        removed_size=-50.0,
    )

    assert pulled_wall is wall
    assert event is not None
    assert wall.current_size == pytest.approx(100.0)
    assert wall.estimated_pulled_size == pytest.approx(0.0)
    assert wall.pull_ratio == pytest.approx(0.0)


def test_persistence_tracker_upsert_many_keeps_event_order_and_returns_all_walls(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    tracker = make_tracker(spoofing_config)

    snapshots = [
        orderbook_snapshot_factory(price=99.9, size=1_000.0),
        orderbook_snapshot_factory(price=99.8, size=1_100.0),
        orderbook_snapshot_factory(price=99.7, size=1_200.0),
    ]

    walls, events = tracker.upsert_many(snapshots)

    assert len(walls) == 3
    assert len(events) >= 3
    assert [wall.price for wall in walls] == [99.9, 99.8, 99.7]

    state = tracker.snapshot_state()
    assert len(state) == 3


# =============================================================================
# PersistenceTracker: indexes / limits / cleanup / snapshots
# =============================================================================


def test_persistence_tracker_enforces_max_walls_per_symbol_without_cross_symbol_deletion(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    """
    Вразливість: symbol limit не має видаляти walls з іншого symbol/exchange.
    """

    spoofing_config.persistence.max_walls_per_symbol = 2
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    for price in [99.9, 99.8, 99.7]:
        tracker.upsert_snapshot(
            orderbook_snapshot_factory(
                symbol="BTCUSDT",
                exchange="binance",
                price=price,
                size=1_000.0,
            )
        )

    tracker.upsert_snapshot(
        orderbook_snapshot_factory(
            symbol="ETHUSDT",
            exchange="binance",
            price=2_000.0,
            size=1_000.0,
        )
    )

    btc_walls = tracker.get_walls_for_symbol(
        exchange="binance",
        symbol="BTCUSDT",
    )
    eth_walls = tracker.get_walls_for_symbol(
        exchange="binance",
        symbol="ETHUSDT",
    )

    assert len(btc_walls) == 2
    assert len(eth_walls) == 1
    assert all(wall.symbol == "BTCUSDT" for wall in btc_walls)
    assert eth_walls[0].symbol == "ETHUSDT"


def test_persistence_tracker_get_walls_for_symbol_filters_side_and_state(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    tracker = make_tracker(spoofing_config)

    bid_wall, _ = tracker.upsert_snapshot(
        orderbook_snapshot_factory(
            side=SpoofingSide.BID,
            price=99.9,
            size=1_000.0,
        )
    )
    ask_wall, _ = tracker.upsert_snapshot(
        orderbook_snapshot_factory(
            side=SpoofingSide.ASK,
            price=100.1,
            size=1_000.0,
        )
    )

    tracker.mark_pulled(
        exchange=ask_wall.exchange,
        symbol=ask_wall.symbol,
        side=ask_wall.side,
        price=ask_wall.price,
        removed_size=900.0,
    )

    bid_only = tracker.get_walls_for_symbol(
        exchange="binance",
        symbol="BTCUSDT",
        side=SpoofingSide.BID,
    )
    pulled_only = tracker.get_walls_for_symbol(
        exchange="binance",
        symbol="BTCUSDT",
        state=OrderbookWallState.PULLED,
    )

    assert [wall.wall_id for wall in bid_only] == [bid_wall.wall_id]
    assert [wall.wall_id for wall in pulled_only] == [ask_wall.wall_id]


def test_persistence_tracker_snapshot_state_returns_copies_not_mutable_internal_references(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    """
    Вразливість: зовнішній код не має змінити internal tracker state через
    snapshot_state().
    """

    tracker = make_tracker(spoofing_config)

    original_wall, _ = tracker.upsert_snapshot(
        orderbook_snapshot_factory(
            price=99.9,
            size=1_000.0,
        )
    )

    snapshot = tracker.snapshot_state()
    assert len(snapshot) == 1
    assert snapshot[0] is not original_wall

    snapshot[0].current_size = -999.0
    snapshot[0].state = OrderbookWallState.FILLED

    internal_wall = tracker.get_wall(original_wall.wall_id)

    assert internal_wall is original_wall
    assert internal_wall.current_size == 1_000.0
    assert internal_wall.state == OrderbookWallState.ACTIVE


def test_persistence_tracker_get_recent_history_respects_limit_and_never_returns_negative_slice(
    spoofing_config,
    orderbook_snapshot_factory,
    fixed_now,
) -> None:
    tracker = make_tracker(spoofing_config)

    wall, _ = tracker.upsert_snapshot(
        orderbook_snapshot_factory(
            price=99.9,
            size=1_000.0,
            timestamp=fixed_now,
        )
    )

    for index in range(10):
        tracker.upsert_snapshot(
            orderbook_snapshot_factory(
                price=99.9,
                size=1_000.0 + index,
                timestamp=fixed_now + timedelta(milliseconds=index + 1),
            )
        )

    history_limit_3 = tracker.get_recent_history(
        exchange=wall.exchange,
        symbol=wall.symbol,
        side=wall.side,
        price=wall.price,
        limit=3,
    )
    history_limit_0 = tracker.get_recent_history(
        exchange=wall.exchange,
        symbol=wall.symbol,
        side=wall.side,
        price=wall.price,
        limit=0,
    )
    history_limit_negative = tracker.get_recent_history(
        exchange=wall.exchange,
        symbol=wall.symbol,
        side=wall.side,
        price=wall.price,
        limit=-5,
    )

    assert len(history_limit_3) == 3
    assert history_limit_0 == []
    assert history_limit_negative == []


def test_persistence_tracker_cleanup_removes_expired_walls_and_indexes(
    spoofing_config,
    tracked_wall_factory,
    fixed_now,
) -> None:
    """
    Вразливість: cleanup може видалити wall з _walls_by_id, але лишити
    stale wall_id у _wall_ids_by_symbol.
    """

    spoofing_config.persistence.wall_ttl_ms = 1_000
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    expired = tracked_wall_factory(
        wall_id="expired-wall",
        price=99.9,
        lifetime_ms=10_000.0,
    )
    expired.last_seen_at = fixed_now - timedelta(milliseconds=10_000)

    active = tracked_wall_factory(
        wall_id="active-wall",
        price=99.8,
        lifetime_ms=100.0,
    )
    active.last_seen_at = fixed_now

    store_wall_directly(tracker, expired)
    store_wall_directly(tracker, active)

    removed_count = tracker.cleanup_expired(now=fixed_now)

    assert removed_count == 1
    assert tracker.get_wall("expired-wall") is None
    assert tracker.get_wall("active-wall") is active

    symbol_walls = tracker.get_walls_for_symbol(
        exchange="binance",
        symbol="BTCUSDT",
    )
    assert [wall.wall_id for wall in symbol_walls] == ["active-wall"]


def test_persistence_tracker_cleanup_disabled_is_noop(
    spoofing_config,
    tracked_wall_factory,
    fixed_now,
) -> None:
    spoofing_config.persistence.enabled = False

    tracker = make_tracker(spoofing_config)

    expired = tracked_wall_factory(
        wall_id="expired-wall",
        price=99.9,
    )
    expired.last_seen_at = fixed_now - timedelta(milliseconds=999_999)

    store_wall_directly(tracker, expired)

    assert tracker.cleanup_expired(now=fixed_now) == 0
    assert tracker.get_wall("expired-wall") is expired


def test_persistence_tracker_stats_reflect_state_history_and_cleanup_time(
    spoofing_config,
    orderbook_snapshot_factory,
    fixed_now,
) -> None:
    tracker = make_tracker(spoofing_config)

    wall, _ = tracker.upsert_snapshot(
        orderbook_snapshot_factory(price=99.9, size=1_000.0)
    )
    tracker.mark_pulled(
        exchange=wall.exchange,
        symbol=wall.symbol,
        side=wall.side,
        price=wall.price,
        removed_size=900.0,
    )
    tracker.cleanup_expired(now=fixed_now)

    stats = tracker.stats()

    assert stats["tracked_walls"] == 1
    assert stats["symbols"] == 1
    assert stats["history_levels"] >= 1
    assert stats["last_cleanup_at"] is not None


# =============================================================================
# SpoofingScoreEngine: filtering / no-signal cases
# =============================================================================


def test_score_engine_returns_none_when_disabled(
    spoofing_config,
    detector_result_factory,
) -> None:
    spoofing_config.scoring.enabled = False

    engine = make_score_engine(spoofing_config)

    result = detector_result_factory(score=1.0, confidence=1.0)

    assert engine.score([result], symbol="BTCUSDT", exchange="binance") is None
    assert engine.build_signal([result], symbol="BTCUSDT", exchange="binance") is None


def test_score_engine_returns_none_for_empty_or_all_negative_results(
    spoofing_config,
    detector_result_factory,
) -> None:
    engine = make_score_engine(spoofing_config)

    negative = detector_result_factory(
        decision=DetectorDecision.NEGATIVE,
        score=0.99,
        confidence=0.99,
    )

    assert engine.score([], symbol="BTCUSDT", exchange="binance") is None
    assert engine.score([negative], symbol="BTCUSDT", exchange="binance") is None


def test_score_engine_filters_results_without_features_and_wrong_market(
    spoofing_config,
    detector_result_factory,
) -> None:
    """
    Вразливість: result із wrong symbol/exchange або features=None не має
    вплинути на score.
    """

    engine = make_score_engine(spoofing_config)

    wrong_symbol = detector_result_factory(
        symbol="ETHUSDT",
        exchange="binance",
        score=1.0,
        confidence=1.0,
    )
    wrong_exchange = detector_result_factory(
        symbol="BTCUSDT",
        exchange="bybit",
        score=1.0,
        confidence=1.0,
    )
    no_features = detector_result_factory(
        score=1.0,
        confidence=1.0,
        features=None,
    )
    no_features.features = None

    assert engine.score(
        [wrong_symbol, wrong_exchange, no_features],
        symbol="BTCUSDT",
        exchange="binance",
    ) is None


def test_score_engine_uses_only_positive_results_from_requested_market(
    spoofing_config,
    detector_result_factory,
) -> None:
    engine = make_score_engine(spoofing_config)

    valid = detector_result_factory(
        symbol="BTCUSDT",
        exchange="binance",
        score=0.7,
        confidence=0.7,
        pull_ratio=0.8,
        fill_ratio=0.05,
    )
    poison_wrong_market = detector_result_factory(
        symbol="ETHUSDT",
        exchange="binance",
        score=1.0,
        confidence=1.0,
        pull_ratio=1.0,
        fill_ratio=0.0,
    )
    poison_negative = detector_result_factory(
        symbol="BTCUSDT",
        exchange="binance",
        decision=DetectorDecision.NEGATIVE,
        score=1.0,
        confidence=1.0,
    )

    score = engine.score(
        [valid, poison_wrong_market, poison_negative],
        symbol="BTCUSDT",
        exchange="binance",
    )

    assert score is not None
    assert score.metadata["detector_count"] == 1
    assert score.metadata["symbol"] == "BTCUSDT"
    assert score.metadata["exchange"] == "binance"


# =============================================================================
# SpoofingScoreEngine: score / confidence / severity
# =============================================================================


def test_score_engine_builds_score_with_clamped_score_confidence_and_contributions(
    spoofing_config,
    detector_result_factory,
) -> None:
    """
    Вразливість: extreme feature values не мають давати score/confidence > 1.
    """

    spoofing_config.scoring.detection_threshold = 0.10
    spoofing_config.validate()

    engine = make_score_engine(spoofing_config)

    extreme = detector_result_factory(
        score=10.0,
        confidence=10.0,
        wall_size_ratio=999_999.0,
        pull_ratio=99.0,
        fill_ratio=-10.0,
        price_reaction_bps=999_999.0,
        pressure_flip_strength=99.0,
        layering_score=99.0,
    )

    score = engine.score(
        [extreme],
        symbol="BTCUSDT",
        exchange="binance",
    )

    assert score is not None
    assert 0.0 <= score.total_score <= 1.0
    assert 0.0 <= score.confidence <= spoofing_config.scoring.max_confidence
    assert score.contributions
    assert all(0.0 <= item.value <= 1.0 for item in score.contributions)
    assert all(item.weight >= 0.0 for item in score.contributions)


@pytest.mark.parametrize(
    "total_score, expected_severity",
    [
        (0.10, SpoofingSeverity.LOW),
        (0.50, SpoofingSeverity.MEDIUM),
        (0.80, SpoofingSeverity.HIGH),
        (0.95, SpoofingSeverity.CRITICAL),
    ],
)
def test_score_engine_resolves_severity_boundaries(
    spoofing_config,
    detector_result_factory,
    total_score: float,
    expected_severity: SpoofingSeverity,
) -> None:
    """
    Тут не monkeypatch-имо public API, а напряму тестуємо internal resolver,
    бо severity boundary bugs дуже дорогі для alerting/risk.
    """

    spoofing_config.scoring.detection_threshold = 0.30
    spoofing_config.scoring.high_severity_threshold = 0.75
    spoofing_config.scoring.critical_severity_threshold = 0.90
    spoofing_config.validate()

    engine = make_score_engine(spoofing_config)

    assert engine._resolve_severity(total_score) == expected_severity


def test_score_engine_passed_flag_uses_detection_threshold(
    spoofing_config,
    detector_result_factory,
) -> None:
    spoofing_config.scoring.detection_threshold = 0.95
    spoofing_config.validate()

    engine = make_score_engine(spoofing_config)

    weak_but_positive = detector_result_factory(
        score=0.30,
        confidence=0.60,
        wall_size_ratio=2.0,
        pull_ratio=0.30,
        fill_ratio=0.30,
        price_reaction_bps=1.0,
    )

    score = engine.score(
        [weak_but_positive],
        symbol="BTCUSDT",
        exchange="binance",
    )

    assert score is not None
    assert score.threshold == 0.95
    assert score.passed is False
    assert score.total_score < score.threshold


def test_score_engine_confidence_boosts_on_detector_agreement_but_stays_clamped(
    spoofing_config,
    detector_result_factory,
) -> None:
    spoofing_config.scoring.confidence_boost_on_detector_agreement = 0.25
    spoofing_config.scoring.max_confidence = 0.90
    spoofing_config.validate()

    engine = make_score_engine(spoofing_config)

    results = [
        detector_result_factory(
            detector=SpoofingComponent.ORDERBOOK_WALL_DETECTOR,
            score=0.9,
            confidence=0.95,
        ),
        detector_result_factory(
            detector=SpoofingComponent.ORDER_PULL_DETECTOR,
            score=0.9,
            confidence=0.95,
        ),
        detector_result_factory(
            detector=SpoofingComponent.FAKE_LIQUIDITY_DETECTOR,
            score=0.9,
            confidence=0.95,
        ),
    ]

    score = engine.score(
        results,
        symbol="BTCUSDT",
        exchange="binance",
    )

    assert score is not None
    assert score.metadata["detector_count"] == 3
    assert score.metadata["agreement_ratio"] > 0.0
    assert score.confidence <= spoofing_config.scoring.max_confidence


def test_score_engine_merge_features_keeps_strongest_values_from_conflicting_detectors(
    spoofing_config,
    detector_result_factory,
) -> None:
    """
    Вразливість: якщо detector-и дають різні feature values, aggregator не має
    обнулити сильні сигнали слабшим result-ом.
    """

    engine = make_score_engine(spoofing_config)

    weak = detector_result_factory(
        detector=SpoofingComponent.ORDERBOOK_WALL_DETECTOR,
        score=0.4,
        confidence=0.5,
        wall_size_ratio=2.0,
        pull_ratio=0.1,
        fill_ratio=0.2,
        price_reaction_bps=1.0,
        pressure_flip_strength=0.0,
        layering_score=0.0,
    )
    strong = detector_result_factory(
        detector=SpoofingComponent.FLIP_PRESSURE_DETECTOR,
        score=0.9,
        confidence=0.9,
        wall_size_ratio=10.0,
        pull_ratio=0.95,
        fill_ratio=0.02,
        price_reaction_bps=12.0,
        pressure_flip_strength=0.9,
        layering_score=0.0,
    )

    signal = engine.build_signal(
        [weak, strong],
        symbol="BTCUSDT",
        exchange="binance",
    )

    assert signal is not None
    assert signal.features.wall_size_ratio >= 10.0
    assert signal.features.pull_ratio >= 0.95
    assert signal.features.fill_ratio <= 0.02
    assert signal.features.price_reaction_bps >= 12.0
    assert signal.features.pressure_flip_strength >= 0.9
    assert signal.metadata["detector_count"] == 2


def test_score_engine_primary_pattern_prefers_stronger_detector_result(
    spoofing_config,
    detector_result_factory,
) -> None:
    engine = make_score_engine(spoofing_config)

    weak_wall = detector_result_factory(
        detector=SpoofingComponent.ORDERBOOK_WALL_DETECTOR,
        pattern=SpoofingPattern.SINGLE_LEVEL_SPOOF,
        score=0.30,
        confidence=0.50,
    )
    strong_flip = detector_result_factory(
        detector=SpoofingComponent.FLIP_PRESSURE_DETECTOR,
        pattern=SpoofingPattern.PRESSURE_BLUFF,
        score=0.95,
        confidence=0.90,
        pressure_flip_strength=0.9,
        price_reaction_bps=15.0,
    )

    signal = engine.build_signal(
        [weak_wall, strong_flip],
        symbol="BTCUSDT",
        exchange="binance",
    )

    assert signal is not None
    assert signal.pattern == SpoofingPattern.PRESSURE_BLUFF


# =============================================================================
# SpoofingScoreEngine: build_signal / signal stability
# =============================================================================


def test_score_engine_build_signal_contains_score_breakdown_detector_results_and_metadata(
    spoofing_config,
    detector_result_factory,
) -> None:
    spoofing_config.scoring.detection_threshold = 0.10
    spoofing_config.validate()

    engine = make_score_engine(spoofing_config)

    result = detector_result_factory(
        detector=SpoofingComponent.ORDER_PULL_DETECTOR,
        pattern=SpoofingPattern.PULL_AND_REVERSAL,
        wall_id="wall-1",
        score=0.9,
        confidence=0.85,
        price=100.0,
    )

    signal = engine.build_signal(
        [result],
        symbol="BTCUSDT",
        exchange="binance",
        status=SpoofingStatus.DETECTED,
    )

    assert signal is not None
    assert signal.signal_id
    assert signal.symbol == "BTCUSDT"
    assert signal.exchange == "binance"
    assert signal.wall_id == "wall-1"
    assert signal.status == SpoofingStatus.DETECTED
    assert signal.score_breakdown is not None
    assert signal.detector_results == [result]
    assert signal.metadata["detector_count"] == 1
    assert signal.metadata["threshold"] == spoofing_config.scoring.detection_threshold
    assert signal.metadata["passed"] == signal.score_breakdown.passed


def test_score_engine_build_signal_can_return_signal_even_when_score_does_not_pass_threshold(
    spoofing_config,
    detector_result_factory,
) -> None:
    """
    Поточний контракт ScoreEngine: build_signal() будує signal, а passed flag
    лежить у score_breakdown/metadata. Analyzer уже вирішує, що публікувати.
    """

    spoofing_config.scoring.detection_threshold = 0.99
    spoofing_config.validate()

    engine = make_score_engine(spoofing_config)

    result = detector_result_factory(
        score=0.30,
        confidence=0.50,
        wall_size_ratio=2.0,
        pull_ratio=0.30,
        fill_ratio=0.20,
        price_reaction_bps=1.0,
    )

    signal = engine.build_signal(
        [result],
        symbol="BTCUSDT",
        exchange="binance",
    )

    assert signal is not None
    assert signal.score_breakdown is not None
    assert signal.score_breakdown.passed is False
    assert signal.metadata["passed"] is False


def test_score_engine_signal_id_is_stable_for_same_market_wall_pattern_and_price(
    spoofing_config,
    detector_result_factory,
) -> None:
    """
    Вразливість: нестабільний signal_id створить duplicate alerts/events.
    """

    engine = make_score_engine(spoofing_config)

    result_1 = detector_result_factory(
        wall_id="same-wall",
        pattern=SpoofingPattern.PULL_AND_REVERSAL,
        price=100.0,
        score=0.9,
        confidence=0.8,
    )
    result_2 = detector_result_factory(
        wall_id="same-wall",
        pattern=SpoofingPattern.PULL_AND_REVERSAL,
        price=100.0,
        score=0.7,
        confidence=0.6,
    )

    signal_1 = engine.build_signal(
        [result_1],
        symbol="BTCUSDT",
        exchange="binance",
    )
    signal_2 = engine.build_signal(
        [result_2],
        symbol="BTCUSDT",
        exchange="binance",
    )

    assert signal_1 is not None
    assert signal_2 is not None
    assert signal_1.signal_id == signal_2.signal_id


def test_score_engine_signal_id_changes_for_different_wall_or_pattern(
    spoofing_config,
    detector_result_factory,
) -> None:
    engine = make_score_engine(spoofing_config)

    base = detector_result_factory(
        wall_id="wall-1",
        pattern=SpoofingPattern.PULL_AND_REVERSAL,
        price=100.0,
    )
    other_wall = detector_result_factory(
        wall_id="wall-2",
        pattern=SpoofingPattern.PULL_AND_REVERSAL,
        price=100.0,
    )
    other_pattern = detector_result_factory(
        wall_id="wall-1",
        pattern=SpoofingPattern.PRESSURE_BLUFF,
        price=100.0,
    )

    signal_base = engine.build_signal([base], symbol="BTCUSDT", exchange="binance")
    signal_other_wall = engine.build_signal([other_wall], symbol="BTCUSDT", exchange="binance")
    signal_other_pattern = engine.build_signal([other_pattern], symbol="BTCUSDT", exchange="binance")

    assert signal_base is not None
    assert signal_other_wall is not None
    assert signal_other_pattern is not None

    assert signal_base.signal_id != signal_other_wall.signal_id
    assert signal_base.signal_id != signal_other_pattern.signal_id


def test_score_engine_should_emit_detection_uses_score_passed_flag(
    spoofing_config,
    detector_result_factory,
) -> None:
    engine = make_score_engine(spoofing_config)

    spoofing_config.scoring.detection_threshold = 0.95
    weak = detector_result_factory(
        score=0.20,
        confidence=0.50,
        wall_size_ratio=1.5,
        pull_ratio=0.20,
        fill_ratio=0.30,
        price_reaction_bps=0.5,
    )
    assert engine.should_emit_detection(
        [weak],
        symbol="BTCUSDT",
        exchange="binance",
    ) is False

    spoofing_config.scoring.detection_threshold = 0.10
    strong = detector_result_factory(
        score=0.95,
        confidence=0.90,
        wall_size_ratio=10.0,
        pull_ratio=0.95,
        fill_ratio=0.01,
        price_reaction_bps=20.0,
    )
    assert engine.should_emit_detection(
        [strong],
        symbol="BTCUSDT",
        exchange="binance",
    ) is True


# =============================================================================
# Scoring adversarial cases
# =============================================================================


def test_score_engine_does_not_let_wrong_market_poison_primary_wall_id(
    spoofing_config,
    detector_result_factory,
) -> None:
    """
    Вразливість: wrong-market high score result не має визначати final wall_id.
    """

    engine = make_score_engine(spoofing_config)

    wrong_market = detector_result_factory(
        symbol="ETHUSDT",
        exchange="binance",
        wall_id="eth-wall-poison",
        score=1.0,
        confidence=1.0,
    )
    valid = detector_result_factory(
        symbol="BTCUSDT",
        exchange="binance",
        wall_id="btc-wall-valid",
        score=0.7,
        confidence=0.7,
    )

    signal = engine.build_signal(
        [wrong_market, valid],
        symbol="BTCUSDT",
        exchange="binance",
    )

    assert signal is not None
    assert signal.wall_id == "btc-wall-valid"
    assert signal.metadata["detector_count"] == 1


def test_score_engine_handles_duplicate_detector_results_without_crashing(
    spoofing_config,
    detector_result_factory,
) -> None:
    """
    Вразливість: повторний event/retry може принести дублікати result-ів.
    Scoring не має падати або давати invalid math.
    """

    engine = make_score_engine(spoofing_config)

    duplicated = detector_result_factory(
        detector=SpoofingComponent.ORDER_PULL_DETECTOR,
        wall_id="same-wall",
        score=0.85,
        confidence=0.80,
    )

    score = engine.score(
        [duplicated, duplicated, duplicated],
        symbol="BTCUSDT",
        exchange="binance",
    )

    assert score is not None
    assert score.metadata["detector_count"] == 3
    assert 0.0 <= score.total_score <= 1.0
    assert 0.0 <= score.confidence <= spoofing_config.scoring.max_confidence


def test_score_engine_rejects_results_with_features_but_unknown_symbol_when_market_is_required(
    spoofing_config,
    detector_result_factory,
    spoofing_features_factory,
) -> None:
    engine = make_score_engine(spoofing_config)

    unknown_market_features = spoofing_features_factory(
        symbol="",
        exchange="",
        price=100.0,
    )
    result = detector_result_factory(
        features=unknown_market_features,
        score=0.9,
        confidence=0.9,
    )

    assert engine.score(
        [result],
        symbol="BTCUSDT",
        exchange="binance",
    ) is None


def test_score_engine_without_explicit_market_uses_result_market(
    spoofing_config,
    detector_result_factory,
) -> None:
    """
    Якщо caller не передав symbol/exchange, ScoreEngine має взяти market
    з positive features, а не повертати None.
    """

    engine = make_score_engine(spoofing_config)

    result = detector_result_factory(
        symbol="BTCUSDT",
        exchange="binance",
        score=0.9,
        confidence=0.9,
    )

    score = engine.score([result])

    assert score is not None
    assert score.metadata["symbol"] == "BTCUSDT"
    assert score.metadata["exchange"] == "binance"


def test_score_engine_handles_missing_wall_id_by_still_building_signal_id(
    spoofing_config,
    detector_result_factory,
) -> None:
    """
    Wall detector може повернути wall_id=None без PersistenceTracker.
    ScoreEngine не має падати.
    """

    engine = make_score_engine(spoofing_config)

    result = detector_result_factory(
        detector=SpoofingComponent.ORDERBOOK_WALL_DETECTOR,
        pattern=SpoofingPattern.SINGLE_LEVEL_SPOOF,
        wall_id=None,
        price=99.9,
        score=0.8,
        confidence=0.8,
    )

    signal = engine.build_signal(
        [result],
        symbol="BTCUSDT",
        exchange="binance",
    )

    assert signal is not None
    assert signal.wall_id is None
    assert signal.signal_id
    assert "none" not in signal.signal_id.lower()


def test_score_engine_output_metadata_is_eventbus_safe_top_level(
    spoofing_config,
    detector_result_factory,
) -> None:
    """
    Analyzer публікує score/signal у EventBus. Top-level metadata має бути
    простим для JSON/event serialization.
    """

    engine = make_score_engine(spoofing_config)

    result = detector_result_factory(
        score=0.9,
        confidence=0.8,
    )

    score = engine.score(
        [result],
        symbol="BTCUSDT",
        exchange="binance",
    )
    signal = engine.build_signal(
        [result],
        symbol="BTCUSDT",
        exchange="binance",
    )

    assert score is not None
    assert signal is not None

    for metadata in [score.metadata, signal.metadata]:
        for key, value in metadata.items():
            assert isinstance(key, str)
            assert isinstance(
                value,
                (str, int, float, bool, type(None)),
            ), f"Non-eventbus-safe metadata: {key!r}={value!r}"


# =============================================================================
# Cross-component persistence + scoring scenario
# =============================================================================


def test_persistence_tracker_and_score_engine_realistic_pull_to_signal_flow(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    """
    Реальний сценарій без analyzer:
    snapshot -> tracker wall -> mark_pulled -> synthetic detector result
    -> score engine -> signal.

    Це перевіряє, що lifecycle state сумісний із scoring expectations.
    """

    tracker = make_tracker(spoofing_config)
    engine = make_score_engine(spoofing_config)

    wall, _ = tracker.upsert_snapshot(
        orderbook_snapshot_factory(
            side=SpoofingSide.ASK,
            price=100.0,
            size=1_000.0,
            mid_price=100.0,
        )
    )
    pulled_wall, _ = tracker.mark_pulled(
        exchange=wall.exchange,
        symbol=wall.symbol,
        side=wall.side,
        price=wall.price,
        removed_size=950.0,
    )

    assert pulled_wall is wall
    assert wall.pull_ratio == pytest.approx(0.95)

    features = engine._merge_features(
        [
            DetectorResult(
                detector=SpoofingComponent.ORDER_PULL_DETECTOR,
                decision=DetectorDecision.POSITIVE,
                score=0.9,
                confidence=0.85,
                reason="realistic pull result",
                features=replace(
                    # reuse feature factory-like data from wall manually
                    engine._merge_features([]) if False else None
                ),
                wall_id=wall.wall_id,
                pattern=SpoofingPattern.PULL_AND_REVERSAL,
                metadata={},
            )
        ]
    )