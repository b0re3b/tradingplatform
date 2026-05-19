# tests/analytics/spoofing/test_spoofing_scoring_and_persistence.py

from __future__ import annotations

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


def make_key(
    component,
    *,
    exchange: str = "binance",
    market_type: str = "perpetual",
    symbol: str = "BTCUSDT",
    timeframe: str = "realtime",
):
    return component.make_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )


def store_wall_directly(tracker: PersistenceTracker, wall) -> None:
    """
    Test helper для scenarios, де треба напряму підкласти synthetic wall
    у state tracker-а.

    Production це робить upsert_snapshot(), але cleanup/limit/history tests
    інколи потребують точного контролю timestamp/state.

    Важливо: новий canonical index — _wall_ids_by_key, не _wall_ids_by_symbol.
    """

    tracker._walls_by_id[wall.wall_id] = wall
    tracker._wall_ids_by_key[wall.key].add(wall.wall_id)


def assert_wall_scope(
    wall,
    *,
    exchange: str = "binance",
    market_type: str = "perpetual",
    symbol: str = "BTCUSDT",
    timeframe: str = "realtime",
) -> None:
    assert wall.exchange == exchange
    assert wall.market_type == market_type
    assert wall.symbol == symbol
    assert wall.timeframe == timeframe
    assert wall.key == (exchange, market_type, symbol, timeframe)


def assert_score_scope(
    score,
    *,
    exchange: str = "binance",
    market_type: str = "perpetual",
    symbol: str = "BTCUSDT",
    timeframe: str = "realtime",
) -> None:
    assert score.metadata["exchange"] == exchange
    assert score.metadata["market_type"] == market_type
    assert score.metadata["symbol"] == symbol
    assert score.metadata["timeframe"] == timeframe
    assert score.metadata["scope"] == {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
    }


def assert_signal_scope(
    signal,
    *,
    exchange: str = "binance",
    market_type: str = "perpetual",
    symbol: str = "BTCUSDT",
    timeframe: str = "realtime",
) -> None:
    assert signal.exchange == exchange
    assert signal.market_type == market_type
    assert signal.symbol == symbol
    assert signal.timeframe == timeframe
    assert signal.key == (exchange, market_type, symbol, timeframe)
    assert signal.metadata["scope"] == {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
    }


# =============================================================================
# PersistenceTracker: creation / update / lifecycle
# =============================================================================


def test_persistence_tracker_creates_wall_with_key_index_and_lifecycle_event(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    """
    Wall має потрапити в canonical key index і lifecycle history.

    Якщо wall створився тільки в _walls_by_id, detector-и через analyze_key()
    його не знайдуть.
    """

    tracker = make_tracker(spoofing_config)

    snapshot = orderbook_snapshot_factory(
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
        side=SpoofingSide.BID,
        price=99.9,
        size=1_000.0,
        mid_price=100.0,
    )

    wall, events = tracker.upsert_snapshot(snapshot)

    assert wall.wall_id
    assert_wall_scope(wall)
    assert wall.side == SpoofingSide.BID
    assert wall.price == 99.9

    assert wall.initial_size == 1_000.0
    assert wall.current_size == 1_000.0
    assert wall.max_size == 1_000.0
    assert wall.min_size == 1_000.0
    assert wall.state == OrderbookWallState.ACTIVE

    assert events
    assert tracker.get_wall(wall.wall_id) is wall
    assert tracker.get_wall_by_snapshot(snapshot) is wall

    key_walls = tracker.get_walls_for_key(snapshot.key)
    assert [item.wall_id for item in key_walls] == [wall.wall_id]

    scope_walls = tracker.get_walls_for_scope(
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
    )
    assert [item.wall_id for item in scope_walls] == [wall.wall_id]

    history = tracker.get_recent_history(
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
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
    Той самий scoped price level не має створювати дублікати wall_id.
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
    assert tracker.get_walls_for_key(first.key) == [wall_3]

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
        market_type=wall_3.market_type,
        symbol=wall_3.symbol,
        timeframe=wall_3.timeframe,
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
    Float-noise у price не має створювати різні wall_id для того самого рівня.
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
    assert len(tracker.get_walls_for_key(first.key)) == 1


def test_persistence_tracker_very_close_prices_are_separate_when_rounding_allows_it(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    """
    Контр-тест: tracker не має агресивно merge-ити distinct price levels.
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
    assert len(tracker.get_walls_for_key(first.key)) == 2


def test_persistence_tracker_mark_pulled_updates_state_ratios_and_history(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    """
    mark_pulled має синхронно оновити current_size, estimated_pulled_size,
    total_removed_size, state і lifecycle history.
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
        market_type=wall.market_type,
        symbol=wall.symbol,
        timeframe=wall.timeframe,
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
        market_type=wall.market_type,
        symbol=wall.symbol,
        timeframe=wall.timeframe,
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
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
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
    removed_size більший за current_size не має робити current_size від'ємним.
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
        market_type=wall.market_type,
        symbol=wall.symbol,
        timeframe=wall.timeframe,
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
    negative removed_size не має збільшувати liquidity або ламати state.
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
        market_type=wall.market_type,
        symbol=wall.symbol,
        timeframe=wall.timeframe,
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
    assert len(tracker.snapshot_state()) == 3
    assert len(tracker.get_walls_for_key(snapshots[0].key)) == 3


# =============================================================================
# PersistenceTracker: indexes / limits / cleanup / snapshots
# =============================================================================


def test_persistence_tracker_enforces_max_walls_per_key_without_cross_key_deletion(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    """
    max_walls_per_key не має видаляти walls з іншого exchange/market_type/symbol/timeframe.
    """

    spoofing_config.persistence.max_walls_per_key = 2
    spoofing_config.persistence.max_walls_per_symbol = 2
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    for price in [99.9, 99.8, 99.7]:
        tracker.upsert_snapshot(
            orderbook_snapshot_factory(
                symbol="BTCUSDT",
                exchange="binance",
                market_type="perpetual",
                timeframe="realtime",
                price=price,
                size=1_000.0,
            )
        )

    eth_wall, _ = tracker.upsert_snapshot(
        orderbook_snapshot_factory(
            symbol="ETHUSDT",
            exchange="binance",
            market_type="perpetual",
            timeframe="realtime",
            price=2_000.0,
            size=1_000.0,
        )
    )

    bybit_wall, _ = tracker.upsert_snapshot(
        orderbook_snapshot_factory(
            symbol="BTCUSDT",
            exchange="bybit",
            market_type="perpetual",
            timeframe="realtime",
            price=99.9,
            size=1_000.0,
        )
    )

    btc_key = tracker.make_key(
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
    )
    eth_key = tracker.make_key(
        exchange="binance",
        market_type="perpetual",
        symbol="ETHUSDT",
        timeframe="realtime",
    )
    bybit_key = tracker.make_key(
        exchange="bybit",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
    )

    btc_walls = tracker.get_walls_for_key(btc_key)
    eth_walls = tracker.get_walls_for_key(eth_key)
    bybit_walls = tracker.get_walls_for_key(bybit_key)

    assert len(btc_walls) == 2
    assert [wall.symbol for wall in btc_walls] == ["BTCUSDT", "BTCUSDT"]
    assert eth_walls == [eth_wall]
    assert bybit_walls == [bybit_wall]


def test_persistence_tracker_get_walls_for_key_filters_side_and_state(
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
        market_type=ask_wall.market_type,
        symbol=ask_wall.symbol,
        timeframe=ask_wall.timeframe,
        side=ask_wall.side,
        price=ask_wall.price,
        removed_size=900.0,
    )

    bid_only = tracker.get_walls_for_key(
        bid_wall.key,
        side=SpoofingSide.BID,
    )
    pulled_only = tracker.get_walls_for_key(
        bid_wall.key,
        state=OrderbookWallState.PULLED,
    )

    assert [wall.wall_id for wall in bid_only] == [bid_wall.wall_id]
    assert [wall.wall_id for wall in pulled_only] == [ask_wall.wall_id]


def test_persistence_tracker_get_walls_for_symbol_legacy_filters_full_scope_when_provided(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    """
    Legacy helper get_walls_for_symbol лишається для міграції, але якщо
    market_type/timeframe передані, він має поводитись як scoped filter.
    """

    tracker = make_tracker(spoofing_config)

    target, _ = tracker.upsert_snapshot(
        orderbook_snapshot_factory(
            exchange="binance",
            market_type="perpetual",
            symbol="BTCUSDT",
            timeframe="realtime",
            price=99.9,
            size=1_000.0,
        )
    )
    tracker.upsert_snapshot(
        orderbook_snapshot_factory(
            exchange="binance",
            market_type="linear",
            symbol="BTCUSDT",
            timeframe="realtime",
            price=99.8,
            size=1_000.0,
        )
    )
    tracker.upsert_snapshot(
        orderbook_snapshot_factory(
            exchange="binance",
            market_type="perpetual",
            symbol="BTCUSDT",
            timeframe="1m",
            price=99.7,
            size=1_000.0,
        )
    )

    scoped = tracker.get_walls_for_symbol(
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
    )

    assert [wall.wall_id for wall in scoped] == [target.wall_id]


def test_persistence_tracker_snapshot_state_returns_copies_not_mutable_internal_references(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    """
    Зовнішній код не має змінити internal tracker state через snapshot_state().
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


def test_persistence_tracker_snapshot_state_can_be_scoped_by_key(
    spoofing_config,
    orderbook_snapshot_factory,
) -> None:
    tracker = make_tracker(spoofing_config)

    target, _ = tracker.upsert_snapshot(
        orderbook_snapshot_factory(
            exchange="binance",
            market_type="perpetual",
            symbol="BTCUSDT",
            timeframe="realtime",
            price=99.9,
            size=1_000.0,
        )
    )
    tracker.upsert_snapshot(
        orderbook_snapshot_factory(
            exchange="bybit",
            market_type="perpetual",
            symbol="BTCUSDT",
            timeframe="realtime",
            price=99.9,
            size=1_000.0,
        )
    )

    scoped_state = tracker.snapshot_state(key=target.key)

    assert len(scoped_state) == 1
    assert scoped_state[0].wall_id == target.wall_id
    assert scoped_state[0] is not target


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
        market_type=wall.market_type,
        symbol=wall.symbol,
        timeframe=wall.timeframe,
        side=wall.side,
        price=wall.price,
        limit=3,
    )
    history_limit_0 = tracker.get_recent_history(
        exchange=wall.exchange,
        market_type=wall.market_type,
        symbol=wall.symbol,
        timeframe=wall.timeframe,
        side=wall.side,
        price=wall.price,
        limit=0,
    )
    history_limit_negative = tracker.get_recent_history(
        exchange=wall.exchange,
        market_type=wall.market_type,
        symbol=wall.symbol,
        timeframe=wall.timeframe,
        side=wall.side,
        price=wall.price,
        limit=-5,
    )

    assert len(history_limit_3) == 3
    assert history_limit_0 == []
    assert history_limit_negative == []


def test_persistence_tracker_cleanup_removes_expired_walls_and_key_indexes(
    spoofing_config,
    tracked_wall_factory,
    fixed_now,
) -> None:
    """
    cleanup не має лишати stale wall_id у _wall_ids_by_key.
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

    key_walls = tracker.get_walls_for_key(active.key)
    assert [wall.wall_id for wall in key_walls] == ["active-wall"]


def test_persistence_tracker_cleanup_does_not_delete_active_walls_from_other_keys(
    spoofing_config,
    tracked_wall_factory,
    fixed_now,
) -> None:
    spoofing_config.persistence.wall_ttl_ms = 1_000
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)

    expired = tracked_wall_factory(
        wall_id="expired-binance",
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
    )
    expired.last_seen_at = fixed_now - timedelta(milliseconds=10_000)

    active_other_key = tracked_wall_factory(
        wall_id="active-bybit",
        exchange="bybit",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
    )
    active_other_key.last_seen_at = fixed_now

    store_wall_directly(tracker, expired)
    store_wall_directly(tracker, active_other_key)

    removed = tracker.cleanup_expired(now=fixed_now)

    assert removed == 1
    assert tracker.get_wall("expired-binance") is None
    assert tracker.get_wall("active-bybit") is active_other_key
    assert tracker.get_walls_for_key(active_other_key.key) == [active_other_key]


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


def test_persistence_tracker_stats_reflect_key_state_history_and_cleanup_time(
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
        market_type=wall.market_type,
        symbol=wall.symbol,
        timeframe=wall.timeframe,
        side=wall.side,
        price=wall.price,
        removed_size=900.0,
    )
    tracker.cleanup_expired(now=fixed_now)

    stats = tracker.stats()

    assert stats["tracked_walls"] == 1
    assert stats["history_levels"] >= 1
    assert stats["last_cleanup_at"] is not None
    assert stats["scope"] == "exchange:market_type:symbol:timeframe"
    assert stats["keys"] == {
        "binance:perpetual:BTCUSDT:realtime": 1,
    }
    assert stats["markets"] == {
        "binance:perpetual:BTCUSDT": 1,
    }
    assert stats["states"][OrderbookWallState.PULLED.value] == 1


# =============================================================================
# SpoofingScoreEngine: filtering / no-signal cases
# =============================================================================


def test_score_engine_returns_none_when_disabled(
    spoofing_config,
    detector_result_factory,
) -> None:
    spoofing_config.scoring.enabled = False
    spoofing_config.validate()

    engine = make_score_engine(spoofing_config)
    key = make_key(engine)

    result = detector_result_factory(score=0.9, confidence=0.9)

    assert engine.score([result], key=key) is None
    assert engine.build_signal([result], key=key) is None


def test_score_engine_returns_none_for_empty_results(
    spoofing_config,
) -> None:
    engine = make_score_engine(spoofing_config)
    key = make_key(engine)

    assert engine.score([], key=key) is None
    assert engine.build_signal([], key=key) is None


def test_score_engine_uses_only_positive_results_from_requested_key(
    spoofing_config,
    detector_result_factory,
) -> None:
    engine = make_score_engine(spoofing_config)
    key = make_key(engine)

    valid = detector_result_factory(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="perpetual",
        timeframe="realtime",
        score=0.7,
        confidence=0.7,
        pull_ratio=0.8,
        fill_ratio=0.05,
    )
    poison_wrong_symbol = detector_result_factory(
        symbol="ETHUSDT",
        exchange="binance",
        market_type="perpetual",
        timeframe="realtime",
        score=1.0,
        confidence=1.0,
        pull_ratio=1.0,
        fill_ratio=0.0,
    )
    poison_wrong_market_type = detector_result_factory(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="linear",
        timeframe="realtime",
        score=1.0,
        confidence=1.0,
        pull_ratio=1.0,
        fill_ratio=0.0,
    )
    poison_negative = detector_result_factory(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="perpetual",
        timeframe="realtime",
        decision=DetectorDecision.NEGATIVE,
        score=1.0,
        confidence=1.0,
    )

    score = engine.score(
        [
            valid,
            poison_wrong_symbol,
            poison_wrong_market_type,
            poison_negative,
        ],
        key=key,
    )

    assert score is not None
    assert score.metadata["detector_count"] == 1
    assert_score_scope(score)


def test_score_engine_legacy_filters_match_key_filter_for_same_scope(
    spoofing_config,
    detector_result_factory,
) -> None:
    engine = make_score_engine(spoofing_config)
    key = make_key(engine)

    valid = detector_result_factory(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="perpetual",
        timeframe="realtime",
        score=0.8,
        confidence=0.8,
    )
    wrong_timeframe = detector_result_factory(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="perpetual",
        timeframe="1m",
        score=1.0,
        confidence=1.0,
    )

    key_score = engine.score([valid, wrong_timeframe], key=key)
    legacy_score = engine.score(
        [valid, wrong_timeframe],
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
    )

    assert key_score is not None
    assert legacy_score is not None
    assert key_score.metadata["detector_count"] == legacy_score.metadata["detector_count"] == 1
    assert key_score.total_score == pytest.approx(legacy_score.total_score)


# =============================================================================
# SpoofingScoreEngine: score / confidence / severity
# =============================================================================


def test_score_engine_builds_score_with_clamped_score_confidence_and_contributions(
    spoofing_config,
    detector_result_factory,
) -> None:
    """
    Extreme feature values не мають давати score/confidence > 1.
    """

    spoofing_config.scoring.detection_threshold = 0.10
    spoofing_config.validate()

    engine = make_score_engine(spoofing_config)
    key = make_key(engine)

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

    score = engine.score([extreme], key=key)

    assert score is not None
    assert 0.0 <= score.total_score <= 1.0
    assert 0.0 <= score.confidence <= spoofing_config.scoring.max_confidence
    assert score.contributions
    assert all(0.0 <= item.value <= 1.0 for item in score.contributions)
    assert all(item.weight >= 0.0 for item in score.contributions)
    assert_score_scope(score)


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
    total_score: float,
    expected_severity: SpoofingSeverity,
) -> None:
    """
    Severity boundary bugs дуже дорогі для alerting/risk.
    """

    engine = make_score_engine(spoofing_config)

    severity = engine._resolve_severity(total_score)

    assert severity == expected_severity


def test_score_engine_confidence_is_boosted_by_detector_agreement_but_clamped(
    spoofing_config,
    detector_result_factory,
) -> None:
    spoofing_config.scoring.confidence_boost_on_detector_agreement = 0.25
    spoofing_config.scoring.max_confidence = 0.90
    spoofing_config.validate()

    engine = make_score_engine(spoofing_config)
    key = make_key(engine)

    results = [
        detector_result_factory(
            detector=SpoofingComponent.ORDER_PULL_DETECTOR,
            score=0.8,
            confidence=0.89,
            wall_id="wall-1",
        ),
        detector_result_factory(
            detector=SpoofingComponent.FAKE_LIQUIDITY_DETECTOR,
            score=0.8,
            confidence=0.89,
            wall_id="wall-1",
        ),
        detector_result_factory(
            detector=SpoofingComponent.FLIP_PRESSURE_DETECTOR,
            score=0.8,
            confidence=0.89,
            wall_id="wall-1",
        ),
    ]

    score = engine.score(results, key=key)

    assert score is not None
    assert score.confidence <= spoofing_config.scoring.max_confidence
    assert score.metadata["detector_count"] == 3
    assert score.metadata["agreement_ratio"] > 0.0


def test_score_engine_merges_features_by_strongest_values(
    spoofing_config,
    detector_result_factory,
) -> None:
    engine = make_score_engine(spoofing_config)
    key = make_key(engine)

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

    signal = engine.build_signal([weak, strong], key=key)

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
    key = make_key(engine)

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

    signal = engine.build_signal([weak_wall, strong_flip], key=key)

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
    key = make_key(engine)

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
        key=key,
        status=SpoofingStatus.DETECTED,
    )

    assert signal is not None
    assert signal.signal_id
    assert_signal_scope(signal)
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
    лежить у score_breakdown/metadata. Analyzer вирішує, що публікувати.
    """

    spoofing_config.scoring.detection_threshold = 0.99
    spoofing_config.validate()

    engine = make_score_engine(spoofing_config)
    key = make_key(engine)

    result = detector_result_factory(
        score=0.30,
        confidence=0.50,
        wall_size_ratio=2.0,
        pull_ratio=0.30,
        fill_ratio=0.20,
        price_reaction_bps=1.0,
    )

    signal = engine.build_signal([result], key=key)

    assert signal is not None
    assert signal.score_breakdown is not None
    assert signal.score_breakdown.passed is False
    assert signal.metadata["passed"] is False


def test_score_engine_signal_id_is_stable_for_same_scope_wall_pattern_and_price(
    spoofing_config,
    detector_result_factory,
) -> None:
    engine = make_score_engine(spoofing_config)
    key = make_key(engine)

    result_1 = detector_result_factory(
        detector=SpoofingComponent.ORDER_PULL_DETECTOR,
        pattern=SpoofingPattern.PULL_AND_REVERSAL,
        wall_id="wall-stable",
        price=100.0,
        score=0.9,
        confidence=0.8,
    )
    result_2 = detector_result_factory(
        detector=SpoofingComponent.ORDER_PULL_DETECTOR,
        pattern=SpoofingPattern.PULL_AND_REVERSAL,
        wall_id="wall-stable",
        price=100.0,
        score=0.7,
        confidence=0.6,
    )

    signal_1 = engine.build_signal([result_1], key=key)
    signal_2 = engine.build_signal([result_2], key=key)

    assert signal_1 is not None
    assert signal_2 is not None
    assert signal_1.signal_id == signal_2.signal_id


def test_score_engine_signal_id_changes_for_different_key_scope(
    spoofing_config,
    detector_result_factory,
) -> None:
    engine = make_score_engine(spoofing_config)

    binance_key = make_key(engine, exchange="binance")
    bybit_key = make_key(engine, exchange="bybit")

    result = detector_result_factory(
        detector=SpoofingComponent.ORDER_PULL_DETECTOR,
        pattern=SpoofingPattern.PULL_AND_REVERSAL,
        wall_id="same-wall-id",
        price=100.0,
        score=0.9,
        confidence=0.8,
    )

    binance_signal = engine.build_signal([result], key=binance_key)
    bybit_signal = engine.build_signal([result], key=bybit_key)

    assert binance_signal is not None
    assert bybit_signal is not None
    assert binance_signal.signal_id != bybit_signal.signal_id


def test_score_engine_handles_missing_wall_id_by_still_building_signal_id(
    spoofing_config,
    detector_result_factory,
) -> None:
    """
    Wall detector може повернути wall_id=None без PersistenceTracker.
    ScoreEngine не має падати.
    """

    engine = make_score_engine(spoofing_config)
    key = make_key(engine)

    result = detector_result_factory(
        detector=SpoofingComponent.ORDERBOOK_WALL_DETECTOR,
        pattern=SpoofingPattern.SINGLE_LEVEL_SPOOF,
        wall_id=None,
        price=99.9,
        score=0.8,
        confidence=0.8,
    )

    signal = engine.build_signal([result], key=key)

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
    key = make_key(engine)

    result = detector_result_factory(
        score=0.9,
        confidence=0.8,
    )

    score = engine.score([result], key=key)
    signal = engine.build_signal([result], key=key)

    assert score is not None
    assert signal is not None

    for metadata in [score.metadata, signal.metadata]:
        for item_key, value in metadata.items():
            assert isinstance(item_key, str)
            assert isinstance(
                value,
                (str, int, float, bool, type(None), dict),
            ), f"Non-eventbus-safe metadata: {item_key!r}={value!r}"

        assert isinstance(metadata["scope"], dict)
        assert metadata["scope"] == {
            "exchange": "binance",
            "market_type": "perpetual",
            "symbol": "BTCUSDT",
            "timeframe": "realtime",
        }


def test_score_engine_result_with_no_positive_detector_results_returns_none(
    spoofing_config,
    detector_result_factory,
) -> None:
    engine = make_score_engine(spoofing_config)
    key = make_key(engine)

    negative = detector_result_factory(
        decision=DetectorDecision.NEGATIVE,
        score=1.0,
        confidence=1.0,
    )

    assert engine.score([negative], key=key) is None
    assert engine.build_signal([negative], key=key) is None


# =============================================================================
# Cross-component persistence + scoring scenario
# =============================================================================


def test_persistence_tracker_and_score_engine_realistic_pull_to_signal_flow(
    spoofing_config,
    orderbook_snapshot_factory,
    detector_result_factory,
) -> None:
    """
    Реальний сценарій без analyzer:
    snapshot -> tracker wall -> mark_pulled -> synthetic detector result
    -> score engine -> signal.

    Перевіряє, що lifecycle state сумісний із scoring expectations.
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
        market_type=wall.market_type,
        symbol=wall.symbol,
        timeframe=wall.timeframe,
        side=wall.side,
        price=wall.price,
        removed_size=950.0,
    )

    assert pulled_wall is wall
    assert wall.pull_ratio == pytest.approx(0.95)

    result = detector_result_factory(
        detector=SpoofingComponent.ORDER_PULL_DETECTOR,
        pattern=SpoofingPattern.PULL_AND_REVERSAL,
        wall_id=wall.wall_id,
        symbol=wall.symbol,
        exchange=wall.exchange,
        market_type=wall.market_type,
        timeframe=wall.timeframe,
        side=wall.side,
        price=wall.price,
        wall_size=wall.max_size,
        pull_ratio=wall.pull_ratio,
        fill_ratio=wall.fill_ratio,
        score=0.90,
        confidence=0.85,
    )

    signal = engine.build_signal([result], key=wall.key)

    assert signal is not None
    assert_signal_scope(signal)
    assert signal.wall_id == wall.wall_id
    assert signal.side == SpoofingSide.ASK
    assert signal.price_level == wall.price
    assert signal.score_breakdown is not None
    assert signal.score_breakdown.passed == signal.metadata["passed"]


def test_persistence_tracker_cleanup_then_score_engine_does_not_reference_removed_wall(
    spoofing_config,
    tracked_wall_factory,
    detector_result_factory,
    fixed_now,
) -> None:
    """
    Якщо wall видалено cleanup-ом, ScoreEngine не має потребувати live wall state.
    Сигнал може бути побудований лише з immutable DetectorResult.
    """

    spoofing_config.persistence.wall_ttl_ms = 1_000
    spoofing_config.validate()

    tracker = make_tracker(spoofing_config)
    engine = make_score_engine(spoofing_config)

    expired = tracked_wall_factory(
        wall_id="expired-wall",
        price=100.0,
        lifetime_ms=10_000.0,
    )
    expired.last_seen_at = fixed_now - timedelta(milliseconds=10_000)
    store_wall_directly(tracker, expired)

    result = detector_result_factory(
        detector=SpoofingComponent.ORDER_PULL_DETECTOR,
        pattern=SpoofingPattern.PULL_AND_REVERSAL,
        wall_id=expired.wall_id,
        symbol=expired.symbol,
        exchange=expired.exchange,
        market_type=expired.market_type,
        timeframe=expired.timeframe,
        price=expired.price,
        score=0.90,
        confidence=0.85,
    )

    assert tracker.cleanup_expired(now=fixed_now) == 1
    assert tracker.get_wall(expired.wall_id) is None

    signal = engine.build_signal([result], key=expired.key)

    assert signal is not None
    assert signal.wall_id == "expired-wall"
    assert_signal_scope(signal)


def test_persistence_tracker_key_isolation_plus_score_engine_key_filtering(
    spoofing_config,
    tracked_wall_factory,
    detector_result_factory,
) -> None:
    """
    Комплексний key isolation:
    tracker містить два однакові symbols на різних exchanges;
    score engine має використати тільки result для requested key.
    """

    tracker = make_tracker(spoofing_config)
    engine = make_score_engine(spoofing_config)

    binance_wall = tracked_wall_factory(
        wall_id="binance-wall",
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
    )
    bybit_wall = tracked_wall_factory(
        wall_id="bybit-wall",
        exchange="bybit",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
    )

    store_wall_directly(tracker, binance_wall)
    store_wall_directly(tracker, bybit_wall)

    binance_key = binance_wall.key

    binance_result = detector_result_factory(
        wall_id=binance_wall.wall_id,
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
        score=0.70,
        confidence=0.70,
    )
    bybit_poison = detector_result_factory(
        wall_id=bybit_wall.wall_id,
        exchange="bybit",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
        score=1.0,
        confidence=1.0,
    )

    assert len(tracker.get_walls_for_key(binance_wall.key)) == 1
    assert len(tracker.get_walls_for_key(bybit_wall.key)) == 1

    score = engine.score([binance_result, bybit_poison], key=binance_key)
    signal = engine.build_signal([binance_result, bybit_poison], key=binance_key)

    assert score is not None
    assert signal is not None
    assert score.metadata["detector_count"] == 1
    assert signal.wall_id == binance_wall.wall_id
    assert_signal_scope(signal, exchange="binance")


def test_score_engine_does_not_let_wrong_key_poison_primary_pattern(
    spoofing_config,
    detector_result_factory,
) -> None:
    """
    Високий score з іншого key не має визначати primary pattern.
    """

    engine = make_score_engine(spoofing_config)
    key = make_key(engine)

    valid = detector_result_factory(
        detector=SpoofingComponent.ORDER_PULL_DETECTOR,
        pattern=SpoofingPattern.PULL_AND_REVERSAL,
        exchange="binance",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
        score=0.60,
        confidence=0.60,
    )
    poison = detector_result_factory(
        detector=SpoofingComponent.FLIP_PRESSURE_DETECTOR,
        pattern=SpoofingPattern.PRESSURE_BLUFF,
        exchange="bybit",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="realtime",
        score=1.0,
        confidence=1.0,
        pressure_flip_strength=1.0,
    )

    signal = engine.build_signal([valid, poison], key=key)

    assert signal is not None
    assert signal.pattern == SpoofingPattern.PULL_AND_REVERSAL
    assert signal.metadata["detector_count"] == 1