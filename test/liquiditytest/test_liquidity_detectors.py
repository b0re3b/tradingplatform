# tests/analytics/liquidity/test_liquidity_detectors.py

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from analytics.liquidity.config import LiquidityConfig
from analytics.liquidity.enums import (
    LiquidityLevelType,
    LiquiditySide,
    LiquidityStatus,
    SweepStatus,
)
from analytics.liquidity.equal_highs_lows import EqualHighsLowsDetector
from analytics.liquidity.models import (
    DEFAULT_TIMEFRAME,
    EqualLevel,
    LiquidityLevel,
    StopCluster,
    make_liquidity_key,
)
from analytics.liquidity.scoring import LiquidityScorer
from analytics.liquidity.stop_clusters import StopClustersDetector


# ---------------------------------------------------------------------
# Canonical futures scope
# ---------------------------------------------------------------------


TEST_EXCHANGE = "binance"
TEST_MARKET_TYPE = "usdm_futures"
ALT_EXCHANGE = "bybit"
ALT_MARKET_TYPE = "linear"


# ---------------------------------------------------------------------
# Local factories
# ---------------------------------------------------------------------


def _make_candle(
    *,
    index: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 100.0,
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    as_strings: bool = False,
) -> dict[str, Any]:
    open_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc) + timedelta(
        minutes=index
    )
    close_time = open_time + timedelta(minutes=1)

    candle: dict[str, Any] = {
        "exchange": TEST_EXCHANGE,
        "market_type": TEST_MARKET_TYPE,
        "symbol": symbol,
        "timeframe": timeframe,
        "open_time": open_time,
        "close_time": close_time,
        "timestamp": close_time,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }

    if as_strings:
        for key in ("open", "high", "low", "close", "volume"):
            candle[key] = str(candle[key])

    return candle


def _make_candles_from_ohlc(
    rows: list[tuple[float, float, float, float]],
    *,
    volume: float = 100.0,
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    as_strings: bool = False,
) -> list[dict[str, Any]]:
    return [
        _make_candle(
            index=index,
            open_=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            symbol=symbol,
            timeframe=timeframe,
            as_strings=as_strings,
        )
        for index, (open_, high, low, close) in enumerate(rows)
    ]


def _deterministic_equal_high_candles(
    *,
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    as_strings: bool = False,
) -> list[dict[str, Any]]:
    """
    Серія з трьома дуже чіткими pivot highs біля 110.

    Піки достатньо вищі за сусідні candles, щоб не залежати від
    випадкових edge-case умов pivot_lookback / pivot_lookforward.
    """

    return _make_candles_from_ohlc(
        [
            (100.0, 101.0, 99.0, 100.5),
            (100.5, 103.0, 99.8, 102.0),
            (102.0, 105.0, 100.5, 103.0),
            (103.0, 110.00, 101.0, 104.0),  # pivot high
            (104.0, 105.0, 100.5, 101.5),
            (101.5, 103.0, 99.5, 100.5),
            (100.5, 102.0, 98.5, 101.0),
            (101.0, 105.0, 99.8, 103.0),
            (103.0, 110.05, 101.2, 104.0),  # pivot high
            (104.0, 105.5, 100.8, 102.0),
            (102.0, 103.0, 99.2, 100.0),
            (100.0, 102.0, 98.8, 101.0),
            (101.0, 104.5, 99.5, 103.0),
            (103.0, 109.96, 101.0, 104.0),  # pivot high
            (104.0, 105.0, 100.0, 101.0),
            (101.0, 102.0, 98.5, 100.0),
            (100.0, 101.0, 97.8, 99.5),
        ],
        volume=150.0,
        symbol=symbol,
        timeframe=timeframe,
        as_strings=as_strings,
    )


def _deterministic_equal_low_candles(
    *,
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    as_strings: bool = False,
) -> list[dict[str, Any]]:
    """
    Серія з трьома дуже чіткими pivot lows біля 90.
    """

    return _make_candles_from_ohlc(
        [
            (100.0, 101.0, 98.5, 99.0),
            (99.0, 100.0, 96.0, 97.5),
            (97.5, 99.0, 94.0, 95.0),
            (95.0, 97.0, 90.00, 92.0),  # pivot low
            (92.0, 96.5, 92.2, 95.0),
            (95.0, 99.0, 94.5, 98.0),
            (98.0, 100.0, 95.0, 96.5),
            (96.5, 98.0, 93.5, 94.0),
            (94.0, 96.0, 90.04, 92.5),  # pivot low
            (92.5, 97.0, 92.8, 96.0),
            (96.0, 100.0, 95.0, 99.0),
            (99.0, 101.0, 96.0, 97.0),
            (97.0, 99.0, 94.5, 95.0),
            (95.0, 97.0, 89.96, 92.0),  # pivot low
            (92.0, 96.0, 92.5, 95.0),
            (95.0, 99.0, 94.0, 98.0),
            (98.0, 101.0, 97.0, 100.0),
        ],
        volume=140.0,
        symbol=symbol,
        timeframe=timeframe,
        as_strings=as_strings,
    )


def _deterministic_two_sided_candles(
    *,
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
) -> list[dict[str, Any]]:
    rows = _deterministic_equal_high_candles(symbol=symbol, timeframe=timeframe)

    lows = [
        98.5,
        96.0,
        94.0,
        90.00,
        92.2,
        94.5,
        95.0,
        93.5,
        90.04,
        92.8,
        95.0,
        96.0,
        94.5,
        89.96,
        92.5,
        94.0,
        97.0,
    ]

    result: list[dict[str, Any]] = []
    for candle, low in zip(rows, lows, strict=True):
        cloned = dict(candle)
        cloned["low"] = low
        result.append(cloned)

    return result


def _object_style_candles(candles: list[dict[str, Any]]) -> list[SimpleNamespace]:
    return [SimpleNamespace(**candle) for candle in candles]


def _hostile_candles(
    *,
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
) -> list[dict[str, Any]]:
    candles = _deterministic_equal_high_candles(symbol=symbol, timeframe=timeframe)
    candles[3] = {**candles[3], "high": "not-a-number"}
    candles[8] = {**candles[8], "low": None}
    return candles


def _scope_level(
    level: LiquidityLevel,
    *,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
) -> LiquidityLevel:
    cloned = deepcopy(level)
    cloned.exchange = exchange
    cloned.market_type = market_type
    cloned.symbol = symbol
    cloned.timeframe = timeframe
    cloned.metadata = dict(cloned.metadata or {})
    cloned.metadata["scope"] = {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
    }
    cloned.metadata["scope_key"] = f"{exchange}:{market_type}:{symbol}:{timeframe}"
    return cloned


def _scope_levels(
    levels: list[LiquidityLevel],
    *,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
) -> list[LiquidityLevel]:
    return [
        _scope_level(
            level,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        for level in levels
    ]


def _move_level_price(level: LiquidityLevel, price: float) -> LiquidityLevel:
    cloned = deepcopy(level)
    cloned.price = price
    return cloned


def _expected_key(
    *,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
):
    return make_liquidity_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )


# ---------------------------------------------------------------------
# Local assertions
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


def _assert_level_scope(
    level: LiquidityLevel,
    *,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
    symbol: str,
    timeframe: str,
) -> None:
    assert level.exchange == exchange
    assert level.market_type == market_type
    assert level.symbol == symbol
    assert level.timeframe == timeframe
    assert level.liquidity_key == _expected_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )


def _assert_cluster_scope(
    cluster: StopCluster,
    *,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
    symbol: str,
    timeframe: str,
) -> None:
    assert cluster.exchange == exchange
    assert cluster.market_type == market_type
    assert cluster.symbol == symbol
    assert cluster.timeframe == timeframe
    assert cluster.liquidity_key == _expected_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )

    for source_level in cluster.source_levels:
        _assert_level_scope(
            source_level,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )


def _assert_level_payload_contract(
    level: LiquidityLevel,
    *,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
    symbol: str,
    timeframe: str,
) -> None:
    _assert_level_scope(
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

    assert payload["exchange"] == exchange
    assert payload["market_type"] == market_type
    assert payload["symbol"] == symbol
    assert payload["timeframe"] == timeframe
    assert payload["level_type"] == level.level_type.value
    assert payload["side"] == level.side.value
    assert payload["price"] == level.price
    assert payload["status"] == level.status.value
    assert payload["sweep_status"] == level.sweep_status.value
    assert payload["scope"]["exchange"] == exchange
    assert payload["scope"]["market_type"] == market_type
    assert payload["scope"]["symbol"] == symbol
    assert payload["scope"]["timeframe"] == timeframe
    assert payload["scope_key"] == f"{exchange}:{market_type}:{symbol}:{timeframe}"
    assert payload["liquidity_key"] == _expected_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )
    assert isinstance(payload["metadata"], dict)


def _assert_cluster_payload_contract(
    cluster: StopCluster,
    *,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
    symbol: str,
    timeframe: str,
) -> None:
    _assert_cluster_scope(
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

    assert payload["exchange"] == exchange
    assert payload["market_type"] == market_type
    assert payload["symbol"] == symbol
    assert payload["timeframe"] == timeframe
    assert payload["side"] == cluster.side.value
    assert payload["low_price"] == cluster.low_price
    assert payload["high_price"] == cluster.high_price
    assert payload["center_price"] == cluster.center_price
    assert payload["scope"]["exchange"] == exchange
    assert payload["scope"]["market_type"] == market_type
    assert payload["scope_key"] == f"{exchange}:{market_type}:{symbol}:{timeframe}"
    assert payload["liquidity_key"] == _expected_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )
    assert isinstance(payload["source_levels"], list)
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
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_with_equal_highs,
            current_price=100.0,
        )

        equal_highs = _levels_by_type(levels, LiquidityLevelType.EQUAL_HIGHS)

        assert equal_highs
        assert all(level.side == LiquiditySide.BUY_SIDE for level in equal_highs)

        strongest = max(equal_highs, key=lambda level: level.confidence)

        assert strongest.price == pytest.approx(105.0, rel=0.003)
        assert strongest.touches_count >= 2
        assert strongest.is_active()

        _assert_level_payload_contract(
            strongest,
            symbol=symbol,
            timeframe=timeframe,
        )

    def test_detects_equal_lows_from_repeated_pivot_lows(
        self,
        equal_detector: EqualHighsLowsDetector,
        candles_with_equal_lows: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = equal_detector.detect(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_with_equal_lows,
            current_price=100.0,
        )

        equal_lows = _levels_by_type(levels, LiquidityLevelType.EQUAL_LOWS)

        assert equal_lows
        assert all(level.side == LiquiditySide.SELL_SIDE for level in equal_lows)

        strongest = max(equal_lows, key=lambda level: level.confidence)

        assert strongest.price == pytest.approx(95.0, rel=0.003)
        assert strongest.touches_count >= 2
        assert strongest.is_active()

        _assert_level_payload_contract(
            strongest,
            symbol=symbol,
            timeframe=timeframe,
        )

    def test_detects_both_equal_highs_and_equal_lows_from_two_sided_structure(
        self,
        equal_detector: EqualHighsLowsDetector,
        candles_with_both_sides: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = equal_detector.detect(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_with_both_sides,
            current_price=100.0,
        )

        equal_highs = _levels_by_type(levels, LiquidityLevelType.EQUAL_HIGHS)
        equal_lows = _levels_by_type(levels, LiquidityLevelType.EQUAL_LOWS)

        assert equal_highs
        assert equal_lows

        assert any(
            level.price == pytest.approx(105.0, rel=0.003)
            for level in equal_highs
        )
        assert any(
            level.price == pytest.approx(95.0, rel=0.003)
            for level in equal_lows
        )

    def test_accepts_object_style_candles(
        self,
        equal_detector: EqualHighsLowsDetector,
        candles_with_equal_highs: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        candles = _object_style_candles(candles_with_equal_highs)

        levels = equal_detector.detect(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            candles=candles,
            current_price=100.0,
        )

        equal_highs = _levels_by_type(levels, LiquidityLevelType.EQUAL_HIGHS)

        assert equal_highs
        for level in equal_highs:
            _assert_level_payload_contract(
                level,
                symbol=symbol,
                timeframe=timeframe,
            )

    def test_accepts_string_numeric_candle_values(
        self,
        equal_detector: EqualHighsLowsDetector,
        candles_with_equal_lows: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        candles = []

        for candle in candles_with_equal_lows:
            cloned = dict(candle)
            for key in ("open", "high", "low", "close", "volume"):
                cloned[key] = str(cloned[key])
            candles.append(cloned)

        levels = equal_detector.detect(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            candles=candles,
            current_price="100.0",
        )

        equal_lows = _levels_by_type(levels, LiquidityLevelType.EQUAL_LOWS)

        assert equal_lows
        assert all(level.price > 0 for level in equal_lows)

        for level in equal_lows:
            _assert_level_payload_contract(
                level,
                symbol=symbol,
                timeframe=timeframe,
            )

    def test_returns_empty_when_not_enough_candles(
        self,
        equal_detector: EqualHighsLowsDetector,
        too_few_candles: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = equal_detector.detect(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
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
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
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
        with pytest.raises(ValueError, match="symbol"):
            equal_detector.detect(
                exchange=TEST_EXCHANGE,
                market_type=TEST_MARKET_TYPE,
                symbol="",
                timeframe=timeframe,
                candles=candles_with_equal_highs,
                current_price=100.0,
            )

    def test_rejects_missing_timeframe_or_normalizes_to_default_contract(
        self,
        equal_detector: EqualHighsLowsDetector,
        candles_with_equal_highs: list[dict[str, Any]],
        symbol: str,
    ) -> None:
        try:
            levels = equal_detector.detect(
                exchange=TEST_EXCHANGE,
                market_type=TEST_MARKET_TYPE,
                symbol=symbol,
                timeframe="",
                candles=candles_with_equal_highs,
                current_price=100.0,
            )
        except ValueError as exc:
            assert "timeframe" in str(exc)
            return

        assert all(level.timeframe == DEFAULT_TIMEFRAME for level in levels)

    def test_detector_respects_disabled_config(
        self,
        disabled_liquidity_config: LiquidityConfig,
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
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_with_equal_highs,
            current_price=100.0,
        )

        assert levels == []

    def test_min_equal_touches_filter_removes_under_touched_levels(
        self,
        liquidity_config: LiquidityConfig,
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
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
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
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
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
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
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
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_with_both_sides,
            current_price=100.0,
        )

        sort_keys = [(level.price, level.level_type.value) for level in levels]

        assert sort_keys == sorted(sort_keys)

    def test_all_detected_confidences_are_clamped_and_payload_safe(
        self,
        equal_detector: EqualHighsLowsDetector,
        candles_with_both_sides: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = equal_detector.detect(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_with_both_sides,
            current_price=100.0,
        )

        assert levels

        for level in levels:
            _assert_level_payload_contract(
                level,
                symbol=symbol,
                timeframe=timeframe,
            )

    def test_hostile_ohlc_payload_raises_or_returns_payload_safe_result(
        self,
        equal_detector: EqualHighsLowsDetector,
        candles_with_equal_highs: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        candles = deepcopy(candles_with_equal_highs)
        candles[3]["high"] = "not-a-number"
        candles[8]["low"] = None

        try:
            levels = equal_detector.detect(
                exchange=TEST_EXCHANGE,
                market_type=TEST_MARKET_TYPE,
                symbol=symbol,
                timeframe=timeframe,
                candles=candles,
                current_price=100.0,
            )
        except Exception as exc:
            assert isinstance(exc, (ValueError, TypeError, RuntimeError))
            return

        for level in levels:
            _assert_level_payload_contract(
                level,
                symbol=symbol,
                timeframe=timeframe,
            )

    def test_same_symbol_timeframe_is_scoped_by_exchange_and_market_type(
        self,
        equal_detector: EqualHighsLowsDetector,
        candles_with_equal_highs: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        binance_levels = equal_detector.detect(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_with_equal_highs,
            current_price=100.0,
        )
        bybit_levels = equal_detector.detect(
            exchange=ALT_EXCHANGE,
            market_type=ALT_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_with_equal_highs,
            current_price=100.0,
        )

        assert binance_levels
        assert bybit_levels

        assert {level.liquidity_key for level in binance_levels} == {
            _expected_key(
                exchange=TEST_EXCHANGE,
                market_type=TEST_MARKET_TYPE,
                symbol=symbol,
                timeframe=timeframe,
            )
        }
        assert {level.liquidity_key for level in bybit_levels} == {
            _expected_key(
                exchange=ALT_EXCHANGE,
                market_type=ALT_MARKET_TYPE,
                symbol=symbol,
                timeframe=timeframe,
            )
        }


# ---------------------------------------------------------------------
# Stop clusters detector
# ---------------------------------------------------------------------


class TestStopClustersDetector:
    def test_builds_stop_cluster_from_close_buy_side_levels(
        self,
        stop_detector: StopClustersDetector,
        buy_side_levels: list[LiquidityLevel],
        orderbook_near_buy_side_cluster: dict[str, list[list[float]]],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = _scope_levels(
            buy_side_levels,
            symbol=symbol,
            timeframe=timeframe,
        )

        clusters = stop_detector.detect_from_levels(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            levels=levels,
            current_price=100.0,
            candles=_deterministic_equal_high_candles(
                symbol=symbol,
                timeframe=timeframe,
            ),
            orderbook=orderbook_near_buy_side_cluster,
        )

        buy_clusters = _clusters_by_side(clusters, LiquiditySide.BUY_SIDE)

        assert buy_clusters
        assert len(buy_clusters) == 1

        cluster = buy_clusters[0]

        assert cluster.center_price == pytest.approx(105.0, rel=0.01)
        assert cluster.low_price < cluster.center_price
        assert cluster.high_price > cluster.center_price
        assert cluster.source_levels
        assert len(cluster.source_levels) == len(levels)

        _assert_cluster_payload_contract(
            cluster,
            symbol=symbol,
            timeframe=timeframe,
        )

    def test_builds_stop_cluster_from_close_sell_side_levels(
        self,
        stop_detector: StopClustersDetector,
        sell_side_levels: list[LiquidityLevel],
        orderbook_near_sell_side_cluster: dict[str, list[list[float]]],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = _scope_levels(
            sell_side_levels,
            symbol=symbol,
            timeframe=timeframe,
        )

        clusters = stop_detector.detect_from_levels(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            levels=levels,
            current_price=100.0,
            candles=_deterministic_equal_low_candles(
                symbol=symbol,
                timeframe=timeframe,
            ),
            orderbook=orderbook_near_sell_side_cluster,
        )

        sell_clusters = _clusters_by_side(clusters, LiquiditySide.SELL_SIDE)

        assert sell_clusters
        assert len(sell_clusters) == 1

        cluster = sell_clusters[0]

        assert cluster.center_price == pytest.approx(95.0, rel=0.01)
        assert cluster.low_price < cluster.center_price
        assert cluster.high_price > cluster.center_price
        assert cluster.source_levels
        assert len(cluster.source_levels) == len(levels)

        _assert_cluster_payload_contract(
            cluster,
            symbol=symbol,
            timeframe=timeframe,
        )

    def test_merges_overlapping_candidates_into_single_cluster(
        self,
        stop_detector: StopClustersDetector,
        buy_side_levels: list[LiquidityLevel],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = _scope_levels(
            buy_side_levels,
            symbol=symbol,
            timeframe=timeframe,
        )

        clusters = stop_detector.detect_from_levels(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            levels=levels,
            current_price=100.0,
            candles=_deterministic_equal_high_candles(
                symbol=symbol,
                timeframe=timeframe,
            ),
            orderbook=None,
        )

        assert len(_clusters_by_side(clusters, LiquiditySide.BUY_SIDE)) == 1

    def test_keeps_buy_and_sell_side_clusters_separate(
        self,
        stop_detector: StopClustersDetector,
        mixed_side_levels: list[LiquidityLevel],
        balanced_orderbook: dict[str, list[list[float]]],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = _scope_levels(
            mixed_side_levels,
            symbol=symbol,
            timeframe=timeframe,
        )

        clusters = stop_detector.detect_from_levels(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            levels=levels,
            current_price=100.0,
            candles=_deterministic_two_sided_candles(
                symbol=symbol,
                timeframe=timeframe,
            ),
            orderbook=balanced_orderbook,
        )

        buy_clusters = _clusters_by_side(clusters, LiquiditySide.BUY_SIDE)
        sell_clusters = _clusters_by_side(clusters, LiquiditySide.SELL_SIDE)

        assert buy_clusters
        assert sell_clusters
        assert all(cluster.side == LiquiditySide.BUY_SIDE for cluster in buy_clusters)
        assert all(cluster.side == LiquiditySide.SELL_SIDE for cluster in sell_clusters)

    def test_returns_empty_for_empty_levels(
        self,
        stop_detector: StopClustersDetector,
        symbol: str,
        timeframe: str,
    ) -> None:
        clusters = stop_detector.detect_from_levels(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            levels=[],
            current_price=100.0,
            candles=[],
            orderbook=None,
        )

        assert clusters == []

    @pytest.mark.parametrize("bad_price", [0.0, -1.0, None, "not-a-price"])
    def test_rejects_invalid_current_price(
        self,
        stop_detector: StopClustersDetector,
        buy_side_levels: list[LiquidityLevel],
        symbol: str,
        timeframe: str,
        bad_price: Any,
    ) -> None:
        with pytest.raises(ValueError, match="current_price"):
            stop_detector.detect_from_levels(
                exchange=TEST_EXCHANGE,
                market_type=TEST_MARKET_TYPE,
                symbol=symbol,
                timeframe=timeframe,
                levels=buy_side_levels,
                current_price=bad_price,
                candles=[],
                orderbook=None,
            )

    def test_rejects_missing_symbol(
        self,
        stop_detector: StopClustersDetector,
        buy_side_levels: list[LiquidityLevel],
        timeframe: str,
    ) -> None:
        with pytest.raises(ValueError, match="symbol"):
            stop_detector.detect_from_levels(
                exchange=TEST_EXCHANGE,
                market_type=TEST_MARKET_TYPE,
                symbol="",
                timeframe=timeframe,
                levels=buy_side_levels,
                current_price=100.0,
                candles=[],
                orderbook=None,
            )

    def test_rejects_missing_timeframe_or_normalizes_to_default_contract(
        self,
        stop_detector: StopClustersDetector,
        buy_side_levels: list[LiquidityLevel],
        symbol: str,
    ) -> None:
        try:
            clusters = stop_detector.detect_from_levels(
                exchange=TEST_EXCHANGE,
                market_type=TEST_MARKET_TYPE,
                symbol=symbol,
                timeframe="",
                levels=_scope_levels(
                    buy_side_levels,
                    symbol=symbol,
                    timeframe=DEFAULT_TIMEFRAME,
                ),
                current_price=100.0,
                candles=[],
                orderbook=None,
            )
        except ValueError as exc:
            assert "timeframe" in str(exc)
            return

        assert all(cluster.timeframe == DEFAULT_TIMEFRAME for cluster in clusters)

    def test_respects_disabled_config(
        self,
        disabled_liquidity_config: LiquidityConfig,
        scorer: LiquidityScorer,
        buy_side_levels: list[LiquidityLevel],
        symbol: str,
        timeframe: str,
    ) -> None:
        detector = StopClustersDetector(
            config=disabled_liquidity_config,
            scorer=scorer,
        )

        clusters = detector.detect_from_levels(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            levels=buy_side_levels,
            current_price=100.0,
            candles=[],
            orderbook=None,
        )

        assert clusters == []

    def test_ignores_invalidated_and_expired_levels(
        self,
        stop_detector: StopClustersDetector,
        invalidated_buy_side_level: LiquidityLevel,
        expired_sell_side_level: LiquidityLevel,
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = _scope_levels(
            [invalidated_buy_side_level, expired_sell_side_level],
            symbol=symbol,
            timeframe=timeframe,
        )

        clusters = stop_detector.detect_from_levels(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            levels=levels,
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
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = _scope_levels(
            [
                partially_swept_buy_side_level,
                *buy_side_levels,
            ],
            symbol=symbol,
            timeframe=timeframe,
        )

        clusters = stop_detector.detect_from_levels(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            levels=levels,
            current_price=100.0,
            candles=_deterministic_equal_high_candles(
                symbol=symbol,
                timeframe=timeframe,
            ),
            orderbook=None,
        )

        assert clusters

        cluster = _clusters_by_side(clusters, LiquiditySide.BUY_SIDE)[0]

        assert any(level.is_partially_swept() for level in cluster.source_levels)
        assert cluster.swept_at is not None or any(
            level.swept_at is not None for level in cluster.source_levels
        )
        _assert_cluster_payload_contract(
            cluster,
            symbol=symbol,
            timeframe=timeframe,
        )

    def test_swept_source_levels_are_preserved_in_cluster_context(
        self,
        stop_detector: StopClustersDetector,
        buy_side_levels: list[LiquidityLevel],
        swept_buy_side_level: LiquidityLevel,
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = _scope_levels(
            [
                swept_buy_side_level,
                *buy_side_levels,
            ],
            symbol=symbol,
            timeframe=timeframe,
        )

        clusters = stop_detector.detect_from_levels(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            levels=levels,
            current_price=100.0,
            candles=_deterministic_equal_high_candles(
                symbol=symbol,
                timeframe=timeframe,
            ),
            orderbook=None,
        )

        assert clusters

        cluster = _clusters_by_side(clusters, LiquiditySide.BUY_SIDE)[0]

        assert any(level.is_swept() for level in cluster.source_levels)
        assert cluster.swept_at is not None or any(
            level.swept_at is not None for level in cluster.source_levels
        )

    def test_orderbook_near_cluster_increases_or_preserves_density(
        self,
        stop_detector: StopClustersDetector,
        buy_side_levels: list[LiquidityLevel],
        orderbook_near_buy_side_cluster: dict[str, list[list[float]]],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = _scope_levels(
            buy_side_levels,
            symbol=symbol,
            timeframe=timeframe,
        )

        without_orderbook = stop_detector.detect_from_levels(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            levels=deepcopy(levels),
            current_price=100.0,
            candles=_deterministic_equal_high_candles(
                symbol=symbol,
                timeframe=timeframe,
            ),
            orderbook=None,
        )

        with_orderbook = stop_detector.detect_from_levels(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            levels=deepcopy(levels),
            current_price=100.0,
            candles=_deterministic_equal_high_candles(
                symbol=symbol,
                timeframe=timeframe,
            ),
            orderbook=orderbook_near_buy_side_cluster,
        )

        assert without_orderbook
        assert with_orderbook

        base_cluster = _clusters_by_side(without_orderbook, LiquiditySide.BUY_SIDE)[0]
        enhanced_cluster = _clusters_by_side(with_orderbook, LiquiditySide.BUY_SIDE)[0]

        assert enhanced_cluster.estimated_stop_density >= base_cluster.estimated_stop_density
        assert enhanced_cluster.confidence >= base_cluster.confidence

    def test_string_numeric_orderbook_payload_is_accepted(
        self,
        stop_detector: StopClustersDetector,
        buy_side_levels: list[LiquidityLevel],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = _scope_levels(
            buy_side_levels,
            symbol=symbol,
            timeframe=timeframe,
        )

        orderbook = {
            "bids": [["99.8", "8.0"], ["99.5", "10.0"]],
            "asks": [["104.9", "45.0"], ["105.0", "60.0"], ["105.1", "42.0"]],
        }

        clusters = stop_detector.detect_from_levels(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            levels=levels,
            current_price="100.0",
            candles=_deterministic_equal_high_candles(
                symbol=symbol,
                timeframe=timeframe,
            ),
            orderbook=orderbook,
        )

        assert clusters
        for cluster in clusters:
            _assert_cluster_payload_contract(
                cluster,
                symbol=symbol,
                timeframe=timeframe,
            )

    def test_build_stop_zones_returns_merged_price_ranges(
        self,
        stop_detector: StopClustersDetector,
        buy_side_levels: list[LiquidityLevel],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = _scope_levels(
            buy_side_levels,
            symbol=symbol,
            timeframe=timeframe,
        )

        zones = stop_detector.build_stop_zones(levels)

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
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = _scope_levels(
            [invalidated_buy_side_level, expired_sell_side_level],
            symbol=symbol,
            timeframe=timeframe,
        )

        zones = stop_detector.build_stop_zones(levels)

        assert zones == []

    def test_far_apart_same_side_levels_create_separate_clusters(
        self,
        stop_detector: StopClustersDetector,
        buy_side_levels: list[LiquidityLevel],
        symbol: str,
        timeframe: str,
    ) -> None:
        near_levels = _scope_levels(
            buy_side_levels,
            symbol=symbol,
            timeframe=timeframe,
        )

        far_levels = [
            _move_level_price(near_levels[0], 120.00),
            _move_level_price(near_levels[1], 120.05),
        ]

        clusters = stop_detector.detect_from_levels(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            levels=[*near_levels, *far_levels],
            current_price=100.0,
            candles=_deterministic_equal_high_candles(
                symbol=symbol,
                timeframe=timeframe,
            ),
            orderbook=None,
        )

        buy_clusters = _clusters_by_side(clusters, LiquiditySide.BUY_SIDE)

        assert len(buy_clusters) >= 2

    def test_same_price_different_exchange_is_rescoped_to_requested_detector_scope(
        self,
        stop_detector: StopClustersDetector,
        buy_side_levels: list[LiquidityLevel],
        symbol: str,
        timeframe: str,
    ) -> None:
        stale_levels = _scope_levels(
            buy_side_levels,
            exchange=ALT_EXCHANGE,
            market_type=ALT_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
        )

        clusters = stop_detector.detect_from_levels(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            levels=stale_levels,
            current_price=100.0,
            candles=_deterministic_equal_high_candles(
                symbol=symbol,
                timeframe=timeframe,
            ),
            orderbook=None,
        )

        assert clusters

        for cluster in clusters:
            _assert_cluster_payload_contract(
                cluster,
                exchange=TEST_EXCHANGE,
                market_type=TEST_MARKET_TYPE,
                symbol=symbol,
                timeframe=timeframe,
            )

        for level in stale_levels:
            assert level.exchange == TEST_EXCHANGE
            assert level.market_type == TEST_MARKET_TYPE

    def test_same_symbol_timeframe_clusters_are_isolated_by_detector_scope(
        self,
        stop_detector: StopClustersDetector,
        buy_side_levels: list[LiquidityLevel],
        symbol: str,
        timeframe: str,
    ) -> None:
        binance_levels = _scope_levels(
            buy_side_levels,
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
        )
        bybit_levels = _scope_levels(
            buy_side_levels,
            exchange=ALT_EXCHANGE,
            market_type=ALT_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
        )

        binance_clusters = stop_detector.detect_from_levels(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            levels=binance_levels,
            current_price=100.0,
            candles=_deterministic_equal_high_candles(
                symbol=symbol,
                timeframe=timeframe,
            ),
            orderbook=None,
        )
        bybit_clusters = stop_detector.detect_from_levels(
            exchange=ALT_EXCHANGE,
            market_type=ALT_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            levels=bybit_levels,
            current_price=100.0,
            candles=_deterministic_equal_high_candles(
                symbol=symbol,
                timeframe=timeframe,
            ),
            orderbook=None,
        )

        assert binance_clusters
        assert bybit_clusters

        assert {cluster.liquidity_key for cluster in binance_clusters} == {
            _expected_key(
                exchange=TEST_EXCHANGE,
                market_type=TEST_MARKET_TYPE,
                symbol=symbol,
                timeframe=timeframe,
            )
        }
        assert {cluster.liquidity_key for cluster in bybit_clusters} == {
            _expected_key(
                exchange=ALT_EXCHANGE,
                market_type=ALT_MARKET_TYPE,
                symbol=symbol,
                timeframe=timeframe,
            )
        }

    def test_duplicate_levels_do_not_create_duplicate_source_context_explosion(
        self,
        stop_detector: StopClustersDetector,
        buy_side_levels: list[LiquidityLevel],
        symbol: str,
        timeframe: str,
    ) -> None:
        levels = _scope_levels(
            [
                deepcopy(buy_side_levels[0]),
                deepcopy(buy_side_levels[0]),
                deepcopy(buy_side_levels[1]),
            ],
            symbol=symbol,
            timeframe=timeframe,
        )

        clusters = stop_detector.detect_from_levels(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            levels=levels,
            current_price=100.0,
            candles=_deterministic_equal_high_candles(
                symbol=symbol,
                timeframe=timeframe,
            ),
            orderbook=None,
        )

        assert clusters

        cluster = _clusters_by_side(clusters, LiquiditySide.BUY_SIDE)[0]

        assert len(cluster.source_levels) <= len(levels)
        _assert_cluster_payload_contract(
            cluster,
            symbol=symbol,
            timeframe=timeframe,
        )

    def test_detect_from_equal_levels_wrapper_matches_detect_from_levels_when_available(
        self,
        stop_detector: StopClustersDetector,
        equal_high_level: EqualLevel,
        symbol: str,
        timeframe: str,
    ) -> None:
        if not hasattr(stop_detector, "detect_from_equal_levels"):
            pytest.skip("StopClustersDetector.detect_from_equal_levels is not available")

        levels = [
            _scope_level(
                deepcopy(equal_high_level),
                symbol=symbol,
                timeframe=timeframe,
            ),
            _scope_level(
                deepcopy(equal_high_level),
                symbol=symbol,
                timeframe=timeframe,
            ),
            _scope_level(
                deepcopy(equal_high_level),
                symbol=symbol,
                timeframe=timeframe,
            ),
        ]

        levels[0].price = 104.95
        levels[1].price = 105.00
        levels[2].price = 105.05

        from_levels = stop_detector.detect_from_levels(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            levels=deepcopy(levels),
            current_price=100.0,
            candles=_deterministic_equal_high_candles(
                symbol=symbol,
                timeframe=timeframe,
            ),
            orderbook=None,
        )

        from_equal_levels = stop_detector.detect_from_equal_levels(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            equal_levels=deepcopy(levels),
            current_price=100.0,
            candles=_deterministic_equal_high_candles(
                symbol=symbol,
                timeframe=timeframe,
            ),
            orderbook=None,
        )

        assert len(from_equal_levels) == len(from_levels)

        if from_levels:
            assert from_equal_levels[0].side == from_levels[0].side
            assert from_equal_levels[0].center_price == pytest.approx(
                from_levels[0].center_price
            )

    def test_detector_does_not_require_event_bus_or_scheduler(
        self,
        liquidity_config: LiquidityConfig,
        scorer: LiquidityScorer,
        buy_side_levels: list[LiquidityLevel],
        symbol: str,
        timeframe: str,
    ) -> None:
        detector = StopClustersDetector(
            config=liquidity_config,
            scorer=scorer,
        )

        clusters = detector.detect_from_levels(
            exchange=TEST_EXCHANGE,
            market_type=TEST_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe,
            levels=_scope_levels(
                buy_side_levels,
                symbol=symbol,
                timeframe=timeframe,
            ),
            current_price=100.0,
            candles=[],
            orderbook=None,
        )

        assert clusters
        assert not hasattr(detector, "_event_bus")
        assert not hasattr(detector, "_scheduler")