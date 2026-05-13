# tests/analytics/open_interest/test_oi_feature_builder.py

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

import pytest

from analytics.open_interest.config import (
    OIAnalyzerConfig,
    OIMaintenanceConfig,
    OIThresholds,
    OIWindows,
)
from analytics.open_interest.enums import OIDirection
from analytics.open_interest.models import OIMarketContext, OISnapshot
from analytics.open_interest.oi_features import (
    OIFeatureBuilder,
    OISeriesInput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EPS = 1e-9


def assert_close(actual: float | None, expected: float | None, *, rel: float = 1e-9) -> None:
    if expected is None:
        assert actual is None
        return

    assert actual is not None
    assert math.isfinite(actual)
    assert actual == pytest.approx(expected, rel=rel, abs=EPS)


def pct_change(current: float, previous: float) -> float:
    return ((current - previous) / abs(previous)) * 100.0


def liquidation_imbalance(long_liq: float, short_liq: float) -> float:
    total = long_liq + short_liq
    if total <= 0:
        return 0.0
    return (short_liq - long_liq) / total


def aggressive_flow_imbalance(buy: float, sell: float) -> float:
    total = buy + sell
    if total <= 0:
        return 0.0
    return (buy - sell) / total


def moving_average(values: Sequence[float], window: int) -> float:
    subset = list(values)[-window:]
    return sum(subset) / len(subset)


def population_std(values: Sequence[float], window: int) -> float:
    subset = list(values)[-window:]
    return statistics.pstdev(subset)


def zscore(values: Sequence[float], window: int) -> float:
    subset = list(values)[-window:]
    mu = sum(subset) / len(subset)
    sigma = statistics.pstdev(subset)
    return (subset[-1] - mu) / sigma


def snapshot(
    *,
    symbol: str = "btcusdt",
    exchange: str = "BINANCE",
    timestamp: float = 1_700_000_060.0,
    oi: float = 1_180.0,
) -> OISnapshot:
    return OISnapshot(
        symbol=symbol,
        exchange=exchange,
        timestamp=timestamp,
        oi=oi,
    )


def market_context(
    *,
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    timestamp: float = 1_700_000_060.0,
    price: float | None = 30_180.0,
    price_delta: float | None = None,
    price_delta_pct: float | None = None,
    volume: float | None = 1_700.0,
    volume_ma: float | None = None,
    volume_ratio: float | None = None,
    funding_rate: float | None = 0.0065,
    long_liquidations: float | None = 220.0,
    short_liquidations: float | None = 580.0,
    cvd_delta: float | None = 125.0,
    aggressive_buy_volume: float | None = 930.0,
    aggressive_sell_volume: float | None = 470.0,
) -> OIMarketContext:
    return OIMarketContext(
        symbol=symbol,
        exchange=exchange,
        timestamp=timestamp,
        price=price,
        price_delta=price_delta,
        price_delta_pct=price_delta_pct,
        volume=volume,
        volume_ma=volume_ma,
        volume_ratio=volume_ratio,
        funding_rate=funding_rate,
        long_liquidations=long_liquidations,
        short_liquidations=short_liquidations,
        cvd_delta=cvd_delta,
        aggressive_buy_volume=aggressive_buy_volume,
        aggressive_sell_volume=aggressive_sell_volume,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def config() -> OIAnalyzerConfig:
    return OIAnalyzerConfig(
        source_name="test_oi_feature_builder",
        require_price_context=False,
        require_volume_confirmation=True,
        thresholds=OIThresholds(
            min_oi_change_pct=0.25,
            min_price_change_pct=0.20,
            volume_confirmation_ratio=1.15,
            aggressive_flow_confirmation=0.10,
        ),
        windows=OIWindows(
            history_size=40,
            fast_window=3,
            slow_window=6,
            zscore_window=6,
            divergence_window=5,
            pressure_window=3,
            volume_window=4,
        ),
        maintenance=OIMaintenanceConfig(
            enable_periodic_cleanup=False,
            enable_metrics_emit=False,
        ),
    )


@pytest.fixture()
def builder(config: OIAnalyzerConfig) -> OIFeatureBuilder:
    return OIFeatureBuilder(config)


# ---------------------------------------------------------------------------
# Direct vulnerable method tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("current", "previous", "expected_delta", "expected_pct"),
    [
        (100.0, 90.0, 10.0, 11.11111111111111),
        (90.0, 100.0, -10.0, -10.0),
        (100.0, 0.0, 100.0, 0.0),       # zero previous OI must not explode
        (100.0, None, 0.0, 0.0),        # missing previous OI must degrade safely
        (None, 100.0, 0.0, 0.0),        # missing current OI must degrade safely
        ("105.5", "100.0", 5.5, 5.5),   # string numerics from raw feeds
        ("nan", 100.0, 0.0, 0.0),       # non-finite numeric strings ignored
        ("inf", 100.0, 0.0, 0.0),
    ],
)
def test_compute_oi_delta_and_pct_are_safe_for_dirty_numeric_inputs(
    builder: OIFeatureBuilder,
    current,
    previous,
    expected_delta: float,
    expected_pct: float,
) -> None:
    assert_close(builder.compute_oi_delta(current, previous), expected_delta)
    assert_close(builder.compute_oi_delta_pct(current, previous), expected_pct)


@pytest.mark.parametrize(
    ("current_volume", "volume_ma", "expected"),
    [
        (100.0, 50.0, 2.0),
        (100.0, 0.0, None),
        (100.0, -10.0, None),
        (None, 10.0, None),
        ("100.0", "25.0", 4.0),
        ("nan", 25.0, None),
        (100.0, "inf", None),
    ],
)
def test_compute_volume_ratio_rejects_zero_negative_missing_and_non_finite_inputs(
    builder: OIFeatureBuilder,
    current_volume,
    volume_ma,
    expected: float | None,
) -> None:
    assert_close(builder.compute_volume_ratio(current_volume, volume_ma), expected)


@pytest.mark.parametrize(
    ("long_liq", "short_liq", "expected"),
    [
        (0.0, 0.0, 0.0),
        (100.0, 300.0, 0.5),
        (300.0, 100.0, -0.5),
        (None, 100.0, None),
        (100.0, None, None),
        ("100", "300", 0.5),
        ("nan", 300.0, None),
    ],
)
def test_compute_liquidation_imbalance_handles_zero_total_and_dirty_values(
    builder: OIFeatureBuilder,
    long_liq,
    short_liq,
    expected: float | None,
) -> None:
    assert_close(builder.compute_liquidation_imbalance(long_liq, short_liq), expected)


@pytest.mark.parametrize(
    ("buy", "sell", "expected"),
    [
        (0.0, 0.0, 0.0),
        (750.0, 250.0, 0.5),
        (250.0, 750.0, -0.5),
        (None, 250.0, None),
        (250.0, None, None),
        ("750", "250", 0.5),
        ("inf", 250.0, None),
    ],
)
def test_compute_aggressive_flow_imbalance_handles_zero_total_and_dirty_values(
    builder: OIFeatureBuilder,
    buy,
    sell,
    expected: float | None,
) -> None:
    assert_close(builder.compute_aggressive_flow_imbalance(buy, sell), expected)


def test_compute_velocity_and_acceleration_reject_non_positive_time_delta(
    builder: OIFeatureBuilder,
) -> None:
    assert builder.compute_velocity([100.0, 110.0], [10.0, 10.0]) is None
    assert builder.compute_velocity([100.0, 110.0], [11.0, 10.0]) is None

    assert builder.compute_acceleration([100.0, 110.0, 130.0], [10.0, 10.0, 12.0]) is None
    assert builder.compute_acceleration([100.0, 110.0, 130.0], [10.0, 12.0, 11.0]) is None


# ---------------------------------------------------------------------------
# Build features happy-path with hard assertions
# ---------------------------------------------------------------------------

def test_build_features_from_full_context_calculates_all_core_metrics(
    builder: OIFeatureBuilder,
    config: OIAnalyzerConfig,
) -> None:
    oi_values = [1_000.0, 1_020.0, 1_010.0, 1_050.0, 1_080.0, 1_120.0, 1_180.0]
    oi_timestamps = [
        1_700_000_000.0,
        1_700_000_010.0,
        1_700_000_020.0,
        1_700_000_030.0,
        1_700_000_040.0,
        1_700_000_050.0,
        1_700_000_060.0,
    ]

    price_values = [30_000.0, 30_050.0, 30_020.0, 30_100.0, 30_120.0, 30_140.0, 30_180.0]
    price_timestamps = oi_timestamps[:]

    volume_values = [900.0, 1_000.0, 1_100.0, 1_300.0, 1_500.0, 1_600.0, 1_700.0]
    volume_timestamps = oi_timestamps[:]

    ctx = market_context(
        price=30_180.0,
        price_delta=None,
        price_delta_pct=None,
        volume=1_700.0,
        volume_ratio=None,
        long_liquidations=220.0,
        short_liquidations=580.0,
        aggressive_buy_volume=930.0,
        aggressive_sell_volume=470.0,
    )

    features = builder.build_from_raw_inputs(
        snapshot=snapshot(oi=1_180.0),
        context=ctx,
        oi_values=oi_values,
        oi_timestamps=oi_timestamps,
        price_values=price_values,
        price_timestamps=price_timestamps,
        volume_values=volume_values,
        volume_timestamps=volume_timestamps,
    )

    previous_oi = 1_120.0
    previous_price = 30_140.0
    expected_volume_ma = moving_average(volume_values, config.windows.volume_window)

    assert_close(features.oi, 1_180.0)
    assert_close(features.oi_delta, 60.0)
    assert_close(features.oi_delta_pct, pct_change(1_180.0, previous_oi))

    assert_close(features.oi_ma_fast, moving_average(oi_values, config.windows.fast_window))
    assert_close(features.oi_ma_slow, moving_average(oi_values, config.windows.slow_window))
    assert_close(features.oi_std, population_std(oi_values, config.windows.zscore_window))
    assert_close(features.oi_zscore, zscore(oi_values, config.windows.zscore_window))

    assert_close(features.oi_velocity, 6.0)
    assert_close(features.oi_acceleration, 0.2)

    assert_close(features.price, 30_180.0)
    assert_close(features.price_delta, 40.0)
    assert_close(features.price_delta_pct, pct_change(30_180.0, previous_price))

    assert_close(features.volume, 1_700.0)
    assert_close(features.volume_ma, expected_volume_ma)
    assert_close(features.volume_ratio, 1_700.0 / expected_volume_ma)

    assert_close(features.liquidation_imbalance, liquidation_imbalance(220.0, 580.0))
    assert_close(features.aggressive_flow_imbalance, aggressive_flow_imbalance(930.0, 470.0))
    assert_close(features.oi_change_per_volume, 60.0 / 1_700.0)
    assert_close(
        features.oi_price_efficiency,
        features.oi_delta_pct / features.price_delta_pct,
    )

    assert features.oi_direction is OIDirection.UP
    assert features.price_direction is OIDirection.UP
    assert features.oi_pressure_score is not None
    assert -1.0 <= features.oi_pressure_score <= 1.0


# ---------------------------------------------------------------------------
# Context precedence / fallback behavior
# ---------------------------------------------------------------------------

def test_context_price_delta_and_volume_ratio_override_series_derived_values(
    builder: OIFeatureBuilder,
) -> None:
    oi_values = [1_000.0, 1_100.0, 1_200.0]
    timestamps = [100.0, 110.0, 120.0]

    price_values = [10_000.0, 10_100.0, 10_200.0]
    volume_values = [100.0, 200.0, 300.0]

    ctx = market_context(
        price=12_000.0,
        price_delta=-999.0,
        price_delta_pct=-8.88,
        volume=9_999.0,
        volume_ratio=7.77,
    )

    features = builder.build_from_raw_inputs(
        snapshot=snapshot(timestamp=120.0, oi=1_200.0),
        context=ctx,
        oi_values=oi_values,
        oi_timestamps=timestamps,
        price_values=price_values,
        price_timestamps=timestamps,
        volume_values=volume_values,
        volume_timestamps=timestamps,
    )

    assert_close(features.price, 12_000.0)
    assert_close(features.price_delta, -999.0)
    assert_close(features.price_delta_pct, -8.88)
    assert_close(features.volume, 9_999.0)
    assert_close(features.volume_ratio, 7.77)
    assert features.price_direction is OIDirection.DOWN


def test_build_features_falls_back_to_series_when_context_is_missing(
    builder: OIFeatureBuilder,
) -> None:
    oi_values = [500.0, 525.0, 550.0]
    price_values = [100.0, 105.0, 111.0]
    volume_values = [10.0, 20.0, 40.0, 80.0]
    timestamps = [1.0, 2.0, 3.0]
    volume_timestamps = [1.0, 2.0, 3.0, 4.0]

    features = builder.build_from_raw_inputs(
        snapshot=snapshot(timestamp=3.0, oi=550.0),
        context=None,
        oi_values=oi_values,
        oi_timestamps=timestamps,
        price_values=price_values,
        price_timestamps=timestamps,
        volume_values=volume_values,
        volume_timestamps=volume_timestamps,
    )

    assert_close(features.price, 111.0)
    assert_close(features.price_delta, 6.0)
    assert_close(features.price_delta_pct, pct_change(111.0, 105.0))

    assert_close(features.volume, 80.0)
    assert_close(features.volume_ma, moving_average(volume_values, 4))
    assert_close(features.volume_ratio, 80.0 / moving_average(volume_values, 4))

    assert features.funding_rate is None
    assert features.liquidation_imbalance is None
    assert features.aggressive_flow_imbalance is None
    assert features.oi_pressure_score is not None


def test_build_minimal_features_produces_safe_bootstrap_features_without_context(
    builder: OIFeatureBuilder,
) -> None:
    features = builder.build_minimal_features(
        snapshot=snapshot(timestamp=2.0, oi=1_000.0),
        oi_values=[1_000.0],
        oi_timestamps=[2.0],
    )

    assert_close(features.oi, 1_000.0)
    assert_close(features.oi_delta, 0.0)
    assert_close(features.oi_delta_pct, 0.0)
    assert features.price is None
    assert features.price_delta is None
    assert features.price_delta_pct is None
    assert features.volume is None
    assert features.volume_ratio is None
    assert features.oi_velocity is None
    assert features.oi_acceleration is None
    assert features.oi_pressure_score == 0.0 or features.oi_pressure_score is None
    assert features.oi_direction is OIDirection.FLAT
    assert features.price_direction is OIDirection.UNKNOWN


# ---------------------------------------------------------------------------
# Dirty data / validation stress tests
# ---------------------------------------------------------------------------

def test_build_features_ignores_nan_inf_and_unparseable_numeric_values_without_desynchronizing_pairs(
    builder: OIFeatureBuilder,
) -> None:
    raw_oi_values = [
        1_000.0,
        "bad-value",
        1_050.0,
        float("nan"),
        1_100.0,
        float("inf"),
        1_200.0,
    ]
    raw_oi_timestamps = [
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
    ]

    raw_price_values = [100.0, "bad", 103.0, float("nan"), 106.0]
    raw_price_timestamps = [10.0, 20.0, 30.0, 40.0, 50.0]

    raw_volume_values = [10.0, "bad", 20.0, float("inf"), 40.0, 80.0]
    raw_volume_timestamps = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

    features = builder.build_from_raw_inputs(
        snapshot=snapshot(timestamp=70.0, oi=1_200.0),
        context=None,
        oi_values=raw_oi_values,
        oi_timestamps=raw_oi_timestamps,
        price_values=raw_price_values,
        price_timestamps=raw_price_timestamps,
        volume_values=raw_volume_values,
        volume_timestamps=raw_volume_timestamps,
    )

    cleaned_oi_values = [1_000.0, 1_050.0, 1_100.0, 1_200.0]
    cleaned_price_values = [100.0, 103.0, 106.0]
    cleaned_volume_values = [10.0, 20.0, 40.0, 80.0]

    assert_close(features.oi, 1_200.0)
    assert_close(features.oi_delta, 100.0)
    assert_close(features.oi_delta_pct, pct_change(1_200.0, 1_100.0))
    assert_close(features.price, 106.0)
    assert_close(features.price_delta, 3.0)
    assert_close(features.price_delta_pct, pct_change(106.0, 103.0))
    assert_close(features.volume, 80.0)
    assert_close(features.volume_ma, moving_average(cleaned_volume_values, 4))
    assert_close(features.volume_ratio, 80.0 / moving_average(cleaned_volume_values, 4))
    assert_close(features.oi_ma_fast, moving_average(cleaned_oi_values, 3))
    assert_close(features.oi_ma_slow, moving_average(cleaned_oi_values, 6))


@pytest.mark.parametrize(
    ("oi_values", "oi_timestamps", "message"),
    [
        ([100.0, 110.0], [1.0], "identical length"),
        ([100.0, 110.0, 120.0], [1.0, 3.0, 2.0], "chronological"),
    ],
)
def test_build_features_rejects_invalid_oi_series_shape_or_order(
    builder: OIFeatureBuilder,
    oi_values: list[float],
    oi_timestamps: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        builder.build_from_raw_inputs(
            snapshot=snapshot(timestamp=3.0, oi=120.0),
            context=None,
            oi_values=oi_values,
            oi_timestamps=oi_timestamps,
        )


@pytest.mark.parametrize(
    ("price_values", "price_timestamps"),
    [
        ([100.0, 105.0, 106.0], [1.0, 3.0, 2.0]),
        ([100.0, "bad", 106.0, 107.0], [1.0, 2.0, 4.0, 3.0]),
    ],
)
def test_build_features_rejects_non_chronological_price_series_after_cleaning(
    builder: OIFeatureBuilder,
    price_values,
    price_timestamps,
) -> None:
    with pytest.raises(ValueError, match="price_timestamps must be in chronological order"):
        builder.build_from_raw_inputs(
            snapshot=snapshot(timestamp=4.0, oi=120.0),
            context=None,
            oi_values=[100.0, 110.0, 120.0],
            oi_timestamps=[1.0, 2.0, 4.0],
            price_values=price_values,
            price_timestamps=price_timestamps,
        )


@pytest.mark.parametrize(
    ("volume_values", "volume_timestamps"),
    [
        ([10.0, 20.0, 30.0], [1.0, 3.0, 2.0]),
        ([10.0, "bad", 30.0, 40.0], [1.0, 2.0, 4.0, 3.0]),
    ],
)
def test_build_features_rejects_non_chronological_volume_series_after_cleaning(
    builder: OIFeatureBuilder,
    volume_values,
    volume_timestamps,
) -> None:
    with pytest.raises(ValueError, match="volume_timestamps must be in chronological order"):
        builder.build_from_raw_inputs(
            snapshot=snapshot(timestamp=4.0, oi=120.0),
            context=None,
            oi_values=[100.0, 110.0, 120.0],
            oi_timestamps=[1.0, 2.0, 4.0],
            volume_values=volume_values,
            volume_timestamps=volume_timestamps,
        )


def test_build_features_raises_when_all_oi_points_are_invalid_after_cleaning(
    builder: OIFeatureBuilder,
) -> None:
    with pytest.raises(ValueError, match="oi_values must contain at least one value"):
        builder.build_from_raw_inputs(
            snapshot=snapshot(timestamp=10.0, oi=100.0),
            context=None,
            oi_values=["bad", float("nan"), float("inf")],
            oi_timestamps=[1.0, 2.0, 3.0],
        )


def test_build_features_handles_zero_previous_oi_without_infinite_pct_or_efficiency(
    builder: OIFeatureBuilder,
) -> None:
    features = builder.build_from_raw_inputs(
        snapshot=snapshot(timestamp=2.0, oi=100.0),
        context=market_context(
            timestamp=2.0,
            price=101.0,
            price_delta=None,
            price_delta_pct=None,
            volume=50.0,
            volume_ratio=None,
            long_liquidations=0.0,
            short_liquidations=0.0,
            aggressive_buy_volume=0.0,
            aggressive_sell_volume=0.0,
        ),
        oi_values=[0.0, 100.0],
        oi_timestamps=[1.0, 2.0],
        price_values=[100.0, 101.0],
        price_timestamps=[1.0, 2.0],
        volume_values=[0.0, 50.0],
        volume_timestamps=[1.0, 2.0],
    )

    assert_close(features.oi_delta, 100.0)
    assert_close(features.oi_delta_pct, 0.0)
    assert_close(features.price_delta, 1.0)
    assert_close(features.price_delta_pct, 1.0)
    assert_close(features.liquidation_imbalance, 0.0)
    assert_close(features.aggressive_flow_imbalance, 0.0)
    assert_close(features.oi_change_per_volume, 2.0)
    assert_close(features.oi_price_efficiency, 0.0)

    assert math.isfinite(features.oi_delta)
    assert math.isfinite(features.oi_delta_pct)
    assert features.oi_pressure_score is not None
    assert -1.0 <= features.oi_pressure_score <= 1.0


def test_build_features_does_not_create_volume_ratio_when_volume_ma_is_zero(
    builder: OIFeatureBuilder,
) -> None:
    features = builder.build_from_raw_inputs(
        snapshot=snapshot(timestamp=3.0, oi=120.0),
        context=None,
        oi_values=[100.0, 110.0, 120.0],
        oi_timestamps=[1.0, 2.0, 3.0],
        volume_values=[0.0, 0.0, 0.0, 0.0],
        volume_timestamps=[1.0, 2.0, 3.0, 4.0],
    )

    assert_close(features.volume, 0.0)
    assert_close(features.volume_ma, 0.0)
    assert features.volume_ratio is None
    assert features.oi_change_per_volume is None


# ---------------------------------------------------------------------------
# Snapshot/current-value inference traps
# ---------------------------------------------------------------------------

def test_build_features_uses_snapshot_oi_as_current_even_when_history_tail_is_stale(
    builder: OIFeatureBuilder,
) -> None:
    """
    Analyzer normally appends snapshot.oi before calling the builder.
    This test intentionally violates that assumption to catch fragile logic:
    latest OI must be snapshot.oi, not blindly the last historical value.
    """

    features = builder.build_from_raw_inputs(
        snapshot=snapshot(timestamp=4.0, oi=1_500.0),
        context=None,
        oi_values=[1_000.0, 1_050.0, 1_100.0],
        oi_timestamps=[1.0, 2.0, 3.0],
    )

    assert_close(features.oi, 1_500.0)
    assert_close(features.oi_delta, 400.0)
    assert_close(features.oi_delta_pct, pct_change(1_500.0, 1_100.0))
    assert features.oi_direction is OIDirection.UP


def test_build_features_uses_second_last_history_value_as_previous_when_snapshot_is_already_tail(
    builder: OIFeatureBuilder,
) -> None:
    features = builder.build_from_raw_inputs(
        snapshot=snapshot(timestamp=4.0, oi=1_100.0),
        context=None,
        oi_values=[1_000.0, 1_050.0, 1_100.0],
        oi_timestamps=[1.0, 2.0, 4.0],
    )

    assert_close(features.oi, 1_100.0)
    assert_close(features.oi_delta, 50.0)
    assert_close(features.oi_delta_pct, pct_change(1_100.0, 1_050.0))
    assert features.oi_direction is OIDirection.UP


def test_build_features_can_represent_negative_pressure_without_overflow_or_wrong_direction(
    builder: OIFeatureBuilder,
) -> None:
    oi_values = [2_000.0, 1_970.0, 1_930.0, 1_860.0, 1_800.0, 1_720.0]
    price_values = [30_000.0, 29_900.0, 29_700.0, 29_500.0, 29_200.0, 28_900.0]
    volume_values = [500.0, 700.0, 1_000.0, 1_300.0, 1_700.0, 2_000.0]
    timestamps = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    features = builder.build_from_raw_inputs(
        snapshot=snapshot(timestamp=6.0, oi=1_720.0),
        context=market_context(
            timestamp=6.0,
            price=28_900.0,
            funding_rate=-0.012,
            long_liquidations=900.0,
            short_liquidations=100.0,
            aggressive_buy_volume=200.0,
            aggressive_sell_volume=900.0,
        ),
        oi_values=oi_values,
        oi_timestamps=timestamps,
        price_values=price_values,
        price_timestamps=timestamps,
        volume_values=volume_values,
        volume_timestamps=timestamps,
    )

    assert features.oi_direction is OIDirection.DOWN
    assert features.price_direction is OIDirection.DOWN
    assert features.oi_delta < 0
    assert features.oi_delta_pct < 0
    assert features.price_delta is not None and features.price_delta < 0
    assert features.price_delta_pct is not None and features.price_delta_pct < 0
    assert features.liquidation_imbalance is not None and features.liquidation_imbalance < 0
    assert features.aggressive_flow_imbalance is not None and features.aggressive_flow_imbalance < 0
    assert features.oi_pressure_score is not None
    assert -1.0 <= features.oi_pressure_score <= 1.0
    assert features.oi_pressure_score < 0


# ---------------------------------------------------------------------------
# Constructor-level OISeriesInput validation used by analyzer-facing method
# ---------------------------------------------------------------------------

def test_oi_series_input_rejects_mismatched_price_and_volume_pair_lengths() -> None:
    with pytest.raises(ValueError, match="price_values and price_timestamps"):
        OISeriesInput(
            oi_values=[1.0],
            oi_timestamps=[1.0],
            price_values=[1.0, 2.0],
            price_timestamps=[1.0],
        )

    with pytest.raises(ValueError, match="volume_values and volume_timestamps"):
        OISeriesInput(
            oi_values=[1.0],
            oi_timestamps=[1.0],
            volume_values=[1.0, 2.0],
            volume_timestamps=[1.0],
        )