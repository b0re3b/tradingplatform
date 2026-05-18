# tests/analytics/funding/test_funding_detectors_logic.py

from __future__ import annotations

from typing import Callable

import pytest

from analytics.funding.enums import (
    FundingBias,
    FundingDataSource,
    FundingDivergenceType,
    FundingExtremeType,
    FundingFlipType,
    FundingPressureDirection,
    FundingPressureLevel,
    FundingRegime,
    FundingTimeframe,
)
from analytics.funding.funding_divergence import (
    FundingDivergenceConfig,
    FundingDivergenceDetector,
)
from analytics.funding.funding_extremes import (
    FundingExtremesConfig,
    FundingExtremesDetector,
)
from analytics.funding.funding_flip_detector import (
    FundingFlipDetector,
    FundingFlipDetectorConfig,
)
from analytics.funding.funding_pressure import (
    FundingPressureAnalyzer,
    FundingPressureConfig,
)
from analytics.funding.funding_regime_detector import (
    FundingRegimeDetector,
    FundingRegimeDetectorConfig,
)
from analytics.funding.models import (
    FundingDivergenceEvent,
    FundingExtremeEvent,
    FundingFlipEvent,
    FundingPressureState,
    FundingRegimeState,
    FundingSnapshot,
    FundingStatistics,
    funding_key_to_dict,
)


# =============================================================================
# Local helpers
# =============================================================================

def _assert_common_scope(model: object, snapshot: FundingSnapshot) -> None:
    assert getattr(model, "symbol") == snapshot.symbol
    assert getattr(model, "exchange") == snapshot.exchange
    assert getattr(model, "market_type") == snapshot.market_type
    assert getattr(model, "timeframe") == snapshot.timeframe
    assert getattr(model, "exchange_symbol") == snapshot.exchange_symbol
    assert getattr(model, "key") == snapshot.key


def _assert_probability(value: float | None) -> None:
    assert value is not None
    assert 0.0 <= value <= 1.0


def _assert_confidence(value: float) -> None:
    assert 0.0 <= value <= 1.0


def _make_statistics_for_rate(
    make_statistics: Callable[..., FundingStatistics],
    *,
    rate: float,
    percentile: float,
    zscore: float,
    sample_size: int = 100,
) -> FundingStatistics:
    return make_statistics(
        current_rate=rate,
        percentile=percentile,
        zscore=zscore,
        sample_size=sample_size,
        mean_rate=0.0,
        median_rate=0.0,
        std_rate=abs(rate) / max(abs(zscore), 1.0) if zscore else abs(rate),
        min_rate=min(rate, 0.0),
        max_rate=max(rate, 0.0),
    )


# =============================================================================
# FundingRegimeDetector
# =============================================================================

@pytest.mark.parametrize(
    (
        "funding_rate",
        "percentile",
        "zscore",
        "expected_regime",
        "expected_bias",
    ),
    [
        (0.0, 50.0, 0.0, FundingRegime.NEUTRAL, FundingBias.NEUTRAL),
        (0.000009, 50.0, 0.0, FundingRegime.NEUTRAL, FundingBias.NEUTRAL),
        (-0.000009, 50.0, 0.0, FundingRegime.NEUTRAL, FundingBias.NEUTRAL),
        (0.00005, 55.0, 0.1, FundingRegime.POSITIVE, FundingBias.LONG_BIAS),
        (-0.00005, 45.0, -0.1, FundingRegime.NEGATIVE, FundingBias.SHORT_BIAS),
        (0.00012, 85.0, 1.0, FundingRegime.POSITIVE, FundingBias.OVERCROWDED_LONGS),
        (-0.00012, 15.0, -1.0, FundingRegime.NEGATIVE, FundingBias.OVERCROWDED_SHORTS),
        (0.00012, 95.0, 1.0, FundingRegime.EXTREME_POSITIVE, FundingBias.SQUEEZE_RISK_LONGS),
        (-0.00012, 5.0, -1.0, FundingRegime.EXTREME_NEGATIVE, FundingBias.SQUEEZE_RISK_SHORTS),
        (0.00031, 70.0, 0.2, FundingRegime.EXTREME_POSITIVE, FundingBias.SQUEEZE_RISK_LONGS),
        (-0.00031, 30.0, -0.2, FundingRegime.EXTREME_NEGATIVE, FundingBias.SQUEEZE_RISK_SHORTS),
        (0.00008, 60.0, 2.0, FundingRegime.EXTREME_POSITIVE, FundingBias.SQUEEZE_RISK_LONGS),
        (-0.00008, 40.0, -2.0, FundingRegime.EXTREME_NEGATIVE, FundingBias.SQUEEZE_RISK_SHORTS),
    ],
)
def test_regime_detector_classifies_boundaries_and_extremes(
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    funding_rate: float,
    percentile: float,
    zscore: float,
    expected_regime: FundingRegime,
    expected_bias: FundingBias,
) -> None:
    detector = FundingRegimeDetector(
        FundingRegimeDetectorConfig(
            neutral_abs_threshold=0.00001,
            positive_abs_threshold=0.00005,
            extreme_abs_threshold=0.00030,
            crowded_percentile_threshold=85.0,
            squeeze_percentile_threshold=95.0,
            extreme_positive_zscore=2.0,
            extreme_negative_zscore=-2.0,
            min_confidence_for_change=0.15,
        )
    )

    snapshot = make_snapshot(
        funding_rate=funding_rate,
        market_type="usdm_futures",
        exchange_symbol="BTC/USDT:USDT",
    )
    statistics = _make_statistics_for_rate(
        make_statistics,
        rate=funding_rate,
        percentile=percentile,
        zscore=zscore,
        sample_size=100,
    )

    state = detector.detect(snapshot=snapshot, statistics=statistics)

    _assert_common_scope(state, snapshot)
    assert state.regime == expected_regime
    assert state.bias == expected_bias
    assert state.current_rate == pytest.approx(funding_rate)
    assert state.percentile == pytest.approx(percentile)
    assert state.zscore == pytest.approx(zscore)
    assert state.changed is False
    assert state.previous_regime is None
    _assert_confidence(state.confidence)
    assert state.metadata["sample_size"] == 100
    assert state.metadata["funding_sign"] == snapshot.funding_sign
    assert state.metadata["scope"] == funding_key_to_dict(snapshot.key)


def test_regime_detector_marks_change_only_with_previous_state_and_sufficient_confidence(
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    make_regime_state: Callable[..., FundingRegimeState],
) -> None:
    detector = FundingRegimeDetector(
        FundingRegimeDetectorConfig(min_confidence_for_change=0.20)
    )

    previous = make_regime_state(
        regime=FundingRegime.NEUTRAL,
        bias=FundingBias.NEUTRAL,
        confidence=0.8,
    )
    snapshot = make_snapshot(funding_rate=0.00035)
    statistics = _make_statistics_for_rate(
        make_statistics,
        rate=0.00035,
        percentile=98.0,
        zscore=3.0,
        sample_size=100,
    )

    state = detector.detect(
        snapshot=snapshot,
        statistics=statistics,
        previous_state=previous,
    )

    assert state.regime == FundingRegime.EXTREME_POSITIVE
    assert state.previous_regime == FundingRegime.NEUTRAL
    assert state.changed is True
    assert state.confidence >= detector.config.min_confidence_for_change


def test_regime_detector_does_not_mark_change_without_previous_state(
    regime_detector: FundingRegimeDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
) -> None:
    snapshot = make_snapshot(funding_rate=0.00035)
    statistics = _make_statistics_for_rate(
        make_statistics,
        rate=0.00035,
        percentile=98.0,
        zscore=3.0,
        sample_size=100,
    )

    state = regime_detector.detect(snapshot=snapshot, statistics=statistics)

    assert state.regime == FundingRegime.EXTREME_POSITIVE
    assert state.previous_regime is None
    assert state.changed is False


def test_regime_detector_does_not_mark_change_when_regime_is_same(
    regime_detector: FundingRegimeDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    make_regime_state: Callable[..., FundingRegimeState],
) -> None:
    previous = make_regime_state(
        regime=FundingRegime.POSITIVE,
        bias=FundingBias.LONG_BIAS,
    )
    snapshot = make_snapshot(funding_rate=0.00008)
    statistics = _make_statistics_for_rate(
        make_statistics,
        rate=0.00008,
        percentile=65.0,
        zscore=0.5,
        sample_size=100,
    )

    state = regime_detector.detect(
        snapshot=snapshot,
        statistics=statistics,
        previous_state=previous,
    )

    assert state.regime == FundingRegime.POSITIVE
    assert state.previous_regime == FundingRegime.POSITIVE
    assert state.changed is False


def test_regime_detector_suppresses_change_when_confidence_is_too_low(
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    make_regime_state: Callable[..., FundingRegimeState],
) -> None:
    detector = FundingRegimeDetector(
        FundingRegimeDetectorConfig(min_confidence_for_change=0.95)
    )

    previous = make_regime_state(
        regime=FundingRegime.NEUTRAL,
        bias=FundingBias.NEUTRAL,
    )
    snapshot = make_snapshot(funding_rate=0.00006)
    statistics = _make_statistics_for_rate(
        make_statistics,
        rate=0.00006,
        percentile=55.0,
        zscore=0.1,
        sample_size=10,
    )

    state = detector.detect(
        snapshot=snapshot,
        statistics=statistics,
        previous_state=previous,
    )

    assert state.regime == FundingRegime.POSITIVE
    assert state.confidence < detector.config.min_confidence_for_change
    assert state.changed is False


def test_regime_detector_confidence_is_clamped_to_unit_interval() -> None:
    detector = FundingRegimeDetector()

    assert detector.calculate_confidence(
        current_rate=0.01,
        percentile=100.0,
        zscore=20.0,
        sample_size=10_000,
    ) == pytest.approx(1.0)

    assert detector.calculate_confidence(
        current_rate=0.0,
        percentile=None,
        zscore=None,
        sample_size=0,
    ) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("previous_regime", "new_regime", "confidence", "expected"),
    [
        (FundingRegime.NEUTRAL, FundingRegime.POSITIVE, 0.01, False),
        (FundingRegime.NEUTRAL, FundingRegime.POSITIVE, 0.99, True),
        (FundingRegime.POSITIVE, FundingRegime.POSITIVE, 0.99, False),
    ],
)
def test_regime_detector_has_regime_changed_helper_is_strict(
    regime_detector: FundingRegimeDetector,
    make_regime_state: Callable[..., FundingRegimeState],
    previous_regime: FundingRegime,
    new_regime: FundingRegime,
    confidence: float,
    expected: bool,
) -> None:
    previous = make_regime_state(regime=previous_regime)

    assert regime_detector.has_regime_changed(
        previous_state=previous,
        new_regime=new_regime,
        confidence=confidence,
    ) is expected


# =============================================================================
# FundingPressureAnalyzer
# =============================================================================

def test_pressure_analyzer_returns_low_neutral_pressure_when_context_is_missing(
    pressure_analyzer: FundingPressureAnalyzer,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    make_regime_state: Callable[..., FundingRegimeState],
) -> None:
    snapshot = make_snapshot(
        funding_rate=0.0,
        open_interest=None,
        mark_price=None,
    )
    statistics = _make_statistics_for_rate(
        make_statistics,
        rate=0.0,
        percentile=50.0,
        zscore=0.0,
        sample_size=100,
    )
    regime_state = make_regime_state(
        regime=FundingRegime.NEUTRAL,
        bias=FundingBias.NEUTRAL,
        current_rate=0.0,
    )

    state = pressure_analyzer.analyze(
        snapshot=snapshot,
        statistics=statistics,
        regime_state=regime_state,
        previous_open_interest=None,
        current_price=None,
        previous_price=None,
    )

    _assert_common_scope(state, snapshot)
    assert state.level == FundingPressureLevel.LOW
    assert state.direction == FundingPressureDirection.NEUTRAL
    assert state.bias == FundingBias.NEUTRAL
    assert state.oi_confirmation is False
    assert state.price_stall_confirmation is False
    _assert_probability(state.pressure_score)
    _assert_probability(state.squeeze_probability)
    _assert_probability(state.mean_reversion_probability)
    assert state.metadata["oi_change_pct"] is None
    assert state.metadata["price_change_pct"] is None


@pytest.mark.parametrize(
    ("score", "expected_level"),
    [
        (0.0, FundingPressureLevel.LOW),
        (0.4499, FundingPressureLevel.LOW),
        (0.45, FundingPressureLevel.MODERATE),
        (0.6999, FundingPressureLevel.MODERATE),
        (0.70, FundingPressureLevel.HIGH),
        (0.8999, FundingPressureLevel.HIGH),
        (0.90, FundingPressureLevel.EXTREME),
        (1.0, FundingPressureLevel.EXTREME),
    ],
)
def test_pressure_level_boundaries_are_not_off_by_one(
    pressure_analyzer: FundingPressureAnalyzer,
    score: float,
    expected_level: FundingPressureLevel,
) -> None:
    assert pressure_analyzer._detect_pressure_level(score) == expected_level


def test_pressure_analyzer_detects_extreme_long_crowding_with_oi_growth_and_price_stall(
    pressure_analyzer: FundingPressureAnalyzer,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    make_regime_state: Callable[..., FundingRegimeState],
) -> None:
    snapshot = make_snapshot(
        funding_rate=0.00035,
        open_interest=1_100_000.0,
        mark_price=50_010.0,
    )
    statistics = _make_statistics_for_rate(
        make_statistics,
        rate=0.00035,
        percentile=98.0,
        zscore=3.0,
        sample_size=100,
    )
    regime_state = make_regime_state(
        regime=FundingRegime.EXTREME_POSITIVE,
        bias=FundingBias.SQUEEZE_RISK_LONGS,
        current_rate=0.00035,
        confidence=0.95,
    )

    state = pressure_analyzer.analyze(
        snapshot=snapshot,
        statistics=statistics,
        regime_state=regime_state,
        previous_open_interest=1_000_000.0,
        current_price=50_010.0,
        previous_price=50_000.0,
    )

    assert state.direction == FundingPressureDirection.LONG
    assert state.level == FundingPressureLevel.EXTREME
    assert state.bias == FundingBias.SQUEEZE_RISK_LONGS
    assert state.oi_confirmation is True
    assert state.price_stall_confirmation is True
    assert state.metadata["oi_change_pct"] == pytest.approx(0.10)
    assert state.metadata["price_change_pct"] == pytest.approx(0.0002)
    assert pressure_analyzer.is_high_pressure(state) is True
    assert pressure_analyzer.is_long_crowded(state) is True
    assert pressure_analyzer.is_short_crowded(state) is False
    assert pressure_analyzer.is_squeeze_risk(state, threshold=0.65) is True
    assert "BTCUSDT" in pressure_analyzer.build_summary(state)


def test_pressure_analyzer_detects_extreme_short_crowding_with_oi_growth_and_price_stall(
    pressure_analyzer: FundingPressureAnalyzer,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    make_regime_state: Callable[..., FundingRegimeState],
) -> None:
    snapshot = make_snapshot(
        funding_rate=-0.00035,
        open_interest=1_100_000.0,
        mark_price=49_990.0,
    )
    statistics = _make_statistics_for_rate(
        make_statistics,
        rate=-0.00035,
        percentile=2.0,
        zscore=-3.0,
        sample_size=100,
    )
    regime_state = make_regime_state(
        regime=FundingRegime.EXTREME_NEGATIVE,
        bias=FundingBias.SQUEEZE_RISK_SHORTS,
        current_rate=-0.00035,
        confidence=0.95,
    )

    state = pressure_analyzer.analyze(
        snapshot=snapshot,
        statistics=statistics,
        regime_state=regime_state,
        previous_open_interest=1_000_000.0,
        current_price=49_990.0,
        previous_price=50_000.0,
    )

    assert state.direction == FundingPressureDirection.SHORT
    assert state.level == FundingPressureLevel.EXTREME
    assert state.bias == FundingBias.SQUEEZE_RISK_SHORTS
    assert pressure_analyzer.is_high_pressure(state) is True
    assert pressure_analyzer.is_short_crowded(state) is True
    assert pressure_analyzer.is_long_crowded(state) is False


def test_pressure_analyzer_resolves_direction_from_neutral_regime_when_rate_is_extreme(
    pressure_analyzer: FundingPressureAnalyzer,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    make_regime_state: Callable[..., FundingRegimeState],
) -> None:
    snapshot = make_snapshot(
        funding_rate=0.00035,
        open_interest=1_100_000.0,
        mark_price=50_010.0,
    )
    statistics = _make_statistics_for_rate(
        make_statistics,
        rate=0.00035,
        percentile=98.0,
        zscore=3.0,
        sample_size=100,
    )
    regime_state = make_regime_state(
        regime=FundingRegime.NEUTRAL,
        bias=FundingBias.NEUTRAL,
        current_rate=0.00035,
        confidence=0.50,
    )

    state = pressure_analyzer.analyze(
        snapshot=snapshot,
        statistics=statistics,
        regime_state=regime_state,
        previous_open_interest=1_000_000.0,
        current_price=50_010.0,
        previous_price=50_000.0,
    )

    assert state.direction == FundingPressureDirection.LONG
    assert state.bias == FundingBias.OVERCROWDED_LONGS
    assert state.level in {FundingPressureLevel.HIGH, FundingPressureLevel.EXTREME}


def test_pressure_analyzer_handles_missing_oi_and_price_without_fake_confirmation(
    pressure_analyzer: FundingPressureAnalyzer,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    make_regime_state: Callable[..., FundingRegimeState],
) -> None:
    snapshot = make_snapshot(
        funding_rate=0.00012,
        open_interest=None,
        mark_price=None,
    )
    statistics = _make_statistics_for_rate(
        make_statistics,
        rate=0.00012,
        percentile=80.0,
        zscore=1.0,
        sample_size=100,
    )
    regime_state = make_regime_state(
        regime=FundingRegime.POSITIVE,
        bias=FundingBias.LONG_BIAS,
        current_rate=0.00012,
    )

    state = pressure_analyzer.analyze(
        snapshot=snapshot,
        statistics=statistics,
        regime_state=regime_state,
        previous_open_interest=None,
        current_price=None,
        previous_price=None,
    )

    assert state.direction == FundingPressureDirection.LONG
    assert state.oi_confirmation is False
    assert state.price_stall_confirmation is False
    assert state.metadata["oi_change_pct"] is None
    assert state.metadata["price_change_pct"] is None


def test_pressure_analyzer_config_can_make_high_pressure_harder_to_reach(
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    make_regime_state: Callable[..., FundingRegimeState],
) -> None:
    analyzer = FundingPressureAnalyzer(
        FundingPressureConfig(
            high_pressure_score_threshold=0.95,
            extreme_pressure_score_threshold=0.99,
        )
    )

    snapshot = make_snapshot(
        funding_rate=0.00035,
        open_interest=1_100_000.0,
        mark_price=50_010.0,
    )
    statistics = _make_statistics_for_rate(
        make_statistics,
        rate=0.00035,
        percentile=98.0,
        zscore=3.0,
        sample_size=100,
    )
    regime_state = make_regime_state(
        regime=FundingRegime.EXTREME_POSITIVE,
        bias=FundingBias.SQUEEZE_RISK_LONGS,
        current_rate=0.00035,
    )

    state = analyzer.analyze(
        snapshot=snapshot,
        statistics=statistics,
        regime_state=regime_state,
        previous_open_interest=1_000_000.0,
        current_price=50_010.0,
        previous_price=50_000.0,
    )

    assert analyzer.is_high_pressure(state) is False


# =============================================================================
# FundingFlipDetector
# =============================================================================

def test_flip_detector_returns_none_without_previous_snapshot(
    flip_detector: FundingFlipDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
) -> None:
    current = make_snapshot(funding_rate=0.0001)
    statistics = _make_statistics_for_rate(
        make_statistics,
        rate=0.0001,
        percentile=80.0,
        zscore=1.2,
    )

    event = flip_detector.detect(
        current_snapshot=current,
        previous_snapshot=None,
        statistics=statistics,
    )

    assert event is None


@pytest.mark.parametrize(
    ("previous_rate", "current_rate"),
    [
        (0.00008, 0.00012),
        (-0.00008, -0.00012),
        (0.0, 0.000005),
        (-0.000005, 0.000005),
        (0.000001, -0.000001),
    ],
)
def test_flip_detector_returns_none_without_meaningful_sign_change(
    flip_detector: FundingFlipDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    previous_rate: float,
    current_rate: float,
) -> None:
    previous = make_snapshot(funding_rate=previous_rate)
    current = make_snapshot(funding_rate=current_rate)
    statistics = _make_statistics_for_rate(
        make_statistics,
        rate=current_rate,
        percentile=50.0,
        zscore=0.0,
    )

    event = flip_detector.detect(
        current_snapshot=current,
        previous_snapshot=previous,
        statistics=statistics,
    )

    assert event is None


@pytest.mark.parametrize(
    ("previous_rate", "current_rate", "expected_type", "bullish", "bearish"),
    [
        (-0.00010, 0.00020, FundingFlipType.NEGATIVE_TO_POSITIVE, True, False),
        (0.00010, -0.00020, FundingFlipType.POSITIVE_TO_NEGATIVE, False, True),
    ],
)
def test_flip_detector_detects_meaningful_flips_and_direction_helpers(
    flip_detector: FundingFlipDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    previous_rate: float,
    current_rate: float,
    expected_type: FundingFlipType,
    bullish: bool,
    bearish: bool,
) -> None:
    previous = make_snapshot(funding_rate=previous_rate)
    current = make_snapshot(funding_rate=current_rate)
    statistics = _make_statistics_for_rate(
        make_statistics,
        rate=current_rate,
        percentile=90.0 if current_rate > 0 else 10.0,
        zscore=2.0 if current_rate > 0 else -2.0,
    )

    event = flip_detector.detect(
        current_snapshot=current,
        previous_snapshot=previous,
        statistics=statistics,
    )

    assert event is not None
    _assert_common_scope(event, current)
    assert event.flip_type == expected_type
    assert event.previous_rate == pytest.approx(previous_rate)
    assert event.current_rate == pytest.approx(current_rate)
    assert event.flip_magnitude == pytest.approx(abs(current_rate - previous_rate))
    _assert_confidence(event.confidence)
    assert flip_detector.is_bullish_flip(event) is bullish
    assert flip_detector.is_bearish_flip(event) is bearish
    assert expected_type.value in flip_detector.build_summary(event)
    assert event.metadata["scope"] == funding_key_to_dict(current.key)


def test_flip_detector_min_flip_magnitude_is_enforced(
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
) -> None:
    detector = FundingFlipDetector(
        FundingFlipDetectorConfig(min_flip_magnitude=0.00020)
    )

    previous = make_snapshot(funding_rate=-0.00005)
    current = make_snapshot(funding_rate=0.00006)
    statistics = _make_statistics_for_rate(
        make_statistics,
        rate=0.00006,
        percentile=60.0,
        zscore=0.5,
    )

    assert detector.detect(
        current_snapshot=current,
        previous_snapshot=previous,
        statistics=statistics,
    ) is None


def test_flip_detector_confidence_is_clamped() -> None:
    detector = FundingFlipDetector()

    confidence = detector.calculate_confidence(
        previous_rate=-0.01,
        current_rate=0.01,
        percentile=100.0,
        zscore=20.0,
    )

    assert confidence == pytest.approx(1.0)


def test_flip_detector_helpers_return_false_for_none(
    flip_detector: FundingFlipDetector,
) -> None:
    assert flip_detector.is_bullish_flip(None) is False
    assert flip_detector.is_bearish_flip(None) is False


# =============================================================================
# FundingExtremesDetector
# =============================================================================

@pytest.mark.parametrize(
    ("rate", "percentile", "zscore", "expected_type", "expected_reversal", "expected_squeeze"),
    [
        (0.00035, 50.0, 0.0, FundingExtremeType.GLOBAL_HIGH, True, True),
        (-0.00035, 50.0, 0.0, FundingExtremeType.GLOBAL_LOW, True, True),
        (0.00012, 98.0, 0.5, FundingExtremeType.PERCENTILE_HIGH, True, True),
        (-0.00012, 2.0, -0.5, FundingExtremeType.PERCENTILE_LOW, True, True),
        (0.00012, 50.0, 3.0, FundingExtremeType.ZSCORE_HIGH, True, True),
        (-0.00012, 50.0, -3.0, FundingExtremeType.ZSCORE_LOW, True, True),
    ],
)
def test_extremes_detector_detects_absolute_percentile_and_zscore_extremes(
    extremes_detector: FundingExtremesDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    make_regime_state: Callable[..., FundingRegimeState],
    rate: float,
    percentile: float,
    zscore: float,
    expected_type: FundingExtremeType,
    expected_reversal: bool,
    expected_squeeze: bool,
) -> None:
    snapshot = make_snapshot(funding_rate=rate)
    statistics = _make_statistics_for_rate(
        make_statistics,
        rate=rate,
        percentile=percentile,
        zscore=zscore,
        sample_size=100,
    )
    regime_state = make_regime_state(
        regime=FundingRegime.EXTREME_POSITIVE if rate > 0 else FundingRegime.EXTREME_NEGATIVE,
        bias=FundingBias.SQUEEZE_RISK_LONGS if rate > 0 else FundingBias.SQUEEZE_RISK_SHORTS,
        current_rate=rate,
    )

    event = extremes_detector.detect(
        snapshot=snapshot,
        statistics=statistics,
        regime_state=regime_state,
    )

    assert event is not None
    _assert_common_scope(event, snapshot)
    assert event.extreme_type == expected_type
    assert event.funding_rate == pytest.approx(rate)
    assert event.percentile == pytest.approx(percentile)
    assert event.zscore == pytest.approx(zscore)
    assert event.is_reversal_risk is expected_reversal
    assert event.is_squeeze_risk is expected_squeeze
    _assert_confidence(event.severity)
    assert event.metadata["scope"] == funding_key_to_dict(snapshot.key)
    assert expected_type.value in extremes_detector.build_summary(event)


def test_extremes_detector_returns_none_for_normal_funding_context(
    extremes_detector: FundingExtremesDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    make_regime_state: Callable[..., FundingRegimeState],
) -> None:
    snapshot = make_snapshot(funding_rate=0.00003)
    statistics = _make_statistics_for_rate(
        make_statistics,
        rate=0.00003,
        percentile=50.0,
        zscore=0.0,
        sample_size=100,
    )
    regime_state = make_regime_state(
        regime=FundingRegime.NEUTRAL,
        bias=FundingBias.NEUTRAL,
        current_rate=0.00003,
    )

    event = extremes_detector.detect(
        snapshot=snapshot,
        statistics=statistics,
        regime_state=regime_state,
    )

    assert event is None


def test_extremes_detector_min_sample_size_is_enforced(
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    make_regime_state: Callable[..., FundingRegimeState],
) -> None:
    detector = FundingExtremesDetector(
        FundingExtremesConfig(min_samples=50)
    )

    snapshot = make_snapshot(funding_rate=0.00035)
    statistics = _make_statistics_for_rate(
        make_statistics,
        rate=0.00035,
        percentile=99.0,
        zscore=4.0,
        sample_size=10,
    )
    regime_state = make_regime_state(
        regime=FundingRegime.EXTREME_POSITIVE,
        bias=FundingBias.SQUEEZE_RISK_LONGS,
        current_rate=0.00035,
    )

    event = detector.detect(
        snapshot=snapshot,
        statistics=statistics,
        regime_state=regime_state,
    )

    assert event is None


def test_extremes_detector_severity_is_clamped() -> None:
    detector = FundingExtremesDetector()

    severity = detector.calculate_severity(
        funding_rate=0.01,
        percentile=100.0,
        zscore=20.0,
        extreme_type=FundingExtremeType.PERCENTILE_HIGH,
    )

    assert severity == pytest.approx(1.0)


def test_extremes_detector_helpers_return_false_for_none(
    extremes_detector: FundingExtremesDetector,
) -> None:
    assert extremes_detector.is_high_extreme(None) is False
    assert extremes_detector.is_low_extreme(None) is False
    assert extremes_detector.is_reversal_risk(None) is False
    assert extremes_detector.is_squeeze_risk(None) is False


# =============================================================================
# FundingDivergenceDetector
# =============================================================================

@pytest.mark.parametrize(
    (
        "funding_rate",
        "price_change_pct",
        "oi_change_pct",
        "cvd_change",
        "long_liquidations",
        "short_liquidations",
        "expected_type",
        "bullish",
        "bearish",
    ),
    [
        (
            -0.00012,
            0.006,
            None,
            None,
            None,
            None,
            FundingDivergenceType.PRICE_UP_FUNDING_DOWN,
            True,
            False,
        ),
        (
            0.00012,
            -0.006,
            None,
            None,
            None,
            None,
            FundingDivergenceType.PRICE_DOWN_FUNDING_UP,
            False,
            True,
        ),
        (
            -0.00012,
            None,
            0.02,
            None,
            None,
            None,
            FundingDivergenceType.OI_UP_FUNDING_DOWN,
            True,
            False,
        ),
        (
            0.00012,
            0.0001,
            0.02,
            None,
            None,
            None,
            FundingDivergenceType.OI_UP_FUNDING_UP_PRICE_STALLED,
            False,
            True,
        ),
        (
            -0.00012,
            None,
            None,
            10_000.0,
            None,
            None,
            FundingDivergenceType.CVD_UP_FUNDING_DOWN,
            True,
            False,
        ),
        (
            0.00012,
            None,
            None,
            -10_000.0,
            None,
            None,
            FundingDivergenceType.CVD_DOWN_FUNDING_UP,
            False,
            True,
        ),
        (
            0.00012,
            None,
            None,
            None,
            100_000.0,
            None,
            FundingDivergenceType.LIQUIDATIONS_LONGS_WITH_POSITIVE_FUNDING,
            False,
            True,
        ),
        (
            -0.00012,
            None,
            None,
            None,
            None,
            100_000.0,
            FundingDivergenceType.LIQUIDATIONS_SHORTS_WITH_NEGATIVE_FUNDING,
            True,
            False,
        ),
    ],
)
def test_divergence_detector_detects_supported_divergence_types(
    divergence_detector: FundingDivergenceDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    funding_rate: float,
    price_change_pct: float | None,
    oi_change_pct: float | None,
    cvd_change: float | None,
    long_liquidations: float | None,
    short_liquidations: float | None,
    expected_type: FundingDivergenceType,
    bullish: bool,
    bearish: bool,
) -> None:
    snapshot = make_snapshot(funding_rate=funding_rate)
    statistics = _make_statistics_for_rate(
        make_statistics,
        rate=funding_rate,
        percentile=98.0 if funding_rate > 0 else 2.0,
        zscore=3.0 if funding_rate > 0 else -3.0,
        sample_size=100,
    )

    event = divergence_detector.detect(
        snapshot=snapshot,
        statistics=statistics,
        price_change_pct=price_change_pct,
        oi_change_pct=oi_change_pct,
        cvd_change=cvd_change,
        long_liquidations=long_liquidations,
        short_liquidations=short_liquidations,
    )

    assert event is not None
    _assert_common_scope(event, snapshot)
    assert event.divergence_type == expected_type
    assert event.funding_rate == pytest.approx(funding_rate)
    assert event.price_change_pct == price_change_pct
    assert event.oi_change_pct == oi_change_pct
    assert event.cvd_change == cvd_change
    assert event.long_liquidations == long_liquidations
    assert event.short_liquidations == short_liquidations
    _assert_confidence(event.confidence)
    assert divergence_detector.is_bullish_divergence(event) is bullish
    assert divergence_detector.is_bearish_divergence(event) is bearish
    assert event.metadata["scope"] == funding_key_to_dict(snapshot.key)
    assert expected_type.value in divergence_detector.build_summary(event)


def test_divergence_detector_returns_none_without_external_contradiction(
    divergence_detector: FundingDivergenceDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
) -> None:
    snapshot = make_snapshot(funding_rate=0.00012)
    statistics = _make_statistics_for_rate(
        make_statistics,
        rate=0.00012,
        percentile=70.0,
        zscore=0.5,
        sample_size=100,
    )

    event = divergence_detector.detect(
        snapshot=snapshot,
        statistics=statistics,
        price_change_pct=0.006,
        oi_change_pct=None,
        cvd_change=10_000.0,
        long_liquidations=None,
        short_liquidations=None,
    )

    assert event is None


def test_divergence_detector_min_funding_abs_is_enforced(
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
) -> None:
    detector = FundingDivergenceDetector(
        FundingDivergenceConfig(min_funding_abs=0.00020)
    )

    snapshot = make_snapshot(funding_rate=0.00012)
    statistics = _make_statistics_for_rate(
        make_statistics,
        rate=0.00012,
        percentile=98.0,
        zscore=3.0,
        sample_size=100,
    )

    event = detector.detect(
        snapshot=snapshot,
        statistics=statistics,
        price_change_pct=-0.01,
        oi_change_pct=None,
        cvd_change=None,
        long_liquidations=None,
        short_liquidations=None,
    )

    assert event is None


def test_divergence_detector_priority_prefers_price_divergence_over_other_inputs(
    divergence_detector: FundingDivergenceDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
) -> None:
    snapshot = make_snapshot(funding_rate=0.00012)
    statistics = _make_statistics_for_rate(
        make_statistics,
        rate=0.00012,
        percentile=98.0,
        zscore=3.0,
        sample_size=100,
    )

    event = divergence_detector.detect(
        snapshot=snapshot,
        statistics=statistics,
        price_change_pct=-0.01,
        oi_change_pct=0.05,
        cvd_change=-50_000.0,
        long_liquidations=200_000.0,
        short_liquidations=None,
    )

    assert event is not None
    assert event.divergence_type == FundingDivergenceType.PRICE_DOWN_FUNDING_UP


def test_divergence_detector_confidence_is_clamped() -> None:
    detector = FundingDivergenceDetector()

    confidence = detector.calculate_confidence(
        funding_rate=0.01,
        divergence_type=FundingDivergenceType.PRICE_DOWN_FUNDING_UP,
        percentile=100.0,
        zscore=20.0,
        price_change_pct=-0.50,
        oi_change_pct=1.0,
        cvd_change=-1_000_000.0,
        long_liquidations=10_000_000.0,
        short_liquidations=None,
    )

    assert confidence == pytest.approx(1.0)


def test_divergence_detector_helpers_return_false_for_none(
    divergence_detector: FundingDivergenceDetector,
) -> None:
    assert divergence_detector.is_bullish_divergence(None) is False
    assert divergence_detector.is_bearish_divergence(None) is False


# =============================================================================
# Cross-detector scope / purity contracts
# =============================================================================

def test_all_detectors_return_models_with_full_futures_scope(
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    make_regime_state: Callable[..., FundingRegimeState],
) -> None:
    snapshot = make_snapshot(
        exchange=FundingDataSource.BINANCE,
        market_type="usdm_futures",
        symbol="BTCUSDT",
        timeframe=FundingTimeframe.H1,
        exchange_symbol="BTC/USDT:USDT",
        funding_rate=0.00035,
        open_interest=1_100_000.0,
        mark_price=50_010.0,
    )
    statistics = _make_statistics_for_rate(
        make_statistics,
        rate=0.00035,
        percentile=98.0,
        zscore=3.0,
        sample_size=100,
    )

    regime_detector = FundingRegimeDetector()
    pressure_analyzer = FundingPressureAnalyzer()
    flip_detector = FundingFlipDetector()
    extremes_detector = FundingExtremesDetector()
    divergence_detector = FundingDivergenceDetector()

    previous_snapshot = make_snapshot(
        exchange=FundingDataSource.BINANCE,
        market_type="usdm_futures",
        symbol="BTCUSDT",
        timeframe=FundingTimeframe.H1,
        exchange_symbol="BTC/USDT:USDT",
        funding_rate=-0.00010,
        open_interest=1_000_000.0,
        mark_price=50_000.0,
    )

    previous_regime = make_regime_state(
        exchange=FundingDataSource.BINANCE,
        market_type="usdm_futures",
        symbol="BTCUSDT",
        timeframe=FundingTimeframe.H1,
        exchange_symbol="BTC/USDT:USDT",
        regime=FundingRegime.NEUTRAL,
        bias=FundingBias.NEUTRAL,
    )

    regime_state = regime_detector.detect(
        snapshot=snapshot,
        statistics=statistics,
        previous_state=previous_regime,
    )
    pressure_state = pressure_analyzer.analyze(
        snapshot=snapshot,
        statistics=statistics,
        regime_state=regime_state,
        previous_snapshot=previous_snapshot,
        previous_open_interest=1_000_000.0,
        current_price=50_010.0,
        previous_price=50_000.0,
    )
    flip_event = flip_detector.detect(
        current_snapshot=snapshot,
        previous_snapshot=previous_snapshot,
        statistics=statistics,
    )
    extreme_event = extremes_detector.detect(
        snapshot=snapshot,
        statistics=statistics,
        regime_state=regime_state,
    )
    divergence_event = divergence_detector.detect(
        snapshot=snapshot,
        statistics=statistics,
        price_change_pct=-0.006,
        oi_change_pct=0.10,
        cvd_change=-25_000.0,
        long_liquidations=150_000.0,
        short_liquidations=None,
    )

    _assert_common_scope(regime_state, snapshot)
    _assert_common_scope(pressure_state, snapshot)

    assert flip_event is not None
    assert extreme_event is not None
    assert divergence_event is not None

    _assert_common_scope(flip_event, snapshot)
    _assert_common_scope(extreme_event, snapshot)
    _assert_common_scope(divergence_event, snapshot)

    assert regime_state.metadata["scope"] == funding_key_to_dict(snapshot.key)
    assert pressure_state.metadata["scope"] == funding_key_to_dict(snapshot.key)
    assert flip_event.metadata["scope"] == funding_key_to_dict(snapshot.key)
    assert extreme_event.metadata["scope"] == funding_key_to_dict(snapshot.key)
    assert divergence_event.metadata["scope"] == funding_key_to_dict(snapshot.key)


def test_detectors_are_pure_components_without_eventbus_or_scheduler_attributes() -> None:
    detectors = [
        FundingRegimeDetector(),
        FundingPressureAnalyzer(),
        FundingFlipDetector(),
        FundingExtremesDetector(),
        FundingDivergenceDetector(),
    ]

    for detector in detectors:
        assert not hasattr(detector, "event_bus")
        assert not hasattr(detector, "scheduler")
        assert not hasattr(detector, "register")
        assert not hasattr(detector, "start")
        assert not hasattr(detector, "stop")