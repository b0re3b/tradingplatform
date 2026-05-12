# tests/analytics/funding/test_funding_detectors_logic.py

from __future__ import annotations

from typing import Callable

import pytest

from analytics.funding.enums import (
    FundingBias,
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
)


# ---------------------------------------------------------------------------
# FundingRegimeDetector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("funding_rate", "percentile", "zscore", "expected_regime", "expected_bias"),
    [
        (
            0.0,
            50.0,
            0.0,
            FundingRegime.NEUTRAL,
            FundingBias.NEUTRAL,
        ),
        (
            0.00008,
            60.0,
            0.5,
            FundingRegime.POSITIVE,
            FundingBias.LONG_BIAS,
        ),
        (
            -0.00008,
            40.0,
            -0.5,
            FundingRegime.NEGATIVE,
            FundingBias.SHORT_BIAS,
        ),
        (
            0.00012,
            90.0,
            1.0,
            FundingRegime.POSITIVE,
            FundingBias.OVERCROWDED_LONGS,
        ),
        (
            -0.00012,
            10.0,
            -1.0,
            FundingRegime.NEGATIVE,
            FundingBias.OVERCROWDED_SHORTS,
        ),
        (
            0.00012,
            96.0,
            1.2,
            FundingRegime.EXTREME_POSITIVE,
            FundingBias.SQUEEZE_RISK_LONGS,
        ),
        (
            -0.00012,
            4.0,
            -1.2,
            FundingRegime.EXTREME_NEGATIVE,
            FundingBias.SQUEEZE_RISK_SHORTS,
        ),
        (
            0.00012,
            70.0,
            2.2,
            FundingRegime.EXTREME_POSITIVE,
            FundingBias.SQUEEZE_RISK_LONGS,
        ),
        (
            -0.00012,
            30.0,
            -2.2,
            FundingRegime.EXTREME_NEGATIVE,
            FundingBias.SQUEEZE_RISK_SHORTS,
        ),
    ],
)
def test_regime_detector_classifies_regime_and_bias(
    regime_detector: FundingRegimeDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    funding_rate: float,
    percentile: float,
    zscore: float,
    expected_regime: FundingRegime,
    expected_bias: FundingBias,
) -> None:
    snapshot = make_snapshot(funding_rate=funding_rate)
    statistics = make_statistics(
        current_rate=funding_rate,
        percentile=percentile,
        zscore=zscore,
        sample_size=100,
    )

    state = regime_detector.detect(snapshot=snapshot, statistics=statistics)

    assert state.symbol == snapshot.symbol
    assert state.exchange == snapshot.exchange
    assert state.timeframe == FundingTimeframe.H1
    assert state.regime == expected_regime
    assert state.bias == expected_bias
    assert state.current_rate == funding_rate
    assert state.percentile == percentile
    assert state.zscore == zscore
    assert 0.0 <= state.confidence <= 1.0
    assert state.changed is False
    assert state.previous_regime is None
    assert state.metadata["sample_size"] == 100
    assert state.metadata["funding_sign"] == snapshot.funding_sign


def test_regime_detector_detects_changed_when_previous_regime_differs(
    regime_detector: FundingRegimeDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    make_regime_state: Callable[..., FundingRegimeState],
) -> None:
    previous_state = make_regime_state(
        regime=FundingRegime.NEUTRAL,
        bias=FundingBias.NEUTRAL,
        confidence=0.8,
    )
    snapshot = make_snapshot(funding_rate=0.00035)
    statistics = make_statistics(
        current_rate=0.00035,
        percentile=98.0,
        zscore=3.0,
        sample_size=100,
    )

    state = regime_detector.detect(
        snapshot=snapshot,
        statistics=statistics,
        previous_state=previous_state,
    )

    assert state.regime == FundingRegime.EXTREME_POSITIVE
    assert state.previous_regime == FundingRegime.NEUTRAL
    assert state.changed is True
    assert state.confidence >= regime_detector.config.min_confidence_for_change


def test_regime_detector_does_not_mark_changed_without_previous_state(
    regime_detector: FundingRegimeDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
) -> None:
    snapshot = make_snapshot(funding_rate=0.00035)
    statistics = make_statistics(
        current_rate=0.00035,
        percentile=98.0,
        zscore=3.0,
        sample_size=100,
    )

    state = regime_detector.detect(snapshot=snapshot, statistics=statistics)

    assert state.regime == FundingRegime.EXTREME_POSITIVE
    assert state.changed is False
    assert state.previous_regime is None


def test_regime_detector_does_not_mark_changed_when_regime_is_same(
    regime_detector: FundingRegimeDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    make_regime_state: Callable[..., FundingRegimeState],
) -> None:
    previous_state = make_regime_state(
        regime=FundingRegime.POSITIVE,
        bias=FundingBias.LONG_BIAS,
    )
    snapshot = make_snapshot(funding_rate=0.00008)
    statistics = make_statistics(
        current_rate=0.00008,
        percentile=70.0,
        zscore=0.5,
        sample_size=100,
    )

    state = regime_detector.detect(
        snapshot=snapshot,
        statistics=statistics,
        previous_state=previous_state,
    )

    assert state.regime == FundingRegime.POSITIVE
    assert state.previous_regime == FundingRegime.POSITIVE
    assert state.changed is False


def test_regime_detector_suppresses_changed_when_confidence_is_too_low(
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    make_regime_state: Callable[..., FundingRegimeState],
) -> None:
    detector = FundingRegimeDetector(
        FundingRegimeDetectorConfig(min_confidence_for_change=0.95)
    )

    previous_state = make_regime_state(
        regime=FundingRegime.NEUTRAL,
        bias=FundingBias.NEUTRAL,
    )
    snapshot = make_snapshot(funding_rate=0.00006)
    statistics = make_statistics(
        current_rate=0.00006,
        percentile=55.0,
        zscore=0.1,
        sample_size=10,
    )

    state = detector.detect(
        snapshot=snapshot,
        statistics=statistics,
        previous_state=previous_state,
    )

    assert state.regime == FundingRegime.POSITIVE
    assert state.confidence < detector.config.min_confidence_for_change
    assert state.changed is False


def test_regime_detector_confidence_is_clamped() -> None:
    detector = FundingRegimeDetector()

    confidence = detector.calculate_confidence(
        current_rate=0.01,
        percentile=100.0,
        zscore=20.0,
        sample_size=1_000,
    )

    assert confidence == pytest.approx(1.0)


def test_regime_detector_direct_helper_for_low_confidence_change(
    regime_detector: FundingRegimeDetector,
    make_regime_state: Callable[..., FundingRegimeState],
) -> None:
    previous_state = make_regime_state(regime=FundingRegime.NEUTRAL)

    changed = regime_detector.has_regime_changed(
        previous_state=previous_state,
        new_regime=FundingRegime.POSITIVE,
        confidence=0.01,
    )

    assert changed is False


# ---------------------------------------------------------------------------
# FundingPressureAnalyzer
# ---------------------------------------------------------------------------


def test_pressure_analyzer_returns_low_pressure_for_neutral_context(
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
    statistics = make_statistics(
        current_rate=0.0,
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

    assert state.level == FundingPressureLevel.LOW
    assert state.direction == FundingPressureDirection.NEUTRAL
    assert state.bias == FundingBias.NEUTRAL
    assert state.oi_confirmation is False
    assert state.price_stall_confirmation is False
    assert 0.0 <= state.pressure_score <= 1.0
    assert 0.0 <= state.squeeze_probability <= 1.0
    assert 0.0 <= state.mean_reversion_probability <= 1.0
    assert state.metadata["oi_change_pct"] is None
    assert state.metadata["price_change_pct"] is None


def test_pressure_analyzer_detects_extreme_long_pressure_with_oi_and_price_stall(
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
    statistics = make_statistics(
        current_rate=0.00035,
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

    assert state.level == FundingPressureLevel.EXTREME
    assert state.direction == FundingPressureDirection.LONG
    assert state.bias == FundingBias.SQUEEZE_RISK_LONGS
    assert state.oi_confirmation is True
    assert state.price_stall_confirmation is True
    assert state.pressure_score >= pressure_analyzer.config.extreme_pressure_score_threshold
    assert pressure_analyzer.is_high_pressure(state) is True
    assert pressure_analyzer.is_long_crowded(state) is True
    assert pressure_analyzer.is_squeeze_risk(state, threshold=0.65) is True
    assert 0.0 <= state.squeeze_probability <= 1.0
    assert 0.0 <= state.mean_reversion_probability <= 1.0
    assert state.metadata["oi_change_pct"] == pytest.approx(0.10)
    assert state.metadata["price_change_pct"] == pytest.approx(0.0002)


def test_pressure_analyzer_detects_short_direction_from_negative_bias(
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
    statistics = make_statistics(
        current_rate=-0.00035,
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
    assert state.bias == FundingBias.SQUEEZE_RISK_SHORTS
    assert pressure_analyzer.is_high_pressure(state) is True
    assert pressure_analyzer.is_short_crowded(state) is True


def test_pressure_analyzer_resolves_bias_from_neutral_regime_when_pressure_is_high(
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
    statistics = make_statistics(
        current_rate=0.00035,
        percentile=98.0,
        zscore=3.0,
        sample_size=100,
    )
    regime_state = make_regime_state(
        regime=FundingRegime.NEUTRAL,
        bias=FundingBias.NEUTRAL,
        current_rate=0.00035,
        confidence=0.5,
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
    assert state.level in {FundingPressureLevel.HIGH, FundingPressureLevel.EXTREME}
    assert state.bias == FundingBias.OVERCROWDED_LONGS


def test_pressure_analyzer_handles_missing_context_without_crash(
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
    statistics = make_statistics(
        current_rate=0.00012,
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
    )

    assert state.direction == FundingPressureDirection.LONG
    assert state.oi_confirmation is False
    assert state.price_stall_confirmation is False
    assert state.metadata["oi_change_pct"] is None
    assert state.metadata["price_change_pct"] is None


@pytest.mark.parametrize(
    ("score", "expected_level"),
    [
        (0.0, FundingPressureLevel.LOW),
        (0.4499, FundingPressureLevel.LOW),
        (0.45, FundingPressureLevel.MODERATE),
        (0.70, FundingPressureLevel.HIGH),
        (0.90, FundingPressureLevel.EXTREME),
        (1.0, FundingPressureLevel.EXTREME),
    ],
)
def test_pressure_level_boundaries(
    pressure_analyzer: FundingPressureAnalyzer,
    score: float,
    expected_level: FundingPressureLevel,
) -> None:
    assert pressure_analyzer._detect_pressure_level(score) == expected_level


# ---------------------------------------------------------------------------
# FundingFlipDetector
# ---------------------------------------------------------------------------


def test_flip_detector_returns_none_without_previous_snapshot(
    flip_detector: FundingFlipDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
) -> None:
    current = make_snapshot(funding_rate=0.0001)
    statistics = make_statistics(current_rate=0.0001, percentile=80.0, zscore=1.2)

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
    ],
)
def test_flip_detector_returns_none_without_real_sign_change(
    flip_detector: FundingFlipDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    previous_rate: float,
    current_rate: float,
) -> None:
    previous = make_snapshot(funding_rate=previous_rate)
    current = make_snapshot(funding_rate=current_rate)
    statistics = make_statistics(
        current_rate=current_rate,
        percentile=50.0,
        zscore=0.0,
        sample_size=100,
    )

    event = flip_detector.detect(
        current_snapshot=current,
        previous_snapshot=previous,
        statistics=statistics,
    )

    assert event is None


@pytest.mark.parametrize(
    ("previous_rate", "current_rate", "expected_type", "is_bullish", "is_bearish"),
    [
        (
            -0.00010,
            0.00020,
            FundingFlipType.NEGATIVE_TO_POSITIVE,
            True,
            False,
        ),
        (
            0.00010,
            -0.00020,
            FundingFlipType.POSITIVE_TO_NEGATIVE,
            False,
            True,
        ),
    ],
)
def test_flip_detector_detects_meaningful_flip(
    flip_detector: FundingFlipDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    previous_rate: float,
    current_rate: float,
    expected_type: FundingFlipType,
    is_bullish: bool,
    is_bearish: bool,
) -> None:
    previous = make_snapshot(funding_rate=previous_rate)
    current = make_snapshot(funding_rate=current_rate)
    statistics = make_statistics(
        current_rate=current_rate,
        percentile=98.0 if current_rate > 0 else 2.0,
        zscore=3.0 if current_rate > 0 else -3.0,
        sample_size=100,
    )

    event = flip_detector.detect(
        current_snapshot=current,
        previous_snapshot=previous,
        statistics=statistics,
        extra_metadata={"test_case": "meaningful_flip"},
    )

    assert event is not None
    assert event.flip_type == expected_type
    assert event.previous_rate == previous_rate
    assert event.current_rate == current_rate
    assert event.flip_magnitude == pytest.approx(abs(current_rate - previous_rate))
    assert event.confidence >= flip_detector.config.min_confidence
    assert event.metadata["previous_sign"] == previous.funding_sign
    assert event.metadata["current_sign"] == current.funding_sign
    assert event.metadata["test_case"] == "meaningful_flip"
    assert flip_detector.is_bullish_flip(event) is is_bullish
    assert flip_detector.is_bearish_flip(event) is is_bearish
    assert current.symbol in flip_detector.build_summary(event)


def test_flip_detector_filters_flip_below_min_magnitude(
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
) -> None:
    detector = FundingFlipDetector(
        FundingFlipDetectorConfig(min_flip_magnitude=0.00020)
    )

    previous = make_snapshot(funding_rate=-0.00003)
    current = make_snapshot(funding_rate=0.00003)
    statistics = make_statistics(
        current_rate=0.00003,
        percentile=80.0,
        zscore=1.0,
        sample_size=100,
    )

    event = detector.detect(
        current_snapshot=current,
        previous_snapshot=previous,
        statistics=statistics,
    )

    assert event is None


def test_flip_detector_filters_low_confidence_flip(
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
) -> None:
    detector = FundingFlipDetector(
        FundingFlipDetectorConfig(min_confidence=0.99)
    )

    previous = make_snapshot(funding_rate=-0.00003)
    current = make_snapshot(funding_rate=0.00004)
    statistics = make_statistics(
        current_rate=0.00004,
        percentile=55.0,
        zscore=0.1,
        sample_size=20,
    )

    event = detector.detect(
        current_snapshot=current,
        previous_snapshot=previous,
        statistics=statistics,
    )

    assert event is None


def test_flip_detector_direct_flip_type_helper(
    flip_detector: FundingFlipDetector,
) -> None:
    assert (
        flip_detector.detect_flip_type(-0.00008, 0.00008)
        == FundingFlipType.NEGATIVE_TO_POSITIVE
    )
    assert (
        flip_detector.detect_flip_type(0.00008, -0.00008)
        == FundingFlipType.POSITIVE_TO_NEGATIVE
    )
    assert (
        flip_detector.detect_flip_type(0.00008, 0.00009)
        == FundingFlipType.NONE
    )
    assert (
        flip_detector.detect_flip_type(0.000001, -0.000001)
        == FundingFlipType.NONE
    )


def test_flip_detector_confidence_is_clamped(
    flip_detector: FundingFlipDetector,
) -> None:
    confidence = flip_detector.calculate_confidence(
        previous_rate=-0.01,
        current_rate=0.01,
        flip_magnitude=0.02,
        percentile=100.0,
        zscore=20.0,
    )

    assert confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# FundingExtremesDetector
# ---------------------------------------------------------------------------


def test_extremes_detector_skips_when_sample_size_too_low(
    extremes_detector: FundingExtremesDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    make_regime_state: Callable[..., FundingRegimeState],
) -> None:
    snapshot = make_snapshot(funding_rate=0.001)
    statistics = make_statistics(
        current_rate=0.001,
        min_rate=-0.001,
        max_rate=0.001,
        percentile=100.0,
        zscore=5.0,
        sample_size=5,
    )
    regime_state = make_regime_state(regime=FundingRegime.EXTREME_POSITIVE)

    event = extremes_detector.detect(
        snapshot=snapshot,
        statistics=statistics,
        regime_state=regime_state,
    )

    assert event is None


@pytest.mark.parametrize(
    ("current_rate", "min_rate", "max_rate", "expected_type"),
    [
        (
            0.00035,
            -0.00020,
            0.00035,
            FundingExtremeType.GLOBAL_HIGH,
        ),
        (
            -0.00035,
            -0.00035,
            0.00020,
            FundingExtremeType.GLOBAL_LOW,
        ),
    ],
)
def test_extremes_detector_detects_global_extremes_with_priority(
    extremes_detector: FundingExtremesDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    make_regime_state: Callable[..., FundingRegimeState],
    current_rate: float,
    min_rate: float,
    max_rate: float,
    expected_type: FundingExtremeType,
) -> None:
    snapshot = make_snapshot(funding_rate=current_rate)
    statistics = make_statistics(
        current_rate=current_rate,
        min_rate=min_rate,
        max_rate=max_rate,
        percentile=50.0,
        zscore=0.0,
        sample_size=100,
    )
    regime_state = make_regime_state(
        regime=FundingRegime.EXTREME_POSITIVE
        if current_rate > 0
        else FundingRegime.EXTREME_NEGATIVE
    )

    event = extremes_detector.detect(
        snapshot=snapshot,
        statistics=statistics,
        regime_state=regime_state,
    )

    assert event is not None
    assert event.extreme_type == expected_type
    assert event.funding_rate == current_rate
    assert event.severity >= extremes_detector.config.min_severity
    assert extremes_detector.is_high_severity(event, threshold=0.20) is True
    assert snapshot.symbol in extremes_detector.build_summary(event)


@pytest.mark.parametrize(
    ("current_rate", "percentile", "expected_type"),
    [
        (0.00020, 98.0, FundingExtremeType.PERCENTILE_HIGH),
        (-0.00020, 2.0, FundingExtremeType.PERCENTILE_LOW),
    ],
)
def test_extremes_detector_detects_percentile_extremes_when_not_global(
    extremes_detector: FundingExtremesDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    current_rate: float,
    percentile: float,
    expected_type: FundingExtremeType,
) -> None:
    snapshot = make_snapshot(funding_rate=current_rate)
    statistics = make_statistics(
        current_rate=current_rate,
        min_rate=-0.001,
        max_rate=0.001,
        percentile=percentile,
        zscore=0.0,
        sample_size=100,
    )

    event = extremes_detector.detect(
        snapshot=snapshot,
        statistics=statistics,
        regime_state=None,
    )

    assert event is not None
    assert event.extreme_type == expected_type
    assert event.percentile == percentile
    assert event.regime in {
        FundingRegime.POSITIVE,
        FundingRegime.NEGATIVE,
        FundingRegime.EXTREME_POSITIVE,
        FundingRegime.EXTREME_NEGATIVE,
    }


@pytest.mark.parametrize(
    ("current_rate", "zscore", "expected_type"),
    [
        (0.00020, 3.0, FundingExtremeType.ZSCORE_HIGH),
        (-0.00020, -3.0, FundingExtremeType.ZSCORE_LOW),
    ],
)
def test_extremes_detector_detects_zscore_extremes_when_not_global_or_percentile(
    extremes_detector: FundingExtremesDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    current_rate: float,
    zscore: float,
    expected_type: FundingExtremeType,
) -> None:
    snapshot = make_snapshot(funding_rate=current_rate)
    statistics = make_statistics(
        current_rate=current_rate,
        min_rate=-0.001,
        max_rate=0.001,
        percentile=50.0,
        zscore=zscore,
        sample_size=100,
    )

    event = extremes_detector.detect(
        snapshot=snapshot,
        statistics=statistics,
    )

    assert event is not None
    assert event.extreme_type == expected_type
    assert event.zscore == zscore


@pytest.mark.parametrize(
    ("current_rate", "expected_type"),
    [
        (0.00035, FundingExtremeType.LOCAL_HIGH),
        (-0.00035, FundingExtremeType.LOCAL_LOW),
    ],
)
def test_extremes_detector_detects_absolute_extremes_when_other_modes_disabled(
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    current_rate: float,
    expected_type: FundingExtremeType,
) -> None:
    detector = FundingExtremesDetector(
        FundingExtremesConfig(
            enable_local_extremes=False,
            enable_percentile_extremes=False,
            enable_zscore_extremes=False,
            enable_absolute_extremes=True,
        )
    )

    snapshot = make_snapshot(funding_rate=current_rate)
    statistics = make_statistics(
        current_rate=current_rate,
        min_rate=-0.001,
        max_rate=0.001,
        percentile=50.0,
        zscore=0.0,
        sample_size=100,
    )

    event = detector.detect(snapshot=snapshot, statistics=statistics)

    assert event is not None
    assert event.extreme_type == expected_type


def test_extremes_detector_filters_low_severity_extreme(
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
) -> None:
    detector = FundingExtremesDetector(
        FundingExtremesConfig(
            min_sample_size=20,
            min_severity=0.95,
        )
    )

    snapshot = make_snapshot(funding_rate=0.00012)
    statistics = make_statistics(
        current_rate=0.00012,
        min_rate=-0.001,
        max_rate=0.001,
        percentile=91.0,
        zscore=1.0,
        sample_size=100,
    )

    event = detector.detect(snapshot=snapshot, statistics=statistics)

    assert event is None


def test_extremes_detector_helper_methods(
    extremes_detector: FundingExtremesDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
    make_regime_state: Callable[..., FundingRegimeState],
) -> None:
    positive_snapshot = make_snapshot(funding_rate=0.00035)
    statistics = make_statistics(
        current_rate=0.00035,
        min_rate=-0.001,
        max_rate=0.001,
        percentile=98.0,
        zscore=3.0,
        sample_size=100,
    )
    regime_state = make_regime_state(
        regime=FundingRegime.EXTREME_POSITIVE,
        bias=FundingBias.SQUEEZE_RISK_LONGS,
    )

    event = extremes_detector.detect(
        snapshot=positive_snapshot,
        statistics=statistics,
        regime_state=regime_state,
    )

    assert event is not None
    assert extremes_detector.is_positive_extreme(event) is True
    assert extremes_detector.is_negative_extreme(event) is False
    assert extremes_detector.is_high_severity(event, threshold=0.50) is True
    assert event.is_reversal_risk is True
    assert event.is_squeeze_risk is True


def test_extremes_detector_severity_is_clamped(
    extremes_detector: FundingExtremesDetector,
) -> None:
    severity = extremes_detector.calculate_severity(
        current_rate=0.01,
        percentile=100.0,
        zscore=20.0,
        extreme_type=FundingExtremeType.GLOBAL_HIGH,
    )

    assert severity == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# FundingDivergenceDetector
# ---------------------------------------------------------------------------


def test_divergence_detector_returns_none_when_funding_is_too_small(
    divergence_detector: FundingDivergenceDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
) -> None:
    snapshot = make_snapshot(funding_rate=0.000001)
    statistics = make_statistics(
        current_rate=0.000001,
        percentile=50.0,
        zscore=0.0,
        sample_size=100,
    )

    event = divergence_detector.detect(
        snapshot=snapshot,
        statistics=statistics,
        price_change_pct=-0.01,
        oi_change_pct=0.02,
        cvd_change=-10_000.0,
        long_liquidations=100_000.0,
    )

    assert event is None


@pytest.mark.parametrize(
    (
        "funding_rate",
        "price_change_pct",
        "oi_change_pct",
        "cvd_change",
        "long_liquidations",
        "short_liquidations",
        "expected_type",
        "is_bullish",
        "is_bearish",
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
    is_bullish: bool,
    is_bearish: bool,
) -> None:
    snapshot = make_snapshot(funding_rate=funding_rate)
    statistics = make_statistics(
        current_rate=funding_rate,
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
        extra_metadata={"test_case": expected_type.value},
    )

    assert event is not None
    assert event.divergence_type == expected_type
    assert event.funding_rate == funding_rate
    assert event.confidence >= divergence_detector.config.min_confidence
    assert event.metadata["funding_sign"] == snapshot.funding_sign
    assert event.metadata["test_case"] == expected_type.value
    assert divergence_detector.is_bullish_divergence(event) is is_bullish
    assert divergence_detector.is_bearish_divergence(event) is is_bearish
    assert snapshot.symbol in divergence_detector.build_summary(event)


def test_divergence_detector_prioritizes_liquidation_divergence_over_price(
    divergence_detector: FundingDivergenceDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
) -> None:
    snapshot = make_snapshot(funding_rate=0.00012)
    statistics = make_statistics(
        current_rate=0.00012,
        percentile=98.0,
        zscore=3.0,
        sample_size=100,
    )

    event = divergence_detector.detect(
        snapshot=snapshot,
        statistics=statistics,
        price_change_pct=-0.01,
        long_liquidations=100_000.0,
    )

    assert event is not None
    assert (
        event.divergence_type
        == FundingDivergenceType.LIQUIDATIONS_LONGS_WITH_POSITIVE_FUNDING
    )


def test_divergence_detector_respects_disabled_price_divergence(
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
) -> None:
    detector = FundingDivergenceDetector(
        FundingDivergenceConfig(
            enable_price_funding_divergence=False,
            enable_oi_funding_divergence=False,
            enable_cvd_funding_divergence=False,
            enable_liquidation_funding_divergence=False,
        )
    )
    snapshot = make_snapshot(funding_rate=0.00012)
    statistics = make_statistics(
        current_rate=0.00012,
        percentile=98.0,
        zscore=3.0,
        sample_size=100,
    )

    event = detector.detect(
        snapshot=snapshot,
        statistics=statistics,
        price_change_pct=-0.01,
        oi_change_pct=0.02,
        cvd_change=-10_000.0,
        long_liquidations=100_000.0,
    )

    assert event is None


def test_divergence_detector_filters_low_confidence_divergence(
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
) -> None:
    detector = FundingDivergenceDetector(
        FundingDivergenceConfig(min_confidence=0.99)
    )
    snapshot = make_snapshot(funding_rate=0.00004)
    statistics = make_statistics(
        current_rate=0.00004,
        percentile=55.0,
        zscore=0.1,
        sample_size=20,
    )

    event = detector.detect(
        snapshot=snapshot,
        statistics=statistics,
        price_change_pct=-0.003,
    )

    assert event is None


def test_divergence_detector_direct_type_helper_returns_none_without_inputs(
    divergence_detector: FundingDivergenceDetector,
) -> None:
    divergence_type = divergence_detector.detect_divergence_type(
        funding_rate=0.00012,
        price_change_pct=None,
        oi_change_pct=None,
        cvd_change=None,
        long_liquidations=None,
        short_liquidations=None,
    )

    assert divergence_type == FundingDivergenceType.NONE


def test_divergence_detector_confidence_is_clamped(
    divergence_detector: FundingDivergenceDetector,
) -> None:
    confidence = divergence_detector.calculate_confidence(
        funding_rate=0.01,
        divergence_type=FundingDivergenceType.LIQUIDATIONS_LONGS_WITH_POSITIVE_FUNDING,
        percentile=100.0,
        zscore=20.0,
        price_change_pct=-0.50,
        oi_change_pct=1.0,
        cvd_change=-1_000_000.0,
        long_liquidations=10_000_000.0,
        short_liquidations=None,
    )

    assert confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Cross-detector sanity checks
# ---------------------------------------------------------------------------


def test_all_detector_outputs_keep_probability_like_values_in_valid_ranges(
    regime_detector: FundingRegimeDetector,
    pressure_analyzer: FundingPressureAnalyzer,
    flip_detector: FundingFlipDetector,
    extremes_detector: FundingExtremesDetector,
    divergence_detector: FundingDivergenceDetector,
    make_snapshot: Callable[..., FundingSnapshot],
    make_statistics: Callable[..., FundingStatistics],
) -> None:
    previous_snapshot = make_snapshot(funding_rate=-0.00012)
    current_snapshot = make_snapshot(
        funding_rate=0.00035,
        open_interest=1_100_000.0,
        mark_price=50_010.0,
    )
    statistics = make_statistics(
        current_rate=0.00035,
        min_rate=-0.00020,
        max_rate=0.00050,
        percentile=98.0,
        zscore=3.0,
        sample_size=100,
    )

    regime_state = regime_detector.detect(
        snapshot=current_snapshot,
        statistics=statistics,
    )
    pressure_state = pressure_analyzer.analyze(
        snapshot=current_snapshot,
        statistics=statistics,
        regime_state=regime_state,
        previous_snapshot=previous_snapshot,
        previous_open_interest=1_000_000.0,
        current_price=50_010.0,
        previous_price=50_000.0,
    )
    flip_event = flip_detector.detect(
        current_snapshot=current_snapshot,
        previous_snapshot=previous_snapshot,
        statistics=statistics,
    )
    extreme_event = extremes_detector.detect(
        snapshot=current_snapshot,
        statistics=statistics,
        regime_state=regime_state,
    )
    divergence_event = divergence_detector.detect(
        snapshot=current_snapshot,
        statistics=statistics,
        price_change_pct=-0.006,
        oi_change_pct=0.10,
        cvd_change=-20_000.0,
        long_liquidations=100_000.0,
    )

    assert 0.0 <= regime_state.confidence <= 1.0

    assert 0.0 <= pressure_state.pressure_score <= 1.0
    assert pressure_state.squeeze_probability is not None
    assert 0.0 <= pressure_state.squeeze_probability <= 1.0
    assert pressure_state.mean_reversion_probability is not None
    assert 0.0 <= pressure_state.mean_reversion_probability <= 1.0

    assert flip_event is not None
    assert 0.0 <= flip_event.confidence <= 1.0

    assert extreme_event is not None
    assert 0.0 <= extreme_event.severity <= 1.0

    assert divergence_event is not None
    assert 0.0 <= divergence_event.confidence <= 1.0