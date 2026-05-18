# tests/analytics/open_interest/test_oi_detectors.py

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from analytics.open_interest.config import (
    OIAnalyzerConfig,
    OIMaintenanceConfig,
    OIThresholds,
    OIWindows,
)
from analytics.open_interest.enums import (
    OIAnomalyType,
    OIDirection,
    OIDivergenceType,
    OIRegime,
    OISignalStrength,
)
from analytics.open_interest.models import (
    OIAnomalyResult,
    OIDivergenceResult,
    OIFeatures,
    OIRegimeResult,
)
from analytics.open_interest.oi_anomaly_detector import (
    AnomalyCandidate,
    OIAnomalyDetector,
)
from analytics.open_interest.oi_divergence import (
    DivergenceCandidate,
    OIDivergenceDetector,
)
from analytics.open_interest.oi_regime_detector import (
    OIRegimeDetector,
    RegimeCandidate,
)


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

DEFAULT_EXCHANGE = "binance"
DEFAULT_MARKET_TYPE = "usdm_futures"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_TIMEFRAME = "1m"
DEFAULT_TIMESTAMP = 1_700_000_000.0


def assert_probability(value: float) -> None:
    assert math.isfinite(value)
    assert 0.0 <= value <= 1.0


def assert_optional_score(value: float | None) -> None:
    if value is not None:
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0


def assert_scope(
    features: OIFeatures,
    *,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_MARKET_TYPE,
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
) -> None:
    assert features.exchange == exchange.lower()
    assert features.market_type == market_type.lower()
    assert features.symbol == symbol.upper()
    assert features.timeframe == timeframe
    assert features.key == (
        exchange.lower(),
        market_type.lower(),
        symbol.upper(),
        timeframe,
    )
    assert features.scope_key == (
        f"{exchange.lower()}:{market_type.lower()}:{symbol.upper()}:{timeframe}"
    )


def assert_detected_regime(result: OIRegimeResult, expected: OIRegime) -> None:
    assert result.regime is expected
    assert_probability(result.confidence)
    assert result.score is not None
    assert result.score >= 0.35
    assert result.reasons
    assert "no_strong_regime_signal" not in result.reasons


def assert_neutral_regime(result: OIRegimeResult) -> None:
    assert result.regime is OIRegime.NEUTRAL
    assert_probability(result.confidence)
    assert result.confidence <= 0.40
    assert result.score is not None
    assert result.score < 0.35
    assert result.reasons == ["no_strong_regime_signal"]


def assert_detected_divergence(
    result: OIDivergenceResult | None,
    expected: OIDivergenceType,
) -> None:
    assert result is not None
    assert result.detected is True
    assert result.divergence_type is expected
    assert_probability(result.confidence)
    assert result.confidence >= 0.55
    assert result.score is not None
    assert result.score >= 0.35
    assert result.window_size >= 3
    assert result.reasons
    assert "no_divergence_detected" not in result.reasons
    assert "divergence_below_confidence_threshold" not in result.reasons


def assert_no_divergence(result: OIDivergenceResult | None) -> None:
    assert result is not None
    assert result.detected is False
    assert result.divergence_type is OIDivergenceType.NONE
    assert result.confidence == pytest.approx(0.0)
    assert result.window_size >= 3
    assert result.reasons


def assert_detected_anomaly(
    result: OIAnomalyResult | None,
    expected: OIAnomalyType,
) -> None:
    assert result is not None
    assert result.detected is True
    assert result.anomaly_type is expected
    assert result.strength is not OISignalStrength.LOW or (result.score or 0.0) >= 0.35
    assert_probability(result.confidence)
    assert result.score is not None
    assert result.score >= 0.35
    assert result.reasons
    assert "no_strong_anomaly_detected" not in result.reasons


def assert_no_anomaly(result: OIAnomalyResult | None) -> None:
    assert result is not None
    assert result.detected is False
    assert result.anomaly_type is OIAnomalyType.NONE
    assert result.strength is OISignalStrength.LOW
    assert result.confidence == pytest.approx(0.0)
    assert result.score is not None
    assert result.score < 0.35


# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------

@pytest.fixture()
def config() -> OIAnalyzerConfig:
    """
    Чутливий тестовий конфіг:
    - низькі min movement thresholds для конкуренції rule combinations;
    - divergence_min_confidence достатньо високий, щоб ловити reject;
    - коротке divergence_window, щоб будувати компактні adversarial windows.
    """

    return OIAnalyzerConfig(
        source_name="test_oi_detectors",
        require_price_context=False,
        require_volume_confirmation=True,
        thresholds=OIThresholds(
            min_oi_change_pct=0.25,
            min_price_change_pct=0.20,
            volume_confirmation_ratio=1.15,
            aggressive_flow_confirmation=0.10,
            funding_extreme_positive=0.010,
            funding_extreme_negative=-0.010,
            divergence_min_price_move_pct=0.35,
            divergence_max_oi_response_pct=0.10,
            divergence_min_confidence=0.55,
            anomaly_zscore_threshold=2.5,
            extreme_anomaly_zscore_threshold=3.5,
            overheated_zscore_threshold=2.8,
            capitulation_price_move_pct=1.25,
            capitulation_oi_drop_pct=1.00,
            deleveraging_oi_drop_pct=1.50,
            squeeze_funding_abs_threshold=0.015,
            squeeze_oi_build_pct=0.75,
            pressure_score_trend_threshold=0.35,
            pressure_score_exhaustion_threshold=0.75,
        ),
        windows=OIWindows(
            history_size=60,
            fast_window=3,
            slow_window=7,
            zscore_window=8,
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
def regime_detector(config: OIAnalyzerConfig) -> OIRegimeDetector:
    return OIRegimeDetector(config)


@pytest.fixture()
def divergence_detector(config: OIAnalyzerConfig) -> OIDivergenceDetector:
    return OIDivergenceDetector(config)


@pytest.fixture()
def anomaly_detector(config: OIAnalyzerConfig) -> OIAnomalyDetector:
    return OIAnomalyDetector(config)


def f(
    *,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_MARKET_TYPE,
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    timestamp: float = DEFAULT_TIMESTAMP,
    exchange_symbol: str | None = None,
    oi: float = 1_000.0,
    oi_delta: float = 0.0,
    oi_delta_pct: float = 0.0,
    open_interest_value: float | None = None,
    oi_ma_fast: float | None = 1_000.0,
    oi_ma_slow: float | None = 1_000.0,
    oi_std: float | None = 10.0,
    oi_zscore: float | None = 0.0,
    oi_velocity: float | None = 0.0,
    oi_acceleration: float | None = 0.0,
    price: float | None = 30_000.0,
    price_delta: float | None = 0.0,
    price_delta_pct: float | None = 0.0,
    volume: float | None = 1_000.0,
    quote_volume: float | None = None,
    volume_ma: float | None = 900.0,
    volume_ratio: float | None = 1.0,
    funding_rate: float | None = 0.0,
    predicted_funding_rate: float | None = None,
    long_liquidations: float | None = 0.0,
    short_liquidations: float | None = 0.0,
    liquidation_imbalance: float | None = 0.0,
    cvd_delta: float | None = 0.0,
    aggressive_buy_volume: float | None = 500.0,
    aggressive_sell_volume: float | None = 500.0,
    aggressive_flow_imbalance: float | None = 0.0,
    oi_change_per_volume: float | None = 0.0,
    oi_price_efficiency: float | None = 1.0,
    oi_pressure_score: float | None = 0.0,
    oi_direction: OIDirection | str = OIDirection.FLAT,
    price_direction: OIDirection | str = OIDirection.FLAT,
    metadata: dict | None = None,
) -> OIFeatures:
    return OIFeatures(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
        exchange_symbol=exchange_symbol,
        timestamp=timestamp,
        oi=oi,
        oi_delta=oi_delta,
        oi_delta_pct=oi_delta_pct,
        open_interest_value=open_interest_value,
        oi_ma_fast=oi_ma_fast,
        oi_ma_slow=oi_ma_slow,
        oi_std=oi_std,
        oi_zscore=oi_zscore,
        oi_velocity=oi_velocity,
        oi_acceleration=oi_acceleration,
        price=price,
        price_delta=price_delta,
        price_delta_pct=price_delta_pct,
        volume=volume,
        quote_volume=quote_volume,
        volume_ma=volume_ma,
        volume_ratio=volume_ratio,
        funding_rate=funding_rate,
        predicted_funding_rate=predicted_funding_rate,
        long_liquidations=long_liquidations,
        short_liquidations=short_liquidations,
        liquidation_imbalance=liquidation_imbalance,
        cvd_delta=cvd_delta,
        aggressive_buy_volume=aggressive_buy_volume,
        aggressive_sell_volume=aggressive_sell_volume,
        aggressive_flow_imbalance=aggressive_flow_imbalance,
        oi_change_per_volume=oi_change_per_volume,
        oi_price_efficiency=oi_price_efficiency,
        oi_pressure_score=oi_pressure_score,
        oi_direction=oi_direction,
        price_direction=price_direction,
        metadata=dict(metadata or {}),
    )


def history_from_moves(
    *,
    price_moves: Sequence[float | None],
    oi_moves: Sequence[float | None],
    volume_ratio: float | None = 1.2,
    oi_zscore: float | None = 0.0,
    pressure: float | None = 0.0,
    funding: float | None = 0.0,
    liquidation: float | None = 0.0,
    flow: float | None = 0.0,
    efficiency: float | None = 1.0,
    market_type: str = DEFAULT_MARKET_TYPE,
    timeframe: str = DEFAULT_TIMEFRAME,
) -> list[OIFeatures]:
    assert len(price_moves) == len(oi_moves)

    out: list[OIFeatures] = []
    price = 30_000.0
    oi = 1_000.0

    for idx, (price_delta_pct, oi_delta_pct) in enumerate(
        zip(price_moves, oi_moves, strict=True)
    ):
        clean_price_move = float(price_delta_pct or 0.0)
        clean_oi_move = float(oi_delta_pct or 0.0)

        price = price * (1.0 + clean_price_move / 100.0)
        oi = max(0.0, oi * (1.0 + clean_oi_move / 100.0))

        out.append(
            f(
                market_type=market_type,
                timeframe=timeframe,
                timestamp=DEFAULT_TIMESTAMP + idx,
                oi=oi,
                oi_delta=clean_oi_move,
                oi_delta_pct=oi_delta_pct if oi_delta_pct is not None else 0.0,
                oi_ma_fast=1_010.0 + idx,
                oi_ma_slow=1_000.0,
                oi_zscore=oi_zscore,
                oi_velocity=clean_oi_move,
                oi_acceleration=0.01 * clean_oi_move,
                price=price,
                price_delta=clean_price_move,
                price_delta_pct=price_delta_pct,
                volume_ratio=volume_ratio,
                funding_rate=funding,
                liquidation_imbalance=liquidation,
                aggressive_flow_imbalance=flow,
                oi_price_efficiency=efficiency,
                oi_pressure_score=pressure,
                oi_direction=(
                    OIDirection.UP
                    if clean_oi_move > 0
                    else OIDirection.DOWN
                    if clean_oi_move < 0
                    else OIDirection.FLAT
                ),
                price_direction=(
                    OIDirection.UP
                    if clean_price_move > 0
                    else OIDirection.DOWN
                    if clean_price_move < 0
                    else OIDirection.FLAT
                ),
            )
        )

    return out


# ---------------------------------------------------------------------------
# Factory / futures scope contract
# ---------------------------------------------------------------------------

def test_feature_factory_builds_canonical_futures_scope() -> None:
    features = f(
        exchange="BINANCE",
        market_type="USDM_FUTURES",
        symbol="btcusdt",
        timeframe="5m",
        exchange_symbol="BTCUSDT",
    )

    assert_scope(
        features,
        exchange="binance",
        market_type="usdm_futures",
        symbol="BTCUSDT",
        timeframe="5m",
    )
    assert features.exchange_symbol == "BTCUSDT"


@pytest.mark.parametrize(
    ("market_type", "timeframe"),
    [
        ("usdm_futures", "1m"),
        ("coinm_futures", "5m"),
        ("linear", "15m"),
        ("swap", "1h"),
    ],
)
def test_detectors_accept_futures_scoped_features_without_touching_scope(
    regime_detector: OIRegimeDetector,
    anomaly_detector: OIAnomalyDetector,
    divergence_detector: OIDivergenceDetector,
    market_type: str,
    timeframe: str,
) -> None:
    features = f(
        market_type=market_type,
        timeframe=timeframe,
        price_delta_pct=0.80,
        oi_delta_pct=0.90,
        volume_ratio=1.50,
        aggressive_flow_imbalance=0.20,
        oi_pressure_score=0.55,
        oi_ma_fast=1_080.0,
        oi_ma_slow=1_000.0,
    )

    history = history_from_moves(
        price_moves=[0.30, 0.35, 0.40],
        oi_moves=[0.20, 0.25, 0.30],
        market_type=market_type,
        timeframe=timeframe,
    )

    regime = regime_detector.detect(features)
    anomaly = anomaly_detector.detect(features)
    divergence = divergence_detector.detect(history)

    assert regime is not None
    assert anomaly is not None
    assert divergence is not None
    assert_scope(features, market_type=market_type, timeframe=timeframe)


# ---------------------------------------------------------------------------
# Regime detector: weak/noisy/boundary cases
# ---------------------------------------------------------------------------

def test_regime_returns_neutral_for_weak_noisy_features_without_throwing(
    regime_detector: OIRegimeDetector,
) -> None:
    result = regime_detector.detect(
        f(
            oi_delta_pct=0.01,
            price_delta_pct=-0.01,
            volume_ratio=None,
            funding_rate=None,
            liquidation_imbalance=None,
            aggressive_flow_imbalance=None,
            oi_pressure_score=None,
            oi_ma_fast=None,
            oi_ma_slow=None,
            oi_zscore=None,
            oi_velocity=None,
            oi_acceleration=None,
        )
    )

    assert_neutral_regime(result)


@pytest.mark.parametrize(
    ("features", "expected_reasons"),
    [
        (
            f(
                price_delta_pct=0.20,
                oi_delta_pct=0.25,
                volume_ratio=1.15,
                aggressive_flow_imbalance=0.10,
                oi_pressure_score=0.35,
                funding_rate=0.001,
                oi_ma_fast=1_001.0,
                oi_ma_slow=1_000.0,
            ),
            {"price_up", "oi_up", "volume_confirmation"},
        ),
        (
            f(
                price_delta_pct=-0.20,
                oi_delta_pct=0.25,
                volume_ratio=1.15,
                aggressive_flow_imbalance=-0.10,
                oi_pressure_score=-0.35,
                funding_rate=-0.001,
                oi_ma_fast=1_001.0,
                oi_ma_slow=1_000.0,
            ),
            {"price_down", "oi_up", "volume_confirmation"},
        ),
    ],
)
def test_regime_threshold_boundaries_are_inclusive(
    regime_detector: OIRegimeDetector,
    features: OIFeatures,
    expected_reasons: set[str],
) -> None:
    result = regime_detector.detect(features)

    assert result.score is not None
    assert result.score >= 0.35
    assert expected_reasons.issubset(set(result.reasons))


@pytest.mark.parametrize(
    ("features", "expected", "required_reasons"),
    [
        (
            f(
                price_delta_pct=0.90,
                oi_delta_pct=0.95,
                volume_ratio=1.60,
                funding_rate=0.004,
                aggressive_flow_imbalance=0.22,
                oi_pressure_score=0.55,
                oi_ma_fast=1_080.0,
                oi_ma_slow=1_020.0,
                oi_direction=OIDirection.UP,
                price_direction=OIDirection.UP,
            ),
            OIRegime.LONG_BUILDUP,
            {"price_up", "oi_up", "volume_confirmation"},
        ),
        (
            f(
                price_delta_pct=-0.95,
                oi_delta_pct=0.95,
                volume_ratio=1.60,
                funding_rate=-0.004,
                aggressive_flow_imbalance=-0.22,
                oi_pressure_score=-0.55,
                oi_ma_fast=1_080.0,
                oi_ma_slow=1_020.0,
                oi_direction=OIDirection.UP,
                price_direction=OIDirection.DOWN,
            ),
            OIRegime.SHORT_BUILDUP,
            {"price_down", "oi_up", "volume_confirmation"},
        ),
        (
            f(
                price_delta_pct=1.15,
                oi_delta_pct=-0.90,
                volume_ratio=1.45,
                funding_rate=-0.003,
                aggressive_flow_imbalance=0.25,
                oi_pressure_score=0.45,
                oi_ma_fast=980.0,
                oi_ma_slow=1_020.0,
                oi_direction=OIDirection.DOWN,
                price_direction=OIDirection.UP,
            ),
            OIRegime.SHORT_COVERING,
            {"price_up", "oi_down"},
        ),
        (
            f(
                price_delta_pct=-1.15,
                oi_delta_pct=-0.90,
                volume_ratio=1.45,
                funding_rate=0.003,
                aggressive_flow_imbalance=-0.25,
                oi_pressure_score=-0.45,
                oi_ma_fast=980.0,
                oi_ma_slow=1_020.0,
                oi_direction=OIDirection.DOWN,
                price_direction=OIDirection.DOWN,
            ),
            OIRegime.LONG_UNWIND,
            {"price_down", "oi_down"},
        ),
    ],
)
def test_regime_detects_directional_states_under_conflicting_but_valid_inputs(
    regime_detector: OIRegimeDetector,
    features: OIFeatures,
    expected: OIRegime,
    required_reasons: set[str],
) -> None:
    result = regime_detector.detect(features)

    assert_detected_regime(result, expected)
    assert required_reasons.issubset(set(result.reasons))


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        (
            f(
                price_delta_pct=1.60,
                oi_delta_pct=1.20,
                volume_ratio=1.80,
                funding_rate=0.006,
                aggressive_flow_imbalance=0.20,
                oi_pressure_score=0.60,
                oi_ma_fast=1_080.0,
                oi_ma_slow=1_000.0,
            ),
            OIRegime.TREND_CONFIRMATION,
        ),
        (
            f(
                price_delta_pct=1.80,
                oi_delta_pct=0.95,
                volume_ratio=1.35,
                funding_rate=0.018,
                aggressive_flow_imbalance=0.06,
                oi_pressure_score=0.82,
                oi_zscore=2.2,
                liquidation_imbalance=0.10,
                oi_ma_fast=1_080.0,
                oi_ma_slow=1_020.0,
            ),
            OIRegime.TREND_EXHAUSTION,
        ),
        (
            f(
                price_delta_pct=0.35,
                oi_delta_pct=1.40,
                volume_ratio=1.40,
                funding_rate=0.020,
                aggressive_flow_imbalance=0.18,
                oi_pressure_score=0.80,
                oi_zscore=3.2,
                oi_ma_fast=1_100.0,
                oi_ma_slow=1_000.0,
            ),
            OIRegime.SQUEEZE_SETUP,
        ),
        (
            f(
                price_delta_pct=-1.80,
                oi_delta_pct=-1.65,
                volume_ratio=2.20,
                funding_rate=-0.020,
                aggressive_flow_imbalance=-0.25,
                liquidation_imbalance=-0.55,
                oi_pressure_score=-0.85,
                oi_zscore=-3.4,
                oi_ma_fast=900.0,
                oi_ma_slow=1_000.0,
            ),
            OIRegime.CAPITULATION,
        ),
        (
            f(
                price_delta_pct=0.80,
                oi_delta_pct=1.70,
                volume_ratio=1.75,
                funding_rate=0.020,
                aggressive_flow_imbalance=0.22,
                liquidation_imbalance=0.45,
                oi_pressure_score=0.86,
                oi_zscore=3.7,
                oi_ma_fast=1_150.0,
                oi_ma_slow=1_000.0,
            ),
            OIRegime.OVERHEATED,
        ),
    ],
)
def test_regime_detects_risk_and_exhaustion_states_from_dense_signals(
    regime_detector: OIRegimeDetector,
    features: OIFeatures,
    expected: OIRegime,
) -> None:
    result = regime_detector.detect(features)

    assert_detected_regime(result, expected)


def test_regime_numeric_deltas_win_over_contradictory_direction_enums(
    regime_detector: OIRegimeDetector,
) -> None:
    features = f(
        price_delta_pct=0.90,
        oi_delta_pct=0.95,
        volume_ratio=1.60,
        funding_rate=0.004,
        aggressive_flow_imbalance=0.22,
        oi_pressure_score=0.55,
        oi_ma_fast=1_080.0,
        oi_ma_slow=1_000.0,
        price_direction=OIDirection.DOWN,
        oi_direction=OIDirection.DOWN,
    )

    result = regime_detector.detect(features)

    assert_detected_regime(result, OIRegime.LONG_BUILDUP)
    assert "price_up" in result.reasons
    assert "oi_up" in result.reasons


def test_regime_volume_confirmation_config_changes_outcome_on_same_feature_vector(
    config: OIAnalyzerConfig,
) -> None:
    features = f(
        price_delta_pct=0.80,
        oi_delta_pct=0.80,
        volume_ratio=0.75,
        funding_rate=0.004,
        aggressive_flow_imbalance=0.18,
        oi_pressure_score=0.55,
        oi_ma_fast=1_070.0,
        oi_ma_slow=1_000.0,
    )

    strict_detector = OIRegimeDetector(config)
    strict_result = strict_detector.detect(features)

    relaxed_config = OIAnalyzerConfig(
        source_name=config.source_name,
        require_price_context=config.require_price_context,
        require_volume_confirmation=False,
        thresholds=config.thresholds,
        windows=config.windows,
        maintenance=config.maintenance,
    )
    relaxed_detector = OIRegimeDetector(relaxed_config)
    relaxed_result = relaxed_detector.detect(features)

    assert relaxed_result.score is not None
    assert strict_result.score is not None
    assert relaxed_result.score > strict_result.score
    assert "volume_confirmation" in relaxed_result.reasons
    assert "volume_confirmation" not in strict_result.reasons


def test_regime_priority_prefers_risk_regime_over_directional_regime_on_equal_score(
    regime_detector: OIRegimeDetector,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        regime_detector,
        "_build_candidates",
        lambda _features: [
            RegimeCandidate(OIRegime.LONG_BUILDUP, 0.80, ["same_score"]),
            RegimeCandidate(OIRegime.TREND_CONFIRMATION, 0.80, ["same_score"]),
            RegimeCandidate(OIRegime.OVERHEATED, 0.80, ["same_score"]),
            RegimeCandidate(OIRegime.CAPITULATION, 0.80, ["same_score"]),
        ],
    )

    result = regime_detector.detect(f())

    assert result.regime is OIRegime.CAPITULATION
    assert result.score == pytest.approx(0.80)


def test_regime_candidate_score_is_clamped_and_cannot_overflow_confidence(
    regime_detector: OIRegimeDetector,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        regime_detector,
        "_build_candidates",
        lambda _features: [
            RegimeCandidate(OIRegime.OVERHEATED, 100.0, ["absurd_score"]),
        ],
    )

    result = regime_detector.detect(f())

    assert result.regime is OIRegime.OVERHEATED
    assert result.score == pytest.approx(1.0)
    assert result.confidence == pytest.approx(0.98)


# ---------------------------------------------------------------------------
# Divergence detector: window preparation, dirty history, confidence gates
# ---------------------------------------------------------------------------

def test_divergence_returns_none_for_empty_or_too_short_history(
    divergence_detector: OIDivergenceDetector,
) -> None:
    assert divergence_detector.detect([]) is None
    assert divergence_detector.detect([f()]) is None
    assert divergence_detector.detect([f(), f()]) is None


def test_divergence_describe_window_ignores_none_items_and_reports_unknown_when_too_short(
    divergence_detector: OIDivergenceDetector,
) -> None:
    summary = divergence_detector.describe_window([None, f(), None])  # type: ignore[list-item]

    assert summary["window_size"] == 0
    assert summary["price_direction"] == OIDirection.UNKNOWN.value
    assert summary["oi_direction"] == OIDirection.UNKNOWN.value


def test_divergence_window_uses_only_configured_tail_not_full_history(
    divergence_detector: OIDivergenceDetector,
) -> None:
    history = history_from_moves(
        price_moves=[-10.0, -10.0, 0.30, 0.35, 0.40, 0.45, 0.50],
        oi_moves=[10.0, 10.0, -0.10, -0.12, -0.11, -0.13, -0.14],
        volume_ratio=1.30,
        pressure=-0.10,
        flow=-0.08,
        efficiency=0.20,
    )

    summary = divergence_detector.describe_window(history)

    assert summary["window_size"] == 5
    assert summary["price_delta_pct_total"] == pytest.approx(2.0)
    assert summary["oi_delta_pct_total"] == pytest.approx(-0.60)


@pytest.mark.parametrize(
    ("history", "expected", "required_reason_fragment"),
    [
        (
            history_from_moves(
                price_moves=[0.30, 0.45, 0.40, 0.35, 0.50],
                oi_moves=[-0.10, -0.18, -0.12, -0.20, -0.16],
                volume_ratio=1.25,
                pressure=-0.10,
                flow=-0.08,
                efficiency=0.20,
            ),
            OIDivergenceType.PRICE_UP_OI_DOWN,
            "price",
        ),
        (
            history_from_moves(
                price_moves=[-0.35, -0.45, -0.50, -0.40, -0.35],
                oi_moves=[-0.12, -0.14, -0.10, -0.18, -0.15],
                volume_ratio=1.30,
                pressure=0.10,
                flow=0.08,
                efficiency=0.20,
            ),
            OIDivergenceType.PRICE_DOWN_OI_DOWN,
            "price",
        ),
        (
            history_from_moves(
                price_moves=[0.30, 0.40, 0.35, 0.30, 0.45],
                oi_moves=[0.01, -0.01, 0.02, -0.02, 0.00],
                volume_ratio=1.30,
                pressure=-0.05,
                flow=-0.05,
                efficiency=0.05,
            ),
            OIDivergenceType.PRICE_UP_OI_FLAT,
            "flat",
        ),
        (
            history_from_moves(
                price_moves=[-0.30, -0.40, -0.35, -0.30, -0.45],
                oi_moves=[0.01, -0.01, 0.02, -0.02, 0.00],
                volume_ratio=1.30,
                pressure=0.05,
                flow=0.05,
                efficiency=0.05,
            ),
            OIDivergenceType.PRICE_DOWN_OI_FLAT,
            "flat",
        ),
        (
            history_from_moves(
                price_moves=[0.50, 0.55, 0.60, 0.50, 0.45],
                oi_moves=[0.02, 0.01, -0.02, 0.00, 0.01],
                volume_ratio=0.80,
                pressure=0.05,
                flow=-0.16,
                efficiency=0.04,
            ),
            OIDivergenceType.WEAK_BREAKOUT_UP,
            "weak",
        ),
        (
            history_from_moves(
                price_moves=[-0.50, -0.55, -0.60, -0.50, -0.45],
                oi_moves=[0.02, 0.01, -0.02, 0.00, 0.01],
                volume_ratio=0.80,
                pressure=-0.05,
                flow=0.16,
                efficiency=0.04,
            ),
            OIDivergenceType.WEAK_BREAKOUT_DOWN,
            "weak",
        ),
        (
            history_from_moves(
                price_moves=[0.55, 0.60, 0.70, 0.65, 0.55],
                oi_moves=[0.45, 0.55, 0.60, 0.50, 0.45],
                volume_ratio=1.50,
                oi_zscore=3.2,
                pressure=0.82,
                funding=0.018,
                liquidation=0.40,
                flow=0.24,
                efficiency=1.40,
            ),
            OIDivergenceType.EXHAUSTION_UP,
            "exhaustion",
        ),
        (
            history_from_moves(
                price_moves=[-0.55, -0.60, -0.70, -0.65, -0.55],
                oi_moves=[-0.45, -0.55, -0.60, -0.50, -0.45],
                volume_ratio=1.50,
                oi_zscore=-3.2,
                pressure=-0.82,
                funding=-0.018,
                liquidation=-0.40,
                flow=-0.24,
                efficiency=1.40,
            ),
            OIDivergenceType.EXHAUSTION_DOWN,
            "exhaustion",
        ),
    ],
)
def test_divergence_detects_expected_pattern_from_adversarial_window(
    divergence_detector: OIDivergenceDetector,
    history: list[OIFeatures],
    expected: OIDivergenceType,
    required_reason_fragment: str,
) -> None:
    result = divergence_detector.detect(history)

    assert_detected_divergence(result, expected)
    assert any(required_reason_fragment in reason for reason in result.reasons)


def test_divergence_rejects_real_signal_when_confidence_is_below_configured_threshold(
    config: OIAnalyzerConfig,
) -> None:
    strict_config = OIAnalyzerConfig(
        source_name=config.source_name,
        require_price_context=config.require_price_context,
        require_volume_confirmation=config.require_volume_confirmation,
        thresholds=OIThresholds(
            min_oi_change_pct=config.thresholds.min_oi_change_pct,
            min_price_change_pct=config.thresholds.min_price_change_pct,
            volume_confirmation_ratio=config.thresholds.volume_confirmation_ratio,
            aggressive_flow_confirmation=config.thresholds.aggressive_flow_confirmation,
            funding_extreme_positive=config.thresholds.funding_extreme_positive,
            funding_extreme_negative=config.thresholds.funding_extreme_negative,
            divergence_min_price_move_pct=config.thresholds.divergence_min_price_move_pct,
            divergence_max_oi_response_pct=config.thresholds.divergence_max_oi_response_pct,
            divergence_min_confidence=0.95,
            anomaly_zscore_threshold=config.thresholds.anomaly_zscore_threshold,
            extreme_anomaly_zscore_threshold=config.thresholds.extreme_anomaly_zscore_threshold,
            overheated_zscore_threshold=config.thresholds.overheated_zscore_threshold,
            capitulation_price_move_pct=config.thresholds.capitulation_price_move_pct,
            capitulation_oi_drop_pct=config.thresholds.capitulation_oi_drop_pct,
            deleveraging_oi_drop_pct=config.thresholds.deleveraging_oi_drop_pct,
            squeeze_funding_abs_threshold=config.thresholds.squeeze_funding_abs_threshold,
            squeeze_oi_build_pct=config.thresholds.squeeze_oi_build_pct,
            pressure_score_trend_threshold=config.thresholds.pressure_score_trend_threshold,
            pressure_score_exhaustion_threshold=config.thresholds.pressure_score_exhaustion_threshold,
        ),
        windows=config.windows,
        maintenance=config.maintenance,
    )
    detector = OIDivergenceDetector(strict_config)

    history = history_from_moves(
        price_moves=[0.30, 0.45, 0.40, 0.35, 0.50],
        oi_moves=[-0.10, -0.18, -0.12, -0.20, -0.16],
        volume_ratio=1.25,
        pressure=-0.10,
        flow=-0.08,
        efficiency=0.20,
    )

    result = detector.detect(history)

    assert_no_divergence(result)
    assert result.reasons == ["divergence_below_confidence_threshold"]
    assert result.score is not None
    assert result.score >= 0.35


def test_divergence_priority_prefers_higher_confidence_then_score_then_priority(
    divergence_detector: OIDivergenceDetector,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    low_priority_high_confidence = OIDivergenceResult(
        detected=True,
        divergence_type=OIDivergenceType.PRICE_UP_OI_DOWN,
        confidence=0.81,
        reasons=["high_confidence"],
        window_size=5,
        score=0.70,
    )
    high_priority_lower_confidence = OIDivergenceResult(
        detected=True,
        divergence_type=OIDivergenceType.EXHAUSTION_UP,
        confidence=0.80,
        reasons=["lower_confidence"],
        window_size=5,
        score=0.99,
    )

    monkeypatch.setattr(
        divergence_detector,
        "_build_candidates",
        lambda _window: [
            DivergenceCandidate(high_priority_lower_confidence),
            DivergenceCandidate(low_priority_high_confidence),
        ],
    )

    result = divergence_detector.detect(
        history_from_moves(
            price_moves=[1, 1, 1],
            oi_moves=[-1, -1, -1],
        )
    )

    assert_detected_divergence(result, OIDivergenceType.PRICE_UP_OI_DOWN)
    assert result.confidence == pytest.approx(0.81)


def test_divergence_result_score_and_confidence_are_bounded_when_candidate_overflows(
    divergence_detector: OIDivergenceDetector,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overflowing = OIDivergenceResult(
        detected=True,
        divergence_type=OIDivergenceType.EXHAUSTION_UP,
        confidence=100.0,
        reasons=["overflow"],
        window_size=5,
        score=100.0,
    )

    monkeypatch.setattr(
        divergence_detector,
        "_build_candidates",
        lambda _window: [DivergenceCandidate(overflowing)],
    )

    result = divergence_detector.detect(
        history_from_moves(
            price_moves=[1, 1, 1],
            oi_moves=[1, 1, 1],
        )
    )

    assert_detected_divergence(result, OIDivergenceType.EXHAUSTION_UP)
    assert result.confidence == pytest.approx(1.0)
    assert result.score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Anomaly detector: weak/noisy/boundary/rule scenarios
# ---------------------------------------------------------------------------

def test_anomaly_returns_not_detected_for_weak_noise(
    anomaly_detector: OIAnomalyDetector,
) -> None:
    result = anomaly_detector.detect(
        f(
            oi_delta_pct=0.01,
            oi_zscore=0.1,
            price_delta_pct=0.01,
            volume_ratio=1.0,
            funding_rate=0.0,
            liquidation_imbalance=0.0,
            aggressive_flow_imbalance=0.0,
            oi_pressure_score=0.0,
        )
    )

    assert_no_anomaly(result)
    assert result.reasons == ["no_strong_anomaly_detected"]


@pytest.mark.parametrize(
    ("features", "expected", "required_reason_fragment"),
    [
        (
            f(
                oi_delta_pct=1.30,
                oi_zscore=3.1,
                oi_velocity=0.50,
                oi_acceleration=0.20,
                volume_ratio=1.60,
                price_delta_pct=0.20,
                oi_ma_fast=1_150.0,
                oi_ma_slow=1_000.0,
            ),
            OIAnomalyType.OI_SPIKE,
            "positive",
        ),
        (
            f(
                oi_delta_pct=-1.40,
                oi_zscore=-3.1,
                oi_velocity=-0.50,
                oi_acceleration=-0.20,
                volume_ratio=1.60,
                price_delta_pct=-0.20,
                oi_ma_fast=900.0,
                oi_ma_slow=1_000.0,
            ),
            OIAnomalyType.OI_COLLAPSE,
            "negative",
        ),
        (
            f(
                price_delta_pct=2.20,
                oi_delta_pct=0.02,
                volume_ratio=0.85,
                oi_price_efficiency=0.03,
                oi_change_per_volume=0.0000001,
                aggressive_flow_imbalance=-0.18,
                oi_pressure_score=-0.35,
            ),
            OIAnomalyType.OI_PRICE_DISLOCATION,
            "price",
        ),
        (
            f(
                oi_delta_pct=1.20,
                oi_zscore=2.8,
                volume_ratio=0.65,
                oi_change_per_volume=0.0000001,
                price_delta_pct=0.05,
                oi_price_efficiency=1.8,
            ),
            OIAnomalyType.OI_VOLUME_DISLOCATION,
            "volume",
        ),
        (
            f(
                oi_delta_pct=-1.70,
                oi_zscore=-3.3,
                price_delta_pct=-1.60,
                volume_ratio=2.20,
                liquidation_imbalance=-0.65,
                aggressive_flow_imbalance=-0.30,
                oi_pressure_score=-0.85,
            ),
            OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP,
            "liquidation",
        ),
        (
            f(
                oi_delta_pct=1.70,
                oi_zscore=3.4,
                price_delta_pct=0.80,
                volume_ratio=1.70,
                funding_rate=0.020,
                liquidation_imbalance=0.45,
                aggressive_flow_imbalance=0.25,
                oi_pressure_score=0.86,
                oi_ma_fast=1_150.0,
                oi_ma_slow=1_000.0,
            ),
            OIAnomalyType.EXTREME_CROWDING,
            "crowding",
        ),
        (
            f(
                oi_delta_pct=-1.90,
                oi_zscore=-3.8,
                price_delta_pct=-1.80,
                volume_ratio=2.20,
                funding_rate=-0.020,
                liquidation_imbalance=-0.55,
                aggressive_flow_imbalance=-0.25,
                oi_pressure_score=-0.85,
                oi_ma_fast=900.0,
                oi_ma_slow=1_000.0,
            ),
            OIAnomalyType.SUDDEN_DELEVERAGING,
            "deleveraging",
        ),
        (
            f(
                oi_delta_pct=1.10,
                oi_zscore=2.9,
                price_delta_pct=0.35,
                volume_ratio=1.50,
                funding_rate=0.030,
                aggressive_flow_imbalance=0.12,
                oi_pressure_score=0.70,
                oi_ma_fast=1_100.0,
                oi_ma_slow=1_000.0,
            ),
            OIAnomalyType.FUNDING_OI_IMBALANCE,
            "funding",
        ),
    ],
)
def test_anomaly_detects_expected_pattern_from_dense_features(
    anomaly_detector: OIAnomalyDetector,
    features: OIFeatures,
    expected: OIAnomalyType,
    required_reason_fragment: str,
) -> None:
    result = anomaly_detector.detect(features)

    assert_detected_anomaly(result, expected)
    assert any(required_reason_fragment in reason for reason in result.reasons)


def test_anomaly_threshold_boundaries_are_inclusive(
    anomaly_detector: OIAnomalyDetector,
    config: OIAnalyzerConfig,
) -> None:
    features = f(
        oi_delta_pct=config.thresholds.squeeze_oi_build_pct,
        oi_zscore=config.thresholds.anomaly_zscore_threshold,
        volume_ratio=config.thresholds.volume_confirmation_ratio,
        funding_rate=config.thresholds.squeeze_funding_abs_threshold,
        aggressive_flow_imbalance=config.thresholds.aggressive_flow_confirmation,
        oi_pressure_score=config.thresholds.pressure_score_trend_threshold,
        oi_velocity=0.10,
        oi_acceleration=0.10,
        oi_ma_fast=1_050.0,
        oi_ma_slow=1_000.0,
    )

    result = anomaly_detector.detect(features)

    assert result is not None
    assert result.score is not None
    assert result.score >= 0.35
    assert_probability(result.confidence)


def test_anomaly_priority_prefers_risk_critical_candidate_on_equal_score(
    anomaly_detector: OIAnomalyDetector,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        anomaly_detector,
        "_build_candidates",
        lambda _features: [
            AnomalyCandidate(OIAnomalyType.OI_SPIKE, 0.80, ["same"]),
            AnomalyCandidate(OIAnomalyType.EXTREME_CROWDING, 0.80, ["same"]),
            AnomalyCandidate(OIAnomalyType.SUDDEN_DELEVERAGING, 0.80, ["same"]),
            AnomalyCandidate(OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP, 0.80, ["same"]),
        ],
    )

    result = anomaly_detector.detect(f())

    assert_detected_anomaly(result, OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP)
    assert result.score == pytest.approx(0.80)


@pytest.mark.parametrize(
    ("score", "expected_strength"),
    [
        (0.35, OISignalStrength.LOW),
        (0.50, OISignalStrength.MEDIUM),
        (0.70, OISignalStrength.HIGH),
        (0.90, OISignalStrength.EXTREME),
        (100.0, OISignalStrength.EXTREME),
    ],
)
def test_anomaly_candidate_score_is_clamped_and_strength_thresholds_are_stable(
    anomaly_detector: OIAnomalyDetector,
    monkeypatch: pytest.MonkeyPatch,
    score: float,
    expected_strength: OISignalStrength,
) -> None:
    monkeypatch.setattr(
        anomaly_detector,
        "_build_candidates",
        lambda _features: [
            AnomalyCandidate(OIAnomalyType.EXTREME_CROWDING, score, ["score_test"]),
        ],
    )

    result = anomaly_detector.detect(f())

    assert_detected_anomaly(result, OIAnomalyType.EXTREME_CROWDING)
    assert result.strength is expected_strength
    assert result.confidence <= 0.99
    assert result.score == pytest.approx(min(score, 1.0))


def test_anomaly_context_description_deduplicates_and_handles_missing_fields(
    anomaly_detector: OIAnomalyDetector,
) -> None:
    features = f(
        oi_delta_pct=2.0,
        oi_zscore=4.0,
        volume_ratio=2.0,
        funding_rate=0.030,
        liquidation_imbalance=0.60,
        aggressive_flow_imbalance=0.35,
        oi_pressure_score=0.95,
        oi_ma_fast=None,
        oi_ma_slow=None,
        oi_velocity=None,
        oi_acceleration=None,
        oi_change_per_volume=None,
        oi_price_efficiency=None,
    )

    reasons = anomaly_detector.describe_anomaly_context(features)

    assert len(reasons) == len(set(reasons))
    assert "extreme_positive_oi_zscore" in reasons
    assert "strong_positive_oi_shift" in reasons
    assert "extreme_positive_funding" in reasons
    assert "high_volume" in reasons
    assert "short_liquidation_pressure" in reasons
    assert "aggressive_buy_imbalance" in reasons
    assert "extreme_positive_pressure" in reasons


# ---------------------------------------------------------------------------
# Cross-detector consistency / sparse inputs / serialization
# ---------------------------------------------------------------------------

def test_same_extreme_snapshot_can_be_regime_and_anomaly_but_not_force_divergence(
    regime_detector: OIRegimeDetector,
    anomaly_detector: OIAnomalyDetector,
    divergence_detector: OIDivergenceDetector,
) -> None:
    extreme = f(
        price_delta_pct=0.80,
        oi_delta_pct=1.60,
        oi_zscore=4.1,
        volume_ratio=1.85,
        funding_rate=0.030,
        liquidation_imbalance=0.50,
        aggressive_flow_imbalance=0.30,
        oi_pressure_score=0.90,
        oi_ma_fast=1_200.0,
        oi_ma_slow=1_000.0,
        oi_price_efficiency=1.5,
    )

    regime = regime_detector.detect(extreme)
    anomaly = anomaly_detector.detect(extreme)
    divergence = divergence_detector.detect([extreme, extreme, extreme, extreme, extreme])

    assert regime.regime in {
        OIRegime.SQUEEZE_SETUP,
        OIRegime.OVERHEATED,
        OIRegime.TREND_EXHAUSTION,
    }
    assert anomaly is not None
    assert anomaly.detected is True
    assert anomaly.anomaly_type in {
        OIAnomalyType.EXTREME_CROWDING,
        OIAnomalyType.OVERHEATED_BUILDUP,
        OIAnomalyType.FUNDING_OI_IMBALANCE,
        OIAnomalyType.OI_SPIKE,
    }

    assert divergence is not None
    if divergence.detected:
        assert divergence.divergence_type in {
            OIDivergenceType.EXHAUSTION_UP,
            OIDivergenceType.WEAK_BREAKOUT_UP,
            OIDivergenceType.BEARISH,
        }


@pytest.mark.parametrize(
    "features",
    [
        f(
            price_delta_pct=None,
            price_delta=None,
            price=None,
            oi_delta_pct=1.0,
            volume_ratio=None,
            funding_rate=None,
            liquidation_imbalance=None,
            aggressive_flow_imbalance=None,
            oi_pressure_score=None,
            oi_zscore=None,
            oi_ma_fast=None,
            oi_ma_slow=None,
            oi_velocity=None,
            oi_acceleration=None,
            oi_change_per_volume=None,
            oi_price_efficiency=None,
            price_direction=OIDirection.UNKNOWN,
        ),
        f(
            price_delta_pct=-0.0,
            oi_delta_pct=-0.0,
            volume_ratio=0.0,
            funding_rate=0.0,
            liquidation_imbalance=0.0,
            aggressive_flow_imbalance=0.0,
            oi_pressure_score=0.0,
            oi_zscore=0.0,
            oi_change_per_volume=0.0,
            oi_price_efficiency=0.0,
        ),
    ],
)
def test_detectors_do_not_crash_on_sparse_or_zeroish_features(
    regime_detector: OIRegimeDetector,
    anomaly_detector: OIAnomalyDetector,
    divergence_detector: OIDivergenceDetector,
    features: OIFeatures,
) -> None:
    regime = regime_detector.detect(features)
    anomaly = anomaly_detector.detect(features)
    divergence = divergence_detector.detect([features, features, features])

    assert regime is not None
    assert_probability(regime.confidence)

    assert anomaly is not None
    assert_probability(anomaly.confidence)

    assert divergence is None or 0.0 <= divergence.confidence <= 1.0


def test_detectors_do_not_emit_strong_false_positive_on_only_oi_delta_without_context(
    regime_detector: OIRegimeDetector,
    anomaly_detector: OIAnomalyDetector,
    divergence_detector: OIDivergenceDetector,
) -> None:
    features = f(
        price=None,
        price_delta=None,
        price_delta_pct=None,
        volume=None,
        volume_ratio=None,
        funding_rate=None,
        liquidation_imbalance=None,
        aggressive_flow_imbalance=None,
        oi_pressure_score=None,
        oi_zscore=None,
        oi_ma_fast=None,
        oi_ma_slow=None,
        oi_velocity=None,
        oi_acceleration=None,
        oi_delta_pct=1.0,
        oi_delta=10.0,
        oi_direction=OIDirection.UP,
        price_direction=OIDirection.UNKNOWN,
    )

    regime = regime_detector.detect(features)
    anomaly = anomaly_detector.detect(features)
    divergence = divergence_detector.detect([features, features, features])

    assert regime.score is not None
    assert regime.score < 0.60

    assert anomaly is not None
    assert anomaly.confidence < 0.75

    assert divergence is None or divergence.detected is False


def test_all_detector_outputs_round_trip_through_model_contracts(
    regime_detector: OIRegimeDetector,
    anomaly_detector: OIAnomalyDetector,
    divergence_detector: OIDivergenceDetector,
) -> None:
    features = f(
        price_delta_pct=0.90,
        oi_delta_pct=1.20,
        volume_ratio=1.60,
        funding_rate=0.018,
        liquidation_imbalance=0.45,
        aggressive_flow_imbalance=0.25,
        oi_pressure_score=0.85,
        oi_zscore=3.4,
        oi_ma_fast=1_100.0,
        oi_ma_slow=1_000.0,
    )
    history = history_from_moves(
        price_moves=[0.50, 0.55, 0.60, 0.50, 0.45],
        oi_moves=[0.02, 0.01, -0.02, 0.00, 0.01],
        volume_ratio=0.80,
        pressure=0.05,
        flow=-0.16,
        efficiency=0.04,
    )

    regime = regime_detector.detect(features)
    anomaly = anomaly_detector.detect(features)
    divergence = divergence_detector.detect(history)

    regime_round_trip = OIRegimeResult.from_dict(regime.to_dict())
    anomaly_round_trip = (
        OIAnomalyResult.from_dict(anomaly.to_dict())
        if anomaly is not None
        else None
    )
    divergence_round_trip = (
        OIDivergenceResult.from_dict(divergence.to_dict())
        if divergence is not None
        else None
    )

    assert regime_round_trip.regime is regime.regime
    assert regime_round_trip.confidence == pytest.approx(regime.confidence)
    assert regime_round_trip.reasons == regime.reasons

    assert anomaly_round_trip is not None
    assert anomaly is not None
    assert anomaly_round_trip.detected is anomaly.detected
    assert anomaly_round_trip.anomaly_type is anomaly.anomaly_type
    assert anomaly_round_trip.strength is anomaly.strength
    assert anomaly_round_trip.confidence == pytest.approx(anomaly.confidence)
    assert anomaly_round_trip.reasons == anomaly.reasons

    assert divergence_round_trip is not None
    assert divergence is not None
    assert divergence_round_trip.detected is divergence.detected
    assert divergence_round_trip.divergence_type is divergence.divergence_type
    assert divergence_round_trip.confidence == pytest.approx(divergence.confidence)
    assert divergence_round_trip.reasons == divergence.reasons


def test_detector_outputs_stay_serializable_and_bounded(
    regime_detector: OIRegimeDetector,
    anomaly_detector: OIAnomalyDetector,
    divergence_detector: OIDivergenceDetector,
) -> None:
    features = f(
        price_delta_pct=0.90,
        oi_delta_pct=1.20,
        volume_ratio=1.60,
        funding_rate=0.018,
        liquidation_imbalance=0.45,
        aggressive_flow_imbalance=0.25,
        oi_pressure_score=0.85,
        oi_zscore=3.4,
        oi_ma_fast=1_100.0,
        oi_ma_slow=1_000.0,
    )
    history = history_from_moves(
        price_moves=[0.50, 0.55, 0.60, 0.50, 0.45],
        oi_moves=[0.02, 0.01, -0.02, 0.00, 0.01],
        volume_ratio=0.80,
        pressure=0.05,
        flow=-0.16,
        efficiency=0.04,
    )

    regime = regime_detector.detect(features)
    anomaly = anomaly_detector.detect(features)
    divergence = divergence_detector.detect(history)

    regime_dict = regime.to_dict()
    anomaly_dict = anomaly.to_dict() if anomaly is not None else None
    divergence_dict = divergence.to_dict() if divergence is not None else None

    assert regime_dict["regime"] in {item.value for item in OIRegime}
    assert 0.0 <= regime_dict["confidence"] <= 1.0
    assert isinstance(regime_dict["reasons"], list)

    assert anomaly_dict is not None
    assert anomaly_dict["anomaly_type"] in {item.value for item in OIAnomalyType}
    assert 0.0 <= anomaly_dict["confidence"] <= 1.0
    assert isinstance(anomaly_dict["reasons"], list)

    assert divergence_dict is not None
    assert divergence_dict["divergence_type"] in {item.value for item in OIDivergenceType}
    assert 0.0 <= divergence_dict["confidence"] <= 1.0
    assert isinstance(divergence_dict["reasons"], list)


# ---------------------------------------------------------------------------
# Pure-service contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "detector",
    [
        OIRegimeDetector,
        OIDivergenceDetector,
        OIAnomalyDetector,
    ],
)
def test_detectors_are_pure_domain_services_without_core_infrastructure(
    detector,
    config: OIAnalyzerConfig,
) -> None:
    instance = detector(config)

    assert not hasattr(instance, "event_bus")
    assert not hasattr(instance, "scheduler")
    assert not hasattr(instance, "logger")
    assert not hasattr(instance, "register")
    assert not hasattr(instance, "start")
    assert not hasattr(instance, "stop")


def test_candidate_reason_lists_are_deduplicated_and_scores_are_clamped() -> None:
    regime = RegimeCandidate(
        regime=OIRegime.OVERHEATED,
        score=10.0,
        reasons=["a", "a", "b"],
    )
    anomaly = AnomalyCandidate(
        anomaly_type=OIAnomalyType.OI_SPIKE,
        score=10.0,
        reasons=["a", "a", "b"],
    )
    divergence = DivergenceCandidate(
        OIDivergenceResult(
            detected=True,
            divergence_type=OIDivergenceType.EXHAUSTION_UP,
            confidence=10.0,
            reasons=["a", "a", "b"],
            window_size=5,
            score=10.0,
        )
    )

    assert regime.score == pytest.approx(1.0)
    assert regime.reasons == ["a", "b"]

    assert anomaly.score == pytest.approx(1.0)
    assert anomaly.reasons == ["a", "b"]

    assert divergence.confidence == pytest.approx(1.0)
    assert divergence.score == pytest.approx(1.0)
    assert divergence.reasons_count == 3