from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .config import OIAnalyzerConfig
from .enums import OIDirection, OIDivergenceType
from .models import OIDivergenceResult, OIFeatures


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _mean(values: Sequence[float]) -> float | None:
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def _safe_abs(value: float | None) -> float:
    return abs(value) if value is not None else 0.0


def _sum(values: Sequence[float | None]) -> float:
    return float(sum(v for v in values if v is not None))


def _trend_delta(values: Sequence[float | None]) -> float | None:
    cleaned = [float(v) for v in values if v is not None]
    if len(cleaned) < 2:
        return None
    return cleaned[-1] - cleaned[0]


def _direction_from_delta(delta: float | None, flat_epsilon: float = 1e-12) -> OIDirection:
    if delta is None:
        return OIDirection.UNKNOWN
    if abs(delta) <= flat_epsilon:
        return OIDirection.FLAT
    return OIDirection.UP if delta > 0 else OIDirection.DOWN


@dataclass(slots=True)
class DivergenceWindowSummary:
    window_size: int
    price_delta_pct_total: float | None
    oi_delta_pct_total: float | None
    avg_volume_ratio: float | None
    avg_oi_zscore: float | None
    avg_pressure_score: float | None
    avg_funding_rate: float | None
    avg_liquidation_imbalance: float | None
    avg_aggressive_flow_imbalance: float | None
    avg_oi_price_efficiency: float | None
    latest: OIFeatures

    @property
    def price_direction(self) -> OIDirection:
        return _direction_from_delta(self.price_delta_pct_total)

    @property
    def oi_direction(self) -> OIDirection:
        return _direction_from_delta(self.oi_delta_pct_total)


class OIDivergenceDetector:
    """
    Детектор дивергенцій між рухом ціни та поведінкою OI.

    Ідея:
    - дивимось не на 1 snapshot, а на rolling window
    - оцінюємо cumulative move по price
    - оцінюємо response по OI
    - враховуємо volume / funding / liquidation / aggressive flow
    """

    def __init__(self, config: OIAnalyzerConfig) -> None:
        self.config = config
        self.thresholds = config.thresholds
        self.windows = config.windows

    def detect(
        self,
        history: Sequence[OIFeatures],
    ) -> OIDivergenceResult | None:
        if not history:
            return None

        window = self._prepare_window(history)
        if window is None:
            return None

        candidates = [
            self._detect_price_up_oi_down(window),
            self._detect_price_down_oi_down(window),
            self._detect_price_up_oi_flat(window),
            self._detect_price_down_oi_flat(window),
            self._detect_weak_breakout_up(window),
            self._detect_weak_breakout_down(window),
            self._detect_exhaustion_up(window),
            self._detect_exhaustion_down(window),
            self._detect_bullish(window),
            self._detect_bearish(window),
        ]

        best = max(candidates, key=lambda x: x.confidence, default=None)
        if best is None:
            return None

        if not best.detected:
            return best

        if best.confidence < self.thresholds.divergence_min_confidence:
            return OIDivergenceResult(
                detected=False,
                divergence_type=OIDivergenceType.NONE,
                confidence=0.0,
                reasons=["divergence_below_confidence_threshold"],
                window_size=window.window_size,
                score=best.score,
            )

        return best

    def _prepare_window(
        self,
        history: Sequence[OIFeatures],
    ) -> DivergenceWindowSummary | None:
        window_size = min(len(history), self.windows.divergence_window)
        if window_size < 3:
            return None

        window = list(history[-window_size:])
        latest = window[-1]

        price_moves = [f.price_delta_pct for f in window]
        oi_moves = [f.oi_delta_pct for f in window]
        volume_ratios = [f.volume_ratio for f in window if f.volume_ratio is not None]
        oi_zscores = [f.oi_zscore for f in window if f.oi_zscore is not None]
        pressure_scores = [f.oi_pressure_score for f in window if f.oi_pressure_score is not None]
        funding_rates = [f.funding_rate for f in window if f.funding_rate is not None]
        liq_imbalances = [
            f.liquidation_imbalance for f in window if f.liquidation_imbalance is not None
        ]
        flow_imbalances = [
            f.aggressive_flow_imbalance
            for f in window
            if f.aggressive_flow_imbalance is not None
        ]
        oi_price_efficiencies = [
            f.oi_price_efficiency for f in window if f.oi_price_efficiency is not None
        ]

        return DivergenceWindowSummary(
            window_size=window_size,
            price_delta_pct_total=_sum(price_moves),
            oi_delta_pct_total=_sum(oi_moves),
            avg_volume_ratio=_mean(volume_ratios),
            avg_oi_zscore=_mean(oi_zscores),
            avg_pressure_score=_mean(pressure_scores),
            avg_funding_rate=_mean(funding_rates),
            avg_liquidation_imbalance=_mean(liq_imbalances),
            avg_aggressive_flow_imbalance=_mean(flow_imbalances),
            avg_oi_price_efficiency=_mean(oi_price_efficiencies),
            latest=latest,
        )

    def _base_not_detected(self, window: DivergenceWindowSummary) -> OIDivergenceResult:
        return OIDivergenceResult(
            detected=False,
            divergence_type=OIDivergenceType.NONE,
            confidence=0.0,
            reasons=["no_divergence_detected"],
            window_size=window.window_size,
            score=0.0,
        )

    def _make_result(
        self,
        *,
        window: DivergenceWindowSummary,
        divergence_type: OIDivergenceType,
        score: float,
        reasons: list[str],
    ) -> OIDivergenceResult:
        score = _clamp(score)
        detected = score >= 0.35

        return OIDivergenceResult(
            detected=detected,
            divergence_type=divergence_type if detected else OIDivergenceType.NONE,
            confidence=score if detected else 0.0,
            reasons=reasons if detected else ["signal_too_weak_for_divergence"],
            window_size=window.window_size,
            score=score,
        )

    def _is_strong_price_move(self, price_delta_pct_total: float | None) -> bool:
        return (
            price_delta_pct_total is not None
            and abs(price_delta_pct_total) >= self.thresholds.divergence_min_price_move_pct
        )

    def _is_flat_oi_response(self, oi_delta_pct_total: float | None) -> bool:
        return (
            oi_delta_pct_total is not None
            and abs(oi_delta_pct_total) <= self.thresholds.divergence_max_oi_response_pct
        )

    def _is_down_oi_response(self, oi_delta_pct_total: float | None) -> bool:
        return (
            oi_delta_pct_total is not None
            and oi_delta_pct_total <= -self.thresholds.divergence_max_oi_response_pct
        )

    def _is_weak_volume(self, avg_volume_ratio: float | None) -> bool:
        return avg_volume_ratio is not None and avg_volume_ratio < 1.0

    def _is_strong_volume(self, avg_volume_ratio: float | None) -> bool:
        return (
            avg_volume_ratio is not None
            and avg_volume_ratio >= self.thresholds.volume_confirmation_ratio
        )

    def _detect_price_up_oi_down(
        self,
        window: DivergenceWindowSummary,
    ) -> OIDivergenceResult:
        score = 0.0
        reasons: list[str] = []

        if (
            window.price_delta_pct_total is not None
            and window.price_delta_pct_total >= self.thresholds.divergence_min_price_move_pct
        ):
            score += 0.34
            reasons.append("price_trending_up")

        if self._is_down_oi_response(window.oi_delta_pct_total):
            score += 0.32
            reasons.append("oi_trending_down")

        if self._is_weak_volume(window.avg_volume_ratio):
            score += 0.10
            reasons.append("weak_volume_follow_through")

        if (
            window.avg_pressure_score is not None
            and window.avg_pressure_score < 0.15
        ):
            score += 0.08
            reasons.append("weak_directional_pressure")

        if (
            window.avg_oi_price_efficiency is not None
            and abs(window.avg_oi_price_efficiency) < 0.35
        ):
            score += 0.08
            reasons.append("price_move_not_supported_by_oi")

        if (
            window.avg_aggressive_flow_imbalance is not None
            and window.avg_aggressive_flow_imbalance < 0.05
        ):
            score += 0.04
            reasons.append("lack_of_aggressive_buy_confirmation")

        return self._make_result(
            window=window,
            divergence_type=OIDivergenceType.PRICE_UP_OI_DOWN,
            score=score,
            reasons=reasons,
        )

    def _detect_price_down_oi_down(
        self,
        window: DivergenceWindowSummary,
    ) -> OIDivergenceResult:
        score = 0.0
        reasons: list[str] = []

        if (
            window.price_delta_pct_total is not None
            and window.price_delta_pct_total <= -self.thresholds.divergence_min_price_move_pct
        ):
            score += 0.32
            reasons.append("price_trending_down")

        if self._is_down_oi_response(window.oi_delta_pct_total):
            score += 0.28
            reasons.append("oi_trending_down")

        if (
            window.avg_liquidation_imbalance is not None
            and window.avg_liquidation_imbalance < -0.15
        ):
            score += 0.12
            reasons.append("long_liquidation_pressure")

        if (
            window.avg_pressure_score is not None
            and window.avg_pressure_score > -0.35
        ):
            score += 0.08
            reasons.append("sell_pressure_not_persistent")

        if self._is_weak_volume(window.avg_volume_ratio):
            score += 0.08
            reasons.append("weak_sell_volume_follow_through")

        if (
            window.avg_oi_price_efficiency is not None
            and abs(window.avg_oi_price_efficiency) < 0.45
        ):
            score += 0.06
            reasons.append("oi_not_expanding_with_down_move")

        return self._make_result(
            window=window,
            divergence_type=OIDivergenceType.PRICE_DOWN_OI_DOWN,
            score=score,
            reasons=reasons,
        )

    def _detect_price_up_oi_flat(
        self,
        window: DivergenceWindowSummary,
    ) -> OIDivergenceResult:
        score = 0.0
        reasons: list[str] = []

        if (
            window.price_delta_pct_total is not None
            and window.price_delta_pct_total >= self.thresholds.divergence_min_price_move_pct
        ):
            score += 0.36
            reasons.append("price_trending_up")

        if self._is_flat_oi_response(window.oi_delta_pct_total):
            score += 0.28
            reasons.append("oi_flat_response")

        if self._is_weak_volume(window.avg_volume_ratio):
            score += 0.10
            reasons.append("weak_volume")

        if (
            window.avg_pressure_score is not None
            and window.avg_pressure_score < 0.20
        ):
            score += 0.08
            reasons.append("pressure_not_confirming_uptrend")

        if (
            window.avg_oi_zscore is not None
            and abs(window.avg_oi_zscore) < 0.75
        ):
            score += 0.05
            reasons.append("oi_not_statistically_expanding")

        return self._make_result(
            window=window,
            divergence_type=OIDivergenceType.PRICE_UP_OI_FLAT,
            score=score,
            reasons=reasons,
        )

    def _detect_price_down_oi_flat(
        self,
        window: DivergenceWindowSummary,
    ) -> OIDivergenceResult:
        score = 0.0
        reasons: list[str] = []

        if (
            window.price_delta_pct_total is not None
            and window.price_delta_pct_total <= -self.thresholds.divergence_min_price_move_pct
        ):
            score += 0.36
            reasons.append("price_trending_down")

        if self._is_flat_oi_response(window.oi_delta_pct_total):
            score += 0.28
            reasons.append("oi_flat_response")

        if self._is_weak_volume(window.avg_volume_ratio):
            score += 0.10
            reasons.append("weak_volume")

        if (
            window.avg_pressure_score is not None
            and window.avg_pressure_score > -0.20
        ):
            score += 0.08
            reasons.append("pressure_not_confirming_downtrend")

        if (
            window.avg_oi_zscore is not None
            and abs(window.avg_oi_zscore) < 0.75
        ):
            score += 0.05
            reasons.append("oi_not_statistically_expanding")

        return self._make_result(
            window=window,
            divergence_type=OIDivergenceType.PRICE_DOWN_OI_FLAT,
            score=score,
            reasons=reasons,
        )

    def _detect_weak_breakout_up(
        self,
        window: DivergenceWindowSummary,
    ) -> OIDivergenceResult:
        latest = window.latest
        score = 0.0
        reasons: list[str] = []

        if (
            latest.price_delta_pct is not None
            and latest.price_delta_pct >= self.thresholds.divergence_min_price_move_pct
        ):
            score += 0.26
            reasons.append("latest_up_move")

        if latest.price_direction == OIDirection.UP:
            score += 0.10
            reasons.append("latest_price_direction_up")

        if latest.oi_delta_pct <= self.thresholds.divergence_max_oi_response_pct:
            score += 0.24
            reasons.append("latest_oi_not_confirming_breakout")

        if latest.volume_ratio is not None and latest.volume_ratio < 1.0:
            score += 0.12
            reasons.append("latest_breakout_on_weak_volume")

        if latest.aggressive_flow_imbalance is not None and latest.aggressive_flow_imbalance < 0.08:
            score += 0.08
            reasons.append("insufficient_aggressive_buy_flow")

        if latest.oi_price_efficiency is not None and abs(latest.oi_price_efficiency) < 0.35:
            score += 0.08
            reasons.append("weak_oi_price_efficiency")

        if latest.oi_zscore is not None and abs(latest.oi_zscore) < 0.75:
            score += 0.06
            reasons.append("no_oi_expansion_extreme")

        return self._make_result(
            window=window,
            divergence_type=OIDivergenceType.WEAK_BREAKOUT_UP,
            score=score,
            reasons=reasons,
        )

    def _detect_weak_breakout_down(
        self,
        window: DivergenceWindowSummary,
    ) -> OIDivergenceResult:
        latest = window.latest
        score = 0.0
        reasons: list[str] = []

        if (
            latest.price_delta_pct is not None
            and latest.price_delta_pct <= -self.thresholds.divergence_min_price_move_pct
        ):
            score += 0.26
            reasons.append("latest_down_move")

        if latest.price_direction == OIDirection.DOWN:
            score += 0.10
            reasons.append("latest_price_direction_down")

        if latest.oi_delta_pct >= -self.thresholds.divergence_max_oi_response_pct:
            score += 0.24
            reasons.append("latest_oi_not_confirming_breakdown")

        if latest.volume_ratio is not None and latest.volume_ratio < 1.0:
            score += 0.12
            reasons.append("latest_breakdown_on_weak_volume")

        if latest.aggressive_flow_imbalance is not None and latest.aggressive_flow_imbalance > -0.08:
            score += 0.08
            reasons.append("insufficient_aggressive_sell_flow")

        if latest.oi_price_efficiency is not None and abs(latest.oi_price_efficiency) < 0.35:
            score += 0.08
            reasons.append("weak_oi_price_efficiency")

        if latest.oi_zscore is not None and abs(latest.oi_zscore) < 0.75:
            score += 0.06
            reasons.append("no_oi_expansion_extreme")

        return self._make_result(
            window=window,
            divergence_type=OIDivergenceType.WEAK_BREAKOUT_DOWN,
            score=score,
            reasons=reasons,
        )

    def _detect_exhaustion_up(
        self,
        window: DivergenceWindowSummary,
    ) -> OIDivergenceResult:
        latest = window.latest
        score = 0.0
        reasons: list[str] = []

        if (
            window.price_delta_pct_total is not None
            and window.price_delta_pct_total > self.thresholds.divergence_min_price_move_pct
        ):
            score += 0.24
            reasons.append("uptrend_present")

        if self._is_down_oi_response(window.oi_delta_pct_total) or self._is_flat_oi_response(
            window.oi_delta_pct_total
        ):
            score += 0.22
            reasons.append("oi_failing_to_support_uptrend")

        if latest.oi_acceleration is not None and latest.oi_acceleration < 0:
            score += 0.10
            reasons.append("negative_oi_acceleration")

        if latest.oi_velocity is not None and latest.oi_velocity < 0:
            score += 0.08
            reasons.append("negative_oi_velocity")

        if self._is_weak_volume(window.avg_volume_ratio):
            score += 0.10
            reasons.append("fading_volume")

        if (
            window.avg_funding_rate is not None
            and window.avg_funding_rate >= self.thresholds.funding_extreme_positive
        ):
            score += 0.10
            reasons.append("crowded_positive_funding")

        if (
            window.avg_pressure_score is not None
            and window.avg_pressure_score < 0.20
        ):
            score += 0.08
            reasons.append("pressure_fading")

        if (
            latest.aggressive_flow_imbalance is not None
            and latest.aggressive_flow_imbalance < 0.05
        ):
            score += 0.05
            reasons.append("buy_aggression_fading")

        return self._make_result(
            window=window,
            divergence_type=OIDivergenceType.EXHAUSTION_UP,
            score=score,
            reasons=reasons,
        )

    def _detect_exhaustion_down(
        self,
        window: DivergenceWindowSummary,
    ) -> OIDivergenceResult:
        latest = window.latest
        score = 0.0
        reasons: list[str] = []

        if (
            window.price_delta_pct_total is not None
            and window.price_delta_pct_total < -self.thresholds.divergence_min_price_move_pct
        ):
            score += 0.24
            reasons.append("downtrend_present")

        if self._is_down_oi_response(window.oi_delta_pct_total) or self._is_flat_oi_response(
            window.oi_delta_pct_total
        ):
            score += 0.18
            reasons.append("oi_not_expanding_with_downtrend")

        if latest.oi_acceleration is not None and latest.oi_acceleration > 0:
            score += 0.10
            reasons.append("positive_oi_acceleration")

        if latest.oi_velocity is not None and latest.oi_velocity > 0:
            score += 0.08
            reasons.append("positive_oi_velocity")

        if self._is_weak_volume(window.avg_volume_ratio):
            score += 0.10
            reasons.append("fading_volume")

        if (
            window.avg_funding_rate is not None
            and window.avg_funding_rate <= self.thresholds.funding_extreme_negative
        ):
            score += 0.10
            reasons.append("crowded_negative_funding")

        if (
            window.avg_pressure_score is not None
            and window.avg_pressure_score > -0.20
        ):
            score += 0.08
            reasons.append("pressure_fading")

        if (
            latest.aggressive_flow_imbalance is not None
            and latest.aggressive_flow_imbalance > -0.05
        ):
            score += 0.05
            reasons.append("sell_aggression_fading")

        return self._make_result(
            window=window,
            divergence_type=OIDivergenceType.EXHAUSTION_DOWN,
            score=score,
            reasons=reasons,
        )

    def _detect_bullish(
        self,
        window: DivergenceWindowSummary,
    ) -> OIDivergenceResult:
        score = 0.0
        reasons: list[str] = []

        if (
            window.price_delta_pct_total is not None
            and window.price_delta_pct_total <= -self.thresholds.divergence_min_price_move_pct
        ):
            score += 0.28
            reasons.append("price_under_pressure")

        if self._is_down_oi_response(window.oi_delta_pct_total):
            score += 0.20
            reasons.append("oi_declining_on_selloff")

        if (
            window.avg_liquidation_imbalance is not None
            and window.avg_liquidation_imbalance <= -0.20
        ):
            score += 0.12
            reasons.append("long_flush_context")

        if (
            window.avg_pressure_score is not None
            and window.avg_pressure_score > -0.15
        ):
            score += 0.10
            reasons.append("bearish_pressure_not_persistent")

        if (
            window.avg_aggressive_flow_imbalance is not None
            and window.avg_aggressive_flow_imbalance > -0.05
        ):
            score += 0.08
            reasons.append("sell_aggression_not_dominant")

        if self._is_weak_volume(window.avg_volume_ratio):
            score += 0.08
            reasons.append("selloff_lacks_volume_expansion")

        if (
            window.avg_funding_rate is not None
            and window.avg_funding_rate < 0
        ):
            score += 0.05
            reasons.append("negative_funding_crowding")

        return self._make_result(
            window=window,
            divergence_type=OIDivergenceType.BULLISH,
            score=score,
            reasons=reasons,
        )

    def _detect_bearish(
        self,
        window: DivergenceWindowSummary,
    ) -> OIDivergenceResult:
        score = 0.0
        reasons: list[str] = []

        if (
            window.price_delta_pct_total is not None
            and window.price_delta_pct_total >= self.thresholds.divergence_min_price_move_pct
        ):
            score += 0.28
            reasons.append("price_strength_present")

        if self._is_down_oi_response(window.oi_delta_pct_total) or self._is_flat_oi_response(
            window.oi_delta_pct_total
        ):
            score += 0.20
            reasons.append("oi_not_supporting_rally")

        if (
            window.avg_pressure_score is not None
            and window.avg_pressure_score < 0.15
        ):
            score += 0.10
            reasons.append("bullish_pressure_not_persistent")

        if (
            window.avg_aggressive_flow_imbalance is not None
            and window.avg_aggressive_flow_imbalance < 0.05
        ):
            score += 0.08
            reasons.append("buy_aggression_not_dominant")

        if self._is_weak_volume(window.avg_volume_ratio):
            score += 0.08
            reasons.append("rally_lacks_volume_expansion")

        if (
            window.avg_funding_rate is not None
            and window.avg_funding_rate > 0
        ):
            score += 0.05
            reasons.append("positive_funding_crowding")

        if (
            window.avg_oi_zscore is not None
            and window.avg_oi_zscore < 1.0
        ):
            score += 0.05
            reasons.append("oi_not_statistically_supportive")

        return self._make_result(
            window=window,
            divergence_type=OIDivergenceType.BEARISH,
            score=score,
            reasons=reasons,
        )

    def describe_window(
        self,
        history: Sequence[OIFeatures],
    ) -> dict[str, float | int | str | None]:
        """
        Helper для debug/logging/inspection.
        """
        window = self._prepare_window(history)
        if window is None:
            return {
                "window_size": 0,
                "price_delta_pct_total": None,
                "oi_delta_pct_total": None,
                "price_direction": "UNKNOWN",
                "oi_direction": "UNKNOWN",
            }

        return {
            "window_size": window.window_size,
            "price_delta_pct_total": window.price_delta_pct_total,
            "oi_delta_pct_total": window.oi_delta_pct_total,
            "avg_volume_ratio": window.avg_volume_ratio,
            "avg_oi_zscore": window.avg_oi_zscore,
            "avg_pressure_score": window.avg_pressure_score,
            "avg_funding_rate": window.avg_funding_rate,
            "avg_liquidation_imbalance": window.avg_liquidation_imbalance,
            "avg_aggressive_flow_imbalance": window.avg_aggressive_flow_imbalance,
            "avg_oi_price_efficiency": window.avg_oi_price_efficiency,
            "price_direction": window.price_direction.value,
            "oi_direction": window.oi_direction.value,
        }