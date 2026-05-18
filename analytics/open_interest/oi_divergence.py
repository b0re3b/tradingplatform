from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from .config import OIAnalyzerConfig
from .enums import OIDirection, OIDivergenceType
from .models import OIDivergenceResult, OIFeatures


MIN_DIVERGENCE_SCORE: Final[float] = 0.35
MIN_DIVERGENCE_WINDOW_SIZE: Final[int] = 3

WEAK_VOLUME_RATIO: Final[float] = 1.0
WEAK_PRESSURE_UPPER: Final[float] = 0.20
WEAK_PRESSURE_LOWER: Final[float] = -0.20

# Must stay small: confidence should remain the primary selector key.
EXHAUSTION_SELECTION_BONUS: Final[float] = 0.005

EXHAUSTION_EFFICIENCY_THRESHOLD: Final[float] = 1.25
EXHAUSTION_LIQUIDATION_THRESHOLD: Final[float] = 0.35

DIVERGENCE_PRIORITY: Final[dict[OIDivergenceType, int]] = {
    OIDivergenceType.EXHAUSTION_UP: 100,
    OIDivergenceType.EXHAUSTION_DOWN: 100,

    OIDivergenceType.WEAK_BREAKOUT_UP: 90,
    OIDivergenceType.WEAK_BREAKOUT_DOWN: 90,

    OIDivergenceType.PRICE_UP_OI_DOWN: 80,
    OIDivergenceType.PRICE_DOWN_OI_DOWN: 80,
    OIDivergenceType.PRICE_UP_OI_FLAT: 75,
    OIDivergenceType.PRICE_DOWN_OI_FLAT: 75,

    OIDivergenceType.BULLISH: 65,
    OIDivergenceType.BEARISH: 65,

    OIDivergenceType.NONE: 0,
}


def _to_float(value: float | int | str | None) -> float | None:
    if value is None:
        return None

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(result):
        return None

    return result


def _clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    if low > high:
        raise ValueError("low must be <= high")

    if not math.isfinite(float(low)) or not math.isfinite(float(high)):
        raise ValueError("clamp bounds must be finite")

    number = _to_float(value)
    if number is None:
        return low

    return max(low, min(high, number))


def _mean(values: Sequence[float | None]) -> float | None:
    cleaned = [_to_float(value) for value in values]
    cleaned = [value for value in cleaned if value is not None]

    if not cleaned:
        return None

    return sum(cleaned) / len(cleaned)


def _safe_abs(value: float | None) -> float:
    value = _to_float(value)
    return abs(value) if value is not None else 0.0


def _sum(values: Sequence[float | None]) -> float | None:
    cleaned = [_to_float(value) for value in values]
    cleaned = [value for value in cleaned if value is not None]

    if not cleaned:
        return None

    return float(sum(cleaned))


def _direction_from_delta(
    delta: float | None,
    *,
    flat_epsilon: float = 1e-12,
) -> OIDirection:
    delta = _to_float(delta)

    if delta is None:
        return OIDirection.UNKNOWN

    if abs(delta) <= flat_epsilon:
        return OIDirection.FLAT

    return OIDirection.UP if delta > 0 else OIDirection.DOWN


@dataclass(slots=True)
class DivergenceWindowSummary:
    """
    Aggregated rolling-window context for OI/price divergence detection.
    """

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

    def __post_init__(self) -> None:
        if self.window_size < MIN_DIVERGENCE_WINDOW_SIZE:
            raise ValueError(
                f"window_size must be >= {MIN_DIVERGENCE_WINDOW_SIZE}"
            )

        self.price_delta_pct_total = _to_float(self.price_delta_pct_total)
        self.oi_delta_pct_total = _to_float(self.oi_delta_pct_total)
        self.avg_volume_ratio = _to_float(self.avg_volume_ratio)
        self.avg_oi_zscore = _to_float(self.avg_oi_zscore)
        self.avg_pressure_score = _to_float(self.avg_pressure_score)
        self.avg_funding_rate = _to_float(self.avg_funding_rate)
        self.avg_liquidation_imbalance = _to_float(
            self.avg_liquidation_imbalance
        )
        self.avg_aggressive_flow_imbalance = _to_float(
            self.avg_aggressive_flow_imbalance
        )
        self.avg_oi_price_efficiency = _to_float(
            self.avg_oi_price_efficiency
        )

    @property
    def price_direction(self) -> OIDirection:
        return _direction_from_delta(self.price_delta_pct_total)

    @property
    def oi_direction(self) -> OIDirection:
        return _direction_from_delta(self.oi_delta_pct_total)


@dataclass(slots=True)
class DivergenceCandidate:
    """
    Internal candidate used for deterministic divergence selection.
    """

    result: OIDivergenceResult

    @property
    def divergence_type(self) -> OIDivergenceType:
        return self.result.divergence_type

    @property
    def confidence(self) -> float:
        return _clamp(self.result.confidence)

    @property
    def score(self) -> float:
        return _clamp(float(self.result.score or 0.0))

    @property
    def priority(self) -> int:
        return DIVERGENCE_PRIORITY.get(self.divergence_type, 0)

    @property
    def reasons_count(self) -> int:
        return len(self.result.reasons)


class OIDivergenceDetector:
    """
    Rule-based detector for divergences between price action and Open Interest.

    This is a pure domain service:
    - no EventBus;
    - no Scheduler;
    - no logger;
    - no side effects.

    It receives historical OIFeatures and returns OIDivergenceResult.
    """

    def __init__(self, config: OIAnalyzerConfig) -> None:
        self.config = config
        self.thresholds = config.thresholds
        self.windows = config.windows

    def detect(
        self,
        history: Sequence[OIFeatures],
    ) -> OIDivergenceResult | None:
        """
        Detect the strongest divergence in the configured rolling window.

        Returns:
            - None when there is not enough data.
            - OIDivergenceResult(detected=False, ...) when data exists but no
              valid divergence passes thresholds.
            - OIDivergenceResult(detected=True, ...) for the best signal.
        """
        if not history:
            return None

        window = self._prepare_window(history)
        if window is None:
            return None

        candidates = self._build_candidates(window)
        best = self._select_best_candidate(candidates)

        if best is None:
            return self._base_not_detected(window)

        result = best.result

        if not result.detected:
            return result

        if result.confidence < self.thresholds.divergence_min_confidence:
            return OIDivergenceResult(
                detected=False,
                divergence_type=OIDivergenceType.NONE,
                confidence=0.0,
                reasons=["divergence_below_confidence_threshold"],
                window_size=window.window_size,
                score=result.score,
            )

        return result

    def describe_window(
        self,
        history: Sequence[OIFeatures],
    ) -> dict[str, float | int | str | None]:
        """
        Helper for debug/inspection by OIAnalyzer, dashboard, or tests.

        This method does not log by itself.
        """
        window = self._prepare_window(history)
        if window is None:
            return {
                "window_size": 0,
                "price_delta_pct_total": None,
                "oi_delta_pct_total": None,
                "price_direction": OIDirection.UNKNOWN.value,
                "oi_direction": OIDirection.UNKNOWN.value,
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

    def _prepare_window(
        self,
        history: Sequence[OIFeatures],
    ) -> DivergenceWindowSummary | None:
        cleaned_history = [item for item in history if item is not None]
        window_size = min(len(cleaned_history), self.windows.divergence_window)

        if window_size < MIN_DIVERGENCE_WINDOW_SIZE:
            return None

        window = list(cleaned_history[-window_size:])
        latest = window[-1]

        price_moves = [features.price_delta_pct for features in window]
        oi_moves = [features.oi_delta_pct for features in window]

        return DivergenceWindowSummary(
            window_size=window_size,
            price_delta_pct_total=_sum(price_moves),
            oi_delta_pct_total=_sum(oi_moves),
            avg_volume_ratio=_mean(
                [features.volume_ratio for features in window]
            ),
            avg_oi_zscore=_mean(
                [features.oi_zscore for features in window]
            ),
            avg_pressure_score=_mean(
                [features.oi_pressure_score for features in window]
            ),
            avg_funding_rate=_mean(
                [features.funding_rate for features in window]
            ),
            avg_liquidation_imbalance=_mean(
                [features.liquidation_imbalance for features in window]
            ),
            avg_aggressive_flow_imbalance=_mean(
                [features.aggressive_flow_imbalance for features in window]
            ),
            avg_oi_price_efficiency=_mean(
                [features.oi_price_efficiency for features in window]
            ),
            latest=latest,
        )

    def _build_candidates(
        self,
        window: DivergenceWindowSummary,
    ) -> list[DivergenceCandidate]:
        return [
            DivergenceCandidate(self._detect_price_up_oi_down(window)),
            DivergenceCandidate(self._detect_price_down_oi_down(window)),
            DivergenceCandidate(self._detect_price_up_oi_flat(window)),
            DivergenceCandidate(self._detect_price_down_oi_flat(window)),
            DivergenceCandidate(self._detect_weak_breakout_up(window)),
            DivergenceCandidate(self._detect_weak_breakout_down(window)),
            DivergenceCandidate(self._detect_exhaustion_up(window)),
            DivergenceCandidate(self._detect_exhaustion_down(window)),
            DivergenceCandidate(self._detect_bullish(window)),
            DivergenceCandidate(self._detect_bearish(window)),
        ]

    def _select_best_candidate(
        self,
        candidates: list[DivergenceCandidate],
    ) -> DivergenceCandidate | None:
        if not candidates:
            return None

        def effective_confidence(candidate: DivergenceCandidate) -> float:
            confidence = candidate.confidence

            if (
                candidate.divergence_type
                in {
                    OIDivergenceType.EXHAUSTION_UP,
                    OIDivergenceType.EXHAUSTION_DOWN,
                }
                and candidate.score >= MIN_DIVERGENCE_SCORE
            ):
                confidence += EXHAUSTION_SELECTION_BONUS

            return _clamp(confidence)

        return max(
            candidates,
            key=lambda candidate: (
                effective_confidence(candidate),
                candidate.score,
                candidate.priority,
                candidate.reasons_count,
            ),
        )

    def _base_not_detected(
        self,
        window: DivergenceWindowSummary,
    ) -> OIDivergenceResult:
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
        detected = score >= MIN_DIVERGENCE_SCORE
        normalized_reasons = list(dict.fromkeys(reasons or []))

        return OIDivergenceResult(
            detected=detected,
            divergence_type=divergence_type if detected else OIDivergenceType.NONE,
            confidence=score if detected else 0.0,
            reasons=(
                normalized_reasons
                if detected
                else ["signal_too_weak_for_divergence"]
            ),
            window_size=window.window_size,
            score=score,
        )

    # ------------------------------------------------------------------
    # Shared predicates
    # ------------------------------------------------------------------

    def _is_flat_oi_response(
        self,
        oi_delta_pct_total: float | None,
    ) -> bool:
        return (
            oi_delta_pct_total is not None
            and abs(oi_delta_pct_total)
            <= self.thresholds.divergence_max_oi_response_pct
        )

    def _is_down_oi_response(
        self,
        oi_delta_pct_total: float | None,
    ) -> bool:
        return (
            oi_delta_pct_total is not None
            and oi_delta_pct_total
            <= -self.thresholds.divergence_max_oi_response_pct
        )

    def _is_up_oi_response(
        self,
        oi_delta_pct_total: float | None,
    ) -> bool:
        return (
            oi_delta_pct_total is not None
            and oi_delta_pct_total
            >= self.thresholds.divergence_max_oi_response_pct
        )

    def _is_weak_volume(
        self,
        avg_volume_ratio: float | None,
    ) -> bool:
        return (
            avg_volume_ratio is not None
            and avg_volume_ratio < WEAK_VOLUME_RATIO
        )

    def _is_strong_volume(
        self,
        avg_volume_ratio: float | None,
    ) -> bool:
        return (
            avg_volume_ratio is not None
            and avg_volume_ratio >= self.thresholds.volume_confirmation_ratio
        )

    def _is_strong_price_move(
        self,
        price_delta_pct_total: float | None,
    ) -> bool:
        return (
            price_delta_pct_total is not None
            and abs(price_delta_pct_total)
            >= self.thresholds.divergence_min_price_move_pct
        )

    def _is_extreme_positive_funding(
        self,
        avg_funding_rate: float | None,
    ) -> bool:
        return (
            avg_funding_rate is not None
            and avg_funding_rate >= self.thresholds.squeeze_funding_abs_threshold
        )

    def _is_extreme_negative_funding(
        self,
        avg_funding_rate: float | None,
    ) -> bool:
        return (
            avg_funding_rate is not None
            and avg_funding_rate <= -self.thresholds.squeeze_funding_abs_threshold
        )

    def _is_extreme_positive_pressure(
        self,
        avg_pressure_score: float | None,
    ) -> bool:
        return (
            avg_pressure_score is not None
            and avg_pressure_score >= self.thresholds.pressure_score_exhaustion_threshold
        )

    def _is_extreme_negative_pressure(
        self,
        avg_pressure_score: float | None,
    ) -> bool:
        return (
            avg_pressure_score is not None
            and avg_pressure_score <= -self.thresholds.pressure_score_exhaustion_threshold
        )

    # ------------------------------------------------------------------
    # Divergence rules
    # ------------------------------------------------------------------

    def _detect_price_up_oi_down(
        self,
        window: DivergenceWindowSummary,
    ) -> OIDivergenceResult:
        score = 0.0
        reasons: list[str] = []

        if (
            window.price_delta_pct_total is not None
            and window.price_delta_pct_total
            >= self.thresholds.divergence_min_price_move_pct
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
            and window.price_delta_pct_total
            <= -self.thresholds.divergence_min_price_move_pct
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

        if (
            self._is_extreme_negative_pressure(window.avg_pressure_score)
            or self._is_extreme_negative_funding(window.avg_funding_rate)
            or (
                window.avg_oi_zscore is not None
                and window.avg_oi_zscore <= -self.thresholds.anomaly_zscore_threshold
            )
        ):
            score -= 0.06
            reasons.append("exhaustion_specific_context_penalty")

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
            and window.price_delta_pct_total
            >= self.thresholds.divergence_min_price_move_pct
        ):
            score += 0.36
            reasons.append("price_trending_up")

        if self._is_flat_oi_response(window.oi_delta_pct_total):
            score += 0.30
            reasons.append("oi_flat_response")

        if self._is_weak_volume(window.avg_volume_ratio):
            score += 0.10
            reasons.append("weak_volume")

        if (
            window.avg_pressure_score is not None
            and window.avg_pressure_score < WEAK_PRESSURE_UPPER
        ):
            score += 0.08
            reasons.append("pressure_not_confirming_uptrend")

        if (
            window.avg_oi_zscore is not None
            and abs(window.avg_oi_zscore) < 0.75
        ):
            score += 0.05
            reasons.append("oi_not_statistically_expanding")

        # If the latest bar is a clear weak breakout, this window-level label
        # should not suppress WEAK_BREAKOUT_UP.
        latest = window.latest
        if (
            latest.price_delta_pct is not None
            and latest.price_delta_pct >= self.thresholds.divergence_min_price_move_pct
            and latest.volume_ratio is not None
            and latest.volume_ratio < WEAK_VOLUME_RATIO
        ):
            score -= 0.10
            reasons.append("latest_weak_breakout_specific_context_penalty")

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
            and window.price_delta_pct_total
            <= -self.thresholds.divergence_min_price_move_pct
        ):
            score += 0.36
            reasons.append("price_trending_down")

        if self._is_flat_oi_response(window.oi_delta_pct_total):
            score += 0.30
            reasons.append("oi_flat_response")

        if self._is_weak_volume(window.avg_volume_ratio):
            score += 0.10
            reasons.append("weak_volume")

        if (
            window.avg_pressure_score is not None
            and window.avg_pressure_score > WEAK_PRESSURE_LOWER
        ):
            score += 0.08
            reasons.append("pressure_not_confirming_downtrend")

        if (
            window.avg_oi_zscore is not None
            and abs(window.avg_oi_zscore) < 0.75
        ):
            score += 0.05
            reasons.append("oi_not_statistically_expanding")

        # If the latest bar is a clear weak breakdown, this window-level label
        # should not suppress WEAK_BREAKOUT_DOWN.
        latest = window.latest
        if (
            latest.price_delta_pct is not None
            and latest.price_delta_pct <= -self.thresholds.divergence_min_price_move_pct
            and latest.volume_ratio is not None
            and latest.volume_ratio < WEAK_VOLUME_RATIO
        ):
            score -= 0.10
            reasons.append("latest_weak_breakdown_specific_context_penalty")

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
            and latest.price_delta_pct
            >= self.thresholds.divergence_min_price_move_pct
        ):
            score += 0.27
            reasons.append("weak_latest_up_move")

        if latest.price_direction == OIDirection.UP:
            score += 0.08
            reasons.append("weak_latest_price_direction_up")

        if latest.oi_delta_pct <= self.thresholds.divergence_max_oi_response_pct:
            score += 0.22
            reasons.append("weak_latest_oi_not_confirming_breakout")

        if latest.volume_ratio is not None and latest.volume_ratio < WEAK_VOLUME_RATIO:
            score += 0.14
            reasons.append("weak_latest_breakout_on_weak_volume")

        if (
            latest.aggressive_flow_imbalance is not None
            and latest.aggressive_flow_imbalance < 0.08
        ):
            score += 0.09
            reasons.append("weak_insufficient_aggressive_buy_flow")

        if (
            latest.oi_price_efficiency is not None
            and abs(latest.oi_price_efficiency) < 0.35
        ):
            score += 0.09
            reasons.append("weak_oi_price_efficiency")

        if latest.oi_zscore is not None and abs(latest.oi_zscore) < 0.75:
            score += 0.05
            reasons.append("weak_no_oi_expansion_extreme")

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
            and latest.price_delta_pct
            <= -self.thresholds.divergence_min_price_move_pct
        ):
            score += 0.27
            reasons.append("weak_latest_down_move")

        if latest.price_direction == OIDirection.DOWN:
            score += 0.08
            reasons.append("weak_latest_price_direction_down")

        if latest.oi_delta_pct >= -self.thresholds.divergence_max_oi_response_pct:
            score += 0.22
            reasons.append("weak_latest_oi_not_confirming_breakdown")

        if latest.volume_ratio is not None and latest.volume_ratio < WEAK_VOLUME_RATIO:
            score += 0.14
            reasons.append("weak_latest_breakdown_on_weak_volume")

        if (
            latest.aggressive_flow_imbalance is not None
            and latest.aggressive_flow_imbalance > -0.08
        ):
            score += 0.09
            reasons.append("weak_insufficient_aggressive_sell_flow")

        if (
            latest.oi_price_efficiency is not None
            and abs(latest.oi_price_efficiency) < 0.35
        ):
            score += 0.09
            reasons.append("weak_oi_price_efficiency")

        if latest.oi_zscore is not None and abs(latest.oi_zscore) < 0.75:
            score += 0.05
            reasons.append("weak_no_oi_expansion_extreme")

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
        score = 0.0
        reasons: list[str] = []

        if (
            window.price_direction is OIDirection.UP
            and self._is_strong_price_move(window.price_delta_pct_total)
        ):
            score += 0.18
            reasons.append("exhaustion_price_trending_up")

        if window.oi_direction is OIDirection.UP:
            score += 0.12
            reasons.append("exhaustion_oi_expanding")

        if (
            window.avg_oi_zscore is not None
            and window.avg_oi_zscore >= self.thresholds.anomaly_zscore_threshold
        ):
            score += 0.16
            reasons.append("exhaustion_elevated_oi_zscore")

        if self._is_extreme_positive_pressure(window.avg_pressure_score):
            score += 0.18
            reasons.append("exhaustion_extreme_positive_pressure")

        if self._is_extreme_positive_funding(window.avg_funding_rate):
            score += 0.14
            reasons.append("exhaustion_extreme_positive_funding")

        if (
            window.avg_liquidation_imbalance is not None
            and window.avg_liquidation_imbalance >= EXHAUSTION_LIQUIDATION_THRESHOLD
        ):
            score += 0.10
            reasons.append("exhaustion_short_liquidation_pressure")

        if (
            window.avg_aggressive_flow_imbalance is not None
            and window.avg_aggressive_flow_imbalance
            >= self.thresholds.aggressive_flow_confirmation
        ):
            score += 0.08
            reasons.append("exhaustion_aggressive_buy_flow")

        if (
            window.avg_oi_price_efficiency is not None
            and window.avg_oi_price_efficiency >= EXHAUSTION_EFFICIENCY_THRESHOLD
        ):
            score += 0.08
            reasons.append("exhaustion_high_oi_price_efficiency")

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
        score = 0.0
        reasons: list[str] = []

        if (
            window.price_direction is OIDirection.DOWN
            and self._is_strong_price_move(window.price_delta_pct_total)
        ):
            score += 0.18
            reasons.append("exhaustion_price_trending_down")

        if window.oi_direction in {OIDirection.DOWN, OIDirection.UP}:
            score += 0.12
            reasons.append("exhaustion_oi_positioning_extreme_or_contracting")

        if (
            window.avg_oi_zscore is not None
            and window.avg_oi_zscore <= -self.thresholds.anomaly_zscore_threshold
        ):
            score += 0.16
            reasons.append("exhaustion_negative_oi_zscore")

        if self._is_extreme_negative_pressure(window.avg_pressure_score):
            score += 0.18
            reasons.append("exhaustion_extreme_negative_pressure")

        if self._is_extreme_negative_funding(window.avg_funding_rate):
            score += 0.14
            reasons.append("exhaustion_extreme_negative_funding")

        if (
            window.avg_liquidation_imbalance is not None
            and window.avg_liquidation_imbalance <= -EXHAUSTION_LIQUIDATION_THRESHOLD
        ):
            score += 0.10
            reasons.append("exhaustion_long_liquidation_pressure")

        if (
            window.avg_aggressive_flow_imbalance is not None
            and window.avg_aggressive_flow_imbalance
            <= -self.thresholds.aggressive_flow_confirmation
        ):
            score += 0.08
            reasons.append("exhaustion_aggressive_sell_flow")

        if (
            window.avg_oi_price_efficiency is not None
            and abs(window.avg_oi_price_efficiency) >= EXHAUSTION_EFFICIENCY_THRESHOLD
        ):
            score += 0.08
            reasons.append("exhaustion_high_oi_price_efficiency")

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
            and window.price_delta_pct_total
            <= -self.thresholds.divergence_min_price_move_pct
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

        if window.avg_funding_rate is not None and window.avg_funding_rate < 0:
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
            and window.price_delta_pct_total
            >= self.thresholds.divergence_min_price_move_pct
        ):
            score += 0.28
            reasons.append("price_strength_present")

        if (
            self._is_down_oi_response(window.oi_delta_pct_total)
            or self._is_flat_oi_response(window.oi_delta_pct_total)
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

        if window.avg_funding_rate is not None and window.avg_funding_rate > 0:
            score += 0.05
            reasons.append("positive_funding_crowding")

        if window.avg_oi_zscore is not None and window.avg_oi_zscore < 1.0:
            score += 0.05
            reasons.append("oi_not_statistically_supportive")

        return self._make_result(
            window=window,
            divergence_type=OIDivergenceType.BEARISH,
            score=score,
            reasons=reasons,
        )