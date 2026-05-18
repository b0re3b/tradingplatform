from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

from .config import OIAnalyzerConfig
from .enums import OIDirection, OIRegime
from .models import OIFeatures, OIRegimeResult


MIN_REGIME_SCORE: Final[float] = 0.35
DEFAULT_NEUTRAL_CONFIDENCE: Final[float] = 0.25
MAX_NEUTRAL_CONFIDENCE: Final[float] = 0.40
MAX_RULE_CONFIDENCE: Final[float] = 0.98

# Selection tuning.
CAPITULATION_SELECTION_BONUS: Final[float] = 0.16
OVERHEATED_SELECTION_BONUS: Final[float] = 0.16
TREND_EXHAUSTION_SELECTION_BONUS: Final[float] = 0.14
SQUEEZE_SELECTION_BONUS: Final[float] = 0.08
TREND_CONFIRMATION_SELECTION_BONUS: Final[float] = 0.04

# Generic buildup should not dominate dense risk contexts.
RISK_CONTEXT_BUILDUP_PENALTY: Final[float] = 0.22

# Trend confirmation should mean a stronger trend, not just any price/OI alignment.
MIN_TREND_CONFIRMATION_PRICE_MOVE_PCT: Final[float] = 1.0

# Dense overheated context thresholds.
OVERHEATED_LIQUIDATION_CONTEXT_ABS: Final[float] = 0.30
LIQUIDATION_CONTEXT_ABS: Final[float] = 0.20


REGIME_PRIORITY: Final[dict[OIRegime, int]] = {
    # Risk / liquidation regimes should win close calls over generic regimes.
    OIRegime.CAPITULATION: 100,
    OIRegime.OVERHEATED: 96,
    OIRegime.SQUEEZE_SETUP: 94,
    OIRegime.TREND_EXHAUSTION: 92,

    # Directional unwind / covering regimes are more specific than generic trend.
    OIRegime.SHORT_COVERING: 80,
    OIRegime.LONG_UNWIND: 80,

    # Generic trend confirmation.
    OIRegime.TREND_CONFIRMATION: 70,

    # Build-up regimes.
    OIRegime.LONG_BUILDUP: 60,
    OIRegime.SHORT_BUILDUP: 60,

    OIRegime.NEUTRAL: 0,
}


def _clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    if low > high:
        raise ValueError("low must be <= high")

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return low

    if not math.isfinite(number):
        return low

    return max(low, min(high, number))


def _safe_abs(value: float | None) -> float:
    if value is None:
        return 0.0

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0

    if not math.isfinite(number):
        return 0.0

    return abs(number)


def _is_positive(value: float | None) -> bool:
    return value is not None and value > 0


def _is_negative(value: float | None) -> bool:
    return value is not None and value < 0


@dataclass(slots=True)
class RegimeCandidate:
    """
    Internal score container for rule-based OI regime classification.

    This is intentionally a pure value object:
    - no EventBus;
    - no Scheduler;
    - no logger;
    - no side effects.
    """

    regime: OIRegime
    score: float
    reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.score = _clamp(self.score)
        self.reasons = list(dict.fromkeys(self.reasons or []))

    @property
    def priority(self) -> int:
        return REGIME_PRIORITY.get(self.regime, 0)

    @property
    def reasons_count(self) -> int:
        return len(self.reasons)


class OIRegimeDetector:
    """
    Rule-based detector for futures Open Interest market regimes.

    This is a pure domain service:
    - no EventBus;
    - no Scheduler;
    - no logger;
    - no IO;
    - no mutable runtime state.

    It receives OIFeatures and returns OIRegimeResult.
    """

    def __init__(self, config: OIAnalyzerConfig) -> None:
        self.config = config
        self.thresholds = config.thresholds

    def detect(self, features: OIFeatures) -> OIRegimeResult:
        """
        Detect current OI regime from a single feature snapshot.

        The detector evaluates every supported regime independently and then
        selects the best candidate using an effective score.

        Important selection behavior:
        - generic buildup regimes are penalized in dense risk contexts;
        - risk regimes get granular selection bonuses;
        - trend confirmation is penalized in extreme/risk contexts;
        - final output confidence remains based on the raw bounded rule score.
        """
        candidates = self._build_candidates(features)
        best = self._select_best_candidate(candidates)

        if best is None or best.score < MIN_REGIME_SCORE:
            score = 0.0 if best is None else best.score
            confidence = (
                DEFAULT_NEUTRAL_CONFIDENCE
                if best is None
                else min(MAX_NEUTRAL_CONFIDENCE, score)
            )

            return OIRegimeResult(
                regime=OIRegime.NEUTRAL,
                confidence=confidence,
                reasons=["no_strong_regime_signal"],
                score=score,
            )

        return OIRegimeResult(
            regime=best.regime,
            confidence=self._score_to_confidence(best.score),
            reasons=best.reasons,
            score=best.score,
        )

    def _build_candidates(self, features: OIFeatures) -> list[RegimeCandidate]:
        return [
            self._detect_long_buildup(features),
            self._detect_short_buildup(features),
            self._detect_short_covering(features),
            self._detect_long_unwind(features),
            self._detect_trend_confirmation(features),
            self._detect_trend_exhaustion(features),
            self._detect_squeeze_setup(features),
            self._detect_capitulation(features),
            self._detect_overheated(features),
        ]

    def _select_best_candidate(
        self,
        candidates: list[RegimeCandidate],
    ) -> RegimeCandidate | None:
        if not candidates:
            return None

        def effective_score(candidate: RegimeCandidate) -> float:
            score = candidate.score

            if candidate.score < MIN_REGIME_SCORE:
                return score

            if candidate.regime is OIRegime.CAPITULATION:
                score += CAPITULATION_SELECTION_BONUS

            elif candidate.regime is OIRegime.OVERHEATED:
                score += OVERHEATED_SELECTION_BONUS

            elif candidate.regime is OIRegime.TREND_EXHAUSTION:
                score += TREND_EXHAUSTION_SELECTION_BONUS

            elif candidate.regime is OIRegime.SQUEEZE_SETUP:
                score += SQUEEZE_SELECTION_BONUS

            elif candidate.regime is OIRegime.TREND_CONFIRMATION:
                score += TREND_CONFIRMATION_SELECTION_BONUS

            return _clamp(score)

        return max(
            candidates,
            key=lambda candidate: (
                effective_score(candidate),
                candidate.priority,
                candidate.score,
                candidate.reasons_count,
            ),
        )

    def _score_to_confidence(self, score: float) -> float:
        """
        Convert bounded rule score into bounded model confidence.
        """
        score = _clamp(score)

        if score <= 0:
            return 0.0

        if score >= 1.0:
            return MAX_RULE_CONFIDENCE

        return _clamp(0.15 + score * 0.8)

    # ------------------------------------------------------------------
    # Shared predicates
    # ------------------------------------------------------------------

    def _has_volume_confirmation(self, features: OIFeatures) -> bool:
        if not self.config.require_volume_confirmation:
            return True

        return (
            features.volume_ratio is not None
            and features.volume_ratio >= self.thresholds.volume_confirmation_ratio
        )

    @staticmethod
    def _has_price_context(features: OIFeatures) -> bool:
        return features.price_delta_pct is not None

    def _price_up(self, features: OIFeatures) -> bool:
        return (
            features.price_delta_pct is not None
            and features.price_delta_pct >= self.thresholds.min_price_change_pct
        )

    def _price_down(self, features: OIFeatures) -> bool:
        return (
            features.price_delta_pct is not None
            and features.price_delta_pct <= -self.thresholds.min_price_change_pct
        )

    def _strong_price_up(self, features: OIFeatures) -> bool:
        return (
            features.price_delta_pct is not None
            and features.price_delta_pct >= MIN_TREND_CONFIRMATION_PRICE_MOVE_PCT
        )

    def _strong_price_down(self, features: OIFeatures) -> bool:
        return (
            features.price_delta_pct is not None
            and features.price_delta_pct <= -MIN_TREND_CONFIRMATION_PRICE_MOVE_PCT
        )

    def _strong_price_move(self, features: OIFeatures) -> bool:
        return (
            features.price_delta_pct is not None
            and abs(features.price_delta_pct) >= MIN_TREND_CONFIRMATION_PRICE_MOVE_PCT
        )

    def _oi_up(self, features: OIFeatures) -> bool:
        return features.oi_delta_pct >= self.thresholds.min_oi_change_pct

    def _oi_down(self, features: OIFeatures) -> bool:
        return features.oi_delta_pct <= -self.thresholds.min_oi_change_pct

    def _strong_oi_up(self, features: OIFeatures) -> bool:
        return features.oi_delta_pct >= self.thresholds.squeeze_oi_build_pct

    def _strong_oi_down(self, features: OIFeatures) -> bool:
        return features.oi_delta_pct <= -self.thresholds.capitulation_oi_drop_pct

    def _positive_pressure(self, features: OIFeatures) -> bool:
        return (
            features.oi_pressure_score is not None
            and features.oi_pressure_score >= self.thresholds.pressure_score_trend_threshold
        )

    def _negative_pressure(self, features: OIFeatures) -> bool:
        return (
            features.oi_pressure_score is not None
            and features.oi_pressure_score <= -self.thresholds.pressure_score_trend_threshold
        )

    def _extreme_pressure(self, features: OIFeatures) -> bool:
        return (
            features.oi_pressure_score is not None
            and abs(features.oi_pressure_score)
            >= self.thresholds.pressure_score_exhaustion_threshold
        )

    def _funding_extreme_positive(self, features: OIFeatures) -> bool:
        return (
            features.funding_rate is not None
            and features.funding_rate >= self.thresholds.funding_extreme_positive
        )

    def _funding_extreme_negative(self, features: OIFeatures) -> bool:
        return (
            features.funding_rate is not None
            and features.funding_rate <= self.thresholds.funding_extreme_negative
        )

    def _has_extreme_funding(self, features: OIFeatures) -> bool:
        return (
            self._funding_extreme_positive(features)
            or self._funding_extreme_negative(features)
        )

    def _oi_extreme(self, features: OIFeatures) -> bool:
        return (
            features.oi_zscore is not None
            and abs(features.oi_zscore) >= self.thresholds.overheated_zscore_threshold
        )

    def _oi_anomalous_positive(self, features: OIFeatures) -> bool:
        return (
            features.oi_zscore is not None
            and features.oi_zscore >= self.thresholds.anomaly_zscore_threshold
        )

    def _oi_anomalous_negative(self, features: OIFeatures) -> bool:
        return (
            features.oi_zscore is not None
            and features.oi_zscore <= -self.thresholds.anomaly_zscore_threshold
        )

    @staticmethod
    def _high_liquidation_pressure_up(features: OIFeatures) -> bool:
        return (
            features.liquidation_imbalance is not None
            and features.liquidation_imbalance >= 0.35
        )

    @staticmethod
    def _high_liquidation_pressure_down(features: OIFeatures) -> bool:
        return (
            features.liquidation_imbalance is not None
            and features.liquidation_imbalance <= -0.35
        )

    def _aggressive_buy_confirmation(self, features: OIFeatures) -> bool:
        return (
            features.aggressive_flow_imbalance is not None
            and features.aggressive_flow_imbalance
            >= self.thresholds.aggressive_flow_confirmation
        )

    def _aggressive_sell_confirmation(self, features: OIFeatures) -> bool:
        return (
            features.aggressive_flow_imbalance is not None
            and features.aggressive_flow_imbalance
            <= -self.thresholds.aggressive_flow_confirmation
        )

    @staticmethod
    def _fast_oi_above_slow_oi(features: OIFeatures) -> bool:
        return (
            features.oi_ma_fast is not None
            and features.oi_ma_slow is not None
            and features.oi_ma_fast > features.oi_ma_slow
        )

    @staticmethod
    def _fast_oi_below_slow_oi(features: OIFeatures) -> bool:
        return (
            features.oi_ma_fast is not None
            and features.oi_ma_slow is not None
            and features.oi_ma_fast < features.oi_ma_slow
        )

    def _risk_context_up(self, features: OIFeatures) -> bool:
        return (
            self._funding_extreme_positive(features)
            or self._oi_extreme(features)
            or self._extreme_pressure(features)
            or self._high_liquidation_pressure_up(features)
        )

    def _risk_context_down(self, features: OIFeatures) -> bool:
        return (
            self._funding_extreme_negative(features)
            or self._oi_extreme(features)
            or self._extreme_pressure(features)
            or self._high_liquidation_pressure_down(features)
        )

    def _dense_overheated_context(self, features: OIFeatures) -> bool:
        return (
                self._oi_extreme(features)
                and self._extreme_pressure(features)
                and self._has_extreme_funding(features)
                and (
                        (
                                features.liquidation_imbalance is not None
                                and abs(features.liquidation_imbalance)
                                >= OVERHEATED_LIQUIDATION_CONTEXT_ABS
                        )
                        or (
                                features.aggressive_flow_imbalance is not None
                                and abs(features.aggressive_flow_imbalance) >= 0.20
                        )
                )
        )

    def _dense_overheated_with_liquidations(self, features: OIFeatures) -> bool:
        return (
            self._dense_overheated_context(features)
            and features.liquidation_imbalance is not None
            and abs(features.liquidation_imbalance)
            >= OVERHEATED_LIQUIDATION_CONTEXT_ABS
        )

    # ------------------------------------------------------------------
    # Regime rules
    # ------------------------------------------------------------------

    def _detect_long_buildup(self, features: OIFeatures) -> RegimeCandidate:
        score = 0.0
        reasons: list[str] = []

        if self._price_up(features):
            score += 0.28
            reasons.append("price_up")

        if self._oi_up(features):
            score += 0.28
            reasons.append("oi_up")

        if self._has_volume_confirmation(features):
            score += 0.14
            reasons.append("volume_confirmation")

        if self._positive_pressure(features):
            score += 0.12
            reasons.append("positive_pressure")

        if self._aggressive_buy_confirmation(features):
            score += 0.10
            reasons.append("aggressive_buy_confirmation")

        if (
            _is_positive(features.funding_rate)
            and not self._funding_extreme_positive(features)
        ):
            score += 0.05
            reasons.append("healthy_positive_funding")

        if self._fast_oi_above_slow_oi(features):
            score += 0.06
            reasons.append("fast_oi_above_slow_oi")

        if self._risk_context_up(features):
            score -= RISK_CONTEXT_BUILDUP_PENALTY
            reasons.append("risk_context_penalty")

        return RegimeCandidate(
            regime=OIRegime.LONG_BUILDUP,
            score=score,
            reasons=reasons,
        )

    def _detect_short_buildup(self, features: OIFeatures) -> RegimeCandidate:
        score = 0.0
        reasons: list[str] = []

        if self._price_down(features):
            score += 0.28
            reasons.append("price_down")

        if self._oi_up(features):
            score += 0.28
            reasons.append("oi_up")

        if self._has_volume_confirmation(features):
            score += 0.14
            reasons.append("volume_confirmation")

        if self._negative_pressure(features):
            score += 0.12
            reasons.append("negative_pressure")

        if self._aggressive_sell_confirmation(features):
            score += 0.10
            reasons.append("aggressive_sell_confirmation")

        if (
            _is_negative(features.funding_rate)
            and not self._funding_extreme_negative(features)
        ):
            score += 0.05
            reasons.append("healthy_negative_funding")

        if self._fast_oi_above_slow_oi(features):
            score += 0.06
            reasons.append("fast_oi_above_slow_oi")

        if self._risk_context_down(features):
            score -= RISK_CONTEXT_BUILDUP_PENALTY
            reasons.append("risk_context_penalty")

        return RegimeCandidate(
            regime=OIRegime.SHORT_BUILDUP,
            score=score,
            reasons=reasons,
        )

    def _detect_short_covering(self, features: OIFeatures) -> RegimeCandidate:
        score = 0.0
        reasons: list[str] = []

        if self._price_up(features):
            score += 0.34
            reasons.append("price_up")

        if self._oi_down(features):
            score += 0.34
            reasons.append("oi_down")

        if self._high_liquidation_pressure_up(features):
            score += 0.12
            reasons.append("short_liquidation_pressure")

        if self._aggressive_buy_confirmation(features):
            score += 0.08
            reasons.append("aggressive_buy_confirmation")

        if _is_negative(features.funding_rate):
            score += 0.06
            reasons.append("negative_funding_context")

        if features.oi_velocity is not None and features.oi_velocity < 0:
            score += 0.06
            reasons.append("negative_oi_velocity")

        return RegimeCandidate(
            regime=OIRegime.SHORT_COVERING,
            score=score,
            reasons=reasons,
        )

    def _detect_long_unwind(self, features: OIFeatures) -> RegimeCandidate:
        score = 0.0
        reasons: list[str] = []

        if self._price_down(features):
            score += 0.34
            reasons.append("price_down")

        if self._oi_down(features):
            score += 0.34
            reasons.append("oi_down")

        if self._high_liquidation_pressure_down(features):
            score += 0.12
            reasons.append("long_liquidation_pressure")

        if self._aggressive_sell_confirmation(features):
            score += 0.08
            reasons.append("aggressive_sell_confirmation")

        if _is_positive(features.funding_rate):
            score += 0.06
            reasons.append("positive_funding_context")

        if features.oi_velocity is not None and features.oi_velocity < 0:
            score += 0.06
            reasons.append("negative_oi_velocity")

        return RegimeCandidate(
            regime=OIRegime.LONG_UNWIND,
            score=score,
            reasons=reasons,
        )

    def _detect_trend_confirmation(self, features: OIFeatures) -> RegimeCandidate:
        score = 0.0
        reasons: list[str] = []

        if self._strong_price_up(features) and self._oi_up(features):
            score += 0.32
            reasons.append("strong_price_up_oi_up")

        elif self._strong_price_down(features) and self._oi_up(features):
            score += 0.32
            reasons.append("strong_price_down_oi_up")

        if self._has_volume_confirmation(features):
            score += 0.18
            reasons.append("volume_confirmation")

        if self._fast_oi_above_slow_oi(features):
            score += 0.12
            reasons.append("fast_oi_above_slow_oi")

        if features.oi_zscore is not None and features.oi_zscore > 0.5:
            score += 0.08
            reasons.append("positive_oi_zscore")

        if (
            features.oi_price_efficiency is not None
            and abs(features.oi_price_efficiency) >= 0.75
        ):
            score += 0.14
            reasons.append("oi_price_efficiency_confirmation")

        if (
            features.oi_pressure_score is not None
            and abs(features.oi_pressure_score)
            >= self.thresholds.pressure_score_trend_threshold
        ):
            score += 0.16
            reasons.append("pressure_score_confirmation")

        if features.volume_ratio is not None and features.volume_ratio >= 1.5:
            score += 0.08
            reasons.append("strong_volume")

        if self._aggressive_buy_confirmation(features) or self._aggressive_sell_confirmation(features):
            score += 0.10
            reasons.append("aggressive_flow_confirmation")

        if (
            features.funding_rate is not None
            and not self._has_extreme_funding(features)
            and abs(features.funding_rate) > 0
        ):
            score += 0.06
            reasons.append("healthy_funding_context")

        # Trend confirmation is not meant to classify extreme crowding or
        # exhaustion. Dense risk context should be handled by risk regimes.
        if self._oi_extreme(features) or self._has_extreme_funding(features):
            score -= 0.20
            reasons.append("risk_context_penalty")

        if self._extreme_pressure(features):
            score -= 0.12
            reasons.append("extreme_pressure_penalty")

        if self._dense_overheated_with_liquidations(features):
            score -= 0.10
            reasons.append("overheated_context_penalty")

        return RegimeCandidate(
            regime=OIRegime.TREND_CONFIRMATION,
            score=score,
            reasons=reasons,
        )

    def _detect_trend_exhaustion(self, features: OIFeatures) -> RegimeCandidate:
        score = 0.0
        reasons: list[str] = []

        if self._has_price_context(features):
            score += 0.05
            reasons.append("price_context_present")

        if self._strong_price_move(features):
            score += 0.14
            reasons.append("strong_price_move")

        if self._oi_extreme(features):
            score += 0.18
            reasons.append("oi_extreme")

        elif (
            features.oi_zscore is not None
            and abs(features.oi_zscore) >= self.thresholds.anomaly_zscore_threshold
        ):
            score += 0.12
            reasons.append("oi_anomalous")

        if self._extreme_pressure(features):
            score += 0.22
            reasons.append("extreme_pressure")

        if self._has_extreme_funding(features):
            score += 0.14
            reasons.append("extreme_funding_crowding")

        if self._strong_price_move(features) and self._extreme_pressure(features):
            score += 0.12
            reasons.append("exhaustion_price_pressure_combo")

        if self._has_extreme_funding(features) and self._extreme_pressure(features):
            score += 0.10
            reasons.append("exhaustion_funding_pressure_combo")

        if (
            features.oi_acceleration is not None
            and (
                (
                    self._price_up(features)
                    and features.oi_acceleration < 0
                )
                or (
                    self._price_down(features)
                    and features.oi_acceleration > 0
                )
            )
        ):
            score += 0.12
            reasons.append("oi_acceleration_divergence")

        if (
            features.oi_price_efficiency is not None
            and abs(features.oi_price_efficiency) < 0.25
            and _safe_abs(features.price_delta_pct)
            >= self.thresholds.min_price_change_pct
        ):
            score += 0.14
            reasons.append("weak_oi_support_for_price_move")

        if (
            features.volume_ratio is not None
            and features.volume_ratio < 1.0
            and _safe_abs(features.price_delta_pct)
            >= self.thresholds.min_price_change_pct
        ):
            score += 0.08
            reasons.append("weak_volume_follow_through")

        if (
            self._price_up(features)
            and self._oi_up(features)
            and self._extreme_pressure(features)
        ):
            score += 0.12
            reasons.append("uptrend_crowding_exhaustion")

        if (
            self._price_down(features)
            and self._oi_up(features)
            and self._extreme_pressure(features)
        ):
            score += 0.12
            reasons.append("downtrend_crowding_exhaustion")

        if (
            self._oi_down(features)
            and _safe_abs(features.price_delta_pct)
            >= self.thresholds.min_price_change_pct
        ):
            score += 0.10
            reasons.append("oi_falling_while_price_still_trending")

        return RegimeCandidate(
            regime=OIRegime.TREND_EXHAUSTION,
            score=score,
            reasons=reasons,
        )

    def _detect_squeeze_setup(self, features: OIFeatures) -> RegimeCandidate:
        score = 0.0
        reasons: list[str] = []

        has_funding_context = features.funding_rate is not None
        require_funding_for_squeeze = bool(
            getattr(self.config, "require_funding_for_squeeze", False)
        )

        if require_funding_for_squeeze and not has_funding_context:
            return RegimeCandidate(
                regime=OIRegime.SQUEEZE_SETUP,
                score=0.0,
                reasons=["funding_required_for_squeeze_but_missing"],
            )

        if self._strong_oi_up(features):
            score += 0.24
            reasons.append("strong_oi_buildup")

        if self._oi_extreme(features):
            score += 0.16
            reasons.append("oi_extreme")

        if self._funding_extreme_positive(features):
            score += 0.18
            reasons.append("extreme_positive_funding")

        elif self._funding_extreme_negative(features):
            score += 0.18
            reasons.append("extreme_negative_funding")

        if self._extreme_pressure(features):
            score += 0.12
            reasons.append("extreme_pressure")

        if (
            features.volume_ratio is not None
            and features.volume_ratio >= self.thresholds.volume_confirmation_ratio
        ):
            score += 0.10
            reasons.append("elevated_volume")

        if (
            features.price_delta_pct is not None
            and abs(features.price_delta_pct) < max(
                self.thresholds.min_price_change_pct * 2.0,
                0.50,
            )
        ):
            score += 0.12
            reasons.append("price_compression")

        if (
            features.oi_price_efficiency is not None
            and abs(features.oi_price_efficiency) > 1.25
        ):
            score += 0.08
            reasons.append("oi_outpacing_price")

        if (
            features.aggressive_flow_imbalance is not None
            and abs(features.aggressive_flow_imbalance)
            >= self.thresholds.aggressive_flow_confirmation
        ):
            score += 0.08
            reasons.append("directional_aggressive_flow")

        if (
            features.liquidation_imbalance is not None
            and abs(features.liquidation_imbalance) >= LIQUIDATION_CONTEXT_ABS
        ):
            score += 0.08
            reasons.append("liquidation_pressure_present")

        # If the context is already densely overheated, OVERHEATED should win
        # over generic squeeze setup.
        if self._dense_overheated_with_liquidations(features):
            score -= 0.18
            reasons.append("overheated_context_penalty")

        return RegimeCandidate(
            regime=OIRegime.SQUEEZE_SETUP,
            score=score,
            reasons=reasons,
        )

    def _detect_capitulation(self, features: OIFeatures) -> RegimeCandidate:
        score = 0.0
        reasons: list[str] = []

        if (
            features.price_delta_pct is not None
            and abs(features.price_delta_pct)
            >= self.thresholds.capitulation_price_move_pct
        ):
            score += 0.28
            reasons.append("large_price_move")

        if self._strong_oi_down(features):
            score += 0.28
            reasons.append("sharp_oi_drop")

        if (
            features.liquidation_imbalance is not None
            and abs(features.liquidation_imbalance) >= 0.45
        ):
            score += 0.16
            reasons.append("heavy_liquidation_imbalance")

        if features.volume_ratio is not None and features.volume_ratio >= 1.5:
            score += 0.10
            reasons.append("high_volume")

        if self._oi_anomalous_negative(features):
            score += 0.10
            reasons.append("negative_oi_zscore_extreme")

        if features.oi_velocity is not None and features.oi_velocity < 0:
            score += 0.08
            reasons.append("negative_oi_velocity")

        return RegimeCandidate(
            regime=OIRegime.CAPITULATION,
            score=score,
            reasons=reasons,
        )

    def _detect_overheated(self, features: OIFeatures) -> RegimeCandidate:
        score = 0.0
        reasons: list[str] = []

        if self._oi_extreme(features):
            score += 0.24
            reasons.append("oi_extreme")

        if self._extreme_pressure(features):
            score += 0.20
            reasons.append("extreme_pressure")

        if self._funding_extreme_positive(features):
            score += 0.16
            reasons.append("extreme_positive_funding")

        elif self._funding_extreme_negative(features):
            score += 0.16
            reasons.append("extreme_negative_funding")

        if self._dense_overheated_context(features):
            score += 0.16
            reasons.append("overheated_combo_context")

        if features.volume_ratio is not None and features.volume_ratio >= 1.4:
            score += 0.08
            reasons.append("high_volume")

        if self._strong_oi_up(features):
            score += 0.12
            reasons.append("strong_oi_expansion")

        if self._fast_oi_above_slow_oi(features):
            score += 0.08
            reasons.append("fast_oi_above_slow_oi")

        if (
            features.oi_price_efficiency is not None
            and abs(features.oi_price_efficiency) > 1.5
        ):
            score += 0.08
            reasons.append("oi_much_stronger_than_price")

        if (
            features.aggressive_flow_imbalance is not None
            and abs(features.aggressive_flow_imbalance) >= 0.20
        ):
            score += 0.06
            reasons.append("aggressive_flow_imbalance")

        if (
            features.liquidation_imbalance is not None
            and abs(features.liquidation_imbalance)
            >= OVERHEATED_LIQUIDATION_CONTEXT_ABS
        ):
            score += 0.08
            reasons.append("liquidation_imbalance")

        return RegimeCandidate(
            regime=OIRegime.OVERHEATED,
            score=score,
            reasons=reasons,
        )