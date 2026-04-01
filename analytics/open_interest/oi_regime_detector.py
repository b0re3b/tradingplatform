from __future__ import annotations

from dataclasses import dataclass

from .config import OIAnalyzerConfig
from .enums import OIDirection, OIRegime
from .models import OIFeatures, OIRegimeResult


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_abs(value: float | None) -> float:
    return abs(value) if value is not None else 0.0


@dataclass(slots=True)
class RegimeCandidate:
    regime: OIRegime
    score: float
    reasons: list[str]


class OIRegimeDetector:
    """
    Rule-based детектор режимів ринку на основі OI features.

    Логіка побудована так:
    1. Для кожного можливого режиму рахується score.
    2. Обирається режим із найбільшим score.
    3. Якщо score замалий — повертається NEUTRAL.
    """

    def __init__(self, config: OIAnalyzerConfig) -> None:
        self.config = config
        self.thresholds = config.thresholds

    def detect(self, features: OIFeatures) -> OIRegimeResult:
        candidates = [
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

        best = max(candidates, key=lambda c: c.score, default=None)

        if best is None or best.score < 0.35:
            return OIRegimeResult(
                regime=OIRegime.NEUTRAL,
                confidence=0.25 if best is None else min(0.4, best.score),
                reasons=["no_strong_regime_signal"],
                score=0.0 if best is None else best.score,
            )

        return OIRegimeResult(
            regime=best.regime,
            confidence=self._score_to_confidence(best.score),
            reasons=best.reasons,
            score=best.score,
        )

    def _score_to_confidence(self, score: float) -> float:
        """
        Перетворення rule score -> confidence.
        """
        if score <= 0:
            return 0.0
        if score >= 1.0:
            return 0.98
        return _clamp_confidence(0.15 + score * 0.8)

    def _has_volume_confirmation(self, features: OIFeatures) -> bool:
        if not self.config.require_volume_confirmation:
            return True
        return (
            features.volume_ratio is not None
            and features.volume_ratio >= self.thresholds.volume_confirmation_ratio
        )

    def _has_price_context(self, features: OIFeatures) -> bool:
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

    def _oi_up(self, features: OIFeatures) -> bool:
        return features.oi_delta_pct >= self.thresholds.min_oi_change_pct

    def _oi_down(self, features: OIFeatures) -> bool:
        return features.oi_delta_pct <= -self.thresholds.min_oi_change_pct

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

    def _oi_extreme(self, features: OIFeatures) -> bool:
        return (
            features.oi_zscore is not None
            and abs(features.oi_zscore) >= self.thresholds.overheated_zscore_threshold
        )

    def _high_liquidation_pressure_up(self, features: OIFeatures) -> bool:
        return (
            features.liquidation_imbalance is not None
            and features.liquidation_imbalance >= 0.35
        )

    def _high_liquidation_pressure_down(self, features: OIFeatures) -> bool:
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
            features.funding_rate is not None
            and features.funding_rate > 0
            and not self._funding_extreme_positive(features)
        ):
            score += 0.05
            reasons.append("healthy_positive_funding")

        if (
            features.oi_ma_fast is not None
            and features.oi_ma_slow is not None
            and features.oi_ma_fast > features.oi_ma_slow
        ):
            score += 0.06
            reasons.append("fast_oi_above_slow_oi")

        return RegimeCandidate(
            regime=OIRegime.LONG_BUILDUP,
            score=min(score, 1.0),
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
            features.funding_rate is not None
            and features.funding_rate < 0
            and not self._funding_extreme_negative(features)
        ):
            score += 0.05
            reasons.append("healthy_negative_funding")

        if (
            features.oi_ma_fast is not None
            and features.oi_ma_slow is not None
            and features.oi_ma_fast > features.oi_ma_slow
        ):
            score += 0.06
            reasons.append("fast_oi_above_slow_oi")

        return RegimeCandidate(
            regime=OIRegime.SHORT_BUILDUP,
            score=min(score, 1.0),
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

        if (
            features.funding_rate is not None
            and features.funding_rate < 0
        ):
            score += 0.06
            reasons.append("negative_funding_context")

        if (
            features.oi_velocity is not None
            and features.oi_velocity < 0
        ):
            score += 0.06
            reasons.append("negative_oi_velocity")

        return RegimeCandidate(
            regime=OIRegime.SHORT_COVERING,
            score=min(score, 1.0),
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

        if (
            features.funding_rate is not None
            and features.funding_rate > 0
        ):
            score += 0.06
            reasons.append("positive_funding_context")

        if (
            features.oi_velocity is not None
            and features.oi_velocity < 0
        ):
            score += 0.06
            reasons.append("negative_oi_velocity")

        return RegimeCandidate(
            regime=OIRegime.LONG_UNWIND,
            score=min(score, 1.0),
            reasons=reasons,
        )

    def _detect_trend_confirmation(self, features: OIFeatures) -> RegimeCandidate:
        score = 0.0
        reasons: list[str] = []

        if self._price_up(features) and self._oi_up(features):
            score += 0.22
            reasons.append("price_up_oi_up")
        elif self._price_down(features) and self._oi_up(features):
            score += 0.22
            reasons.append("price_down_oi_up")

        if self._has_volume_confirmation(features):
            score += 0.16
            reasons.append("volume_confirmation")

        if (
            features.oi_ma_fast is not None
            and features.oi_ma_slow is not None
            and features.oi_ma_fast > features.oi_ma_slow
        ):
            score += 0.12
            reasons.append("fast_oi_above_slow_oi")

        if (
            features.oi_zscore is not None
            and features.oi_zscore > 0.5
        ):
            score += 0.08
            reasons.append("positive_oi_zscore")

        if features.oi_price_efficiency is not None:
            if abs(features.oi_price_efficiency) >= 0.75:
                score += 0.10
                reasons.append("oi_price_efficiency_confirmation")

        if features.oi_pressure_score is not None:
            if abs(features.oi_pressure_score) >= self.thresholds.pressure_score_trend_threshold:
                score += 0.14
                reasons.append("pressure_score_confirmation")

        if (
            features.volume_ratio is not None
            and features.volume_ratio >= 1.5
        ):
            score += 0.08
            reasons.append("strong_volume")

        return RegimeCandidate(
            regime=OIRegime.TREND_CONFIRMATION,
            score=min(score, 1.0),
            reasons=reasons,
        )

    def _detect_trend_exhaustion(self, features: OIFeatures) -> RegimeCandidate:
        score = 0.0
        reasons: list[str] = []

        if self._has_price_context(features):
            score += 0.05
            reasons.append("price_context_present")

        if self._oi_extreme(features):
            score += 0.20
            reasons.append("oi_extreme")

        if self._extreme_pressure(features):
            score += 0.20
            reasons.append("extreme_pressure")

        if (
            features.oi_acceleration is not None
            and (
                (features.price_direction == OIDirection.UP and features.oi_acceleration < 0)
                or (features.price_direction == OIDirection.DOWN and features.oi_acceleration > 0)
            )
        ):
            score += 0.14
            reasons.append("oi_acceleration_divergence")

        if (
            features.oi_price_efficiency is not None
            and abs(features.oi_price_efficiency) < 0.25
            and _safe_abs(features.price_delta_pct) >= self.thresholds.min_price_change_pct
        ):
            score += 0.15
            reasons.append("weak_oi_support_for_price_move")

        if (
            features.volume_ratio is not None
            and features.volume_ratio < 1.0
            and _safe_abs(features.price_delta_pct) >= self.thresholds.min_price_change_pct
        ):
            score += 0.10
            reasons.append("weak_volume_follow_through")

        if (
            self._funding_extreme_positive(features) and features.price_direction == OIDirection.UP
        ) or (
            self._funding_extreme_negative(features) and features.price_direction == OIDirection.DOWN
        ):
            score += 0.12
            reasons.append("extreme_funding_crowding")

        if (
            features.oi_direction == OIDirection.DOWN
            and _safe_abs(features.price_delta_pct) >= self.thresholds.min_price_change_pct
        ):
            score += 0.10
            reasons.append("oi_falling_while_price_still_trending")

        return RegimeCandidate(
            regime=OIRegime.TREND_EXHAUSTION,
            score=min(score, 1.0),
            reasons=reasons,
        )

    def _detect_squeeze_setup(self, features: OIFeatures) -> RegimeCandidate:
        score = 0.0
        reasons: list[str] = []

        if features.oi_delta_pct >= self.thresholds.squeeze_oi_build_pct:
            score += 0.22
            reasons.append("strong_oi_buildup")

        if self._oi_extreme(features):
            score += 0.16
            reasons.append("oi_extreme")

        if self._funding_extreme_positive(features):
            score += 0.16
            reasons.append("extreme_positive_funding")
        elif self._funding_extreme_negative(features):
            score += 0.16
            reasons.append("extreme_negative_funding")

        if (
            features.volume_ratio is not None
            and features.volume_ratio >= self.thresholds.volume_confirmation_ratio
        ):
            score += 0.10
            reasons.append("elevated_volume")

        if (
            features.price_direction == OIDirection.FLAT
            or (
                features.price_delta_pct is not None
                and abs(features.price_delta_pct) < self.thresholds.min_price_change_pct
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
            and abs(features.liquidation_imbalance) >= 0.20
        ):
            score += 0.08
            reasons.append("liquidation_pressure_present")

        return RegimeCandidate(
            regime=OIRegime.SQUEEZE_SETUP,
            score=min(score, 1.0),
            reasons=reasons,
        )

    def _detect_capitulation(self, features: OIFeatures) -> RegimeCandidate:
        score = 0.0
        reasons: list[str] = []

        if (
            features.price_delta_pct is not None
            and abs(features.price_delta_pct) >= self.thresholds.capitulation_price_move_pct
        ):
            score += 0.28
            reasons.append("large_price_move")

        if features.oi_delta_pct <= -self.thresholds.capitulation_oi_drop_pct:
            score += 0.28
            reasons.append("sharp_oi_drop")

        if (
            features.liquidation_imbalance is not None
            and abs(features.liquidation_imbalance) >= 0.45
        ):
            score += 0.16
            reasons.append("heavy_liquidation_imbalance")

        if (
            features.volume_ratio is not None
            and features.volume_ratio >= 1.5
        ):
            score += 0.10
            reasons.append("high_volume")

        if (
            features.oi_zscore is not None
            and features.oi_zscore <= -self.thresholds.anomaly_zscore_threshold
        ):
            score += 0.10
            reasons.append("negative_oi_zscore_extreme")

        if (
            features.oi_velocity is not None
            and features.oi_velocity < 0
        ):
            score += 0.08
            reasons.append("negative_oi_velocity")

        return RegimeCandidate(
            regime=OIRegime.CAPITULATION,
            score=min(score, 1.0),
            reasons=reasons,
        )

    def _detect_overheated(self, features: OIFeatures) -> RegimeCandidate:
        score = 0.0
        reasons: list[str] = []

        if self._oi_extreme(features):
            score += 0.24
            reasons.append("oi_extreme")

        if self._extreme_pressure(features):
            score += 0.18
            reasons.append("extreme_pressure")

        if self._funding_extreme_positive(features):
            score += 0.14
            reasons.append("extreme_positive_funding")
        elif self._funding_extreme_negative(features):
            score += 0.14
            reasons.append("extreme_negative_funding")

        if (
            features.volume_ratio is not None
            and features.volume_ratio >= 1.4
        ):
            score += 0.08
            reasons.append("high_volume")

        if (
            features.oi_delta_pct >= self.thresholds.squeeze_oi_build_pct
        ):
            score += 0.10
            reasons.append("strong_oi_expansion")

        if (
            features.oi_ma_fast is not None
            and features.oi_ma_slow is not None
            and features.oi_ma_fast > features.oi_ma_slow
        ):
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
            and abs(features.liquidation_imbalance) >= 0.30
        ):
            score += 0.04
            reasons.append("liquidation_imbalance")

        return RegimeCandidate(
            regime=OIRegime.OVERHEATED,
            score=min(score, 1.0),
            reasons=reasons,
        )