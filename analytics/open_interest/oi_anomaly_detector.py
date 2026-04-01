from __future__ import annotations

from dataclasses import dataclass

from .config import OIAnalyzerConfig
from .enums import OIAnomalyType, OISignalStrength
from .models import OIAnomalyResult, OIFeatures


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _safe_abs(value: float | None) -> float:
    return abs(value) if value is not None else 0.0


@dataclass(slots=True)
class AnomalyCandidate:
    anomaly_type: OIAnomalyType
    score: float
    reasons: list[str]


class OIAnomalyDetector:
    """
    Детектор аномалій для Open Interest.

    Основна ідея:
    - оцінити кілька типів аномалій незалежно
    - для кожної зібрати score
    - вибрати найсильнішу
    - якщо score нижче порога -> anomaly не детектиться
    """

    def __init__(self, config: OIAnalyzerConfig) -> None:
        self.config = config
        self.thresholds = config.thresholds

    def detect(self, features: OIFeatures) -> OIAnomalyResult | None:
        candidates = [
            self._detect_oi_spike(features),
            self._detect_oi_collapse(features),
            self._detect_oi_price_dislocation(features),
            self._detect_oi_volume_dislocation(features),
            self._detect_liquidation_driven_oi_drop(features),
            self._detect_overheated_buildup(features),
            self._detect_sudden_deleveraging(features),
            self._detect_funding_oi_imbalance(features),
            self._detect_extreme_crowding(features),
        ]

        best = max(candidates, key=lambda c: c.score, default=None)
        if best is None:
            return None

        if best.score < 0.35:
            return OIAnomalyResult(
                detected=False,
                anomaly_type=OIAnomalyType.NONE,
                strength=OISignalStrength.LOW,
                confidence=0.0,
                reasons=["no_strong_anomaly_detected"],
                score=best.score,
            )

        return OIAnomalyResult(
            detected=True,
            anomaly_type=best.anomaly_type,
            strength=self._score_to_strength(best.score),
            confidence=self._score_to_confidence(best.score),
            reasons=best.reasons,
            score=best.score,
        )

    def _score_to_confidence(self, score: float) -> float:
        if score <= 0:
            return 0.0
        if score >= 1.0:
            return 0.99
        return _clamp(0.15 + score * 0.8)

    def _score_to_strength(self, score: float) -> OISignalStrength:
        if score >= 0.9:
            return OISignalStrength.EXTREME
        if score >= 0.7:
            return OISignalStrength.HIGH
        if score >= 0.5:
            return OISignalStrength.MEDIUM
        return OISignalStrength.LOW

    def _is_positive_oi_extreme(self, features: OIFeatures) -> bool:
        return (
            features.oi_zscore is not None
            and features.oi_zscore >= self.thresholds.anomaly_zscore_threshold
        )

    def _is_negative_oi_extreme(self, features: OIFeatures) -> bool:
        return (
            features.oi_zscore is not None
            and features.oi_zscore <= -self.thresholds.anomaly_zscore_threshold
        )

    def _is_extreme_oi_extreme(self, features: OIFeatures) -> bool:
        return (
            features.oi_zscore is not None
            and abs(features.oi_zscore) >= self.thresholds.extreme_anomaly_zscore_threshold
        )

    def _detect_oi_spike(self, features: OIFeatures) -> AnomalyCandidate:
        score = 0.0
        reasons: list[str] = []

        if self._is_positive_oi_extreme(features):
            score += 0.34
            reasons.append("positive_oi_zscore_extreme")

        if features.oi_delta_pct >= self.thresholds.min_oi_change_pct * 2.0:
            score += 0.24
            reasons.append("large_positive_oi_change")

        if (
            features.oi_velocity is not None
            and features.oi_velocity > 0
        ):
            score += 0.08
            reasons.append("positive_oi_velocity")

        if (
            features.oi_acceleration is not None
            and features.oi_acceleration > 0
        ):
            score += 0.08
            reasons.append("positive_oi_acceleration")

        if (
            features.volume_ratio is not None
            and features.volume_ratio >= self.thresholds.volume_confirmation_ratio
        ):
            score += 0.08
            reasons.append("volume_confirmation")

        if (
            features.oi_ma_fast is not None
            and features.oi_ma_slow is not None
            and features.oi_ma_fast > features.oi_ma_slow
        ):
            score += 0.06
            reasons.append("fast_oi_above_slow_oi")

        if (
            features.oi_pressure_score is not None
            and abs(features.oi_pressure_score) >= self.thresholds.pressure_score_trend_threshold
        ):
            score += 0.06
            reasons.append("pressure_score_elevated")

        return AnomalyCandidate(
            anomaly_type=OIAnomalyType.OI_SPIKE,
            score=min(score, 1.0),
            reasons=reasons,
        )

    def _detect_oi_collapse(self, features: OIFeatures) -> AnomalyCandidate:
        score = 0.0
        reasons: list[str] = []

        if self._is_negative_oi_extreme(features):
            score += 0.34
            reasons.append("negative_oi_zscore_extreme")

        if features.oi_delta_pct <= -(self.thresholds.min_oi_change_pct * 2.0):
            score += 0.24
            reasons.append("large_negative_oi_change")

        if (
            features.oi_velocity is not None
            and features.oi_velocity < 0
        ):
            score += 0.08
            reasons.append("negative_oi_velocity")

        if (
            features.oi_acceleration is not None
            and features.oi_acceleration < 0
        ):
            score += 0.08
            reasons.append("negative_oi_acceleration")

        if (
            features.volume_ratio is not None
            and features.volume_ratio >= self.thresholds.volume_confirmation_ratio
        ):
            score += 0.08
            reasons.append("volume_confirmation")

        if (
            features.oi_ma_fast is not None
            and features.oi_ma_slow is not None
            and features.oi_ma_fast < features.oi_ma_slow
        ):
            score += 0.06
            reasons.append("fast_oi_below_slow_oi")

        if (
            features.oi_pressure_score is not None
            and abs(features.oi_pressure_score) >= self.thresholds.pressure_score_trend_threshold
        ):
            score += 0.06
            reasons.append("pressure_score_elevated")

        return AnomalyCandidate(
            anomaly_type=OIAnomalyType.OI_COLLAPSE,
            score=min(score, 1.0),
            reasons=reasons,
        )

    def _detect_oi_price_dislocation(self, features: OIFeatures) -> AnomalyCandidate:
        score = 0.0
        reasons: list[str] = []

        if (
            features.price_delta_pct is not None
            and abs(features.price_delta_pct) >= self.thresholds.min_price_change_pct
        ):
            score += 0.20
            reasons.append("meaningful_price_move")

        if (
            features.oi_price_efficiency is not None
            and abs(features.oi_price_efficiency) < 0.20
        ):
            score += 0.26
            reasons.append("oi_not_supporting_price_move")

        if (
            features.oi_delta_pct is not None
            and abs(features.oi_delta_pct) <= self.thresholds.divergence_max_oi_response_pct
        ):
            score += 0.18
            reasons.append("flat_or_weak_oi_response")

        if (
            features.volume_ratio is not None
            and features.volume_ratio < 1.0
        ):
            score += 0.08
            reasons.append("weak_volume_context")

        if (
            features.oi_pressure_score is not None
            and abs(features.oi_pressure_score) < 0.20
        ):
            score += 0.10
            reasons.append("weak_pressure_context")

        if (
            features.aggressive_flow_imbalance is not None
            and abs(features.aggressive_flow_imbalance) < 0.08
        ):
            score += 0.08
            reasons.append("lack_of_aggressive_flow_confirmation")

        if (
            features.oi_zscore is not None
            and abs(features.oi_zscore) < 0.75
        ):
            score += 0.05
            reasons.append("oi_not_statistically_expanding")

        return AnomalyCandidate(
            anomaly_type=OIAnomalyType.OI_PRICE_DISLOCATION,
            score=min(score, 1.0),
            reasons=reasons,
        )

    def _detect_oi_volume_dislocation(self, features: OIFeatures) -> AnomalyCandidate:
        score = 0.0
        reasons: list[str] = []

        if (
            features.volume_ratio is not None
            and features.volume_ratio >= self.thresholds.volume_confirmation_ratio
        ):
            score += 0.24
            reasons.append("elevated_volume")

        if (
            features.oi_change_per_volume is not None
            and abs(features.oi_change_per_volume) < 1e-6
        ):
            score += 0.22
            reasons.append("almost_no_oi_change_per_volume")

        elif (
            features.oi_change_per_volume is not None
            and abs(features.oi_change_per_volume) < 0.001
        ):
            score += 0.16
            reasons.append("weak_oi_change_relative_to_volume")

        if (
            features.oi_delta_pct is not None
            and abs(features.oi_delta_pct) <= self.thresholds.divergence_max_oi_response_pct
        ):
            score += 0.14
            reasons.append("flat_oi_response")

        if (
            features.price_delta_pct is not None
            and abs(features.price_delta_pct) >= self.thresholds.min_price_change_pct
        ):
            score += 0.08
            reasons.append("price_also_moving")

        if (
            features.aggressive_flow_imbalance is not None
            and abs(features.aggressive_flow_imbalance) >= self.thresholds.aggressive_flow_confirmation
        ):
            score += 0.06
            reasons.append("aggressive_flow_present")

        if (
            features.oi_zscore is not None
            and abs(features.oi_zscore) < 0.75
        ):
            score += 0.06
            reasons.append("oi_not_expanding_statistically")

        return AnomalyCandidate(
            anomaly_type=OIAnomalyType.OI_VOLUME_DISLOCATION,
            score=min(score, 1.0),
            reasons=reasons,
        )

    def _detect_liquidation_driven_oi_drop(self, features: OIFeatures) -> AnomalyCandidate:
        score = 0.0
        reasons: list[str] = []

        if features.oi_delta_pct <= -self.thresholds.capitulation_oi_drop_pct:
            score += 0.28
            reasons.append("sharp_oi_drop")

        if (
            features.liquidation_imbalance is not None
            and abs(features.liquidation_imbalance) >= 0.40
        ):
            score += 0.26
            reasons.append("strong_liquidation_imbalance")

        if (
            features.price_delta_pct is not None
            and abs(features.price_delta_pct) >= self.thresholds.capitulation_price_move_pct
        ):
            score += 0.16
            reasons.append("large_price_move")

        if (
            features.volume_ratio is not None
            and features.volume_ratio >= 1.5
        ):
            score += 0.10
            reasons.append("high_volume")

        if (
            features.oi_velocity is not None
            and features.oi_velocity < 0
        ):
            score += 0.06
            reasons.append("negative_oi_velocity")

        if self._is_negative_oi_extreme(features):
            score += 0.06
            reasons.append("negative_oi_zscore_extreme")

        return AnomalyCandidate(
            anomaly_type=OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP,
            score=min(score, 1.0),
            reasons=reasons,
        )

    def _detect_overheated_buildup(self, features: OIFeatures) -> AnomalyCandidate:
        score = 0.0
        reasons: list[str] = []

        if (
            features.oi_zscore is not None
            and features.oi_zscore >= self.thresholds.overheated_zscore_threshold
        ):
            score += 0.26
            reasons.append("overheated_oi_zscore")

        if features.oi_delta_pct >= self.thresholds.squeeze_oi_build_pct:
            score += 0.18
            reasons.append("strong_oi_buildup")

        if (
            features.funding_rate is not None
            and abs(features.funding_rate) >= self.thresholds.squeeze_funding_abs_threshold
        ):
            score += 0.16
            reasons.append("extreme_funding")

        if (
            features.oi_pressure_score is not None
            and abs(features.oi_pressure_score) >= self.thresholds.pressure_score_exhaustion_threshold
        ):
            score += 0.14
            reasons.append("extreme_pressure_score")

        if (
            features.volume_ratio is not None
            and features.volume_ratio >= 1.3
        ):
            score += 0.08
            reasons.append("elevated_volume")

        if (
            features.oi_ma_fast is not None
            and features.oi_ma_slow is not None
            and features.oi_ma_fast > features.oi_ma_slow
        ):
            score += 0.06
            reasons.append("fast_oi_above_slow_oi")

        if (
            features.oi_price_efficiency is not None
            and abs(features.oi_price_efficiency) > 1.5
        ):
            score += 0.06
            reasons.append("oi_outpacing_price")

        return AnomalyCandidate(
            anomaly_type=OIAnomalyType.OVERHEATED_BUILDUP,
            score=min(score, 1.0),
            reasons=reasons,
        )

    def _detect_sudden_deleveraging(self, features: OIFeatures) -> AnomalyCandidate:
        score = 0.0
        reasons: list[str] = []

        if features.oi_delta_pct <= -self.thresholds.deleveraging_oi_drop_pct:
            score += 0.30
            reasons.append("sudden_oi_drop")

        if (
            features.volume_ratio is not None
            and features.volume_ratio >= 1.4
        ):
            score += 0.12
            reasons.append("high_volume")

        if (
            features.price_delta_pct is not None
            and abs(features.price_delta_pct) >= self.thresholds.min_price_change_pct
        ):
            score += 0.10
            reasons.append("price_reaction_present")

        if (
            features.oi_velocity is not None
            and features.oi_velocity < 0
        ):
            score += 0.08
            reasons.append("negative_oi_velocity")

        if (
            features.oi_acceleration is not None
            and features.oi_acceleration < 0
        ):
            score += 0.08
            reasons.append("negative_oi_acceleration")

        if (
            features.liquidation_imbalance is not None
            and abs(features.liquidation_imbalance) >= 0.25
        ):
            score += 0.10
            reasons.append("liquidation_component_present")

        if self._is_negative_oi_extreme(features):
            score += 0.08
            reasons.append("negative_oi_zscore_extreme")

        return AnomalyCandidate(
            anomaly_type=OIAnomalyType.SUDDEN_DELEVERAGING,
            score=min(score, 1.0),
            reasons=reasons,
        )

    def _detect_funding_oi_imbalance(self, features: OIFeatures) -> AnomalyCandidate:
        score = 0.0
        reasons: list[str] = []

        if (
            features.funding_rate is not None
            and abs(features.funding_rate) >= self.thresholds.squeeze_funding_abs_threshold
        ):
            score += 0.28
            reasons.append("extreme_funding")

        if (
            features.oi_delta_pct is not None
            and abs(features.oi_delta_pct) <= self.thresholds.divergence_max_oi_response_pct
        ):
            score += 0.18
            reasons.append("oi_not_following_funding_extreme")

        elif (
            features.oi_delta_pct is not None
            and abs(features.oi_delta_pct) < self.thresholds.min_oi_change_pct
        ):
            score += 0.12
            reasons.append("weak_oi_response_to_funding")

        if (
            features.price_delta_pct is not None
            and abs(features.price_delta_pct) < self.thresholds.min_price_change_pct
        ):
            score += 0.10
            reasons.append("price_compression")

        if (
            features.oi_zscore is not None
            and abs(features.oi_zscore) < 1.0
        ):
            score += 0.08
            reasons.append("oi_not_statistically_extreme")

        if (
            features.oi_pressure_score is not None
            and abs(features.oi_pressure_score) < 0.30
        ):
            score += 0.08
            reasons.append("pressure_not_confirming_crowding")

        if (
            features.volume_ratio is not None
            and features.volume_ratio < 1.1
        ):
            score += 0.06
            reasons.append("limited_volume_confirmation")

        return AnomalyCandidate(
            anomaly_type=OIAnomalyType.FUNDING_OI_IMBALANCE,
            score=min(score, 1.0),
            reasons=reasons,
        )

    def _detect_extreme_crowding(self, features: OIFeatures) -> AnomalyCandidate:
        score = 0.0
        reasons: list[str] = []

        if self._is_extreme_oi_extreme(features):
            score += 0.24
            reasons.append("extreme_oi_zscore")

        if (
            features.funding_rate is not None
            and abs(features.funding_rate) >= self.thresholds.squeeze_funding_abs_threshold
        ):
            score += 0.18
            reasons.append("extreme_funding")

        if (
            features.oi_pressure_score is not None
            and abs(features.oi_pressure_score) >= self.thresholds.pressure_score_exhaustion_threshold
        ):
            score += 0.16
            reasons.append("extreme_pressure")

        if (
            features.oi_delta_pct >= self.thresholds.squeeze_oi_build_pct
            or features.oi_delta_pct <= -self.thresholds.squeeze_oi_build_pct
        ):
            score += 0.12
            reasons.append("strong_oi_shift")

        if (
            features.volume_ratio is not None
            and features.volume_ratio >= 1.2
        ):
            score += 0.08
            reasons.append("elevated_volume")

        if (
            features.aggressive_flow_imbalance is not None
            and abs(features.aggressive_flow_imbalance) >= 0.20
        ):
            score += 0.08
            reasons.append("aggressive_flow_imbalance")

        if (
            features.liquidation_imbalance is not None
            and abs(features.liquidation_imbalance) >= 0.25
        ):
            score += 0.06
            reasons.append("liquidation_imbalance")

        if (
            features.oi_price_efficiency is not None
            and abs(features.oi_price_efficiency) > 1.25
        ):
            score += 0.06
            reasons.append("oi_outpacing_price")

        return AnomalyCandidate(
            anomaly_type=OIAnomalyType.EXTREME_CROWDING,
            score=min(score, 1.0),
            reasons=reasons,
        )

    def describe_anomaly_context(self, features: OIFeatures) -> list[str]:
        """
        Допоміжний метод для логування / debug / audit trail.
        """
        reasons: list[str] = []

        if features.oi_zscore is not None:
            if features.oi_zscore >= self.thresholds.extreme_anomaly_zscore_threshold:
                reasons.append("extreme_positive_oi_zscore")
            elif features.oi_zscore <= -self.thresholds.extreme_anomaly_zscore_threshold:
                reasons.append("extreme_negative_oi_zscore")
            elif abs(features.oi_zscore) >= self.thresholds.anomaly_zscore_threshold:
                reasons.append("anomalous_oi_zscore")

        if features.oi_delta_pct >= self.thresholds.squeeze_oi_build_pct:
            reasons.append("strong_positive_oi_shift")
        elif features.oi_delta_pct <= -self.thresholds.deleveraging_oi_drop_pct:
            reasons.append("strong_negative_oi_shift")

        if features.funding_rate is not None:
            if features.funding_rate >= self.thresholds.squeeze_funding_abs_threshold:
                reasons.append("extreme_positive_funding")
            elif features.funding_rate <= -self.thresholds.squeeze_funding_abs_threshold:
                reasons.append("extreme_negative_funding")

        if features.volume_ratio is not None:
            if features.volume_ratio >= 1.5:
                reasons.append("high_volume")
            elif features.volume_ratio < 1.0:
                reasons.append("weak_volume")

        if features.liquidation_imbalance is not None:
            if features.liquidation_imbalance >= 0.35:
                reasons.append("short_liquidation_pressure")
            elif features.liquidation_imbalance <= -0.35:
                reasons.append("long_liquidation_pressure")

        if features.aggressive_flow_imbalance is not None:
            if features.aggressive_flow_imbalance >= 0.20:
                reasons.append("aggressive_buy_imbalance")
            elif features.aggressive_flow_imbalance <= -0.20:
                reasons.append("aggressive_sell_imbalance")

        if features.oi_pressure_score is not None:
            if features.oi_pressure_score >= self.thresholds.pressure_score_exhaustion_threshold:
                reasons.append("extreme_positive_pressure")
            elif features.oi_pressure_score <= -self.thresholds.pressure_score_exhaustion_threshold:
                reasons.append("extreme_negative_pressure")

        return reasons