from __future__ import annotations
from core.logger import get_logger

import math
from dataclasses import dataclass, field
from typing import Final

from .config import OIAnalyzerConfig
from .enums import OIAnomalyType, OISignalStrength
from .models import OIAnomalyResult, OIFeatures


MIN_ANOMALY_SCORE: Final[float] = 0.35
MAX_ANOMALY_CONFIDENCE: Final[float] = 0.99

VERY_SMALL_OI_PER_VOLUME: Final[float] = 1e-6
WEAK_OI_PER_VOLUME: Final[float] = 0.001

WEAK_VOLUME_RATIO: Final[float] = 1.0
LIMITED_VOLUME_CONFIRMATION_RATIO: Final[float] = 1.1
ELEVATED_VOLUME_RATIO: Final[float] = 1.2
HIGH_VOLUME_RATIO: Final[float] = 1.5

WEAK_PRICE_EFFICIENCY: Final[float] = 0.20
WEAK_OI_PRICE_EFFICIENCY: Final[float] = 0.35
STRONG_OI_PRICE_EFFICIENCY: Final[float] = 1.25
EXTREME_OI_PRICE_EFFICIENCY: Final[float] = 1.5

WEAK_PRESSURE_ABS: Final[float] = 0.30
WEAK_FLOW_ABS: Final[float] = 0.08
AGGRESSIVE_FLOW_EXTREME_ABS: Final[float] = 0.20

LIQUIDATION_COMPONENT_ABS: Final[float] = 0.25
STRONG_LIQUIDATION_IMBALANCE_ABS: Final[float] = 0.40

# Selection tuning.
# Broad OI_SPIKE / OI_COLLAPSE rules are useful, but they should not always
# suppress more specific structural or risk-critical labels.
SPECIFIC_ANOMALY_BONUS: Final[dict[OIAnomalyType, float]] = {
    OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP: 0.08,
    OIAnomalyType.SUDDEN_DELEVERAGING: 0.10,
    OIAnomalyType.EXTREME_CROWDING: 0.07,
    OIAnomalyType.OVERHEATED_BUILDUP: 0.05,
    OIAnomalyType.FUNDING_OI_IMBALANCE: 0.08,
    OIAnomalyType.OI_PRICE_DISLOCATION: 0.05,
    OIAnomalyType.OI_VOLUME_DISLOCATION: 0.05,
    OIAnomalyType.OI_COLLAPSE: 0.03,
    OIAnomalyType.OI_SPIKE: 0.0,
    OIAnomalyType.NONE: 0.0,
}

ANOMALY_PRIORITY: Final[dict[OIAnomalyType, int]] = {
    # Найбільш risk-critical аномалії.
    OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP: 100,
    OIAnomalyType.SUDDEN_DELEVERAGING: 95,
    OIAnomalyType.OI_COLLAPSE: 90,

    # Crowding / overheated стани.
    OIAnomalyType.EXTREME_CROWDING: 85,
    OIAnomalyType.OVERHEATED_BUILDUP: 80,
    OIAnomalyType.FUNDING_OI_IMBALANCE: 75,

    # Структурні dislocation-и.
    OIAnomalyType.OI_PRICE_DISLOCATION: 70,
    OIAnomalyType.OI_VOLUME_DISLOCATION: 65,

    # Простий spike менш специфічний, тому нижче.
    OIAnomalyType.OI_SPIKE: 60,

    OIAnomalyType.NONE: 0,
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
class AnomalyCandidate:
    """
    Internal score container for rule-based OI anomaly classification.

    Pure value object:
    - no EventBus;
    - no Scheduler;
    - no logger;
    - no side effects.
    """

    anomaly_type: OIAnomalyType
    score: float
    reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__post_init__", _analytics_args)
        except Exception:
            pass
        self.score = _clamp(self.score)
        self.reasons = list(dict.fromkeys(self.reasons or []))

    @property
    def priority(self) -> int:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "priority", _analytics_args)
        except Exception:
            pass
        return ANOMALY_PRIORITY.get(self.anomaly_type, 0)

    @property
    def reasons_count(self) -> int:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "reasons_count", _analytics_args)
        except Exception:
            pass
        return len(self.reasons)


class OIAnomalyDetector:
    """
    Rule-based detector for futures Open Interest anomalies.

    This is a pure domain service:
    - no EventBus;
    - no Scheduler;
    - no logger;
    - no IO;
    - no mutable runtime state.

    It receives OIFeatures and returns OIAnomalyResult.
    """

    def __init__(self, config: OIAnalyzerConfig) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__init__", _analytics_args)
        except Exception:
            pass
        self.config = config
        self.thresholds = config.thresholds

    def detect(self, features: OIFeatures) -> OIAnomalyResult | None:
        """
        Detect the strongest Open Interest anomaly for a single features snapshot.

        Returns:
            - None only when candidate construction is impossible.
            - OIAnomalyResult(detected=False, ...) when no anomaly passes threshold.
            - OIAnomalyResult(detected=True, ...) for the strongest anomaly.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "detect", _analytics_args)
        except Exception:
            pass
        candidates = self._build_candidates(features)
        best = self._select_best_candidate(candidates)

        if best is None:
            return None

        if best.score < MIN_ANOMALY_SCORE:
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

    def describe_anomaly_context(self, features: OIFeatures) -> list[str]:
        """
        Helper for debug/audit trail by OIAnalyzer, dashboard, or tests.

        This method does not log by itself.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "describe_anomaly_context", _analytics_args)
        except Exception:
            pass
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
            if features.volume_ratio >= HIGH_VOLUME_RATIO:
                reasons.append("high_volume")
            elif features.volume_ratio < WEAK_VOLUME_RATIO:
                reasons.append("weak_volume")

        if features.liquidation_imbalance is not None:
            if features.liquidation_imbalance >= 0.35:
                reasons.append("short_liquidation_pressure")
            elif features.liquidation_imbalance <= -0.35:
                reasons.append("long_liquidation_pressure")

        if features.aggressive_flow_imbalance is not None:
            if features.aggressive_flow_imbalance >= AGGRESSIVE_FLOW_EXTREME_ABS:
                reasons.append("aggressive_buy_imbalance")
            elif features.aggressive_flow_imbalance <= -AGGRESSIVE_FLOW_EXTREME_ABS:
                reasons.append("aggressive_sell_imbalance")

        if features.oi_pressure_score is not None:
            if features.oi_pressure_score >= self.thresholds.pressure_score_exhaustion_threshold:
                reasons.append("extreme_positive_pressure")
            elif features.oi_pressure_score <= -self.thresholds.pressure_score_exhaustion_threshold:
                reasons.append("extreme_negative_pressure")

        return list(dict.fromkeys(reasons))

    def _build_candidates(self, features: OIFeatures) -> list[AnomalyCandidate]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_candidates", _analytics_args)
        except Exception:
            pass
        return [
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

    def _select_best_candidate(
        self,
        candidates: list[AnomalyCandidate],
    ) -> AnomalyCandidate | None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_select_best_candidate", _analytics_args)
        except Exception:
            pass
        if not candidates:
            return None

        def effective_score(candidate: AnomalyCandidate) -> float:
            if candidate.score < MIN_ANOMALY_SCORE:
                return candidate.score

            return _clamp(
                candidate.score
                + SPECIFIC_ANOMALY_BONUS.get(candidate.anomaly_type, 0.0)
            )

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
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_score_to_confidence", _analytics_args)
        except Exception:
            pass
        score = _clamp(score)

        if score <= 0:
            return 0.0

        if score >= 1.0:
            return MAX_ANOMALY_CONFIDENCE

        return _clamp(0.15 + score * 0.8)

    @staticmethod
    def _score_to_strength(score: float) -> OISignalStrength:
        try:
            _analytics_class_name = "OIAnomalyDetector"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_score_to_strength", _analytics_args)
        except Exception:
            pass
        score = _clamp(score)

        if score >= 0.90:
            return OISignalStrength.EXTREME

        if score >= 0.70:
            return OISignalStrength.HIGH

        if score >= 0.50:
            return OISignalStrength.MEDIUM

        return OISignalStrength.LOW

    # ------------------------------------------------------------------
    # Shared predicates
    # ------------------------------------------------------------------

    def _is_positive_oi_extreme(self, features: OIFeatures) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_is_positive_oi_extreme", _analytics_args)
        except Exception:
            pass
        return (
            features.oi_zscore is not None
            and features.oi_zscore >= self.thresholds.anomaly_zscore_threshold
        )

    def _is_negative_oi_extreme(self, features: OIFeatures) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_is_negative_oi_extreme", _analytics_args)
        except Exception:
            pass
        return (
            features.oi_zscore is not None
            and features.oi_zscore <= -self.thresholds.anomaly_zscore_threshold
        )

    def _is_extreme_oi_extreme(self, features: OIFeatures) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_is_extreme_oi_extreme", _analytics_args)
        except Exception:
            pass
        return (
            features.oi_zscore is not None
            and abs(features.oi_zscore) >= self.thresholds.extreme_anomaly_zscore_threshold
        )

    @staticmethod
    def _fast_oi_above_slow_oi(features: OIFeatures) -> bool:
        try:
            _analytics_class_name = "OIAnomalyDetector"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_fast_oi_above_slow_oi", _analytics_args)
        except Exception:
            pass
        return (
            features.oi_ma_fast is not None
            and features.oi_ma_slow is not None
            and features.oi_ma_fast > features.oi_ma_slow
        )

    @staticmethod
    def _fast_oi_below_slow_oi(features: OIFeatures) -> bool:
        try:
            _analytics_class_name = "OIAnomalyDetector"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_fast_oi_below_slow_oi", _analytics_args)
        except Exception:
            pass
        return (
            features.oi_ma_fast is not None
            and features.oi_ma_slow is not None
            and features.oi_ma_fast < features.oi_ma_slow
        )

    def _has_volume_confirmation(self, features: OIFeatures) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_has_volume_confirmation", _analytics_args)
        except Exception:
            pass
        return (
            features.volume_ratio is not None
            and features.volume_ratio >= self.thresholds.volume_confirmation_ratio
        )

    def _has_elevated_volume(self, features: OIFeatures) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_has_elevated_volume", _analytics_args)
        except Exception:
            pass
        return (
            features.volume_ratio is not None
            and features.volume_ratio >= ELEVATED_VOLUME_RATIO
        )

    def _has_high_volume(self, features: OIFeatures) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_has_high_volume", _analytics_args)
        except Exception:
            pass
        return (
            features.volume_ratio is not None
            and features.volume_ratio >= HIGH_VOLUME_RATIO
        )

    def _has_weak_volume(self, features: OIFeatures) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_has_weak_volume", _analytics_args)
        except Exception:
            pass
        return (
            features.volume_ratio is not None
            and features.volume_ratio < WEAK_VOLUME_RATIO
        )

    def _has_elevated_pressure(self, features: OIFeatures) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_has_elevated_pressure", _analytics_args)
        except Exception:
            pass
        return (
            features.oi_pressure_score is not None
            and abs(features.oi_pressure_score)
            >= self.thresholds.pressure_score_trend_threshold
        )

    def _has_extreme_pressure(self, features: OIFeatures) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_has_extreme_pressure", _analytics_args)
        except Exception:
            pass
        return (
            features.oi_pressure_score is not None
            and abs(features.oi_pressure_score)
            >= self.thresholds.pressure_score_exhaustion_threshold
        )

    def _has_extreme_funding(self, features: OIFeatures) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_has_extreme_funding", _analytics_args)
        except Exception:
            pass
        return (
            features.funding_rate is not None
            and abs(features.funding_rate)
            >= self.thresholds.squeeze_funding_abs_threshold
        )

    def _has_positive_extreme_funding(self, features: OIFeatures) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_has_positive_extreme_funding", _analytics_args)
        except Exception:
            pass
        return (
            features.funding_rate is not None
            and features.funding_rate >= self.thresholds.squeeze_funding_abs_threshold
        )

    def _has_negative_extreme_funding(self, features: OIFeatures) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_has_negative_extreme_funding", _analytics_args)
        except Exception:
            pass
        return (
            features.funding_rate is not None
            and features.funding_rate <= -self.thresholds.squeeze_funding_abs_threshold
        )

    @staticmethod
    def _has_large_liquidation_component(features: OIFeatures) -> bool:
        try:
            _analytics_class_name = "OIAnomalyDetector"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_has_large_liquidation_component", _analytics_args)
        except Exception:
            pass
        return (
            features.liquidation_imbalance is not None
            and abs(features.liquidation_imbalance) >= LIQUIDATION_COMPONENT_ABS
        )

    @staticmethod
    def _has_strong_liquidation_imbalance(features: OIFeatures) -> bool:
        try:
            _analytics_class_name = "OIAnomalyDetector"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_has_strong_liquidation_imbalance", _analytics_args)
        except Exception:
            pass
        return (
            features.liquidation_imbalance is not None
            and abs(features.liquidation_imbalance) >= STRONG_LIQUIDATION_IMBALANCE_ABS
        )

    @staticmethod
    def _has_extreme_aggressive_flow(features: OIFeatures) -> bool:
        try:
            _analytics_class_name = "OIAnomalyDetector"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_has_extreme_aggressive_flow", _analytics_args)
        except Exception:
            pass
        return (
            features.aggressive_flow_imbalance is not None
            and abs(features.aggressive_flow_imbalance) >= AGGRESSIVE_FLOW_EXTREME_ABS
        )

    def _is_positive_oi_shift(self, features: OIFeatures) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_is_positive_oi_shift", _analytics_args)
        except Exception:
            pass
        return features.oi_delta_pct >= self.thresholds.squeeze_oi_build_pct

    def _is_negative_oi_drop(self, features: OIFeatures) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_is_negative_oi_drop", _analytics_args)
        except Exception:
            pass
        return features.oi_delta_pct <= -self.thresholds.deleveraging_oi_drop_pct

    # ------------------------------------------------------------------
    # Anomaly rules
    # ------------------------------------------------------------------

    def _detect_oi_spike(self, features: OIFeatures) -> AnomalyCandidate:
        """
        Broad positive OI spike.

        This rule is intentionally broad, but its score is dampened when
        the snapshot better matches structural volume dislocation, extreme
        crowding, or funding imbalance.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_detect_oi_spike", _analytics_args)
        except Exception:
            pass
        score = 0.0
        reasons: list[str] = []

        if self._is_positive_oi_extreme(features):
            score += 0.30
            reasons.append("positive_oi_zscore_extreme")

        if features.oi_delta_pct >= self.thresholds.min_oi_change_pct * 2.0:
            score += 0.22
            reasons.append("large_positive_oi_change")

        if _is_positive(features.oi_velocity):
            score += 0.07
            reasons.append("positive_oi_velocity")

        if _is_positive(features.oi_acceleration):
            score += 0.07
            reasons.append("positive_oi_acceleration")

        if self._has_volume_confirmation(features):
            score += 0.07
            reasons.append("volume_confirmation")

        if self._fast_oi_above_slow_oi(features):
            score += 0.05
            reasons.append("fast_oi_above_slow_oi")

        if self._has_elevated_pressure(features):
            score += 0.05
            reasons.append("pressure_score_elevated")

        if (
            self._has_weak_volume(features)
            and features.oi_price_efficiency is not None
            and abs(features.oi_price_efficiency) >= STRONG_OI_PRICE_EFFICIENCY
        ):
            score -= 0.10
            reasons.append("structural_volume_dislocation_penalty")

        if self._has_extreme_funding(features):
            score -= 0.06
            reasons.append("funding_specific_context_penalty")

        if self._has_extreme_pressure(features) and self._has_large_liquidation_component(features):
            score -= 0.08
            reasons.append("crowding_specific_context_penalty")

        return AnomalyCandidate(
            anomaly_type=OIAnomalyType.OI_SPIKE,
            score=score,
            reasons=reasons,
        )

    def _detect_oi_collapse(self, features: OIFeatures) -> AnomalyCandidate:
        """
        Broad negative OI collapse.

        Risk-critical liquidation/deleveraging rules receive selector bonuses,
        so this rule remains a broad fallback.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_detect_oi_collapse", _analytics_args)
        except Exception:
            pass
        score = 0.0
        reasons: list[str] = []

        if self._is_negative_oi_extreme(features):
            score += 0.32
            reasons.append("negative_oi_zscore_extreme")

        if features.oi_delta_pct <= -(self.thresholds.min_oi_change_pct * 2.0):
            score += 0.22
            reasons.append("large_negative_oi_change")

        if _is_negative(features.oi_velocity):
            score += 0.07
            reasons.append("negative_oi_velocity")

        if _is_negative(features.oi_acceleration):
            score += 0.07
            reasons.append("negative_oi_acceleration")

        if self._has_volume_confirmation(features):
            score += 0.07
            reasons.append("volume_confirmation")

        if self._fast_oi_below_slow_oi(features):
            score += 0.05
            reasons.append("fast_oi_below_slow_oi")

        if self._has_elevated_pressure(features):
            score += 0.05
            reasons.append("pressure_score_elevated")

        if self._has_strong_liquidation_imbalance(features):
            score -= 0.06
            reasons.append("liquidation_specific_context_penalty")

        if features.oi_delta_pct <= -self.thresholds.deleveraging_oi_drop_pct:
            score -= 0.05
            reasons.append("deleveraging_specific_context_penalty")

        return AnomalyCandidate(
            anomaly_type=OIAnomalyType.OI_COLLAPSE,
            score=score,
            reasons=reasons,
        )

    def _detect_oi_price_dislocation(
        self,
        features: OIFeatures,
    ) -> AnomalyCandidate:
        """
        Price moves meaningfully, but OI does not confirm the move.

        Critical fix:
        Without a meaningful price move this rule must not trigger. Otherwise
        low-context noise can be misclassified as OI_PRICE_DISLOCATION.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_detect_oi_price_dislocation", _analytics_args)
        except Exception:
            pass
        if (
            features.price_delta_pct is None
            or abs(features.price_delta_pct) < self.thresholds.min_price_change_pct
        ):
            return AnomalyCandidate(
                anomaly_type=OIAnomalyType.OI_PRICE_DISLOCATION,
                score=0.0,
                reasons=["price_move_not_meaningful"],
            )

        score = 0.0
        reasons: list[str] = []

        score += 0.20
        reasons.append("meaningful_price_move")

        if (
            features.oi_price_efficiency is not None
            and abs(features.oi_price_efficiency) < WEAK_PRICE_EFFICIENCY
        ):
            score += 0.26
            reasons.append("oi_not_supporting_price_move")

        if (
            abs(features.oi_delta_pct)
            <= self.thresholds.divergence_max_oi_response_pct
        ):
            score += 0.18
            reasons.append("flat_or_weak_oi_response")

        if self._has_weak_volume(features):
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
            and abs(features.aggressive_flow_imbalance) < WEAK_FLOW_ABS
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
            score=score,
            reasons=reasons,
        )

    def _detect_oi_volume_dislocation(
        self,
        features: OIFeatures,
    ) -> AnomalyCandidate:
        """
        Large OI movement with weak volume or abnormal OI-per-volume behavior.

        This should beat broad OI_SPIKE when the setup is structural rather
        than simply momentum-driven.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_detect_oi_volume_dislocation", _analytics_args)
        except Exception:
            pass
        score = 0.0
        reasons: list[str] = []

        weak_volume = self._has_weak_volume(features)
        strong_oi_shift = (
            abs(features.oi_delta_pct) >= self.thresholds.squeeze_oi_build_pct
        )

        if weak_volume and strong_oi_shift:
            score += 0.34
            reasons.append("large_oi_shift_on_weak_volume")

        if (
            features.oi_change_per_volume is not None
            and abs(features.oi_change_per_volume) <= WEAK_OI_PER_VOLUME
        ):
            score += 0.18
            reasons.append("abnormal_oi_per_volume")

        if (
            features.oi_price_efficiency is not None
            and abs(features.oi_price_efficiency) >= STRONG_OI_PRICE_EFFICIENCY
        ):
            score += 0.20
            reasons.append("high_oi_price_efficiency")

        if (
            features.price_delta_pct is not None
            and abs(features.price_delta_pct) < self.thresholds.min_price_change_pct
        ):
            score += 0.12
            reasons.append("price_compression")

        if self._is_positive_oi_extreme(features) or self._is_negative_oi_extreme(features):
            score += 0.06
            reasons.append("oi_zscore_extreme")

        if features.volume_ratio is None:
            score -= 0.10
            reasons.append("missing_volume_ratio_penalty")

        return AnomalyCandidate(
            anomaly_type=OIAnomalyType.OI_VOLUME_DISLOCATION,
            score=score,
            reasons=reasons,
        )

    def _detect_liquidation_driven_oi_drop(
        self,
        features: OIFeatures,
    ) -> AnomalyCandidate:
        """
        OI drop driven by liquidation pressure.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_detect_liquidation_driven_oi_drop", _analytics_args)
        except Exception:
            pass
        score = 0.0
        reasons: list[str] = []

        if features.oi_delta_pct <= -self.thresholds.capitulation_oi_drop_pct:
            score += 0.28
            reasons.append("sharp_oi_drop")

        if self._has_strong_liquidation_imbalance(features):
            score += 0.28
            reasons.append("strong_liquidation_imbalance")

        if (
            _safe_abs(features.price_delta_pct)
            >= self.thresholds.capitulation_price_move_pct
        ):
            score += 0.16
            reasons.append("large_price_move")

        if self._has_high_volume(features):
            score += 0.10
            reasons.append("high_volume")

        if _is_negative(features.oi_velocity):
            score += 0.06
            reasons.append("negative_oi_velocity")

        if self._is_negative_oi_extreme(features):
            score += 0.06
            reasons.append("negative_oi_zscore_extreme")

        return AnomalyCandidate(
            anomaly_type=OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP,
            score=score,
            reasons=reasons,
        )

    def _detect_overheated_buildup(
        self,
        features: OIFeatures,
    ) -> AnomalyCandidate:
        """
        Strong OI buildup that is overheated but not necessarily full crowding.

        If funding-only imbalance is dominant, this rule is penalized so
        FUNDING_OI_IMBALANCE can win. If funding + pressure + liquidation are
        all extreme, this rule is penalized so EXTREME_CROWDING can win.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_detect_overheated_buildup", _analytics_args)
        except Exception:
            pass
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

        if self._has_extreme_funding(features):
            score += 0.13
            reasons.append("extreme_funding")

        if self._has_extreme_pressure(features):
            score += 0.12
            reasons.append("extreme_pressure_score")

        if features.volume_ratio is not None and features.volume_ratio >= 1.3:
            score += 0.08
            reasons.append("elevated_volume")

        if self._fast_oi_above_slow_oi(features):
            score += 0.06
            reasons.append("fast_oi_above_slow_oi")

        if (
            features.oi_price_efficiency is not None
            and abs(features.oi_price_efficiency) > EXTREME_OI_PRICE_EFFICIENCY
        ):
            score += 0.06
            reasons.append("oi_outpacing_price")

        if (
            self._has_extreme_funding(features)
            and not self._has_extreme_pressure(features)
        ):
            score -= 0.16
            reasons.append("funding_imbalance_context_penalty")

        if (
            self._has_extreme_funding(features)
            and self._has_extreme_pressure(features)
            and self._has_large_liquidation_component(features)
        ):
            score -= 0.12
            reasons.append("extreme_crowding_context_penalty")

        return AnomalyCandidate(
            anomaly_type=OIAnomalyType.OVERHEATED_BUILDUP,
            score=score,
            reasons=reasons,
        )

    def _detect_sudden_deleveraging(
        self,
        features: OIFeatures,
    ) -> AnomalyCandidate:
        """
        Sudden market-wide deleveraging signature.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_detect_sudden_deleveraging", _analytics_args)
        except Exception:
            pass
        score = 0.0
        reasons: list[str] = []

        if features.oi_delta_pct <= -self.thresholds.deleveraging_oi_drop_pct:
            score += 0.34
            reasons.append("deleveraging_sudden_oi_drop")

        if features.volume_ratio is not None and features.volume_ratio >= 1.4:
            score += 0.12
            reasons.append("deleveraging_high_volume")

        if _safe_abs(features.price_delta_pct) >= self.thresholds.min_price_change_pct:
            score += 0.10
            reasons.append("deleveraging_price_reaction_present")

        if _is_negative(features.oi_velocity):
            score += 0.08
            reasons.append("deleveraging_negative_oi_velocity")

        if _is_negative(features.oi_acceleration):
            score += 0.08
            reasons.append("deleveraging_negative_oi_acceleration")

        if self._has_large_liquidation_component(features):
            score += 0.11
            reasons.append("deleveraging_liquidation_component_present")

        if self._is_negative_oi_extreme(features):
            score += 0.09
            reasons.append("deleveraging_negative_oi_zscore_extreme")

        if self._has_negative_extreme_funding(features):
            score += 0.10
            reasons.append("deleveraging_extreme_negative_funding")

        if (
            features.oi_pressure_score is not None
            and features.oi_pressure_score <= -self.thresholds.pressure_score_exhaustion_threshold
        ):
            score += 0.10
            reasons.append("deleveraging_extreme_negative_pressure")

        return AnomalyCandidate(
            anomaly_type=OIAnomalyType.SUDDEN_DELEVERAGING,
            score=score,
            reasons=reasons,
        )

    def _detect_funding_oi_imbalance(
        self,
        features: OIFeatures,
    ) -> AnomalyCandidate:
        """
        Funding/OI imbalance.

        Critical fix:
        This anomaly is hard-gated by extreme funding. Without extreme funding,
        compressed price + weak OI response must not produce a funding anomaly.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_detect_funding_oi_imbalance", _analytics_args)
        except Exception:
            pass
        score = 0.0
        reasons: list[str] = []

        if not self._has_extreme_funding(features):
            return AnomalyCandidate(
                anomaly_type=OIAnomalyType.FUNDING_OI_IMBALANCE,
                score=0.0,
                reasons=["funding_not_extreme"],
            )

        score += 0.36
        reasons.append("funding_extreme")

        if (
            features.funding_rate is not None
            and features.funding_rate > 0
            and features.oi_delta_pct < self.thresholds.min_oi_change_pct
        ):
            score += 0.20
            reasons.append("funding_positive_without_oi_expansion")

        if (
            features.funding_rate is not None
            and features.funding_rate < 0
            and features.oi_delta_pct > -self.thresholds.min_oi_change_pct
        ):
            score += 0.20
            reasons.append("funding_negative_without_oi_contraction")

        if (
            features.price_delta_pct is not None
            and abs(features.price_delta_pct) < self.thresholds.min_price_change_pct
        ):
            score += 0.12
            reasons.append("funding_price_compression")

        if (
            features.oi_zscore is not None
            and abs(features.oi_zscore) < self.thresholds.anomaly_zscore_threshold
        ):
            score += 0.08
            reasons.append("funding_oi_not_statistically_extreme")

        if (
            features.oi_pressure_score is not None
            and abs(features.oi_pressure_score) < self.thresholds.pressure_score_exhaustion_threshold
        ):
            score += 0.10
            reasons.append("funding_without_extreme_pressure")

        if (
            features.oi_pressure_score is not None
            and abs(features.oi_pressure_score) < self.thresholds.pressure_score_trend_threshold
        ):
            score += 0.08
            reasons.append("funding_pressure_not_confirming_crowding")

        if (
            features.volume_ratio is not None
            and features.volume_ratio < LIMITED_VOLUME_CONFIRMATION_RATIO
        ):
            score += 0.07
            reasons.append("funding_limited_volume_confirmation")

        return AnomalyCandidate(
            anomaly_type=OIAnomalyType.FUNDING_OI_IMBALANCE,
            score=score,
            reasons=reasons,
        )

    def _detect_extreme_crowding(
        self,
        features: OIFeatures,
    ) -> AnomalyCandidate:
        """
        Extreme crowding/crowded positioning context.

        This should win over OVERHEATED_BUILDUP when funding, pressure and
        liquidation/flow context are all extreme. It should not steal obvious
        negative deleveraging events.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_detect_extreme_crowding", _analytics_args)
        except Exception:
            pass
        score = 0.0
        reasons: list[str] = []

        if self._is_extreme_oi_extreme(features):
            score += 0.24
            reasons.append("extreme_crowding_oi_zscore")

        if self._has_extreme_funding(features):
            score += 0.18
            reasons.append("extreme_crowding_funding")

        if self._has_extreme_pressure(features):
            score += 0.17
            reasons.append("extreme_crowding_pressure")

        if self._has_extreme_funding(features) and self._has_extreme_pressure(features):
            score += 0.10
            reasons.append("extreme_crowding_funding_pressure_combo")

        if (
            features.oi_delta_pct >= self.thresholds.squeeze_oi_build_pct
            or features.oi_delta_pct <= -self.thresholds.squeeze_oi_build_pct
        ):
            score += 0.12
            reasons.append("extreme_crowding_strong_oi_shift")

        if self._has_elevated_volume(features):
            score += 0.08
            reasons.append("extreme_crowding_elevated_volume")

        if self._has_extreme_aggressive_flow(features):
            score += 0.08
            reasons.append("extreme_crowding_aggressive_flow_imbalance")

        if self._has_large_liquidation_component(features):
            score += 0.07
            reasons.append("extreme_crowding_liquidation_imbalance")

        if (
            features.oi_price_efficiency is not None
            and abs(features.oi_price_efficiency) > STRONG_OI_PRICE_EFFICIENCY
        ):
            score += 0.06
            reasons.append("extreme_crowding_oi_outpacing_price")

        if features.oi_delta_pct <= -self.thresholds.deleveraging_oi_drop_pct:
            score -= 0.25
            reasons.append("deleveraging_context_penalty")

        return AnomalyCandidate(
            anomaly_type=OIAnomalyType.EXTREME_CROWDING,
            score=score,
            reasons=reasons,
        )