from __future__ import annotations
from core.logger import get_logger

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .config import BaseSpreadConfig
from .enums import SpreadRegime
from .models import RollingStats, SpreadSnapshot


DECIMAL_ZERO = Decimal("0")
DEFAULT_ELEVATED_THRESHOLD = Decimal("1.5")
DEFAULT_COMPRESSED_THRESHOLD = Decimal("0.5")
DEFAULT_DISLOCATION_MULTIPLIER = Decimal("1.5")


# ============================================================
# Result models
# ============================================================

@dataclass(slots=True)
class RegimeDetectionResult:
    """
    Результат класифікації spread regime.

    Не містить runtime-залежностей і може безпечно використовуватись:
    - analyzer-ами;
    - SpreadSignalEngine;
    - storage/dashboard;
    - EventBus payload metadata.
    """

    regime: SpreadRegime
    zscore: Decimal | None = None
    abs_zscore: Decimal | None = None

    is_compressed: bool = False
    is_elevated: bool = False
    is_extreme: bool = False
    is_dislocated: bool = False

    threshold_compressed: Decimal | None = None
    threshold_elevated: Decimal | None = None
    threshold_extreme: Decimal | None = None
    threshold_dislocated: Decimal | None = None

    reason: str | None = None

    @property
    def is_normal(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_normal", _analytics_args)
        except Exception:
            pass
        return self.regime == SpreadRegime.NORMAL

    @property
    def is_abnormal(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_abnormal", _analytics_args)
        except Exception:
            pass
        return self.regime in {
            SpreadRegime.ELEVATED,
            SpreadRegime.EXTREME,
            SpreadRegime.DISLOCATED,
        }

    @property
    def is_high_risk(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_high_risk", _analytics_args)
        except Exception:
            pass
        return self.regime in {
            SpreadRegime.EXTREME,
            SpreadRegime.DISLOCATED,
        }

    def to_payload(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_payload", _analytics_args)
        except Exception:
            pass
        return {
            "regime": self.regime.value,
            "zscore": _decimal_to_payload(self.zscore),
            "abs_zscore": _decimal_to_payload(self.abs_zscore),
            "is_normal": self.is_normal,
            "is_abnormal": self.is_abnormal,
            "is_high_risk": self.is_high_risk,
            "is_compressed": self.is_compressed,
            "is_elevated": self.is_elevated,
            "is_extreme": self.is_extreme,
            "is_dislocated": self.is_dislocated,
            "threshold_compressed": _decimal_to_payload(self.threshold_compressed),
            "threshold_elevated": _decimal_to_payload(self.threshold_elevated),
            "threshold_extreme": _decimal_to_payload(self.threshold_extreme),
            "threshold_dislocated": _decimal_to_payload(self.threshold_dislocated),
            "reason": self.reason,
        }


@dataclass(slots=True)
class RegimeShiftResult:
    """
    Результат порівняння попереднього та поточного regime.
    """

    changed: bool
    previous_regime: SpreadRegime | None = None
    current_regime: SpreadRegime | None = None

    previous_zscore: Decimal | None = None
    current_zscore: Decimal | None = None
    zscore_delta: Decimal | None = None

    previous_rank: int | None = None
    current_rank: int | None = None
    rank_delta: int | None = None

    reason: str | None = None

    @property
    def is_shift_up(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_shift_up", _analytics_args)
        except Exception:
            pass
        if self.previous_rank is None or self.current_rank is None:
            return False
        return self.current_rank > self.previous_rank

    @property
    def is_shift_down(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_shift_down", _analytics_args)
        except Exception:
            pass
        if self.previous_rank is None or self.current_rank is None:
            return False
        return self.current_rank < self.previous_rank

    @property
    def is_high_risk_shift(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_high_risk_shift", _analytics_args)
        except Exception:
            pass
        return (
            self.changed
            and self.current_regime
            in {
                SpreadRegime.EXTREME,
                SpreadRegime.DISLOCATED,
            }
        )

    def to_payload(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_payload", _analytics_args)
        except Exception:
            pass
        return {
            "changed": self.changed,
            "previous_regime": self.previous_regime.value if self.previous_regime else None,
            "current_regime": self.current_regime.value if self.current_regime else None,
            "previous_zscore": _decimal_to_payload(self.previous_zscore),
            "current_zscore": _decimal_to_payload(self.current_zscore),
            "zscore_delta": _decimal_to_payload(self.zscore_delta),
            "previous_rank": self.previous_rank,
            "current_rank": self.current_rank,
            "rank_delta": self.rank_delta,
            "is_shift_up": self.is_shift_up,
            "is_shift_down": self.is_shift_down,
            "is_high_risk_shift": self.is_high_risk_shift,
            "reason": self.reason,
        }


# ============================================================
# Detector
# ============================================================

class SpreadRegimeDetector:
    """
    Чистий доменний сервіс для класифікації spread regime.

    Відповідальність:
    - визначити regime за z-score / RollingStats;
    - визначити compressed/elevated/extreme/dislocated стани;
    - порівняти два snapshots і знайти regime shift;
    - повернути стабільні result-моделі.

    Не відповідає за:
    - EventBus publish;
    - Scheduler jobs;
    - logging;
    - storage;
    - lifecycle analyzer-а.
    """

    def __init__(
        self,
        config: BaseSpreadConfig,
        *,
        elevated_threshold: Decimal = DEFAULT_ELEVATED_THRESHOLD,
        compressed_threshold: Decimal = DEFAULT_COMPRESSED_THRESHOLD,
        extreme_threshold: Decimal | None = None,
        dislocated_threshold: Decimal | None = None,
    ) -> None:
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
        self._config = config

        self._compressed_threshold = _validate_positive_decimal(
            "compressed_threshold",
            compressed_threshold,
        )
        self._elevated_threshold = _validate_positive_decimal(
            "elevated_threshold",
            elevated_threshold,
        )
        self._extreme_threshold = _validate_positive_decimal(
            "extreme_threshold",
            extreme_threshold or config.anomaly_zscore_threshold,
        )
        self._dislocated_threshold = _validate_positive_decimal(
            "dislocated_threshold",
            dislocated_threshold
            or (self._extreme_threshold * DEFAULT_DISLOCATION_MULTIPLIER),
        )

        self._validate_threshold_order()

    # ------------------------------------------------------------------
    # Public detection API
    # ------------------------------------------------------------------

    def detect_from_snapshot(
        self,
        snapshot: SpreadSnapshot | None,
    ) -> RegimeDetectionResult:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "detect_from_snapshot", _analytics_args)
        except Exception:
            pass
        if snapshot is None:
            return self._missing_result(reason="missing_snapshot")

        return self.detect_from_stats(snapshot.stats)

    def detect_from_stats(
        self,
        stats: RollingStats | None,
    ) -> RegimeDetectionResult:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "detect_from_stats", _analytics_args)
        except Exception:
            pass
        if stats is None:
            return self._missing_result(reason="missing_stats")

        return self.detect_from_zscore(stats.zscore)

    def detect_from_zscore(
        self,
        zscore: Decimal | None,
    ) -> RegimeDetectionResult:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "detect_from_zscore", _analytics_args)
        except Exception:
            pass
        if zscore is None:
            return self._missing_result(reason="missing_zscore")

        abs_zscore = abs(zscore)
        regime = self._resolve_regime(abs_zscore)

        is_dislocated = abs_zscore >= self._dislocated_threshold
        is_extreme = abs_zscore >= self._extreme_threshold
        is_elevated = abs_zscore >= self._elevated_threshold
        is_compressed = abs_zscore <= self._compressed_threshold

        return RegimeDetectionResult(
            regime=regime,
            zscore=zscore,
            abs_zscore=abs_zscore,
            is_compressed=is_compressed,
            is_elevated=is_elevated,
            is_extreme=is_extreme,
            is_dislocated=is_dislocated,
            threshold_compressed=self._compressed_threshold,
            threshold_elevated=self._elevated_threshold,
            threshold_extreme=self._extreme_threshold,
            threshold_dislocated=self._dislocated_threshold,
            reason=self._build_detection_reason(
                regime=regime,
                zscore=zscore,
                abs_zscore=abs_zscore,
            ),
        )

    def detect_regime(
        self,
        zscore: Decimal | None,
    ) -> SpreadRegime:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "detect_regime", _analytics_args)
        except Exception:
            pass
        return self.detect_from_zscore(zscore).regime

    # ------------------------------------------------------------------
    # Shift detection API
    # ------------------------------------------------------------------

    def detect_shift(
        self,
        previous_snapshot: SpreadSnapshot | None,
        current_snapshot: SpreadSnapshot | None,
    ) -> RegimeShiftResult:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "detect_shift", _analytics_args)
        except Exception:
            pass
        previous_regime = previous_snapshot.regime if previous_snapshot else None
        current_regime = current_snapshot.regime if current_snapshot else None

        previous_zscore = self._extract_zscore(previous_snapshot)
        current_zscore = self._extract_zscore(current_snapshot)

        return self.detect_shift_from_values(
            previous_regime=previous_regime,
            current_regime=current_regime,
            previous_zscore=previous_zscore,
            current_zscore=current_zscore,
        )

    def detect_shift_from_stats(
        self,
        previous_stats: RollingStats | None,
        current_stats: RollingStats | None,
    ) -> RegimeShiftResult:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "detect_shift_from_stats", _analytics_args)
        except Exception:
            pass
        previous_detection = self.detect_from_stats(previous_stats)
        current_detection = self.detect_from_stats(current_stats)

        return self.detect_shift_from_values(
            previous_regime=previous_detection.regime,
            current_regime=current_detection.regime,
            previous_zscore=previous_detection.zscore,
            current_zscore=current_detection.zscore,
        )

    def detect_shift_from_zscores(
        self,
        previous_zscore: Decimal | None,
        current_zscore: Decimal | None,
    ) -> RegimeShiftResult:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "detect_shift_from_zscores", _analytics_args)
        except Exception:
            pass
        previous_detection = self.detect_from_zscore(previous_zscore)
        current_detection = self.detect_from_zscore(current_zscore)

        return self.detect_shift_from_values(
            previous_regime=previous_detection.regime,
            current_regime=current_detection.regime,
            previous_zscore=previous_zscore,
            current_zscore=current_zscore,
        )

    def detect_shift_from_values(
        self,
        previous_regime: SpreadRegime | None,
        current_regime: SpreadRegime | None,
        previous_zscore: Decimal | None = None,
        current_zscore: Decimal | None = None,
    ) -> RegimeShiftResult:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "detect_shift_from_values", _analytics_args)
        except Exception:
            pass
        previous_rank = _regime_rank(previous_regime) if previous_regime else None
        current_rank = _regime_rank(current_regime) if current_regime else None

        zscore_delta = _safe_delta(current_zscore, previous_zscore)
        rank_delta = (
            current_rank - previous_rank
            if current_rank is not None and previous_rank is not None
            else None
        )

        if previous_regime is None or current_regime is None:
            return RegimeShiftResult(
                changed=False,
                previous_regime=previous_regime,
                current_regime=current_regime,
                previous_zscore=previous_zscore,
                current_zscore=current_zscore,
                zscore_delta=zscore_delta,
                previous_rank=previous_rank,
                current_rank=current_rank,
                rank_delta=rank_delta,
                reason="missing_regime",
            )

        changed = previous_regime != current_regime

        return RegimeShiftResult(
            changed=changed,
            previous_regime=previous_regime,
            current_regime=current_regime,
            previous_zscore=previous_zscore,
            current_zscore=current_zscore,
            zscore_delta=zscore_delta,
            previous_rank=previous_rank,
            current_rank=current_rank,
            rank_delta=rank_delta,
            reason=self._build_shift_reason(
                changed=changed,
                previous_regime=previous_regime,
                current_regime=current_regime,
            ),
        )

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    def is_compressed(self, stats: RollingStats | None) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_compressed", _analytics_args)
        except Exception:
            pass
        return self.detect_from_stats(stats).is_compressed

    def is_elevated(self, stats: RollingStats | None) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_elevated", _analytics_args)
        except Exception:
            pass
        return self.detect_from_stats(stats).is_elevated

    def is_extreme(self, stats: RollingStats | None) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_extreme", _analytics_args)
        except Exception:
            pass
        return self.detect_from_stats(stats).is_extreme

    def is_dislocated(self, stats: RollingStats | None) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_dislocated", _analytics_args)
        except Exception:
            pass
        return self.detect_from_stats(stats).is_dislocated

    def is_abnormal(self, stats: RollingStats | None) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_abnormal", _analytics_args)
        except Exception:
            pass
        return self.detect_from_stats(stats).is_abnormal

    def is_high_risk(self, stats: RollingStats | None) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_high_risk", _analytics_args)
        except Exception:
            pass
        return self.detect_from_stats(stats).is_high_risk

    # ------------------------------------------------------------------
    # Threshold accessors
    # ------------------------------------------------------------------

    @property
    def compressed_threshold(self) -> Decimal:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "compressed_threshold", _analytics_args)
        except Exception:
            pass
        return self._compressed_threshold

    @property
    def elevated_threshold(self) -> Decimal:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "elevated_threshold", _analytics_args)
        except Exception:
            pass
        return self._elevated_threshold

    @property
    def extreme_threshold(self) -> Decimal:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "extreme_threshold", _analytics_args)
        except Exception:
            pass
        return self._extreme_threshold

    @property
    def dislocated_threshold(self) -> Decimal:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "dislocated_threshold", _analytics_args)
        except Exception:
            pass
        return self._dislocated_threshold

    def thresholds_payload(self) -> dict[str, str]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "thresholds_payload", _analytics_args)
        except Exception:
            pass
        return {
            "compressed": str(self._compressed_threshold),
            "elevated": str(self._elevated_threshold),
            "extreme": str(self._extreme_threshold),
            "dislocated": str(self._dislocated_threshold),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _missing_result(self, *, reason: str) -> RegimeDetectionResult:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_missing_result", _analytics_args)
        except Exception:
            pass
        return RegimeDetectionResult(
            regime=SpreadRegime.NORMAL,
            zscore=None,
            abs_zscore=None,
            is_compressed=False,
            is_elevated=False,
            is_extreme=False,
            is_dislocated=False,
            threshold_compressed=self._compressed_threshold,
            threshold_elevated=self._elevated_threshold,
            threshold_extreme=self._extreme_threshold,
            threshold_dislocated=self._dislocated_threshold,
            reason=reason,
        )

    def _resolve_regime(self, abs_zscore: Decimal) -> SpreadRegime:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_resolve_regime", _analytics_args)
        except Exception:
            pass
        if abs_zscore >= self._dislocated_threshold:
            return SpreadRegime.DISLOCATED

        if abs_zscore >= self._extreme_threshold:
            return SpreadRegime.EXTREME

        if abs_zscore >= self._elevated_threshold:
            return SpreadRegime.ELEVATED

        if abs_zscore <= self._compressed_threshold:
            return SpreadRegime.COMPRESSED

        return SpreadRegime.NORMAL

    def _validate_threshold_order(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_threshold_order", _analytics_args)
        except Exception:
            pass
        if self._compressed_threshold >= self._elevated_threshold:
            raise ValueError("compressed_threshold must be < elevated_threshold")

        if self._elevated_threshold > self._extreme_threshold:
            raise ValueError("elevated_threshold must be <= extreme_threshold")

        if self._extreme_threshold > self._dislocated_threshold:
            raise ValueError("extreme_threshold must be <= dislocated_threshold")

    @staticmethod
    def _extract_zscore(snapshot: SpreadSnapshot | None) -> Decimal | None:
        try:
            _analytics_class_name = "SpreadRegimeDetector"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_extract_zscore", _analytics_args)
        except Exception:
            pass
        if snapshot is None or snapshot.stats is None:
            return None
        return snapshot.stats.zscore

    def _build_detection_reason(
        self,
        *,
        regime: SpreadRegime,
        zscore: Decimal | None,
        abs_zscore: Decimal | None,
    ) -> str:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_detection_reason", _analytics_args)
        except Exception:
            pass
        return (
            f"regime={regime.value};"
            f"zscore={zscore};"
            f"abs_zscore={abs_zscore};"
            f"compressed_threshold={self._compressed_threshold};"
            f"elevated_threshold={self._elevated_threshold};"
            f"extreme_threshold={self._extreme_threshold};"
            f"dislocated_threshold={self._dislocated_threshold}"
        )

    @staticmethod
    def _build_shift_reason(
        *,
        changed: bool,
        previous_regime: SpreadRegime,
        current_regime: SpreadRegime,
    ) -> str:
        try:
            _analytics_class_name = "SpreadRegimeDetector"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_shift_reason", _analytics_args)
        except Exception:
            pass
        if not changed:
            return f"no_shift:{previous_regime.value}"

        if _regime_rank(current_regime) > _regime_rank(previous_regime):
            return f"shift_up:{previous_regime.value}->{current_regime.value}"

        return f"shift_down:{previous_regime.value}->{current_regime.value}"


# ============================================================
# Module helpers
# ============================================================

def _validate_positive_decimal(name: str, value: Decimal) -> Decimal:
    if value <= DECIMAL_ZERO:
        raise ValueError(f"{name} must be > 0")
    return value


def _regime_rank(regime: SpreadRegime | None) -> int:
    if regime is None:
        return -1

    rank_attr = getattr(regime, "rank", None)
    if isinstance(rank_attr, int):
        return rank_attr

    order = {
        SpreadRegime.COMPRESSED: 0,
        SpreadRegime.NORMAL: 1,
        SpreadRegime.ELEVATED: 2,
        SpreadRegime.EXTREME: 3,
        SpreadRegime.DISLOCATED: 4,
    }
    return order[regime]


def _safe_delta(
    current_value: Decimal | None,
    previous_value: Decimal | None,
) -> Decimal | None:
    if current_value is None or previous_value is None:
        return None
    return current_value - previous_value


def _decimal_to_payload(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


__all__ = [
    "RegimeDetectionResult",
    "RegimeShiftResult",
    "SpreadRegimeDetector",
]