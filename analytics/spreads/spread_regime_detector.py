from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .config import BaseSpreadConfig
from .enums import SpreadRegime
from .models import RollingStats, SpreadSnapshot


DECIMAL_ZERO = Decimal("0")


@dataclass(slots=True)
class RegimeDetectionResult:
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


@dataclass(slots=True)
class RegimeShiftResult:
    changed: bool
    previous_regime: SpreadRegime | None = None
    current_regime: SpreadRegime | None = None

    previous_zscore: Decimal | None = None
    current_zscore: Decimal | None = None
    zscore_delta: Decimal | None = None

    reason: str | None = None

    @property
    def is_shift_up(self) -> bool:
        if self.previous_regime is None or self.current_regime is None:
            return False
        return _regime_rank(self.current_regime) > _regime_rank(self.previous_regime)

    @property
    def is_shift_down(self) -> bool:
        if self.previous_regime is None or self.current_regime is None:
            return False
        return _regime_rank(self.current_regime) < _regime_rank(self.previous_regime)


class SpreadRegimeDetector:
    """
    Доменний сервіс для класифікації spread regime та виявлення regime shifts.

    Основні задачі:
    - визначити regime за z-score / stats
    - визначити dislocation
    - порівняти два стани та знайти regime shift
    """

    def __init__(
        self,
        config: BaseSpreadConfig,
        elevated_threshold: Decimal = Decimal("1.5"),
        compressed_threshold: Decimal = Decimal("0.5"),
        dislocated_threshold: Decimal | None = None,
    ) -> None:
        self._config = config
        self._elevated_threshold = elevated_threshold
        self._compressed_threshold = compressed_threshold
        self._extreme_threshold = config.anomaly_zscore_threshold
        self._dislocated_threshold = (
            dislocated_threshold
            if dislocated_threshold is not None
            else config.anomaly_zscore_threshold * Decimal("1.5")
        )

    def detect_from_snapshot(self, snapshot: SpreadSnapshot) -> RegimeDetectionResult:
        return self.detect_from_stats(snapshot.stats)

    def detect_from_stats(self, stats: RollingStats | None) -> RegimeDetectionResult:
        if stats is None or stats.zscore is None:
            return RegimeDetectionResult(
                regime=SpreadRegime.NORMAL,
                zscore=None,
                abs_zscore=None,
                threshold_compressed=self._compressed_threshold,
                threshold_elevated=self._elevated_threshold,
                threshold_extreme=self._extreme_threshold,
                threshold_dislocated=self._dislocated_threshold,
                reason="missing_zscore",
            )

        zscore = stats.zscore
        abs_zscore = abs(zscore)

        is_dislocated = abs_zscore >= self._dislocated_threshold
        is_extreme = abs_zscore >= self._extreme_threshold
        is_elevated = abs_zscore >= self._elevated_threshold
        is_compressed = abs_zscore <= self._compressed_threshold

        regime = self._resolve_regime(abs_zscore)

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
            reason=self._build_reason(
                regime=regime,
                zscore=zscore,
                abs_zscore=abs_zscore,
            ),
        )

    def detect_regime(self, zscore: Decimal | None) -> SpreadRegime:
        if zscore is None:
            return SpreadRegime.NORMAL
        return self._resolve_regime(abs(zscore))

    def detect_shift(
        self,
        previous_snapshot: SpreadSnapshot | None,
        current_snapshot: SpreadSnapshot | None,
    ) -> RegimeShiftResult:
        previous_regime = previous_snapshot.regime if previous_snapshot is not None else None
        current_regime = current_snapshot.regime if current_snapshot is not None else None

        previous_zscore = (
            previous_snapshot.stats.zscore
            if previous_snapshot is not None and previous_snapshot.stats is not None
            else None
        )
        current_zscore = (
            current_snapshot.stats.zscore
            if current_snapshot is not None and current_snapshot.stats is not None
            else None
        )

        return self.detect_shift_from_values(
            previous_regime=previous_regime,
            current_regime=current_regime,
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
        if previous_regime is None or current_regime is None:
            return RegimeShiftResult(
                changed=False,
                previous_regime=previous_regime,
                current_regime=current_regime,
                previous_zscore=previous_zscore,
                current_zscore=current_zscore,
                zscore_delta=_safe_delta(current_zscore, previous_zscore),
                reason="missing_regime",
            )

        changed = previous_regime != current_regime

        return RegimeShiftResult(
            changed=changed,
            previous_regime=previous_regime,
            current_regime=current_regime,
            previous_zscore=previous_zscore,
            current_zscore=current_zscore,
            zscore_delta=_safe_delta(current_zscore, previous_zscore),
            reason=self._build_shift_reason(
                changed=changed,
                previous_regime=previous_regime,
                current_regime=current_regime,
            ),
        )

    def is_compressed(self, stats: RollingStats | None) -> bool:
        result = self.detect_from_stats(stats)
        return result.is_compressed

    def is_elevated(self, stats: RollingStats | None) -> bool:
        result = self.detect_from_stats(stats)
        return result.is_elevated

    def is_extreme(self, stats: RollingStats | None) -> bool:
        result = self.detect_from_stats(stats)
        return result.is_extreme

    def is_dislocated(self, stats: RollingStats | None) -> bool:
        result = self.detect_from_stats(stats)
        return result.is_dislocated

    def _resolve_regime(self, abs_zscore: Decimal) -> SpreadRegime:
        if abs_zscore >= self._dislocated_threshold:
            return SpreadRegime.DISLOCATED

        if abs_zscore >= self._extreme_threshold:
            return SpreadRegime.EXTREME

        if abs_zscore >= self._elevated_threshold:
            return SpreadRegime.ELEVATED

        if abs_zscore <= self._compressed_threshold:
            return SpreadRegime.COMPRESSED

        return SpreadRegime.NORMAL

    def _build_reason(
        self,
        regime: SpreadRegime,
        zscore: Decimal | None,
        abs_zscore: Decimal | None,
    ) -> str:
        return (
            f"regime={regime.value}, "
            f"zscore={zscore}, "
            f"abs_zscore={abs_zscore}, "
            f"compressed_threshold={self._compressed_threshold}, "
            f"elevated_threshold={self._elevated_threshold}, "
            f"extreme_threshold={self._extreme_threshold}, "
            f"dislocated_threshold={self._dislocated_threshold}"
        )

    def _build_shift_reason(
        self,
        changed: bool,
        previous_regime: SpreadRegime,
        current_regime: SpreadRegime,
    ) -> str:
        if not changed:
            return f"no_shift:{previous_regime.value}"

        if _regime_rank(current_regime) > _regime_rank(previous_regime):
            return f"shift_up:{previous_regime.value}->{current_regime.value}"

        return f"shift_down:{previous_regime.value}->{current_regime.value}"


def _regime_rank(regime: SpreadRegime) -> int:
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