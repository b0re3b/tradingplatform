from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.logger import get_logger

from .enums import (
    FundingExtremeType,
    FundingRegime,
    FundingTimeframe,
)
from .models import (
    FundingExtremeEvent,
    FundingRegimeState,
    FundingSnapshot,
    FundingStatistics,
)


@dataclass(slots=True)
class FundingExtremesConfig:
    """
    Конфігурація детектора екстремумів funding.

    Детектор шукає:
    - абсолютні екстремуми
    - percentile extremes
    - z-score extremes
    - локальні high/low відносно історичного вікна
    """

    default_timeframe: FundingTimeframe = FundingTimeframe.H1

    neutral_abs_threshold: float = 0.00001
    elevated_abs_threshold: float = 0.00008
    extreme_abs_threshold: float = 0.00030

    elevated_zscore_threshold: float = 1.5
    extreme_zscore_threshold: float = 2.5

    elevated_percentile_high: float = 90.0
    extreme_percentile_high: float = 97.5

    elevated_percentile_low: float = 10.0
    extreme_percentile_low: float = 2.5

    min_sample_size: int = 20
    min_severity: float = 0.20

    enable_local_extremes: bool = True
    enable_percentile_extremes: bool = True
    enable_zscore_extremes: bool = True
    enable_absolute_extremes: bool = True


class FundingExtremesDetector:
    """
    Детектор екстремальних funding-станів.

    Клас:
    - не працює з EventBus напряму
    - не зберігає історію
    - приймає snapshot/statistics/regime_state
    - повертає FundingExtremeEvent або None

    Це pure analytics detector для використання у FundingAnalyzer.
    """

    def __init__(
        self,
        config: FundingExtremesConfig | None = None,
    ) -> None:
        self.config = config or FundingExtremesConfig()
        self.logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        snapshot: FundingSnapshot,
        statistics: FundingStatistics,
        regime_state: FundingRegimeState | None = None,
        timeframe: FundingTimeframe | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> FundingExtremeEvent | None:
        """
        Виявляє funding extreme event.

        Parameters
        ----------
        snapshot:
            Поточний funding snapshot.
        statistics:
            Агрегована funding-статистика по вікну.
        regime_state:
            Поточний regime state. Необов'язковий, але бажаний.
        timeframe:
            Таймфрейм аналізу.
        extra_metadata:
            Додаткові metadata.

        Returns
        -------
        FundingExtremeEvent | None
        """
        if statistics.sample_size < self.config.min_sample_size:
            self.logger.debug(
                "Funding extreme skipped due to low sample size: symbol=%s sample_size=%s",
                snapshot.symbol,
                statistics.sample_size,
            )
            return None

        tf = timeframe or statistics.timeframe or self.config.default_timeframe
        regime = regime_state.regime if regime_state is not None else self._infer_regime(snapshot.funding_rate)

        extreme_type = self.detect_extreme_type(
            current_rate=snapshot.funding_rate,
            min_rate=statistics.min_rate,
            max_rate=statistics.max_rate,
            percentile=statistics.percentile,
            zscore=statistics.zscore,
        )

        if extreme_type == FundingExtremeType.NONE:
            return None

        severity = self.calculate_severity(
            current_rate=snapshot.funding_rate,
            percentile=statistics.percentile,
            zscore=statistics.zscore,
            extreme_type=extreme_type,
        )

        if severity < self.config.min_severity:
            self.logger.debug(
                "Funding extreme ignored due to low severity: symbol=%s type=%s severity=%.4f",
                snapshot.symbol,
                extreme_type.value,
                severity,
            )
            return None

        is_reversal_risk = self.is_reversal_risk(
            extreme_type=extreme_type,
            regime=regime,
            severity=severity,
        )
        is_squeeze_risk = self.is_squeeze_risk(
            extreme_type=extreme_type,
            regime=regime,
            severity=severity,
        )

        metadata: dict[str, Any] = {
            "sample_size": statistics.sample_size,
            "mean_rate": statistics.mean_rate,
            "median_rate": statistics.median_rate,
            "std_rate": statistics.std_rate,
            "min_rate": statistics.min_rate,
            "max_rate": statistics.max_rate,
            "percentile": statistics.percentile,
            "zscore": statistics.zscore,
            "current_rate_abs": abs(snapshot.funding_rate),
            "basis": snapshot.basis,
            "funding_sign": snapshot.funding_sign,
        }

        if regime_state is not None:
            metadata["regime_confidence"] = regime_state.confidence
            metadata["bias"] = regime_state.bias.value

        if extra_metadata:
            metadata.update(extra_metadata)

        event = FundingExtremeEvent(
            symbol=snapshot.symbol,
            exchange=snapshot.exchange,
            timeframe=tf,
            extreme_type=extreme_type,
            regime=regime,
            funding_rate=snapshot.funding_rate,
            zscore=statistics.zscore,
            percentile=statistics.percentile,
            severity=severity,
            is_reversal_risk=is_reversal_risk,
            is_squeeze_risk=is_squeeze_risk,
            event_time=snapshot.event_time,
            metadata=metadata,
        )

        self.logger.debug(
            "Funding extreme detected: symbol=%s exchange=%s type=%s regime=%s "
            "rate=%.8f percentile=%s zscore=%s severity=%.4f reversal_risk=%s squeeze_risk=%s",
            event.symbol,
            event.exchange.value,
            event.extreme_type.value,
            event.regime.value,
            event.funding_rate,
            f"{event.percentile:.2f}" if event.percentile is not None else "None",
            f"{event.zscore:.4f}" if event.zscore is not None else "None",
            event.severity,
            event.is_reversal_risk,
            event.is_squeeze_risk,
        )

        return event

    def detect_extreme_type(
        self,
        current_rate: float,
        min_rate: float,
        max_rate: float,
        percentile: float | None,
        zscore: float | None,
    ) -> FundingExtremeType:
        """
        Визначає тип funding extreme.

        Пріоритет:
        1. Local/global extremes
        2. Percentile extremes
        3. Z-score extremes
        4. Absolute extremes
        """
        if self.config.enable_local_extremes:
            if current_rate >= max_rate and current_rate > self.config.neutral_abs_threshold:
                return FundingExtremeType.GLOBAL_HIGH
            if current_rate <= min_rate and current_rate < -self.config.neutral_abs_threshold:
                return FundingExtremeType.GLOBAL_LOW

        if self.config.enable_percentile_extremes and percentile is not None:
            if percentile >= self.config.extreme_percentile_high:
                return FundingExtremeType.PERCENTILE_HIGH
            if percentile <= self.config.extreme_percentile_low:
                return FundingExtremeType.PERCENTILE_LOW

        if self.config.enable_zscore_extremes and zscore is not None:
            if zscore >= self.config.extreme_zscore_threshold:
                return FundingExtremeType.ZSCORE_HIGH
            if zscore <= -self.config.extreme_zscore_threshold:
                return FundingExtremeType.ZSCORE_LOW

        if self.config.enable_absolute_extremes:
            abs_rate = abs(current_rate)
            if abs_rate >= self.config.extreme_abs_threshold:
                if current_rate > 0:
                    return FundingExtremeType.LOCAL_HIGH
                if current_rate < 0:
                    return FundingExtremeType.LOCAL_LOW

        return FundingExtremeType.NONE

    def calculate_severity(
        self,
        current_rate: float,
        percentile: float | None,
        zscore: float | None,
        extreme_type: FundingExtremeType,
    ) -> float:
        """
        Severity в діапазоні [0, 1].

        Враховуються:
        - абсолютна величина funding
        - percentile distance
        - z-score abnormality
        - бонус за тип extreme
        """
        magnitude_score = self._calc_magnitude_score(current_rate)
        percentile_score = self._calc_percentile_score(percentile)
        zscore_score = self._calc_zscore_score(zscore)
        type_bonus = self._calc_type_bonus(extreme_type)

        severity = (
            0.35 * magnitude_score
            + 0.25 * percentile_score
            + 0.25 * zscore_score
            + 0.15 * type_bonus
        )

        return max(0.0, min(1.0, severity))

    def is_reversal_risk(
        self,
        extreme_type: FundingExtremeType,
        regime: FundingRegime,
        severity: float,
    ) -> bool:
        """
        Extreme funding часто означає ризик mean reversion / reversal.
        """
        if severity < 0.45:
            return False

        if extreme_type in {
            FundingExtremeType.GLOBAL_HIGH,
            FundingExtremeType.GLOBAL_LOW,
            FundingExtremeType.PERCENTILE_HIGH,
            FundingExtremeType.PERCENTILE_LOW,
            FundingExtremeType.ZSCORE_HIGH,
            FundingExtremeType.ZSCORE_LOW,
        }:
            return True

        if regime in {
            FundingRegime.EXTREME_POSITIVE,
            FundingRegime.EXTREME_NEGATIVE,
        } and severity >= 0.55:
            return True

        return False

    def is_squeeze_risk(
        self,
        extreme_type: FundingExtremeType,
        regime: FundingRegime,
        severity: float,
    ) -> bool:
        """
        Extreme funding також може означати ризик squeeze.
        """
        if severity < 0.50:
            return False

        if regime in {
            FundingRegime.EXTREME_POSITIVE,
            FundingRegime.EXTREME_NEGATIVE,
        }:
            return True

        if extreme_type in {
            FundingExtremeType.GLOBAL_HIGH,
            FundingExtremeType.GLOBAL_LOW,
            FundingExtremeType.PERCENTILE_HIGH,
            FundingExtremeType.PERCENTILE_LOW,
        }:
            return severity >= 0.60

        return False

    # ------------------------------------------------------------------
    # Helper API
    # ------------------------------------------------------------------

    def is_positive_extreme(self, event: FundingExtremeEvent | None) -> bool:
        if event is None:
            return False

        return event.funding_rate > 0 and event.extreme_type in {
            FundingExtremeType.LOCAL_HIGH,
            FundingExtremeType.GLOBAL_HIGH,
            FundingExtremeType.ZSCORE_HIGH,
            FundingExtremeType.PERCENTILE_HIGH,
        }

    def is_negative_extreme(self, event: FundingExtremeEvent | None) -> bool:
        if event is None:
            return False

        return event.funding_rate < 0 and event.extreme_type in {
            FundingExtremeType.LOCAL_LOW,
            FundingExtremeType.GLOBAL_LOW,
            FundingExtremeType.ZSCORE_LOW,
            FundingExtremeType.PERCENTILE_LOW,
        }

    def is_high_severity(self, event: FundingExtremeEvent | None, threshold: float = 0.70) -> bool:
        if event is None:
            return False
        return event.severity >= threshold

    def build_summary(self, event: FundingExtremeEvent) -> str:
        return (
            f"Funding extreme for {event.symbol}: "
            f"type={event.extreme_type.value}, "
            f"regime={event.regime.value}, "
            f"funding_rate={event.funding_rate:.8f}, "
            f"severity={event.severity:.4f}, "
            f"reversal_risk={event.is_reversal_risk}, "
            f"squeeze_risk={event.is_squeeze_risk}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_regime(self, funding_rate: float) -> FundingRegime:
        abs_rate = abs(funding_rate)

        if abs_rate <= self.config.neutral_abs_threshold:
            return FundingRegime.NEUTRAL

        if funding_rate > 0:
            if abs_rate >= self.config.extreme_abs_threshold:
                return FundingRegime.EXTREME_POSITIVE
            return FundingRegime.POSITIVE

        if funding_rate < 0:
            if abs_rate >= self.config.extreme_abs_threshold:
                return FundingRegime.EXTREME_NEGATIVE
            return FundingRegime.NEGATIVE

        return FundingRegime.UNKNOWN

    def _calc_magnitude_score(self, current_rate: float) -> float:
        abs_rate = abs(current_rate)

        if abs_rate <= self.config.neutral_abs_threshold:
            return 0.0

        denominator = max(
            self.config.extreme_abs_threshold - self.config.neutral_abs_threshold,
            1e-12,
        )
        normalized = (abs_rate - self.config.neutral_abs_threshold) / denominator
        return max(0.0, min(1.0, normalized))

    def _calc_percentile_score(self, percentile: float | None) -> float:
        if percentile is None:
            return 0.0

        distance_from_center = abs(percentile - 50.0) / 50.0
        return max(0.0, min(1.0, distance_from_center))

    def _calc_zscore_score(self, zscore: float | None) -> float:
        if zscore is None:
            return 0.0

        normalized = abs(zscore) / max(self.config.extreme_zscore_threshold, 1e-12)
        return max(0.0, min(1.0, normalized))

    def _calc_type_bonus(self, extreme_type: FundingExtremeType) -> float:
        if extreme_type in {
            FundingExtremeType.GLOBAL_HIGH,
            FundingExtremeType.GLOBAL_LOW,
        }:
            return 1.0

        if extreme_type in {
            FundingExtremeType.PERCENTILE_HIGH,
            FundingExtremeType.PERCENTILE_LOW,
            FundingExtremeType.ZSCORE_HIGH,
            FundingExtremeType.ZSCORE_LOW,
        }:
            return 0.85

        if extreme_type in {
            FundingExtremeType.LOCAL_HIGH,
            FundingExtremeType.LOCAL_LOW,
        }:
            return 0.65

        return 0.0