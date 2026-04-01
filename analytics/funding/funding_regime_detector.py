from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.logger import get_logger

from .enums import (
    FundingBias,
    FundingRegime,
    FundingTimeframe,
)
from .models import (
    FundingRegimeState,
    FundingSnapshot,
    FundingStatistics,
)


@dataclass(slots=True)
class FundingRegimeDetectorConfig:
    """
    Конфігурація детектора funding regime.

    Порогові значення можна потім винести у central Config,
    але вже зараз клас повністю готовий до роботи.
    """

    neutral_abs_threshold: float = 0.00001
    positive_abs_threshold: float = 0.00005
    extreme_abs_threshold: float = 0.00030

    crowded_percentile_threshold: float = 85.0
    squeeze_percentile_threshold: float = 95.0

    extreme_positive_zscore: float = 2.0
    extreme_negative_zscore: float = -2.0

    min_confidence_for_change: float = 0.15
    default_timeframe: FundingTimeframe = FundingTimeframe.H1


class FundingRegimeDetector:
    """
    Детектор режимів funding rate.

    Основна відповідальність:
    - інтерпретувати snapshot + statistics
    - визначати regime
    - визначати bias
    - обчислювати confidence
    - визначати факт regime change
    - повертати FundingRegimeState

    Клас не зберігає історію самостійно.
    Історію та попередній state має підтримувати analyzer.
    """

    def __init__(
        self,
        config: FundingRegimeDetectorConfig | None = None,
    ) -> None:
        self.config = config or FundingRegimeDetectorConfig()
        self.logger = get_logger(__name__)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def detect(
        self,
        snapshot: FundingSnapshot,
        statistics: FundingStatistics,
        previous_state: FundingRegimeState | None = None,
        timeframe: FundingTimeframe | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> FundingRegimeState:
        """
        Побудова FundingRegimeState на основі snapshot + statistics.

        Parameters
        ----------
        snapshot:
            Поточний funding snapshot.
        statistics:
            Агрегована статистика по funding.
        previous_state:
            Попередній regime state для визначення changed / previous_regime.
        timeframe:
            Таймфрейм аналізу. Якщо не заданий, береться з config.
        extra_metadata:
            Додаткові службові поля для metadata.

        Returns
        -------
        FundingRegimeState
        """
        tf = timeframe or statistics.timeframe or self.config.default_timeframe

        regime = self.detect_regime(
            current_rate=snapshot.funding_rate,
            percentile=statistics.percentile,
            zscore=statistics.zscore,
        )

        bias = self.detect_bias(
            current_rate=snapshot.funding_rate,
            percentile=statistics.percentile,
            regime=regime,
        )

        confidence = self.calculate_confidence(
            current_rate=snapshot.funding_rate,
            percentile=statistics.percentile,
            zscore=statistics.zscore,
            sample_size=statistics.sample_size,
        )

        previous_regime = previous_state.regime if previous_state is not None else None
        changed = self.has_regime_changed(
            previous_state=previous_state,
            new_regime=regime,
            confidence=confidence,
        )

        metadata: dict[str, Any] = {
            "sample_size": statistics.sample_size,
            "mean_rate": statistics.mean_rate,
            "median_rate": statistics.median_rate,
            "std_rate": statistics.std_rate,
            "min_rate": statistics.min_rate,
            "max_rate": statistics.max_rate,
            "current_rate_abs": abs(snapshot.funding_rate),
            "basis": snapshot.basis,
            "funding_sign": snapshot.funding_sign,
        }

        if extra_metadata:
            metadata.update(extra_metadata)

        state = FundingRegimeState(
            symbol=snapshot.symbol,
            exchange=snapshot.exchange,
            timeframe=tf,
            regime=regime,
            bias=bias,
            current_rate=snapshot.funding_rate,
            mean_rate=statistics.mean_rate,
            zscore=statistics.zscore,
            percentile=statistics.percentile,
            confidence=confidence,
            changed=changed,
            previous_regime=previous_regime,
            event_time=snapshot.event_time,
            metadata=metadata,
        )

        self.logger.debug(
            "Funding regime detected: symbol=%s exchange=%s regime=%s bias=%s "
            "rate=%.8f percentile=%s zscore=%s confidence=%.4f changed=%s",
            state.symbol,
            state.exchange.value,
            state.regime.value,
            state.bias.value,
            state.current_rate,
            f"{state.percentile:.2f}" if state.percentile is not None else "None",
            f"{state.zscore:.4f}" if state.zscore is not None else "None",
            state.confidence,
            state.changed,
        )

        return state

    def detect_regime(
        self,
        current_rate: float,
        percentile: float | None,
        zscore: float | None,
    ) -> FundingRegime:
        """
        Визначає режим funding.

        Пріоритет:
        1. Percentile extremes
        2. Z-score extremes
        3. Absolute thresholds
        """
        abs_rate = abs(current_rate)

        # 1. Percentile-based extremes
        if percentile is not None:
            if (
                current_rate > 0
                and percentile >= self.config.squeeze_percentile_threshold
            ):
                return FundingRegime.EXTREME_POSITIVE

            if (
                current_rate < 0
                and percentile <= (100.0 - self.config.squeeze_percentile_threshold)
            ):
                return FundingRegime.EXTREME_NEGATIVE

        # 2. Z-score-based extremes
        if zscore is not None:
            if (
                current_rate > 0
                and zscore >= self.config.extreme_positive_zscore
            ):
                return FundingRegime.EXTREME_POSITIVE

            if (
                current_rate < 0
                and zscore <= self.config.extreme_negative_zscore
            ):
                return FundingRegime.EXTREME_NEGATIVE

        # 3. Absolute thresholds
        if abs_rate <= self.config.neutral_abs_threshold:
            return FundingRegime.NEUTRAL

        if current_rate > 0:
            if abs_rate >= self.config.extreme_abs_threshold:
                return FundingRegime.EXTREME_POSITIVE
            if abs_rate >= self.config.positive_abs_threshold:
                return FundingRegime.POSITIVE
            return FundingRegime.POSITIVE

        if current_rate < 0:
            if abs_rate >= self.config.extreme_abs_threshold:
                return FundingRegime.EXTREME_NEGATIVE
            if abs_rate >= self.config.positive_abs_threshold:
                return FundingRegime.NEGATIVE
            return FundingRegime.NEGATIVE

        return FundingRegime.UNKNOWN

    def detect_bias(
        self,
        current_rate: float,
        percentile: float | None,
        regime: FundingRegime,
    ) -> FundingBias:
        """
        Визначає ринковий bias на основі funding regime та положення в розподілі.
        """
        abs_rate = abs(current_rate)

        if regime == FundingRegime.NEUTRAL or abs_rate <= self.config.neutral_abs_threshold:
            return FundingBias.NEUTRAL

        if current_rate > 0:
            if regime == FundingRegime.EXTREME_POSITIVE:
                return FundingBias.SQUEEZE_RISK_LONGS

            if percentile is not None:
                if percentile >= self.config.squeeze_percentile_threshold:
                    return FundingBias.SQUEEZE_RISK_LONGS
                if percentile >= self.config.crowded_percentile_threshold:
                    return FundingBias.OVERCROWDED_LONGS

            return FundingBias.LONG_BIAS

        if current_rate < 0:
            if regime == FundingRegime.EXTREME_NEGATIVE:
                return FundingBias.SQUEEZE_RISK_SHORTS

            if percentile is not None:
                if percentile <= (100.0 - self.config.squeeze_percentile_threshold):
                    return FundingBias.SQUEEZE_RISK_SHORTS
                if percentile <= (100.0 - self.config.crowded_percentile_threshold):
                    return FundingBias.OVERCROWDED_SHORTS

            return FundingBias.SHORT_BIAS

        return FundingBias.NEUTRAL

    def calculate_confidence(
        self,
        current_rate: float,
        percentile: float | None,
        zscore: float | None,
        sample_size: int,
    ) -> float:
        """
        Обчислює впевненість у regime classification.

        Компоненти:
        - absolute magnitude funding
        - position in historical distribution
        - z-score abnormality
        - sample size quality
        """
        magnitude_score = self._calc_magnitude_score(current_rate)
        percentile_score = self._calc_percentile_score(percentile)
        zscore_score = self._calc_zscore_score(zscore)
        sample_quality_score = self._calc_sample_quality_score(sample_size)

        confidence = (
            0.40 * magnitude_score
            + 0.25 * percentile_score
            + 0.20 * zscore_score
            + 0.15 * sample_quality_score
        )

        return max(0.0, min(1.0, confidence))

    def has_regime_changed(
        self,
        previous_state: FundingRegimeState | None,
        new_regime: FundingRegime,
        confidence: float,
    ) -> bool:
        """
        Визначає, чи є новий regime справжньою зміною,
        а не шумом на межі порогів.
        """
        if previous_state is None:
            return False

        if previous_state.regime == new_regime:
            return False

        if confidence < self.config.min_confidence_for_change:
            return False

        return True

    # ---------------------------------------------------------------------
    # Internal scoring helpers
    # ---------------------------------------------------------------------

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

        normalized = abs(zscore) / 3.0
        return max(0.0, min(1.0, normalized))

    def _calc_sample_quality_score(self, sample_size: int) -> float:
        """
        Оцінка якості статистики по розміру історії.
        """
        if sample_size <= 1:
            return 0.0
        if sample_size >= 100:
            return 1.0

        return max(0.0, min(1.0, sample_size / 100.0))