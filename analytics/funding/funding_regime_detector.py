from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.logger import get_logger

from analytics.funding.enums import (
    FundingBias,
    FundingRegime,
    FundingTimeframe,
)
from analytics.funding.models import (
    FundingRegimeState,
    FundingSnapshot,
    FundingStatistics,
    funding_key_to_dict,
)


@dataclass(slots=True)
class FundingRegimeDetectorConfig:
    """
    Конфігурація pure detector-а funding regime.

    Цей config не містить EventBus/Scheduler налаштувань, бо FundingRegimeDetector
    не є runtime-компонентом. Runtime lifecycle, subscriptions і publish logic
    належать FundingAnalyzer.

    Scope:
        exchange + market_type + symbol + timeframe
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

    service_name: str = "funding_regime_detector"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.neutral_abs_threshold < 0:
            raise ValueError("neutral_abs_threshold must be >= 0")

        if self.positive_abs_threshold < 0:
            raise ValueError("positive_abs_threshold must be >= 0")

        if self.extreme_abs_threshold < 0:
            raise ValueError("extreme_abs_threshold must be >= 0")

        if self.neutral_abs_threshold > self.positive_abs_threshold:
            raise ValueError(
                "neutral_abs_threshold must be <= positive_abs_threshold"
            )

        if self.positive_abs_threshold > self.extreme_abs_threshold:
            raise ValueError(
                "positive_abs_threshold must be <= extreme_abs_threshold"
            )

        if not 0.0 <= self.crowded_percentile_threshold <= 100.0:
            raise ValueError("crowded_percentile_threshold must be in [0, 100]")

        if not 0.0 <= self.squeeze_percentile_threshold <= 100.0:
            raise ValueError("squeeze_percentile_threshold must be in [0, 100]")

        if self.crowded_percentile_threshold > self.squeeze_percentile_threshold:
            raise ValueError(
                "crowded_percentile_threshold must be <= squeeze_percentile_threshold"
            )

        if self.extreme_negative_zscore > 0:
            raise ValueError("extreme_negative_zscore must be <= 0")

        if self.extreme_positive_zscore < 0:
            raise ValueError("extreme_positive_zscore must be >= 0")

        if not 0.0 <= self.min_confidence_for_change <= 1.0:
            raise ValueError("min_confidence_for_change must be in [0, 1]")


class FundingRegimeDetector:
    """
    Pure detector для класифікації funding regime.

    Відповідальність:
    - інтерпретує FundingSnapshot + FundingStatistics;
    - визначає FundingRegime;
    - визначає FundingBias;
    - рахує confidence;
    - визначає changed / previous_regime;
    - повертає FundingRegimeState з повним futures scope.

    Correct architecture:
        FundingAnalyzer
            -> FundingRegimeDetector.detect(...)
            -> FundingRegimeState
            -> FundingAnalyzer публікує analytics.funding.*

    Важливо:
    - не слухає EventBus;
    - не публікує EventBus events;
    - не має Scheduler jobs;
    - не читає exchange/data caches напряму;
    - не зберігає історію самостійно.
    """

    def __init__(
        self,
        config: FundingRegimeDetectorConfig | None = None,
    ) -> None:
        self.config = config or FundingRegimeDetectorConfig()
        self.config.validate()

        self.logger = get_logger(
            __name__,
            service_name=self.config.service_name,
            event_type="funding_regime_detector",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        Очікується, що snapshot і statistics вже належать одному scope:
            exchange + market_type + symbol + timeframe

        FundingAnalyzer відповідає за:
        - збереження history;
        - передачу previous_state;
        - EventBus publish;
        - cleanup.
        """
        self._validate_snapshot_statistics_scope(
            snapshot=snapshot,
            statistics=statistics,
        )

        tf = timeframe or statistics.timeframe or snapshot.timeframe or self.config.default_timeframe

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
            "scope": funding_key_to_dict(snapshot.key),
            "exchange_symbol": snapshot.exchange_symbol,
            "sample_size": statistics.sample_size,
            "mean_rate": statistics.mean_rate,
            "median_rate": statistics.median_rate,
            "std_rate": statistics.std_rate,
            "min_rate": statistics.min_rate,
            "max_rate": statistics.max_rate,
            "current_rate_abs": abs(snapshot.funding_rate),
            "basis": snapshot.basis,
            "funding_sign": snapshot.funding_sign,
            "statistics_window_start": (
                statistics.window_start.isoformat()
                if statistics.window_start is not None
                else None
            ),
            "statistics_window_end": (
                statistics.window_end.isoformat()
                if statistics.window_end is not None
                else None
            ),
        }

        if extra_metadata:
            metadata.update(extra_metadata)

        state = FundingRegimeState(
            symbol=snapshot.symbol,
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            timeframe=tf,
            exchange_symbol=snapshot.exchange_symbol,
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
            "Funding regime detected | exchange=%s market_type=%s symbol=%s "
            "timeframe=%s regime=%s bias=%s rate=%.8f percentile=%s "
            "zscore=%s confidence=%.4f changed=%s",
            state.exchange.value,
            state.market_type,
            state.symbol,
            state.timeframe.value,
            state.regime.value,
            state.bias.value,
            state.current_rate,
            f"{state.percentile:.2f}" if state.percentile is not None else "None",
            f"{state.zscore:.4f}" if state.zscore is not None else "None",
            state.confidence,
            state.changed,
            extra={
                "scope": funding_key_to_dict(state.key),
                "exchange_symbol": state.exchange_symbol,
            },
        )

        return state

    def detect_regime(
        self,
        current_rate: float,
        percentile: float | None,
        zscore: float | None,
    ) -> FundingRegime:
        """
        Визначає funding regime.

        Пріоритет:
        1. percentile extremes;
        2. z-score extremes;
        3. absolute thresholds.
        """
        rate = float(current_rate)
        abs_rate = abs(rate)

        if percentile is not None:
            percentile = self._clamp(percentile, 0.0, 100.0)

            if rate > 0 and percentile >= self.config.squeeze_percentile_threshold:
                return FundingRegime.EXTREME_POSITIVE

            if rate < 0 and percentile <= (100.0 - self.config.squeeze_percentile_threshold):
                return FundingRegime.EXTREME_NEGATIVE

        if zscore is not None:
            zscore = float(zscore)

            if rate > 0 and zscore >= self.config.extreme_positive_zscore:
                return FundingRegime.EXTREME_POSITIVE

            if rate < 0 and zscore <= self.config.extreme_negative_zscore:
                return FundingRegime.EXTREME_NEGATIVE

        if abs_rate <= self.config.neutral_abs_threshold:
            return FundingRegime.NEUTRAL

        if rate > 0:
            if abs_rate >= self.config.extreme_abs_threshold:
                return FundingRegime.EXTREME_POSITIVE
            return FundingRegime.POSITIVE

        if rate < 0:
            if abs_rate >= self.config.extreme_abs_threshold:
                return FundingRegime.EXTREME_NEGATIVE
            return FundingRegime.NEGATIVE

        return FundingRegime.UNKNOWN

    def detect_bias(
        self,
        current_rate: float,
        percentile: float | None,
        regime: FundingRegime,
    ) -> FundingBias:
        """
        Визначає ринковий bias на основі funding regime і позиції
        в історичному розподілі.

        Positive funding:
            long crowding / long squeeze risk.

        Negative funding:
            short crowding / short squeeze risk.
        """
        rate = float(current_rate)
        abs_rate = abs(rate)

        if regime == FundingRegime.NEUTRAL or abs_rate <= self.config.neutral_abs_threshold:
            return FundingBias.NEUTRAL

        normalized_percentile = (
            self._clamp(percentile, 0.0, 100.0)
            if percentile is not None
            else None
        )

        if rate > 0:
            if regime == FundingRegime.EXTREME_POSITIVE:
                return FundingBias.SQUEEZE_RISK_LONGS

            if normalized_percentile is not None:
                if normalized_percentile >= self.config.squeeze_percentile_threshold:
                    return FundingBias.SQUEEZE_RISK_LONGS

                if normalized_percentile >= self.config.crowded_percentile_threshold:
                    return FundingBias.OVERCROWDED_LONGS

            return FundingBias.LONG_BIAS

        if rate < 0:
            if regime == FundingRegime.EXTREME_NEGATIVE:
                return FundingBias.SQUEEZE_RISK_SHORTS

            if normalized_percentile is not None:
                if normalized_percentile <= (100.0 - self.config.squeeze_percentile_threshold):
                    return FundingBias.SQUEEZE_RISK_SHORTS

                if normalized_percentile <= (100.0 - self.config.crowded_percentile_threshold):
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
        Обчислює confidence класифікації regime.

        Компоненти:
        - absolute magnitude funding;
        - position in historical distribution;
        - z-score abnormality;
        - sample size quality.
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

        return self._clamp(confidence, 0.0, 1.0)

    def has_regime_changed(
        self,
        previous_state: FundingRegimeState | None,
        new_regime: FundingRegime,
        confidence: float,
    ) -> bool:
        """
        Визначає, чи є новий regime реальною зміною, а не шумом
        на межі порогів.
        """
        if previous_state is None:
            return False

        if previous_state.regime == new_regime:
            return False

        if confidence < self.config.min_confidence_for_change:
            return False

        return True

    # ------------------------------------------------------------------
    # Internal scoring helpers
    # ------------------------------------------------------------------

    def _calc_magnitude_score(self, current_rate: float) -> float:
        abs_rate = abs(float(current_rate))

        if abs_rate <= self.config.neutral_abs_threshold:
            return 0.0

        denominator = max(
            self.config.extreme_abs_threshold - self.config.neutral_abs_threshold,
            1e-12,
        )
        normalized = (abs_rate - self.config.neutral_abs_threshold) / denominator
        return self._clamp(normalized, 0.0, 1.0)

    def _calc_percentile_score(self, percentile: float | None) -> float:
        if percentile is None:
            return 0.0

        normalized_percentile = self._clamp(float(percentile), 0.0, 100.0)
        distance_from_center = abs(normalized_percentile - 50.0) / 50.0
        return self._clamp(distance_from_center, 0.0, 1.0)

    def _calc_zscore_score(self, zscore: float | None) -> float:
        if zscore is None:
            return 0.0

        normalized = abs(float(zscore)) / 3.0
        return self._clamp(normalized, 0.0, 1.0)

    def _calc_sample_quality_score(self, sample_size: int) -> float:
        """
        Оцінка якості статистики за розміром історії.
        """
        sample_size = max(0, int(sample_size))

        if sample_size <= 1:
            return 0.0

        if sample_size >= 100:
            return 1.0

        return self._clamp(sample_size / 100.0, 0.0, 1.0)

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, float(value)))

    @staticmethod
    def _validate_snapshot_statistics_scope(
        *,
        snapshot: FundingSnapshot,
        statistics: FundingStatistics,
    ) -> None:
        """
        FundingAnalyzer має передавати snapshot/statistics одного scope.

        Тут робимо fail-fast, бо змішування різних exchange/market_type/timeframe
        у funding regime дасть некоректний сигнал для strategy.
        """
        if snapshot.key != statistics.key:
            raise ValueError(
                "FundingSnapshot and FundingStatistics scope mismatch: "
                f"snapshot={funding_key_to_dict(snapshot.key)} "
                f"statistics={funding_key_to_dict(statistics.key)}"
            )


__all__ = [
    "FundingRegimeDetectorConfig",
    "FundingRegimeDetector",
]