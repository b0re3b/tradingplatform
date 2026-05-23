from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.logger import get_logger

from analytics.funding.enums import (
    FundingExtremeType,
    FundingRegime,
    FundingTimeframe,
)
from analytics.funding.models import (
    FundingExtremeEvent,
    FundingRegimeState,
    FundingSnapshot,
    FundingStatistics,
    funding_key_to_dict,
)


@dataclass(slots=True)
class FundingExtremesConfig:
    """
    Конфігурація pure detector-а funding extremes.

    FundingExtremesDetector не є runtime-компонентом:
    - не слухає EventBus;
    - не має Scheduler jobs;
    - не читає exchange/data cache напряму;
    - не зберігає історію.

    Runtime orchestration виконує FundingAnalyzer.

    Detector шукає:
    - absolute extremes;
    - percentile extremes;
    - z-score extremes;
    - local/global high/low відносно статистичного вікна.
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

    reversal_risk_min_severity: float = 0.45
    squeeze_risk_min_severity: float = 0.50
    regime_squeeze_min_severity: float = 0.50
    percentile_squeeze_min_severity: float = 0.60

    enable_local_extremes: bool = True
    enable_percentile_extremes: bool = True
    enable_zscore_extremes: bool = True
    enable_absolute_extremes: bool = True

    service_name: str = "funding_extremes_detector"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.neutral_abs_threshold < 0:
            raise ValueError("neutral_abs_threshold must be >= 0")

        if self.elevated_abs_threshold < 0:
            raise ValueError("elevated_abs_threshold must be >= 0")

        if self.extreme_abs_threshold < 0:
            raise ValueError("extreme_abs_threshold must be >= 0")

        if self.neutral_abs_threshold > self.elevated_abs_threshold:
            raise ValueError("neutral_abs_threshold must be <= elevated_abs_threshold")

        if self.elevated_abs_threshold > self.extreme_abs_threshold:
            raise ValueError("elevated_abs_threshold must be <= extreme_abs_threshold")

        if self.elevated_zscore_threshold < 0:
            raise ValueError("elevated_zscore_threshold must be >= 0")

        if self.extreme_zscore_threshold < 0:
            raise ValueError("extreme_zscore_threshold must be >= 0")

        if self.elevated_zscore_threshold > self.extreme_zscore_threshold:
            raise ValueError("elevated_zscore_threshold must be <= extreme_zscore_threshold")

        self._validate_percentile("elevated_percentile_high", self.elevated_percentile_high)
        self._validate_percentile("extreme_percentile_high", self.extreme_percentile_high)
        self._validate_percentile("elevated_percentile_low", self.elevated_percentile_low)
        self._validate_percentile("extreme_percentile_low", self.extreme_percentile_low)

        if self.elevated_percentile_high > self.extreme_percentile_high:
            raise ValueError("elevated_percentile_high must be <= extreme_percentile_high")

        if self.extreme_percentile_low > self.elevated_percentile_low:
            raise ValueError("extreme_percentile_low must be <= elevated_percentile_low")

        if self.min_sample_size <= 0:
            raise ValueError("min_sample_size must be > 0")

        self._validate_ratio("min_severity", self.min_severity)
        self._validate_ratio("reversal_risk_min_severity", self.reversal_risk_min_severity)
        self._validate_ratio("squeeze_risk_min_severity", self.squeeze_risk_min_severity)
        self._validate_ratio("regime_squeeze_min_severity", self.regime_squeeze_min_severity)
        self._validate_ratio(
            "percentile_squeeze_min_severity",
            self.percentile_squeeze_min_severity,
        )

    @staticmethod
    def _validate_percentile(name: str, value: float) -> None:
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"{name} must be in [0, 100]")

    @staticmethod
    def _validate_ratio(name: str, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")


class FundingExtremesDetector:
    """
    Pure detector для екстремальних funding-станів.

    Відповідальність:
    - приймає FundingSnapshot + FundingStatistics;
    - опційно приймає FundingRegimeState;
    - визначає FundingExtremeType;
    - рахує severity;
    - визначає reversal/squeeze risk;
    - повертає FundingExtremeEvent з повним futures scope.

    Correct architecture:
        FundingAnalyzer
            -> FundingExtremesDetector.detect(...)
            -> FundingExtremeEvent | None
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
        config: FundingExtremesConfig | None = None,
    ) -> None:
        self.config = config or FundingExtremesConfig()
        self.config.validate()

        self.logger = get_logger(
            __name__,
            service_name=self.config.service_name,
            event_type="funding_extremes_detector",
        )

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

        Очікується, що snapshot/statistics/regime_state належать одному scope:
            exchange + market_type + symbol + timeframe

        FundingAnalyzer відповідає за:
        - EventBus subscriptions;
        - history/statistics;
        - regime_state;
        - publish output events.
        """
        self._validate_input_scope(
            snapshot=snapshot,
            statistics=statistics,
            regime_state=regime_state,
        )

        if statistics.sample_size < self.config.min_sample_size:
            self.logger.debug(
                "Funding extreme skipped due to low sample size | exchange=%s "
                "market_type=%s symbol=%s timeframe=%s sample_size=%s",
                snapshot.exchange.value,
                snapshot.market_type,
                snapshot.symbol,
                snapshot.timeframe.value,
                statistics.sample_size,
                extra={
                    "scope": funding_key_to_dict(snapshot.key),
                    "exchange_symbol": snapshot.exchange_symbol,
                },
            )
            return None

        tf = (
            timeframe
            or statistics.timeframe
            or snapshot.timeframe
            or self.config.default_timeframe
        )
        regime = (
            regime_state.regime
            if regime_state is not None
            else self._infer_regime(snapshot.funding_rate)
        )

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
                "Funding extreme ignored due to low severity | exchange=%s "
                "market_type=%s symbol=%s timeframe=%s type=%s severity=%.4f",
                snapshot.exchange.value,
                snapshot.market_type,
                snapshot.symbol,
                snapshot.timeframe.value,
                extreme_type.value,
                severity,
                extra={
                    "scope": funding_key_to_dict(snapshot.key),
                    "exchange_symbol": snapshot.exchange_symbol,
                },
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
            "scope": funding_key_to_dict(snapshot.key),
            "exchange_symbol": snapshot.exchange_symbol,
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

        if regime_state is not None:
            metadata["regime"] = regime_state.regime.value
            metadata["regime_confidence"] = regime_state.confidence
            metadata["bias"] = regime_state.bias.value

        if extra_metadata:
            metadata.update(extra_metadata)

        event = FundingExtremeEvent(
            symbol=snapshot.symbol,
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            timeframe=tf,
            exchange_symbol=snapshot.exchange_symbol,
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
            "Funding extreme detected | exchange=%s market_type=%s symbol=%s "
            "timeframe=%s type=%s regime=%s rate=%.8f percentile=%s "
            "zscore=%s severity=%.4f reversal_risk=%s squeeze_risk=%s",
            event.exchange.value,
            event.market_type,
            event.symbol,
            event.timeframe.value,
            event.extreme_type.value,
            event.regime.value,
            event.funding_rate,
            f"{event.percentile:.2f}" if event.percentile is not None else "None",
            f"{event.zscore:.4f}" if event.zscore is not None else "None",
            event.severity,
            event.is_reversal_risk,
            event.is_squeeze_risk,
            extra={
                "scope": funding_key_to_dict(event.key),
                "exchange_symbol": event.exchange_symbol,
            },
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
        1. Local/global extremes;
        2. Percentile extremes;
        3. Z-score extremes;
        4. Absolute extremes.
        """
        current_rate = float(current_rate)
        min_rate = float(min_rate)
        max_rate = float(max_rate)

        if self.config.enable_local_extremes:
            if current_rate >= max_rate and current_rate > self.config.neutral_abs_threshold:
                return FundingExtremeType.GLOBAL_HIGH

            if current_rate <= min_rate and current_rate < -self.config.neutral_abs_threshold:
                return FundingExtremeType.GLOBAL_LOW

        if self.config.enable_percentile_extremes and percentile is not None:
            percentile = self._clamp(float(percentile), 0.0, 100.0)

            if percentile >= self.config.extreme_percentile_high:
                return FundingExtremeType.PERCENTILE_HIGH

            if percentile <= self.config.extreme_percentile_low:
                return FundingExtremeType.PERCENTILE_LOW

        if self.config.enable_zscore_extremes and zscore is not None:
            zscore = float(zscore)

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

        Компоненти:
        - absolute funding magnitude;
        - percentile distance from center;
        - z-score abnormality;
        - bonus за тип extreme.
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

        return self._clamp_0_1(severity)

    def is_reversal_risk(
        self,
        extreme_type: FundingExtremeType,
        regime: FundingRegime,
        severity: float,
    ) -> bool:
        """
        Extreme funding часто означає ризик mean reversion / reversal.
        """
        severity = self._clamp_0_1(severity)

        if severity < self.config.reversal_risk_min_severity:
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
        } and severity >= max(
            self.config.reversal_risk_min_severity,
            self.config.regime_squeeze_min_severity,
        ):
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
        severity = self._clamp_0_1(severity)

        if severity < self.config.squeeze_risk_min_severity:
            return False

        if regime in {
            FundingRegime.EXTREME_POSITIVE,
            FundingRegime.EXTREME_NEGATIVE,
        }:
            return severity >= self.config.regime_squeeze_min_severity

        if extreme_type in {
            FundingExtremeType.GLOBAL_HIGH,
            FundingExtremeType.GLOBAL_LOW,
            FundingExtremeType.PERCENTILE_HIGH,
            FundingExtremeType.PERCENTILE_LOW,
        }:
            return severity >= self.config.percentile_squeeze_min_severity

        return False

    # ------------------------------------------------------------------
    # Helper API
    # ------------------------------------------------------------------

    def is_positive_extreme(
        self,
        event: FundingExtremeEvent | None,
    ) -> bool:
        if event is None:
            return False

        return event.funding_rate > 0 and event.extreme_type in {
            FundingExtremeType.LOCAL_HIGH,
            FundingExtremeType.GLOBAL_HIGH,
            FundingExtremeType.ZSCORE_HIGH,
            FundingExtremeType.PERCENTILE_HIGH,
        }

    def is_negative_extreme(
        self,
        event: FundingExtremeEvent | None,
    ) -> bool:
        if event is None:
            return False

        return event.funding_rate < 0 and event.extreme_type in {
            FundingExtremeType.LOCAL_LOW,
            FundingExtremeType.GLOBAL_LOW,
            FundingExtremeType.ZSCORE_LOW,
            FundingExtremeType.PERCENTILE_LOW,
        }

    def is_high_severity(
        self,
        event: FundingExtremeEvent | None,
        threshold: float = 0.70,
    ) -> bool:
        if event is None:
            return False

        return event.severity >= self._clamp_0_1(threshold)

    def build_summary(
        self,
        event: FundingExtremeEvent,
    ) -> str:
        return (
            f"Funding extreme for "
            f"{event.exchange.value}:{event.market_type}:{event.symbol}:{event.timeframe.value}: "
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

    def _infer_regime(
        self,
        funding_rate: float,
    ) -> FundingRegime:
        funding_rate = float(funding_rate)
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

    def _calc_magnitude_score(
        self,
        current_rate: float,
    ) -> float:
        abs_rate = abs(float(current_rate))

        if abs_rate <= self.config.neutral_abs_threshold:
            return 0.0

        denominator = max(
            self.config.extreme_abs_threshold - self.config.neutral_abs_threshold,
            1e-12,
        )
        normalized = (abs_rate - self.config.neutral_abs_threshold) / denominator
        return self._clamp_0_1(normalized)

    def _calc_percentile_score(
        self,
        percentile: float | None,
    ) -> float:
        if percentile is None:
            return 0.0

        normalized_percentile = self._clamp(float(percentile), 0.0, 100.0)
        distance_from_center = abs(normalized_percentile - 50.0) / 50.0

        if normalized_percentile >= self.config.extreme_percentile_high:
            return 1.0

        if normalized_percentile <= self.config.extreme_percentile_low:
            return 1.0

        if normalized_percentile >= self.config.elevated_percentile_high:
            return max(0.75, self._clamp_0_1(distance_from_center))

        if normalized_percentile <= self.config.elevated_percentile_low:
            return max(0.75, self._clamp_0_1(distance_from_center))

        return self._clamp_0_1(distance_from_center)

    def _calc_zscore_score(
        self,
        zscore: float | None,
    ) -> float:
        if zscore is None:
            return 0.0

        abs_zscore = abs(float(zscore))

        if abs_zscore >= self.config.extreme_zscore_threshold:
            return 1.0

        if abs_zscore >= self.config.elevated_zscore_threshold:
            denominator = max(self.config.extreme_zscore_threshold, 1e-12)
            return max(0.75, self._clamp_0_1(abs_zscore / denominator))

        denominator = max(self.config.extreme_zscore_threshold, 1e-12)
        return self._clamp_0_1(abs_zscore / denominator)

    def _calc_type_bonus(
        self,
        extreme_type: FundingExtremeType,
    ) -> float:
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

    # ------------------------------------------------------------------
    # Validation / helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input_scope(
        *,
        snapshot: FundingSnapshot,
        statistics: FundingStatistics,
        regime_state: FundingRegimeState | None,
    ) -> None:
        """
        Funding extremes не можна рахувати на змішаних scope-ах.

        Всі основні моделі мають належати одному:
            exchange + market_type + symbol + timeframe
        """
        if snapshot.key != statistics.key:
            raise ValueError(
                "FundingSnapshot and FundingStatistics scope mismatch: "
                f"snapshot={funding_key_to_dict(snapshot.key)} "
                f"statistics={funding_key_to_dict(statistics.key)}"
            )

        if regime_state is not None and snapshot.key != regime_state.key:
            raise ValueError(
                "FundingSnapshot and FundingRegimeState scope mismatch: "
                f"snapshot={funding_key_to_dict(snapshot.key)} "
                f"regime_state={funding_key_to_dict(regime_state.key)}"
            )

    @staticmethod
    def _clamp_0_1(
        value: float,
    ) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _clamp(
        value: float,
        lower: float,
        upper: float,
    ) -> float:
        return max(lower, min(upper, float(value)))


__all__ = [
    "FundingExtremesConfig",
    "FundingExtremesDetector",
]