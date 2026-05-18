from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.logger import get_logger

from .enums import (
    FundingFlipType,
    FundingTimeframe,
)
from .models import (
    FundingFlipEvent,
    FundingSnapshot,
    FundingStatistics,
    funding_key_to_dict,
)


@dataclass(slots=True)
class FundingFlipDetectorConfig:
    """
    Конфігурація pure detector-а funding flip.

    Funding flip — це зміна знаку funding, яка може сигналізувати:
    - зміну positioning regime;
    - зміну funding sentiment;
    - squeeze / reversal context.

    FundingFlipDetector не є runtime-компонентом:
    - не слухає EventBus;
    - не має Scheduler jobs;
    - не читає exchange/data cache напряму;
    - не зберігає історію.

    Runtime orchestration виконує FundingAnalyzer.
    """

    default_timeframe: FundingTimeframe = FundingTimeframe.H1

    # Funding біля нуля вважається шумом.
    neutral_abs_threshold: float = 0.00001

    # Мінімальна амплітуда між previous/current funding.
    min_flip_magnitude: float = 0.00002

    # Історична аномальність.
    elevated_percentile_threshold: float = 80.0
    extreme_percentile_threshold: float = 95.0

    elevated_zscore_threshold: float = 1.5
    extreme_zscore_threshold: float = 2.5

    # Мінімальна confidence для валідного flip event.
    min_confidence: float = 0.15

    service_name: str = "funding_flip_detector"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.neutral_abs_threshold < 0:
            raise ValueError("neutral_abs_threshold must be >= 0")

        if self.min_flip_magnitude < 0:
            raise ValueError("min_flip_magnitude must be >= 0")

        self._validate_percentile(
            "elevated_percentile_threshold",
            self.elevated_percentile_threshold,
        )
        self._validate_percentile(
            "extreme_percentile_threshold",
            self.extreme_percentile_threshold,
        )

        if self.elevated_percentile_threshold > self.extreme_percentile_threshold:
            raise ValueError(
                "elevated_percentile_threshold must be <= extreme_percentile_threshold"
            )

        if self.elevated_zscore_threshold < 0:
            raise ValueError("elevated_zscore_threshold must be >= 0")

        if self.extreme_zscore_threshold < 0:
            raise ValueError("extreme_zscore_threshold must be >= 0")

        if self.elevated_zscore_threshold > self.extreme_zscore_threshold:
            raise ValueError(
                "elevated_zscore_threshold must be <= extreme_zscore_threshold"
            )

        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")

    @staticmethod
    def _validate_percentile(name: str, value: float) -> None:
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"{name} must be in [0, 100]")


class FundingFlipDetector:
    """
    Pure detector для зміни знаку funding rate.

    Відповідальність:
    - приймає current/previous FundingSnapshot;
    - опційно приймає FundingStatistics для confidence scoring;
    - визначає FundingFlipType;
    - повертає FundingFlipEvent з повним futures scope.

    Correct architecture:
        FundingAnalyzer
            -> FundingFlipDetector.detect(...)
            -> FundingFlipEvent | None
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
        config: FundingFlipDetectorConfig | None = None,
    ) -> None:
        self.config = config or FundingFlipDetectorConfig()
        self.config.validate()

        self.logger = get_logger(
            __name__,
            service_name=self.config.service_name,
            event_type="funding_flip_detector",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        current_snapshot: FundingSnapshot,
        previous_snapshot: FundingSnapshot | None,
        statistics: FundingStatistics | None = None,
        timeframe: FundingTimeframe | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> FundingFlipEvent | None:
        """
        Визначає, чи відбувся значущий funding flip.

        Очікується, що current_snapshot, previous_snapshot і statistics
        належать одному scope:
            exchange + market_type + symbol + timeframe

        FundingAnalyzer відповідає за:
        - EventBus subscriptions;
        - funding history;
        - previous_snapshot;
        - statistics;
        - publish output events.
        """
        if previous_snapshot is None:
            return None

        self._validate_input_scope(
            current_snapshot=current_snapshot,
            previous_snapshot=previous_snapshot,
            statistics=statistics,
        )

        flip_type = self.detect_flip_type(
            previous_rate=previous_snapshot.funding_rate,
            current_rate=current_snapshot.funding_rate,
        )

        if flip_type == FundingFlipType.NONE:
            return None

        flip_magnitude = abs(
            current_snapshot.funding_rate - previous_snapshot.funding_rate
        )

        if flip_magnitude < self.config.min_flip_magnitude:
            self.logger.debug(
                "Funding flip ignored due to low magnitude | exchange=%s "
                "market_type=%s symbol=%s timeframe=%s previous_rate=%.8f "
                "current_rate=%.8f magnitude=%.8f",
                current_snapshot.exchange.value,
                current_snapshot.market_type,
                current_snapshot.symbol,
                current_snapshot.timeframe.value,
                previous_snapshot.funding_rate,
                current_snapshot.funding_rate,
                flip_magnitude,
                extra={
                    "scope": funding_key_to_dict(current_snapshot.key),
                    "exchange_symbol": current_snapshot.exchange_symbol,
                },
            )
            return None

        confidence = self.calculate_confidence(
            previous_rate=previous_snapshot.funding_rate,
            current_rate=current_snapshot.funding_rate,
            flip_magnitude=flip_magnitude,
            percentile=statistics.percentile if statistics is not None else None,
            zscore=statistics.zscore if statistics is not None else None,
        )

        if confidence < self.config.min_confidence:
            self.logger.debug(
                "Funding flip ignored due to low confidence | exchange=%s "
                "market_type=%s symbol=%s timeframe=%s flip_type=%s "
                "confidence=%.4f",
                current_snapshot.exchange.value,
                current_snapshot.market_type,
                current_snapshot.symbol,
                current_snapshot.timeframe.value,
                flip_type.value,
                confidence,
                extra={
                    "scope": funding_key_to_dict(current_snapshot.key),
                    "exchange_symbol": current_snapshot.exchange_symbol,
                },
            )
            return None

        tf = (
            timeframe
            or (statistics.timeframe if statistics is not None else None)
            or current_snapshot.timeframe
            or self.config.default_timeframe
        )

        metadata: dict[str, Any] = {
            "scope": funding_key_to_dict(current_snapshot.key),
            "exchange_symbol": current_snapshot.exchange_symbol,
            "previous_sign": previous_snapshot.funding_sign,
            "current_sign": current_snapshot.funding_sign,
            "neutral_abs_threshold": self.config.neutral_abs_threshold,
            "min_flip_magnitude": self.config.min_flip_magnitude,
            "previous_event_time": previous_snapshot.event_time.isoformat(),
            "current_event_time": current_snapshot.event_time.isoformat(),
        }

        if statistics is not None:
            metadata.update(
                {
                    "percentile": statistics.percentile,
                    "zscore": statistics.zscore,
                    "sample_size": statistics.sample_size,
                    "mean_rate": statistics.mean_rate,
                    "std_rate": statistics.std_rate,
                    "window_start": (
                        statistics.window_start.isoformat()
                        if statistics.window_start is not None
                        else None
                    ),
                    "window_end": (
                        statistics.window_end.isoformat()
                        if statistics.window_end is not None
                        else None
                    ),
                }
            )

        if extra_metadata:
            metadata.update(extra_metadata)

        event = FundingFlipEvent(
            symbol=current_snapshot.symbol,
            exchange=current_snapshot.exchange,
            market_type=current_snapshot.market_type,
            timeframe=tf,
            exchange_symbol=current_snapshot.exchange_symbol,
            flip_type=flip_type,
            previous_rate=previous_snapshot.funding_rate,
            current_rate=current_snapshot.funding_rate,
            flip_magnitude=flip_magnitude,
            confidence=confidence,
            event_time=current_snapshot.event_time,
            metadata=metadata,
        )

        self.logger.debug(
            "Funding flip detected | exchange=%s market_type=%s symbol=%s "
            "timeframe=%s flip_type=%s previous_rate=%.8f current_rate=%.8f "
            "magnitude=%.8f confidence=%.4f",
            event.exchange.value,
            event.market_type,
            event.symbol,
            event.timeframe.value,
            event.flip_type.value,
            event.previous_rate,
            event.current_rate,
            event.flip_magnitude,
            event.confidence,
            extra={
                "scope": funding_key_to_dict(event.key),
                "exchange_symbol": event.exchange_symbol,
            },
        )

        return event

    def detect_flip_type(
        self,
        previous_rate: float,
        current_rate: float,
    ) -> FundingFlipType:
        """
        Визначає напрям зміни знаку funding.

        Логіка:
        - якщо обидва значення в neutral zone -> NONE;
        - якщо знаки не змінились -> NONE;
        - якщо negative -> positive -> NEGATIVE_TO_POSITIVE;
        - якщо positive -> negative -> POSITIVE_TO_NEGATIVE.
        """
        previous_rate = float(previous_rate)
        current_rate = float(current_rate)

        prev_sign = self._sign_with_neutral_zone(previous_rate)
        curr_sign = self._sign_with_neutral_zone(current_rate)

        if prev_sign == 0 and curr_sign == 0:
            return FundingFlipType.NONE

        if prev_sign == curr_sign:
            return FundingFlipType.NONE

        if prev_sign < 0 and curr_sign > 0:
            return FundingFlipType.NEGATIVE_TO_POSITIVE

        if prev_sign > 0 and curr_sign < 0:
            return FundingFlipType.POSITIVE_TO_NEGATIVE

        # Дозволяємо сценарії через нейтральну зону,
        # якщо фактичний previous/current funding уже по різні боки threshold.
        if (
            previous_rate < -self.config.neutral_abs_threshold
            and current_rate > self.config.neutral_abs_threshold
        ):
            return FundingFlipType.NEGATIVE_TO_POSITIVE

        if (
            previous_rate > self.config.neutral_abs_threshold
            and current_rate < -self.config.neutral_abs_threshold
        ):
            return FundingFlipType.POSITIVE_TO_NEGATIVE

        return FundingFlipType.NONE

    def calculate_confidence(
        self,
        previous_rate: float,
        current_rate: float,
        flip_magnitude: float,
        percentile: float | None,
        zscore: float | None,
    ) -> float:
        """
        Оцінка confidence для funding flip.

        Компоненти:
        - magnitude самого flip;
        - відстань current funding від neutral zone;
        - historical percentile abnormality;
        - z-score abnormality.
        """
        magnitude_score = self._calc_magnitude_score(flip_magnitude)
        current_distance_score = self._calc_current_distance_score(current_rate)
        percentile_score = self._calc_percentile_score(percentile)
        zscore_score = self._calc_zscore_score(zscore)

        confidence = (
            0.45 * magnitude_score
            + 0.25 * current_distance_score
            + 0.15 * percentile_score
            + 0.15 * zscore_score
        )

        return self._clamp_0_1(confidence)

    # ------------------------------------------------------------------
    # Helper API
    # ------------------------------------------------------------------

    def is_bullish_flip(
        self,
        event: FundingFlipEvent | None,
    ) -> bool:
        """
        Funding negative -> positive.

        Сам по собі це не automatic bullish trade signal, але часто означає
        зміщення positioning у сторону longs.
        """
        return (
            event is not None
            and event.flip_type == FundingFlipType.NEGATIVE_TO_POSITIVE
        )

    def is_bearish_flip(
        self,
        event: FundingFlipEvent | None,
    ) -> bool:
        """
        Funding positive -> negative.
        """
        return (
            event is not None
            and event.flip_type == FundingFlipType.POSITIVE_TO_NEGATIVE
        )

    def build_summary(
        self,
        event: FundingFlipEvent,
    ) -> str:
        return (
            f"Funding flip for "
            f"{event.exchange.value}:{event.market_type}:{event.symbol}:{event.timeframe.value}: "
            f"{event.flip_type.value}, "
            f"previous_rate={event.previous_rate:.8f}, "
            f"current_rate={event.current_rate:.8f}, "
            f"magnitude={event.flip_magnitude:.8f}, "
            f"confidence={event.confidence:.4f}"
        )

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------

    def _sign_with_neutral_zone(
        self,
        value: float,
    ) -> int:
        """
        Повертає:
        -1 для негативного funding;
         0 для neutral/noise zone;
         1 для позитивного funding.
        """
        value = float(value)

        if value > self.config.neutral_abs_threshold:
            return 1

        if value < -self.config.neutral_abs_threshold:
            return -1

        return 0

    def _calc_magnitude_score(
        self,
        flip_magnitude: float,
    ) -> float:
        """
        Нормалізація сили flip.
        """
        flip_magnitude = abs(float(flip_magnitude))

        if flip_magnitude <= self.config.min_flip_magnitude:
            return 0.0

        denominator = max(
            self.config.min_flip_magnitude * 10.0,
            1e-12,
        )
        normalized = flip_magnitude / denominator
        return self._clamp_0_1(normalized)

    def _calc_current_distance_score(
        self,
        current_rate: float,
    ) -> float:
        """
        Наскільки новий funding відійшов від neutral zone.
        Якщо current_rate майже біля 0 — flip слабший.
        """
        abs_rate = abs(float(current_rate))

        if abs_rate <= self.config.neutral_abs_threshold:
            return 0.0

        denominator = max(
            self.config.neutral_abs_threshold * 10.0,
            1e-12,
        )
        normalized = abs_rate / denominator
        return self._clamp_0_1(normalized)

    def _calc_percentile_score(
        self,
        percentile: float | None,
    ) -> float:
        if percentile is None:
            return 0.0

        normalized_percentile = self._clamp(float(percentile), 0.0, 100.0)
        distance_from_center = abs(normalized_percentile - 50.0) / 50.0

        if normalized_percentile >= self.config.extreme_percentile_threshold:
            return 1.0

        if normalized_percentile <= 100.0 - self.config.extreme_percentile_threshold:
            return 1.0

        if normalized_percentile >= self.config.elevated_percentile_threshold:
            return max(0.75, self._clamp_0_1(distance_from_center))

        if normalized_percentile <= 100.0 - self.config.elevated_percentile_threshold:
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

    # ------------------------------------------------------------------
    # Validation / helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input_scope(
        *,
        current_snapshot: FundingSnapshot,
        previous_snapshot: FundingSnapshot,
        statistics: FundingStatistics | None,
    ) -> None:
        """
        Funding flip не можна рахувати на змішаних scope-ах.

        Всі моделі мають належати одному:
            exchange + market_type + symbol + timeframe
        """
        if current_snapshot.key != previous_snapshot.key:
            raise ValueError(
                "current_snapshot and previous_snapshot scope mismatch: "
                f"current={funding_key_to_dict(current_snapshot.key)} "
                f"previous={funding_key_to_dict(previous_snapshot.key)}"
            )

        if statistics is not None and current_snapshot.key != statistics.key:
            raise ValueError(
                "current_snapshot and statistics scope mismatch: "
                f"current={funding_key_to_dict(current_snapshot.key)} "
                f"statistics={funding_key_to_dict(statistics.key)}"
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
    "FundingFlipDetectorConfig",
    "FundingFlipDetector",
]