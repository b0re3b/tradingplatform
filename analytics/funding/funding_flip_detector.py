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
)


@dataclass(slots=True)
class FundingFlipDetectorConfig:
    """
    Конфігурація детектора funding flip.

    Funding flip — це зміна знаку funding, яка може сигналізувати:
    - зміну ринкового сентименту
    - перехід у новий positioning regime
    - потенційний squeeze / reversal context
    """

    default_timeframe: FundingTimeframe = FundingTimeframe.H1

    # Значення funding біля нуля вважаються шумом
    neutral_abs_threshold: float = 0.00001

    # Мінімальна амплітуда між previous/current funding,
    # щоб flip вважався реальним, а не випадковим шумом.
    min_flip_magnitude: float = 0.00002

    # Підсилення довіри, якщо funding вже в історичному екстремумі
    elevated_percentile_threshold: float = 80.0
    extreme_percentile_threshold: float = 95.0

    elevated_zscore_threshold: float = 1.5
    extreme_zscore_threshold: float = 2.5

    # Мінімальна confidence для валідного flip event
    min_confidence: float = 0.15


class FundingFlipDetector:
    """
    Детектор зміни знаку funding rate.

    Клас:
    - не працює з EventBus напряму
    - не зберігає історію
    - приймає current/previous snapshot
    - повертає FundingFlipEvent або None

    Це чистий analytics detector, який має викликатися з FundingAnalyzer.
    """

    def __init__(
        self,
        config: FundingFlipDetectorConfig | None = None,
    ) -> None:
        self.config = config or FundingFlipDetectorConfig()
        self.logger = get_logger(__name__)

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

        Parameters
        ----------
        current_snapshot:
            Поточний funding snapshot.
        previous_snapshot:
            Попередній funding snapshot.
        statistics:
            Поточна статистика funding. Необов'язкова, але корисна
            для кращого confidence scoring.
        timeframe:
            Таймфрейм аналізу.
        extra_metadata:
            Додаткові metadata-поля.

        Returns
        -------
        FundingFlipEvent | None
        """
        if previous_snapshot is None:
            return None

        flip_type = self.detect_flip_type(
            previous_rate=previous_snapshot.funding_rate,
            current_rate=current_snapshot.funding_rate,
        )

        if flip_type == FundingFlipType.NONE:
            return None

        flip_magnitude = abs(current_snapshot.funding_rate - previous_snapshot.funding_rate)

        if flip_magnitude < self.config.min_flip_magnitude:
            self.logger.debug(
                "Funding flip ignored due to low magnitude: symbol=%s exchange=%s "
                "previous_rate=%.8f current_rate=%.8f magnitude=%.8f",
                current_snapshot.symbol,
                current_snapshot.exchange.value,
                previous_snapshot.funding_rate,
                current_snapshot.funding_rate,
                flip_magnitude,
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
                "Funding flip ignored due to low confidence: symbol=%s exchange=%s "
                "flip_type=%s confidence=%.4f",
                current_snapshot.symbol,
                current_snapshot.exchange.value,
                flip_type.value,
                confidence,
            )
            return None

        tf = timeframe or (
            statistics.timeframe if statistics is not None else self.config.default_timeframe
        )

        metadata: dict[str, Any] = {
            "previous_sign": previous_snapshot.funding_sign,
            "current_sign": current_snapshot.funding_sign,
            "neutral_abs_threshold": self.config.neutral_abs_threshold,
            "min_flip_magnitude": self.config.min_flip_magnitude,
        }

        if statistics is not None:
            metadata.update(
                {
                    "percentile": statistics.percentile,
                    "zscore": statistics.zscore,
                    "sample_size": statistics.sample_size,
                    "mean_rate": statistics.mean_rate,
                    "std_rate": statistics.std_rate,
                }
            )

        if extra_metadata:
            metadata.update(extra_metadata)

        event = FundingFlipEvent(
            symbol=current_snapshot.symbol,
            exchange=current_snapshot.exchange,
            timeframe=tf,
            flip_type=flip_type,
            previous_rate=previous_snapshot.funding_rate,
            current_rate=current_snapshot.funding_rate,
            flip_magnitude=flip_magnitude,
            confidence=confidence,
            event_time=current_snapshot.event_time,
            metadata=metadata,
        )

        self.logger.debug(
            "Funding flip detected: symbol=%s exchange=%s flip_type=%s "
            "previous_rate=%.8f current_rate=%.8f magnitude=%.8f confidence=%.4f",
            event.symbol,
            event.exchange.value,
            event.flip_type.value,
            event.previous_rate,
            event.current_rate,
            event.flip_magnitude,
            event.confidence,
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
        - якщо обидва значення занадто близькі до нуля -> NONE
        - якщо знаки не змінились -> NONE
        - якщо negative -> positive -> NEGATIVE_TO_POSITIVE
        - якщо positive -> negative -> POSITIVE_TO_NEGATIVE
        """
        prev_sign = self._sign_with_neutral_zone(previous_rate)
        curr_sign = self._sign_with_neutral_zone(current_rate)

        if prev_sign == 0 and curr_sign == 0:
            return FundingFlipType.NONE

        if prev_sign == curr_sign:
            return FundingFlipType.NONE

        # Через нейтральну зону теж рахуємо flip, якщо реальний перехід є
        if prev_sign < 0 and curr_sign > 0:
            return FundingFlipType.NEGATIVE_TO_POSITIVE

        if prev_sign > 0 and curr_sign < 0:
            return FundingFlipType.POSITIVE_TO_NEGATIVE

        # Дозволяємо сценарії на кшталт:
        # -0.00002 -> 0.0 -> +0.00003
        # або
        # +0.00002 -> 0.0 -> -0.00003
        if previous_rate < -self.config.neutral_abs_threshold and current_rate > self.config.neutral_abs_threshold:
            return FundingFlipType.NEGATIVE_TO_POSITIVE

        if previous_rate > self.config.neutral_abs_threshold and current_rate < -self.config.neutral_abs_threshold:
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
        Оцінка впевненості для funding flip.

        Компоненти:
        - наскільки великий сам flip
        - наскільки current funding віддалений від нейтральної зони
        - чи є історична аномальність через percentile/zscore
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

        return max(0.0, min(1.0, confidence))

    # ------------------------------------------------------------------
    # Helper API
    # ------------------------------------------------------------------

    def is_bullish_flip(self, event: FundingFlipEvent | None) -> bool:
        """
        Funding negative -> positive.
        Сам по собі не є bullish signal автоматично,
        але часто означає зміщення позиціонування в сторону longs.
        """
        return event is not None and event.flip_type == FundingFlipType.NEGATIVE_TO_POSITIVE

    def is_bearish_flip(self, event: FundingFlipEvent | None) -> bool:
        """
        Funding positive -> negative.
        """
        return event is not None and event.flip_type == FundingFlipType.POSITIVE_TO_NEGATIVE

    def build_summary(self, event: FundingFlipEvent) -> str:
        return (
            f"Funding flip for {event.symbol}: "
            f"{event.flip_type.value}, "
            f"previous_rate={event.previous_rate:.8f}, "
            f"current_rate={event.current_rate:.8f}, "
            f"magnitude={event.flip_magnitude:.8f}, "
            f"confidence={event.confidence:.4f}"
        )

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------

    def _sign_with_neutral_zone(self, value: float) -> int:
        """
        Повертає:
        -1 для негативного funding
         0 для neutral/noise zone
         1 для позитивного funding
        """
        if value > self.config.neutral_abs_threshold:
            return 1
        if value < -self.config.neutral_abs_threshold:
            return -1
        return 0

    def _calc_magnitude_score(self, flip_magnitude: float) -> float:
        """
        Нормалізація сили flip.
        """
        if flip_magnitude <= self.config.min_flip_magnitude:
            return 0.0

        denominator = max(
            self.config.extreme_percentile_threshold * self.config.min_flip_magnitude,
            1e-12,
        )
        normalized = flip_magnitude / denominator

        # Більш практичний кап:
        return max(0.0, min(1.0, normalized * 10.0))

    def _calc_current_distance_score(self, current_rate: float) -> float:
        """
        Наскільки новий funding уже відійшов від нейтральної зони.
        Якщо current_rate майже біля 0 — flip слабший.
        """
        abs_rate = abs(current_rate)

        if abs_rate <= self.config.neutral_abs_threshold:
            return 0.0

        denominator = max(
            self.config.extreme_percentile_threshold * self.config.neutral_abs_threshold,
            1e-12,
        )
        normalized = abs_rate / denominator

        return max(0.0, min(1.0, normalized * 5.0))

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