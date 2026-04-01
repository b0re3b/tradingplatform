from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.logger import get_logger

from .enums import (
    FundingDivergenceType,
    FundingTimeframe,
)
from .models import (
    FundingDivergenceEvent,
    FundingSnapshot,
    FundingStatistics,
)


@dataclass(slots=True)
class FundingDivergenceConfig:
    """
    Конфігурація детектора funding divergence.

    Дивергенція тут означає, що funding-позиціонування ринку
    не узгоджується з іншими компонентами:
    - ціною
    - open interest
    - CVD
    - ліквідаціями
    """

    default_timeframe: FundingTimeframe = FundingTimeframe.H1

    neutral_abs_threshold: float = 0.00001

    # Мінімальні зміни для значущої дивергенції
    min_price_change_pct: float = 0.0020
    min_oi_change_pct: float = 0.0050
    min_cvd_change: float = 0.0
    min_liquidations_value: float = 0.0

    # Funding має бути достатньо вираженим
    min_funding_for_divergence: float = 0.00003

    # Статистичні підсилювачі
    elevated_percentile_threshold: float = 80.0
    extreme_percentile_threshold: float = 95.0
    elevated_zscore_threshold: float = 1.5
    extreme_zscore_threshold: float = 2.5

    # Мінімальна впевненість
    min_confidence: float = 0.20

    # Увімкнення окремих типів дивергенцій
    enable_price_funding_divergence: bool = True
    enable_oi_funding_divergence: bool = True
    enable_cvd_funding_divergence: bool = True
    enable_liquidation_funding_divergence: bool = True


class FundingDivergenceDetector:
    """
    Детектор дивергенцій між funding і price/OI/CVD/liquidations.

    Клас:
    - не зберігає історію
    - працює з уже підготовленими змінами/метриками
    - повертає FundingDivergenceEvent або None
    """

    def __init__(
        self,
        config: FundingDivergenceConfig | None = None,
    ) -> None:
        self.config = config or FundingDivergenceConfig()
        self.logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        snapshot: FundingSnapshot,
        statistics: FundingStatistics | None = None,
        price_change_pct: float | None = None,
        oi_change_pct: float | None = None,
        cvd_change: float | None = None,
        long_liquidations: float | None = None,
        short_liquidations: float | None = None,
        timeframe: FundingTimeframe | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> FundingDivergenceEvent | None:
        """
        Пошук найбільш релевантної funding-divergence події.

        Parameters
        ----------
        snapshot:
            Поточний funding snapshot.
        statistics:
            Funding statistics для percentile/zscore context.
        price_change_pct:
            Відносна зміна ціни за вікно аналізу.
        oi_change_pct:
            Відносна зміна open interest за вікно аналізу.
        cvd_change:
            Зміна cumulative volume delta.
        long_liquidations:
            Обсяг long liquidations.
        short_liquidations:
            Обсяг short liquidations.
        timeframe:
            Таймфрейм аналізу.
        extra_metadata:
            Додаткові metadata.

        Returns
        -------
        FundingDivergenceEvent | None
        """
        if abs(snapshot.funding_rate) < self.config.min_funding_for_divergence:
            return None

        divergence_type = self.detect_divergence_type(
            funding_rate=snapshot.funding_rate,
            price_change_pct=price_change_pct,
            oi_change_pct=oi_change_pct,
            cvd_change=cvd_change,
            long_liquidations=long_liquidations,
            short_liquidations=short_liquidations,
        )

        if divergence_type == FundingDivergenceType.NONE:
            return None

        confidence = self.calculate_confidence(
            funding_rate=snapshot.funding_rate,
            divergence_type=divergence_type,
            percentile=statistics.percentile if statistics is not None else None,
            zscore=statistics.zscore if statistics is not None else None,
            price_change_pct=price_change_pct,
            oi_change_pct=oi_change_pct,
            cvd_change=cvd_change,
            long_liquidations=long_liquidations,
            short_liquidations=short_liquidations,
        )

        if confidence < self.config.min_confidence:
            self.logger.debug(
                "Funding divergence ignored due to low confidence: "
                "symbol=%s exchange=%s type=%s confidence=%.4f",
                snapshot.symbol,
                snapshot.exchange.value,
                divergence_type.value,
                confidence,
            )
            return None

        tf = timeframe or (
            statistics.timeframe if statistics is not None else self.config.default_timeframe
        )

        metadata: dict[str, Any] = {
            "funding_sign": snapshot.funding_sign,
            "funding_abs": abs(snapshot.funding_rate),
            "basis": snapshot.basis,
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

        event = FundingDivergenceEvent(
            symbol=snapshot.symbol,
            exchange=snapshot.exchange,
            timeframe=tf,
            divergence_type=divergence_type,
            funding_rate=snapshot.funding_rate,
            price_change_pct=price_change_pct,
            oi_change_pct=oi_change_pct,
            cvd_change=cvd_change,
            long_liquidations=long_liquidations,
            short_liquidations=short_liquidations,
            confidence=confidence,
            event_time=snapshot.event_time,
            metadata=metadata,
        )

        self.logger.debug(
            "Funding divergence detected: symbol=%s exchange=%s type=%s "
            "funding_rate=%.8f price_change_pct=%s oi_change_pct=%s cvd_change=%s "
            "long_liquidations=%s short_liquidations=%s confidence=%.4f",
            event.symbol,
            event.exchange.value,
            event.divergence_type.value,
            event.funding_rate,
            f"{event.price_change_pct:.6f}" if event.price_change_pct is not None else "None",
            f"{event.oi_change_pct:.6f}" if event.oi_change_pct is not None else "None",
            f"{event.cvd_change:.6f}" if event.cvd_change is not None else "None",
            f"{event.long_liquidations:.6f}" if event.long_liquidations is not None else "None",
            f"{event.short_liquidations:.6f}" if event.short_liquidations is not None else "None",
            event.confidence,
        )

        return event

    def detect_divergence_type(
        self,
        funding_rate: float,
        price_change_pct: float | None,
        oi_change_pct: float | None,
        cvd_change: float | None,
        long_liquidations: float | None,
        short_liquidations: float | None,
    ) -> FundingDivergenceType:
        """
        Визначення типу дивергенції.

        Пріоритет:
        1. liquidation divergence
        2. price/funding divergence
        3. oi/funding divergence
        4. cvd/funding divergence
        """
        # --------------------------------------------------------------
        # 1. Liquidations vs funding
        # --------------------------------------------------------------
        if self.config.enable_liquidation_funding_divergence:
            if (
                funding_rate > self.config.min_funding_for_divergence
                and long_liquidations is not None
                and long_liquidations > self.config.min_liquidations_value
            ):
                return FundingDivergenceType.LIQUIDATIONS_LONGS_WITH_POSITIVE_FUNDING

            if (
                funding_rate < -self.config.min_funding_for_divergence
                and short_liquidations is not None
                and short_liquidations > self.config.min_liquidations_value
            ):
                return FundingDivergenceType.LIQUIDATIONS_SHORTS_WITH_NEGATIVE_FUNDING

        # --------------------------------------------------------------
        # 2. Price vs funding
        # --------------------------------------------------------------
        if self.config.enable_price_funding_divergence and price_change_pct is not None:
            if (
                price_change_pct >= self.config.min_price_change_pct
                and funding_rate < -self.config.min_funding_for_divergence
            ):
                return FundingDivergenceType.PRICE_UP_FUNDING_DOWN

            if (
                price_change_pct <= -self.config.min_price_change_pct
                and funding_rate > self.config.min_funding_for_divergence
            ):
                return FundingDivergenceType.PRICE_DOWN_FUNDING_UP

        # --------------------------------------------------------------
        # 3. OI vs funding
        # --------------------------------------------------------------
        if self.config.enable_oi_funding_divergence and oi_change_pct is not None:
            if (
                oi_change_pct >= self.config.min_oi_change_pct
                and funding_rate < -self.config.min_funding_for_divergence
            ):
                return FundingDivergenceType.OI_UP_FUNDING_DOWN

            # Funding позитивний, OI росте, але ціна стоїть або слабка
            if (
                oi_change_pct >= self.config.min_oi_change_pct
                and funding_rate > self.config.min_funding_for_divergence
                and price_change_pct is not None
                and abs(price_change_pct) < self.config.min_price_change_pct
            ):
                return FundingDivergenceType.OI_UP_FUNDING_UP_PRICE_STALLED

        # --------------------------------------------------------------
        # 4. CVD vs funding
        # --------------------------------------------------------------
        if self.config.enable_cvd_funding_divergence and cvd_change is not None:
            if (
                cvd_change > self.config.min_cvd_change
                and funding_rate < -self.config.min_funding_for_divergence
            ):
                return FundingDivergenceType.CVD_UP_FUNDING_DOWN

            if (
                cvd_change < -self.config.min_cvd_change
                and funding_rate > self.config.min_funding_for_divergence
            ):
                return FundingDivergenceType.CVD_DOWN_FUNDING_UP

        return FundingDivergenceType.NONE

    def calculate_confidence(
        self,
        funding_rate: float,
        divergence_type: FundingDivergenceType,
        percentile: float | None,
        zscore: float | None,
        price_change_pct: float | None,
        oi_change_pct: float | None,
        cvd_change: float | None,
        long_liquidations: float | None,
        short_liquidations: float | None,
    ) -> float:
        """
        Оцінка впевненості у дивергенції.

        Компоненти:
        - сила funding
        - сила протиріччя в зовнішній метриці
        - percentile/zscore abnormality
        """
        funding_score = self._calc_funding_score(funding_rate)
        percentile_score = self._calc_percentile_score(percentile)
        zscore_score = self._calc_zscore_score(zscore)

        divergence_strength = self._calc_divergence_strength(
            divergence_type=divergence_type,
            price_change_pct=price_change_pct,
            oi_change_pct=oi_change_pct,
            cvd_change=cvd_change,
            long_liquidations=long_liquidations,
            short_liquidations=short_liquidations,
        )

        confidence = (
            0.35 * funding_score
            + 0.35 * divergence_strength
            + 0.15 * percentile_score
            + 0.15 * zscore_score
        )

        return max(0.0, min(1.0, confidence))

    # ------------------------------------------------------------------
    # Helper API
    # ------------------------------------------------------------------

    def is_bullish_divergence(self, event: FundingDivergenceEvent | None) -> bool:
        """
        Буліш-контекст:
        - price up while funding negative
        - CVD up while funding negative
        - shorts under pressure
        """
        if event is None:
            return False

        return event.divergence_type in {
            FundingDivergenceType.PRICE_UP_FUNDING_DOWN,
            FundingDivergenceType.CVD_UP_FUNDING_DOWN,
            FundingDivergenceType.OI_UP_FUNDING_DOWN,
            FundingDivergenceType.LIQUIDATIONS_SHORTS_WITH_NEGATIVE_FUNDING,
        }

    def is_bearish_divergence(self, event: FundingDivergenceEvent | None) -> bool:
        """
        Беаріш-контекст:
        - price down while funding positive
        - CVD down while funding positive
        - longs under pressure
        """
        if event is None:
            return False

        return event.divergence_type in {
            FundingDivergenceType.PRICE_DOWN_FUNDING_UP,
            FundingDivergenceType.CVD_DOWN_FUNDING_UP,
            FundingDivergenceType.OI_UP_FUNDING_UP_PRICE_STALLED,
            FundingDivergenceType.LIQUIDATIONS_LONGS_WITH_POSITIVE_FUNDING,
        }

    def build_summary(self, event: FundingDivergenceEvent) -> str:
        return (
            f"Funding divergence for {event.symbol}: "
            f"type={event.divergence_type.value}, "
            f"funding_rate={event.funding_rate:.8f}, "
            f"price_change_pct={event.price_change_pct if event.price_change_pct is not None else 'None'}, "
            f"oi_change_pct={event.oi_change_pct if event.oi_change_pct is not None else 'None'}, "
            f"cvd_change={event.cvd_change if event.cvd_change is not None else 'None'}, "
            f"confidence={event.confidence:.4f}"
        )

    # ------------------------------------------------------------------
    # Internal scoring helpers
    # ------------------------------------------------------------------

    def _calc_funding_score(self, funding_rate: float) -> float:
        abs_rate = abs(funding_rate)

        if abs_rate <= self.config.neutral_abs_threshold:
            return 0.0

        denominator = max(
            self.config.min_funding_for_divergence,
            1e-12,
        )
        normalized = abs_rate / denominator
        return max(0.0, min(1.0, normalized / 3.0))

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

    def _calc_divergence_strength(
        self,
        divergence_type: FundingDivergenceType,
        price_change_pct: float | None,
        oi_change_pct: float | None,
        cvd_change: float | None,
        long_liquidations: float | None,
        short_liquidations: float | None,
    ) -> float:
        if divergence_type in {
            FundingDivergenceType.PRICE_UP_FUNDING_DOWN,
            FundingDivergenceType.PRICE_DOWN_FUNDING_UP,
        }:
            return self._normalize_by_threshold(
                value=abs(price_change_pct or 0.0),
                threshold=self.config.min_price_change_pct,
                scale=3.0,
            )

        if divergence_type in {
            FundingDivergenceType.OI_UP_FUNDING_DOWN,
            FundingDivergenceType.OI_UP_FUNDING_UP_PRICE_STALLED,
        }:
            return self._normalize_by_threshold(
                value=abs(oi_change_pct or 0.0),
                threshold=self.config.min_oi_change_pct,
                scale=3.0,
            )

        if divergence_type in {
            FundingDivergenceType.CVD_UP_FUNDING_DOWN,
            FundingDivergenceType.CVD_DOWN_FUNDING_UP,
        }:
            return self._normalize_cvd(cvd_change)

        if divergence_type == FundingDivergenceType.LIQUIDATIONS_LONGS_WITH_POSITIVE_FUNDING:
            return self._normalize_liquidations(long_liquidations)

        if divergence_type == FundingDivergenceType.LIQUIDATIONS_SHORTS_WITH_NEGATIVE_FUNDING:
            return self._normalize_liquidations(short_liquidations)

        return 0.0

    def _normalize_by_threshold(
        self,
        value: float,
        threshold: float,
        scale: float = 3.0,
    ) -> float:
        if threshold <= 0:
            return 0.0

        normalized = value / threshold
        return max(0.0, min(1.0, normalized / scale))

    def _normalize_cvd(self, cvd_change: float | None) -> float:
        if cvd_change is None:
            return 0.0

        value = abs(cvd_change)
        if value <= self.config.min_cvd_change:
            return 0.0

        # Нормалізація м'яка, бо шкала CVD сильно залежить від ринку/символа.
        # Далі в analyzer можна додати symbol-aware scaling.
        return max(0.0, min(1.0, value / max(abs(value), 1.0)))

    def _normalize_liquidations(self, liquidations: float | None) -> float:
        if liquidations is None:
            return 0.0

        value = abs(liquidations)
        if value <= self.config.min_liquidations_value:
            return 0.0

        # Логарифмічна-like м'яка нормалізація без math.log,
        # щоб не ускладнювати і не прив'язуватись до шкали джерела.
        return max(0.0, min(1.0, value / (value + 1.0)))