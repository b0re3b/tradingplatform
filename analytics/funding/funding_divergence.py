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
    funding_key_to_dict,
)


@dataclass(slots=True)
class FundingDivergenceConfig:
    """
    Конфігурація pure detector-а funding divergence.

    FundingDivergenceDetector не є runtime-компонентом:
    - не слухає EventBus;
    - не має Scheduler jobs;
    - не читає exchange/data cache напряму;
    - не зберігає історію.

    Runtime orchestration виконує FundingAnalyzer.

    Дивергенція означає, що funding-позиціонування ринку
    не узгоджується з іншими компонентами:
    - price;
    - open interest;
    - CVD;
    - liquidations.
    """

    default_timeframe: FundingTimeframe = FundingTimeframe.H1

    neutral_abs_threshold: float = 0.00001

    # Мінімальні зміни для значущої дивергенції.
    min_price_change_pct: float = 0.0020
    min_oi_change_pct: float = 0.0050
    min_cvd_change: float = 0.0
    min_liquidations_value: float = 0.0

    # Funding має бути достатньо вираженим.
    min_funding_for_divergence: float = 0.00003

    # Статистичні підсилювачі.
    elevated_percentile_threshold: float = 80.0
    extreme_percentile_threshold: float = 95.0
    elevated_zscore_threshold: float = 1.5
    extreme_zscore_threshold: float = 2.5

    # Мінімальна впевненість.
    min_confidence: float = 0.20

    # Увімкнення окремих типів дивергенцій.
    enable_price_funding_divergence: bool = True
    enable_oi_funding_divergence: bool = True
    enable_cvd_funding_divergence: bool = True
    enable_liquidation_funding_divergence: bool = True

    service_name: str = "funding_divergence_detector"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.neutral_abs_threshold < 0:
            raise ValueError("neutral_abs_threshold must be >= 0")

        if self.min_price_change_pct < 0:
            raise ValueError("min_price_change_pct must be >= 0")

        if self.min_oi_change_pct < 0:
            raise ValueError("min_oi_change_pct must be >= 0")

        if self.min_cvd_change < 0:
            raise ValueError("min_cvd_change must be >= 0")

        if self.min_liquidations_value < 0:
            raise ValueError("min_liquidations_value must be >= 0")

        if self.min_funding_for_divergence < 0:
            raise ValueError("min_funding_for_divergence must be >= 0")

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


class FundingDivergenceDetector:
    """
    Pure detector для дивергенцій між funding і price/OI/CVD/liquidations.

    Відповідальність:
    - приймає FundingSnapshot;
    - опційно приймає FundingStatistics;
    - приймає вже підготовлені context metrics:
      price_change_pct / oi_change_pct / cvd_change / liquidations;
    - визначає FundingDivergenceType;
    - рахує confidence;
    - повертає FundingDivergenceEvent з повним futures scope.

    Correct architecture:
        FundingAnalyzer
            -> FundingDivergenceDetector.detect(...)
            -> FundingDivergenceEvent | None
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
        config: FundingDivergenceConfig | None = None,
    ) -> None:
        self.config = config or FundingDivergenceConfig()
        self.config.validate()

        self.logger = get_logger(
            __name__,
            service_name=self.config.service_name,
            event_type="funding_divergence_detector",
        )

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
        Виявляє funding divergence event.

        Очікується, що snapshot/statistics належать одному scope:
            exchange + market_type + symbol + timeframe

        FundingAnalyzer відповідає за:
        - EventBus subscriptions;
        - market context cache;
        - розрахунок price_change_pct / oi_change_pct / cvd_change;
        - агрегацію liquidation context;
        - publish output events.
        """
        self._validate_input_scope(
            snapshot=snapshot,
            statistics=statistics,
        )

        if abs(snapshot.funding_rate) < self.config.min_funding_for_divergence:
            return None

        normalized_price_change_pct = self._to_optional_float(price_change_pct)
        normalized_oi_change_pct = self._to_optional_float(oi_change_pct)
        normalized_cvd_change = self._to_optional_float(cvd_change)
        normalized_long_liquidations = self._non_negative_optional_float(long_liquidations)
        normalized_short_liquidations = self._non_negative_optional_float(short_liquidations)

        divergence_type = self.detect_divergence_type(
            funding_rate=snapshot.funding_rate,
            price_change_pct=normalized_price_change_pct,
            oi_change_pct=normalized_oi_change_pct,
            cvd_change=normalized_cvd_change,
            long_liquidations=normalized_long_liquidations,
            short_liquidations=normalized_short_liquidations,
        )

        if divergence_type == FundingDivergenceType.NONE:
            return None

        confidence = self.calculate_confidence(
            funding_rate=snapshot.funding_rate,
            divergence_type=divergence_type,
            percentile=statistics.percentile if statistics is not None else None,
            zscore=statistics.zscore if statistics is not None else None,
            price_change_pct=normalized_price_change_pct,
            oi_change_pct=normalized_oi_change_pct,
            cvd_change=normalized_cvd_change,
            long_liquidations=normalized_long_liquidations,
            short_liquidations=normalized_short_liquidations,
        )

        if confidence < self.config.min_confidence:
            self.logger.debug(
                "Funding divergence ignored due to low confidence | exchange=%s "
                "market_type=%s symbol=%s timeframe=%s type=%s confidence=%.4f",
                snapshot.exchange.value,
                snapshot.market_type,
                snapshot.symbol,
                snapshot.timeframe.value,
                divergence_type.value,
                confidence,
                extra={
                    "scope": funding_key_to_dict(snapshot.key),
                    "exchange_symbol": snapshot.exchange_symbol,
                },
            )
            return None

        tf = (
            timeframe
            or (statistics.timeframe if statistics is not None else None)
            or snapshot.timeframe
            or self.config.default_timeframe
        )

        metadata: dict[str, Any] = {
            "scope": funding_key_to_dict(snapshot.key),
            "exchange_symbol": snapshot.exchange_symbol,
            "funding_sign": snapshot.funding_sign,
            "funding_abs": abs(snapshot.funding_rate),
            "basis": snapshot.basis,
            "price_change_pct": normalized_price_change_pct,
            "oi_change_pct": normalized_oi_change_pct,
            "cvd_change": normalized_cvd_change,
            "long_liquidations": normalized_long_liquidations,
            "short_liquidations": normalized_short_liquidations,
            "enabled_detectors": {
                "price_funding": self.config.enable_price_funding_divergence,
                "oi_funding": self.config.enable_oi_funding_divergence,
                "cvd_funding": self.config.enable_cvd_funding_divergence,
                "liquidation_funding": (
                    self.config.enable_liquidation_funding_divergence
                ),
            },
        }

        if statistics is not None:
            metadata.update(
                {
                    "percentile": statistics.percentile,
                    "zscore": statistics.zscore,
                    "sample_size": statistics.sample_size,
                    "mean_rate": statistics.mean_rate,
                    "median_rate": statistics.median_rate,
                    "std_rate": statistics.std_rate,
                    "min_rate": statistics.min_rate,
                    "max_rate": statistics.max_rate,
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

        event = FundingDivergenceEvent(
            symbol=snapshot.symbol,
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            timeframe=tf,
            exchange_symbol=snapshot.exchange_symbol,
            divergence_type=divergence_type,
            funding_rate=snapshot.funding_rate,
            price_change_pct=normalized_price_change_pct,
            oi_change_pct=normalized_oi_change_pct,
            cvd_change=normalized_cvd_change,
            long_liquidations=normalized_long_liquidations,
            short_liquidations=normalized_short_liquidations,
            confidence=confidence,
            event_time=snapshot.event_time,
            metadata=metadata,
        )

        self.logger.debug(
            "Funding divergence detected | exchange=%s market_type=%s symbol=%s "
            "timeframe=%s type=%s funding_rate=%.8f price_change_pct=%s "
            "oi_change_pct=%s cvd_change=%s long_liquidations=%s "
            "short_liquidations=%s confidence=%.4f",
            event.exchange.value,
            event.market_type,
            event.symbol,
            event.timeframe.value,
            event.divergence_type.value,
            event.funding_rate,
            self._format_optional_float(event.price_change_pct),
            self._format_optional_float(event.oi_change_pct),
            self._format_optional_float(event.cvd_change),
            self._format_optional_float(event.long_liquidations),
            self._format_optional_float(event.short_liquidations),
            event.confidence,
            extra={
                "scope": funding_key_to_dict(event.key),
                "exchange_symbol": event.exchange_symbol,
            },
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
        1. liquidation divergence;
        2. price/funding divergence;
        3. OI/funding divergence;
        4. CVD/funding divergence.
        """
        funding_rate = float(funding_rate)

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

            # Funding позитивний, OI росте, але ціна стоїть або слабка.
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
        - сила funding;
        - сила протиріччя в зовнішній метриці;
        - percentile abnormality;
        - z-score abnormality.
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

        return self._clamp_0_1(confidence)

    # ------------------------------------------------------------------
    # Helper API
    # ------------------------------------------------------------------

    def is_bullish_divergence(
        self,
        event: FundingDivergenceEvent | None,
    ) -> bool:
        """
        Буліш-контекст:
        - price up while funding negative;
        - CVD up while funding negative;
        - OI up while funding negative;
        - shorts under pressure.
        """
        if event is None:
            return False

        return event.divergence_type in {
            FundingDivergenceType.PRICE_UP_FUNDING_DOWN,
            FundingDivergenceType.CVD_UP_FUNDING_DOWN,
            FundingDivergenceType.OI_UP_FUNDING_DOWN,
            FundingDivergenceType.LIQUIDATIONS_SHORTS_WITH_NEGATIVE_FUNDING,
        }

    def is_bearish_divergence(
        self,
        event: FundingDivergenceEvent | None,
    ) -> bool:
        """
        Беаріш-контекст:
        - price down while funding positive;
        - CVD down while funding positive;
        - OI up + positive funding + price stalled;
        - longs under pressure.
        """
        if event is None:
            return False

        return event.divergence_type in {
            FundingDivergenceType.PRICE_DOWN_FUNDING_UP,
            FundingDivergenceType.CVD_DOWN_FUNDING_UP,
            FundingDivergenceType.OI_UP_FUNDING_UP_PRICE_STALLED,
            FundingDivergenceType.LIQUIDATIONS_LONGS_WITH_POSITIVE_FUNDING,
        }

    def build_summary(
        self,
        event: FundingDivergenceEvent,
    ) -> str:
        return (
            f"Funding divergence for "
            f"{event.exchange.value}:{event.market_type}:{event.symbol}:{event.timeframe.value}: "
            f"type={event.divergence_type.value}, "
            f"funding_rate={event.funding_rate:.8f}, "
            f"price_change_pct={event.price_change_pct if event.price_change_pct is not None else 'None'}, "
            f"oi_change_pct={event.oi_change_pct if event.oi_change_pct is not None else 'None'}, "
            f"cvd_change={event.cvd_change if event.cvd_change is not None else 'None'}, "
            f"long_liquidations={event.long_liquidations if event.long_liquidations is not None else 'None'}, "
            f"short_liquidations={event.short_liquidations if event.short_liquidations is not None else 'None'}, "
            f"confidence={event.confidence:.4f}"
        )

    # ------------------------------------------------------------------
    # Internal scoring helpers
    # ------------------------------------------------------------------

    def _calc_funding_score(
        self,
        funding_rate: float,
    ) -> float:
        abs_rate = abs(float(funding_rate))

        if abs_rate <= self.config.neutral_abs_threshold:
            return 0.0

        denominator = max(
            self.config.min_funding_for_divergence,
            1e-12,
        )
        normalized = abs_rate / denominator
        return self._clamp_0_1(normalized / 3.0)

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

        normalized = abs(float(value)) / threshold
        return self._clamp_0_1(normalized / max(scale, 1e-12))

    def _normalize_cvd(
        self,
        cvd_change: float | None,
    ) -> float:
        if cvd_change is None:
            return 0.0

        value = abs(float(cvd_change))
        if value <= self.config.min_cvd_change:
            return 0.0

        # М'яка normalization без symbol-specific scale.
        # Symbol-aware scaling може бути доданий у FundingAnalyzer/context layer.
        denominator = max(value, 1.0)
        return self._clamp_0_1(value / denominator)

    def _normalize_liquidations(
        self,
        liquidations: float | None,
    ) -> float:
        if liquidations is None:
            return 0.0

        value = abs(float(liquidations))
        if value <= self.config.min_liquidations_value:
            return 0.0

        # М'яка bounded normalization без припущень щодо масштабу конкретного ринку.
        return self._clamp_0_1(value / (value + 1.0))

    # ------------------------------------------------------------------
    # Validation / helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input_scope(
        *,
        snapshot: FundingSnapshot,
        statistics: FundingStatistics | None,
    ) -> None:
        """
        Funding divergence не можна рахувати на змішаних scope-ах.

        Якщо statistics переданий, він має належати тому самому:
            exchange + market_type + symbol + timeframe
        """
        if statistics is not None and snapshot.key != statistics.key:
            raise ValueError(
                "FundingSnapshot and FundingStatistics scope mismatch: "
                f"snapshot={funding_key_to_dict(snapshot.key)} "
                f"statistics={funding_key_to_dict(statistics.key)}"
            )

    @staticmethod
    def _to_optional_float(
        value: float | int | str | None,
    ) -> float | None:
        if value is None:
            return None

        try:
            result = float(value)
        except (TypeError, ValueError):
            return None

        if result != result:  # NaN
            return None

        return result

    @classmethod
    def _non_negative_optional_float(
        cls,
        value: float | int | str | None,
    ) -> float | None:
        result = cls._to_optional_float(value)
        if result is None:
            return None
        return max(0.0, result)

    @staticmethod
    def _format_optional_float(
        value: float | None,
    ) -> str:
        return f"{value:.6f}" if value is not None else "None"

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
    "FundingDivergenceConfig",
    "FundingDivergenceDetector",
]