from __future__ import annotations

from enum import Enum


class FundingRegime(str, Enum):
    """
    Загальний режим funding rate для інструмента.
    """

    UNKNOWN = "unknown"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    EXTREME_POSITIVE = "extreme_positive"
    EXTREME_NEGATIVE = "extreme_negative"
    FLIPPING = "flipping"


class FundingBias(str, Enum):
    """
    Інтерпретація перекосу позиціонування натовпу.
    """

    NEUTRAL = "neutral"
    LONG_BIAS = "long_bias"
    SHORT_BIAS = "short_bias"
    OVERCROWDED_LONGS = "overcrowded_longs"
    OVERCROWDED_SHORTS = "overcrowded_shorts"
    SQUEEZE_RISK_LONGS = "squeeze_risk_longs"
    SQUEEZE_RISK_SHORTS = "squeeze_risk_shorts"


class FundingSignalType(str, Enum):
    """
    Тип funding-сигналу, який може бути згенерований аналітичним шаром.
    """

    REGIME_CHANGE = "regime_change"
    EXTREME_DETECTED = "extreme_detected"
    DIVERGENCE_DETECTED = "divergence_detected"
    FLIP_DETECTED = "flip_detected"
    PRESSURE_BUILDUP = "pressure_buildup"
    PRESSURE_RELEASE = "pressure_release"
    CROWDING_WARNING = "crowding_warning"
    SQUEEZE_WARNING = "squeeze_warning"
    REVERSION_SETUP = "reversion_setup"
    TREND_CONFIRMATION = "trend_confirmation"


class FundingExtremeType(str, Enum):
    """
    Конкретний тип екстремуму funding.
    """

    NONE = "none"
    LOCAL_HIGH = "local_high"
    LOCAL_LOW = "local_low"
    GLOBAL_HIGH = "global_high"
    GLOBAL_LOW = "global_low"
    ZSCORE_HIGH = "zscore_high"
    ZSCORE_LOW = "zscore_low"
    PERCENTILE_HIGH = "percentile_high"
    PERCENTILE_LOW = "percentile_low"


class FundingDivergenceType(str, Enum):
    """
    Тип дивергенції між funding та іншими ринковими компонентами.
    """

    NONE = "none"
    PRICE_UP_FUNDING_DOWN = "price_up_funding_down"
    PRICE_DOWN_FUNDING_UP = "price_down_funding_up"
    OI_UP_FUNDING_DOWN = "oi_up_funding_down"
    OI_UP_FUNDING_UP_PRICE_STALLED = "oi_up_funding_up_price_stalled"
    CVD_UP_FUNDING_DOWN = "cvd_up_funding_down"
    CVD_DOWN_FUNDING_UP = "cvd_down_funding_up"
    LIQUIDATIONS_LONGS_WITH_POSITIVE_FUNDING = (
        "liquidations_longs_with_positive_funding"
    )
    LIQUIDATIONS_SHORTS_WITH_NEGATIVE_FUNDING = (
        "liquidations_shorts_with_negative_funding"
    )


class FundingFlipType(str, Enum):
    """
    Тип зміни знаку funding.
    """

    NONE = "none"
    NEGATIVE_TO_POSITIVE = "negative_to_positive"
    POSITIVE_TO_NEGATIVE = "positive_to_negative"


class FundingPressureLevel(str, Enum):
    """
    Рівень тиску позиціонування.
    """

    UNKNOWN = "unknown"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class FundingPressureDirection(str, Enum):
    """
    Напрямок тиску.
    """

    NEUTRAL = "neutral"
    LONG = "long"
    SHORT = "short"
    TWO_SIDED = "two_sided"


class FundingTimeframe(str, Enum):
    """
    Таймфрейм або горизонт оцінки funding-аналітики.
    """

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    H8 = "8h"
    D1 = "1d"


class FundingEventType(str, Enum):
    """
    Внутрішня класифікація funding events для EventBus/analytics.
    """

    SNAPSHOT = "snapshot"
    REGIME = "regime"
    EXTREME = "extreme"
    DIVERGENCE = "divergence"
    FLIP = "flip"
    PRESSURE = "pressure"
    SIGNAL = "signal"


class FundingDataSource(str, Enum):
    """
    Джерело funding-даних.
    """

    UNKNOWN = "unknown"
    BINANCE = "binance"
    BYBIT = "bybit"
    OKX = "okx"
    MEXC = "mexc"
    GATE = "gate"
    AGGREGATED = "aggregated"
    INTERNAL = "internal"