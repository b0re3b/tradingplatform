from __future__ import annotations

from enum import Enum


class OIRegime(str, Enum):
    """
    Основний ринковий режим з точки зору взаємодії:
    - ціни
    - open interest
    - обсягу
    - ліквідацій
    - funding
    """

    NEUTRAL = "NEUTRAL"

    LONG_BUILDUP = "LONG_BUILDUP"
    SHORT_BUILDUP = "SHORT_BUILDUP"

    SHORT_COVERING = "SHORT_COVERING"
    LONG_UNWIND = "LONG_UNWIND"

    TREND_CONFIRMATION = "TREND_CONFIRMATION"
    TREND_EXHAUSTION = "TREND_EXHAUSTION"

    SQUEEZE_SETUP = "SQUEEZE_SETUP"
    CAPITULATION = "CAPITULATION"
    OVERHEATED = "OVERHEATED"


class OIDirection(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    FLAT = "FLAT"
    UNKNOWN = "UNKNOWN"


class OIDivergenceType(str, Enum):
    """
    Типи дивергенцій між price та OI.
    """

    NONE = "NONE"

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"

    PRICE_UP_OI_DOWN = "PRICE_UP_OI_DOWN"
    PRICE_DOWN_OI_DOWN = "PRICE_DOWN_OI_DOWN"

    PRICE_UP_OI_FLAT = "PRICE_UP_OI_FLAT"
    PRICE_DOWN_OI_FLAT = "PRICE_DOWN_OI_FLAT"

    WEAK_BREAKOUT_UP = "WEAK_BREAKOUT_UP"
    WEAK_BREAKOUT_DOWN = "WEAK_BREAKOUT_DOWN"

    EXHAUSTION_UP = "EXHAUSTION_UP"
    EXHAUSTION_DOWN = "EXHAUSTION_DOWN"


class OIAnomalyType(str, Enum):
    """
    Аномальні події/стани по OI.
    """

    NONE = "NONE"

    OI_SPIKE = "OI_SPIKE"
    OI_COLLAPSE = "OI_COLLAPSE"

    OI_PRICE_DISLOCATION = "OI_PRICE_DISLOCATION"
    OI_VOLUME_DISLOCATION = "OI_VOLUME_DISLOCATION"

    LIQUIDATION_DRIVEN_OI_DROP = "LIQUIDATION_DRIVEN_OI_DROP"
    OVERHEATED_BUILDUP = "OVERHEATED_BUILDUP"
    SUDDEN_DELEVERAGING = "SUDDEN_DELEVERAGING"

    FUNDING_OI_IMBALANCE = "FUNDING_OI_IMBALANCE"
    EXTREME_CROWDING = "EXTREME_CROWDING"


class OISignalStrength(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class OIConfidenceBand(str, Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class OIEventType(str, Enum):
    """
    Уніфіковані event names, які буде зручно використовувати
    в analyzer / strategies / dashboard.
    """

    UPDATED = "analytics.oi.updated"
    REGIME_CHANGED = "analytics.oi.regime.changed"
    DIVERGENCE_DETECTED = "analytics.oi.divergence.detected"
    ANOMALY_DETECTED = "analytics.oi.anomaly.detected"
    SQUEEZE_SETUP = "analytics.oi.squeeze_setup"
    CAPITULATION_DETECTED = "analytics.oi.capitulation.detected"