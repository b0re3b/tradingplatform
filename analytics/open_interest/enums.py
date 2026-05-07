from __future__ import annotations

from enum import Enum


class OIRegime(str, Enum):
    """
    Основний ринковий режим з точки зору взаємодії:
    - price
    - open interest
    - volume
    - liquidations
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

    @property
    def is_directional(self) -> bool:
        return self in {
            OIRegime.LONG_BUILDUP,
            OIRegime.SHORT_BUILDUP,
            OIRegime.SHORT_COVERING,
            OIRegime.LONG_UNWIND,
            OIRegime.TREND_CONFIRMATION,
        }

    @property
    def is_risk_regime(self) -> bool:
        return self in {
            OIRegime.SQUEEZE_SETUP,
            OIRegime.CAPITULATION,
            OIRegime.OVERHEATED,
            OIRegime.TREND_EXHAUSTION,
        }


class OIDirection(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    FLAT = "FLAT"
    UNKNOWN = "UNKNOWN"

    @property
    def is_known(self) -> bool:
        return self is not OIDirection.UNKNOWN

    @property
    def is_directional(self) -> bool:
        return self in {OIDirection.UP, OIDirection.DOWN}


class OIDivergenceType(str, Enum):
    """
    Типи дивергенцій між price та open interest.
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

    @property
    def is_detected(self) -> bool:
        return self is not OIDivergenceType.NONE

    @property
    def is_bullish_context(self) -> bool:
        return self in {
            OIDivergenceType.BULLISH,
            OIDivergenceType.PRICE_DOWN_OI_DOWN,
            OIDivergenceType.PRICE_DOWN_OI_FLAT,
            OIDivergenceType.WEAK_BREAKOUT_DOWN,
            OIDivergenceType.EXHAUSTION_DOWN,
        }

    @property
    def is_bearish_context(self) -> bool:
        return self in {
            OIDivergenceType.BEARISH,
            OIDivergenceType.PRICE_UP_OI_DOWN,
            OIDivergenceType.PRICE_UP_OI_FLAT,
            OIDivergenceType.WEAK_BREAKOUT_UP,
            OIDivergenceType.EXHAUSTION_UP,
        }


class OIAnomalyType(str, Enum):
    """
    Аномальні події / стани по open interest.
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

    @property
    def is_detected(self) -> bool:
        return self is not OIAnomalyType.NONE

    @property
    def is_risk_anomaly(self) -> bool:
        return self in {
            OIAnomalyType.OI_COLLAPSE,
            OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP,
            OIAnomalyType.OVERHEATED_BUILDUP,
            OIAnomalyType.SUDDEN_DELEVERAGING,
            OIAnomalyType.FUNDING_OI_IMBALANCE,
            OIAnomalyType.EXTREME_CROWDING,
        }


class OISignalStrength(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

    @property
    def weight(self) -> float:
        if self is OISignalStrength.EXTREME:
            return 1.0
        if self is OISignalStrength.HIGH:
            return 0.75
        if self is OISignalStrength.MEDIUM:
            return 0.5
        return 0.25


class OIConfidenceBand(str, Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"

    @property
    def min_confidence(self) -> float:
        if self is OIConfidenceBand.VERY_HIGH:
            return 0.90
        if self is OIConfidenceBand.HIGH:
            return 0.75
        if self is OIConfidenceBand.MEDIUM:
            return 0.50
        if self is OIConfidenceBand.LOW:
            return 0.25
        return 0.0


class OIEventType(str, Enum):
    """
    Уніфіковані EventBus topic names для Open Interest analytics.

    Ці значення використовуються в:
    - analytics/open_interest/oi_analyzer.py
    - strategy/open_interest/*
    - dashboard subscribers
    - storage subscribers
    - bots / alert modules
    """

    UPDATED = "analytics.oi.updated"

    REGIME_CHANGED = "analytics.oi.regime.changed"
    DIVERGENCE_DETECTED = "analytics.oi.divergence.detected"
    ANOMALY_DETECTED = "analytics.oi.anomaly.detected"

    SQUEEZE_SETUP = "analytics.oi.squeeze_setup"
    CAPITULATION_DETECTED = "analytics.oi.capitulation.detected"

    METRICS = "analytics.oi.metrics"
    HEALTH = "analytics.oi.health"
    STATE_CLEANED = "analytics.oi.state.cleaned"

    @property
    def topic(self) -> str:
        return self.value


class OIMarketEventType(str, Enum):
    """
    Market topics, які OIAnalyzer слухає через EventBus.subscribe().
    """

    OPEN_INTEREST = "market.open_interest"
    CANDLE = "market.candle"
    TRADE = "market.trade"
    FUNDING = "market.funding"
    LIQUIDATION = "market.liquidation"
    ORDERFLOW_UPDATED = "analytics.orderflow.updated"

    @property
    def topic(self) -> str:
        return self.value