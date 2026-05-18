from __future__ import annotations

from enum import Enum


class OIRegime(str, Enum):
    """
    Основний futures/perpetual ринковий режим з точки зору взаємодії:
    - price
    - open interest
    - volume
    - liquidations
    - funding
    - orderflow

    У проєкті open-interest analytics розглядається тільки для futures/perps,
    не для spot.
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

    @property
    def is_neutral(self) -> bool:
        return self is OIRegime.NEUTRAL


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

    @property
    def sign(self) -> int:
        if self is OIDirection.UP:
            return 1
        if self is OIDirection.DOWN:
            return -1
        return 0


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
    Аномальні події / стани по futures open interest.
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

    Ці події публікує analytics/open_interest/OIAnalyzer.

    Downstream consumers:
    - strategy
    - risk context
    - dashboard
    - storage
    - bots / alerts
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
    Data-layer / analytics-layer topics, які OIAnalyzer слухає через EventBus.

    Важливо:
    OIAnalyzer не має слухати raw exchange events напряму.
    Він має отримувати вже нормалізовані futures events із data cache layer
    або з інших analytics-пакетів.

    Правильний потік:
        exchanges -> data caches -> market.*.updated / market.candle.closed
        -> analytics.open_interest -> analytics.oi.*

    Scope кожного payload:
        exchange + market_type + symbol + timeframe

    Futures-only market_type examples:
        binance: usdm_futures
        bybit: linear
        okx: swap
        mexc: usdm_futures
    """

    # Main OI trigger from OpenInterestCache.
    OPEN_INTEREST_UPDATED = "market.open_interest.updated"

    # Price/volume context from CandlesCache.
    CANDLE_CLOSED = "market.candle.closed"
    CANDLES_UPDATED = "market.candles.updated"

    # Optional fallback volume/trade context from TradesCache.
    TRADES_UPDATED = "market.trades.updated"

    # Funding context from FundingCache.
    FUNDING_UPDATED = "market.funding.updated"

    # Preferred context from analytics packages.
    ORDERFLOW_UPDATED = "analytics.orderflow.updated"
    LIQUIDATIONS_UPDATED = "analytics.liquidations.updated"

    # Backward-compatible aliases for old OIAnalyzer code.
    # Після оновлення oi_analyzer.py краще використовувати explicit *_UPDATED names.
    OPEN_INTEREST = OPEN_INTEREST_UPDATED
    CANDLE = CANDLE_CLOSED
    TRADE = TRADES_UPDATED
    FUNDING = FUNDING_UPDATED
    LIQUIDATION = LIQUIDATIONS_UPDATED

    @property
    def topic(self) -> str:
        return self.value

    @property
    def is_primary_oi_trigger(self) -> bool:
        return self is OIMarketEventType.OPEN_INTEREST_UPDATED

    @property
    def is_context_event(self) -> bool:
        return self in {
            OIMarketEventType.CANDLE_CLOSED,
            OIMarketEventType.CANDLES_UPDATED,
            OIMarketEventType.TRADES_UPDATED,
            OIMarketEventType.FUNDING_UPDATED,
            OIMarketEventType.ORDERFLOW_UPDATED,
            OIMarketEventType.LIQUIDATIONS_UPDATED,
        }