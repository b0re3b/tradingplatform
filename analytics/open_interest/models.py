from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .enums import (
    OIAnomalyType,
    OIConfidenceBand,
    OIDirection,
    OIDivergenceType,
    OIRegime,
    OISignalStrength,
)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _confidence_to_band(confidence: float) -> OIConfidenceBand:
    if confidence >= 0.9:
        return OIConfidenceBand.VERY_HIGH
    if confidence >= 0.75:
        return OIConfidenceBand.HIGH
    if confidence >= 0.5:
        return OIConfidenceBand.MEDIUM
    if confidence >= 0.25:
        return OIConfidenceBand.LOW
    return OIConfidenceBand.VERY_LOW


@dataclass(slots=True)
class OISnapshot:
    """
    Сирий snapshot open interest.
    """

    symbol: str
    exchange: str
    timestamp: float
    oi: float

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper().strip()
        self.exchange = self.exchange.lower().strip()
        self.timestamp = float(self.timestamp)
        self.oi = float(self.oi)

        if not self.symbol:
            raise ValueError("OISnapshot.symbol must not be empty")
        if not self.exchange:
            raise ValueError("OISnapshot.exchange must not be empty")
        if self.timestamp <= 0:
            raise ValueError("OISnapshot.timestamp must be > 0")
        if self.oi < 0:
            raise ValueError("OISnapshot.oi must be >= 0")

    @property
    def key(self) -> tuple[str, str]:
        return self.exchange, self.symbol

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timestamp": self.timestamp,
            "oi": self.oi,
        }


@dataclass(slots=True)
class OIMarketContext:
    """
    Контекст ринку на момент оцінки OI.
    Частина полів може бути відсутня, якщо відповідні стріми
    ще не прийшли або не підключені.
    """

    symbol: str
    exchange: str
    timestamp: float

    price: float | None = None
    price_delta: float | None = None
    price_delta_pct: float | None = None

    volume: float | None = None
    volume_ma: float | None = None
    volume_ratio: float | None = None

    funding_rate: float | None = None

    long_liquidations: float | None = None
    short_liquidations: float | None = None

    cvd_delta: float | None = None
    aggressive_buy_volume: float | None = None
    aggressive_sell_volume: float | None = None

    mark_price: float | None = None
    index_price: float | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper().strip()
        self.exchange = self.exchange.lower().strip()
        self.timestamp = float(self.timestamp)

        if not self.symbol:
            raise ValueError("OIMarketContext.symbol must not be empty")
        if not self.exchange:
            raise ValueError("OIMarketContext.exchange must not be empty")
        if self.timestamp <= 0:
            raise ValueError("OIMarketContext.timestamp must be > 0")

        for attr in (
            "price",
            "price_delta",
            "price_delta_pct",
            "volume",
            "volume_ma",
            "volume_ratio",
            "funding_rate",
            "long_liquidations",
            "short_liquidations",
            "cvd_delta",
            "aggressive_buy_volume",
            "aggressive_sell_volume",
            "mark_price",
            "index_price",
        ):
            setattr(self, attr, _safe_float(getattr(self, attr)))

    @property
    def key(self) -> tuple[str, str]:
        return self.exchange, self.symbol

    @property
    def liquidation_imbalance(self) -> float | None:
        if self.long_liquidations is None or self.short_liquidations is None:
            return None

        total = self.long_liquidations + self.short_liquidations
        if total <= 0:
            return 0.0

        return (self.short_liquidations - self.long_liquidations) / total

    @property
    def aggressive_flow_imbalance(self) -> float | None:
        if self.aggressive_buy_volume is None or self.aggressive_sell_volume is None:
            return None

        total = self.aggressive_buy_volume + self.aggressive_sell_volume
        if total <= 0:
            return 0.0

        return (self.aggressive_buy_volume - self.aggressive_sell_volume) / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timestamp": self.timestamp,
            "price": self.price,
            "price_delta": self.price_delta,
            "price_delta_pct": self.price_delta_pct,
            "volume": self.volume,
            "volume_ma": self.volume_ma,
            "volume_ratio": self.volume_ratio,
            "funding_rate": self.funding_rate,
            "long_liquidations": self.long_liquidations,
            "short_liquidations": self.short_liquidations,
            "liquidation_imbalance": self.liquidation_imbalance,
            "cvd_delta": self.cvd_delta,
            "aggressive_buy_volume": self.aggressive_buy_volume,
            "aggressive_sell_volume": self.aggressive_sell_volume,
            "aggressive_flow_imbalance": self.aggressive_flow_imbalance,
            "mark_price": self.mark_price,
            "index_price": self.index_price,
            "extra": dict(self.extra),
        }


@dataclass(slots=True)
class OIFeatures:
    """
    Розраховані фічі для інтерпретації OI.
    """

    oi: float
    oi_delta: float
    oi_delta_pct: float

    oi_ma_fast: float | None = None
    oi_ma_slow: float | None = None
    oi_std: float | None = None
    oi_zscore: float | None = None

    oi_velocity: float | None = None
    oi_acceleration: float | None = None

    price: float | None = None
    price_delta: float | None = None
    price_delta_pct: float | None = None

    volume: float | None = None
    volume_ma: float | None = None
    volume_ratio: float | None = None

    funding_rate: float | None = None

    long_liquidations: float | None = None
    short_liquidations: float | None = None
    liquidation_imbalance: float | None = None

    cvd_delta: float | None = None
    aggressive_buy_volume: float | None = None
    aggressive_sell_volume: float | None = None
    aggressive_flow_imbalance: float | None = None

    oi_change_per_volume: float | None = None
    oi_price_efficiency: float | None = None
    oi_pressure_score: float | None = None

    oi_direction: OIDirection = OIDirection.UNKNOWN
    price_direction: OIDirection = OIDirection.UNKNOWN

    def __post_init__(self) -> None:
        for attr in (
            "oi",
            "oi_delta",
            "oi_delta_pct",
            "oi_ma_fast",
            "oi_ma_slow",
            "oi_std",
            "oi_zscore",
            "oi_velocity",
            "oi_acceleration",
            "price",
            "price_delta",
            "price_delta_pct",
            "volume",
            "volume_ma",
            "volume_ratio",
            "funding_rate",
            "long_liquidations",
            "short_liquidations",
            "liquidation_imbalance",
            "cvd_delta",
            "aggressive_buy_volume",
            "aggressive_sell_volume",
            "aggressive_flow_imbalance",
            "oi_change_per_volume",
            "oi_price_efficiency",
            "oi_pressure_score",
        ):
            setattr(self, attr, _safe_float(getattr(self, attr)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "oi": self.oi,
            "oi_delta": self.oi_delta,
            "oi_delta_pct": self.oi_delta_pct,
            "oi_ma_fast": self.oi_ma_fast,
            "oi_ma_slow": self.oi_ma_slow,
            "oi_std": self.oi_std,
            "oi_zscore": self.oi_zscore,
            "oi_velocity": self.oi_velocity,
            "oi_acceleration": self.oi_acceleration,
            "price": self.price,
            "price_delta": self.price_delta,
            "price_delta_pct": self.price_delta_pct,
            "volume": self.volume,
            "volume_ma": self.volume_ma,
            "volume_ratio": self.volume_ratio,
            "funding_rate": self.funding_rate,
            "long_liquidations": self.long_liquidations,
            "short_liquidations": self.short_liquidations,
            "liquidation_imbalance": self.liquidation_imbalance,
            "cvd_delta": self.cvd_delta,
            "aggressive_buy_volume": self.aggressive_buy_volume,
            "aggressive_sell_volume": self.aggressive_sell_volume,
            "aggressive_flow_imbalance": self.aggressive_flow_imbalance,
            "oi_change_per_volume": self.oi_change_per_volume,
            "oi_price_efficiency": self.oi_price_efficiency,
            "oi_pressure_score": self.oi_pressure_score,
            "oi_direction": self.oi_direction.value,
            "price_direction": self.price_direction.value,
        }


@dataclass(slots=True)
class OIRegimeResult:
    regime: OIRegime
    confidence: float
    reasons: list[str] = field(default_factory=list)
    score: float | None = None

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.score = _safe_float(self.score)

    @property
    def confidence_band(self) -> OIConfidenceBand:
        return _confidence_to_band(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "confidence_band": self.confidence_band.value,
            "score": self.score,
            "reasons": list(self.reasons),
        }


@dataclass(slots=True)
class OIDivergenceResult:
    detected: bool
    divergence_type: OIDivergenceType = OIDivergenceType.NONE
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    window_size: int | None = None
    score: float | None = None

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.score = _safe_float(self.score)

        if not self.detected:
            self.divergence_type = OIDivergenceType.NONE
            self.confidence = 0.0

    @property
    def confidence_band(self) -> OIConfidenceBand:
        return _confidence_to_band(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "divergence_type": self.divergence_type.value,
            "confidence": self.confidence,
            "confidence_band": self.confidence_band.value,
            "window_size": self.window_size,
            "score": self.score,
            "reasons": list(self.reasons),
        }


@dataclass(slots=True)
class OIAnomalyResult:
    detected: bool
    anomaly_type: OIAnomalyType = OIAnomalyType.NONE
    strength: OISignalStrength = OISignalStrength.LOW
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    score: float | None = None

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.score = _safe_float(self.score)

        if not self.detected:
            self.anomaly_type = OIAnomalyType.NONE
            self.confidence = 0.0

    @property
    def confidence_band(self) -> OIConfidenceBand:
        return _confidence_to_band(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "anomaly_type": self.anomaly_type.value,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "confidence_band": self.confidence_band.value,
            "score": self.score,
            "reasons": list(self.reasons),
        }


@dataclass(slots=True)
class OIAnalysisResult:
    """
    Фінальний результат повного аналізу OI.
    """

    symbol: str
    exchange: str
    timestamp: float

    snapshot: OISnapshot
    context: OIMarketContext
    features: OIFeatures
    regime: OIRegimeResult
    divergence: OIDivergenceResult | None = None
    anomaly: OIAnomalyResult | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper().strip()
        self.exchange = self.exchange.lower().strip()
        self.timestamp = float(self.timestamp)

        if not self.symbol:
            raise ValueError("OIAnalysisResult.symbol must not be empty")
        if not self.exchange:
            raise ValueError("OIAnalysisResult.exchange must not be empty")
        if self.timestamp <= 0:
            raise ValueError("OIAnalysisResult.timestamp must be > 0")

    @property
    def key(self) -> tuple[str, str]:
        return self.exchange, self.symbol

    @property
    def has_divergence(self) -> bool:
        return self.divergence is not None and self.divergence.detected

    @property
    def has_anomaly(self) -> bool:
        return self.anomaly is not None and self.anomaly.detected

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timestamp": self.timestamp,
            "snapshot": self.snapshot.to_dict(),
            "context": self.context.to_dict(),
            "features": self.features.to_dict(),
            "regime": self.regime.to_dict(),
            "divergence": self.divergence.to_dict() if self.divergence else None,
            "anomaly": self.anomaly.to_dict() if self.anomaly else None,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class OIState:
    """
    Поточний стан по конкретному symbol/exchange,
    який буде вести OIAnalyzer.
    """

    symbol: str
    exchange: str

    last_snapshot: OISnapshot | None = None
    last_context: OIMarketContext | None = None
    last_features: OIFeatures | None = None
    last_analysis: OIAnalysisResult | None = None
    last_regime: OIRegime = OIRegime.NEUTRAL

    last_update_ts: float | None = None

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper().strip()
        self.exchange = self.exchange.lower().strip()

        if not self.symbol:
            raise ValueError("OIState.symbol must not be empty")
        if not self.exchange:
            raise ValueError("OIState.exchange must not be empty")

        self.last_update_ts = _safe_float(self.last_update_ts)

    @property
    def key(self) -> tuple[str, str]:
        return self.exchange, self.symbol

    def touch(self, timestamp: float) -> None:
        self.last_update_ts = float(timestamp)