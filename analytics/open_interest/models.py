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


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _normalize_symbol(symbol: str) -> str:
    normalized = str(symbol or "").upper().strip()
    if not normalized:
        raise ValueError("symbol must not be empty")
    return normalized


def _normalize_exchange(exchange: str) -> str:
    normalized = str(exchange or "").lower().strip()
    if not normalized:
        raise ValueError("exchange must not be empty")
    return normalized


def _confidence_to_band(confidence: float) -> OIConfidenceBand:
    confidence = _clamp(confidence)

    if confidence >= 0.90:
        return OIConfidenceBand.VERY_HIGH
    if confidence >= 0.75:
        return OIConfidenceBand.HIGH
    if confidence >= 0.50:
        return OIConfidenceBand.MEDIUM
    if confidence >= 0.25:
        return OIConfidenceBand.LOW
    return OIConfidenceBand.VERY_LOW


def _coerce_oi_regime(value: OIRegime | str) -> OIRegime:
    if isinstance(value, OIRegime):
        return value
    return OIRegime(str(value))


def _coerce_oi_direction(value: OIDirection | str | None) -> OIDirection:
    if value is None:
        return OIDirection.UNKNOWN
    if isinstance(value, OIDirection):
        return value
    return OIDirection(str(value))


def _coerce_divergence_type(
    value: OIDivergenceType | str | None,
) -> OIDivergenceType:
    if value is None:
        return OIDivergenceType.NONE
    if isinstance(value, OIDivergenceType):
        return value
    return OIDivergenceType(str(value))


def _coerce_anomaly_type(value: OIAnomalyType | str | None) -> OIAnomalyType:
    if value is None:
        return OIAnomalyType.NONE
    if isinstance(value, OIAnomalyType):
        return value
    return OIAnomalyType(str(value))


def _coerce_signal_strength(
    value: OISignalStrength | str | None,
) -> OISignalStrength:
    if value is None:
        return OISignalStrength.LOW
    if isinstance(value, OISignalStrength):
        return value
    return OISignalStrength(str(value))


@dataclass(slots=True)
class OISnapshot:
    """
    Сирий snapshot open interest з market data layer.
    """

    symbol: str
    exchange: str
    timestamp: float
    oi: float

    def __post_init__(self) -> None:
        self.symbol = _normalize_symbol(self.symbol)
        self.exchange = _normalize_exchange(self.exchange)
        self.timestamp = float(self.timestamp)
        self.oi = float(self.oi)

        if self.timestamp <= 0:
            raise ValueError("OISnapshot.timestamp must be > 0")

        if self.oi < 0:
            raise ValueError("OISnapshot.oi must be >= 0")

    @property
    def key(self) -> tuple[str, str]:
        return self.exchange, self.symbol

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OISnapshot:
        return cls(
            symbol=str(data["symbol"]),
            exchange=str(data["exchange"]),
            timestamp=float(data["timestamp"]),
            oi=float(data["oi"]),
        )

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
    Контекст ринку на момент оцінки Open Interest.

    Частина полів може бути None, якщо відповідні стріми ще не прийшли
    або не підключені.
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
        self.symbol = _normalize_symbol(self.symbol)
        self.exchange = _normalize_exchange(self.exchange)
        self.timestamp = float(self.timestamp)

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

        self.extra = dict(self.extra or {})

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

    def is_stale(self, now_ts: float, max_age_sec: float) -> bool:
        if max_age_sec <= 0:
            raise ValueError("max_age_sec must be > 0")
        return float(now_ts) - self.timestamp > max_age_sec

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OIMarketContext:
        return cls(
            symbol=str(data["symbol"]),
            exchange=str(data["exchange"]),
            timestamp=float(data["timestamp"]),
            price=data.get("price"),
            price_delta=data.get("price_delta"),
            price_delta_pct=data.get("price_delta_pct"),
            volume=data.get("volume"),
            volume_ma=data.get("volume_ma"),
            volume_ratio=data.get("volume_ratio"),
            funding_rate=data.get("funding_rate"),
            long_liquidations=data.get("long_liquidations"),
            short_liquidations=data.get("short_liquidations"),
            cvd_delta=data.get("cvd_delta"),
            aggressive_buy_volume=data.get("aggressive_buy_volume"),
            aggressive_sell_volume=data.get("aggressive_sell_volume"),
            mark_price=data.get("mark_price"),
            index_price=data.get("index_price"),
            extra=dict(data.get("extra") or {}),
        )

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
    Розраховані фічі для інтерпретації Open Interest.
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

        if self.oi is None:
            raise ValueError("OIFeatures.oi must not be None")
        if self.oi_delta is None:
            raise ValueError("OIFeatures.oi_delta must not be None")
        if self.oi_delta_pct is None:
            raise ValueError("OIFeatures.oi_delta_pct must not be None")

        if self.oi < 0:
            raise ValueError("OIFeatures.oi must be >= 0")

        self.oi_direction = _coerce_oi_direction(self.oi_direction)
        self.price_direction = _coerce_oi_direction(self.price_direction)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OIFeatures:
        return cls(
            oi=data["oi"],
            oi_delta=data["oi_delta"],
            oi_delta_pct=data["oi_delta_pct"],
            oi_ma_fast=data.get("oi_ma_fast"),
            oi_ma_slow=data.get("oi_ma_slow"),
            oi_std=data.get("oi_std"),
            oi_zscore=data.get("oi_zscore"),
            oi_velocity=data.get("oi_velocity"),
            oi_acceleration=data.get("oi_acceleration"),
            price=data.get("price"),
            price_delta=data.get("price_delta"),
            price_delta_pct=data.get("price_delta_pct"),
            volume=data.get("volume"),
            volume_ma=data.get("volume_ma"),
            volume_ratio=data.get("volume_ratio"),
            funding_rate=data.get("funding_rate"),
            long_liquidations=data.get("long_liquidations"),
            short_liquidations=data.get("short_liquidations"),
            liquidation_imbalance=data.get("liquidation_imbalance"),
            cvd_delta=data.get("cvd_delta"),
            aggressive_buy_volume=data.get("aggressive_buy_volume"),
            aggressive_sell_volume=data.get("aggressive_sell_volume"),
            aggressive_flow_imbalance=data.get("aggressive_flow_imbalance"),
            oi_change_per_volume=data.get("oi_change_per_volume"),
            oi_price_efficiency=data.get("oi_price_efficiency"),
            oi_pressure_score=data.get("oi_pressure_score"),
            oi_direction=_coerce_oi_direction(data.get("oi_direction")),
            price_direction=_coerce_oi_direction(data.get("price_direction")),
        )

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
        self.regime = _coerce_oi_regime(self.regime)
        self.confidence = _clamp(float(self.confidence))
        self.score = _safe_float(self.score)
        self.reasons = list(self.reasons or [])

    @property
    def confidence_band(self) -> OIConfidenceBand:
        return _confidence_to_band(self.confidence)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OIRegimeResult:
        return cls(
            regime=_coerce_oi_regime(data["regime"]),
            confidence=float(data.get("confidence", 0.0)),
            reasons=list(data.get("reasons") or []),
            score=data.get("score"),
        )

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
        self.detected = bool(self.detected)
        self.divergence_type = _coerce_divergence_type(self.divergence_type)
        self.confidence = _clamp(float(self.confidence))
        self.window_size = _safe_int(self.window_size)
        self.score = _safe_float(self.score)
        self.reasons = list(self.reasons or [])

        if not self.detected:
            self.divergence_type = OIDivergenceType.NONE
            self.confidence = 0.0

        if self.window_size is not None and self.window_size < 0:
            raise ValueError("OIDivergenceResult.window_size must be >= 0")

    @property
    def confidence_band(self) -> OIConfidenceBand:
        return _confidence_to_band(self.confidence)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OIDivergenceResult:
        return cls(
            detected=bool(data.get("detected", False)),
            divergence_type=_coerce_divergence_type(data.get("divergence_type")),
            confidence=float(data.get("confidence", 0.0)),
            reasons=list(data.get("reasons") or []),
            window_size=data.get("window_size"),
            score=data.get("score"),
        )

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
        self.detected = bool(self.detected)
        self.anomaly_type = _coerce_anomaly_type(self.anomaly_type)
        self.strength = _coerce_signal_strength(self.strength)
        self.confidence = _clamp(float(self.confidence))
        self.score = _safe_float(self.score)
        self.reasons = list(self.reasons or [])

        if not self.detected:
            self.anomaly_type = OIAnomalyType.NONE
            self.strength = OISignalStrength.LOW
            self.confidence = 0.0

    @property
    def confidence_band(self) -> OIConfidenceBand:
        return _confidence_to_band(self.confidence)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OIAnomalyResult:
        return cls(
            detected=bool(data.get("detected", False)),
            anomaly_type=_coerce_anomaly_type(data.get("anomaly_type")),
            strength=_coerce_signal_strength(data.get("strength")),
            confidence=float(data.get("confidence", 0.0)),
            reasons=list(data.get("reasons") or []),
            score=data.get("score"),
        )

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
    Фінальний результат повного аналізу Open Interest.
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
        self.symbol = _normalize_symbol(self.symbol)
        self.exchange = _normalize_exchange(self.exchange)
        self.timestamp = float(self.timestamp)

        if self.timestamp <= 0:
            raise ValueError("OIAnalysisResult.timestamp must be > 0")

        if self.snapshot.key != self.key:
            raise ValueError("OIAnalysisResult.snapshot key does not match result key")

        if self.context.key != self.key:
            raise ValueError("OIAnalysisResult.context key does not match result key")

        self.metadata = dict(self.metadata or {})

    @property
    def key(self) -> tuple[str, str]:
        return self.exchange, self.symbol

    @property
    def has_divergence(self) -> bool:
        return self.divergence is not None and self.divergence.detected

    @property
    def has_anomaly(self) -> bool:
        return self.anomaly is not None and self.anomaly.detected

    @property
    def confidence(self) -> float:
        values = [self.regime.confidence]

        if self.divergence is not None and self.divergence.detected:
            values.append(self.divergence.confidence)

        if self.anomaly is not None and self.anomaly.detected:
            values.append(self.anomaly.confidence)

        return sum(values) / len(values)

    @property
    def confidence_band(self) -> OIConfidenceBand:
        return _confidence_to_band(self.confidence)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OIAnalysisResult:
        divergence_data = data.get("divergence")
        anomaly_data = data.get("anomaly")

        return cls(
            symbol=str(data["symbol"]),
            exchange=str(data["exchange"]),
            timestamp=float(data["timestamp"]),
            snapshot=OISnapshot.from_dict(dict(data["snapshot"])),
            context=OIMarketContext.from_dict(dict(data["context"])),
            features=OIFeatures.from_dict(dict(data["features"])),
            regime=OIRegimeResult.from_dict(dict(data["regime"])),
            divergence=(
                OIDivergenceResult.from_dict(dict(divergence_data))
                if divergence_data is not None
                else None
            ),
            anomaly=(
                OIAnomalyResult.from_dict(dict(anomaly_data))
                if anomaly_data is not None
                else None
            ),
            metadata=dict(data.get("metadata") or {}),
        )

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
            "has_divergence": self.has_divergence,
            "has_anomaly": self.has_anomaly,
            "confidence": self.confidence,
            "confidence_band": self.confidence_band.value,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class OIState:
    """
    Поточний стан по конкретному symbol/exchange, який веде OIAnalyzer.
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
        self.symbol = _normalize_symbol(self.symbol)
        self.exchange = _normalize_exchange(self.exchange)
        self.last_regime = _coerce_oi_regime(self.last_regime)
        self.last_update_ts = _safe_float(self.last_update_ts)

    @property
    def key(self) -> tuple[str, str]:
        return self.exchange, self.symbol

    @property
    def has_snapshot(self) -> bool:
        return self.last_snapshot is not None

    @property
    def has_context(self) -> bool:
        return self.last_context is not None

    @property
    def has_analysis(self) -> bool:
        return self.last_analysis is not None

    def touch(self, timestamp: float) -> None:
        timestamp = float(timestamp)
        if timestamp <= 0:
            raise ValueError("timestamp must be > 0")
        self.last_update_ts = timestamp

    def is_stale(self, now_ts: float, max_age_sec: float) -> bool:
        if max_age_sec <= 0:
            raise ValueError("max_age_sec must be > 0")

        if self.last_update_ts is None:
            return True

        return float(now_ts) - self.last_update_ts > max_age_sec

    def to_dict(self, *, include_full_analysis: bool = False) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "last_regime": self.last_regime.value,
            "last_update_ts": self.last_update_ts,
            "has_snapshot": self.has_snapshot,
            "has_context": self.has_context,
            "has_features": self.last_features is not None,
            "has_analysis": self.has_analysis,
            "last_snapshot": (
                self.last_snapshot.to_dict()
                if self.last_snapshot is not None
                else None
            ),
            "last_context": (
                self.last_context.to_dict()
                if self.last_context is not None
                else None
            ),
            "last_features": (
                self.last_features.to_dict()
                if self.last_features is not None
                else None
            ),
            "last_analysis": (
                self.last_analysis.to_dict()
                if include_full_analysis and self.last_analysis is not None
                else None
            ),
        }