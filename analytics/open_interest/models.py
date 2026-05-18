from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, TypeAlias

from .enums import (
    OIAnomalyType,
    OIConfidenceBand,
    OIDirection,
    OIDivergenceType,
    OIRegime,
    OISignalStrength,
)


DEFAULT_EXCHANGE = "unknown"
DEFAULT_MARKET_TYPE = "perpetual"
DEFAULT_TIMEFRAME = "1m"

OIKey: TypeAlias = tuple[str, str, str, str]
# exchange, market_type, symbol, timeframe


# =============================================================================
# Numeric helpers
# =============================================================================

def safe_float(value: Any, default: float | None = None) -> float | None:
    """
    Convert value to finite float.

    Returns default for:
    - None
    - unparseable values
    - NaN
    - +inf / -inf

    This helper is intentionally strict because OI features, scores and
    downstream strategy/risk logic must never receive non-finite numbers.
    """
    if value is None:
        return default

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default

    if not math.isfinite(result):
        return default

    return result


def safe_int(value: Any, default: int | None = None) -> int | None:
    """
    Convert value to int only when it is finite and parseable.
    """
    if value is None:
        return default

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default

    if not math.isfinite(result):
        return default

    try:
        return int(result)
    except (TypeError, ValueError, OverflowError):
        return default


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """
    Clamp finite numeric value to [low, high].

    Raises:
        ValueError: when bounds or value are non-finite.
    """
    if low > high:
        raise ValueError("low must be <= high")

    if not math.isfinite(float(low)) or not math.isfinite(float(high)):
        raise ValueError("clamp bounds must be finite")

    number = safe_float(value)
    if number is None:
        raise ValueError("value must be finite")

    return max(low, min(high, number))


def required_float(value: Any, field_name: str) -> float:
    """
    Convert required numeric value to finite float.
    """
    result = safe_float(value)
    if result is None:
        raise ValueError(f"{field_name} must be a finite number")
    return result


def positive_float(value: Any, field_name: str) -> float:
    """
    Convert required numeric value to finite float and require > 0.
    """
    result = required_float(value, field_name)
    if result <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return result


def non_negative_float(value: Any, field_name: str) -> float:
    """
    Convert required numeric value to finite float and require >= 0.
    """
    result = required_float(value, field_name)
    if result < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return result


def optional_non_negative_float(
    value: Any,
    field_name: str,
) -> float | None:
    """
    Convert optional numeric value and require >= 0 when present.
    """
    result = safe_float(value)
    if result is None:
        return None

    if result < 0:
        raise ValueError(f"{field_name} must be >= 0")

    return result


def optional_positive_float(
    value: Any,
    field_name: str,
) -> float | None:
    """
    Convert optional numeric value and require > 0 when present.
    """
    result = safe_float(value)
    if result is None:
        return None

    if result <= 0:
        raise ValueError(f"{field_name} must be > 0")

    return result


def normalize_optional_score(value: Any) -> float | None:
    """
    Normalize optional model score.

    Direct result construction, deserialization and tests may pass invalid
    or out-of-range values. Scores must remain finite and bounded.
    """
    score = safe_float(value)
    if score is None:
        return None
    return clamp(score)


# Backward-compatible aliases.
_safe_float = safe_float
_safe_int = safe_int
_clamp = clamp


# =============================================================================
# Scope helpers
# =============================================================================

def normalize_symbol(symbol: Any) -> str:
    normalized = str(symbol or "").upper().strip()
    if not normalized:
        raise ValueError("symbol must not be empty")
    return normalized


def normalize_exchange(exchange: Any) -> str:
    normalized = str(exchange or DEFAULT_EXCHANGE).lower().strip()
    if not normalized:
        return DEFAULT_EXCHANGE
    return normalized


def normalize_market_type(market_type: Any) -> str:
    normalized = str(market_type or DEFAULT_MARKET_TYPE).lower().strip()
    if not normalized:
        return DEFAULT_MARKET_TYPE
    return normalized


def normalize_timeframe(timeframe: Any) -> str:
    normalized = str(timeframe or DEFAULT_TIMEFRAME).strip()
    if not normalized:
        return DEFAULT_TIMEFRAME
    return normalized


def normalize_exchange_symbol(
    exchange_symbol: Any,
    *,
    fallback_symbol: str,
) -> str:
    normalized = str(exchange_symbol or "").strip()
    return normalized if normalized else fallback_symbol


def make_oi_key(
    *,
    exchange: Any = DEFAULT_EXCHANGE,
    market_type: Any = DEFAULT_MARKET_TYPE,
    symbol: Any,
    timeframe: Any = DEFAULT_TIMEFRAME,
) -> OIKey:
    return (
        normalize_exchange(exchange),
        normalize_market_type(market_type),
        normalize_symbol(symbol),
        normalize_timeframe(timeframe),
    )


def oi_key_to_dict(key: OIKey) -> dict[str, str]:
    exchange, market_type, symbol, timeframe = key
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
    }


def oi_key_to_string(key: OIKey) -> str:
    scope = oi_key_to_dict(key)
    return (
        f"{scope['exchange']}:"
        f"{scope['market_type']}:"
        f"{scope['symbol']}:"
        f"{scope['timeframe']}"
    )


def make_scope_payload(
    *,
    exchange: Any = DEFAULT_EXCHANGE,
    market_type: Any = DEFAULT_MARKET_TYPE,
    symbol: Any,
    timeframe: Any = DEFAULT_TIMEFRAME,
    exchange_symbol: Any | None = None,
) -> dict[str, Any]:
    key = make_oi_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )
    scope = oi_key_to_dict(key)

    return {
        "exchange": scope["exchange"],
        "market_type": scope["market_type"],
        "symbol": scope["symbol"],
        "timeframe": scope["timeframe"],
        "exchange_symbol": normalize_exchange_symbol(
            exchange_symbol,
            fallback_symbol=scope["symbol"],
        ),
        "scope": scope,
        "scope_key": oi_key_to_string(key),
        "oi_key": key,
        "key": list(key),
    }


def normalize_scope_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    key: OIKey,
    exchange_symbol: str,
) -> dict[str, Any]:
    result = dict(metadata or {})
    result.setdefault("scope", oi_key_to_dict(key))
    result.setdefault("scope_key", oi_key_to_string(key))
    result.setdefault("exchange_symbol", exchange_symbol)
    return result


# Backward-compatible private aliases.
_normalize_symbol = normalize_symbol
_normalize_exchange = normalize_exchange
_normalize_market_type = normalize_market_type
_normalize_timeframe = normalize_timeframe
_normalize_exchange_symbol = normalize_exchange_symbol


# =============================================================================
# Enum coercion helpers
# =============================================================================

def _confidence_to_band(confidence: float) -> OIConfidenceBand:
    confidence = clamp(confidence)

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


# =============================================================================
# Base scoped mixin
# =============================================================================

@dataclass(slots=True)
class OIScopedModel:
    """
    Shared scope behavior for Open Interest domain models.

    Canonical scope:
        exchange + market_type + symbol + timeframe
    """

    symbol: str
    exchange: str = DEFAULT_EXCHANGE
    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    def __post_init__(self) -> None:
        self.symbol = normalize_symbol(self.symbol)
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

    @property
    def key(self) -> OIKey:
        return make_oi_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def oi_key(self) -> OIKey:
        return self.key

    @property
    def scope(self) -> dict[str, str]:
        return oi_key_to_dict(self.key)

    @property
    def scope_key(self) -> str:
        return oi_key_to_string(self.key)

    @property
    def market_key(self) -> tuple[str, str, str]:
        return self.exchange, self.market_type, self.symbol

    def scope_payload(self) -> dict[str, Any]:
        return make_scope_payload(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
            exchange_symbol=self.exchange_symbol,
        )


# =============================================================================
# Core input models
# =============================================================================

@dataclass(slots=True)
class OISnapshot:
    """
    Нормалізований futures open-interest snapshot з data layer.

    Correct source:
        OpenInterestCache -> market.open_interest.updated -> OIAnalyzer

    Scope:
        exchange + market_type + symbol + timeframe

    Для самого OI біржа зазвичай не дає timeframe, але analyzer використовує
    OI разом із candle/price context, тому snapshot прив'язується до
    timeframe аналізу.
    """

    symbol: str
    exchange: str
    timestamp: float
    oi: float

    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    open_interest_value: float | None = None
    mark_price: float | None = None
    index_price: float | None = None

    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = normalize_symbol(self.symbol)
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

        self.timestamp = positive_float(
            self.timestamp,
            "OISnapshot.timestamp",
        )
        self.oi = non_negative_float(
            self.oi,
            "OISnapshot.oi",
        )
        self.open_interest_value = optional_non_negative_float(
            self.open_interest_value,
            "OISnapshot.open_interest_value",
        )
        self.mark_price = optional_positive_float(
            self.mark_price,
            "OISnapshot.mark_price",
        )
        self.index_price = optional_positive_float(
            self.index_price,
            "OISnapshot.index_price",
        )

        self.metadata = normalize_scope_metadata(
            self.metadata,
            key=self.key,
            exchange_symbol=self.exchange_symbol,
        )

    @property
    def key(self) -> OIKey:
        return make_oi_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def oi_key(self) -> OIKey:
        return self.key

    @property
    def scope(self) -> dict[str, str]:
        return oi_key_to_dict(self.key)

    @property
    def scope_key(self) -> str:
        return oi_key_to_string(self.key)

    @property
    def market_key(self) -> tuple[str, str, str]:
        return self.exchange, self.market_type, self.symbol

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OISnapshot:
        oi_value = (
            data.get("oi")
            if data.get("oi") is not None
            else data.get("open_interest")
        )

        timestamp = (
            data.get("timestamp")
            if data.get("timestamp") is not None
            else data.get("timestamp_ms")
        )

        return cls(
            symbol=str(data["symbol"]),
            exchange=str(data.get("exchange") or DEFAULT_EXCHANGE),
            market_type=str(data.get("market_type") or data.get("category") or DEFAULT_MARKET_TYPE),
            timeframe=str(data.get("timeframe") or DEFAULT_TIMEFRAME),
            exchange_symbol=data.get("exchange_symbol"),
            timestamp=timestamp,
            oi=oi_value,
            open_interest_value=data.get("open_interest_value"),
            mark_price=data.get("mark_price"),
            index_price=data.get("index_price"),
            source=data.get("source"),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.scope_payload(),
            "timestamp": self.timestamp,
            "oi": self.oi,
            "open_interest": self.oi,
            "open_interest_value": self.open_interest_value,
            "mark_price": self.mark_price,
            "index_price": self.index_price,
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    def scope_payload(self) -> dict[str, Any]:
        return make_scope_payload(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
            exchange_symbol=self.exchange_symbol,
        )


@dataclass(slots=True)
class OIMarketContext:
    """
    Futures market context на момент оцінки Open Interest.

    Джерела:
    - CandlesCache -> market.candle.closed / market.candles.updated
    - TradesCache -> market.trades.updated
    - FundingCache -> market.funding.updated
    - OrderflowAnalyzer -> analytics.orderflow.updated
    - LiquidationsAnalyzer -> analytics.liquidations.updated
    """

    symbol: str
    exchange: str
    timestamp: float

    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    price: float | None = None
    price_delta: float | None = None
    price_delta_pct: float | None = None

    volume: float | None = None
    quote_volume: float | None = None
    volume_ma: float | None = None
    volume_ratio: float | None = None

    funding_rate: float | None = None
    predicted_funding_rate: float | None = None
    next_funding_time_ms: float | None = None

    long_liquidations: float | None = None
    short_liquidations: float | None = None

    cvd_delta: float | None = None
    aggressive_buy_volume: float | None = None
    aggressive_sell_volume: float | None = None

    mark_price: float | None = None
    index_price: float | None = None

    source: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = normalize_symbol(self.symbol)
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

        self.timestamp = positive_float(
            self.timestamp,
            "OIMarketContext.timestamp",
        )

        for attr in (
            "price",
            "price_delta",
            "price_delta_pct",
            "volume",
            "quote_volume",
            "volume_ma",
            "volume_ratio",
            "funding_rate",
            "predicted_funding_rate",
            "next_funding_time_ms",
            "long_liquidations",
            "short_liquidations",
            "cvd_delta",
            "aggressive_buy_volume",
            "aggressive_sell_volume",
            "mark_price",
            "index_price",
        ):
            setattr(self, attr, safe_float(getattr(self, attr)))

        for attr in (
            "price",
            "volume",
            "quote_volume",
            "volume_ma",
            "long_liquidations",
            "short_liquidations",
            "aggressive_buy_volume",
            "aggressive_sell_volume",
            "mark_price",
            "index_price",
        ):
            value = getattr(self, attr)
            if value is not None and value < 0:
                raise ValueError(f"OIMarketContext.{attr} must be >= 0")

        for attr in ("price", "mark_price", "index_price"):
            value = getattr(self, attr)
            if value is not None and value <= 0:
                raise ValueError(f"OIMarketContext.{attr} must be > 0")

        self.extra = normalize_scope_metadata(
            self.extra,
            key=self.key,
            exchange_symbol=self.exchange_symbol,
        )

    @property
    def key(self) -> OIKey:
        return make_oi_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def oi_key(self) -> OIKey:
        return self.key

    @property
    def scope(self) -> dict[str, str]:
        return oi_key_to_dict(self.key)

    @property
    def scope_key(self) -> str:
        return oi_key_to_string(self.key)

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
        return required_float(now_ts, "now_ts") - self.timestamp > max_age_sec

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OIMarketContext:
        timestamp = (
            data.get("timestamp")
            if data.get("timestamp") is not None
            else data.get("timestamp_ms")
        )

        return cls(
            symbol=str(data["symbol"]),
            exchange=str(data.get("exchange") or DEFAULT_EXCHANGE),
            market_type=str(data.get("market_type") or data.get("category") or DEFAULT_MARKET_TYPE),
            timeframe=str(data.get("timeframe") or DEFAULT_TIMEFRAME),
            exchange_symbol=data.get("exchange_symbol"),
            timestamp=timestamp,
            price=data.get("price"),
            price_delta=data.get("price_delta"),
            price_delta_pct=data.get("price_delta_pct"),
            volume=data.get("volume"),
            quote_volume=data.get("quote_volume"),
            volume_ma=data.get("volume_ma"),
            volume_ratio=data.get("volume_ratio"),
            funding_rate=data.get("funding_rate"),
            predicted_funding_rate=data.get("predicted_funding_rate"),
            next_funding_time_ms=data.get("next_funding_time_ms"),
            long_liquidations=data.get("long_liquidations"),
            short_liquidations=data.get("short_liquidations"),
            cvd_delta=data.get("cvd_delta"),
            aggressive_buy_volume=data.get("aggressive_buy_volume"),
            aggressive_sell_volume=data.get("aggressive_sell_volume"),
            mark_price=data.get("mark_price"),
            index_price=data.get("index_price"),
            source=data.get("source"),
            extra=dict(data.get("extra") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.scope_payload(),
            "timestamp": self.timestamp,
            "price": self.price,
            "price_delta": self.price_delta,
            "price_delta_pct": self.price_delta_pct,
            "volume": self.volume,
            "quote_volume": self.quote_volume,
            "volume_ma": self.volume_ma,
            "volume_ratio": self.volume_ratio,
            "funding_rate": self.funding_rate,
            "predicted_funding_rate": self.predicted_funding_rate,
            "next_funding_time_ms": self.next_funding_time_ms,
            "long_liquidations": self.long_liquidations,
            "short_liquidations": self.short_liquidations,
            "liquidation_imbalance": self.liquidation_imbalance,
            "cvd_delta": self.cvd_delta,
            "aggressive_buy_volume": self.aggressive_buy_volume,
            "aggressive_sell_volume": self.aggressive_sell_volume,
            "aggressive_flow_imbalance": self.aggressive_flow_imbalance,
            "mark_price": self.mark_price,
            "index_price": self.index_price,
            "source": self.source,
            "extra": dict(self.extra),
        }

    def scope_payload(self) -> dict[str, Any]:
        return make_scope_payload(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
            exchange_symbol=self.exchange_symbol,
        )


# =============================================================================
# Feature model
# =============================================================================

@dataclass(slots=True)
class OIFeatures:
    """
    Розраховані фічі для інтерпретації futures Open Interest.

    Це pure value object. Він не має EventBus/Scheduler/logger/IO.
    """

    oi: float
    oi_delta: float
    oi_delta_pct: float

    exchange: str
    market_type: str
    symbol: str
    timeframe: str
    timestamp: float

    exchange_symbol: str | None = None

    open_interest_value: float | None = None

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
    quote_volume: float | None = None
    volume_ma: float | None = None
    volume_ratio: float | None = None

    funding_rate: float | None = None
    predicted_funding_rate: float | None = None

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

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.symbol = normalize_symbol(self.symbol)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

        self.timestamp = positive_float(
            self.timestamp,
            "OIFeatures.timestamp",
        )

        for attr in (
            "oi",
            "oi_delta",
            "oi_delta_pct",
            "open_interest_value",
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
            "quote_volume",
            "volume_ma",
            "volume_ratio",
            "funding_rate",
            "predicted_funding_rate",
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
            setattr(self, attr, safe_float(getattr(self, attr)))

        if self.oi is None:
            raise ValueError("OIFeatures.oi must not be None")
        if self.oi_delta is None:
            raise ValueError("OIFeatures.oi_delta must not be None")
        if self.oi_delta_pct is None:
            raise ValueError("OIFeatures.oi_delta_pct must not be None")

        if self.oi < 0:
            raise ValueError("OIFeatures.oi must be >= 0")

        for attr in (
            "open_interest_value",
            "price",
            "volume",
            "quote_volume",
            "volume_ma",
            "long_liquidations",
            "short_liquidations",
            "aggressive_buy_volume",
            "aggressive_sell_volume",
        ):
            value = getattr(self, attr)
            if value is not None and value < 0:
                raise ValueError(f"OIFeatures.{attr} must be >= 0")

        if self.price is not None and self.price <= 0:
            raise ValueError("OIFeatures.price must be > 0")

        if self.liquidation_imbalance is not None:
            self.liquidation_imbalance = clamp(
                self.liquidation_imbalance,
                low=-1.0,
                high=1.0,
            )

        if self.aggressive_flow_imbalance is not None:
            self.aggressive_flow_imbalance = clamp(
                self.aggressive_flow_imbalance,
                low=-1.0,
                high=1.0,
            )

        if self.oi_pressure_score is not None:
            self.oi_pressure_score = clamp(
                self.oi_pressure_score,
                low=-1.0,
                high=1.0,
            )

        self.oi_direction = _coerce_oi_direction(self.oi_direction)
        self.price_direction = _coerce_oi_direction(self.price_direction)
        self.metadata = normalize_scope_metadata(
            self.metadata,
            key=self.key,
            exchange_symbol=self.exchange_symbol,
        )

    @property
    def key(self) -> OIKey:
        return make_oi_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def oi_key(self) -> OIKey:
        return self.key

    @property
    def scope(self) -> dict[str, str]:
        return oi_key_to_dict(self.key)

    @property
    def scope_key(self) -> str:
        return oi_key_to_string(self.key)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OIFeatures:
        timestamp = (
            data.get("timestamp")
            if data.get("timestamp") is not None
            else data.get("timestamp_ms")
        )

        return cls(
            exchange=str(data.get("exchange") or DEFAULT_EXCHANGE),
            market_type=str(data.get("market_type") or data.get("category") or DEFAULT_MARKET_TYPE),
            symbol=str(data["symbol"]),
            timeframe=str(data.get("timeframe") or DEFAULT_TIMEFRAME),
            exchange_symbol=data.get("exchange_symbol"),
            timestamp=timestamp,
            oi=data["oi"],
            oi_delta=data["oi_delta"],
            oi_delta_pct=data["oi_delta_pct"],
            open_interest_value=data.get("open_interest_value"),
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
            quote_volume=data.get("quote_volume"),
            volume_ma=data.get("volume_ma"),
            volume_ratio=data.get("volume_ratio"),
            funding_rate=data.get("funding_rate"),
            predicted_funding_rate=data.get("predicted_funding_rate"),
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
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.scope_payload(),
            "timestamp": self.timestamp,
            "oi": self.oi,
            "oi_delta": self.oi_delta,
            "oi_delta_pct": self.oi_delta_pct,
            "open_interest_value": self.open_interest_value,
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
            "quote_volume": self.quote_volume,
            "volume_ma": self.volume_ma,
            "volume_ratio": self.volume_ratio,
            "funding_rate": self.funding_rate,
            "predicted_funding_rate": self.predicted_funding_rate,
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
            "metadata": dict(self.metadata),
        }

    def scope_payload(self) -> dict[str, Any]:
        return make_scope_payload(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
            exchange_symbol=self.exchange_symbol,
        )


# =============================================================================
# Result models
# =============================================================================

@dataclass(slots=True)
class OIRegimeResult:
    regime: OIRegime
    confidence: float
    reasons: list[str] = field(default_factory=list)
    score: float | None = None

    def __post_init__(self) -> None:
        self.regime = _coerce_oi_regime(self.regime)
        self.confidence = clamp(
            required_float(
                self.confidence,
                "OIRegimeResult.confidence",
            )
        )
        self.score = normalize_optional_score(self.score)
        self.reasons = list(dict.fromkeys(self.reasons or []))

    @property
    def confidence_band(self) -> OIConfidenceBand:
        return _confidence_to_band(self.confidence)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OIRegimeResult:
        return cls(
            regime=_coerce_oi_regime(data["regime"]),
            confidence=data.get("confidence", 0.0),
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
        self.confidence = clamp(
            required_float(
                self.confidence,
                "OIDivergenceResult.confidence",
            )
        )
        self.window_size = safe_int(self.window_size)
        self.score = normalize_optional_score(self.score)
        self.reasons = list(dict.fromkeys(self.reasons or []))

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
            confidence=data.get("confidence", 0.0),
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
        self.confidence = clamp(
            required_float(
                self.confidence,
                "OIAnomalyResult.confidence",
            )
        )
        self.score = normalize_optional_score(self.score)
        self.reasons = list(dict.fromkeys(self.reasons or []))

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
            confidence=data.get("confidence", 0.0),
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
    Фінальний результат повного futures Open Interest аналізу.
    """

    symbol: str
    exchange: str
    timestamp: float

    snapshot: OISnapshot
    context: OIMarketContext
    features: OIFeatures
    regime: OIRegimeResult

    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    divergence: OIDivergenceResult | None = None
    anomaly: OIAnomalyResult | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = normalize_symbol(self.symbol)
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

        self.timestamp = positive_float(
            self.timestamp,
            "OIAnalysisResult.timestamp",
        )

        if self.snapshot.key != self.key:
            raise ValueError(
                "OIAnalysisResult.snapshot key does not match result key: "
                f"snapshot={oi_key_to_dict(self.snapshot.key)} result={self.scope}"
            )

        if self.context.key != self.key:
            raise ValueError(
                "OIAnalysisResult.context key does not match result key: "
                f"context={oi_key_to_dict(self.context.key)} result={self.scope}"
            )

        if self.features.key != self.key:
            raise ValueError(
                "OIAnalysisResult.features key does not match result key: "
                f"features={oi_key_to_dict(self.features.key)} result={self.scope}"
            )

        self.metadata = normalize_scope_metadata(
            self.metadata,
            key=self.key,
            exchange_symbol=self.exchange_symbol,
        )

    @property
    def key(self) -> OIKey:
        return make_oi_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def oi_key(self) -> OIKey:
        return self.key

    @property
    def scope(self) -> dict[str, str]:
        return oi_key_to_dict(self.key)

    @property
    def scope_key(self) -> str:
        return oi_key_to_string(self.key)

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
            exchange=str(data.get("exchange") or DEFAULT_EXCHANGE),
            market_type=str(data.get("market_type") or data.get("category") or DEFAULT_MARKET_TYPE),
            symbol=str(data["symbol"]),
            timeframe=str(data.get("timeframe") or DEFAULT_TIMEFRAME),
            exchange_symbol=data.get("exchange_symbol"),
            timestamp=data["timestamp"],
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
            **self.scope_payload(),
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

    def scope_payload(self) -> dict[str, Any]:
        return make_scope_payload(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
            exchange_symbol=self.exchange_symbol,
        )


# =============================================================================
# Runtime state model
# =============================================================================

@dataclass(slots=True)
class OIState:
    """
    Поточний стан по конкретному futures scope:

        exchange + market_type + symbol + timeframe

    Цей клас веде OIAnalyzer. Він не має EventBus/Scheduler/logger.
    """

    symbol: str
    exchange: str

    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    last_snapshot: OISnapshot | None = None
    last_context: OIMarketContext | None = None
    last_features: OIFeatures | None = None
    last_analysis: OIAnalysisResult | None = None
    last_regime: OIRegime = OIRegime.NEUTRAL

    last_update_ts: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = normalize_symbol(self.symbol)
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

        self.last_regime = _coerce_oi_regime(self.last_regime)
        self.last_update_ts = safe_float(self.last_update_ts)

        self.metadata = normalize_scope_metadata(
            self.metadata,
            key=self.key,
            exchange_symbol=self.exchange_symbol,
        )

        self._validate_existing_children()

    @property
    def key(self) -> OIKey:
        return make_oi_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def oi_key(self) -> OIKey:
        return self.key

    @property
    def scope(self) -> dict[str, str]:
        return oi_key_to_dict(self.key)

    @property
    def scope_key(self) -> str:
        return oi_key_to_string(self.key)

    @property
    def has_snapshot(self) -> bool:
        return self.last_snapshot is not None

    @property
    def has_context(self) -> bool:
        return self.last_context is not None

    @property
    def has_features(self) -> bool:
        return self.last_features is not None

    @property
    def has_analysis(self) -> bool:
        return self.last_analysis is not None

    def apply_snapshot(self, snapshot: OISnapshot) -> None:
        if snapshot.key != self.key:
            raise ValueError(
                "OISnapshot key does not match OIState key: "
                f"snapshot={oi_key_to_dict(snapshot.key)} state={self.scope}"
            )
        self.last_snapshot = snapshot
        self.exchange_symbol = snapshot.exchange_symbol
        self.touch(snapshot.timestamp)

    def apply_context(self, context: OIMarketContext) -> None:
        if context.key != self.key:
            raise ValueError(
                "OIMarketContext key does not match OIState key: "
                f"context={oi_key_to_dict(context.key)} state={self.scope}"
            )
        self.last_context = context
        self.exchange_symbol = context.exchange_symbol
        self.touch(context.timestamp)

    def apply_features(self, features: OIFeatures) -> None:
        if features.key != self.key:
            raise ValueError(
                "OIFeatures key does not match OIState key: "
                f"features={oi_key_to_dict(features.key)} state={self.scope}"
            )
        self.last_features = features
        self.exchange_symbol = features.exchange_symbol
        self.touch(features.timestamp)

    def apply_analysis(self, analysis: OIAnalysisResult) -> None:
        if analysis.key != self.key:
            raise ValueError(
                "OIAnalysisResult key does not match OIState key: "
                f"analysis={oi_key_to_dict(analysis.key)} state={self.scope}"
            )

        self.last_analysis = analysis
        self.last_snapshot = analysis.snapshot
        self.last_context = analysis.context
        self.last_features = analysis.features
        self.last_regime = analysis.regime.regime
        self.exchange_symbol = analysis.exchange_symbol
        self.touch(analysis.timestamp)

    def touch(self, timestamp: float) -> None:
        self.last_update_ts = positive_float(
            timestamp,
            "OIState.last_update_ts",
        )

    def is_stale(self, now_ts: float, max_age_sec: float) -> bool:
        if max_age_sec <= 0:
            raise ValueError("max_age_sec must be > 0")

        if self.last_update_ts is None:
            return True

        return required_float(now_ts, "now_ts") - self.last_update_ts > max_age_sec

    def reset_runtime(self) -> None:
        self.last_snapshot = None
        self.last_context = None
        self.last_features = None
        self.last_analysis = None
        self.last_regime = OIRegime.NEUTRAL
        self.last_update_ts = None
        self.metadata.clear()
        self.metadata.update(
            normalize_scope_metadata(
                {},
                key=self.key,
                exchange_symbol=self.exchange_symbol or self.symbol,
            )
        )

    def to_dict(self, *, include_full_analysis: bool = False) -> dict[str, Any]:
        return {
            **self.scope_payload(),
            "last_regime": self.last_regime.value,
            "last_update_ts": self.last_update_ts,
            "has_snapshot": self.has_snapshot,
            "has_context": self.has_context,
            "has_features": self.has_features,
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
            "metadata": dict(self.metadata),
        }

    def scope_payload(self) -> dict[str, Any]:
        return make_scope_payload(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
            exchange_symbol=self.exchange_symbol,
        )

    def _validate_existing_children(self) -> None:
        if self.last_snapshot is not None and self.last_snapshot.key != self.key:
            raise ValueError(
                "last_snapshot key does not match OIState key: "
                f"snapshot={oi_key_to_dict(self.last_snapshot.key)} state={self.scope}"
            )

        if self.last_context is not None and self.last_context.key != self.key:
            raise ValueError(
                "last_context key does not match OIState key: "
                f"context={oi_key_to_dict(self.last_context.key)} state={self.scope}"
            )

        if self.last_features is not None and self.last_features.key != self.key:
            raise ValueError(
                "last_features key does not match OIState key: "
                f"features={oi_key_to_dict(self.last_features.key)} state={self.scope}"
            )

        if self.last_analysis is not None and self.last_analysis.key != self.key:
            raise ValueError(
                "last_analysis key does not match OIState key: "
                f"analysis={oi_key_to_dict(self.last_analysis.key)} state={self.scope}"
            )


# =============================================================================
# Generic payload helper
# =============================================================================

def model_to_payload(model: Any) -> dict[str, Any]:
    """
    Єдиний helper для EventBus/storage/dashboard serialization.
    """
    if hasattr(model, "to_dict") and callable(model.to_dict):
        return model.to_dict()

    if hasattr(model, "to_payload") and callable(model.to_payload):
        payload = model.to_payload()
        if isinstance(payload, Mapping):
            return dict(payload)

    if isinstance(model, Mapping):
        return dict(model)

    raise TypeError(f"Unsupported OI model type: {type(model)!r}")


__all__ = [
    "DEFAULT_EXCHANGE",
    "DEFAULT_MARKET_TYPE",
    "DEFAULT_TIMEFRAME",
    "OIKey",

    # helpers
    "safe_float",
    "safe_int",
    "clamp",
    "required_float",
    "positive_float",
    "non_negative_float",
    "optional_non_negative_float",
    "optional_positive_float",
    "normalize_optional_score",
    "normalize_symbol",
    "normalize_exchange",
    "normalize_market_type",
    "normalize_timeframe",
    "normalize_exchange_symbol",
    "make_oi_key",
    "oi_key_to_dict",
    "oi_key_to_string",
    "make_scope_payload",
    "normalize_scope_metadata",
    "model_to_payload",

    # scoped base
    "OIScopedModel",

    # models
    "OISnapshot",
    "OIMarketContext",
    "OIFeatures",
    "OIRegimeResult",
    "OIDivergenceResult",
    "OIAnomalyResult",
    "OIAnalysisResult",
    "OIState",
]