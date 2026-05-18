from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Mapping, TypeAlias

from .enums import (
    CascadeDirection,
    CascadeSeverity,
    LiquidationEventType,
    LiquidationSide,
    LiquidationStatus,
)


DECIMAL_ZERO = Decimal("0")
DEFAULT_LARGE_LIQUIDATION_THRESHOLD_USD = Decimal("100000")

DEFAULT_MARKET_TYPE = "perpetual"
DEFAULT_TIMEFRAME = "realtime"
DEFAULT_EXCHANGE_SYMBOL = ""

LiquidationKey: TypeAlias = tuple[str, str, str, str]
# exchange, market_type, symbol, timeframe


# =============================================================================
# Time / serialization helpers
# =============================================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _utc_now() -> datetime:
    """
    Backward-compatible local default_factory alias.
    """
    return utc_now()


def _ensure_utc(dt: datetime) -> datetime:
    """
    Backward-compatible local datetime normalization alias.
    """
    return ensure_utc(dt)


def _decimal_to_str(value: Any) -> Any:
    """
    JSON-friendly serializer для Decimal / datetime / enum / dataclass / dict / list.
    """
    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, datetime):
        return ensure_utc(value).isoformat()

    if hasattr(value, "value"):
        return value.value

    if is_dataclass(value):
        return _decimal_to_str(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): _decimal_to_str(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_decimal_to_str(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_decimal_to_str(item) for item in value)

    return value


# =============================================================================
# Scope / normalization helpers
# =============================================================================


def normalize_exchange(exchange: object | None) -> str:
    value = str(exchange or "").strip().lower()
    if not value:
        raise ValueError("exchange must not be empty")
    return value


def normalize_symbol(symbol: object | None) -> str:
    value = (
        str(symbol or "")
        .strip()
        .upper()
        .replace("-", "")
        .replace("/", "")
        .replace("_", "")
    )
    if not value:
        raise ValueError("symbol must not be empty")
    return value


def normalize_market_type(market_type: object | None = None) -> str:
    value = str(market_type or DEFAULT_MARKET_TYPE).strip().lower()
    return value or DEFAULT_MARKET_TYPE


def normalize_timeframe(timeframe: object | None = None) -> str:
    value = str(timeframe or DEFAULT_TIMEFRAME).strip().lower()
    return value or DEFAULT_TIMEFRAME


def normalize_exchange_symbol(
    exchange_symbol: object | None,
    *,
    fallback_symbol: str,
) -> str:
    value = str(exchange_symbol or "").strip()
    return value or fallback_symbol


def make_liquidation_key(
    *,
    exchange: object | None,
    market_type: object | None,
    symbol: object,
    timeframe: object | None = None,
) -> LiquidationKey:
    return (
        normalize_exchange(exchange),
        normalize_market_type(market_type),
        normalize_symbol(symbol),
        normalize_timeframe(timeframe),
    )


def liquidation_key_to_dict(key: LiquidationKey) -> dict[str, str]:
    exchange, market_type, symbol, timeframe = key
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
    }


def scoped_metadata(
    *,
    exchange: object | None,
    market_type: object | None,
    symbol: object,
    timeframe: object | None = None,
    exchange_symbol: object | None = None,
) -> dict[str, str]:
    key = make_liquidation_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )
    scope = liquidation_key_to_dict(key)
    scope["exchange_symbol"] = normalize_exchange_symbol(
        exchange_symbol,
        fallback_symbol=scope["symbol"],
    )
    return scope


def _normalize_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    key: LiquidationKey,
    exchange_symbol: str,
) -> dict[str, Any]:
    result = dict(metadata or {})
    result.setdefault("scope", liquidation_key_to_dict(key))
    result.setdefault("exchange_symbol", exchange_symbol)
    return result


# =============================================================================
# Base scoped model
# =============================================================================


@dataclass(slots=True)
class LiquidationScopedModel:
    """
    Базова модель для liquidation scope.

    Canonical scope:
        exchange + market_type + symbol + timeframe

    `exchange_symbol` зберігає нативний символ біржі:
        - Binance USDM: BTCUSDT
        - Binance COINM: BTCUSD_PERP / BTCUSD
        - OKX swap: BTC-USDT-SWAP
        - Bybit linear: BTCUSDT
    """

    exchange: str
    symbol: str
    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.exchange = normalize_exchange(self.exchange)
        self.symbol = normalize_symbol(self.symbol)
        self.market_type = normalize_market_type(self.market_type)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )
        self.metadata = _normalize_metadata(
            self.metadata,
            key=self.key,
            exchange_symbol=self.exchange_symbol,
        )

    @property
    def normalized_exchange(self) -> str:
        return self.exchange

    @property
    def normalized_symbol(self) -> str:
        return self.symbol

    @property
    def key(self) -> LiquidationKey:
        return make_liquidation_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def liquidation_key(self) -> LiquidationKey:
        return self.key

    @property
    def symbol_key(self) -> tuple[str, str]:
        """
        Backward-compatible alias.

        Новий код має використовувати `.key`.
        """
        return self.normalized_exchange, self.normalized_symbol

    @property
    def scope(self) -> dict[str, str]:
        return scoped_metadata(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
            exchange_symbol=self.exchange_symbol,
        )

    def _base_payload(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "exchange_symbol": self.exchange_symbol,
            "scope": liquidation_key_to_dict(self.key),
            "metadata": dict(self.metadata),
        }


# =============================================================================
# Domain models
# =============================================================================


@dataclass(slots=True, frozen=True)
class LiquidationEvent:
    """
    Нормалізована атомарна liquidation-подія.

    Створюється на stream/ingestion рівні після парсингу raw payload біржі.

    Correct production flow:
        exchanges/*_ws.py
            -> EventBus.emit("market.liquidation", raw normalized payload)
            -> LiquidationStream
            -> LiquidationEvent
            -> market.liquidation.normalized / market.liquidations.updated

    Ця модель:
    - не публікує події самостійно;
    - не знає про EventBus;
    - не читає біржі;
    - не містить торгової логіки.
    """

    exchange: str
    symbol: str
    side: LiquidationSide
    price: Decimal
    quantity: Decimal
    notional_usd: Decimal
    timestamp: datetime

    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    event_type: LiquidationEventType = LiquidationEventType.NORMALIZED

    trade_id: str | None = None
    order_id: str | None = None
    event_id: str | None = None
    correlation_id: str | None = None

    source: str | None = None
    received_at: datetime = field(default_factory=utc_now)

    raw_payload_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_exchange = normalize_exchange(self.exchange)
        normalized_symbol = normalize_symbol(self.symbol)
        normalized_market_type = normalize_market_type(self.market_type)
        normalized_timeframe = normalize_timeframe(self.timeframe)
        normalized_exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=normalized_symbol,
        )

        key = make_liquidation_key(
            exchange=normalized_exchange,
            market_type=normalized_market_type,
            symbol=normalized_symbol,
            timeframe=normalized_timeframe,
        )

        object.__setattr__(self, "exchange", normalized_exchange)
        object.__setattr__(self, "symbol", normalized_symbol)
        object.__setattr__(self, "market_type", normalized_market_type)
        object.__setattr__(self, "timeframe", normalized_timeframe)
        object.__setattr__(self, "exchange_symbol", normalized_exchange_symbol)

        object.__setattr__(self, "price", Decimal(str(self.price)))
        object.__setattr__(self, "quantity", Decimal(str(self.quantity)))
        object.__setattr__(self, "notional_usd", Decimal(str(self.notional_usd)))
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        object.__setattr__(self, "received_at", ensure_utc(self.received_at))

        object.__setattr__(
            self,
            "metadata",
            _normalize_metadata(
                self.metadata,
                key=key,
                exchange_symbol=normalized_exchange_symbol,
            ),
        )

    @property
    def normalized_exchange(self) -> str:
        return self.exchange

    @property
    def normalized_symbol(self) -> str:
        return self.symbol

    @property
    def key(self) -> LiquidationKey:
        return make_liquidation_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def liquidation_key(self) -> LiquidationKey:
        return self.key

    @property
    def symbol_key(self) -> tuple[str, str]:
        """
        Backward-compatible alias.

        Новий код має використовувати `.key`, щоб не змішувати market_type/timeframe.
        """
        return self.normalized_exchange, self.normalized_symbol

    @property
    def scope(self) -> dict[str, str]:
        return scoped_metadata(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
            exchange_symbol=self.exchange_symbol,
        )

    @property
    def is_large(self) -> bool:
        """
        Backward-compatible helper.

        Для production threshold краще використовувати:
        LiquidationStreamConfig.large_liquidation_threshold_usd.
        """
        return self.notional_usd >= DEFAULT_LARGE_LIQUIDATION_THRESHOLD_USD

    def is_large_at(self, threshold_usd: Decimal) -> bool:
        threshold = Decimal(str(threshold_usd))
        return threshold > DECIMAL_ZERO and self.notional_usd >= threshold

    @property
    def is_valid(self) -> bool:
        return (
            bool(self.normalized_exchange)
            and bool(self.normalized_symbol)
            and bool(self.market_type)
            and bool(self.timeframe)
            and self.side.is_known
            and self.price > DECIMAL_ZERO
            and self.quantity > DECIMAL_ZERO
            and self.notional_usd > DECIMAL_ZERO
        )

    @property
    def pressure_direction(self) -> CascadeDirection:
        return CascadeDirection.from_side(self.side)

    @property
    def age_seconds(self) -> float:
        now = utc_now()
        return max(0.0, (now - ensure_utc(self.timestamp)).total_seconds())

    def to_dict(self, *, serialize: bool = True) -> dict[str, Any]:
        data = asdict(self)

        data["side"] = self.side.value
        data["event_type"] = self.event_type.value
        data["pressure_direction"] = self.pressure_direction.value
        data["normalized_exchange"] = self.normalized_exchange
        data["normalized_symbol"] = self.normalized_symbol
        data["scope"] = liquidation_key_to_dict(self.key)
        data["liquidation_key"] = self.key

        if serialize:
            return _decimal_to_str(data)

        return data


@dataclass(slots=True)
class LiquidationCluster(LiquidationScopedModel):
    """
    Агрегований кластер liquidation events у часовому вікні.

    Це доменна модель для detector-а. Вона не виконує detection самостійно,
    а лише зберігає результат агрегації.
    """

    side: LiquidationSide = LiquidationSide.UNKNOWN
    start_time: datetime = field(default_factory=utc_now)
    end_time: datetime = field(default_factory=utc_now)

    event_count: int = 0
    total_notional_usd: Decimal = DECIMAL_ZERO
    total_quantity: Decimal = DECIMAL_ZERO

    avg_price: Decimal = DECIMAL_ZERO
    min_price: Decimal = DECIMAL_ZERO
    max_price: Decimal = DECIMAL_ZERO

    direction: CascadeDirection = CascadeDirection.UNKNOWN

    severity: CascadeSeverity = CascadeSeverity.LOW
    status: LiquidationStatus = LiquidationStatus.NEW

    cluster_id: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        LiquidationScopedModel.__post_init__(self)

        self.start_time = ensure_utc(self.start_time)
        self.end_time = ensure_utc(self.end_time)

        self.event_count = max(0, int(self.event_count))
        self.total_notional_usd = Decimal(str(self.total_notional_usd))
        self.total_quantity = Decimal(str(self.total_quantity))
        self.avg_price = Decimal(str(self.avg_price))
        self.min_price = Decimal(str(self.min_price))
        self.max_price = Decimal(str(self.max_price))

    @property
    def duration_seconds(self) -> float:
        return max(
            0.0,
            (ensure_utc(self.end_time) - ensure_utc(self.start_time)).total_seconds(),
        )

    @property
    def price_range(self) -> Decimal:
        return max(DECIMAL_ZERO, self.max_price - self.min_price)

    @property
    def price_range_pct(self) -> float:
        if self.min_price <= DECIMAL_ZERO:
            return 0.0
        return float((self.max_price - self.min_price) / self.min_price) * 100.0

    @property
    def avg_notional_per_event(self) -> Decimal:
        if self.event_count <= 0:
            return DECIMAL_ZERO
        return self.total_notional_usd / Decimal(self.event_count)

    @property
    def is_confirmed(self) -> bool:
        return self.status is LiquidationStatus.CONFIRMED

    @property
    def is_actionable_severity(self) -> bool:
        return self.severity.is_actionable

    def to_dict(self, *, serialize: bool = True) -> dict[str, Any]:
        data = asdict(self)

        data["side"] = self.side.value
        data["direction"] = self.direction.value
        data["severity"] = self.severity.value
        data["status"] = self.status.value
        data["duration_seconds"] = self.duration_seconds
        data["price_range"] = self.price_range
        data["price_range_pct"] = self.price_range_pct
        data["avg_notional_per_event"] = self.avg_notional_per_event
        data["normalized_exchange"] = self.normalized_exchange
        data["normalized_symbol"] = self.normalized_symbol
        data["scope"] = liquidation_key_to_dict(self.key)
        data["liquidation_key"] = self.key

        if serialize:
            return _decimal_to_str(data)

        return data


@dataclass(slots=True)
class CascadeDetectionResult(LiquidationScopedModel):
    """
    Результат детекції liquidation cascade.

    Це analytics-level висновок detector-а. Strategy/Risk мають сприймати його
    як вхідний аналітичний сигнал, а не як готове торгове рішення.
    """

    side: LiquidationSide = LiquidationSide.UNKNOWN
    direction: CascadeDirection = CascadeDirection.UNKNOWN
    detected_at: datetime = field(default_factory=utc_now)
    cluster: LiquidationCluster | None = None

    intensity_score: float = 0.0
    confidence: float = 0.0
    continuation_bias: float = 0.0
    exhaustion_bias: float = 0.0

    event_count: int = 0
    total_notional_usd: Decimal = DECIMAL_ZERO
    window_seconds: int = 0
    price_range_pct: float = 0.0

    severity: CascadeSeverity = CascadeSeverity.LOW
    status: LiquidationStatus = LiquidationStatus.CONFIRMED

    signal_id: str | None = None
    correlation_id: str | None = None
    source: str | None = "cascade_detector"

    def __post_init__(self) -> None:
        LiquidationScopedModel.__post_init__(self)

        self.detected_at = ensure_utc(self.detected_at)
        self.intensity_score = max(0.0, min(1.0, float(self.intensity_score)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.continuation_bias = max(0.0, min(1.0, float(self.continuation_bias)))
        self.exhaustion_bias = max(0.0, min(1.0, float(self.exhaustion_bias)))
        self.event_count = max(0, int(self.event_count))
        self.total_notional_usd = Decimal(str(self.total_notional_usd))
        self.window_seconds = max(0, int(self.window_seconds))
        self.price_range_pct = max(0.0, float(self.price_range_pct))

        if self.cluster is not None and self.cluster.key != self.key:
            raise ValueError(
                "CascadeDetectionResult cluster scope mismatch: "
                f"result={liquidation_key_to_dict(self.key)} "
                f"cluster={liquidation_key_to_dict(self.cluster.key)}"
            )

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.8

    @property
    def is_confirmed(self) -> bool:
        return self.status is LiquidationStatus.CONFIRMED

    @property
    def is_actionable_severity(self) -> bool:
        return self.severity.is_actionable

    @property
    def favors_continuation(self) -> bool:
        return self.continuation_bias > self.exhaustion_bias

    @property
    def favors_exhaustion(self) -> bool:
        return self.exhaustion_bias > self.continuation_bias

    @property
    def bias_delta(self) -> float:
        return abs(self.continuation_bias - self.exhaustion_bias)

    @property
    def event_type(self) -> LiquidationEventType:
        if self.favors_exhaustion:
            return LiquidationEventType.EXHAUSTION
        return LiquidationEventType.CASCADE

    def to_dict(self, *, serialize: bool = True) -> dict[str, Any]:
        data = asdict(self)

        data["side"] = self.side.value
        data["direction"] = self.direction.value
        data["severity"] = self.severity.value
        data["status"] = self.status.value
        data["event_type"] = self.event_type.value
        data["is_high_confidence"] = self.is_high_confidence
        data["is_actionable_severity"] = self.is_actionable_severity
        data["favors_continuation"] = self.favors_continuation
        data["favors_exhaustion"] = self.favors_exhaustion
        data["bias_delta"] = self.bias_delta
        data["normalized_exchange"] = self.normalized_exchange
        data["normalized_symbol"] = self.normalized_symbol
        data["scope"] = liquidation_key_to_dict(self.key)
        data["liquidation_key"] = self.key

        if serialize:
            return _decimal_to_str(data)

        return data


@dataclass(slots=True)
class LiquidationWindowStats(LiquidationScopedModel):
    """
    Статистика по liquidation events у конкретному sliding window.

    Scope має відповідати одному:
        exchange + market_type + symbol + timeframe
    """

    window_start: datetime = field(default_factory=utc_now)
    window_end: datetime = field(default_factory=utc_now)

    total_events: int = 0
    long_events: int = 0
    short_events: int = 0

    total_notional_usd: Decimal = DECIMAL_ZERO
    long_notional_usd: Decimal = DECIMAL_ZERO
    short_notional_usd: Decimal = DECIMAL_ZERO

    min_price: Decimal | None = None
    max_price: Decimal | None = None

    def __post_init__(self) -> None:
        LiquidationScopedModel.__post_init__(self)

        self.window_start = ensure_utc(self.window_start)
        self.window_end = ensure_utc(self.window_end)

        self.total_events = max(0, int(self.total_events))
        self.long_events = max(0, int(self.long_events))
        self.short_events = max(0, int(self.short_events))

        self.total_notional_usd = Decimal(str(self.total_notional_usd))
        self.long_notional_usd = Decimal(str(self.long_notional_usd))
        self.short_notional_usd = Decimal(str(self.short_notional_usd))

        if self.min_price is not None:
            self.min_price = Decimal(str(self.min_price))
        if self.max_price is not None:
            self.max_price = Decimal(str(self.max_price))

    @property
    def duration_seconds(self) -> float:
        return max(
            0.0,
            (ensure_utc(self.window_end) - ensure_utc(self.window_start)).total_seconds(),
        )

    @property
    def dominant_side(self) -> LiquidationSide:
        if self.long_notional_usd > self.short_notional_usd:
            return LiquidationSide.LONG
        if self.short_notional_usd > self.long_notional_usd:
            return LiquidationSide.SHORT
        return LiquidationSide.UNKNOWN

    @property
    def dominant_notional_usd(self) -> Decimal:
        if self.dominant_side is LiquidationSide.LONG:
            return self.long_notional_usd
        if self.dominant_side is LiquidationSide.SHORT:
            return self.short_notional_usd
        return DECIMAL_ZERO

    @property
    def dominant_events_count(self) -> int:
        if self.dominant_side is LiquidationSide.LONG:
            return self.long_events
        if self.dominant_side is LiquidationSide.SHORT:
            return self.short_events
        return 0

    @property
    def side_imbalance_ratio(self) -> float:
        if self.total_notional_usd <= DECIMAL_ZERO:
            return 0.0
        return float(self.dominant_notional_usd / self.total_notional_usd)

    @property
    def event_imbalance_ratio(self) -> float:
        if self.total_events <= 0:
            return 0.0
        return self.dominant_events_count / self.total_events

    @property
    def price_range(self) -> Decimal:
        if self.min_price is None or self.max_price is None:
            return DECIMAL_ZERO
        return max(DECIMAL_ZERO, self.max_price - self.min_price)

    @property
    def price_range_pct(self) -> float:
        if self.min_price is None or self.min_price <= DECIMAL_ZERO or self.max_price is None:
            return 0.0
        return float((self.max_price - self.min_price) / self.min_price) * 100.0

    @property
    def avg_notional_per_event(self) -> Decimal:
        if self.total_events <= 0:
            return DECIMAL_ZERO
        return self.total_notional_usd / Decimal(self.total_events)

    @property
    def has_known_dominant_side(self) -> bool:
        return self.dominant_side.is_known

    def to_dict(self, *, serialize: bool = True) -> dict[str, Any]:
        data = asdict(self)

        data["dominant_side"] = self.dominant_side.value
        data["dominant_notional_usd"] = self.dominant_notional_usd
        data["dominant_events_count"] = self.dominant_events_count
        data["side_imbalance_ratio"] = self.side_imbalance_ratio
        data["event_imbalance_ratio"] = self.event_imbalance_ratio
        data["price_range"] = self.price_range
        data["price_range_pct"] = self.price_range_pct
        data["avg_notional_per_event"] = self.avg_notional_per_event
        data["duration_seconds"] = self.duration_seconds
        data["normalized_exchange"] = self.normalized_exchange
        data["normalized_symbol"] = self.normalized_symbol
        data["scope"] = liquidation_key_to_dict(self.key)
        data["liquidation_key"] = self.key

        if serialize:
            return _decimal_to_str(data)

        return data


@dataclass(slots=True)
class LiquidationBufferSnapshot(LiquidationScopedModel):
    """
    Знімок буфера/state для діагностики, dashboard, storage та metrics.
    """

    total_buffered_events: int = 0
    long_buffered_events: int = 0
    short_buffered_events: int = 0

    first_event_at: datetime | None = None
    last_event_at: datetime | None = None
    last_cascade_at: datetime | None = None
    cooldown_until: datetime | None = None

    max_events: int | None = None
    total_events_seen: int = 0

    def __post_init__(self) -> None:
        LiquidationScopedModel.__post_init__(self)

        self.total_buffered_events = max(0, int(self.total_buffered_events))
        self.long_buffered_events = max(0, int(self.long_buffered_events))
        self.short_buffered_events = max(0, int(self.short_buffered_events))
        self.total_events_seen = max(0, int(self.total_events_seen))

        if self.max_events is not None:
            self.max_events = max(0, int(self.max_events))

        if self.first_event_at is not None:
            self.first_event_at = ensure_utc(self.first_event_at)
        if self.last_event_at is not None:
            self.last_event_at = ensure_utc(self.last_event_at)
        if self.last_cascade_at is not None:
            self.last_cascade_at = ensure_utc(self.last_cascade_at)
        if self.cooldown_until is not None:
            self.cooldown_until = ensure_utc(self.cooldown_until)

    @property
    def is_empty(self) -> bool:
        return self.total_buffered_events <= 0

    @property
    def is_in_cooldown(self) -> bool:
        if self.cooldown_until is None:
            return False
        return utc_now() < ensure_utc(self.cooldown_until)

    @property
    def dominant_buffer_side(self) -> LiquidationSide:
        if self.long_buffered_events > self.short_buffered_events:
            return LiquidationSide.LONG
        if self.short_buffered_events > self.long_buffered_events:
            return LiquidationSide.SHORT
        return LiquidationSide.UNKNOWN

    def to_dict(self, *, serialize: bool = True) -> dict[str, Any]:
        data = asdict(self)

        data["dominant_buffer_side"] = self.dominant_buffer_side.value
        data["is_empty"] = self.is_empty
        data["is_in_cooldown"] = self.is_in_cooldown
        data["normalized_exchange"] = self.normalized_exchange
        data["normalized_symbol"] = self.normalized_symbol
        data["scope"] = liquidation_key_to_dict(self.key)
        data["liquidation_key"] = self.key

        if serialize:
            return _decimal_to_str(data)

        return data


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

    if is_dataclass(model):
        return _decimal_to_str(asdict(model))

    if isinstance(model, Mapping):
        return _decimal_to_str(dict(model))

    raise TypeError(f"Unsupported liquidation model type: {type(model)!r}")


__all__ = [
    "DECIMAL_ZERO",
    "DEFAULT_LARGE_LIQUIDATION_THRESHOLD_USD",
    "DEFAULT_MARKET_TYPE",
    "DEFAULT_TIMEFRAME",
    "DEFAULT_EXCHANGE_SYMBOL",
    "LiquidationKey",
    "utc_now",
    "ensure_utc",
    "normalize_exchange",
    "normalize_symbol",
    "normalize_market_type",
    "normalize_timeframe",
    "normalize_exchange_symbol",
    "make_liquidation_key",
    "liquidation_key_to_dict",
    "scoped_metadata",
    "LiquidationScopedModel",
    "LiquidationEvent",
    "LiquidationCluster",
    "CascadeDetectionResult",
    "LiquidationWindowStats",
    "LiquidationBufferSnapshot",
    "model_to_payload",
]