from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Mapping, Sequence


DEFAULT_MARKET_TYPE = "usdm_futures"
DEFAULT_TIMEFRAME = "1m"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def now_ms() -> int:
    return int(time.time() * 1000)


def ensure_aware_utc(value: datetime | int | float | str | None) -> datetime:
    if value is None:
        return utcnow()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 10_000_000_000:
            ts = ts / 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return utcnow()
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            parsed = datetime.fromisoformat(text)
            return ensure_aware_utc(parsed)
        except ValueError:
            return utcnow()
    return utcnow()


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        parsed = float(Decimal(str(value)))
    except (InvalidOperation, ValueError, TypeError):
        return default
    if parsed != parsed:  # NaN
        return default
    return parsed


def safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(Decimal(str(value)))
    except (InvalidOperation, ValueError, TypeError):
        return default


def first_present(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload.get(key) is not None:
            return payload.get(key)
    return None


def nested_get(payload: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = payload
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return default
    return current


def normalize_exchange(value: Any) -> str:
    text = str(value or "binance").strip().lower()
    return text or "binance"


def normalize_market_type(value: Any) -> str:
    text = str(value or DEFAULT_MARKET_TYPE).strip().lower()
    if text in {"perp", "perpetual", "futures", "future", "linear"}:
        return DEFAULT_MARKET_TYPE
    if text in {"swap", "usdm", "usd_m"}:
        return DEFAULT_MARKET_TYPE
    return text or DEFAULT_MARKET_TYPE


def normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def normalize_timeframe(value: Any | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_side(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    if text in {"b", "buy", "bid", "long", "buyer", "taker_buy"}:
        return "buy"
    if text in {"s", "sell", "ask", "short", "seller", "taker_sell"}:
        return "sell"
    return text or "unknown"


class MarketDataKind(str, Enum):
    TRADE = "trade"
    TRADES_BATCH = "trades_batch"
    CANDLE = "candle"
    CANDLES_BATCH = "candles_batch"
    ORDERBOOK_DELTA = "orderbook_delta"
    ORDERBOOK_SNAPSHOT = "orderbook_snapshot"
    FUNDING = "funding"
    OPEN_INTEREST = "open_interest"
    LIQUIDATION = "liquidation"
    PRICE = "price"


class DirtyReason(str, Enum):
    TRADE = "trade"
    TRADES_BATCH = "trades_batch"
    ORDERBOOK = "orderbook"
    ORDERBOOK_RESYNC_REQUIRED = "orderbook_resync_required"
    CANDLE = "candle"
    CANDLE_CLOSED = "candle_closed"
    FUNDING = "funding"
    OPEN_INTEREST = "open_interest"
    LIQUIDATION = "liquidation"
    PRICE = "price"
    WARMUP = "warmup"
    REST_SNAPSHOT = "rest_snapshot"


@dataclass(frozen=True, slots=True)
class MarketScope:
    exchange: str
    market_type: str
    symbol: str
    timeframe: str | None = None
    exchange_symbol: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "exchange", normalize_exchange(self.exchange))
        object.__setattr__(self, "market_type", normalize_market_type(self.market_type))
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        object.__setattr__(self, "timeframe", normalize_timeframe(self.timeframe))
        if self.exchange_symbol is not None:
            object.__setattr__(self, "exchange_symbol", str(self.exchange_symbol).strip().upper() or None)

    @property
    def key(self) -> str:
        timeframe = self.timeframe or "_"
        return f"{self.exchange}:{self.market_type}:{self.symbol}:{timeframe}"

    @property
    def symbol_key(self) -> str:
        return f"{self.exchange}:{self.market_type}:{self.symbol}"

    def with_timeframe(self, timeframe: str | None) -> "MarketScope":
        return MarketScope(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=timeframe,
            exchange_symbol=self.exchange_symbol,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "exchange_symbol": self.exchange_symbol or self.symbol,
            "key": [self.exchange, self.market_type, self.symbol, self.timeframe],
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        default_exchange: str = "binance",
        default_market_type: str = DEFAULT_MARKET_TYPE,
        default_timeframe: str | None = None,
    ) -> "MarketScope":
        symbol = first_present(payload, "symbol", "s", "instId", "instrument", "pair")
        exchange_symbol = first_present(payload, "exchange_symbol", "raw_symbol", "instId", "instrument")
        return cls(
            exchange=first_present(payload, "exchange", "source_exchange") or default_exchange,
            market_type=first_present(payload, "market_type", "category", "inst_type") or default_market_type,
            symbol=symbol,
            timeframe=first_present(payload, "timeframe", "interval", "tf") or default_timeframe,
            exchange_symbol=exchange_symbol,
        )


@dataclass(slots=True)
class PriceUpdate:
    scope: MarketScope
    price: float
    source: str = "unknown"
    timestamp_ms: int = field(default_factory=now_ms)
    received_at_ms: int = field(default_factory=now_ms)
    mark_price: float | None = None
    index_price: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TradeUpdate:
    scope: MarketScope
    price: float
    quantity: float
    side: str = "unknown"
    aggressor_side: str = "unknown"
    trade_id: str | None = None
    timestamp_ms: int = field(default_factory=now_ms)
    received_at_ms: int = field(default_factory=now_ms)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TradeUpdate | None":
        scope = MarketScope.from_payload(payload)
        price = safe_float(first_present(payload, "price", "p", "last_price"))
        quantity = safe_float(first_present(payload, "quantity", "qty", "q", "size", "volume"), 0.0)
        if not scope.symbol or price is None or price <= 0:
            return None
        side = normalize_side(first_present(payload, "side", "S", "direction"))
        aggressor = normalize_side(first_present(payload, "aggressor_side", "taker_side", "side", "S"))
        return cls(
            scope=scope,
            price=price,
            quantity=quantity or 0.0,
            side=side,
            aggressor_side=aggressor,
            trade_id=str(first_present(payload, "trade_id", "id", "t") or "") or None,
            timestamp_ms=safe_int(first_present(payload, "timestamp_ms", "trade_time", "event_time", "T", "E", "timestamp"), now_ms()) or now_ms(),
            received_at_ms=safe_int(payload.get("received_at_ms"), now_ms()) or now_ms(),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(slots=True)
class CandleUpdate:
    scope: MarketScope
    open_time_ms: int
    close_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool = False
    timestamp_ms: int = field(default_factory=now_ms)
    received_at_ms: int = field(default_factory=now_ms)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CandleUpdate | None":
        scope = MarketScope.from_payload(payload, default_timeframe=DEFAULT_TIMEFRAME)
        open_price = safe_float(first_present(payload, "open", "o"))
        high = safe_float(first_present(payload, "high", "h"))
        low = safe_float(first_present(payload, "low", "l"))
        close = safe_float(first_present(payload, "close", "c"))
        if not scope.symbol or open_price is None or high is None or low is None or close is None:
            return None
        open_time_ms = safe_int(first_present(payload, "open_time_ms", "open_time", "start", "t"), now_ms()) or now_ms()
        close_time_ms = safe_int(first_present(payload, "close_time_ms", "close_time", "end", "T"), open_time_ms) or open_time_ms
        return cls(
            scope=scope,
            open_time_ms=open_time_ms,
            close_time_ms=close_time_ms,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=safe_float(first_present(payload, "volume", "v"), 0.0) or 0.0,
            is_closed=bool(first_present(payload, "is_closed", "closed", "x") or False),
            timestamp_ms=safe_int(first_present(payload, "timestamp_ms", "event_time", "timestamp"), now_ms()) or now_ms(),
            received_at_ms=safe_int(payload.get("received_at_ms"), now_ms()) or now_ms(),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(slots=True)
class OrderBookDeltaUpdate:
    scope: MarketScope
    bids: list[tuple[float, float]] = field(default_factory=list)
    asks: list[tuple[float, float]] = field(default_factory=list)
    first_update_id: int | None = None
    final_update_id: int | None = None
    previous_final_update_id: int | None = None
    timestamp_ms: int = field(default_factory=now_ms)
    received_at_ms: int = field(default_factory=now_ms)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "OrderBookDeltaUpdate | None":
        scope = MarketScope.from_payload(payload)
        if not scope.symbol:
            return None
        return cls(
            scope=scope,
            bids=normalize_levels(first_present(payload, "bids", "b") or []),
            asks=normalize_levels(first_present(payload, "asks", "a") or []),
            first_update_id=safe_int(first_present(payload, "first_update_id", "U")),
            final_update_id=safe_int(first_present(payload, "final_update_id", "u", "sequence", "update_id")),
            previous_final_update_id=safe_int(first_present(payload, "previous_final_update_id", "pu", "prev_sequence")),
            timestamp_ms=safe_int(first_present(payload, "timestamp_ms", "event_time", "E", "timestamp"), now_ms()) or now_ms(),
            received_at_ms=safe_int(payload.get("received_at_ms"), now_ms()) or now_ms(),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(slots=True)
class OrderBookSnapshotUpdate:
    scope: MarketScope
    bids: list[tuple[float, float]] = field(default_factory=list)
    asks: list[tuple[float, float]] = field(default_factory=list)
    last_update_id: int | None = None
    timestamp_ms: int = field(default_factory=now_ms)
    received_at_ms: int = field(default_factory=now_ms)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "OrderBookSnapshotUpdate | None":
        scope = MarketScope.from_payload(payload)
        if not scope.symbol:
            return None
        return cls(
            scope=scope,
            bids=normalize_levels(first_present(payload, "bids", "b") or []),
            asks=normalize_levels(first_present(payload, "asks", "a") or []),
            last_update_id=safe_int(first_present(payload, "last_update_id", "lastUpdateId", "sequence", "update_id")),
            timestamp_ms=safe_int(first_present(payload, "timestamp_ms", "event_time", "E", "timestamp"), now_ms()) or now_ms(),
            received_at_ms=safe_int(payload.get("received_at_ms"), now_ms()) or now_ms(),
            metadata=dict(payload.get("metadata") or {}),
        )


def normalize_levels(levels: Sequence[Any]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for item in levels:
        price: Any = None
        qty: Any = None
        if isinstance(item, Mapping):
            price = first_present(item, "price", "p", "px")
            qty = first_present(item, "quantity", "qty", "q", "size", "sz")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            price, qty = item[0], item[1]
        p = safe_float(price)
        q = safe_float(qty)
        if p is None or q is None or p <= 0:
            continue
        result.append((p, q))
    return result


@dataclass(slots=True)
class FundingUpdate:
    scope: MarketScope
    funding_rate: float | None = None
    next_funding_time_ms: int | None = None
    mark_price: float | None = None
    index_price: float | None = None
    timestamp_ms: int = field(default_factory=now_ms)
    received_at_ms: int = field(default_factory=now_ms)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FundingUpdate | None":
        scope = MarketScope.from_payload(payload)
        if not scope.symbol:
            return None
        return cls(
            scope=scope,
            funding_rate=safe_float(first_present(payload, "funding_rate", "lastFundingRate", "rate")),
            next_funding_time_ms=safe_int(first_present(payload, "next_funding_time_ms", "nextFundingTime", "funding_time")),
            mark_price=safe_float(first_present(payload, "mark_price", "markPrice", "current_price", "price")),
            index_price=safe_float(first_present(payload, "index_price", "indexPrice")),
            timestamp_ms=safe_int(first_present(payload, "timestamp_ms", "event_time", "time", "timestamp"), now_ms()) or now_ms(),
            received_at_ms=safe_int(payload.get("received_at_ms"), now_ms()) or now_ms(),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(slots=True)
class OpenInterestUpdate:
    scope: MarketScope
    open_interest: float
    open_interest_value: float | None = None
    mark_price: float | None = None
    index_price: float | None = None
    timestamp_ms: int = field(default_factory=now_ms)
    received_at_ms: int = field(default_factory=now_ms)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "OpenInterestUpdate | None":
        scope = MarketScope.from_payload(payload)
        value = safe_float(first_present(payload, "open_interest", "openInterest", "oi"))
        if not scope.symbol or value is None:
            return None
        return cls(
            scope=scope,
            open_interest=value,
            open_interest_value=safe_float(first_present(payload, "open_interest_value", "openInterestValue", "notional_value")),
            mark_price=safe_float(first_present(payload, "mark_price", "markPrice", "current_price", "price")),
            index_price=safe_float(first_present(payload, "index_price", "indexPrice")),
            timestamp_ms=safe_int(first_present(payload, "timestamp_ms", "event_time", "time", "timestamp"), now_ms()) or now_ms(),
            received_at_ms=safe_int(payload.get("received_at_ms"), now_ms()) or now_ms(),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(slots=True)
class LiquidationUpdate:
    scope: MarketScope
    price: float
    quantity: float
    side: str
    order_id: str | None = None
    timestamp_ms: int = field(default_factory=now_ms)
    received_at_ms: int = field(default_factory=now_ms)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LiquidationUpdate | None":
        scope = MarketScope.from_payload(payload)
        price = safe_float(first_present(payload, "price", "p", "average_price", "avg_price"))
        quantity = safe_float(first_present(payload, "quantity", "qty", "q", "size"), 0.0)
        if not scope.symbol or price is None or price <= 0:
            return None

        metadata = dict(payload.get("metadata") or {})
        for key in (
            "notional_usd",
            "notional",
            "avg_price",
            "average_price",
            "limit_price",
            "last_filled_qty",
            "accumulated_filled_qty",
            "order_status",
            "order_type",
            "time_in_force",
            "exchange_symbol",
            "liquidation_side",
            "trade_time",
            "event_time",
            "source",
        ):
            if key in payload and payload.get(key) is not None and key not in metadata:
                metadata[key] = payload.get(key)

        notional = safe_float(first_present(payload, "notional_usd", "notional"))
        if notional is None and quantity is not None and price is not None:
            notional = float(quantity or 0.0) * float(price)
        if notional is not None:
            metadata.setdefault("notional_usd", notional)
            metadata.setdefault("notional", notional)

        raw_side = first_present(payload, "liquidation_side", "side", "S", "position_side")
        metadata.setdefault("raw_side", raw_side)

        return cls(
            scope=scope,
            price=price,
            quantity=quantity or 0.0,
            side=normalize_side(raw_side),
            order_id=str(first_present(payload, "order_id", "id", "i", "trade_id") or "") or None,
            timestamp_ms=safe_int(first_present(payload, "timestamp_ms", "event_time", "T", "E", "timestamp"), now_ms()) or now_ms(),
            received_at_ms=safe_int(payload.get("received_at_ms"), now_ms()) or now_ms(),
            metadata=metadata,
        )
