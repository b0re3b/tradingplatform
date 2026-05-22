from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, TypeAlias

from .enums import (
    InstrumentType,
    OpportunityStatus,
    PricingSource,
    QuoteValidity,
    SpreadDirection,
    SpreadRegime,
    SpreadSignalType,
    SpreadType,
)


DECIMAL_ZERO = Decimal("0")
DECIMAL_TWO = Decimal("2")

DEFAULT_TIMEFRAME = "realtime"
DEFAULT_SPOT_MARKET_TYPE = "spot"
DEFAULT_PERPETUAL_MARKET_TYPE = "perpetual"
DEFAULT_FUTURES_MARKET_TYPE = "futures"

SpreadKey: TypeAlias = tuple[str, str, str, str]
# exchange, market_type, symbol, timeframe


# ============================================================
# Internal helpers
# ============================================================

def _utcnow() -> datetime:
    """
    Єдина точка створення timestamp.

    Залишаємо naive UTC для сумісності з існуючим пакетом.
    Якщо весь проєкт пізніше переходить на timezone-aware UTC, достатньо
    змінити цю функцію централізовано.
    """
    return datetime.utcnow()


def _normalize_exchange(exchange: object) -> str:
    value = str(exchange or "").strip().lower()
    if not value:
        raise ValueError("exchange must not be empty")
    return value


def _normalize_symbol(symbol: object) -> str:
    value = str(symbol or "").replace("-", "").replace("/", "").replace("_", "").upper().strip()
    if not value:
        raise ValueError("symbol must not be empty")
    return value


def _normalize_exchange_symbol(
    exchange_symbol: object | None,
    *,
    fallback_symbol: str,
) -> str:
    value = str(exchange_symbol or "").strip()
    return value if value else fallback_symbol


def _normalize_timeframe(timeframe: object | None = DEFAULT_TIMEFRAME) -> str:
    value = str(timeframe or DEFAULT_TIMEFRAME).strip()
    return value if value else DEFAULT_TIMEFRAME


def _normalize_market_type(
    market_type: object | None = None,
    *,
    instrument_type: InstrumentType | str | None = None,
) -> str:
    if market_type is not None:
        value = str(market_type).strip().lower()
        if value:
            return value

    parsed_instrument = _parse_instrument_type(instrument_type)
    if parsed_instrument == InstrumentType.SPOT:
        return DEFAULT_SPOT_MARKET_TYPE
    if parsed_instrument == InstrumentType.PERPETUAL:
        return DEFAULT_PERPETUAL_MARKET_TYPE
    if parsed_instrument == InstrumentType.FUTURES:
        return DEFAULT_FUTURES_MARKET_TYPE

    return DEFAULT_PERPETUAL_MARKET_TYPE


def _parse_instrument_type(value: InstrumentType | str | None) -> InstrumentType:
    if isinstance(value, InstrumentType):
        return value

    if value is None:
        return InstrumentType.UNKNOWN

    raw = str(value).strip().lower()
    for item in InstrumentType:
        if item.value == raw:
            return item

    return InstrumentType.UNKNOWN


def _infer_instrument_type_from_payload(payload: Mapping[str, Any]) -> InstrumentType:
    """
    Infer spread instrument type from production market-data payloads.

    OrderBookCache / market.orderbook.updated may not carry an explicit
    `instrument_type`, while futures/perpetual exchange adapters usually carry
    `market_type` values such as `usdm_futures`, `linear`, `swap`, `contract`,
    or exchange-specific futures symbols. QuoteSnapshot must not receive
    InstrumentType.UNKNOWN, so this helper provides a safe futures-first
    inference without changing the strict QuoteSnapshot validation.

    Explicit instrument fields always win. For ambiguous missing metadata we
    default to PERPETUAL because this runtime uses futures/perpetual market-data
    adapters.
    """
    explicit = _parse_instrument_type(
        _first_present(payload, "instrument_type", "instrument", "type")
    )
    if explicit != InstrumentType.UNKNOWN:
        return explicit

    market_type = _first_present(
        payload,
        "market_type",
        "market",
        "contract_type",
        "category",
        "inst_type",
        "instrument_kind",
    )
    raw_market = str(market_type or "").strip().lower()

    if raw_market:
        if raw_market in {"spot", "cash"}:
            return InstrumentType.SPOT

        if any(
            token in raw_market
            for token in (
                "perp",
                "perpetual",
                "swap",
                "linear",
                "inverse",
                "usdm",
                "usd-m",
                "coinm",
                "coin-m",
                "contract",
            )
        ):
            return InstrumentType.PERPETUAL

        if "future" in raw_market or "futures" in raw_market:
            return InstrumentType.FUTURES

    exchange_symbol = str(
        _first_present(payload, "exchange_symbol", "raw_symbol", "symbol") or ""
    ).strip().upper()

    if exchange_symbol.endswith("-SWAP") or "_PERP" in exchange_symbol or "PERP" in exchange_symbol:
        return InstrumentType.PERPETUAL

    # Futures-first runtime fallback. This prevents production orderbook events
    # without explicit instrument metadata from being dropped by spread analyzers.
    return InstrumentType.PERPETUAL


def _validate_non_empty(name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _to_decimal(value: Any, *, default: Decimal | None = None) -> Decimal | None:
    if value is None:
        return default

    if isinstance(value, Decimal):
        return value

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def _validate_non_negative_decimal(name: str, value: Decimal | None) -> None:
    if value is not None and value < DECIMAL_ZERO:
        raise ValueError(f"{name} must be >= 0")


def _validate_positive_decimal(name: str, value: Decimal | None) -> None:
    if value is not None and value <= DECIMAL_ZERO:
        raise ValueError(f"{name} must be > 0")


def _decimal_to_payload(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def _datetime_to_payload(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _datetime_from_payload(value: Any) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    if isinstance(value, (int, float)):
        try:
            # Most market-data payloads use milliseconds. Keep seconds support
            # for already-normalized payloads and tests.
            timestamp = float(value)
            if timestamp > 10_000_000_000:
                timestamp /= 1000
            return datetime.utcfromtimestamp(timestamp)
        except (OverflowError, OSError, ValueError):
            return None

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None

        try:
            numeric = float(raw)
            if numeric > 10_000_000_000:
                numeric /= 1000
            return datetime.utcfromtimestamp(numeric)
        except (OverflowError, OSError, ValueError):
            pass

        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    return None


def _enum_to_payload(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _metadata_copy(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


def make_spread_key(
    *,
    exchange: object,
    market_type: object,
    symbol: object,
    timeframe: object = DEFAULT_TIMEFRAME,
) -> SpreadKey:
    """
    Canonical key для spread analytics.

    Scope:
        exchange + market_type + symbol + timeframe
    """
    return (
        _normalize_exchange(exchange),
        _normalize_market_type(market_type),
        _normalize_symbol(symbol),
        _normalize_timeframe(timeframe),
    )


def spread_key_to_dict(key: SpreadKey) -> dict[str, str]:
    exchange, market_type, symbol, timeframe = key
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
    }


def scoped_metadata(
    *,
    exchange: object,
    market_type: object,
    symbol: object,
    timeframe: object = DEFAULT_TIMEFRAME,
    exchange_symbol: object | None = None,
) -> dict[str, str]:
    key = make_spread_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )
    data = spread_key_to_dict(key)
    data["exchange_symbol"] = _normalize_exchange_symbol(
        exchange_symbol,
        fallback_symbol=data["symbol"],
    )
    return data


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "value"):
        return value.value

    if is_dataclass(value):
        return {
            key: _serialize_value(item)
            for key, item in asdict(value).items()
        }

    if isinstance(value, Mapping):
        return {
            str(key): _serialize_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_serialize_value(item) for item in value]

    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]

    if isinstance(value, set):
        return sorted(_serialize_value(item) for item in value)

    return value



# ============================================================
# Orderbook -> QuoteSnapshot normalization helpers
# ============================================================

def _first_present(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload.get(key) is not None:
            return payload.get(key)
    return None


def _to_optional_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _level_value(level: Any, *keys: str) -> Any:
    """
    Підтримує різні формати orderbook level:
    - {"price": "100", "quantity": "1.2"}
    - {"p": "100", "q": "1.2"}
    - ["100", "1.2"] / ("100", "1.2")
    """
    if level is None:
        return None

    if isinstance(level, Mapping):
        for key in keys:
            if key in level and level.get(key) is not None:
                return level.get(key)
        return None

    if isinstance(level, (list, tuple)):
        if any(key in {"price", "p", "px", "rate"} for key in keys):
            return level[0] if len(level) > 0 else None
        if any(key in {"quantity", "qty", "size", "amount", "volume", "q"} for key in keys):
            return level[1] if len(level) > 1 else None

    return None


def _best_level(payload: Mapping[str, Any], side: str) -> Any:
    levels = payload.get(side)
    if isinstance(levels, (list, tuple)) and levels:
        return levels[0]
    return None


def _extract_best_bid(payload: Mapping[str, Any]) -> tuple[Decimal | None, Decimal | None]:
    best_bid_level = _best_level(payload, "bids")

    price = _to_decimal(
        _first_present(
            payload,
            "best_bid",
            "best_bid_price",
            "bid",
            "bid_price",
        )
    )
    if price is None:
        price = _to_decimal(_level_value(best_bid_level, "price", "p", "px", "rate"))

    size = _to_decimal(
        _first_present(
            payload,
            "best_bid_size",
            "bid_size",
            "bid_qty",
            "bid_quantity",
        )
    )
    if size is None:
        size = _to_decimal(
            _level_value(best_bid_level, "quantity", "qty", "size", "amount", "volume", "q")
        )

    return price, size


def _extract_best_ask(payload: Mapping[str, Any]) -> tuple[Decimal | None, Decimal | None]:
    best_ask_level = _best_level(payload, "asks")

    price = _to_decimal(
        _first_present(
            payload,
            "best_ask",
            "best_ask_price",
            "ask",
            "ask_price",
        )
    )
    if price is None:
        price = _to_decimal(_level_value(best_ask_level, "price", "p", "px", "rate"))

    size = _to_decimal(
        _first_present(
            payload,
            "best_ask_size",
            "ask_size",
            "ask_qty",
            "ask_quantity",
        )
    )
    if size is None:
        size = _to_decimal(
            _level_value(best_ask_level, "quantity", "qty", "size", "amount", "volume", "q")
        )

    return price, size


def _payload_timestamp(
    payload: Mapping[str, Any],
    *keys: str,
    default: datetime | None = None,
) -> datetime:
    for key in keys:
        value = payload.get(key)
        parsed = _datetime_from_payload(value)
        if parsed is not None:
            return parsed

    return default or _utcnow()


def _payload_metadata(payload: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    metadata = _metadata_copy(payload.get("metadata"))
    metadata.setdefault("source", source)

    for key in (
        "depth",
        "checksum",
        "sequence_id",
        "first_update_id",
        "last_update_id",
        "update_id",
        "is_snapshot",
        "is_resync_required",
        "validity",
    ):
        if key in payload and payload.get(key) is not None:
            metadata.setdefault(key, payload.get(key))

    return metadata


def quote_snapshot_from_orderbook_payload(
    payload: Mapping[str, Any],
) -> "QuoteSnapshot":
    """
    Нормалізує payload з data/orderbook_cache.py у внутрішній QuoteSnapshot.

    Production flow без QuoteCache:
        exchange adapter -> market.orderbook
        -> OrderBookCache -> market.orderbook.updated
        -> analytics.spreads -> QuoteSnapshot

    Підтримує payload-и з явними best_bid/best_ask полями або з bids/asks
    рівнями, де перший level є top-of-book.
    """
    bid, bid_size = _extract_best_bid(payload)
    ask, ask_size = _extract_best_ask(payload)

    instrument_type = _infer_instrument_type_from_payload(payload)
    market_type = _first_present(payload, "market_type", "market", "contract_type")

    if market_type is None:
        market_type = _normalize_market_type(instrument_type=instrument_type)

    return QuoteSnapshot(
        exchange=str(payload["exchange"]),
        symbol=str(payload["symbol"]),
        instrument_type=instrument_type,
        market_type=market_type,
        timeframe=str(_first_present(payload, "timeframe", "interval") or DEFAULT_TIMEFRAME),
        exchange_symbol=_first_present(payload, "exchange_symbol", "raw_symbol"),
        bid=bid,
        ask=ask,
        bid_size=bid_size,
        ask_size=ask_size,
        last_price=_to_decimal(_first_present(payload, "last_price", "last", "price")),
        mark_price=_to_decimal(_first_present(payload, "mark_price", "mark")),
        index_price=_to_decimal(_first_present(payload, "index_price", "index")),
        timestamp=_payload_timestamp(
            payload,
            "timestamp",
            "timestamp_ms",
            "updated_at",
            "updated_at_ms",
            "event_time",
            "event_time_ms",
        ),
        received_at=_payload_timestamp(
            payload,
            "received_at",
            "received_at_ms",
            "ingested_at",
            "ingested_at_ms",
            default=_utcnow(),
        ),
        sequence_id=_to_optional_int(
            _first_present(
                payload,
                "sequence_id",
                "last_update_id",
                "update_id",
                "seq",
            )
        ),
        metadata=_payload_metadata(payload, source="market.orderbook.updated"),
    )

# ============================================================
# Market Data Models
# ============================================================

@dataclass(slots=True)
class QuoteSnapshot:
    """
    Normalized top-of-book snapshot для spread analytics.

    У production ця модель будується всередині analytics.spreads із payload-у
    data-layer події market.orderbook.updated. Окремий QuoteCache не потрібен:
    OrderBookCache є джерелом bid/ask/top-of-book, а QuoteSnapshot лишається
    внутрішнім контрактом analyzer-ів.

    Correct input flow:
        exchange adapter -> market.orderbook
        -> OrderBookCache -> market.orderbook.updated
        -> analytics.spreads -> QuoteSnapshot -> SpreadSnapshot/Signal

    Для backward compatibility from_payload() також підтримує старий dict-формат
    із bid/ask полями.
    """

    exchange: str
    symbol: str
    instrument_type: InstrumentType

    bid: Decimal | None = None
    ask: Decimal | None = None
    bid_size: Decimal | None = None
    ask_size: Decimal | None = None

    last_price: Decimal | None = None
    mark_price: Decimal | None = None
    index_price: Decimal | None = None

    market_type: str | None = None
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    timestamp: datetime = field(default_factory=_utcnow)
    received_at: datetime = field(default_factory=_utcnow)

    sequence_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.instrument_type = _parse_instrument_type(self.instrument_type)
        if self.instrument_type == InstrumentType.UNKNOWN:
            raise ValueError("instrument_type must not be UNKNOWN for QuoteSnapshot")

        self.exchange = _normalize_exchange(self.exchange)
        self.symbol = _normalize_symbol(self.symbol)
        self.market_type = _normalize_market_type(
            self.market_type,
            instrument_type=self.instrument_type,
        )
        self.timeframe = _normalize_timeframe(self.timeframe)
        self.exchange_symbol = _normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

        self.bid = _to_decimal(self.bid)
        self.ask = _to_decimal(self.ask)
        self.bid_size = _to_decimal(self.bid_size)
        self.ask_size = _to_decimal(self.ask_size)
        self.last_price = _to_decimal(self.last_price)
        self.mark_price = _to_decimal(self.mark_price)
        self.index_price = _to_decimal(self.index_price)

        self.metadata = _metadata_copy(self.metadata)
        self.metadata.setdefault("scope", spread_key_to_dict(self.key))
        self.metadata.setdefault("exchange_symbol", self.exchange_symbol)

        _validate_positive_decimal("bid", self.bid)
        _validate_positive_decimal("ask", self.ask)
        _validate_non_negative_decimal("bid_size", self.bid_size)
        _validate_non_negative_decimal("ask_size", self.ask_size)
        _validate_positive_decimal("last_price", self.last_price)
        _validate_positive_decimal("mark_price", self.mark_price)
        _validate_positive_decimal("index_price", self.index_price)

        if self.bid is not None and self.ask is not None and self.bid > self.ask:
            raise ValueError("bid must be <= ask")

        if self.sequence_id is not None and self.sequence_id < 0:
            raise ValueError("sequence_id must be >= 0")

    @property
    def key(self) -> SpreadKey:
        return make_spread_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

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
    def mid_price(self) -> Decimal | None:
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / DECIMAL_TWO

    @property
    def spread(self) -> Decimal | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid

    @property
    def is_complete(self) -> bool:
        return self.bid is not None and self.ask is not None

    @property
    def age_ms(self) -> int:
        delta = _utcnow() - self.timestamp
        return max(int(delta.total_seconds() * 1000), 0)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> QuoteSnapshot:
        """
        Backward-compatible constructor для вже нормалізованого quote payload.

        Якщо payload схожий на market.orderbook.updated — містить bids/asks або
        best_bid/best_ask — делегує нормалізацію в
        quote_snapshot_from_orderbook_payload().
        """
        if any(
            key in payload
            for key in (
                "bids",
                "asks",
                "best_bid",
                "best_ask",
                "best_bid_price",
                "best_ask_price",
            )
        ):
            return quote_snapshot_from_orderbook_payload(payload)

        instrument_type = _parse_instrument_type(payload.get("instrument_type"))

        return cls(
            exchange=str(payload["exchange"]),
            symbol=str(payload["symbol"]),
            instrument_type=instrument_type,
            market_type=payload.get("market_type"),
            timeframe=str(payload.get("timeframe") or DEFAULT_TIMEFRAME),
            exchange_symbol=payload.get("exchange_symbol"),
            bid=_to_decimal(payload.get("bid")),
            ask=_to_decimal(payload.get("ask")),
            bid_size=_to_decimal(payload.get("bid_size")),
            ask_size=_to_decimal(payload.get("ask_size")),
            last_price=_to_decimal(payload.get("last_price")),
            mark_price=_to_decimal(payload.get("mark_price")),
            index_price=_to_decimal(payload.get("index_price")),
            timestamp=_payload_timestamp(payload, "timestamp", "timestamp_ms"),
            received_at=_payload_timestamp(payload, "received_at", "received_at_ms"),
            sequence_id=_to_optional_int(payload.get("sequence_id")),
            metadata=_metadata_copy(payload.get("metadata")),
        )

    @classmethod
    def from_orderbook_payload(cls, payload: Mapping[str, Any]) -> QuoteSnapshot:
        """
        Явний constructor для market.orderbook.updated payload.
        """
        return quote_snapshot_from_orderbook_payload(payload)

    def to_payload(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "exchange_symbol": self.exchange_symbol,
            "instrument_type": self.instrument_type.value,
            "bid": _decimal_to_payload(self.bid),
            "ask": _decimal_to_payload(self.ask),
            "bid_size": _decimal_to_payload(self.bid_size),
            "ask_size": _decimal_to_payload(self.ask_size),
            "last_price": _decimal_to_payload(self.last_price),
            "mark_price": _decimal_to_payload(self.mark_price),
            "index_price": _decimal_to_payload(self.index_price),
            "mid_price": _decimal_to_payload(self.mid_price),
            "spread": _decimal_to_payload(self.spread),
            "timestamp": _datetime_to_payload(self.timestamp),
            "received_at": _datetime_to_payload(self.received_at),
            "sequence_id": self.sequence_id,
            "scope": spread_key_to_dict(self.key),
            "metadata": _metadata_copy(self.metadata),
        }


@dataclass(slots=True)
class FundingSnapshot:
    """
    Funding snapshot для spot/futures або futures basis spread analytics.

    Production source:
        FundingCache -> market.funding.updated -> analytics.spreads.*
    """

    exchange: str
    symbol: str
    funding_rate: Decimal

    market_type: str = DEFAULT_PERPETUAL_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    timestamp: datetime = field(default_factory=_utcnow)
    next_funding_time: datetime | None = None
    predicted_rate: Decimal | None = None
    interval_hours: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.exchange = _normalize_exchange(self.exchange)
        self.symbol = _normalize_symbol(self.symbol)
        self.market_type = _normalize_market_type(self.market_type)
        self.timeframe = _normalize_timeframe(self.timeframe)
        self.exchange_symbol = _normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

        self.funding_rate = _to_decimal(self.funding_rate, default=DECIMAL_ZERO) or DECIMAL_ZERO
        self.predicted_rate = _to_decimal(self.predicted_rate)

        self.metadata = _metadata_copy(self.metadata)
        self.metadata.setdefault("scope", spread_key_to_dict(self.key))
        self.metadata.setdefault("exchange_symbol", self.exchange_symbol)

        if self.interval_hours is not None and self.interval_hours <= 0:
            raise ValueError("interval_hours must be > 0")

    @property
    def key(self) -> SpreadKey:
        return make_spread_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def age_ms(self) -> int:
        delta = _utcnow() - self.timestamp
        return max(int(delta.total_seconds() * 1000), 0)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> FundingSnapshot:
        return cls(
            exchange=str(payload["exchange"]),
            symbol=str(payload["symbol"]),
            market_type=str(payload.get("market_type") or DEFAULT_PERPETUAL_MARKET_TYPE),
            timeframe=str(payload.get("timeframe") or DEFAULT_TIMEFRAME),
            exchange_symbol=payload.get("exchange_symbol"),
            funding_rate=_to_decimal(payload.get("funding_rate"), default=DECIMAL_ZERO) or DECIMAL_ZERO,
            timestamp=_datetime_from_payload(payload.get("timestamp")) or _utcnow(),
            next_funding_time=_datetime_from_payload(payload.get("next_funding_time")),
            predicted_rate=_to_decimal(payload.get("predicted_rate")),
            interval_hours=payload.get("interval_hours"),
            metadata=_metadata_copy(payload.get("metadata")),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "exchange_symbol": self.exchange_symbol,
            "funding_rate": _decimal_to_payload(self.funding_rate),
            "timestamp": _datetime_to_payload(self.timestamp),
            "next_funding_time": _datetime_to_payload(self.next_funding_time),
            "predicted_rate": _decimal_to_payload(self.predicted_rate),
            "interval_hours": self.interval_hours,
            "scope": spread_key_to_dict(self.key),
            "metadata": _metadata_copy(self.metadata),
        }


# ============================================================
# Analytics Models
# ============================================================

@dataclass(slots=True)
class RollingStats:
    """
    Rolling statistical state for spread values.
    """

    count: int = 0
    mean: Decimal | None = None
    std: Decimal | None = None
    min_value: Decimal | None = None
    max_value: Decimal | None = None
    ema: Decimal | None = None
    last_value: Decimal | None = None
    zscore: Decimal | None = None
    percentile_rank: Decimal | None = None

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("count must be >= 0")

        self.mean = _to_decimal(self.mean)
        self.std = _to_decimal(self.std)
        self.min_value = _to_decimal(self.min_value)
        self.max_value = _to_decimal(self.max_value)
        self.ema = _to_decimal(self.ema)
        self.last_value = _to_decimal(self.last_value)
        self.zscore = _to_decimal(self.zscore)
        self.percentile_rank = _to_decimal(self.percentile_rank)

        _validate_non_negative_decimal("std", self.std)

        if self.percentile_rank is not None:
            if self.percentile_rank < DECIMAL_ZERO or self.percentile_rank > Decimal("100"):
                raise ValueError("percentile_rank must be between 0 and 100")

    @property
    def has_enough_data(self) -> bool:
        return self.count > 1

    def to_payload(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": _decimal_to_payload(self.mean),
            "std": _decimal_to_payload(self.std),
            "min_value": _decimal_to_payload(self.min_value),
            "max_value": _decimal_to_payload(self.max_value),
            "ema": _decimal_to_payload(self.ema),
            "last_value": _decimal_to_payload(self.last_value),
            "zscore": _decimal_to_payload(self.zscore),
            "percentile_rank": _decimal_to_payload(self.percentile_rank),
            "has_enough_data": self.has_enough_data,
        }


@dataclass(slots=True)
class SpreadSnapshot:
    """
    Canonical spread analytics snapshot.

    Analyzer-и публікують цю модель у:
    - analytics.spreads.spot_futures.updated
    - analytics.spreads.cross_exchange.updated
    """

    spread_type: SpreadType
    symbol: str

    leg_a_exchange: str
    leg_b_exchange: str
    leg_a_type: InstrumentType
    leg_b_type: InstrumentType

    leg_a_market_type: str | None = None
    leg_b_market_type: str | None = None
    timeframe: str = DEFAULT_TIMEFRAME

    leg_a_exchange_symbol: str | None = None
    leg_b_exchange_symbol: str | None = None

    pricing_source: PricingSource = PricingSource.BID_ASK

    raw_spread: Decimal | None = None
    spread_pct: Decimal | None = None
    spread_bps: Decimal | None = None

    net_spread: Decimal | None = None
    basis: Decimal | None = None
    funding_adjusted_spread: Decimal | None = None

    direction: SpreadDirection = SpreadDirection.FLAT
    regime: SpreadRegime = SpreadRegime.NORMAL

    stats: RollingStats | None = None

    leg_a_bid: Decimal | None = None
    leg_a_ask: Decimal | None = None
    leg_b_bid: Decimal | None = None
    leg_b_ask: Decimal | None = None

    leg_a_mid: Decimal | None = None
    leg_b_mid: Decimal | None = None

    estimated_fees: Decimal | None = None
    estimated_slippage: Decimal | None = None

    quote_validity: QuoteValidity = QuoteValidity.VALID

    timestamp: datetime = field(default_factory=_utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.spread_type = self.spread_type if isinstance(self.spread_type, SpreadType) else SpreadType(str(self.spread_type))
        self.leg_a_type = _parse_instrument_type(self.leg_a_type)
        self.leg_b_type = _parse_instrument_type(self.leg_b_type)

        if self.leg_a_type == InstrumentType.UNKNOWN:
            raise ValueError("leg_a_type must not be UNKNOWN")
        if self.leg_b_type == InstrumentType.UNKNOWN:
            raise ValueError("leg_b_type must not be UNKNOWN")

        self.symbol = _normalize_symbol(self.symbol)
        self.leg_a_exchange = _normalize_exchange(self.leg_a_exchange)
        self.leg_b_exchange = _normalize_exchange(self.leg_b_exchange)

        self.leg_a_market_type = _normalize_market_type(
            self.leg_a_market_type,
            instrument_type=self.leg_a_type,
        )
        self.leg_b_market_type = _normalize_market_type(
            self.leg_b_market_type,
            instrument_type=self.leg_b_type,
        )
        self.timeframe = _normalize_timeframe(self.timeframe)

        self.leg_a_exchange_symbol = _normalize_exchange_symbol(
            self.leg_a_exchange_symbol,
            fallback_symbol=self.symbol,
        )
        self.leg_b_exchange_symbol = _normalize_exchange_symbol(
            self.leg_b_exchange_symbol,
            fallback_symbol=self.symbol,
        )

        self.raw_spread = _to_decimal(self.raw_spread)
        self.spread_pct = _to_decimal(self.spread_pct)
        self.spread_bps = _to_decimal(self.spread_bps)
        self.net_spread = _to_decimal(self.net_spread)
        self.basis = _to_decimal(self.basis)
        self.funding_adjusted_spread = _to_decimal(self.funding_adjusted_spread)

        self.leg_a_bid = _to_decimal(self.leg_a_bid)
        self.leg_a_ask = _to_decimal(self.leg_a_ask)
        self.leg_b_bid = _to_decimal(self.leg_b_bid)
        self.leg_b_ask = _to_decimal(self.leg_b_ask)
        self.leg_a_mid = _to_decimal(self.leg_a_mid)
        self.leg_b_mid = _to_decimal(self.leg_b_mid)

        self.estimated_fees = _to_decimal(self.estimated_fees)
        self.estimated_slippage = _to_decimal(self.estimated_slippage)

        self.metadata = _metadata_copy(self.metadata)
        self.metadata.setdefault("leg_a_scope", spread_key_to_dict(self.leg_a_key))
        self.metadata.setdefault("leg_b_scope", spread_key_to_dict(self.leg_b_key))

        _validate_non_negative_decimal("estimated_fees", self.estimated_fees)
        _validate_non_negative_decimal("estimated_slippage", self.estimated_slippage)

        _validate_positive_decimal("leg_a_bid", self.leg_a_bid)
        _validate_positive_decimal("leg_a_ask", self.leg_a_ask)
        _validate_positive_decimal("leg_b_bid", self.leg_b_bid)
        _validate_positive_decimal("leg_b_ask", self.leg_b_ask)
        _validate_positive_decimal("leg_a_mid", self.leg_a_mid)
        _validate_positive_decimal("leg_b_mid", self.leg_b_mid)

        if self.leg_a_bid is not None and self.leg_a_ask is not None and self.leg_a_bid > self.leg_a_ask:
            raise ValueError("leg_a_bid must be <= leg_a_ask")

        if self.leg_b_bid is not None and self.leg_b_ask is not None and self.leg_b_bid > self.leg_b_ask:
            raise ValueError("leg_b_bid must be <= leg_b_ask")

    @property
    def leg_a_key(self) -> SpreadKey:
        return make_spread_key(
            exchange=self.leg_a_exchange,
            market_type=self.leg_a_market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def leg_b_key(self) -> SpreadKey:
        return make_spread_key(
            exchange=self.leg_b_exchange,
            market_type=self.leg_b_market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def has_edge(self) -> bool:
        return self.net_spread is not None and self.net_spread > DECIMAL_ZERO

    @property
    def abs_spread_bps(self) -> Decimal | None:
        return abs(self.spread_bps) if self.spread_bps is not None else None

    @property
    def pair_key(self) -> tuple[str, str, str, str, str, InstrumentType, InstrumentType]:
        return (
            self.symbol,
            self.leg_a_exchange,
            self.leg_b_exchange,
            self.leg_a_market_type,
            self.leg_b_market_type,
            self.leg_a_type,
            self.leg_b_type,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "spread_type": self.spread_type.value,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "leg_a_exchange": self.leg_a_exchange,
            "leg_b_exchange": self.leg_b_exchange,
            "leg_a_market_type": self.leg_a_market_type,
            "leg_b_market_type": self.leg_b_market_type,
            "leg_a_exchange_symbol": self.leg_a_exchange_symbol,
            "leg_b_exchange_symbol": self.leg_b_exchange_symbol,
            "leg_a_type": self.leg_a_type.value,
            "leg_b_type": self.leg_b_type.value,
            "leg_a_scope": spread_key_to_dict(self.leg_a_key),
            "leg_b_scope": spread_key_to_dict(self.leg_b_key),
            "pricing_source": self.pricing_source.value,
            "raw_spread": _decimal_to_payload(self.raw_spread),
            "spread_pct": _decimal_to_payload(self.spread_pct),
            "spread_bps": _decimal_to_payload(self.spread_bps),
            "net_spread": _decimal_to_payload(self.net_spread),
            "basis": _decimal_to_payload(self.basis),
            "funding_adjusted_spread": _decimal_to_payload(self.funding_adjusted_spread),
            "direction": self.direction.value,
            "regime": self.regime.value,
            "stats": self.stats.to_payload() if self.stats is not None else None,
            "leg_a_bid": _decimal_to_payload(self.leg_a_bid),
            "leg_a_ask": _decimal_to_payload(self.leg_a_ask),
            "leg_b_bid": _decimal_to_payload(self.leg_b_bid),
            "leg_b_ask": _decimal_to_payload(self.leg_b_ask),
            "leg_a_mid": _decimal_to_payload(self.leg_a_mid),
            "leg_b_mid": _decimal_to_payload(self.leg_b_mid),
            "estimated_fees": _decimal_to_payload(self.estimated_fees),
            "estimated_slippage": _decimal_to_payload(self.estimated_slippage),
            "quote_validity": self.quote_validity.value,
            "timestamp": _datetime_to_payload(self.timestamp),
            "metadata": _metadata_copy(self.metadata),
            "has_edge": self.has_edge,
        }


# ============================================================
# Signals
# ============================================================

@dataclass(slots=True)
class SpreadSignal:
    """
    Canonical spread signal.

    Strategy layer може слухати analytics.spreads.signal.generated
    і отримувати саме цю модель або її payload.
    """

    signal_type: SpreadSignalType
    spread_type: SpreadType
    symbol: str

    message: str

    value: Decimal | None = None
    threshold: Decimal | None = None
    confidence: Decimal | None = None

    exchange_a: str | None = None
    exchange_b: str | None = None

    market_type_a: str | None = None
    market_type_b: str | None = None
    timeframe: str = DEFAULT_TIMEFRAME

    exchange_symbol_a: str | None = None
    exchange_symbol_b: str | None = None

    timestamp: datetime = field(default_factory=_utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty("symbol", self.symbol)
        _validate_non_empty("message", self.message)

        self.symbol = _normalize_symbol(self.symbol)
        self.timeframe = _normalize_timeframe(self.timeframe)

        if self.exchange_a is not None:
            self.exchange_a = _normalize_exchange(self.exchange_a)
            self.market_type_a = _normalize_market_type(self.market_type_a)
            self.exchange_symbol_a = _normalize_exchange_symbol(
                self.exchange_symbol_a,
                fallback_symbol=self.symbol,
            )

        if self.exchange_b is not None:
            self.exchange_b = _normalize_exchange(self.exchange_b)
            self.market_type_b = _normalize_market_type(self.market_type_b)
            self.exchange_symbol_b = _normalize_exchange_symbol(
                self.exchange_symbol_b,
                fallback_symbol=self.symbol,
            )

        self.value = _to_decimal(self.value)
        self.threshold = _to_decimal(self.threshold)
        self.confidence = _to_decimal(self.confidence)

        self.metadata = _metadata_copy(self.metadata)

        if self.exchange_a and self.market_type_a:
            self.metadata.setdefault("leg_a_scope", spread_key_to_dict(self.leg_a_key))
        if self.exchange_b and self.market_type_b:
            self.metadata.setdefault("leg_b_scope", spread_key_to_dict(self.leg_b_key))

        if self.confidence is not None:
            if self.confidence < DECIMAL_ZERO or self.confidence > Decimal("1"):
                raise ValueError("confidence must be between 0 and 1")

    @property
    def leg_a_key(self) -> SpreadKey:
        if self.exchange_a is None:
            raise ValueError("exchange_a is required to build leg_a_key")
        return make_spread_key(
            exchange=self.exchange_a,
            market_type=self.market_type_a,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def leg_b_key(self) -> SpreadKey:
        if self.exchange_b is None:
            raise ValueError("exchange_b is required to build leg_b_key")
        return make_spread_key(
            exchange=self.exchange_b,
            market_type=self.market_type_b,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def signal_key(self) -> str:
        exchange_a = self.exchange_a or "na"
        exchange_b = self.exchange_b or "na"
        market_type_a = self.market_type_a or "na"
        market_type_b = self.market_type_b or "na"

        return (
            f"{self.signal_type.value}|"
            f"{self.spread_type.value}|"
            f"{self.symbol}|"
            f"{exchange_a}|{market_type_a}|"
            f"{exchange_b}|{market_type_b}|"
            f"{self.timeframe}"
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "signal_type": self.signal_type.value,
            "spread_type": self.spread_type.value,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "message": self.message,
            "value": _decimal_to_payload(self.value),
            "threshold": _decimal_to_payload(self.threshold),
            "confidence": _decimal_to_payload(self.confidence),
            "exchange_a": self.exchange_a,
            "exchange_b": self.exchange_b,
            "market_type_a": self.market_type_a,
            "market_type_b": self.market_type_b,
            "exchange_symbol_a": self.exchange_symbol_a,
            "exchange_symbol_b": self.exchange_symbol_b,
            "leg_a_scope": spread_key_to_dict(self.leg_a_key) if self.exchange_a else None,
            "leg_b_scope": spread_key_to_dict(self.leg_b_key) if self.exchange_b else None,
            "timestamp": _datetime_to_payload(self.timestamp),
            "metadata": _metadata_copy(self.metadata),
            "signal_key": self.signal_key,
        }


# ============================================================
# Arbitrage
# ============================================================

@dataclass(slots=True)
class ArbitrageOpportunity:
    """
    Cross-exchange arbitrage opportunity.

    Публікується analyzer-ом у:
        analytics.spreads.arbitrage.opportunity
    """

    symbol: str

    buy_exchange: str
    sell_exchange: str

    buy_instrument_type: InstrumentType
    sell_instrument_type: InstrumentType

    buy_price: Decimal
    sell_price: Decimal

    gross_edge: Decimal

    buy_market_type: str | None = None
    sell_market_type: str | None = None
    timeframe: str = DEFAULT_TIMEFRAME

    buy_exchange_symbol: str | None = None
    sell_exchange_symbol: str | None = None

    estimated_fees: Decimal = Decimal("0")
    estimated_slippage: Decimal = Decimal("0")
    net_edge: Decimal = Decimal("0")

    spread_pct: Decimal | None = None
    spread_bps: Decimal | None = None
    confidence: Decimal | None = None

    status: OpportunityStatus = OpportunityStatus.ACTIVE

    timestamp: datetime = field(default_factory=_utcnow)
    expires_at: datetime | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = _normalize_symbol(self.symbol)
        self.buy_exchange = _normalize_exchange(self.buy_exchange)
        self.sell_exchange = _normalize_exchange(self.sell_exchange)

        self.buy_instrument_type = _parse_instrument_type(self.buy_instrument_type)
        self.sell_instrument_type = _parse_instrument_type(self.sell_instrument_type)

        if self.buy_instrument_type == InstrumentType.UNKNOWN:
            raise ValueError("buy_instrument_type must not be UNKNOWN")
        if self.sell_instrument_type == InstrumentType.UNKNOWN:
            raise ValueError("sell_instrument_type must not be UNKNOWN")

        self.buy_market_type = _normalize_market_type(
            self.buy_market_type,
            instrument_type=self.buy_instrument_type,
        )
        self.sell_market_type = _normalize_market_type(
            self.sell_market_type,
            instrument_type=self.sell_instrument_type,
        )
        self.timeframe = _normalize_timeframe(self.timeframe)

        self.buy_exchange_symbol = _normalize_exchange_symbol(
            self.buy_exchange_symbol,
            fallback_symbol=self.symbol,
        )
        self.sell_exchange_symbol = _normalize_exchange_symbol(
            self.sell_exchange_symbol,
            fallback_symbol=self.symbol,
        )

        self.buy_price = _to_decimal(self.buy_price, default=DECIMAL_ZERO) or DECIMAL_ZERO
        self.sell_price = _to_decimal(self.sell_price, default=DECIMAL_ZERO) or DECIMAL_ZERO
        self.gross_edge = _to_decimal(self.gross_edge, default=DECIMAL_ZERO) or DECIMAL_ZERO
        self.estimated_fees = _to_decimal(self.estimated_fees, default=DECIMAL_ZERO) or DECIMAL_ZERO
        self.estimated_slippage = _to_decimal(self.estimated_slippage, default=DECIMAL_ZERO) or DECIMAL_ZERO
        self.net_edge = _to_decimal(self.net_edge, default=DECIMAL_ZERO) or DECIMAL_ZERO
        self.spread_pct = _to_decimal(self.spread_pct)
        self.spread_bps = _to_decimal(self.spread_bps)
        self.confidence = _to_decimal(self.confidence)

        self.metadata = _metadata_copy(self.metadata)
        self.metadata.setdefault("buy_scope", spread_key_to_dict(self.buy_key))
        self.metadata.setdefault("sell_scope", spread_key_to_dict(self.sell_key))

        _validate_positive_decimal("buy_price", self.buy_price)
        _validate_positive_decimal("sell_price", self.sell_price)
        _validate_non_negative_decimal("estimated_fees", self.estimated_fees)
        _validate_non_negative_decimal("estimated_slippage", self.estimated_slippage)

        if self.confidence is not None:
            if self.confidence < DECIMAL_ZERO or self.confidence > Decimal("1"):
                raise ValueError("confidence must be between 0 and 1")

        if self.expires_at is not None and self.expires_at < self.timestamp:
            raise ValueError("expires_at must be >= timestamp")

    @property
    def buy_key(self) -> SpreadKey:
        return make_spread_key(
            exchange=self.buy_exchange,
            market_type=self.buy_market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def sell_key(self) -> SpreadKey:
        return make_spread_key(
            exchange=self.sell_exchange,
            market_type=self.sell_market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def is_profitable(self) -> bool:
        return self.net_edge > DECIMAL_ZERO

    @property
    def is_active(self) -> bool:
        return self.status == OpportunityStatus.ACTIVE and not self.is_expired

    @property
    def is_expired(self) -> bool:
        return self.expires_at is not None and _utcnow() >= self.expires_at

    @property
    def total_costs(self) -> Decimal:
        return self.estimated_fees + self.estimated_slippage

    @property
    def edge_after_costs(self) -> Decimal:
        return self.gross_edge - self.total_costs

    @property
    def notional(self) -> Decimal:
        return self.buy_price

    @property
    def opportunity_key(self) -> str:
        return (
            f"{self.symbol}|"
            f"{self.buy_exchange}|{self.buy_market_type}|"
            f"{self.sell_exchange}|{self.sell_market_type}|"
            f"{self.buy_instrument_type.value}|"
            f"{self.sell_instrument_type.value}|"
            f"{self.timeframe}"
        )

    def mark_expired(self) -> None:
        self.status = OpportunityStatus.EXPIRED

    def mark_rejected(self) -> None:
        self.status = OpportunityStatus.REJECTED

    def mark_executed(self) -> None:
        self.status = OpportunityStatus.EXECUTED

    def to_payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "buy_exchange": self.buy_exchange,
            "sell_exchange": self.sell_exchange,
            "buy_market_type": self.buy_market_type,
            "sell_market_type": self.sell_market_type,
            "buy_exchange_symbol": self.buy_exchange_symbol,
            "sell_exchange_symbol": self.sell_exchange_symbol,
            "buy_scope": spread_key_to_dict(self.buy_key),
            "sell_scope": spread_key_to_dict(self.sell_key),
            "buy_instrument_type": self.buy_instrument_type.value,
            "sell_instrument_type": self.sell_instrument_type.value,
            "buy_price": _decimal_to_payload(self.buy_price),
            "sell_price": _decimal_to_payload(self.sell_price),
            "gross_edge": _decimal_to_payload(self.gross_edge),
            "estimated_fees": _decimal_to_payload(self.estimated_fees),
            "estimated_slippage": _decimal_to_payload(self.estimated_slippage),
            "total_costs": _decimal_to_payload(self.total_costs),
            "net_edge": _decimal_to_payload(self.net_edge),
            "edge_after_costs": _decimal_to_payload(self.edge_after_costs),
            "spread_pct": _decimal_to_payload(self.spread_pct),
            "spread_bps": _decimal_to_payload(self.spread_bps),
            "confidence": _decimal_to_payload(self.confidence),
            "status": self.status.value,
            "timestamp": _datetime_to_payload(self.timestamp),
            "expires_at": _datetime_to_payload(self.expires_at),
            "metadata": _metadata_copy(self.metadata),
            "is_profitable": self.is_profitable,
            "is_active": self.is_active,
            "is_expired": self.is_expired,
            "opportunity_key": self.opportunity_key,
        }


# ============================================================
# Generic serialization helper
# ============================================================

def model_to_payload(model: Any) -> dict[str, Any]:
    """
    Універсальний helper для EventBus/storage/dashboard.

    Якщо модель має власний to_payload() — використовує його.
    Якщо ні — fallback через dataclasses.asdict() з безпечною серіалізацією.
    """
    to_payload = getattr(model, "to_payload", None)
    if callable(to_payload):
        return to_payload()

    if is_dataclass(model):
        return _serialize_value(asdict(model))

    if isinstance(model, Mapping):
        return _serialize_value(model)

    raise TypeError(f"Unsupported model type for payload serialization: {type(model)!r}")


__all__ = [
    "DECIMAL_ZERO",
    "DECIMAL_TWO",
    "DEFAULT_TIMEFRAME",
    "DEFAULT_SPOT_MARKET_TYPE",
    "DEFAULT_PERPETUAL_MARKET_TYPE",
    "DEFAULT_FUTURES_MARKET_TYPE",
    "SpreadKey",
    "make_spread_key",
    "spread_key_to_dict",
    "scoped_metadata",
    "quote_snapshot_from_orderbook_payload",
    "QuoteSnapshot",
    "FundingSnapshot",
    "RollingStats",
    "SpreadSnapshot",
    "SpreadSignal",
    "ArbitrageOpportunity",
    "model_to_payload",
]