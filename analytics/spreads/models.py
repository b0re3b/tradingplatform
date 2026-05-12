from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

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


# ============================================================
# Internal helpers
# ============================================================

def _utcnow() -> datetime:
    """
    Єдина точка створення timestamp.

    Залишаємо naive UTC, бо існуючий код пакета вже використовує
    datetime.utcnow(). Якщо пізніше весь проєкт переводитиметься на
    timezone-aware datetime, це можна буде змінити централізовано.
    """
    return datetime.utcnow()


def _normalize_exchange(exchange: str) -> str:
    return exchange.strip().lower()


def _normalize_symbol(symbol: str) -> str:
    return symbol.replace("-", "").replace("/", "").replace("_", "").upper().strip()


def _validate_non_empty(name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")


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


def _enum_to_payload(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _metadata_copy(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Повертає shallow copy metadata, щоб зовнішній код не мутував модель
    через посилання.
    """
    return dict(metadata)


# ============================================================
# Market Data Models
# ============================================================

@dataclass(slots=True)
class QuoteSnapshot:
    """
    Normalized quote snapshot для spread analytics.

    Використовується як payload для market.quote.updated або як внутрішній
    state analyzer-ів. Модель не залежить від EventBus напряму.
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

    timestamp: datetime = field(default_factory=_utcnow)
    received_at: datetime = field(default_factory=_utcnow)

    sequence_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty("exchange", self.exchange)
        _validate_non_empty("symbol", self.symbol)

        self.exchange = _normalize_exchange(self.exchange)
        self.symbol = _normalize_symbol(self.symbol)
        self.metadata = _metadata_copy(self.metadata)

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

    def to_payload(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
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
            "metadata": _metadata_copy(self.metadata),
        }


@dataclass(slots=True)
class FundingSnapshot:
    """
    Funding snapshot для spot/futures spread analytics.
    """

    exchange: str
    symbol: str
    funding_rate: Decimal

    timestamp: datetime = field(default_factory=_utcnow)
    next_funding_time: datetime | None = None
    predicted_rate: Decimal | None = None
    interval_hours: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty("exchange", self.exchange)
        _validate_non_empty("symbol", self.symbol)

        self.exchange = _normalize_exchange(self.exchange)
        self.symbol = _normalize_symbol(self.symbol)
        self.metadata = _metadata_copy(self.metadata)

        if self.interval_hours is not None and self.interval_hours <= 0:
            raise ValueError("interval_hours must be > 0")

    @property
    def age_ms(self) -> int:
        delta = _utcnow() - self.timestamp
        return max(int(delta.total_seconds() * 1000), 0)

    def to_payload(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "funding_rate": _decimal_to_payload(self.funding_rate),
            "timestamp": _datetime_to_payload(self.timestamp),
            "next_funding_time": _datetime_to_payload(self.next_funding_time),
            "predicted_rate": _decimal_to_payload(self.predicted_rate),
            "interval_hours": self.interval_hours,
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

    Саме цю модель analyzer-и публікують у:
    - analytics.spreads.spot_futures.updated
    - analytics.spreads.cross_exchange.updated
    """

    spread_type: SpreadType
    symbol: str

    leg_a_exchange: str
    leg_b_exchange: str
    leg_a_type: InstrumentType
    leg_b_type: InstrumentType

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
        _validate_non_empty("symbol", self.symbol)
        _validate_non_empty("leg_a_exchange", self.leg_a_exchange)
        _validate_non_empty("leg_b_exchange", self.leg_b_exchange)

        self.symbol = _normalize_symbol(self.symbol)
        self.leg_a_exchange = _normalize_exchange(self.leg_a_exchange)
        self.leg_b_exchange = _normalize_exchange(self.leg_b_exchange)
        self.metadata = _metadata_copy(self.metadata)

        _validate_non_negative_decimal("estimated_fees", self.estimated_fees)
        _validate_non_negative_decimal("estimated_slippage", self.estimated_slippage)

        _validate_positive_decimal("leg_a_bid", self.leg_a_bid)
        _validate_positive_decimal("leg_a_ask", self.leg_a_ask)
        _validate_positive_decimal("leg_b_bid", self.leg_b_bid)
        _validate_positive_decimal("leg_b_ask", self.leg_b_ask)
        _validate_positive_decimal("leg_a_mid", self.leg_a_mid)
        _validate_positive_decimal("leg_b_mid", self.leg_b_mid)

        if (
            self.leg_a_bid is not None
            and self.leg_a_ask is not None
            and self.leg_a_bid > self.leg_a_ask
        ):
            raise ValueError("leg_a_bid must be <= leg_a_ask")

        if (
            self.leg_b_bid is not None
            and self.leg_b_ask is not None
            and self.leg_b_bid > self.leg_b_ask
        ):
            raise ValueError("leg_b_bid must be <= leg_b_ask")

    @property
    def has_edge(self) -> bool:
        return self.net_spread is not None and self.net_spread > DECIMAL_ZERO

    @property
    def abs_spread_bps(self) -> Decimal | None:
        return abs(self.spread_bps) if self.spread_bps is not None else None

    @property
    def pair_key(self) -> tuple[str, str, str, InstrumentType, InstrumentType]:
        return (
            self.symbol,
            self.leg_a_exchange,
            self.leg_b_exchange,
            self.leg_a_type,
            self.leg_b_type,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "spread_type": self.spread_type.value,
            "symbol": self.symbol,
            "leg_a_exchange": self.leg_a_exchange,
            "leg_b_exchange": self.leg_b_exchange,
            "leg_a_type": self.leg_a_type.value,
            "leg_b_type": self.leg_b_type.value,
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
    і отримувати саме цю модель як payload.
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

    timestamp: datetime = field(default_factory=_utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty("symbol", self.symbol)
        _validate_non_empty("message", self.message)

        self.symbol = _normalize_symbol(self.symbol)

        if self.exchange_a is not None:
            self.exchange_a = _normalize_exchange(self.exchange_a)

        if self.exchange_b is not None:
            self.exchange_b = _normalize_exchange(self.exchange_b)

        self.metadata = _metadata_copy(self.metadata)

        if self.confidence is not None:
            if self.confidence < DECIMAL_ZERO or self.confidence > Decimal("1"):
                raise ValueError("confidence must be between 0 and 1")

    @property
    def signal_key(self) -> str:
        exchange_a = self.exchange_a or "na"
        exchange_b = self.exchange_b or "na"
        return (
            f"{self.signal_type.value}|"
            f"{self.spread_type.value}|"
            f"{self.symbol}|"
            f"{exchange_a}|"
            f"{exchange_b}"
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "signal_type": self.signal_type.value,
            "spread_type": self.spread_type.value,
            "symbol": self.symbol,
            "message": self.message,
            "value": _decimal_to_payload(self.value),
            "threshold": _decimal_to_payload(self.threshold),
            "confidence": _decimal_to_payload(self.confidence),
            "exchange_a": self.exchange_a,
            "exchange_b": self.exchange_b,
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
        _validate_non_empty("symbol", self.symbol)
        _validate_non_empty("buy_exchange", self.buy_exchange)
        _validate_non_empty("sell_exchange", self.sell_exchange)

        self.symbol = _normalize_symbol(self.symbol)
        self.buy_exchange = _normalize_exchange(self.buy_exchange)
        self.sell_exchange = _normalize_exchange(self.sell_exchange)
        self.metadata = _metadata_copy(self.metadata)

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
            f"{self.buy_exchange}|"
            f"{self.sell_exchange}|"
            f"{self.buy_instrument_type.value}|"
            f"{self.sell_instrument_type.value}"
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
            "buy_exchange": self.buy_exchange,
            "sell_exchange": self.sell_exchange,
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
    Якщо ні — fallback через dataclasses.asdict().
    """
    to_payload = getattr(model, "to_payload", None)
    if callable(to_payload):
        return to_payload()

    return asdict(model)