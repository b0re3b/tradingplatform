"""
Backtesting models and DTOs.

This module contains data structures used by the offline backtesting pipeline:
historical market data, replay events, simulated orders/fills/positions,
equity curve points, signal/risk/execution records, metrics containers and
final backtest results.

The backtesting package should reuse production analytics/strategy/risk logic,
but it needs its own simulation and reporting models because historical replay
and simulated execution have additional metadata that does not exist in live
trading.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Iterable, Mapping
from uuid import uuid4

from backtesting.enums import (
    BacktestArtifactType,
    BacktestDataType,
    BacktestEventType,
    BacktestMode,
    BacktestStatus,
    BacktestWarningLevel,
    CandleExecutionPath,
    CommissionModel,
    DataAlignmentPolicy,
    DataGapPolicy,
    FillModel,
    FundingSimulationMode,
    HistoricalDataFormat,
    LatencyModel,
    LiquidityModel,
    MetricAggregation,
    OptimizationDirection,
    OptimizationMetric,
    OrderRejectionReason,
    PnLAccountingMode,
    PositionAccountingMode,
    ReplayEventPriority,
    ReplayMode,
    ReplayOrdering,
    ReplaySpeed,
    ReportFormat,
    SimulatedOrderStatus,
    SimulatedPositionStatus,
    SignalOutcome,
    SlippageModel,
    TradeOutcome,
    WalkForwardMode,
    WalkForwardWindowType,
)
from backtesting.exceptions import (
    BacktestConfigurationError,
    HistoricalDataValidationError,
    SimulatedOrderValidationError,
    SimulatedPositionValidationError,
)


# ============================================================================
# Utility helpers
# ============================================================================


Number = int | float | Decimal


def utcnow() -> datetime:
    """
    Return timezone-aware UTC datetime.
    """

    return datetime.now(tz=timezone.utc)


def ensure_aware_utc(value: datetime) -> datetime:
    """
    Normalize datetime to timezone-aware UTC.
    """

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def timestamp_ms(value: datetime | int | float) -> int:
    """
    Convert datetime or numeric timestamp to milliseconds.
    """

    if isinstance(value, datetime):
        return int(ensure_aware_utc(value).timestamp() * 1000)

    numeric_value = float(value)

    # Treat very small values as seconds, large values as milliseconds.
    if numeric_value < 10_000_000_000:
        return int(numeric_value * 1000)

    return int(numeric_value)


def datetime_from_ms(value: int | float) -> datetime:
    """
    Build timezone-aware UTC datetime from millisecond timestamp.
    """

    return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)


def new_id(prefix: str) -> str:
    """
    Create deterministic-looking unique ID for backtesting records.
    """

    return f"{prefix}_{uuid4().hex}"


def safe_div(
    numerator: Number,
    denominator: Number,
    *,
    default: float = 0.0,
) -> float:
    """
    Safe float division helper.
    """

    denominator_float = float(denominator)
    if denominator_float == 0.0:
        return default
    return float(numerator) / denominator_float


def clamp(value: Number, minimum: Number, maximum: Number) -> float:
    """
    Clamp a numeric value.
    """

    return max(float(minimum), min(float(maximum), float(value)))


def _serialize_value(value: Any) -> Any:
    """
    Convert nested dataclasses/enums/datetimes/decimals into JSON-safe values.
    """

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return ensure_aware_utc(value).isoformat()

    if isinstance(value, Decimal):
        return float(value)

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
        return [_serialize_value(item) for item in sorted(value, key=str)]

    return value


class SerializableMixin:
    """
    Small mixin for dataclass DTOs.
    """

    def to_dict(self) -> dict[str, Any]:
        return _serialize_value(self)

    def copy_with(self, **changes: Any) -> Any:
        payload = asdict(self)
        payload.update(changes)
        return self.__class__(**payload)


# ============================================================================
# Backtest identity / period / instrument / dataset metadata
# ============================================================================


@dataclass(slots=True, frozen=True)
class BacktestRunId(SerializableMixin):
    """
    Backtest run identity.
    """

    value: str = field(default_factory=lambda: new_id("bt"))

    def __str__(self) -> str:
        return self.value


@dataclass(slots=True, frozen=True)
class BacktestPeriod(SerializableMixin):
    """
    Time range for a backtest run.
    """

    start: datetime
    end: datetime
    warmup_start: datetime | None = None

    def __post_init__(self) -> None:
        start = ensure_aware_utc(self.start)
        end = ensure_aware_utc(self.end)

        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

        if self.warmup_start is not None:
            object.__setattr__(self, "warmup_start", ensure_aware_utc(self.warmup_start))

        if end <= start:
            raise BacktestConfigurationError(
                "BacktestPeriod.end must be greater than BacktestPeriod.start.",
                details={
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                },
            )

        if self.warmup_start is not None and self.warmup_start > start:
            raise BacktestConfigurationError(
                "BacktestPeriod.warmup_start cannot be after BacktestPeriod.start.",
                details={
                    "warmup_start": self.warmup_start.isoformat(),
                    "start": start.isoformat(),
                },
            )

    @property
    def start_ms(self) -> int:
        return timestamp_ms(self.start)

    @property
    def end_ms(self) -> int:
        return timestamp_ms(self.end)

    @property
    def warmup_start_ms(self) -> int:
        return timestamp_ms(self.warmup_start or self.start)

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    @property
    def duration_seconds(self) -> float:
        return self.duration.total_seconds()

    def contains(
        self,
        value: datetime | int | float,
        *,
        include_warmup: bool = False,
    ) -> bool:
        current_ms = timestamp_ms(value)
        lower_bound = self.warmup_start_ms if include_warmup else self.start_ms
        return lower_bound <= current_ms <= self.end_ms

    def is_warmup(self, value: datetime | int | float) -> bool:
        if self.warmup_start is None:
            return False
        current_ms = timestamp_ms(value)
        return self.warmup_start_ms <= current_ms < self.start_ms


@dataclass(slots=True, frozen=True)
class BacktestInstrument(SerializableMixin):
    """
    Instrument tested by the backtesting system.
    """

    exchange: str
    symbol: str
    market_type: str = "usdm_futures"
    exchange_symbol: str | None = None
    base_asset: str | None = None
    quote_asset: str = "USDT"
    tick_size: float | None = None
    step_size: float | None = None
    min_qty: float | None = None
    min_notional: float | None = None
    contract_size: float = 1.0

    def __post_init__(self) -> None:
        if not self.exchange:
            raise BacktestConfigurationError("BacktestInstrument.exchange is required.")

        if not self.symbol:
            raise BacktestConfigurationError("BacktestInstrument.symbol is required.")

        object.__setattr__(self, "exchange", self.exchange.lower())
        object.__setattr__(self, "symbol", self.symbol.upper())
        object.__setattr__(self, "market_type", self.market_type.lower())

        if self.exchange_symbol is None:
            object.__setattr__(self, "exchange_symbol", self.symbol)

    @property
    def key(self) -> str:
        return f"{self.exchange}:{self.market_type}:{self.symbol}"


@dataclass(slots=True, frozen=True)
class BacktestDataSource(SerializableMixin):
    """
    Source description for historical market data.
    """

    data_type: BacktestDataType
    format: HistoricalDataFormat
    path: str | None = None
    table: str | None = None
    exchange: str | None = None
    symbol: str | None = None
    market_type: str | None = None
    timeframe: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BacktestDatasetInfo(SerializableMixin):
    """
    Dataset metadata and validation summary.
    """

    dataset_id: str = field(default_factory=lambda: new_id("dataset"))
    instruments: list[BacktestInstrument] = field(default_factory=list)
    data_sources: list[BacktestDataSource] = field(default_factory=list)
    period: BacktestPeriod | None = None
    data_types: set[BacktestDataType] = field(default_factory=set)
    total_events: int = 0
    first_event_time: datetime | None = None
    last_event_time: datetime | None = None
    gap_policy: DataGapPolicy = DataGapPolicy.WARN
    alignment_policy: DataAlignmentPolicy = DataAlignmentPolicy.EVENT_TIME
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def validate(self) -> None:
        if self.total_events < 0:
            raise HistoricalDataValidationError(
                "Dataset total_events cannot be negative.",
                details={"total_events": self.total_events},
            )

        if self.first_event_time and self.last_event_time:
            first = ensure_aware_utc(self.first_event_time)
            last = ensure_aware_utc(self.last_event_time)
            if last < first:
                raise HistoricalDataValidationError(
                    "Dataset last_event_time cannot be before first_event_time.",
                    details={
                        "first_event_time": first.isoformat(),
                        "last_event_time": last.isoformat(),
                    },
                )


# ============================================================================
# Historical market data records
# ============================================================================


def _normalize_backtest_exchange(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        raise HistoricalDataValidationError("exchange is required.")
    return normalized


def _normalize_backtest_symbol(value: str) -> str:
    normalized = str(value or "").strip().upper()
    if not normalized:
        raise HistoricalDataValidationError("symbol is required.")
    return normalized


def _normalize_backtest_market_type(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        raise HistoricalDataValidationError("market_type is required.")
    return normalized


def _normalize_backtest_timeframe(value: str | None, *, default: str = "1m") -> str:
    normalized = str(value or default).strip()
    return normalized or default


def _compact_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


@dataclass(slots=True, frozen=True)
class HistoricalMarketRecord(SerializableMixin):
    """
    Base historical market record.

    This is a backtesting DTO only. It does not emit EventBus events by itself.
    MarketReplay converts these records into raw production-compatible market.*
    payloads and sends them through core.EventBus.
    """

    exchange: str
    symbol: str
    market_type: str
    timestamp_ms: int
    received_at_ms: int | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "exchange", _normalize_backtest_exchange(self.exchange))
        object.__setattr__(self, "symbol", _normalize_backtest_symbol(self.symbol))
        object.__setattr__(self, "market_type", _normalize_backtest_market_type(self.market_type))

        if int(self.timestamp_ms) <= 0:
            raise HistoricalDataValidationError(
                "Historical record timestamp_ms must be positive.",
                details={"timestamp_ms": self.timestamp_ms},
            )

        object.__setattr__(self, "timestamp_ms", int(self.timestamp_ms))

        if self.received_at_ms is None:
            object.__setattr__(self, "received_at_ms", self.timestamp_ms)
        else:
            object.__setattr__(self, "received_at_ms", int(self.received_at_ms))

        object.__setattr__(self, "metadata", _compact_metadata(self.metadata))

    @property
    def event_time(self) -> datetime:
        return datetime_from_ms(self.timestamp_ms)

    @property
    def instrument_key(self) -> str:
        return f"{self.exchange}:{self.market_type}:{self.symbol}"

    def _base_market_payload(
        self,
        *,
        data_type: str,
        timeframe: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "market_type": self.market_type,
            "timestamp_ms": self.timestamp_ms,
            "received_at_ms": self.received_at_ms,
            "source": self.source or "backtest",
            "data_type": data_type,
            "metadata": {
                **self.metadata,
                "backtest": True,
                "instrument_key": self.instrument_key,
                "record_type": self.__class__.__name__,
                "data_type": data_type,
            },
        }

        if timeframe is not None:
            payload["timeframe"] = timeframe
            payload["metadata"]["timeframe"] = timeframe

        return payload


@dataclass(slots=True, frozen=True)
class HistoricalCandle(HistoricalMarketRecord):
    """
    Historical OHLCV candle.

    Converts to raw market.candle payload. Production CandlesCache should then
    emit market.candles.updated and market.candle.closed.
    """

    timeframe: str = "1m"
    open_time_ms: int = 0
    close_time_ms: int = 0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    quote_volume: float = 0.0
    trades_count: int = 0
    is_closed: bool = True

    def __post_init__(self) -> None:
        HistoricalMarketRecord.__post_init__(self)

        object.__setattr__(self, "timeframe", _normalize_backtest_timeframe(self.timeframe))

        open_time_ms = int(self.open_time_ms or self.timestamp_ms)
        close_time_ms = int(self.close_time_ms or self.timestamp_ms)

        object.__setattr__(self, "open_time_ms", open_time_ms)
        object.__setattr__(self, "close_time_ms", close_time_ms)

        if open_time_ms <= 0 or close_time_ms <= 0:
            raise HistoricalDataValidationError(
                "HistoricalCandle open_time_ms and close_time_ms must be positive.",
                details={
                    "open_time_ms": open_time_ms,
                    "close_time_ms": close_time_ms,
                },
            )

        if close_time_ms < open_time_ms:
            raise HistoricalDataValidationError(
                "HistoricalCandle close_time_ms cannot be before open_time_ms.",
                details={
                    "open_time_ms": open_time_ms,
                    "close_time_ms": close_time_ms,
                },
            )

        if self.high < self.low:
            raise HistoricalDataValidationError(
                "HistoricalCandle high cannot be less than low.",
                details={
                    "high": self.high,
                    "low": self.low,
                },
            )

        if min(self.open, self.high, self.low, self.close) < 0:
            raise HistoricalDataValidationError(
                "HistoricalCandle OHLC values cannot be negative.",
                details={
                    "open": self.open,
                    "high": self.high,
                    "low": self.low,
                    "close": self.close,
                },
            )

        if self.volume < 0 or self.quote_volume < 0:
            raise HistoricalDataValidationError(
                "HistoricalCandle volume values cannot be negative.",
                details={
                    "volume": self.volume,
                    "quote_volume": self.quote_volume,
                },
            )

    def to_market_event_payload(self) -> dict[str, Any]:
        payload = self._base_market_payload(
            data_type=BacktestDataType.CANDLES.value,
            timeframe=self.timeframe,
        )

        payload.update(
            {
                "open_time_ms": self.open_time_ms,
                "close_time_ms": self.close_time_ms,
                "open": float(self.open),
                "high": float(self.high),
                "low": float(self.low),
                "close": float(self.close),
                "volume": float(self.volume),
                "quote_volume": float(self.quote_volume),
                "trades_count": int(self.trades_count),
                "is_closed": bool(self.is_closed),
            }
        )

        return payload


@dataclass(slots=True, frozen=True)
class HistoricalTrade(HistoricalMarketRecord):
    """
    Historical trade record.

    Converts to raw market.trade payload. Production TradesCache should then
    emit market.trades.updated.
    """

    trade_id: str | int | None = None
    price: float = 0.0
    quantity: float = 0.0
    quote_quantity: float | None = None
    side: str | None = None
    aggressor_side: str | None = None
    buyer_maker: bool | None = None
    timeframe: str | None = None

    def __post_init__(self) -> None:
        HistoricalMarketRecord.__post_init__(self)

        if self.price <= 0:
            raise HistoricalDataValidationError(
                "HistoricalTrade.price must be positive.",
                details={"price": self.price},
            )

        if self.quantity <= 0:
            raise HistoricalDataValidationError(
                "HistoricalTrade.quantity must be positive.",
                details={"quantity": self.quantity},
            )

        if self.quote_quantity is None:
            object.__setattr__(self, "quote_quantity", float(self.price) * float(self.quantity))
        elif self.quote_quantity < 0:
            raise HistoricalDataValidationError(
                "HistoricalTrade.quote_quantity cannot be negative.",
                details={"quote_quantity": self.quote_quantity},
            )

        if self.timeframe is not None:
            object.__setattr__(self, "timeframe", _normalize_backtest_timeframe(self.timeframe))

    def to_market_event_payload(self) -> dict[str, Any]:
        payload = self._base_market_payload(
            data_type=BacktestDataType.TRADES.value,
            timeframe=self.timeframe,
        )

        notional = float(self.quote_quantity or (self.price * self.quantity))

        payload.update(
            {
                "trade_id": str(self.trade_id) if self.trade_id is not None else None,
                "price": float(self.price),
                "quantity": float(self.quantity),
                "qty": float(self.quantity),
                "quote_quantity": notional,
                "quote_volume": notional,
                "notional": notional,
                "side": self.side,
                "aggressor_side": self.aggressor_side or self.side,
                "buyer_maker": self.buyer_maker,
                "is_buyer_maker": self.buyer_maker,
            }
        )

        return payload


@dataclass(slots=True, frozen=True)
class HistoricalOrderBookLevel(SerializableMixin):
    """
    Historical order book level.
    """

    price: float
    quantity: float

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise HistoricalDataValidationError(
                "Order book level price must be positive.",
                details={"price": self.price},
            )

        if self.quantity < 0:
            raise HistoricalDataValidationError(
                "Order book level quantity cannot be negative.",
                details={"quantity": self.quantity},
            )


@dataclass(slots=True, frozen=True)
class HistoricalOrderBookSnapshot(HistoricalMarketRecord):
    """
    Historical order book snapshot.

    Converts to raw market.orderbook snapshot payload. Production OrderBookCache
    should then emit market.orderbook.updated.
    """

    bids: list[HistoricalOrderBookLevel] = field(default_factory=list)
    asks: list[HistoricalOrderBookLevel] = field(default_factory=list)
    sequence: int | None = None
    depth: int | None = None
    timeframe: str | None = None

    def __post_init__(self) -> None:
        HistoricalMarketRecord.__post_init__(self)

        if not self.bids and not self.asks:
            raise HistoricalDataValidationError(
                "HistoricalOrderBookSnapshot must contain bids or asks."
            )

        if self.depth is None:
            object.__setattr__(self, "depth", max(len(self.bids), len(self.asks)))
        else:
            object.__setattr__(self, "depth", int(self.depth))

        if self.timeframe is not None:
            object.__setattr__(self, "timeframe", _normalize_backtest_timeframe(self.timeframe))

    @property
    def best_bid(self) -> float | None:
        return max((level.price for level in self.bids), default=None)

    @property
    def best_ask(self) -> float | None:
        return min((level.price for level in self.asks), default=None)

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    def to_market_event_payload(self) -> dict[str, Any]:
        payload = self._base_market_payload(
            data_type=BacktestDataType.ORDERBOOK_SNAPSHOT.value,
            timeframe=self.timeframe,
        )

        payload.update(
            {
                "type": "snapshot",
                "update_type": "snapshot",
                "is_snapshot": True,
                "sequence": self.sequence,
                "last_update_id": self.sequence,
                "depth": self.depth,
                "bids": [[float(level.price), float(level.quantity)] for level in self.bids],
                "asks": [[float(level.price), float(level.quantity)] for level in self.asks],
                "best_bid": self.best_bid,
                "best_ask": self.best_ask,
                "spread": self.spread,
            }
        )

        return payload


@dataclass(slots=True, frozen=True)
class HistoricalFundingRecord(HistoricalMarketRecord):
    """
    Historical funding rate record.

    Converts to raw market.funding payload. Production FundingCache should then
    emit market.funding.updated.
    """

    funding_rate: float = 0.0
    predicted_rate: float | None = None
    mark_price: float | None = None
    index_price: float | None = None
    next_funding_time_ms: int | None = None
    timeframe: str | None = None

    def __post_init__(self) -> None:
        HistoricalMarketRecord.__post_init__(self)

        if self.timeframe is not None:
            object.__setattr__(self, "timeframe", _normalize_backtest_timeframe(self.timeframe))

        if self.next_funding_time_ms is not None:
            object.__setattr__(self, "next_funding_time_ms", int(self.next_funding_time_ms))

    def to_market_event_payload(self) -> dict[str, Any]:
        payload = self._base_market_payload(
            data_type=BacktestDataType.FUNDING.value,
            timeframe=self.timeframe,
        )

        payload.update(
            {
                "funding_rate": float(self.funding_rate),
                "rate": float(self.funding_rate),
                "predicted_rate": self.predicted_rate,
                "mark_price": self.mark_price,
                "index_price": self.index_price,
                "next_funding_time_ms": self.next_funding_time_ms,
            }
        )

        return payload


@dataclass(slots=True, frozen=True)
class HistoricalOpenInterestRecord(HistoricalMarketRecord):
    """
    Historical open interest record.

    Converts to raw market.open_interest payload. Production OpenInterestCache
    should then emit market.open_interest.updated.
    """

    open_interest: float = 0.0
    open_interest_value: float | None = None
    mark_price: float | None = None
    timeframe: str | None = None

    def __post_init__(self) -> None:
        HistoricalMarketRecord.__post_init__(self)

        if self.open_interest < 0:
            raise HistoricalDataValidationError(
                "HistoricalOpenInterestRecord.open_interest cannot be negative.",
                details={"open_interest": self.open_interest},
            )

        if self.open_interest_value is None and self.mark_price is not None:
            object.__setattr__(
                self,
                "open_interest_value",
                float(self.open_interest) * float(self.mark_price),
            )

        if self.timeframe is not None:
            object.__setattr__(self, "timeframe", _normalize_backtest_timeframe(self.timeframe))

    def to_market_event_payload(self) -> dict[str, Any]:
        payload = self._base_market_payload(
            data_type=BacktestDataType.OPEN_INTEREST.value,
            timeframe=self.timeframe,
        )

        payload.update(
            {
                "open_interest": float(self.open_interest),
                "open_interest_qty": float(self.open_interest),
                "open_interest_value": self.open_interest_value,
                "mark_price": self.mark_price,
            }
        )

        return payload


@dataclass(slots=True, frozen=True)
class HistoricalLiquidationRecord(HistoricalMarketRecord):
    """
    Historical liquidation event.

    Converts to raw market.liquidation payload. Production liquidation analytics
    or liquidation data cache should then emit analytics/liquidation updates
    according to the production pipeline.
    """

    liquidation_id: str | int | None = None
    side: str = ""
    price: float = 0.0
    quantity: float = 0.0
    notional: float | None = None
    timeframe: str | None = None

    def __post_init__(self) -> None:
        HistoricalMarketRecord.__post_init__(self)

        if self.price <= 0:
            raise HistoricalDataValidationError(
                "HistoricalLiquidationRecord.price must be positive.",
                details={"price": self.price},
            )

        if self.quantity <= 0:
            raise HistoricalDataValidationError(
                "HistoricalLiquidationRecord.quantity must be positive.",
                details={"quantity": self.quantity},
            )

        if self.notional is None:
            object.__setattr__(self, "notional", float(self.price) * float(self.quantity))
        elif self.notional < 0:
            raise HistoricalDataValidationError(
                "HistoricalLiquidationRecord.notional cannot be negative.",
                details={"notional": self.notional},
            )

        if self.timeframe is not None:
            object.__setattr__(self, "timeframe", _normalize_backtest_timeframe(self.timeframe))

    def to_market_event_payload(self) -> dict[str, Any]:
        payload = self._base_market_payload(
            data_type=BacktestDataType.LIQUIDATIONS.value,
            timeframe=self.timeframe,
        )

        payload.update(
            {
                "liquidation_id": str(self.liquidation_id)
                if self.liquidation_id is not None
                else None,
                "side": self.side,
                "price": float(self.price),
                "quantity": float(self.quantity),
                "qty": float(self.quantity),
                "notional": float(self.notional or 0.0),
            }
        )

        return payload


# ============================================================================
# Replay events / dataset
# ============================================================================


@dataclass(slots=True, frozen=True)
class BacktestEvent(SerializableMixin):
    """
    Generic event recorded or emitted during a backtest run.

    For market replay, topic must be production-compatible raw market topic:
    market.candle, market.trade, market.orderbook, market.funding,
    market.open_interest, market.liquidation.

    BacktestEvent itself does not emit EventBus events.
    """

    event_id: str = field(default_factory=lambda: new_id("evt"))
    run_id: str | None = None
    event_type: BacktestEventType = BacktestEventType.SYSTEM
    topic: str = ""
    timestamp_ms: int = 0
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = "backtest"
    sequence: int | None = None
    priority: ReplayEventPriority | None = None
    is_warmup: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.timestamp_ms) < 0:
            raise HistoricalDataValidationError(
                "BacktestEvent.timestamp_ms cannot be negative.",
                details={"timestamp_ms": self.timestamp_ms},
            )

        object.__setattr__(self, "timestamp_ms", int(self.timestamp_ms))
        object.__setattr__(self, "metadata", _compact_metadata(self.metadata))
        object.__setattr__(self, "payload", dict(self.payload or {}))

        if self.event_type == BacktestEventType.MARKET:
            if not self.topic.startswith("market."):
                raise HistoricalDataValidationError(
                    "Market BacktestEvent.topic must start with 'market.'.",
                    details={
                        "topic": self.topic,
                        "event_id": self.event_id,
                    },
                )

            if self.topic.endswith(".updated"):
                raise HistoricalDataValidationError(
                    "MarketReplay events must be raw market.* topics, not market.*.updated.",
                    details={
                        "topic": self.topic,
                        "event_id": self.event_id,
                    },
                )

    @property
    def event_time(self) -> datetime:
        return datetime_from_ms(self.timestamp_ms)

    def with_run_id(self, run_id: str) -> BacktestEvent:
        return self.copy_with(run_id=run_id)

    @classmethod
    def from_market_record(
        cls,
        record: HistoricalMarketRecord,
        *,
        topic: str,
        data_type: BacktestDataType,
        run_id: str | None = None,
        sequence: int | None = None,
        priority: ReplayEventPriority | None = None,
        is_warmup: bool = False,
    ) -> BacktestEvent:
        payload_method = getattr(record, "to_market_event_payload")
        payload = payload_method()

        return cls(
            run_id=run_id,
            event_type=BacktestEventType.MARKET,
            topic=topic,
            timestamp_ms=record.timestamp_ms,
            payload=payload,
            source="market_replay",
            sequence=sequence,
            priority=priority,
            is_warmup=is_warmup,
            metadata={
                "data_type": data_type.value,
                "instrument_key": record.instrument_key,
                "record_type": record.__class__.__name__,
            },
        )


@dataclass(slots=True)
class ReplayEventBatch(SerializableMixin):
    """
    Batch of replay events sharing a timestamp.
    """

    batch_id: str = field(default_factory=lambda: new_id("batch"))
    timestamp_ms: int = 0
    events: list[BacktestEvent] = field(default_factory=list)
    sequence_start: int | None = None
    sequence_end: int | None = None
    is_warmup: bool = False

    def __post_init__(self) -> None:
        if int(self.timestamp_ms) < 0:
            raise HistoricalDataValidationError(
                "ReplayEventBatch.timestamp_ms cannot be negative.",
                details={"timestamp_ms": self.timestamp_ms},
            )

        self.events.sort(
            key=lambda event: (
                event.timestamp_ms,
                event.sequence if event.sequence is not None else 0,
            )
        )

    @property
    def size(self) -> int:
        return len(self.events)

    @property
    def event_time(self) -> datetime:
        return datetime_from_ms(self.timestamp_ms)


@dataclass(slots=True)
class BacktestDataset(SerializableMixin):
    """
    Replay-ready historical dataset.

    Dataset stores BacktestEvent objects. It does not emit EventBus events.
    MarketReplay is responsible for replaying these events into core.EventBus.
    """

    info: BacktestDatasetInfo = field(default_factory=BacktestDatasetInfo)
    events: list[BacktestEvent] = field(default_factory=list)
    ordering: ReplayOrdering = ReplayOrdering.TIMESTAMP_THEN_PRIORITY
    replay_mode: ReplayMode = ReplayMode.FULL_RUN
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sort_events()
        self.info.total_events = len(self.events)

        if self.events:
            self.info.first_event_time = self.events[0].event_time
            self.info.last_event_time = self.events[-1].event_time

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterable[BacktestEvent]:
        return iter(self.events)

    @property
    def is_empty(self) -> bool:
        return len(self.events) == 0

    def sort_events(self) -> None:
        priority_order = {
            ReplayEventPriority.ORDERBOOK: 10,
            ReplayEventPriority.TRADE: 20,
            ReplayEventPriority.CANDLE: 30,
            ReplayEventPriority.FUNDING: 40,
            ReplayEventPriority.OPEN_INTEREST: 50,
            ReplayEventPriority.LIQUIDATION: 60,
            ReplayEventPriority.MARK_PRICE: 70,
            ReplayEventPriority.INDEX_PRICE: 80,
            None: 999,
        }

        if self.ordering == ReplayOrdering.TIMESTAMP_ASC:
            self.events.sort(
                key=lambda event: (
                    event.timestamp_ms,
                    event.sequence if event.sequence is not None else 0,
                )
            )
            return

        if self.ordering == ReplayOrdering.TIMESTAMP_THEN_PRIORITY:
            self.events.sort(
                key=lambda event: (
                    event.timestamp_ms,
                    priority_order.get(event.priority, 999),
                    event.sequence if event.sequence is not None else 0,
                )
            )
            return

        if self.ordering == ReplayOrdering.STREAM_PRIORITY_THEN_TIMESTAMP:
            self.events.sort(
                key=lambda event: (
                    priority_order.get(event.priority, 999),
                    event.timestamp_ms,
                    event.sequence if event.sequence is not None else 0,
                )
            )
            return

    def batches_by_timestamp(self) -> list[ReplayEventBatch]:
        batches: list[ReplayEventBatch] = []
        current_timestamp: int | None = None
        current_events: list[BacktestEvent] = []

        for event in self.events:
            if current_timestamp is None:
                current_timestamp = event.timestamp_ms

            if event.timestamp_ms != current_timestamp:
                batches.append(
                    ReplayEventBatch(
                        timestamp_ms=current_timestamp,
                        events=current_events,
                        sequence_start=current_events[0].sequence if current_events else None,
                        sequence_end=current_events[-1].sequence if current_events else None,
                        is_warmup=all(item.is_warmup for item in current_events),
                    )
                )
                current_timestamp = event.timestamp_ms
                current_events = []

            current_events.append(event)

        if current_timestamp is not None and current_events:
            batches.append(
                ReplayEventBatch(
                    timestamp_ms=current_timestamp,
                    events=current_events,
                    sequence_start=current_events[0].sequence if current_events else None,
                    sequence_end=current_events[-1].sequence if current_events else None,
                    is_warmup=all(item.is_warmup for item in current_events),
                )
            )

        return batches


# ============================================================================
# Replay events / dataset
# ============================================================================


@dataclass(slots=True, frozen=True)
class BacktestEvent(SerializableMixin):
    """
    Generic event recorded or emitted during a backtest run.

    For market replay, topic should be production-compatible:
    market.candle, market.trade, market.orderbook, market.funding,
    market.open_interest, etc.
    """

    event_id: str = field(default_factory=lambda: new_id("evt"))
    run_id: str | None = None
    event_type: BacktestEventType = BacktestEventType.SYSTEM
    topic: str = ""
    timestamp_ms: int = 0
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = "backtest"
    sequence: int | None = None
    priority: ReplayEventPriority | None = None
    is_warmup: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0:
            raise HistoricalDataValidationError(
                "BacktestEvent.timestamp_ms cannot be negative.",
                details={"timestamp_ms": self.timestamp_ms},
            )

    @property
    def event_time(self) -> datetime:
        return datetime_from_ms(self.timestamp_ms)

    def with_run_id(self, run_id: str) -> BacktestEvent:
        return self.copy_with(run_id=run_id)

    @classmethod
    def from_market_record(
        cls,
        record: HistoricalMarketRecord,
        *,
        topic: str,
        data_type: BacktestDataType,
        run_id: str | None = None,
        sequence: int | None = None,
        priority: ReplayEventPriority | None = None,
        is_warmup: bool = False,
    ) -> BacktestEvent:
        payload_method = getattr(record, "to_market_event_payload")
        payload = payload_method()

        return cls(
            run_id=run_id,
            event_type=BacktestEventType.MARKET,
            topic=topic,
            timestamp_ms=record.timestamp_ms,
            payload=payload,
            source="market_replay",
            sequence=sequence,
            priority=priority,
            is_warmup=is_warmup,
            metadata={
                "data_type": data_type.value,
                "instrument_key": record.instrument_key,
            },
        )


@dataclass(slots=True)
class ReplayEventBatch(SerializableMixin):
    """
    Batch of replay events sharing a timestamp or grouped for faster emission.
    """

    batch_id: str = field(default_factory=lambda: new_id("batch"))
    timestamp_ms: int = 0
    events: list[BacktestEvent] = field(default_factory=list)
    sequence_start: int | None = None
    sequence_end: int | None = None
    is_warmup: bool = False

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0:
            raise HistoricalDataValidationError(
                "ReplayEventBatch.timestamp_ms cannot be negative.",
                details={"timestamp_ms": self.timestamp_ms},
            )

        if self.events:
            self.events.sort(key=lambda event: (event.timestamp_ms, event.sequence or 0))

    @property
    def size(self) -> int:
        return len(self.events)

    @property
    def event_time(self) -> datetime:
        return datetime_from_ms(self.timestamp_ms)


@dataclass(slots=True)
class BacktestDataset(SerializableMixin):
    """
    Replay-ready historical dataset.
    """

    info: BacktestDatasetInfo = field(default_factory=BacktestDatasetInfo)
    events: list[BacktestEvent] = field(default_factory=list)
    ordering: ReplayOrdering = ReplayOrdering.TIMESTAMP_THEN_PRIORITY
    replay_mode: ReplayMode = ReplayMode.FULL_RUN
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sort_events()
        self.info.total_events = len(self.events)

        if self.events:
            self.info.first_event_time = self.events[0].event_time
            self.info.last_event_time = self.events[-1].event_time

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterable[BacktestEvent]:
        return iter(self.events)

    @property
    def is_empty(self) -> bool:
        return len(self.events) == 0

    def sort_events(self) -> None:
        priority_order = {
            ReplayEventPriority.ORDERBOOK: 10,
            ReplayEventPriority.TRADE: 20,
            ReplayEventPriority.CANDLE: 30,
            ReplayEventPriority.FUNDING: 40,
            ReplayEventPriority.OPEN_INTEREST: 50,
            ReplayEventPriority.LIQUIDATION: 60,
            ReplayEventPriority.MARK_PRICE: 70,
            ReplayEventPriority.INDEX_PRICE: 80,
            None: 999,
        }

        if self.ordering == ReplayOrdering.TIMESTAMP_ASC:
            self.events.sort(key=lambda event: (event.timestamp_ms, event.sequence or 0))
            return

        if self.ordering == ReplayOrdering.TIMESTAMP_THEN_PRIORITY:
            self.events.sort(
                key=lambda event: (
                    event.timestamp_ms,
                    priority_order.get(event.priority, 999),
                    event.sequence or 0,
                )
            )
            return

        if self.ordering == ReplayOrdering.STREAM_PRIORITY_THEN_TIMESTAMP:
            self.events.sort(
                key=lambda event: (
                    priority_order.get(event.priority, 999),
                    event.timestamp_ms,
                    event.sequence or 0,
                )
            )

    def batches_by_timestamp(self) -> list[ReplayEventBatch]:
        batches: list[ReplayEventBatch] = []
        current_timestamp: int | None = None
        current_events: list[BacktestEvent] = []

        for event in self.events:
            if current_timestamp is None:
                current_timestamp = event.timestamp_ms

            if event.timestamp_ms != current_timestamp:
                batches.append(
                    ReplayEventBatch(
                        timestamp_ms=current_timestamp,
                        events=current_events,
                        is_warmup=all(item.is_warmup for item in current_events),
                    )
                )
                current_timestamp = event.timestamp_ms
                current_events = []

            current_events.append(event)

        if current_timestamp is not None and current_events:
            batches.append(
                ReplayEventBatch(
                    timestamp_ms=current_timestamp,
                    events=current_events,
                    is_warmup=all(item.is_warmup for item in current_events),
                )
            )

        return batches


# ============================================================================
# Backtest clock state
# ============================================================================


@dataclass(slots=True)
class BacktestClockState(SerializableMixin):
    """
    Runtime state of simulated backtest time.
    """

    period: BacktestPeriod
    current_time: datetime | None = None
    current_timestamp_ms: int | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    events_processed: int = 0
    total_events: int = 0
    replay_speed: ReplaySpeed = ReplaySpeed.MAX_SPEED

    def __post_init__(self) -> None:
        if self.current_time is None:
            self.current_time = self.period.warmup_start or self.period.start

        self.current_time = ensure_aware_utc(self.current_time)

        if self.current_timestamp_ms is None:
            self.current_timestamp_ms = timestamp_ms(self.current_time)

    @property
    def now(self) -> datetime:
        return self.current_time or self.period.start

    @property
    def progress_pct(self) -> float:
        if self.total_events > 0:
            return clamp(self.events_processed / self.total_events * 100.0, 0.0, 100.0)

        total = self.period.end_ms - self.period.warmup_start_ms
        if total <= 0:
            return 0.0

        current = (self.current_timestamp_ms or self.period.warmup_start_ms) - self.period.warmup_start_ms
        return clamp(current / total * 100.0, 0.0, 100.0)

    @property
    def is_warmup(self) -> bool:
        return self.period.is_warmup(self.now)


# ============================================================================
# Simulation orders / fills / execution
# ============================================================================


@dataclass(slots=True)
class SimulatedOrder(SerializableMixin):
    """
    Simulated order tracked by ExecutionSimulator.
    """

    order_id: str = field(default_factory=lambda: new_id("sim_order"))
    client_order_id: str | None = None
    run_id: str | None = None
    signal_id: str | None = None
    strategy_name: str | None = None
    exchange: str = "binance"
    symbol: str = ""
    market_type: str = "usdm_futures"
    side: str = ""
    order_type: str = "market"
    status: SimulatedOrderStatus = SimulatedOrderStatus.CREATED
    quantity: float = 0.0
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    price: float | None = None
    stop_price: float | None = None
    average_fill_price: float | None = None
    reduce_only: bool = False
    close_position: bool = False
    time_in_force: str | None = None
    leverage: float | None = None
    submitted_at_ms: int | None = None
    accepted_at_ms: int | None = None
    filled_at_ms: int | None = None
    cancelled_at_ms: int | None = None
    rejected_at_ms: int | None = None
    rejection_reason: OrderRejectionReason = OrderRejectionReason.NONE
    rejection_message: str | None = None
    fees: float = 0.0
    slippage: float = 0.0
    latency_ms: int = 0
    source_payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol:
            raise SimulatedOrderValidationError("SimulatedOrder.symbol is required.")

        if self.quantity <= 0:
            raise SimulatedOrderValidationError(
                "SimulatedOrder.quantity must be positive.",
                details={"quantity": self.quantity},
            )

        if self.filled_quantity < 0:
            raise SimulatedOrderValidationError(
                "SimulatedOrder.filled_quantity cannot be negative.",
                details={"filled_quantity": self.filled_quantity},
            )

        if self.remaining_quantity == 0.0:
            self.remaining_quantity = max(0.0, self.quantity - self.filled_quantity)

        self.exchange = self.exchange.lower()
        self.symbol = self.symbol.upper()
        self.market_type = self.market_type.lower()

    @property
    def is_active(self) -> bool:
        return self.status in {
            SimulatedOrderStatus.CREATED,
            SimulatedOrderStatus.SUBMITTED,
            SimulatedOrderStatus.ACCEPTED,
            SimulatedOrderStatus.PARTIALLY_FILLED,
        }

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            SimulatedOrderStatus.FILLED,
            SimulatedOrderStatus.CANCELLED,
            SimulatedOrderStatus.REJECTED,
            SimulatedOrderStatus.EXPIRED,
            SimulatedOrderStatus.FAILED,
        }

    @property
    def notional(self) -> float:
        reference_price = self.average_fill_price or self.price or 0.0
        return abs(self.quantity * reference_price)

    @property
    def filled_notional(self) -> float:
        reference_price = self.average_fill_price or self.price or 0.0
        return abs(self.filled_quantity * reference_price)

    def mark_submitted(self, timestamp_ms_value: int) -> None:
        self.status = SimulatedOrderStatus.SUBMITTED
        self.submitted_at_ms = timestamp_ms_value

    def mark_accepted(self, timestamp_ms_value: int) -> None:
        self.status = SimulatedOrderStatus.ACCEPTED
        self.accepted_at_ms = timestamp_ms_value

    def mark_rejected(
        self,
        reason: OrderRejectionReason,
        *,
        message: str | None = None,
        timestamp_ms_value: int | None = None,
    ) -> None:
        self.status = SimulatedOrderStatus.REJECTED
        self.rejection_reason = reason
        self.rejection_message = message
        self.rejected_at_ms = timestamp_ms_value

    def apply_fill(self, fill: SimulatedFill) -> None:
        previous_notional = (self.average_fill_price or 0.0) * self.filled_quantity
        new_notional = fill.price * fill.quantity
        new_quantity = self.filled_quantity + fill.quantity

        self.filled_quantity = new_quantity
        self.remaining_quantity = max(0.0, self.quantity - self.filled_quantity)

        if new_quantity > 0:
            self.average_fill_price = (previous_notional + new_notional) / new_quantity

        self.fees += fill.fee
        self.slippage += fill.slippage

        if self.remaining_quantity <= 0:
            self.status = SimulatedOrderStatus.FILLED
            self.filled_at_ms = fill.timestamp_ms
        else:
            self.status = SimulatedOrderStatus.PARTIALLY_FILLED


@dataclass(slots=True)
class SimulatedFill(SerializableMixin):
    """
    Simulated execution fill.
    """

    fill_id: str = field(default_factory=lambda: new_id("sim_fill"))
    order_id: str = ""
    run_id: str | None = None
    signal_id: str | None = None
    position_id: str | None = None
    exchange: str = "binance"
    symbol: str = ""
    market_type: str = "usdm_futures"
    side: str = ""
    price: float = 0.0
    quantity: float = 0.0
    notional: float = 0.0
    fee: float = 0.0
    fee_asset: str = "USDT"
    slippage: float = 0.0
    slippage_bps: float = 0.0
    liquidity_type: str | None = None
    timestamp_ms: int = 0
    source_event_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.order_id:
            raise SimulatedOrderValidationError("SimulatedFill.order_id is required.")

        if not self.symbol:
            raise SimulatedOrderValidationError("SimulatedFill.symbol is required.")

        if self.price <= 0:
            raise SimulatedOrderValidationError(
                "SimulatedFill.price must be positive.",
                details={"price": self.price},
            )

        if self.quantity <= 0:
            raise SimulatedOrderValidationError(
                "SimulatedFill.quantity must be positive.",
                details={"quantity": self.quantity},
            )

        if self.timestamp_ms <= 0:
            raise SimulatedOrderValidationError(
                "SimulatedFill.timestamp_ms must be positive.",
                details={"timestamp_ms": self.timestamp_ms},
            )

        if self.notional == 0.0:
            self.notional = abs(self.price * self.quantity)

        self.exchange = self.exchange.lower()
        self.symbol = self.symbol.upper()
        self.market_type = self.market_type.lower()

    @property
    def event_time(self) -> datetime:
        return datetime_from_ms(self.timestamp_ms)


@dataclass(slots=True)
class BacktestExecutionRecord(SerializableMixin):
    """
    Audit record for execution simulator events.
    """

    record_id: str = field(default_factory=lambda: new_id("exec_rec"))
    run_id: str | None = None
    timestamp_ms: int = 0
    topic: str = ""
    order_id: str | None = None
    fill_id: str | None = None
    signal_id: str | None = None
    strategy_name: str | None = None
    symbol: str | None = None
    status: SimulatedOrderStatus | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Simulated positions / trades / balance / equity
# ============================================================================


@dataclass(slots=True)
class SimulatedPosition(SerializableMixin):
    """
    Simulated futures position tracked by PositionSimulator.
    """

    position_id: str = field(default_factory=lambda: new_id("sim_pos"))
    run_id: str | None = None
    signal_id: str | None = None
    strategy_name: str | None = None
    exchange: str = "binance"
    symbol: str = ""
    market_type: str = "usdm_futures"
    side: str = ""
    status: SimulatedPositionStatus = SimulatedPositionStatus.NONE
    quantity: float = 0.0
    entry_price: float = 0.0
    mark_price: float = 0.0
    exit_price: float | None = None
    leverage: float = 1.0
    margin: float = 0.0
    notional: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees_paid: float = 0.0
    funding_paid: float = 0.0
    funding_received: float = 0.0
    slippage_paid: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    liquidation_price: float | None = None
    opened_at_ms: int | None = None
    updated_at_ms: int | None = None
    closed_at_ms: int | None = None
    close_reason: str | None = None
    source_order_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol:
            raise SimulatedPositionValidationError("SimulatedPosition.symbol is required.")

        if self.quantity < 0:
            raise SimulatedPositionValidationError(
                "SimulatedPosition.quantity cannot be negative.",
                details={"quantity": self.quantity},
            )

        if self.entry_price < 0:
            raise SimulatedPositionValidationError(
                "SimulatedPosition.entry_price cannot be negative.",
                details={"entry_price": self.entry_price},
            )

        if self.leverage <= 0:
            raise SimulatedPositionValidationError(
                "SimulatedPosition.leverage must be positive.",
                details={"leverage": self.leverage},
            )

        self.exchange = self.exchange.lower()
        self.symbol = self.symbol.upper()
        self.market_type = self.market_type.lower()

        if self.mark_price == 0.0 and self.entry_price > 0:
            self.mark_price = self.entry_price

        if self.notional == 0.0 and self.entry_price > 0:
            self.notional = abs(self.quantity * self.entry_price)

        if self.margin == 0.0 and self.notional > 0:
            self.margin = self.notional / self.leverage

    @property
    def is_open(self) -> bool:
        return self.status in {
            SimulatedPositionStatus.OPENING,
            SimulatedPositionStatus.OPEN,
            SimulatedPositionStatus.REDUCING,
            SimulatedPositionStatus.CLOSING,
        }

    @property
    def net_funding(self) -> float:
        return self.funding_received - self.funding_paid

    @property
    def total_costs(self) -> float:
        return self.fees_paid + self.slippage_paid + self.funding_paid - self.funding_received

    @property
    def net_realized_pnl(self) -> float:
        return self.realized_pnl - self.fees_paid - self.slippage_paid + self.net_funding

    @property
    def holding_time_seconds(self) -> float:
        if self.opened_at_ms is None:
            return 0.0

        end = self.closed_at_ms or self.updated_at_ms or self.opened_at_ms
        return max(0.0, (end - self.opened_at_ms) / 1000.0)

    def update_mark_price(self, price: float, timestamp_ms_value: int) -> None:
        if price <= 0:
            raise SimulatedPositionValidationError(
                "Mark price must be positive.",
                details={"price": price},
            )

        self.mark_price = price
        self.updated_at_ms = timestamp_ms_value

        if self.side.lower() in {"buy", "long"}:
            self.unrealized_pnl = (self.mark_price - self.entry_price) * self.quantity
        elif self.side.lower() in {"sell", "short"}:
            self.unrealized_pnl = (self.entry_price - self.mark_price) * self.quantity

    def close(
        self,
        *,
        exit_price: float,
        timestamp_ms_value: int,
        reason: str,
    ) -> None:
        if exit_price <= 0:
            raise SimulatedPositionValidationError(
                "Exit price must be positive.",
                details={"exit_price": exit_price},
            )

        self.exit_price = exit_price
        self.mark_price = exit_price
        self.closed_at_ms = timestamp_ms_value
        self.updated_at_ms = timestamp_ms_value
        self.close_reason = reason
        self.status = SimulatedPositionStatus.CLOSED

        if self.side.lower() in {"buy", "long"}:
            self.realized_pnl = (exit_price - self.entry_price) * self.quantity
        elif self.side.lower() in {"sell", "short"}:
            self.realized_pnl = (self.entry_price - exit_price) * self.quantity

        self.unrealized_pnl = 0.0


@dataclass(slots=True)
class SimulatedTrade(SerializableMixin):
    """
    Completed or open trade reconstructed from simulated position lifecycle.
    """

    trade_id: str = field(default_factory=lambda: new_id("sim_trade"))
    run_id: str | None = None
    position_id: str | None = None
    signal_id: str | None = None
    strategy_name: str | None = None
    exchange: str = "binance"
    symbol: str = ""
    market_type: str = "usdm_futures"
    side: str = ""
    quantity: float = 0.0
    entry_price: float = 0.0
    exit_price: float | None = None
    opened_at_ms: int | None = None
    closed_at_ms: int | None = None
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    pnl_pct: float = 0.0
    r_multiple: float | None = None
    fees: float = 0.0
    slippage: float = 0.0
    funding: float = 0.0
    outcome: TradeOutcome = TradeOutcome.OPEN
    close_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_closed(self) -> bool:
        return self.outcome != TradeOutcome.OPEN and self.closed_at_ms is not None

    @property
    def holding_time_seconds(self) -> float:
        if self.opened_at_ms is None or self.closed_at_ms is None:
            return 0.0
        return max(0.0, (self.closed_at_ms - self.opened_at_ms) / 1000.0)


@dataclass(slots=True)
class SimulatedBalance(SerializableMixin):
    """
    Simulated account balance state.
    """

    currency: str = "USDT"
    initial_balance: float = 0.0
    cash_balance: float = 0.0
    available_balance: float = 0.0
    margin_used: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    equity: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_funding: float = 0.0
    updated_at_ms: int | None = None

    def __post_init__(self) -> None:
        if self.initial_balance < 0:
            raise SimulatedPositionValidationError(
                "Initial balance cannot be negative.",
                details={"initial_balance": self.initial_balance},
            )

        if self.cash_balance == 0.0:
            self.cash_balance = self.initial_balance

        if self.available_balance == 0.0:
            self.available_balance = self.cash_balance

        if self.equity == 0.0:
            self.equity = self.cash_balance + self.unrealized_pnl

    @property
    def net_profit(self) -> float:
        return self.equity - self.initial_balance

    @property
    def net_profit_pct(self) -> float:
        return safe_div(self.net_profit, self.initial_balance) * 100.0


@dataclass(slots=True)
class SimulatedEquityPoint(SerializableMixin):
    """
    Equity curve point.
    """

    timestamp_ms: int
    equity: float
    balance: float
    available_balance: float
    margin_used: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    drawdown: float = 0.0
    drawdown_pct: float = 0.0
    open_positions: int = 0
    source: str = "position_simulator"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def event_time(self) -> datetime:
        return datetime_from_ms(self.timestamp_ms)


@dataclass(slots=True)
class BacktestPositionRecord(SerializableMixin):
    """
    Audit record for position simulator events.
    """

    record_id: str = field(default_factory=lambda: new_id("pos_rec"))
    run_id: str | None = None
    timestamp_ms: int = 0
    topic: str = ""
    position_id: str | None = None
    signal_id: str | None = None
    strategy_name: str | None = None
    symbol: str | None = None
    status: SimulatedPositionStatus | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Signal / risk audit models
# ============================================================================


@dataclass(slots=True)
class BacktestSignalRecord(SerializableMixin):
    """
    Signal lifecycle record in the backtest.
    """

    record_id: str = field(default_factory=lambda: new_id("sig_rec"))
    run_id: str | None = None
    signal_id: str | None = None
    strategy_name: str | None = None
    symbol: str | None = None
    timeframe: str | None = None
    side: str | None = None
    setup_type: str | None = None
    confidence: float | None = None
    strength: float | None = None
    generated_at_ms: int | None = None
    confirmed_at_ms: int | None = None
    rejected_at_ms: int | None = None
    opened_at_ms: int | None = None
    closed_at_ms: int | None = None
    outcome: SignalOutcome = SignalOutcome.GENERATED
    pnl: float = 0.0
    r_multiple: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BacktestRiskDecisionRecord(SerializableMixin):
    """
    Risk decision audit record.
    """

    record_id: str = field(default_factory=lambda: new_id("risk_rec"))
    run_id: str | None = None
    signal_id: str | None = None
    strategy_name: str | None = None
    symbol: str | None = None
    timestamp_ms: int = 0
    approved: bool = False
    blocked: bool = False
    reason: str | None = None
    risk_amount: float | None = None
    final_size: float | None = None
    final_leverage: float | None = None
    final_margin: float | None = None
    final_notional: float | None = None
    reservation_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Cost / model diagnostics
# ============================================================================


@dataclass(slots=True)
class TradingCostBreakdown(SerializableMixin):
    """
    Detailed cost breakdown for an order, fill, trade or whole run.
    """

    commission: float = 0.0
    slippage: float = 0.0
    spread_cost: float = 0.0
    funding_paid: float = 0.0
    funding_received: float = 0.0
    borrow_cost: float = 0.0
    liquidation_penalty: float = 0.0
    other_costs: float = 0.0

    @property
    def net_funding(self) -> float:
        return self.funding_received - self.funding_paid

    @property
    def total_cost(self) -> float:
        return (
            self.commission
            + self.slippage
            + self.spread_cost
            + self.funding_paid
            - self.funding_received
            + self.borrow_cost
            + self.liquidation_penalty
            + self.other_costs
        )


@dataclass(slots=True)
class SimulationModelSnapshot(SerializableMixin):
    """
    Snapshot of simulation models used in a run.
    """

    fill_model: FillModel = FillModel.NEXT_CANDLE_OPEN
    candle_execution_path: CandleExecutionPath = CandleExecutionPath.CONSERVATIVE
    slippage_model: SlippageModel = SlippageModel.FIXED_BPS
    commission_model: CommissionModel = CommissionModel.MAKER_TAKER
    liquidity_model: LiquidityModel = LiquidityModel.CANDLE_VOLUME_PERCENT
    latency_model: LatencyModel = LatencyModel.NONE
    funding_mode: FundingSimulationMode = FundingSimulationMode.APPLY_ON_FUNDING_TIMESTAMP
    position_accounting_mode: PositionAccountingMode = PositionAccountingMode.NETTING
    pnl_accounting_mode: PnLAccountingMode = PnLAccountingMode.REALIZED_AND_UNREALIZED
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Performance models
# ============================================================================


@dataclass(slots=True)
class DrawdownPeriod(SerializableMixin):
    """
    One drawdown period.
    """

    start_ms: int
    valley_ms: int
    recovery_ms: int | None = None
    peak_equity: float = 0.0
    valley_equity: float = 0.0
    drawdown: float = 0.0
    drawdown_pct: float = 0.0

    @property
    def is_recovered(self) -> bool:
        return self.recovery_ms is not None

    @property
    def duration_seconds(self) -> float:
        end = self.recovery_ms or self.valley_ms
        return max(0.0, (end - self.start_ms) / 1000.0)


@dataclass(slots=True)
class TradeStats(SerializableMixin):
    """
    Aggregated trade statistics.
    """

    total_trades: int = 0
    open_trades: int = 0
    closed_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    win_rate: float = 0.0
    loss_rate: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    average_trade: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    median_trade: float = 0.0
    expectancy: float = 0.0
    expectancy_r: float | None = None
    profit_factor: float = 0.0
    payoff_ratio: float = 0.0
    average_holding_time_seconds: float = 0.0


@dataclass(slots=True)
class RiskStats(SerializableMixin):
    """
    Aggregated risk pipeline statistics.
    """

    signals_received: int = 0
    signals_confirmed: int = 0
    signals_blocked: int = 0
    confirmation_rate: float = 0.0
    block_rate: float = 0.0
    position_blocked_events: int = 0
    kill_switch_events: int = 0
    limit_warnings: int = 0
    max_margin_used: float = 0.0
    max_exposure: float = 0.0
    max_leverage_used: float = 0.0
    reservations_created: int = 0
    reservations_released: int = 0
    reservations_expired: int = 0


@dataclass(slots=True)
class ExecutionStatsSnapshot(SerializableMixin):
    """
    Aggregated simulated execution statistics.
    """

    orders_submitted: int = 0
    orders_accepted: int = 0
    orders_rejected: int = 0
    orders_cancelled: int = 0
    orders_filled: int = 0
    orders_partially_filled: int = 0
    fills: int = 0
    rejection_rate: float = 0.0
    fill_rate: float = 0.0
    partial_fill_rate: float = 0.0
    average_slippage: float = 0.0
    average_slippage_bps: float = 0.0
    average_latency_ms: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0


@dataclass(slots=True)
class PerformanceSummary(SerializableMixin):
    """
    Main performance summary for system, strategy, symbol or window.
    """

    aggregation: MetricAggregation = MetricAggregation.SYSTEM
    key: str = "system"
    initial_balance: float = 0.0
    final_balance: float = 0.0
    final_equity: float = 0.0
    net_profit: float = 0.0
    net_profit_pct: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    average_drawdown: float = 0.0
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    calmar_ratio: float | None = None
    recovery_factor: float | None = None
    exposure_time_pct: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_funding: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyBacktestResult(SerializableMixin):
    """
    Performance result for a single strategy.
    """

    strategy_name: str
    summary: PerformanceSummary = field(default_factory=PerformanceSummary)
    trade_stats: TradeStats = field(default_factory=TradeStats)
    risk_stats: RiskStats = field(default_factory=RiskStats)
    execution_stats: ExecutionStatsSnapshot = field(default_factory=ExecutionStatsSnapshot)
    signals: list[BacktestSignalRecord] = field(default_factory=list)
    trades: list[SimulatedTrade] = field(default_factory=list)
    positions: list[SimulatedPosition] = field(default_factory=list)
    equity_curve: list[SimulatedEquityPoint] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SymbolBacktestResult(SerializableMixin):
    """
    Performance result for one symbol.
    """

    symbol: str
    exchange: str = "binance"
    market_type: str = "usdm_futures"
    summary: PerformanceSummary = field(default_factory=PerformanceSummary)
    trade_stats: TradeStats = field(default_factory=TradeStats)
    trades: list[SimulatedTrade] = field(default_factory=list)
    positions: list[SimulatedPosition] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PortfolioBacktestResult(SerializableMixin):
    """
    Portfolio-level backtest result.
    """

    summary: PerformanceSummary = field(default_factory=PerformanceSummary)
    trade_stats: TradeStats = field(default_factory=TradeStats)
    risk_stats: RiskStats = field(default_factory=RiskStats)
    execution_stats: ExecutionStatsSnapshot = field(default_factory=ExecutionStatsSnapshot)
    strategy_results: dict[str, StrategyBacktestResult] = field(default_factory=dict)
    symbol_results: dict[str, SymbolBacktestResult] = field(default_factory=dict)
    equity_curve: list[SimulatedEquityPoint] = field(default_factory=list)
    drawdowns: list[DrawdownPeriod] = field(default_factory=list)
    costs: TradingCostBreakdown = field(default_factory=TradingCostBreakdown)
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Analytics / attribution models
# ============================================================================


@dataclass(slots=True)
class SignalQualityStats(SerializableMixin):
    """
    Signal quality and conversion statistics.
    """

    signals_generated: int = 0
    signals_confirmed: int = 0
    signals_blocked_by_risk: int = 0
    signals_executed: int = 0
    signals_profitable: int = 0
    signals_unprofitable: int = 0
    confirmation_rate: float = 0.0
    execution_rate: float = 0.0
    profitable_signal_rate: float = 0.0
    average_signal_pnl: float = 0.0
    average_signal_r: float | None = None


@dataclass(slots=True)
class StrategyAttribution(SerializableMixin):
    """
    Attribution record for strategy-level contribution.
    """

    strategy_name: str
    net_profit: float = 0.0
    profit_share_pct: float = 0.0
    drawdown_contribution: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    signals: int = 0
    blocked_signals: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RegimePerformanceStats(SerializableMixin):
    """
    Performance grouped by market regime.
    """

    regime: str
    trades: int = 0
    net_profit: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    average_r: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BacktestModelAnalytics(SerializableMixin):
    """
    Higher-level diagnostics from model_analytics.py.
    """

    signal_quality: SignalQualityStats = field(default_factory=SignalQualityStats)
    strategy_attribution: list[StrategyAttribution] = field(default_factory=list)
    regime_performance: list[RegimePerformanceStats] = field(default_factory=list)
    feature_stats: dict[str, Any] = field(default_factory=dict)
    risk_decision_stats: dict[str, Any] = field(default_factory=dict)
    execution_quality_stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ============================================================================
# Report / artifact / warning models
# ============================================================================


@dataclass(slots=True)
class BacktestWarning(SerializableMixin):
    """
    Warning generated during backtesting, analytics or reporting.
    """

    message: str
    level: BacktestWarningLevel = BacktestWarningLevel.WARNING
    code: str | None = None
    timestamp: datetime = field(default_factory=utcnow)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BacktestArtifact(SerializableMixin):
    """
    Output artifact generated by report_builder or strategy_tester.
    """

    artifact_type: BacktestArtifactType
    path: str
    format: ReportFormat | None = None
    created_at: datetime = field(default_factory=utcnow)
    size_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BacktestReport(SerializableMixin):
    """
    Report descriptor.
    """

    report_id: str = field(default_factory=lambda: new_id("report"))
    run_id: str | None = None
    title: str = "Backtest Report"
    format: ReportFormat = ReportFormat.MARKDOWN
    path: str | None = None
    summary: PerformanceSummary | None = None
    artifacts: list[BacktestArtifact] = field(default_factory=list)
    created_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Walk-forward / optimization models
# ============================================================================


@dataclass(slots=True, frozen=True)
class WalkForwardWindow(SerializableMixin):
    """
    One walk-forward train/validation/test window.
    """

    window_id: str
    window_type: WalkForwardWindowType
    period: BacktestPeriod
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WalkForwardIterationResult(SerializableMixin):
    """
    Result of one walk-forward iteration.
    """

    iteration: int
    train_window: WalkForwardWindow | None = None
    validation_window: WalkForwardWindow | None = None
    test_window: WalkForwardWindow | None = None
    train_result: BacktestResult | None = None
    validation_result: BacktestResult | None = None
    test_result: BacktestResult | None = None
    selected_parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WalkForwardResult(SerializableMixin):
    """
    Aggregated walk-forward result.
    """

    run_id: str = field(default_factory=lambda: new_id("wf"))
    mode: WalkForwardMode = WalkForwardMode.ROLLING
    iterations: list[WalkForwardIterationResult] = field(default_factory=list)
    aggregated_summary: PerformanceSummary = field(default_factory=PerformanceSummary)
    stability_score: float | None = None
    overfitting_score: float | None = None
    warnings: list[BacktestWarning] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OptimizationParameter(SerializableMixin):
    """
    Parameter definition for optimizer.
    """

    name: str
    values: list[Any] | None = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    distribution: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OptimizationTrialResult(SerializableMixin):
    """
    One optimizer trial result.
    """

    trial_id: str = field(default_factory=lambda: new_id("trial"))
    index: int = 0
    parameters: dict[str, Any] = field(default_factory=dict)
    objective_metric: OptimizationMetric = OptimizationMetric.NET_PROFIT
    objective_value: float = 0.0
    direction: OptimizationDirection = OptimizationDirection.MAXIMIZE
    backtest_result: BacktestResult | None = None
    status: BacktestStatus = BacktestStatus.CREATED
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OptimizationResult(SerializableMixin):
    """
    Aggregated optimization result.
    """

    optimization_id: str = field(default_factory=lambda: new_id("opt"))
    trials: list[OptimizationTrialResult] = field(default_factory=list)
    best_trial: OptimizationTrialResult | None = None
    objective_metric: OptimizationMetric = OptimizationMetric.NET_PROFIT
    direction: OptimizationDirection = OptimizationDirection.MAXIMIZE
    overfitting_score: float | None = None
    parameter_importance: dict[str, float] = field(default_factory=dict)
    warnings: list[BacktestWarning] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def rank_trials(self) -> list[OptimizationTrialResult]:
        reverse = self.direction == OptimizationDirection.MAXIMIZE
        return sorted(
            self.trials,
            key=lambda trial: trial.objective_value,
            reverse=reverse,
        )


# ============================================================================
# Final backtest result
# ============================================================================


@dataclass(slots=True)
class BacktestResult(SerializableMixin):
    """
    Final result of a complete backtest run.
    """

    run_id: str = field(default_factory=lambda: str(BacktestRunId()))
    run_name: str = "backtest"
    mode: BacktestMode = BacktestMode.MULTI_STRATEGY
    status: BacktestStatus = BacktestStatus.CREATED
    period: BacktestPeriod | None = None
    dataset_info: BacktestDatasetInfo | None = None
    simulation_models: SimulationModelSnapshot = field(default_factory=SimulationModelSnapshot)

    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float = 0.0

    initial_balance: float = 0.0
    final_balance: float = 0.0
    final_equity: float = 0.0

    portfolio: PortfolioBacktestResult = field(default_factory=PortfolioBacktestResult)
    analytics: BacktestModelAnalytics = field(default_factory=BacktestModelAnalytics)

    signals: list[BacktestSignalRecord] = field(default_factory=list)
    risk_decisions: list[BacktestRiskDecisionRecord] = field(default_factory=list)
    execution_records: list[BacktestExecutionRecord] = field(default_factory=list)
    position_records: list[BacktestPositionRecord] = field(default_factory=list)

    orders: list[SimulatedOrder] = field(default_factory=list)
    fills: list[SimulatedFill] = field(default_factory=list)
    positions: list[SimulatedPosition] = field(default_factory=list)
    trades: list[SimulatedTrade] = field(default_factory=list)
    equity_curve: list[SimulatedEquityPoint] = field(default_factory=list)

    reports: list[BacktestReport] = field(default_factory=list)
    artifacts: list[BacktestArtifact] = field(default_factory=list)
    warnings: list[BacktestWarning] = field(default_factory=list)

    error: str | None = None
    error_details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def completed_successfully(self) -> bool:
        return self.status == BacktestStatus.COMPLETED and self.error is None

    @property
    def net_profit(self) -> float:
        return self.final_equity - self.initial_balance

    @property
    def net_profit_pct(self) -> float:
        return safe_div(self.net_profit, self.initial_balance) * 100.0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def open_positions(self) -> list[SimulatedPosition]:
        return [position for position in self.positions if position.is_open]

    def mark_started(self) -> None:
        self.started_at = utcnow()
        self.status = BacktestStatus.RUNNING

    def mark_completed(self) -> None:
        self.finished_at = utcnow()
        self.status = BacktestStatus.COMPLETED

        if self.started_at is not None:
            self.duration_seconds = (
                ensure_aware_utc(self.finished_at) - ensure_aware_utc(self.started_at)
            ).total_seconds()

    def mark_failed(
        self,
        error: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.finished_at = utcnow()
        self.status = BacktestStatus.FAILED
        self.error = error
        self.error_details = details or {}

        if self.started_at is not None:
            self.duration_seconds = (
                ensure_aware_utc(self.finished_at) - ensure_aware_utc(self.started_at)
            ).total_seconds()

    def add_warning(
        self,
        message: str,
        *,
        level: BacktestWarningLevel = BacktestWarningLevel.WARNING,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.warnings.append(
            BacktestWarning(
                message=message,
                level=level,
                code=code,
                details=details or {},
            )
        )

    def compact_summary(self) -> dict[str, Any]:
        """
        Return compact user-facing result summary.
        """

        summary = self.portfolio.summary

        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "mode": self.mode.value,
            "status": self.status.value,
            "initial_balance": self.initial_balance,
            "final_balance": self.final_balance,
            "final_equity": self.final_equity,
            "net_profit": self.net_profit,
            "net_profit_pct": self.net_profit_pct,
            "total_trades": self.total_trades,
            "win_rate": summary.win_rate,
            "profit_factor": summary.profit_factor,
            "max_drawdown": summary.max_drawdown,
            "max_drawdown_pct": summary.max_drawdown_pct,
            "sharpe_ratio": summary.sharpe_ratio,
            "sortino_ratio": summary.sortino_ratio,
            "total_fees": summary.total_fees,
            "total_slippage": summary.total_slippage,
            "total_funding": summary.total_funding,
            "signals_generated": len(self.signals),
            "signals_confirmed": self.portfolio.risk_stats.signals_confirmed,
            "signals_blocked": self.portfolio.risk_stats.signals_blocked,
            "orders": len(self.orders),
            "fills": len(self.fills),
            "positions": len(self.positions),
            "open_positions": len(self.open_positions),
            "warnings": len(self.warnings),
            "duration_seconds": self.duration_seconds,
        }


__all__ = [
    # Utility
    "Number",
    "SerializableMixin",
    "utcnow",
    "ensure_aware_utc",
    "timestamp_ms",
    "datetime_from_ms",
    "new_id",
    "safe_div",
    "clamp",
    # Identity / metadata
    "BacktestRunId",
    "BacktestPeriod",
    "BacktestInstrument",
    "BacktestDataSource",
    "BacktestDatasetInfo",
    # Historical records
    "HistoricalMarketRecord",
    "HistoricalCandle",
    "HistoricalTrade",
    "HistoricalOrderBookLevel",
    "HistoricalOrderBookSnapshot",
    "HistoricalFundingRecord",
    "HistoricalOpenInterestRecord",
    "HistoricalLiquidationRecord",
    # Replay
    "BacktestEvent",
    "ReplayEventBatch",
    "BacktestDataset",
    "BacktestClockState",
    # Execution simulation
    "SimulatedOrder",
    "SimulatedFill",
    "BacktestExecutionRecord",
    # Position simulation
    "SimulatedPosition",
    "SimulatedTrade",
    "SimulatedBalance",
    "SimulatedEquityPoint",
    "BacktestPositionRecord",
    # Signal / risk
    "BacktestSignalRecord",
    "BacktestRiskDecisionRecord",
    # Costs / model snapshots
    "TradingCostBreakdown",
    "SimulationModelSnapshot",
    # Performance
    "DrawdownPeriod",
    "TradeStats",
    "RiskStats",
    "ExecutionStatsSnapshot",
    "PerformanceSummary",
    "StrategyBacktestResult",
    "SymbolBacktestResult",
    "PortfolioBacktestResult",
    # Analytics
    "SignalQualityStats",
    "StrategyAttribution",
    "RegimePerformanceStats",
    "BacktestModelAnalytics",
    # Reports
    "BacktestWarning",
    "BacktestArtifact",
    "BacktestReport",
    # Walk-forward / optimization
    "WalkForwardWindow",
    "WalkForwardIterationResult",
    "WalkForwardResult",
    "OptimizationParameter",
    "OptimizationTrialResult",
    "OptimizationResult",
    # Final result
    "BacktestResult",
]