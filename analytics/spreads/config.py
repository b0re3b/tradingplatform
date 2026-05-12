from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any, Mapping

from .enums import InstrumentType


# ============================================================
# Constants
# ============================================================

DEFAULT_QUOTE_EVENT_TOPIC = "market.quote.updated"
DEFAULT_FUNDING_EVENT_TOPIC = "market.funding.updated"

DEFAULT_SPOT_FUTURES_SNAPSHOT_TOPIC = "analytics.spreads.spot_futures.updated"
DEFAULT_CROSS_EXCHANGE_SNAPSHOT_TOPIC = "analytics.spreads.cross_exchange.updated"
DEFAULT_SPREAD_SIGNAL_TOPIC = "analytics.spreads.signal.generated"
DEFAULT_ARBITRAGE_OPPORTUNITY_TOPIC = "analytics.spreads.arbitrage.opportunity"

DEFAULT_ANALYZER_STARTED_TOPIC = "analytics.spreads.analyzer.started"
DEFAULT_ANALYZER_STOPPED_TOPIC = "analytics.spreads.analyzer.stopped"
DEFAULT_ANALYZER_HEARTBEAT_TOPIC = "analytics.spreads.analyzer.heartbeat"

DECIMAL_ZERO = Decimal("0")


# ============================================================
# Helpers
# ============================================================

def _normalize_exchange_set(values: set[str] | list[str] | tuple[str, ...] | None) -> set[str]:
    if not values:
        return set()
    return {value.strip().lower() for value in values if value and value.strip()}


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _validate_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _validate_positive_decimal(name: str, value: Decimal) -> None:
    if value <= DECIMAL_ZERO:
        raise ValueError(f"{name} must be > 0")


def _validate_non_negative_decimal(name: str, value: Decimal) -> None:
    if value < DECIMAL_ZERO:
        raise ValueError(f"{name} must be >= 0")


def _validate_positive_float(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


# ============================================================
# Base Config
# ============================================================

@dataclass(slots=True)
class BaseSpreadConfig:
    """
    Базовий config-контракт для analytics/spreads.

    Відповідальність:
    - runtime enable/disable;
    - topic names для EventBus;
    - quote freshness / alignment rules;
    - rolling statistics parameters;
    - throttling / cooldown;
    - scheduler intervals;
    - cache cleanup limits;
    - signal thresholds;
    - metadata для розширення без зміни контракту.

    Не відповідає за:
    - створення EventBus;
    - створення Scheduler;
    - читання .env напряму;
    - logging;
    - бізнес-логіку analyzer-ів.
    """

    # Runtime
    enabled: bool = True
    service_name: str = "spread_analyzer"

    # Input EventBus topics
    quote_event_topic: str = DEFAULT_QUOTE_EVENT_TOPIC
    funding_event_topic: str = DEFAULT_FUNDING_EVENT_TOPIC

    # Common output EventBus topics
    signal_event_topic: str = DEFAULT_SPREAD_SIGNAL_TOPIC
    analyzer_started_event_topic: str = DEFAULT_ANALYZER_STARTED_TOPIC
    analyzer_stopped_event_topic: str = DEFAULT_ANALYZER_STOPPED_TOPIC
    analyzer_heartbeat_event_topic: str = DEFAULT_ANALYZER_HEARTBEAT_TOPIC

    # Quote freshness / alignment
    max_quote_age_ms: int = 2_000
    max_quote_skew_ms: int = 1_000

    # Rolling stats
    rolling_window_size: int = 500
    ema_alpha: Decimal = Decimal("0.2")

    # Emit throttling / signal cooldown
    min_emit_interval_ms: int = 250
    cooldown_seconds: int = 10

    # Scheduler / maintenance
    cleanup_interval_seconds: float = 30.0
    heartbeat_interval_seconds: float = 60.0
    stale_state_ttl_seconds: float = 300.0

    # Cache safety limits
    max_cached_quotes: int = 50_000
    max_cached_snapshots: int = 25_000
    max_cached_windows: int = 25_000

    # Signal thresholds
    anomaly_zscore_threshold: Decimal = Decimal("2.5")
    widening_bps_threshold: Decimal = Decimal("8")

    # Extensibility
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _validate_positive_int("max_quote_age_ms", self.max_quote_age_ms)
        _validate_positive_int("max_quote_skew_ms", self.max_quote_skew_ms)
        _validate_positive_int("rolling_window_size", self.rolling_window_size)
        _validate_non_negative_int("min_emit_interval_ms", self.min_emit_interval_ms)
        _validate_non_negative_int("cooldown_seconds", self.cooldown_seconds)

        _validate_positive_float("cleanup_interval_seconds", self.cleanup_interval_seconds)
        _validate_positive_float("heartbeat_interval_seconds", self.heartbeat_interval_seconds)
        _validate_positive_float("stale_state_ttl_seconds", self.stale_state_ttl_seconds)

        _validate_positive_int("max_cached_quotes", self.max_cached_quotes)
        _validate_positive_int("max_cached_snapshots", self.max_cached_snapshots)
        _validate_positive_int("max_cached_windows", self.max_cached_windows)

        _validate_positive_decimal("ema_alpha", self.ema_alpha)
        if self.ema_alpha > Decimal("1"):
            raise ValueError("ema_alpha must be <= 1")

        _validate_positive_decimal("anomaly_zscore_threshold", self.anomaly_zscore_threshold)
        _validate_positive_decimal("widening_bps_threshold", self.widening_bps_threshold)

        self._validate_topic("quote_event_topic", self.quote_event_topic)
        self._validate_topic("funding_event_topic", self.funding_event_topic)
        self._validate_topic("signal_event_topic", self.signal_event_topic)
        self._validate_topic("analyzer_started_event_topic", self.analyzer_started_event_topic)
        self._validate_topic("analyzer_stopped_event_topic", self.analyzer_stopped_event_topic)
        self._validate_topic("analyzer_heartbeat_event_topic", self.analyzer_heartbeat_event_topic)

    @staticmethod
    def _validate_topic(field_name: str, value: str) -> None:
        if not value or not value.strip():
            raise ValueError(f"{field_name} must not be empty")

    def with_metadata(self, **metadata: Any) -> BaseSpreadConfig:
        """
        Повертає копію config з оновленим metadata.

        Корисно, коли bootstrap/container хоче додати runtime context,
        не мутуючи оригінальний config.
        """
        return replace(
            self,
            metadata={
                **self.metadata,
                **metadata,
            },
        )


# ============================================================
# Spot-Futures Config
# ============================================================

@dataclass(slots=True)
class SpotFuturesSpreadConfig(BaseSpreadConfig):
    """
    Config для SpotFuturesSpreadAnalyzer.
    """

    service_name: str = "spot_futures_spread_analyzer"

    # Output topics
    snapshot_event_topic: str = DEFAULT_SPOT_FUTURES_SNAPSHOT_TOPIC

    # Strategy/signal thresholds
    mean_reversion_zscore_threshold: Decimal = Decimal("2.0")
    regime_shift_zscore_threshold: Decimal = Decimal("3.0")

    # Funding adjustment
    notional_for_funding_adjustment: Decimal | None = None

    # Optional filters/default routing
    default_spot_exchange: str | None = None
    default_futures_exchange: str | None = None

    allowed_spot_exchanges: set[str] = field(default_factory=set)
    allowed_futures_exchanges: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.allowed_spot_exchanges = _normalize_exchange_set(
            self.allowed_spot_exchanges
        )
        self.allowed_futures_exchanges = _normalize_exchange_set(
            self.allowed_futures_exchanges
        )

        if self.default_spot_exchange:
            self.default_spot_exchange = self.default_spot_exchange.strip().lower()

        if self.default_futures_exchange:
            self.default_futures_exchange = self.default_futures_exchange.strip().lower()

        # Не використовуємо super().__post_init__() у dataclass(slots=True)
        # inheritance, бо zero-argument super() може падати з:
        # TypeError: super(type, obj): obj must be an instance or subtype of type
        BaseSpreadConfig.__post_init__(self)

        self._validate_topic("snapshot_event_topic", self.snapshot_event_topic)

        _validate_positive_decimal(
            "mean_reversion_zscore_threshold",
            self.mean_reversion_zscore_threshold,
        )
        _validate_positive_decimal(
            "regime_shift_zscore_threshold",
            self.regime_shift_zscore_threshold,
        )

        if self.notional_for_funding_adjustment is not None:
            _validate_positive_decimal(
                "notional_for_funding_adjustment",
                self.notional_for_funding_adjustment,
            )

    def is_spot_exchange_allowed(self, exchange: str) -> bool:
        normalized = exchange.strip().lower()
        return not self.allowed_spot_exchanges or normalized in self.allowed_spot_exchanges

    def is_futures_exchange_allowed(self, exchange: str) -> bool:
        normalized = exchange.strip().lower()
        return (
            not self.allowed_futures_exchanges
            or normalized in self.allowed_futures_exchanges
        )


# ============================================================
# Cross-Exchange Config
# ============================================================

@dataclass(slots=True)
class CrossExchangeSpreadConfig(BaseSpreadConfig):
    """
    Config для CrossExchangeSpreadAnalyzer.
    """

    service_name: str = "cross_exchange_spread_analyzer"

    # Output topics
    snapshot_event_topic: str = DEFAULT_CROSS_EXCHANGE_SNAPSHOT_TOPIC
    opportunity_event_topic: str = DEFAULT_ARBITRAGE_OPPORTUNITY_TOPIC

    # Arbitrage threshold
    arbitrage_min_bps: Decimal = Decimal("5")

    # Trade sizing
    default_trade_size: Decimal = Decimal("1")
    min_trade_size: Decimal | None = None
    max_trade_size: Decimal | None = None

    # Cost model
    slippage_max_bps: Decimal = Decimal("5")
    safety_buffer_bps: Decimal = Decimal("1")
    default_taker_fee_rate: Decimal = Decimal("0.001")
    default_maker_fee_rate: Decimal = Decimal("0.0005")

    # Opportunity lifecycle
    opportunity_ttl_seconds: float = 10.0
    max_cached_opportunities: int = 10_000

    # Filters
    allowed_instrument_types: set[InstrumentType] = field(
        default_factory=lambda: {
            InstrumentType.SPOT,
            InstrumentType.PERPETUAL,
            InstrumentType.FUTURES,
        }
    )
    preferred_exchanges: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.preferred_exchanges = _normalize_exchange_set(self.preferred_exchanges)

        if not self.allowed_instrument_types:
            raise ValueError("allowed_instrument_types must not be empty")

        if InstrumentType.UNKNOWN in self.allowed_instrument_types:
            raise ValueError(
                "allowed_instrument_types must not include InstrumentType.UNKNOWN"
            )

        # Не використовуємо super().__post_init__() у dataclass(slots=True)
        # inheritance, бо zero-argument super() може падати з:
        # TypeError: super(type, obj): obj must be an instance or subtype of type
        BaseSpreadConfig.__post_init__(self)

        self._validate_topic("snapshot_event_topic", self.snapshot_event_topic)
        self._validate_topic("opportunity_event_topic", self.opportunity_event_topic)

        _validate_positive_decimal("arbitrage_min_bps", self.arbitrage_min_bps)
        _validate_positive_decimal("default_trade_size", self.default_trade_size)

        _validate_non_negative_decimal("slippage_max_bps", self.slippage_max_bps)
        _validate_non_negative_decimal("safety_buffer_bps", self.safety_buffer_bps)
        _validate_non_negative_decimal("default_taker_fee_rate", self.default_taker_fee_rate)
        _validate_non_negative_decimal("default_maker_fee_rate", self.default_maker_fee_rate)

        _validate_positive_float("opportunity_ttl_seconds", self.opportunity_ttl_seconds)
        _validate_positive_int("max_cached_opportunities", self.max_cached_opportunities)

        if self.min_trade_size is not None:
            _validate_positive_decimal("min_trade_size", self.min_trade_size)

        if self.max_trade_size is not None:
            _validate_positive_decimal("max_trade_size", self.max_trade_size)

        if (
            self.min_trade_size is not None
            and self.max_trade_size is not None
            and self.min_trade_size > self.max_trade_size
        ):
            raise ValueError("min_trade_size must be <= max_trade_size")

        if self.default_trade_size is not None:
            if (
                self.min_trade_size is not None
                and self.default_trade_size < self.min_trade_size
            ):
                raise ValueError("default_trade_size must be >= min_trade_size")

            if (
                self.max_trade_size is not None
                and self.default_trade_size > self.max_trade_size
            ):
                raise ValueError("default_trade_size must be <= max_trade_size")

    def is_exchange_preferred(self, exchange: str) -> bool:
        normalized = exchange.strip().lower()
        return not self.preferred_exchanges or normalized in self.preferred_exchanges

    def is_instrument_type_allowed(self, instrument_type: InstrumentType) -> bool:
        return instrument_type in self.allowed_instrument_types

    def fee_rates_from_metadata(self) -> Mapping[str, Any]:
        """
        Повертає fee override map з metadata.

        Очікуваний формат:
            metadata = {
                "fee_rates": {
                    "binance": {"buy": "0.001", "sell": "0.001"},
                    "bybit": {"buy": "0.0008", "sell": "0.0008"},
                }
            }
        """
        value = self.metadata.get("fee_rates", {})
        if isinstance(value, Mapping):
            return value
        return {}