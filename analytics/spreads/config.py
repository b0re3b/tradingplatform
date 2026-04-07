from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from .enums import InstrumentType


# =========================
# Base Config
# =========================

@dataclass(slots=True)
class BaseSpreadConfig:
    enabled: bool = True

    max_quote_age_ms: int = 2_000
    max_quote_skew_ms: int = 1_000

    rolling_window_size: int = 500
    ema_alpha: Decimal = Decimal("0.2")

    min_emit_interval_ms: int = 250
    cooldown_seconds: int = 10

    anomaly_zscore_threshold: Decimal = Decimal("2.5")
    widening_bps_threshold: Decimal = Decimal("8")

    metadata: dict[str, Any] = field(default_factory=dict)


# =========================
# Spot-Futures Config
# =========================

@dataclass(slots=True)
class SpotFuturesSpreadConfig(BaseSpreadConfig):
    mean_reversion_zscore_threshold: Decimal = Decimal("2.0")
    regime_shift_zscore_threshold: Decimal = Decimal("3.0")

    notional_for_funding_adjustment: Decimal | None = None

    default_spot_exchange: str | None = None
    default_futures_exchange: str | None = None


# =========================
# Cross-Exchange Config
# =========================

@dataclass(slots=True)
class CrossExchangeSpreadConfig(BaseSpreadConfig):
    arbitrage_min_bps: Decimal = Decimal("5")

    default_trade_size: Decimal = Decimal("1")

    slippage_max_bps: Decimal = Decimal("5")
    safety_buffer_bps: Decimal = Decimal("1")

    default_taker_fee_rate: Decimal = Decimal("0.001")
    default_maker_fee_rate: Decimal = Decimal("0.0005")

    allowed_instrument_types: set[InstrumentType] = field(
        default_factory=lambda: {
            InstrumentType.SPOT,
            InstrumentType.PERPETUAL,
            InstrumentType.FUTURES,
        }
    )

    preferred_exchanges: set[str] = field(default_factory=set)