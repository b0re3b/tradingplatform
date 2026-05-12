from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping

from .config import CrossExchangeSpreadConfig
from .models import QuoteSnapshot
from .spread_utils import (
    DECIMAL_10_000,
    DECIMAL_ONE,
    DECIMAL_ZERO,
    normalize_exchange,
    safe_div,
    to_decimal,
)


# ============================================================
# Enums
# ============================================================

class CostSide(str, Enum):
    """
    Side для cost calculation.

    BUY:
        Вхід через ask-side.
    SELL:
        Вихід через bid-side.
    """

    BUY = "buy"
    SELL = "sell"

    @classmethod
    def from_value(cls, value: str | CostSide) -> CostSide:
        if isinstance(value, cls):
            return value

        normalized = value.strip().lower()
        if normalized == cls.BUY.value:
            return cls.BUY
        if normalized == cls.SELL.value:
            return cls.SELL

        raise ValueError(f"Unsupported cost side: {value!r}")


class LiquiditySide(str, Enum):
    """
    Side стакана, який використовується для slippage estimation.
    """

    ASK = "ask"
    BID = "bid"
    AUTO = "auto"


# ============================================================
# Models
# ============================================================

@dataclass(slots=True)
class SpreadCostBreakdown:
    """
    Canonical cost breakdown для cross-exchange arbitrage.

    gross_edge:
        Теоретичний edge до витрат.

    estimated_fees:
        Комісії обох legs.

    estimated_slippage:
        Орієнтовний slippage обох legs.

    safety_buffer:
        Додатковий buffer проти latency / stale quote / execution drift.

    net_edge:
        gross_edge - estimated_fees - estimated_slippage - safety_buffer.
    """

    gross_edge: Decimal = DECIMAL_ZERO
    estimated_fees: Decimal = DECIMAL_ZERO
    estimated_slippage: Decimal = DECIMAL_ZERO
    safety_buffer: Decimal = DECIMAL_ZERO
    net_edge: Decimal = DECIMAL_ZERO

    def __post_init__(self) -> None:
        self.gross_edge = _decimal_or_zero(self.gross_edge)
        self.estimated_fees = _non_negative_decimal(self.estimated_fees)
        self.estimated_slippage = _non_negative_decimal(self.estimated_slippage)
        self.safety_buffer = _non_negative_decimal(self.safety_buffer)
        self.net_edge = _decimal_or_zero(self.net_edge)

    @property
    def total_costs(self) -> Decimal:
        return self.estimated_fees + self.estimated_slippage + self.safety_buffer

    @property
    def is_profitable(self) -> bool:
        return self.net_edge > DECIMAL_ZERO

    @property
    def cost_to_gross_ratio(self) -> Decimal:
        ratio = safe_div(self.total_costs, self.gross_edge, default=DECIMAL_ZERO)
        return ratio if ratio is not None else DECIMAL_ZERO

    def edge_bps(self, reference_notional: Decimal | None) -> Decimal:
        return edge_bps_after_costs(
            net_edge=self.net_edge,
            reference_notional=reference_notional,
        )

    def to_payload(self) -> dict[str, str | bool]:
        return {
            "gross_edge": str(self.gross_edge),
            "estimated_fees": str(self.estimated_fees),
            "estimated_slippage": str(self.estimated_slippage),
            "safety_buffer": str(self.safety_buffer),
            "total_costs": str(self.total_costs),
            "net_edge": str(self.net_edge),
            "is_profitable": self.is_profitable,
            "cost_to_gross_ratio": str(self.cost_to_gross_ratio),
        }


# ============================================================
# Internal helpers
# ============================================================

def _decimal_or_zero(value: Decimal | int | float | str | None) -> Decimal:
    resolved = to_decimal(value, default=DECIMAL_ZERO)
    return resolved if resolved is not None else DECIMAL_ZERO


def _non_negative_decimal(value: Decimal | int | float | str | None) -> Decimal:
    resolved = _decimal_or_zero(value)
    if resolved < DECIMAL_ZERO:
        return DECIMAL_ZERO
    return resolved


def _positive_decimal_or_none(value: Decimal | int | float | str | None) -> Decimal | None:
    resolved = to_decimal(value, default=None)
    if resolved is None or resolved <= DECIMAL_ZERO:
        return None
    return resolved


def _normalize_side_name(side: str | CostSide) -> str:
    return CostSide.from_value(side).value


def _fee_override_map(config: CrossExchangeSpreadConfig) -> Mapping[str, Any]:
    """
    Повертає fee override map із config.metadata.

    Очікуваний формат:
        {
            "fee_rates": {
                "binance": {"buy": "0.001", "sell": "0.001"},
                "bybit": {"buy": "0.0008", "sell": "0.0009"},
            }
        }
    """
    fee_rates_from_metadata = getattr(config, "fee_rates_from_metadata", None)
    if callable(fee_rates_from_metadata):
        return fee_rates_from_metadata()

    value = config.metadata.get("fee_rates", {})
    if isinstance(value, Mapping):
        return value
    return {}


def _resolve_trade_size_bounds(
    config: CrossExchangeSpreadConfig,
) -> tuple[Decimal | None, Decimal | None]:
    min_trade_size = getattr(config, "min_trade_size", None)
    max_trade_size = getattr(config, "max_trade_size", None)

    return (
        _positive_decimal_or_none(min_trade_size),
        _positive_decimal_or_none(max_trade_size),
    )


def _quote_reference_price(
    quote: QuoteSnapshot,
    *,
    side: LiquiditySide = LiquiditySide.AUTO,
) -> Decimal | None:
    if side == LiquiditySide.ASK:
        return quote.ask or quote.mid_price or quote.bid

    if side == LiquiditySide.BID:
        return quote.bid or quote.mid_price or quote.ask

    return quote.mid_price or quote.ask or quote.bid


def _quote_top_book_size(
    quote: QuoteSnapshot,
    *,
    side: LiquiditySide = LiquiditySide.AUTO,
) -> Decimal | None:
    if side == LiquiditySide.ASK:
        return quote.ask_size

    if side == LiquiditySide.BID:
        return quote.bid_size

    return quote.ask_size or quote.bid_size


# ============================================================
# Public fee helpers
# ============================================================

def normalize_fee_overrides(
    fee_overrides: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, dict[str, Decimal]]:
    """
    Нормалізує fee-конфіг до вигляду:

        {
            "binance": {"buy": Decimal("0.001"), "sell": Decimal("0.001")},
            "bybit": {"buy": Decimal("0.0008"), "sell": Decimal("0.0009")},
        }
    """
    normalized: dict[str, dict[str, Decimal]] = {}

    if not fee_overrides:
        return normalized

    for exchange, side_map in fee_overrides.items():
        exchange_key = normalize_exchange(exchange)
        normalized[exchange_key] = {}

        for side_name, raw_value in side_map.items():
            try:
                side = _normalize_side_name(side_name)
            except ValueError:
                continue

            value = _non_negative_decimal(raw_value)
            normalized[exchange_key][side] = value

    return normalized


def get_fee_rate(
    exchange: str,
    side: str | CostSide,
    config: CrossExchangeSpreadConfig,
    *,
    use_maker_fee: bool = False,
) -> Decimal:
    """
    Повертає fee rate для exchange/side.

    Пріоритет:
    1. config.metadata["fee_rates"][exchange][side]
    2. config.default_maker_fee_rate, якщо use_maker_fee=True
    3. config.default_taker_fee_rate
    """
    exchange_key = normalize_exchange(exchange)
    side_key = _normalize_side_name(side)

    fee_overrides = normalize_fee_overrides(_fee_override_map(config))
    exchange_rates = fee_overrides.get(exchange_key, {})

    if side_key in exchange_rates:
        return exchange_rates[side_key]

    default_rate = (
        config.default_maker_fee_rate
        if use_maker_fee
        else config.default_taker_fee_rate
    )
    return _non_negative_decimal(default_rate)


def estimate_fee_cost(
    price: Decimal | None,
    quantity: Decimal | None,
    fee_rate: Decimal | None,
) -> Decimal:
    price_value = _positive_decimal_or_none(price)
    quantity_value = _positive_decimal_or_none(quantity)
    fee_rate_value = _non_negative_decimal(fee_rate)

    if price_value is None or quantity_value is None:
        return DECIMAL_ZERO

    return price_value * quantity_value * fee_rate_value


def estimate_total_fees(
    buy_price: Decimal | None,
    sell_price: Decimal | None,
    quantity: Decimal | None,
    buy_exchange: str,
    sell_exchange: str,
    config: CrossExchangeSpreadConfig,
    *,
    use_maker_fee: bool = False,
) -> Decimal:
    buy_fee_rate = get_fee_rate(
        buy_exchange,
        side=CostSide.BUY,
        config=config,
        use_maker_fee=use_maker_fee,
    )
    sell_fee_rate = get_fee_rate(
        sell_exchange,
        side=CostSide.SELL,
        config=config,
        use_maker_fee=use_maker_fee,
    )

    return estimate_fee_cost(
        price=buy_price,
        quantity=quantity,
        fee_rate=buy_fee_rate,
    ) + estimate_fee_cost(
        price=sell_price,
        quantity=quantity,
        fee_rate=sell_fee_rate,
    )


# ============================================================
# Slippage helpers
# ============================================================

def estimate_simple_slippage_ratio(
    quantity: Decimal | None,
    top_book_size: Decimal | None,
    max_slippage_bps: Decimal | None,
) -> Decimal:
    """
    Повертає slippage ratio, а не абсолютну вартість.

    Приклад:
        quantity=5
        top_book_size=10
        max_slippage_bps=5

        participation = 0.5
        slippage = 0.5 * 5 bps = 2.5 bps
        ratio = 2.5 / 10000 = 0.00025
    """
    quantity_value = _positive_decimal_or_none(quantity)
    top_book_value = _positive_decimal_or_none(top_book_size)
    max_slippage_value = _positive_decimal_or_none(max_slippage_bps)

    if quantity_value is None or top_book_value is None or max_slippage_value is None:
        return DECIMAL_ZERO

    participation = safe_div(quantity_value, top_book_value, default=DECIMAL_ZERO)
    if participation is None or participation <= DECIMAL_ZERO:
        return DECIMAL_ZERO

    capped_participation = min(participation, DECIMAL_ONE)
    slippage_bps = capped_participation * max_slippage_value

    return slippage_bps / DECIMAL_10_000


# Backward-compatible alias for previous API name.
def estimate_simple_slippage(
    quantity: Decimal | None,
    top_book_size: Decimal | None,
    max_slippage_bps: Decimal | None,
) -> Decimal:
    return estimate_simple_slippage_ratio(
        quantity=quantity,
        top_book_size=top_book_size,
        max_slippage_bps=max_slippage_bps,
    )


def estimate_slippage_cost_from_quote(
    quote: QuoteSnapshot,
    quantity: Decimal | None,
    max_slippage_bps: Decimal | None,
    use_ask_side: bool | None = None,
) -> Decimal:
    quantity_value = _positive_decimal_or_none(quantity)
    if quantity_value is None:
        return DECIMAL_ZERO

    if use_ask_side is True:
        side = LiquiditySide.ASK
    elif use_ask_side is False:
        side = LiquiditySide.BID
    else:
        side = LiquiditySide.AUTO

    reference_price = _quote_reference_price(quote, side=side)
    reference_price = _positive_decimal_or_none(reference_price)
    if reference_price is None:
        return DECIMAL_ZERO

    top_book_size = _quote_top_book_size(quote, side=side)

    slippage_ratio = estimate_simple_slippage_ratio(
        quantity=quantity_value,
        top_book_size=top_book_size,
        max_slippage_bps=max_slippage_bps,
    )

    return reference_price * quantity_value * slippage_ratio


def estimate_total_slippage(
    buy_quote: QuoteSnapshot,
    sell_quote: QuoteSnapshot,
    quantity: Decimal | None,
    config: CrossExchangeSpreadConfig,
) -> Decimal:
    buy_cost = estimate_slippage_cost_from_quote(
        quote=buy_quote,
        quantity=quantity,
        max_slippage_bps=config.slippage_max_bps,
        use_ask_side=True,
    )

    sell_cost = estimate_slippage_cost_from_quote(
        quote=sell_quote,
        quantity=quantity,
        max_slippage_bps=config.slippage_max_bps,
        use_ask_side=False,
    )

    return buy_cost + sell_cost


# ============================================================
# Edge / buffer helpers
# ============================================================

def estimate_safety_buffer_cost(
    reference_price: Decimal | None,
    quantity: Decimal | None,
    safety_buffer_bps: Decimal | None,
) -> Decimal:
    reference_price_value = _positive_decimal_or_none(reference_price)
    quantity_value = _positive_decimal_or_none(quantity)
    safety_buffer_value = _positive_decimal_or_none(safety_buffer_bps)

    if (
        reference_price_value is None
        or quantity_value is None
        or safety_buffer_value is None
    ):
        return DECIMAL_ZERO

    return reference_price_value * quantity_value * (
        safety_buffer_value / DECIMAL_10_000
    )


def gross_edge_from_prices(
    buy_price: Decimal | None,
    sell_price: Decimal | None,
    quantity: Decimal | None,
) -> Decimal:
    buy_price_value = _positive_decimal_or_none(buy_price)
    sell_price_value = _positive_decimal_or_none(sell_price)
    quantity_value = _positive_decimal_or_none(quantity)

    if (
        buy_price_value is None
        or sell_price_value is None
        or quantity_value is None
    ):
        return DECIMAL_ZERO

    return (sell_price_value - buy_price_value) * quantity_value


def net_edge_after_costs(
    gross_edge: Decimal | None,
    fees: Decimal | None = None,
    slippage: Decimal | None = None,
    safety_buffer: Decimal | None = None,
) -> Decimal:
    gross = _decimal_or_zero(gross_edge)
    fee_cost = _non_negative_decimal(fees)
    slippage_cost = _non_negative_decimal(slippage)
    buffer_cost = _non_negative_decimal(safety_buffer)

    return gross - fee_cost - slippage_cost - buffer_cost


def edge_bps_after_costs(
    net_edge: Decimal | None,
    reference_notional: Decimal | None,
) -> Decimal:
    net_edge_value = _decimal_or_zero(net_edge)
    reference_notional_value = _positive_decimal_or_none(reference_notional)

    if reference_notional_value is None:
        return DECIMAL_ZERO

    ratio = safe_div(
        net_edge_value,
        reference_notional_value,
        default=DECIMAL_ZERO,
    )
    if ratio is None:
        return DECIMAL_ZERO

    return ratio * DECIMAL_10_000


def reference_notional_from_quote(
    quote: QuoteSnapshot,
    quantity: Decimal | None,
    *,
    prefer_ask: bool = True,
) -> Decimal:
    quantity_value = _positive_decimal_or_none(quantity)
    if quantity_value is None:
        return DECIMAL_ZERO

    reference_price = (
        quote.ask or quote.mid_price or quote.bid
        if prefer_ask
        else quote.bid or quote.mid_price or quote.ask
    )
    reference_price_value = _positive_decimal_or_none(reference_price)
    if reference_price_value is None:
        return DECIMAL_ZERO

    return reference_price_value * quantity_value


# ============================================================
# Quantity helpers
# ============================================================

def resolve_trade_quantity(
    config: CrossExchangeSpreadConfig,
    quantity: Decimal | None = None,
) -> Decimal:
    """
    Визначає trade quantity з урахуванням config bounds.

    Пріоритет:
    1. quantity argument
    2. config.default_trade_size
    3. Decimal("1")

    Якщо config має min_trade_size / max_trade_size, кількість clamp-иться
    у цей діапазон.
    """
    resolved = quantity if quantity is not None else config.default_trade_size
    resolved = _positive_decimal_or_none(resolved) or Decimal("1")

    min_trade_size, max_trade_size = _resolve_trade_size_bounds(config)

    if min_trade_size is not None and resolved < min_trade_size:
        resolved = min_trade_size

    if max_trade_size is not None and resolved > max_trade_size:
        resolved = max_trade_size

    return resolved


# ============================================================
# Main cost breakdown
# ============================================================

def calculate_cost_breakdown(
    buy_quote: QuoteSnapshot,
    sell_quote: QuoteSnapshot,
    quantity: Decimal,
    buy_exchange: str,
    sell_exchange: str,
    config: CrossExchangeSpreadConfig,
    *,
    use_maker_fee: bool = False,
) -> SpreadCostBreakdown:
    """
    Розраховує повний cost breakdown для cross-exchange arbitrage.

    buy leg:
        buy_quote.ask

    sell leg:
        sell_quote.bid
    """
    trade_quantity = resolve_trade_quantity(config, quantity)

    buy_price = buy_quote.ask
    sell_price = sell_quote.bid

    if buy_price is None or sell_price is None:
        return SpreadCostBreakdown()

    gross_edge = gross_edge_from_prices(
        buy_price=buy_price,
        sell_price=sell_price,
        quantity=trade_quantity,
    )

    estimated_fees = estimate_total_fees(
        buy_price=buy_price,
        sell_price=sell_price,
        quantity=trade_quantity,
        buy_exchange=buy_exchange,
        sell_exchange=sell_exchange,
        config=config,
        use_maker_fee=use_maker_fee,
    )

    estimated_slippage = estimate_total_slippage(
        buy_quote=buy_quote,
        sell_quote=sell_quote,
        quantity=trade_quantity,
        config=config,
    )

    reference_price = buy_quote.mid_price or sell_quote.mid_price or buy_price
    safety_buffer = estimate_safety_buffer_cost(
        reference_price=reference_price,
        quantity=trade_quantity,
        safety_buffer_bps=config.safety_buffer_bps,
    )

    net_edge = net_edge_after_costs(
        gross_edge=gross_edge,
        fees=estimated_fees,
        slippage=estimated_slippage,
        safety_buffer=safety_buffer,
    )

    return SpreadCostBreakdown(
        gross_edge=gross_edge,
        estimated_fees=estimated_fees,
        estimated_slippage=estimated_slippage,
        safety_buffer=safety_buffer,
        net_edge=net_edge,
    )


__all__ = [
    "CostSide",
    "LiquiditySide",
    "SpreadCostBreakdown",
    "normalize_fee_overrides",
    "get_fee_rate",
    "estimate_fee_cost",
    "estimate_total_fees",
    "estimate_simple_slippage_ratio",
    "estimate_simple_slippage",
    "estimate_slippage_cost_from_quote",
    "estimate_total_slippage",
    "estimate_safety_buffer_cost",
    "gross_edge_from_prices",
    "net_edge_after_costs",
    "edge_bps_after_costs",
    "reference_notional_from_quote",
    "resolve_trade_quantity",
    "calculate_cost_breakdown",
]