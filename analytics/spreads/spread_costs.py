from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

from .config import CrossExchangeSpreadConfig
from .models import QuoteSnapshot
from .spread_utils import DECIMAL_ZERO, safe_div, to_decimal


@dataclass(slots=True)
class SpreadCostBreakdown:
    gross_edge: Decimal = Decimal("0")
    estimated_fees: Decimal = Decimal("0")
    estimated_slippage: Decimal = Decimal("0")
    safety_buffer: Decimal = Decimal("0")
    net_edge: Decimal = Decimal("0")

    @property
    def total_costs(self) -> Decimal:
        return self.estimated_fees + self.estimated_slippage + self.safety_buffer

    @property
    def is_profitable(self) -> bool:
        return self.net_edge > DECIMAL_ZERO


def estimate_fee_cost(
    price: Decimal | None,
    quantity: Decimal | None,
    fee_rate: Decimal | None,
) -> Decimal:
    if price is None or quantity is None or fee_rate is None:
        return DECIMAL_ZERO

    if price <= DECIMAL_ZERO or quantity <= DECIMAL_ZERO or fee_rate < DECIMAL_ZERO:
        return DECIMAL_ZERO

    return price * quantity * fee_rate


def estimate_simple_slippage(
    quantity: Decimal | None,
    top_book_size: Decimal | None,
    max_slippage_bps: Decimal | None,
) -> Decimal:
    """
    Повертає slippage ratio, а не абсолютну вартість.

    Наприклад:
    - quantity=5
    - top_book_size=10
    - max_slippage_bps=5

    => ratio = 0.5 * 5bps = 2.5bps = 0.00025
    """
    if quantity is None or top_book_size is None or max_slippage_bps is None:
        return DECIMAL_ZERO

    if quantity <= DECIMAL_ZERO or top_book_size <= DECIMAL_ZERO or max_slippage_bps <= DECIMAL_ZERO:
        return DECIMAL_ZERO

    participation = safe_div(quantity, top_book_size, default=DECIMAL_ZERO)
    if participation is None:
        return DECIMAL_ZERO

    if participation <= DECIMAL_ZERO:
        return DECIMAL_ZERO

    capped_participation = min(participation, Decimal("1"))
    slippage_bps = capped_participation * max_slippage_bps

    return slippage_bps / Decimal("10000")


def estimate_slippage_cost_from_quote(
    quote: QuoteSnapshot,
    quantity: Decimal | None,
    max_slippage_bps: Decimal | None,
    use_ask_side: bool | None = None,
) -> Decimal:
    if quantity is None or quantity <= DECIMAL_ZERO:
        return DECIMAL_ZERO

    reference_price = quote.mid_price or quote.ask or quote.bid
    if reference_price is None or reference_price <= DECIMAL_ZERO:
        return DECIMAL_ZERO

    if use_ask_side is True:
        top_book_size = quote.ask_size
    elif use_ask_side is False:
        top_book_size = quote.bid_size
    else:
        top_book_size = quote.ask_size or quote.bid_size

    slippage_ratio = estimate_simple_slippage(
        quantity=quantity,
        top_book_size=top_book_size,
        max_slippage_bps=max_slippage_bps,
    )
    return reference_price * quantity * slippage_ratio


def get_fee_rate(
    exchange: str,
    side: str,
    config: CrossExchangeSpreadConfig,
) -> Decimal:
    fee_overrides = config.metadata.get("fee_rates", {})
    exchange_rates = fee_overrides.get(exchange, {})

    if side == "buy":
        raw_value = exchange_rates.get("buy", config.default_taker_fee_rate)
        return Decimal(str(raw_value))

    if side == "sell":
        raw_value = exchange_rates.get("sell", config.default_taker_fee_rate)
        return Decimal(str(raw_value))

    return config.default_taker_fee_rate


def estimate_total_fees(
    buy_price: Decimal | None,
    sell_price: Decimal | None,
    quantity: Decimal | None,
    buy_exchange: str,
    sell_exchange: str,
    config: CrossExchangeSpreadConfig,
) -> Decimal:
    buy_fee_rate = get_fee_rate(buy_exchange, side="buy", config=config)
    sell_fee_rate = get_fee_rate(sell_exchange, side="sell", config=config)

    buy_fee = estimate_fee_cost(
        price=buy_price,
        quantity=quantity,
        fee_rate=buy_fee_rate,
    )
    sell_fee = estimate_fee_cost(
        price=sell_price,
        quantity=quantity,
        fee_rate=sell_fee_rate,
    )

    return buy_fee + sell_fee


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


def estimate_safety_buffer_cost(
    reference_price: Decimal | None,
    quantity: Decimal | None,
    safety_buffer_bps: Decimal | None,
) -> Decimal:
    if reference_price is None or quantity is None or safety_buffer_bps is None:
        return DECIMAL_ZERO

    if reference_price <= DECIMAL_ZERO or quantity <= DECIMAL_ZERO or safety_buffer_bps <= DECIMAL_ZERO:
        return DECIMAL_ZERO

    return reference_price * quantity * (safety_buffer_bps / Decimal("10000"))


def net_edge_after_costs(
    gross_edge: Decimal | None,
    fees: Decimal | None = None,
    slippage: Decimal | None = None,
    safety_buffer: Decimal | None = None,
) -> Decimal:
    gross = gross_edge if gross_edge is not None else DECIMAL_ZERO
    fee_cost = fees if fees is not None else DECIMAL_ZERO
    slippage_cost = slippage if slippage is not None else DECIMAL_ZERO
    buffer_cost = safety_buffer if safety_buffer is not None else DECIMAL_ZERO

    return gross - fee_cost - slippage_cost - buffer_cost


def calculate_cost_breakdown(
    buy_quote: QuoteSnapshot,
    sell_quote: QuoteSnapshot,
    quantity: Decimal,
    buy_exchange: str,
    sell_exchange: str,
    config: CrossExchangeSpreadConfig,
) -> SpreadCostBreakdown:
    buy_price = buy_quote.ask
    sell_price = sell_quote.bid

    if buy_price is None or sell_price is None:
        return SpreadCostBreakdown()

    gross_edge = (sell_price - buy_price) * quantity

    estimated_fees = estimate_total_fees(
        buy_price=buy_price,
        sell_price=sell_price,
        quantity=quantity,
        buy_exchange=buy_exchange,
        sell_exchange=sell_exchange,
        config=config,
    )

    estimated_slippage = estimate_total_slippage(
        buy_quote=buy_quote,
        sell_quote=sell_quote,
        quantity=quantity,
        config=config,
    )

    reference_price = buy_quote.mid_price or sell_quote.mid_price or buy_price
    safety_buffer = estimate_safety_buffer_cost(
        reference_price=reference_price,
        quantity=quantity,
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


def edge_bps_after_costs(
    net_edge: Decimal | None,
    reference_notional: Decimal | None,
) -> Decimal:
    if net_edge is None or reference_notional is None:
        return DECIMAL_ZERO

    if reference_notional <= DECIMAL_ZERO:
        return DECIMAL_ZERO

    ratio = safe_div(net_edge, reference_notional, default=DECIMAL_ZERO)
    if ratio is None:
        return DECIMAL_ZERO

    return ratio * Decimal("10000")


def resolve_trade_quantity(
    config: CrossExchangeSpreadConfig,
    quantity: Decimal | None = None,
) -> Decimal:
    resolved = quantity if quantity is not None else config.default_trade_size
    if resolved <= DECIMAL_ZERO:
        return Decimal("1")
    return resolved


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
        exchange_key = exchange.strip().lower()
        normalized[exchange_key] = {}

        for side_name, raw_value in side_map.items():
            value = to_decimal(raw_value, default=DECIMAL_ZERO)
            if value is None or value < DECIMAL_ZERO:
                value = DECIMAL_ZERO
            normalized[exchange_key][side_name] = value

    return normalized