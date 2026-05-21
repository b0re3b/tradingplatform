from __future__ import annotations

import hashlib
import math
import time
from decimal import ROUND_DOWN, ROUND_HALF_UP
from typing import Any, Mapping
from uuid import uuid4

from execution.enums import OrderSide, OrderStatus, OrderType, TimeInForce
from execution.exceptions import ExecutionRejectedError
from risk.enums import MarginMode, OrderIntent, PositionSide

_EPSILON: float = 1e-12
_BPS_DENOMINATOR: float = 10_000.0


# ---------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------

from decimal import Decimal, InvalidOperation


def _decimal_from_number(value: float | int | str | Decimal) -> Decimal:
    if isinstance(value, Decimal):
        return value

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid decimal value: {value!r}") from exc

def now_ms() -> int:
    """
    Return current Unix timestamp in milliseconds.
    """
    return int(time.time() * 1000)


def now_ts() -> float:
    """
    Return current Unix timestamp in seconds.
    """
    return time.time()


# ---------------------------------------------------------------------
# Generic numeric helpers
# ---------------------------------------------------------------------


def is_finite_number(value: float | int | None) -> bool:
    """
    Strict finite number check.

    None, NaN and +/-inf are considered invalid.
    """
    if value is None:
        return False

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def require_finite_number(value: float | int | None, field_name: str) -> float:
    """
    Return value as float or raise ValueError.
    """
    if value is None:
        raise ValueError(f"{field_name} is required")

    if not is_finite_number(value):
        raise ValueError(f"{field_name} must be finite")

    return float(value)

def require_positive_number(value: float | int | None, field_name: str) -> float:
    value_f = require_finite_number(value, field_name)

    if value_f <= 0:
        raise ValueError(f"{field_name} must be > 0")

    return value_f


def require_non_negative_number(value: float | int | None, field_name: str) -> float:
    value_f = require_finite_number(value, field_name)

    if value_f < 0:
        raise ValueError(f"{field_name} must be >= 0")

    return value_f


def safe_float(value: Any, default: float | None = None) -> float | None:
    """
    Convert arbitrary runtime value to finite float.

    Returns default for None, empty strings, NaN, infinities and invalid values.
    """
    if value is None or value == "":
        return default

    try:
        value_f = float(value)
    except (TypeError, ValueError, OverflowError):
        return default

    if not math.isfinite(value_f):
        return default

    return value_f


def safe_int(value: Any, default: int | None = None) -> int | None:
    """
    Convert arbitrary runtime value to int.
    """
    if value is None or value == "":
        return default

    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Zero-safe division.
    """
    numerator_f = require_finite_number(numerator, "numerator")
    denominator_f = require_finite_number(denominator, "denominator")
    default_f = require_finite_number(default, "default")

    if abs(denominator_f) <= _EPSILON:
        return default_f

    return numerator_f / denominator_f


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp value into [min_value, max_value].
    """
    value_f = require_finite_number(value, "value")
    min_f = require_finite_number(min_value, "min_value")
    max_f = require_finite_number(max_value, "max_value")

    if min_f > max_f:
        raise ValueError("min_value must be <= max_value")

    return max(min_f, min(value_f, max_f))


# ---------------------------------------------------------------------
# Symbol / market helpers
# ---------------------------------------------------------------------


def normalize_symbol(symbol: str) -> str:
    """
    Normalize symbol for Binance USD-M Futures.

    Examples:
    - btcusdt -> BTCUSDT
    - BTC/USDT -> BTCUSDT
    - BTC-USDT -> BTCUSDT
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError("symbol is required")

    normalized = (
        symbol.strip()
        .upper()
        .replace("/", "")
        .replace("-", "")
        .replace("_", "")
        .replace(":", "")
    )

    if not normalized:
        raise ValueError("symbol is empty after normalization")

    return normalized


def normalize_exchange(exchange: str | None, default: str = "binance") -> str:
    """
    Normalize exchange id.
    """
    raw = exchange or default
    normalized = raw.strip().lower()

    if not normalized:
        raise ValueError("exchange is required")

    return normalized


def normalize_market_type(market_type: str | None, default: str = "usdm_futures") -> str:
    """
    Normalize futures market type.

    Current execution package is Binance USD-M Futures-first.
    """
    raw = market_type or default
    normalized = raw.strip().lower()

    aliases = {
        "usd-m": "usdm_futures",
        "usdm": "usdm_futures",
        "usd_m": "usdm_futures",
        "binance_usdm": "usdm_futures",
        "binance_usdm_futures": "usdm_futures",
        "linear": "usdm_futures",
        "perp": "usdm_futures",
        "perpetual": "usdm_futures",
        "swap": "usdm_futures",
    }

    return aliases.get(normalized, normalized)


def is_supported_execution_market(
    *,
    exchange: str | None,
    market_type: str | None,
) -> bool:
    """
    Return whether the current execution target is supported.

    For now: Binance USD-M Futures only.
    """
    return (
        normalize_exchange(exchange) == "binance"
        and normalize_market_type(market_type) == "usdm_futures"
    )


# ---------------------------------------------------------------------
# Decimal / rounding helpers
# ---------------------------------------------------------------------




def format_decimal(value: float | int | Decimal, *, normalize: bool = True) -> str:
    """
    Format number for Binance REST params.

    Binance accepts decimal strings without trailing zeros.
    """
    value_d = _decimal_from_number(value)

    if normalize:
        formatted = format(value_d.normalize(), "f")
    else:
        formatted = format(value_d, "f")

    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")

    return formatted or "0"


def round_down_to_step(value: float, step: float | None) -> float:
    """
    Round value down to exchange step size.

    Used for quantity rounding.
    """
    value_f = require_non_negative_number(value, "value")

    if step is None:
        return value_f

    step_f = require_positive_number(step, "step")

    value_d = _decimal_from_number(value_f)
    step_d = _decimal_from_number(step_f)

    rounded = (value_d / step_d).to_integral_value(rounding=ROUND_DOWN) * step_d

    return float(rounded)


def round_to_tick(value: float, tick_size: float | None) -> float:
    """
    Round price to nearest exchange tick size.
    """
    value_f = require_positive_number(value, "value")

    if tick_size is None:
        return value_f

    tick_f = require_positive_number(tick_size, "tick_size")

    value_d = _decimal_from_number(value_f)
    tick_d = _decimal_from_number(tick_f)

    rounded = (value_d / tick_d).to_integral_value(rounding=ROUND_HALF_UP) * tick_d

    return float(rounded)


def round_price(price: float, tick_size: float | None = None) -> float:
    return round_to_tick(price, tick_size)


def round_quantity(quantity: float, step_size: float | None = None) -> float:
    return round_down_to_step(quantity, step_size)


def validate_min_notional(
    *,
    quantity: float,
    price: float | None,
    min_notional: float | None,
) -> bool:
    """
    Validate notional against exchange minimum.

    Market orders may not always have a known final price, so caller may pass
    estimated mark/entry price.
    """
    if min_notional is None:
        return True

    quantity_f = require_positive_number(quantity, "quantity")
    price_f = require_positive_number(price, "price")
    min_notional_f = require_positive_number(min_notional, "min_notional")

    return quantity_f * price_f >= min_notional_f


# ---------------------------------------------------------------------
# Client order id helpers
# ---------------------------------------------------------------------


def sanitize_client_order_id(value: str) -> str:
    """
    Keep client order id compatible with Binance common constraints.

    Binance accepts a limited character set for newClientOrderId. This helper
    keeps ids compact and deterministic enough for execution tracking.
    """
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    cleaned = "".join(ch for ch in value if ch in allowed)

    return cleaned.strip("-_")


def short_hash(value: str, length: int = 8) -> str:
    """
    Stable short hash for client order ids.
    """
    if length <= 0:
        raise ValueError("length must be > 0")

    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def shorten_client_order_id(client_order_id: str, max_length: int = 36) -> str:
    """
    Shorten client order id while preserving uniqueness reasonably.
    """
    cleaned = sanitize_client_order_id(client_order_id)

    if len(cleaned) <= max_length:
        return cleaned

    digest = short_hash(cleaned, length=8)
    prefix_length = max(1, max_length - len(digest) - 1)

    return f"{cleaned[:prefix_length]}-{digest}"


def build_client_order_id(
    *,
    prefix: str = "ts",
    signal_id: str | None = None,
    strategy_name: str | None = None,
    symbol: str | None = None,
    order_intent: str | None = None,
    leg_index: int | None = None,
    timestamp_ms: int | None = None,
    max_length: int = 36,
) -> str:
    """
    Build Binance-compatible deterministic-ish client order id.

    Kept in utils.py to avoid a separate order_id_factory.py.
    """
    timestamp = timestamp_ms or now_ms()

    raw_parts = [
        prefix,
        normalize_symbol(symbol) if symbol else None,
        _compact_text(strategy_name, max_chars=8) if strategy_name else None,
        _compact_text(signal_id, max_chars=8) if signal_id else None,
        _compact_text(order_intent, max_chars=8) if order_intent else None,
        str(leg_index) if leg_index is not None else None,
        str(timestamp),
        uuid4().hex[:6],
    ]

    raw = "-".join(part for part in raw_parts if part)
    cleaned = sanitize_client_order_id(raw)

    return shorten_client_order_id(cleaned, max_length=max_length)


def _compact_text(value: str, *, max_chars: int) -> str:
    cleaned = sanitize_client_order_id(value.replace(" ", "_"))

    if not cleaned:
        return ""

    if len(cleaned) <= max_chars:
        return cleaned

    return cleaned[:max_chars]


# ---------------------------------------------------------------------
# Side / intent mapping helpers
# ---------------------------------------------------------------------


def normalize_order_side(value: OrderSide | str) -> OrderSide:
    if isinstance(value, OrderSide):
        return value

    return OrderSide.from_raw(value)


def normalize_order_type(value: OrderType | str) -> OrderType:
    if isinstance(value, OrderType):
        return value

    return OrderType.from_raw(value)


def normalize_order_status(value: OrderStatus | str | None) -> OrderStatus:
    if isinstance(value, OrderStatus):
        return value

    return OrderStatus.from_raw(value)


def normalize_time_in_force(value: TimeInForce | str | None) -> TimeInForce | None:
    if value is None:
        return None

    if isinstance(value, TimeInForce):
        return value

    return TimeInForce.from_raw(value)


def opposite_order_side(side: OrderSide | str) -> OrderSide:
    side_n = normalize_order_side(side)

    if side_n is OrderSide.BUY:
        return OrderSide.SELL

    return OrderSide.BUY


def order_side_for_position_open(side: PositionSide) -> OrderSide:
    """
    Map risk PositionSide to Binance order side for opening/increasing position.
    """
    if side is PositionSide.LONG:
        return OrderSide.BUY

    if side is PositionSide.SHORT:
        return OrderSide.SELL

    raise ValueError(f"Unsupported position side for open: {side!r}")


def order_side_for_position_close(side: PositionSide) -> OrderSide:
    """
    Map risk PositionSide to Binance order side for reducing/closing position.
    """
    if side is PositionSide.LONG:
        return OrderSide.SELL

    if side is PositionSide.SHORT:
        return OrderSide.BUY

    raise ValueError(f"Unsupported position side for close: {side!r}")


def order_side_for_intent(
    *,
    position_side: PositionSide,
    order_intent: OrderIntent,
) -> OrderSide:
    """
    Resolve exchange OrderSide from risk position side + order intent.
    """
    if order_intent.increases_risk:
        return order_side_for_position_open(position_side)

    if order_intent.reduces_risk:
        return order_side_for_position_close(position_side)

    if order_intent is OrderIntent.OPEN:
        return order_side_for_position_open(position_side)

    if order_intent is OrderIntent.CLOSE:
        return order_side_for_position_close(position_side)

    raise ValueError(
        f"Unsupported order intent for side mapping: {order_intent!r}"
    )


def binance_position_side(side: PositionSide | str | None) -> str | None:
    """
    Convert risk PositionSide to Binance positionSide.

    Returns None for one-way mode when caller intentionally omits positionSide.
    """
    if side is None:
        return None

    raw_side = getattr(side, "value", side)
    normalized = str(raw_side).strip().upper()

    if normalized in {"LONG", "SHORT", "BOTH"}:
        return normalized

    raise ValueError(f"Unsupported Binance position side: {side!r}")


def binance_margin_type(margin_mode: MarginMode | str | None) -> str | None:
    """
    Convert risk MarginMode to Binance marginType.
    """
    if margin_mode is None:
        return None

    raw_margin_mode = getattr(margin_mode, "value", margin_mode)
    normalized = str(raw_margin_mode).strip().upper()

    aliases = {
        "CROSS": "CROSSED",
        "CROSSED": "CROSSED",
        "ISOLATED": "ISOLATED",
    }

    if normalized not in aliases:
        raise ValueError(f"Unsupported margin mode: {margin_mode!r}")

    return aliases[normalized]


def is_reduce_intent(order_intent: OrderIntent) -> bool:
    return bool(order_intent.reduces_risk)


def is_increase_intent(order_intent: OrderIntent) -> bool:
    return bool(order_intent.increases_risk)


# ---------------------------------------------------------------------
# Order status helpers
# ---------------------------------------------------------------------


def is_terminal_order_status(status: OrderStatus | str | None) -> bool:
    return normalize_order_status(status).is_terminal


def is_open_order_status(status: OrderStatus | str | None) -> bool:
    return normalize_order_status(status).is_open


def is_filled_order_status(status: OrderStatus | str | None) -> bool:
    return normalize_order_status(status) is OrderStatus.FILLED


def is_cancelled_order_status(status: OrderStatus | str | None) -> bool:
    return normalize_order_status(status).is_cancelled


def order_status_from_exchange_payload(payload: Mapping[str, Any]) -> OrderStatus:
    return normalize_order_status(payload.get("status"))


# ---------------------------------------------------------------------
# Reduce-only / protective order helpers
# ---------------------------------------------------------------------


def should_be_reduce_only(
    *,
    order_intent: OrderIntent | None = None,
    trigger_type: str | None = None,
    close_position: bool | None = None,
) -> bool:
    """
    Return whether an order should use reduce-only semantics.
    """
    if close_position:
        return True

    if order_intent is not None and order_intent.reduces_risk:
        return True

    if trigger_type is not None:
        normalized = trigger_type.strip().lower()
        if normalized in {
            "stop_loss",
            "take_profit",
            "trailing_stop",
            "breakeven_stop",
            "partial_take_profit",
            "manual_close",
            "risk_close",
            "risk_reduce",
            "liquidation_protection",
        }:
            return True

    return False


def validate_reduce_only_order(
    *,
    order_intent: OrderIntent | None,
    reduce_only: bool | None,
    close_position: bool | None = None,
    trigger_type: str | None = None,
) -> None:
    """
    Validate reduce-only requirements for risk-reducing/protective orders.
    """
    required = should_be_reduce_only(
        order_intent=order_intent,
        trigger_type=trigger_type,
        close_position=close_position,
    )

    if required and not reduce_only and not close_position:
        raise ExecutionRejectedError(
            "reduce_only=True or close_position=True is required for risk-reducing/protective orders"
        )


def validate_close_position_order(
    *,
    order_type: OrderType | str,
    quantity: float | None,
    close_position: bool | None,
) -> None:
    """
    Validate Binance closePosition semantics.

    Binance closePosition is supported for STOP_MARKET / TAKE_PROFIT_MARKET.
    When closePosition=True, quantity should not be sent.
    """
    order_type_n = normalize_order_type(order_type)

    if not close_position:
        return

    if not order_type_n.supports_close_position:
        raise ExecutionRejectedError(
            f"close_position=True is not supported for order type {order_type_n.value}"
        )

    if quantity is not None:
        raise ExecutionRejectedError(
            "quantity must be omitted when close_position=True"
        )


# ---------------------------------------------------------------------
# Price / slippage / fill helpers
# ---------------------------------------------------------------------


def calculate_notional(price: float, quantity: float) -> float:
    price_f = require_positive_number(price, "price")
    quantity_f = require_non_negative_number(quantity, "quantity")

    return price_f * quantity_f


def calculate_slippage_bps(
    *,
    expected_price: float,
    actual_price: float,
    side: OrderSide | str | None = None,
) -> float:
    """
    Calculate absolute slippage in basis points.

    If side is provided, positive slippage means worse execution:
    - BUY worse when actual > expected;
    - SELL worse when actual < expected.

    Without side, returns absolute distance in bps.
    """
    expected = require_positive_number(expected_price, "expected_price")
    actual = require_positive_number(actual_price, "actual_price")

    if side is None:
        return abs(actual - expected) / expected * _BPS_DENOMINATOR

    side_n = normalize_order_side(side)

    if side_n is OrderSide.BUY:
        return (actual - expected) / expected * _BPS_DENOMINATOR

    return (expected - actual) / expected * _BPS_DENOMINATOR


def calculate_spread_bps(
    *,
    bid: float,
    ask: float,
    mid_price: float | None = None,
) -> float:
    bid_f = require_positive_number(bid, "bid")
    ask_f = require_positive_number(ask, "ask")

    if ask_f < bid_f:
        raise ValueError("ask must be >= bid")

    mid = mid_price if mid_price is not None else (bid_f + ask_f) / 2.0
    mid_f = require_positive_number(mid, "mid_price")

    return (ask_f - bid_f) / mid_f * _BPS_DENOMINATOR


def calculate_average_fill_price(
    fills: list[Mapping[str, Any]],
    *,
    price_key: str = "price",
    quantity_key: str = "qty",
) -> float | None:
    """
    Calculate weighted average fill price from normalized fills.
    """
    total_qty = 0.0
    total_quote = 0.0

    for fill in fills:
        price = safe_float(fill.get(price_key))
        qty = safe_float(fill.get(quantity_key))

        if price is None or qty is None or price <= 0 or qty <= 0:
            continue

        total_qty += qty
        total_quote += price * qty

    if total_qty <= 0:
        return None

    return total_quote / total_qty


def calculate_order_avg_price_from_payload(payload: Mapping[str, Any]) -> float | None:
    """
    Resolve average order price from Binance-normalized order payload.
    """
    avg_price = safe_float(payload.get("avg_price"))

    if avg_price is not None and avg_price > 0:
        return avg_price

    executed_qty = safe_float(payload.get("executed_qty")) or safe_float(payload.get("cum_qty"))
    cum_quote = (
        safe_float(payload.get("cum_quote"))
        or safe_float(payload.get("cumulative_quote_qty"))
    )

    if executed_qty is None or executed_qty <= 0:
        return None

    if cum_quote is None or cum_quote <= 0:
        return None

    return cum_quote / executed_qty


def calculate_fill_ratio(
    *,
    executed_qty: float | None,
    original_qty: float | None,
) -> float:
    executed = safe_float(executed_qty, default=0.0) or 0.0
    original = safe_float(original_qty, default=0.0) or 0.0

    if original <= 0:
        return 0.0

    return clamp(executed / original, 0.0, 1.0)


# ---------------------------------------------------------------------
# Binance normalized payload helpers
# ---------------------------------------------------------------------


def extract_order_id(payload: Mapping[str, Any]) -> str | None:
    order_id = payload.get("order_id")

    if order_id is None:
        order_id = payload.get("orderId")

    if order_id is None:
        return None

    return str(order_id)


def extract_client_order_id(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("client_order_id")

    if value is None:
        value = payload.get("clientOrderId")

    if value is None:
        value = payload.get("orig_client_order_id")

    if value is None:
        return None

    return str(value)


def extract_symbol(payload: Mapping[str, Any]) -> str:
    symbol = payload.get("symbol")

    if not symbol:
        raise ValueError("payload.symbol is required")

    return normalize_symbol(str(symbol))


def extract_executed_quantity(payload: Mapping[str, Any]) -> float:
    """
    Extract executed/cumulative quantity from either:
    - Binance normalized REST payload;
    - execution.order_* event payload produced by OrderResult.to_event_payload().
    """
    value = payload.get("executed_quantity")

    if value is None:
        value = payload.get("executed_qty")

    if value is None:
        value = payload.get("executedQty")

    if value is None:
        value = payload.get("cum_qty")

    if value is None:
        value = payload.get("cumQty")

    if value is None:
        value = payload.get("cumulative_quantity")

    return safe_float(value, default=0.0) or 0.0


def extract_original_quantity(payload: Mapping[str, Any]) -> float:
    """
    Extract original requested quantity from either:
    - Binance normalized REST payload;
    - execution.order_* event payload produced by OrderResult.to_event_payload().
    """
    value = payload.get("original_quantity")

    if value is None:
        value = payload.get("orig_qty")

    if value is None:
        value = payload.get("origQty")

    if value is None:
        value = payload.get("quantity")

    return safe_float(value, default=0.0) or 0.0


def extract_order_price(payload: Mapping[str, Any]) -> float | None:
    value = payload.get("price")

    return safe_float(value)


def extract_order_status(payload: Mapping[str, Any]) -> OrderStatus:
    return normalize_order_status(payload.get("status"))


def extract_order_side(payload: Mapping[str, Any]) -> OrderSide | None:
    side = payload.get("side")

    if side is None:
        return None

    return normalize_order_side(str(side))


def extract_order_type(payload: Mapping[str, Any]) -> OrderType | None:
    order_type = payload.get("type") or payload.get("orig_type")

    if order_type is None:
        return None

    return normalize_order_type(str(order_type))


# ---------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------


def signed_position_size(
    *,
    side: PositionSide,
    size: float,
) -> float:
    size_f = require_non_negative_number(size, "size")

    if side is PositionSide.LONG:
        return size_f

    if side is PositionSide.SHORT:
        return -size_f

    return 0.0


def abs_position_size(position_amt: float | int | None) -> float:
    value = safe_float(position_amt, default=0.0) or 0.0

    return abs(value)


def infer_position_side_from_amount(position_amt: float | int | None) -> PositionSide | None:
    amount = safe_float(position_amt, default=0.0) or 0.0

    if amount > _EPSILON:
        return PositionSide.LONG

    if amount < -_EPSILON:
        return PositionSide.SHORT

    return None


def is_flat_position(
    *,
    size: float | None = None,
    position_amt: float | None = None,
    epsilon: float = _EPSILON,
) -> bool:
    value = size if size is not None else position_amt
    value_f = safe_float(value, default=0.0) or 0.0

    return abs(value_f) <= epsilon


def calculate_unrealized_pnl_pct(
    *,
    unrealized_pnl: float,
    margin_used: float,
) -> float:
    return safe_div(unrealized_pnl, margin_used, default=0.0)


# ---------------------------------------------------------------------
# Event payload helpers
# ---------------------------------------------------------------------


def base_execution_payload(
    *,
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    symbol: str | None = None,
    signal_id: str | None = None,
    strategy_name: str | None = None,
    reservation_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Shared payload base for execution.* events.
    """
    payload: dict[str, Any] = {
        "exchange": normalize_exchange(exchange),
        "market_type": normalize_market_type(market_type),
        "timestamp": now_ms(),
    }

    if symbol is not None:
        payload["symbol"] = normalize_symbol(symbol)

    if signal_id is not None:
        payload["signal_id"] = signal_id

    if strategy_name is not None:
        payload["strategy_name"] = strategy_name

    if reservation_id is not None:
        payload["reservation_id"] = reservation_id

    if metadata:
        payload["metadata"] = dict(metadata)

    return payload


def merge_metadata(
    *items: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    Merge metadata mappings, skipping None.
    Later mappings override earlier keys.
    """
    result: dict[str, Any] = {}

    for item in items:
        if not item:
            continue

        result.update(dict(item))

    return result


__all__ = [
    "now_ms",
    "now_ts",
    "is_finite_number",
    "require_finite_number",
    "require_positive_number",
    "require_non_negative_number",
    "safe_float",
    "safe_int",
    "safe_div",
    "clamp",
    "normalize_symbol",
    "normalize_exchange",
    "normalize_market_type",
    "is_supported_execution_market",
    "format_decimal",
    "round_down_to_step",
    "round_to_tick",
    "round_price",
    "round_quantity",
    "validate_min_notional",
    "sanitize_client_order_id",
    "short_hash",
    "shorten_client_order_id",
    "build_client_order_id",
    "normalize_order_side",
    "normalize_order_type",
    "normalize_order_status",
    "normalize_time_in_force",
    "opposite_order_side",
    "order_side_for_position_open",
    "order_side_for_position_close",
    "order_side_for_intent",
    "binance_position_side",
    "binance_margin_type",
    "is_reduce_intent",
    "is_increase_intent",
    "is_terminal_order_status",
    "is_open_order_status",
    "is_filled_order_status",
    "is_cancelled_order_status",
    "order_status_from_exchange_payload",
    "should_be_reduce_only",
    "validate_reduce_only_order",
    "validate_close_position_order",
    "calculate_notional",
    "calculate_slippage_bps",
    "calculate_spread_bps",
    "calculate_average_fill_price",
    "calculate_order_avg_price_from_payload",
    "calculate_fill_ratio",
    "extract_order_id",
    "extract_client_order_id",
    "extract_symbol",
    "extract_executed_quantity",
    "extract_original_quantity",
    "extract_order_price",
    "extract_order_status",
    "extract_order_side",
    "extract_order_type",
    "signed_position_size",
    "abs_position_size",
    "infer_position_side_from_amount",
    "is_flat_position",
    "calculate_unrealized_pnl_pct",
    "base_execution_payload",
    "merge_metadata",
]