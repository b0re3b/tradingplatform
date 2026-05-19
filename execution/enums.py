from __future__ import annotations

from enum import Enum


class OrderSide(str, Enum):
    """
    Exchange order side.

    For Binance USD-M Futures:
    - BUY opens/increases LONG or reduces SHORT.
    - SELL opens/increases SHORT or reduces LONG.

    Domain position direction should normally come from risk.enums.PositionSide.
    """

    BUY = "BUY"
    SELL = "SELL"

    @property
    def is_buy(self) -> bool:
        return self is OrderSide.BUY

    @property
    def is_sell(self) -> bool:
        return self is OrderSide.SELL

    @classmethod
    def from_raw(cls, value: str) -> "OrderSide":
        normalized = value.strip().upper()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(f"Unsupported order side: {value!r}") from exc


class OrderType(str, Enum):
    """
    Binance USD-M Futures compatible order types.

    Keep this execution-specific. Strategy and Risk should not depend on these
    low-level exchange order types directly.
    """

    MARKET = "MARKET"
    LIMIT = "LIMIT"

    STOP = "STOP"
    STOP_MARKET = "STOP_MARKET"

    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"

    TRAILING_STOP_MARKET = "TRAILING_STOP_MARKET"

    @property
    def is_market(self) -> bool:
        return self in {
            OrderType.MARKET,
            OrderType.STOP_MARKET,
            OrderType.TAKE_PROFIT_MARKET,
            OrderType.TRAILING_STOP_MARKET,
        }

    @property
    def is_limit(self) -> bool:
        return self in {
            OrderType.LIMIT,
            OrderType.STOP,
            OrderType.TAKE_PROFIT,
        }

    @property
    def is_trigger_order(self) -> bool:
        return self in {
            OrderType.STOP,
            OrderType.STOP_MARKET,
            OrderType.TAKE_PROFIT,
            OrderType.TAKE_PROFIT_MARKET,
            OrderType.TRAILING_STOP_MARKET,
        }

    @property
    def requires_price(self) -> bool:
        return self in {
            OrderType.LIMIT,
            OrderType.STOP,
            OrderType.TAKE_PROFIT,
        }

    @property
    def requires_stop_price(self) -> bool:
        return self in {
            OrderType.STOP,
            OrderType.STOP_MARKET,
            OrderType.TAKE_PROFIT,
            OrderType.TAKE_PROFIT_MARKET,
        }

    @property
    def supports_close_position(self) -> bool:
        return self in {
            OrderType.STOP_MARKET,
            OrderType.TAKE_PROFIT_MARKET,
        }

    @classmethod
    def from_raw(cls, value: str) -> "OrderType":
        normalized = value.strip().upper()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(f"Unsupported order type: {value!r}") from exc


class OrderStatus(str, Enum):
    """
    Normalized order lifecycle statuses.

    Values intentionally match Binance order status names where possible.
    """

    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"

    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    EXPIRED_IN_MATCH = "EXPIRED_IN_MATCH"

    PENDING_CANCEL = "PENDING_CANCEL"
    UNKNOWN = "UNKNOWN"

    @property
    def is_open(self) -> bool:
        return self in {
            OrderStatus.NEW,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.PENDING_CANCEL,
        }

    @property
    def is_terminal(self) -> bool:
        return self in {
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.EXPIRED_IN_MATCH,
        }

    @property
    def is_successful_fill(self) -> bool:
        return self is OrderStatus.FILLED

    @property
    def is_failure(self) -> bool:
        return self in {
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.EXPIRED_IN_MATCH,
        }

    @property
    def is_cancelled(self) -> bool:
        return self is OrderStatus.CANCELED

    @classmethod
    def from_raw(cls, value: str | None) -> "OrderStatus":
        if value is None:
            return cls.UNKNOWN

        normalized = value.strip().upper()
        if not normalized:
            return cls.UNKNOWN

        try:
            return cls(normalized)
        except ValueError:
            return cls.UNKNOWN


class TimeInForce(str, Enum):
    """
    Binance-compatible time in force values.
    """

    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"
    GTX = "GTX"

    @property
    def is_post_only(self) -> bool:
        return self is TimeInForce.GTX

    @classmethod
    def from_raw(cls, value: str) -> "TimeInForce":
        normalized = value.strip().upper()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(f"Unsupported time in force: {value!r}") from exc


class ExecutionStatus(str, Enum):
    """
    High-level execution plan / trade execution lifecycle.

    This is not the same thing as exchange order status. One execution may
    contain one or multiple exchange orders.
    """

    PENDING = "pending"
    ACCEPTED = "accepted"
    PLANNING = "planning"
    PLANNED = "planned"

    SUBMITTING = "submitting"
    SUBMITTED = "submitted"

    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"

    COMPLETED = "completed"

    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

    BLOCKED = "blocked"
    KILL_SWITCHED = "kill_switched"

    @property
    def is_active(self) -> bool:
        return self in {
            ExecutionStatus.PENDING,
            ExecutionStatus.ACCEPTED,
            ExecutionStatus.PLANNING,
            ExecutionStatus.PLANNED,
            ExecutionStatus.SUBMITTING,
            ExecutionStatus.SUBMITTED,
            ExecutionStatus.PARTIALLY_FILLED,
        }

    @property
    def is_terminal(self) -> bool:
        return self in {
            ExecutionStatus.COMPLETED,
            ExecutionStatus.REJECTED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.EXPIRED,
            ExecutionStatus.BLOCKED,
            ExecutionStatus.KILL_SWITCHED,
        }

    @property
    def is_successful(self) -> bool:
        return self is ExecutionStatus.COMPLETED

    @property
    def is_failure(self) -> bool:
        return self in {
            ExecutionStatus.REJECTED,
            ExecutionStatus.FAILED,
            ExecutionStatus.EXPIRED,
            ExecutionStatus.BLOCKED,
            ExecutionStatus.KILL_SWITCHED,
        }


class ExecutionMode(str, Enum):
    """
    Defines how SmartExecution should build an ExecutionPlan.

    Risk decides whether trade is allowed and final size/leverage.
    ExecutionMode only controls technical order placement.
    """

    MARKET = "market"
    LIMIT = "limit"
    POST_ONLY = "post_only"
    SMART = "smart"
    LIQUIDITY_AWARE = "liquidity_aware"
    TWAP = "twap"

    @property
    def is_immediate(self) -> bool:
        return self is ExecutionMode.MARKET

    @property
    def is_passive(self) -> bool:
        return self in {
            ExecutionMode.LIMIT,
            ExecutionMode.POST_ONLY,
        }

    @property
    def may_split_orders(self) -> bool:
        return self in {
            ExecutionMode.SMART,
            ExecutionMode.LIQUIDITY_AWARE,
            ExecutionMode.TWAP,
        }


class TriggerType(str, Enum):
    """
    Trigger intent for protective or conditional futures orders.
    """

    NONE = "none"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    LIQUIDATION_PROTECTION = "liquidation_protection"
    MANUAL_CLOSE = "manual_close"
    RISK_CLOSE = "risk_close"
    RISK_REDUCE = "risk_reduce"

    @property
    def is_protective(self) -> bool:
        return self in {
            TriggerType.STOP_LOSS,
            TriggerType.TAKE_PROFIT,
            TriggerType.TRAILING_STOP,
            TriggerType.LIQUIDATION_PROTECTION,
        }

    @property
    def is_risk_driven(self) -> bool:
        return self in {
            TriggerType.RISK_CLOSE,
            TriggerType.RISK_REDUCE,
            TriggerType.LIQUIDATION_PROTECTION,
        }


class WorkingType(str, Enum):
    """
    Binance Futures working type for trigger orders.
    """

    MARK_PRICE = "MARK_PRICE"
    CONTRACT_PRICE = "CONTRACT_PRICE"

    @classmethod
    def from_raw(cls, value: str) -> "WorkingType":
        normalized = value.strip().upper()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(f"Unsupported working type: {value!r}") from exc


class SLTPType(str, Enum):
    """
    Protective order type managed by SLTPManager.

    These are execution-level protective intents, not exchange order types.
    SLTPManager maps them into OrderType + Binance futures params.
    """

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    BREAKEVEN_STOP = "breakeven_stop"
    PARTIAL_TAKE_PROFIT = "partial_take_profit"

    @property
    def is_stop(self) -> bool:
        return self in {
            SLTPType.STOP_LOSS,
            SLTPType.TRAILING_STOP,
            SLTPType.BREAKEVEN_STOP,
        }

    @property
    def is_take_profit(self) -> bool:
        return self in {
            SLTPType.TAKE_PROFIT,
            SLTPType.PARTIAL_TAKE_PROFIT,
        }

    @property
    def requires_reduce_only(self) -> bool:
        return True


__all__ = [
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    "ExecutionStatus",
    "ExecutionMode",
    "TriggerType",
    "WorkingType",
    "SLTPType",
]