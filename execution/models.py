from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping
from uuid import uuid4

from execution.enums import (
    ExecutionMode,
    ExecutionStatus,
    OrderSide,
    OrderStatus,
    OrderType,
    SLTPType,
    TimeInForce,
    TriggerType,
    WorkingType,
)
from execution.exceptions import (
    ExecutionPlanValidationError,
    ExecutionRejectedError,
    OrderStateError,
    PositionStateError,
)
from execution.utils import (
    abs_position_size,
    base_execution_payload,
    binance_position_side,
    calculate_fill_ratio,
    calculate_notional,
    calculate_order_avg_price_from_payload,
    extract_client_order_id,
    extract_executed_quantity,
    extract_order_id,
    extract_order_price,
    extract_order_side,
    extract_order_status,
    extract_order_type,
    extract_original_quantity,
    extract_symbol,
    infer_position_side_from_amount,
    merge_metadata,
    normalize_exchange,
    normalize_market_type,
    normalize_order_side,
    normalize_order_status,
    normalize_order_type,
    normalize_symbol,
    normalize_time_in_force,
    now_ts,
    order_side_for_intent,
    require_non_negative_number,
    require_positive_number,
    safe_float,
    safe_int,
    validate_close_position_order,
    validate_reduce_only_order,
)
from risk.enums import MarginMode, OrderIntent, PositionSide, RiskMode, TradeTier
from risk.models import RiskDecision


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _get_decision_value(decision: RiskDecision, name: str, default: Any = None) -> Any:
    return getattr(decision, name, default)


def _get_nested_value(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    return getattr(obj, name, default)


# ---------------------------------------------------------------------
# Execution intent / plan models
# ---------------------------------------------------------------------


@dataclass(slots=True)
class ExecutionIntent:
    """
    Risk-approved execution intent.

    This is the main handoff model from RiskManager to execution.

    ExecutionIntent must be built from signal.confirmed / RiskDecision payload.
    It must not recalculate risk, size, leverage or tier.
    """

    symbol: str
    side: PositionSide
    order_intent: OrderIntent

    final_size: float
    final_leverage: float
    final_tier: TradeTier | None
    final_risk_amount: float
    final_margin: float
    final_notional: float

    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None

    signal_id: str | None = None
    strategy_name: str | None = None

    reservation_id: str | None = None
    reservation_expires_at: float | None = None

    risk_mode: RiskMode = RiskMode.NORMAL
    margin_mode: MarginMode = MarginMode.ISOLATED

    exchange: str = "binance"
    market_type: str = "usdm_futures"

    reduce_only: bool = False
    close_position: bool = False

    execution_id: str = field(default_factory=lambda: _new_id("exec"))
    created_at: float = field(default_factory=now_ts)

    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_risk_decision(
        cls,
        decision: RiskDecision,
        *,
        exchange: str = "binance",
        market_type: str = "usdm_futures",
        metadata: Mapping[str, Any] | None = None,
    ) -> "ExecutionIntent":
        """
        Build ExecutionIntent from risk-approved RiskDecision.

        The method intentionally uses getattr fallbacks because RiskDecision may
        evolve while preserving the same semantic contract.
        """
        request = _get_decision_value(decision, "request", None)

        symbol = (
            _get_decision_value(decision, "symbol", None)
            or _get_nested_value(request, "symbol", None)
        )
        side = (
            _get_decision_value(decision, "side", None)
            or _get_nested_value(request, "side", None)
        )
        order_intent = (
            _get_decision_value(decision, "order_intent", None)
            or _get_nested_value(request, "order_intent", OrderIntent.OPEN)
        )

        final_size = _get_decision_value(decision, "final_size", None)
        final_leverage = _get_decision_value(decision, "final_leverage", None)
        final_tier = _get_decision_value(decision, "final_tier", None)
        final_risk_amount = _get_decision_value(decision, "final_risk_amount", 0.0)
        final_margin = _get_decision_value(decision, "final_margin", 0.0)
        final_notional = _get_decision_value(decision, "final_notional", 0.0)

        intent = cls(
            symbol=symbol,
            side=side,
            order_intent=order_intent,
            final_size=final_size,
            final_leverage=final_leverage,
            final_tier=final_tier,
            final_risk_amount=final_risk_amount,
            final_margin=final_margin,
            final_notional=final_notional,
            entry_price=(
                _get_decision_value(decision, "entry_price", None)
                or _get_nested_value(request, "entry_price", None)
            ),
            stop_loss=(
                _get_decision_value(decision, "stop_loss", None)
                or _get_nested_value(request, "stop_loss", None)
            ),
            take_profit=(
                _get_decision_value(decision, "take_profit", None)
                or _get_nested_value(request, "take_profit", None)
            ),
            signal_id=(
                _get_decision_value(decision, "signal_id", None)
                or _get_nested_value(request, "signal_id", None)
            ),
            strategy_name=(
                _get_decision_value(decision, "strategy_name", None)
                or _get_nested_value(request, "strategy_name", None)
            ),
            reservation_id=_get_decision_value(decision, "reservation_id", None),
            reservation_expires_at=_get_decision_value(
                decision,
                "reservation_expires_at",
                None,
            ),
            risk_mode=_get_decision_value(decision, "risk_mode", RiskMode.NORMAL),
            margin_mode=(
                _get_decision_value(decision, "margin_mode", None)
                or _get_nested_value(request, "margin_mode", MarginMode.ISOLATED)
            ),
            exchange=exchange,
            market_type=market_type,
            reduce_only=bool(
                _get_decision_value(decision, "reduce_only", False)
                or _get_nested_value(request, "reduce_only", False)
                or getattr(order_intent, "reduces_risk", False)
            ),
            close_position=bool(_get_decision_value(decision, "close_position", False)),
            metadata=merge_metadata(
                _get_decision_value(decision, "metadata", None),
                metadata,
            ),
        )
        intent.validate()
        return intent

    @property
    def is_reduce_only(self) -> bool:
        return bool(self.reduce_only or getattr(self.order_intent, "reduces_risk", False))

    @property
    def increases_risk(self) -> bool:
        return bool(getattr(self.order_intent, "increases_risk", False))

    @property
    def reduces_risk(self) -> bool:
        return bool(getattr(self.order_intent, "reduces_risk", False))

    @property
    def is_reservation_expired(self) -> bool:
        if self.reservation_expires_at is None:
            return False
        return now_ts() >= self.reservation_expires_at

    @property
    def order_side(self) -> OrderSide:
        return order_side_for_intent(
            position_side=self.side,
            order_intent=self.order_intent,
        )

    @property
    def binance_position_side(self) -> str | None:
        return binance_position_side(self.side)

    def validate(self) -> None:
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.symbol = normalize_symbol(self.symbol)

        if self.exchange != "binance":
            raise ExecutionRejectedError("Only Binance execution is supported currently")

        if self.market_type != "usdm_futures":
            raise ExecutionRejectedError("Only Binance USD-M Futures execution is supported currently")

        self.final_size = require_positive_number(self.final_size, "final_size")
        self.final_leverage = require_positive_number(self.final_leverage, "final_leverage")
        self.final_risk_amount = require_non_negative_number(
            self.final_risk_amount,
            "final_risk_amount",
        )
        self.final_margin = require_non_negative_number(self.final_margin, "final_margin")
        self.final_notional = require_non_negative_number(
            self.final_notional,
            "final_notional",
        )

        if self.entry_price is not None:
            self.entry_price = require_positive_number(self.entry_price, "entry_price")

        if self.stop_loss is not None:
            self.stop_loss = require_positive_number(self.stop_loss, "stop_loss")

        if self.take_profit is not None:
            self.take_profit = require_positive_number(self.take_profit, "take_profit")

        validate_reduce_only_order(
            order_intent=self.order_intent,
            reduce_only=self.reduce_only,
            close_position=self.close_position,
            trigger_type=None,
        )

    def to_event_payload(self) -> dict[str, Any]:
        return {
            **base_execution_payload(
                exchange=self.exchange,
                market_type=self.market_type,
                symbol=self.symbol,
                signal_id=self.signal_id,
                strategy_name=self.strategy_name,
                reservation_id=self.reservation_id,
                metadata=self.metadata,
            ),
            "execution_id": self.execution_id,
            "side": self.side.value,
            "order_side": self.order_side.value,
            "order_intent": self.order_intent.value,
            "risk_mode": self.risk_mode.value,
            "margin_mode": self.margin_mode.value,
            "final_size": self.final_size,
            "final_leverage": self.final_leverage,
            "final_tier": self.final_tier.value if self.final_tier else None,
            "final_risk_amount": self.final_risk_amount,
            "final_margin": self.final_margin,
            "final_notional": self.final_notional,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "reduce_only": self.reduce_only,
            "close_position": self.close_position,
            "reservation_expires_at": self.reservation_expires_at,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class ExecutionLeg:
    """
    One executable leg inside an ExecutionPlan.

    SmartExecution may produce one or multiple legs. OrderManager receives each
    leg as OrderRequest.
    """

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float | None = None

    price: float | None = None
    stop_price: float | None = None

    position_side: PositionSide | None = None
    time_in_force: TimeInForce | None = None

    reduce_only: bool = False
    close_position: bool = False

    activation_price: float | None = None
    callback_rate: float | None = None
    working_type: WorkingType | None = None
    price_protect: bool | None = None

    trigger_type: TriggerType = TriggerType.NONE

    client_order_id: str | None = None
    sequence: int = 0
    leg_id: str = field(default_factory=lambda: _new_id("leg"))

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.symbol = normalize_symbol(self.symbol)
        self.side = normalize_order_side(self.side)
        self.order_type = normalize_order_type(self.order_type)

        if self.quantity is not None:
            self.quantity = require_positive_number(self.quantity, "quantity")

        if self.price is not None:
            self.price = require_positive_number(self.price, "price")

        if self.stop_price is not None:
            self.stop_price = require_positive_number(self.stop_price, "stop_price")

        if self.activation_price is not None:
            self.activation_price = require_positive_number(
                self.activation_price,
                "activation_price",
            )

        if self.callback_rate is not None:
            self.callback_rate = require_positive_number(
                self.callback_rate,
                "callback_rate",
            )

        if self.order_type.requires_price and self.price is None:
            raise ExecutionPlanValidationError(
                f"price is required for order type {self.order_type.value}"
            )

        if self.order_type.requires_stop_price and self.stop_price is None:
            raise ExecutionPlanValidationError(
                f"stop_price is required for order type {self.order_type.value}"
            )

        if self.order_type is OrderType.TRAILING_STOP_MARKET and self.callback_rate is None:
            raise ExecutionPlanValidationError(
                "callback_rate is required for TRAILING_STOP_MARKET"
            )

        validate_close_position_order(
            order_type=self.order_type,
            quantity=self.quantity,
            close_position=self.close_position,
        )

        validate_reduce_only_order(
            order_intent=None,
            reduce_only=self.reduce_only,
            close_position=self.close_position,
            trigger_type=self.trigger_type.value,
        )

    def to_order_request(
        self,
        *,
        execution_id: str,
        signal_id: str | None = None,
        strategy_name: str | None = None,
        reservation_id: str | None = None,
        exchange: str = "binance",
        market_type: str = "usdm_futures",
    ) -> "OrderRequest":
        self.validate()
        return OrderRequest(
            execution_id=execution_id,
            leg_id=self.leg_id,
            exchange=exchange,
            market_type=market_type,
            symbol=self.symbol,
            side=self.side,
            order_type=self.order_type,
            quantity=self.quantity,
            price=self.price,
            position_side=self.position_side,
            time_in_force=self.time_in_force,
            reduce_only=self.reduce_only,
            close_position=self.close_position,
            client_order_id=self.client_order_id,
            stop_price=self.stop_price,
            activation_price=self.activation_price,
            callback_rate=self.callback_rate,
            working_type=self.working_type,
            price_protect=self.price_protect,
            trigger_type=self.trigger_type,
            signal_id=signal_id,
            strategy_name=strategy_name,
            reservation_id=reservation_id,
            metadata=dict(self.metadata),
        )


@dataclass(slots=True)
class ExecutionPlan:
    """
    Technical execution plan produced by SmartExecution.

    It does not approve risk. It only describes how to execute an already
    risk-approved ExecutionIntent.
    """

    intent: ExecutionIntent
    mode: ExecutionMode
    legs: list[ExecutionLeg]

    status: ExecutionStatus = ExecutionStatus.PLANNED
    plan_id: str = field(default_factory=lambda: _new_id("plan"))

    created_at: float = field(default_factory=now_ts)
    expires_at: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def execution_id(self) -> str:
        return self.intent.execution_id

    @property
    def symbol(self) -> str:
        return self.intent.symbol

    @property
    def total_quantity(self) -> float:
        return sum(leg.quantity or 0.0 for leg in self.legs)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return now_ts() >= self.expires_at

    def validate(self) -> None:
        self.intent.validate()

        if not self.legs:
            raise ExecutionPlanValidationError("ExecutionPlan must contain at least one leg")

        for index, leg in enumerate(self.legs):
            leg.sequence = index
            leg.validate()

            if leg.symbol != self.intent.symbol:
                raise ExecutionPlanValidationError(
                    f"Leg symbol {leg.symbol} does not match intent symbol {self.intent.symbol}"
                )

        if not self.intent.close_position and self.total_quantity <= 0:
            raise ExecutionPlanValidationError("ExecutionPlan total quantity must be > 0")

        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ExecutionPlanValidationError("ExecutionPlan expires_at must be after created_at")

    def to_order_requests(self) -> list["OrderRequest"]:
        """
        Convert plan legs into OrderRequest objects.

        Important:
        Intent-level risk-approved fields must be propagated into every
        OrderRequest metadata, because OrderManager -> OrderResult ->
        execution.order_filled -> PositionManager depends on this metadata
        to build position.opened payload with stop_loss / take_profit /
        leverage / margin / risk amount.

        Without this propagation SLTPManager receives position.opened without
        protective levels and therefore does not place SL/TP orders.
        """
        self.validate()

        requests: list[OrderRequest] = []

        intent_metadata = {
            "execution_id": self.intent.execution_id,
            "signal_id": self.intent.signal_id,
            "strategy_name": self.intent.strategy_name,
            "reservation_id": self.intent.reservation_id,
            "side": self.intent.side.value,
            "order_intent": self.intent.order_intent.value,
            "risk_mode": self.intent.risk_mode.value,
            "margin_mode": self.intent.margin_mode.value,
            "final_size": self.intent.final_size,
            "final_leverage": self.intent.final_leverage,
            "final_tier": self.intent.final_tier.value if self.intent.final_tier else None,
            "final_risk_amount": self.intent.final_risk_amount,
            "final_margin": self.intent.final_margin,
            "final_notional": self.intent.final_notional,
            "entry_price": self.intent.entry_price,
            "stop_loss": self.intent.stop_loss,
            "take_profit": self.intent.take_profit,
            "reduce_only": self.intent.reduce_only,
            "close_position": self.intent.close_position,
            "reservation_expires_at": self.intent.reservation_expires_at,
            "plan_id": self.plan_id,
            "execution_mode": self.mode.value,
        }

        for leg in self.legs:
            request = leg.to_order_request(
                execution_id=self.intent.execution_id,
                signal_id=self.intent.signal_id,
                strategy_name=self.intent.strategy_name,
                reservation_id=self.intent.reservation_id,
                exchange=self.intent.exchange,
                market_type=self.intent.market_type,
            )

            request.metadata = merge_metadata(
                self.intent.metadata,
                intent_metadata,
                self.metadata,
                leg.metadata,
                request.metadata,
            )

            requests.append(request)

        return requests

    def to_event_payload(self) -> dict[str, Any]:
        return {
            **self.intent.to_event_payload(),
            "plan_id": self.plan_id,
            "execution_mode": self.mode.value,
            "execution_status": self.status.value,
            "legs_count": len(self.legs),
            "total_quantity": self.total_quantity,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "metadata": merge_metadata(self.intent.metadata, self.metadata),
        }


# ---------------------------------------------------------------------
# Order models
# ---------------------------------------------------------------------


@dataclass(slots=True)
class OrderRequest:
    """
    Internal execution order request.

    OrderManager maps this model into BinanceRestClient.create_order(...) params.
    """

    execution_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType

    quantity: float | None = None
    price: float | None = None

    exchange: str = "binance"
    market_type: str = "usdm_futures"

    position_side: PositionSide | None = None
    time_in_force: TimeInForce | None = None

    reduce_only: bool = False
    close_position: bool = False

    client_order_id: str | None = None
    stop_price: float | None = None
    activation_price: float | None = None
    callback_rate: float | None = None
    working_type: WorkingType | None = None
    price_protect: bool | None = None

    new_order_resp_type: str = "RESULT"

    leg_id: str | None = None
    signal_id: str | None = None
    strategy_name: str | None = None
    reservation_id: str | None = None

    trigger_type: TriggerType = TriggerType.NONE

    request_id: str = field(default_factory=lambda: _new_id("order_req"))
    created_at: float = field(default_factory=now_ts)

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.symbol = normalize_symbol(self.symbol)
        self.side = normalize_order_side(self.side)
        self.order_type = normalize_order_type(self.order_type)
        self.time_in_force = normalize_time_in_force(self.time_in_force)

        if self.quantity is not None:
            self.quantity = require_positive_number(self.quantity, "quantity")

        if self.price is not None:
            self.price = require_positive_number(self.price, "price")

        if self.stop_price is not None:
            self.stop_price = require_positive_number(self.stop_price, "stop_price")

        if self.activation_price is not None:
            self.activation_price = require_positive_number(
                self.activation_price,
                "activation_price",
            )

        if self.callback_rate is not None:
            self.callback_rate = require_positive_number(
                self.callback_rate,
                "callback_rate",
            )

        if self.order_type.requires_price and self.price is None:
            raise ExecutionRejectedError(f"price is required for {self.order_type.value}")

        if self.order_type.requires_stop_price and self.stop_price is None:
            raise ExecutionRejectedError(f"stop_price is required for {self.order_type.value}")

        if self.order_type is OrderType.TRAILING_STOP_MARKET and self.callback_rate is None:
            raise ExecutionRejectedError("callback_rate is required for TRAILING_STOP_MARKET")

        validate_close_position_order(
            order_type=self.order_type,
            quantity=self.quantity,
            close_position=self.close_position,
        )

        validate_reduce_only_order(
            order_intent=None,
            reduce_only=self.reduce_only,
            close_position=self.close_position,
            trigger_type=self.trigger_type.value,
        )

    def to_binance_params(self) -> dict[str, Any]:
        """
        Convert to BinanceRestClient.create_order(...) kwargs.
        """
        self.validate()

        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "position_side": binance_position_side(self.position_side),
            "time_in_force": self.time_in_force.value if self.time_in_force else None,
            "reduce_only": self.reduce_only if self.reduce_only else None,
            "new_client_order_id": self.client_order_id,
            "stop_price": self.stop_price,
            "close_position": self.close_position if self.close_position else None,
            "activation_price": self.activation_price,
            "callback_rate": self.callback_rate,
            "working_type": self.working_type.value if self.working_type else None,
            "price_protect": self.price_protect,
            "new_order_resp_type": self.new_order_resp_type,
        }

    def to_event_payload(self) -> dict[str, Any]:
        return {
            **base_execution_payload(
                exchange=self.exchange,
                market_type=self.market_type,
                symbol=self.symbol,
                signal_id=self.signal_id,
                strategy_name=self.strategy_name,
                reservation_id=self.reservation_id,
                metadata=self.metadata,
            ),
            "request_id": self.request_id,
            "execution_id": self.execution_id,
            "leg_id": self.leg_id,
            "client_order_id": self.client_order_id,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "position_side": self.position_side.value if self.position_side else None,
            "time_in_force": self.time_in_force.value if self.time_in_force else None,
            "reduce_only": self.reduce_only,
            "close_position": self.close_position,
            "stop_price": self.stop_price,
            "activation_price": self.activation_price,
            "callback_rate": self.callback_rate,
            "working_type": self.working_type.value if self.working_type else None,
            "price_protect": self.price_protect,
            "trigger_type": self.trigger_type.value,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class OrderResult:
    """
    Result of submitting/cancelling/fetching an order.
    """

    symbol: str
    status: OrderStatus

    exchange: str = "binance"
    market_type: str = "usdm_futures"

    order_id: str | None = None
    client_order_id: str | None = None

    side: OrderSide | None = None
    order_type: OrderType | None = None

    price: float | None = None
    avg_price: float | None = None
    original_quantity: float = 0.0
    executed_quantity: float = 0.0
    cumulative_quote_quantity: float = 0.0

    position_side: str | None = None
    reduce_only: bool | None = None
    close_position: bool | None = None

    stop_price: float | None = None
    working_type: str | None = None

    update_time: int | None = None
    exchange_time: int | None = None

    request_id: str | None = None
    execution_id: str | None = None
    leg_id: str | None = None
    signal_id: str | None = None
    strategy_name: str | None = None
    reservation_id: str | None = None

    raw: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    received_at: float = field(default_factory=now_ts)

    @classmethod
    def from_exchange_order(
        cls,
        payload: Mapping[str, Any],
        *,
        request: OrderRequest | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "OrderResult":
        symbol = extract_symbol(payload)

        result = cls(
            exchange=normalize_exchange(str(payload.get("exchange") or "binance")),
            market_type=normalize_market_type(str(payload.get("market_type") or "usdm_futures")),
            symbol=symbol,
            order_id=extract_order_id(payload),
            client_order_id=extract_client_order_id(payload),
            status=extract_order_status(payload),
            side=extract_order_side(payload),
            order_type=extract_order_type(payload),
            price=extract_order_price(payload),
            avg_price=calculate_order_avg_price_from_payload(payload),
            original_quantity=extract_original_quantity(payload),
            executed_quantity=extract_executed_quantity(payload),
            cumulative_quote_quantity=(
                safe_float(payload.get("cum_quote"))
                or safe_float(payload.get("cumulative_quote_qty"))
                or 0.0
            ),
            position_side=payload.get("position_side") or payload.get("positionSide"),
            reduce_only=payload.get("reduce_only") if payload.get("reduce_only") is not None else payload.get("reduceOnly"),
            close_position=(
                payload.get("close_position")
                if payload.get("close_position") is not None
                else payload.get("closePosition")
            ),
            stop_price=safe_float(payload.get("stop_price") or payload.get("stopPrice")),
            working_type=payload.get("working_type") or payload.get("workingType"),
            update_time=safe_int(payload.get("update_time") or payload.get("updateTime")),
            exchange_time=safe_int(payload.get("time")),
            request_id=request.request_id if request else None,
            execution_id=request.execution_id if request else None,
            leg_id=request.leg_id if request else None,
            signal_id=request.signal_id if request else None,
            strategy_name=request.strategy_name if request else None,
            reservation_id=request.reservation_id if request else None,
            raw=dict(payload),
            metadata=merge_metadata(request.metadata if request else None, metadata),
        )
        result.validate()
        return result

    @property
    def fill_ratio(self) -> float:
        return calculate_fill_ratio(
            executed_qty=self.executed_quantity,
            original_qty=self.original_quantity,
        )

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def is_filled(self) -> bool:
        return self.status is OrderStatus.FILLED

    def validate(self) -> None:
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.symbol = normalize_symbol(self.symbol)
        self.status = normalize_order_status(self.status)

        self.original_quantity = require_non_negative_number(
            self.original_quantity,
            "original_quantity",
        )
        self.executed_quantity = require_non_negative_number(
            self.executed_quantity,
            "executed_quantity",
        )
        self.cumulative_quote_quantity = require_non_negative_number(
            self.cumulative_quote_quantity,
            "cumulative_quote_quantity",
        )

        if self.executed_quantity - self.original_quantity > 1e-12 and self.original_quantity > 0:
            raise OrderStateError("executed_quantity cannot exceed original_quantity")

    def to_event_payload(self) -> dict[str, Any]:
        return {
            **base_execution_payload(
                exchange=self.exchange,
                market_type=self.market_type,
                symbol=self.symbol,
                signal_id=self.signal_id,
                strategy_name=self.strategy_name,
                reservation_id=self.reservation_id,
                metadata=self.metadata,
            ),
            "request_id": self.request_id,
            "execution_id": self.execution_id,
            "leg_id": self.leg_id,
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "status": self.status.value,
            "side": self.side.value if self.side else None,
            "order_type": self.order_type.value if self.order_type else None,
            "price": self.price,
            "avg_price": self.avg_price,
            "original_quantity": self.original_quantity,
            "executed_quantity": self.executed_quantity,
            "cumulative_quote_quantity": self.cumulative_quote_quantity,
            "fill_ratio": self.fill_ratio,
            "position_side": self.position_side,
            "reduce_only": self.reduce_only,
            "close_position": self.close_position,
            "stop_price": self.stop_price,
            "working_type": self.working_type,
            "update_time": self.update_time,
            "exchange_time": self.exchange_time,
            "received_at": self.received_at,
        }


@dataclass(slots=True)
class OrderUpdate:
    """
    Normalized order update from exchange REST/WS or reconciliation.
    """

    result: OrderResult
    previous_status: OrderStatus | None = None
    update_reason: str | None = None

    update_id: str = field(default_factory=lambda: _new_id("order_update"))
    received_at: float = field(default_factory=now_ts)

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> OrderStatus:
        return self.result.status

    @property
    def is_status_changed(self) -> bool:
        return self.previous_status is not None and self.previous_status is not self.status

    def to_event_payload(self) -> dict[str, Any]:
        payload = self.result.to_event_payload()
        payload.update(
            {
                "update_id": self.update_id,
                "previous_status": self.previous_status.value if self.previous_status else None,
                "update_reason": self.update_reason,
                "status_changed": self.is_status_changed,
                "received_at": self.received_at,
                "metadata": merge_metadata(self.result.metadata, self.metadata),
            }
        )
        return payload


@dataclass(slots=True)
class OrderFill:
    """
    Normalized fill event.

    OrderManager publishes execution.order_filled /
    execution.order_partially_filled using this model.
    """

    symbol: str
    side: OrderSide
    quantity: float
    price: float

    exchange: str = "binance"
    market_type: str = "usdm_futures"

    order_id: str | None = None
    client_order_id: str | None = None
    trade_id: str | None = None

    position_side: PositionSide | None = None

    quote_quantity: float | None = None
    commission: float | None = None
    commission_asset: str | None = None
    realized_pnl: float | None = None

    maker: bool | None = None

    execution_id: str | None = None
    signal_id: str | None = None
    strategy_name: str | None = None
    reservation_id: str | None = None

    fill_time: int | None = None
    fill_id: str = field(default_factory=lambda: _new_id("fill"))

    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_user_trade(
        cls,
        payload: Mapping[str, Any],
        *,
        execution_id: str | None = None,
        signal_id: str | None = None,
        strategy_name: str | None = None,
        reservation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "OrderFill":
        side = payload.get("side")
        symbol = payload.get("symbol")

        fill = cls(
            exchange=normalize_exchange(str(payload.get("exchange") or "binance")),
            market_type=normalize_market_type(str(payload.get("market_type") or "usdm_futures")),
            symbol=normalize_symbol(str(symbol)),
            side=normalize_order_side(str(side)),
            quantity=require_positive_number(payload.get("qty"), "qty"),
            price=require_positive_number(payload.get("price"), "price"),
            order_id=str(payload.get("order_id") or payload.get("orderId") or ""),
            client_order_id=payload.get("client_order_id") or payload.get("clientOrderId"),
            trade_id=str(payload.get("id")) if payload.get("id") is not None else None,
            position_side=_position_side_from_binance(payload.get("position_side") or payload.get("positionSide")),
            quote_quantity=safe_float(payload.get("quote_qty") or payload.get("quoteQty")),
            commission=safe_float(payload.get("commission")),
            commission_asset=payload.get("commission_asset") or payload.get("commissionAsset"),
            realized_pnl=safe_float(payload.get("realized_pnl") or payload.get("realizedPnl")),
            maker=payload.get("maker"),
            execution_id=execution_id,
            signal_id=signal_id,
            strategy_name=strategy_name,
            reservation_id=reservation_id,
            fill_time=safe_int(payload.get("time")),
            metadata=merge_metadata(metadata),
            raw=dict(payload),
        )
        fill.validate()
        return fill

    @property
    def notional(self) -> float:
        return self.quote_quantity if self.quote_quantity is not None else self.price * self.quantity

    def validate(self) -> None:
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.symbol = normalize_symbol(self.symbol)
        self.side = normalize_order_side(self.side)
        self.quantity = require_positive_number(self.quantity, "quantity")
        self.price = require_positive_number(self.price, "price")

        if self.quote_quantity is not None:
            self.quote_quantity = require_non_negative_number(
                self.quote_quantity,
                "quote_quantity",
            )

        if self.commission is not None:
            self.commission = require_non_negative_number(self.commission, "commission")

    def to_event_payload(self) -> dict[str, Any]:
        return {
            **base_execution_payload(
                exchange=self.exchange,
                market_type=self.market_type,
                symbol=self.symbol,
                signal_id=self.signal_id,
                strategy_name=self.strategy_name,
                reservation_id=self.reservation_id,
                metadata=self.metadata,
            ),
            "fill_id": self.fill_id,
            "execution_id": self.execution_id,
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "trade_id": self.trade_id,
            "side": self.side.value,
            "position_side": self.position_side.value if self.position_side else None,
            "quantity": self.quantity,
            "price": self.price,
            "notional": self.notional,
            "quote_quantity": self.quote_quantity,
            "commission": self.commission,
            "commission_asset": self.commission_asset,
            "realized_pnl": self.realized_pnl,
            "maker": self.maker,
            "fill_time": self.fill_time,
        }


@dataclass(slots=True)
class OrderState:
    """
    Runtime local order state owned by OrderManager.
    """

    symbol: str
    status: OrderStatus

    exchange: str = "binance"
    market_type: str = "usdm_futures"

    order_id: str | None = None
    client_order_id: str | None = None

    execution_id: str | None = None
    leg_id: str | None = None
    signal_id: str | None = None
    strategy_name: str | None = None
    reservation_id: str | None = None

    side: OrderSide | None = None
    order_type: OrderType | None = None

    original_quantity: float = 0.0
    executed_quantity: float = 0.0
    avg_price: float | None = None

    reduce_only: bool | None = None
    close_position: bool | None = None
    position_side: str | None = None

    created_at: float = field(default_factory=now_ts)
    updated_at: float = field(default_factory=now_ts)

    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(cls, result: OrderResult) -> "OrderState":
        return cls(
            exchange=result.exchange,
            market_type=result.market_type,
            symbol=result.symbol,
            order_id=result.order_id,
            client_order_id=result.client_order_id,
            execution_id=result.execution_id,
            leg_id=result.leg_id,
            signal_id=result.signal_id,
            strategy_name=result.strategy_name,
            reservation_id=result.reservation_id,
            status=result.status,
            side=result.side,
            order_type=result.order_type,
            original_quantity=result.original_quantity,
            executed_quantity=result.executed_quantity,
            avg_price=result.avg_price,
            reduce_only=result.reduce_only,
            close_position=result.close_position,
            position_side=result.position_side,
            updated_at=now_ts(),
            metadata=dict(result.metadata),
        )

    @property
    def is_open(self) -> bool:
        return self.status.is_open

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def fill_ratio(self) -> float:
        return calculate_fill_ratio(
            executed_qty=self.executed_quantity,
            original_qty=self.original_quantity,
        )

    def apply_result(self, result: OrderResult) -> None:
        if self.order_id and result.order_id and self.order_id != result.order_id:
            raise OrderStateError("Cannot apply result for a different order_id")

        if self.client_order_id and result.client_order_id and self.client_order_id != result.client_order_id:
            raise OrderStateError("Cannot apply result for a different client_order_id")

        if result.executed_quantity + 1e-12 < self.executed_quantity:
            raise OrderStateError("executed_quantity cannot decrease")

        self.status = result.status
        self.order_id = result.order_id or self.order_id
        self.client_order_id = result.client_order_id or self.client_order_id
        self.executed_quantity = result.executed_quantity
        self.original_quantity = result.original_quantity or self.original_quantity
        self.avg_price = result.avg_price or self.avg_price
        self.reduce_only = result.reduce_only
        self.close_position = result.close_position
        self.position_side = result.position_side
        self.updated_at = now_ts()
        self.metadata.update(result.metadata)

    def snapshot(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "execution_id": self.execution_id,
            "leg_id": self.leg_id,
            "signal_id": self.signal_id,
            "strategy_name": self.strategy_name,
            "reservation_id": self.reservation_id,
            "status": self.status.value,
            "side": self.side.value if self.side else None,
            "order_type": self.order_type.value if self.order_type else None,
            "original_quantity": self.original_quantity,
            "executed_quantity": self.executed_quantity,
            "fill_ratio": self.fill_ratio,
            "avg_price": self.avg_price,
            "reduce_only": self.reduce_only,
            "close_position": self.close_position,
            "position_side": self.position_side,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------
# Position models
# ---------------------------------------------------------------------


@dataclass(slots=True)
class PositionSnapshot:
    """
    Normalized exchange position snapshot.

    Built from BinanceRestClient.get_positions() normalized payload.
    """

    symbol: str
    size: float
    side: PositionSide | None

    exchange: str = "binance"
    market_type: str = "usdm_futures"

    entry_price: float | None = None
    break_even_price: float | None = None
    mark_price: float | None = None

    unrealized_pnl: float = 0.0
    liquidation_price: float | None = None

    leverage: float | None = None
    margin_type: str | None = None
    margin_used: float = 0.0
    notional_value: float = 0.0

    update_time: int | None = None
    snapshot_time: float = field(default_factory=now_ts)

    raw: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_exchange_position(
        cls,
        payload: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "PositionSnapshot":
        position_amt = safe_float(payload.get("position_amt") or payload.get("positionAmt"), 0.0) or 0.0
        side = _position_side_from_binance(payload.get("position_side") or payload.get("positionSide"))

        if side is None:
            side = infer_position_side_from_amount(position_amt)

        snapshot = cls(
            exchange=normalize_exchange(str(payload.get("exchange") or "binance")),
            market_type=normalize_market_type(str(payload.get("market_type") or "usdm_futures")),
            symbol=normalize_symbol(str(payload.get("symbol"))),
            size=abs_position_size(position_amt),
            side=side,
            entry_price=safe_float(payload.get("entry_price") or payload.get("entryPrice")),
            break_even_price=safe_float(payload.get("break_even_price") or payload.get("breakEvenPrice")),
            mark_price=safe_float(payload.get("mark_price") or payload.get("markPrice")),
            unrealized_pnl=safe_float(
                payload.get("unrealized_profit") or payload.get("unRealizedProfit"),
                0.0,
            ) or 0.0,
            liquidation_price=safe_float(
                payload.get("liquidation_price") or payload.get("liquidationPrice")
            ),
            leverage=safe_float(payload.get("leverage")),
            margin_type=payload.get("margin_type") or payload.get("marginType"),
            margin_used=(
                safe_float(payload.get("isolated_margin") or payload.get("isolatedMargin"), 0.0)
                or 0.0
            ),
            notional_value=abs(safe_float(payload.get("notional"), 0.0) or 0.0),
            update_time=safe_int(payload.get("update_time") or payload.get("updateTime")),
            raw=dict(payload),
            metadata=merge_metadata(metadata),
        )
        snapshot.validate()
        return snapshot

    @property
    def is_open(self) -> bool:
        return self.side is not None and self.size > 0

    def validate(self) -> None:
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.symbol = normalize_symbol(self.symbol)
        self.size = require_non_negative_number(self.size, "size")
        self.unrealized_pnl = require_non_negative_number(abs(self.unrealized_pnl), "abs_unrealized_pnl") * (
            -1 if self.unrealized_pnl < 0 else 1
        )
        self.margin_used = require_non_negative_number(self.margin_used, "margin_used")
        self.notional_value = require_non_negative_number(self.notional_value, "notional_value")

        if self.size > 0 and self.side is None:
            raise PositionStateError("Open position snapshot requires side")

    def to_event_payload(self) -> dict[str, Any]:
        return {
            **base_execution_payload(
                exchange=self.exchange,
                market_type=self.market_type,
                symbol=self.symbol,
                metadata=self.metadata,
            ),
            "side": self.side.value if self.side else None,
            "size": self.size,
            "entry_price": self.entry_price,
            "break_even_price": self.break_even_price,
            "mark_price": self.mark_price,
            "unrealized_pnl": self.unrealized_pnl,
            "liquidation_price": self.liquidation_price,
            "leverage": self.leverage,
            "margin_type": self.margin_type,
            "margin_used": self.margin_used,
            "notional_value": self.notional_value,
            "update_time": self.update_time,
            "snapshot_time": self.snapshot_time,
        }


@dataclass(slots=True)
class PositionUpdate:
    """
    Position lifecycle update emitted by PositionManager.
    """

    symbol: str
    side: PositionSide | None
    size: float

    exchange: str = "binance"
    market_type: str = "usdm_futures"

    previous_size: float = 0.0
    previous_side: PositionSide | None = None

    entry_price: float | None = None
    mark_price: float | None = None

    notional_value: float = 0.0
    leverage: float | None = None
    margin_used: float = 0.0
    risk_amount: float = 0.0

    stop_loss: float | None = None
    take_profit: float | None = None
    tier: TradeTier | None = None

    signal_id: str | None = None
    strategy_name: str | None = None
    reservation_id: str | None = None
    position_id: str | None = None

    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    update_type: str = "updated"
    update_id: str = field(default_factory=lambda: _new_id("position_update"))
    updated_at: float = field(default_factory=now_ts)

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_open_event(self) -> bool:
        return self.update_type == "opened"

    @property
    def is_close_event(self) -> bool:
        return self.update_type == "closed"

    def to_event_payload(self) -> dict[str, Any]:
        return {
            **base_execution_payload(
                exchange=self.exchange,
                market_type=self.market_type,
                symbol=self.symbol,
                signal_id=self.signal_id,
                strategy_name=self.strategy_name,
                reservation_id=self.reservation_id,
                metadata=self.metadata,
            ),
            "update_id": self.update_id,
            "update_type": self.update_type,
            "position_id": self.position_id,
            "side": self.side.value if self.side else None,
            "previous_side": self.previous_side.value if self.previous_side else None,
            "size": self.size,
            "previous_size": self.previous_size,
            "entry_price": self.entry_price,
            "mark_price": self.mark_price,
            "notional_value": self.notional_value,
            "leverage": self.leverage,
            "margin_used": self.margin_used,
            "risk_amount": self.risk_amount,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "tier": self.tier.value if self.tier else None,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class PositionState:
    """
    Runtime local position state owned by PositionManager.

    Payload snapshots from this model must be compatible with risk PortfolioPosition.
    """

    symbol: str
    side: PositionSide | None = None
    size: float = 0.0

    exchange: str = "binance"
    market_type: str = "usdm_futures"

    entry_price: float | None = None
    mark_price: float | None = None

    notional_value: float = 0.0
    leverage: float | None = None
    margin_used: float = 0.0
    risk_amount: float = 0.0

    stop_loss: float | None = None
    take_profit: float | None = None
    tier: TradeTier | None = None

    signal_id: str | None = None
    strategy_name: str | None = None
    reservation_id: str | None = None
    position_id: str = field(default_factory=lambda: _new_id("pos"))

    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    opened_at: float | None = None
    updated_at: float = field(default_factory=now_ts)
    closed_at: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.side is not None and self.size > 0

    @property
    def is_flat(self) -> bool:
        return not self.is_open

    def apply_snapshot(self, snapshot: PositionSnapshot) -> PositionUpdate:
        if normalize_symbol(snapshot.symbol) != normalize_symbol(self.symbol):
            raise PositionStateError("Cannot apply snapshot for a different symbol")

        previous_size = self.size
        previous_side = self.side

        self.side = snapshot.side
        self.size = snapshot.size
        self.entry_price = snapshot.entry_price or self.entry_price
        self.mark_price = snapshot.mark_price or self.mark_price
        self.notional_value = snapshot.notional_value
        self.leverage = snapshot.leverage or self.leverage
        self.margin_used = snapshot.margin_used
        self.unrealized_pnl = snapshot.unrealized_pnl
        self.updated_at = now_ts()
        self.metadata.update(snapshot.metadata)

        if previous_size <= 0 and self.size > 0:
            self.opened_at = self.opened_at or self.updated_at
            update_type = "opened"
        elif previous_size > 0 and self.size <= 0:
            self.closed_at = self.updated_at
            update_type = "closed"
        elif self.size < previous_size:
            update_type = "reduced"
        elif self.size > previous_size:
            update_type = "updated"
        else:
            update_type = "updated"

        return self._build_update(
            previous_size=previous_size,
            previous_side=previous_side,
            update_type=update_type,
        )

    def apply_fill(self, fill: OrderFill) -> PositionUpdate:
        if normalize_symbol(fill.symbol) != normalize_symbol(self.symbol):
            raise PositionStateError("Cannot apply fill for a different symbol")

        previous_size = self.size
        previous_side = self.side

        fill_side = fill.position_side
        if fill_side is None:
            fill_side = infer_position_side_from_amount(
                fill.quantity if fill.side is OrderSide.BUY else -fill.quantity
            )

        if fill_side is None:
            raise PositionStateError("Cannot infer position side from fill")

        if self.side is None or self.size <= 0:
            self.side = fill_side
            self.size = fill.quantity
            self.entry_price = fill.price
            self.opened_at = self.opened_at or now_ts()
            update_type = "opened"
        elif self.side is fill_side:
            total_qty = self.size + fill.quantity
            if self.entry_price is not None and total_qty > 0:
                self.entry_price = (
                    (self.entry_price * self.size) + (fill.price * fill.quantity)
                ) / total_qty
            self.size = total_qty
            update_type = "updated"
        else:
            if fill.quantity + 1e-12 < self.size:
                self.size -= fill.quantity
                update_type = "reduced"
            elif abs(fill.quantity - self.size) <= 1e-12:
                self.size = 0.0
                self.side = None
                self.closed_at = now_ts()
                update_type = "closed"
            else:
                self.side = fill_side
                self.size = fill.quantity - self.size
                self.entry_price = fill.price
                update_type = "reversed"

        self.mark_price = fill.price
        self.notional_value = calculate_notional(fill.price, self.size) if self.size > 0 else 0.0

        if fill.realized_pnl is not None:
            self.realized_pnl += fill.realized_pnl

        self.updated_at = now_ts()

        return self._build_update(
            previous_size=previous_size,
            previous_side=previous_side,
            update_type=update_type,
        )

    def _build_update(
        self,
        *,
        previous_size: float,
        previous_side: PositionSide | None,
        update_type: str,
    ) -> PositionUpdate:
        return PositionUpdate(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            side=self.side,
            size=self.size,
            previous_size=previous_size,
            previous_side=previous_side,
            entry_price=self.entry_price,
            mark_price=self.mark_price,
            notional_value=self.notional_value,
            leverage=self.leverage,
            margin_used=self.margin_used,
            risk_amount=self.risk_amount,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            tier=self.tier,
            signal_id=self.signal_id,
            strategy_name=self.strategy_name,
            reservation_id=self.reservation_id,
            position_id=self.position_id,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            update_type=update_type,
            metadata=dict(self.metadata),
        )

    def to_portfolio_position_payload(self) -> dict[str, Any]:
        """
        Payload shape expected by RiskManager/RiskState PortfolioPosition updates.
        """
        return {
            "position_id": self.position_id,
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "side": self.side.value if self.side else None,
            "size": self.size,
            "entry_price": self.entry_price,
            "mark_price": self.mark_price,
            "notional_value": self.notional_value,
            "leverage": self.leverage,
            "margin_used": self.margin_used,
            "risk_amount": self.risk_amount,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "tier": self.tier.value if self.tier else None,
            "strategy_name": self.strategy_name,
            "signal_id": self.signal_id,
            "reservation_id": self.reservation_id,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "opened_at": self.opened_at,
            "updated_at": self.updated_at,
            "closed_at": self.closed_at,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------
# SL/TP models
# ---------------------------------------------------------------------


@dataclass(slots=True)
class SLTPPlan:
    """
    Protective order plan for one position.
    """

    symbol: str
    position_side: PositionSide
    size: float

    exchange: str = "binance"
    market_type: str = "usdm_futures"

    stop_loss: float | None = None
    take_profit: float | None = None

    trailing_activation_price: float | None = None
    trailing_callback_rate: float | None = None

    working_type: WorkingType = WorkingType.MARK_PRICE
    price_protect: bool = True

    use_close_position_for_stop: bool = True
    use_close_position_for_take_profit: bool = False

    signal_id: str | None = None
    strategy_name: str | None = None
    reservation_id: str | None = None
    position_id: str | None = None

    plan_id: str = field(default_factory=lambda: _new_id("sltp_plan"))
    created_at: float = field(default_factory=now_ts)

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.symbol = normalize_symbol(self.symbol)
        self.size = require_positive_number(self.size, "size")

        if self.stop_loss is not None:
            self.stop_loss = require_positive_number(self.stop_loss, "stop_loss")

        if self.take_profit is not None:
            self.take_profit = require_positive_number(self.take_profit, "take_profit")

        if self.trailing_activation_price is not None:
            self.trailing_activation_price = require_positive_number(
                self.trailing_activation_price,
                "trailing_activation_price",
            )

        if self.trailing_callback_rate is not None:
            self.trailing_callback_rate = require_positive_number(
                self.trailing_callback_rate,
                "trailing_callback_rate",
            )

        if (
            self.stop_loss is None
            and self.take_profit is None
            and self.trailing_callback_rate is None
        ):
            raise ExecutionRejectedError("SLTPPlan must contain stop_loss, take_profit or trailing stop")

    def to_event_payload(self) -> dict[str, Any]:
        return {
            **base_execution_payload(
                exchange=self.exchange,
                market_type=self.market_type,
                symbol=self.symbol,
                signal_id=self.signal_id,
                strategy_name=self.strategy_name,
                reservation_id=self.reservation_id,
                metadata=self.metadata,
            ),
            "plan_id": self.plan_id,
            "position_id": self.position_id,
            "position_side": self.position_side.value,
            "size": self.size,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trailing_activation_price": self.trailing_activation_price,
            "trailing_callback_rate": self.trailing_callback_rate,
            "working_type": self.working_type.value,
            "price_protect": self.price_protect,
            "use_close_position_for_stop": self.use_close_position_for_stop,
            "use_close_position_for_take_profit": self.use_close_position_for_take_profit,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class ProtectiveOrderState:
    """
    Runtime state of one protective SL/TP/trailing order.
    """

    symbol: str
    sltp_type: SLTPType
    status: OrderStatus

    exchange: str = "binance"
    market_type: str = "usdm_futures"

    order_id: str | None = None
    client_order_id: str | None = None

    position_id: str | None = None
    position_side: PositionSide | None = None

    price: float | None = None
    stop_price: float | None = None
    quantity: float | None = None

    reduce_only: bool = True
    close_position: bool = False

    created_at: float = field(default_factory=now_ts)
    updated_at: float = field(default_factory=now_ts)

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status.is_open

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    def apply_order_result(self, result: OrderResult) -> None:
        self.order_id = result.order_id or self.order_id
        self.client_order_id = result.client_order_id or self.client_order_id
        self.status = result.status
        self.updated_at = now_ts()
        self.metadata.update(result.metadata)

    def snapshot(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "sltp_type": self.sltp_type.value,
            "status": self.status.value,
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "position_id": self.position_id,
            "position_side": self.position_side.value if self.position_side else None,
            "price": self.price,
            "stop_price": self.stop_price,
            "quantity": self.quantity,
            "reduce_only": self.reduce_only,
            "close_position": self.close_position,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------
# Stats models
# ---------------------------------------------------------------------


@dataclass(slots=True)
class ExecutionStats:
    accepted: int = 0
    rejected: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    kill_switch_rejections: int = 0

    active: int = 0
    peak_active: int = 0

    last_execution_id: str | None = None
    last_error: str | None = None
    updated_at: float = field(default_factory=now_ts)

    def register_started(self, execution_id: str | None = None) -> None:
        self.accepted += 1
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        self.last_execution_id = execution_id
        self.updated_at = now_ts()

    def register_completed(self, execution_id: str | None = None) -> None:
        self.completed += 1
        self.active = max(0, self.active - 1)
        self.last_execution_id = execution_id
        self.updated_at = now_ts()

    def register_failed(self, error: str | None = None, execution_id: str | None = None) -> None:
        self.failed += 1
        self.active = max(0, self.active - 1)
        self.last_error = error
        self.last_execution_id = execution_id
        self.updated_at = now_ts()

    def register_rejected(self, *, kill_switch: bool = False, error: str | None = None) -> None:
        self.rejected += 1
        if kill_switch:
            self.kill_switch_rejections += 1
        self.last_error = error
        self.updated_at = now_ts()

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OrderManagerStats:
    submitted: int = 0
    acknowledged: int = 0
    partially_filled: int = 0
    filled: int = 0
    rejected: int = 0
    failed: int = 0
    cancelled: int = 0
    expired: int = 0

    reconciliation_runs: int = 0
    reconciliation_failures: int = 0

    active_orders: int = 0
    peak_active_orders: int = 0

    last_order_id: str | None = None
    last_client_order_id: str | None = None
    last_error: str | None = None
    updated_at: float = field(default_factory=now_ts)

    def register_submit(self, result: OrderResult) -> None:
        self.submitted += 1
        self.active_orders += 1
        self.peak_active_orders = max(self.peak_active_orders, self.active_orders)
        self.last_order_id = result.order_id
        self.last_client_order_id = result.client_order_id
        self.updated_at = now_ts()

    def register_update(self, result: OrderResult) -> None:
        if result.status is OrderStatus.PARTIALLY_FILLED:
            self.partially_filled += 1
        elif result.status is OrderStatus.FILLED:
            self.filled += 1
            self.active_orders = max(0, self.active_orders - 1)
        elif result.status is OrderStatus.CANCELED:
            self.cancelled += 1
            self.active_orders = max(0, self.active_orders - 1)
        elif result.status is OrderStatus.REJECTED:
            self.rejected += 1
            self.active_orders = max(0, self.active_orders - 1)
        elif result.status in {OrderStatus.EXPIRED, OrderStatus.EXPIRED_IN_MATCH}:
            self.expired += 1
            self.active_orders = max(0, self.active_orders - 1)

        self.last_order_id = result.order_id
        self.last_client_order_id = result.client_order_id
        self.updated_at = now_ts()

    def register_failure(self, error: str | None = None) -> None:
        self.failed += 1
        self.last_error = error
        self.updated_at = now_ts()

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PositionManagerStats:
    opened: int = 0
    updated: int = 0
    reduced: int = 0
    closed: int = 0
    reversed: int = 0

    reconciliation_runs: int = 0
    reconciliation_failures: int = 0

    open_positions: int = 0
    peak_open_positions: int = 0

    last_position_id: str | None = None
    last_error: str | None = None
    updated_at: float = field(default_factory=now_ts)

    def register_update(self, update: PositionUpdate) -> None:
        if update.update_type == "opened":
            self.opened += 1
            self.open_positions += 1
            self.peak_open_positions = max(self.peak_open_positions, self.open_positions)
        elif update.update_type == "closed":
            self.closed += 1
            self.open_positions = max(0, self.open_positions - 1)
        elif update.update_type == "reduced":
            self.reduced += 1
        elif update.update_type == "reversed":
            self.reversed += 1
        else:
            self.updated += 1

        self.last_position_id = update.position_id
        self.updated_at = now_ts()

    def register_failure(self, error: str | None = None) -> None:
        self.last_error = error
        self.updated_at = now_ts()

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SLTPManagerStats:
    stop_loss_placed: int = 0
    take_profit_placed: int = 0
    trailing_stop_placed: int = 0

    updated: int = 0
    cancelled: int = 0
    triggered: int = 0
    failed: int = 0

    active_protective_orders: int = 0
    peak_active_protective_orders: int = 0

    last_order_id: str | None = None
    last_error: str | None = None
    updated_at: float = field(default_factory=now_ts)

    def register_placed(self, state: ProtectiveOrderState) -> None:
        if state.sltp_type is SLTPType.STOP_LOSS:
            self.stop_loss_placed += 1
        elif state.sltp_type is SLTPType.TAKE_PROFIT:
            self.take_profit_placed += 1
        elif state.sltp_type is SLTPType.TRAILING_STOP:
            self.trailing_stop_placed += 1

        self.active_protective_orders += 1
        self.peak_active_protective_orders = max(
            self.peak_active_protective_orders,
            self.active_protective_orders,
        )
        self.last_order_id = state.order_id
        self.updated_at = now_ts()

    def register_cancelled(self) -> None:
        self.cancelled += 1
        self.active_protective_orders = max(0, self.active_protective_orders - 1)
        self.updated_at = now_ts()

    def register_failure(self, error: str | None = None) -> None:
        self.failed += 1
        self.last_error = error
        self.updated_at = now_ts()

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SmartExecutionStats:
    plans_created: int = 0
    plans_failed: int = 0
    split_plans_created: int = 0
    market_plans_created: int = 0
    limit_plans_created: int = 0

    last_plan_id: str | None = None
    last_error: str | None = None
    updated_at: float = field(default_factory=now_ts)

    def register_plan(self, plan: ExecutionPlan) -> None:
        self.plans_created += 1
        self.last_plan_id = plan.plan_id

        if len(plan.legs) > 1:
            self.split_plans_created += 1

        if plan.mode is ExecutionMode.MARKET:
            self.market_plans_created += 1

        if plan.mode in {ExecutionMode.LIMIT, ExecutionMode.POST_ONLY}:
            self.limit_plans_created += 1

        self.updated_at = now_ts()

    def register_failure(self, error: str | None = None) -> None:
        self.plans_failed += 1
        self.last_error = error
        self.updated_at = now_ts()

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _position_side_from_binance(value: Any) -> PositionSide | None:
    if value is None:
        return None

    normalized = str(value).strip().upper()

    if normalized == "LONG":
        return PositionSide.LONG

    if normalized == "SHORT":
        return PositionSide.SHORT

    return None


__all__ = [
    "ExecutionIntent",
    "ExecutionLeg",
    "ExecutionPlan",
    "OrderRequest",
    "OrderResult",
    "OrderUpdate",
    "OrderFill",
    "OrderState",
    "PositionSnapshot",
    "PositionUpdate",
    "PositionState",
    "SLTPPlan",
    "ProtectiveOrderState",
    "ExecutionStats",
    "OrderManagerStats",
    "PositionManagerStats",
    "SLTPManagerStats",
    "SmartExecutionStats",
]