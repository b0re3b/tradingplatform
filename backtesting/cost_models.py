"""
Trading cost models for backtesting.

This module contains pure calculation helpers for simulated trading costs:
commissions, slippage, spread cost, funding payments, borrow costs, liquidation
price estimation and total cost breakdowns.

Important:
- No EventBus usage here.
- No strategy/risk/execution orchestration here.
- No live exchange calls here.
- This module only calculates costs for execution_simulator.py and
  position_simulator.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from backtesting.config import CostModelConfig
from backtesting.enums import CommissionModel, FundingSimulationMode, SlippageModel
from backtesting.exceptions import (
    CommissionCalculationError,
    FundingCostCalculationError,
    InvalidCostModelInputError,
    LiquidationPriceCalculationError,
    SlippageCalculationError,
    SpreadCostCalculationError,
)
from backtesting.models import (
    HistoricalCandle,
    HistoricalFundingRecord,
    HistoricalOrderBookSnapshot,
    SimulatedFill,
    TradingCostBreakdown,
    ensure_aware_utc,
    safe_div,
    timestamp_ms,
)


@dataclass(slots=True, frozen=True)
class CommissionInput:
    """
    Input for commission calculation.
    """

    notional: float
    is_maker: bool = False
    fee_bps: float | None = None
    fixed_fee: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class SlippageInput:
    """
    Input for slippage calculation.
    """

    side: str
    intended_price: float
    quantity: float
    reference_price: float | None = None
    candle: HistoricalCandle | None = None
    orderbook: HistoricalOrderBookSnapshot | None = None
    participation_pct: float | None = None
    volatility_pct: float | None = None
    spread: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class FundingInput:
    """
    Input for funding calculation.
    """

    side: str
    notional: float
    funding_rate: float
    timestamp_ms: int
    next_funding_time_ms: int | None = None
    holding_seconds: float | None = None
    funding_interval_hours: int = 8
    metadata: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class LiquidationInput:
    """
    Input for futures liquidation price estimation.
    """

    side: str
    entry_price: float
    quantity: float
    leverage: float
    margin: float | None = None
    maintenance_margin_rate: float = 0.005
    wallet_balance: float | None = None
    unrealized_pnl: float = 0.0
    buffer_bps: float = 0.0
    metadata: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class TradeCostInput:
    """
    Full trading cost input for one fill/order/trade leg.
    """

    side: str
    intended_price: float
    fill_price: float
    quantity: float
    notional: float | None = None
    is_maker: bool = False
    funding_rate: float | None = None
    holding_seconds: float | None = None
    candle: HistoricalCandle | None = None
    orderbook: HistoricalOrderBookSnapshot | None = None
    metadata: dict[str, Any] | None = None


class CommissionCalculator:
    """
    Commission / fee calculator.
    """

    def __init__(self, config: CostModelConfig | None = None) -> None:
        self.config = config or CostModelConfig()
        self.config.validate()

    def calculate(self, data: CommissionInput) -> float:
        """
        Calculate commission amount in quote currency.
        """

        if data.notional < 0:
            raise CommissionCalculationError(
                "CommissionInput.notional cannot be negative.",
                details={"notional": data.notional},
            )

        if data.notional == 0:
            return 0.0

        if not self.config.include_commissions:
            return 0.0

        if self.config.commission_model == CommissionModel.NONE:
            return 0.0

        if self.config.commission_model == CommissionModel.FIXED:
            return float(data.fixed_fee if data.fixed_fee is not None else self.config.fixed_fee)

        if self.config.commission_model == CommissionModel.PERCENTAGE:
            fee_bps = data.fee_bps if data.fee_bps is not None else self.config.default_fee_bps
            return self._bps_cost(data.notional, fee_bps)

        if self.config.commission_model == CommissionModel.MAKER_TAKER:
            fee_bps = self.config.maker_fee_bps if data.is_maker else self.config.taker_fee_bps
            if data.fee_bps is not None:
                fee_bps = data.fee_bps
            return self._bps_cost(data.notional, fee_bps)

        if self.config.commission_model == CommissionModel.EXCHANGE_SPECIFIC:
            # Binance USD-M futures-compatible default if no external fee table
            # is injected later.
            fee_bps = self.config.maker_fee_bps if data.is_maker else self.config.taker_fee_bps
            return self._bps_cost(data.notional, fee_bps)

        raise CommissionCalculationError(
            "Unsupported commission model.",
            details={"model": self.config.commission_model.value},
        )

    @staticmethod
    def _bps_cost(notional: float, bps: float) -> float:
        if bps < 0:
            raise CommissionCalculationError(
                "Fee bps cannot be negative.",
                details={"fee_bps": bps},
            )

        return abs(notional) * bps / 10_000.0


class SlippageCalculator:
    """
    Slippage calculator.

    The returned value is a signed price delta:
    - positive means worse price for buy / higher fill price;
    - negative means worse price for sell / lower fill price.

    For cost aggregation, use abs(delta * quantity).
    """

    def __init__(self, config: CostModelConfig | None = None) -> None:
        self.config = config or CostModelConfig()
        self.config.validate()

    def calculate_price_delta(self, data: SlippageInput) -> float:
        """
        Calculate signed slippage price delta.
        """

        self._validate(data)

        if not self.config.include_slippage:
            return 0.0

        if self.config.slippage_model == SlippageModel.NONE:
            return 0.0

        side_sign = self._side_sign(data.side)

        if self.config.slippage_model == SlippageModel.FIXED_BPS:
            delta = data.intended_price * self.config.fixed_slippage_bps / 10_000.0
            return side_sign * delta

        if self.config.slippage_model == SlippageModel.FIXED_PRICE:
            return side_sign * self.config.fixed_slippage_price

        if self.config.slippage_model == SlippageModel.PERCENT_OF_SPREAD:
            spread = self._resolve_spread(data)
            delta = spread * self.config.spread_slippage_fraction
            return side_sign * delta

        if self.config.slippage_model == SlippageModel.VOLUME_BASED:
            participation = data.participation_pct
            if participation is None:
                participation = self._estimate_participation_pct(data)

            delta_bps = self.config.fixed_slippage_bps + (
                max(0.0, participation) * self.config.volume_slippage_multiplier
            )
            return side_sign * data.intended_price * delta_bps / 10_000.0

        if self.config.slippage_model == SlippageModel.VOLATILITY_BASED:
            volatility_pct = data.volatility_pct
            if volatility_pct is None:
                volatility_pct = self._estimate_candle_volatility_pct(data.candle)

            delta = data.intended_price * volatility_pct * self.config.volatility_slippage_multiplier / 100.0
            return side_sign * delta

        if self.config.slippage_model == SlippageModel.ORDERBOOK_DEPTH:
            return self._orderbook_depth_slippage(data)

        if self.config.slippage_model == SlippageModel.ADVERSE_SELECTION:
            base_delta = data.intended_price * self.config.fixed_slippage_bps / 10_000.0
            adverse_delta = data.intended_price * self.config.adverse_selection_bps / 10_000.0
            return side_sign * (base_delta + adverse_delta)

        raise SlippageCalculationError(
            "Unsupported slippage model.",
            details={"model": self.config.slippage_model.value},
        )

    def calculate_cost(self, data: SlippageInput) -> float:
        """
        Calculate absolute slippage cost in quote currency.
        """

        delta = self.calculate_price_delta(data)
        return abs(delta * data.quantity)

    def apply_to_price(self, data: SlippageInput) -> float:
        """
        Return intended price adjusted by simulated slippage.
        """

        return data.intended_price + self.calculate_price_delta(data)

    def calculate_bps(self, data: SlippageInput) -> float:
        """
        Calculate absolute slippage in basis points.
        """

        delta = abs(self.calculate_price_delta(data))
        return safe_div(delta, data.intended_price) * 10_000.0

    def _validate(self, data: SlippageInput) -> None:
        if data.intended_price <= 0:
            raise SlippageCalculationError(
                "SlippageInput.intended_price must be positive.",
                details={"intended_price": data.intended_price},
            )

        if data.quantity <= 0:
            raise SlippageCalculationError(
                "SlippageInput.quantity must be positive.",
                details={"quantity": data.quantity},
            )

        if self._side_sign(data.side) == 0:
            raise SlippageCalculationError(
                "Unsupported order side for slippage calculation.",
                details={"side": data.side},
            )

    @staticmethod
    def _side_sign(side: str) -> int:
        value = side.lower()

        if value in {"buy", "long"}:
            return 1

        if value in {"sell", "short"}:
            return -1

        return 0

    def _resolve_spread(self, data: SlippageInput) -> float:
        if data.spread is not None:
            return max(0.0, data.spread)

        if data.orderbook is not None and data.orderbook.spread is not None:
            return max(0.0, data.orderbook.spread)

        if data.candle is not None:
            return max(0.0, data.candle.high - data.candle.low) * 0.05

        return data.intended_price * self.config.fixed_slippage_bps / 10_000.0

    @staticmethod
    def _estimate_candle_volatility_pct(candle: HistoricalCandle | None) -> float:
        if candle is None or candle.close <= 0:
            return 0.0

        return abs(candle.high - candle.low) / candle.close * 100.0

    def _estimate_participation_pct(self, data: SlippageInput) -> float:
        if data.candle is None or data.candle.volume <= 0:
            return 0.0

        return abs(data.quantity) / data.candle.volume * 100.0

    def _orderbook_depth_slippage(self, data: SlippageInput) -> float:
        if data.orderbook is None:
            # Conservative fallback.
            side_sign = self._side_sign(data.side)
            delta = data.intended_price * self.config.fixed_slippage_bps / 10_000.0
            return side_sign * delta

        side_sign = self._side_sign(data.side)
        levels = data.orderbook.asks if side_sign > 0 else data.orderbook.bids

        if not levels:
            raise SlippageCalculationError(
                "Orderbook has no levels for requested side.",
                details={
                    "side": data.side,
                    "symbol": data.orderbook.symbol,
                    "timestamp_ms": data.orderbook.timestamp_ms,
                },
            )

        remaining = data.quantity
        notional = 0.0
        filled_qty = 0.0

        sorted_levels = (
            sorted(levels, key=lambda level: level.price)
            if side_sign > 0
            else sorted(levels, key=lambda level: level.price, reverse=True)
        )

        for level in sorted_levels:
            if remaining <= 0:
                break

            fill_qty = min(remaining, level.quantity)
            notional += fill_qty * level.price
            filled_qty += fill_qty
            remaining -= fill_qty

        if filled_qty <= 0:
            raise SlippageCalculationError(
                "Orderbook depth produced zero simulated fill.",
                details={"side": data.side, "quantity": data.quantity},
            )

        average_price = notional / filled_qty
        delta = average_price - data.intended_price

        # If depth is insufficient, add adverse residual penalty.
        if remaining > 0:
            residual_penalty = data.intended_price * self.config.fixed_slippage_bps / 10_000.0
            delta += side_sign * residual_penalty

        return delta


class SpreadCostCalculator:
    """
    Spread cost calculator.
    """

    def __init__(self, config: CostModelConfig | None = None) -> None:
        self.config = config or CostModelConfig()
        self.config.validate()

    def calculate(
        self,
        *,
        quantity: float,
        mid_price: float | None = None,
        bid: float | None = None,
        ask: float | None = None,
        spread: float | None = None,
        orderbook: HistoricalOrderBookSnapshot | None = None,
    ) -> float:
        """
        Calculate approximate spread cost.

        Cost is half-spread * quantity by default.
        """

        if quantity <= 0:
            raise SpreadCostCalculationError(
                "quantity must be positive.",
                details={"quantity": quantity},
            )

        if not self.config.include_spread_cost:
            return 0.0

        resolved_spread = spread

        if resolved_spread is None and orderbook is not None:
            resolved_spread = orderbook.spread

        if resolved_spread is None and bid is not None and ask is not None:
            resolved_spread = ask - bid

        if resolved_spread is None:
            if mid_price is None or mid_price <= 0:
                return 0.0
            resolved_spread = mid_price * self.config.fixed_slippage_bps / 10_000.0

        if resolved_spread < 0:
            raise SpreadCostCalculationError(
                "Spread cannot be negative.",
                details={"spread": resolved_spread},
            )

        return resolved_spread * 0.5 * quantity


class FundingCostCalculator:
    """
    Funding payment calculator for perpetual futures.
    """

    def __init__(self, config: CostModelConfig | None = None) -> None:
        self.config = config or CostModelConfig()
        self.config.validate()

    def calculate(self, data: FundingInput) -> float:
        """
        Calculate signed funding cashflow.

        Return convention:
        - positive value means funding received;
        - negative value means funding paid.
        """

        if data.notional < 0:
            raise FundingCostCalculationError(
                "FundingInput.notional cannot be negative.",
                details={"notional": data.notional},
            )

        if not self.config.include_funding:
            return 0.0

        if self.config.funding_mode == FundingSimulationMode.DISABLED:
            return 0.0

        side_sign = self._side_sign(data.side)

        if side_sign == 0:
            raise FundingCostCalculationError(
                "Unsupported position side for funding calculation.",
                details={"side": data.side},
            )

        if self.config.funding_mode == FundingSimulationMode.APPLY_ON_FUNDING_TIMESTAMP:
            # Positive funding rate means longs pay shorts.
            return -side_sign * data.notional * data.funding_rate

        if self.config.funding_mode == FundingSimulationMode.PRORATED_CONTINUOUS:
            if data.holding_seconds is None:
                raise FundingCostCalculationError(
                    "holding_seconds is required for prorated continuous funding."
                )

            interval_seconds = data.funding_interval_hours * 60 * 60
            fraction = data.holding_seconds / interval_seconds
            return -side_sign * data.notional * data.funding_rate * fraction

        if self.config.funding_mode == FundingSimulationMode.ESTIMATED_FROM_RATE:
            holding_seconds = data.holding_seconds or 0.0
            interval_seconds = data.funding_interval_hours * 60 * 60
            fraction = max(1.0, holding_seconds / interval_seconds) if holding_seconds > 0 else 1.0
            return -side_sign * data.notional * data.funding_rate * fraction

        raise FundingCostCalculationError(
            "Unsupported funding simulation mode.",
            details={"mode": self.config.funding_mode.value},
        )

    def calculate_from_record(
        self,
        *,
        side: str,
        notional: float,
        funding: HistoricalFundingRecord,
        holding_seconds: float | None = None,
    ) -> float:
        """
        Calculate funding from HistoricalFundingRecord.
        """

        return self.calculate(
            FundingInput(
                side=side,
                notional=notional,
                funding_rate=funding.funding_rate,
                timestamp_ms=funding.timestamp_ms,
                next_funding_time_ms=funding.next_funding_time_ms,
                holding_seconds=holding_seconds,
                funding_interval_hours=self.config.funding_interval_hours,
            )
        )

    @staticmethod
    def _side_sign(side: str) -> int:
        value = side.lower()

        if value in {"buy", "long"}:
            return 1

        if value in {"sell", "short"}:
            return -1

        return 0


class BorrowCostCalculator:
    """
    Placeholder borrow cost calculator.

    For USD-M perpetual futures this usually stays zero. It is included for
    compatibility with future margin/spot backtests.
    """

    def __init__(self, config: CostModelConfig | None = None) -> None:
        self.config = config or CostModelConfig()
        self.config.validate()

    def calculate(
        self,
        *,
        notional: float,
        annual_rate: float = 0.0,
        holding_seconds: float = 0.0,
    ) -> float:
        if notional < 0:
            raise InvalidCostModelInputError(
                "notional cannot be negative.",
                details={"notional": notional},
            )

        if annual_rate < 0:
            raise InvalidCostModelInputError(
                "annual_rate cannot be negative.",
                details={"annual_rate": annual_rate},
            )

        if holding_seconds < 0:
            raise InvalidCostModelInputError(
                "holding_seconds cannot be negative.",
                details={"holding_seconds": holding_seconds},
            )

        seconds_per_year = 365 * 24 * 60 * 60
        return notional * annual_rate * holding_seconds / seconds_per_year


class LiquidationPriceCalculator:
    """
    Approximate futures liquidation price calculator.

    This is an estimation layer for backtesting. Real exchange liquidation
    logic may include tiered maintenance margin, wallet balance sharing,
    fee buffers and exchange-specific formulas.
    """

    def __init__(self, config: CostModelConfig | None = None) -> None:
        self.config = config or CostModelConfig()
        self.config.validate()

    def calculate(self, data: LiquidationInput) -> float:
        """
        Estimate liquidation price.

        For isolated-style approximation:

        long:
            liquidation ~= entry * (1 - 1/leverage + mmr)

        short:
            liquidation ~= entry * (1 + 1/leverage - mmr)

        Optional buffer_bps moves liquidation closer to entry for conservative
        simulation.
        """

        if data.entry_price <= 0:
            raise LiquidationPriceCalculationError(
                "entry_price must be positive.",
                details={"entry_price": data.entry_price},
            )

        if data.quantity <= 0:
            raise LiquidationPriceCalculationError(
                "quantity must be positive.",
                details={"quantity": data.quantity},
            )

        if data.leverage <= 0:
            raise LiquidationPriceCalculationError(
                "leverage must be positive.",
                details={"leverage": data.leverage},
            )

        if not 0 <= data.maintenance_margin_rate < 1:
            raise LiquidationPriceCalculationError(
                "maintenance_margin_rate must be in [0, 1).",
                details={"maintenance_margin_rate": data.maintenance_margin_rate},
            )

        side = data.side.lower()
        leverage_component = 1.0 / data.leverage
        buffer = data.entry_price * max(0.0, data.buffer_bps) / 10_000.0

        if side in {"buy", "long"}:
            price = data.entry_price * (1.0 - leverage_component + data.maintenance_margin_rate)
            price += buffer
            return max(0.0, price)

        if side in {"sell", "short"}:
            price = data.entry_price * (1.0 + leverage_component - data.maintenance_margin_rate)
            price -= buffer
            return max(0.0, price)

        raise LiquidationPriceCalculationError(
            "Unsupported side for liquidation price calculation.",
            details={"side": data.side},
        )

    def is_liquidated(
        self,
        *,
        side: str,
        mark_price: float,
        liquidation_price: float,
    ) -> bool:
        """
        Check whether mark price has crossed liquidation price.
        """

        if mark_price <= 0:
            raise LiquidationPriceCalculationError(
                "mark_price must be positive.",
                details={"mark_price": mark_price},
            )

        if liquidation_price <= 0:
            return False

        side_value = side.lower()

        if side_value in {"buy", "long"}:
            return mark_price <= liquidation_price

        if side_value in {"sell", "short"}:
            return mark_price >= liquidation_price

        raise LiquidationPriceCalculationError(
            "Unsupported side for liquidation check.",
            details={"side": side},
        )


class TradingCostModel:
    """
    Facade for all trading cost calculators.

    This class is used by execution_simulator.py and position_simulator.py
    to keep cost calculations centralized and consistent.
    """

    def __init__(self, config: CostModelConfig | None = None) -> None:
        self.config = config or CostModelConfig()
        self.config.validate()

        self.commissions = CommissionCalculator(self.config)
        self.slippage = SlippageCalculator(self.config)
        self.spread = SpreadCostCalculator(self.config)
        self.funding = FundingCostCalculator(self.config)
        self.borrow = BorrowCostCalculator(self.config)
        self.liquidation = LiquidationPriceCalculator(self.config)

    def calculate_fill_price(
        self,
        *,
        side: str,
        intended_price: float,
        quantity: float,
        candle: HistoricalCandle | None = None,
        orderbook: HistoricalOrderBookSnapshot | None = None,
        participation_pct: float | None = None,
        volatility_pct: float | None = None,
        spread: float | None = None,
    ) -> float:
        """
        Calculate simulated fill price after slippage.
        """

        return self.slippage.apply_to_price(
            SlippageInput(
                side=side,
                intended_price=intended_price,
                quantity=quantity,
                candle=candle,
                orderbook=orderbook,
                participation_pct=participation_pct,
                volatility_pct=volatility_pct,
                spread=spread,
            )
        )

    def calculate_trade_costs(
        self,
        data: TradeCostInput,
    ) -> TradingCostBreakdown:
        """
        Calculate full cost breakdown for a simulated trade leg.
        """

        self._validate_trade_input(data)

        notional = data.notional
        if notional is None:
            notional = abs(data.fill_price * data.quantity)

        commission = self.commissions.calculate(
            CommissionInput(
                notional=notional,
                is_maker=data.is_maker,
            )
        )

        slippage_cost = abs((data.fill_price - data.intended_price) * data.quantity)

        spread_cost = self.spread.calculate(
            quantity=data.quantity,
            mid_price=data.intended_price,
            orderbook=data.orderbook,
        )

        funding_paid = 0.0
        funding_received = 0.0

        if data.funding_rate is not None:
            funding_cashflow = self.funding.calculate(
                FundingInput(
                    side=data.side,
                    notional=notional,
                    funding_rate=data.funding_rate,
                    timestamp_ms=timestamp_ms(datetime.now()),
                    holding_seconds=data.holding_seconds,
                    funding_interval_hours=self.config.funding_interval_hours,
                )
            )

            if funding_cashflow >= 0:
                funding_received = funding_cashflow
            else:
                funding_paid = abs(funding_cashflow)

        return TradingCostBreakdown(
            commission=commission,
            slippage=slippage_cost,
            spread_cost=spread_cost,
            funding_paid=funding_paid,
            funding_received=funding_received,
        )

    def estimate_net_pnl(
        self,
        *,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        entry_costs: TradingCostBreakdown | None = None,
        exit_costs: TradingCostBreakdown | None = None,
    ) -> float:
        """
        Estimate net PnL after costs.
        """

        if entry_price <= 0 or exit_price <= 0:
            raise InvalidCostModelInputError(
                "entry_price and exit_price must be positive.",
                details={
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                },
            )

        if quantity <= 0:
            raise InvalidCostModelInputError(
                "quantity must be positive.",
                details={"quantity": quantity},
            )

        side_value = side.lower()

        if side_value in {"buy", "long"}:
            gross_pnl = (exit_price - entry_price) * quantity
        elif side_value in {"sell", "short"}:
            gross_pnl = (entry_price - exit_price) * quantity
        else:
            raise InvalidCostModelInputError(
                "Unsupported side for PnL estimation.",
                details={"side": side},
            )

        total_costs = 0.0

        if entry_costs is not None:
            total_costs += entry_costs.total_cost

        if exit_costs is not None:
            total_costs += exit_costs.total_cost

        return gross_pnl - total_costs

    def build_fill_cost_breakdown(
        self,
        *,
        fill: SimulatedFill,
        intended_price: float,
        is_maker: bool = False,
        candle: HistoricalCandle | None = None,
        orderbook: HistoricalOrderBookSnapshot | None = None,
    ) -> TradingCostBreakdown:
        """
        Build costs for an already created SimulatedFill.
        """

        return self.calculate_trade_costs(
            TradeCostInput(
                side=fill.side,
                intended_price=intended_price,
                fill_price=fill.price,
                quantity=fill.quantity,
                notional=fill.notional,
                is_maker=is_maker,
                candle=candle,
                orderbook=orderbook,
            )
        )

    def calculate_liquidation_price(
        self,
        *,
        side: str,
        entry_price: float,
        quantity: float,
        leverage: float,
        margin: float | None = None,
        maintenance_margin_rate: float = 0.005,
        buffer_bps: float | None = None,
    ) -> float:
        """
        Estimate liquidation price using configured calculator.
        """

        return self.liquidation.calculate(
            LiquidationInput(
                side=side,
                entry_price=entry_price,
                quantity=quantity,
                leverage=leverage,
                margin=margin,
                maintenance_margin_rate=maintenance_margin_rate,
                buffer_bps=self.config.liquidation_penalty_bps if buffer_bps is None else buffer_bps,
            )
        )

    def apply_funding_record(
        self,
        *,
        side: str,
        notional: float,
        funding: HistoricalFundingRecord,
        holding_seconds: float | None = None,
    ) -> TradingCostBreakdown:
        """
        Convert a funding record into a cost breakdown.
        """

        cashflow = self.funding.calculate_from_record(
            side=side,
            notional=notional,
            funding=funding,
            holding_seconds=holding_seconds,
        )

        if cashflow >= 0:
            return TradingCostBreakdown(funding_received=cashflow)

        return TradingCostBreakdown(funding_paid=abs(cashflow))

    def calculate_total_cost(
        self,
        breakdowns: list[TradingCostBreakdown],
    ) -> TradingCostBreakdown:
        """
        Aggregate multiple cost breakdowns.
        """

        result = TradingCostBreakdown()

        for item in breakdowns:
            result.commission += item.commission
            result.slippage += item.slippage
            result.spread_cost += item.spread_cost
            result.funding_paid += item.funding_paid
            result.funding_received += item.funding_received
            result.borrow_cost += item.borrow_cost
            result.liquidation_penalty += item.liquidation_penalty
            result.other_costs += item.other_costs

        return result

    @staticmethod
    def _validate_trade_input(data: TradeCostInput) -> None:
        if data.intended_price <= 0:
            raise InvalidCostModelInputError(
                "TradeCostInput.intended_price must be positive.",
                details={"intended_price": data.intended_price},
            )

        if data.fill_price <= 0:
            raise InvalidCostModelInputError(
                "TradeCostInput.fill_price must be positive.",
                details={"fill_price": data.fill_price},
            )

        if data.quantity <= 0:
            raise InvalidCostModelInputError(
                "TradeCostInput.quantity must be positive.",
                details={"quantity": data.quantity},
            )

        if data.notional is not None and data.notional < 0:
            raise InvalidCostModelInputError(
                "TradeCostInput.notional cannot be negative.",
                details={"notional": data.notional},
            )


# ============================================================================
# Convenience helpers
# ============================================================================


def calculate_slippage_bps(
    *,
    intended_price: float,
    fill_price: float,
) -> float:
    """
    Calculate absolute slippage in bps.
    """

    if intended_price <= 0:
        raise SlippageCalculationError(
            "intended_price must be positive.",
            details={"intended_price": intended_price},
        )

    return abs(fill_price - intended_price) / intended_price * 10_000.0


def calculate_return_pct(
    *,
    side: str,
    entry_price: float,
    exit_price: float,
) -> float:
    """
    Calculate raw return percentage for long/short.
    """

    if entry_price <= 0 or exit_price <= 0:
        raise InvalidCostModelInputError(
            "entry_price and exit_price must be positive.",
            details={
                "entry_price": entry_price,
                "exit_price": exit_price,
            },
        )

    side_value = side.lower()

    if side_value in {"buy", "long"}:
        return (exit_price - entry_price) / entry_price * 100.0

    if side_value in {"sell", "short"}:
        return (entry_price - exit_price) / entry_price * 100.0

    raise InvalidCostModelInputError(
        "Unsupported side for return calculation.",
        details={"side": side},
    )


def calculate_r_multiple(
    *,
    pnl: float,
    initial_risk: float,
) -> float | None:
    """
    Calculate R-multiple.
    """

    if initial_risk <= 0:
        return None

    return pnl / initial_risk


def split_funding_cashflow(cashflow: float) -> tuple[float, float]:
    """
    Convert signed funding cashflow into paid/received tuple.

    Returns:
        (funding_paid, funding_received)
    """

    if cashflow >= 0:
        return 0.0, cashflow

    return abs(cashflow), 0.0


def estimate_holding_seconds(
    *,
    opened_at_ms: int | None,
    closed_at_ms: int | None,
    fallback_end_ms: int | None = None,
) -> float:
    """
    Estimate holding time in seconds.
    """

    if opened_at_ms is None:
        return 0.0

    end_ms = closed_at_ms or fallback_end_ms

    if end_ms is None:
        return 0.0

    return max(0.0, (end_ms - opened_at_ms) / 1000.0)


def normalize_cost_timestamp(value: datetime | int | float | None) -> int:
    """
    Normalize optional timestamp value for cost diagnostics.
    """

    if value is None:
        return timestamp_ms(ensure_aware_utc(datetime.now()))

    if isinstance(value, datetime):
        return timestamp_ms(ensure_aware_utc(value))

    return timestamp_ms(value)


__all__ = [
    "CommissionInput",
    "SlippageInput",
    "FundingInput",
    "LiquidationInput",
    "TradeCostInput",
    "CommissionCalculator",
    "SlippageCalculator",
    "SpreadCostCalculator",
    "FundingCostCalculator",
    "BorrowCostCalculator",
    "LiquidationPriceCalculator",
    "TradingCostModel",
    "calculate_slippage_bps",
    "calculate_return_pct",
    "calculate_r_multiple",
    "split_funding_cashflow",
    "estimate_holding_seconds",
    "normalize_cost_timestamp",
]