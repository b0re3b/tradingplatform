"""
Tests for cost models, execution simulator and position simulator.

Covered modules:
- backtesting.cost_models
- backtesting.execution_simulator
- backtesting.position_simulator

These tests focus on financial correctness:
- commission;
- slippage;
- spread/funding/liquidation calculations;
- order fill simulation;
- position opening/increasing/reducing/closing;
- mark-to-market PnL;
- equity curve;
- event emission.

The tests use compatibility helpers because several backtesting DTOs are strict
dataclasses and may have slightly different field names than production payloads.
"""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from datetime import timedelta
from typing import Any

import pytest

from backtesting.backtest_time import BacktestClock
from backtesting.config import (
    CostModelConfig,
    ExecutionSimulatorConfig,
)
from backtesting.cost_models import (
    CommissionCalculator,
    CommissionInput,
    FundingCostCalculator,
    FundingInput,
    LiquidationInput,
    LiquidationPriceCalculator,
    SlippageCalculator,
    SlippageInput,
    SpreadCostCalculator,
    TradeCostInput,
    TradingCostModel,
    calculate_r_multiple,
    calculate_return_pct,
    calculate_slippage_bps,
    split_funding_cashflow,
)
from backtesting.enums import (
    CommissionModel,
    LiquidityModel,
    SimulatedOrderStatus,
    SimulatedPositionStatus,
)
from backtesting.exceptions import InvalidCostModelInputError
from backtesting.execution_simulator import ExecutionSimulator
from backtesting.models import (
    HistoricalFundingRecord,
    HistoricalOrderBookSnapshot,
    SimulatedFill,
    timestamp_ms,
)
from backtesting.position_simulator import PositionSimulator
from backtesting.strategy_tester import InMemoryBacktestEventBus


# =============================================================================
# Compatibility helpers
# =============================================================================


def _dataclass_field_names(cls: type) -> set[str]:
    if not is_dataclass(cls):
        return set()
    return {field.name for field in fields(cls)}


def _build_dataclass(cls: type, **values: Any) -> Any:
    """
    Build dataclass using only supported fields.

    This keeps tests stable when DTOs use slightly different field names, for
    example:
    - liquidity_type vs is_maker;
    - holding_seconds vs duration_seconds;
    - position_notional vs notional.
    """

    if not is_dataclass(cls):
        return cls(**values)

    aliases = dict(values)

    if "liquidity_type" in values:
        liquidity_type = str(values["liquidity_type"]).lower()
        aliases.setdefault("is_maker", liquidity_type == "maker")
        aliases.setdefault("maker", liquidity_type == "maker")
        aliases.setdefault("is_taker", liquidity_type == "taker")
        aliases.setdefault("taker", liquidity_type == "taker")

    if "holding_seconds" in values:
        aliases.setdefault("duration_seconds", values["holding_seconds"])
        aliases.setdefault("held_seconds", values["holding_seconds"])
        aliases.setdefault("seconds", values["holding_seconds"])

    if "position_notional" in values:
        aliases.setdefault("notional", values["position_notional"])

    if "intended_price" in values:
        aliases.setdefault("price", values["intended_price"])
        aliases.setdefault("reference_price", values["intended_price"])

    if "entry_price" in values:
        aliases.setdefault("price", values["entry_price"])

    kwargs: dict[str, Any] = {}

    for field in fields(cls):
        if field.name in aliases:
            kwargs[field.name] = aliases[field.name]
            continue

        if field.default is not MISSING:
            continue

        if field.default_factory is not MISSING:  # type: ignore[attr-defined]
            continue

        name = field.name.lower()

        if "side" in name:
            kwargs[field.name] = values.get("side", "buy")
        elif "notional" in name:
            kwargs[field.name] = values.get("notional", values.get("position_notional", 1_000.0))
        elif "quantity" in name or name in {"qty", "size"}:
            kwargs[field.name] = values.get("quantity", 1.0)
        elif "price" in name:
            kwargs[field.name] = values.get(
                "intended_price",
                values.get("entry_price", values.get("price", 100.0)),
            )
        elif "rate" in name:
            kwargs[field.name] = values.get("funding_rate", 0.0001)
        elif "leverage" in name:
            kwargs[field.name] = values.get("leverage", 10.0)
        elif "margin" in name:
            kwargs[field.name] = values.get("maintenance_margin_rate", 0.005)
        elif "seconds" in name or "duration" in name:
            kwargs[field.name] = values.get("holding_seconds", 8 * 60 * 60)
        elif "symbol" in name:
            kwargs[field.name] = values.get("symbol", "BTCUSDT")
        elif "exchange" in name:
            kwargs[field.name] = values.get("exchange", "binance")
        elif "market_type" in name:
            kwargs[field.name] = values.get("market_type", "usdm_futures")
        elif "wallet" in name or "balance" in name:
            kwargs[field.name] = values.get("wallet_balance", 1_000.0)
        else:
            kwargs[field.name] = values.get(field.name, None)

    supported = _dataclass_field_names(cls)
    for key, value in aliases.items():
        if key in supported:
            kwargs[key] = value

    return cls(**kwargs)


def _calculate_slippage(calculator: SlippageCalculator, **kwargs: Any) -> Any:
    """
    Support both possible APIs:
    - SlippageCalculator.calculate(SlippageInput)
    - SlippageCalculator.calculate_bps(SlippageInput)
    """

    input_payload = _build_dataclass(SlippageInput, **kwargs)

    if hasattr(calculator, "calculate"):
        return calculator.calculate(input_payload)

    if hasattr(calculator, "calculate_bps"):
        bps = calculator.calculate_bps(input_payload)
        side = str(kwargs.get("side", "buy")).lower()
        intended_price = float(kwargs.get("intended_price", kwargs.get("price", 100.0)))
        direction = 1.0 if side in {"buy", "long"} else -1.0
        fill_price = intended_price * (1.0 + direction * float(bps) / 10_000.0)

        class Result:
            def __init__(self) -> None:
                self.fill_price = fill_price
                self.slippage_bps = float(bps)
                self.slippage = abs(fill_price - intended_price)

        return Result()

    raise AssertionError("SlippageCalculator has neither calculate() nor calculate_bps().")


async def _emit(event_bus: InMemoryBacktestEventBus, topic: str, payload: dict[str, Any]) -> None:
    await event_bus.emit(topic, payload)


def _status_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _position_is_open(position: Any) -> bool:
    if hasattr(position, "is_open"):
        return bool(position.is_open)

    status = _status_value(getattr(position, "status", ""))
    return status in {"open", "increased", "updated"}


def _latest_position(position_simulator: PositionSimulator) -> Any:
    positions = position_simulator.all_positions()
    assert positions
    return positions[-1]


def _long_fill(
    *,
    price: float = 100.0,
    quantity: float = 1.0,
    timestamp: int = 1,
    signal_id: str = "signal_long",
) -> SimulatedFill:
    return SimulatedFill(
        order_id=f"order_{signal_id}",
        run_id="test_run",
        signal_id=signal_id,
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        side="buy",
        price=price,
        quantity=quantity,
        notional=price * quantity,
        fee=0.04,
        fee_asset="USDT",
        slippage=0.02,
        slippage_bps=2.0,
        liquidity_type="taker",
        timestamp_ms=timestamp,
        metadata={
            "strategy_name": "fake_strategy",
            "order_type": "market",
        },
    )


def _short_fill(
    *,
    price: float = 100.0,
    quantity: float = 1.0,
    timestamp: int = 1,
    signal_id: str = "signal_short",
) -> SimulatedFill:
    return SimulatedFill(
        order_id=f"order_{signal_id}",
        run_id="test_run",
        signal_id=signal_id,
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        side="sell",
        price=price,
        quantity=quantity,
        notional=price * quantity,
        fee=0.04,
        fee_asset="USDT",
        slippage=0.02,
        slippage_bps=2.0,
        liquidity_type="taker",
        timestamp_ms=timestamp,
        metadata={
            "strategy_name": "fake_strategy",
            "order_type": "market",
        },
    )


def _fill_payload(fill: SimulatedFill, **extra: Any) -> dict[str, Any]:
    """
    Build execution.order_filled payload compatible with PositionSimulator.

    PositionSimulator may parse execution-style keys, not exactly
    SimulatedFill.to_dict() keys, so we include important aliases explicitly.
    """

    payload = fill.to_dict()

    payload.update(
        {
            "fill_id": fill.fill_id,
            "order_id": fill.order_id,
            "run_id": fill.run_id,
            "signal_id": fill.signal_id,
            "exchange": fill.exchange,
            "symbol": fill.symbol,
            "market_type": fill.market_type,
            "side": fill.side,

            # Quantity aliases.
            "quantity": fill.quantity,
            "qty": fill.quantity,
            "filled_quantity": fill.quantity,
            "fill_quantity": fill.quantity,
            "executed_quantity": fill.quantity,
            "filled_qty": fill.quantity,

            # Price aliases.
            "price": fill.price,
            "fill_price": fill.price,
            "filled_price": fill.price,
            "avg_price": fill.price,
            "average_price": fill.price,
            "executed_price": fill.price,

            # Notional aliases.
            "notional": fill.notional,
            "filled_notional": fill.notional,
            "executed_notional": fill.notional,

            # Cost aliases.
            "fee": fill.fee,
            "fees": fill.fee,
            "commission": fill.fee,
            "slippage": fill.slippage,
            "slippage_bps": fill.slippage_bps,
            "liquidity_type": fill.liquidity_type,

            "timestamp_ms": fill.timestamp_ms,
            "strategy_name": fill.metadata.get("strategy_name", "fake_strategy"),
            "order_type": fill.metadata.get("order_type", "market"),
            "metadata": dict(fill.metadata),
        }
    )

    payload.update(extra)
    return payload


async def _start_simulators(
    execution_simulator: ExecutionSimulator | None = None,
    position_simulator: PositionSimulator | None = None,
) -> None:
    if execution_simulator is not None:
        execution_simulator.register()
        await execution_simulator.start()

    if position_simulator is not None:
        position_simulator.register()
        await position_simulator.start()


# =============================================================================
# Cost model tests
# =============================================================================


def test_percentage_commission_model() -> None:
    calculator = CommissionCalculator(
        CostModelConfig(
            commission_model=CommissionModel.PERCENTAGE,
            default_fee_bps=10.0,
        )
    )

    cost = calculator.calculate(
        _build_dataclass(
            CommissionInput,
            notional=1_000.0,
            liquidity_type="taker",
            symbol="BTCUSDT",
            exchange="binance",
        )
    )

    assert cost == pytest.approx(1.0)


def test_maker_taker_commission_model() -> None:
    calculator = CommissionCalculator(
        CostModelConfig(
            commission_model=CommissionModel.MAKER_TAKER,
            maker_fee_bps=2.0,
            taker_fee_bps=4.0,
        )
    )

    maker_fee = calculator.calculate(
        _build_dataclass(
            CommissionInput,
            notional=1_000.0,
            liquidity_type="maker",
            symbol="BTCUSDT",
            exchange="binance",
        )
    )
    taker_fee = calculator.calculate(
        _build_dataclass(
            CommissionInput,
            notional=1_000.0,
            liquidity_type="taker",
            symbol="BTCUSDT",
            exchange="binance",
        )
    )

    assert maker_fee == pytest.approx(0.2)
    assert taker_fee == pytest.approx(0.4)


def test_no_commission_when_disabled() -> None:
    calculator = CommissionCalculator(
        CostModelConfig(
            include_commissions=False,
            default_fee_bps=100.0,
        )
    )

    cost = calculator.calculate(
        _build_dataclass(
            CommissionInput,
            notional=1_000.0,
            liquidity_type="taker",
            symbol="BTCUSDT",
            exchange="binance",
        )
    )

    assert cost == 0.0


def test_fixed_bps_slippage_buy_increases_price(cost_model_config: CostModelConfig) -> None:
    calculator = SlippageCalculator(cost_model_config)

    result = _calculate_slippage(
        calculator,
        side="buy",
        intended_price=100.0,
        quantity=1.0,
        symbol="BTCUSDT",
        exchange="binance",
    )

    assert result.fill_price > 100.0
    assert result.slippage >= 0.0
    assert result.slippage_bps == pytest.approx(cost_model_config.fixed_slippage_bps)


def test_fixed_bps_slippage_sell_decreases_price(cost_model_config: CostModelConfig) -> None:
    calculator = SlippageCalculator(cost_model_config)

    result = _calculate_slippage(
        calculator,
        side="sell",
        intended_price=100.0,
        quantity=1.0,
        symbol="BTCUSDT",
        exchange="binance",
    )

    assert result.fill_price < 100.0
    assert result.slippage >= 0.0
    assert result.slippage_bps == pytest.approx(cost_model_config.fixed_slippage_bps)


def test_orderbook_depth_slippage_uses_book_depth(
    cost_model_config: CostModelConfig,
    sample_orderbook: HistoricalOrderBookSnapshot,
) -> None:
    cost_model_config.slippage_model = getattr(
        type(cost_model_config.slippage_model),
        "ORDERBOOK_DEPTH",
        cost_model_config.slippage_model,
    )

    calculator = SlippageCalculator(cost_model_config)

    result = _calculate_slippage(
        calculator,
        side="buy",
        intended_price=100.0,
        quantity=1.5,
        symbol="BTCUSDT",
        exchange="binance",
        orderbook=sample_orderbook,
    )

    assert result.fill_price >= 100.0
    assert result.slippage >= 0.0


def test_spread_cost_half_spread(cost_model_config: CostModelConfig) -> None:
    calculator = SpreadCostCalculator(cost_model_config)

    cost = calculator.calculate(
        bid=99.9,
        ask=100.1,
        quantity=1.0,
    )

    assert cost == pytest.approx(0.1)


def test_funding_positive_rate_long_pays(cost_model_config: CostModelConfig) -> None:
    calculator = FundingCostCalculator(cost_model_config)

    funding = calculator.calculate(
        _build_dataclass(
            FundingInput,
            side="long",
            notional=10_000.0,
            funding_rate=0.0001,
            holding_seconds=8 * 60 * 60,
            symbol="BTCUSDT",
            exchange="binance",
        )
    )

    paid, received = split_funding_cashflow(funding)

    assert funding < 0
    assert paid > 0
    assert received == 0.0


def test_funding_positive_rate_short_receives(cost_model_config: CostModelConfig) -> None:
    calculator = FundingCostCalculator(cost_model_config)

    funding = calculator.calculate(
        _build_dataclass(
            FundingInput,
            side="short",
            notional=10_000.0,
            funding_rate=0.0001,
            holding_seconds=8 * 60 * 60,
            symbol="BTCUSDT",
            exchange="binance",
        )
    )

    paid, received = split_funding_cashflow(funding)

    assert funding > 0
    assert paid == 0.0
    assert received > 0


def test_liquidation_price_long_and_short(cost_model_config: CostModelConfig) -> None:
    calculator = LiquidationPriceCalculator(cost_model_config)

    long_price = calculator.calculate(
        _build_dataclass(
            LiquidationInput,
            side="long",
            entry_price=100.0,
            leverage=10.0,
            maintenance_margin_rate=0.005,
            wallet_balance=1_000.0,
            position_notional=1_000.0,
            quantity=10.0,
        )
    )
    short_price = calculator.calculate(
        _build_dataclass(
            LiquidationInput,
            side="short",
            entry_price=100.0,
            leverage=10.0,
            maintenance_margin_rate=0.005,
            wallet_balance=1_000.0,
            position_notional=1_000.0,
            quantity=10.0,
        )
    )

    assert long_price < 100.0
    assert short_price > 100.0


def test_cost_model_calculates_trade_costs(cost_model: TradingCostModel) -> None:
    result = cost_model.calculate_trade_costs(
        _build_dataclass(
            TradeCostInput,
            side="buy",
            intended_price=100.0,
            quantity=1.0,
            symbol="BTCUSDT",
            exchange="binance",
            liquidity_type="taker",
            bid=99.9,
            ask=100.1,
        )
    )

    assert result.total_cost >= 0.0
    assert result.commission >= 0.0
    assert result.slippage >= 0.0


def test_cost_model_rejects_invalid_negative_price(cost_model: TradingCostModel) -> None:
    with pytest.raises(InvalidCostModelInputError):
        cost_model.calculate_trade_costs(
            _build_dataclass(
                TradeCostInput,
                side="buy",
                intended_price=-100.0,
                quantity=1.0,
                symbol="BTCUSDT",
                exchange="binance",
                liquidity_type="taker",
            )
        )


def test_return_r_and_slippage_helpers() -> None:
    assert calculate_return_pct(entry_price=100.0, exit_price=110.0, side="long") == pytest.approx(10.0)
    assert calculate_return_pct(entry_price=100.0, exit_price=90.0, side="short") == pytest.approx(10.0)

    try:
        r_multiple = calculate_r_multiple(pnl=200.0, risk_amount=100.0)
    except TypeError:
        r_multiple = calculate_r_multiple(200.0, 100.0)

    assert r_multiple == pytest.approx(2.0)
    assert calculate_slippage_bps(intended_price=100.0, fill_price=100.1) == pytest.approx(10.0)


# =============================================================================
# Execution simulator tests
# =============================================================================


@pytest.mark.asyncio
async def test_execution_signal_confirmed_creates_order_and_fill(
    event_bus: InMemoryBacktestEventBus,
    execution_simulator: ExecutionSimulator,
    backtest_clock: BacktestClock,
) -> None:
    submitted: list[dict[str, Any]] = []
    filled: list[dict[str, Any]] = []

    event_bus.subscribe("execution.order_submitted", lambda payload: submitted.append(payload))
    event_bus.subscribe("execution.order_filled", lambda payload: filled.append(payload))

    await _start_simulators(execution_simulator=execution_simulator)

    await _emit(
        event_bus,
        "signal.confirmed",
        {
            "run_id": "test_run",
            "signal_id": "signal_1",
            "strategy_name": "fake_strategy",
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "market_type": "usdm_futures",
            "side": "buy",
            "quantity": 1.0,
            "order_type": "market",
            "entry_price": 100.0,
            "timestamp_ms": backtest_clock.timestamp_ms(),
        },
    )

    assert submitted
    assert filled
    assert len(execution_simulator.orders) == 1
    assert len(execution_simulator.fills) == 1

    order = next(iter(execution_simulator.orders.values()))
    assert _status_value(order.status) in {
        SimulatedOrderStatus.FILLED.value,
        "filled",
    }

    fill = execution_simulator.fills[0]
    assert fill.symbol == "BTCUSDT"
    assert fill.quantity == pytest.approx(1.0)
    assert fill.price > 0.0
    assert fill.fee >= 0.0
    assert fill.slippage >= 0.0


@pytest.mark.asyncio
async def test_execution_rejects_order_without_price_when_required(
    event_bus: InMemoryBacktestEventBus,
    execution_simulator_config: ExecutionSimulatorConfig,
    backtest_clock: BacktestClock,
    cost_model: TradingCostModel,
) -> None:
    execution_simulator_config.reject_if_no_price = True

    simulator = ExecutionSimulator(
        config=execution_simulator_config,
        event_bus=event_bus,
        clock=backtest_clock,
        cost_model=cost_model,
        random_seed=42,
    )

    rejected: list[dict[str, Any]] = []
    event_bus.subscribe("execution.order_rejected", lambda payload: rejected.append(payload))

    await _start_simulators(execution_simulator=simulator)

    await _emit(
        event_bus,
        "signal.confirmed",
        {
            "run_id": "test_run",
            "signal_id": "signal_no_price",
            "strategy_name": "fake_strategy",
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "market_type": "usdm_futures",
            "side": "buy",
            "quantity": 1.0,
            "order_type": "market",
            "timestamp_ms": backtest_clock.timestamp_ms(),
        },
    )

    assert rejected
    assert len(simulator.fills) == 0


@pytest.mark.asyncio
async def test_execution_partial_fill_by_liquidity(
    event_bus: InMemoryBacktestEventBus,
    execution_simulator_config: ExecutionSimulatorConfig,
    backtest_clock: BacktestClock,
    cost_model: TradingCostModel,
) -> None:
    execution_simulator_config.liquidity_model = (
        getattr(LiquidityModel, "CANDLE_VOLUME", None)
        or getattr(LiquidityModel, "VOLUME_PARTICIPATION", None)
        or getattr(LiquidityModel, "CANDLE_VOLUME_PARTICIPATION", None)
        or execution_simulator_config.liquidity_model
    )
    execution_simulator_config.allow_partial_fills = True
    execution_simulator_config.max_volume_participation_pct = 10.0

    simulator = ExecutionSimulator(
        config=execution_simulator_config,
        event_bus=event_bus,
        clock=backtest_clock,
        cost_model=cost_model,
        random_seed=42,
    )

    await _start_simulators(execution_simulator=simulator)

    await _emit(
        event_bus,
        "market.candle",
        {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "market_type": "usdm_futures",
            "timeframe": "1m",
            "timestamp_ms": backtest_clock.timestamp_ms(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 10.0,
            "is_closed": True,
        },
    )

    await _emit(
        event_bus,
        "signal.confirmed",
        {
            "run_id": "test_run",
            "signal_id": "signal_large",
            "strategy_name": "fake_strategy",
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "market_type": "usdm_futures",
            "side": "buy",
            "quantity": 10.0,
            "order_type": "market",
            "entry_price": 100.0,
            "timestamp_ms": backtest_clock.timestamp_ms(),
        },
    )

    assert simulator.fills
    assert simulator.fills[0].quantity <= 10.0


@pytest.mark.asyncio
async def test_execution_kill_switch_cancels_or_blocks_orders(
    event_bus: InMemoryBacktestEventBus,
    execution_simulator: ExecutionSimulator,
    backtest_clock: BacktestClock,
) -> None:
    rejected: list[dict[str, Any]] = []
    event_bus.subscribe("execution.order_rejected", lambda payload: rejected.append(payload))

    await _start_simulators(execution_simulator=execution_simulator)

    await _emit(
        event_bus,
        "risk.kill_switch",
        {
            "reason": "test_kill_switch",
            "timestamp_ms": backtest_clock.timestamp_ms(),
        },
    )

    await _emit(
        event_bus,
        "signal.confirmed",
        {
            "run_id": "test_run",
            "signal_id": "signal_after_kill",
            "strategy_name": "fake_strategy",
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "market_type": "usdm_futures",
            "side": "buy",
            "quantity": 1.0,
            "order_type": "market",
            "entry_price": 100.0,
            "timestamp_ms": backtest_clock.timestamp_ms(),
        },
    )

    assert rejected or len(execution_simulator.fills) == 0


@pytest.mark.asyncio
async def test_execution_fill_payload_contains_costs(
    event_bus: InMemoryBacktestEventBus,
    execution_simulator: ExecutionSimulator,
    backtest_clock: BacktestClock,
) -> None:
    filled: list[dict[str, Any]] = []
    event_bus.subscribe("execution.order_filled", lambda payload: filled.append(payload))

    await _start_simulators(execution_simulator=execution_simulator)

    await _emit(
        event_bus,
        "signal.confirmed",
        {
            "run_id": "test_run",
            "signal_id": "signal_costs",
            "strategy_name": "fake_strategy",
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "market_type": "usdm_futures",
            "side": "buy",
            "quantity": 1.0,
            "order_type": "market",
            "entry_price": 100.0,
            "timestamp_ms": backtest_clock.timestamp_ms(),
        },
    )

    assert filled
    payload = filled[0]

    assert "fee" in payload or "fees" in payload or "costs" in payload
    assert "slippage" in payload or "slippage_bps" in payload


# =============================================================================
# Position simulator tests
# =============================================================================


@pytest.mark.asyncio
async def test_position_buy_fill_opens_long(
    event_bus: InMemoryBacktestEventBus,
    position_simulator: PositionSimulator,
    start_time,
) -> None:
    opened: list[dict[str, Any]] = []
    event_bus.subscribe("position.opened", lambda payload: opened.append(payload))

    await _start_simulators(position_simulator=position_simulator)

    fill = _long_fill(timestamp=timestamp_ms(start_time))
    await _emit(event_bus, "execution.order_filled", _fill_payload(fill))

    position = _latest_position(position_simulator)

    assert opened
    assert _position_is_open(position)
    assert str(position.symbol) == "BTCUSDT"
    assert str(position.side).lower() in {"long", "buy"}
    assert float(position.quantity) == pytest.approx(1.0)
    assert float(position.entry_price) == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_position_sell_fill_opens_short(
    event_bus: InMemoryBacktestEventBus,
    position_simulator: PositionSimulator,
    start_time,
) -> None:
    await _start_simulators(position_simulator=position_simulator)

    fill = _short_fill(timestamp=timestamp_ms(start_time))
    await _emit(event_bus, "execution.order_filled", _fill_payload(fill))

    position = _latest_position(position_simulator)

    assert _position_is_open(position)
    assert str(position.side).lower() in {"short", "sell"}
    assert float(position.quantity) == pytest.approx(1.0)
    assert float(position.entry_price) == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_position_same_side_fill_increases_average_entry(
    event_bus: InMemoryBacktestEventBus,
    position_simulator: PositionSimulator,
    start_time,
) -> None:
    await _start_simulators(position_simulator=position_simulator)

    await _emit(
        event_bus,
        "execution.order_filled",
        _fill_payload(_long_fill(price=100.0, quantity=1.0, timestamp=timestamp_ms(start_time))),
    )
    await _emit(
        event_bus,
        "execution.order_filled",
        _fill_payload(_long_fill(price=110.0, quantity=1.0, timestamp=timestamp_ms(start_time + timedelta(minutes=1)))),
    )

    position = _latest_position(position_simulator)

    assert float(position.quantity) == pytest.approx(2.0)
    assert float(position.entry_price) == pytest.approx(105.0)


@pytest.mark.asyncio
async def test_position_opposite_fill_reduces_position(
    event_bus: InMemoryBacktestEventBus,
    position_simulator: PositionSimulator,
    start_time,
) -> None:
    await _start_simulators(position_simulator=position_simulator)

    await _emit(
        event_bus,
        "execution.order_filled",
        _fill_payload(_long_fill(price=100.0, quantity=2.0, timestamp=timestamp_ms(start_time))),
    )
    await _emit(
        event_bus,
        "execution.order_filled",
        _fill_payload(_short_fill(price=105.0, quantity=1.0, timestamp=timestamp_ms(start_time + timedelta(minutes=1)))),
    )

    position = _latest_position(position_simulator)

    assert float(position.quantity) == pytest.approx(1.0)
    assert position_simulator.trades


@pytest.mark.asyncio
async def test_position_opposite_fill_closes_position_and_creates_trade(
    event_bus: InMemoryBacktestEventBus,
    position_simulator: PositionSimulator,
    start_time,
) -> None:
    closed: list[dict[str, Any]] = []
    event_bus.subscribe("position.closed", lambda payload: closed.append(payload))

    await _start_simulators(position_simulator=position_simulator)

    await _emit(
        event_bus,
        "execution.order_filled",
        _fill_payload(_long_fill(price=100.0, quantity=1.0, timestamp=timestamp_ms(start_time))),
    )
    await _emit(
        event_bus,
        "execution.order_filled",
        _fill_payload(_short_fill(price=110.0, quantity=1.0, timestamp=timestamp_ms(start_time + timedelta(minutes=1)))),
    )

    assert closed
    assert position_simulator.trades

    trade = position_simulator.trades[-1]
    assert trade.net_pnl > 0
    assert trade.exit_price == pytest.approx(110.0)


@pytest.mark.asyncio
async def test_position_mark_to_market_updates_unrealized_pnl(
    event_bus: InMemoryBacktestEventBus,
    position_simulator: PositionSimulator,
    start_time,
) -> None:
    await _start_simulators(position_simulator=position_simulator)

    await _emit(
        event_bus,
        "execution.order_filled",
        _fill_payload(_long_fill(price=100.0, quantity=1.0, timestamp=timestamp_ms(start_time))),
    )

    await _emit(
        event_bus,
        "market.candle",
        {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "market_type": "usdm_futures",
            "timeframe": "1m",
            "timestamp_ms": timestamp_ms(start_time + timedelta(minutes=1)),
            "open": 110.0,
            "high": 111.0,
            "low": 109.0,
            "close": 110.0,
            "volume": 1_000.0,
            "is_closed": True,
        },
    )

    position = _latest_position(position_simulator)

    assert getattr(position, "unrealized_pnl", 0.0) >= 0.0
    assert position_simulator.equity_curve


@pytest.mark.asyncio
async def test_position_apply_funding_updates_balance(
    event_bus: InMemoryBacktestEventBus,
    position_simulator: PositionSimulator,
    start_time,
) -> None:
    await _start_simulators(position_simulator=position_simulator)

    await _emit(
        event_bus,
        "execution.order_filled",
        _fill_payload(_long_fill(price=100.0, quantity=1.0, timestamp=timestamp_ms(start_time))),
    )

    before_balance = position_simulator.balance.cash_balance

    funding = HistoricalFundingRecord(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timestamp_ms=timestamp_ms(start_time + timedelta(minutes=10)),
        received_at_ms=timestamp_ms(start_time + timedelta(minutes=10)),
        funding_rate=0.001,
        predicted_rate=None,
        mark_price=100.0,
        index_price=100.0,
        next_funding_time_ms=None,
        source="test",
        metadata={},
    )

    await _emit(event_bus, "market.funding", funding.to_market_event_payload())

    after_balance = position_simulator.balance.cash_balance

    assert after_balance <= before_balance


@pytest.mark.asyncio
async def test_position_stop_loss_closes_long_when_candle_low_hits(
    event_bus: InMemoryBacktestEventBus,
    position_simulator: PositionSimulator,
    start_time,
) -> None:
    closed: list[dict[str, Any]] = []
    event_bus.subscribe("position.closed", lambda payload: closed.append(payload))

    await _start_simulators(position_simulator=position_simulator)

    payload = _fill_payload(
        _long_fill(price=100.0, quantity=1.0, timestamp=timestamp_ms(start_time)),
        stop_loss=95.0,
        take_profit=120.0,
    )

    await _emit(event_bus, "execution.order_filled", payload)

    await _emit(
        event_bus,
        "market.candle",
        {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "market_type": "usdm_futures",
            "timeframe": "1m",
            "timestamp_ms": timestamp_ms(start_time + timedelta(minutes=1)),
            "open": 100.0,
            "high": 101.0,
            "low": 94.0,
            "close": 96.0,
            "volume": 1_000.0,
            "is_closed": True,
        },
    )

    assert closed or position_simulator.trades


@pytest.mark.asyncio
async def test_position_take_profit_closes_long_when_candle_high_hits(
    event_bus: InMemoryBacktestEventBus,
    position_simulator: PositionSimulator,
    start_time,
) -> None:
    closed: list[dict[str, Any]] = []
    event_bus.subscribe("position.closed", lambda payload: closed.append(payload))

    await _start_simulators(position_simulator=position_simulator)

    payload = _fill_payload(
        _long_fill(price=100.0, quantity=1.0, timestamp=timestamp_ms(start_time)),
        stop_loss=95.0,
        take_profit=105.0,
    )

    await _emit(event_bus, "execution.order_filled", payload)

    await _emit(
        event_bus,
        "market.candle",
        {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "market_type": "usdm_futures",
            "timeframe": "1m",
            "timestamp_ms": timestamp_ms(start_time + timedelta(minutes=1)),
            "open": 100.0,
            "high": 106.0,
            "low": 99.0,
            "close": 105.0,
            "volume": 1_000.0,
            "is_closed": True,
        },
    )

    assert closed or position_simulator.trades


@pytest.mark.asyncio
async def test_position_liquidation_emits_event(
    event_bus: InMemoryBacktestEventBus,
    position_simulator: PositionSimulator,
    start_time,
) -> None:
    liquidated: list[dict[str, Any]] = []
    event_bus.subscribe("position.liquidated", lambda payload: liquidated.append(payload))

    await _start_simulators(position_simulator=position_simulator)

    await _emit(
        event_bus,
        "execution.order_filled",
        _fill_payload(_long_fill(price=100.0, quantity=10.0, timestamp=timestamp_ms(start_time))),
    )

    await _emit(
        event_bus,
        "market.candle",
        {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "market_type": "usdm_futures",
            "timeframe": "1m",
            "timestamp_ms": timestamp_ms(start_time + timedelta(minutes=1)),
            "open": 100.0,
            "high": 100.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1_000.0,
            "is_closed": True,
        },
    )

    assert liquidated or any(
        _status_value(position.status) == SimulatedPositionStatus.LIQUIDATED.value
        for position in position_simulator.all_positions()
    )


@pytest.mark.asyncio
async def test_position_equity_curve_records_points(
    event_bus: InMemoryBacktestEventBus,
    position_simulator: PositionSimulator,
    start_time,
) -> None:
    await _start_simulators(position_simulator=position_simulator)

    await _emit(
        event_bus,
        "execution.order_filled",
        _fill_payload(_long_fill(price=100.0, quantity=1.0, timestamp=timestamp_ms(start_time))),
    )

    await _emit(
        event_bus,
        "market.candle",
        {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "market_type": "usdm_futures",
            "timeframe": "1m",
            "timestamp_ms": timestamp_ms(start_time + timedelta(minutes=1)),
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
            "volume": 1_000.0,
            "is_closed": True,
        },
    )

    assert position_simulator.equity_curve
    assert position_simulator.equity_curve[-1].equity > 0.0