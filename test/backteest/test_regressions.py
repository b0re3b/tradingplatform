"""
Regression tests for backtesting package integration edges.

Covered modules:
- backtesting.config
- backtesting.market_replay
- backtesting.execution_simulator
- backtesting.position_simulator
- backtesting.strategy_tester
- backtesting.report_builder

These tests lock down bugs that are easy to reintroduce while refactoring:
- futures defaults and shared config propagation;
- production-compatible market replay payload metadata;
- risk-approved execution-intent payload aliases;
- nested fill payload parsing by PositionSimulator;
- signal outcome progression in StrategyTester collectors;
- no execution when risk blocks a signal;
- deterministic repeated pipeline runs;
- report export filename/path stability.
"""

from __future__ import annotations

import copy
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from backtesting.config import BacktestConfig, ExecutionSimulatorConfig, ReportBuilderConfig
from backtesting.enums import BacktestDataType, BacktestMode, BacktestStatus, ReportFormat, SignalOutcome
from backtesting.execution_simulator import ExecutionSimulator
from backtesting.market_replay import MarketReplay, build_replay_event_from_record
from backtesting.models import (
    BacktestDataset,
    BacktestPeriod,
    BacktestResult,
    HistoricalCandle,
    SimulatedFill,
    timestamp_ms,
)
from backtesting.position_simulator import PositionSimulator
from backtesting.report_builder import ReportBuilder
from backtesting.strategy_tester import InMemoryBacktestEventBus, StrategyTester, run_backtest


# =============================================================================
# Compatibility helpers
# =============================================================================


def _status_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _outcome_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _disable_reports(config: BacktestConfig) -> BacktestConfig:
    config.save_report = False
    config.report_builder.enabled = False
    return config


def _clone_backtest_config(config: BacktestConfig, **changes: Any) -> BacktestConfig:
    clone = copy.deepcopy(config)
    for key, value in changes.items():
        setattr(clone, key, value)
    return clone


def _fill_event_payload(fill: SimulatedFill, **extra: Any) -> dict[str, Any]:
    payload = fill.to_dict()
    payload.update(
        {
            "topic": "execution.order_filled",
            "event_topic": "execution.order_filled",
            "run_id": fill.run_id,
            "order_id": fill.order_id,
            "fill_id": fill.fill_id,
            "signal_id": fill.signal_id,
            "exchange": fill.exchange,
            "market_type": fill.market_type,
            "symbol": fill.symbol,
            "side": fill.side,
            "price": fill.price,
            "fill_price": fill.price,
            "average_fill_price": fill.price,
            "quantity": fill.quantity,
            "fill_quantity": fill.quantity,
            "filled_quantity": fill.quantity,
            "filled_qty": fill.quantity,
            "notional": fill.notional,
            "fill_notional": fill.notional,
            "fee": fill.fee,
            "fees": fill.fee,
            "slippage": fill.slippage,
            "fill_slippage": fill.slippage,
            "slippage_bps": fill.slippage_bps,
            "fill_slippage_bps": fill.slippage_bps,
            "liquidity_type": fill.liquidity_type,
            "timestamp_ms": fill.timestamp_ms,
            "strategy_name": fill.metadata.get("strategy_name", "fake_strategy"),
            "order_type": fill.metadata.get("order_type", "market"),
            "metadata": dict(fill.metadata),
        }
    )
    payload.update(extra)
    return payload


async def _start_position_simulator(simulator: PositionSimulator) -> None:
    simulator.register()
    await simulator.start()


async def _start_execution_simulator(simulator: ExecutionSimulator) -> None:
    simulator.register()
    await simulator.start()


# =============================================================================
# Config / replay regressions
# =============================================================================


def test_backtest_config_propagates_futures_shared_settings(
    backtest_config: BacktestConfig,
) -> None:
    config = _clone_backtest_config(
        backtest_config,
        run_name="Regression Futures Run",
        exchange="BINANCE",
        market_type="USDM_FUTURES",
        symbols=["btcusdt", "BTCUSDT", "ethusdt"],
        timeframes=["1m", "5m", "1m"],
        quote_currency="USDT",
    )

    config.validate()

    assert config.exchange == "binance"
    assert config.market_type == "usdm_futures"
    assert config.symbols == ["BTCUSDT", "ETHUSDT"]
    assert config.timeframes == ["1m", "5m"]

    assert config.history_downloader.exchange == "binance"
    assert config.history_downloader.market_type == "usdm_futures"
    assert config.data_loader.market_type == "usdm_futures"
    assert config.execution_simulator.market_type == "usdm_futures"
    assert config.strategy_tester.market_type == "usdm_futures"
    assert config.position_simulator.quote_currency == "USDT"

    instruments = config.instruments()
    assert {item.market_type for item in instruments} == {"usdm_futures"}
    assert [item.symbol for item in instruments] == ["BTCUSDT", "ETHUSDT"]


def test_default_binance_futures_config_stays_futures_only(start_time, end_time) -> None:
    config = BacktestConfig.default_binance_futures(
        symbols=["btcusdt"],
        start_time=start_time,
        end_time=end_time,
        timeframes=["1m"],
        initial_balance=5_000.0,
    )

    assert config.exchange == "binance"
    assert config.market_type == "usdm_futures"
    assert config.use_candles is True
    assert config.use_funding is True
    assert config.use_open_interest is True
    assert config.use_trades is False
    assert config.execution_simulator.market_type == "usdm_futures"
    assert config.strategy_tester.market_type == "usdm_futures"


def test_market_replay_payload_preserves_domain_fields_and_adds_metadata(
    event_bus: InMemoryBacktestEventBus,
    backtest_clock,
    market_replay_config,
    sample_candles: list[HistoricalCandle],
    backtest_period: BacktestPeriod,
) -> None:
    candle = sample_candles[0]
    event = build_replay_event_from_record(
        candle,
        data_type=BacktestDataType.CANDLES,
        period=backtest_period,
        run_id="test_run",
        sequence=7,
    )
    event.payload["metadata"] = {"user_key": "keep_me"}
    event.payload["source"] = "fixture"

    replay = MarketReplay(
        config=market_replay_config,
        event_bus=event_bus,
        clock=backtest_clock,
    )

    payload = replay._build_payload(event)

    assert payload["symbol"] == "BTCUSDT"
    assert payload["market_type"] == "usdm_futures"
    assert payload["source"] == "fixture"
    assert payload["replay_sequence"] == 7
    assert payload["metadata"]["user_key"] == "keep_me"
    assert payload["metadata"]["backtest"] is True
    assert payload["metadata"]["replay_event_id"] == event.event_id


# =============================================================================
# Execution / position payload compatibility regressions
# =============================================================================


@pytest.mark.asyncio
async def test_execution_simulator_accepts_nested_execution_intent_payload(
    event_bus: InMemoryBacktestEventBus,
    execution_simulator_config: ExecutionSimulatorConfig,
    backtest_clock,
    cost_model,
) -> None:
    simulator = ExecutionSimulator(
        config=execution_simulator_config,
        event_bus=event_bus,
        clock=backtest_clock,
        cost_model=cost_model,
        random_seed=42,
    )

    filled: list[dict[str, Any]] = []
    event_bus.subscribe("execution.order_filled", lambda payload: filled.append(payload))

    await _start_execution_simulator(simulator)

    await event_bus.emit(
        "signal.confirmed",
        {
            "run_id": "test_run",
            "signal_id": "signal_nested_intent",
            "strategy_name": "fake_strategy",
            "execution_intent": {
                "symbol": "BTCUSDT",
                "exchange": "binance",
                "market_type": "usdm_futures",
                "side": "buy",
                "final_size": 1.25,
                "order_type": "market_entry",
            },
            "entry_plan": {
                "entry_price": 100.0,
            },
            "timestamp_ms": backtest_clock.timestamp_ms(),
        },
    )

    assert filled
    assert len(simulator.fills) == 1
    fill = simulator.fills[0]
    assert fill.symbol == "BTCUSDT"
    assert fill.market_type == "usdm_futures"
    assert fill.quantity == pytest.approx(1.25)
    assert fill.price > 0.0


@pytest.mark.asyncio
async def test_position_simulator_accepts_nested_fill_payload(
    event_bus: InMemoryBacktestEventBus,
    position_simulator: PositionSimulator,
    sample_long_fill: SimulatedFill,
) -> None:
    opened: list[dict[str, Any]] = []
    event_bus.subscribe("position.opened", lambda payload: opened.append(payload))

    await _start_position_simulator(position_simulator)

    await event_bus.emit(
        "execution.order_filled",
        {
            "topic": "execution.order_filled",
            "run_id": "test_run",
            "signal_id": sample_long_fill.signal_id,
            "strategy_name": "fake_strategy",
            "order_type": "market",
            "fill": {
                "fill_id": "nested_fill_1",
                "order_id": "order_nested_1",
                "run_id": "test_run",
                "signal_id": sample_long_fill.signal_id,
                "exchange": "binance",
                "market_type": "usdm_futures",
                "symbol": "BTCUSDT",
                "side": "buy",
                "price": 100.0,
                "quantity": 1.0,
                "notional": 100.0,
                "fee": 0.04,
                "slippage": 0.02,
                "slippage_bps": 2.0,
                "timestamp_ms": sample_long_fill.timestamp_ms,
                "metadata": {"strategy_name": "fake_strategy"},
            },
        },
    )

    assert opened
    positions = position_simulator.all_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "BTCUSDT"
    assert positions[0].market_type == "usdm_futures"
    assert positions[0].quantity == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_position_simulator_flat_fill_payload_aliases_do_not_zero_quantity(
    event_bus: InMemoryBacktestEventBus,
    position_simulator: PositionSimulator,
    sample_long_fill: SimulatedFill,
) -> None:
    await _start_position_simulator(position_simulator)

    payload = _fill_event_payload(sample_long_fill)
    payload.pop("fill", None)

    await event_bus.emit("execution.order_filled", payload)

    positions = position_simulator.all_positions()
    assert len(positions) == 1
    assert positions[0].quantity == pytest.approx(sample_long_fill.quantity)
    assert positions[0].entry_price == pytest.approx(sample_long_fill.price)


# =============================================================================
# StrategyTester collector and pipeline regressions
# =============================================================================


@pytest.mark.asyncio
async def test_strategy_tester_collectors_progress_signal_to_closed_win(
    backtest_config: BacktestConfig,
    sample_dataset: BacktestDataset,
    event_bus: InMemoryBacktestEventBus,
    fake_strategy_engine,
    fake_risk_manager_approve,
    sample_long_fill: SimulatedFill,
) -> None:
    _disable_reports(backtest_config)

    tester = StrategyTester(
        backtest_config,
        dataset=sample_dataset,
        event_bus=event_bus,
        strategy_engine=fake_strategy_engine,
        risk_manager=fake_risk_manager_approve,
    )
    tester.register()

    await event_bus.emit(
        "signal.generated",
        {
            "run_id": "test_run",
            "signal_id": "signal_regression",
            "strategy_name": "fake_strategy",
            "symbol": "BTCUSDT",
            "timeframe": "1m",
            "side": "buy",
            "confidence": 0.8,
            "strength": 0.7,
            "timestamp_ms": sample_long_fill.timestamp_ms,
        },
    )
    await event_bus.emit(
        "signal.confirmed",
        {
            "run_id": "test_run",
            "signal_id": "signal_regression",
            "strategy_name": "fake_strategy",
            "symbol": "BTCUSDT",
            "side": "buy",
            "final_size": 1.0,
            "timestamp_ms": sample_long_fill.timestamp_ms,
        },
    )
    await event_bus.emit(
        "execution.order_filled",
        {
            **_fill_event_payload(sample_long_fill, signal_id="signal_regression"),
            "signal_id": "signal_regression",
        },
    )
    await event_bus.emit(
        "position.opened",
        {
            "run_id": "test_run",
            "position_id": "position_1",
            "signal_id": "signal_regression",
            "strategy_name": "fake_strategy",
            "symbol": "BTCUSDT",
            "status": "open",
            "timestamp_ms": sample_long_fill.timestamp_ms,
        },
    )
    await event_bus.emit(
        "position.closed",
        {
            "run_id": "test_run",
            "position_id": "position_1",
            "signal_id": "signal_regression",
            "strategy_name": "fake_strategy",
            "symbol": "BTCUSDT",
            "status": "closed",
            "net_realized_pnl": 12.5,
            "timestamp_ms": sample_long_fill.timestamp_ms,
        },
    )

    assert len(tester.collectors.signals) == 1
    signal = tester.collectors.signals[0]
    assert _outcome_value(signal.outcome) == SignalOutcome.POSITION_CLOSED_WIN.value
    assert signal.pnl == pytest.approx(12.5)
    assert signal.confirmed_at_ms is not None
    assert signal.opened_at_ms is not None
    assert signal.closed_at_ms is not None
    assert len(tester.collectors.risk_decisions) == 1
    assert len(tester.collectors.execution_records) == 1
    assert len(tester.collectors.position_records) == 2


@pytest.mark.asyncio
async def test_full_pipeline_blocked_by_risk_does_not_emit_execution(
    backtest_config: BacktestConfig,
    sample_dataset: BacktestDataset,
    event_bus: InMemoryBacktestEventBus,
    fake_strategy_engine,
    fake_risk_manager_block,
) -> None:
    _disable_reports(backtest_config)

    execution_events: list[dict[str, Any]] = []
    event_bus.subscribe("execution.*", lambda payload: execution_events.append(payload))

    tester = StrategyTester(
        backtest_config,
        dataset=sample_dataset,
        event_bus=event_bus,
        strategy_engine=fake_strategy_engine,
        risk_manager=fake_risk_manager_block,
    )

    result = await tester.run()

    assert result.status == BacktestStatus.COMPLETED
    assert result.signals
    assert result.risk_decisions
    assert all(decision.blocked for decision in result.risk_decisions)
    assert result.orders == []
    assert result.fills == []
    assert execution_events == []


@pytest.mark.asyncio
async def test_run_backtest_is_deterministic_for_same_dataset(
    backtest_config: BacktestConfig,
    sample_dataset: BacktestDataset,
    fake_strategy_engine,
    fake_risk_manager_approve,
) -> None:
    config_1 = _disable_reports(_clone_backtest_config(backtest_config, run_name="regression_run_1"))
    config_2 = _disable_reports(_clone_backtest_config(backtest_config, run_name="regression_run_2"))

    event_bus_1 = InMemoryBacktestEventBus()
    engine_1 = type(fake_strategy_engine)(event_bus_1)
    risk_1 = type(fake_risk_manager_approve)(event_bus_1)

    result_1 = await run_backtest(
        config_1,
        dataset=sample_dataset,
        event_bus=event_bus_1,
        strategy_engine=engine_1,
        risk_manager=risk_1,
    )

    event_bus_2 = InMemoryBacktestEventBus()
    engine_2 = type(fake_strategy_engine)(event_bus_2)
    risk_2 = type(fake_risk_manager_approve)(event_bus_2)

    result_2 = await run_backtest(
        config_2,
        dataset=sample_dataset,
        event_bus=event_bus_2,
        strategy_engine=engine_2,
        risk_manager=risk_2,
    )

    assert result_1.status == BacktestStatus.COMPLETED
    assert result_2.status == BacktestStatus.COMPLETED
    assert len(result_1.signals) == len(result_2.signals) == 1
    assert len(result_1.orders) == len(result_2.orders) == 1
    assert len(result_1.fills) == len(result_2.fills) == 1
    assert result_1.fills[0].quantity == pytest.approx(result_2.fills[0].quantity)
    assert result_1.fills[0].price == pytest.approx(result_2.fills[0].price)


# =============================================================================
# Report/export regressions
# =============================================================================


def test_report_builder_safe_filename_removes_path_separators() -> None:
    name = ReportBuilder._safe_filename("../unsafe run/name:BTCUSDT")

    assert "/" not in name
    assert "\\" not in name
    assert ":" not in name
    assert " " not in name
    assert name


def test_report_builder_disabled_does_not_write_files(
    report_builder_config: ReportBuilderConfig,
    backtest_config: BacktestConfig,
    tmp_path: Path,
) -> None:
    report_builder_config.enabled = False
    report_builder_config.output_dir = str(tmp_path / "disabled_reports")
    builder = ReportBuilder(report_builder_config)

    result = BacktestResult(
        run_name="disabled_report_regression",
        mode=BacktestMode.SINGLE_STRATEGY,
        status=BacktestStatus.COMPLETED,
        period=backtest_config.period(),
        initial_balance=backtest_config.initial_balance,
        final_balance=backtest_config.initial_balance,
        final_equity=backtest_config.initial_balance,
    )

    report = builder.build(result)

    assert report.metadata["enabled"] is False
    assert report.path is None
    assert not Path(report_builder_config.output_dir).exists()


def test_report_builder_json_export_is_json_safe(
    report_builder_config: ReportBuilderConfig,
    backtest_config: BacktestConfig,
    sample_trade,
    sample_signal_records,
    tmp_path: Path,
) -> None:
    report_builder_config.enabled = True
    report_builder_config.output_dir = str(tmp_path / "reports")
    report_builder_config.formats = [ReportFormat.JSON]
    report_builder_config.save_result_json = True
    report_builder_config.save_trades_csv = False
    report_builder_config.save_positions_csv = False
    report_builder_config.save_equity_curve_csv = False
    report_builder_config.save_events_jsonl = False

    result = BacktestResult(
        run_name="json_safe_regression",
        mode=BacktestMode.SINGLE_STRATEGY,
        status=BacktestStatus.COMPLETED,
        period=backtest_config.period(),
        initial_balance=backtest_config.initial_balance,
        final_balance=backtest_config.initial_balance + sample_trade.net_pnl,
        final_equity=backtest_config.initial_balance + sample_trade.net_pnl,
        trades=[sample_trade],
        signals=list(sample_signal_records),
        metadata={"set_value": {"a", "b"}},
    )

    report = ReportBuilder(report_builder_config).build(result)

    assert report.artifacts
    json_artifact = next(item for item in report.artifacts if str(item.path).endswith(".json"))
    payload = json.loads(Path(json_artifact.path).read_text(encoding="utf-8"))

    assert payload["run_name"] == "json_safe_regression"
    assert payload["status"] == BacktestStatus.COMPLETED.value
    assert payload["metadata"]["set_value"] == ["a", "b"]