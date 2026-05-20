"""
Shared pytest fixtures for backtesting package tests.

These fixtures intentionally use small deterministic datasets and fake pipeline
components so tests can verify backtesting behavior without live exchanges,
real strategy engines or real risk/execution services.
"""

from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from backtesting.backtest_time import BacktestClock
from backtesting.config import (
    BacktestConfig,
    BacktestTimeConfig,
    CostModelConfig,
    DataLoaderConfig,
    ExecutionSimulatorConfig,
    MarketReplayConfig,
    ModelAnalyticsConfig,
    OptimizerConfig,
    PerformanceMetricsConfig,
    PositionSimulatorConfig,
    ReportBuilderConfig,
    StrategyTesterConfig,
    WalkForwardConfig,
)
from backtesting.cost_models import TradingCostModel
from backtesting.enums import (
    BacktestDataType,
    BacktestEventType,
    BacktestMode,
    BacktestStatus,
    CandleExecutionPath,
    CommissionModel,
    DataGapPolicy,
    DataValidationLevel,
    FillModel,
    FundingSimulationMode,
    HistoricalDataFormat,
    LatencyModel,
    LiquidityModel,
    OptimizationDirection,
    OptimizationMethod,
    OptimizationMetric,
    PnLAccountingMode,
    PositionAccountingMode,
    ReplayEventPriority,
    ReplayMode,
    ReplayOrdering,
    ReportFormat,
    ReportSection,
    SignalOutcome,
    SlippageModel,
)
from backtesting.execution_simulator import ExecutionSimulator
from backtesting.market_replay import build_replay_event_from_record
from backtesting.models import (
    BacktestDataset,
    BacktestDatasetInfo,
    BacktestEvent,
    BacktestInstrument,
    BacktestPeriod,
    BacktestRiskDecisionRecord,
    BacktestSignalRecord,
    HistoricalCandle,
    HistoricalFundingRecord,
    HistoricalOpenInterestRecord,
    HistoricalOrderBookLevel,
    HistoricalOrderBookSnapshot,
    HistoricalTrade,
    OptimizationResult,
    OptimizationTrialResult,
    PerformanceSummary,
    PortfolioBacktestResult,
    SimulatedEquityPoint,
    SimulatedFill,
    SimulatedOrder,
    SimulatedPosition,
    SimulatedTrade,
    TradeStats,
    timestamp_ms,
)
from backtesting.position_simulator import PositionSimulator
from backtesting.strategy_tester import InMemoryBacktestEventBus


# =============================================================================
# Pytest / asyncio
# =============================================================================


@pytest.fixture(scope="session")
def event_loop() -> asyncio.AbstractEventLoop:
    """
    Session-level event loop for pytest-asyncio compatibility.

    pytest-asyncio auto mode usually creates loops itself, but this keeps tests
    stable when running in older local environments.
    """

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Basic time fixtures
# =============================================================================


@pytest.fixture
def start_time() -> datetime:
    return datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def end_time(start_time: datetime) -> datetime:
    return start_time + timedelta(hours=2)


@pytest.fixture
def warmup_start_time(start_time: datetime) -> datetime:
    return start_time - timedelta(minutes=30)


@pytest.fixture
def backtest_period(
    start_time: datetime,
    end_time: datetime,
    warmup_start_time: datetime,
) -> BacktestPeriod:
    return BacktestPeriod(
        start=start_time,
        end=end_time,
        warmup_start=warmup_start_time,
    )


@pytest.fixture
def backtest_clock(backtest_period: BacktestPeriod) -> BacktestClock:
    clock = BacktestClock(
        period=backtest_period,
        config=BacktestTimeConfig(),
    )
    clock.start(total_events=100)
    return clock


# =============================================================================
# Config fixtures
# =============================================================================


@pytest.fixture
def cost_model_config() -> CostModelConfig:
    return CostModelConfig(
        commission_model=CommissionModel.MAKER_TAKER,
        maker_fee_bps=2.0,
        taker_fee_bps=4.0,
        default_fee_bps=4.0,
        slippage_model=SlippageModel.FIXED_BPS,
        fixed_slippage_bps=2.0,
        include_commissions=True,
        include_slippage=True,
        include_spread_cost=True,
        include_funding=True,
        funding_mode=FundingSimulationMode.APPLY_ON_FUNDING_TIMESTAMP,
    )


@pytest.fixture
def execution_simulator_config() -> ExecutionSimulatorConfig:
    return ExecutionSimulatorConfig(
        exchange="binance",
        market_type="usdm_futures",
        fill_model=FillModel.INSTANT,
        liquidity_model=LiquidityModel.UNLIMITED,
        latency_model=LatencyModel.NONE,
        candle_execution_path=CandleExecutionPath.CONSERVATIVE,
        reject_if_no_price=False,
        reject_if_no_liquidity=False,
        allow_market_orders=True,
        allow_limit_orders=True,
        allow_stop_orders=True,
        allow_reduce_only=True,
        allow_partial_fills=True,
        record_orders=True,
        record_fills=True,
    )


@pytest.fixture
def position_simulator_config() -> PositionSimulatorConfig:
    return PositionSimulatorConfig(
        initial_balance=10_000.0,
        quote_currency="USDT",
        default_leverage=2.0,
        max_leverage=10.0,
        maintenance_margin_rate=0.005,
        liquidation_buffer_bps=5.0,
        position_accounting_mode=PositionAccountingMode.NETTING,
        pnl_accounting_mode=PnLAccountingMode.MARK_TO_MARKET,
        enable_mark_to_market=True,
        enable_funding_application=True,
        enable_liquidation_check=True,
        enable_stop_loss=True,
        enable_take_profit=True,
        record_positions=True,
        record_equity_curve=True,
        emit_position_events=True,
    )


@pytest.fixture
def market_replay_config() -> MarketReplayConfig:
    return MarketReplayConfig(
        replay_mode=ReplayMode.FULL_RUN,
        batch_events_by_timestamp=False,
        emit_replay_lifecycle_events=True,
        emit_market_candles=True,
        emit_market_trades=True,
        emit_market_funding=True,
        emit_market_open_interest=True,
        emit_market_liquidations=True,
        fail_on_emit_error=True,
    )


@pytest.fixture
def performance_metrics_config() -> PerformanceMetricsConfig:
    return PerformanceMetricsConfig(
        calculate_trade_stats=True,
        calculate_drawdowns=True,
        calculate_ratios=True,
        calculate_risk_stats=True,
        calculate_execution_stats=True,
        calculate_cost_breakdown=True,
        calculate_strategy_breakdown=True,
        calculate_symbol_breakdown=True,
        min_trades_for_ratios=1,
    )


@pytest.fixture
def model_analytics_config() -> ModelAnalyticsConfig:
    return ModelAnalyticsConfig(
        analyze_signal_quality=True,
        analyze_strategy_attribution=True,
        analyze_regime_performance=True,
        analyze_feature_importance=True,
        analyze_risk_decisions=True,
        analyze_execution_quality=True,
    )


@pytest.fixture
def report_builder_config(tmp_path: Path) -> ReportBuilderConfig:
    return ReportBuilderConfig(
        enabled=True,
        output_dir=str(tmp_path / "reports"),
        report_title="Backtest Test Report",
        formats=[ReportFormat.MARKDOWN, ReportFormat.JSON, ReportFormat.CSV],
        sections=[
            ReportSection.SUMMARY,
            ReportSection.TRADES,
            ReportSection.POSITIONS,
            ReportSection.STRATEGIES,
            ReportSection.RISK,
            ReportSection.EXECUTION,
            ReportSection.COSTS,
            ReportSection.SIGNALS,
            ReportSection.WARNINGS,
        ],
        save_result_json=True,
        save_trades_csv=True,
        save_positions_csv=True,
        save_equity_curve_csv=True,
        save_events_jsonl=True,
    )


@pytest.fixture
def strategy_tester_config() -> StrategyTesterConfig:
    return StrategyTesterConfig(
        run_name="test_backtest",
        mode=BacktestMode.SINGLE_STRATEGY,
        require_strategy_engine=True,
        require_signal_processor=False,
        require_risk_manager=True,
        require_analytics=False,
        fail_if_live_execution_detected=True,
        collect_signal_records=True,
        collect_risk_records=True,
        collect_execution_records=True,
        collect_position_records=True,
        cleanup_after_run=False,
        stop_on_first_error=True,
    )


@pytest.fixture
def optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(
        enabled=True,
        method=OptimizationMethod.GRID_SEARCH,
        objective_metric=OptimizationMetric.NET_PROFIT_PCT,
        direction=OptimizationDirection.MAXIMIZE,
        max_trials=20,
        parallel_jobs=1,
        parameter_space={
            "cost_model.fixed_slippage_bps": [1.0, 2.0],
            "position_simulator.default_leverage": [1.0, 2.0],
        },
        min_trades_required=0,
        stop_on_trial_error=False,
    )


@pytest.fixture
def walk_forward_config() -> WalkForwardConfig:
    return WalkForwardConfig(
        enabled=True,
        train_window=timedelta(minutes=40),
        validation_window=timedelta(minutes=20),
        test_window=timedelta(minutes=20),
        step_size=timedelta(minutes=20),
        max_iterations=3,
        aggregate_results=True,
        calculate_stability_score=True,
        calculate_overfitting_score=True,
    )


@pytest.fixture
def backtest_config(
    start_time: datetime,
    end_time: datetime,
    warmup_start_time: datetime,
    cost_model_config: CostModelConfig,
    execution_simulator_config: ExecutionSimulatorConfig,
    position_simulator_config: PositionSimulatorConfig,
    market_replay_config: MarketReplayConfig,
    performance_metrics_config: PerformanceMetricsConfig,
    model_analytics_config: ModelAnalyticsConfig,
    report_builder_config: ReportBuilderConfig,
    strategy_tester_config: StrategyTesterConfig,
    optimizer_config: OptimizerConfig,
    walk_forward_config: WalkForwardConfig,
) -> BacktestConfig:
    return BacktestConfig(
        run_name="test_backtest",
        mode=BacktestMode.SINGLE_STRATEGY,
        start_time=start_time,
        end_time=end_time,
        warmup_start_time=warmup_start_time,
        initial_balance=10_000.0,
        symbols=["BTCUSDT"],
        timeframes=["1m"],
        exchange="binance",
        market_type="usdm_futures",
        save_report=True,
        cost_model=cost_model_config,
        execution_simulator=execution_simulator_config,
        position_simulator=position_simulator_config,
        market_replay=market_replay_config,
        performance_metrics=performance_metrics_config,
        model_analytics=model_analytics_config,
        report_builder=report_builder_config,
        strategy_tester=strategy_tester_config,
        optimizer=optimizer_config,
        walk_forward=walk_forward_config,
    )


@pytest.fixture
def data_loader_config(tmp_path: Path) -> DataLoaderConfig:
    return DataLoaderConfig(
        data_dir=str(tmp_path / "history"),
        input_format=HistoricalDataFormat.CSV,
        exchange="binance",
        market_type="usdm_futures",
        symbols=["BTCUSDT"],
        timeframes=["1m"],
        data_types={BacktestDataType.CANDLES},
        require_candles=True,
        require_trades=False,
        require_funding=False,
        require_open_interest=False,
        require_orderbook=False,
        allow_empty_optional_streams=True,
        validation_level=DataValidationLevel.BASIC,
        gap_policy=DataGapPolicy.WARN,
        max_allowed_gap_seconds=120,
        drop_duplicate_events=True,
    )


# =============================================================================
# Event bus / cost model / simulators
# =============================================================================


@pytest.fixture
def event_bus() -> InMemoryBacktestEventBus:
    return InMemoryBacktestEventBus()


@pytest.fixture
def cost_model(cost_model_config: CostModelConfig) -> TradingCostModel:
    return TradingCostModel(cost_model_config)


@pytest.fixture
def execution_simulator(
    execution_simulator_config: ExecutionSimulatorConfig,
    event_bus: InMemoryBacktestEventBus,
    backtest_clock: BacktestClock,
    cost_model: TradingCostModel,
) -> ExecutionSimulator:
    return ExecutionSimulator(
        config=execution_simulator_config,
        event_bus=event_bus,
        clock=backtest_clock,
        cost_model=cost_model,
        random_seed=42,
    )


@pytest.fixture
def position_simulator(
    position_simulator_config: PositionSimulatorConfig,
    event_bus: InMemoryBacktestEventBus,
    backtest_clock: BacktestClock,
    cost_model: TradingCostModel,
) -> PositionSimulator:
    return PositionSimulator(
        config=position_simulator_config,
        event_bus=event_bus,
        clock=backtest_clock,
        cost_model=cost_model,
    )


# =============================================================================
# Historical data fixtures
# =============================================================================


@pytest.fixture
def sample_candles(start_time: datetime) -> list[HistoricalCandle]:
    candles: list[HistoricalCandle] = []
    base_price = 100.0

    for index in range(10):
        open_time = start_time + timedelta(minutes=index)
        close_time = open_time + timedelta(minutes=1)
        open_time_ms = timestamp_ms(open_time)
        close_time_ms = timestamp_ms(close_time)

        open_price = base_price + index
        close_price = open_price + 1.0

        candles.append(
            HistoricalCandle(
                exchange="binance",
                symbol="BTCUSDT",
                market_type="usdm_futures",
                timeframe="1m",
                timestamp_ms=close_time_ms,
                received_at_ms=close_time_ms,
                open_time_ms=open_time_ms,
                close_time_ms=close_time_ms,
                open=open_price,
                high=open_price + 2.0,
                low=open_price - 1.0,
                close=close_price,
                volume=1000.0 + index,
                quote_volume=(1000.0 + index) * close_price,
                trades_count=100 + index,
                is_closed=True,
                source="test",
                metadata={"index": index},
            )
        )

    return candles


@pytest.fixture
def sample_trades(start_time: datetime) -> list[HistoricalTrade]:
    trades: list[HistoricalTrade] = []

    for index in range(5):
        event_time = start_time + timedelta(minutes=index, seconds=10)

        trades.append(
            HistoricalTrade(
                exchange="binance",
                symbol="BTCUSDT",
                market_type="usdm_futures",
                timestamp_ms=timestamp_ms(event_time),
                received_at_ms=timestamp_ms(event_time),
                trade_id=f"trade_{index}",
                price=100.0 + index,
                quantity=0.1 + index * 0.01,
                side="buy" if index % 2 == 0 else "sell",
                aggressor_side="buy" if index % 2 == 0 else "sell",
                buyer_maker=index % 2 == 1,
                source="test",
                metadata={"index": index},
            )
        )

    return trades


@pytest.fixture
def sample_funding(start_time: datetime) -> list[HistoricalFundingRecord]:
    return [
        HistoricalFundingRecord(
            exchange="binance",
            symbol="BTCUSDT",
            market_type="usdm_futures",
            timestamp_ms=timestamp_ms(start_time + timedelta(minutes=30)),
            received_at_ms=timestamp_ms(start_time + timedelta(minutes=30)),
            funding_rate=0.0001,
            predicted_rate=0.00009,
            mark_price=105.0,
            index_price=104.8,
            next_funding_time_ms=timestamp_ms(start_time + timedelta(hours=8)),
            source="test",
            metadata={},
        )
    ]


@pytest.fixture
def sample_open_interest(start_time: datetime) -> list[HistoricalOpenInterestRecord]:
    return [
        HistoricalOpenInterestRecord(
            exchange="binance",
            symbol="BTCUSDT",
            market_type="usdm_futures",
            timestamp_ms=timestamp_ms(start_time + timedelta(minutes=5)),
            received_at_ms=timestamp_ms(start_time + timedelta(minutes=5)),
            open_interest=100_000.0,
            open_interest_value=10_000_000.0,
            mark_price=101.0,
            source="test",
            metadata={},
        )
    ]


@pytest.fixture
def sample_orderbook(start_time: datetime) -> HistoricalOrderBookSnapshot:
    event_time_ms = timestamp_ms(start_time + timedelta(minutes=1))

    return HistoricalOrderBookSnapshot(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        timestamp_ms=event_time_ms,
        received_at_ms=event_time_ms,
        bids=[
            HistoricalOrderBookLevel(price=99.9, quantity=1.0),
            HistoricalOrderBookLevel(price=99.8, quantity=2.0),
        ],
        asks=[
            HistoricalOrderBookLevel(price=100.1, quantity=1.0),
            HistoricalOrderBookLevel(price=100.2, quantity=2.0),
        ],
        sequence=1,
        depth=2,
        source="test",
        metadata={},
    )


@pytest.fixture
def sample_dataset(
    sample_candles: list[HistoricalCandle],
    backtest_period: BacktestPeriod,
) -> BacktestDataset:
    events: list[BacktestEvent] = []

    for sequence, candle in enumerate(sample_candles):
        events.append(
            build_replay_event_from_record(
                candle,
                data_type=BacktestDataType.CANDLES,
                period=backtest_period,
                run_id="test_run",
                sequence=sequence,
            )
        )

    dataset = BacktestDataset(
        events=events,
        ordering=ReplayOrdering.TIMESTAMP_THEN_PRIORITY,
        replay_mode=ReplayMode.FULL_RUN,
        metadata={"source": "conftest"},
    )
    dataset.sort_events()
    dataset.info = BacktestDatasetInfo(
        period=backtest_period,
        instruments=[
            BacktestInstrument(
                exchange="binance",
                symbol="BTCUSDT",
                market_type="usdm_futures",
            )
        ],
        data_types={BacktestDataType.CANDLES},
        total_events=len(events),
        first_event_time=events[0].event_time,
        last_event_time=events[-1].event_time,
        metadata={"source": "conftest"},
    )
    return dataset


@pytest.fixture
def multi_stream_dataset(
    sample_candles: list[HistoricalCandle],
    sample_trades: list[HistoricalTrade],
    sample_funding: list[HistoricalFundingRecord],
    sample_open_interest: list[HistoricalOpenInterestRecord],
    backtest_period: BacktestPeriod,
) -> BacktestDataset:
    events: list[BacktestEvent] = []
    sequence = 0

    streams: list[tuple[BacktestDataType, list[Any]]] = [
        (BacktestDataType.CANDLES, sample_candles),
        (BacktestDataType.TRADES, sample_trades),
        (BacktestDataType.FUNDING, sample_funding),
        (BacktestDataType.OPEN_INTEREST, sample_open_interest),
    ]

    for data_type, records in streams:
        for record in records:
            events.append(
                build_replay_event_from_record(
                    record,
                    data_type=data_type,
                    period=backtest_period,
                    run_id="test_run",
                    sequence=sequence,
                )
            )
            sequence += 1

    dataset = BacktestDataset(
        events=events,
        ordering=ReplayOrdering.TIMESTAMP_THEN_PRIORITY,
        replay_mode=ReplayMode.FULL_RUN,
        metadata={"source": "conftest_multi_stream"},
    )
    dataset.sort_events()
    dataset.info = BacktestDatasetInfo(
        period=backtest_period,
        instruments=[
            BacktestInstrument(
                exchange="binance",
                symbol="BTCUSDT",
                market_type="usdm_futures",
            )
        ],
        data_types={
            BacktestDataType.CANDLES,
            BacktestDataType.TRADES,
            BacktestDataType.FUNDING,
            BacktestDataType.OPEN_INTEREST,
        },
        total_events=len(events),
        first_event_time=dataset.events[0].event_time,
        last_event_time=dataset.events[-1].event_time,
        metadata={"source": "conftest_multi_stream"},
    )
    return dataset


# =============================================================================
# Temporary historical file fixtures
# =============================================================================


@pytest.fixture
def temp_history_dir(tmp_path: Path, sample_candles: list[HistoricalCandle]) -> Path:
    """
    Create local CSV historical candles in the DataLoader default folder layout.
    """

    path = (
        tmp_path
        / "history"
        / "binance"
        / "usdm_futures"
        / "candles"
        / "BTCUSDT"
        / "1m"
        / "BTCUSDT_1m.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "exchange",
                "symbol",
                "market_type",
                "timeframe",
                "timestamp_ms",
                "received_at_ms",
                "open_time_ms",
                "close_time_ms",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "trades_count",
                "is_closed",
            ],
        )
        writer.writeheader()

        for candle in sample_candles:
            writer.writerow(
                {
                    "exchange": candle.exchange,
                    "symbol": candle.symbol,
                    "market_type": candle.market_type,
                    "timeframe": candle.timeframe,
                    "timestamp_ms": candle.timestamp_ms,
                    "received_at_ms": candle.received_at_ms,
                    "open_time_ms": candle.open_time_ms,
                    "close_time_ms": candle.close_time_ms,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                    "quote_volume": candle.quote_volume,
                    "trades_count": candle.trades_count,
                    "is_closed": candle.is_closed,
                }
            )

    return tmp_path / "history"


@pytest.fixture
def temp_history_data_loader_config(
    data_loader_config: DataLoaderConfig,
    temp_history_dir: Path,
) -> DataLoaderConfig:
    data_loader_config.data_dir = str(temp_history_dir)
    return data_loader_config


# =============================================================================
# Simulated execution / position records
# =============================================================================


@pytest.fixture
def sample_long_fill(start_time: datetime) -> SimulatedFill:
    return SimulatedFill(
        order_id="order_1",
        run_id="test_run",
        signal_id="signal_1",
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        side="buy",
        price=100.0,
        quantity=1.0,
        notional=100.0,
        fee=0.04,
        fee_asset="USDT",
        slippage=0.02,
        slippage_bps=2.0,
        liquidity_type="taker",
        timestamp_ms=timestamp_ms(start_time + timedelta(minutes=1)),
        metadata={
            "strategy_name": "fake_strategy",
            "order_type": "market",
        },
    )


@pytest.fixture
def sample_short_fill(start_time: datetime) -> SimulatedFill:
    return SimulatedFill(
        order_id="order_2",
        run_id="test_run",
        signal_id="signal_2",
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        side="sell",
        price=100.0,
        quantity=1.0,
        notional=100.0,
        fee=0.04,
        fee_asset="USDT",
        slippage=0.02,
        slippage_bps=2.0,
        liquidity_type="taker",
        timestamp_ms=timestamp_ms(start_time + timedelta(minutes=1)),
        metadata={
            "strategy_name": "fake_strategy",
            "order_type": "market",
        },
    )


@pytest.fixture
def sample_trade(start_time: datetime) -> SimulatedTrade:
    return SimulatedTrade(
        run_id="test_run",
        position_id="position_1",
        signal_id="signal_1",
        strategy_name="fake_strategy",
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        side="long",
        quantity=1.0,
        entry_price=100.0,
        exit_price=110.0,
        opened_at_ms=timestamp_ms(start_time),
        closed_at_ms=timestamp_ms(start_time + timedelta(minutes=10)),
        gross_pnl=10.0,
        net_pnl=9.9,
        pnl_pct=10.0,
        r_multiple=2.0,
        fees=0.08,
        slippage=0.02,
        funding=0.0,
        close_reason="take_profit",
        metadata={
            "market_regime": "trend",
            "source_features": ["breakout", "volume"],
        },
    )


@pytest.fixture
def losing_trade(start_time: datetime) -> SimulatedTrade:
    return SimulatedTrade(
        run_id="test_run",
        position_id="position_2",
        signal_id="signal_2",
        strategy_name="fake_strategy",
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        side="long",
        quantity=1.0,
        entry_price=100.0,
        exit_price=95.0,
        opened_at_ms=timestamp_ms(start_time + timedelta(minutes=20)),
        closed_at_ms=timestamp_ms(start_time + timedelta(minutes=30)),
        gross_pnl=-5.0,
        net_pnl=-5.1,
        pnl_pct=-5.0,
        r_multiple=-1.0,
        fees=0.08,
        slippage=0.02,
        funding=0.0,
        close_reason="stop_loss",
        metadata={
            "market_regime": "range",
            "source_features": ["fake_breakout"],
        },
    )


@pytest.fixture
def sample_equity_curve(start_time: datetime) -> list[SimulatedEquityPoint]:
    values = [10_000.0, 10_100.0, 10_050.0, 10_200.0, 10_150.0, 10_300.0]

    return [
        SimulatedEquityPoint(
            timestamp_ms=timestamp_ms(start_time + timedelta(minutes=index)),
            equity=value,
            balance=value,
            available_balance=value,
            realized_pnl=value - 10_000.0,
            drawdown=max(0.0, max(values[: index + 1]) - value),
            drawdown_pct=0.0,
            open_positions=0,
            source="test",
        )
        for index, value in enumerate(values)
    ]


@pytest.fixture
def sample_signal_records(start_time: datetime) -> list[BacktestSignalRecord]:
    return [
        BacktestSignalRecord(
            run_id="test_run",
            signal_id="signal_1",
            strategy_name="fake_strategy",
            symbol="BTCUSDT",
            timeframe="1m",
            side="buy",
            setup_type="breakout",
            confidence=0.8,
            strength=0.75,
            generated_at_ms=timestamp_ms(start_time),
            confirmed_at_ms=timestamp_ms(start_time + timedelta(seconds=1)),
            opened_at_ms=timestamp_ms(start_time + timedelta(seconds=2)),
            closed_at_ms=timestamp_ms(start_time + timedelta(minutes=10)),
            outcome=SignalOutcome.POSITION_CLOSED_WIN,
            pnl=9.9,
            r_multiple=2.0,
            payload={
                "source_features": ["breakout", "volume"],
                "market_regime": "trend",
            },
            metadata={
                "market_regime": "trend",
                "source_features": ["breakout", "volume"],
            },
        ),
        BacktestSignalRecord(
            run_id="test_run",
            signal_id="signal_2",
            strategy_name="fake_strategy",
            symbol="BTCUSDT",
            timeframe="1m",
            side="buy",
            setup_type="breakout",
            confidence=0.6,
            strength=0.55,
            generated_at_ms=timestamp_ms(start_time + timedelta(minutes=20)),
            outcome=SignalOutcome.BLOCKED_BY_RISK,
            payload={
                "source_features": ["fake_breakout"],
                "market_regime": "range",
            },
            metadata={
                "market_regime": "range",
                "source_features": ["fake_breakout"],
            },
        ),
    ]


@pytest.fixture
def sample_risk_decisions(start_time: datetime) -> list[BacktestRiskDecisionRecord]:
    return [
        BacktestRiskDecisionRecord(
            run_id="test_run",
            signal_id="signal_1",
            strategy_name="fake_strategy",
            symbol="BTCUSDT",
            timestamp_ms=timestamp_ms(start_time + timedelta(seconds=1)),
            approved=True,
            blocked=False,
            reason=None,
            risk_amount=100.0,
            final_size=1.0,
            final_leverage=2.0,
            final_margin=50.0,
            final_notional=100.0,
            reservation_id="reservation_1",
            payload={},
            metadata={},
        ),
        BacktestRiskDecisionRecord(
            run_id="test_run",
            signal_id="signal_2",
            strategy_name="fake_strategy",
            symbol="BTCUSDT",
            timestamp_ms=timestamp_ms(start_time + timedelta(minutes=20)),
            approved=False,
            blocked=True,
            reason="max_daily_loss",
            payload={},
            metadata={},
        ),
    ]


# =============================================================================
# Fake production-like pipeline components
# =============================================================================


class FakeStrategyEngine:
    """
    Minimal strategy-engine replacement for integration tests.

    It listens to market.candle and emits signal.generated once per configured
    candle direction.
    """

    def __init__(
        self,
        event_bus: InMemoryBacktestEventBus,
        *,
        strategy_name: str = "fake_strategy",
        emit_side: str = "buy",
        max_signals: int = 1,
    ) -> None:
        self.event_bus = event_bus
        self.strategy_name = strategy_name
        self.emit_side = emit_side
        self.max_signals = max_signals
        self.generated = 0
        self.started = False
        self.registered = False

    def register(self) -> None:
        if self.registered:
            return
        self.event_bus.subscribe("market.candle", self._on_market_candle)
        self.registered = True

    async def start(self) -> None:
        self.started = True
        self.register()

    async def stop(self) -> None:
        self.started = False

    async def _on_market_candle(self, payload: dict[str, Any]) -> None:
        if not self.started:
            return

        if self.generated >= self.max_signals:
            return

        if bool(payload.get("is_warmup")):
            return

        self.generated += 1

        await self.event_bus.emit(
            "signal.generated",
            {
                "run_id": payload.get("run_id", "test_run"),
                "signal_id": f"signal_{self.generated}",
                "strategy_name": self.strategy_name,
                "symbol": payload["symbol"],
                "exchange": payload.get("exchange", "binance"),
                "market_type": payload.get("market_type", "usdm_futures"),
                "timeframe": payload.get("timeframe", "1m"),
                "side": self.emit_side,
                "direction": self.emit_side,
                "confidence": 0.8,
                "strength": 0.75,
                "quantity": 1.0,
                "order_type": "market",
                "timestamp_ms": payload.get("timestamp_ms"),
                "entry_price": payload.get("close"),
                "metadata": {
                    "source": "FakeStrategyEngine",
                    "market_regime": "trend",
                    "source_features": ["fake_signal"],
                },
            },
        )


class FakeRiskManagerApprove:
    """
    Fake RiskManager that approves every signal.generated event.
    """

    def __init__(self, event_bus: InMemoryBacktestEventBus) -> None:
        self.event_bus = event_bus
        self.started = False
        self.registered = False
        self.decisions = 0

    def register(self) -> None:
        if self.registered:
            return
        self.event_bus.subscribe("signal.generated", self._on_signal_generated)
        self.registered = True

    async def start(self) -> None:
        self.started = True
        self.register()

    async def stop(self) -> None:
        self.started = False

    async def _on_signal_generated(self, payload: dict[str, Any]) -> None:
        if not self.started:
            return

        self.decisions += 1

        await self.event_bus.emit(
            "signal.confirmed",
            {
                **payload,
                "approved": True,
                "blocked": False,
                "final_size": float(payload.get("quantity") or 1.0),
                "final_leverage": 2.0,
                "final_margin": 50.0,
                "final_notional": 100.0,
                "final_risk_amount": 100.0,
                "reservation_id": f"reservation_{self.decisions}",
                "timestamp_ms": payload.get("timestamp_ms"),
                "metadata": {
                    **dict(payload.get("metadata") or {}),
                    "source": "FakeRiskManagerApprove",
                },
            },
        )


class FakeRiskManagerBlock:
    """
    Fake RiskManager that blocks every signal.generated event.
    """

    def __init__(self, event_bus: InMemoryBacktestEventBus, reason: str = "test_block") -> None:
        self.event_bus = event_bus
        self.reason = reason
        self.started = False
        self.registered = False
        self.decisions = 0

    def register(self) -> None:
        if self.registered:
            return
        self.event_bus.subscribe("signal.generated", self._on_signal_generated)
        self.registered = True

    async def start(self) -> None:
        self.started = True
        self.register()

    async def stop(self) -> None:
        self.started = False

    async def _on_signal_generated(self, payload: dict[str, Any]) -> None:
        if not self.started:
            return

        self.decisions += 1

        await self.event_bus.emit(
            "risk.position_blocked",
            {
                **payload,
                "approved": False,
                "blocked": True,
                "reason": self.reason,
                "block_reason": self.reason,
                "timestamp_ms": payload.get("timestamp_ms"),
                "metadata": {
                    **dict(payload.get("metadata") or {}),
                    "source": "FakeRiskManagerBlock",
                },
            },
        )


class FakeFailingComponent:
    """
    Component that fails on start; useful for StrategyTester failure tests.
    """

    def register(self) -> None:
        return None

    async def start(self) -> None:
        raise RuntimeError("fake component failure")

    async def stop(self) -> None:
        return None


@pytest.fixture
def fake_strategy_engine(event_bus: InMemoryBacktestEventBus) -> FakeStrategyEngine:
    return FakeStrategyEngine(event_bus)


@pytest.fixture
def fake_strategy_engine_two_signals(event_bus: InMemoryBacktestEventBus) -> FakeStrategyEngine:
    return FakeStrategyEngine(
        event_bus,
        max_signals=2,
    )


@pytest.fixture
def fake_short_strategy_engine(event_bus: InMemoryBacktestEventBus) -> FakeStrategyEngine:
    return FakeStrategyEngine(
        event_bus,
        emit_side="sell",
        max_signals=1,
    )


@pytest.fixture
def fake_risk_manager_approve(event_bus: InMemoryBacktestEventBus) -> FakeRiskManagerApprove:
    return FakeRiskManagerApprove(event_bus)


@pytest.fixture
def fake_risk_manager_block(event_bus: InMemoryBacktestEventBus) -> FakeRiskManagerBlock:
    return FakeRiskManagerBlock(event_bus)


@pytest.fixture
def fake_failing_component() -> FakeFailingComponent:
    return FakeFailingComponent()


# =============================================================================
# Fake tester factory for optimizer / walk-forward tests
# =============================================================================


class FakeStrategyTester:
    """
    Minimal StrategyTester replacement for optimizer/walk-forward tests.

    It returns deterministic BacktestResult objects without running the full
    pipeline. This keeps optimizer/walk-forward unit tests fast and isolated.
    """

    def __init__(
        self,
        config: BacktestConfig,
        dataset: BacktestDataset,
        *,
        net_profit_pct: float | None = None,
        should_fail: bool = False,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.net_profit_pct = net_profit_pct
        self.should_fail = should_fail

    async def run(self) -> Any:
        from backtesting.models import BacktestResult

        if self.should_fail:
            result = BacktestResult(
                run_name=self.config.run_name,
                mode=self.config.mode,
                status=BacktestStatus.FAILED,
                period=self.config.period(),
                initial_balance=self.config.initial_balance,
                final_balance=self.config.initial_balance,
                final_equity=self.config.initial_balance,
                error="fake failure",
            )
            return result

        profit_pct = self.net_profit_pct

        if profit_pct is None:
            leverage = float(getattr(self.config.position_simulator, "default_leverage", 1.0))
            slippage = float(getattr(self.config.cost_model, "fixed_slippage_bps", 0.0))
            profit_pct = leverage * 2.0 - slippage * 0.1

        net_profit = self.config.initial_balance * profit_pct / 100.0
        final_equity = self.config.initial_balance + net_profit

        summary = PerformanceSummary(
            key="system",
            initial_balance=self.config.initial_balance,
            final_balance=final_equity,
            final_equity=final_equity,
            net_profit=net_profit,
            net_profit_pct=profit_pct,
            gross_profit=max(net_profit, 0.0),
            gross_loss=abs(min(net_profit, 0.0)),
            profit_factor=2.0 if net_profit >= 0 else 0.5,
            expectancy=net_profit,
            total_trades=2,
            win_rate=50.0,
            max_drawdown=100.0,
            max_drawdown_pct=1.0,
            recovery_factor=net_profit / 100.0 if net_profit else 0.0,
        )

        portfolio = PortfolioBacktestResult(
            summary=summary,
            trade_stats=TradeStats(
                total_trades=2,
                closed_trades=2,
                winning_trades=1,
                losing_trades=1,
                win_rate=50.0,
                net_profit=net_profit,
            ),
        )

        result = BacktestResult(
            run_name=self.config.run_name,
            mode=self.config.mode,
            status=BacktestStatus.COMPLETED,
            period=self.config.period(),
            initial_balance=self.config.initial_balance,
            final_balance=final_equity,
            final_equity=final_equity,
            portfolio=portfolio,
        )
        result.mark_completed()
        return result


@pytest.fixture
def fake_tester_factory() -> Any:
    def factory(config: BacktestConfig, dataset: BacktestDataset) -> FakeStrategyTester:
        return FakeStrategyTester(config, dataset)

    return factory


@pytest.fixture
def failing_fake_tester_factory() -> Any:
    def factory(config: BacktestConfig, dataset: BacktestDataset) -> FakeStrategyTester:
        return FakeStrategyTester(config, dataset, should_fail=True)

    return factory