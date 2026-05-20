"""
Backtesting configuration models.

This module defines typed dataclass configs for the offline backtesting package:
history download, data loading, market replay, cost simulation, simulated
execution, simulated positions, metrics, reports, walk-forward and optimization.

Important architectural rule:
backtesting config does not replace production strategy/risk/execution configs.
It only aggregates them or stores backtesting-specific simulation settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from backtesting.enums import (
    BacktestDataType,
    BacktestMode,
    CandleExecutionPath,
    CommissionModel,
    DataAlignmentPolicy,
    DataGapPolicy,
    DataValidationLevel,
    EquityUpdateMode,
    FillModel,
    FundingSimulationMode,
    HistoricalDataFormat,
    LatencyModel,
    LiquidityModel,
    OptimizationDirection,
    OptimizationMethod,
    OptimizationMetric,
    OverfittingCheckMode,
    PnLAccountingMode,
    PositionAccountingMode,
    ReplayMode,
    ReplayOrdering,
    ReplaySpeed,
    ReportFormat,
    ReportSection,
    SlippageModel,
    WalkForwardMode,
    WarmupPolicy,
)
from backtesting.exceptions import BacktestConfigurationError
from backtesting.models import BacktestInstrument, BacktestPeriod, ensure_aware_utc


# ============================================================================
# Helpers
# ============================================================================


def _ensure_positive(
    value: int | float,
    name: str,
    *,
    allow_zero: bool = False,
) -> None:
    if allow_zero:
        valid = value >= 0
    else:
        valid = value > 0

    if not valid:
        raise BacktestConfigurationError(
            f"{name} must be {'non-negative' if allow_zero else 'positive'}.",
            details={name: value},
        )


def _ensure_between(
    value: int | float,
    name: str,
    minimum: int | float,
    maximum: int | float,
) -> None:
    if not minimum <= value <= maximum:
        raise BacktestConfigurationError(
            f"{name} must be between {minimum} and {maximum}.",
            details={
                name: value,
                "minimum": minimum,
                "maximum": maximum,
            },
        )


def _ensure_dir_path(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser()
    if not str(path):
        raise BacktestConfigurationError(f"{name} cannot be empty.")
    return path


def _normalize_symbols(symbols: list[str]) -> list[str]:
    normalized = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
    if not normalized:
        raise BacktestConfigurationError("At least one symbol is required.")
    return list(dict.fromkeys(normalized))


def _normalize_timeframes(timeframes: list[str]) -> list[str]:
    normalized = [timeframe.strip() for timeframe in timeframes if timeframe.strip()]
    if not normalized:
        raise BacktestConfigurationError("At least one timeframe is required.")
    return list(dict.fromkeys(normalized))


# ============================================================================
# History downloader
# ============================================================================


@dataclass(slots=True)
class HistoryDownloaderConfig:
    """
    Configuration for history_downloader.py.

    Downloader is allowed to call exchange REST APIs to fetch historical data.
    During actual replay, live exchange calls must not be used.
    """

    enabled: bool = False

    exchange: str = "binance"
    market_type: str = "usdm_futures"
    symbols: list[str] = field(default_factory=list)
    timeframes: list[str] = field(default_factory=lambda: ["1m"])

    data_types: set[BacktestDataType] = field(
        default_factory=lambda: {
            BacktestDataType.CANDLES,
            BacktestDataType.FUNDING,
            BacktestDataType.OPEN_INTEREST,
        }
    )

    output_dir: str | Path = "data/history"
    output_format: HistoricalDataFormat = HistoricalDataFormat.PARQUET

    overwrite_existing: bool = False
    skip_existing: bool = True
    validate_after_download: bool = True

    request_limit: int = 1000
    max_retries: int = 5
    retry_delay_seconds: float = 1.0
    request_timeout_seconds: float = 30.0
    rate_limit_sleep_seconds: float = 0.25

    candle_limit_per_request: int = 1000
    trade_limit_per_request: int = 1000
    funding_limit_per_request: int = 1000
    open_interest_limit_per_request: int = 500

    include_mark_price: bool = True
    include_index_price: bool = True
    include_liquidations: bool = False
    include_orderbook_snapshots: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.exchange = self.exchange.lower()
        self.market_type = self.market_type.lower()
        self.output_dir = _ensure_dir_path(self.output_dir, "HistoryDownloaderConfig.output_dir")

        if self.enabled:
            self.symbols = _normalize_symbols(self.symbols)
            self.timeframes = _normalize_timeframes(self.timeframes)

        if not self.data_types:
            raise BacktestConfigurationError("HistoryDownloaderConfig.data_types cannot be empty.")

        _ensure_positive(self.request_limit, "HistoryDownloaderConfig.request_limit")
        _ensure_positive(self.max_retries, "HistoryDownloaderConfig.max_retries", allow_zero=True)
        _ensure_positive(
            self.retry_delay_seconds,
            "HistoryDownloaderConfig.retry_delay_seconds",
            allow_zero=True,
        )
        _ensure_positive(
            self.request_timeout_seconds,
            "HistoryDownloaderConfig.request_timeout_seconds",
        )
        _ensure_positive(
            self.rate_limit_sleep_seconds,
            "HistoryDownloaderConfig.rate_limit_sleep_seconds",
            allow_zero=True,
        )


# ============================================================================
# Data loader
# ============================================================================


@dataclass(slots=True)
class DataLoaderConfig:
    """
    Configuration for data_loader.py.
    """

    data_dir: str | Path = "data/history"
    input_format: HistoricalDataFormat = HistoricalDataFormat.PARQUET

    exchange: str = "binance"
    market_type: str = "usdm_futures"
    symbols: list[str] = field(default_factory=list)
    timeframes: list[str] = field(default_factory=lambda: ["1m"])

    data_types: set[BacktestDataType] = field(
        default_factory=lambda: {
            BacktestDataType.CANDLES,
            BacktestDataType.FUNDING,
            BacktestDataType.OPEN_INTEREST,
        }
    )

    validation_level: DataValidationLevel = DataValidationLevel.BASIC
    gap_policy: DataGapPolicy = DataGapPolicy.WARN
    alignment_policy: DataAlignmentPolicy = DataAlignmentPolicy.EVENT_TIME

    require_candles: bool = True
    require_trades: bool = False
    require_orderbook: bool = False
    require_funding: bool = False
    require_open_interest: bool = False

    allow_empty_optional_streams: bool = True
    drop_duplicate_events: bool = True
    sort_events: bool = True

    max_allowed_gap_seconds: int = 60 * 60
    max_events: int | None = None

    preload_into_memory: bool = True
    chunk_size: int = 100_000

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.exchange = self.exchange.lower()
        self.market_type = self.market_type.lower()
        self.data_dir = _ensure_dir_path(self.data_dir, "DataLoaderConfig.data_dir")
        self.symbols = _normalize_symbols(self.symbols)
        self.timeframes = _normalize_timeframes(self.timeframes)

        if not self.data_types:
            raise BacktestConfigurationError("DataLoaderConfig.data_types cannot be empty.")

        _ensure_positive(
            self.max_allowed_gap_seconds,
            "DataLoaderConfig.max_allowed_gap_seconds",
            allow_zero=True,
        )
        _ensure_positive(self.chunk_size, "DataLoaderConfig.chunk_size")

        if self.max_events is not None:
            _ensure_positive(self.max_events, "DataLoaderConfig.max_events")


# ============================================================================
# Market replay
# ============================================================================


@dataclass(slots=True)
class MarketReplayConfig:
    """
    Configuration for market_replay.py.
    """

    replay_mode: ReplayMode = ReplayMode.FULL_RUN
    replay_speed: ReplaySpeed = ReplaySpeed.MAX_SPEED
    ordering: ReplayOrdering = ReplayOrdering.TIMESTAMP_THEN_PRIORITY

    warmup_policy: WarmupPolicy = WarmupPolicy.REPLAY_WITH_TRADING_DISABLED
    emit_warmup_events: bool = True
    mark_warmup_payloads: bool = True

    batch_events_by_timestamp: bool = True
    max_batch_size: int = 10_000
    yield_every_events: int = 50_000

    deterministic_replay: bool = True
    fail_on_emit_error: bool = True
    continue_on_invalid_event: bool = False

    emit_market_candles: bool = True
    emit_market_trades: bool = True
    emit_market_orderbook: bool = True
    emit_market_funding: bool = True
    emit_market_open_interest: bool = True
    emit_market_liquidations: bool = True

    market_candle_topic: str = "market.candle"
    market_trade_topic: str = "market.trade"
    market_orderbook_topic: str = "market.orderbook"
    market_funding_topic: str = "market.funding"
    market_open_interest_topic: str = "market.open_interest"
    market_liquidation_topic: str = "market.liquidation"

    emit_replay_lifecycle_events: bool = True
    replay_started_topic: str = "system.backtest.replay.started"
    replay_finished_topic: str = "system.backtest.replay.finished"
    replay_failed_topic: str = "system.backtest.replay.failed"
    replay_progress_topic: str = "system.backtest.replay.progress"

    progress_interval_events: int = 100_000
    progress_interval_seconds: float = 5.0

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _ensure_positive(self.max_batch_size, "MarketReplayConfig.max_batch_size")
        _ensure_positive(
            self.yield_every_events,
            "MarketReplayConfig.yield_every_events",
            allow_zero=True,
        )
        _ensure_positive(
            self.progress_interval_events,
            "MarketReplayConfig.progress_interval_events",
            allow_zero=True,
        )
        _ensure_positive(
            self.progress_interval_seconds,
            "MarketReplayConfig.progress_interval_seconds",
            allow_zero=True,
        )

        required_topics = {
            "market_candle_topic": self.market_candle_topic,
            "market_trade_topic": self.market_trade_topic,
            "market_orderbook_topic": self.market_orderbook_topic,
            "market_funding_topic": self.market_funding_topic,
            "market_open_interest_topic": self.market_open_interest_topic,
            "market_liquidation_topic": self.market_liquidation_topic,
        }

        for name, topic in required_topics.items():
            if not topic:
                raise BacktestConfigurationError(f"MarketReplayConfig.{name} cannot be empty.")


# ============================================================================
# Cost models
# ============================================================================


@dataclass(slots=True)
class CostModelConfig:
    """
    Configuration for cost_models.py.
    """

    commission_model: CommissionModel = CommissionModel.MAKER_TAKER
    slippage_model: SlippageModel = SlippageModel.FIXED_BPS
    funding_mode: FundingSimulationMode = FundingSimulationMode.APPLY_ON_FUNDING_TIMESTAMP

    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 4.0
    default_fee_bps: float = 4.0
    fixed_fee: float = 0.0

    fixed_slippage_bps: float = 2.0
    fixed_slippage_price: float = 0.0
    spread_slippage_fraction: float = 0.5
    volatility_slippage_multiplier: float = 0.1
    volume_slippage_multiplier: float = 1.0
    adverse_selection_bps: float = 0.0

    include_commissions: bool = True
    include_slippage: bool = True
    include_spread_cost: bool = True
    include_funding: bool = True
    include_liquidation_penalty: bool = True

    funding_interval_hours: int = 8
    fallback_funding_rate: float = 0.0

    liquidation_penalty_bps: float = 0.0

    quote_currency: str = "USDT"

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _ensure_positive(self.maker_fee_bps, "CostModelConfig.maker_fee_bps", allow_zero=True)
        _ensure_positive(self.taker_fee_bps, "CostModelConfig.taker_fee_bps", allow_zero=True)
        _ensure_positive(self.default_fee_bps, "CostModelConfig.default_fee_bps", allow_zero=True)
        _ensure_positive(self.fixed_fee, "CostModelConfig.fixed_fee", allow_zero=True)

        _ensure_positive(
            self.fixed_slippage_bps,
            "CostModelConfig.fixed_slippage_bps",
            allow_zero=True,
        )
        _ensure_positive(
            self.fixed_slippage_price,
            "CostModelConfig.fixed_slippage_price",
            allow_zero=True,
        )
        _ensure_between(
            self.spread_slippage_fraction,
            "CostModelConfig.spread_slippage_fraction",
            0.0,
            10.0,
        )
        _ensure_positive(
            self.volatility_slippage_multiplier,
            "CostModelConfig.volatility_slippage_multiplier",
            allow_zero=True,
        )
        _ensure_positive(
            self.volume_slippage_multiplier,
            "CostModelConfig.volume_slippage_multiplier",
            allow_zero=True,
        )
        _ensure_positive(
            self.adverse_selection_bps,
            "CostModelConfig.adverse_selection_bps",
            allow_zero=True,
        )
        _ensure_positive(
            self.funding_interval_hours,
            "CostModelConfig.funding_interval_hours",
        )
        _ensure_positive(
            self.liquidation_penalty_bps,
            "CostModelConfig.liquidation_penalty_bps",
            allow_zero=True,
        )

        if not self.quote_currency:
            raise BacktestConfigurationError("CostModelConfig.quote_currency cannot be empty.")


# ============================================================================
# Execution simulator
# ============================================================================


@dataclass(slots=True)
class ExecutionSimulatorConfig:
    """
    Configuration for execution_simulator.py.

    This simulator replaces live exchange execution during backtests.
    It must only execute risk-confirmed signals or explicit risk close/reduce
    requests.
    """

    enabled: bool = True

    exchange: str = "binance"
    market_type: str = "usdm_futures"

    fill_model: FillModel = FillModel.NEXT_CANDLE_OPEN
    candle_execution_path: CandleExecutionPath = CandleExecutionPath.CONSERVATIVE
    liquidity_model: LiquidityModel = LiquidityModel.CANDLE_VOLUME_PERCENT
    latency_model: LatencyModel = LatencyModel.NONE

    allow_market_orders: bool = True
    allow_limit_orders: bool = True
    allow_stop_orders: bool = True
    allow_reduce_only: bool = True
    allow_partial_fills: bool = True
    allow_order_rejections: bool = True

    max_volume_participation_pct: float = 10.0
    min_fill_ratio: float = 0.05
    partial_fill_probability: float = 0.0

    fixed_latency_ms: int = 0
    random_latency_min_ms: int = 0
    random_latency_max_ms: int = 0

    reject_if_no_price: bool = True
    reject_if_no_liquidity: bool = True
    reject_if_price_outside_candle: bool = True

    market_order_topic: str = "execution.order_submitted"
    order_rejected_topic: str = "execution.order_rejected"
    order_failed_topic: str = "execution.order_failed"
    order_cancelled_topic: str = "execution.order_cancelled"
    order_filled_topic: str = "execution.order_filled"
    order_partially_filled_topic: str = "execution.order_partially_filled"

    listen_signal_confirmed: bool = True
    signal_confirmed_topic: str = "signal.confirmed"
    position_close_requested_topic: str = "risk.position_close_requested"
    position_reduce_requested_topic: str = "risk.position_reduce_requested"
    kill_switch_topic: str = "risk.kill_switch"

    emit_execution_events: bool = True
    record_orders: bool = True
    record_fills: bool = True

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.exchange = self.exchange.lower()
        self.market_type = self.market_type.lower()

        _ensure_between(
            self.max_volume_participation_pct,
            "ExecutionSimulatorConfig.max_volume_participation_pct",
            0.0,
            100.0,
        )
        _ensure_between(
            self.min_fill_ratio,
            "ExecutionSimulatorConfig.min_fill_ratio",
            0.0,
            1.0,
        )
        _ensure_between(
            self.partial_fill_probability,
            "ExecutionSimulatorConfig.partial_fill_probability",
            0.0,
            1.0,
        )
        _ensure_positive(
            self.fixed_latency_ms,
            "ExecutionSimulatorConfig.fixed_latency_ms",
            allow_zero=True,
        )
        _ensure_positive(
            self.random_latency_min_ms,
            "ExecutionSimulatorConfig.random_latency_min_ms",
            allow_zero=True,
        )
        _ensure_positive(
            self.random_latency_max_ms,
            "ExecutionSimulatorConfig.random_latency_max_ms",
            allow_zero=True,
        )

        if self.random_latency_max_ms < self.random_latency_min_ms:
            raise BacktestConfigurationError(
                "ExecutionSimulatorConfig.random_latency_max_ms cannot be less than random_latency_min_ms.",
                details={
                    "random_latency_min_ms": self.random_latency_min_ms,
                    "random_latency_max_ms": self.random_latency_max_ms,
                },
            )

        if not self.allow_market_orders and not self.allow_limit_orders and not self.allow_stop_orders:
            raise BacktestConfigurationError(
                "At least one simulated order type must be allowed."
            )


# ============================================================================
# Position simulator
# ============================================================================


@dataclass(slots=True)
class PositionSimulatorConfig:
    """
    Configuration for position_simulator.py.
    """

    enabled: bool = True

    initial_balance: float = 10_000.0
    quote_currency: str = "USDT"

    position_accounting_mode: PositionAccountingMode = PositionAccountingMode.NETTING
    pnl_accounting_mode: PnLAccountingMode = PnLAccountingMode.REALIZED_AND_UNREALIZED
    equity_update_mode: EquityUpdateMode = EquityUpdateMode.ON_CANDLE_CLOSE

    default_leverage: float = 1.0
    max_leverage: float = 20.0
    maintenance_margin_rate: float = 0.005
    liquidation_buffer_bps: float = 10.0

    allow_hedge_positions: bool = False
    allow_position_reversal: bool = True
    close_opposite_position_on_reverse: bool = True

    enable_mark_to_market: bool = True
    enable_stop_loss: bool = True
    enable_take_profit: bool = True
    enable_trailing_stop: bool = True
    enable_liquidation_check: bool = True
    enable_funding_application: bool = True

    mark_price_source: str = "candle_close"
    equity_update_interval_seconds: int = 60

    listen_order_filled: bool = True
    listen_order_partially_filled: bool = True
    listen_position_close_requested: bool = True
    listen_position_reduce_requested: bool = True

    order_filled_topic: str = "execution.order_filled"
    order_partially_filled_topic: str = "execution.order_partially_filled"
    position_opened_topic: str = "position.opened"
    position_updated_topic: str = "position.updated"
    position_closed_topic: str = "position.closed"
    position_liquidated_topic: str = "position.liquidated"

    emit_position_events: bool = True
    record_positions: bool = True
    record_trades: bool = True
    record_equity_curve: bool = True

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _ensure_positive(self.initial_balance, "PositionSimulatorConfig.initial_balance")
        _ensure_positive(self.default_leverage, "PositionSimulatorConfig.default_leverage")
        _ensure_positive(self.max_leverage, "PositionSimulatorConfig.max_leverage")

        if self.default_leverage > self.max_leverage:
            raise BacktestConfigurationError(
                "PositionSimulatorConfig.default_leverage cannot be greater than max_leverage.",
                details={
                    "default_leverage": self.default_leverage,
                    "max_leverage": self.max_leverage,
                },
            )

        _ensure_between(
            self.maintenance_margin_rate,
            "PositionSimulatorConfig.maintenance_margin_rate",
            0.0,
            1.0,
        )
        _ensure_positive(
            self.liquidation_buffer_bps,
            "PositionSimulatorConfig.liquidation_buffer_bps",
            allow_zero=True,
        )
        _ensure_positive(
            self.equity_update_interval_seconds,
            "PositionSimulatorConfig.equity_update_interval_seconds",
        )

        if not self.quote_currency:
            raise BacktestConfigurationError("PositionSimulatorConfig.quote_currency cannot be empty.")


# ============================================================================
# Performance metrics
# ============================================================================


@dataclass(slots=True)
class PerformanceMetricsConfig:
    """
    Configuration for performance_metrics.py.
    """

    enabled: bool = True

    risk_free_rate: float = 0.0
    annualization_periods: int = 365
    trading_days_per_year: int = 365

    calculate_trade_stats: bool = True
    calculate_drawdowns: bool = True
    calculate_ratios: bool = True
    calculate_strategy_breakdown: bool = True
    calculate_symbol_breakdown: bool = True
    calculate_timeframe_breakdown: bool = True
    calculate_regime_breakdown: bool = True
    calculate_cost_breakdown: bool = True
    calculate_execution_stats: bool = True
    calculate_risk_stats: bool = True

    min_trades_for_ratios: int = 5
    max_drawdown_periods: int = 100

    use_log_returns: bool = False
    include_open_positions_in_equity: bool = True
    include_unrealized_pnl: bool = True

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _ensure_positive(
            self.annualization_periods,
            "PerformanceMetricsConfig.annualization_periods",
        )
        _ensure_positive(
            self.trading_days_per_year,
            "PerformanceMetricsConfig.trading_days_per_year",
        )
        _ensure_positive(
            self.min_trades_for_ratios,
            "PerformanceMetricsConfig.min_trades_for_ratios",
            allow_zero=True,
        )
        _ensure_positive(
            self.max_drawdown_periods,
            "PerformanceMetricsConfig.max_drawdown_periods",
            allow_zero=True,
        )


# ============================================================================
# Model analytics
# ============================================================================


@dataclass(slots=True)
class ModelAnalyticsConfig:
    """
    Configuration for model_analytics.py.
    """

    enabled: bool = True

    analyze_signal_quality: bool = True
    analyze_strategy_attribution: bool = True
    analyze_regime_performance: bool = True
    analyze_feature_importance: bool = True
    analyze_risk_decisions: bool = True
    analyze_execution_quality: bool = True

    min_signals_for_strategy_stats: int = 10
    min_trades_for_strategy_stats: int = 5
    min_trades_for_regime_stats: int = 5

    include_blocked_signals: bool = True
    include_rejected_orders: bool = True
    include_open_trades: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _ensure_positive(
            self.min_signals_for_strategy_stats,
            "ModelAnalyticsConfig.min_signals_for_strategy_stats",
            allow_zero=True,
        )
        _ensure_positive(
            self.min_trades_for_strategy_stats,
            "ModelAnalyticsConfig.min_trades_for_strategy_stats",
            allow_zero=True,
        )
        _ensure_positive(
            self.min_trades_for_regime_stats,
            "ModelAnalyticsConfig.min_trades_for_regime_stats",
            allow_zero=True,
        )


# ============================================================================
# Report builder
# ============================================================================


@dataclass(slots=True)
class ReportBuilderConfig:
    """
    Configuration for report_builder.py.
    """

    enabled: bool = True

    output_dir: str | Path = "reports/backtests"
    formats: list[ReportFormat] = field(default_factory=lambda: [ReportFormat.MARKDOWN, ReportFormat.JSON])

    sections: set[ReportSection] = field(
        default_factory=lambda: {
            ReportSection.SUMMARY,
            ReportSection.EQUITY_CURVE,
            ReportSection.DRAWDOWN,
            ReportSection.TRADES,
            ReportSection.STRATEGIES,
            ReportSection.RISK,
            ReportSection.EXECUTION,
            ReportSection.COSTS,
            ReportSection.SIGNALS,
            ReportSection.WARNINGS,
        }
    )

    include_full_trade_list: bool = True
    include_full_signal_list: bool = False
    include_full_event_log: bool = False
    include_charts: bool = True
    include_config_snapshot: bool = True
    include_metadata: bool = True

    max_trades_in_markdown: int = 500
    max_signals_in_markdown: int = 500
    max_warnings_in_report: int = 100

    save_result_json: bool = True
    save_trades_csv: bool = True
    save_positions_csv: bool = True
    save_equity_curve_csv: bool = True
    save_events_jsonl: bool = False

    report_title: str = "Backtest Report"

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.output_dir = _ensure_dir_path(self.output_dir, "ReportBuilderConfig.output_dir")

        if not self.formats:
            raise BacktestConfigurationError("ReportBuilderConfig.formats cannot be empty.")

        if not self.sections:
            raise BacktestConfigurationError("ReportBuilderConfig.sections cannot be empty.")

        _ensure_positive(
            self.max_trades_in_markdown,
            "ReportBuilderConfig.max_trades_in_markdown",
            allow_zero=True,
        )
        _ensure_positive(
            self.max_signals_in_markdown,
            "ReportBuilderConfig.max_signals_in_markdown",
            allow_zero=True,
        )
        _ensure_positive(
            self.max_warnings_in_report,
            "ReportBuilderConfig.max_warnings_in_report",
            allow_zero=True,
        )

        if not self.report_title:
            raise BacktestConfigurationError("ReportBuilderConfig.report_title cannot be empty.")


# ============================================================================
# Strategy tester
# ============================================================================


@dataclass(slots=True)
class StrategyTesterConfig:
    """
    Configuration for strategy_tester.py.

    StrategyTester is the main backtest orchestrator. It should wire production
    data caches, analytics, StrategyEngine, SignalProcessor and RiskManager,
    then replace live execution with ExecutionSimulator + PositionSimulator.
    """

    run_name: str = "backtest"
    mode: BacktestMode = BacktestMode.MULTI_STRATEGY

    exchange: str = "binance"
    market_type: str = "usdm_futures"
    symbols: list[str] = field(default_factory=list)
    timeframes: list[str] = field(default_factory=lambda: ["1m"])

    strategies: list[str] = field(default_factory=list)
    strategy_preset: str | None = None
    test_all_registered_strategies: bool = True

    require_risk_manager: bool = True
    require_strategy_engine: bool = True
    require_signal_processor: bool = True
    require_analytics: bool = True

    use_production_data_caches: bool = True
    use_production_analytics: bool = True
    use_production_strategy_engine: bool = True
    use_production_risk_manager: bool = True

    disable_live_exchange_execution: bool = True
    fail_if_live_execution_detected: bool = True

    collect_event_log: bool = True
    collect_signal_records: bool = True
    collect_risk_records: bool = True
    collect_execution_records: bool = True
    collect_position_records: bool = True

    stop_on_first_error: bool = False
    cleanup_after_run: bool = True

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.run_name:
            raise BacktestConfigurationError("StrategyTesterConfig.run_name cannot be empty.")

        self.exchange = self.exchange.lower()
        self.market_type = self.market_type.lower()
        self.symbols = _normalize_symbols(self.symbols)
        self.timeframes = _normalize_timeframes(self.timeframes)

        if not self.test_all_registered_strategies and not self.strategies and not self.strategy_preset:
            raise BacktestConfigurationError(
                "StrategyTesterConfig requires strategies, strategy_preset, or test_all_registered_strategies=True."
            )

        if not self.disable_live_exchange_execution:
            raise BacktestConfigurationError(
                "Live exchange execution must be disabled during backtesting."
            )


# ============================================================================
# Backtest clock / time
# ============================================================================


@dataclass(slots=True)
class BacktestTimeConfig:
    """
    Configuration for backtest_time.py.
    """

    use_simulated_time: bool = True
    freeze_system_time: bool = False

    timezone: str = "UTC"

    scheduler_tick_interval_ms: int = 1000
    run_scheduler_jobs: bool = True
    run_interval_jobs_during_replay: bool = True
    run_due_jobs_after_each_event: bool = True

    allow_time_travel_backwards: bool = False
    fail_on_time_out_of_range: bool = True

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _ensure_positive(
            self.scheduler_tick_interval_ms,
            "BacktestTimeConfig.scheduler_tick_interval_ms",
        )

        if not self.timezone:
            raise BacktestConfigurationError("BacktestTimeConfig.timezone cannot be empty.")


# ============================================================================
# Walk-forward
# ============================================================================


@dataclass(slots=True)
class WalkForwardConfig:
    """
    Configuration for walk_forward.py.
    """

    enabled: bool = False

    mode: WalkForwardMode = WalkForwardMode.ROLLING

    train_window: timedelta = timedelta(days=90)
    validation_window: timedelta | None = None
    test_window: timedelta = timedelta(days=30)
    step_size: timedelta = timedelta(days=30)

    min_train_days: int = 30
    min_test_days: int = 7

    optimize_on_train: bool = True
    validate_before_test: bool = False
    carry_best_parameters_forward: bool = True

    aggregate_results: bool = True
    calculate_stability_score: bool = True
    calculate_overfitting_score: bool = True

    max_iterations: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.train_window.total_seconds() <= 0:
            raise BacktestConfigurationError("WalkForwardConfig.train_window must be positive.")

        if self.validation_window is not None and self.validation_window.total_seconds() < 0:
            raise BacktestConfigurationError(
                "WalkForwardConfig.validation_window cannot be negative."
            )

        if self.test_window.total_seconds() <= 0:
            raise BacktestConfigurationError("WalkForwardConfig.test_window must be positive.")

        if self.step_size.total_seconds() <= 0:
            raise BacktestConfigurationError("WalkForwardConfig.step_size must be positive.")

        _ensure_positive(self.min_train_days, "WalkForwardConfig.min_train_days")
        _ensure_positive(self.min_test_days, "WalkForwardConfig.min_test_days")

        if self.max_iterations is not None:
            _ensure_positive(self.max_iterations, "WalkForwardConfig.max_iterations")


# ============================================================================
# Optimizer
# ============================================================================


@dataclass(slots=True)
class OptimizerConfig:
    """
    Configuration for optimizer.py.
    """

    enabled: bool = False

    method: OptimizationMethod = OptimizationMethod.GRID_SEARCH
    objective_metric: OptimizationMetric = OptimizationMetric.NET_PROFIT
    direction: OptimizationDirection = OptimizationDirection.MAXIMIZE

    parameter_space: dict[str, Any] = field(default_factory=dict)

    max_trials: int = 100
    random_seed: int | None = 42

    parallel_jobs: int = 1
    stop_on_trial_error: bool = False

    overfitting_check: OverfittingCheckMode = OverfittingCheckMode.TRAIN_TEST_SPLIT
    use_walk_forward_for_overfitting_check: bool = False

    min_trades_required: int = 20
    max_drawdown_pct_limit: float | None = None
    min_profit_factor: float | None = None
    min_win_rate: float | None = None

    save_all_trials: bool = True
    save_best_config: bool = True

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.enabled and not self.parameter_space:
            raise BacktestConfigurationError(
                "OptimizerConfig.parameter_space cannot be empty when optimizer is enabled."
            )

        _ensure_positive(self.max_trials, "OptimizerConfig.max_trials")
        _ensure_positive(self.parallel_jobs, "OptimizerConfig.parallel_jobs")
        _ensure_positive(
            self.min_trades_required,
            "OptimizerConfig.min_trades_required",
            allow_zero=True,
        )

        if self.max_drawdown_pct_limit is not None:
            _ensure_between(
                self.max_drawdown_pct_limit,
                "OptimizerConfig.max_drawdown_pct_limit",
                0.0,
                100.0,
            )

        if self.min_profit_factor is not None:
            _ensure_positive(
                self.min_profit_factor,
                "OptimizerConfig.min_profit_factor",
                allow_zero=True,
            )

        if self.min_win_rate is not None:
            _ensure_between(
                self.min_win_rate,
                "OptimizerConfig.min_win_rate",
                0.0,
                100.0,
            )


# ============================================================================
# Main aggregate config
# ============================================================================


@dataclass(slots=True)
class BacktestConfig:
    """
    Main aggregate configuration for a full backtest run.

    This config wires all backtesting sub-configs and can also carry snapshots
    or direct references to production configs for strategy/risk/execution.
    """

    run_name: str = "backtest"
    mode: BacktestMode = BacktestMode.MULTI_STRATEGY

    exchange: str = "binance"
    market_type: str = "usdm_futures"
    symbols: list[str] = field(default_factory=list)
    timeframes: list[str] = field(default_factory=lambda: ["1m"])

    start_time: datetime | None = None
    end_time: datetime | None = None
    warmup_start_time: datetime | None = None
    warmup_bars: int = 500

    initial_balance: float = 10_000.0
    quote_currency: str = "USDT"

    strategies: list[str] = field(default_factory=list)
    strategy_preset: str | None = None
    test_all_registered_strategies: bool = True

    data_dir: str | Path = "data/history"
    output_dir: str | Path = "reports/backtests"

    use_candles: bool = True
    use_trades: bool = False
    use_orderbook: bool = False
    use_funding: bool = True
    use_open_interest: bool = True
    use_liquidations: bool = False
    use_mark_price: bool = True
    use_index_price: bool = True

    download_missing_data: bool = False

    deterministic: bool = True
    random_seed: int | None = 42

    save_trades: bool = True
    save_positions: bool = True
    save_equity_curve: bool = True
    save_events: bool = False
    save_report: bool = True

    fail_fast: bool = False
    allow_partial_results: bool = True

    # Backtesting sub-configs
    history_downloader: HistoryDownloaderConfig = field(default_factory=HistoryDownloaderConfig)
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    market_replay: MarketReplayConfig = field(default_factory=MarketReplayConfig)
    cost_model: CostModelConfig = field(default_factory=CostModelConfig)
    execution_simulator: ExecutionSimulatorConfig = field(default_factory=ExecutionSimulatorConfig)
    position_simulator: PositionSimulatorConfig = field(default_factory=PositionSimulatorConfig)
    performance_metrics: PerformanceMetricsConfig = field(default_factory=PerformanceMetricsConfig)
    model_analytics: ModelAnalyticsConfig = field(default_factory=ModelAnalyticsConfig)
    report_builder: ReportBuilderConfig = field(default_factory=ReportBuilderConfig)
    strategy_tester: StrategyTesterConfig = field(default_factory=StrategyTesterConfig)
    backtest_time: BacktestTimeConfig = field(default_factory=BacktestTimeConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # Production config snapshots or direct config objects.
    # These stay intentionally untyped to avoid hard dependency cycles.
    core_config: Any | None = None
    strategy_config: Any | None = None
    risk_config: Any | None = None
    execution_config: Any | None = None
    analytics_config: Any | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate and propagate shared settings to sub-configs.
        """

        if not self.run_name:
            raise BacktestConfigurationError("BacktestConfig.run_name cannot be empty.")

        self.exchange = self.exchange.lower()
        self.market_type = self.market_type.lower()
        self.symbols = _normalize_symbols(self.symbols)
        self.timeframes = _normalize_timeframes(self.timeframes)

        _ensure_positive(self.initial_balance, "BacktestConfig.initial_balance")
        _ensure_positive(self.warmup_bars, "BacktestConfig.warmup_bars", allow_zero=True)

        if not self.quote_currency:
            raise BacktestConfigurationError("BacktestConfig.quote_currency cannot be empty.")

        if self.start_time is None or self.end_time is None:
            raise BacktestConfigurationError(
                "BacktestConfig.start_time and BacktestConfig.end_time are required."
            )

        self.start_time = ensure_aware_utc(self.start_time)
        self.end_time = ensure_aware_utc(self.end_time)

        if self.warmup_start_time is not None:
            self.warmup_start_time = ensure_aware_utc(self.warmup_start_time)

        if self.end_time <= self.start_time:
            raise BacktestConfigurationError(
                "BacktestConfig.end_time must be greater than start_time.",
                details={
                    "start_time": self.start_time.isoformat(),
                    "end_time": self.end_time.isoformat(),
                },
            )

        if self.warmup_start_time is not None and self.warmup_start_time > self.start_time:
            raise BacktestConfigurationError(
                "BacktestConfig.warmup_start_time cannot be after start_time.",
                details={
                    "warmup_start_time": self.warmup_start_time.isoformat(),
                    "start_time": self.start_time.isoformat(),
                },
            )

        if not any(
            [
                self.use_candles,
                self.use_trades,
                self.use_orderbook,
                self.use_funding,
                self.use_open_interest,
                self.use_liquidations,
                self.use_mark_price,
                self.use_index_price,
            ]
        ):
            raise BacktestConfigurationError("At least one historical data stream must be enabled.")

        if not self.test_all_registered_strategies and not self.strategies and not self.strategy_preset:
            raise BacktestConfigurationError(
                "BacktestConfig requires strategies, strategy_preset, or test_all_registered_strategies=True."
            )

        self.data_dir = _ensure_dir_path(self.data_dir, "BacktestConfig.data_dir")
        self.output_dir = _ensure_dir_path(self.output_dir, "BacktestConfig.output_dir")

        self._propagate_shared_settings()
        self._validate_subconfigs()

    def period(self) -> BacktestPeriod:
        """
        Build BacktestPeriod from this config.
        """

        if self.start_time is None or self.end_time is None:
            raise BacktestConfigurationError(
                "Cannot build BacktestPeriod without start_time and end_time."
            )

        return BacktestPeriod(
            start=self.start_time,
            end=self.end_time,
            warmup_start=self.warmup_start_time,
        )

    def instruments(self) -> list[BacktestInstrument]:
        """
        Build BacktestInstrument list from shared symbol settings.
        """

        return [
            BacktestInstrument(
                exchange=self.exchange,
                symbol=symbol,
                market_type=self.market_type,
                quote_asset=self.quote_currency,
            )
            for symbol in self.symbols
        ]

    def enabled_data_types(self) -> set[BacktestDataType]:
        """
        Return enabled historical data stream types.
        """

        data_types: set[BacktestDataType] = set()

        if self.use_candles:
            data_types.add(BacktestDataType.CANDLES)
        if self.use_trades:
            data_types.add(BacktestDataType.TRADES)
        if self.use_orderbook:
            data_types.add(BacktestDataType.ORDERBOOK)
        if self.use_funding:
            data_types.add(BacktestDataType.FUNDING)
        if self.use_open_interest:
            data_types.add(BacktestDataType.OPEN_INTEREST)
        if self.use_liquidations:
            data_types.add(BacktestDataType.LIQUIDATIONS)
        if self.use_mark_price:
            data_types.add(BacktestDataType.MARK_PRICE)
        if self.use_index_price:
            data_types.add(BacktestDataType.INDEX_PRICE)

        return data_types

    def _propagate_shared_settings(self) -> None:
        """
        Push top-level settings into sub-configs where appropriate.
        """

        data_types = self.enabled_data_types()

        # Downloader
        self.history_downloader.enabled = self.download_missing_data
        self.history_downloader.exchange = self.exchange
        self.history_downloader.market_type = self.market_type
        self.history_downloader.symbols = list(self.symbols)
        self.history_downloader.timeframes = list(self.timeframes)
        self.history_downloader.data_types = set(data_types)
        self.history_downloader.output_dir = self.data_dir
        self.history_downloader.include_mark_price = self.use_mark_price
        self.history_downloader.include_index_price = self.use_index_price
        self.history_downloader.include_liquidations = self.use_liquidations
        self.history_downloader.include_orderbook_snapshots = self.use_orderbook

        # Loader
        self.data_loader.data_dir = self.data_dir
        self.data_loader.exchange = self.exchange
        self.data_loader.market_type = self.market_type
        self.data_loader.symbols = list(self.symbols)
        self.data_loader.timeframes = list(self.timeframes)
        self.data_loader.data_types = set(data_types)
        self.data_loader.require_candles = self.use_candles
        self.data_loader.require_trades = self.use_trades
        self.data_loader.require_orderbook = self.use_orderbook
        self.data_loader.require_funding = self.use_funding
        self.data_loader.require_open_interest = self.use_open_interest

        # Cost / execution / position
        self.cost_model.quote_currency = self.quote_currency

        self.execution_simulator.exchange = self.exchange
        self.execution_simulator.market_type = self.market_type

        self.position_simulator.initial_balance = self.initial_balance
        self.position_simulator.quote_currency = self.quote_currency

        # Report
        self.report_builder.output_dir = self.output_dir

        # Strategy tester
        self.strategy_tester.run_name = self.run_name
        self.strategy_tester.mode = self.mode
        self.strategy_tester.exchange = self.exchange
        self.strategy_tester.market_type = self.market_type
        self.strategy_tester.symbols = list(self.symbols)
        self.strategy_tester.timeframes = list(self.timeframes)
        self.strategy_tester.strategies = list(self.strategies)
        self.strategy_tester.strategy_preset = self.strategy_preset
        self.strategy_tester.test_all_registered_strategies = self.test_all_registered_strategies

    def _validate_subconfigs(self) -> None:
        self.history_downloader.validate()
        self.data_loader.validate()
        self.market_replay.validate()
        self.cost_model.validate()
        self.execution_simulator.validate()
        self.position_simulator.validate()
        self.performance_metrics.validate()
        self.model_analytics.validate()
        self.report_builder.validate()
        self.strategy_tester.validate()
        self.backtest_time.validate()
        self.walk_forward.validate()
        self.optimizer.validate()

    @classmethod
    def default_binance_futures(
        cls,
        *,
        symbols: list[str],
        start_time: datetime,
        end_time: datetime,
        timeframes: list[str] | None = None,
        initial_balance: float = 10_000.0,
        run_name: str = "binance_futures_backtest",
    ) -> BacktestConfig:
        """
        Convenient Binance USD-M Futures default config.
        """

        config = cls(
            run_name=run_name,
            exchange="binance",
            market_type="usdm_futures",
            symbols=symbols,
            timeframes=timeframes or ["1m"],
            start_time=start_time,
            end_time=end_time,
            initial_balance=initial_balance,
            quote_currency="USDT",
            use_candles=True,
            use_funding=True,
            use_open_interest=True,
            use_trades=False,
            use_orderbook=False,
            use_liquidations=False,
        )
        config.validate()
        return config


# ============================================================================
# Preset builders
# ============================================================================


def build_fast_candle_backtest_config(
    *,
    symbols: list[str],
    start_time: datetime,
    end_time: datetime,
    timeframes: list[str] | None = None,
    initial_balance: float = 10_000.0,
) -> BacktestConfig:
    """
    Fast candle-only-ish backtest preset.

    Uses candles + funding + open interest, max-speed replay and simple
    next-candle execution. This is the best first MVP mode.
    """

    config = BacktestConfig.default_binance_futures(
        symbols=symbols,
        start_time=start_time,
        end_time=end_time,
        timeframes=timeframes or ["1m"],
        initial_balance=initial_balance,
        run_name="fast_candle_backtest",
    )

    config.use_trades = False
    config.use_orderbook = False
    config.use_liquidations = False

    config.market_replay.replay_speed = ReplaySpeed.MAX_SPEED
    config.execution_simulator.fill_model = FillModel.NEXT_CANDLE_OPEN
    config.execution_simulator.liquidity_model = LiquidityModel.CANDLE_VOLUME_PERCENT
    config.cost_model.slippage_model = SlippageModel.FIXED_BPS
    config.validate()
    return config


def build_realistic_futures_backtest_config(
    *,
    symbols: list[str],
    start_time: datetime,
    end_time: datetime,
    timeframes: list[str] | None = None,
    initial_balance: float = 10_000.0,
) -> BacktestConfig:
    """
    More realistic Binance futures preset.

    Enables trades, funding, open interest and liquidation checks. Order book
    remains optional because historical depth can be heavy.
    """

    config = BacktestConfig.default_binance_futures(
        symbols=symbols,
        start_time=start_time,
        end_time=end_time,
        timeframes=timeframes or ["1m", "5m"],
        initial_balance=initial_balance,
        run_name="realistic_futures_backtest",
    )

    config.use_trades = True
    config.use_orderbook = False
    config.use_funding = True
    config.use_open_interest = True
    config.use_liquidations = True

    config.execution_simulator.fill_model = FillModel.NEXT_TICK
    config.execution_simulator.allow_partial_fills = True
    config.execution_simulator.max_volume_participation_pct = 5.0

    config.position_simulator.enable_liquidation_check = True
    config.position_simulator.enable_funding_application = True

    config.cost_model.commission_model = CommissionModel.MAKER_TAKER
    config.cost_model.slippage_model = SlippageModel.VOLUME_BASED
    config.cost_model.include_funding = True

    config.validate()
    return config


def build_walk_forward_backtest_config(
    *,
    symbols: list[str],
    start_time: datetime,
    end_time: datetime,
    train_days: int = 90,
    test_days: int = 30,
    timeframes: list[str] | None = None,
    initial_balance: float = 10_000.0,
) -> BacktestConfig:
    """
    Walk-forward preset.
    """

    config = build_fast_candle_backtest_config(
        symbols=symbols,
        start_time=start_time,
        end_time=end_time,
        timeframes=timeframes,
        initial_balance=initial_balance,
    )

    config.run_name = "walk_forward_backtest"
    config.mode = BacktestMode.WALK_FORWARD
    config.walk_forward.enabled = True
    config.walk_forward.train_window = timedelta(days=train_days)
    config.walk_forward.test_window = timedelta(days=test_days)
    config.walk_forward.step_size = timedelta(days=test_days)

    config.validate()
    return config


__all__ = [
    "HistoryDownloaderConfig",
    "DataLoaderConfig",
    "MarketReplayConfig",
    "CostModelConfig",
    "ExecutionSimulatorConfig",
    "PositionSimulatorConfig",
    "PerformanceMetricsConfig",
    "ModelAnalyticsConfig",
    "ReportBuilderConfig",
    "StrategyTesterConfig",
    "BacktestTimeConfig",
    "WalkForwardConfig",
    "OptimizerConfig",
    "BacktestConfig",
    "build_fast_candle_backtest_config",
    "build_realistic_futures_backtest_config",
    "build_walk_forward_backtest_config",
]