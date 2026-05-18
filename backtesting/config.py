# trading_system/backtesting/config.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .enums import (
    BacktestMode,
    DrawdownMode,
    DuplicateHandlingPolicy,
    EquityCurveMode,
    ExchangeName,
    ExecutionSimulationMode,
    FeeModelType,
    GapHandlingPolicy,
    HistoryDataType,
    HistorySourceType,
    LatencyModelType,
    MarketType,
    OptimizationMode,
    OptimizationObjective,
    OrderSimulationType,
    ReplayMode,
    ReportFormat,
    SlippageModelType,
    StorageFormat,
    WalkForwardWindowMode,
)
from .exceptions import BacktestConfigError, build_error_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_positive_int(value: int, field_name: str) -> None:
    if value <= 0:
        raise BacktestConfigError(
            f"{field_name} must be positive",
            context=build_error_context(**{field_name: value}),
        )


def _ensure_non_negative_float(value: float, field_name: str) -> None:
    if value < 0:
        raise BacktestConfigError(
            f"{field_name} must be non-negative",
            context=build_error_context(**{field_name: value}),
        )


def _ensure_time_range(start_time_ms: int, end_time_ms: int) -> None:
    if start_time_ms <= 0:
        raise BacktestConfigError(
            "start_time_ms must be positive",
            context=build_error_context(start_time_ms=start_time_ms),
        )

    if end_time_ms <= 0:
        raise BacktestConfigError(
            "end_time_ms must be positive",
            context=build_error_context(end_time_ms=end_time_ms),
        )

    if start_time_ms >= end_time_ms:
        raise BacktestConfigError(
            "start_time_ms must be lower than end_time_ms",
            context=build_error_context(
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
            ),
        )


def _ensure_non_empty_list(values: list[Any], field_name: str) -> None:
    if not values:
        raise BacktestConfigError(
            f"{field_name} must not be empty",
            context=build_error_context(**{field_name: values}),
        )


def _as_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


# ---------------------------------------------------------------------------
# History download config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HistoryDownloadConfig:
    """
    Configuration for downloading futures/perpetual historical data
    directly from exchanges/providers and writing it to local storage.

    This config is used by history_downloader.py.
    """

    exchange: ExchangeName | str = ExchangeName.BINANCE
    market_type: MarketType | str = MarketType.USDM_FUTURES

    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT"])
    timeframes: list[str] = field(default_factory=lambda: ["1m"])

    start_time_ms: int = 0
    end_time_ms: int = 0

    data_types: list[HistoryDataType | str] = field(
        default_factory=lambda: [
            HistoryDataType.CANDLES,
            HistoryDataType.FUNDING,
            HistoryDataType.OPEN_INTEREST,
        ]
    )

    source_type: HistorySourceType | str = HistorySourceType.EXCHANGE_REST
    storage_format: StorageFormat | str = StorageFormat.PARQUET

    output_dir: str = "data/history"

    overwrite_existing: bool = False
    skip_existing: bool = True
    validate_after_download: bool = True

    request_timeout_sec: float = 15.0
    max_retries: int = 5
    retry_delay_sec: float = 1.0
    rate_limit_delay_sec: float = 0.25

    max_rows_per_request: int = 1000
    max_concurrent_symbols: int = 1

    user_agent: str = "trading-system-backtesting"
    extra_headers: dict[str, str] = field(default_factory=dict)

    provider_api_key_env: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _ensure_non_empty_list(self.symbols, "symbols")
        _ensure_non_empty_list(self.data_types, "data_types")
        _ensure_time_range(self.start_time_ms, self.end_time_ms)

        if HistoryDataType.CANDLES in self.data_types or "candles" in self.data_types:
            _ensure_non_empty_list(self.timeframes, "timeframes")

        _ensure_non_negative_float(self.request_timeout_sec, "request_timeout_sec")
        _ensure_positive_int(self.max_retries, "max_retries")
        _ensure_non_negative_float(self.retry_delay_sec, "retry_delay_sec")
        _ensure_non_negative_float(self.rate_limit_delay_sec, "rate_limit_delay_sec")
        _ensure_positive_int(self.max_rows_per_request, "max_rows_per_request")
        _ensure_positive_int(self.max_concurrent_symbols, "max_concurrent_symbols")

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange": _as_value(self.exchange),
            "market_type": _as_value(self.market_type),
            "symbols": list(self.symbols),
            "timeframes": list(self.timeframes),
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "data_types": [_as_value(item) for item in self.data_types],
            "source_type": _as_value(self.source_type),
            "storage_format": _as_value(self.storage_format),
            "output_dir": self.output_dir,
            "overwrite_existing": self.overwrite_existing,
            "skip_existing": self.skip_existing,
            "validate_after_download": self.validate_after_download,
            "request_timeout_sec": self.request_timeout_sec,
            "max_retries": self.max_retries,
            "retry_delay_sec": self.retry_delay_sec,
            "rate_limit_delay_sec": self.rate_limit_delay_sec,
            "max_rows_per_request": self.max_rows_per_request,
            "max_concurrent_symbols": self.max_concurrent_symbols,
            "user_agent": self.user_agent,
            "provider_api_key_env": self.provider_api_key_env,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Data loading config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BacktestDataConfig:
    """
    Configuration for reading local historical data and building
    HistoricalMarketEvent stream.

    This config is used by data_loader.py.
    """

    data_dir: str = "data/history"

    exchange: ExchangeName | str = ExchangeName.BINANCE
    market_type: MarketType | str = MarketType.USDM_FUTURES

    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT"])
    timeframes: list[str] = field(default_factory=lambda: ["1m"])

    start_time_ms: int = 0
    end_time_ms: int = 0

    data_types: list[HistoryDataType | str] = field(
        default_factory=lambda: [
            HistoryDataType.CANDLES,
            HistoryDataType.FUNDING,
            HistoryDataType.OPEN_INTEREST,
        ]
    )

    storage_format: StorageFormat | str = StorageFormat.PARQUET

    preload_into_memory: bool = False
    sort_events: bool = True
    enforce_chronological_order: bool = True

    validate_schema: bool = True
    validate_quality: bool = True

    gap_policy: GapHandlingPolicy | str = GapHandlingPolicy.WARN
    duplicate_policy: DuplicateHandlingPolicy | str = DuplicateHandlingPolicy.KEEP_LAST

    max_events: int | None = None
    batch_size: int = 10_000

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _ensure_non_empty_list(self.symbols, "symbols")
        _ensure_non_empty_list(self.data_types, "data_types")
        _ensure_time_range(self.start_time_ms, self.end_time_ms)
        _ensure_positive_int(self.batch_size, "batch_size")

        if self.max_events is not None:
            _ensure_positive_int(self.max_events, "max_events")

        if HistoryDataType.CANDLES in self.data_types or "candles" in self.data_types:
            _ensure_non_empty_list(self.timeframes, "timeframes")

        if not Path(self.data_dir).exists:
            raise BacktestConfigError(
                "data_dir does not exist",
                context=build_error_context(data_path=self.data_dir),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_dir": self.data_dir,
            "exchange": _as_value(self.exchange),
            "market_type": _as_value(self.market_type),
            "symbols": list(self.symbols),
            "timeframes": list(self.timeframes),
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "data_types": [_as_value(item) for item in self.data_types],
            "storage_format": _as_value(self.storage_format),
            "preload_into_memory": self.preload_into_memory,
            "sort_events": self.sort_events,
            "enforce_chronological_order": self.enforce_chronological_order,
            "validate_schema": self.validate_schema,
            "validate_quality": self.validate_quality,
            "gap_policy": _as_value(self.gap_policy),
            "duplicate_policy": _as_value(self.duplicate_policy),
            "max_events": self.max_events,
            "batch_size": self.batch_size,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Replay config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BacktestReplayConfig:
    """
    Configuration for BacktestMarketReplay.
    """

    mode: ReplayMode | str = ReplayMode.FAST

    emit_system_events: bool = True
    emit_progress_events: bool = True
    progress_interval_events: int = 10_000
    progress_interval_ms: int | None = None

    enforce_chronological_order: bool = True
    allow_same_timestamp_reordering: bool = True

    stop_on_event_error: bool = True
    continue_on_missing_optional_data: bool = True

    real_time_speed_multiplier: float = 1.0
    step_wait_for_ack: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _ensure_positive_int(self.progress_interval_events, "progress_interval_events")

        if self.progress_interval_ms is not None:
            _ensure_positive_int(self.progress_interval_ms, "progress_interval_ms")

        if self.real_time_speed_multiplier <= 0:
            raise BacktestConfigError(
                "real_time_speed_multiplier must be positive",
                context=build_error_context(
                    real_time_speed_multiplier=self.real_time_speed_multiplier
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": _as_value(self.mode),
            "emit_system_events": self.emit_system_events,
            "emit_progress_events": self.emit_progress_events,
            "progress_interval_events": self.progress_interval_events,
            "progress_interval_ms": self.progress_interval_ms,
            "enforce_chronological_order": self.enforce_chronological_order,
            "allow_same_timestamp_reordering": self.allow_same_timestamp_reordering,
            "stop_on_event_error": self.stop_on_event_error,
            "continue_on_missing_optional_data": self.continue_on_missing_optional_data,
            "real_time_speed_multiplier": self.real_time_speed_multiplier,
            "step_wait_for_ack": self.step_wait_for_ack,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Cost / execution config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BacktestCostConfig:
    """
    Fee, slippage and latency model configuration.

    This config is used by cost_models.py and execution_simulator.py.
    """

    fee_model: FeeModelType | str = FeeModelType.BINANCE_FUTURES
    slippage_model: SlippageModelType | str = SlippageModelType.FIXED_BPS
    latency_model: LatencyModelType | str = LatencyModelType.NONE

    maker_fee_rate: float = 0.0002
    taker_fee_rate: float = 0.0004

    fixed_slippage_ticks: float = 0.0
    fixed_slippage_bps: float = 1.0
    percent_slippage: float = 0.0

    volatility_slippage_multiplier: float = 1.0

    fixed_latency_ms: int = 0
    random_latency_min_ms: int = 0
    random_latency_max_ms: int = 0

    fee_asset: str = "USDT"

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _ensure_non_negative_float(self.maker_fee_rate, "maker_fee_rate")
        _ensure_non_negative_float(self.taker_fee_rate, "taker_fee_rate")
        _ensure_non_negative_float(self.fixed_slippage_ticks, "fixed_slippage_ticks")
        _ensure_non_negative_float(self.fixed_slippage_bps, "fixed_slippage_bps")
        _ensure_non_negative_float(self.percent_slippage, "percent_slippage")
        _ensure_non_negative_float(
            self.volatility_slippage_multiplier,
            "volatility_slippage_multiplier",
        )

        if self.fixed_latency_ms < 0:
            raise BacktestConfigError(
                "fixed_latency_ms must be non-negative",
                context=build_error_context(fixed_latency_ms=self.fixed_latency_ms),
            )

        if self.random_latency_min_ms < 0 or self.random_latency_max_ms < 0:
            raise BacktestConfigError(
                "random latency bounds must be non-negative",
                context=build_error_context(
                    random_latency_min_ms=self.random_latency_min_ms,
                    random_latency_max_ms=self.random_latency_max_ms,
                ),
            )

        if self.random_latency_min_ms > self.random_latency_max_ms:
            raise BacktestConfigError(
                "random_latency_min_ms cannot be greater than random_latency_max_ms",
                context=build_error_context(
                    random_latency_min_ms=self.random_latency_min_ms,
                    random_latency_max_ms=self.random_latency_max_ms,
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fee_model": _as_value(self.fee_model),
            "slippage_model": _as_value(self.slippage_model),
            "latency_model": _as_value(self.latency_model),
            "maker_fee_rate": self.maker_fee_rate,
            "taker_fee_rate": self.taker_fee_rate,
            "fixed_slippage_ticks": self.fixed_slippage_ticks,
            "fixed_slippage_bps": self.fixed_slippage_bps,
            "percent_slippage": self.percent_slippage,
            "volatility_slippage_multiplier": self.volatility_slippage_multiplier,
            "fixed_latency_ms": self.fixed_latency_ms,
            "random_latency_min_ms": self.random_latency_min_ms,
            "random_latency_max_ms": self.random_latency_max_ms,
            "fee_asset": self.fee_asset,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class BacktestExecutionConfig:
    """
    Configuration for simulated execution and position handling.
    """

    simulation_mode: ExecutionSimulationMode | str = ExecutionSimulationMode.REALISTIC
    order_simulation_type: OrderSimulationType | str = OrderSimulationType.NEXT_TICK

    initial_balance: float = 10_000.0
    quote_asset: str = "USDT"

    leverage: float = 3.0
    max_leverage: float = 20.0

    allow_short: bool = True
    allow_long: bool = True

    allow_position_increase: bool = True
    allow_position_reduce: bool = True
    allow_multiple_positions_per_symbol: bool = False

    default_order_quantity: float | None = None
    default_order_notional: float | None = None

    min_notional: float = 5.0
    quantity_precision: int = 3
    price_precision: int = 2

    simulate_partial_fills: bool = False
    max_fill_ratio_per_event: float = 1.0

    reject_on_insufficient_margin: bool = True
    reject_on_missing_price: bool = True

    close_positions_on_backtest_end: bool = True

    maintenance_margin_rate: float = 0.005
    liquidation_fee_rate: float = 0.002

    cost: BacktestCostConfig = field(default_factory=BacktestCostConfig)

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _ensure_non_negative_float(self.initial_balance, "initial_balance")
        _ensure_non_negative_float(self.leverage, "leverage")
        _ensure_non_negative_float(self.max_leverage, "max_leverage")
        _ensure_non_negative_float(self.min_notional, "min_notional")
        _ensure_non_negative_float(
            self.maintenance_margin_rate,
            "maintenance_margin_rate",
        )
        _ensure_non_negative_float(self.liquidation_fee_rate, "liquidation_fee_rate")

        if self.initial_balance <= 0:
            raise BacktestConfigError(
                "initial_balance must be greater than zero",
                context=build_error_context(initial_balance=self.initial_balance),
            )

        if self.leverage <= 0:
            raise BacktestConfigError(
                "leverage must be greater than zero",
                context=build_error_context(leverage=self.leverage),
            )

        if self.leverage > self.max_leverage:
            raise BacktestConfigError(
                "leverage cannot be greater than max_leverage",
                context=build_error_context(
                    leverage=self.leverage,
                    max_leverage=self.max_leverage,
                ),
            )

        if not self.allow_long and not self.allow_short:
            raise BacktestConfigError(
                "At least one of allow_long or allow_short must be enabled"
            )

        if self.default_order_quantity is not None:
            _ensure_non_negative_float(
                self.default_order_quantity,
                "default_order_quantity",
            )

        if self.default_order_notional is not None:
            _ensure_non_negative_float(
                self.default_order_notional,
                "default_order_notional",
            )

        if not 0 < self.max_fill_ratio_per_event <= 1:
            raise BacktestConfigError(
                "max_fill_ratio_per_event must be between 0 and 1",
                context=build_error_context(
                    max_fill_ratio_per_event=self.max_fill_ratio_per_event
                ),
            )

        if self.quantity_precision < 0 or self.price_precision < 0:
            raise BacktestConfigError(
                "quantity_precision and price_precision must be non-negative",
                context=build_error_context(
                    quantity_precision=self.quantity_precision,
                    price_precision=self.price_precision,
                ),
            )

        self.cost.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "simulation_mode": _as_value(self.simulation_mode),
            "order_simulation_type": _as_value(self.order_simulation_type),
            "initial_balance": self.initial_balance,
            "quote_asset": self.quote_asset,
            "leverage": self.leverage,
            "max_leverage": self.max_leverage,
            "allow_short": self.allow_short,
            "allow_long": self.allow_long,
            "allow_position_increase": self.allow_position_increase,
            "allow_position_reduce": self.allow_position_reduce,
            "allow_multiple_positions_per_symbol": self.allow_multiple_positions_per_symbol,
            "default_order_quantity": self.default_order_quantity,
            "default_order_notional": self.default_order_notional,
            "min_notional": self.min_notional,
            "quantity_precision": self.quantity_precision,
            "price_precision": self.price_precision,
            "simulate_partial_fills": self.simulate_partial_fills,
            "max_fill_ratio_per_event": self.max_fill_ratio_per_event,
            "reject_on_insufficient_margin": self.reject_on_insufficient_margin,
            "reject_on_missing_price": self.reject_on_missing_price,
            "close_positions_on_backtest_end": self.close_positions_on_backtest_end,
            "maintenance_margin_rate": self.maintenance_margin_rate,
            "liquidation_fee_rate": self.liquidation_fee_rate,
            "cost": self.cost.to_dict(),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Metrics / analytics / report config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BacktestMetricsConfig:
    """
    Configuration for performance_metrics.py.
    """

    enabled: bool = True

    equity_curve_mode: EquityCurveMode | str = EquityCurveMode.ON_POSITION_UPDATE
    drawdown_mode: DrawdownMode | str = DrawdownMode.EQUITY_BASED

    calculate_sharpe: bool = True
    calculate_sortino: bool = True
    calculate_calmar: bool = True
    calculate_expectancy: bool = True

    risk_free_rate_annual: float = 0.0
    periods_per_year: int = 365

    track_symbol_stats: bool = True
    track_strategy_stats: bool = True
    track_long_short_stats: bool = True

    emit_metrics_events: bool = True
    metrics_update_interval_events: int = 10_000

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _ensure_positive_int(self.periods_per_year, "periods_per_year")
        _ensure_positive_int(
            self.metrics_update_interval_events,
            "metrics_update_interval_events",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "equity_curve_mode": _as_value(self.equity_curve_mode),
            "drawdown_mode": _as_value(self.drawdown_mode),
            "calculate_sharpe": self.calculate_sharpe,
            "calculate_sortino": self.calculate_sortino,
            "calculate_calmar": self.calculate_calmar,
            "calculate_expectancy": self.calculate_expectancy,
            "risk_free_rate_annual": self.risk_free_rate_annual,
            "periods_per_year": self.periods_per_year,
            "track_symbol_stats": self.track_symbol_stats,
            "track_strategy_stats": self.track_strategy_stats,
            "track_long_short_stats": self.track_long_short_stats,
            "emit_metrics_events": self.emit_metrics_events,
            "metrics_update_interval_events": self.metrics_update_interval_events,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ModelAnalyticsConfig:
    """
    Configuration for model_analytics.py.
    """

    enabled: bool = True

    track_signal_quality: bool = True
    track_confidence_calibration: bool = True
    track_regime_performance: bool = True
    track_symbol_performance: bool = True
    track_timeframe_performance: bool = True
    track_strategy_performance: bool = True

    track_mfe_mae: bool = True
    track_signal_decay: bool = False
    track_llm_decisions: bool = True

    confidence_buckets: list[tuple[float, float]] = field(
        default_factory=lambda: [
            (0.0, 0.2),
            (0.2, 0.4),
            (0.4, 0.6),
            (0.6, 0.8),
            (0.8, 1.0),
        ]
    )

    signal_decay_horizons_ms: list[int] = field(
        default_factory=lambda: [
            60_000,
            3 * 60_000,
            5 * 60_000,
            15 * 60_000,
            30 * 60_000,
        ]
    )

    min_predictions_for_calibration: int = 30

    emit_model_analytics_events: bool = True

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.min_predictions_for_calibration < 1:
            raise BacktestConfigError(
                "min_predictions_for_calibration must be positive",
                context=build_error_context(
                    min_predictions_for_calibration=self.min_predictions_for_calibration
                ),
            )

        for start, end in self.confidence_buckets:
            if start < 0 or end > 1 or start >= end:
                raise BacktestConfigError(
                    "Invalid confidence bucket",
                    context=build_error_context(
                        confidence_bucket_start=start,
                        confidence_bucket_end=end,
                    ),
                )

        for horizon in self.signal_decay_horizons_ms:
            _ensure_positive_int(horizon, "signal_decay_horizon_ms")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "track_signal_quality": self.track_signal_quality,
            "track_confidence_calibration": self.track_confidence_calibration,
            "track_regime_performance": self.track_regime_performance,
            "track_symbol_performance": self.track_symbol_performance,
            "track_timeframe_performance": self.track_timeframe_performance,
            "track_strategy_performance": self.track_strategy_performance,
            "track_mfe_mae": self.track_mfe_mae,
            "track_signal_decay": self.track_signal_decay,
            "track_llm_decisions": self.track_llm_decisions,
            "confidence_buckets": list(self.confidence_buckets),
            "signal_decay_horizons_ms": list(self.signal_decay_horizons_ms),
            "min_predictions_for_calibration": self.min_predictions_for_calibration,
            "emit_model_analytics_events": self.emit_model_analytics_events,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class BacktestReportConfig:
    """
    Configuration for report_builder.py.
    """

    enabled: bool = True

    output_dir: str = "data/backtest_reports"
    formats: list[ReportFormat | str] = field(
        default_factory=lambda: [ReportFormat.JSON, ReportFormat.MARKDOWN]
    )

    include_config_snapshot: bool = True
    include_trades: bool = True
    include_orders: bool = True
    include_fills: bool = True
    include_positions: bool = True
    include_signals: bool = True
    include_risk_blocks: bool = True
    include_equity_curve: bool = True
    include_drawdown_curve: bool = True
    include_model_analytics: bool = True

    write_artifacts: bool = True
    artifact_format: StorageFormat | str = StorageFormat.PARQUET

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.enabled:
            _ensure_non_empty_list(self.formats, "formats")
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "output_dir": self.output_dir,
            "formats": [_as_value(item) for item in self.formats],
            "include_config_snapshot": self.include_config_snapshot,
            "include_trades": self.include_trades,
            "include_orders": self.include_orders,
            "include_fills": self.include_fills,
            "include_positions": self.include_positions,
            "include_signals": self.include_signals,
            "include_risk_blocks": self.include_risk_blocks,
            "include_equity_curve": self.include_equity_curve,
            "include_drawdown_curve": self.include_drawdown_curve,
            "include_model_analytics": self.include_model_analytics,
            "write_artifacts": self.write_artifacts,
            "artifact_format": _as_value(self.artifact_format),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Simulated time config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BacktestTimeConfig:
    """
    Configuration for backtest_time.py.
    """

    enabled: bool = True

    use_simulated_clock: bool = True
    allow_time_travel_backwards: bool = False

    scheduler_enabled: bool = True
    scheduler_emit_events: bool = True

    max_jobs_per_tick: int = 1000
    fail_on_job_error: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _ensure_positive_int(self.max_jobs_per_tick, "max_jobs_per_tick")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "use_simulated_clock": self.use_simulated_clock,
            "allow_time_travel_backwards": self.allow_time_travel_backwards,
            "scheduler_enabled": self.scheduler_enabled,
            "scheduler_emit_events": self.scheduler_emit_events,
            "max_jobs_per_tick": self.max_jobs_per_tick,
            "fail_on_job_error": self.fail_on_job_error,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Walk-forward / optimizer config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OptimizerConfig:
    """
    Configuration for optimizer.py.
    """

    enabled: bool = False

    mode: OptimizationMode | str = OptimizationMode.GRID_SEARCH
    objective: OptimizationObjective | str = OptimizationObjective.NET_PNL

    maximize_objective: bool = True

    max_trials: int = 100
    random_seed: int | None = 42

    fail_fast: bool = False
    parallel_trials: int = 1

    parameter_grid: dict[str, list[Any]] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.enabled:
            return

        _ensure_positive_int(self.max_trials, "max_trials")
        _ensure_positive_int(self.parallel_trials, "parallel_trials")

        if not self.parameter_grid:
            raise BacktestConfigError(
                "parameter_grid must not be empty when optimizer is enabled"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": _as_value(self.mode),
            "objective": _as_value(self.objective),
            "maximize_objective": self.maximize_objective,
            "max_trials": self.max_trials,
            "random_seed": self.random_seed,
            "fail_fast": self.fail_fast,
            "parallel_trials": self.parallel_trials,
            "parameter_grid": dict(self.parameter_grid),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class WalkForwardConfig:
    """
    Configuration for walk_forward.py.
    """

    enabled: bool = False

    window_mode: WalkForwardWindowMode | str = WalkForwardWindowMode.ROLLING

    train_window_ms: int = 90 * 24 * 60 * 60 * 1000
    test_window_ms: int = 30 * 24 * 60 * 60 * 1000
    step_ms: int = 30 * 24 * 60 * 60 * 1000

    require_full_windows: bool = True
    optimize_each_window: bool = True

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.enabled:
            return

        _ensure_positive_int(self.train_window_ms, "train_window_ms")
        _ensure_positive_int(self.test_window_ms, "test_window_ms")
        _ensure_positive_int(self.step_ms, "step_ms")

        if self.optimize_each_window:
            self.optimizer.enabled = True

        self.optimizer.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "window_mode": _as_value(self.window_mode),
            "train_window_ms": self.train_window_ms,
            "test_window_ms": self.test_window_ms,
            "step_ms": self.step_ms,
            "require_full_windows": self.require_full_windows,
            "optimize_each_window": self.optimize_each_window,
            "optimizer": self.optimizer.to_dict(),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Main backtest config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BacktestConfig:
    """
    Main configuration for one backtest run.

    This is the object passed into StrategyTester.
    """

    name: str = "backtest"

    mode: BacktestMode | str = BacktestMode.CANDLE

    exchange: ExchangeName | str = ExchangeName.BINANCE
    market_type: MarketType | str = MarketType.USDM_FUTURES

    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT"])
    timeframes: list[str] = field(default_factory=lambda: ["1m"])

    start_time_ms: int = 0
    end_time_ms: int = 0

    strategy_names: list[str] = field(default_factory=list)
    model_names: list[str] = field(default_factory=list)

    auto_download_missing_history: bool = False

    data: BacktestDataConfig = field(default_factory=BacktestDataConfig)
    replay: BacktestReplayConfig = field(default_factory=BacktestReplayConfig)
    execution: BacktestExecutionConfig = field(default_factory=BacktestExecutionConfig)
    metrics: BacktestMetricsConfig = field(default_factory=BacktestMetricsConfig)
    model_analytics: ModelAnalyticsConfig = field(default_factory=ModelAnalyticsConfig)
    report: BacktestReportConfig = field(default_factory=BacktestReportConfig)
    time: BacktestTimeConfig = field(default_factory=BacktestTimeConfig)
    history_download: HistoryDownloadConfig | None = None
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)

    emit_lifecycle_events: bool = True
    fail_fast: bool = True

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _ensure_non_empty_list(self.symbols, "symbols")
        _ensure_non_empty_list(self.timeframes, "timeframes")
        _ensure_time_range(self.start_time_ms, self.end_time_ms)

        self._sync_child_configs()

        self.data.validate()
        self.replay.validate()
        self.execution.validate()
        self.metrics.validate()
        self.model_analytics.validate()
        self.report.validate()
        self.time.validate()
        self.walk_forward.validate()

        if self.history_download is not None:
            self.history_download.validate()

    def _sync_child_configs(self) -> None:
        """
        Keep child configs aligned with main config.

        This avoids subtle bugs where BacktestConfig says BTCUSDT but
        BacktestDataConfig points to another symbol/date range.
        """

        self.data.exchange = self.exchange
        self.data.market_type = self.market_type
        self.data.symbols = list(self.symbols)
        self.data.timeframes = list(self.timeframes)
        self.data.start_time_ms = self.start_time_ms
        self.data.end_time_ms = self.end_time_ms

        if self.history_download is not None:
            self.history_download.exchange = self.exchange
            self.history_download.market_type = self.market_type
            self.history_download.symbols = list(self.symbols)
            self.history_download.timeframes = list(self.timeframes)
            self.history_download.start_time_ms = self.start_time_ms
            self.history_download.end_time_ms = self.end_time_ms

    def build_history_download_config(self) -> HistoryDownloadConfig:
        """
        Build a default HistoryDownloadConfig from the current backtest config.
        """

        config = self.history_download or HistoryDownloadConfig()

        config.exchange = self.exchange
        config.market_type = self.market_type
        config.symbols = list(self.symbols)
        config.timeframes = list(self.timeframes)
        config.start_time_ms = self.start_time_ms
        config.end_time_ms = self.end_time_ms
        config.output_dir = self.data.data_dir
        config.data_types = list(self.data.data_types)

        config.validate()
        return config

    def config_snapshot(self) -> dict[str, Any]:
        """
        Snapshot safe to store with BacktestRun and reports.

        Do not put secrets or API keys here.
        """

        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": _as_value(self.mode),
            "exchange": _as_value(self.exchange),
            "market_type": _as_value(self.market_type),
            "symbols": list(self.symbols),
            "timeframes": list(self.timeframes),
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "strategy_names": list(self.strategy_names),
            "model_names": list(self.model_names),
            "auto_download_missing_history": self.auto_download_missing_history,
            "data": self.data.to_dict(),
            "replay": self.replay.to_dict(),
            "execution": self.execution.to_dict(),
            "metrics": self.metrics.to_dict(),
            "model_analytics": self.model_analytics.to_dict(),
            "report": self.report.to_dict(),
            "time": self.time.to_dict(),
            "history_download": (
                self.history_download.to_dict()
                if self.history_download is not None
                else None
            ),
            "walk_forward": self.walk_forward.to_dict(),
            "emit_lifecycle_events": self.emit_lifecycle_events,
            "fail_fast": self.fail_fast,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def build_default_backtest_config(
    *,
    exchange: ExchangeName | str = ExchangeName.BINANCE,
    market_type: MarketType | str = MarketType.USDM_FUTURES,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    start_time_ms: int,
    end_time_ms: int,
    initial_balance: float = 10_000.0,
    leverage: float = 3.0,
    mode: BacktestMode | str = BacktestMode.CANDLE,
    data_dir: str = "data/history",
) -> BacktestConfig:
    """
    Convenient factory for a standard futures backtest config.
    """

    config = BacktestConfig(
        mode=mode,
        exchange=exchange,
        market_type=market_type,
        symbols=symbols or ["BTCUSDT"],
        timeframes=timeframes or ["1m"],
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
    )

    config.data.data_dir = data_dir
    config.execution.initial_balance = initial_balance
    config.execution.leverage = leverage

    if mode == BacktestMode.CANDLE or str(mode) == BacktestMode.CANDLE.value:
        config.data.data_types = [
            HistoryDataType.CANDLES,
            HistoryDataType.FUNDING,
            HistoryDataType.OPEN_INTEREST,
        ]
    elif mode == BacktestMode.TRADE_LEVEL or str(mode) == BacktestMode.TRADE_LEVEL.value:
        config.data.data_types = [
            HistoryDataType.CANDLES,
            HistoryDataType.AGG_TRADES,
            HistoryDataType.FUNDING,
            HistoryDataType.OPEN_INTEREST,
            HistoryDataType.LIQUIDATIONS,
        ]
    elif mode == BacktestMode.ORDERBOOK or str(mode) == BacktestMode.ORDERBOOK.value:
        config.data.data_types = [
            HistoryDataType.CANDLES,
            HistoryDataType.AGG_TRADES,
            HistoryDataType.ORDERBOOK_SNAPSHOTS,
            HistoryDataType.ORDERBOOK_DELTAS,
            HistoryDataType.FUNDING,
            HistoryDataType.OPEN_INTEREST,
            HistoryDataType.LIQUIDATIONS,
        ]

    config.validate()
    return config


def build_history_download_config_from_backtest(
    config: BacktestConfig,
) -> HistoryDownloadConfig:
    """
    Helper for StrategyTester or CLI.

    Uses the same symbol/date/data scope as the backtest config.
    """

    config.validate()
    return config.build_history_download_config()