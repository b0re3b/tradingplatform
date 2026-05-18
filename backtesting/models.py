# trading_system/backtesting/models.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from backtesting.enums import (
    BacktestArtifactType,
    BacktestMode,
    BacktestStatus,
    ConfidenceBucket,
    DataQualityStatus,
    DrawdownMode,
    EquityCurveMode,
    ExchangeName,
    ExecutionSimulationMode,
    FeeModelType,
    FillType,
    FundingPaymentSide,
    GapHandlingPolicy,
    HistoryDataType,
    HistoryDownloadStatus,
    HistorySourceType,
    LatencyModelType,
    LiquidityRole,
    MarginMode,
    MarketRegime,
    MarketType,
    ModelAnalyticsScope,
    OptimizationMode,
    OptimizationObjective,
    OrderRejectReason,
    OrderSide,
    OrderSimulationType,
    OrderStatus,
    OrderType,
    PositionCloseReason,
    PositionSide,
    PositionStatus,
    PredictionOutcome,
    ReplayMode,
    ReportFormat,
    SignalDecision,
    SlippageModelType,
    StorageFormat,
    TimeInForce,
    WalkForwardWindowMode,
)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


# ---------------------------------------------------------------------------
# Historical events / replay models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HistoricalMarketEvent:
    """
    Canonical historical event replayed into EventBus.

    This is the bridge between local historical data and the live-style
    event-driven architecture.

    Example:
        topic="market.candle"
        payload={... normalized candle payload ...}
    """

    topic: str
    timestamp_ms: int
    payload: dict[str, Any]
    source: str = "backtest.history"
    sequence: int | None = None

    exchange: str | None = None
    market_type: str | None = None
    symbol: str | None = None
    timeframe: str | None = None
    data_type: str | None = None

    def to_event_payload(self) -> dict[str, Any]:
        return dict(self.payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "timestamp_ms": self.timestamp_ms,
            "payload": dict(self.payload),
            "source": self.source,
            "sequence": self.sequence,
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "data_type": self.data_type,
        }


@dataclass(slots=True)
class ReplayProgress:
    run_id: str
    started_at_ms: int
    current_timestamp_ms: int | None = None
    completed_at_ms: int | None = None

    total_events: int | None = None
    processed_events: int = 0

    start_time_ms: int | None = None
    end_time_ms: int | None = None

    progress_pct: float = 0.0
    last_topic: str | None = None
    last_symbol: str | None = None

    def mark_processed(self, event: HistoricalMarketEvent) -> None:
        self.processed_events += 1
        self.current_timestamp_ms = event.timestamp_ms
        self.last_topic = event.topic
        self.last_symbol = event.symbol

        if self.total_events and self.total_events > 0:
            self.progress_pct = min(
                100.0,
                (self.processed_events / self.total_events) * 100.0,
            )


# ---------------------------------------------------------------------------
# History download / ingestion models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HistoryDownloadRequest:
    """
    Request for downloading historical futures data from exchange/provider APIs.
    """

    exchange: ExchangeName | str
    market_type: MarketType | str
    symbols: list[str]
    start_time_ms: int
    end_time_ms: int

    data_types: list[HistoryDataType | str]
    timeframes: list[str] = field(default_factory=lambda: ["1m"])

    source_type: HistorySourceType | str = HistorySourceType.EXCHANGE_REST
    storage_format: StorageFormat | str = StorageFormat.PARQUET
    output_dir: str = "data/history"

    overwrite_existing: bool = False
    validate_after_download: bool = True

    request_id: str = field(default_factory=lambda: _new_id("history_req"))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HistoryDownloadProgress:
    request_id: str
    exchange: str
    market_type: str
    symbol: str
    data_type: str

    status: HistoryDownloadStatus | str = HistoryDownloadStatus.PENDING

    timeframe: str | None = None
    started_at_ms: int | None = None
    completed_at_ms: int | None = None

    current_start_time_ms: int | None = None
    current_end_time_ms: int | None = None

    downloaded_rows: int = 0
    written_rows: int = 0
    failed_rows: int = 0

    progress_pct: float = 0.0
    message: str | None = None


@dataclass(slots=True)
class HistoryDownloadResult:
    request_id: str
    exchange: str
    market_type: str

    status: HistoryDownloadStatus | str

    symbols: list[str]
    data_types: list[str]
    timeframes: list[str]

    start_time_ms: int
    end_time_ms: int

    output_dir: str
    files_written: list[str] = field(default_factory=list)

    downloaded_rows: int = 0
    written_rows: int = 0
    failed_rows: int = 0

    started_at_ms: int | None = None
    completed_at_ms: int | None = None

    errors: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return str(self.status) == HistoryDownloadStatus.COMPLETED.value and not self.errors


@dataclass(slots=True)
class RawHistoryBatch:
    """
    Raw rows received from an exchange/provider before normalization.
    """

    exchange: str
    market_type: str
    symbol: str
    data_type: str

    rows: list[Any]

    timeframe: str | None = None
    start_time_ms: int | None = None
    end_time_ms: int | None = None

    source: str = "exchange_rest"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NormalizedHistoryBatch:
    """
    Normalized rows ready for local storage and later conversion into
    HistoricalMarketEvent.
    """

    exchange: str
    market_type: str
    symbol: str
    data_type: str

    rows: list[dict[str, Any]]

    timeframe: str | None = None
    start_time_ms: int | None = None
    end_time_ms: int | None = None

    storage_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DataQualityReport:
    """
    Data quality summary for downloaded or loaded history.
    """

    exchange: str
    market_type: str
    symbol: str
    data_type: str

    status: DataQualityStatus | str

    timeframe: str | None = None
    start_time_ms: int | None = None
    end_time_ms: int | None = None

    rows: int = 0
    duplicate_rows: int = 0
    gap_count: int = 0
    out_of_order_rows: int = 0
    invalid_rows: int = 0

    gap_policy: GapHandlingPolicy | str = GapHandlingPolicy.WARN
    duplicate_policy: str = "keep_last"

    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_usable(self) -> bool:
        return self.status in {
            DataQualityStatus.VALID,
            DataQualityStatus.PARTIAL,
            DataQualityStatus.HAS_GAPS,
            DataQualityStatus.HAS_DUPLICATES,
        }


# ---------------------------------------------------------------------------
# Backtest run / lifecycle models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BacktestRun:
    """
    Runtime descriptor of one backtest session.
    """

    run_id: str = field(default_factory=lambda: _new_id("backtest"))

    status: BacktestStatus | str = BacktestStatus.CREATED
    mode: BacktestMode | str = BacktestMode.CANDLE
    replay_mode: ReplayMode | str = ReplayMode.FAST

    exchange: str = ExchangeName.BINANCE.value
    market_type: str = MarketType.USDM_FUTURES.value
    symbols: list[str] = field(default_factory=list)
    timeframes: list[str] = field(default_factory=lambda: ["1m"])

    start_time_ms: int | None = None
    end_time_ms: int | None = None

    initial_balance: float = 0.0
    final_balance: float | None = None

    strategy_names: list[str] = field(default_factory=list)
    model_names: list[str] = field(default_factory=list)

    created_at_ms: int | None = None
    started_at_ms: int | None = None
    completed_at_ms: int | None = None

    config_snapshot: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def mark_status(self, status: BacktestStatus | str) -> None:
        self.status = status


@dataclass(slots=True)
class BacktestArtifact:
    run_id: str
    artifact_type: BacktestArtifactType | str
    path: str

    storage_format: StorageFormat | str | None = None
    created_at_ms: int | None = None
    rows: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BacktestResult:
    """
    Final output of StrategyTester.
    """

    run: BacktestRun

    status: BacktestStatus | str
    started_at_ms: int | None = None
    completed_at_ms: int | None = None

    initial_balance: float = 0.0
    final_balance: float = 0.0
    net_pnl: float = 0.0
    total_return_pct: float = 0.0

    metrics: dict[str, Any] = field(default_factory=dict)
    model_analytics: dict[str, Any] = field(default_factory=dict)

    artifacts: list[BacktestArtifact] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == BacktestStatus.COMPLETED and not self.errors


# ---------------------------------------------------------------------------
# Signals / model prediction records
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BacktestSignalRecord:
    """
    Signal lifecycle record used by metrics, model analytics and reports.
    """

    signal_id: str
    run_id: str

    timestamp_ms: int
    exchange: str
    market_type: str
    symbol: str

    side: PositionSide | str
    decision: SignalDecision | str = SignalDecision.GENERATED

    strategy_name: str | None = None
    model_name: str | None = None
    signal_type: str | None = None
    timeframe: str | None = None

    confidence: float | None = None
    score: float | None = None

    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None

    reason: str | None = None
    features: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BacktestRiskBlock:
    """
    Record of a signal blocked by risk.
    """

    block_id: str = field(default_factory=lambda: _new_id("risk_block"))

    run_id: str = ""
    signal_id: str | None = None

    timestamp_ms: int = 0
    exchange: str | None = None
    market_type: str | None = None
    symbol: str | None = None

    reason: str = ""
    risk_rule: str | None = None
    severity: str | None = None

    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelPredictionRecord:
    """
    One prediction/decision produced by strategy, statistical model or LLM.
    """

    prediction_id: str = field(default_factory=lambda: _new_id("prediction"))

    run_id: str = ""
    timestamp_ms: int = 0

    exchange: str = ExchangeName.BINANCE.value
    market_type: str = MarketType.USDM_FUTURES.value
    symbol: str = ""

    model_name: str = ""
    strategy_name: str | None = None
    signal_id: str | None = None

    side: PositionSide | str | None = None
    confidence: float | None = None
    score: float | None = None

    predicted_outcome: str | None = None
    market_regime: MarketRegime | str | None = None
    timeframe: str | None = None

    features: dict[str, Any] = field(default_factory=dict)
    explanation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelOutcomeRecord:
    """
    Realized outcome linked to a model prediction or signal.
    """

    prediction_id: str

    run_id: str = ""
    signal_id: str | None = None
    trade_id: str | None = None
    position_id: str | None = None

    timestamp_ms: int = 0

    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0
    was_profitable: bool = False

    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

    time_to_peak_ms: int | None = None
    time_to_close_ms: int | None = None

    outcome_label: PredictionOutcome | str = PredictionOutcome.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Execution simulation models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BacktestOrder:
    """
    Simulated order record.
    """

    order_id: str = field(default_factory=lambda: _new_id("order"))

    run_id: str = ""
    signal_id: str | None = None
    position_id: str | None = None

    exchange: str = ExchangeName.BINANCE.value
    market_type: str = MarketType.USDM_FUTURES.value
    symbol: str = ""

    side: OrderSide | str = OrderSide.BUY
    position_side: PositionSide | str | None = None

    order_type: OrderType | str = OrderType.MARKET
    time_in_force: TimeInForce | str = TimeInForce.GTC

    status: OrderStatus | str = OrderStatus.CREATED

    quantity: float = 0.0
    price: float | None = None
    stop_price: float | None = None

    reduce_only: bool = False
    post_only: bool = False

    created_at_ms: int = 0
    submitted_at_ms: int | None = None
    accepted_at_ms: int | None = None
    filled_at_ms: int | None = None
    cancelled_at_ms: int | None = None
    rejected_at_ms: int | None = None

    filled_quantity: float = 0.0
    avg_fill_price: float | None = None

    reject_reason: OrderRejectReason | str | None = None
    reject_message: str | None = None

    simulation_type: OrderSimulationType | str | None = None
    liquidity_role: LiquidityRole | str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_quantity(self) -> float:
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }


@dataclass(slots=True)
class BacktestFill:
    """
    Simulated fill produced by ExecutionSimulator.
    """

    fill_id: str = field(default_factory=lambda: _new_id("fill"))

    run_id: str = ""
    order_id: str = ""
    signal_id: str | None = None
    position_id: str | None = None

    exchange: str = ExchangeName.BINANCE.value
    market_type: str = MarketType.USDM_FUTURES.value
    symbol: str = ""

    side: OrderSide | str = OrderSide.BUY
    position_side: PositionSide | str | None = None

    fill_type: FillType | str = FillType.FULL
    liquidity_role: LiquidityRole | str = LiquidityRole.TAKER

    timestamp_ms: int = 0

    quantity: float = 0.0
    price: float = 0.0
    notional: float = 0.0

    fee: float = 0.0
    fee_asset: str = "USDT"

    slippage: float = 0.0
    slippage_pct: float = 0.0

    latency_ms: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BacktestTrade:
    """
    Closed trade record used for performance metrics.
    """

    trade_id: str = field(default_factory=lambda: _new_id("trade"))

    run_id: str = ""
    signal_id: str | None = None
    position_id: str | None = None

    exchange: str = ExchangeName.BINANCE.value
    market_type: str = MarketType.USDM_FUTURES.value
    symbol: str = ""

    side: PositionSide | str = PositionSide.LONG

    strategy_name: str | None = None
    model_name: str | None = None
    timeframe: str | None = None

    opened_at_ms: int = 0
    closed_at_ms: int = 0

    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0

    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    pnl_pct: float = 0.0

    fees_paid: float = 0.0
    funding_paid: float = 0.0
    funding_received: float = 0.0
    slippage_cost: float = 0.0

    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

    close_reason: PositionCloseReason | str = PositionCloseReason.SIGNAL_EXIT
    was_liquidated: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def was_profitable(self) -> bool:
        return self.net_pnl > 0.0

    @property
    def holding_time_ms(self) -> int:
        return max(0, self.closed_at_ms - self.opened_at_ms)


# ---------------------------------------------------------------------------
# Position / portfolio simulation models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BacktestPosition:
    """
    Simulated futures position.
    """

    position_id: str = field(default_factory=lambda: _new_id("position"))

    run_id: str = ""
    signal_id: str | None = None

    exchange: str = ExchangeName.BINANCE.value
    market_type: str = MarketType.USDM_FUTURES.value
    symbol: str = ""

    side: PositionSide | str = PositionSide.LONG
    status: PositionStatus | str = PositionStatus.OPEN
    margin_mode: MarginMode | str = MarginMode.ISOLATED

    leverage: float = 1.0

    quantity: float = 0.0
    entry_price: float = 0.0
    mark_price: float = 0.0

    notional: float = 0.0
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0

    liquidation_price: float | None = None

    stop_loss: float | None = None
    take_profit: float | None = None
    trailing_stop: float | None = None

    opened_at_ms: int = 0
    updated_at_ms: int = 0
    closed_at_ms: int | None = None

    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    fees_paid: float = 0.0
    funding_paid: float = 0.0
    funding_received: float = 0.0

    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

    close_reason: PositionCloseReason | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.status in {PositionStatus.OPEN, PositionStatus.PARTIALLY_CLOSED}

    @property
    def net_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl - self.fees_paid - self.funding_paid + self.funding_received


@dataclass(slots=True)
class FundingPaymentRecord:
    payment_id: str = field(default_factory=lambda: _new_id("funding"))

    run_id: str = ""
    position_id: str = ""

    exchange: str = ExchangeName.BINANCE.value
    market_type: str = MarketType.USDM_FUTURES.value
    symbol: str = ""

    timestamp_ms: int = 0

    side: FundingPaymentSide | str = FundingPaymentSide.NONE
    funding_rate: float = 0.0
    notional: float = 0.0
    amount: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Equity / drawdown / metrics models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BacktestEquityPoint:
    run_id: str
    timestamp_ms: int

    balance: float
    equity: float

    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    fees_paid: float = 0.0
    funding_paid: float = 0.0
    funding_received: float = 0.0

    open_positions: int = 0
    mode: EquityCurveMode | str = EquityCurveMode.ON_POSITION_UPDATE

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BacktestDrawdownPoint:
    run_id: str
    timestamp_ms: int

    equity: float
    peak_equity: float

    drawdown: float
    drawdown_pct: float

    mode: DrawdownMode | str = DrawdownMode.EQUITY_BASED


@dataclass(slots=True)
class PerformanceSummary:
    """
    Aggregated financial performance result.
    """

    run_id: str

    initial_balance: float = 0.0
    final_balance: float = 0.0

    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    total_return_pct: float = 0.0

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0

    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0

    average_win: float = 0.0
    average_loss: float = 0.0

    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    sharpe: float | None = None
    sortino: float | None = None
    calmar: float | None = None

    fees_paid: float = 0.0
    funding_paid: float = 0.0
    funding_received: float = 0.0
    slippage_cost: float = 0.0

    liquidation_count: int = 0
    risk_blocks: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model analytics models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ConfidenceBucketStats:
    bucket: ConfidenceBucket | str

    predictions: int = 0
    executed_predictions: int = 0

    win_rate: float = 0.0
    avg_pnl: float = 0.0
    total_pnl: float = 0.0

    directional_accuracy: float = 0.0
    avg_confidence: float = 0.0


@dataclass(slots=True)
class ModelAnalyticsResult:
    """
    Aggregated quality analysis for model/strategy/signal decisions.
    """

    run_id: str
    model_name: str

    scope: ModelAnalyticsScope | str = ModelAnalyticsScope.MODEL

    total_predictions: int = 0
    executed_predictions: int = 0

    win_rate: float = 0.0
    directional_accuracy: float = 0.0

    avg_realized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    profit_factor: float = 0.0

    confidence_correlation: float | None = None
    calibration_error: float | None = None

    avg_mfe: float = 0.0
    avg_mae: float = 0.0

    false_positive_rate: float | None = None
    false_negative_rate: float | None = None

    by_confidence_bucket: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_symbol: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_timeframe: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_regime: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_strategy: dict[str, dict[str, Any]] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cost model records
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FeeCalculationResult:
    fee_model: FeeModelType | str

    fee: float
    fee_asset: str = "USDT"

    liquidity_role: LiquidityRole | str = LiquidityRole.TAKER
    rate: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SlippageCalculationResult:
    slippage_model: SlippageModelType | str

    requested_price: float
    executed_price: float

    slippage: float
    slippage_pct: float

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LatencyCalculationResult:
    latency_model: LatencyModelType | str

    latency_ms: int
    requested_timestamp_ms: int
    effective_timestamp_ms: int

    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Report / optimizer / walk-forward models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BacktestReport:
    run_id: str
    format: ReportFormat | str

    path: str | None = None
    generated_at_ms: int | None = None

    summary: dict[str, Any] = field(default_factory=dict)
    sections: dict[str, Any] = field(default_factory=dict)
    artifacts: list[BacktestArtifact] = field(default_factory=list)


@dataclass(slots=True)
class ParameterGrid:
    parameters: dict[str, list[Any]]

    def total_combinations(self) -> int:
        total = 1
        for values in self.parameters.values():
            total *= max(1, len(values))
        return total


@dataclass(slots=True)
class OptimizationTrial:
    trial_id: str = field(default_factory=lambda: _new_id("trial"))

    parameter_values: dict[str, Any] = field(default_factory=dict)

    objective: OptimizationObjective | str = OptimizationObjective.NET_PNL
    objective_value: float | None = None

    run_id: str | None = None
    status: BacktestStatus | str = BacktestStatus.CREATED

    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class OptimizationResult:
    optimization_id: str = field(default_factory=lambda: _new_id("optimization"))

    mode: OptimizationMode | str = OptimizationMode.GRID_SEARCH
    objective: OptimizationObjective | str = OptimizationObjective.NET_PNL

    trials: list[OptimizationTrial] = field(default_factory=list)
    best_trial: OptimizationTrial | None = None

    started_at_ms: int | None = None
    completed_at_ms: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WalkForwardWindow:
    window_id: str = field(default_factory=lambda: _new_id("wf_window"))

    train_start_ms: int = 0
    train_end_ms: int = 0

    test_start_ms: int = 0
    test_end_ms: int = 0

    mode: WalkForwardWindowMode | str = WalkForwardWindowMode.ROLLING

    optimization_result: OptimizationResult | None = None
    test_result: BacktestResult | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WalkForwardResult:
    walk_forward_id: str = field(default_factory=lambda: _new_id("walk_forward"))

    mode: WalkForwardWindowMode | str = WalkForwardWindowMode.ROLLING

    windows: list[WalkForwardWindow] = field(default_factory=list)

    aggregate_metrics: dict[str, Any] = field(default_factory=dict)
    started_at_ms: int | None = None
    completed_at_ms: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)