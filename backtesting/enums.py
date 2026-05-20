"""
Backtesting-specific enums.

This module contains only enums that belong to the offline backtesting,
market replay, simulation, reporting, walk-forward and optimization layers.

It must not duplicate strategy/risk/execution domain enums unless the enum
describes a backtesting-only concept.
"""

from __future__ import annotations

from enum import Enum


class BacktestMode(str, Enum):
    """
    Main operating mode for a backtest run.
    """

    SINGLE_STRATEGY = "single_strategy"
    MULTI_STRATEGY = "multi_strategy"
    PORTFOLIO = "portfolio"
    WALK_FORWARD = "walk_forward"
    OPTIMIZATION = "optimization"
    STRESS_TEST = "stress_test"


class BacktestStatus(str, Enum):
    """
    Runtime status of a backtest run.
    """

    CREATED = "created"
    CONFIGURING = "configuring"
    LOADING_DATA = "loading_data"
    WARMING_UP = "warming_up"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BacktestPhase(str, Enum):
    """
    Internal phase of the backtesting pipeline.
    """

    INIT = "init"
    DATA_DOWNLOAD = "data_download"
    DATA_LOAD = "data_load"
    DATA_VALIDATION = "data_validation"
    COMPONENT_BOOTSTRAP = "component_bootstrap"
    WARMUP = "warmup"
    MARKET_REPLAY = "market_replay"
    SIGNAL_PROCESSING = "signal_processing"
    RISK_PROCESSING = "risk_processing"
    EXECUTION_SIMULATION = "execution_simulation"
    POSITION_SIMULATION = "position_simulation"
    METRICS_CALCULATION = "metrics_calculation"
    REPORT_BUILDING = "report_building"
    CLEANUP = "cleanup"


class BacktestDataType(str, Enum):
    """
    Historical data streams supported by the backtesting package.
    """

    CANDLES = "candles"
    TRADES = "trades"
    ORDERBOOK = "orderbook"
    ORDERBOOK_SNAPSHOT = "orderbook_snapshot"
    FUNDING = "funding"
    OPEN_INTEREST = "open_interest"
    LIQUIDATIONS = "liquidations"
    MARK_PRICE = "mark_price"
    INDEX_PRICE = "index_price"
    MULTI_STREAM = "multi_stream"


class HistoricalDataFormat(str, Enum):
    """
    Supported historical data file/storage formats.
    """

    PARQUET = "parquet"
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    POSTGRES = "postgres"
    REDIS_SNAPSHOT = "redis_snapshot"


class DataValidationLevel(str, Enum):
    """
    Strictness level for validating historical data before replay.
    """

    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    FAIL_FAST = "fail_fast"


class DataGapPolicy(str, Enum):
    """
    Policy for handling gaps in historical data.
    """

    IGNORE = "ignore"
    WARN = "warn"
    SKIP_RANGE = "skip_range"
    FORWARD_FILL = "forward_fill"
    FAIL = "fail"


class DataAlignmentPolicy(str, Enum):
    """
    Policy for aligning multiple historical streams by timestamp.
    """

    EVENT_TIME = "event_time"
    RECEIVED_TIME = "received_time"
    CANDLE_CLOSE_TIME = "candle_close_time"
    NEAREST = "nearest"
    STRICT_TIMESTAMP = "strict_timestamp"


class ReplayMode(str, Enum):
    """
    Market replay mode.
    """

    FULL_RUN = "full_run"
    STEP_BY_STEP = "step_by_step"
    BATCHED = "batched"
    EVENT_BY_EVENT = "event_by_event"


class ReplaySpeed(str, Enum):
    """
    Replay speed mode.

    MAX_SPEED should be the default for offline deterministic tests.
    REALTIME is mostly useful for debugging dashboards or event flow.
    """

    REALTIME = "realtime"
    FAST = "fast"
    MAX_SPEED = "max_speed"
    STEP_BY_STEP = "step_by_step"


class ReplayOrdering(str, Enum):
    """
    Ordering rule for replaying mixed historical streams.
    """

    TIMESTAMP_ASC = "timestamp_asc"
    TIMESTAMP_THEN_PRIORITY = "timestamp_then_priority"
    STREAM_PRIORITY_THEN_TIMESTAMP = "stream_priority_then_timestamp"


class ReplayEventPriority(str, Enum):
    """
    Priority of market events when several events share the same timestamp.

    This helps make replay deterministic.
    """

    ORDERBOOK = "orderbook"
    TRADE = "trade"
    CANDLE = "candle"
    FUNDING = "funding"
    OPEN_INTEREST = "open_interest"
    LIQUIDATION = "liquidation"
    MARK_PRICE = "mark_price"
    INDEX_PRICE = "index_price"


class WarmupPolicy(str, Enum):
    """
    Policy for warmup period handling.
    """

    NONE = "none"
    LOAD_ONLY = "load_only"
    REPLAY_WITHOUT_TRADING = "replay_without_trading"
    REPLAY_WITH_TRADING_DISABLED = "replay_with_trading_disabled"


class FillModel(str, Enum):
    """
    Order fill simulation model.
    """

    INSTANT = "instant"
    NEXT_TICK = "next_tick"
    NEXT_CANDLE_OPEN = "next_candle_open"
    NEXT_CANDLE_CLOSE = "next_candle_close"
    VWAP = "vwap"
    OHLC_PATH = "ohlc_path"
    ORDERBOOK_DEPTH = "orderbook_depth"
    PROBABILISTIC = "probabilistic"


class CandleExecutionPath(str, Enum):
    """
    Assumed intrabar path when only OHLC candles are available.

    This affects whether SL or TP is considered hit first inside one candle.
    """

    OPEN_HIGH_LOW_CLOSE = "open_high_low_close"
    OPEN_LOW_HIGH_CLOSE = "open_low_high_close"
    CONSERVATIVE = "conservative"
    OPTIMISTIC = "optimistic"
    RANDOMIZED = "randomized"


class SlippageModel(str, Enum):
    """
    Slippage simulation model.
    """

    NONE = "none"
    FIXED_BPS = "fixed_bps"
    FIXED_PRICE = "fixed_price"
    PERCENT_OF_SPREAD = "percent_of_spread"
    VOLUME_BASED = "volume_based"
    VOLATILITY_BASED = "volatility_based"
    ORDERBOOK_DEPTH = "orderbook_depth"
    ADVERSE_SELECTION = "adverse_selection"


class CommissionModel(str, Enum):
    """
    Commission/fee simulation model.
    """

    NONE = "none"
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    MAKER_TAKER = "maker_taker"
    EXCHANGE_SPECIFIC = "exchange_specific"


class FundingSimulationMode(str, Enum):
    """
    Funding cost simulation mode for perpetual futures.
    """

    DISABLED = "disabled"
    APPLY_ON_FUNDING_TIMESTAMP = "apply_on_funding_timestamp"
    PRORATED_CONTINUOUS = "prorated_continuous"
    ESTIMATED_FROM_RATE = "estimated_from_rate"


class LiquidityModel(str, Enum):
    """
    Liquidity constraint model for simulated fills.
    """

    UNLIMITED = "unlimited"
    CANDLE_VOLUME_PERCENT = "candle_volume_percent"
    TRADE_VOLUME_PERCENT = "trade_volume_percent"
    ORDERBOOK_DEPTH = "orderbook_depth"
    PROBABILISTIC = "probabilistic"


class LatencyModel(str, Enum):
    """
    Simulated latency model.
    """

    NONE = "none"
    FIXED_MS = "fixed_ms"
    RANDOM_MS = "random_ms"
    DISTRIBUTION = "distribution"


class OrderRejectionReason(str, Enum):
    """
    Backtesting-specific order rejection reasons.
    """

    NONE = "none"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    INSUFFICIENT_MARGIN = "insufficient_margin"
    INSUFFICIENT_LIQUIDITY = "insufficient_liquidity"
    PRICE_OUT_OF_RANGE = "price_out_of_range"
    MARKET_CLOSED = "market_closed"
    KILL_SWITCH_ACTIVE = "kill_switch_active"
    RISK_REJECTED = "risk_rejected"
    INVALID_ORDER = "invalid_order"
    DUPLICATE_ORDER = "duplicate_order"
    SIMULATION_ERROR = "simulation_error"


class SimulatedOrderStatus(str, Enum):
    """
    Status of a simulated order inside the backtesting execution layer.

    This is intentionally separate from production execution enums because
    backtesting may need additional simulation-only states.
    """

    CREATED = "created"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class SimulatedPositionStatus(str, Enum):
    """
    Status of a simulated position.
    """

    NONE = "none"
    OPENING = "opening"
    OPEN = "open"
    REDUCING = "reducing"
    CLOSING = "closing"
    CLOSED = "closed"
    LIQUIDATED = "liquidated"


class PositionAccountingMode(str, Enum):
    """
    Position accounting mode.
    """

    NETTING = "netting"
    HEDGE = "hedge"


class PnLAccountingMode(str, Enum):
    """
    PnL accounting mode.
    """

    REALIZED_ONLY = "realized_only"
    MARK_TO_MARKET = "mark_to_market"
    REALIZED_AND_UNREALIZED = "realized_and_unrealized"


class EquityUpdateMode(str, Enum):
    """
    How often the equity curve should be updated.
    """

    ON_EVERY_EVENT = "on_every_event"
    ON_TRADE = "on_trade"
    ON_POSITION_UPDATE = "on_position_update"
    ON_CANDLE_CLOSE = "on_candle_close"
    INTERVAL = "interval"


class BacktestEventType(str, Enum):
    """
    Internal backtesting event categories.

    These are not replacements for production EventBus topics. They are used
    for recording, reporting and diagnostics.
    """

    MARKET = "market"
    ANALYTICS = "analytics"
    STRATEGY = "strategy"
    SIGNAL = "signal"
    RISK = "risk"
    EXECUTION = "execution"
    POSITION = "position"
    COST = "cost"
    METRIC = "metric"
    SYSTEM = "system"


class SignalOutcome(str, Enum):
    """
    Final observed outcome of a generated signal in backtest analytics.
    """

    GENERATED = "generated"
    REJECTED_BY_STRATEGY = "rejected_by_strategy"
    CONFIRMED_BY_RISK = "confirmed_by_risk"
    BLOCKED_BY_RISK = "blocked_by_risk"
    ORDER_REJECTED = "order_rejected"
    ORDER_FILLED = "order_filled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED_WIN = "position_closed_win"
    POSITION_CLOSED_LOSS = "position_closed_loss"
    POSITION_CLOSED_BREAKEVEN = "position_closed_breakeven"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


class TradeOutcome(str, Enum):
    """
    Final outcome of a completed simulated trade.
    """

    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    LIQUIDATED = "liquidated"
    CANCELLED = "cancelled"
    OPEN = "open"


class MetricAggregation(str, Enum):
    """
    Aggregation level for performance metrics.
    """

    SYSTEM = "system"
    STRATEGY = "strategy"
    SYMBOL = "symbol"
    TIMEFRAME = "timeframe"
    SETUP_TYPE = "setup_type"
    MARKET_REGIME = "market_regime"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class PerformanceMetric(str, Enum):
    """
    Common performance metrics used for reporting, ranking and optimization.
    """

    NET_PROFIT = "net_profit"
    NET_PROFIT_PCT = "net_profit_pct"
    GROSS_PROFIT = "gross_profit"
    GROSS_LOSS = "gross_loss"
    TOTAL_TRADES = "total_trades"
    WIN_RATE = "win_rate"
    LOSS_RATE = "loss_rate"
    PROFIT_FACTOR = "profit_factor"
    EXPECTANCY = "expectancy"
    EXPECTANCY_R = "expectancy_r"
    AVERAGE_TRADE = "average_trade"
    AVERAGE_WIN = "average_win"
    AVERAGE_LOSS = "average_loss"
    BEST_TRADE = "best_trade"
    WORST_TRADE = "worst_trade"
    MAX_DRAWDOWN = "max_drawdown"
    MAX_DRAWDOWN_PCT = "max_drawdown_pct"
    AVERAGE_DRAWDOWN = "average_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    RECOVERY_FACTOR = "recovery_factor"
    PAYOFF_RATIO = "payoff_ratio"
    EXPOSURE_TIME_PCT = "exposure_time_pct"
    TOTAL_FEES = "total_fees"
    TOTAL_SLIPPAGE = "total_slippage"
    TOTAL_FUNDING = "total_funding"


class OptimizationMetric(str, Enum):
    """
    Objective metrics supported by optimizer.
    """

    NET_PROFIT = "net_profit"
    NET_PROFIT_PCT = "net_profit_pct"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    PROFIT_FACTOR = "profit_factor"
    EXPECTANCY = "expectancy"
    EXPECTANCY_R = "expectancy_r"
    MAX_DRAWDOWN = "max_drawdown"
    MAX_DRAWDOWN_PCT = "max_drawdown_pct"
    WIN_RATE = "win_rate"
    RECOVERY_FACTOR = "recovery_factor"
    CUSTOM = "custom"


class OptimizationDirection(str, Enum):
    """
    Optimization direction for a metric.
    """

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class OptimizationMethod(str, Enum):
    """
    Supported optimization algorithms.
    """

    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    HYPERBAND = "hyperband"


class ParameterSampling(str, Enum):
    """
    Parameter sampling distribution for optimizer.
    """

    GRID = "grid"
    UNIFORM = "uniform"
    LOG_UNIFORM = "log_uniform"
    NORMAL = "normal"
    CHOICE = "choice"
    BOOLEAN = "boolean"


class OverfittingCheckMode(str, Enum):
    """
    Overfitting detection mode for optimized strategies.
    """

    DISABLED = "disabled"
    TRAIN_TEST_SPLIT = "train_test_split"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"
    PARAMETER_STABILITY = "parameter_stability"


class WalkForwardMode(str, Enum):
    """
    Walk-forward evaluation mode.
    """

    ANCHORED = "anchored"
    ROLLING = "rolling"
    EXPANDING = "expanding"


class WalkForwardWindowType(str, Enum):
    """
    Walk-forward window role.
    """

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ReportFormat(str, Enum):
    """
    Supported report output formats.
    """

    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    CSV = "csv"
    PARQUET = "parquet"


class ReportSection(str, Enum):
    """
    Sections that may be included in a backtest report.
    """

    SUMMARY = "summary"
    EQUITY_CURVE = "equity_curve"
    DRAWDOWN = "drawdown"
    TRADES = "trades"
    POSITIONS = "positions"
    STRATEGIES = "strategies"
    SYMBOLS = "symbols"
    TIMEFRAMES = "timeframes"
    RISK = "risk"
    EXECUTION = "execution"
    COSTS = "costs"
    FUNDING = "funding"
    SIGNALS = "signals"
    REGIMES = "regimes"
    WALK_FORWARD = "walk_forward"
    OPTIMIZATION = "optimization"
    WARNINGS = "warnings"


class BacktestArtifactType(str, Enum):
    """
    Output artifact types produced by a backtest run.
    """

    RESULT_JSON = "result_json"
    REPORT_MARKDOWN = "report_markdown"
    REPORT_HTML = "report_html"
    TRADES_CSV = "trades_csv"
    POSITIONS_CSV = "positions_csv"
    EQUITY_CURVE_CSV = "equity_curve_csv"
    EVENTS_JSONL = "events_jsonl"
    METRICS_JSON = "metrics_json"
    OPTIMIZATION_RESULTS = "optimization_results"


class BacktestWarningLevel(str, Enum):
    """
    Severity level for warnings generated during a backtest.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class StressScenario(str, Enum):
    """
    Optional stress scenarios for robustness testing.
    """

    NONE = "none"
    HIGH_SLIPPAGE = "high_slippage"
    HIGH_FEES = "high_fees"
    LOW_LIQUIDITY = "low_liquidity"
    LATENCY_SPIKE = "latency_spike"
    FUNDING_SPIKE = "funding_spike"
    GAP_MOVE = "gap_move"
    FLASH_CRASH = "flash_crash"
    EXCHANGE_OUTAGE = "exchange_outage"


__all__ = [
    "BacktestMode",
    "BacktestStatus",
    "BacktestPhase",
    "BacktestDataType",
    "HistoricalDataFormat",
    "DataValidationLevel",
    "DataGapPolicy",
    "DataAlignmentPolicy",
    "ReplayMode",
    "ReplaySpeed",
    "ReplayOrdering",
    "ReplayEventPriority",
    "WarmupPolicy",
    "FillModel",
    "CandleExecutionPath",
    "SlippageModel",
    "CommissionModel",
    "FundingSimulationMode",
    "LiquidityModel",
    "LatencyModel",
    "OrderRejectionReason",
    "SimulatedOrderStatus",
    "SimulatedPositionStatus",
    "PositionAccountingMode",
    "PnLAccountingMode",
    "EquityUpdateMode",
    "BacktestEventType",
    "SignalOutcome",
    "TradeOutcome",
    "MetricAggregation",
    "PerformanceMetric",
    "OptimizationMetric",
    "OptimizationDirection",
    "OptimizationMethod",
    "ParameterSampling",
    "OverfittingCheckMode",
    "WalkForwardMode",
    "WalkForwardWindowType",
    "ReportFormat",
    "ReportSection",
    "BacktestArtifactType",
    "BacktestWarningLevel",
    "StressScenario",
]