# trading_system/backtesting/enums.py

from __future__ import annotations

from enum import StrEnum


class BacktestMode(StrEnum):
    """
    High-level backtest mode.

    Defines how deep the market replay should be.
    """

    CANDLE = "candle"
    TRADE_LEVEL = "trade_level"
    ORDERBOOK = "orderbook"
    MULTI_EXCHANGE = "multi_exchange"


class ReplayMode(StrEnum):
    """
    Replay behavior for historical market events.
    """

    FAST = "fast"
    REAL_TIME = "real_time"
    STEPPED = "stepped"


class BacktestStatus(StrEnum):
    """
    Lifecycle status of a backtest run.
    """

    CREATED = "created"
    INITIALIZING = "initializing"
    DOWNLOADING_HISTORY = "downloading_history"
    LOADING_DATA = "loading_data"
    REPLAYING = "replaying"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HistoryDataType(StrEnum):
    """
    Historical data types used by downloader, local storage and replay.

    All types are futures/perpetual-oriented.
    """

    CANDLES = "candles"
    TRADES = "trades"
    AGG_TRADES = "agg_trades"
    ORDERBOOK_SNAPSHOTS = "orderbook_snapshots"
    ORDERBOOK_DELTAS = "orderbook_deltas"
    FUNDING = "funding"
    OPEN_INTEREST = "open_interest"
    LIQUIDATIONS = "liquidations"
    MARK_PRICE = "mark_price"
    INDEX_PRICE = "index_price"


class HistorySourceType(StrEnum):
    """
    Source from which historical data is loaded or downloaded.
    """

    EXCHANGE_REST = "exchange_rest"
    EXCHANGE_DATA_PORTAL = "exchange_data_portal"
    PARQUET = "parquet"
    POSTGRES = "postgres"
    CSV = "csv"
    PROVIDER_API = "provider_api"


class HistoryDownloadStatus(StrEnum):
    """
    Status of one historical download task.
    """

    PENDING = "pending"
    RUNNING = "running"
    PARTIALLY_COMPLETED = "partially_completed"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RATE_LIMITED = "rate_limited"


class ExchangeName(StrEnum):
    """
    Supported futures exchanges for backtesting ingestion.

    Keep values aligned with normalized payload field `exchange`.
    """

    BINANCE = "binance"
    BYBIT = "bybit"
    OKX = "okx"
    MEXC = "mexc"


class MarketType(StrEnum):
    """
    Futures/perpetual market types.

    The project is futures-focused, so spot is intentionally not included.
    """

    USDM_FUTURES = "usdm_futures"
    COINM_FUTURES = "coinm_futures"
    LINEAR = "linear"
    INVERSE = "inverse"
    SWAP = "swap"


class HistoricalEventTopic(StrEnum):
    """
    Canonical market topics emitted by BacktestMarketReplay.

    These must match the live EventBus topics used by exchange adapters and data caches.
    """

    MARKET_CANDLE = "market.candle"
    MARKET_TRADE = "market.trade"
    MARKET_TRADES_SNAPSHOT = "market.trades.snapshot"
    MARKET_ORDERBOOK = "market.orderbook"
    MARKET_ORDERBOOK_SNAPSHOT = "market.orderbook.snapshot"
    MARKET_FUNDING = "market.funding"
    MARKET_FUNDING_SNAPSHOT = "market.funding.snapshot"
    MARKET_OPEN_INTEREST = "market.open_interest"
    MARKET_OPEN_INTEREST_SNAPSHOT = "market.open_interest.snapshot"
    MARKET_LIQUIDATION = "market.liquidation"
    MARKET_MARK_PRICE = "market.mark_price"
    MARKET_INDEX_PRICE = "market.index_price"


class BacktestSystemTopic(StrEnum):
    """
    Backtest-specific system topics emitted through EventBus.
    """

    BACKTEST_STARTED = "system.backtest.started"
    BACKTEST_COMPLETED = "system.backtest.completed"
    BACKTEST_FAILED = "system.backtest.failed"
    BACKTEST_CANCELLED = "system.backtest.cancelled"

    REPLAY_STARTED = "system.backtest.replay_started"
    REPLAY_PROGRESS = "system.backtest.progress"
    REPLAY_COMPLETED = "system.backtest.replay_completed"
    REPLAY_FAILED = "system.backtest.replay_failed"

    HISTORY_DOWNLOAD_STARTED = "system.backtest.history_download_started"
    HISTORY_DOWNLOAD_PROGRESS = "system.backtest.history_download_progress"
    HISTORY_DOWNLOAD_COMPLETED = "system.backtest.history_download_completed"
    HISTORY_DOWNLOAD_FAILED = "system.backtest.history_download_failed"


class SignalDecision(StrEnum):
    """
    Backtest-side signal decision state.

    This is useful for signal records, model analytics and reports.
    """

    GENERATED = "generated"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    EXPIRED = "expired"
    EXECUTED = "executed"
    CANCELLED = "cancelled"


class OrderSide(StrEnum):
    """
    Normalized trading side.
    """

    BUY = "buy"
    SELL = "sell"


class PositionSide(StrEnum):
    """
    Normalized futures position side.
    """

    LONG = "long"
    SHORT = "short"


class OrderType(StrEnum):
    """
    Simulated order type.
    """

    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT_MARKET = "take_profit_market"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


class TimeInForce(StrEnum):
    """
    Simulated time-in-force policy.
    """

    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    POST_ONLY = "post_only"


class OrderStatus(StrEnum):
    """
    Simulated order lifecycle status.
    """

    CREATED = "created"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderRejectReason(StrEnum):
    """
    Common simulated rejection reasons.
    """

    INSUFFICIENT_BALANCE = "insufficient_balance"
    INSUFFICIENT_MARGIN = "insufficient_margin"
    MAX_POSITION_LIMIT = "max_position_limit"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    KILL_SWITCH_ACTIVE = "kill_switch_active"
    INVALID_ORDER = "invalid_order"
    PRICE_OUT_OF_BOUNDS = "price_out_of_bounds"
    MIN_NOTIONAL_NOT_MET = "min_notional_not_met"
    RATE_LIMITED = "rate_limited"
    EXCHANGE_UNAVAILABLE = "exchange_unavailable"
    UNKNOWN = "unknown"


class ExecutionSimulationMode(StrEnum):
    """
    How strict the execution simulator should be.
    """

    IDEAL = "ideal"
    REALISTIC = "realistic"
    CONSERVATIVE = "conservative"
    ORDERBOOK_BASED = "orderbook_based"


class OrderSimulationType(StrEnum):
    """
    How simulated fills should be calculated.
    """

    INSTANT_FILL = "instant_fill"
    NEXT_TICK = "next_tick"
    NEXT_CANDLE_OPEN = "next_candle_open"
    NEXT_CANDLE_CLOSE = "next_candle_close"
    LIMIT_TOUCH = "limit_touch"
    ORDERBOOK_DEPTH = "orderbook_depth"


class FillType(StrEnum):
    """
    Type of simulated fill.
    """

    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"


class LiquidityRole(StrEnum):
    """
    Whether an order is simulated as maker or taker.
    """

    MAKER = "maker"
    TAKER = "taker"


class FeeModelType(StrEnum):
    """
    Fee model used by cost_models.py.
    """

    NONE = "none"
    FIXED = "fixed"
    BINANCE_FUTURES = "binance_futures"
    BYBIT_FUTURES = "bybit_futures"
    OKX_FUTURES = "okx_futures"
    MEXC_FUTURES = "mexc_futures"
    CUSTOM = "custom"


class SlippageModelType(StrEnum):
    """
    Slippage model used by cost_models.py.
    """

    NONE = "none"
    FIXED_TICKS = "fixed_ticks"
    FIXED_BPS = "fixed_bps"
    PERCENT = "percent"
    VOLATILITY_BASED = "volatility_based"
    ORDERBOOK_BASED = "orderbook_based"
    CUSTOM = "custom"


class LatencyModelType(StrEnum):
    """
    Latency model used by cost_models.py.
    """

    NONE = "none"
    FIXED = "fixed"
    RANDOM = "random"
    EXCHANGE_PROFILE = "exchange_profile"
    CUSTOM = "custom"


class PositionStatus(StrEnum):
    """
    Simulated position lifecycle status.
    """

    OPEN = "open"
    PARTIALLY_CLOSED = "partially_closed"
    CLOSED = "closed"
    LIQUIDATED = "liquidated"


class PositionCloseReason(StrEnum):
    """
    Why a simulated position was closed.
    """

    SIGNAL_EXIT = "signal_exit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    LIQUIDATION = "liquidation"
    MANUAL = "manual"
    END_OF_BACKTEST = "end_of_backtest"
    RISK_REDUCTION = "risk_reduction"


class MarginMode(StrEnum):
    """
    Futures margin mode.
    """

    CROSS = "cross"
    ISOLATED = "isolated"


class FundingPaymentSide(StrEnum):
    """
    Direction of funding payment from the account perspective.
    """

    PAID = "paid"
    RECEIVED = "received"
    NONE = "none"


class MetricPeriod(StrEnum):
    """
    Aggregation period for performance metrics.
    """

    TRADE = "trade"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    FULL_RUN = "full_run"


class ReportFormat(StrEnum):
    """
    Supported report export formats.
    """

    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    CSV = "csv"
    PARQUET = "parquet"


class EquityCurveMode(StrEnum):
    """
    How equity curve should be updated.
    """

    ON_TRADE_CLOSE = "on_trade_close"
    ON_POSITION_UPDATE = "on_position_update"
    ON_EVERY_MARKET_EVENT = "on_every_market_event"


class DrawdownMode(StrEnum):
    """
    How drawdown should be calculated.
    """

    REALIZED_ONLY = "realized_only"
    EQUITY_BASED = "equity_based"


class ModelAnalyticsScope(StrEnum):
    """
    Scope for model analytics aggregation.
    """

    MODEL = "model"
    STRATEGY = "strategy"
    SIGNAL_TYPE = "signal_type"
    SYMBOL = "symbol"
    TIMEFRAME = "timeframe"
    MARKET_REGIME = "market_regime"
    CONFIDENCE_BUCKET = "confidence_bucket"


class PredictionOutcome(StrEnum):
    """
    Normalized outcome label for model predictions.
    """

    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    TRUE_NEGATIVE = "true_negative"
    FALSE_NEGATIVE = "false_negative"
    PROFITABLE = "profitable"
    LOSING = "losing"
    BREAKEVEN = "breakeven"
    MISSED_OPPORTUNITY = "missed_opportunity"
    UNKNOWN = "unknown"


class ConfidenceBucket(StrEnum):
    """
    Default confidence buckets for calibration analysis.
    """

    VERY_LOW = "0.00_0.20"
    LOW = "0.20_0.40"
    MEDIUM = "0.40_0.60"
    HIGH = "0.60_0.80"
    VERY_HIGH = "0.80_1.00"


class MarketRegime(StrEnum):
    """
    Common market regimes used by model analytics.

    Analytics modules may produce more specific regime labels,
    but these are the default normalized buckets.
    """

    UNKNOWN = "unknown"
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    LIQUIDATION_CASCADE = "liquidation_cascade"
    FUNDING_EXTREME = "funding_extreme"
    OPEN_INTEREST_EXPANSION = "open_interest_expansion"
    OPEN_INTEREST_COMPRESSION = "open_interest_compression"


class OptimizationMode(StrEnum):
    """
    Parameter optimization mode.
    """

    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    WALK_FORWARD = "walk_forward"


class OptimizationObjective(StrEnum):
    """
    Objective used by optimizer.py.
    """

    NET_PNL = "net_pnl"
    TOTAL_RETURN = "total_return"
    SHARPE = "sharpe"
    SORTINO = "sortino"
    CALMAR = "calmar"
    PROFIT_FACTOR = "profit_factor"
    MAX_DRAWDOWN = "max_drawdown"
    EXPECTANCY = "expectancy"
    WIN_RATE = "win_rate"
    MODEL_ACCURACY = "model_accuracy"
    CUSTOM = "custom"


class WalkForwardWindowMode(StrEnum):
    """
    Walk-forward window behavior.
    """

    ROLLING = "rolling"
    EXPANDING = "expanding"
    ANCHORED = "anchored"


class DataQualityStatus(StrEnum):
    """
    Data quality status for downloaded or loaded history.
    """

    VALID = "valid"
    PARTIAL = "partial"
    EMPTY = "empty"
    HAS_GAPS = "has_gaps"
    HAS_DUPLICATES = "has_duplicates"
    OUT_OF_ORDER = "out_of_order"
    INVALID_SCHEMA = "invalid_schema"


class GapHandlingPolicy(StrEnum):
    """
    Policy used when historical data has missing intervals.
    """

    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"
    FORWARD_FILL = "forward_fill"
    BACK_FILL = "back_fill"


class DuplicateHandlingPolicy(StrEnum):
    """
    Policy used when historical data contains duplicate records.
    """

    FAIL = "fail"
    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"
    MERGE = "merge"


class StorageFormat(StrEnum):
    """
    Supported storage formats for historical data and results.
    """

    PARQUET = "parquet"
    CSV = "csv"
    JSON = "json"
    POSTGRES = "postgres"


class BacktestArtifactType(StrEnum):
    """
    Types of artifacts produced by a backtest run.
    """

    CONFIG_SNAPSHOT = "config_snapshot"
    TRADES = "trades"
    ORDERS = "orders"
    FILLS = "fills"
    POSITIONS = "positions"
    SIGNALS = "signals"
    RISK_BLOCKS = "risk_blocks"
    EQUITY_CURVE = "equity_curve"
    DRAWDOWN_CURVE = "drawdown_curve"
    PERFORMANCE_METRICS = "performance_metrics"
    MODEL_ANALYTICS = "model_analytics"
    REPORT = "report"