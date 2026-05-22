from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Literal

from backtesting.exceptions import BacktestConfigurationError, BacktestSafetyError


@dataclass(slots=True)
class BacktestConfig:
    """
    Configuration for a production-pipeline paper backtest.

    Defaults are Binance USD-M Futures and safe paper-only execution.
    """

    exchange: str = "binance"
    market_type: str = "usdm_futures"
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "RIVERUSDT")
    timeframes: tuple[str, ...] = ("1m", "15m")
    lookback_days: int = 7

    initial_balance_usd: Decimal = Decimal("1000")
    execution_mode: Literal["paper"] = "paper"

    enable_funding: bool = True
    enable_open_interest: bool = True
    enable_orderflow: bool = True
    enable_agg_trades: bool = False
    enable_mark_price: bool = True
    enable_liquidations: bool = False
    enable_news: bool = False

    # Important for deterministic backtesting:
    # production scheduler jobs such as cleanup/health/heartbeat/parquet flush
    # should not run on wall-clock time during historical replay.
    disable_scheduler_loop: bool = True

    report_format: Literal["html", "markdown"] = "html"
    fill_model: Literal["next_candle_open", "conservative_intrabar"] = "next_candle_open"

    maker_fee_bps: Decimal = Decimal("2.0")
    taker_fee_bps: Decimal = Decimal("4.0")
    slippage_bps: Decimal = Decimal("1.0")

    deterministic: bool = True
    replay_speed: Literal["fast", "realistic"] = "fast"
    accelerated_delay_multiplier: float = 0.0
    replay_drain_every_events: int = 5000
    replay_batch_drain_timeout_seconds: float = 120.0
    replay_final_drain_timeout_seconds: float = 600.0
    low_queue_size_threshold: int = 100

    historical_cache_dir: Path = Path("data/historical/binance/usdm_futures")
    reports_dir: Path = Path("reports/backtests")

    backtest_id: str | None = None
    backtest_mode_env_guard: str = "BACKTEST_MODE"

    require_event_bus_join: bool = False
    allow_private_event_bus_queue_join_fallback: bool = True

    event_bus_max_queue_size: int = 250_000
    event_bus_worker_count: int = 12
    event_bus_overflow_policy: str = "block"

    request_retries: int = 5
    request_timeout_seconds: float = 20.0
    request_retry_delay_seconds: float = 2.0
    request_delay_seconds: float = 0.25
    binance_public_base_url: str = "https://fapi.binance.com"

    max_klines_per_request: int = 1500
    max_agg_trades_per_request: int = 1000
    max_open_interest_points_per_request: int = 500
    max_funding_points_per_request: int = 1000

    disabled_strategy_domains: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        self.exchange = self.exchange.lower().strip()
        self.market_type = self.market_type.strip()

        if self.exchange != "binance":
            raise BacktestConfigurationError(
                "Only Binance historical loading is implemented in this module."
            )

        if self.market_type != "usdm_futures":
            raise BacktestConfigurationError(
                "Backtest is futures-only and supports market_type='usdm_futures'."
            )

        if self.execution_mode != "paper":
            raise BacktestSafetyError("Backtest can only run with execution_mode='paper'.")

        if self.lookback_days <= 0:
            raise BacktestConfigurationError("lookback_days must be positive.")

        if not self.symbols:
            raise BacktestConfigurationError("At least one symbol is required.")

        if not self.timeframes:
            raise BacktestConfigurationError("At least one timeframe is required.")

        allowed_timeframes = {"1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"}
        for timeframe in self.timeframes:
            if timeframe not in allowed_timeframes:
                raise BacktestConfigurationError(f"Unsupported timeframe: {timeframe}")

        if self.initial_balance_usd <= 0:
            raise BacktestConfigurationError("initial_balance_usd must be positive.")

        if self.replay_drain_every_events <= 0:
            raise BacktestConfigurationError("replay_drain_every_events must be > 0.")

        if self.replay_batch_drain_timeout_seconds <= 0:
            raise BacktestConfigurationError("replay_batch_drain_timeout_seconds must be > 0.")

        if self.replay_final_drain_timeout_seconds <= 0:
            raise BacktestConfigurationError("replay_final_drain_timeout_seconds must be > 0.")

        if self.low_queue_size_threshold < 0:
            raise BacktestConfigurationError("low_queue_size_threshold must be >= 0.")

        if self.event_bus_max_queue_size <= 0:
            raise BacktestConfigurationError("event_bus_max_queue_size must be > 0.")

        if self.event_bus_worker_count <= 0:
            raise BacktestConfigurationError("event_bus_worker_count must be > 0.")

        if self.request_retries < 1:
            raise BacktestConfigurationError("request_retries must be >= 1.")

        if self.request_timeout_seconds <= 0:
            raise BacktestConfigurationError("request_timeout_seconds must be > 0.")

        if self.request_retry_delay_seconds <= 0:
            raise BacktestConfigurationError("request_retry_delay_seconds must be > 0.")

        if self.request_delay_seconds < 0:
            raise BacktestConfigurationError("request_delay_seconds must be >= 0.")

        for name in ("maker_fee_bps", "taker_fee_bps", "slippage_bps"):
            if getattr(self, name) < 0:
                raise BacktestConfigurationError(f"{name} must be non-negative.")

        if self.replay_speed == "realistic" and self.accelerated_delay_multiplier < 0:
            raise BacktestConfigurationError("accelerated_delay_multiplier must be >= 0.")
