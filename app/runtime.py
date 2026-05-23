from __future__ import annotations

import asyncio
import inspect
import os
import signal
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from core.config import Config
from core.event_bus import EventBus, QueueFullPolicy
from core.scheduler import Scheduler


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_list(name: str, default: Iterable[str]) -> list[str]:
    value = os.getenv(name)
    if value is None:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def call_if_exists(component: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(component, method_name, None)
    if not callable(method):
        return None
    return await maybe_await(method(*args, **kwargs))


async def register_component(component: Any) -> None:
    await call_if_exists(component, "register")


async def start_component(component: Any) -> None:
    await call_if_exists(component, "start")


async def stop_component(component: Any) -> None:
    await call_if_exists(component, "stop")


def build_event_bus(config: Config) -> EventBus:
    policy_value = str(config.event_bus.queue_full_policy).lower()
    try:
        policy = QueueFullPolicy(policy_value)
    except Exception:
        policy = QueueFullPolicy.DROP_OLDEST

    return EventBus(
        max_queue_size=config.event_bus.max_queue_size,
        worker_count=config.event_bus.worker_count,
        queue_full_policy=policy,
        max_retries=config.event_bus.max_retries,
        retry_delay=config.event_bus.retry_delay,
        enable_metrics=config.event_bus.enable_metrics,
    )


def build_scheduler(config: Config, event_bus: EventBus) -> Scheduler:
    return Scheduler(
        event_bus=event_bus,
        tick_interval=config.scheduler.tick_interval,
    )


@dataclass(slots=True)
class RuntimeSettings:
    """
    Runtime settings are the single source of truth for app-level bootstrap.

    All exchange names, symbol filters, timeframes, stream names/channels,
    market types, warmup settings and polling parameters are read from .env
    through from_env(). Values below are safe fallbacks only.
    """

    market_data_exchanges: list[str] = field(default_factory=lambda: ["binance", "bybit", "okx", "mexc"])
    discover_all_symbols: bool = True
    symbol_allowlist: list[str] = field(default_factory=list)
    symbol_blocklist: list[str] = field(default_factory=list)
    quote_asset: str = "USDT"
    timeframes: list[str] = field(default_factory=lambda: ["1m", "15m"])
    execution_exchange: str = "binance"
    execution_market_type: str = "usdm_futures"
    execution_mode: str = "paper"
    live_trading_enabled: bool = False
    enable_market_data: bool = True
    enable_analytics: bool = True
    enable_strategy: bool = True
    enable_risk: bool = True
    enable_execution: bool = True
    enable_telegram: bool = True
    enable_news: bool = True

    # Exchange discovery / instrument filters.
    binance_contract_type: str = "PERPETUAL"
    binance_required_status: str = "TRADING"
    bybit_category: str = "linear"
    bybit_limit: int = 1000
    bybit_contract_types: list[str] = field(default_factory=lambda: ["linearperpetual", "perpetual", "swap"])
    okx_inst_type: str = "SWAP"
    okx_symbol_suffix: str = "SWAP"
    mexc_inactive_states: list[str] = field(default_factory=lambda: ["offline", "disabled", "disable", "closed", "delisted", "suspended", "suspend"])

    # WebSocket stream config.
    binance_public_ws_url: str = "wss://fstream.binance.com/stream"
    binance_private_ws_base_url: str = "wss://fstream.binance.com/ws"
    binance_rest_url: str = "https://fapi.binance.com"
    binance_ws_streams: list[str] = field(default_factory=lambda: ["trade", "depth", "kline", "forceorder"])
    binance_liquidation_stream_name: str = "forceorder"
    binance_ws_depth_level: str = "20"
    binance_ws_kline_interval: str = "1m"
    binance_orderbook_emit_min_interval_ms: int = 500
    binance_orderbook_batch_max_size: int = 2000
    binance_trade_emit_min_interval_ms: int = 500
    binance_trade_batch_max_size: int = 2000
    binance_enable_private_stream: bool = False

    bybit_ws_streams: list[str] = field(default_factory=lambda: ["trade", "orderbook", "kline", "liquidation"])
    bybit_liquidation_stream_name: str = "liquidation"
    bybit_orderbook_depth: int = 50
    bybit_kline_interval: str = "1"
    bybit_orderbook_emit_min_interval_ms: int = 100
    bybit_orderbook_batch_max_size: int = 500
    bybit_trade_emit_min_interval_ms: int = 250
    bybit_trade_batch_max_size: int = 1000
    bybit_enable_private_stream: bool = False

    okx_ws_streams: list[str] = field(default_factory=lambda: ["trades", "books", "candle"])
    okx_orderbook_channel: str = "books5"
    okx_candle_channel: str = "candle1m"
    okx_orderbook_emit_min_interval_ms: int = 100
    okx_orderbook_batch_max_size: int = 500
    okx_trade_emit_min_interval_ms: int = 250
    okx_trade_batch_max_size: int = 1000
    okx_enable_private_stream: bool = False

    mexc_ws_streams: list[str] = field(default_factory=lambda: ["deal", "depth", "kline"])
    mexc_kline_interval: str = "Min1"
    mexc_orderbook_emit_min_interval_ms: int = 100
    mexc_orderbook_batch_max_size: int = 500
    mexc_trade_emit_min_interval_ms: int = 250
    mexc_trade_batch_max_size: int = 1000
    mexc_enable_private_stream: bool = False

    ws_shard_size_binance: int = 80
    ws_shard_size_bybit: int = 50
    ws_shard_size_okx: int = 80
    ws_shard_size_mexc: int = 50

    # Analytics config.
    analytics_exchange: str = "binance"
    analytics_market_type: str = "usdm_futures"
    analytics_symbols: list[str] = field(default_factory=list)
    orderflow_default_exchange: str = "binance"
    orderflow_default_market_type: str = "usdm_futures"
    orderflow_default_timeframe: str = "1m"
    price_action_exchange: str = "binance"
    price_action_market_type: str = "usdm_futures"
    price_action_symbols: list[str] = field(default_factory=list)
    price_action_timeframes: list[str] = field(default_factory=lambda: ["1m", "15m"])
    liquidity_candles_updated_topics: list[str] = field(default_factory=lambda: ["market.candles.updated"])
    liquidity_min_candles_for_snapshot: int = 5

    strategy_preset_name: str = "scalping"
    strategy_use_required_features: bool = False
    strategy_registry_strict: bool = False

    # Startup warmup gate. Trading components are not started until historical
    # candle/funding data has been fetched and propagated through caches + analytics.
    startup_warmup_enabled: bool = True
    startup_warmup_required: bool = True
    startup_warmup_exchange: str = "binance"
    startup_warmup_timeframes: list[str] = field(default_factory=lambda: ["1m", "15m"])
    startup_warmup_kline_limit: int = 500
    startup_warmup_funding_limit: int = 24
    startup_warmup_concurrency: int = 8
    startup_warmup_batch_size: int = 50
    startup_warmup_eventbus_idle_timeout: float = 30.0
    startup_warmup_settle_seconds: float = 0.25
    startup_warmup_persist_enabled: bool = True
    startup_warmup_persist_required: bool = True
    startup_warmup_flush_storage_before_trading: bool = True

    # Live derivative REST snapshot polling.
    derivative_snapshot_exchange: str = "binance"
    derivative_snapshot_poll_interval_seconds: float = 60.0
    derivative_snapshot_poll_concurrency: int = 8
    derivative_snapshot_poll_batch_size: int = 50
    derivative_snapshot_min_interval_seconds: float = 10.0
    derivative_snapshot_funding_limit: int = 1

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        market_timeframes = env_list("MARKET_DATA_TIMEFRAMES", ["1m", "15m"])
        analytics_symbols = [s.upper() for s in env_list("ANALYTICS_SYMBOLS", [])]
        return cls(
            market_data_exchanges=[e.lower() for e in env_list("MARKET_DATA_EXCHANGES", ["binance", "bybit", "okx", "mexc"])],
            discover_all_symbols=env_bool("MARKET_DATA_DISCOVER_ALL_SYMBOLS", True),
            symbol_allowlist=[s.upper() for s in env_list("MARKET_DATA_SYMBOL_ALLOWLIST", [])],
            symbol_blocklist=[s.upper() for s in env_list("MARKET_DATA_SYMBOL_BLOCKLIST", [])],
            quote_asset=env_str("MARKET_DATA_QUOTE_ASSET", "USDT").upper(),
            timeframes=market_timeframes,
            execution_exchange=env_str("EXECUTION_EXCHANGE", "binance").lower(),
            execution_market_type=env_str("EXECUTION_MARKET_TYPE", "usdm_futures"),
            execution_mode=env_str("EXECUTION_MODE", "paper").lower(),
            live_trading_enabled=env_bool("EXECUTION_LIVE_TRADING_ENABLED", False),
            enable_market_data=env_bool("MARKET_DATA_ENABLED", True),
            enable_analytics=env_bool("ANALYTICS_ENABLED", True),
            enable_strategy=env_bool("STRATEGY_ENABLED", True),
            enable_risk=env_bool("RISK_ENABLED", True),
            enable_execution=env_bool("EXECUTION_ENABLED", True),
            enable_telegram=env_bool("TELEGRAM_BOT_ENABLED", True),
            enable_news=env_bool("NEWS_AI_ENABLED", True),

            binance_contract_type=env_str("BINANCE_CONTRACT_TYPE", "PERPETUAL").upper(),
            binance_required_status=env_str("BINANCE_REQUIRED_STATUS", "TRADING").upper(),
            bybit_category=env_str("BYBIT_CATEGORY", "linear"),
            bybit_limit=max(1, env_int("BYBIT_DISCOVERY_LIMIT", 1000)),
            bybit_contract_types=[v.lower() for v in env_list("BYBIT_CONTRACT_TYPES", ["linearperpetual", "perpetual", "swap"])],
            okx_inst_type=env_str("OKX_INST_TYPE", "SWAP").upper(),
            okx_symbol_suffix=env_str("OKX_SYMBOL_SUFFIX", "SWAP").upper(),
            mexc_inactive_states=[v.lower() for v in env_list("MEXC_INACTIVE_STATES", ["offline", "disabled", "disable", "closed", "delisted", "suspended", "suspend"])],

            binance_public_ws_url=env_str("BINANCE_PUBLIC_WS_URL", "wss://fstream.binance.com/stream"),
            binance_private_ws_base_url=env_str("BINANCE_PRIVATE_WS_BASE_URL", "wss://fstream.binance.com/ws"),
            binance_rest_url=env_str("BINANCE_REST_URL", "https://fapi.binance.com"),
            binance_ws_streams=env_list("BINANCE_WS_STREAMS", ["trade", "depth", "kline", "forceorder"]),
            binance_liquidation_stream_name=env_str("BINANCE_LIQUIDATION_STREAM_NAME", "forceorder"),
            binance_ws_depth_level=env_str("BINANCE_WS_DEPTH_LEVEL", "20"),
            binance_ws_kline_interval=env_str("BINANCE_WS_KLINE_INTERVAL", env_str("MARKET_DATA_PRIMARY_TIMEFRAME", "1m")),
            binance_orderbook_emit_min_interval_ms=max(0, env_int("BINANCE_ORDERBOOK_EMIT_MIN_INTERVAL_MS", 500)),
            binance_orderbook_batch_max_size=max(1, env_int("BINANCE_ORDERBOOK_BATCH_MAX_SIZE", 2000)),
            binance_trade_emit_min_interval_ms=max(0, env_int("BINANCE_TRADE_EMIT_MIN_INTERVAL_MS", 500)),
            binance_trade_batch_max_size=max(1, env_int("BINANCE_TRADE_BATCH_MAX_SIZE", 2000)),
            binance_enable_private_stream=env_bool("BINANCE_ENABLE_PRIVATE_STREAM", False),

            bybit_ws_streams=env_list("BYBIT_WS_STREAMS", ["trade", "orderbook", "kline", "liquidation"]),
            bybit_liquidation_stream_name=env_str("BYBIT_LIQUIDATION_STREAM_NAME", "liquidation"),
            bybit_orderbook_depth=max(1, env_int("BYBIT_ORDERBOOK_DEPTH", 50)),
            bybit_kline_interval=env_str("BYBIT_KLINE_INTERVAL", "1"),
            bybit_orderbook_emit_min_interval_ms=max(0, env_int("BYBIT_ORDERBOOK_EMIT_MIN_INTERVAL_MS", 100)),
            bybit_orderbook_batch_max_size=max(1, env_int("BYBIT_ORDERBOOK_BATCH_MAX_SIZE", 500)),
            bybit_trade_emit_min_interval_ms=max(0, env_int("BYBIT_TRADE_EMIT_MIN_INTERVAL_MS", 250)),
            bybit_trade_batch_max_size=max(1, env_int("BYBIT_TRADE_BATCH_MAX_SIZE", 1000)),
            bybit_enable_private_stream=env_bool("BYBIT_ENABLE_PRIVATE_STREAM", False),

            okx_ws_streams=env_list("OKX_WS_STREAMS", ["trades", "books", "candle"]),
            okx_orderbook_channel=env_str("OKX_ORDERBOOK_CHANNEL", "books5"),
            okx_candle_channel=env_str("OKX_CANDLE_CHANNEL", "candle1m"),
            okx_orderbook_emit_min_interval_ms=max(0, env_int("OKX_ORDERBOOK_EMIT_MIN_INTERVAL_MS", 100)),
            okx_orderbook_batch_max_size=max(1, env_int("OKX_ORDERBOOK_BATCH_MAX_SIZE", 500)),
            okx_trade_emit_min_interval_ms=max(0, env_int("OKX_TRADE_EMIT_MIN_INTERVAL_MS", 250)),
            okx_trade_batch_max_size=max(1, env_int("OKX_TRADE_BATCH_MAX_SIZE", 1000)),
            okx_enable_private_stream=env_bool("OKX_ENABLE_PRIVATE_STREAM", False),

            mexc_ws_streams=env_list("MEXC_WS_STREAMS", ["deal", "depth", "kline"]),
            mexc_kline_interval=env_str("MEXC_KLINE_INTERVAL", "Min1"),
            mexc_orderbook_emit_min_interval_ms=max(0, env_int("MEXC_ORDERBOOK_EMIT_MIN_INTERVAL_MS", 100)),
            mexc_orderbook_batch_max_size=max(1, env_int("MEXC_ORDERBOOK_BATCH_MAX_SIZE", 500)),
            mexc_trade_emit_min_interval_ms=max(0, env_int("MEXC_TRADE_EMIT_MIN_INTERVAL_MS", 250)),
            mexc_trade_batch_max_size=max(1, env_int("MEXC_TRADE_BATCH_MAX_SIZE", 1000)),
            mexc_enable_private_stream=env_bool("MEXC_ENABLE_PRIVATE_STREAM", False),

            ws_shard_size_binance=max(1, env_int("BINANCE_WS_SHARD_SIZE", 80)),
            ws_shard_size_bybit=max(1, env_int("BYBIT_WS_SHARD_SIZE", 50)),
            ws_shard_size_okx=max(1, env_int("OKX_WS_SHARD_SIZE", 80)),
            ws_shard_size_mexc=max(1, env_int("MEXC_WS_SHARD_SIZE", 50)),

            analytics_exchange=env_str("ANALYTICS_EXCHANGE", "binance").lower(),
            analytics_market_type=env_str("ANALYTICS_MARKET_TYPE", "usdm_futures"),
            analytics_symbols=analytics_symbols,
            orderflow_default_exchange=env_str("ORDERFLOW_DEFAULT_EXCHANGE", env_str("ANALYTICS_EXCHANGE", "binance")).lower(),
            orderflow_default_market_type=env_str("ORDERFLOW_DEFAULT_MARKET_TYPE", env_str("ANALYTICS_MARKET_TYPE", "usdm_futures")),
            orderflow_default_timeframe=env_str("ORDERFLOW_DEFAULT_TIMEFRAME", env_str("MARKET_DATA_PRIMARY_TIMEFRAME", market_timeframes[0] if market_timeframes else "1m")),
            price_action_exchange=env_str("PRICE_ACTION_EXCHANGE", env_str("ANALYTICS_EXCHANGE", "binance")).lower(),
            price_action_market_type=env_str("PRICE_ACTION_MARKET_TYPE", env_str("ANALYTICS_MARKET_TYPE", "usdm_futures")),
            price_action_symbols=[s.upper() for s in env_list("PRICE_ACTION_SYMBOLS", analytics_symbols)],
            price_action_timeframes=env_list("PRICE_ACTION_TIMEFRAMES", market_timeframes),
            liquidity_candles_updated_topics=env_list("LIQUIDITY_CANDLES_UPDATED_TOPICS", ["market.candles.updated"]),
            liquidity_min_candles_for_snapshot=max(1, env_int("LIQUIDITY_MIN_CANDLES_FOR_SNAPSHOT", 5)),

            strategy_preset_name=env_str("STRATEGY_PRESET_NAME", "scalping"),
            strategy_use_required_features=env_bool("STRATEGY_USE_REQUIRED_FEATURES", False),
            strategy_registry_strict=env_bool("STRATEGY_REGISTRY_STRICT", False),

            startup_warmup_enabled=env_bool("STARTUP_WARMUP_ENABLED", True),
            startup_warmup_required=env_bool("STARTUP_WARMUP_REQUIRED", True),
            startup_warmup_exchange=env_str("STARTUP_WARMUP_EXCHANGE", "binance").lower(),
            startup_warmup_timeframes=env_list("STARTUP_WARMUP_TIMEFRAMES", market_timeframes),
            startup_warmup_kline_limit=max(1, env_int("STARTUP_WARMUP_KLINE_LIMIT", 500)),
            startup_warmup_funding_limit=max(1, env_int("STARTUP_WARMUP_FUNDING_LIMIT", 24)),
            startup_warmup_concurrency=max(1, env_int("STARTUP_WARMUP_CONCURRENCY", 8)),
            startup_warmup_batch_size=max(1, env_int("STARTUP_WARMUP_BATCH_SIZE", 50)),
            startup_warmup_eventbus_idle_timeout=env_float("STARTUP_WARMUP_EVENTBUS_IDLE_TIMEOUT", 30.0),
            startup_warmup_settle_seconds=env_float("STARTUP_WARMUP_SETTLE_SECONDS", 0.25),
            startup_warmup_persist_enabled=env_bool("STARTUP_WARMUP_PERSIST_ENABLED", True),
            startup_warmup_persist_required=env_bool("STARTUP_WARMUP_PERSIST_REQUIRED", True),
            startup_warmup_flush_storage_before_trading=env_bool("STARTUP_WARMUP_FLUSH_STORAGE_BEFORE_TRADING", True),

            derivative_snapshot_exchange=env_str("DERIVATIVE_SNAPSHOT_EXCHANGE", env_str("STARTUP_WARMUP_EXCHANGE", "binance")).lower(),
            derivative_snapshot_poll_interval_seconds=env_float("DERIVATIVE_SNAPSHOT_POLL_INTERVAL_SECONDS", 60.0),
            derivative_snapshot_poll_concurrency=max(1, env_int("DERIVATIVE_SNAPSHOT_POLL_CONCURRENCY", 8)),
            derivative_snapshot_poll_batch_size=max(1, env_int("DERIVATIVE_SNAPSHOT_POLL_BATCH_SIZE", 50)),
            derivative_snapshot_min_interval_seconds=max(1.0, env_float("DERIVATIVE_SNAPSHOT_MIN_INTERVAL_SECONDS", 10.0)),
            derivative_snapshot_funding_limit=max(1, env_int("DERIVATIVE_SNAPSHOT_FUNDING_LIMIT", 1)),
        )

    def validate(self) -> None:
        supported = {"binance", "bybit", "okx", "mexc"}
        unknown = set(self.market_data_exchanges) - supported
        if unknown:
            raise ValueError(f"Unsupported market data exchanges: {sorted(unknown)}")

        if self.execution_exchange != "binance":
            raise ValueError("Execution exchange must be binance for Binance USD-M Futures execution")

        if self.execution_market_type != "usdm_futures":
            raise ValueError("Execution market type must be usdm_futures")

        if self.live_trading_enabled and self.execution_mode != "live":
            raise ValueError("EXECUTION_LIVE_TRADING_ENABLED=true requires EXECUTION_MODE=live")

        if self.execution_mode == "live" and not self.live_trading_enabled:
            raise ValueError("EXECUTION_MODE=live requires EXECUTION_LIVE_TRADING_ENABLED=true")

        for name in (self.startup_warmup_exchange, self.derivative_snapshot_exchange):
            if name not in supported:
                raise ValueError(f"Unsupported startup/derivative snapshot exchange: {name}")

        if self.startup_warmup_exchange != "binance":
            raise ValueError("Startup warmup REST implementation currently supports STARTUP_WARMUP_EXCHANGE=binance only")

        if self.derivative_snapshot_exchange != "binance":
            raise ValueError("Derivative snapshot polling currently supports DERIVATIVE_SNAPSHOT_EXCHANGE=binance only")

def install_signal_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass


def chunked(items: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]