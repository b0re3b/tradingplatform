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
    market_data_exchanges: list[str] = field(default_factory=lambda: ["binance", "bybit", "okx", "mexc"])
    discover_all_symbols: bool = True
    symbol_allowlist: list[str] = field(default_factory=list)
    symbol_blocklist: list[str] = field(default_factory=list)
    quote_asset: str = "USDT"
    timeframes: list[str] = field(default_factory=lambda: ["1m", "15m"])
    execution_exchange: str = "binance"
    execution_mode: str = "paper"
    live_trading_enabled: bool = False
    enable_market_data: bool = True
    enable_analytics: bool = True
    enable_strategy: bool = True
    enable_risk: bool = True
    enable_execution: bool = True
    enable_telegram: bool = True
    enable_news: bool = True
    ws_shard_size_binance: int = 80
    ws_shard_size_bybit: int = 50
    ws_shard_size_okx: int = 80
    ws_shard_size_mexc: int = 50

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        return cls(
            market_data_exchanges=[e.lower() for e in env_list("MARKET_DATA_EXCHANGES", ["binance", "bybit", "okx", "mexc"])],
            discover_all_symbols=env_bool("MARKET_DATA_DISCOVER_ALL_SYMBOLS", True),
            symbol_allowlist=[s.upper() for s in env_list("MARKET_DATA_SYMBOL_ALLOWLIST", [])],
            symbol_blocklist=[s.upper() for s in env_list("MARKET_DATA_SYMBOL_BLOCKLIST", [])],
            quote_asset=os.getenv("MARKET_DATA_QUOTE_ASSET", "USDT").upper(),
            timeframes=env_list("MARKET_DATA_TIMEFRAMES", ["1m", "15m"]),
            execution_exchange=os.getenv("EXECUTION_EXCHANGE", "binance").lower(),
            execution_mode=os.getenv("EXECUTION_MODE", "paper").lower(),
            live_trading_enabled=env_bool("EXECUTION_LIVE_TRADING_ENABLED", False),
            enable_market_data=env_bool("MARKET_DATA_ENABLED", True),
            enable_analytics=env_bool("ANALYTICS_ENABLED", True),
            enable_strategy=env_bool("STRATEGY_ENABLED", True),
            enable_risk=env_bool("RISK_ENABLED", True),
            enable_execution=env_bool("EXECUTION_ENABLED", True),
            enable_telegram=env_bool("TELEGRAM_BOT_ENABLED", True),
            enable_news=env_bool("NEWS_AI_ENABLED", True),
            ws_shard_size_binance=env_int("BINANCE_WS_SHARD_SIZE", 80),
            ws_shard_size_bybit=env_int("BYBIT_WS_SHARD_SIZE", 50),
            ws_shard_size_okx=env_int("OKX_WS_SHARD_SIZE", 80),
            ws_shard_size_mexc=env_int("MEXC_WS_SHARD_SIZE", 50),
        )

    def validate(self) -> None:
        supported = {"binance", "bybit", "okx", "mexc"}
        unknown = set(self.market_data_exchanges) - supported
        if unknown:
            raise ValueError(f"Unsupported market data exchanges: {sorted(unknown)}")
        if self.execution_exchange != "binance":
            raise ValueError("Live execution is currently allowed only on Binance USD-M Futures")
        if self.live_trading_enabled and self.execution_mode != "live":
            raise ValueError("EXECUTION_LIVE_TRADING_ENABLED=true requires EXECUTION_MODE=live")
        if self.execution_mode == "live" and not self.live_trading_enabled:
            raise ValueError("EXECUTION_MODE=live requires EXECUTION_LIVE_TRADING_ENABLED=true")


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