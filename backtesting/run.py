# backtesting/run_backtest_full.py

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import inspect
import os
import pkgutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from core.config import Config
from core.event_bus import EventBus
from core.logger import get_logger
from core.scheduler import Scheduler

from backtesting.config import BacktestConfig
from backtesting.exceptions import DataLoadError
from backtesting.data_loader import DataLoader
from backtesting.enums import (
    BacktestDataType,
    BacktestMode,
    DataGapPolicy,
    DataValidationLevel,
    FillModel,
    HistoricalDataFormat,
    LiquidityModel,
    PnLAccountingMode,
    PositionAccountingMode,
    ReplayMode,
    ReportFormat,
    ReportSection,
    SlippageModel,
)
from backtesting.strategy_tester import StrategyTester

from data.candles_cache import CandlesCache
from data.funding_cache import FundingCache
from data.open_interest_cache import OpenInterestCache
from data.orderbook_cache import OrderBookCache
from data.trades_cache import TradesCache

from risk.config import RiskConfig
from risk.risk_manager import RiskManager

from strategy.base import BaseStrategy
from strategy.config import StrategyConfig
from strategy.state import StrategyRuntimeState
from strategy.engine import StrategyEngine
from strategy.presets import build_default_strategy_config, build_default_strategy_registry
from strategy.processor import SignalProcessor


logger = get_logger("backtesting.run_backtest_full")


# =============================================================================
# DataLoader compatibility
# =============================================================================


class SortingDataLoader(DataLoader):
    """
    DataLoader compatibility layer for multi-symbol datasets.

    Current DataLoader.load_bundle() validates record ordering before calling
    _sort_bundle(). With multi-symbol files, records are loaded per file/symbol
    and are often grouped as BTC -> DOGE -> SOL rather than globally sorted by
    timestamp. This subclass sorts before validation, then lets the original
    DataLoader flow continue.
    """

    def validate_bundle(self, bundle: Any, *, period: Any | None = None) -> None:
        self._sort_bundle(bundle)
        super().validate_bundle(bundle, period=period)


# =============================================================================
# Paths
# =============================================================================


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _candidate_history_dirs() -> list[Path]:
    """
    Candidate historical data roots.

    The downloader may be executed from project root or from the backtesting
    folder. This runner resolves both layouts:
    - <project>/data/history
    - <project>/backtesting/data/history
    """

    candidates: list[Path] = []

    env_dir = os.getenv("BACKTEST_DATA_DIR")
    if env_dir and env_dir.strip():
        candidates.append(Path(env_dir).expanduser())

    candidates.extend(
        [
            Path("data/history"),
            PROJECT_ROOT / "data" / "history",
            SCRIPT_DIR / "data" / "history",
        ]
    )

    unique: list[Path] = []
    seen: set[str] = set()

    for path in candidates:
        resolved = path.resolve()
        key = str(resolved)
        if key in seen:
            continue
        unique.append(resolved)
        seen.add(key)

    return unique


def _history_dir_has_required_candles(
    root: Path,
    *,
    exchange: str,
    market_type: str,
    symbols: list[str],
    timeframes: list[str],
) -> bool:
    timeframe = timeframes[0]

    for symbol in symbols:
        path_without_suffix = (
            root
            / exchange
            / market_type
            / "candles"
            / symbol
            / timeframe
            / f"{symbol}_{timeframe}"
        )

        if not _file_exists_any(path_without_suffix):
            return False

    return True


def resolve_history_dir(
    *,
    exchange: str,
    market_type: str,
    symbols: list[str],
    timeframes: list[str],
) -> Path:
    """
    Resolve the first data directory that contains required candles for every
    requested symbol.
    """

    candidates = _candidate_history_dirs()

    for candidate in candidates:
        if _history_dir_has_required_candles(
            candidate,
            exchange=exchange,
            market_type=market_type,
            symbols=symbols,
            timeframes=timeframes,
        ):
            return candidate

    checked = "\n".join(f"- {item}" for item in candidates)
    raise FileNotFoundError(
        "Could not find historical candles for all requested symbols.\n"
        f"symbols={symbols}\n"
        f"timeframes={timeframes}\n"
        f"Checked data dirs:\n{checked}\n"
        "Expected layout:\n"
        "<data_dir>/binance/usdm_futures/candles/<SYMBOL>/1m/<SYMBOL>_1m.csv"
    )


# =============================================================================
# Constants
# =============================================================================


DEFAULT_RUN_NAME = "btc_doge_sol_last_2d_full_pipeline"
DEFAULT_EXCHANGE = "binance"
DEFAULT_MARKET_TYPE = "usdm_futures"
DEFAULT_SYMBOLS = ["BTCUSDT", "DOGEUSDT", "SOLUSDT"]
DEFAULT_TIMEFRAMES = ["1m"]

DEFAULT_BACKTEST_DAYS = 2


def _rolling_end_time() -> datetime:
    return datetime.now(timezone.utc).replace(second=0, microsecond=0)


def _rolling_start_time(days: int = DEFAULT_BACKTEST_DAYS) -> datetime:
    return _rolling_end_time() - timedelta(days=days)


# =============================================================================
# Runtime config
# =============================================================================


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return list(default)
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def _env_path(name: str, default: str | Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


@dataclass(slots=True)
class FullBacktestRunConfig:
    """
    Runtime settings for the final full-pipeline backtest runner.

    This script assumes you already downloaded historical data with
    backtesting.history_downloader.HistoryDownloader.

    Default expected data:
        data/history/binance/usdm_futures/candles/BTCUSDT/1m/BTCUSDT_1m.csv
        data/history/binance/usdm_futures/candles/DOGEUSDT/1m/DOGEUSDT_1m.csv
        data/history/binance/usdm_futures/candles/SOLUSDT/1m/SOLUSDT_1m.csv
        data/history/binance/usdm_futures/funding/<SYMBOL>/<SYMBOL>.csv
        data/history/binance/usdm_futures/open_interest/<SYMBOL>/5m/<SYMBOL>_5m.csv

    Optional:
        data/history/binance/usdm_futures/open_interest/BTCUSDT/5m/BTCUSDT_5m.csv
        data/history/binance/usdm_futures/trades/BTCUSDT/BTCUSDT.csv
    """

    run_name: str = DEFAULT_RUN_NAME
    exchange: str = DEFAULT_EXCHANGE
    market_type: str = DEFAULT_MARKET_TYPE
    symbols: list[str] = field(default_factory=lambda: list(DEFAULT_SYMBOLS))
    timeframes: list[str] = field(default_factory=lambda: list(DEFAULT_TIMEFRAMES))

    start_time: datetime = field(default_factory=lambda: _rolling_start_time(DEFAULT_BACKTEST_DAYS))
    end_time: datetime = field(default_factory=_rolling_end_time)
    warmup_start_time: datetime | None = None

    data_dir: Path = Path("data/history")
    output_dir: Path = Path("reports/backtests")

    input_format: HistoricalDataFormat = HistoricalDataFormat.CSV
    initial_balance: float = 10_000.0
    backtest_days: int = DEFAULT_BACKTEST_DAYS

    include_candles: bool = True
    include_funding: bool = True
    include_open_interest: bool = True
    include_trades: bool = False
    include_orderbook: bool = False
    include_liquidations: bool = False

    require_analytics: bool = True
    stop_on_first_error: bool = True
    cleanup_after_run: bool = True
    fail_if_live_execution_detected: bool = True

    use_all_registered_strategies: bool = True
    strategy_preset: str | None = None
    strategies: list[str] = field(default_factory=list)

    analytics_specs: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "FullBacktestRunConfig":
        input_format_raw = os.getenv("BACKTEST_INPUT_FORMAT", "csv").strip().lower()
        try:
            input_format = HistoricalDataFormat(input_format_raw)
        except ValueError:
            raise ValueError(
                "BACKTEST_INPUT_FORMAT must be one of: "
                f"{', '.join(item.value for item in HistoricalDataFormat)}"
            ) from None

        analytics_specs_raw = os.getenv("ANALYTICS_COMPONENTS", "").strip()
        analytics_specs = [
            item.strip()
            for item in analytics_specs_raw.split(",")
            if item.strip()
        ]

        strategies_raw = os.getenv("BACKTEST_STRATEGIES", "").strip()
        strategies = [
            item.strip()
            for item in strategies_raw.split(",")
            if item.strip()
        ]

        backtest_days_raw = os.getenv("BACKTEST_DAYS", str(DEFAULT_BACKTEST_DAYS)).strip()
        backtest_days = int(backtest_days_raw)

        end_time = _rolling_end_time()
        start_time = end_time - timedelta(days=backtest_days)

        symbols = _env_list("SYMBOLS", DEFAULT_SYMBOLS)
        timeframes = [item.lower() for item in _env_list("TIMEFRAMES", DEFAULT_TIMEFRAMES)]

        data_dir = resolve_history_dir(
            exchange=DEFAULT_EXCHANGE,
            market_type=DEFAULT_MARKET_TYPE,
            symbols=symbols,
            timeframes=timeframes,
        )

        return cls(
            run_name=os.getenv("BACKTEST_RUN_NAME", DEFAULT_RUN_NAME).strip() or DEFAULT_RUN_NAME,
            symbols=symbols,
            timeframes=timeframes,
            data_dir=data_dir,
            output_dir=_env_path("BACKTEST_OUTPUT_DIR", "reports/backtests"),
            input_format=input_format,
            backtest_days=backtest_days,
            start_time=start_time,
            end_time=end_time,
            include_candles=_env_bool("INCLUDE_CANDLES", True),
            include_funding=_env_bool("INCLUDE_FUNDING", True),
            include_open_interest=_env_bool("INCLUDE_OPEN_INTEREST", True),
            include_trades=_env_bool("INCLUDE_TRADES", False),
            include_orderbook=_env_bool("INCLUDE_ORDERBOOK", True),
            include_liquidations=_env_bool("INCLUDE_LIQUIDATIONS", True),
            require_analytics=_env_bool("REQUIRE_ANALYTICS", True),
            stop_on_first_error=_env_bool("STOP_ON_FIRST_ERROR", True),
            cleanup_after_run=_env_bool("CLEANUP_AFTER_RUN", True),
            fail_if_live_execution_detected=_env_bool("FAIL_IF_LIVE_EXECUTION_DETECTED", True),
            use_all_registered_strategies=_env_bool("TEST_ALL_REGISTERED_STRATEGIES", True),
            strategy_preset=os.getenv("STRATEGY_PRESET") or None,
            strategies=strategies,
            analytics_specs=analytics_specs,
        )


# =============================================================================
# Generic helpers
# =============================================================================


@dataclass(slots=True)
class InlineSubscription:
    pattern: str
    handler: Any
    name: str = "anonymous_handler"

    @property
    def topic(self) -> str:
        return self.pattern


class InlineBacktestEvent(dict):
    """
    Dict-compatible event object for inline backtesting.

    It supports both handler styles:
    - payload-style handlers: event.get("symbol")
    - core Event-style handlers: event.payload / event.topic

    The dict itself contains the payload fields plus a `topic` key.
    The `.payload` property returns the original payload dict.
    """

    def __init__(self, topic: str, payload: Any, **metadata: Any) -> None:
        payload_dict = dict(payload or {}) if isinstance(payload, dict) else {"payload": payload}
        super().__init__(payload_dict)
        self["topic"] = topic
        self._topic = topic
        self._payload = payload_dict
        self._metadata = metadata

    @property
    def topic(self) -> str:
        return self._topic

    @property
    def payload(self) -> dict[str, Any]:
        return self._payload

    @property
    def correlation_id(self) -> str | None:
        return self._metadata.get("correlation_id")

    @property
    def source(self) -> str | None:
        return self._metadata.get("source")

    @property
    def headers(self) -> dict[str, Any]:
        return dict(self._metadata.get("headers") or {})


class InlineBacktestEventBus:
    """
    Synchronous/inline EventBus for deterministic backtests.

    The production core.EventBus is queue/worker based. That is good for live
    runtime, but in backtesting we need every MarketReplay.emit() to complete
    the whole downstream chain before moving on to the next event:

        market.* -> data cache -> analytics.* -> strategy -> risk -> execution -> position

    This bus keeps the same basic API:
    - subscribe(pattern, handler, name=None)
    - unsubscribe(subscription)
    - emit(topic, payload, **kwargs)
    - publish(event_or_topic, payload=None)
    - start()/stop()

    It supports exact topics and simple suffix wildcards like "signal.*".
    """

    def __init__(self) -> None:
        self._subscriptions: list[InlineSubscription] = []
        self._running = False
        self._metrics: dict[str, Any] = {
            "published": 0,
            "processed": 0,
            "failed": 0,
            "subscriptions": 0,
            "topic_published": {},
            "topic_processed": {},
            "handler_errors": {},
        }

    async def start(self) -> None:
        self._running = True

    async def stop(self, *, drain: bool = True, timeout: float = 10.0) -> None:
        self._running = False

    def subscribe(
        self,
        pattern: str,
        handler: Any,
        *,
        name: str | None = None,
    ) -> InlineSubscription:
        subscription = InlineSubscription(
            pattern=pattern,
            handler=handler,
            name=name or getattr(handler, "__name__", "anonymous_handler"),
        )
        self._subscriptions.append(subscription)
        self._metrics["subscriptions"] = len(self._subscriptions)
        return subscription

    def unsubscribe(self, subscription: Any, handler: Any | None = None) -> None:
        if isinstance(subscription, InlineSubscription):
            self._subscriptions = [
                item for item in self._subscriptions if item is not subscription
            ]
        elif isinstance(subscription, str) and handler is not None:
            self._subscriptions = [
                item
                for item in self._subscriptions
                if not (item.pattern == subscription and item.handler == handler)
            ]
        self._metrics["subscriptions"] = len(self._subscriptions)

    def add_middleware(self, middleware: Any) -> None:
        # Compatibility no-op. Inline bus is intentionally minimal.
        return None

    def set_error_handler(self, handler: Any) -> None:
        self._error_handler = handler

    async def emit(
        self,
        topic: str,
        payload: Any,
        *,
        priority: Any = None,
        source: str | None = None,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        event = InlineBacktestEvent(
            topic,
            payload,
            source=source,
            correlation_id=correlation_id,
            headers=headers or {},
            priority=priority,
        )

        self._metrics["published"] += 1
        self._metrics["topic_published"][topic] = (
            self._metrics["topic_published"].get(topic, 0) + 1
        )

        handlers = self._matching_handlers(topic)

        for subscription in handlers:
            try:
                result = subscription.handler(event)
                if inspect.isawaitable(result):
                    await result

                self._metrics["processed"] += 1
                self._metrics["topic_processed"][topic] = (
                    self._metrics["topic_processed"].get(topic, 0) + 1
                )

            except Exception as exc:
                self._metrics["failed"] += 1
                key = f"{subscription.pattern}:{subscription.name}"
                self._metrics["handler_errors"][key] = (
                    self._metrics["handler_errors"].get(key, 0) + 1
                )
                raise

        return True

    async def publish(self, event_or_topic: Any, payload: Any | None = None, **kwargs: Any) -> bool:
        if isinstance(event_or_topic, str):
            return await self.emit(event_or_topic, payload, **kwargs)

        topic = getattr(event_or_topic, "topic", None)
        event_payload = getattr(event_or_topic, "payload", payload)

        if topic is None:
            raise TypeError("publish() expects topic string or object with .topic")

        return await self.emit(
            topic,
            event_payload,
            source=getattr(event_or_topic, "source", None),
            correlation_id=getattr(event_or_topic, "correlation_id", None),
            headers=getattr(event_or_topic, "headers", None) or {},
        )

    def _matching_handlers(self, topic: str) -> list[InlineSubscription]:
        result: list[InlineSubscription] = []

        for subscription in list(self._subscriptions):
            pattern = subscription.pattern

            if pattern == topic:
                result.append(subscription)
                continue

            if pattern.endswith(".*") and topic.startswith(pattern[:-1]):
                result.append(subscription)
                continue

            if pattern == "*":
                result.append(subscription)

        return result

    def stats(self) -> dict[str, Any]:
        return dict(self._metrics)


class BacktestSchedulerCompatAdapter:
    """
    Compatibility wrapper around core Scheduler for backtesting components.

    Supported component-side call styles:
    - add_interval_job(name=..., func=..., interval=...)
    - add_interval_job(name=..., callback=..., interval_seconds=...)
    - add_interval_job(name, func, interval)
    - add_interval_job(name, callback, interval_seconds)

    Current core Scheduler contract is:
        add_interval_job(name, func, *, interval: float, ...)

    The adapter forwards every other attribute/method to the wrapped scheduler.
    """

    def __init__(self, scheduler: Scheduler) -> None:
        self._scheduler = scheduler

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scheduler, name)

    def add_interval_job(self, *args: Any, **kwargs: Any) -> Any:
        """
        Normalize all scheduler call styles used across the project.

        Supported forms:
        - add_interval_job(name, func, interval)
        - add_interval_job(func, interval_seconds=..., name=...)
        - add_interval_job(name=..., func=..., interval=...)
        - add_interval_job(name=..., callback=..., interval_seconds=...)
        - add_interval_job(name=..., coro=..., interval_seconds=...)
        """

        name = kwargs.pop("name", None)

        func = kwargs.pop("func", None)
        callback = kwargs.pop("callback", None)
        coro = kwargs.pop("coro", None)
        coroutine = kwargs.pop("coroutine", None)
        job_func = kwargs.pop("job_func", None)
        handler = kwargs.pop("handler", None)

        resolved_func = func or callback or coro or coroutine or job_func or handler

        interval = kwargs.pop("interval", None)
        interval_seconds = kwargs.pop("interval_seconds", None)
        resolved_interval = interval if interval is not None else interval_seconds

        remaining_args = list(args)

        # RiskManager uses: add_interval_job(callback, interval_seconds=..., name=...)
        if remaining_args:
            first = remaining_args.pop(0)

            if name is not None and callable(first) and resolved_func is None:
                resolved_func = first
            elif name is None:
                name = first
            elif resolved_func is None:
                resolved_func = first

        if remaining_args and resolved_func is None:
            resolved_func = remaining_args.pop(0)

        if remaining_args and resolved_interval is None:
            resolved_interval = remaining_args.pop(0)

        if name is None:
            raise TypeError("add_interval_job() missing required argument: 'name'")

        if resolved_func is None:
            raise TypeError("add_interval_job() missing required argument: 'func'/'callback'/'coro'")

        if resolved_interval is None:
            raise TypeError("add_interval_job() missing required argument: 'interval' or 'interval_seconds'")

        if isinstance(resolved_interval, timedelta):
            resolved_interval = resolved_interval.total_seconds()

        resolved_interval = float(resolved_interval)

        supported = _scheduler_add_interval_job_parameters(self._scheduler)
        forwarded: dict[str, Any] = {}

        # Alias cleanup.
        if "overlap" in kwargs and "allow_overlap" in supported and "allow_overlap" not in kwargs:
            kwargs["allow_overlap"] = kwargs.pop("overlap")

        for key in (
            "args",
            "kwargs",
            "run_immediately",
            "max_retries",
            "retry_delay",
            "timeout",
            "allow_overlap",
            "enabled",
        ):
            if key in kwargs and key in supported:
                forwarded[key] = kwargs.pop(key)

        # Drop legacy/non-core fields.
        kwargs.pop("job_id", None)
        kwargs.pop("metadata", None)
        kwargs.pop("max_runs", None)
        kwargs.pop("start_at", None)

        for key in list(kwargs):
            if key in supported:
                forwarded[key] = kwargs.pop(key)
            else:
                kwargs.pop(key)

        return self._scheduler.add_interval_job(
            str(name),
            resolved_func,
            interval=resolved_interval,
            **forwarded,
        )

    def add_delayed_job(self, *args: Any, **kwargs: Any) -> Any:
        name = kwargs.pop("name", None)

        func = kwargs.pop("func", None)
        callback = kwargs.pop("callback", None)
        coro = kwargs.pop("coro", None)
        coroutine = kwargs.pop("coroutine", None)
        job_func = kwargs.pop("job_func", None)
        handler = kwargs.pop("handler", None)

        resolved_func = func or callback or coro or coroutine or job_func or handler

        delay = kwargs.pop("delay", None)
        delay_seconds = kwargs.pop("delay_seconds", None)
        resolved_delay = delay if delay is not None else delay_seconds

        remaining_args = list(args)

        if remaining_args:
            first = remaining_args.pop(0)

            if name is not None and callable(first) and resolved_func is None:
                resolved_func = first
            elif name is None:
                name = first
            elif resolved_func is None:
                resolved_func = first

        if remaining_args and resolved_func is None:
            resolved_func = remaining_args.pop(0)

        if remaining_args and resolved_delay is None:
            resolved_delay = remaining_args.pop(0)

        if name is None:
            raise TypeError("add_delayed_job() missing required argument: 'name'")

        if resolved_func is None:
            raise TypeError("add_delayed_job() missing required argument: 'func'/'callback'/'coro'")

        if resolved_delay is None:
            raise TypeError("add_delayed_job() missing required argument: 'delay' or 'delay_seconds'")

        if isinstance(resolved_delay, timedelta):
            resolved_delay = resolved_delay.total_seconds()

        resolved_delay = float(resolved_delay)

        supported = _scheduler_add_delayed_job_parameters(self._scheduler)

        forwarded: dict[str, Any] = {}

        for key in (
            "args",
            "kwargs",
            "max_retries",
            "retry_delay",
            "timeout",
            "enabled",
        ):
            if key in kwargs and key in supported:
                forwarded[key] = kwargs.pop(key)

        kwargs.pop("job_id", None)
        kwargs.pop("metadata", None)

        for key in list(kwargs):
            if key in supported:
                forwarded[key] = kwargs.pop(key)
            else:
                kwargs.pop(key)

        return self._scheduler.add_delayed_job(
            str(name),
            resolved_func,
            delay=resolved_delay,
            **forwarded,
        )


def _scheduler_add_interval_job_parameters(scheduler: Scheduler) -> set[str]:
    try:
        return set(inspect.signature(scheduler.add_interval_job).parameters)
    except Exception:
        return set()


def _scheduler_add_delayed_job_parameters(scheduler: Scheduler) -> set[str]:
    try:
        return set(inspect.signature(scheduler.add_delayed_job).parameters)
    except Exception:
        return set()


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def start_if_supported(component: Any) -> None:
    method = getattr(component, "start", None)
    if callable(method):
        await maybe_await(method())


async def stop_if_supported(component: Any) -> None:
    method = getattr(component, "stop", None)
    if callable(method):
        await maybe_await(method())


def stats_if_supported(component: Any) -> dict[str, Any]:
    method = getattr(component, "stats", None)
    if callable(method):
        try:
            value = method()
            return value if isinstance(value, dict) else {"value": value}
        except Exception as exc:
            return {"error": str(exc)}
    return {}


def _get_attr(component: Any, names: Iterable[str]) -> Any | None:
    for name in names:
        if hasattr(component, name):
            return getattr(component, name)
    return None


def _instantiate_flexibly(cls: type, **available_kwargs: Any) -> Any:
    """
    Instantiate a class using only kwargs supported by its __init__ signature.

    This keeps the run script stable when project components have slightly
    different constructor names, while still preserving constructor dependency
    injection.
    """

    signature = inspect.signature(cls)
    kwargs: dict[str, Any] = {}

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    if accepts_var_kwargs:
        kwargs = {
            key: value
            for key, value in available_kwargs.items()
            if value is not None
        }
        return cls(**kwargs)

    for name, parameter in signature.parameters.items():
        if name == "self":
            continue

        if name in available_kwargs and available_kwargs[name] is not None:
            kwargs[name] = available_kwargs[name]
            continue

        # Common aliases used across packages.
        aliases = {
            "app_config": ["app_config", "config"],
            "core_config": ["app_config", "config"],
            "settings": ["app_config", "config"],
            "cfg": ["config", "app_config"],
            "strategy_config": ["strategy_config", "config"],
            "risk_config": ["risk_config", "config"],
            "event_bus": ["event_bus"],
            "bus": ["event_bus"],
            "scheduler": ["scheduler"],
            "clock": ["clock"],
            "candles_cache": ["candles_cache"],
            "candle_cache": ["candles_cache"],
            "trades_cache": ["trades_cache"],
            "trade_cache": ["trades_cache"],
            "funding_cache": ["funding_cache"],
            "open_interest_cache": ["open_interest_cache"],
            "oi_cache": ["open_interest_cache"],
            "orderbook_cache": ["orderbook_cache"],
            "order_book_cache": ["orderbook_cache"],
            "data_caches": ["data_caches"],
            "caches": ["data_caches"],
        }

        for alias in aliases.get(name, []):
            if alias in available_kwargs and available_kwargs[alias] is not None:
                kwargs[name] = available_kwargs[alias]
                break

    return cls(**kwargs)


def _try_build_config_from_annotation(annotation: Any) -> Any | None:
    """
    Try to build a domain-specific config object from an __init__ annotation.

    Many analytics classes have signatures like:
        __init__(config: FundingAnalyticsConfig | None = None, ...)

    Passing core.Config into such classes is wrong. This helper builds the
    annotated config when it can be constructed without arguments.
    """

    if annotation is inspect.Signature.empty:
        return None

    candidates: list[Any] = []

    # Python 3.10 union annotations expose __args__.
    args = getattr(annotation, "__args__", None)
    if args:
        candidates.extend(item for item in args if item is not type(None))  # noqa: E721
    else:
        candidates.append(annotation)

    for candidate in candidates:
        if not isinstance(candidate, type):
            continue

        name = candidate.__name__.lower()
        if not name.endswith("config"):
            continue

        try:
            return candidate()
        except Exception:
            continue

    return None


def _instantiate_runtime_component(
    cls: type,
    *,
    app_config: Config,
    event_bus: EventBus,
    scheduler: Scheduler,
    **available_kwargs: Any,
) -> Any:
    """
    Instantiate runtime project components safely.

    Important:
    - Do NOT pass core.Config as the generic `config` argument into analytics
      components. Analytics modules usually have their own config dataclasses.
    - If a constructor's `config` annotation looks like a domain config class,
      instantiate that config class.
    - Still pass core Config via `app_config` / `core_config` if supported.
    """

    signature = inspect.signature(cls)
    kwargs: dict[str, Any] = {}

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    if accepts_var_kwargs:
        # Keep this conservative: do not include generic config=app_config.
        kwargs.update(
            {
                "app_config": app_config,
                "event_bus": event_bus,
                "scheduler": scheduler,
                **{
                    key: value
                    for key, value in available_kwargs.items()
                    if value is not None
                },
            }
        )
        return cls(**kwargs)

    for name, parameter in signature.parameters.items():
        if name == "self":
            continue

        if name == "config":
            domain_config = _try_build_config_from_annotation(parameter.annotation)
            if domain_config is not None:
                kwargs[name] = domain_config
            # If config has a default and we cannot infer the domain config,
            # leave it unset. This is safer than passing core.Config.
            continue

        if name in {"app_config", "core_config", "settings"}:
            kwargs[name] = app_config
            continue

        if name in {"event_bus", "bus"}:
            kwargs[name] = event_bus
            continue

        if name == "scheduler":
            kwargs[name] = scheduler
            continue

        if name in available_kwargs and available_kwargs[name] is not None:
            kwargs[name] = available_kwargs[name]
            continue

        aliases = {
            "candles_cache": ["candles_cache"],
            "candle_cache": ["candles_cache"],
            "trades_cache": ["trades_cache"],
            "trade_cache": ["trades_cache"],
            "funding_cache": ["funding_cache"],
            "open_interest_cache": ["open_interest_cache"],
            "oi_cache": ["open_interest_cache"],
            "orderbook_cache": ["orderbook_cache"],
            "order_book_cache": ["orderbook_cache"],
            "data_caches": ["data_caches"],
            "caches": ["data_caches"],
        }

        for alias in aliases.get(name, []):
            if alias in available_kwargs and available_kwargs[alias] is not None:
                kwargs[name] = available_kwargs[alias]
                break

    return cls(**kwargs)


# =============================================================================
# Historical stream detection
# =============================================================================


def _file_exists_any(path_without_suffix: Path) -> bool:
    return any(
        path_without_suffix.with_suffix(suffix).exists()
        for suffix in (".csv", ".parquet", ".json", ".jsonl")
    )


def _stream_exists(
    *,
    root: Path,
    exchange: str,
    market_type: str,
    data_type: BacktestDataType,
    symbol: str,
    timeframe: str | None = None,
) -> bool:
    base = root / exchange / market_type

    if data_type == BacktestDataType.CANDLES:
        if timeframe is None:
            return False
        return _file_exists_any(
            base / "candles" / symbol / timeframe / f"{symbol}_{timeframe}"
        )

    if data_type == BacktestDataType.FUNDING:
        return _file_exists_any(base / "funding" / symbol / symbol)

    if data_type == BacktestDataType.OPEN_INTEREST:
        # HistoryDownloader may store OI either as:
        #   open_interest/SYMBOL/5m/SYMBOL_5m.csv
        # or:
        #   open_interest/SYMBOL/SYMBOL.csv
        oi_root = base / "open_interest" / symbol

        if timeframe is not None and _file_exists_any(oi_root / timeframe / f"{symbol}_{timeframe}"):
            return True

        if _file_exists_any(oi_root / symbol):
            return True

        return any(oi_root.glob(f"**/{symbol}*.*"))

    if data_type == BacktestDataType.TRADES:
        return _file_exists_any(base / "trades" / symbol / symbol)

    if data_type in {BacktestDataType.ORDERBOOK, BacktestDataType.ORDERBOOK_SNAPSHOT}:
        # DataLoader discovers this stream as data_type="orderbook" and expects:
        #   orderbook/<SYMBOL>/<SYMBOL>.<format>
        #
        # Do not treat orderbook_snapshot/<SYMBOL>/<SYMBOL> as compatible here,
        # because DataLoader will still ask for "orderbook" and fail.
        return _file_exists_any(base / "orderbook" / symbol / symbol)

    if data_type == BacktestDataType.LIQUIDATIONS:
        return _file_exists_any(base / "liquidations" / symbol / symbol)

    return False


def _copy_file_if_needed(source_base: Path, target_base: Path) -> bool:
    """
    Copy source_base.* to target_base.* if target does not exist.

    Returns True when a compatible file exists or was created.
    """

    for suffix in (".csv", ".parquet", ".json", ".jsonl"):
        source = source_base.with_suffix(suffix)
        target = target_base.with_suffix(suffix)

        if target.exists():
            return True

        if not source.exists():
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        return True

    return False


def ensure_open_interest_timeframe_compat(runtime: FullBacktestRunConfig) -> None:
    """
    DataLoader currently discovers open_interest with runtime.timeframes, which
    is usually ["1m"] for candle replay. Binance OI history is downloaded at
    5m, so the files are often stored as:

        open_interest/SYMBOL/5m/SYMBOL_5m.csv

    while DataLoader asks for:

        open_interest/SYMBOL/1m/SYMBOL_1m.csv

    This helper creates lightweight compatibility copies for the configured
    loader timeframe. The data remains 5m OI internally; this only satisfies
    DataLoader's file discovery convention.
    """

    if not runtime.include_open_interest:
        return

    target_timeframe = runtime.timeframes[0]
    if target_timeframe == "5m":
        return

    root = runtime.data_dir / runtime.exchange / runtime.market_type / "open_interest"

    for symbol in runtime.symbols:
        symbol_root = root / symbol

        target_base = symbol_root / target_timeframe / f"{symbol}_{target_timeframe}"

        if _file_exists_any(target_base):
            continue

        # Most common HistoryDownloader layout.
        source_candidates = [
            symbol_root / "5m" / f"{symbol}_5m",
            symbol_root / symbol,
        ]

        # Fallback: any file under open_interest/SYMBOL matching symbol.
        source_candidates.extend(
            candidate.with_suffix("")
            for candidate in symbol_root.glob(f"**/{symbol}*.*")
        )

        for source_base in source_candidates:
            if _copy_file_if_needed(source_base, target_base):
                logger.warning(
                    "Created open_interest timeframe compatibility file for DataLoader",
                    extra={
                        "symbol": symbol,
                        "source_base": str(source_base),
                        "target_base": str(target_base),
                        "target_timeframe": target_timeframe,
                    },
                )
                break


def detect_available_data_types(runtime: FullBacktestRunConfig) -> set[BacktestDataType]:
    """
    Detect streams available for all requested symbols.

    Candles are required for every symbol. Optional streams are enabled only
    when local files exist for every requested symbol, so DataLoader does not
    fail on a partially downloaded multi-symbol dataset.
    """

    selected: set[BacktestDataType] = set()
    timeframe = runtime.timeframes[0]

    checks = [
        (runtime.include_candles, BacktestDataType.CANDLES, timeframe),
        (runtime.include_funding, BacktestDataType.FUNDING, None),
        (runtime.include_open_interest, BacktestDataType.OPEN_INTEREST, timeframe),
        (runtime.include_trades, BacktestDataType.TRADES, None),
        (runtime.include_orderbook, BacktestDataType.ORDERBOOK, None),
        (runtime.include_liquidations, BacktestDataType.LIQUIDATIONS, None),
    ]

    for enabled, data_type, stream_timeframe in checks:
        if not enabled:
            continue

        missing_symbols: list[str] = []

        for symbol in runtime.symbols:
            exists = _stream_exists(
                root=runtime.data_dir,
                exchange=runtime.exchange,
                market_type=runtime.market_type,
                data_type=data_type,
                symbol=symbol,
                timeframe=stream_timeframe,
            )

            if not exists:
                missing_symbols.append(symbol)

        if not missing_symbols:
            selected.add(data_type)
            continue

        if data_type == BacktestDataType.CANDLES:
            expected = (
                f"{runtime.data_dir}/{runtime.exchange}/{runtime.market_type}/"
                f"candles/<SYMBOL>/{timeframe}/<SYMBOL>_{timeframe}.{runtime.input_format.value}"
            )
            raise FileNotFoundError(
                "Required candles history was not found for all symbols. "
                f"missing_symbols={missing_symbols}. Expected layout: {expected}"
            )

        logger.warning(
            "Historical stream is enabled but local files are missing for some symbols; "
            "disabling this stream for the run",
            extra={
                "data_type": data_type.value,
                "missing_symbols": missing_symbols,
                "timeframe": stream_timeframe,
            },
        )

    return selected


# =============================================================================
# Backtest config
# =============================================================================


def build_backtest_config(runtime: FullBacktestRunConfig) -> BacktestConfig:
    data_types = detect_available_data_types(runtime)

    config = BacktestConfig.default_binance_futures(
        run_name=runtime.run_name,
        symbols=runtime.symbols,
        timeframes=runtime.timeframes,
        start_time=runtime.start_time,
        end_time=runtime.end_time,
        initial_balance=runtime.initial_balance,
    )

    config.mode = BacktestMode.MULTI_STRATEGY
    config.warmup_start_time = runtime.warmup_start_time
    config.data_dir = runtime.data_dir
    config.output_dir = runtime.output_dir

    config.strategies = list(runtime.strategies)
    config.strategy_preset = runtime.strategy_preset
    config.test_all_registered_strategies = runtime.use_all_registered_strategies

    config.use_candles = BacktestDataType.CANDLES in data_types
    config.use_funding = BacktestDataType.FUNDING in data_types
    config.use_open_interest = BacktestDataType.OPEN_INTEREST in data_types
    config.use_trades = BacktestDataType.TRADES in data_types
    config.use_orderbook = BacktestDataType.ORDERBOOK in data_types
    config.use_liquidations = BacktestDataType.LIQUIDATIONS in data_types
    config.use_mark_price = True
    config.use_index_price = True

    config.data_loader.data_dir = runtime.data_dir
    config.data_loader.input_format = runtime.input_format
    config.data_loader.exchange = runtime.exchange
    config.data_loader.market_type = runtime.market_type
    config.data_loader.symbols = list(runtime.symbols)
    config.data_loader.timeframes = list(runtime.timeframes)
    # Keep DataLoader strictly aligned with detected local files.
    # This prevents defaults from BacktestConfig.default_binance_futures()
    # from re-introducing unavailable streams such as orderbook.
    config.data_loader.data_types = set(data_types)
    config.data_loader.require_orderbook = False
    config.data_loader.require_trades = False
    config.data_loader.require_funding = False
    config.data_loader.require_open_interest = False

    config.data_loader.require_candles = True
    config.data_loader.require_funding = False
    config.data_loader.require_open_interest = False
    config.data_loader.require_trades = False
    config.data_loader.require_orderbook = False
    config.data_loader.allow_empty_optional_streams = True
    config.data_loader.validation_level = DataValidationLevel.BASIC
    config.data_loader.gap_policy = DataGapPolicy.WARN
    config.data_loader.drop_duplicate_events = True

    config.market_replay.replay_mode = ReplayMode.FULL_RUN
    config.market_replay.batch_events_by_timestamp = True
    config.market_replay.deterministic_replay = True
    config.market_replay.fail_on_emit_error = True
    config.market_replay.emit_replay_lifecycle_events = True
    config.market_replay.emit_market_candles = BacktestDataType.CANDLES in data_types
    config.market_replay.emit_market_trades = BacktestDataType.TRADES in data_types
    config.market_replay.emit_market_orderbook = BacktestDataType.ORDERBOOK in data_types
    config.market_replay.emit_market_funding = BacktestDataType.FUNDING in data_types
    config.market_replay.emit_market_open_interest = BacktestDataType.OPEN_INTEREST in data_types
    config.market_replay.emit_market_liquidations = BacktestDataType.LIQUIDATIONS in data_types

    config.cost_model.slippage_model = SlippageModel.FIXED_BPS
    config.cost_model.fixed_slippage_bps = 2.0
    config.cost_model.maker_fee_bps = 2.0
    config.cost_model.taker_fee_bps = 4.0
    config.cost_model.include_commissions = True
    config.cost_model.include_slippage = True
    config.cost_model.include_spread_cost = True
    config.cost_model.include_funding = BacktestDataType.FUNDING in data_types

    config.execution_simulator.exchange = runtime.exchange
    config.execution_simulator.market_type = runtime.market_type
    config.execution_simulator.fill_model = FillModel.NEXT_CANDLE_OPEN
    config.execution_simulator.liquidity_model = LiquidityModel.CANDLE_VOLUME_PERCENT
    config.execution_simulator.max_volume_participation_pct = 10.0
    config.execution_simulator.allow_market_orders = True
    config.execution_simulator.allow_limit_orders = True
    config.execution_simulator.allow_stop_orders = True
    config.execution_simulator.allow_reduce_only = True
    config.execution_simulator.allow_partial_fills = True
    config.execution_simulator.reject_if_no_price = True
    config.execution_simulator.reject_if_no_liquidity = False
    config.execution_simulator.record_orders = True
    config.execution_simulator.record_fills = True

    config.position_simulator.initial_balance = runtime.initial_balance
    config.position_simulator.quote_currency = "USDT"
    config.position_simulator.position_accounting_mode = PositionAccountingMode.NETTING
    config.position_simulator.pnl_accounting_mode = PnLAccountingMode.REALIZED_AND_UNREALIZED
    config.position_simulator.default_leverage = 2.0
    config.position_simulator.max_leverage = 10.0
    config.position_simulator.maintenance_margin_rate = 0.005
    config.position_simulator.enable_mark_to_market = True
    config.position_simulator.enable_funding_application = BacktestDataType.FUNDING in data_types
    config.position_simulator.enable_liquidation_check = True
    config.position_simulator.emit_position_events = True
    config.position_simulator.record_positions = True
    config.position_simulator.record_equity_curve = True

    config.report_builder.enabled = True
    config.report_builder.output_dir = runtime.output_dir
    config.report_builder.report_title = "BTC/DOGE/SOL Last 2 Days Full Pipeline Backtest"
    config.report_builder.formats = [
        ReportFormat.MARKDOWN,
        ReportFormat.JSON,
        ReportFormat.CSV,
    ]
    config.report_builder.sections = [
        ReportSection.SUMMARY,
        ReportSection.TRADES,
        ReportSection.POSITIONS,
        ReportSection.STRATEGIES,
        ReportSection.RISK,
        ReportSection.EXECUTION,
        ReportSection.COSTS,
        ReportSection.SIGNALS,
        ReportSection.WARNINGS,
    ]
    config.report_builder.save_result_json = True
    config.report_builder.save_trades_csv = True
    config.report_builder.save_positions_csv = True
    config.report_builder.save_equity_curve_csv = True
    config.report_builder.save_events_jsonl = True

    config.strategy_tester.run_name = config.run_name
    config.strategy_tester.mode = BacktestMode.MULTI_STRATEGY
    config.strategy_tester.exchange = runtime.exchange
    config.strategy_tester.market_type = runtime.market_type
    config.strategy_tester.symbols = list(runtime.symbols)
    config.strategy_tester.timeframes = list(runtime.timeframes)
    config.strategy_tester.strategies = list(runtime.strategies)
    config.strategy_tester.strategy_preset = runtime.strategy_preset
    config.strategy_tester.test_all_registered_strategies = runtime.use_all_registered_strategies

    config.strategy_tester.require_strategy_engine = True
    config.strategy_tester.require_signal_processor = True
    config.strategy_tester.require_risk_manager = True
    config.strategy_tester.require_analytics = runtime.require_analytics

    config.strategy_tester.use_production_data_caches = True
    config.strategy_tester.use_production_analytics = True
    config.strategy_tester.use_production_strategy_engine = True
    config.strategy_tester.use_production_risk_manager = True

    config.strategy_tester.disable_live_exchange_execution = True
    config.strategy_tester.fail_if_live_execution_detected = runtime.fail_if_live_execution_detected
    config.strategy_tester.collect_event_log = True
    config.strategy_tester.collect_signal_records = True
    config.strategy_tester.collect_risk_records = True
    config.strategy_tester.collect_execution_records = True
    config.strategy_tester.collect_position_records = True
    config.strategy_tester.cleanup_after_run = runtime.cleanup_after_run
    config.strategy_tester.stop_on_first_error = runtime.stop_on_first_error

    config.metadata.update(
        {
            "runner": "backtesting.run_backtest_full",
            "detected_data_types": sorted(item.value for item in data_types),
            "period": "last_2d",
            "backtest_days": runtime.backtest_days,
        }
    )

    config.validate()
    return config


# =============================================================================
# Component builders
# =============================================================================


def build_data_caches(
    *,
    app_config: Config,
    event_bus: EventBus,
    scheduler: Scheduler,
    enabled_data_types: set[BacktestDataType],
) -> list[Any]:
    caches: list[Any] = [
        CandlesCache(
            config=app_config,
            event_bus=event_bus,
            scheduler=scheduler,
        )
    ]

    if BacktestDataType.TRADES in enabled_data_types:
        caches.append(
            TradesCache(
                config=app_config,
                event_bus=event_bus,
                scheduler=scheduler,
            )
        )

    if BacktestDataType.FUNDING in enabled_data_types:
        caches.append(
            FundingCache(
                config=app_config,
                event_bus=event_bus,
                scheduler=scheduler,
            )
        )

    if BacktestDataType.OPEN_INTEREST in enabled_data_types:
        caches.append(
            OpenInterestCache(
                config=app_config,
                event_bus=event_bus,
                scheduler=scheduler,
            )
        )

    if BacktestDataType.ORDERBOOK in enabled_data_types:
        caches.append(
            OrderBookCache(
                config=app_config,
                event_bus=event_bus,
                scheduler=scheduler,
            )
        )

    return caches


DEFAULT_ANALYTICS_CANDIDATES: list[tuple[str, str]] = [
    # Common direct module/class layouts.
    ("analytics.price_action.engine", "PriceActionAnalytics"),
    ("analytics.price_action.analyzer", "PriceActionAnalytics"),
    ("analytics.price_action.market_structure", "MarketStructureAnalytics"),
    ("analytics.price_action.market_structure_analyzer", "MarketStructureAnalyzer"),
    ("analytics.funding.engine", "FundingAnalytics"),
    ("analytics.funding.analyzer", "FundingAnalytics"),
    ("analytics.open_interest.engine", "OpenInterestAnalytics"),
    ("analytics.open_interest.analyzer", "OpenInterestAnalytics"),
    ("analytics.orderflow.engine", "OrderflowAnalytics"),
    ("analytics.orderflow.analyzer", "OrderflowAnalytics"),
    ("analytics.liquidations.engine", "LiquidationsAnalytics"),
    ("analytics.liquidations.analyzer", "LiquidationsAnalytics"),
    ("analytics.liquidity.engine", "LiquidityAnalytics"),
    ("analytics.liquidity.analyzer", "LiquidityAnalytics"),
    ("analytics.spreads.engine", "SpreadsAnalytics"),
    ("analytics.spreads.analyzer", "SpreadsAnalytics"),
    ("analytics.spoofing.engine", "SpoofingAnalytics"),
    ("analytics.spoofing.analyzer", "SpoofingAnalytics"),
    ("analytics.whales.engine", "WhalesAnalytics"),
    ("analytics.whales.analyzer", "WhalesAnalytics"),

    # File-based layouts that often exist in this project structure.
    ("analytics.price_action.market_structure_detector", "MarketStructureDetector"),
    ("analytics.price_action.fvg_detector", "FVGDetector"),
    ("analytics.price_action.support_resistance_detector", "SupportResistanceDetector"),
    ("analytics.funding.funding_analyzer", "FundingAnalyzer"),
    ("analytics.open_interest.oi_analyzer", "OpenInterestAnalyzer"),
    ("analytics.open_interest.open_interest_analyzer", "OpenInterestAnalyzer"),
    ("analytics.orderflow.cvd_analyzer", "CVDAnalyzer"),
    ("analytics.orderflow.orderflow_analyzer", "OrderflowAnalyzer"),
    ("analytics.liquidity.liquidity_analyzer", "LiquidityAnalyzer"),
    ("analytics.liquidations.liquidation_analyzer", "LiquidationAnalyzer"),
]


def _parse_analytics_specs(specs: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []

    for spec in specs:
        if ":" not in spec:
            raise ValueError(
                "ANALYTICS_COMPONENTS must use module:Class format. "
                f"Invalid item: {spec!r}"
            )

        module_name, class_name = spec.split(":", 1)
        parsed.append((module_name.strip(), class_name.strip()))

    return parsed


def _import_class(module_name: str, class_name: str) -> type | None:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        logger.debug(
            "Analytics module import skipped",
            extra={
                "module_name": module_name,
                "class_name": class_name,
                "error": str(exc),
            },
        )
        return None

    cls = getattr(module, class_name, None)
    if isinstance(cls, type):
        return cls

    logger.debug(
        "Analytics class not found in module",
        extra={
            "module_name": module_name,
            "class_name": class_name,
        },
    )
    return None

def _looks_like_analytics_class(cls: type) -> bool:
    """
    Best-effort filter for auto-discovered analytics components.

    We only instantiate classes that look like runtime analytics components,
    not DTOs/configs/exceptions.
    """

    name = cls.__name__.lower()

    if name.endswith(("config", "state", "snapshot", "record", "event", "result", "error", "exception")):
        return False

    if not any(token in name for token in ("analytics", "analyzer", "detector", "engine")):
        return False

    # Runtime components in this project normally have register/start/stop or
    # an event-driven handler. Keep the check soft because some analytics
    # components only expose register().
    runtime_methods = {"register", "start", "stop", "handle", "on_event"}
    return any(hasattr(cls, method) for method in runtime_methods)


def _discover_analytics_candidates() -> list[tuple[str, str]]:
    """
    Scan the local analytics package and find plausible runtime analytics
    classes. This avoids hard-coding module names in the runner.
    """

    try:
        package = importlib.import_module("analytics")
    except Exception as exc:
        logger.warning(
            "Could not import analytics package",
            extra={"error": str(exc)},
        )
        return []

    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return []

    candidates: list[tuple[str, str]] = []

    for module_info in pkgutil.walk_packages(package_path, prefix="analytics."):
        module_name = module_info.name

        # Skip likely non-runtime modules.
        leaf = module_name.rsplit(".", 1)[-1].lower()
        if leaf in {"models", "enums", "exceptions", "config", "utils", "__init__"}:
            continue

        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            logger.debug(
                "Analytics auto-discovery skipped module",
                extra={"module_name": module_name, "error": str(exc)},
            )
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue

            if _looks_like_analytics_class(obj):
                candidates.append((module_name, obj.__name__))

    # Preserve order and remove duplicates.
    unique: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        unique.append(candidate)
        seen.add(candidate)

    return unique


def build_analytics_components(
    *,
    runtime: FullBacktestRunConfig,
    app_config: Config,
    event_bus: EventBus,
    scheduler: Scheduler,
    data_caches: list[Any],
    enabled_data_types: set[BacktestDataType],
) -> list[Any]:
    """
    Build production analytics components.

    Priority:
    1. ANALYTICS_COMPONENTS env, format:
       analytics.price_action.engine:PriceActionAnalytics,analytics.funding.engine:FundingAnalytics
    2. Default common candidates.

    StrategyTester receives these components only to manage lifecycle.
    The data flow still stays event-driven through EventBus.
    """

    cache_kwargs = {
        "candles_cache": _get_attr_by_type(data_caches, CandlesCache),
        "trades_cache": _get_attr_by_type(data_caches, TradesCache),
        "funding_cache": _get_attr_by_type(data_caches, FundingCache),
        "open_interest_cache": _get_attr_by_type(data_caches, OpenInterestCache),
        "orderbook_cache": _get_attr_by_type(data_caches, OrderBookCache),
        "data_caches": data_caches,
    }

    if runtime.analytics_specs:
        candidates = _parse_analytics_specs(runtime.analytics_specs)
    else:
        discovered = _discover_analytics_candidates()
        candidates = [*DEFAULT_ANALYTICS_CANDIDATES, *discovered]

    components: list[Any] = []
    seen_classes: set[type] = set()

    for module_name, class_name in candidates:
        cls = _import_class(module_name, class_name)
        if cls is None or cls in seen_classes:
            continue

        try:
            component = _instantiate_runtime_component(
                cls,
                app_config=app_config,
                event_bus=event_bus,
                scheduler=scheduler,
                **cache_kwargs,
            )
        except Exception as exc:
            logger.warning(
                "Analytics component could not be instantiated",
                extra={
                    "module_name": module_name,
                    "class_name": class_name,
                    "error": str(exc),
                },
            )
            if _env_bool("PRINT_ANALYTICS_WIRING_ERRORS", False):
                print(
                    "Analytics component could not be instantiated: "
                    f"{module_name}:{class_name} -> {exc}"
                )
            continue

        components.append(component)
        seen_classes.add(cls)

    if runtime.require_analytics and not components:
        raise RuntimeError(
            "No analytics components were loaded. "
            "Set ANALYTICS_COMPONENTS='module.path:ClassName,...', "
            "set REQUIRE_ANALYTICS=0 for a technical smoke run, "
            "or ensure your analytics classes expose register()/start() and "
            "can be instantiated with config/event_bus/scheduler/cache dependencies."
        )

    if not components:
        logger.warning(
            "No analytics components loaded. Backtest can run only if require_analytics=False, "
            "but strategies probably will not receive analytics.* events."
        )

    return components


def _get_attr_by_type(items: list[Any], cls: type) -> Any | None:
    for item in items:
        if isinstance(item, cls):
            return item
    return None


def _camel_to_snake(name: str) -> str:
    chars: list[str] = []

    for index, char in enumerate(name):
        if char.isupper() and index > 0:
            previous = name[index - 1]
            next_char = name[index + 1] if index + 1 < len(name) else ""
            if previous != "_" and (not previous.isupper() or next_char.islower()):
                chars.append("_")
        chars.append(char.lower())

    return "".join(chars)


def _strategy_name_from_class(cls: type[BaseStrategy], module_name: str) -> str:
    class_attr = getattr(cls, "strategy_name", None)
    if isinstance(class_attr, str) and class_attr.strip():
        return class_attr.strip()

    leaf = module_name.rsplit(".", 1)[-1]

    for suffix in ("_strategy", "_strategies"):
        if leaf.endswith(suffix):
            return leaf[: -len(suffix)]

    snake = _camel_to_snake(cls.__name__)
    if snake.endswith("_strategy"):
        snake = snake[: -len("_strategy")]

    return snake


def _discover_strategy_factories() -> dict[str, type[BaseStrategy]]:
    """
    Discover concrete BaseStrategy subclasses under strategy.strategies.

    build_default_strategy_registry() needs concrete strategy factories or
    instances. It does not instantiate strategies from catalog metadata alone.
    """

    try:
        package = importlib.import_module("strategy.strategies")
    except Exception as exc:
        logger.warning(
            "Could not import strategy.strategies package",
            extra={"error": str(exc)},
        )
        return {}

    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return {}

    factories: dict[str, type[BaseStrategy]] = {}

    for module_info in pkgutil.walk_packages(package_path, prefix="strategy.strategies."):
        module_name = module_info.name
        leaf = module_name.rsplit(".", 1)[-1].lower()

        if leaf in {"__init__", "base", "utils", "config", "enums", "models", "exceptions"}:
            continue

        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            logger.debug(
                "Strategy auto-discovery skipped module",
                extra={"module_name": module_name, "error": str(exc)},
            )
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue

            if obj is BaseStrategy:
                continue

            try:
                is_strategy = issubclass(obj, BaseStrategy)
            except TypeError:
                continue

            if not is_strategy:
                continue

            name = obj.__name__.lower()
            if name.startswith("base") or name in {"tradingstrategy", "strategyvalidationmixin"}:
                continue

            if inspect.isabstract(obj):
                continue

            strategy_name = _strategy_name_from_class(obj, module_name)
            factories.setdefault(strategy_name, obj)

    return factories


def _registry_count(registry: Any) -> int:
    count = getattr(registry, "count", None)
    if callable(count):
        return int(count())

    list_all = getattr(registry, "list_all", None)
    if callable(list_all):
        return len(list_all())

    strategies = getattr(registry, "_strategies", None)
    if isinstance(strategies, dict):
        return len(strategies)

    return 0


def build_strategy_stack(
    *,
    runtime: FullBacktestRunConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
) -> tuple[Any, SignalProcessor, StrategyEngine]:
    """
    Build StrategyRegistry + shared StrategyRuntimeState + SignalProcessor + StrategyEngine.

    StrategyConfig contains definitions/presets, but concrete strategy classes
    still have to be registered as factories or instances. This runner discovers
    strategy classes under strategy.strategies and passes them as factories to
    build_default_strategy_registry().
    """

    preset_name = runtime.strategy_preset or "intraday"

    strategy_config = build_default_strategy_config(
        symbols=runtime.symbols,
        preset_name=preset_name,
        use_required_features=False,
    )

    if runtime.strategies:
        strategy_config.preset.enabled_strategy_names = list(runtime.strategies)

    strategy_state = StrategyRuntimeState()
    strategy_factories = _discover_strategy_factories()

    strategy_registry = build_default_strategy_registry(
        config=strategy_config,
        event_bus=event_bus,
        scheduler=scheduler,
        strategy_factories=strategy_factories,
        strict=False,
        emit_events=False,
    )

    if _registry_count(strategy_registry) <= 0:
        discovered_names = ", ".join(sorted(strategy_factories)) or "none"
        configured_names = ", ".join(sorted(strategy_config.strategies.keys())) or "none"
        raise RuntimeError(
            "StrategyRegistry has no registered strategies after bootstrap. "
            "This means the preset/config exists, but no matching concrete strategy "
            "classes were instantiated. "
            f"preset={preset_name!r}; "
            f"configured={configured_names}; "
            f"discovered={discovered_names}. "
            "Check that files under strategy/strategies expose concrete BaseStrategy "
            "subclasses and their file/class names match catalog names."
        )

    signal_processor = SignalProcessor(
        config=strategy_config,
        registry=strategy_registry,
        state=strategy_state,
        event_bus=event_bus,
        scheduler=scheduler,
    )

    strategy_engine = StrategyEngine(
        config=strategy_config,
        event_bus=event_bus,
        scheduler=scheduler,
        registry=strategy_registry,
        state=strategy_state,
        processor=signal_processor,
    )

    return strategy_registry, signal_processor, strategy_engine


def build_risk_manager(
    *,
    app_config: Config,
    event_bus: EventBus,
    scheduler: Scheduler,
) -> RiskManager:
    risk_config = RiskConfig()

    return _instantiate_flexibly(
        RiskManager,
        config=risk_config,
        risk_config=risk_config,
        app_config=app_config,
        event_bus=event_bus,
        scheduler=scheduler,
    )


# =============================================================================
# Event debug counters
# =============================================================================


DEBUG_EVENT_TOPICS: tuple[str, ...] = (
    # Replay/raw market events.
    "market.candle",
    "market.trade",
    "market.orderbook",
    "market.funding",
    "market.open_interest",
    "market.liquidation",

    # Data cache events.
    "market.candles.updated",
    "market.candle.closed",
    "market.trades.updated",
    "market.orderbook.updated",
    "market.funding.updated",
    "market.open_interest.updated",
    "market.liquidations.updated",

    # Common analytics events.
    "analytics.price_action.updated",
    "analytics.price_action.signal",
    "analytics.market_structure.updated",
    "analytics.fvg.updated",
    "analytics.support_resistance.updated",
    "analytics.funding.updated",
    "analytics.funding.signal",
    "analytics.open_interest.updated",
    "analytics.open_interest.signal",
    "analytics.oi.updated",
    "analytics.oi.signal",
    "analytics.orderflow.updated",
    "analytics.orderflow.signal",
    "analytics.liquidity.updated",
    "analytics.liquidity.signal",
    "analytics.liquidations.updated",
    "analytics.liquidations.signal",
    "analytics.spreads.updated",
    "analytics.spreads.signal",
    "analytics.spoofing.updated",
    "analytics.spoofing.signal",
    "analytics.whales.updated",
    "analytics.whales.signal",

    # Strategy/signal/risk/execution/position events.
    "strategy.context.updated",
    "strategy.evaluation.completed",
    "strategy.signal.generated",
    "signal.generated",
    "signal.rejected",
    "signal.updated",
    "signal.confirmed",
    "risk.position_blocked",
    "risk.limit_warning",
    "risk.kill_switch",
    "risk.position_close_requested",
    "risk.position_reduce_requested",
    "execution.order_submitted",
    "execution.order_accepted",
    "execution.order_rejected",
    "execution.order_cancelled",
    "execution.order_filled",
    "execution.order_partially_filled",
    "position.opened",
    "position.updated",
    "position.closed",
)


async def register_event_debug_counters(event_bus: EventBus) -> dict[str, int]:
    """
    Register lightweight counters for critical backtest topics.

    Works with both sync and async EventBus.subscribe() implementations.
    """

    counts: dict[str, int] = {topic: 0 for topic in DEBUG_EVENT_TOPICS}

    def make_handler(topic: str):
        async def handler(payload: Any) -> None:
            counts[topic] = counts.get(topic, 0) + 1

        return handler

    subscribe = getattr(event_bus, "subscribe", None)
    if not callable(subscribe):
        raise RuntimeError("EventBus does not expose subscribe().")

    for topic in DEBUG_EVENT_TOPICS:
        result = subscribe(topic, make_handler(topic))
        if inspect.isawaitable(result):
            await result

    return counts


def print_event_debug_counts(counts: dict[str, int]) -> None:
    print("")
    print("========== EVENT COUNTS ==========")

    groups = (
        (
            "Market replay/raw",
            (
                "market.candle",
                "market.trade",
                "market.orderbook",
                "market.funding",
                "market.open_interest",
                "market.liquidation",
            ),
        ),
        (
            "Data cache",
            (
                "market.candles.updated",
                "market.candle.closed",
                "market.trades.updated",
                "market.orderbook.updated",
                "market.funding.updated",
                "market.open_interest.updated",
                "market.liquidations.updated",
            ),
        ),
        (
            "Analytics",
            (
                "analytics.price_action.updated",
                "analytics.price_action.signal",
                "analytics.market_structure.updated",
                "analytics.fvg.updated",
                "analytics.support_resistance.updated",
                "analytics.funding.updated",
                "analytics.funding.signal",
                "analytics.open_interest.updated",
                "analytics.open_interest.signal",
                "analytics.oi.updated",
                "analytics.oi.signal",
                "analytics.orderflow.updated",
                "analytics.orderflow.signal",
                "analytics.liquidity.updated",
                "analytics.liquidity.signal",
                "analytics.liquidations.updated",
                "analytics.liquidations.signal",
                "analytics.spreads.updated",
                "analytics.spreads.signal",
                "analytics.spoofing.updated",
                "analytics.spoofing.signal",
                "analytics.whales.updated",
                "analytics.whales.signal",
            ),
        ),
        (
            "Strategy / Signal",
            (
                "strategy.context.updated",
                "strategy.evaluation.completed",
                "strategy.signal.generated",
                "signal.generated",
                "signal.rejected",
                "signal.updated",
            ),
        ),
        (
            "Risk",
            (
                "signal.confirmed",
                "risk.position_blocked",
                "risk.limit_warning",
                "risk.kill_switch",
                "risk.position_close_requested",
                "risk.position_reduce_requested",
            ),
        ),
        (
            "Execution / Position",
            (
                "execution.order_submitted",
                "execution.order_accepted",
                "execution.order_rejected",
                "execution.order_cancelled",
                "execution.order_filled",
                "execution.order_partially_filled",
                "position.opened",
                "position.updated",
                "position.closed",
            ),
        ),
    )

    for title, topics in groups:
        print(f"[{title}]")
        for topic in topics:
            print(f"{topic}: {counts.get(topic, 0)}")
        print("")

    print("Non-zero topics:")
    non_zero = {topic: count for topic, count in sorted(counts.items()) if count}
    if not non_zero:
        print("- none")
    else:
        for topic, count in non_zero.items():
            print(f"- {topic}: {count}")

    print("==================================")


def print_replay_stats(tester: Any) -> None:
    """
    Print MarketReplay stats after run.
    This distinguishes "replay did not emit/process events" from
    "diagnostic subscriptions did not capture events".
    """

    replay = getattr(getattr(tester, "components", None), "market_replay", None)

    print("")
    print("========== MARKET REPLAY STATS ==========")

    if replay is None:
        print("MarketReplay: missing")
        print("=========================================")
        return

    stats = getattr(replay, "stats", None)

    try:
        value = stats() if callable(stats) else getattr(replay, "stats_state", None)
    except Exception as exc:
        print(f"Failed to read MarketReplay stats: {exc}")
        print("=========================================")
        return

    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)

    if isinstance(value, dict):
        for key in (
            "status",
            "total_events",
            "processed_events",
            "emitted_events",
            "skipped_events",
            "failed_events",
            "market_candles",
            "market_trades",
            "market_orderbooks",
            "market_funding",
            "market_open_interest",
            "market_liquidations",
            "current_index",
            "progress_pct",
            "last_error",
        ):
            if key in value:
                print(f"{key}: {value[key]}")
    else:
        print(value)

    print("=========================================")


def summarize_event_pipeline(counts: dict[str, int]) -> dict[str, Any]:
    """
    Produce a compact diagnosis-friendly summary.
    """

    market_total = sum(
        counts.get(topic, 0)
        for topic in (
            "market.candle",
            "market.trade",
            "market.orderbook",
            "market.funding",
            "market.open_interest",
            "market.liquidation",
        )
    )

    cache_total = sum(
        counts.get(topic, 0)
        for topic in (
            "market.candles.updated",
            "market.candle.closed",
            "market.trades.updated",
            "market.orderbook.updated",
            "market.funding.updated",
            "market.open_interest.updated",
            "market.liquidations.updated",
        )
    )

    analytics_total = sum(
        count
        for topic, count in counts.items()
        if topic.startswith("analytics.")
    )

    signal_total = (
        counts.get("signal.generated", 0)
        + counts.get("strategy.signal.generated", 0)
        + counts.get("signal.rejected", 0)
        + counts.get("signal.updated", 0)
    )

    risk_total = (
        counts.get("signal.confirmed", 0)
        + counts.get("risk.position_blocked", 0)
        + counts.get("risk.limit_warning", 0)
        + counts.get("risk.kill_switch", 0)
    )

    execution_total = sum(
        counts.get(topic, 0)
        for topic in (
            "execution.order_submitted",
            "execution.order_accepted",
            "execution.order_rejected",
            "execution.order_cancelled",
            "execution.order_filled",
            "execution.order_partially_filled",
        )
    )

    position_total = sum(
        counts.get(topic, 0)
        for topic in (
            "position.opened",
            "position.updated",
            "position.closed",
        )
    )

    if market_total <= 0:
        likely_breakpoint = "market_replay"
    elif cache_total <= 0:
        likely_breakpoint = "data_caches"
    elif analytics_total <= 0:
        likely_breakpoint = "analytics"
    elif signal_total <= 0:
        likely_breakpoint = "strategy_or_signal_processor"
    elif risk_total <= 0:
        likely_breakpoint = "risk_manager"
    elif execution_total <= 0:
        likely_breakpoint = "execution_simulator"
    elif position_total <= 0:
        likely_breakpoint = "position_simulator"
    else:
        likely_breakpoint = "none"

    return {
        "market_total": market_total,
        "cache_total": cache_total,
        "analytics_total": analytics_total,
        "signal_total": signal_total,
        "risk_total": risk_total,
        "execution_total": execution_total,
        "position_total": position_total,
        "likely_breakpoint": likely_breakpoint,
        "counts": dict(counts),
    }


# =============================================================================
# Diagnostics
# =============================================================================


def print_dataset_summary(config: BacktestConfig, dataset: Any) -> None:
    print("")
    print("========== DATASET ==========")
    print(f"Run:            {config.run_name}")
    print(f"Exchange:       {config.exchange}")
    print(f"Market type:    {config.market_type}")
    print(f"Symbols:        {config.symbols}")
    print(f"Timeframes:     {config.timeframes}")
    print(f"Data dir:       {config.data_loader.data_dir}")
    print(f"Period:         {config.start_time.isoformat()} -> {config.end_time.isoformat()}")
    print(f"Data types cfg: {[item.value for item in sorted(config.data_loader.data_types, key=lambda x: x.value)]}")
    print(f"Events:         {len(dataset.events)}")

    if dataset.info:
        print(f"Data types:     {[item.value for item in sorted(dataset.info.data_types, key=lambda x: x.value)]}")
        print(f"First event:    {dataset.info.first_event_time}")
        print(f"Last event:     {dataset.info.last_event_time}")

    print("=============================")


def print_component_summary(
    *,
    data_caches: list[Any],
    analytics_components: list[Any],
    strategy_registry: Any,
    signal_processor: Any,
    strategy_engine: Any,
    risk_manager: Any,
) -> None:
    strategies = StrategyTester._get_registered_strategies(strategy_registry)

    print("")
    print("========== COMPONENTS ==========")
    print("Data caches:")
    for component in data_caches:
        print(f"- {component.__class__.__module__}.{component.__class__.__name__}")

    print("Analytics:")
    if analytics_components:
        for component in analytics_components:
            print(f"- {component.__class__.__module__}.{component.__class__.__name__}")
    else:
        print("- none")

    print("Strategies:")
    print(f"- registered: {len(strategies)}")
    for strategy in strategies[:50]:
        print(f"  - {StrategyTester._strategy_name(strategy)}")
    if len(strategies) > 50:
        print(f"  ... +{len(strategies) - 50} more")

    print("Processor:")
    print(f"- {signal_processor.__class__.__module__}.{signal_processor.__class__.__name__}")

    print("Engine:")
    print(f"- {strategy_engine.__class__.__module__}.{strategy_engine.__class__.__name__}")

    print("Risk:")
    print(f"- {risk_manager.__class__.__module__}.{risk_manager.__class__.__name__}")

    print("================================")


def print_result_summary(result: Any) -> None:
    print("")
    print("========== BACKTEST RESULT ==========")
    print(f"Run name:       {result.run_name}")
    print(f"Run ID:         {result.run_id}")
    print(f"Status:         {result.status.value}")
    print(f"Initial equity: {result.initial_balance:.2f}")
    print(f"Final equity:   {result.final_equity:.2f}")
    print(f"Final balance:  {result.final_balance:.2f}")

    if result.portfolio and result.portfolio.summary:
        summary = result.portfolio.summary
        print("-------------------------------------")
        print(f"Net profit:     {summary.net_profit:.2f}")
        print(f"Net profit %:   {summary.net_profit_pct:.2f}%")
        print(f"Total trades:   {summary.total_trades}")
        print(f"Win rate:       {summary.win_rate:.2f}%")
        print(f"Max DD %:       {summary.max_drawdown_pct:.2f}%")
        print(f"Profit factor:  {summary.profit_factor:.4f}")

    print("-------------------------------------")
    print(f"Signals:        {len(result.signals)}")
    print(f"Risk decisions: {len(result.risk_decisions)}")
    print(f"Executions:     {len(result.execution_records)}")
    print(f"Positions:      {len(result.position_records)}")
    print(f"Trades:         {len(result.trades)}")

    if result.reports:
        print("-------------------------------------")
        print("Reports:")
        for report in result.reports:
            print(f"- {report.path}")

    if result.warnings:
        print("-------------------------------------")
        print("Warnings:")
        for warning in result.warnings[:30]:
            level = getattr(warning.level, "value", warning.level)
            print(f"- [{level}] {warning.message}")
        if len(result.warnings) > 30:
            print(f"... +{len(result.warnings) - 30} more warnings")

    print("=====================================")


# =============================================================================
# Full run
# =============================================================================


async def run_full_backtest() -> Any:
    runtime = FullBacktestRunConfig.from_env()
    ensure_open_interest_timeframe_compat(runtime)

    app_config = Config.from_env()
    app_config.validate()
    app_config.prepare_directories()

    # Use inline bus for deterministic backtesting. core.EventBus is queued and
    # can return from MarketReplay.emit() before downstream handlers finish.
    event_bus = InlineBacktestEventBus()
    scheduler = Scheduler(event_bus=event_bus)

    started_components: list[Any] = []

    try:
        await start_if_supported(event_bus)
        started_components.append(event_bus)

        await start_if_supported(scheduler)
        started_components.append(scheduler)

        component_scheduler = BacktestSchedulerCompatAdapter(scheduler)

        backtest_config = build_backtest_config(runtime)
        enabled_data_types = backtest_config.enabled_data_types()

        data_loader = SortingDataLoader(backtest_config.data_loader)

        try:
            dataset = data_loader.load_dataset(
                period=backtest_config.period(),
                run_id=backtest_config.run_name,
            )
        except DataLoadError as exc:
            print("")
            print("========== DATA LOADER ERROR ==========")
            print(exc)
            details = getattr(exc, "details", None)
            if details:
                print("Details:")
                for key, value in details.items():
                    print(f"- {key}: {value}")
            print("")
            print("DataLoader stats:")
            print(data_loader.stats())
            print("=======================================")
            raise

        print_dataset_summary(backtest_config, dataset)

        data_caches = build_data_caches(
            app_config=app_config,
            event_bus=event_bus,
            scheduler=component_scheduler,
            enabled_data_types=enabled_data_types,
        )

        analytics_components = build_analytics_components(
            runtime=runtime,
            app_config=app_config,
            event_bus=event_bus,
            scheduler=component_scheduler,
            data_caches=data_caches,
            enabled_data_types=enabled_data_types,
        )

        strategy_registry, signal_processor, strategy_engine = build_strategy_stack(
            runtime=runtime,
            event_bus=event_bus,
            scheduler=component_scheduler,
        )

        risk_manager = build_risk_manager(
            app_config=app_config,
            event_bus=event_bus,
            scheduler=component_scheduler,
        )

        print_component_summary(
            data_caches=data_caches,
            analytics_components=analytics_components,
            strategy_registry=strategy_registry,
            signal_processor=signal_processor,
            strategy_engine=strategy_engine,
            risk_manager=risk_manager,
        )

        debug_event_counts = await register_event_debug_counters(event_bus)

        tester = StrategyTester(
            backtest_config,
            dataset=dataset,
            event_bus=event_bus,
            scheduler=component_scheduler,
            data_caches=data_caches,
            analytics_components=analytics_components,
            strategy_registry=strategy_registry,
            strategy_engine=strategy_engine,
            signal_processor=signal_processor,
            risk_manager=risk_manager,
        )

        result = await tester.run()

        event_pipeline_summary = summarize_event_pipeline(debug_event_counts)
        result.metadata.setdefault("event_debug", event_pipeline_summary)

        print_replay_stats(tester)
        print_event_debug_counts(debug_event_counts)
        print("")
        print("Likely pipeline breakpoint:", event_pipeline_summary["likely_breakpoint"])

        # Persist diagnostics independently from ReportBuilder, because reports
        # may already be built inside StrategyTester before runner metadata is added.
        debug_dir = Path(backtest_config.report_builder.output_dir) / f"{result.run_name}_{result.run_id}"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / "event_debug.json"
        debug_path.write_text(
            __import__("json").dumps(event_pipeline_summary, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Event debug saved: {debug_path}")

        print_result_summary(result)
        return result

    finally:
        for component in reversed(started_components):
            try:
                await stop_if_supported(component)
            except Exception as exc:
                logger.exception(
                    "Failed to stop component",
                    extra={
                        "component": component.__class__.__name__,
                        "error": str(exc),
                    },
                )


if __name__ == "__main__":
    asyncio.run(run_full_backtest())