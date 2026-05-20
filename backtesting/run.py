# backtesting/run.py

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
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
from backtesting.data_loader import DataLoader
from backtesting.enums import (
    BacktestDataType,
    BacktestMode,
    CommissionModel,
    DataGapPolicy,
    DataValidationLevel,
    FillModel,
    HistoricalDataFormat,
    LiquidityModel,
    PnLAccountingMode,
    PositionAccountingMode,
    ReportFormat,
    ReportSection,
    SlippageModel,
)
from backtesting.exceptions import DataLoadError
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
from strategy.engine import StrategyEngine
from strategy.presets import (
    build_default_strategy_config,
    build_default_strategy_registry,
)
from strategy.processor import SignalProcessor
from strategy.state import StrategyRuntimeState


logger = get_logger("backtesting.run")


# =============================================================================
# Paths / defaults
# =============================================================================


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

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


def _file_exists_any(path_without_suffix: Path) -> bool:
    return any(
        path_without_suffix.with_suffix(suffix).exists()
        for suffix in (".csv", ".parquet", ".json", ".jsonl")
    )


def _candidate_history_dirs() -> list[Path]:
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
# Runtime config
# =============================================================================


@dataclass(slots=True)
class FullBacktestRunConfig:
    """
    Runtime settings for full-pipeline backtest runner.

    This runner uses:
    - real core.EventBus;
    - real core.Scheduler wrapped by BacktestSchedulerCompatAdapter;
    - production data caches;
    - production analytics;
    - production StrategyEngine / SignalProcessor;
    - production RiskManager;
    - backtesting ExecutionSimulator and PositionSimulator via StrategyTester.
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
    def from_env(cls) -> FullBacktestRunConfig:
        input_format_raw = os.getenv("BACKTEST_INPUT_FORMAT", "csv").strip().lower()

        try:
            input_format = HistoricalDataFormat(input_format_raw)
        except ValueError:
            raise ValueError(
                "BACKTEST_INPUT_FORMAT must be one of: "
                f"{', '.join(item.value for item in HistoricalDataFormat)}"
            ) from None

        backtest_days = int(os.getenv("BACKTEST_DAYS", str(DEFAULT_BACKTEST_DAYS)).strip())
        end_time = _rolling_end_time()
        start_time = end_time - timedelta(days=backtest_days)

        symbols = _env_list("SYMBOLS", DEFAULT_SYMBOLS)
        timeframes = [item.lower() for item in _env_list("TIMEFRAMES", DEFAULT_TIMEFRAMES)]

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

        data_dir = resolve_history_dir(
            exchange=os.getenv("BACKTEST_EXCHANGE", DEFAULT_EXCHANGE).strip().lower() or DEFAULT_EXCHANGE,
            market_type=os.getenv("BACKTEST_MARKET_TYPE", DEFAULT_MARKET_TYPE).strip().lower() or DEFAULT_MARKET_TYPE,
            symbols=symbols,
            timeframes=timeframes,
        )

        return cls(
            run_name=os.getenv("BACKTEST_RUN_NAME", DEFAULT_RUN_NAME).strip() or DEFAULT_RUN_NAME,
            exchange=os.getenv("BACKTEST_EXCHANGE", DEFAULT_EXCHANGE).strip().lower() or DEFAULT_EXCHANGE,
            market_type=os.getenv("BACKTEST_MARKET_TYPE", DEFAULT_MARKET_TYPE).strip().lower() or DEFAULT_MARKET_TYPE,
            symbols=symbols,
            timeframes=timeframes,
            data_dir=data_dir,
            output_dir=_env_path("BACKTEST_OUTPUT_DIR", "reports/backtests"),
            input_format=input_format,
            initial_balance=float(os.getenv("INITIAL_BALANCE", "10000")),
            backtest_days=backtest_days,
            start_time=start_time,
            end_time=end_time,
            include_candles=_env_bool("INCLUDE_CANDLES", True),
            include_funding=_env_bool("INCLUDE_FUNDING", True),
            include_open_interest=_env_bool("INCLUDE_OPEN_INTEREST", True),
            include_trades=_env_bool("INCLUDE_TRADES", False),
            include_orderbook=_env_bool("INCLUDE_ORDERBOOK", False),
            include_liquidations=_env_bool("INCLUDE_LIQUIDATIONS", False),
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


class BacktestSchedulerCompatAdapter:
    """
    Compatibility wrapper around real core.scheduler.Scheduler.

    Backtesting still uses the real Scheduler instance, but this adapter accepts
    older/different call styles from production components without changing
    production code.

    Supported add_interval_job() call styles:

        add_interval_job(name=..., callback=..., interval_seconds=...)
        add_interval_job(name=..., callback=..., interval=...)
        add_interval_job(name=..., callback=..., seconds=...)
        add_interval_job(name, callback, interval_seconds)
    """

    def __init__(self, scheduler: Scheduler) -> None:
        self._scheduler = scheduler

    @property
    def wrapped(self) -> Scheduler:
        return self._scheduler

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scheduler, name)

    async def start(self) -> Any:
        return await maybe_await(self._scheduler.start())

    async def stop(self) -> Any:
        return await maybe_await(self._scheduler.stop())

    def stats(self) -> dict[str, Any]:
        stats = getattr(self._scheduler, "stats", None)

        if callable(stats):
            value = stats()
            return value if isinstance(value, dict) else {"value": value}

        return {}

    def add_interval_job(self, *args: Any, **kwargs: Any) -> Any:
        name = kwargs.pop("name", None)

        callback = (
                kwargs.pop("callback", None)
                or kwargs.pop("func", None)
                or kwargs.pop("job_func", None)
                or kwargs.pop("job", None)
                or kwargs.pop("coro", None)
                or kwargs.pop("coroutine", None)
                or kwargs.pop("handler", None)
                or kwargs.pop("target", None)
        )

        interval_seconds = kwargs.pop("interval_seconds", None)
        interval = kwargs.pop("interval", None)
        seconds = kwargs.pop("seconds", None)
        every_seconds = kwargs.pop("every_seconds", None)

        run_immediately = kwargs.pop("run_immediately", None)
        enabled = kwargs.pop("enabled", None)
        timeout = kwargs.pop("timeout", None)
        retry_count = kwargs.pop("retry_count", None)
        retry_delay = kwargs.pop("retry_delay", None)
        allow_overlap = kwargs.pop("allow_overlap", None)

        if args:
            if name is None and len(args) >= 1 and isinstance(args[0], str):
                name = args[0]

            if callback is None:
                if len(args) >= 2 and isinstance(args[0], str):
                    callback = args[1]
                elif len(args) >= 1 and callable(args[0]):
                    callback = args[0]

            if (
                    interval_seconds is None
                    and interval is None
                    and seconds is None
                    and every_seconds is None
            ):
                if len(args) >= 3 and isinstance(args[0], str):
                    interval_seconds = args[2]
                elif len(args) >= 2 and callable(args[0]):
                    interval_seconds = args[1]

        if name is None:
            raise TypeError("add_interval_job() missing required argument: name")

        if callback is None:
            raise TypeError("add_interval_job() missing required argument: callback")

        normalized_interval = (
            interval_seconds
            if interval_seconds is not None
            else interval
            if interval is not None
            else seconds
            if seconds is not None
            else every_seconds
        )

        if normalized_interval is None:
            raise TypeError(
                "add_interval_job() missing interval; expected one of "
                "interval_seconds, interval, seconds, every_seconds"
            )

        optional_kwargs = {
            "run_immediately": run_immediately,
            "enabled": enabled,
            "timeout": timeout,
            "retry_count": retry_count,
            "retry_delay": retry_delay,
            "allow_overlap": allow_overlap,
            **kwargs,
        }
        optional_kwargs = {
            key: value
            for key, value in optional_kwargs.items()
            if value is not None
        }

        attempts = [
            lambda: self._scheduler.add_interval_job(
                name=name,
                callback=callback,
                interval=normalized_interval,
                **optional_kwargs,
            ),
            lambda: self._scheduler.add_interval_job(
                name=name,
                callback=callback,
                seconds=normalized_interval,
                **optional_kwargs,
            ),
            lambda: self._scheduler.add_interval_job(
                name=name,
                callback=callback,
                every_seconds=normalized_interval,
                **optional_kwargs,
            ),
            lambda: self._scheduler.add_interval_job(
                name=name,
                callback=callback,
                interval_seconds=normalized_interval,
                **optional_kwargs,
            ),
            lambda: self._scheduler.add_interval_job(
                name,
                callback,
                interval=normalized_interval,
                **optional_kwargs,
            ),
            lambda: self._scheduler.add_interval_job(
                name,
                callback,
                seconds=normalized_interval,
                **optional_kwargs,
            ),
            lambda: self._scheduler.add_interval_job(
                name,
                callback,
                every_seconds=normalized_interval,
                **optional_kwargs,
            ),
            lambda: self._scheduler.add_interval_job(
                name,
                callback,
                interval_seconds=normalized_interval,
                **optional_kwargs,
            ),
            lambda: self._scheduler.add_interval_job(
                name=name,
                callback=callback,
                interval=normalized_interval,
            ),
            lambda: self._scheduler.add_interval_job(
                name=name,
                callback=callback,
                seconds=normalized_interval,
            ),
            lambda: self._scheduler.add_interval_job(
                name=name,
                callback=callback,
                every_seconds=normalized_interval,
            ),
            lambda: self._scheduler.add_interval_job(
                name=name,
                callback=callback,
                interval_seconds=normalized_interval,
            ),
            lambda: self._scheduler.add_interval_job(
                name,
                callback,
                interval=normalized_interval,
            ),
            lambda: self._scheduler.add_interval_job(
                name,
                callback,
                seconds=normalized_interval,
            ),
            lambda: self._scheduler.add_interval_job(
                name,
                callback,
                every_seconds=normalized_interval,
            ),
            lambda: self._scheduler.add_interval_job(
                name,
                callback,
                interval_seconds=normalized_interval,
            ),
        ]

        last_error: Exception | None = None

        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_error = exc
                continue

        raise TypeError(
            "Could not adapt add_interval_job() call to core Scheduler signature. "
            f"name={name!r}, interval={normalized_interval!r}, last_error={last_error}"
        ) from last_error

    def add_delayed_job(self, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(self._scheduler, "add_delayed_job"):
            raise AttributeError("Wrapped Scheduler does not expose add_delayed_job")

        delay_seconds = kwargs.pop("delay_seconds", None)

        if delay_seconds is None:
            return self._scheduler.add_delayed_job(*args, **kwargs)

        attempts = [
            lambda: self._scheduler.add_delayed_job(
                *args,
                delay=delay_seconds,
                **kwargs,
            ),
            lambda: self._scheduler.add_delayed_job(
                *args,
                seconds=delay_seconds,
                **kwargs,
            ),
            lambda: self._scheduler.add_delayed_job(
                *args,
                delay_seconds=delay_seconds,
                **kwargs,
            ),
        ]

        last_error: Exception | None = None

        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_error = exc
                continue

        raise TypeError(
            f"Could not adapt add_delayed_job() call. last_error={last_error}"
        ) from last_error


def _get_attr(component: Any, names: Iterable[str]) -> Any | None:
    for name in names:
        if hasattr(component, name):
            return getattr(component, name)

    return None


def _event_payload(event_or_payload: Any) -> dict[str, Any]:
    if isinstance(event_or_payload, dict):
        return dict(event_or_payload)

    payload = getattr(event_or_payload, "payload", None)

    if isinstance(payload, dict):
        return dict(payload)

    return {}


def _event_topic(event_or_payload: Any, fallback: str = "") -> str:
    value = getattr(event_or_payload, "topic", None)

    if isinstance(value, str) and value:
        return value

    if isinstance(event_or_payload, dict):
        value = event_or_payload.get("topic")
        if isinstance(value, str):
            return value

    return fallback


def _subscribe(
    event_bus: EventBus,
    topic: str,
    handler: Any,
    *,
    name: str,
) -> Any:
    try:
        return event_bus.subscribe(topic, handler, name=name)
    except TypeError:
        return event_bus.subscribe(pattern=topic, handler=handler, name=name)


def _instantiate_flexibly(cls: type, **available_kwargs: Any) -> Any:
    signature = inspect.signature(cls)
    kwargs: dict[str, Any] = {}

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    if accepts_var_kwargs:
        return cls(
            **{
                key: value
                for key, value in available_kwargs.items()
                if value is not None
            }
        )

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

    for name, parameter in signature.parameters.items():
        if name == "self":
            continue

        if name in available_kwargs and available_kwargs[name] is not None:
            kwargs[name] = available_kwargs[name]
            continue

        for alias in aliases.get(name, []):
            if alias in available_kwargs and available_kwargs[alias] is not None:
                kwargs[name] = available_kwargs[alias]
                break

    return cls(**kwargs)


def _try_build_config_from_annotation(annotation: Any) -> Any | None:
    if annotation is inspect.Signature.empty:
        return None

    candidates: list[Any] = []

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
    scheduler: Any,
    **available_kwargs: Any,
) -> Any:
    signature = inspect.signature(cls)
    kwargs: dict[str, Any] = {}

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    if accepts_var_kwargs:
        return cls(
            **{
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

    for name, parameter in signature.parameters.items():
        if name == "self":
            continue

        if name == "config":
            domain_config = _try_build_config_from_annotation(parameter.annotation)
            if domain_config is not None:
                kwargs[name] = domain_config
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

        for alias in aliases.get(name, []):
            if alias in available_kwargs and available_kwargs[alias] is not None:
                kwargs[name] = available_kwargs[alias]
                break

    return cls(**kwargs)


# =============================================================================
# Historical stream detection
# =============================================================================


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
        oi_root = base / "open_interest" / symbol

        if timeframe is not None and _file_exists_any(oi_root / timeframe / f"{symbol}_{timeframe}"):
            return True

        if _file_exists_any(oi_root / symbol):
            return True

        return any(oi_root.glob(f"**/{symbol}*.*"))

    if data_type == BacktestDataType.TRADES:
        return _file_exists_any(base / "trades" / symbol / symbol)

    if data_type in {BacktestDataType.ORDERBOOK, BacktestDataType.ORDERBOOK_SNAPSHOT}:
        return _file_exists_any(base / "orderbook" / symbol / symbol)

    if data_type == BacktestDataType.LIQUIDATIONS:
        return _file_exists_any(base / "liquidations" / symbol / symbol)

    return False


def _copy_file_if_needed(source_base: Path, target_base: Path) -> bool:
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

        source_candidates = [
            symbol_root / "5m" / f"{symbol}_5m",
            symbol_root / symbol,
        ]

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
    config.use_mark_price = False
    config.use_index_price = False

    config.data_loader.data_dir = runtime.data_dir
    config.data_loader.input_format = runtime.input_format
    config.data_loader.exchange = runtime.exchange
    config.data_loader.market_type = runtime.market_type
    config.data_loader.symbols = list(runtime.symbols)
    config.data_loader.timeframes = list(runtime.timeframes)
    config.data_loader.data_types = set(data_types)
    config.data_loader.require_candles = True
    config.data_loader.require_orderbook = False
    config.data_loader.require_trades = False
    config.data_loader.require_funding = False
    config.data_loader.require_open_interest = False
    config.data_loader.allow_empty_optional_streams = True
    config.data_loader.validation_level = DataValidationLevel.BASIC
    config.data_loader.gap_policy = DataGapPolicy.WARN
    config.data_loader.drop_duplicate_events = True

    config.market_replay.emit_market_candles = BacktestDataType.CANDLES in data_types
    config.market_replay.emit_market_trades = BacktestDataType.TRADES in data_types
    config.market_replay.emit_market_orderbook = BacktestDataType.ORDERBOOK in data_types
    config.market_replay.emit_market_funding = BacktestDataType.FUNDING in data_types
    config.market_replay.emit_market_open_interest = BacktestDataType.OPEN_INTEREST in data_types
    config.market_replay.emit_market_liquidations = BacktestDataType.LIQUIDATIONS in data_types
    config.market_replay.emit_replay_lifecycle_events = True

    config.cost_model.slippage_model = SlippageModel.FIXED_BPS
    config.cost_model.fixed_slippage_bps = 2.0
    config.cost_model.commission_model = CommissionModel.MAKER_TAKER
    config.cost_model.maker_fee_bps = 2.0
    config.cost_model.taker_fee_bps = 4.0
    config.cost_model.default_fee_bps = 4.0

    config.execution_simulator.fill_model = FillModel.NEXT_CANDLE_CLOSE
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
    config.execution_simulator.emit_execution_events = True

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
    config.report_builder.report_title = "Full Pipeline Backtest"
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
            "runner": "backtesting.run",
            "event_bus": "core.event_bus.EventBus",
            "scheduler": "core.scheduler.Scheduler+BacktestSchedulerCompatAdapter",
            "detected_data_types": sorted(item.value for item in data_types),
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
    scheduler: Any,
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
    ("analytics.price_action.engine", "PriceActionAnalytics"),
    ("analytics.price_action.analyzer", "PriceActionAnalytics"),
    ("analytics.price_action.market_structure", "MarketStructureAnalytics"),
    ("analytics.price_action.market_structure_analyzer", "MarketStructureAnalyzer"),
    ("analytics.funding.engine", "FundingAnalytics"),
    ("analytics.funding.analyzer", "FundingAnalyzer"),
    ("analytics.funding.analyzer", "FundingAnalytics"),
    ("analytics.open_interest.engine", "OpenInterestAnalytics"),
    ("analytics.open_interest.analyzer", "OIAnalyzer"),
    ("analytics.open_interest.analyzer", "OpenInterestAnalytics"),
    ("analytics.orderflow.analyzer", "OrderFlowAnalyzer"),
    ("analytics.orderflow.engine", "OrderFlowAnalyzer"),
    ("analytics.liquidations.engine", "LiquidationsAnalytics"),
    ("analytics.liquidations.analyzer", "LiquidationsAnalytics"),
    ("analytics.liquidity.engine", "LiquidityAnalytics"),
    ("analytics.liquidity.analyzer", "LiquidityAnalytics"),
    ("analytics.spreads.engine", "SpreadAnalyzer"),
    ("analytics.spreads.analyzer", "SpreadAnalyzer"),
    ("analytics.spoofing.engine", "SpoofingAnalytics"),
    ("analytics.spoofing.analyzer", "SpoofingAnalytics"),
    ("analytics.whales.engine", "WhalesAnalytics"),
    ("analytics.whales.analyzer", "WhalesAnalytics"),
]


def _import_class(spec: str) -> type | None:
    if ":" in spec:
        module_name, class_name = spec.split(":", 1)
    else:
        parts = spec.rsplit(".", 1)

        if len(parts) != 2:
            return None

        module_name, class_name = parts

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        logger.debug(
            "Could not import class module",
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
        "Class not found in module",
        extra={
            "module_name": module_name,
            "class_name": class_name,
        },
    )
    return None


def _looks_like_analytics_class(cls: type) -> bool:
    name = cls.__name__.lower()

    if name.endswith(("config", "state", "snapshot", "record", "event", "result", "error", "exception")):
        return False

    if not any(token in name for token in ("analytics", "analyzer", "detector", "engine")):
        return False

    runtime_methods = {"register", "start", "stop", "handle", "on_event"}
    return any(hasattr(cls, method) for method in runtime_methods)


def _discover_analytics_candidates() -> list[tuple[str, str]]:
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
        leaf = module_name.rsplit(".", 1)[-1].lower()

        if leaf in {"models", "enums", "exceptions", "config", "utils", "__init__"}:
            continue

        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            logger.debug(
                "Analytics auto-discovery skipped module",
                extra={
                    "module_name": module_name,
                    "error": str(exc),
                },
            )
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue

            if _looks_like_analytics_class(obj):
                candidates.append((module_name, obj.__name__))

    unique: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for candidate in candidates:
        if candidate in seen:
            continue

        unique.append(candidate)
        seen.add(candidate)

    return unique


def _analytics_class_requires_missing_streams(
    cls: type,
    *,
    enabled_data_types: set[BacktestDataType],
    cache_by_name: dict[str, Any],
) -> str | None:
    name = cls.__name__.lower()
    module_name = getattr(cls, "__module__", "").lower()
    text = f"{module_name}.{name}"

    needs_trades = (
        "orderflow" in text
        or "cvd" in text
        or "volume_delta" in text
        or "aggressive_trade" in text
    )

    needs_orderbook = (
        "orderflow" in text
        or "orderbook" in text
        or "imbalance" in text
        or "liquidity" in text
        or "spoof" in text
    )

    needs_funding = "funding" in text
    needs_open_interest = "open_interest" in text or ".oi" in text or "oi_" in text
    needs_liquidations = "liquidation" in text
    needs_candles = "price_action" in text or "market_structure" in text or "fvg" in text

    if needs_trades and cache_by_name.get("trades_cache") is None:
        return "requires TradesCache, but trades stream is unavailable"

    if needs_orderbook and cache_by_name.get("orderbook_cache") is None:
        return "requires OrderBookCache, but orderbook stream is unavailable"

    if needs_funding and cache_by_name.get("funding_cache") is None:
        return "requires FundingCache, but funding stream is unavailable"

    if needs_open_interest and cache_by_name.get("open_interest_cache") is None:
        return "requires OpenInterestCache, but open_interest stream is unavailable"

    if needs_candles and cache_by_name.get("candles_cache") is None:
        return "requires CandlesCache, but candles stream is unavailable"

    if needs_liquidations and BacktestDataType.LIQUIDATIONS not in enabled_data_types:
        return "requires liquidations stream, but liquidation history is unavailable"

    return None


def build_analytics_components(
    *,
    runtime: FullBacktestRunConfig,
    app_config: Config,
    event_bus: EventBus,
    scheduler: Any,
    data_caches: list[Any],
    enabled_data_types: set[BacktestDataType],
) -> list[Any]:
    cache_by_name = {
        "candles_cache": next((item for item in data_caches if isinstance(item, CandlesCache)), None),
        "trades_cache": next((item for item in data_caches if isinstance(item, TradesCache)), None),
        "funding_cache": next((item for item in data_caches if isinstance(item, FundingCache)), None),
        "open_interest_cache": next((item for item in data_caches if isinstance(item, OpenInterestCache)), None),
        "orderbook_cache": next((item for item in data_caches if isinstance(item, OrderBookCache)), None),
    }

    specs: list[tuple[str, str]] = []

    for spec in runtime.analytics_specs:
        if ":" in spec:
            module_name, class_name = spec.split(":", 1)
        else:
            module_name, class_name = spec.rsplit(".", 1)
        specs.append((module_name, class_name))

    specs.extend(DEFAULT_ANALYTICS_CANDIDATES)
    specs.extend(_discover_analytics_candidates())

    components: list[Any] = []
    seen_classes: set[type] = set()

    available_kwargs = {
        "data_caches": data_caches,
        **cache_by_name,
        "default_exchange": runtime.exchange,
        "default_market_type": runtime.market_type,
        "default_timeframe": runtime.timeframes[0] if runtime.timeframes else "1m",
    }

    for module_name, class_name in specs:
        cls = _import_class(f"{module_name}:{class_name}")

        if cls is None:
            continue

        if cls in seen_classes:
            continue

        seen_classes.add(cls)

        skip_reason = _analytics_class_requires_missing_streams(
            cls,
            enabled_data_types=enabled_data_types,
            cache_by_name=cache_by_name,
        )

        if skip_reason is not None:
            logger.info(
                "Skipping analytics component because required replay stream is unavailable",
                extra={
                    "analytics_module": module_name,
                    "analytics_class": class_name,
                    "reason": skip_reason,
                    "enabled_data_types": sorted(item.value for item in enabled_data_types),
                },
            )
            continue

        try:
            component = _instantiate_runtime_component(
                cls,
                app_config=app_config,
                event_bus=event_bus,
                scheduler=scheduler,
                **available_kwargs,
            )
        except Exception as exc:
            logger.warning(
                "Analytics component could not be instantiated",
                extra={
                    "analytics_module": module_name,
                    "analytics_class": class_name,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            )
            continue

        components.append(component)

    if runtime.require_analytics and not components:
        raise RuntimeError(
            "No analytics components were instantiated. "
            "Set ANALYTICS_COMPONENTS=analytics.funding.analyzer:FundingAnalyzer,... "
            "or fix analytics constructors."
        )

    return components


def _camel_to_snake(name: str) -> str:
    chars: list[str] = []

    for index, char in enumerate(name):
        if char.isupper() and index > 0 and not name[index - 1].isupper():
            chars.append("_")
        chars.append(char.lower())

    return "".join(chars)


def _strategy_name_from_class(cls: type, module_name: str) -> str:
    name = cls.__name__

    for suffix in ("TradingStrategy", "Strategy"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    snake = _camel_to_snake(name)
    leaf = module_name.rsplit(".", 1)[-1]

    if leaf not in {"base", "utils", "__init__"} and leaf.endswith("_strategy"):
        return leaf[: -len("_strategy")]

    return snake


def _discover_strategy_factories() -> dict[str, type[BaseStrategy]]:
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
from strategy.enums import StrategyCategory


def build_strategy_stack(
    *,
    runtime: FullBacktestRunConfig,
    event_bus: EventBus,
    scheduler: Any,
) -> tuple[Any, SignalProcessor, StrategyEngine]:
    preset_name = runtime.strategy_preset or "intraday"

    strategy_config = build_default_strategy_config(
        symbols=runtime.symbols,
        preset_name=preset_name,
        use_required_features=False,
    )
    strategy_config.routing.event_to_categories.update(
        {
            # Funding analytics
            "analytics.funding.updated": [StrategyCategory.FUNDING],
            "analytics.funding.snapshot": [StrategyCategory.FUNDING],
            "analytics.funding.signal": [StrategyCategory.FUNDING],
            "analytics.funding.regime": [StrategyCategory.FUNDING],
            "analytics.funding.flip": [StrategyCategory.FUNDING],

            # OI analytics aliases used by your analytics package
            "analytics.oi.updated": [StrategyCategory.OPEN_INTEREST],
            "analytics.oi.anomaly": [StrategyCategory.OPEN_INTEREST],
            "analytics.oi.capitulation": [StrategyCategory.OPEN_INTEREST],
            "analytics.oi.regime_changed": [StrategyCategory.OPEN_INTEREST],
            "analytics.oi.squeeze_setup": [StrategyCategory.OPEN_INTEREST],

            # Canonical aliases too
            "analytics.open_interest.updated": [StrategyCategory.OPEN_INTEREST],
            "analytics.open_interest.anomaly": [StrategyCategory.OPEN_INTEREST],
            "analytics.open_interest.capitulation": [StrategyCategory.OPEN_INTEREST],
            "analytics.open_interest.regime_changed": [StrategyCategory.OPEN_INTEREST],
            "analytics.open_interest.squeeze_setup": [StrategyCategory.OPEN_INTEREST],
        }
    )
    if not isinstance(strategy_config, StrategyConfig):
        strategy_config = StrategyConfig()

    if runtime.strategies:
        strategy_config.preset.enabled_strategy_names = list(runtime.strategies)

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
            f"preset={preset_name!r}; "
            f"configured={configured_names}; "
            f"discovered={discovered_names}."
        )

    strategy_state = StrategyRuntimeState()

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
    scheduler: Any,
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
# Diagnostics
# =============================================================================


DEBUG_TOPICS: tuple[str, ...] = (
    "market.*",
    "analytics.*",
    "strategy.*",
    "strategy.engine.*",
    "strategy.registry.*",
    "system.strategy.*",
    "system.strategy_engine.*",
    "system.signal_processor.*",
    "signal.*",
    "risk.*",
    "execution.*",
    "position.*",
    "system.backtest.*",
    "system.market_replay.*",
    "system.*",
)


async def register_event_debug_counters(event_bus: EventBus) -> dict[str, int]:
    counts: dict[str, int] = {}

    for pattern in DEBUG_TOPICS:
        async def handler(event: Any, *, pattern: str = pattern) -> None:
            topic = _event_topic(event, fallback=pattern)
            counts[topic] = counts.get(topic, 0) + 1

        _subscribe(
            event_bus,
            pattern,
            handler,
            name=f"backtest_debug_{pattern.replace('*', 'wildcard').replace('.', '_')}",
        )

    return counts


def count_prefix(counts: dict[str, int], prefix: str) -> int:
    return sum(value for topic, value in counts.items() if topic.startswith(prefix))


def summarize_event_pipeline(counts: dict[str, int]) -> dict[str, Any]:
    summary = {
        "market_raw": count_prefix(counts, "market."),
        "market_updated": sum(
            value
            for topic, value in counts.items()
            if topic.startswith("market.") and topic.endswith(".updated")
        ),
        "analytics": count_prefix(counts, "analytics."),
        "strategy": count_prefix(counts, "strategy."),
        "signals": count_prefix(counts, "signal."),
        "risk": count_prefix(counts, "risk."),
        "execution": count_prefix(counts, "execution."),
        "position": count_prefix(counts, "position."),
        "topics": dict(sorted(counts.items())),
    }

    if summary["market_raw"] <= 0:
        breakpoint = "MarketReplay did not emit market.* events"
    elif summary["market_updated"] <= 0:
        breakpoint = "Data caches did not emit market.*.updated events"
    elif summary["analytics"] <= 0:
        breakpoint = "Analytics did not emit analytics.* events"
    elif summary["signals"] <= 0:
        breakpoint = "Strategy did not emit signal.* events"
    elif summary["risk"] <= 0:
        breakpoint = "RiskManager did not emit risk/signal.confirmed events"
    elif summary["execution"] <= 0:
        breakpoint = "ExecutionSimulator did not emit execution.* events"
    elif summary["position"] <= 0:
        breakpoint = "PositionSimulator did not emit position.* events"
    else:
        breakpoint = "No obvious breakpoint"

    summary["likely_breakpoint"] = breakpoint
    return summary


def print_dataset_summary(config: BacktestConfig, dataset: Any) -> None:
    print("")
    print("========== DATASET ==========")
    print(f"Run:         {config.run_name}")
    print(f"Events:      {len(dataset.events)}")
    print(f"Period:      {config.start_time} -> {config.end_time}")
    print(f"Symbols:     {config.symbols}")
    print(f"Timeframes:  {config.timeframes}")
    print(f"Data types:  {sorted(item.value for item in config.enabled_data_types())}")
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
    print("")
    print("========== COMPONENTS ==========")
    print("Data caches:")
    for item in data_caches:
        print(f"- {item.__class__.__name__}")

    print("Analytics:")
    for item in analytics_components:
        print(f"- {item.__class__.__name__}")

    print(f"Strategy registry: {strategy_registry.__class__.__name__}")
    print(f"Signal processor:  {signal_processor.__class__.__name__}")
    print(f"Strategy engine:   {strategy_engine.__class__.__name__}")
    print(f"Risk manager:      {risk_manager.__class__.__name__}")
    print("================================")


def print_replay_stats(tester: StrategyTester) -> None:
    components = tester.components

    print("")
    print("========== PIPELINE STATS ==========")

    for name, component in (
        ("market_replay", components.market_replay),
        ("execution_simulator", components.execution_simulator),
        ("position_simulator", components.position_simulator),
        ("strategy_tester", tester),
    ):
        if component is None:
            continue

        print(f"[{name}]")
        stats = stats_if_supported(component)
        for key, value in stats.items():
            if isinstance(value, (dict, list)):
                continue
            print(f"  {key}: {value}")

    print("====================================")


def print_event_debug_counts(counts: dict[str, int]) -> None:
    print("")
    print("========== EVENT DEBUG ==========")

    for topic, count in sorted(counts.items()):
        print(f"{topic}: {count}")

    print("=================================")


def print_result_summary(result: Any) -> None:
    print("")
    print("========== BACKTEST RESULT ==========")
    print(f"Run:            {result.run_name}")
    print(f"Status:         {result.status.value}")
    print(f"Initial balance:{result.initial_balance}")
    print(f"Final balance:  {result.final_balance}")
    print(f"Final equity:   {result.final_equity}")
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

    event_bus = EventBus()
    core_scheduler = Scheduler(event_bus=event_bus)
    scheduler = BacktestSchedulerCompatAdapter(core_scheduler)

    started_components: list[Any] = []

    try:
        await start_if_supported(event_bus)
        started_components.append(event_bus)

        await start_if_supported(scheduler)
        started_components.append(scheduler)

        backtest_config = build_backtest_config(runtime)
        enabled_data_types = backtest_config.enabled_data_types()

        data_loader = DataLoader(backtest_config.data_loader)

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
            scheduler=scheduler,
            enabled_data_types=enabled_data_types,
        )

        analytics_components = build_analytics_components(
            runtime=runtime,
            app_config=app_config,
            event_bus=event_bus,
            scheduler=scheduler,
            data_caches=data_caches,
            enabled_data_types=enabled_data_types,
        )

        strategy_registry, signal_processor, strategy_engine = build_strategy_stack(
            runtime=runtime,
            event_bus=event_bus,
            scheduler=scheduler,
        )

        risk_manager = build_risk_manager(
            app_config=app_config,
            event_bus=event_bus,
            scheduler=scheduler,
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
            scheduler=scheduler,
            data_caches=data_caches,
            analytics_components=analytics_components,
            strategy_registry=strategy_registry,
            strategy_engine=strategy_engine,
            signal_processor=signal_processor,
            risk_manager=risk_manager,
        )

        result = await tester.run()
        print("")
        print("========== STRATEGY DIAGNOSTICS ==========")
        for name, component in (
                ("strategy_engine", strategy_engine),
                ("signal_processor", signal_processor),
                ("strategy_registry", strategy_registry),
        ):
            print(f"[{name}]")
            stats = stats_if_supported(component)
            if not stats:
                print("  no stats")
            for key, value in stats.items():
                if isinstance(value, (dict, list)):
                    print(f"  {key}: {json.dumps(value, default=str)[:1000]}")
                else:
                    print(f"  {key}: {value}")
        print("==========================================")
        market_replay_stats = (
            tester.components.market_replay.stats()
            if tester.components.market_replay is not None
            else {}
        )
        print("")
        print("========== STRATEGY REGISTRY ==========")
        try:
            strategies = strategy_registry.list_all()
        except Exception:
            strategies = []

        for item in strategies:
            name = getattr(item, "name", None) or getattr(item, "strategy_name", None) or str(item)
            category = getattr(item, "category", None)
            enabled = getattr(item, "enabled", None)
            print(f"- {name} | category={category} | enabled={enabled}")

        print(f"Total strategies: {len(strategies)}")
        print("=======================================")
        print("")
        print("========== MARKET REPLAY HARD STATS ==========")
        for key in (
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
                "status",
        ):
            print(f"{key}: {market_replay_stats.get(key)}")
        print("==============================================")
        event_pipeline_summary = summarize_event_pipeline(debug_event_counts)
        result.metadata.setdefault("event_debug", event_pipeline_summary)

        print_replay_stats(tester)
        print_event_debug_counts(debug_event_counts)
        print("")
        print("Likely pipeline breakpoint:", event_pipeline_summary["likely_breakpoint"])

        debug_dir = Path(backtest_config.report_builder.output_dir) / f"{result.run_name}_{result.run_id}"
        debug_dir.mkdir(parents=True, exist_ok=True)

        debug_path = debug_dir / "event_debug.json"
        debug_path.write_text(
            json.dumps(event_pipeline_summary, indent=2, default=str),
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
                        "component_name": component.__class__.__name__,
                        "error": str(exc),
                    },
                )


if __name__ == "__main__":
    asyncio.run(run_full_backtest())