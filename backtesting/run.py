# backtesting/run.py

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.config import Config
from core.event_bus import EventBus
from core.logger import get_logger
from core.scheduler import Scheduler

from backtesting.bootstrap import (
    BacktestProjectBootstrap,
    BacktestProjectBootstrapConfig,
)
from backtesting.config import BacktestConfig
from backtesting.data_loader import DataLoader
from backtesting.enums import BacktestDataType, BacktestMode, DataGapPolicy, DataValidationLevel
from backtesting.models import BacktestDataset, BacktestResult
from backtesting.strategy_tester import StrategyTester


logger = get_logger("backtesting.run")


# =============================================================================
# Small helpers
# =============================================================================


def _getattr_or(obj: Any, name: str, default: Any) -> Any:
    value = getattr(obj, name, default)
    return default if value is None else value


def _bool_attr(obj: Any, primary: str, fallback: str, default: bool) -> bool:
    """
    Read runtime stream flags from either BacktestProjectBootstrapConfig names
    (enable_*) or older runner-style names (use_* / include_*).
    """

    if hasattr(obj, primary):
        return bool(getattr(obj, primary))

    if hasattr(obj, fallback):
        return bool(getattr(obj, fallback))

    include_name = fallback.replace("use_", "include_", 1)
    if hasattr(obj, include_name):
        return bool(getattr(obj, include_name))

    return default


def _as_path(value: Any, default: str | Path) -> Path:
    return Path(value if value is not None else default).expanduser()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw.strip())


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


def _parse_datetime(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    parsed = datetime.fromisoformat(normalized)

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def _floor_to_minute(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(second=0, microsecond=0)


def _enabled_data_types_from_runtime(
    runtime: BacktestProjectBootstrapConfig,
) -> set[BacktestDataType]:
    data_types: set[BacktestDataType] = set()

    if _bool_attr(runtime, "enable_candles", "use_candles", True):
        data_types.add(BacktestDataType.CANDLES)

    if _bool_attr(runtime, "enable_funding", "use_funding", True):
        data_types.add(BacktestDataType.FUNDING)

    if _bool_attr(runtime, "enable_open_interest", "use_open_interest", True):
        data_types.add(BacktestDataType.OPEN_INTEREST)

    if _bool_attr(runtime, "enable_trades", "use_trades", False):
        data_types.add(BacktestDataType.TRADES)

    if _bool_attr(runtime, "enable_orderbook", "use_orderbook", False):
        data_types.add(BacktestDataType.ORDERBOOK)

    if _bool_attr(runtime, "enable_liquidations", "use_liquidations", False):
        data_types.add(BacktestDataType.LIQUIDATIONS)

    return data_types


# =============================================================================
# Shared system runtime
# =============================================================================


def _construct_event_bus(core_config: Config) -> EventBus:
    """
    Construct the single shared system EventBus used by the entire backtest.

    This is intentionally done in the entrypoint/runtime layer, not inside
    BacktestProjectBootstrap. Bootstrap receives this object and wires all
    components to it.
    """

    event_bus_config = getattr(core_config, "event_bus", None)

    for args, kwargs in (
        ((), {"config": event_bus_config}),
        ((), {"event_bus_config": event_bus_config}),
        ((event_bus_config,), {}),
        ((), {}),
    ):
        try:
            return EventBus(*args, **kwargs)
        except TypeError:
            continue

    raise RuntimeError("Unable to construct shared system EventBus.")


def _construct_scheduler(core_config: Config, event_bus: EventBus) -> Scheduler:
    """
    Construct the single shared system Scheduler used by the entire backtest.

    The same scheduler instance is passed through bootstrap and StrategyTester,
    so scheduled jobs from data/analytics/strategy/risk all run on one runtime.
    """

    scheduler_config = getattr(core_config, "scheduler", None)

    for args, kwargs in (
        ((), {"config": scheduler_config, "event_bus": event_bus}),
        ((), {"scheduler_config": scheduler_config, "event_bus": event_bus}),
        ((), {"event_bus": event_bus, "config": scheduler_config}),
        ((), {"event_bus": event_bus}),
        ((scheduler_config,), {}),
        ((), {}),
    ):
        try:
            return Scheduler(*args, **kwargs)
        except TypeError:
            continue

    raise RuntimeError("Unable to construct shared system Scheduler.")


def build_shared_system_runtime(core_config: Config) -> tuple[EventBus, Scheduler]:
    """
    Build one shared system EventBus/Scheduler pair for the full backtest flow.

    Flow:
        shared EventBus/Scheduler
        -> BacktestProjectBootstrap
        -> data caches / analytics / strategy / risk
        -> StrategyTester / MarketReplay / simulators

    Do not create EventBus/Scheduler inside bootstrap; it must only receive
    these shared instances.
    """

    event_bus = _construct_event_bus(core_config)
    scheduler = _construct_scheduler(core_config, event_bus)
    return event_bus, scheduler


# =============================================================================
# Backtest period / config
# =============================================================================


def resolve_backtest_period(
    runtime: BacktestProjectBootstrapConfig,
) -> tuple[datetime, datetime, datetime | None]:
    """
    Resolve the historical period.

    Supported env:
    - BACKTEST_START_TIME=2026-05-18T19:00:00+00:00
    - BACKTEST_END_TIME=2026-05-20T19:00:00+00:00
    - BACKTEST_DAYS=2
    - BACKTEST_WARMUP_DAYS=0
    """

    env_start = os.getenv("BACKTEST_START_TIME")
    env_end = os.getenv("BACKTEST_END_TIME")

    start_time = (
        _parse_datetime(env_start)
        if env_start and env_start.strip()
        else _getattr_or(runtime, "start_time", None)
    )
    end_time = (
        _parse_datetime(env_end)
        if env_end and env_end.strip()
        else _getattr_or(runtime, "end_time", None)
    )

    if end_time is None:
        end_time = _floor_to_minute(datetime.now(tz=timezone.utc))
    else:
        end_time = _floor_to_minute(end_time)

    if start_time is None:
        days = _env_int("BACKTEST_DAYS", 2)
        start_time = end_time - timedelta(days=days)
    else:
        start_time = _floor_to_minute(start_time)

    warmup_start_time = _getattr_or(runtime, "warmup_start_time", None)

    env_warmup_days = os.getenv("BACKTEST_WARMUP_DAYS")
    if env_warmup_days is not None and env_warmup_days.strip():
        warmup_days = int(env_warmup_days.strip())
        warmup_start_time = start_time - timedelta(days=warmup_days) if warmup_days > 0 else None

    return start_time, end_time, warmup_start_time


def build_backtest_config(
    runtime: BacktestProjectBootstrapConfig,
    *,
    core_config: Config,
) -> BacktestConfig:
    """
    Build a fully validated BacktestConfig from runtime/bootstrap settings.

    BacktestConfig.default_binance_futures(...) is used first because plain
    BacktestConfig().validate() requires symbols and start/end time to be set.
    """

    symbols = list(_getattr_or(runtime, "symbols", []))
    timeframes = list(_getattr_or(runtime, "timeframes", ["1m"]))
    start_time, end_time, warmup_start_time = resolve_backtest_period(runtime)

    if not symbols:
        raise ValueError(
            "BacktestProjectBootstrapConfig.symbols is empty. "
            "Set BACKTEST_SYMBOLS, for example: BTCUSDT,DOGEUSDT,SOLUSDT."
        )

    run_name = str(
        _getattr_or(
            runtime,
            "run_name",
            os.getenv("BACKTEST_RUN_NAME", "btc_doge_sol_last_2d_full_pipeline"),
        )
    )
    initial_balance = float(
        _getattr_or(
            runtime,
            "initial_balance",
            _env_float("BACKTEST_INITIAL_BALANCE", 10_000.0),
        )
    )

    config = BacktestConfig.default_binance_futures(
        run_name=run_name,
        symbols=symbols,
        timeframes=timeframes,
        start_time=start_time,
        end_time=end_time,
        initial_balance=initial_balance,
    )

    data_types = _enabled_data_types_from_runtime(runtime)

    config.mode = BacktestMode.MULTI_STRATEGY
    config.exchange = str(_getattr_or(runtime, "exchange", "binance")).lower()
    config.market_type = str(_getattr_or(runtime, "market_type", "usdm_futures")).lower()
    config.warmup_start_time = warmup_start_time

    config.data_dir = _as_path(
        _getattr_or(runtime, "data_dir", os.getenv("BACKTEST_DATA_DIR", "data/history")),
        "data/history",
    )
    config.output_dir = _as_path(
        _getattr_or(runtime, "output_dir", os.getenv("BACKTEST_OUTPUT_DIR", "reports/backtests")),
        "reports/backtests",
    )

    config.use_candles = BacktestDataType.CANDLES in data_types
    config.use_funding = BacktestDataType.FUNDING in data_types
    config.use_open_interest = BacktestDataType.OPEN_INTEREST in data_types
    config.use_trades = BacktestDataType.TRADES in data_types
    config.use_orderbook = BacktestDataType.ORDERBOOK in data_types
    config.use_liquidations = BacktestDataType.LIQUIDATIONS in data_types
    config.use_mark_price = False
    config.use_index_price = False

    # Loader. DataLoaderConfig has no require_liquidations field; liquidations
    # are controlled only via data_types and allow_empty_optional_streams.
    config.data_loader.data_dir = config.data_dir
    config.data_loader.exchange = config.exchange
    config.data_loader.market_type = config.market_type
    config.data_loader.symbols = list(config.symbols)
    config.data_loader.timeframes = list(config.timeframes)
    config.data_loader.data_types = set(data_types)
    config.data_loader.require_candles = config.use_candles
    config.data_loader.require_funding = False
    config.data_loader.require_open_interest = False
    config.data_loader.require_trades = False
    config.data_loader.require_orderbook = False
    config.data_loader.allow_empty_optional_streams = True
    config.data_loader.validation_level = DataValidationLevel.BASIC
    config.data_loader.gap_policy = DataGapPolicy.WARN
    config.data_loader.drop_duplicate_events = True

    # Keep production config snapshot available to post-run artifacts and
    # simulator/report components that need it.
    config.core_config = core_config

    config.validate()
    return config


def load_dataset(config: BacktestConfig) -> BacktestDataset:
    loader = DataLoader(config.data_loader)
    return loader.load_dataset(
        period=config.period(),
        run_id=config.run_name,
    )


# =============================================================================
# Output helpers
# =============================================================================


def print_dataset_summary(dataset: BacktestDataset, config: BacktestConfig) -> None:
    info = dataset.info
    data_types = sorted(item.value for item in info.data_types)

    print("\n========== DATASET ==========")
    print(f"Run:         {config.run_name}")
    print(f"Events:      {len(dataset.events)}")
    print(f"Period:      {info.period.start} -> {info.period.end}")
    print(f"Symbols:     {config.symbols}")
    print(f"Timeframes:  {config.timeframes}")
    print(f"Data types:  {data_types}")
    print("=============================\n")


def print_components_summary(pipeline: Any) -> None:
    strategy_pipeline = pipeline.strategy_pipeline

    print("\n========== COMPONENTS ==========")
    print("Data caches:")
    for component in pipeline.data_caches:
        print(f"- {component.__class__.__name__}")

    print("Analytics:")
    for component in pipeline.analytics_components:
        print(f"- {component.__class__.__name__}")

    print(f"Strategy registry: {strategy_pipeline.registry.__class__.__name__}")
    print(f"Signal processor:  {strategy_pipeline.signal_processor.__class__.__name__}")
    print(f"Strategy engine:   {strategy_pipeline.strategy_engine.__class__.__name__}")
    print(f"Risk manager:      {pipeline.risk_manager.__class__.__name__}")
    print("================================\n")

    diagnostics = getattr(pipeline, "diagnostics", None)
    if diagnostics is not None:
        format_method = getattr(diagnostics, "format", None)
        if callable(format_method):
            text = format_method()
            if text:
                print(text)


def print_result_summary(result: BacktestResult) -> None:
    summary = result.portfolio.summary

    print("\n========== RESULT ==========")
    print(f"Run:      {result.run_id}")
    print(f"Status:   {result.status.value}")
    print(f"initial_balance: {summary.initial_balance}")
    print(f"final_balance: {summary.final_balance}")
    print(f"final_equity: {summary.final_equity}")
    print(f"net_profit: {summary.net_profit}")
    print(f"net_profit_pct: {summary.net_profit_pct}")
    print(f"max_drawdown_pct: {summary.max_drawdown_pct}")
    print(f"total_trades: {summary.total_trades}")
    print(f"win_rate: {summary.win_rate}")
    print(f"Signals:  {len(result.signals)}")
    print(f"Orders:   {len(result.orders)}")
    print(f"Fills:    {len(result.fills)}")
    print(f"Reports:  {len(result.reports)}")
    print(f"Artifacts:{len(result.artifacts)}")
    print("============================\n")


def _safe_public_value(value: Any) -> Any:
    """
    Convert a debug value into something printable and reasonably compact.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): _safe_public_value(item)
            for key, item in list(value.items())[:80]
        }

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        return [_safe_public_value(item) for item in items[:80]]

    if hasattr(value, "value"):
        try:
            return value.value
        except (AttributeError, TypeError):
            pass

    if hasattr(value, "to_dict"):
        try:
            return _safe_public_value(value.to_dict())
        except (TypeError, ValueError, AttributeError, RuntimeError):
            pass

    if hasattr(value, "__dict__"):
        try:
            return {
                key: _safe_public_value(item)
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        except (TypeError, ValueError):
            pass

    return str(value)


def _call_noarg(obj: Any, name: str) -> Any:
    method = getattr(obj, name, None)
    if callable(method):
        try:
            return method()
        except (TypeError, ValueError, AttributeError, RuntimeError) as exc:
            return f"<{name} failed: {exc}>"
    return None


def _read_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except (TypeError, ValueError, AttributeError, RuntimeError):
        return default


def _print_mapping(title: str, mapping: dict[str, Any]) -> None:
    print(title)
    if not mapping:
        print("- <empty>")
        return

    for key, value in mapping.items():
        print(f"- {key}: {_safe_public_value(value)}")


def _debug_strategy_config(
    strategy_config: Any,
    *,
    observed_analytics_topics: list[str] | None = None,
) -> None:
    """
    Read-only StrategyConfig diagnostics.

    Important:
    run.py must not mutate strategy routing. Routing is production-level config
    and must live in strategy/config.py, strategy presets, or the production
    strategy bootstrap. This function only prints what is configured and compares
    it with analytics topics that were actually observed during this backtest.
    """

    print("\nStrategyConfig:")

    observed_analytics_topics = observed_analytics_topics or []

    routing = _read_attr(strategy_config, "routing")
    if routing is None:
        print("- routing: <missing>")
    else:
        event_to_categories = _read_attr(routing, "event_to_categories", {})
        if isinstance(event_to_categories, dict):
            configured_topics = list(event_to_categories.keys())
            configured_set = set(str(topic) for topic in configured_topics)

            print(f"- routing.event_to_categories.count: {len(configured_topics)}")
            print(f"- routing.event_to_categories.first_50: {configured_topics[:50]}")

            unmatched_observed = [
                topic
                for topic in observed_analytics_topics
                if topic not in configured_set and "analytics.*" not in configured_set
            ]

            print(f"- observed_analytics_topics.count: {len(observed_analytics_topics)}")
            print(f"- observed_analytics_topics.first_50: {observed_analytics_topics[:50]}")
            print(f"- observed_analytics_not_in_routing.count: {len(unmatched_observed)}")
            print(f"- observed_analytics_not_in_routing.first_50: {unmatched_observed[:50]}")
        else:
            print(f"- routing.event_to_categories: {_safe_public_value(event_to_categories)}")

    enabled = _read_attr(strategy_config, "enabled", None)
    if enabled is not None:
        print(f"- enabled: {enabled}")

    min_confidence = _read_attr(strategy_config, "min_confidence", None)
    if min_confidence is not None:
        print(f"- min_confidence: {min_confidence}")

    default_symbols = _read_attr(strategy_config, "symbols", None)
    if default_symbols is not None:
        print(f"- symbols: {_safe_public_value(default_symbols)}")


def _debug_strategy_registry(registry: Any) -> None:
    print("\nStrategyRegistry:")

    count_value = _call_noarg(registry, "count")
    if count_value is None:
        strategies_attr = _read_attr(registry, "strategies", None)
        if isinstance(strategies_attr, dict):
            count_value = len(strategies_attr)
        elif isinstance(strategies_attr, list):
            count_value = len(strategies_attr)
    print(f"- count: {count_value if count_value is not None else '<unknown>'}")

    names = (
        _call_noarg(registry, "list_names")
        or _call_noarg(registry, "names")
        or _call_noarg(registry, "strategy_names")
    )
    if names is not None:
        print(f"- names: {_safe_public_value(names)}")

    categories = (
        _call_noarg(registry, "list_categories")
        or _call_noarg(registry, "categories")
    )
    if categories is not None:
        print(f"- categories: {_safe_public_value(categories)}")

    by_category = _read_attr(registry, "_by_category", None)
    if isinstance(by_category, dict):
        print("- _by_category:")
        for category, values in by_category.items():
            try:
                size = len(values)
            except TypeError:
                size = "<unknown>"
            print(f"  - {category}: {size}")

    raw_strategies = _read_attr(registry, "_strategies", None)
    if isinstance(raw_strategies, dict):
        print(f"- _strategies.count: {len(raw_strategies)}")
        print(f"- _strategies.first_30: {list(raw_strategies.keys())[:30]}")

    select_methods = [
        "select_for_event",
        "select_for_payload",
        "select_for_context",
        "get_by_category",
        "find_by_category",
    ]
    available_selectors = [name for name in select_methods if callable(getattr(registry, name, None))]
    print(f"- available_selectors: {available_selectors}")


def _debug_strategy_engine(engine: Any) -> None:
    print("\nStrategyEngine:")

    stats_candidates = [
        _read_attr(engine, "stats", None),
        _read_attr(engine, "stats_state", None),
        _read_attr(engine, "_stats", None),
    ]
    for stats in stats_candidates:
        if stats is None:
            continue

        summary = _call_noarg(stats, "summary")
        snapshot = _call_noarg(stats, "snapshot")
        to_dict = _call_noarg(stats, "to_dict")

        if summary is not None:
            print(f"- stats.summary: {_safe_public_value(summary)}")
            break
        if snapshot is not None:
            print(f"- stats.snapshot: {_safe_public_value(snapshot)}")
            break
        if to_dict is not None:
            print(f"- stats.to_dict: {_safe_public_value(to_dict)}")
            break

        print(f"- stats: {_safe_public_value(stats)}")
        break
    else:
        print("- stats: <missing>")

    for attr in (
        "status",
        "is_running",
        "started",
        "registered",
        "_registered",
        "_started",
        "_running",
        "events_received",
        "events_processed",
        "events_failed",
        "signals_generated",
        "signals_rejected",
    ):
        value = _read_attr(engine, attr, None)
        if value is not None:
            print(f"- {attr}: {_safe_public_value(value)}")

    subscriptions = _read_attr(engine, "_subscriptions", None)
    if subscriptions is not None:
        try:
            print(f"- _subscriptions.count: {len(subscriptions)}")
        except TypeError:
            print(f"- _subscriptions: {_safe_public_value(subscriptions)}")

    event_handler = _read_attr(engine, "event_handler", None) or _read_attr(engine, "_event_handler", None)
    if event_handler is not None:
        print("- event_handler:")
        for attr in ("_subscriptions", "subscriptions", "_registered", "registered"):
            value = _read_attr(event_handler, attr, None)
            if value is not None:
                try:
                    print(f"  - {attr}.count: {len(value)}")
                    if isinstance(value, (list, tuple, set)):
                        print(f"  - {attr}.sample: {_safe_public_value(list(value)[:10])}")
                except TypeError:
                    print(f"  - {attr}: {_safe_public_value(value)}")

        analytics_topics = _call_noarg(event_handler, "_analytics_topics")
        if analytics_topics is None:
            analytics_topics = _call_noarg(event_handler, "analytics_topics")
        if analytics_topics is not None:
            print(f"  - analytics_topics.count: {len(analytics_topics) if hasattr(analytics_topics, '__len__') else '<unknown>'}")
            print(f"  - analytics_topics.first_50: {_safe_public_value(list(analytics_topics)[:50]) if not isinstance(analytics_topics, str) else analytics_topics}")


def _debug_signal_processor(processor: Any) -> None:
    print("\nSignalProcessor:")

    stats_candidates = [
        _read_attr(processor, "stats", None),
        _read_attr(processor, "stats_state", None),
        _read_attr(processor, "_stats", None),
    ]
    for stats in stats_candidates:
        if stats is None:
            continue

        summary = _call_noarg(stats, "summary")
        snapshot = _call_noarg(stats, "snapshot")
        to_dict = _call_noarg(stats, "to_dict")

        if summary is not None:
            print(f"- stats.summary: {_safe_public_value(summary)}")
            break
        if snapshot is not None:
            print(f"- stats.snapshot: {_safe_public_value(snapshot)}")
            break
        if to_dict is not None:
            print(f"- stats.to_dict: {_safe_public_value(to_dict)}")
            break

        print(f"- stats: {_safe_public_value(stats)}")
        break
    else:
        print("- stats: <missing>")

    for attr in (
        "events_received",
        "events_processed",
        "events_rejected",
        "signals_built",
        "signals_filtered",
        "signals_emitted",
        "batches_processed",
        "last_error",
    ):
        value = _read_attr(processor, attr, None)
        if value is not None:
            print(f"- {attr}: {_safe_public_value(value)}")

    pipeline_parts = {
        "normalizer": _read_attr(processor, "normalizer", None) or _read_attr(processor, "_normalizer", None),
        "router": _read_attr(processor, "router", None) or _read_attr(processor, "_router", None),
        "confluence_engine": _read_attr(processor, "confluence_engine", None) or _read_attr(processor, "_confluence_engine", None),
        "portfolio_coordinator": _read_attr(processor, "portfolio_coordinator", None) or _read_attr(processor, "_portfolio_coordinator", None),
        "scorer": _read_attr(processor, "scorer", None) or _read_attr(processor, "_scorer", None),
        "filters": _read_attr(processor, "filters", None) or _read_attr(processor, "filter_chain", None) or _read_attr(processor, "_filter_chain", None),
        "builder": _read_attr(processor, "builder", None) or _read_attr(processor, "_builder", None),
    }
    print("- pipeline_parts:")
    for name, part in pipeline_parts.items():
        print(f"  - {name}: {part.__class__.__name__ if part is not None else '<missing>'}")


def _debug_runtime_state(runtime_state: Any) -> None:
    print("\nStrategyRuntimeState:")

    if runtime_state is None:
        print("- <missing>")
        return

    snapshot = _call_noarg(runtime_state, "snapshot")
    to_dict = _call_noarg(runtime_state, "to_dict")
    stats = _call_noarg(runtime_state, "stats")

    if snapshot is not None:
        print(f"- snapshot: {_safe_public_value(snapshot)}")
    elif to_dict is not None:
        print(f"- to_dict: {_safe_public_value(to_dict)}")
    elif stats is not None:
        print(f"- stats: {_safe_public_value(stats)}")
    else:
        print(f"- value: {_safe_public_value(runtime_state)}")

    for attr in ("signal_state", "context_store", "cooldown_state", "metrics_state"):
        value = _read_attr(runtime_state, attr, None)
        if value is not None:
            compact = _call_noarg(value, "summary") or _call_noarg(value, "snapshot") or _call_noarg(value, "to_dict") or value
            print(f"- {attr}: {_safe_public_value(compact)}")


def print_strategy_internal_debug(pipeline: Any, result: BacktestResult | None = None) -> None:
    """
    Print internal strategy-layer diagnostics after backtest completion.

    This answers:
    - Did StrategyEventHandler subscribe to real analytics topics?
    - Did StrategyEngine receive/process events?
    - Is StrategyRegistry populated?
    - Did SignalProcessor build/filter/emit anything?
    """

    enabled_raw = os.getenv("BACKTEST_DEBUG_STRATEGY", "1").strip().lower()
    if enabled_raw in {"0", "false", "no", "off"}:
        return

    strategy_pipeline = pipeline.strategy_pipeline

    print("\n========== STRATEGY INTERNAL DEBUG ==========")

    observed_analytics_topics: list[str] = []

    if result is not None:
        debug_payload = result.metadata.get("event_flow_debug") if isinstance(result.metadata, dict) else None
        if isinstance(debug_payload, dict):
            print("EventFlow summary:")
            print(f"- last_stage: {debug_payload.get('last_stage')}")
            print(f"- suspected_breakpoint: {debug_payload.get('suspected_breakpoint')}")
            print(f"- group_counts: {_safe_public_value(debug_payload.get('group_counts'))}")

            topic_counts = debug_payload.get("topic_counts")
            if isinstance(topic_counts, dict):
                observed_analytics_topics = sorted(
                    str(topic)
                    for topic in topic_counts.keys()
                    if str(topic).startswith("analytics.")
                )

    _debug_strategy_config(
        _read_attr(strategy_pipeline, "strategy_config", None)
        or _read_attr(strategy_pipeline.strategy_engine, "config", None),
        observed_analytics_topics=observed_analytics_topics,
    )
    _debug_strategy_registry(strategy_pipeline.registry)
    _debug_strategy_engine(strategy_pipeline.strategy_engine)
    _debug_signal_processor(strategy_pipeline.signal_processor)
    _debug_runtime_state(_read_attr(strategy_pipeline, "runtime_state", None))

    print("=============================================\n")


# =============================================================================
# Entrypoint
# =============================================================================


async def run_full_backtest(
    runtime_config: BacktestProjectBootstrapConfig | None = None,
    *,
    event_bus: EventBus | None = None,
    scheduler: Scheduler | None = None,
) -> BacktestResult:
    """
    Run a full production-style backtest on one shared system runtime.

    If event_bus/scheduler are not provided by a higher-level application,
    this entrypoint creates exactly one shared pair and passes it everywhere.
    BacktestProjectBootstrap must not create its own local runtime.
    """

    core_config = Config.from_env()
    runtime = runtime_config or BacktestProjectBootstrapConfig.from_env()

    if event_bus is None or scheduler is None:
        shared_event_bus, shared_scheduler = build_shared_system_runtime(core_config)
        event_bus = event_bus or shared_event_bus
        scheduler = scheduler or shared_scheduler

    backtest_config = build_backtest_config(runtime, core_config=core_config)
    dataset = load_dataset(backtest_config)

    print_dataset_summary(dataset, backtest_config)

    bootstrap = BacktestProjectBootstrap(
        config=runtime,
        core_config=core_config,
        event_bus=event_bus,
        scheduler=scheduler,
        backtest_config=backtest_config,
    )
    pipeline = bootstrap.build()

    print_components_summary(pipeline)

    tester = StrategyTester(
        config=backtest_config,
        dataset=dataset,
        event_bus=pipeline.event_bus,
        scheduler=pipeline.scheduler,
        data_caches=pipeline.data_caches,
        analytics_components=pipeline.analytics_components,
        strategy_engine=pipeline.strategy_pipeline.strategy_engine,
        signal_processor=pipeline.strategy_pipeline.signal_processor,
        risk_manager=pipeline.risk_manager,
    )

    result = await tester.run(dataset=dataset)
    print_strategy_internal_debug(pipeline, result)
    print_result_summary(result)
    return result


async def main_async() -> BacktestResult:
    return await run_full_backtest()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()