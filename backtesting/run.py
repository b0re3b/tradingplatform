# backtesting/run.py

from __future__ import annotations

import asyncio
import os
from collections import Counter
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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


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


def _safe_public_value(value: Any, _depth: int = 0) -> Any:
    """
    Convert a debug value into something printable and reasonably compact.
    Beyond depth 1 everything collapses to str() to avoid multi-KB dumps.
    """

    if value is None:
        return None

    # Collapse deep nesting immediately — prevents datetime/defaultdict walls
    if _depth > 1:
        return str(value)

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): _safe_public_value(item, _depth + 1)
            for key, item in list(value.items())[:80]
        }

    if isinstance(value, Counter):
        return dict(value.most_common(80))

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        return [_safe_public_value(item, _depth + 1) for item in items[:80]]

    if hasattr(value, "value"):
        try:
            return value.value
        except (AttributeError, TypeError):
            pass

    if hasattr(value, "to_dict"):
        try:
            return _safe_public_value(value.to_dict(), _depth + 1)
        except (TypeError, ValueError, AttributeError, RuntimeError):
            pass

    if hasattr(value, "__dict__"):
        try:
            return {
                key: _safe_public_value(item, _depth + 1)
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


# ---------------------------------------------------------------------------
# Compact stats helper — replaces raw stats dumps in engine/processor
# ---------------------------------------------------------------------------

_STATS_COUNTER_KEYS = (
    "events_received",
    "events_processed",
    "events_failed",
    "events_rejected",
    "signals_generated",
    "signals_rejected",
    "signals_emitted",
    "signals_built",
    "signals_filtered",
    "batches_processed",
    "signal_confirmed_events",
    "close_requested_events",
    "reduce_requested_events",
    "kill_switch_events",
    "orders_created",
    "orders_submitted",
    "orders_accepted",
    "orders_rejected",
    "orders_failed",
    "orders_cancelled",
    "orders_filled",
    "orders_partially_filled",
    "fills_created",
    "fills_processed",
    "positions_opened",
    "positions_updated",
    "positions_closed",
    "positions_liquidated",
)

_STATS_TIME_KEYS = (
    "started_at",
    "updated_at",
    "stopped_at",
)


def _compact_stats(stats: Any) -> dict[str, Any]:
    """
    Extract only scalar counters and timestamps from a stats object.
    Deliberately ignores per-strategy dicts, defaultdicts, and last_*_at maps
    to keep the output concise.
    """
    result: dict[str, Any] = {}

    for key in _STATS_COUNTER_KEYS:
        v = _read_attr(stats, key, None)
        if v is not None:
            result[key] = v

    for key in _STATS_TIME_KEYS:
        v = _read_attr(stats, key, None)
        if v is not None:
            result[key] = str(v)

    status = _read_attr(stats, "status", None)
    if status is not None:
        result["status"] = _safe_public_value(status)

    last_error = _read_attr(stats, "last_error", None)
    if last_error:
        result["last_error"] = str(last_error)

    # Fallback: try summary/snapshot/to_dict but only keep scalar values.
    if not result:
        for method_name in ("summary", "snapshot", "to_dict"):
            raw = _call_noarg(stats, method_name)
            if isinstance(raw, dict):
                result = {
                    k: v
                    for k, v in raw.items()
                    if isinstance(v, (int, float, bool, str))
                }
                if result:
                    break

    return result


# =============================================================================
# Post-run event flow diagnostics
# =============================================================================

TRADING_SIGNAL_TOPICS = (
    "signal.generated",
    "signal.updated",
    "signal.confirmed",
    "signal.rejected",
)

TRADING_RISK_TOPICS = (
    "signal.confirmed",
    "risk.position_blocked",
    "risk.position_close_requested",
    "risk.position_reduce_requested",
    "risk.kill_switch",
    "risk.limit_warning",
)

RISK_LIFECYCLE_TOPICS = (
    "risk.manager.started",
    "risk.manager.stopped",
    "risk.manager.registered",
    "risk.manager.unregistered",
)

EXECUTION_TOPICS = (
    "execution.order_submitted",
    "execution.order_accepted",
    "execution.order_rejected",
    "execution.order_failed",
    "execution.order_cancelled",
    "execution.order_partially_filled",
    "execution.order_filled",
)

POSITION_TOPICS = (
    "position.opened",
    "position.updated",
    "position.closed",
    "position.liquidated",
)

ANALYTICS_NON_TRADE_SUFFIXES = (
    ".heartbeat",
    ".metrics",
    ".state_cleaned",
    ".started",
    ".stopped",
)


def _metadata_dict(result: BacktestResult | None) -> dict[str, Any]:
    if result is None:
        return {}
    metadata = getattr(result, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _event_flow_debug(result: BacktestResult | None) -> dict[str, Any]:
    metadata = _metadata_dict(result)
    debug_payload = metadata.get("event_flow_debug")
    return debug_payload if isinstance(debug_payload, dict) else {}


def _topic_counts_from_result(result: BacktestResult | None) -> Counter[str]:
    debug = _event_flow_debug(result)
    raw = debug.get("topic_counts")
    if isinstance(raw, Counter):
        return Counter({str(k): int(v) for k, v in raw.items()})
    if isinstance(raw, dict):
        counts: Counter[str] = Counter()
        for key, value in raw.items():
            try:
                counts[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        return counts
    return Counter()


def _group_counts_from_result(result: BacktestResult | None) -> dict[str, int]:
    debug = _event_flow_debug(result)
    raw = debug.get("group_counts")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = int(value)
        except (TypeError, ValueError):
            pass
    return out


def _samples_from_result(result: BacktestResult | None) -> dict[str, list[Any]]:
    debug = _event_flow_debug(result)
    raw = debug.get("samples")
    if isinstance(raw, dict):
        return {
            str(key): list(value if isinstance(value, list) else [value])[:5]
            for key, value in raw.items()
        }
    return {}


def _recent_events_from_result(result: BacktestResult | None) -> list[Any]:
    debug = _event_flow_debug(result)
    raw = debug.get("recent_events")
    if isinstance(raw, list):
        return raw[-30:]
    return []


def _count_prefix(topic_counts: Counter[str], prefix: str) -> int:
    return sum(count for topic, count in topic_counts.items() if topic.startswith(prefix))


def _count_exact(topic_counts: Counter[str], topics: tuple[str, ...]) -> int:
    return sum(topic_counts.get(topic, 0) for topic in topics)


def _analytics_non_trade_topics(topic_counts: Counter[str]) -> dict[str, int]:
    return {
        topic: count
        for topic, count in topic_counts.items()
        if topic.startswith("analytics.") and topic.endswith(ANALYTICS_NON_TRADE_SUFFIXES)
    }


def _verdict_from_event_flow(
    topic_counts: Counter[str],
    group_counts: dict[str, int],
) -> str:
    signal_generated = topic_counts.get("signal.generated", 0)
    signal_rejected = topic_counts.get("signal.rejected", 0)
    signal_confirmed = topic_counts.get("signal.confirmed", 0)
    risk_blocked = topic_counts.get("risk.position_blocked", 0)
    close_requested = topic_counts.get("risk.position_close_requested", 0)
    reduce_requested = topic_counts.get("risk.position_reduce_requested", 0)
    execution_events = _count_prefix(topic_counts, "execution.")
    position_events = _count_prefix(topic_counts, "position.")
    trading_risk_events = _count_exact(topic_counts, TRADING_RISK_TOPICS)
    lifecycle_risk_events = _count_exact(topic_counts, RISK_LIFECYCLE_TOPICS)

    if execution_events > 0 and position_events > 0:
        return "OK: execution.* і position.* є, pipeline дійшов до симульованих угод."

    if execution_events > 0 and position_events <= 0:
        return "Breakpoint: execution.* є, але position.* немає — перевір PositionSimulator subscriptions/fill payload."

    if signal_confirmed > 0 or close_requested > 0 or reduce_requested > 0:
        return (
            "Breakpoint: є signal.confirmed / risk close-reduce request, але execution.* = 0 — "
            "перевір ExecutionSimulator register/start/config/listen_signal_confirmed та payload validation."
        )

    if risk_blocked > 0:
        return "Breakpoint: RiskManager блокує позиції — дивись risk.position_blocked samples/reasons."

    if signal_rejected > 0 and signal_generated <= 0 and signal_confirmed <= 0:
        return (
            "Breakpoint: Strategy/SignalProcessor емітив signal.rejected, але не було signal.generated/signal.confirmed. "
            "ExecutionSimulator тут не винен — до нього не дійшов risk-approved intent."
        )

    if signal_generated > 0 and trading_risk_events <= lifecycle_risk_events:
        return (
            "Breakpoint: signal.generated є, але немає торгового risk response — перевір RiskManager subscriptions "
            "на signal.generated і schema risk-ready payload."
        )

    if group_counts.get("strategy", 0) > 0 and group_counts.get("signal", 0) <= 0:
        return "Breakpoint: StrategyEngine обробляє events, але SignalProcessor не емітить signal.*."

    if group_counts.get("analytics", 0) > 0 and group_counts.get("strategy", 0) <= 0:
        return "Breakpoint: analytics.* є, але StrategyEngine не отримує/не обробляє їх."

    if group_counts.get("market.updated", 0) > 0 and group_counts.get("analytics", 0) <= 0:
        return "Breakpoint: market.*.updated є, але analytics не публікує analytics.*."

    if group_counts.get("market.raw", 0) > 0 and group_counts.get("market.updated", 0) <= 0:
        return "Breakpoint: MarketReplay емітить raw market.*, але data caches не публікують market.*.updated."

    return "Недостатньо даних для точного breakpoint; дивись topic matrix нижче."


def _extract_reason_from_payload(payload: Any) -> str | None:
    if payload is None:
        return None

    if isinstance(payload, str):
        # Samples may already be stringified dicts; keep only compact strings.
        return payload[:240]

    if not isinstance(payload, dict):
        to_dict = getattr(payload, "to_dict", None)
        if callable(to_dict):
            try:
                payload = to_dict()
            except Exception:
                return str(payload)[:240]
        else:
            return str(payload)[:240]

    candidates = (
        "reason",
        "reject_reason",
        "rejection_reason",
        "error",
        "message",
        "status_reason",
    )
    for key in candidates:
        value = payload.get(key)
        if value:
            return str(value)[:240]

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in candidates:
            value = metadata.get(key)
            if value:
                return str(value)[:240]

    errors = payload.get("errors") or payload.get("reasons")
    if errors:
        return str(errors)[:240]

    return None


def _extract_reason_from_record(record: Any) -> str | None:
    for attr in (
        "reason",
        "reject_reason",
        "rejection_reason",
        "error",
        "message",
        "outcome",
        "status",
    ):
        value = _read_attr(record, attr, None)
        if value:
            return str(_safe_public_value(value))[:240]

    payload = _read_attr(record, "payload", None)
    reason = _extract_reason_from_payload(payload)
    if reason:
        return reason

    metadata = _read_attr(record, "metadata", None)
    if isinstance(metadata, dict):
        reason = _extract_reason_from_payload(metadata)
        if reason:
            return reason

    return None


def _signal_reason_counts(result: BacktestResult | None, samples: dict[str, list[Any]]) -> Counter[str]:
    reasons: Counter[str] = Counter()

    if result is not None:
        for record in list(getattr(result, "signals", []) or []):
            reason = _extract_reason_from_record(record)
            if reason:
                reasons[reason] += 1

    for sample in samples.get("signal.rejected", []):
        reason = _extract_reason_from_payload(sample)
        if reason:
            reasons[reason] += 1

    return reasons


def _print_topic_matrix(title: str, topic_counts: Counter[str], topics: tuple[str, ...]) -> None:
    print(title)
    for topic in topics:
        print(f"- {topic}: {topic_counts.get(topic, 0)}")


def _component_stats(component: Any) -> dict[str, Any]:
    if component is None:
        return {}

    for attr in ("stats_state", "stats", "_stats"):
        stats = _read_attr(component, attr, None)
        if stats is not None:
            compact = _compact_stats(stats)
            if compact:
                return compact

    stats_method = getattr(component, "stats", None)
    if callable(stats_method):
        try:
            value = stats_method()
            if isinstance(value, dict):
                return {
                    str(k): _safe_public_value(v)
                    for k, v in value.items()
                    if isinstance(v, (int, float, bool, str)) or k in {"status", "last_error"}
                }
            compact = _compact_stats(value)
            if compact:
                return compact
        except Exception as exc:
            return {"stats_error": str(exc)}

    return {}


def print_event_flow_diagnostics(result: BacktestResult | None, tester: StrategyTester | None = None) -> None:
    """
    More precise event-flow diagnostics than StrategyTester.suspected_breakpoint.

    Key fix:
    risk.* lifecycle events are separated from trading risk events. Therefore
    risk.manager.stopped no longer makes the report blame ExecutionSimulator.
    """

    enabled = _env_bool("BACKTEST_DEBUG_EVENT_FLOW", True)
    if not enabled:
        return

    topic_counts = _topic_counts_from_result(result)
    group_counts = _group_counts_from_result(result)
    samples = _samples_from_result(result)
    recent_events = _recent_events_from_result(result)

    print("\n========== PRECISE EVENT FLOW DIAGNOSTICS ==========")
    if group_counts:
        print("Group counts:")
        for key in (
            "market.raw",
            "market.updated",
            "market.candle.closed",
            "analytics",
            "strategy",
            "signal",
            "risk",
            "execution",
            "position",
            "system",
        ):
            print(f"- {key}: {group_counts.get(key, 0)}")

    debug = _event_flow_debug(result)
    if debug:
        print("\nOriginal monitor:")
        print(f"- last_stage: {debug.get('last_stage')}")
        print(f"- suspected_breakpoint: {debug.get('suspected_breakpoint')}")

    print("\nCorrected verdict:")
    print(f"- {_verdict_from_event_flow(topic_counts, group_counts)}")

    print("\nTrading signal topics:")
    _print_topic_matrix("", topic_counts, TRADING_SIGNAL_TOPICS)

    print("\nRisk topics split:")
    _print_topic_matrix("Trading risk topics:", topic_counts, TRADING_RISK_TOPICS)
    _print_topic_matrix("Lifecycle/system risk topics:", topic_counts, RISK_LIFECYCLE_TOPICS)

    print("\nExecution topics:")
    _print_topic_matrix("", topic_counts, EXECUTION_TOPICS)

    print("\nPosition topics:")
    _print_topic_matrix("", topic_counts, POSITION_TOPICS)

    non_trade_analytics = _analytics_non_trade_topics(topic_counts)
    if non_trade_analytics:
        print("\nAnalytics non-trade topics observed by monitor:")
        for topic, count in sorted(non_trade_analytics.items(), key=lambda item: item[0])[:40]:
            print(f"- {topic}: {count}")
        print("Hint: ці topics не мають потрапляти в SignalNormalizer як tradeable analytics payload.")

    reasons = _signal_reason_counts(result, samples)
    if reasons:
        print("\nTop signal reject/status reasons:")
        for reason, count in reasons.most_common(15):
            print(f"- {count}x {reason}")

    interesting_samples = [
        "signal.rejected",
        "signal.generated",
        "signal.confirmed",
        "risk.position_blocked",
        "execution.order_rejected",
        "execution.order_failed",
        "execution.order_filled",
    ]
    print("\nImportant payload samples:")
    printed_any = False
    for topic in interesting_samples:
        topic_samples = samples.get(topic, [])
        if not topic_samples:
            continue
        printed_any = True
        print(f"- {topic}:")
        for sample in topic_samples[:3]:
            print(f"  {_safe_public_value(sample)}")
    if not printed_any:
        print("- <none>")

    if tester is not None:
        components = getattr(tester, "components", None)
        if components is not None:
            print("\nBacktesting simulator stats:")
            execution_simulator = _read_attr(components, "execution_simulator", None)
            position_simulator = _read_attr(components, "position_simulator", None)
            market_replay = _read_attr(components, "market_replay", None)
            collectors = _read_attr(components, "collectors", None)

            print(f"- MarketReplay: {_component_stats(market_replay) or '<missing>'}")
            print(f"- ExecutionSimulator: {_component_stats(execution_simulator) or '<missing>'}")
            print(f"- PositionSimulator: {_component_stats(position_simulator) or '<missing>'}")
            print(f"- Collectors: {_component_stats(collectors) or '<missing>'}")

    if recent_events:
        print("\nRecent events:")
        for event in recent_events[-15:]:
            print(f"- {_safe_public_value(event)}")

    if topic_counts:
        print("\nTop topics:")
        for topic, count in topic_counts.most_common(30):
            print(f"- {topic}: {count}")

    print("====================================================\n")


# =============================================================================
# Strategy internal diagnostics
# =============================================================================


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

            non_trade_observed = [
                topic
                for topic in observed_analytics_topics
                if topic.endswith(ANALYTICS_NON_TRADE_SUFFIXES)
            ]

            print(f"- observed_analytics_topics.count: {len(observed_analytics_topics)}")
            print(f"- observed_analytics_topics.first_50: {observed_analytics_topics[:50]}")
            print(f"- observed_analytics_not_in_routing.count: {len(unmatched_observed)}")
            print(f"- observed_analytics_not_in_routing.first_50: {unmatched_observed[:50]}")
            print(f"- observed_non_trade_analytics.count: {len(non_trade_observed)}")
            print(f"- observed_non_trade_analytics.first_30: {non_trade_observed[:30]}")
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
        compact = _compact_stats(stats)
        if compact:
            print(f"- stats: {compact}")
        else:
            print(f"- stats: {stats.__class__.__name__} (no scalar fields found)")
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
            topic_list = list(analytics_topics) if not isinstance(analytics_topics, str) else [analytics_topics]
            print(f"  - analytics_topics.count: {len(topic_list)}")
            print(f"  - analytics_topics.first_50: {_safe_public_value(topic_list[:50])}")


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
        compact = _compact_stats(stats)
        if compact:
            print(f"- stats: {compact}")
        else:
            print(f"- stats: {stats.__class__.__name__} (no scalar fields found)")
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

    # Only show scalar summary — skip snapshot/to_dict to avoid large dumps.
    for attr in ("signal_state", "context_store", "cooldown_state", "metrics_state"):
        value = _read_attr(runtime_state, attr, None)
        if value is not None:
            class_name = value.__class__.__name__
            size: Any = None
            for size_attr in ("count", "size", "__len__"):
                if size_attr == "__len__":
                    try:
                        size = len(value)
                    except TypeError:
                        pass
                else:
                    size = _read_attr(value, size_attr, None)
                if size is not None:
                    break
            if size is not None:
                print(f"- {attr}: {class_name}(len={size})")
            else:
                print(f"- {attr}: {class_name}")


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
        debug_payload = _event_flow_debug(result)
        if isinstance(debug_payload, dict):
            print("EventFlow summary:")
            print(f"- last_stage: {debug_payload.get('last_stage')}")
            print(f"- original_suspected_breakpoint: {debug_payload.get('suspected_breakpoint')}")
            topic_counts = _topic_counts_from_result(result)
            group_counts = _group_counts_from_result(result)
            print(f"- corrected_breakpoint: {_verdict_from_event_flow(topic_counts, group_counts)}")
            print(f"- group_counts: {_safe_public_value(group_counts)}")

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

    # Print precise event-flow diagnostics before strategy internals so the
    # root breakpoint is visible immediately.
    print_event_flow_diagnostics(result, tester)
    print_strategy_internal_debug(pipeline, result)
    print_result_summary(result)
    return result


async def main_async() -> BacktestResult:
    return await run_full_backtest()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()