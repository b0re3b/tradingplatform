"""
test_signal_processor_diagnostics.py

Diagnostic pytest for strategy SignalProcessor / StrategyEngine failures.

Purpose
-------
This test is intentionally verbose. It is meant to show WHY the pipeline emits
signal.rejected(reason="no_passed_strategy_signals") or another stage-specific
rejection instead of signal.generated.

It prints:
- registry strategies/categories;
- emitted analytics payloads;
- captured signal.rejected payloads;
- selected strategies;
- route.skipped;
- evaluation_reasons;
- batch/debug failure_stage;
- context feature/domain keys.

Run from project root:

    python -m pytest -s test/strategy/test_signal_processor_diagnostics.py

Useful env flags
----------------
STRATEGY_DIAG_STRICT_SIGNAL=0
    Do not fail if no signal.generated is produced. Default: 0.

STRATEGY_DIAG_STRICT_REJECTION_DEBUG=1
    Fail if signal.rejected has no diagnostic debug block. Default: 1.

STRATEGY_DIAG_DIRECT_BATCH=1
    If no signal.generated is captured, call StrategyEngine.process_analytics_event
    directly for every synthetic event and dump ProcessedSignalBatch. Default: 1.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
import traceback
from collections import Counter, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pytest


# =============================================================================
# Project import bootstrap
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Project imports
# =============================================================================

from core.config import Config
from core.event_bus import EventBus, EventPriority
from core.scheduler import Scheduler

from backtesting.bootstrap import BacktestProjectBootstrap, BacktestProjectBootstrapConfig
from backtesting.config import BacktestConfig
from backtesting.enums import BacktestDataType, BacktestMode, DataGapPolicy, DataValidationLevel

from strategy.enums import StrategyCategory


# =============================================================================
# Small helpers
# =============================================================================

def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def safe_value(value: Any, *, max_items: int = 60) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if hasattr(value, "value"):
        try:
            return value.value
        except Exception:
            pass

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(k): safe_value(v, max_items=max_items)
            for k, v in list(value.items())[:max_items]
        }

    if isinstance(value, (list, tuple, set, frozenset)):
        return [safe_value(v, max_items=max_items) for v in list(value)[:max_items]]

    for method_name in ("to_dict", "summary"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return safe_value(method(), max_items=max_items)
            except Exception:
                pass

    return repr(value)


def dump(title: str, value: Any = None) -> None:
    print(f"\n========== {title} ==========", flush=True)
    if value is not None:
        print(safe_value(value), flush=True)
    print("=" * (22 + len(title)), flush=True)


def dump_mapping(title: str, mapping: dict[str, Any] | None, *, max_items: int = 80) -> None:
    print(f"\n========== {title} ==========", flush=True)
    if not mapping:
        print("- <empty>", flush=True)
    else:
        for i, (key, value) in enumerate(mapping.items(), start=1):
            if i > max_items:
                print(f"- ... {len(mapping) - max_items} more", flush=True)
                break
            print(f"- {key}: {safe_value(value)}", flush=True)
    print("=" * (22 + len(title)), flush=True)


def dump_list(title: str, values: Iterable[Any] | None, *, max_items: int = 100) -> None:
    print(f"\n========== {title} ==========", flush=True)
    items = list(values or [])
    if not items:
        print("- <empty>", flush=True)
    else:
        for i, value in enumerate(items, start=1):
            if i > max_items:
                print(f"- ... {len(items) - max_items} more", flush=True)
                break
            print(f"- {safe_value(value)}", flush=True)
    print("=" * (22 + len(title)), flush=True)


@contextmanager
def debug_step(name: str):
    print(f"\n[STEP] {name}", flush=True)
    try:
        yield
    except Exception as exc:
        print(f"[FAIL] {name}: {exc.__class__.__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    else:
        print(f"[OK]   {name}", flush=True)


@asynccontextmanager
async def async_debug_step(name: str):
    print(f"\n[STEP] {name}", flush=True)
    try:
        yield
    except Exception as exc:
        print(f"[FAIL] {name}: {exc.__class__.__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    else:
        print(f"[OK]   {name}", flush=True)


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def call_if_supported(component: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(component, method_name, None)
    if not callable(method):
        return None
    return await maybe_await(method(*args, **kwargs))


def payload_from_event_or_dict(event_or_payload: Any) -> dict[str, Any]:
    if isinstance(event_or_payload, dict):
        return dict(event_or_payload)

    payload = getattr(event_or_payload, "payload", None)
    if isinstance(payload, dict):
        return dict(payload)

    data = getattr(event_or_payload, "data", None)
    if isinstance(data, dict):
        return dict(data)

    return {}


def topic_from_event_or_dict(event_or_payload: Any, fallback: str = "unknown") -> str:
    for attr in ("topic", "name", "event_name", "event_type", "type"):
        value = getattr(event_or_payload, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    if isinstance(event_or_payload, dict):
        for key in ("topic", "event_name", "event_type", "type"):
            value = event_or_payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return fallback


async def emit_event(event_bus: EventBus, topic: str, payload: dict[str, Any]) -> None:
    print(f"\n[EVENT-IN] {topic}", flush=True)
    dump_mapping("payload", payload, max_items=80)

    try:
        result = event_bus.emit(
            topic,
            payload,
            priority=EventPriority.NORMAL,
            source="test.signal_processor_diagnostics",
        )
    except TypeError:
        result = event_bus.emit(topic, payload)

    await maybe_await(result)


async def drain_event_bus(event_bus: EventBus, *, cycles: int = 8) -> None:
    for method_name in ("drain", "join", "flush", "wait_idle", "wait_until_idle"):
        method = getattr(event_bus, method_name, None)
        if callable(method):
            try:
                await maybe_await(method())
                return
            except Exception:
                pass

    for _ in range(max(1, cycles)):
        await asyncio.sleep(0)


async def wait_until(predicate: Callable[[], bool], *, timeout_seconds: float, label: str) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds

    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)

    print(f"[WARN] timed out waiting for {label} after {timeout_seconds:.1f}s", flush=True)
    return False


# =============================================================================
# Runtime / bootstrap
# =============================================================================

def build_shared_system_runtime(core_config: Config) -> tuple[EventBus, Scheduler]:
    event_bus_config = getattr(core_config, "event_bus", None)
    scheduler_config = getattr(core_config, "scheduler", None)

    event_bus_attempts: list[tuple[str, Callable[[], EventBus]]] = [
        ("EventBus(config=core_config.event_bus)", lambda: EventBus(config=event_bus_config)),
        ("EventBus(core_config.event_bus)", lambda: EventBus(event_bus_config)),
        ("EventBus()", lambda: EventBus()),
    ]

    event_bus: EventBus | None = None
    errors: list[str] = []
    for label, factory in event_bus_attempts:
        try:
            event_bus = factory()
            print(f"[OK] EventBus created via {label}", flush=True)
            break
        except Exception as exc:
            errors.append(f"{label}: {exc.__class__.__name__}: {exc}")

    if event_bus is None:
        raise RuntimeError("Cannot create EventBus:\n- " + "\n- ".join(errors))

    scheduler_attempts: list[tuple[str, Callable[[], Scheduler]]] = [
        (
            "Scheduler(config=core_config.scheduler, event_bus=event_bus)",
            lambda: Scheduler(config=scheduler_config, event_bus=event_bus),
        ),
        (
            "Scheduler(scheduler_config=core_config.scheduler, event_bus=event_bus)",
            lambda: Scheduler(scheduler_config=scheduler_config, event_bus=event_bus),
        ),
        ("Scheduler(event_bus=event_bus)", lambda: Scheduler(event_bus=event_bus)),
        ("Scheduler()", lambda: Scheduler()),
    ]

    scheduler: Scheduler | None = None
    scheduler_errors: list[str] = []
    for label, factory in scheduler_attempts:
        try:
            scheduler = factory()
            print(f"[OK] Scheduler created via {label}", flush=True)
            break
        except Exception as exc:
            scheduler_errors.append(f"{label}: {exc.__class__.__name__}: {exc}")

    if scheduler is None:
        raise RuntimeError("Cannot create Scheduler:\n- " + "\n- ".join(scheduler_errors))

    return event_bus, scheduler


def build_diagnostic_runtime() -> BacktestProjectBootstrapConfig:
    return BacktestProjectBootstrapConfig(
        exchange="binance",
        market_type="usdm_futures",
        symbols=["BTCUSDT", "SOLUSDT"],
        timeframes=["1m"],
        enable_candles=True,
        enable_trades=False,
        enable_orderbook=False,
        enable_funding=True,
        enable_open_interest=True,
        enable_liquidations=False,
        enable_spreads=False,
        require_analytics_for_enabled_streams=False,
        verbose_bootstrap_errors=True,
        service_name="test.signal_processor_diagnostics",
    )


def build_diagnostic_backtest_config(
    runtime: BacktestProjectBootstrapConfig,
    core_config: Config,
) -> BacktestConfig:
    now = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)
    start = now - timedelta(days=2)

    try:
        config = BacktestConfig.default_binance_futures(
            run_name="test_signal_processor_diagnostics",
            symbols=list(runtime.symbols),
            timeframes=list(runtime.timeframes),
            start_time=start,
            end_time=now,
            initial_balance=10_000.0,
        )
    except Exception:
        config = BacktestConfig()

    config.mode = BacktestMode.MULTI_STRATEGY
    config.exchange = runtime.exchange
    config.market_type = runtime.market_type
    config.symbols = list(runtime.symbols)
    config.timeframes = list(runtime.timeframes)
    config.start_time = start
    config.end_time = now
    config.initial_balance = 10_000.0

    config.use_candles = True
    config.use_funding = True
    config.use_open_interest = True
    config.use_trades = False
    config.use_orderbook = False
    config.use_liquidations = False
    config.use_mark_price = False
    config.use_index_price = False

    config.data_loader.exchange = config.exchange
    config.data_loader.market_type = config.market_type
    config.data_loader.symbols = list(config.symbols)
    config.data_loader.timeframes = list(config.timeframes)
    config.data_loader.data_types = {
        BacktestDataType.CANDLES,
        BacktestDataType.FUNDING,
        BacktestDataType.OPEN_INTEREST,
    }
    config.data_loader.require_candles = False
    config.data_loader.require_funding = False
    config.data_loader.require_open_interest = False
    config.data_loader.require_trades = False
    config.data_loader.require_orderbook = False
    config.data_loader.allow_empty_optional_streams = True
    config.data_loader.validation_level = DataValidationLevel.BASIC
    config.data_loader.gap_policy = DataGapPolicy.WARN
    config.data_loader.drop_duplicate_events = True
    config.core_config = core_config

    config.validate()
    return config


# =============================================================================
# Captures / monitor
# =============================================================================

@dataclass(slots=True)
class CapturedEvent:
    topic: str
    payload: dict[str, Any]
    raw: Any


@dataclass(slots=True)
class EventCapture:
    events: list[CapturedEvent] = field(default_factory=list)

    async def handle(self, event_or_payload: Any = None, **kwargs: Any) -> None:
        topic = topic_from_event_or_dict(
            event_or_payload,
            fallback=str(kwargs.get("topic") or "unknown"),
        )
        payload = payload_from_event_or_dict(event_or_payload)

        if not payload and isinstance(kwargs.get("payload"), dict):
            payload = dict(kwargs["payload"])

        print(f"\n[EVENT-OUT] {topic}", flush=True)
        dump_mapping("captured payload", payload, max_items=120)

        self.events.append(CapturedEvent(topic=topic, payload=payload, raw=event_or_payload))

    def by_prefix(self, prefix: str) -> list[CapturedEvent]:
        return [event for event in self.events if event.topic.startswith(prefix)]

    def by_topic(self, topic: str) -> list[CapturedEvent]:
        return [event for event in self.events if event.topic == topic]

    def any_signal_event(self) -> bool:
        return bool(self.by_prefix("signal."))

    def any_signal_generated(self) -> bool:
        return bool(self.by_topic("signal.generated"))

    def any_signal_rejected(self) -> bool:
        return bool(self.by_topic("signal.rejected"))


@dataclass(slots=True)
class FlowMonitor:
    max_recent_events: int = 200
    topic_counts: Counter[str] = field(default_factory=Counter)
    group_counts: Counter[str] = field(default_factory=Counter)
    samples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    recent_events: deque[dict[str, Any]] = field(default_factory=deque)
    subscriptions: list[Any] = field(default_factory=list)
    event_bus: EventBus | None = None

    TOPICS: tuple[str, ...] = (
        "analytics.*",
        "strategy.*",
        "signal.*",
        "risk.*",
        "execution.*",
        "position.*",
        "system.*",
        "analytics.oi.divergence",
        "analytics.oi.anomaly",
        "analytics.oi.updated",
        "analytics.funding.updated",
        "analytics.price_action.market_structure.bos",
        "analytics.price_action.trend.trend_alignment",
    )

    def register(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        for topic in self.TOPICS:
            self._subscribe(topic)

    def unregister(self) -> None:
        if self.event_bus is None:
            self.subscriptions.clear()
            return

        for sub in list(self.subscriptions):
            try:
                self.event_bus.unsubscribe(sub)
            except Exception:
                pass
        self.subscriptions.clear()

    def _subscribe(self, pattern: str) -> None:
        assert self.event_bus is not None
        name = f"diag_monitor_{pattern.replace('.', '_').replace('*', 'wildcard')}"

        async def handler(event_or_payload: Any = None, **kwargs: Any) -> None:
            try:
                topic = topic_from_event_or_dict(
                    event_or_payload,
                    fallback=pattern.replace("*", "event"),
                )
                payload = payload_from_event_or_dict(event_or_payload)

                if not payload and isinstance(kwargs.get("payload"), dict):
                    payload = dict(kwargs["payload"])

                self.record(topic, payload)
            except Exception:
                return

        try:
            sub = self.event_bus.subscribe(pattern, handler, name=name)
        except TypeError:
            sub = self.event_bus.subscribe(pattern=pattern, handler=handler, name=name)

        self.subscriptions.append(sub)

    @staticmethod
    def group(topic: str) -> str:
        if topic.startswith("analytics."):
            return "analytics"
        if topic.startswith("strategy."):
            return "strategy"
        if topic.startswith("signal."):
            return "signal"
        if topic.startswith("risk."):
            return "risk"
        if topic.startswith("execution."):
            return "execution"
        if topic.startswith("position."):
            return "position"
        if topic.startswith("system."):
            return "system"
        return "other"

    @staticmethod
    def compact(payload: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "exchange",
            "market_type",
            "symbol",
            "timeframe",
            "source",
            "source_topic",
            "strategy_name",
            "signal_id",
            "side",
            "direction",
            "confidence",
            "score",
            "status",
            "reason",
            "price",
            "timestamp_ms",
            "received_at_ms",
        )
        result = {key: safe_value(payload.get(key)) for key in keys if payload.get(key) is not None}

        for key in ("selected_strategies", "evaluation_reasons", "route_skipped", "debug"):
            value = payload.get(key)
            if value is not None:
                result[key] = safe_value(value, max_items=20)

        return result

    def record(self, topic: str, payload: dict[str, Any]) -> None:
        group = self.group(topic)
        self.topic_counts[topic] += 1
        self.group_counts[group] += 1

        self.samples.setdefault(topic, [])
        if len(self.samples[topic]) < 3:
            self.samples[topic].append(self.compact(payload))

        self.recent_events.append(
            {
                "topic": topic,
                "group": group,
                "symbol": payload.get("symbol"),
                "timeframe": payload.get("timeframe"),
                "strategy_name": payload.get("strategy_name") or payload.get("strategy"),
                "signal_id": payload.get("signal_id") or payload.get("id"),
                "reason": payload.get("reason"),
                "timestamp_ms": payload.get("timestamp_ms"),
            }
        )

        while len(self.recent_events) > self.max_recent_events:
            self.recent_events.popleft()

    @property
    def last_stage(self) -> str:
        for group in ("position", "execution", "risk", "signal", "strategy", "analytics"):
            if self.group_counts.get(group, 0) > 0:
                return group
        return "none"

    def print_summary(self) -> None:
        dump_mapping("FLOW group counts", dict(self.group_counts))
        dump_mapping("FLOW top topics", dict(self.topic_counts.most_common(40)))
        dump_mapping("FLOW samples", self.samples, max_items=40)
        dump_list("FLOW recent events", list(self.recent_events)[-40:])


# =============================================================================
# Synthetic analytics events that should exercise route/evaluation diagnostics
# =============================================================================

def diagnostic_events() -> list[tuple[str, dict[str, Any]]]:
    ts = datetime(2026, 5, 20, 0, 0, 0, 1000, tzinfo=timezone.utc).isoformat()

    base = {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "timestamp": ts,
        "timestamp_ms": 1779235200001,
        "received_at_ms": 1779235200001,
    }

    return [
        (
            "analytics.oi.divergence",
            {
                **base,
                "symbol": "BTCUSDT",
                "exchange_symbol": "BTCUSDT",
                "timeframe": "1m",
                "confidence": 0.74,
                "score": 0.76,
                "price": 77150.0,
                "side": "long",
                "direction": "long",
                "features": {
                    "open_interest": 104_547.32,
                    "oi_delta": 4200.0,
                    "oi_delta_pct": 0.034,
                    "oi_direction": "long",
                    "oi_pressure_score": 0.82,
                    "price_delta_pct": 0.42,
                    "volume_ratio": 1.6,
                },
                "regime": {
                    "regime": "trend_confirmation",
                    "confidence": 0.71,
                    "score": 0.70,
                },
                "divergence": {
                    "detected": True,
                    "divergence_type": "bullish",
                    "direction": "long",
                    "side": "long",
                    "confidence": 0.74,
                    "score": 0.76,
                    "window_size": 8,
                },
                "feature_map": {
                    "score": 0.76,
                    "confidence": 0.74,
                    "direction": "long",
                    "entry_price": 77150.0,
                },
            },
        ),
        (
            "analytics.price_action.market_structure.bos",
            {
                **base,
                "symbol": "BTCUSDT",
                "exchange_symbol": "BTCUSDT",
                "timeframe": "1m",
                "event_type": "bos",
                "direction": "bullish",
                "side": "long",
                "confidence": 0.72,
                "score": 0.74,
                "price": 77161.5,
                "close": 77161.5,
                "market_structure": {
                    "last_break_event": {
                        "event_type": "bos",
                        "side": "long",
                        "direction": "long",
                        "confidence": 0.72,
                        "score": 0.74,
                        "price": 77161.5,
                        "confirmed": True,
                    },
                    "external": {
                        "bias": "bullish",
                        "market_bias": "bullish",
                        "confidence": 0.72,
                        "score": 0.74,
                    },
                    "internal": {
                        "bias": "bullish",
                        "market_bias": "bullish",
                        "confidence": 0.70,
                        "score": 0.70,
                    },
                    "mtf_alignment": 0.72,
                    "trend_strength": 0.74,
                },
                "feature_map": {
                    "market_structure": {"event_type": "bos", "market_bias": "bullish"},
                    "score": 0.74,
                    "confidence": 0.72,
                    "direction": "long",
                    "entry_price": 77161.5,
                },
            },
        ),
        (
            "analytics.price_action.trend.trend_alignment",
            {
                **base,
                "symbol": "BTCUSDT",
                "exchange_symbol": "BTCUSDT",
                "timeframe": "1m",
                "direction": "bullish",
                "side": "long",
                "confidence": 0.70,
                "score": 0.73,
                "price": 77173.0,
                "close": 77173.0,
                "trend": {
                    "last_signal": {
                        "event_type": "trend_alignment",
                        "direction": "long",
                        "trend_direction": "long",
                        "trend_regime": "uptrend",
                        "confidence": 0.70,
                        "score": 0.73,
                        "continuation_probability": 0.82,
                        "confirmed": True,
                    },
                    "external": {
                        "direction": "long",
                        "trend_direction": "long",
                        "trend_regime": "uptrend",
                        "confidence": 0.70,
                        "score": 0.73,
                        "trend_strength": 0.73,
                        "continuation_probability": 0.82,
                    },
                    "internal": {
                        "direction": "long",
                        "trend_direction": "long",
                        "trend_regime": "uptrend",
                        "confidence": 0.68,
                        "score": 0.69,
                        "trend_strength": 0.69,
                        "continuation_probability": 0.76,
                    },
                    "internal_external_alignment": 0.80,
                    "higher_timeframe_alignment": 0.70,
                    "overall_trend_score": 0.74,
                },
                "feature_map": {
                    "score": 0.73,
                    "confidence": 0.70,
                    "direction": "long",
                    "entry_price": 77173.0,
                },
            },
        ),
        (
            "analytics.funding.updated",
            {
                **base,
                "symbol": "SOLUSDT",
                "exchange_symbol": "SOLUSDT",
                "timeframe": "1h",
                "confidence": 0.73,
                "score": 0.75,
                "price": 85.14,
                "side": "long",
                "direction": "long",
                "snapshot": {
                    "funding_rate": -0.0015,
                    "predicted_rate": -0.0012,
                    "next_funding_time_ms": 1779348000000,
                },
                "regime": {
                    "regime": "extreme_negative",
                    "confidence": 0.73,
                    "score": 0.75,
                },
                "pressure": {
                    "side": "long",
                    "bias": "long",
                    "pressure_score": 0.76,
                    "score": 0.76,
                },
                "extreme": {
                    "detected": True,
                    "extreme_type": "negative_extreme",
                    "side": "long",
                    "direction": "long",
                    "score": 0.75,
                    "confidence": 0.73,
                    "severity": 0.75,
                    "reversal_risk": True,
                    "mean_reversion_probability": 0.75,
                    "squeeze_probability": 0.50,
                },
                "signal": {
                    "bias": "long",
                    "side": "long",
                    "direction": "long",
                    "score": 0.75,
                    "confidence": 0.73,
                },
                "funding_rate": -0.0015,
            },
        ),
    ]


# =============================================================================
# Registry / engine / batch inspection
# =============================================================================

def registry_names(registry: Any) -> list[str]:
    for method_name in ("list_names", "names", "strategy_names"):
        method = getattr(registry, method_name, None)
        if callable(method):
            try:
                return [str(item) for item in method()]
            except Exception:
                pass

    list_all = getattr(registry, "list_all", None)
    if callable(list_all):
        try:
            return [str(getattr(item, "strategy_name", item)) for item in list_all()]
        except Exception:
            pass

    raw = getattr(registry, "_strategies", None)
    if isinstance(raw, dict):
        return sorted(str(key) for key in raw.keys())

    return []


def registry_categories(registry: Any) -> dict[str, list[str]]:
    by_category = getattr(registry, "_by_category", None)
    if isinstance(by_category, dict):
        result: dict[str, list[str]] = {}
        for category, names in by_category.items():
            key = getattr(category, "value", str(category))
            result[str(key)] = sorted(str(item) for item in list(names))
        return result
    return {}


def dump_registry(registry: Any) -> None:
    names = registry_names(registry)
    categories = registry_categories(registry)

    dump_list("REGISTRY names", names)
    dump_mapping("REGISTRY categories", categories)
    dump_mapping(
        "REGISTRY required categories presence",
        {
            "price_action": bool(categories.get(StrategyCategory.PRICE_ACTION.value)),
            "open_interest": bool(categories.get(StrategyCategory.OPEN_INTEREST.value)),
            "funding": bool(categories.get(StrategyCategory.FUNDING.value)),
            "count": len(names),
        },
    )


def dump_engine_debug(engine: Any, *, title: str = "ENGINE DEBUG") -> None:
    stats = getattr(engine, "stats", None)
    stats_summary = None

    if stats is not None:
        summary = getattr(stats, "summary", None)
        if callable(summary):
            try:
                stats_summary = summary()
            except Exception:
                stats_summary = safe_value(stats)
        else:
            stats_summary = safe_value(stats)

    event_handler = getattr(engine, "event_handler", None)
    handler_debug = {}
    if event_handler is not None:
        handler_debug = {
            "registered": getattr(event_handler, "is_registered", getattr(event_handler, "_registered", None)),
            "started": getattr(event_handler, "is_started", getattr(event_handler, "_started", None)),
            "subscriptions_count": getattr(event_handler, "subscriptions_count", None),
            "_subscriptions_count": len(getattr(event_handler, "_subscriptions", []) or []),
        }

        for method_name in ("_analytics_topics", "analytics_topics"):
            method = getattr(event_handler, method_name, None)
            if callable(method):
                try:
                    topics = list(method())
                    handler_debug["analytics_topics_count"] = len(topics)
                    handler_debug["analytics_topics_first_80"] = topics[:80]
                    break
                except Exception as exc:
                    handler_debug["analytics_topics_error"] = str(exc)

    dump_mapping(
        title,
        {
            "engine_started": getattr(engine, "is_started", getattr(engine, "_started", None)),
            "engine_registered": getattr(engine, "is_registered", getattr(engine, "_registered", None)),
            "engine_subscriptions_count": len(getattr(engine, "_subscriptions", []) or []),
            "stats": stats_summary,
            "event_handler": handler_debug,
        },
    )


def dump_processed_batch(batch: Any) -> None:
    print("\n========== PROCESSED SIGNAL BATCH ==========", flush=True)
    print(f"accepted={getattr(batch, 'accepted', None)!r}", flush=True)
    print(f"emitted={getattr(batch, 'emitted', None)!r}", flush=True)
    print(f"symbol={getattr(batch, 'symbol', None)!r}", flush=True)
    print(f"reasons={safe_value(getattr(batch, 'reasons', None), max_items=120)}", flush=True)
    print(f"metadata={safe_value(getattr(batch, 'metadata', None), max_items=120)}", flush=True)
    print(f"debug={safe_value(getattr(batch, 'debug', None), max_items=240)}", flush=True)

    normalized = getattr(batch, "normalized", None)
    if normalized is not None:
        print("\n[NORMALIZED]", flush=True)
        print(f"source={getattr(normalized, 'source', None)!r}", flush=True)
        print(f"symbol={getattr(normalized, 'symbol', None)!r}", flush=True)
        print(f"timeframe={getattr(normalized, 'timeframe', None)!r}", flush=True)
        print(f"domain_data={safe_value(getattr(normalized, 'domain_data', None), max_items=120)}", flush=True)
        extra_domain_data = getattr(normalized, "extra_domain_data", None)
        print(f"extra_domain_data={safe_value(extra_domain_data, max_items=120)}", flush=True)
        features = list(getattr(normalized, "features", []) or [])
        print(f"features_count={len(features)}", flush=True)
        print(f"features_first_80={[getattr(item, 'name', repr(item)) for item in features[:80]]}", flush=True)

    route = getattr(batch, "route", None)
    if route is not None:
        print("\n[ROUTE]", flush=True)
        print(f"selected_names={safe_value(getattr(route, 'selected_names', None))}", flush=True)
        print(f"categories_used={safe_value(getattr(route, 'categories_used', None))}", flush=True)
        print(f"matched_features={safe_value(getattr(route, 'matched_features', None))}", flush=True)
        print(f"skipped={safe_value(getattr(route, 'skipped', None), max_items=160)}", flush=True)
        print(f"metadata={safe_value(getattr(route, 'metadata', None), max_items=120)}", flush=True)

    evaluations = list(getattr(batch, "evaluations", []) or [])
    print(f"\n[EVALUATIONS] count={len(evaluations)}", flush=True)
    for ev in evaluations:
        signal = getattr(ev, "signal", None)
        print(
            "-",
            f"strategy={getattr(ev, 'strategy_name', None)!r}",
            f"passed={getattr(ev, 'passed', None)!r}",
            f"score={getattr(ev, 'score', None)!r}",
            f"confidence={getattr(ev, 'confidence', None)!r}",
            f"signal={getattr(signal, 'strategy_name', None)!r}" if signal else "signal=None",
            f"reasons={safe_value(getattr(ev, 'reasons', None), max_items=80)}",
            f"metadata={safe_value(getattr(ev, 'metadata', None), max_items=120)}",
            flush=True,
        )

    for attr in ("raw_signals", "filtered_signals", "final_signals"):
        signals = list(getattr(batch, attr, []) or [])
        print(f"\n[{attr.upper()}] count={len(signals)}", flush=True)
        for signal in signals:
            print(
                "-",
                f"strategy={getattr(signal, 'strategy_name', None)!r}",
                f"side={getattr(signal, 'side', None)!r}",
                f"confidence={getattr(signal, 'confidence', None)!r}",
                f"score={getattr(signal, 'score', None)!r}",
                f"reasons={safe_value(getattr(signal, 'reasons', None), max_items=60)}",
                flush=True,
            )

    print("============================================", flush=True)


async def start_components(pipeline: Any) -> list[Any]:
    started: list[Any] = []
    components: list[Any] = []

    if pipeline.event_bus is not None:
        components.append(pipeline.event_bus)
    if pipeline.scheduler is not None:
        components.append(pipeline.scheduler)

    if pipeline.signal_processor is not None:
        components.append(pipeline.signal_processor)
    if pipeline.strategy_engine is not None:
        components.append(pipeline.strategy_engine)
    if pipeline.risk_manager is not None:
        components.append(pipeline.risk_manager)

    components.extend(list(getattr(pipeline, "data_caches", []) or []))

    seen: set[int] = set()
    deduped: list[Any] = []
    for component in components:
        marker = id(component)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(component)

    for component in deduped:
        print(f"[START] {component.__class__.__name__}", flush=True)
        await call_if_supported(component, "start")
        started.append(component)

    return started


async def stop_components(started: list[Any]) -> None:
    for component in reversed(started):
        try:
            print(f"[STOP] {component.__class__.__name__}", flush=True)
            await call_if_supported(component, "stop")
        except Exception as exc:
            print(f"[WARN] stop failed for {component.__class__.__name__}: {exc}", flush=True)


# =============================================================================
# Test
# =============================================================================

@pytest.mark.asyncio
async def test_signal_processor_rejection_diagnostics_are_visible() -> None:
    core_config: Config | None = None
    event_bus: EventBus | None = None
    scheduler: Scheduler | None = None
    pipeline: Any | None = None
    started: list[Any] = []
    flow_monitor: FlowMonitor | None = None
    capture = EventCapture()

    try:
        with debug_step("build shared runtime"):
            core_config = Config.from_env()
            event_bus, scheduler = build_shared_system_runtime(core_config)

        with debug_step("build diagnostic bootstrap config"):
            runtime = build_diagnostic_runtime()
            backtest_config = build_diagnostic_backtest_config(runtime, core_config)
            dump_mapping(
                "RUNTIME",
                {
                    "symbols": runtime.symbols,
                    "timeframes": runtime.timeframes,
                    "enable_candles": runtime.enable_candles,
                    "enable_funding": runtime.enable_funding,
                    "enable_open_interest": runtime.enable_open_interest,
                },
            )

        with debug_step("build pipeline through BacktestProjectBootstrap"):
            bootstrap = BacktestProjectBootstrap(
                config=runtime,
                core_config=core_config,
                event_bus=event_bus,
                scheduler=scheduler,
                backtest_config=backtest_config,
            )
            pipeline = bootstrap.build()

            diagnostics = getattr(pipeline, "diagnostics", None)
            if diagnostics is not None:
                fmt = getattr(diagnostics, "format", None)
                if callable(fmt):
                    print(fmt(), flush=True)

            dump_registry(pipeline.strategy_registry)

        with debug_step("subscribe capture and monitor"):
            assert event_bus is not None
            flow_monitor = FlowMonitor()
            flow_monitor.register(event_bus)

            for pattern in ("signal.*", "strategy.*", "risk.*"):
                try:
                    event_bus.subscribe(
                        pattern,
                        capture.handle,
                        name=f"diag_capture_{pattern.replace('.', '_').replace('*', 'wildcard')}",
                    )
                except TypeError:
                    event_bus.subscribe(
                        pattern=pattern,
                        handler=capture.handle,
                        name=f"diag_capture_{pattern}",
                    )

        async with async_debug_step("start components"):
            started = await start_components(pipeline)
            await drain_event_bus(event_bus)
            dump_engine_debug(pipeline.strategy_engine, title="ENGINE after start")

        async with async_debug_step("emit diagnostic analytics events"):
            for topic, payload in diagnostic_events():
                await emit_event(event_bus, topic, payload)
                await drain_event_bus(event_bus, cycles=6)

        async with async_debug_step("wait for signal events and print diagnostics"):
            await drain_event_bus(event_bus, cycles=12)
            await wait_until(
                lambda: capture.any_signal_event(),
                timeout_seconds=3.0,
                label="signal.generated or signal.rejected",
            )
            await drain_event_bus(event_bus, cycles=12)

            dump_engine_debug(pipeline.strategy_engine, title="ENGINE after events")
            if flow_monitor is not None:
                flow_monitor.print_summary()

            generated = capture.by_topic("signal.generated")
            rejected = capture.by_topic("signal.rejected")

            dump_mapping(
                "SIGNAL COUNTS",
                {
                    "generated": len(generated),
                    "rejected": len(rejected),
                    "all_signal_events": len(capture.by_prefix("signal.")),
                },
            )

            for event in rejected[-10:]:
                dump_mapping("LAST signal.rejected payload", event.payload, max_items=160)

            for event in generated[-10:]:
                dump_mapping("LAST signal.generated payload", event.payload, max_items=160)

            if env_bool("STRATEGY_DIAG_DIRECT_BATCH", True):
                print("\n[DEBUG] Running direct StrategyEngine batch inspection per event.", flush=True)
                for topic, payload in diagnostic_events():
                    print(f"\n[DIRECT-BATCH] {topic}", flush=True)
                    batch = await pipeline.strategy_engine.process_analytics_event(
                        event_name=topic,
                        payload=payload,
                    )
                    dump_processed_batch(batch)

            if rejected and env_bool("STRATEGY_DIAG_STRICT_REJECTION_DEBUG", True):
                payload = rejected[-1].payload
                assert "debug" in payload, "signal.rejected payload must contain debug"
                assert payload.get("timeframe") is not None, "signal.rejected must contain timeframe"
                assert payload.get("timestamp_ms") is not None, "signal.rejected must contain timestamp_ms"
                assert payload.get("source_topic") is not None, "signal.rejected must contain source_topic"

                debug = payload.get("debug")
                assert isinstance(debug, dict), "signal.rejected.debug must be a dict"
                assert debug.get("failure_stage"), "signal.rejected.debug.failure_stage is required"

            if env_bool("STRATEGY_DIAG_STRICT_SIGNAL", False):
                assert generated, (
                    "Expected at least one signal.generated. "
                    "See signal.rejected.debug / DIRECT-BATCH output above."
                )

    finally:
        print("\n[TEARDOWN]", flush=True)
        if flow_monitor is not None:
            try:
                flow_monitor.unregister()
            except Exception:
                pass

        await stop_components(started)

        if scheduler is not None and all(id(item) != id(scheduler) for item in started):
            try:
                await call_if_supported(scheduler, "stop")
            except Exception:
                pass

        if event_bus is not None and all(id(item) != id(event_bus) for item in started):
            try:
                await call_if_supported(event_bus, "stop")
            except Exception:
                pass