"""
Full-pipeline strategy tester for the backtesting package.

StrategyTester is the backtesting orchestration layer. It preserves the
production event flow and replaces only live market input and live execution:

    BacktestDataset -> MarketReplay -> market.* -> data caches -> analytics
    -> StrategyEngine / SignalProcessor -> RiskManager
    -> ExecutionSimulator -> PositionSimulator -> metrics / reports

Important architectural rules:
- no strategy signal generation happens here;
- no risk approval/sizing happens here;
- no simulated fills or position accounting happen here;
- MarketReplay emits only raw market.* topics;
- ExecutionSimulator starts from risk-approved signal.confirmed events;
- PositionSimulator accounts only simulated execution fills.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.event_bus import EventBus, EventPriority
from core.logger import get_logger
from core.scheduler import Scheduler

from backtesting.backtest_time import BacktestClock
from backtesting.config import BacktestConfig, StrategyTesterConfig
from backtesting.cost_models import TradingCostModel
from backtesting.data_loader import DataLoader
from backtesting.enums import (
    BacktestEventType,
    BacktestStatus,
    BacktestWarningLevel,
    SignalOutcome,
    SimulatedOrderStatus,
    SimulatedPositionStatus,
)
from backtesting.exceptions import (
    BacktestComponentError,
    BacktestDependencyError,
    BacktestLifecycleError,
    BacktestResultCollectionError,
)
from backtesting.execution_simulator import ExecutionSimulator
from backtesting.market_replay import MarketReplay
from backtesting.model_analytics import BacktestModelAnalyticsEngine
from backtesting.models import (
    BacktestDataset,
    BacktestEvent,
    BacktestExecutionRecord,
    BacktestPositionRecord,
    BacktestResult,
    BacktestRiskDecisionRecord,
    BacktestSignalRecord,
    BacktestWarning,
    SerializableMixin,
    SimulationModelSnapshot,
    timestamp_ms,
    utcnow,
)
from backtesting.performance_metrics import (
    PerformanceMetrics,
    build_metrics_input_from_components,
)
from backtesting.position_simulator import PositionSimulator
from backtesting.report_builder import ReportBuilder

ComponentFactory = Callable[[BacktestConfig, EventBus, Any], Any]
AsyncOrSyncFactory = Callable[..., Any]


# =============================================================================
# Generic lifecycle helpers
# =============================================================================


async def maybe_await(value: Any) -> Any:
    """Await value if it is awaitable, otherwise return it as-is."""

    if inspect.isawaitable(value):
        return await value
    return value


async def call_if_supported(component: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    """Call a sync/async method if the component exposes it."""

    method = getattr(component, method_name, None)
    if not callable(method):
        return None
    return await maybe_await(method(*args, **kwargs))


def stats_if_supported(component: Any) -> dict[str, Any]:
    """Best-effort component stats adapter."""

    method = getattr(component, "stats", None)
    if not callable(method):
        return {}

    try:
        value = method()
    except (RuntimeError, ValueError, TypeError) as exc:
        return {"error": str(exc)}

    if isinstance(value, dict):
        return value

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        return converted if isinstance(converted, dict) else {"value": converted}

    return {"value": value}


def payload_from_event_or_dict(event_or_payload: Any) -> dict[str, Any]:
    """Normalize core Event / dict / object-with-payload into a payload dict."""

    if isinstance(event_or_payload, dict):
        return dict(event_or_payload)

    payload = getattr(event_or_payload, "payload", None)
    if isinstance(payload, dict):
        return dict(payload)

    data = getattr(event_or_payload, "data", None)
    if isinstance(data, dict):
        return dict(data)

    return {}


def topic_from_event_or_dict(event_or_payload: Any, fallback: str = "") -> str:
    """Best-effort extraction of an EventBus topic from an event-like object."""

    for attr in ("topic", "name", "event_type", "type"):
        value = getattr(event_or_payload, attr, None)
        if isinstance(value, str) and value:
            return value

    if isinstance(event_or_payload, dict):
        value = event_or_payload.get("topic") or event_or_payload.get("event_type")
        if value is not None:
            return str(value)

    return fallback


def serialize_for_metadata(value: Any) -> Any:
    """Convert common dataclass/enums/datetimes to metadata-safe values."""

    if value is None:
        return None

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()

    if isinstance(value, datetime):
        return value.isoformat()

    enum_value = getattr(value, "value", None)
    if enum_value is not None and value.__class__.__name__.endswith("Enum"):
        return enum_value

    if isinstance(value, dict):
        return {str(key): serialize_for_metadata(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [serialize_for_metadata(item) for item in value]

    return value


# =============================================================================
# Scheduler compatibility adapter
# =============================================================================


class BacktestSchedulerCompatAdapter:
    """
    Compatibility wrapper around core.scheduler.Scheduler.

    Production components in this codebase have historically used a few
    Scheduler.add_interval_job() call styles. This adapter keeps backtesting
    components deterministic while still delegating to the real core Scheduler.
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
        return stats_if_supported(self._scheduler)

    def add_interval_job(self, *args: Any, **kwargs: Any) -> Any:
        name, func, interval, remaining_args, normalized_kwargs = self._normalize_interval_job_args(
            *args,
            **kwargs,
        )

        target = self._scheduler.add_interval_job
        signature = inspect.signature(target)

        call_kwargs: dict[str, Any] = {
            "name": name,
            "func": func,
            "interval": interval,
            "args": normalized_kwargs.pop("args", remaining_args),
            "kwargs": normalized_kwargs.pop("kwargs", None),
            "run_immediately": normalized_kwargs.pop("run_immediately", False),
            "max_retries": normalized_kwargs.pop("max_retries", 0),
            "retry_delay": normalized_kwargs.pop("retry_delay", 1.0),
            "timeout": normalized_kwargs.pop("timeout", None),
            "allow_overlap": normalized_kwargs.pop("allow_overlap", False),
            "enabled": normalized_kwargs.pop("enabled", True),
        }
        call_kwargs.update(normalized_kwargs)

        if "callback" in signature.parameters and "func" not in signature.parameters:
            call_kwargs["callback"] = call_kwargs.pop("func")
        elif "coro" in signature.parameters and "func" not in signature.parameters:
            call_kwargs["coro"] = call_kwargs.pop("func")

        if "interval_seconds" in signature.parameters and "interval" not in signature.parameters:
            call_kwargs["interval_seconds"] = call_kwargs.pop("interval")
        elif "seconds" in signature.parameters and "interval" not in signature.parameters:
            call_kwargs["seconds"] = call_kwargs.pop("interval")

        if "retry_count" in signature.parameters and "max_retries" not in signature.parameters:
            call_kwargs["retry_count"] = call_kwargs.pop("max_retries")

        accepted = {
            key: value
            for key, value in call_kwargs.items()
            if key in signature.parameters
        }
        return target(**accepted)

    @staticmethod
    def _normalize_interval_job_args(*args: Any, **kwargs: Any) -> tuple[str, Any, float, tuple[Any, ...], dict[str, Any]]:
        name = kwargs.pop("name", None)
        func = (
            kwargs.pop("func", None)
            or kwargs.pop("callback", None)
            or kwargs.pop("coro", None)
        )
        interval = (
            kwargs.pop("interval", None)
            or kwargs.pop("interval_seconds", None)
            or kwargs.pop("seconds", None)
        )

        remaining = list(args)
        if name is None and remaining:
            name = remaining.pop(0)
        if func is None and remaining:
            func = remaining.pop(0)
        if interval is None and remaining:
            interval = remaining.pop(0)

        if name is None:
            raise TypeError("Scheduler job name is required.")
        if func is None or not callable(func):
            raise TypeError("Scheduler job callback is required and must be callable.")
        if interval is None:
            raise TypeError("Scheduler job interval is required.")

        interval_seconds = float(interval.total_seconds()) if hasattr(interval, "total_seconds") else float(interval)
        return str(name), func, interval_seconds, tuple(remaining), kwargs


# =============================================================================
# Runtime DTOs
# =============================================================================


@dataclass(slots=True)
class StrategyTesterStats(SerializableMixin):
    """Runtime stats for StrategyTester."""

    status: BacktestStatus = BacktestStatus.CREATED
    started_at: datetime | None = None
    finished_at: datetime | None = None

    prepared: bool = False
    running: bool = False
    stopped: bool = False

    total_events: int = 0
    replayed_events: int = 0
    event_log_records: int = 0
    signal_records: int = 0
    risk_records: int = 0
    execution_records: int = 0
    position_records: int = 0

    orders: int = 0
    fills: int = 0
    positions: int = 0
    trades: int = 0

    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BacktestCollectors(SerializableMixin):
    """
    Passive event collectors for diagnostics and result building.

    The collector subscribes to EventBus topics but never mutates trading state.
    It is intentionally tolerant of evolving production payload schemas.
    """

    run_id: str
    event_log: list[BacktestEvent] = field(default_factory=list)
    signal_records: list[BacktestSignalRecord] = field(default_factory=list)
    risk_records: list[BacktestRiskDecisionRecord] = field(default_factory=list)
    execution_records: list[BacktestExecutionRecord] = field(default_factory=list)
    position_records: list[BacktestPositionRecord] = field(default_factory=list)
    subscriptions: list[Any] = field(default_factory=list)

    _event_bus: EventBus | None = field(default=None, init=False, repr=False)
    _enabled: bool = field(default=False, init=False, repr=False)

    def register(
        self,
        event_bus: EventBus,
        *,
        collect_event_log: bool = True,
        collect_signal_records: bool = True,
        collect_risk_records: bool = True,
        collect_execution_records: bool = True,
        collect_position_records: bool = True,
    ) -> None:
        """Register passive collector subscriptions."""

        if self._enabled:
            return

        self._event_bus = event_bus

        if collect_event_log:
            self._subscribe("market.*", self._record_event_log)
            self._subscribe("analytics.*", self._record_event_log)
            self._subscribe("strategy.*", self._record_event_log)
            self._subscribe("signal.*", self._record_event_log)
            self._subscribe("risk.*", self._record_event_log)
            self._subscribe("execution.*", self._record_event_log)
            self._subscribe("position.*", self._record_event_log)
            self._subscribe("system.backtest.*", self._record_event_log)

        if collect_signal_records:
            self._subscribe("signal.generated", self._record_signal_generated)
            self._subscribe("signal.confirmed", self._record_signal_confirmed)
            self._subscribe("signal.rejected", self._record_signal_rejected)
            self._subscribe("signal.updated", self._record_signal_updated)

        if collect_risk_records:
            self._subscribe("signal.confirmed", self._record_risk_decision)
            self._subscribe("risk.position_blocked", self._record_risk_decision)
            self._subscribe("risk.kill_switch", self._record_risk_decision)

        if collect_execution_records:
            self._subscribe("execution.*", self._record_execution)

        if collect_position_records:
            self._subscribe("position.*", self._record_position)

        self._enabled = True

    def unregister(self) -> None:
        """Unsubscribe all passive collector subscriptions."""

        if self._event_bus is None:
            self.subscriptions.clear()
            self._enabled = False
            return

        for subscription in list(self.subscriptions):
            try:
                self._event_bus.unsubscribe(subscription)
            except (RuntimeError, ValueError, TypeError, AttributeError):
                pass

        self.subscriptions.clear()
        self._enabled = False

    def _subscribe(self, topic: str, handler: Any) -> None:
        assert self._event_bus is not None

        wrapped = self._wrap_handler(topic, handler)
        name = f"backtest_collector_{topic.replace('.', '_').replace('*', 'wildcard')}"

        try:
            subscription = self._event_bus.subscribe(topic, wrapped, name=name)
        except TypeError:
            subscription = self._event_bus.subscribe(pattern=topic, handler=wrapped, name=name)

        self.subscriptions.append(subscription)

    @staticmethod
    def _wrap_handler(topic: str, handler: Any) -> Any:
        async def _wrapped(event_or_payload: Any) -> None:
            payload = payload_from_event_or_dict(event_or_payload)
            real_topic = topic_from_event_or_dict(event_or_payload, fallback=topic.replace("*", "event"))
            result = handler(real_topic, payload, event_or_payload)
            if inspect.isawaitable(result):
                await result

        return _wrapped

    def _record_event_log(self, topic: str, payload: dict[str, Any], event_or_payload: Any) -> None:
        timestamp_value = payload.get("timestamp_ms") or payload.get("received_at_ms") or timestamp_ms(utcnow())
        event = BacktestEvent(
            run_id=self.run_id,
            event_type=self._event_type_for_topic(topic),
            topic=topic,
            timestamp_ms=int(float(timestamp_value)),
            payload=dict(payload),
            source="event_bus_collector",
            metadata={
                "collector": "BacktestCollectors",
                "raw_event_type": event_or_payload.__class__.__name__,
            },
        )
        self.event_log.append(event)

    def _record_signal_generated(self, topic: str, payload: dict[str, Any], _: Any) -> None:
        self.signal_records.append(self._build_signal_record(topic=topic, payload=payload, status="generated"))

    def _record_signal_confirmed(self, topic: str, payload: dict[str, Any], _: Any) -> None:
        # Also useful for signal analytics. Duplicate signal IDs are allowed at
        # this collection layer because outcome analysis may need lifecycle rows.
        self.signal_records.append(self._build_signal_record(topic=topic, payload=payload, status="confirmed"))

    def _record_signal_rejected(self, topic: str, payload: dict[str, Any], _: Any) -> None:
        # Rejected signals are critical for diagnosing strategy/filter/confluence
        # breakpoints. They must be collected even though they never reach risk.
        self.signal_records.append(self._build_signal_record(topic=topic, payload=payload, status="rejected"))

    def _record_signal_updated(self, topic: str, payload: dict[str, Any], _: Any) -> None:
        status = str(payload.get("status") or "").strip().lower()
        if status == "rejected":
            self._record_signal_rejected(topic, payload, _)
        elif status == "confirmed":
            self._record_signal_confirmed(topic, payload, _)

    def _record_risk_decision(self, topic: str, payload: dict[str, Any], _: Any) -> None:
        try:
            self.risk_records.append(self._build_risk_record(topic=topic, payload=payload))
        except (TypeError, ValueError):
            # Keep collector passive; malformed diagnostic payloads should not
            # interrupt replay.
            return

    def _record_execution(self, topic: str, payload: dict[str, Any], _: Any) -> None:
        try:
            self.execution_records.append(self._build_execution_record(topic=topic, payload=payload))
        except (TypeError, ValueError):
            return

    def _record_position(self, topic: str, payload: dict[str, Any], _: Any) -> None:
        try:
            self.position_records.append(self._build_position_record(topic=topic, payload=payload))
        except (TypeError, ValueError):
            return

    def _build_signal_record(self, *, topic: str, payload: dict[str, Any], status: str) -> BacktestSignalRecord:
        kwargs = self._accepted_kwargs(
            BacktestSignalRecord,
            {
                "run_id": self.run_id,
                "signal_id": self._str_first(payload, "signal_id", "id", default=""),
                "strategy_name": self._str_first(payload, "strategy_name", "strategy", default=None),
                "symbol": self._str_first(payload, "symbol", default=None),
                "exchange": self._str_first(payload, "exchange", default=None),
                "market_type": self._str_first(payload, "market_type", default=None),
                "timeframe": self._str_first(payload, "timeframe", default=None),
                "side": self._str_first(payload, "side", "direction", default=None),
                "confidence": self._float_first(payload, "confidence", "score", default=0.0),
                "generated_at_ms": self._int_first(payload, "timestamp_ms", "created_at_ms", default=timestamp_ms(utcnow())) if status == "generated" else None,
                "confirmed_at_ms": self._int_first(payload, "timestamp_ms", "created_at_ms", default=timestamp_ms(utcnow())) if status == "confirmed" else None,
                "outcome": self._signal_outcome_for_status(status),
                "payload": dict(payload),
                "metadata": {"source_topic": topic, "run_id": self.run_id},
            },
        )
        return BacktestSignalRecord(**kwargs)

    @staticmethod
    def _signal_outcome_for_status(status: str) -> SignalOutcome:
        normalized = str(status).strip().lower()

        if normalized == "confirmed":
            return SignalOutcome.CONFIRMED_BY_RISK

        if normalized == "rejected":
            for attr in (
                "REJECTED",
                "REJECTED_BY_STRATEGY",
                "REJECTED_BY_FILTER",
                "BLOCKED",
            ):
                value = getattr(SignalOutcome, attr, None)
                if value is not None:
                    return value

        return SignalOutcome.GENERATED

    def _build_risk_record(self, *, topic: str, payload: dict[str, Any]) -> BacktestRiskDecisionRecord:
        kwargs = self._accepted_kwargs(
            BacktestRiskDecisionRecord,
            {
                "run_id": self.run_id,
                "signal_id": self._str_first(payload, "signal_id", default=None),
                "strategy_name": self._str_first(payload, "strategy_name", "strategy", default=None),
                "symbol": self._str_first(payload, "symbol", default=None),
                "side": self._str_first(payload, "side", "direction", default=None),
                "approved": topic == "signal.confirmed" or bool(payload.get("approved", False)),
                "blocked": topic in {"risk.position_blocked", "risk.kill_switch"} or bool(payload.get("blocked", False)),
                "reason": self._str_first(payload, "reason", "block_reason", default=None),
                "final_size": self._float_first(payload, "final_size", "size", "quantity", default=0.0),
                "final_leverage": self._float_first(payload, "final_leverage", "leverage", default=0.0),
                "final_margin": self._float_first(payload, "final_margin", "margin", default=0.0),
                "final_notional": self._float_first(payload, "final_notional", "notional", default=0.0),
                "timestamp_ms": self._int_first(payload, "timestamp_ms", "created_at_ms", default=timestamp_ms(utcnow())),
                "payload": dict(payload),
                "metadata": {"source_topic": topic, "run_id": self.run_id},
            },
        )
        return BacktestRiskDecisionRecord(**kwargs)

    def _build_execution_record(self, *, topic: str, payload: dict[str, Any]) -> BacktestExecutionRecord:
        kwargs = self._accepted_kwargs(
            BacktestExecutionRecord,
            {
                "run_id": self.run_id,
                "record_id": self._str_first(payload, "record_id", "event_id", default=""),
                "order_id": self._str_first(payload, "order_id", default=None),
                "fill_id": self._str_first(payload, "fill_id", default=None),
                "signal_id": self._str_first(payload, "signal_id", default=None),
                "strategy_name": self._str_first(payload, "strategy_name", "strategy", default=None),
                "symbol": self._str_first(payload, "symbol", default=None),
                "side": self._str_first(payload, "side", default=None),
                "status": self._simulated_order_status(payload, topic),

                "timestamp_ms": self._int_first(payload, "timestamp_ms", "created_at_ms", default=timestamp_ms(utcnow())),
                "payload": dict(payload),
                "metadata": {"source_topic": topic, "run_id": self.run_id},
            },
        )
        return BacktestExecutionRecord(**kwargs)

    def _build_position_record(self, *, topic: str, payload: dict[str, Any]) -> BacktestPositionRecord:
        kwargs = self._accepted_kwargs(
            BacktestPositionRecord,
            {
                "run_id": self.run_id,
                "record_id": self._str_first(payload, "record_id", "event_id", default=""),
                "position_id": self._str_first(payload, "position_id", default=None),
                "signal_id": self._str_first(payload, "signal_id", default=None),
                "strategy_name": self._str_first(payload, "strategy_name", "strategy", default=None),
                "symbol": self._str_first(payload, "symbol", default=None),
                "side": self._str_first(payload, "side", default=None),
                "status": self._simulated_position_status(payload, topic),
                "timestamp_ms": self._int_first(payload, "timestamp_ms", "created_at_ms", default=timestamp_ms(utcnow())),
                "payload": dict(payload),
                "metadata": {"source_topic": topic, "run_id": self.run_id},
            },
        )
        return BacktestPositionRecord(**kwargs)


    @staticmethod
    def _simulated_order_status(payload: dict[str, Any], topic: str) -> SimulatedOrderStatus | None:
        raw = str(payload.get("status") or topic.rsplit(".", 1)[-1]).lower()
        aliases = {
            "order_submitted": SimulatedOrderStatus.SUBMITTED,
            "submitted": SimulatedOrderStatus.SUBMITTED,
            "order_accepted": SimulatedOrderStatus.ACCEPTED,
            "accepted": SimulatedOrderStatus.ACCEPTED,
            "order_rejected": SimulatedOrderStatus.REJECTED,
            "rejected": SimulatedOrderStatus.REJECTED,
            "order_failed": SimulatedOrderStatus.FAILED,
            "failed": SimulatedOrderStatus.FAILED,
            "order_cancelled": SimulatedOrderStatus.CANCELLED,
            "cancelled": SimulatedOrderStatus.CANCELLED,
            "order_filled": SimulatedOrderStatus.FILLED,
            "filled": SimulatedOrderStatus.FILLED,
            "order_partially_filled": SimulatedOrderStatus.PARTIALLY_FILLED,
            "partially_filled": SimulatedOrderStatus.PARTIALLY_FILLED,
        }
        return aliases.get(raw)

    @staticmethod
    def _simulated_position_status(payload: dict[str, Any], topic: str) -> SimulatedPositionStatus | None:
        raw = str(payload.get("status") or topic.rsplit(".", 1)[-1]).lower()
        aliases = {
            "position_opened": SimulatedPositionStatus.OPEN,
            "opened": SimulatedPositionStatus.OPEN,
            "open": SimulatedPositionStatus.OPEN,
            "position_updated": SimulatedPositionStatus.OPEN,
            "updated": SimulatedPositionStatus.OPEN,
            "position_closed": SimulatedPositionStatus.CLOSED,
            "closed": SimulatedPositionStatus.CLOSED,
            "position_liquidated": SimulatedPositionStatus.LIQUIDATED,
            "liquidated": SimulatedPositionStatus.LIQUIDATED,
            "reducing": SimulatedPositionStatus.REDUCING,
            "closing": SimulatedPositionStatus.CLOSING,
        }
        return aliases.get(raw)

    @staticmethod
    def _event_type_for_topic(topic: str) -> BacktestEventType:
        if topic.startswith("market."):
            return BacktestEventType.MARKET
        if topic.startswith("analytics."):
            return BacktestEventType.ANALYTICS
        if topic.startswith("strategy."):
            return BacktestEventType.STRATEGY
        if topic.startswith("signal."):
            return BacktestEventType.SIGNAL
        if topic.startswith("risk."):
            return BacktestEventType.RISK
        if topic.startswith("execution."):
            return BacktestEventType.EXECUTION
        if topic.startswith("position."):
            return BacktestEventType.POSITION
        return BacktestEventType.SYSTEM

    @staticmethod
    def _accepted_kwargs(cls: type[Any], values: dict[str, Any]) -> dict[str, Any]:
        signature = inspect.signature(cls)
        return {
            key: value
            for key, value in values.items()
            if key in signature.parameters
        }

    @staticmethod
    def _str_first(payload: dict[str, Any], *keys: str, default: str | None = None) -> str | None:
        for key in keys:
            value = payload.get(key)
            if value is not None:
                return str(value)
        return default

    @staticmethod
    def _float_first(payload: dict[str, Any], *keys: str, default: float = 0.0) -> float:
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return default

    @staticmethod
    def _int_first(payload: dict[str, Any], *keys: str, default: int = 0) -> int:
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            try:
                return int(float(value))
            except (TypeError, ValueError):
                continue
        return default


@dataclass(slots=True)
class BacktestEventFlowDebugMonitor(SerializableMixin):
    """
    Passive EventBus flow monitor for debugging full-pipeline backtests.

    It answers the most important integration question:

        market.* -> data -> market.*.updated -> analytics.*
        -> signal.* -> risk.* -> execution.* -> position.*

    The monitor never mutates trading state and never raises from handlers.
    """

    run_id: str
    max_samples_per_topic: int = 3
    max_recent_events: int = 80
    print_summary: bool = True

    subscriptions: list[Any] = field(default_factory=list)
    topic_counts: Counter[str] = field(default_factory=Counter)
    group_counts: Counter[str] = field(default_factory=Counter)
    samples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    recent_events: deque[dict[str, Any]] = field(default_factory=deque)
    replay_stats_snapshot: dict[str, Any] = field(default_factory=dict)
    dataset_events: int = 0

    _event_bus: EventBus | None = field(default=None, init=False, repr=False)
    _enabled: bool = field(default=False, init=False, repr=False)
    _recent_event_keys: deque[tuple[Any, ...]] = field(default_factory=deque, init=False, repr=False)
    _recent_event_key_set: set[tuple[Any, ...]] = field(default_factory=set, init=False, repr=False)

    TOPIC_PATTERNS: tuple[str, ...] = (
        # Raw market topics emitted by MarketReplay. Keep explicit topics because
        # not every EventBus wildcard implementation treats "market.*" the same.
        "market.candle",
        "market.trade",
        "market.orderbook",
        "market.orderbook.snapshot",
        "market.funding",
        "market.open_interest",
        "market.liquidation",

        # Data-cache updated topics.
        "market.candles.updated",
        "market.candle.closed",
        "market.trades.updated",
        "market.orderbook.updated",
        "market.funding.updated",
        "market.open_interest.updated",
        "market.liquidations.updated",

        # Wildcards are still useful for broad diagnostics where supported.
        "market.*",
        "analytics.*",
        "strategy.*",
        "signal.*",
        "risk.*",
        "execution.*",
        "position.*",
        "system.backtest.*",
        "system.strategy.*",
        "system.risk.*",
        "system.execution.*",
    )

    def register(self, event_bus: EventBus) -> None:
        if self._enabled:
            return

        self._event_bus = event_bus
        for pattern in self.TOPIC_PATTERNS:
            self._subscribe(pattern)

        self._enabled = True

    def unregister(self) -> None:
        if self._event_bus is None:
            self.subscriptions.clear()
            self._enabled = False
            return

        for subscription in list(self.subscriptions):
            try:
                self._event_bus.unsubscribe(subscription)
            except (RuntimeError, ValueError, TypeError, AttributeError):
                pass

        self.subscriptions.clear()
        self._enabled = False

    def _subscribe(self, pattern: str) -> None:
        assert self._event_bus is not None

        name = f"backtest_flow_debug_{pattern.replace('.', '_').replace('*', 'wildcard')}"
        handler = self._wrap_handler(pattern)

        try:
            subscription = self._event_bus.subscribe(pattern, handler, name=name)
        except TypeError:
            subscription = self._event_bus.subscribe(pattern=pattern, handler=handler, name=name)

        self.subscriptions.append(subscription)

    def _wrap_handler(self, fallback_pattern: str) -> Any:
        async def _wrapped(event_or_payload: Any) -> None:
            try:
                payload = payload_from_event_or_dict(event_or_payload)
                topic = topic_from_event_or_dict(
                    event_or_payload,
                    fallback=fallback_pattern.replace("*", "event"),
                )
                self.record(topic=topic, payload=payload)
            except (RuntimeError, ValueError, TypeError, AttributeError):
                # Debug monitor must never interrupt backtest replay.
                return

        return _wrapped

    def record(self, *, topic: str, payload: dict[str, Any]) -> None:
        event_key = self._event_key(topic=topic, payload=payload)
        if event_key in self._recent_event_key_set:
            return

        self._recent_event_keys.append(event_key)
        self._recent_event_key_set.add(event_key)
        while len(self._recent_event_keys) > max(self.max_recent_events * 4, 256):
            old_key = self._recent_event_keys.popleft()
            self._recent_event_key_set.discard(old_key)

        group = self.group_for_topic(topic)

        self.topic_counts[topic] += 1
        self.group_counts[group] += 1

        if topic not in self.samples:
            self.samples[topic] = []

        if len(self.samples[topic]) < self.max_samples_per_topic:
            self.samples[topic].append(self.compact_payload(payload))

        self.recent_events.append(
            {
                "topic": topic,
                "group": group,
                "symbol": self._first_str(payload, "symbol"),
                "timeframe": self._first_str(payload, "timeframe"),
                "strategy_name": self._first_str(payload, "strategy_name", "strategy"),
                "signal_id": self._first_str(payload, "signal_id", "id"),
                "timestamp_ms": self._first_value(payload, "timestamp_ms", "received_at_ms", "created_at_ms"),
            }
        )

        while len(self.recent_events) > self.max_recent_events:
            self.recent_events.popleft()

    @staticmethod
    def group_for_topic(topic: str) -> str:
        if topic.startswith("market.") and topic.endswith(".updated"):
            return "market.updated"
        if topic == "market.candle.closed":
            return "market.candle.closed"
        if topic.startswith("market."):
            return "market.raw"
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
    def _event_key(*, topic: str, payload: dict[str, Any]) -> tuple[Any, ...]:
        return (
            topic,
            payload.get("event_id"),
            payload.get("signal_id") or payload.get("id"),
            payload.get("order_id"),
            payload.get("position_id"),
            payload.get("symbol"),
            payload.get("timeframe"),
            payload.get("timestamp_ms") or payload.get("received_at_ms") or payload.get("created_at_ms"),
            payload.get("open_time_ms"),
            payload.get("status"),
            payload.get("side") or payload.get("direction"),
            payload.get("price") or payload.get("close"),
        )

    @staticmethod
    def compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "exchange",
            "market_type",
            "symbol",
            "timeframe",
            "strategy_name",
            "strategy",
            "signal_id",
            "id",
            "side",
            "direction",
            "confidence",
            "score",
            "status",
            "reason",
            "timestamp_ms",
            "received_at_ms",
            "created_at_ms",
            "open_time_ms",
            "close_time_ms",
            "price",
            "close",
            "funding_rate",
            "open_interest",
            "open_interest_value",
        )

        compact: dict[str, Any] = {}
        for key in keys:
            if key in payload and payload[key] is not None:
                compact[key] = payload[key]

        # Include lightweight feature/debug hints without dumping huge payloads.
        for key in ("features", "metadata", "source", "source_topic"):
            value = payload.get(key)
            if isinstance(value, dict):
                compact[key] = sorted(str(item) for item in value.keys())[:20]
            elif value is not None and key not in compact:
                compact[key] = str(value)[:160]

        return compact

    @staticmethod
    def _first_value(payload: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            value = payload.get(key)
            if value is not None:
                return value
        return None

    @classmethod
    def _first_str(cls, payload: dict[str, Any], *keys: str) -> str | None:
        value = cls._first_value(payload, *keys)
        return str(value) if value is not None else None

    def attach_replay_stats(
        self,
        *,
        replay_stats: Any | None = None,
        dataset_events: int = 0,
    ) -> None:
        """
        Attach MarketReplay stats as a fallback diagnostic.

        This prevents a false breakpoint like "MarketReplay emitted no market.*"
        when replay did emit events but the debug monitor failed to catch topics
        because of EventBus wildcard/subscription differences.
        """

        self.dataset_events = int(dataset_events or 0)

        if replay_stats is None:
            self.replay_stats_snapshot = {}
            return

        if isinstance(replay_stats, dict):
            self.replay_stats_snapshot = dict(replay_stats)
            return

        to_dict = getattr(replay_stats, "to_dict", None)
        if callable(to_dict):
            try:
                converted = to_dict()
                if isinstance(converted, dict):
                    self.replay_stats_snapshot = converted
                    return
            except (TypeError, ValueError, AttributeError, RuntimeError):
                pass

        snapshot: dict[str, Any] = {}
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
            "current_timestamp_ms",
            "status",
            "last_error",
        ):
            value = getattr(replay_stats, key, None)
            if value is not None:
                if hasattr(value, "value"):
                    value = value.value
                snapshot[key] = value

        self.replay_stats_snapshot = snapshot

    @property
    def replay_emitted_events(self) -> int:
        value = self.replay_stats_snapshot.get("emitted_events")
        if isinstance(value, int):
            return value
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0

    @property
    def replay_processed_events(self) -> int:
        value = self.replay_stats_snapshot.get("processed_events")
        if isinstance(value, int):
            return value
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0

    @property
    def last_stage(self) -> str:
        """
        Return the last *trading-flow* stage reached.

        Lifecycle-only risk/strategy events such as risk.manager.started or
        strategy.engine.started must not make the flow look healthier than it
        is. A risk stage is considered reached only after signal.* has appeared
        or after an explicit trading risk topic is observed.
        """

        market_updated = (
            self.group_counts.get("market.updated", 0)
            + self.group_counts.get("market.candle.closed", 0)
        )

        if self.group_counts.get("position", 0) > 0:
            return "position"
        if self.group_counts.get("execution", 0) > 0:
            return "execution"
        if self.group_counts.get("signal", 0) > 0:
            if self._has_trading_risk_topic():
                return "risk"
            return "signal"
        if self._has_trading_risk_topic():
            return "risk"
        if self.group_counts.get("analytics", 0) > 0:
            return "analytics"
        if market_updated > 0:
            return "market.updated"
        if self.group_counts.get("market.raw", 0) > 0:
            return "market.raw"
        return "none"

    def _has_trading_risk_topic(self) -> bool:
        trading_risk_topics = {
            "signal.confirmed",
            "risk.position_blocked",
            "risk.kill_switch",
            "risk.limit_warning",
        }
        return any(self.topic_counts.get(topic, 0) > 0 for topic in trading_risk_topics)

    @property
    def suspected_breakpoint(self) -> str:
        raw_market = self.group_counts.get("market.raw", 0)
        market_updated = self.group_counts.get("market.updated", 0) + self.group_counts.get("market.candle.closed", 0)
        analytics = self.group_counts.get("analytics", 0)
        signal = self.group_counts.get("signal", 0)
        risk = self.group_counts.get("risk", 0)
        execution = self.group_counts.get("execution", 0)
        position = self.group_counts.get("position", 0)

        # Fallback: MarketReplay can prove it emitted events even when this
        # passive monitor did not catch market.* due to subscription/wildcard
        # differences. In that case the breakpoint is in diagnostics wiring,
        # not in replay itself.
        if raw_market <= 0 and self.replay_emitted_events > 0:
            if market_updated > 0 or analytics > 0:
                # Downstream stages prove market events were emitted.
                pass
            else:
                return (
                    "Debug monitor не зафіксував market.* topics, але "
                    f"MarketReplay.stats.emitted_events={self.replay_emitted_events}. "
                    "Перевір EventBus wildcard/exact subscriptions або порядок реєстрації monitor-а."
                )

        if raw_market <= 0:
            if self.replay_processed_events > 0 or self.dataset_events > 0:
                return (
                    "Debug monitor не зафіксував market.* topics. "
                    f"dataset_events={self.dataset_events}, "
                    f"replay_processed_events={self.replay_processed_events}, "
                    f"replay_emitted_events={self.replay_emitted_events}. "
                    "Це схоже на проблему debug-підписок, а не обов’язково MarketReplay."
                )
            return "MarketReplay не емітив market.* події."

        if market_updated <= 0:
            return "Data cache layer не емітив market.*.updated / market.candle.closed."
        if analytics <= 0:
            return "Analytics layer не емітив analytics.* після data cache updates."
        if signal <= 0:
            return "Strategy layer не емітив signal.* після analytics.*."
        if risk <= 0:
            return "Risk layer не емітив risk.* / signal.confirmed після signal.*."
        if execution <= 0:
            return "Execution simulator не емітив execution.* після risk/signal.confirmed."
        if position <= 0:
            return "Position simulator не емітив position.* після execution fills."
        return "Повний ланцюг дійшов до position.*."

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "group_counts": dict(self.group_counts),
            "topic_counts": dict(self.topic_counts),
            "last_stage": self.last_stage,
            "suspected_breakpoint": self.suspected_breakpoint,
            "samples": self.samples,
            "recent_events": list(self.recent_events),
            "dataset_events": self.dataset_events,
            "replay_stats": dict(self.replay_stats_snapshot),
        }

    def format_summary(self) -> str:
        lines: list[str] = []
        lines.append("")
        lines.append("========== EVENT FLOW DEBUG ==========")
        lines.append("Stage counts:")
        for group in (
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
            lines.append(f"- {group}: {self.group_counts.get(group, 0)}")

        lines.append("")
        lines.append(f"Last reached stage: {self.last_stage}")
        lines.append(f"Suspected breakpoint: {self.suspected_breakpoint}")

        lines.append("")
        lines.append("Replay stats fallback:")
        lines.append(f"- dataset_events: {self.dataset_events}")
        lines.append(f"- replay_processed_events: {self.replay_processed_events}")
        lines.append(f"- replay_emitted_events: {self.replay_emitted_events}")
        lines.append(f"- replay_stats: {self.replay_stats_snapshot}")

        lines.append("")
        lines.append("Top topics:")
        if self.topic_counts:
            for topic, count in self.topic_counts.most_common(25):
                lines.append(f"- {topic}: {count}")
        else:
            lines.append("- <none>")

        lines.append("")
        lines.append("Payload samples:")
        if self.samples:
            for topic, samples in sorted(self.samples.items()):
                lines.append(f"- {topic}:")
                for sample in samples[: self.max_samples_per_topic]:
                    lines.append(f"  {sample}")
        else:
            lines.append("- <none>")

        lines.append("")
        lines.append("Recent events:")
        if self.recent_events:
            for event in list(self.recent_events)[-20:]:
                lines.append(f"- {event}")
        else:
            lines.append("- <none>")

        lines.append("======================================")
        return "\n".join(lines)



@dataclass(slots=True)
class BacktestComponentBundle(SerializableMixin):
    """Runtime component bundle owned or coordinated by StrategyTester."""

    event_bus: EventBus | None = None
    scheduler: Any | None = None
    clock: BacktestClock | None = None
    market_replay: MarketReplay | None = None
    cost_model: TradingCostModel | None = None
    execution_simulator: ExecutionSimulator | None = None
    position_simulator: PositionSimulator | None = None
    data_caches: list[Any] = field(default_factory=list)
    analytics_components: list[Any] = field(default_factory=list)
    strategy_engine: Any | None = None
    signal_processor: Any | None = None
    risk_manager: Any | None = None
    collectors: BacktestCollectors | None = None
    debug_monitor: BacktestEventFlowDebugMonitor | None = None
    owned_components: list[Any] = field(default_factory=list)
    started_components: list[Any] = field(default_factory=list)


# =============================================================================
# StrategyTester
# =============================================================================


class StrategyTester:
    """
    Full-pipeline offline strategy/system tester.

    This class wires production components and backtesting simulators. It does
    not implement analytics, strategy decisioning, risk approval, simulated
    fills or position accounting itself.
    """

    component_name = "StrategyTester"

    def __init__(
        self,
        config: BacktestConfig | None = None,
        dataset: BacktestDataset | None = None,
        *,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | Any | None = None,
        clock: BacktestClock | None = None,
        market_replay: MarketReplay | None = None,
        cost_model: TradingCostModel | None = None,
        execution_simulator: ExecutionSimulator | None = None,
        position_simulator: PositionSimulator | None = None,
        data_caches: Sequence[Any] | None = None,
        analytics_components: Sequence[Any] | None = None,
        strategy_engine: Any | None = None,
        signal_processor: Any | None = None,
        risk_manager: Any | None = None,
        performance_metrics: PerformanceMetrics | None = None,
        model_analytics: BacktestModelAnalyticsEngine | None = None,
        report_builder: ReportBuilder | None = None,
        data_cache_factory: ComponentFactory | None = None,
        analytics_factory: ComponentFactory | None = None,
        strategy_engine_factory: AsyncOrSyncFactory | None = None,
        signal_processor_factory: AsyncOrSyncFactory | None = None,
        risk_manager_factory: AsyncOrSyncFactory | None = None,
        logger_name: str = "backtesting.strategy_tester",
    ) -> None:
        self.config = config or BacktestConfig()
        self.config.validate()
        self.tester_config: StrategyTesterConfig = self.config.strategy_tester

        self.dataset = dataset
        self.logger = get_logger(logger_name)

        self.components = BacktestComponentBundle(
            event_bus=event_bus,
            scheduler=scheduler,
            clock=clock,
            market_replay=market_replay,
            cost_model=cost_model,
            execution_simulator=execution_simulator,
            position_simulator=position_simulator,
            data_caches=list(data_caches or []),
            analytics_components=list(analytics_components or []),
            strategy_engine=strategy_engine,
            signal_processor=signal_processor,
            risk_manager=risk_manager,
        )

        self.performance_metrics = performance_metrics or PerformanceMetrics(self.config.performance_metrics)
        self.model_analytics = model_analytics or BacktestModelAnalyticsEngine(self.config.model_analytics)
        self.report_builder = report_builder or ReportBuilder(self.config.report_builder)

        self.data_cache_factory = data_cache_factory
        self.analytics_factory = analytics_factory
        self.strategy_engine_factory = strategy_engine_factory
        self.signal_processor_factory = signal_processor_factory
        self.risk_manager_factory = risk_manager_factory

        self.stats_state = StrategyTesterStats(status=BacktestStatus.CREATED)
        self.result: BacktestResult | None = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public lifecycle API
    # ------------------------------------------------------------------

    async def prepare(self, dataset: BacktestDataset | None = None) -> None:
        """Build runtime components and register all subscriptions."""

        async with self._lock:
            if self.stats_state.prepared:
                return

            if dataset is not None:
                self.dataset = dataset

            if self.dataset is None:
                self.dataset = DataLoader(self.config.data_loader).load_dataset(
                    period=self.config.period(),
                )

            if self.dataset.is_empty:
                raise BacktestLifecycleError("StrategyTester cannot run with an empty BacktestDataset.")

            self._build_initial_result()
            self._build_core_runtime()
            self._build_clock()
            self._build_market_replay()
            await self._build_or_validate_pipeline_components()
            self._build_simulators()
            self._build_collectors()
            self._build_debug_monitor()

            self._assert_required_components()
            self._assert_no_live_execution_components()
            self._rebind_component_runtime_handles()
            self._prepare_market_replay()
            self._register_components()

            self.stats_state.prepared = True
            self.stats_state.status = BacktestStatus.CONFIGURING
            self.stats_state.total_events = len(self.dataset.events)

    async def start(self) -> None:
        """Start EventBus, Scheduler and all pipeline components."""

        if not self.stats_state.prepared:
            await self.prepare()

        async with self._lock:
            if self.stats_state.running:
                return

            assert self.result is not None
            self.result.mark_started()
            self.stats_state.status = BacktestStatus.RUNNING
            self.stats_state.started_at = utcnow()
            self.stats_state.running = True
            self.stats_state.stopped = False

        await self._start_components()
        await self._emit_best_effort(
            "system.backtest.strategy_tester.started",
            {
                "run_id": self.result.run_id if self.result else None,
                "run_name": self.config.run_name,
                "events": len(self.dataset.events) if self.dataset else 0,
            },
        )

    async def run(
        self,
        dataset: BacktestDataset | None = None,
        *,
        component_kwargs: dict[str, Any] | None = None,
    ) -> BacktestResult:
        """
        Run a full backtest and return BacktestResult.

        component_kwargs is accepted for Optimizer/WalkForward compatibility.
        Values are not interpreted here unless they correspond to injectable
        StrategyTester attributes.
        """

        self._apply_component_kwargs(component_kwargs or {})

        await self.prepare(dataset=dataset)
        await self.start()

        try:
            await self._run_replay()

            # EventBus may deliver handlers through an internal async queue.
            # Replay can finish before downstream analytics/strategy/risk
            # handlers have fully drained, so always wait before diagnostics and
            # result collection.
            await self._drain_event_bus(reason="after_replay")

            self._attach_replay_debug_stats()
            self._print_debug_checkpoint("after_replay")
            result = await self._collect_result(status=BacktestStatus.COMPLETED)
            await self._emit_best_effort(
                "system.backtest.strategy_tester.completed",
                {
                    "run_id": result.run_id,
                    "run_name": result.run_name,
                    "status": result.status.value,
                    "net_profit": result.net_profit,
                    "net_profit_pct": result.net_profit_pct,
                    "trades": len(result.trades),
                },
            )
            await self._drain_event_bus(reason="after_completed_event")
            self._attach_result_debug_metadata()
            return result

        except (BacktestLifecycleError, BacktestDependencyError, BacktestComponentError, BacktestResultCollectionError) as exc:
            result = await self._handle_run_failure(exc)
            if self.config.fail_fast or not self.config.allow_partial_results:
                raise
            return result

        except Exception as exc:
            result = await self._handle_run_failure(exc)
            if self.config.fail_fast or not self.config.allow_partial_results:
                raise BacktestLifecycleError(
                    "StrategyTester run failed.",
                    details={"error": str(exc), "error_type": exc.__class__.__name__},
                ) from exc
            return result

        finally:
            await self.stop()
            if self.tester_config.cleanup_after_run:
                await self.cleanup()

    async def stop(self) -> None:
        """Stop all started components in reverse order."""

        async with self._lock:
            if self.stats_state.stopped:
                return
            self.stats_state.running = False
            self.stats_state.stopped = True
            self.stats_state.finished_at = utcnow()

        await self._stop_components_reverse_order()
        await self._drain_event_bus(reason="after_component_stop")

        if self.components.collectors is not None:
            self.components.collectors.unregister()
        if self.components.debug_monitor is not None:
            self._attach_replay_debug_stats()
            self._attach_result_debug_metadata()
            if self.components.debug_monitor.print_summary:
                print(self.components.debug_monitor.format_summary())
            self.components.debug_monitor.unregister()

        await self._emit_best_effort(
            "system.backtest.strategy_tester.stopped",
            self.stats(),
        )

    async def reset(self) -> None:
        """Reset tester-owned runtime state and simulator state."""

        await self.stop()

        for component in (
            self.components.execution_simulator,
            self.components.position_simulator,
            self.components.market_replay,
        ):
            if component is not None:
                await call_if_supported(component, "reset")

        self.result = None
        self.stats_state = StrategyTesterStats(status=BacktestStatus.CREATED)
        self.components.collectors = None
        self.components.debug_monitor = None
        self.components.started_components.clear()

    async def cleanup(self) -> None:
        """Best-effort cleanup hook for components that expose cleanup()."""

        for component in reversed(self._all_components_for_lifecycle(include_replay=False)):
            await call_if_supported(component, "cleanup")

    def stats(self) -> dict[str, Any]:
        payload = self.stats_state.to_dict()
        payload.update(
            {
                "run_id": self.result.run_id if self.result else None,
                "run_name": self.config.run_name,
                "dataset_events": len(self.dataset.events) if self.dataset else 0,
                "components": {
                    "event_bus": self.components.event_bus.__class__.__name__ if self.components.event_bus else None,
                    "scheduler": self.components.scheduler.__class__.__name__ if self.components.scheduler else None,
                    "market_replay": stats_if_supported(self.components.market_replay),
                    "execution_simulator": stats_if_supported(self.components.execution_simulator),
                    "position_simulator": stats_if_supported(self.components.position_simulator),
                    "data_caches": [stats_if_supported(item) for item in self.components.data_caches],
                    "analytics": [stats_if_supported(item) for item in self.components.analytics_components],
                    "strategy_engine": stats_if_supported(self.components.strategy_engine),
                    "signal_processor": stats_if_supported(self.components.signal_processor),
                    "risk_manager": stats_if_supported(self.components.risk_manager),
                },
                "event_flow_debug": self.components.debug_monitor.to_dict() if self.components.debug_monitor else None,
            }
        )
        return payload

    # ------------------------------------------------------------------
    # Build phase
    # ------------------------------------------------------------------

    def _build_initial_result(self) -> None:
        assert self.dataset is not None

        self.result = BacktestResult(
            run_name=self.config.run_name,
            mode=self.config.mode,
            period=self.dataset.info.period or self.config.period(),
            dataset_info=self.dataset.info,
            initial_balance=self.config.initial_balance,
            final_balance=self.config.initial_balance,
            final_equity=self.config.initial_balance,
            metadata={
                "exchange": self.config.exchange,
                "market_type": self.config.market_type,
                "symbols": list(self.config.symbols),
                "timeframes": list(self.config.timeframes),
                "strategy_tester": dict(self.tester_config.metadata),
            },
        )

    def _build_core_runtime(self) -> None:
        if self.components.event_bus is None:
            self.components.event_bus = EventBus()
            self.components.owned_components.append(self.components.event_bus)

        if self.components.scheduler is None:
            try:
                scheduler = Scheduler(event_bus=self.components.event_bus)
            except TypeError:
                scheduler = Scheduler()
            self.components.scheduler = BacktestSchedulerCompatAdapter(scheduler)
            self.components.owned_components.append(self.components.scheduler)
            return

        if not isinstance(self.components.scheduler, BacktestSchedulerCompatAdapter):
            wrap_external = os.getenv("BACKTEST_WRAP_EXTERNAL_SCHEDULER", "1").strip().lower()
            if wrap_external not in {"0", "false", "no", "off"}:
                self.components.scheduler = BacktestSchedulerCompatAdapter(self.components.scheduler)

    def _build_clock(self) -> None:
        if self.components.clock is not None:
            return

        assert self.dataset is not None
        period = self.dataset.info.period or self.config.period()
        self.components.clock = BacktestClock(period=period, config=self.config.backtest_time)
        self.components.owned_components.append(self.components.clock)

    def _build_market_replay(self) -> None:
        if self.components.market_replay is None:
            self.components.market_replay = MarketReplay(
                config=self.config.market_replay,
                event_bus=self.components.event_bus,
                clock=self.components.clock,
            )
            self.components.owned_components.append(self.components.market_replay)
        else:
            self.components.market_replay.event_bus = self.components.event_bus
            self.components.market_replay.clock = self.components.clock

    async def _build_or_validate_pipeline_components(self) -> None:
        assert self.components.event_bus is not None

        if not self.components.data_caches and self.tester_config.use_production_data_caches:
            if self.data_cache_factory is not None:
                created = await maybe_await(
                    self.data_cache_factory(self.config, self.components.event_bus, self.components.scheduler)
                )
                self.components.data_caches = self._as_component_list(created)
            else:
                self.components.data_caches = self._build_default_data_caches()

        if not self.components.analytics_components and self.tester_config.use_production_analytics:
            if self.analytics_factory is not None:
                created = await maybe_await(
                    self.analytics_factory(self.config, self.components.event_bus, self.components.scheduler)
                )
                self.components.analytics_components = self._as_component_list(created)

        if self.components.signal_processor is None and self.tester_config.use_production_strategy_engine:
            if self.signal_processor_factory is not None:
                self.components.signal_processor = await maybe_await(
                    self.signal_processor_factory(
                        config=self.config,
                        event_bus=self.components.event_bus,
                        scheduler=self.components.scheduler,
                    )
                )

        if self.components.strategy_engine is None and self.tester_config.use_production_strategy_engine:
            if self.strategy_engine_factory is not None:
                self.components.strategy_engine = await maybe_await(
                    self.strategy_engine_factory(
                        config=self.config,
                        event_bus=self.components.event_bus,
                        scheduler=self.components.scheduler,
                        signal_processor=self.components.signal_processor,
                    )
                )

        if self.components.risk_manager is None and self.tester_config.use_production_risk_manager:
            if self.risk_manager_factory is not None:
                self.components.risk_manager = await maybe_await(
                    self.risk_manager_factory(
                        config=self.config,
                        event_bus=self.components.event_bus,
                        scheduler=self.components.scheduler,
                    )
                )

    def _build_default_data_caches(self) -> list[Any]:
        """
        Build production data caches if available.

        Imports are local on purpose: backtesting can still be imported in
        environments where production data package is not installed.
        """

        try:
            from data.candles_cache import CandlesCache
            from data.funding_cache import FundingCache
            from data.open_interest_cache import OpenInterestCache
            from data.orderbook_cache import OrderBookCache
            from data.trades_cache import TradesCache
        except ImportError as exc:
            if self.tester_config.require_analytics or self.tester_config.use_production_data_caches:
                raise BacktestDependencyError(
                    "Production data caches are required but could not be imported.",
                    details={"error": str(exc)},
                ) from exc
            return []

        assert self.components.event_bus is not None

        core_config = self.config.core_config
        caches: list[Any] = []

        constructors = [CandlesCache]
        if self.config.use_trades:
            constructors.append(TradesCache)
        if self.config.use_orderbook:
            constructors.append(OrderBookCache)
        if self.config.use_funding:
            constructors.append(FundingCache)
        if self.config.use_open_interest:
            constructors.append(OpenInterestCache)

        for cls in constructors:
            caches.append(
                self._instantiate_flexibly(
                    cls,
                    config=core_config,
                    event_bus=self.components.event_bus,
                    scheduler=self.components.scheduler,
                )
            )

        self.components.owned_components.extend(caches)
        return caches

    def _build_simulators(self) -> None:
        if self.components.cost_model is None:
            self.components.cost_model = TradingCostModel(self.config.cost_model)
            self.components.owned_components.append(self.components.cost_model)

        if self.components.execution_simulator is None:
            self.components.execution_simulator = ExecutionSimulator(
                config=self.config.execution_simulator,
                event_bus=self.components.event_bus,
                clock=self.components.clock,
                cost_model=self.components.cost_model,
                random_seed=self.config.random_seed,
            )
            self.components.owned_components.append(self.components.execution_simulator)
        else:
            self.components.execution_simulator.event_bus = self.components.event_bus
            self.components.execution_simulator.clock = self.components.clock

        if self.components.position_simulator is None:
            self.components.position_simulator = PositionSimulator(
                config=self.config.position_simulator,
                event_bus=self.components.event_bus,
                clock=self.components.clock,
                cost_model=self.components.cost_model,
            )
            self.components.owned_components.append(self.components.position_simulator)
        else:
            self.components.position_simulator.event_bus = self.components.event_bus
            self.components.position_simulator.clock = self.components.clock

    def _build_collectors(self) -> None:
        assert self.result is not None
        self.components.collectors = BacktestCollectors(run_id=self.result.run_id)
        assert self.components.event_bus is not None
        self.components.collectors.register(
            self.components.event_bus,
            collect_event_log=self.tester_config.collect_event_log,
            collect_signal_records=self.tester_config.collect_signal_records,
            collect_risk_records=self.tester_config.collect_risk_records,
            collect_execution_records=self.tester_config.collect_execution_records,
            collect_position_records=self.tester_config.collect_position_records,
        )

    def _build_debug_monitor(self) -> None:
        """
        Register passive EventBus diagnostics for integration debugging.

        Env:
        - BACKTEST_DEBUG_FLOW=0 disables it.
        - BACKTEST_DEBUG_FLOW_PRINT=0 keeps metadata but suppresses stdout.
        - BACKTEST_DEBUG_FLOW_RECENT=120 changes recent event buffer size.
        """

        assert self.result is not None
        assert self.components.event_bus is not None

        enabled_raw = os.getenv("BACKTEST_DEBUG_FLOW", "1").strip().lower()
        if enabled_raw in {"0", "false", "no", "off"}:
            self.components.debug_monitor = None
            return

        print_raw = os.getenv("BACKTEST_DEBUG_FLOW_PRINT", "1").strip().lower()
        print_summary = print_raw not in {"0", "false", "no", "off"}

        try:
            max_recent = int(os.getenv("BACKTEST_DEBUG_FLOW_RECENT", "80"))
        except ValueError:
            max_recent = 80

        self.components.debug_monitor = BacktestEventFlowDebugMonitor(
            run_id=self.result.run_id,
            max_recent_events=max_recent,
            print_summary=print_summary,
        )
        self.components.debug_monitor.register(self.components.event_bus)

    def _prepare_market_replay(self) -> None:
        assert self.dataset is not None
        assert self.components.market_replay is not None
        self.components.market_replay.prepare(self.dataset, clock=self.components.clock)

    # ------------------------------------------------------------------
    # Validation / registration / lifecycle
    # ------------------------------------------------------------------

    def _assert_required_components(self) -> None:
        if self.tester_config.require_strategy_engine and self.components.strategy_engine is None:
            raise BacktestDependencyError(
                "StrategyTester requires StrategyEngine. Provide strategy_engine or strategy_engine_factory."
            )

        if self.tester_config.require_signal_processor and self.components.signal_processor is None:
            raise BacktestDependencyError(
                "StrategyTester requires SignalProcessor. Provide signal_processor or signal_processor_factory."
            )

        if self.tester_config.require_risk_manager and self.components.risk_manager is None:
            raise BacktestDependencyError(
                "StrategyTester requires RiskManager. Provide risk_manager or risk_manager_factory."
            )

        if self.tester_config.require_analytics and not self.components.analytics_components:
            raise BacktestDependencyError(
                "StrategyTester requires analytics components. Provide analytics_components or analytics_factory."
            )

    def _assert_no_live_execution_components(self) -> None:
        if not self.tester_config.fail_if_live_execution_detected:
            return

        suspicious: list[str] = []
        for component in self._all_components_for_lifecycle(include_replay=False):
            name = component.__class__.__name__.lower()
            module = getattr(component.__class__, "__module__", "").lower()
            if "tradeexecutor" in name or "execution.trade_executor" in module:
                suspicious.append(component.__class__.__name__)

        if suspicious:
            raise BacktestDependencyError(
                "Live execution components must not be used during backtesting.",
                details={"components": suspicious},
            )

    def _register_components(self) -> None:
        for component in self._all_components_for_lifecycle(include_replay=True):
            method = getattr(component, "register", None)
            if callable(method):
                try:
                    method()
                except (RuntimeError, ValueError, TypeError) as exc:
                    raise BacktestComponentError(
                        "Failed to register backtest component.",
                        details={"component": component.__class__.__name__, "error": str(exc)},
                    ) from exc

    async def _start_components(self) -> None:
        ordered = self._components_start_order()
        for component in ordered:
            try:
                await call_if_supported(component, "start")
                self.components.started_components.append(component)
            except (RuntimeError, ValueError, TypeError) as exc:
                raise BacktestComponentError(
                    "Failed to start backtest component.",
                    details={"component": component.__class__.__name__, "error": str(exc)},
                ) from exc

    async def _stop_components_reverse_order(self) -> None:
        for component in reversed(self.components.started_components):
            try:
                await call_if_supported(component, "stop")
            except (RuntimeError, ValueError, TypeError) as exc:
                self.stats_state.errors.append(
                    f"stop_failed:{component.__class__.__name__}:{exc}"
                )
                self.logger.warning(
                    "Failed to stop backtest component",
                    extra={"component": component.__class__.__name__, "error": str(exc)},
                )

        self.components.started_components.clear()

    def _components_start_order(self) -> list[Any]:
        result: list[Any] = []

        # Core runtime first. EventBus/Scheduler must be running before any
        # component registers background jobs or emits lifecycle events.
        if self.components.event_bus is not None:
            result.append(self.components.event_bus)
        if self.components.scheduler is not None:
            result.append(self.components.scheduler)

        # Downstream listeners before upstream producers. This prevents
        # analytics.start() or replay-produced analytics.* events from being
        # emitted before StrategyEventHandler/Risk/Simulators are subscribed.
        for component in (
            self.components.risk_manager,
            self.components.execution_simulator,
            self.components.position_simulator,
            self.components.signal_processor,
            self.components.strategy_engine,
        ):
            if component is not None:
                result.append(component)

        # Data caches must be ready before MarketReplay emits market.*.
        result.extend(self.components.data_caches)

        # Analytics is intentionally last among production components: it is an
        # upstream producer of analytics.* events consumed by StrategyEngine.
        result.extend(self.components.analytics_components)

        # MarketReplay starts lazily inside replay(); do not start it here.
        return self._dedupe_components(result)

    def _all_components_for_lifecycle(self, *, include_replay: bool) -> list[Any]:
        result = self._components_start_order()
        if include_replay and self.components.market_replay is not None:
            result.append(self.components.market_replay)
        return self._dedupe_components(result)

    # ------------------------------------------------------------------
    # Replay / collection
    # ------------------------------------------------------------------

    def _attach_replay_debug_stats(self) -> None:
        monitor = self.components.debug_monitor
        if monitor is None:
            return

        replay_stats = None
        if self.components.market_replay is not None:
            replay_stats = getattr(self.components.market_replay, "stats_state", None)
            if replay_stats is None:
                replay_stats = getattr(self.components.market_replay, "stats", None)
                if callable(replay_stats):
                    try:
                        replay_stats = replay_stats()
                    except (RuntimeError, ValueError, TypeError, AttributeError):
                        replay_stats = None

        dataset_events = len(self.dataset.events) if self.dataset is not None else 0
        monitor.attach_replay_stats(
            replay_stats=replay_stats,
            dataset_events=dataset_events,
        )

    def _print_debug_checkpoint(self, label: str) -> None:
        monitor = self.components.debug_monitor
        if monitor is None or not monitor.print_summary:
            return

        print("")
        print(f"========== BACKTEST CHECKPOINT: {label} ==========")
        print(f"Last reached stage: {monitor.last_stage}")
        print(f"Suspected breakpoint: {monitor.suspected_breakpoint}")
        print("Stage counts:")
        for group in (
            "market.raw",
            "market.updated",
            "market.candle.closed",
            "analytics",
            "strategy",
            "signal",
            "risk",
            "execution",
            "position",
        ):
            print(f"- {group}: {monitor.group_counts.get(group, 0)}")
        print("===============================================")

    async def _drain_event_bus(self, *, reason: str = "") -> None:
        """
        Best-effort wait until queued EventBus handlers have had a chance to run.

        Production EventBus implementations may expose different names for this
        operation. We try the known public/semipublic shapes first and then fall
        back to several event-loop yields. This keeps backtest result collection
        deterministic without depending on a single EventBus implementation.
        """

        event_bus = self.components.event_bus
        if event_bus is None:
            return

        for method_name in (
            "drain",
            "flush",
            "join",
            "wait_idle",
            "wait_until_idle",
            "wait_empty",
        ):
            method = getattr(event_bus, method_name, None)
            if not callable(method):
                continue
            try:
                await maybe_await(method())
                return
            except TypeError:
                try:
                    await maybe_await(method(timeout=5.0))
                    return
                except (RuntimeError, ValueError, TypeError, AttributeError):
                    continue
            except (RuntimeError, ValueError, AttributeError):
                continue

        queue = getattr(event_bus, "_queue", None)
        join = getattr(queue, "join", None)
        if callable(join):
            try:
                await join()
                return
            except (RuntimeError, ValueError, TypeError, AttributeError):
                pass

        # Fallback for EventBus implementations where emit() schedules handler
        # tasks but exposes no drain API. Chained market -> analytics -> strategy
        # -> risk -> execution -> position flows may need more than a few loop
        # ticks, so keep this configurable for deterministic backtests.
        try:
            iterations = int(os.getenv("BACKTEST_EVENT_DRAIN_ITERATIONS", "50"))
        except ValueError:
            iterations = 50

        try:
            sleep_seconds = float(os.getenv("BACKTEST_EVENT_DRAIN_SLEEP_SECONDS", "0"))
        except ValueError:
            sleep_seconds = 0.0

        for _ in range(max(1, iterations)):
            await asyncio.sleep(max(0.0, sleep_seconds))

    def _attach_result_debug_metadata(self) -> None:
        if self.result is None:
            return

        if self.components.debug_monitor is not None:
            self.result.metadata["event_flow_debug"] = self.components.debug_monitor.to_dict()

        self.result.metadata["strategy_tester_stats"] = self.stats_state.to_dict()
        self.result.metadata["market_replay"] = stats_if_supported(self.components.market_replay)
        self.result.metadata["execution_simulator"] = stats_if_supported(self.components.execution_simulator)
        self.result.metadata["position_simulator"] = stats_if_supported(self.components.position_simulator)

    async def _run_replay(self) -> None:
        if self.components.market_replay is None:
            raise BacktestLifecycleError("MarketReplay is not initialized.")

        stats = await self.components.market_replay.replay()
        self.stats_state.replayed_events = int(getattr(stats, "processed_events", 0))

    async def _collect_result(self, *, status: BacktestStatus) -> BacktestResult:
        if self.result is None:
            raise BacktestResultCollectionError("Cannot collect result before BacktestResult exists.")

        execution = self.components.execution_simulator
        position = self.components.position_simulator
        collectors = self.components.collectors

        orders = list(getattr(execution, "orders", {}) or {})
        if isinstance(getattr(execution, "orders", None), dict):
            orders = list(getattr(execution, "orders").values())

        fills = list(getattr(execution, "fills", []) or [])
        execution_records = list(getattr(execution, "records", []) or [])

        open_positions = list((getattr(position, "positions", {}) or {}).values()) if isinstance(getattr(position, "positions", None), dict) else list(getattr(position, "positions", []) or [])
        closed_positions = list(getattr(position, "closed_positions", []) or [])
        positions = [*closed_positions, *open_positions]
        trades = list(getattr(position, "trades", []) or [])
        equity_curve = list(getattr(position, "equity_curve", []) or [])
        position_records = list(getattr(position, "records", []) or [])

        signal_records = list(collectors.signal_records if collectors is not None else [])
        risk_records = list(collectors.risk_records if collectors is not None else [])

        if collectors is not None:
            # Prefer simulator-owned records for execution/positions because they
            # are richer than passive collector rows. Collector records are used
            # as fallback when a custom simulator does not expose records.
            if not execution_records:
                execution_records = list(collectors.execution_records)
            if not position_records:
                position_records = list(collectors.position_records)

        final_balance = self._resolve_final_balance(position)
        final_equity = self._resolve_final_equity(position, final_balance)

        metrics_input = build_metrics_input_from_components(
            initial_balance=self.config.initial_balance,
            final_balance=final_balance,
            final_equity=final_equity,
            trades=trades,
            positions=positions,
            equity_curve=equity_curve,
            signals=signal_records,
            risk_decisions=risk_records,
            orders=orders,
            fills=fills,
            execution_records=execution_records,
            metadata={"run_id": self.result.run_id, "run_name": self.result.run_name},
        )

        self.result.final_balance = final_balance
        self.result.final_equity = final_equity
        self.result.signals = signal_records
        self.result.risk_decisions = risk_records
        self.result.execution_records = execution_records
        self.result.position_records = position_records
        self.result.orders = orders
        self.result.fills = fills
        self.result.positions = positions
        self.result.trades = trades
        self.result.equity_curve = equity_curve
        self.result.portfolio = self.performance_metrics.calculate_portfolio_result(metrics_input)
        self.result.analytics = self.model_analytics.analyze_from_components(
            signals=signal_records,
            risk_decisions=risk_records,
            orders=orders,
            fills=fills,
            positions=positions,
            trades=trades,
            execution_records=execution_records,
            metadata={"run_id": self.result.run_id},
        )
        self.result.simulation_models = self._build_simulation_snapshot()
        self._attach_result_debug_metadata()

        if collectors is not None and self.config.save_events:
            self.result.metadata["event_log_count"] = len(collectors.event_log)

        if self.config.save_report:
            try:
                self.report_builder.build(self.result)
            except (RuntimeError, ValueError, TypeError, OSError) as exc:
                self.result.warnings.append(
                    BacktestWarning(
                        message="Failed to build backtest report.",
                        level=BacktestWarningLevel.WARNING,
                        code="REPORT_BUILD_FAILED",
                        details={"error": str(exc)},
                    )
                )

        if status == BacktestStatus.COMPLETED:
            self.result.mark_completed()
        elif status == BacktestStatus.FAILED and self.result.error is None:
            self.result.mark_failed("Backtest failed.")

        self._update_stats_from_result(self.result)
        return self.result

    async def _handle_run_failure(self, exc: BaseException) -> BacktestResult:
        self.stats_state.status = BacktestStatus.FAILED
        self.stats_state.errors.append(str(exc))

        if self.result is None:
            self._build_initial_result()

        assert self.result is not None
        self.result.mark_failed(
            str(exc),
            details={"error_type": exc.__class__.__name__},
        )

        try:
            await self._collect_result(status=BacktestStatus.FAILED)
        except (BacktestResultCollectionError, RuntimeError, ValueError, TypeError) as collection_exc:
            self.result.warnings.append(
                BacktestWarning(
                    message="Failed to collect partial backtest result.",
                    level=BacktestWarningLevel.ERROR,
                    code="PARTIAL_RESULT_COLLECTION_FAILED",
                    details={"error": str(collection_exc)},
                )
            )

        await self._emit_best_effort(
            "system.backtest.strategy_tester.failed",
            {
                "run_id": self.result.run_id,
                "run_name": self.result.run_name,
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            },
        )
        return self.result

    # ------------------------------------------------------------------
    # Result helpers
    # ------------------------------------------------------------------

    def _build_simulation_snapshot(self) -> SimulationModelSnapshot:
        kwargs = self._accepted_kwargs(
            SimulationModelSnapshot,
            {
                "commission_model": self.config.cost_model.commission_model,
                "slippage_model": self.config.cost_model.slippage_model,
                "funding_mode": self.config.cost_model.funding_mode,
                "fill_model": self.config.execution_simulator.fill_model,
                "latency_model": self.config.execution_simulator.latency_model,
                "liquidity_model": self.config.execution_simulator.liquidity_model,
                "position_accounting_mode": self._config_value(self.config.position_simulator, "position_accounting_mode", "accounting_mode"),
                "pnl_accounting_mode": self._config_value(self.config.position_simulator, "pnl_accounting_mode", "pnl_mode"),
                "metadata": {
                    "deterministic": self.config.deterministic,
                    "random_seed": self.config.random_seed,
                },
            },
        )
        return SimulationModelSnapshot(**kwargs)


    @staticmethod
    def _config_value(config: Any, *names: str, default: Any = None) -> Any:
        """Return the first existing config attribute from a compatibility list."""

        for name in names:
            if hasattr(config, name):
                return getattr(config, name)
        return default

    @staticmethod
    def _resolve_final_balance(position_simulator: Any) -> float:
        balance = getattr(position_simulator, "balance", None)
        if balance is not None:
            for attr in ("available_balance", "wallet_balance", "balance", "equity"):
                value = getattr(balance, attr, None)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        continue
        return 0.0

    def _resolve_final_equity(self, position_simulator: Any, final_balance: float) -> float:
        equity_curve = list(getattr(position_simulator, "equity_curve", []) or [])
        if equity_curve:
            latest = sorted(equity_curve, key=lambda item: getattr(item, "timestamp_ms", 0))[-1]
            value = getattr(latest, "equity", None)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

        if final_balance > 0:
            return final_balance
        return float(self.config.initial_balance)

    def _update_stats_from_result(self, result: BacktestResult) -> None:
        self.stats_state.status = result.status
        self.stats_state.finished_at = result.finished_at
        self.stats_state.duration_seconds = result.duration_seconds
        self.stats_state.signal_records = len(result.signals)
        self.stats_state.risk_records = len(result.risk_decisions)
        self.stats_state.execution_records = len(result.execution_records)
        self.stats_state.position_records = len(result.position_records)
        self.stats_state.orders = len(result.orders)
        self.stats_state.fills = len(result.fills)
        self.stats_state.positions = len(result.positions)
        self.stats_state.trades = len(result.trades)

        collectors = self.components.collectors
        if collectors is not None:
            self.stats_state.event_log_records = len(collectors.event_log)

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    async def _emit_best_effort(self, topic: str, payload: dict[str, Any]) -> None:
        event_bus = self.components.event_bus
        if event_bus is None:
            return

        enriched = {
            **payload,
            "source": "backtesting.strategy_tester",
            "timestamp_ms": timestamp_ms(utcnow()),
        }

        try:
            result = event_bus.emit(topic, enriched, priority=EventPriority.NORMAL)
            await maybe_await(result)
        except TypeError:
            try:
                result = event_bus.emit(topic, enriched)
                await maybe_await(result)
            except (RuntimeError, ValueError, TypeError, AttributeError):
                return
        except (RuntimeError, ValueError, AttributeError):
            return

    def _apply_component_kwargs(self, component_kwargs: dict[str, Any]) -> None:
        if not component_kwargs:
            return

        expanded = dict(component_kwargs)

        for key in ("pipeline", "backtest_pipeline", "built_pipeline"):
            pipeline = expanded.get(key)
            if pipeline is not None:
                self._apply_built_pipeline(pipeline)
                break

        mapping = {
            "event_bus": "event_bus",
            "scheduler": "scheduler",
            "clock": "clock",
            "market_replay": "market_replay",
            "execution_simulator": "execution_simulator",
            "position_simulator": "position_simulator",
            "data_caches": "data_caches",
            "analytics_components": "analytics_components",
            "strategy_engine": "strategy_engine",
            "signal_processor": "signal_processor",
            "risk_manager": "risk_manager",
        }

        list_attrs = {"data_caches", "analytics_components"}

        for key, attr in mapping.items():
            if key not in expanded:
                continue

            value = expanded[key]
            if attr in list_attrs:
                value = self._as_component_list(value)

            setattr(self.components, attr, value)

    def _apply_built_pipeline(self, pipeline: Any) -> None:
        """
        Accept BuiltBacktestPipeline returned by BacktestProjectBootstrap.build().

        This keeps bootstrap.py and StrategyTester compatible: callers can pass
        component_kwargs={"pipeline": pipeline} instead of manually unpacking
        every component.
        """

        event_bus = getattr(pipeline, "event_bus", None)
        scheduler = getattr(pipeline, "scheduler", None)
        data_caches = getattr(pipeline, "data_caches", None)
        analytics_components = getattr(pipeline, "analytics_components", None)
        risk_manager = getattr(pipeline, "risk_manager", None)

        strategy_pipeline = getattr(pipeline, "strategy_pipeline", None)
        signal_processor = (
            getattr(pipeline, "signal_processor", None)
            or getattr(strategy_pipeline, "signal_processor", None)
        )
        strategy_engine = (
            getattr(pipeline, "strategy_engine", None)
            or getattr(strategy_pipeline, "strategy_engine", None)
        )

        if event_bus is not None:
            self.components.event_bus = event_bus
        if scheduler is not None:
            self.components.scheduler = scheduler
        if data_caches is not None:
            self.components.data_caches = self._as_component_list(data_caches)
        if analytics_components is not None:
            self.components.analytics_components = self._as_component_list(analytics_components)
        if signal_processor is not None:
            self.components.signal_processor = signal_processor
        if strategy_engine is not None:
            self.components.strategy_engine = strategy_engine
        if risk_manager is not None:
            self.components.risk_manager = risk_manager

        diagnostics = getattr(pipeline, "diagnostics", None)
        if diagnostics is not None and self.result is not None:
            formatter = getattr(diagnostics, "format", None)
            if callable(formatter):
                self.result.metadata["bootstrap_diagnostics"] = formatter()

    def _rebind_component_runtime_handles(self) -> None:
        """
        Ensure injected/bootstrap-built components share the tester runtime.

        This is best-effort and intentionally conservative: it only updates
        common public attributes when they already exist on a component.
        """

        event_bus = self.components.event_bus
        scheduler = self.components.scheduler

        for component in self._all_components_for_lifecycle(include_replay=False):
            if component is None:
                continue

            if event_bus is not None and hasattr(component, "event_bus"):
                try:
                    setattr(component, "event_bus", event_bus)
                except (RuntimeError, ValueError, TypeError, AttributeError):
                    pass

            if scheduler is not None and hasattr(component, "scheduler"):
                try:
                    setattr(component, "scheduler", scheduler)
                except (RuntimeError, ValueError, TypeError, AttributeError):
                    pass

    @staticmethod
    def _as_component_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    @staticmethod
    def _dedupe_components(components: Iterable[Any]) -> list[Any]:
        result: list[Any] = []
        seen: set[int] = set()
        for component in components:
            if component is None:
                continue
            marker = id(component)
            if marker in seen:
                continue
            result.append(component)
            seen.add(marker)
        return result

    @staticmethod
    def _instantiate_flexibly(cls: type[Any], **kwargs: Any) -> Any:
        signature = inspect.signature(cls)
        accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
        try:
            return cls(**accepted)
        except TypeError:
            fallback = {key: value for key, value in accepted.items() if value is not None}
            return cls(**fallback)

    @staticmethod
    def _accepted_kwargs(cls: type[Any], values: dict[str, Any]) -> dict[str, Any]:
        signature = inspect.signature(cls)
        return {key: value for key, value in values.items() if key in signature.parameters}


# =============================================================================
# Convenience functions
# =============================================================================


async def run_strategy_backtest(
    *,
    config: BacktestConfig,
    dataset: BacktestDataset | None = None,
    component_kwargs: dict[str, Any] | None = None,
    **tester_kwargs: Any,
) -> BacktestResult:
    """Convenience async helper for scripts and notebooks."""

    tester = StrategyTester(config=config, dataset=dataset, **tester_kwargs)
    return await tester.run(component_kwargs=component_kwargs or {})


def run_strategy_backtest_sync(
    *,
    config: BacktestConfig,
    dataset: BacktestDataset | None = None,
    component_kwargs: dict[str, Any] | None = None,
    **tester_kwargs: Any,
) -> BacktestResult:
    """Synchronous wrapper for simple scripts."""

    return asyncio.run(
        run_strategy_backtest(
            config=config,
            dataset=dataset,
            component_kwargs=component_kwargs,
            **tester_kwargs,
        )
    )


__all__ = [
    "BacktestSchedulerCompatAdapter",
    "StrategyTesterStats",
    "BacktestCollectors",
    "BacktestEventFlowDebugMonitor",
    "BacktestComponentBundle",
    "StrategyTester",
    "run_strategy_backtest",
    "run_strategy_backtest_sync",
]