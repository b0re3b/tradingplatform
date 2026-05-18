from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Deque, Generic, Protocol, TypeVar

from core.event_bus import Event, EventBus, EventPriority, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler

from analytics.liquidations.models import (
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    LiquidationKey,
    liquidation_key_to_dict,
    make_liquidation_key,
    normalize_exchange as _normalize_exchange,
    normalize_exchange_symbol as _normalize_exchange_symbol,
    normalize_market_type as _normalize_market_type,
    normalize_symbol as _normalize_symbol,
    normalize_timeframe as _normalize_timeframe,
)


# ============================================================================
# Small shared helpers
# ============================================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_exchange(exchange: str) -> str:
    return _normalize_exchange(exchange)


def normalize_symbol(symbol: str) -> str:
    return _normalize_symbol(symbol)


def normalize_market_type(market_type: str | None = None) -> str:
    return _normalize_market_type(market_type)


def normalize_timeframe(timeframe: str | None = None) -> str:
    return _normalize_timeframe(timeframe)


def normalize_exchange_symbol(
    exchange_symbol: str | None,
    *,
    fallback_symbol: str,
) -> str:
    return _normalize_exchange_symbol(
        exchange_symbol,
        fallback_symbol=fallback_symbol,
    )


def make_strategy_scope_key(
    *,
    exchange: str,
    market_type: str | None,
    symbol: str,
    timeframe: str | None,
) -> LiquidationKey:
    return make_liquidation_key(
        exchange=exchange,
        market_type=market_type or DEFAULT_MARKET_TYPE,
        symbol=symbol,
        timeframe=timeframe or DEFAULT_TIMEFRAME,
    )


def scoped_key_to_string(key: LiquidationKey) -> str:
    scope = liquidation_key_to_dict(key)
    return (
        f"{scope['exchange']}:"
        f"{scope['market_type']}:"
        f"{scope['symbol']}:"
        f"{scope['timeframe']}"
    )


def clamp_float(
    value: float,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> float:
    if min_value > max_value:
        raise ValueError("min_value must be <= max_value")
    return max(min_value, min(max_value, value))


def serialize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, datetime):
        return ensure_utc(value).isoformat()

    if isinstance(value, dict):
        return {
            str(key): serialize_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [serialize_value(item) for item in value]

    if isinstance(value, tuple):
        return [serialize_value(item) for item in value]

    if hasattr(value, "value"):
        return value.value

    return value


def result_scope(result: Any) -> dict[str, str]:
    exchange = normalize_exchange(getattr(result, "exchange"))
    symbol = normalize_symbol(getattr(result, "symbol"))
    market_type = normalize_market_type(
        getattr(result, "market_type", DEFAULT_MARKET_TYPE)
    )
    timeframe = normalize_timeframe(
        getattr(result, "timeframe", DEFAULT_TIMEFRAME)
    )
    exchange_symbol = normalize_exchange_symbol(
        getattr(result, "exchange_symbol", None),
        fallback_symbol=symbol,
    )

    scope = liquidation_key_to_dict(
        make_strategy_scope_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
    )
    scope["exchange_symbol"] = exchange_symbol
    scope["scope_key"] = scoped_key_to_string(
        (
            scope["exchange"],
            scope["market_type"],
            scope["symbol"],
            scope["timeframe"],
        )
    )
    return scope


def signal_scope(signal: Any) -> dict[str, str]:
    exchange = normalize_exchange(getattr(signal, "exchange"))
    symbol = normalize_symbol(getattr(signal, "symbol"))
    market_type = normalize_market_type(
        getattr(signal, "market_type", DEFAULT_MARKET_TYPE)
    )
    timeframe = normalize_timeframe(
        getattr(signal, "timeframe", DEFAULT_TIMEFRAME)
    )
    exchange_symbol = normalize_exchange_symbol(
        getattr(signal, "exchange_symbol", None),
        fallback_symbol=symbol,
    )

    scope_key = make_strategy_scope_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )

    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
        "exchange_symbol": exchange_symbol,
        "scope_key": scoped_key_to_string(scope_key),
    }


# ============================================================================
# Protocols
# ============================================================================


class AnalyticsStrategyConfigProtocol(Protocol):
    """
    Мінімальний контракт config-а для liquidation strategy поверх analytics events.

    Canonical scope:
        exchange + market_type + symbol + timeframe
    """

    enabled: bool

    subscribe_topic: str
    publish_topic_signal_generated: str
    publish_topic_signal_rejected: str

    publish_rejected_events: bool
    publish_diagnostics_snapshots: bool

    diagnostics_topic: str
    diagnostics_interval_seconds: float

    strategy_name: str
    signal_type: str
    service_name: str

    signal_priority: EventPriority
    rejection_priority: EventPriority
    diagnostics_priority: EventPriority

    allowed_exchanges: tuple[str, ...]
    allowed_symbols: tuple[str, ...]
    blocked_symbols: tuple[str, ...]

    # New full-scope filters.
    allowed_market_types: tuple[str, ...]
    blocked_market_types: tuple[str, ...]
    allowed_timeframes: tuple[str, ...]
    blocked_timeframes: tuple[str, ...]

    min_confidence: float
    min_intensity_score: float
    min_total_notional_usd: Decimal
    min_event_count: int
    max_price_range_pct: float | None

    require_high_confidence_only: bool

    symbol_cooldown_seconds: int
    min_seconds_between_same_side_signals: int

    max_signals_per_symbol_window: int
    signal_window_seconds: int

    deduplicate_by_detected_at: bool
    deduplicate_same_cluster_signature: bool

    recent_signals_limit: int
    recent_rejections_limit: int

    def validate(self) -> None:
        ...


class AnalyticsResultProtocol(Protocol):
    """
    Мінімальний контракт analytics result-а.

    Для liquidation strategies це CascadeDetectionResult.
    """

    exchange: str
    market_type: str
    symbol: str
    timeframe: str
    exchange_symbol: str | None

    detected_at: datetime

    confidence: float
    intensity_score: float

    event_count: int
    total_notional_usd: Decimal
    price_range_pct: float

    severity: Any
    direction: Any

    correlation_id: str | None
    metadata: dict[str, Any]

    @property
    def key(self) -> LiquidationKey:
        ...

    @property
    def is_high_confidence(self) -> bool:
        ...


# ============================================================================
# Shared models
# ============================================================================


@dataclass(slots=True)
class BaseStrategyStats:
    started_at: datetime | None = None
    stopped_at: datetime | None = None

    processed_events: int = 0
    emitted_signals: int = 0
    rejected_events: int = 0

    duplicate_skips: int = 0
    cooldown_skips: int = 0
    rate_limit_skips: int = 0
    filter_skips: int = 0
    invalid_payload_skips: int = 0

    last_signal_at: datetime | None = None
    last_error_at: datetime | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "processed_events": self.processed_events,
            "emitted_signals": self.emitted_signals,
            "rejected_events": self.rejected_events,
            "duplicate_skips": self.duplicate_skips,
            "cooldown_skips": self.cooldown_skips,
            "rate_limit_skips": self.rate_limit_skips,
            "filter_skips": self.filter_skips,
            "invalid_payload_skips": self.invalid_payload_skips,
            "last_signal_at": self.last_signal_at.isoformat() if self.last_signal_at else None,
            "last_error_at": self.last_error_at.isoformat() if self.last_error_at else None,
            "last_error": self.last_error,
        }


@dataclass(slots=True)
class StrategyRejection:
    exchange: str
    symbol: str

    rejected_at: datetime
    reason: str
    source_topic: str

    strategy_name: str
    signal_type: str

    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    correlation_id: str | None = None
    source_event_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.exchange = normalize_exchange(self.exchange)
        self.symbol = normalize_symbol(self.symbol)
        self.market_type = normalize_market_type(self.market_type)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )
        self.rejected_at = ensure_utc(self.rejected_at)

    @property
    def key(self) -> LiquidationKey:
        return make_strategy_scope_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def scope(self) -> dict[str, str]:
        scope = liquidation_key_to_dict(self.key)
        scope["exchange_symbol"] = self.exchange_symbol or self.symbol
        return scope

    @property
    def scope_key(self) -> str:
        return scoped_key_to_string(self.key)

    def to_dict(self, *, serialize: bool = True) -> dict[str, Any]:
        data = {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "exchange_symbol": self.exchange_symbol,
            "scope": self.scope,
            "scope_key": self.scope_key,
            "rejected_at": self.rejected_at,
            "reason": self.reason,
            "source_topic": self.source_topic,
            "strategy_name": self.strategy_name,
            "signal_type": self.signal_type,
            "correlation_id": self.correlation_id,
            "source_event_id": self.source_event_id,
            "details": self.details,
        }
        return serialize_value(data) if serialize else data


@dataclass(slots=True)
class BaseSymbolStrategyState:
    exchange: str
    symbol: str

    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    last_signal_at: datetime | None = None
    cooldown_until: datetime | None = None
    last_signal_side: str | None = None
    last_detected_at: datetime | None = None
    last_cluster_signature: str | None = None
    last_signal_score: float | None = None

    total_signals_emitted: int = 0
    signal_timestamps: Deque[datetime] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.exchange = normalize_exchange(self.exchange)
        self.symbol = normalize_symbol(self.symbol)
        self.market_type = normalize_market_type(self.market_type)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

    @property
    def key(self) -> LiquidationKey:
        return make_strategy_scope_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def scope(self) -> dict[str, str]:
        scope = liquidation_key_to_dict(self.key)
        scope["exchange_symbol"] = self.exchange_symbol or self.symbol
        return scope

    @property
    def scope_key(self) -> str:
        return scoped_key_to_string(self.key)

    def is_in_cooldown(self, now: datetime) -> bool:
        return self.cooldown_until is not None and ensure_utc(now) < ensure_utc(self.cooldown_until)

    def remember_signal(
        self,
        *,
        signal_at: datetime,
        signal_side: str,
        score: float,
        cooldown_seconds: int,
        cluster_signature: str | None,
        detected_at: datetime,
        window_seconds: int,
    ) -> None:
        signal_at = ensure_utc(signal_at)

        self.last_signal_at = signal_at
        self.cooldown_until = (
            signal_at + timedelta(seconds=cooldown_seconds)
            if cooldown_seconds > 0
            else None
        )

        self.last_signal_side = signal_side
        self.last_signal_score = score
        self.last_cluster_signature = cluster_signature
        self.last_detected_at = ensure_utc(detected_at)

        self.total_signals_emitted += 1
        self.signal_timestamps.append(signal_at)
        self.prune_old_signal_timestamps(signal_at, window_seconds)

    def prune_old_signal_timestamps(
        self,
        now: datetime,
        window_seconds: int,
    ) -> None:
        min_ts = ensure_utc(now) - timedelta(seconds=window_seconds)

        while self.signal_timestamps and ensure_utc(self.signal_timestamps[0]) < min_ts:
            self.signal_timestamps.popleft()

    def signals_in_window(
        self,
        now: datetime,
        window_seconds: int,
    ) -> int:
        self.prune_old_signal_timestamps(now, window_seconds)
        return len(self.signal_timestamps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "exchange_symbol": self.exchange_symbol,
            "scope": self.scope,
            "scope_key": self.scope_key,
            "last_signal_at": self.last_signal_at.isoformat() if self.last_signal_at else None,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "last_signal_side": self.last_signal_side,
            "last_detected_at": self.last_detected_at.isoformat() if self.last_detected_at else None,
            "last_cluster_signature": self.last_cluster_signature,
            "last_signal_score": self.last_signal_score,
            "total_signals_emitted": self.total_signals_emitted,
            "signals_in_memory_window": len(self.signal_timestamps),
        }


# ============================================================================
# Generic base strategy
# ============================================================================


ResultT = TypeVar("ResultT", bound=AnalyticsResultProtocol)
SignalT = TypeVar("SignalT")
StateT = TypeVar("StateT", bound=BaseSymbolStrategyState)
ConfigT = TypeVar("ConfigT", bound=AnalyticsStrategyConfigProtocol)


@dataclass(slots=True)
class FilterResult:
    rejection_reason: str | None
    cluster_signature: str | None = None


class BaseAnalyticsStrategy(Generic[ResultT, SignalT, StateT, ConfigT], ABC):
    """
    Базовий клас для liquidation strategies поверх analytics events.

    Canonical scope:
        exchange + market_type + symbol + timeframe

    Base відповідає за:
    - lifecycle start/stop/restart;
    - EventBus subscribe/unsubscribe;
    - Scheduler diagnostics job;
    - common stats;
    - recent signals/rejections;
    - common EventBus emit wrapper;
    - common reject publishing;
    - full-scope state;
    - common filters:
        exchange / market_type / symbol / timeframe;
        confidence / intensity / notional / event_count / severity / price_range;
        cooldown / dedup / rate-limit.
    """

    payload_type: type[ResultT]

    def __init__(
        self,
        *,
        event_bus: EventBus,
        config: ConfigT,
        scheduler: Scheduler | None = None,
        service_name: str | None = None,
        component: str,
        payload_type: type[ResultT],
    ) -> None:
        if event_bus is None:
            raise ValueError("event_bus is required")

        self.event_bus = event_bus
        self.scheduler = scheduler
        self.config = config
        self.service_name = service_name or self.config.service_name
        self.payload_type = payload_type

        self.config.validate()

        self.logger = get_logger(
            __name__,
            service_name=self.service_name,
            event_type="strategy",
            strategy=self.config.strategy_name,
            component=component,
        )

        self._running = False
        self._subscription: Subscription | None = None
        self._diagnostics_job_id: str | None = None

        self._states: dict[LiquidationKey, StateT] = {}

        self._recent_signals: Deque[SignalT] = deque(
            maxlen=max(1, self.config.recent_signals_limit)
        )
        self._recent_rejections: Deque[StrategyRejection] = deque(
            maxlen=max(1, self.config.recent_rejections_limit)
        )

        self._stats = BaseStrategyStats()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_registered(self) -> bool:
        return self._subscription is not None

    async def start(self) -> None:
        if self._running:
            self.logger.warning(
                "Strategy already running",
                extra={"strategy": self.config.strategy_name},
            )
            return

        if not self.config.enabled:
            self.logger.warning(
                "Strategy is disabled by config",
                extra={"strategy": self.config.strategy_name},
            )
            return

        self._running = True
        self._stats.started_at = utc_now()
        self._stats.stopped_at = None
        self._stats.last_error = None
        self._stats.last_error_at = None

        self._subscription = self.event_bus.subscribe(
            self.config.subscribe_topic,
            self._on_bus_event,
            name=f"{self.config.strategy_name}.on_analytics_event",
        )

        self._register_scheduler_jobs()

        self.logger.info(
            "Strategy started",
            extra=self._start_log_extra(),
        )

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        self._stats.stopped_at = utc_now()

        self._unsubscribe()
        self._remove_scheduler_jobs()

        self.logger.info(
            "Strategy stopped",
            extra=self.get_stats(),
        )

    async def close(self) -> None:
        await self.stop()

    async def restart(self) -> None:
        await self.stop()
        await self.start()

    def _unsubscribe(self) -> None:
        if self._subscription is None:
            return

        try:
            self.event_bus.unsubscribe(self._subscription)
        except Exception as exc:
            self._record_error(exc)
            self.logger.warning(
                "Failed to unsubscribe strategy from EventBus",
                extra={
                    "strategy": self.config.strategy_name,
                    "topic": self.config.subscribe_topic,
                    "error": repr(exc),
                },
            )
        finally:
            self._subscription = None

    # ------------------------------------------------------------------
    # Main EventBus handler
    # ------------------------------------------------------------------

    async def _on_bus_event(self, bus_event: Event) -> None:
        if not self._running:
            return

        payload = bus_event.payload

        if not isinstance(payload, self.payload_type):
            self._stats.invalid_payload_skips += 1
            self.logger.debug(
                "Unexpected analytics payload ignored",
                extra={
                    "strategy": self.config.strategy_name,
                    "topic": bus_event.topic,
                    "event_id": bus_event.event_id,
                    "payload_type": type(payload).__name__,
                    "expected_payload_type": self.payload_type.__name__,
                },
            )
            return

        try:
            self._stats.processed_events += 1
            await self.process_result(payload, bus_event=bus_event)

        except Exception as exc:
            self._record_error(exc)
            self.logger.exception(
                "Unhandled strategy processing error",
                extra={
                    "strategy": self.config.strategy_name,
                    "topic": bus_event.topic,
                    "event_id": bus_event.event_id,
                    "correlation_id": bus_event.correlation_id,
                    "scope": result_scope(payload),
                    "error": repr(exc),
                },
            )

    @abstractmethod
    async def process_result(
        self,
        result: ResultT,
        *,
        bus_event: Event,
    ) -> None:
        """
        Subclass реалізує domain-specific pipeline:
        - get/create full-scope state;
        - common + custom filters;
        - build signal;
        - emit signal;
        - remember state.
        """

    # ------------------------------------------------------------------
    # Common filters
    # ------------------------------------------------------------------

    def evaluate_common_filters(
        self,
        *,
        result: ResultT,
        state: StateT,
        now: datetime,
    ) -> FilterResult:
        rejection = self.get_common_rejection_reason(
            result=result,
            state=state,
            now=now,
        )

        signature = None
        if rejection is None:
            signature = self.build_cluster_signature(result)

        return FilterResult(
            rejection_reason=rejection,
            cluster_signature=signature,
        )

    def get_common_rejection_reason(
        self,
        *,
        result: ResultT,
        state: StateT,
        now: datetime,
    ) -> str | None:
        scope = result_scope(result)

        allowed_exchanges = {
            normalize_exchange(item)
            for item in getattr(self.config, "allowed_exchanges", ())
            if str(item).strip()
        }
        if allowed_exchanges and scope["exchange"] not in allowed_exchanges:
            self._stats.filter_skips += 1
            return "exchange_not_allowed"

        allowed_market_types = {
            normalize_market_type(item)
            for item in getattr(self.config, "allowed_market_types", ())
            if str(item).strip()
        }
        if allowed_market_types and scope["market_type"] not in allowed_market_types:
            self._stats.filter_skips += 1
            return "market_type_not_allowed"

        blocked_market_types = {
            normalize_market_type(item)
            for item in getattr(self.config, "blocked_market_types", ())
            if str(item).strip()
        }
        if blocked_market_types and scope["market_type"] in blocked_market_types:
            self._stats.filter_skips += 1
            return "market_type_blocked"

        allowed_symbols = {
            normalize_symbol(item)
            for item in getattr(self.config, "allowed_symbols", ())
            if str(item).strip()
        }
        if allowed_symbols and scope["symbol"] not in allowed_symbols:
            self._stats.filter_skips += 1
            return "symbol_not_allowed"

        blocked_symbols = {
            normalize_symbol(item)
            for item in getattr(self.config, "blocked_symbols", ())
            if str(item).strip()
        }
        if blocked_symbols and scope["symbol"] in blocked_symbols:
            self._stats.filter_skips += 1
            return "symbol_blocked"

        allowed_timeframes = {
            normalize_timeframe(item)
            for item in getattr(self.config, "allowed_timeframes", ())
            if str(item).strip()
        }
        if allowed_timeframes and scope["timeframe"] not in allowed_timeframes:
            self._stats.filter_skips += 1
            return "timeframe_not_allowed"

        blocked_timeframes = {
            normalize_timeframe(item)
            for item in getattr(self.config, "blocked_timeframes", ())
            if str(item).strip()
        }
        if blocked_timeframes and scope["timeframe"] in blocked_timeframes:
            self._stats.filter_skips += 1
            return "timeframe_blocked"

        direction = getattr(result, "direction", None)
        if direction is not None and getattr(direction, "value", direction) == "unknown":
            self._stats.filter_skips += 1
            return "unknown_direction"

        if self.config.require_high_confidence_only and not result.is_high_confidence:
            self._stats.filter_skips += 1
            return "not_high_confidence"

        if result.confidence < self.config.min_confidence:
            self._stats.filter_skips += 1
            return "confidence_below_threshold"

        if result.intensity_score < self.config.min_intensity_score:
            self._stats.filter_skips += 1
            return "intensity_below_threshold"

        if result.total_notional_usd < self.config.min_total_notional_usd:
            self._stats.filter_skips += 1
            return "notional_below_threshold"

        if result.event_count < self.config.min_event_count:
            self._stats.filter_skips += 1
            return "event_count_below_threshold"

        allowed_severities = getattr(self.config, "allowed_severities", ())
        if allowed_severities and result.severity not in allowed_severities:
            self._stats.filter_skips += 1
            return "severity_not_allowed"

        if (
            self.config.max_price_range_pct is not None
            and result.price_range_pct > self.config.max_price_range_pct
        ):
            self._stats.filter_skips += 1
            return "price_range_above_threshold"

        if state.is_in_cooldown(now):
            self._stats.cooldown_skips += 1
            return "scope_in_cooldown"

        detected_at = ensure_utc(result.detected_at)

        if self.config.deduplicate_by_detected_at:
            if state.last_detected_at is not None and detected_at <= ensure_utc(state.last_detected_at):
                self._stats.duplicate_skips += 1
                return "duplicate_detected_at"

        cluster_signature = self.build_cluster_signature(result)
        if self.config.deduplicate_same_cluster_signature:
            if cluster_signature and state.last_cluster_signature == cluster_signature:
                self._stats.duplicate_skips += 1
                return "duplicate_cluster_signature"

        trade_side = self.direction_to_trade_side(result)
        if (
            state.last_signal_at is not None
            and state.last_signal_side == trade_side
            and (ensure_utc(now) - ensure_utc(state.last_signal_at)).total_seconds()
            < self.config.min_seconds_between_same_side_signals
        ):
            self._stats.duplicate_skips += 1
            return "same_side_signal_too_soon"

        if self.config.max_signals_per_symbol_window > 0:
            signals_in_window = state.signals_in_window(
                now=now,
                window_seconds=self.config.signal_window_seconds,
            )
            if signals_in_window >= self.config.max_signals_per_symbol_window:
                self._stats.rate_limit_skips += 1
                return "scope_signal_rate_limited"

        return None

    # ------------------------------------------------------------------
    # Emit / reject
    # ------------------------------------------------------------------

    async def emit_event(
        self,
        topic: str,
        payload: Any,
        *,
        priority: EventPriority,
        correlation_id: str | None,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        try:
            return await self.event_bus.emit(
                topic,
                payload,
                priority=priority,
                source=self.config.strategy_name,
                correlation_id=correlation_id,
                headers=headers or {},
            )
        except Exception as exc:
            self._record_error(exc)
            self.logger.exception(
                "EventBus emit failed",
                extra={
                    "strategy": self.config.strategy_name,
                    "topic": topic,
                    "correlation_id": correlation_id,
                    "error": repr(exc),
                },
            )
            return False

    async def emit_signal(
        self,
        signal: SignalT,
        *,
        bus_event: Event,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        scope = signal_scope(signal)

        signal_headers = {
            "strategy": self.config.strategy_name,
            "signal_type": self.config.signal_type,
            "source_event_id": bus_event.event_id,
            "source_topic": bus_event.topic,
            "exchange": scope["exchange"],
            "market_type": scope["market_type"],
            "symbol": scope["symbol"],
            "timeframe": scope["timeframe"],
            "exchange_symbol": scope["exchange_symbol"],
            "scope": scope["scope_key"],
        }

        side = getattr(signal, "side", None)
        if side is not None:
            signal_headers["side"] = str(side)

        if headers:
            signal_headers.update(headers)

        return await self.emit_event(
            self.config.publish_topic_signal_generated,
            signal,
            priority=self.config.signal_priority,
            correlation_id=bus_event.correlation_id or bus_event.event_id,
            headers=signal_headers,
        )

    async def reject_result(
        self,
        *,
        result: ResultT,
        bus_event: Event,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._stats.rejected_events += 1

        scope = result_scope(result)

        rejection = StrategyRejection(
            exchange=scope["exchange"],
            market_type=scope["market_type"],
            symbol=scope["symbol"],
            timeframe=scope["timeframe"],
            exchange_symbol=scope["exchange_symbol"],
            rejected_at=utc_now(),
            reason=reason,
            source_topic=bus_event.topic,
            strategy_name=self.config.strategy_name,
            signal_type=self.config.signal_type,
            correlation_id=bus_event.correlation_id,
            source_event_id=bus_event.event_id,
            details=details or self.build_rejection_details(result),
        )

        self._recent_rejections.append(rejection)

        self.logger.debug(
            "Analytics result rejected by strategy filters",
            extra={
                "strategy": self.config.strategy_name,
                "exchange": scope["exchange"],
                "market_type": scope["market_type"],
                "symbol": scope["symbol"],
                "timeframe": scope["timeframe"],
                "exchange_symbol": scope["exchange_symbol"],
                "scope": scope["scope_key"],
                "reason": reason,
                "event_id": bus_event.event_id,
                "correlation_id": bus_event.correlation_id,
            },
        )

        if self.config.publish_rejected_events:
            await self.emit_event(
                self.config.publish_topic_signal_rejected,
                rejection,
                priority=self.config.rejection_priority,
                correlation_id=bus_event.correlation_id or bus_event.event_id,
                headers={
                    "strategy": self.config.strategy_name,
                    "signal_type": self.config.signal_type,
                    "exchange": scope["exchange"],
                    "market_type": scope["market_type"],
                    "symbol": scope["symbol"],
                    "timeframe": scope["timeframe"],
                    "exchange_symbol": scope["exchange_symbol"],
                    "scope": scope["scope_key"],
                    "reason": reason,
                    "source_event_id": bus_event.event_id,
                    "source_topic": bus_event.topic,
                },
            )

    def remember_emitted_signal(
        self,
        *,
        signal: SignalT,
        state: StateT,
        result: ResultT,
        signal_side: str,
        score: float,
        cluster_signature: str | None,
    ) -> None:
        generated_at = getattr(signal, "generated_at", utc_now())

        state.remember_signal(
            signal_at=generated_at,
            signal_side=signal_side,
            score=score,
            cooldown_seconds=self.config.symbol_cooldown_seconds,
            cluster_signature=cluster_signature,
            detected_at=result.detected_at,
            window_seconds=self.config.signal_window_seconds,
        )

        self._recent_signals.append(signal)

        self._stats.emitted_signals += 1
        self._stats.last_signal_at = ensure_utc(generated_at)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @staticmethod
    def state_key(
        exchange: str,
        symbol: str,
        *,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> LiquidationKey:
        return make_strategy_scope_key(
            exchange=exchange,
            market_type=market_type or DEFAULT_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe or DEFAULT_TIMEFRAME,
        )

    @staticmethod
    def state_key_from_result(result: AnalyticsResultProtocol) -> LiquidationKey:
        return make_strategy_scope_key(
            exchange=result.exchange,
            market_type=getattr(result, "market_type", DEFAULT_MARKET_TYPE),
            symbol=result.symbol,
            timeframe=getattr(result, "timeframe", DEFAULT_TIMEFRAME),
        )

    def get_or_create_state(
        self,
        exchange: str,
        symbol: str,
        *,
        market_type: str | None = None,
        timeframe: str | None = None,
        exchange_symbol: str | None = None,
    ) -> StateT:
        key = self.state_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        state = self._states.get(key)

        if state is None:
            state = self.create_symbol_state(
                exchange=exchange,
                market_type=market_type or DEFAULT_MARKET_TYPE,
                symbol=symbol,
                timeframe=timeframe or DEFAULT_TIMEFRAME,
                exchange_symbol=exchange_symbol,
            )
            self._states[key] = state

        return state

    def get_or_create_state_for_result(self, result: ResultT) -> StateT:
        return self.get_or_create_state(
            exchange=result.exchange,
            market_type=getattr(result, "market_type", DEFAULT_MARKET_TYPE),
            symbol=result.symbol,
            timeframe=getattr(result, "timeframe", DEFAULT_TIMEFRAME),
            exchange_symbol=getattr(result, "exchange_symbol", None),
        )

    def get_state(
        self,
        exchange: str,
        symbol: str,
        *,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> StateT | None:
        return self._states.get(
            self.state_key(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            )
        )

    def get_state_for_result(self, result: ResultT) -> StateT | None:
        return self._states.get(self.state_key_from_result(result))

    @abstractmethod
    def create_symbol_state(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        exchange_symbol: str | None = None,
    ) -> StateT:
        """
        Subclass повертає свій full-scope state.
        """

    # ------------------------------------------------------------------
    # Scheduler / diagnostics
    # ------------------------------------------------------------------

    def _register_scheduler_jobs(self) -> None:
        if self.scheduler is None:
            return

        if not self.config.publish_diagnostics_snapshots:
            return

        self._diagnostics_job_id = self.scheduler.add_interval_job(
            name=f"{self.config.strategy_name}:diagnostics",
            func=self.publish_diagnostics_snapshot,
            interval=self.config.diagnostics_interval_seconds,
            run_immediately=False,
            max_retries=0,
            retry_delay=1.0,
            timeout=10.0,
            allow_overlap=False,
            enabled=True,
        )

    def _remove_scheduler_jobs(self) -> None:
        if self.scheduler is None:
            self._diagnostics_job_id = None
            return

        if self._diagnostics_job_id is None:
            return

        try:
            self.scheduler.remove_job(self._diagnostics_job_id)
        except KeyError:
            pass
        except Exception as exc:
            self._record_error(exc)
            self.logger.warning(
                "Failed to remove diagnostics scheduler job",
                extra={
                    "strategy": self.config.strategy_name,
                    "job_id": self._diagnostics_job_id,
                    "error": repr(exc),
                },
            )
        finally:
            self._diagnostics_job_id = None

    async def publish_diagnostics_snapshot(self) -> None:
        if not self._running:
            return

        snapshot = {
            "strategy_name": self.config.strategy_name,
            "signal_type": self.config.signal_type,
            "created_at": utc_now().isoformat(),
            "scope": "exchange:market_type:symbol:timeframe",
            "stats": self.get_stats(),
            "hot_symbols": self.get_hot_symbols(limit=10),
        }

        await self.emit_event(
            self.config.diagnostics_topic,
            snapshot,
            priority=self.config.diagnostics_priority,
            correlation_id=None,
            headers={
                "strategy": self.config.strategy_name,
                "signal_type": self.config.signal_type,
                "event_type": "strategy_diagnostics",
            },
        )

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_recent_signals(
        self,
        *,
        exchange: str | None = None,
        symbol: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
        limit: int = 50,
    ) -> list[SignalT]:
        if limit <= 0:
            return []

        target_exchange = normalize_exchange(exchange) if exchange else None
        target_symbol = normalize_symbol(symbol) if symbol else None
        target_market_type = normalize_market_type(market_type) if market_type else None
        target_timeframe = normalize_timeframe(timeframe) if timeframe else None

        result: list[SignalT] = []

        for signal in reversed(self._recent_signals):
            scope = signal_scope(signal)

            if target_exchange is not None and scope["exchange"] != target_exchange:
                continue

            if target_market_type is not None and scope["market_type"] != target_market_type:
                continue

            if target_symbol is not None and scope["symbol"] != target_symbol:
                continue

            if target_timeframe is not None and scope["timeframe"] != target_timeframe:
                continue

            result.append(signal)

            if len(result) >= limit:
                break

        return result

    def get_recent_rejections(
        self,
        *,
        exchange: str | None = None,
        symbol: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
        limit: int = 50,
    ) -> list[StrategyRejection]:
        if limit <= 0:
            return []

        target_exchange = normalize_exchange(exchange) if exchange else None
        target_symbol = normalize_symbol(symbol) if symbol else None
        target_market_type = normalize_market_type(market_type) if market_type else None
        target_timeframe = normalize_timeframe(timeframe) if timeframe else None

        result: list[StrategyRejection] = []

        for rejection in reversed(self._recent_rejections):
            if target_exchange is not None and rejection.exchange != target_exchange:
                continue

            if target_market_type is not None and rejection.market_type != target_market_type:
                continue

            if target_symbol is not None and rejection.symbol != target_symbol:
                continue

            if target_timeframe is not None and rejection.timeframe != target_timeframe:
                continue

            result.append(rejection)

            if len(result) >= limit:
                break

        return result

    def get_hot_symbols(
        self,
        *,
        exchange: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []

        target_exchange = normalize_exchange(exchange) if exchange else None
        target_market_type = normalize_market_type(market_type) if market_type else None
        target_timeframe = normalize_timeframe(timeframe) if timeframe else None

        latest_by_key: dict[LiquidationKey, SignalT] = {}

        for signal in self._recent_signals:
            scope = signal_scope(signal)

            if target_exchange is not None and scope["exchange"] != target_exchange:
                continue

            if target_market_type is not None and scope["market_type"] != target_market_type:
                continue

            if target_timeframe is not None and scope["timeframe"] != target_timeframe:
                continue

            key = make_strategy_scope_key(
                exchange=scope["exchange"],
                market_type=scope["market_type"],
                symbol=scope["symbol"],
                timeframe=scope["timeframe"],
            )

            previous = latest_by_key.get(key)
            if previous is None:
                latest_by_key[key] = signal
                continue

            if getattr(signal, "generated_at", datetime.min.replace(tzinfo=timezone.utc)) > getattr(
                previous,
                "generated_at",
                datetime.min.replace(tzinfo=timezone.utc),
            ):
                latest_by_key[key] = signal

        rows = [self.signal_to_hot_symbol_row(signal) for signal in latest_by_key.values()]

        rows.sort(
            key=lambda row: (
                float(row.get("score", 0.0) or 0.0),
                float(row.get("confidence", 0.0) or 0.0),
                float(row.get("intensity_score", 0.0) or 0.0),
            ),
            reverse=True,
        )

        return rows[:limit]

    def signal_to_hot_symbol_row(self, signal: SignalT) -> dict[str, Any]:
        scope = signal_scope(signal)

        return {
            "exchange": scope["exchange"],
            "market_type": scope["market_type"],
            "symbol": scope["symbol"],
            "timeframe": scope["timeframe"],
            "exchange_symbol": scope["exchange_symbol"],
            "scope": {
                "exchange": scope["exchange"],
                "market_type": scope["market_type"],
                "symbol": scope["symbol"],
                "timeframe": scope["timeframe"],
            },
            "scope_key": scope["scope_key"],
            "side": getattr(signal, "side", None),
            "score": getattr(signal, "score", None),
            "confidence": getattr(signal, "confidence", None),
            "severity": getattr(signal, "severity", None),
            "intensity_score": getattr(signal, "intensity_score", None),
            "generated_at": (
                ensure_utc(getattr(signal, "generated_at")).isoformat()
                if getattr(signal, "generated_at", None)
                else None
            ),
            "total_notional_usd": str(getattr(signal, "total_notional_usd", "")),
        }

    def get_stats(self) -> dict[str, Any]:
        data = self._stats.to_dict()
        data.update(
            {
                "running": self._running,
                "registered": self._subscription is not None,
                "strategy_name": self.config.strategy_name,
                "signal_type": self.config.signal_type,
                "subscribe_topic": self.config.subscribe_topic,
                "scope": "exchange:market_type:symbol:timeframe",
                "tracked_scopes": len(self._states),
                "tracked_symbols": len(
                    {
                        (state.exchange, state.symbol)
                        for state in self._states.values()
                    }
                ),
                "recent_signals": len(self._recent_signals),
                "recent_rejections": len(self._recent_rejections),
                "diagnostics_job_registered": self._diagnostics_job_id is not None,
                "diagnostics_job_id": self._diagnostics_job_id,
            }
        )
        return data

    # ------------------------------------------------------------------
    # Shared builders
    # ------------------------------------------------------------------

    def build_rejection_details(self, result: ResultT) -> dict[str, Any]:
        scope = result_scope(result)

        return {
            "exchange": scope["exchange"],
            "market_type": scope["market_type"],
            "symbol": scope["symbol"],
            "timeframe": scope["timeframe"],
            "exchange_symbol": scope["exchange_symbol"],
            "scope": {
                "exchange": scope["exchange"],
                "market_type": scope["market_type"],
                "symbol": scope["symbol"],
                "timeframe": scope["timeframe"],
            },
            "scope_key": scope["scope_key"],
            "severity": serialize_value(getattr(result, "severity", None)),
            "direction": serialize_value(getattr(result, "direction", None)),
            "event_type": serialize_value(getattr(result, "event_type", None)),
            "status": serialize_value(getattr(result, "status", None)),
            "confidence": getattr(result, "confidence", None),
            "intensity_score": getattr(result, "intensity_score", None),
            "continuation_bias": getattr(result, "continuation_bias", None),
            "exhaustion_bias": getattr(result, "exhaustion_bias", None),
            "event_count": getattr(result, "event_count", None),
            "total_notional_usd": str(getattr(result, "total_notional_usd", "")),
            "price_range_pct": getattr(result, "price_range_pct", None),
            "metadata": serialize_value(getattr(result, "metadata", {}) or {}),
        }

    def build_cluster_signature(self, result: ResultT) -> str:
        """
        Stable signature для dedup.

        Важливо:
        signature включає повний futures/liquidation scope, щоб не змішувати:
        - usdm_futures / coinm_futures;
        - linear / inverse / swap;
        - realtime / 1m.
        """
        scope = result_scope(result)
        cluster = getattr(result, "cluster", None)

        parts: dict[str, Any] = {
            "strategy_name": self.config.strategy_name,
            "exchange": scope["exchange"],
            "market_type": scope["market_type"],
            "symbol": scope["symbol"],
            "timeframe": scope["timeframe"],
            "exchange_symbol": scope["exchange_symbol"],
            "direction": serialize_value(getattr(result, "direction", None)),
            "severity": serialize_value(getattr(result, "severity", None)),
            "event_type": serialize_value(getattr(result, "event_type", None)),
            "status": serialize_value(getattr(result, "status", None)),
            "detected_at": ensure_utc(result.detected_at).isoformat(),
            "event_count": result.event_count,
            "total_notional_usd": str(result.total_notional_usd),
        }

        if cluster is not None:
            parts["cluster"] = {
                "exchange": getattr(cluster, "exchange", scope["exchange"]),
                "market_type": getattr(cluster, "market_type", scope["market_type"]),
                "symbol": getattr(cluster, "symbol", scope["symbol"]),
                "timeframe": getattr(cluster, "timeframe", scope["timeframe"]),
                "exchange_symbol": getattr(cluster, "exchange_symbol", scope["exchange_symbol"]),
                "start_time": serialize_value(getattr(cluster, "start_time", None)),
                "end_time": serialize_value(getattr(cluster, "end_time", None)),
                "event_count": getattr(cluster, "event_count", None),
                "total_notional_usd": str(getattr(cluster, "total_notional_usd", "")),
                "avg_price": str(getattr(cluster, "avg_price", "")),
                "min_price": str(getattr(cluster, "min_price", "")),
                "max_price": str(getattr(cluster, "max_price", "")),
            }

        raw = json.dumps(
            serialize_value(parts),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def build_common_signal_metadata(
        self,
        *,
        result: ResultT,
        bus_event: Event,
    ) -> dict[str, Any]:
        scope = result_scope(result)
        cluster = getattr(result, "cluster", None)

        metadata: dict[str, Any] = {
            "scope": {
                "exchange": scope["exchange"],
                "market_type": scope["market_type"],
                "symbol": scope["symbol"],
                "timeframe": scope["timeframe"],
            },
            "scope_key": scope["scope_key"],
            "exchange_symbol": scope["exchange_symbol"],
            "strategy": {
                "strategy_name": self.config.strategy_name,
                "signal_type": self.config.signal_type,
                "min_confidence": self.config.min_confidence,
                "min_intensity_score": self.config.min_intensity_score,
                "min_total_notional_usd": str(self.config.min_total_notional_usd),
                "min_event_count": self.config.min_event_count,
                "allowed_exchanges": getattr(self.config, "allowed_exchanges", ()),
                "allowed_market_types": getattr(self.config, "allowed_market_types", ()),
                "allowed_symbols": getattr(self.config, "allowed_symbols", ()),
                "allowed_timeframes": getattr(self.config, "allowed_timeframes", ()),
                "blocked_market_types": getattr(self.config, "blocked_market_types", ()),
                "blocked_symbols": getattr(self.config, "blocked_symbols", ()),
                "blocked_timeframes": getattr(self.config, "blocked_timeframes", ()),
            },
            "bus_event": {
                "topic": bus_event.topic,
                "event_id": bus_event.event_id,
                "source": bus_event.source,
                "priority": int(bus_event.priority),
                "correlation_id": bus_event.correlation_id,
                "headers": dict(bus_event.headers),
            },
            "analytics": {
                "event_type": serialize_value(getattr(result, "event_type", None)),
                "status": serialize_value(getattr(result, "status", None)),
                "severity": serialize_value(getattr(result, "severity", None)),
                "direction": serialize_value(getattr(result, "direction", None)),
                "confidence": getattr(result, "confidence", None),
                "intensity_score": getattr(result, "intensity_score", None),
                "continuation_bias": getattr(result, "continuation_bias", None),
                "exhaustion_bias": getattr(result, "exhaustion_bias", None),
                "event_count": getattr(result, "event_count", None),
                "total_notional_usd": str(getattr(result, "total_notional_usd", "")),
                "price_range_pct": getattr(result, "price_range_pct", None),
                "metadata": serialize_value(getattr(result, "metadata", {}) or {}),
            },
        }

        if cluster is not None:
            metadata["cluster"] = {
                "exchange": getattr(cluster, "exchange", scope["exchange"]),
                "market_type": getattr(cluster, "market_type", scope["market_type"]),
                "symbol": getattr(cluster, "symbol", scope["symbol"]),
                "timeframe": getattr(cluster, "timeframe", scope["timeframe"]),
                "exchange_symbol": getattr(cluster, "exchange_symbol", scope["exchange_symbol"]),
                "start_time": serialize_value(getattr(cluster, "start_time", None)),
                "end_time": serialize_value(getattr(cluster, "end_time", None)),
                "event_count": getattr(cluster, "event_count", None),
                "total_notional_usd": str(getattr(cluster, "total_notional_usd", "")),
                "avg_price": str(getattr(cluster, "avg_price", "")),
                "min_price": str(getattr(cluster, "min_price", "")),
                "max_price": str(getattr(cluster, "max_price", "")),
                "duration_seconds": getattr(cluster, "duration_seconds", None),
                "avg_notional_per_event": str(getattr(cluster, "avg_notional_per_event", "")),
                "metadata": serialize_value(getattr(cluster, "metadata", {}) or {}),
            }

        return metadata

    def severity_to_score(self, severity: Any) -> float:
        value = getattr(severity, "value", str(severity)).lower()

        if value == "extreme":
            return 1.0
        if value == "high":
            return 0.8
        if value == "medium":
            return 0.6
        if value == "low":
            return 0.4

        rank = getattr(severity, "rank", None)
        if isinstance(rank, (int, float)):
            return clamp_float(float(rank) / 4.0)

        return 0.0

    def extract_analytics_metadata(self, result: ResultT) -> dict[str, Any]:
        metadata = dict(getattr(result, "metadata", {}) or {})

        return {
            "side_imbalance_ratio": metadata.get("side_imbalance_ratio"),
            "event_imbalance_ratio": metadata.get("event_imbalance_ratio"),
            "acceleration_ratio": (
                metadata.get("acceleration_ratio")
                or metadata.get("climax_acceleration_ratio")
            ),
            "scope": metadata.get("scope") or result_scope(result),
            "scope_key": metadata.get("scope_key") or result_scope(result)["scope_key"],
            "raw": metadata,
        }

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _start_log_extra(self) -> dict[str, Any]:
        return {
            "strategy": self.config.strategy_name,
            "signal_type": self.config.signal_type,
            "topic": self.config.subscribe_topic,
            "scope": "exchange:market_type:symbol:timeframe",
            "allowed_exchanges": getattr(self.config, "allowed_exchanges", ()),
            "allowed_market_types": getattr(self.config, "allowed_market_types", ()),
            "allowed_symbols": getattr(self.config, "allowed_symbols", ()),
            "allowed_timeframes": getattr(self.config, "allowed_timeframes", ()),
            "min_confidence": self.config.min_confidence,
            "min_intensity_score": self.config.min_intensity_score,
            "min_total_notional_usd": str(self.config.min_total_notional_usd),
            "scheduler_enabled": self.scheduler is not None,
            "diagnostics_enabled": self.config.publish_diagnostics_snapshots,
        }

    def _record_error(self, exc: Exception) -> None:
        self._stats.last_error_at = utc_now()
        self._stats.last_error = repr(exc)

    @abstractmethod
    def direction_to_trade_side(self, result: ResultT) -> str:
        """
        Subclass визначає, як analytics direction перетворюється в trade side.

        Continuation:
            DOWN -> SHORT
            UP   -> LONG

        Reversal:
            DOWN -> LONG
            UP   -> SHORT
        """