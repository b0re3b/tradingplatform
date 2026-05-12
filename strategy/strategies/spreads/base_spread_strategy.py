from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from analytics.spreads.config import (
    DEFAULT_ARBITRAGE_OPPORTUNITY_TOPIC,
    DEFAULT_CROSS_EXCHANGE_SNAPSHOT_TOPIC,
    DEFAULT_SPOT_FUTURES_SNAPSHOT_TOPIC,
    DEFAULT_SPREAD_SIGNAL_TOPIC,
)
from core.event_bus import Event, EventBus, EventPriority, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler


PayloadHandler = Callable[[Any], None | Awaitable[None]]
EventHandler = Callable[[Event], None | Awaitable[None]]


# ============================================================
# Input events from analytics.spreads
# ============================================================

SPOT_FUTURES_SNAPSHOT_EVENT = DEFAULT_SPOT_FUTURES_SNAPSHOT_TOPIC
CROSS_EXCHANGE_SNAPSHOT_EVENT = DEFAULT_CROSS_EXCHANGE_SNAPSHOT_TOPIC
SPREAD_SIGNAL_EVENT = DEFAULT_SPREAD_SIGNAL_TOPIC
ARBITRAGE_OPPORTUNITY_EVENT = DEFAULT_ARBITRAGE_OPPORTUNITY_TOPIC


# ============================================================
# Output events from strategy layer
# ============================================================

STRATEGY_SIGNAL_GENERATED_EVENT = "signal.generated"
STRATEGY_SIGNAL_UPDATED_EVENT = "signal.updated"
STRATEGY_SIGNAL_REJECTED_EVENT = "signal.rejected"
STRATEGY_SIGNAL_CANCELLED_EVENT = "signal.cancelled"
STRATEGY_SIGNAL_CLOSED_EVENT = "signal.closed"

STRATEGY_STARTED_EVENT = "strategy.spreads.started"
STRATEGY_STOPPED_EVENT = "strategy.spreads.stopped"
STRATEGY_HEARTBEAT_EVENT = "strategy.spreads.heartbeat"


# ============================================================
# State/status constants
# ============================================================

STATE_IDLE = "idle"
STATE_PENDING = "pending"
STATE_OPEN = "open"
STATE_BLOCKED = "blocked"
STATE_CLOSING = "closing"
STATE_CLOSED = "closed"
STATE_CANCELLED = "cancelled"
STATE_REJECTED = "rejected"

ACTIVE_STATES = frozenset({STATE_PENDING, STATE_OPEN, STATE_CLOSING})
CLOSED_STATES = frozenset({STATE_CLOSED, STATE_CANCELLED, STATE_REJECTED})


# ============================================================
# Validation helpers
# ============================================================

DECIMAL_ZERO = Decimal("0")


def _validate_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _validate_positive_float(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _validate_non_negative_decimal(name: str, value: Decimal) -> None:
    if value < DECIMAL_ZERO:
        raise ValueError(f"{name} must be >= 0")


def _normalize_symbol_value(symbol: str | None) -> str:
    if not symbol:
        return ""
    return symbol.replace("-", "").replace("/", "").replace("_", "").upper().strip()


def _normalize_exchange_value(exchange: str | None) -> str:
    if not exchange:
        return ""
    return exchange.strip().lower()


def _normalize_symbol_set(values: set[str] | list[str] | tuple[str, ...] | None) -> set[str]:
    if not values:
        return set()
    return {
        normalized
        for item in values
        if (normalized := _normalize_symbol_value(item))
    }


def _normalize_exchange_set(values: set[str] | list[str] | tuple[str, ...] | None) -> set[str]:
    if not values:
        return set()
    return {
        normalized
        for item in values
        if (normalized := _normalize_exchange_value(item))
    }


def _metadata_copy(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    return dict(metadata)


# ============================================================
# Config
# ============================================================

@dataclass(slots=True)
class BaseSpreadStrategyConfig:
    """
    Базова strategy-layer конфігурація для spread-стратегій.

    Відповідальність:
    - runtime enable/disable;
    - cooldown для strategy-intent публікацій;
    - мінімальна confidence;
    - freshness thresholds для analytics snapshot/signal payload;
    - allowlist symbol/exchange фільтри;
    - cleanup-політика для closed states;
    - metadata для розширення без зміни контракту.

    Не відповідає за:
    - створення EventBus;
    - створення Scheduler;
    - analytics-розрахунки;
    - execution/risk логіку;
    - читання .env напряму.
    """

    enabled: bool = True

    cooldown_seconds: int = 10
    min_confidence: Decimal = Decimal("0")

    max_snapshot_age_ms: int = 3_000
    max_signal_age_ms: int = 5_000

    allowed_symbols: set[str] = field(default_factory=set)
    allowed_exchanges: set[str] = field(default_factory=set)

    cleanup_closed_states_interval_seconds: float = 300.0
    cleanup_closed_states_older_than_seconds: int = 3_600

    emit_lifecycle_events: bool = True

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.allowed_symbols = _normalize_symbol_set(self.allowed_symbols)
        self.allowed_exchanges = _normalize_exchange_set(self.allowed_exchanges)
        self.metadata = _metadata_copy(self.metadata)
        self.validate()

    def validate(self) -> None:
        _validate_non_negative_int("cooldown_seconds", self.cooldown_seconds)
        _validate_non_negative_int("max_snapshot_age_ms", self.max_snapshot_age_ms)
        _validate_non_negative_int("max_signal_age_ms", self.max_signal_age_ms)
        _validate_non_negative_int(
            "cleanup_closed_states_older_than_seconds",
            self.cleanup_closed_states_older_than_seconds,
        )
        _validate_non_negative_decimal("min_confidence", self.min_confidence)

        if self.cleanup_closed_states_interval_seconds < 0:
            raise ValueError("cleanup_closed_states_interval_seconds must be >= 0")

        if self.min_confidence > Decimal("1"):
            raise ValueError("min_confidence must be <= 1")


# ============================================================
# State model
# ============================================================

@dataclass(slots=True)
class SpreadStrategyState:
    """
    Уніфікований state для одного spread setup-а.

    status:
        idle / pending / open / blocked / closing / closed / cancelled / rejected

    bias:
        Strategy-specific bias:
        - arb;
        - LONG_BASIS;
        - SHORT_BASIS;
        - інший domain-specific напрямок.
    """

    key: str
    strategy: str
    symbol: str

    exchange_a: str
    exchange_b: str

    status: str = STATE_IDLE
    bias: str | None = None

    opened_at: datetime | None = None
    updated_at: datetime | None = None
    closed_at: datetime | None = None

    entry_value: Decimal | None = None
    entry_zscore: Decimal | None = None
    entry_net_edge: Decimal | None = None
    confidence: Decimal | None = None

    last_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.key = str(self.key).strip()
        self.strategy = str(self.strategy).strip()
        self.symbol = _normalize_symbol_value(self.symbol)
        self.exchange_a = _normalize_exchange_value(self.exchange_a)
        self.exchange_b = _normalize_exchange_value(self.exchange_b)
        self.metadata = _metadata_copy(self.metadata)

        if not self.key:
            raise ValueError("SpreadStrategyState.key must not be empty")
        if not self.strategy:
            raise ValueError("SpreadStrategyState.strategy must not be empty")
        if not self.symbol:
            raise ValueError("SpreadStrategyState.symbol must not be empty")

        if self.status not in {
            STATE_IDLE,
            STATE_PENDING,
            STATE_OPEN,
            STATE_BLOCKED,
            STATE_CLOSING,
            STATE_CLOSED,
            STATE_CANCELLED,
            STATE_REJECTED,
        }:
            raise ValueError(f"Unsupported spread strategy state status: {self.status!r}")

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_STATES

    @property
    def is_closed(self) -> bool:
        return self.status in CLOSED_STATES

    def to_payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "strategy": self.strategy,
            "symbol": self.symbol,
            "exchange_a": self.exchange_a,
            "exchange_b": self.exchange_b,
            "status": self.status,
            "bias": self.bias,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "entry_value": str(self.entry_value) if self.entry_value is not None else None,
            "entry_zscore": str(self.entry_zscore) if self.entry_zscore is not None else None,
            "entry_net_edge": (
                str(self.entry_net_edge) if self.entry_net_edge is not None else None
            ),
            "confidence": str(self.confidence) if self.confidence is not None else None,
            "last_reason": self.last_reason,
            "metadata": dict(self.metadata),
        }


# ============================================================
# Base strategy
# ============================================================

class BaseSpreadStrategy(ABC):
    """
    Production-grade базовий клас для strategy/spreads компонентів.

    Відповідальність:
    - constructor dependency injection через core.EventBus / core.Scheduler / config;
    - async lifecycle: register / unregister / start / stop;
    - EventBus subscribe wrappers для Event і payload handlers;
    - EventBus.emit() helpers для strategy-level signal intents;
    - Scheduler cleanup job для closed states;
    - centralized logger через core.logger.get_logger;
    - state management для spread setup-ів;
    - cooldown / dedup / freshness / allowlist фільтри;
    - базові stats/telemetry;
    - єдині input constants для analytics.spreads.* topics.

    Не відповідає за:
    - analytics розрахунки;
    - побудову SpreadSnapshot / SpreadSignal / ArbitrageOpportunity;
    - risk approval;
    - execution;
    - storage напряму;
    - прямі виклики analytics/risk/execution модулів.
    """

    SIGNAL_GENERATED_EVENT = STRATEGY_SIGNAL_GENERATED_EVENT
    SIGNAL_UPDATED_EVENT = STRATEGY_SIGNAL_UPDATED_EVENT
    SIGNAL_REJECTED_EVENT = STRATEGY_SIGNAL_REJECTED_EVENT
    SIGNAL_CANCELLED_EVENT = STRATEGY_SIGNAL_CANCELLED_EVENT
    SIGNAL_CLOSED_EVENT = STRATEGY_SIGNAL_CLOSED_EVENT

    STRATEGY_STARTED_EVENT = STRATEGY_STARTED_EVENT
    STRATEGY_STOPPED_EVENT = STRATEGY_STOPPED_EVENT
    STRATEGY_HEARTBEAT_EVENT = STRATEGY_HEARTBEAT_EVENT

    STRATEGY_NAME = "base_spread_strategy"

    def __init__(
        self,
        *,
        event_bus: EventBus,
        config: BaseSpreadStrategyConfig | None = None,
        scheduler: Scheduler | None = None,
        service_name: str | None = None,
    ) -> None:
        if event_bus is None:
            raise ValueError("event_bus must not be None")

        self._event_bus = event_bus
        self._scheduler = scheduler
        self._config = config or BaseSpreadStrategyConfig()

        resolved_service_name = service_name or self.STRATEGY_NAME

        self._logger = get_logger(
            __name__,
            service_name=resolved_service_name,
            event_type="strategy.spreads",
            strategy=self.STRATEGY_NAME,
        )

        self._running = False
        self._registered = False
        self._lock = asyncio.Lock()

        self._states: dict[str, SpreadStrategyState] = {}
        self._last_signal_times: dict[str, datetime] = {}
        self._last_event_times: dict[str, datetime] = {}

        self._subscriptions: list[Subscription] = []
        self._cleanup_job_id: str | None = None

        self._stats: dict[str, int] = self._build_base_stats()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_registered(self) -> bool:
        return self._registered

    @property
    def config(self) -> BaseSpreadStrategyConfig:
        return self._config

    @property
    def states(self) -> dict[str, SpreadStrategyState]:
        return self._states

    @property
    def active_states(self) -> list[SpreadStrategyState]:
        return [state for state in self._states.values() if state.is_active]

    @property
    def closed_states(self) -> list[SpreadStrategyState]:
        return [state for state in self._states.values() if state.is_closed]

    # ------------------------------------------------------------------
    # Required subclass API
    # ------------------------------------------------------------------

    @abstractmethod
    async def _subscribe_events(self) -> None:
        """
        Дочірня strategy сама визначає, які analytics.spreads.* події слухати.

        Приклади:
            await self._subscribe_payload(
                SPOT_FUTURES_SNAPSHOT_EVENT,
                self.on_spot_futures_snapshot,
                name="on_spot_futures_snapshot",
            )

            await self._subscribe_payload(
                ARBITRAGE_OPPORTUNITY_EVENT,
                self.on_arbitrage_opportunity,
                name="on_arbitrage_opportunity",
            )
        """
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """
        Дочірня strategy повертає розширену статистику поверх get_base_stats().
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def register(self) -> None:
        """
        Реєструє EventBus subscriptions і Scheduler cleanup job.

        Idempotent:
        - повторний register() не створює дублікати;
        - якщо _subscribe_events() падає, уже створені subscriptions прибираються.
        """
        if self._registered:
            self._logger.debug(
                "Spread strategy already registered | strategy=%s",
                self.STRATEGY_NAME,
            )
            return

        try:
            await self._subscribe_events()
            self._register_cleanup_job()
        except Exception as exc:
            await self._safe_cleanup_after_failed_register()
            self._mark_exception(
                "Failed to register spread strategy",
                exc,
                strategy=self.STRATEGY_NAME,
            )
            raise

        self._registered = True

        self._logger.info(
            "Spread strategy registered | strategy=%s subscriptions=%s cleanup_job_id=%s",
            self.STRATEGY_NAME,
            len(self._subscriptions),
            self._cleanup_job_id,
        )

    async def unregister(self) -> None:
        """
        Повністю відписує strategy від EventBus і disable-ить cleanup job.

        Використовувати під час shutdown/reconfigure, коли об'єкт більше
        не має отримувати analytics events.
        """
        for subscription in list(self._subscriptions):
            try:
                self._event_bus.unsubscribe(subscription)
            except Exception as exc:
                self._mark_exception(
                    "Failed to unsubscribe spread strategy handler",
                    exc,
                    pattern=getattr(subscription, "pattern", None),
                    handler=getattr(subscription, "name", None),
                )

        self._subscriptions.clear()
        self._disable_cleanup_job()

        self._registered = False

        self._logger.info(
            "Spread strategy unregistered | strategy=%s",
            self.STRATEGY_NAME,
        )

    async def start(self) -> None:
        """
        Запускає strategy.

        start():
        - не створює власних loops;
        - гарантує register(), якщо strategy ще не зареєстрована;
        - не містить trading/execution logic;
        - lifecycle event публікується best-effort.
        """
        if self._running:
            self._logger.warning(
                "Spread strategy already started | strategy=%s",
                self.STRATEGY_NAME,
            )
            return

        if not self._registered:
            await self.register()

        self._running = True

        self._logger.info(
            "Spread strategy started | strategy=%s",
            self.STRATEGY_NAME,
            extra=self._build_start_log_extra(),
        )

        if self._config.emit_lifecycle_events:
            await self._emit_lifecycle_event(
                self.STRATEGY_STARTED_EVENT,
                {
                    "strategy": self.STRATEGY_NAME,
                    "running": self._running,
                    "registered": self._registered,
                },
            )

    async def stop(self, *, unregister: bool = False) -> None:
        """
        Зупиняє strategy.

        За замовчуванням:
        - strategy перестає обробляти handlers через is_running guard у дочірніх класах;
        - subscriptions лишаються, щоб повторний start() був дешевим;
        - cleanup job може залишатися зареєстрованим.

        Якщо unregister=True:
        - EventBus subscriptions знімаються;
        - cleanup job disable-иться;
        - strategy треба буде register() перед повторним start().
        """
        if not self._running and not unregister:
            self._logger.debug(
                "Spread strategy already stopped | strategy=%s",
                self.STRATEGY_NAME,
            )
            return

        self._running = False

        if unregister:
            await self.unregister()

        self._logger.info(
            "Spread strategy stopped | strategy=%s unregister=%s",
            self.STRATEGY_NAME,
            unregister,
            extra=self._build_stop_log_extra(),
        )

        if self._config.emit_lifecycle_events:
            await self._emit_lifecycle_event(
                self.STRATEGY_STOPPED_EVENT,
                {
                    "strategy": self.STRATEGY_NAME,
                    "running": self._running,
                    "registered": self._registered,
                    "stats": self._stats.copy(),
                },
            )

    # ------------------------------------------------------------------
    # Scheduler helpers
    # ------------------------------------------------------------------

    def _register_cleanup_job(self) -> None:
        if self._scheduler is None:
            return

        if self._cleanup_job_id is not None:
            return

        interval = self._config.cleanup_closed_states_interval_seconds
        if interval <= 0:
            return

        try:
            existing_job = self._scheduler.get_job_by_name(
                f"{self.STRATEGY_NAME}.cleanup_closed_states"
            )
            if existing_job is not None:
                self._cleanup_job_id = existing_job.id
                return
        except AttributeError:
            pass
        except Exception:
            self._logger.debug(
                "Unable to check existing cleanup job by name | strategy=%s",
                self.STRATEGY_NAME,
                exc_info=True,
            )

        self._cleanup_job_id = self._scheduler.add_interval_job(
            name=f"{self.STRATEGY_NAME}.cleanup_closed_states",
            func=self._cleanup_closed_states_job,
            interval=interval,
            run_immediately=False,
            max_retries=1,
            retry_delay=1.0,
            timeout=10.0,
            allow_overlap=False,
            enabled=self._config.enabled,
        )

    def _disable_cleanup_job(self) -> None:
        if self._scheduler is None or self._cleanup_job_id is None:
            self._cleanup_job_id = None
            return

        try:
            self._scheduler.disable_job(self._cleanup_job_id)
        except KeyError:
            pass
        except Exception as exc:
            self._mark_exception(
                "Failed to disable spread strategy cleanup job",
                exc,
                cleanup_job_id=self._cleanup_job_id,
            )
        finally:
            self._cleanup_job_id = None

    async def _cleanup_closed_states_job(self) -> None:
        removed = self.cleanup_closed_states(
            older_than_seconds=self._config.cleanup_closed_states_older_than_seconds,
        )
        self._stats["cleanup_runs"] += 1
        self._stats["cleanup_removed_states"] += removed

        if removed:
            self._logger.info(
                "Closed spread strategy states cleaned | strategy=%s removed=%s",
                self.STRATEGY_NAME,
                removed,
            )

    # ------------------------------------------------------------------
    # EventBus helpers
    # ------------------------------------------------------------------

    async def _subscribe_payload(
        self,
        event_name: str,
        handler: PayloadHandler,
        *,
        name: str | None = None,
    ) -> Subscription:
        """
        Підписує payload-handler на EventBus topic.

        core.EventBus передає handler-у Event, тому wrapper передає в
        бізнес-handler тільки event.payload.
        """
        if not event_name or not event_name.strip():
            raise ValueError("event_name must not be empty")

        handler_name = name or getattr(handler, "__name__", "payload_handler")

        async def _event_wrapper(event: Event) -> None:
            try:
                result = handler(event.payload)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                self._mark_exception(
                    "Spread strategy payload handler failed",
                    exc,
                    topic=getattr(event, "topic", None),
                    event_id=getattr(event, "event_id", None),
                    correlation_id=getattr(event, "correlation_id", None),
                    handler=handler_name,
                )

        subscription = self._event_bus.subscribe(
            event_name,
            _event_wrapper,
            name=f"{self.STRATEGY_NAME}.{handler_name}",
        )
        self._subscriptions.append(subscription)

        self._logger.info(
            "Spread strategy subscribed | strategy=%s topic_pattern=%s handler=%s",
            self.STRATEGY_NAME,
            event_name,
            getattr(subscription, "name", handler_name),
        )

        return subscription

    async def _subscribe_event(
        self,
        event_name: str,
        handler: EventHandler,
        *,
        name: str | None = None,
    ) -> Subscription:
        """
        Підписує raw Event-handler на EventBus topic.
        """
        if not event_name or not event_name.strip():
            raise ValueError("event_name must not be empty")

        handler_name = name or getattr(handler, "__name__", "event_handler")

        async def _event_wrapper(event: Event) -> None:
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                self._mark_exception(
                    "Spread strategy event handler failed",
                    exc,
                    topic=getattr(event, "topic", None),
                    event_id=getattr(event, "event_id", None),
                    correlation_id=getattr(event, "correlation_id", None),
                    handler=handler_name,
                )

        subscription = self._event_bus.subscribe(
            event_name,
            _event_wrapper,
            name=f"{self.STRATEGY_NAME}.{handler_name}",
        )
        self._subscriptions.append(subscription)

        self._logger.info(
            "Spread strategy subscribed | strategy=%s topic_pattern=%s handler=%s",
            self.STRATEGY_NAME,
            event_name,
            getattr(subscription, "name", handler_name),
        )

        return subscription

    async def _subscribe(
        self,
        event_name: str,
        handler: PayloadHandler,
    ) -> Subscription:
        """
        Backward-compatible alias для старих дочірніх класів.

        Новий код краще пише явно:
            await self._subscribe_payload(...)
        """
        return await self._subscribe_payload(event_name, handler)

    async def _emit(
        self,
        event_name: str,
        payload: Any,
        *,
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        """
        Єдиний helper для EventBus.emit().

        Не використовує EventBus.publish(topic, payload), бо core.publish()
        у твоїй архітектурі працює з готовим Event object.
        """
        if not event_name or not event_name.strip():
            raise ValueError("event_name must not be empty")

        try:
            accepted = await self._event_bus.emit(
                event_name,
                payload,
                priority=priority,
                source=self.STRATEGY_NAME,
                correlation_id=correlation_id,
                headers=headers or {},
            )
            if not accepted:
                self._stats["events_rejected"] += 1
                self._logger.warning(
                    "Event rejected by EventBus | strategy=%s topic=%s",
                    self.STRATEGY_NAME,
                    event_name,
                )
            return accepted

        except Exception as exc:
            self._stats["events_failed"] += 1
            self._mark_exception(
                "Failed to emit spread strategy event",
                exc,
                topic=event_name,
                priority=getattr(priority, "value", str(priority)),
            )
            return False

    async def _publish(
        self,
        event_name: str,
        payload: Any,
    ) -> bool:
        """
        Backward-compatible alias.
        """
        return await self._emit(event_name, payload)

    async def _emit_lifecycle_event(
        self,
        event_name: str,
        payload: dict[str, Any],
    ) -> None:
        try:
            await self._emit(
                event_name,
                {
                    **payload,
                    "timestamp": self._utcnow().isoformat(),
                },
                priority=EventPriority.LOW,
            )
        except Exception:
            self._logger.exception(
                "Failed to emit spread strategy lifecycle event | strategy=%s topic=%s",
                self.STRATEGY_NAME,
                event_name,
            )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _build_state_key(self, *parts: Any) -> str:
        cleaned: list[str] = []
        for part in parts:
            if part is None:
                cleaned.append("na")
                continue

            value = str(part).strip()
            cleaned.append(value if value else "na")

        return "|".join(cleaned)

    def _get_state(self, key: str) -> SpreadStrategyState | None:
        return self._states.get(key)

    def _get_or_create_state(
        self,
        *,
        key: str,
        symbol: str,
        exchange_a: str,
        exchange_b: str,
        bias: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SpreadStrategyState:
        state = self._states.get(key)
        if state is not None:
            return state

        state = SpreadStrategyState(
            key=key,
            strategy=self.STRATEGY_NAME,
            symbol=symbol,
            exchange_a=exchange_a,
            exchange_b=exchange_b,
            bias=bias,
            metadata=metadata or {},
        )
        self._states[key] = state
        self._stats["state_created"] += 1
        return state

    def _set_state_open(
        self,
        state: SpreadStrategyState,
        *,
        bias: str | None = None,
        reason: str | None = None,
        entry_value: Decimal | None = None,
        entry_zscore: Decimal | None = None,
        entry_net_edge: Decimal | None = None,
        confidence: Decimal | None = None,
        metadata: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> None:
        current_time = now or self._utcnow()

        state.status = STATE_OPEN
        state.bias = bias or state.bias
        state.opened_at = state.opened_at or current_time
        state.updated_at = current_time
        state.closed_at = None

        state.entry_value = entry_value if entry_value is not None else state.entry_value
        state.entry_zscore = entry_zscore if entry_zscore is not None else state.entry_zscore
        state.entry_net_edge = (
            entry_net_edge if entry_net_edge is not None else state.entry_net_edge
        )
        state.confidence = confidence if confidence is not None else state.confidence
        state.last_reason = reason or state.last_reason

        if metadata:
            state.metadata.update(metadata)

        self._stats["state_updated"] += 1

    def _set_state_pending(
        self,
        state: SpreadStrategyState,
        *,
        bias: str | None = None,
        reason: str | None = None,
        confidence: Decimal | None = None,
        metadata: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> None:
        current_time = now or self._utcnow()

        state.status = STATE_PENDING
        state.bias = bias or state.bias
        state.updated_at = current_time
        state.closed_at = None
        state.last_reason = reason or state.last_reason
        state.confidence = confidence if confidence is not None else state.confidence

        if metadata:
            state.metadata.update(metadata)

        self._stats["state_updated"] += 1

    def _set_state_blocked(
        self,
        state: SpreadStrategyState,
        *,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> None:
        current_time = now or self._utcnow()

        state.status = STATE_BLOCKED
        state.updated_at = current_time
        state.last_reason = reason or state.last_reason

        if metadata:
            state.metadata.update(metadata)

        self._stats["state_updated"] += 1

    def _set_state_closing(
        self,
        state: SpreadStrategyState,
        *,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> None:
        current_time = now or self._utcnow()

        state.status = STATE_CLOSING
        state.updated_at = current_time
        state.last_reason = reason or state.last_reason

        if metadata:
            state.metadata.update(metadata)

        self._stats["state_updated"] += 1

    def _set_state_closed(
        self,
        state: SpreadStrategyState,
        *,
        status: str = STATE_CLOSED,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> None:
        if status not in CLOSED_STATES:
            raise ValueError(f"status must be one of {sorted(CLOSED_STATES)}, got {status!r}")

        current_time = now or self._utcnow()

        state.status = status
        state.updated_at = current_time
        state.closed_at = current_time
        state.last_reason = reason or state.last_reason

        if metadata:
            state.metadata.update(metadata)

        self._stats["state_closed"] += 1
        self._stats["state_updated"] += 1

    # ------------------------------------------------------------------
    # Strategy signal payload / emit helpers
    # ------------------------------------------------------------------

    def _build_strategy_payload(
        self,
        *,
        action: str,
        symbol: str,
        state_key: str,
        exchange_a: str | None = None,
        exchange_b: str | None = None,
        reason: str | None = None,
        confidence: Decimal | None = None,
        spread_type: str | None = None,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event_time = timestamp or self._utcnow()

        return {
            "strategy": self.STRATEGY_NAME,
            "action": action,
            "symbol": self._normalize_symbol(symbol),
            "exchange_a": self._normalize_exchange(exchange_a),
            "exchange_b": self._normalize_exchange(exchange_b),
            "state_key": state_key,
            "spread_type": spread_type,
            "reason": reason,
            "confidence": self._to_decimal_str(confidence),
            "timestamp": event_time.isoformat(),
            "metadata": metadata or {},
        }

    async def _publish_strategy_signal(
        self,
        *,
        event_name: str,
        action: str,
        symbol: str,
        state_key: str,
        exchange_a: str | None = None,
        exchange_b: str | None = None,
        reason: str | None = None,
        confidence: Decimal | None = None,
        spread_type: str | None = None,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        priority: EventPriority = EventPriority.HIGH,
    ) -> dict[str, Any]:
        payload = self._build_strategy_payload(
            action=action,
            symbol=symbol,
            state_key=state_key,
            exchange_a=exchange_a,
            exchange_b=exchange_b,
            reason=reason,
            confidence=confidence,
            spread_type=spread_type,
            timestamp=timestamp,
            metadata=metadata,
        )
        await self._emit(event_name, payload, priority=priority)
        return payload

    async def _emit_generated(
        self,
        *,
        action: str,
        symbol: str,
        state_key: str,
        exchange_a: str | None = None,
        exchange_b: str | None = None,
        reason: str | None = None,
        confidence: Decimal | None = None,
        spread_type: str | None = None,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._stats["signals_generated"] += 1
        return await self._publish_strategy_signal(
            event_name=self.SIGNAL_GENERATED_EVENT,
            action=action,
            symbol=symbol,
            state_key=state_key,
            exchange_a=exchange_a,
            exchange_b=exchange_b,
            reason=reason,
            confidence=confidence,
            spread_type=spread_type,
            timestamp=timestamp,
            metadata=metadata,
            priority=EventPriority.HIGH,
        )

    async def _emit_updated(
        self,
        *,
        action: str,
        symbol: str,
        state_key: str,
        exchange_a: str | None = None,
        exchange_b: str | None = None,
        reason: str | None = None,
        confidence: Decimal | None = None,
        spread_type: str | None = None,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._stats["signals_updated"] += 1
        return await self._publish_strategy_signal(
            event_name=self.SIGNAL_UPDATED_EVENT,
            action=action,
            symbol=symbol,
            state_key=state_key,
            exchange_a=exchange_a,
            exchange_b=exchange_b,
            reason=reason,
            confidence=confidence,
            spread_type=spread_type,
            timestamp=timestamp,
            metadata=metadata,
            priority=EventPriority.NORMAL,
        )

    async def _emit_rejected(
        self,
        *,
        action: str,
        symbol: str,
        state_key: str,
        exchange_a: str | None = None,
        exchange_b: str | None = None,
        reason: str | None = None,
        confidence: Decimal | None = None,
        spread_type: str | None = None,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._stats["signals_rejected"] += 1
        return await self._publish_strategy_signal(
            event_name=self.SIGNAL_REJECTED_EVENT,
            action=action,
            symbol=symbol,
            state_key=state_key,
            exchange_a=exchange_a,
            exchange_b=exchange_b,
            reason=reason,
            confidence=confidence,
            spread_type=spread_type,
            timestamp=timestamp,
            metadata=metadata,
            priority=EventPriority.NORMAL,
        )

    async def _emit_cancelled(
        self,
        *,
        action: str,
        symbol: str,
        state_key: str,
        exchange_a: str | None = None,
        exchange_b: str | None = None,
        reason: str | None = None,
        confidence: Decimal | None = None,
        spread_type: str | None = None,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._stats["signals_cancelled"] += 1
        return await self._publish_strategy_signal(
            event_name=self.SIGNAL_CANCELLED_EVENT,
            action=action,
            symbol=symbol,
            state_key=state_key,
            exchange_a=exchange_a,
            exchange_b=exchange_b,
            reason=reason,
            confidence=confidence,
            spread_type=spread_type,
            timestamp=timestamp,
            metadata=metadata,
            priority=EventPriority.NORMAL,
        )

    async def _emit_closed(
        self,
        *,
        action: str,
        symbol: str,
        state_key: str,
        exchange_a: str | None = None,
        exchange_b: str | None = None,
        reason: str | None = None,
        confidence: Decimal | None = None,
        spread_type: str | None = None,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._stats["signals_closed"] += 1
        return await self._publish_strategy_signal(
            event_name=self.SIGNAL_CLOSED_EVENT,
            action=action,
            symbol=symbol,
            state_key=state_key,
            exchange_a=exchange_a,
            exchange_b=exchange_b,
            reason=reason,
            confidence=confidence,
            spread_type=spread_type,
            timestamp=timestamp,
            metadata=metadata,
            priority=EventPriority.NORMAL,
        )

    # ------------------------------------------------------------------
    # Generic filters / helpers
    # ------------------------------------------------------------------

    def _utcnow(self) -> datetime:
        """
        Поки analytics models використовують naive UTC datetime, strategy
        залишає той самий контракт. Перехід на timezone-aware краще робити
        централізовано для всього проєкту.
        """
        return datetime.utcnow()

    def _normalize_symbol(self, symbol: str | None) -> str:
        return _normalize_symbol_value(symbol)

    def _normalize_exchange(self, exchange: str | None) -> str:
        return _normalize_exchange_value(exchange)

    def _to_decimal_str(self, value: Decimal | None) -> str | None:
        return str(value) if value is not None else None

    def _safe_isoformat(self, value: datetime | None) -> str | None:
        return value.isoformat() if value is not None else None

    def _event_age_ms(
        self,
        event_time: datetime | None,
        now: datetime | None = None,
    ) -> int | None:
        if event_time is None:
            return None

        current_time = now or self._utcnow()
        delta = current_time - event_time
        return max(int(delta.total_seconds() * 1000), 0)

    def _is_enabled(self) -> bool:
        return self._config.enabled

    def _is_symbol_allowed(self, symbol: str | None) -> bool:
        allowed = self._config.allowed_symbols
        if not allowed:
            return True

        normalized_symbol = self._normalize_symbol(symbol)
        return normalized_symbol in allowed

    def _are_exchanges_allowed(self, *exchanges: str | None) -> bool:
        allowed = self._config.allowed_exchanges
        if not allowed:
            return True

        for exchange in exchanges:
            normalized_exchange = self._normalize_exchange(exchange)
            if normalized_exchange and normalized_exchange not in allowed:
                return False

        return True

    def _is_confidence_ok(self, confidence: Decimal | None) -> bool:
        if confidence is None:
            return False
        return confidence >= self._config.min_confidence

    def _is_snapshot_fresh(self, timestamp: datetime | None) -> bool:
        age_ms = self._event_age_ms(timestamp)
        if age_ms is None:
            return False
        return age_ms <= self._config.max_snapshot_age_ms

    def _is_signal_fresh(self, timestamp: datetime | None) -> bool:
        age_ms = self._event_age_ms(timestamp)
        if age_ms is None:
            return False
        return age_ms <= self._config.max_signal_age_ms

    def _should_skip_by_cooldown(
        self,
        key: str,
        now: datetime | None = None,
    ) -> bool:
        current_time = now or self._utcnow()
        last_signal_at = self._last_signal_times.get(key)

        if last_signal_at is None:
            self._last_signal_times[key] = current_time
            return False

        cooldown = timedelta(seconds=self._config.cooldown_seconds)
        if (current_time - last_signal_at) < cooldown:
            self._stats["cooldown_skips"] += 1
            return True

        self._last_signal_times[key] = current_time
        return False

    def _mark_event_seen(
        self,
        key: str,
        timestamp: datetime | None = None,
    ) -> bool:
        """
        Повертає True, якщо подія дубльована по timestamp для цього key.
        """
        if timestamp is None:
            return False

        last_seen_at = self._last_event_times.get(key)
        if last_seen_at is not None and last_seen_at == timestamp:
            self._stats["duplicate_skips"] += 1
            return True

        self._last_event_times[key] = timestamp
        return False

    def _record_event_received(self) -> None:
        self._stats["events_received"] += 1

    def _reject_disabled(self) -> bool:
        if self._is_enabled():
            return False
        self._stats["disabled_skips"] += 1
        return True

    def _reject_symbol(self, symbol: str | None) -> bool:
        if self._is_symbol_allowed(symbol):
            return False
        self._stats["symbol_skips"] += 1
        return True

    def _reject_exchanges(self, *exchanges: str | None) -> bool:
        if self._are_exchanges_allowed(*exchanges):
            return False
        self._stats["exchange_skips"] += 1
        return True

    def _reject_confidence(self, confidence: Decimal | None) -> bool:
        if self._is_confidence_ok(confidence):
            return False
        self._stats["confidence_skips"] += 1
        return True

    def _reject_stale_snapshot(self, timestamp: datetime | None) -> bool:
        if self._is_snapshot_fresh(timestamp):
            return False
        self._stats["freshness_skips"] += 1
        return True

    def _reject_stale_signal(self, timestamp: datetime | None) -> bool:
        if self._is_signal_fresh(timestamp):
            return False
        self._stats["freshness_skips"] += 1
        return True

    # ------------------------------------------------------------------
    # Cleanup / stats / exception helpers
    # ------------------------------------------------------------------

    def cleanup_closed_states(
        self,
        *,
        older_than_seconds: int = 3_600,
        now: datetime | None = None,
    ) -> int:
        _validate_non_negative_int("older_than_seconds", older_than_seconds)

        current_time = now or self._utcnow()
        threshold = timedelta(seconds=older_than_seconds)

        keys_to_delete: list[str] = []
        for key, state in self._states.items():
            if not state.is_closed:
                continue
            if state.closed_at is None:
                continue
            if (current_time - state.closed_at) >= threshold:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            self._states.pop(key, None)
            self._last_signal_times.pop(key, None)
            self._last_event_times.pop(key, None)

        return len(keys_to_delete)

    def _build_base_stats(self) -> dict[str, int]:
        return {
            "events_received": 0,
            "events_rejected": 0,
            "events_failed": 0,
            "signals_generated": 0,
            "signals_updated": 0,
            "signals_rejected": 0,
            "signals_cancelled": 0,
            "signals_closed": 0,
            "cooldown_skips": 0,
            "disabled_skips": 0,
            "confidence_skips": 0,
            "freshness_skips": 0,
            "symbol_skips": 0,
            "exchange_skips": 0,
            "duplicate_skips": 0,
            "state_created": 0,
            "state_updated": 0,
            "state_closed": 0,
            "cleanup_runs": 0,
            "cleanup_removed_states": 0,
            "exceptions": 0,
        }

    def _build_start_log_extra(self) -> dict[str, Any]:
        return {
            "strategy": self.STRATEGY_NAME,
            "enabled": self._config.enabled,
            "cooldown_seconds": self._config.cooldown_seconds,
            "min_confidence": str(self._config.min_confidence),
            "max_snapshot_age_ms": self._config.max_snapshot_age_ms,
            "max_signal_age_ms": self._config.max_signal_age_ms,
            "allowed_symbols_count": len(self._config.allowed_symbols),
            "allowed_exchanges_count": len(self._config.allowed_exchanges),
            "cleanup_closed_states_interval_seconds": (
                self._config.cleanup_closed_states_interval_seconds
            ),
            "cleanup_closed_states_older_than_seconds": (
                self._config.cleanup_closed_states_older_than_seconds
            ),
            "subscriptions": len(self._subscriptions),
            "cleanup_job_id": self._cleanup_job_id,
        }

    def _build_stop_log_extra(self) -> dict[str, Any]:
        return {
            "strategy": self.STRATEGY_NAME,
            "stats": self._stats.copy(),
            "active_states": len(self.active_states),
            "closed_states": len(self.closed_states),
            "total_states": len(self._states),
            "registered": self._registered,
            "subscriptions": len(self._subscriptions),
            "cleanup_job_id": self._cleanup_job_id,
        }

    def get_base_stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "running": self._running,
            "registered": self._registered,
            "enabled": self._config.enabled,
            "strategy": self.STRATEGY_NAME,
            "states_total": len(self._states),
            "states_active": len(self.active_states),
            "states_closed": len(self.closed_states),
            "subscriptions": len(self._subscriptions),
            "cleanup_job_id": self._cleanup_job_id,
        }

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def _run_safely(
        self,
        operation_name: str,
        coro: Awaitable[Any],
        **context: Any,
    ) -> Any:
        try:
            return await coro
        except Exception as exc:
            self._mark_exception(
                f"Strategy operation failed: {operation_name}",
                exc,
                **context,
            )
            return None

    def _mark_exception(
        self,
        message: str,
        exc: Exception,
        **context: Any,
    ) -> None:
        self._stats["exceptions"] += 1
        self._logger.exception(
            message,
            extra={
                "strategy": self.STRATEGY_NAME,
                **context,
                "error": str(exc),
            },
        )

    async def _safe_cleanup_after_failed_register(self) -> None:
        for subscription in list(self._subscriptions):
            try:
                self._event_bus.unsubscribe(subscription)
            except Exception:
                self._logger.exception(
                    "Failed to cleanup subscription after register failure | strategy=%s",
                    self.STRATEGY_NAME,
                )

        self._subscriptions.clear()
        self._disable_cleanup_job()
        self._registered = False


__all__ = [
    # Input analytics.spreads events
    "SPOT_FUTURES_SNAPSHOT_EVENT",
    "CROSS_EXCHANGE_SNAPSHOT_EVENT",
    "SPREAD_SIGNAL_EVENT",
    "ARBITRAGE_OPPORTUNITY_EVENT",

    # Output strategy events
    "STRATEGY_SIGNAL_GENERATED_EVENT",
    "STRATEGY_SIGNAL_UPDATED_EVENT",
    "STRATEGY_SIGNAL_REJECTED_EVENT",
    "STRATEGY_SIGNAL_CANCELLED_EVENT",
    "STRATEGY_SIGNAL_CLOSED_EVENT",
    "STRATEGY_STARTED_EVENT",
    "STRATEGY_STOPPED_EVENT",
    "STRATEGY_HEARTBEAT_EVENT",

    # State constants
    "STATE_IDLE",
    "STATE_PENDING",
    "STATE_OPEN",
    "STATE_BLOCKED",
    "STATE_CLOSING",
    "STATE_CLOSED",
    "STATE_CANCELLED",
    "STATE_REJECTED",
    "ACTIVE_STATES",
    "CLOSED_STATES",

    # Types
    "PayloadHandler",
    "EventHandler",
    "BaseSpreadStrategyConfig",
    "SpreadStrategyState",
    "BaseSpreadStrategy",
]