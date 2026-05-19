from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from analytics.spreads.config import (
    DEFAULT_ARBITRAGE_OPPORTUNITY_TOPIC,
    DEFAULT_CROSS_EXCHANGE_SNAPSHOT_TOPIC,
    DEFAULT_SPOT_FUTURES_SNAPSHOT_TOPIC,
    DEFAULT_SPREAD_SIGNAL_TOPIC,
)
from analytics.spreads.enums import (
    OpportunityStatus,
    QuoteValidity,
    SpreadDirection,
    SpreadRegime,
    SpreadSignalType,
    SpreadType,
    parse_instrument_type,
    parse_pricing_source,
    parse_spread_type,
)
from analytics.spreads.models import (
    DEFAULT_TIMEFRAME,
    ArbitrageOpportunity,
    RollingStats,
    SpreadSignal,
    SpreadSnapshot,
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
# Validation / normalization helpers
# ============================================================

DECIMAL_ZERO = Decimal("0")


def _validate_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


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


def _to_decimal(value: Any, default: Decimal | None = None) -> Decimal | None:
    if value is None:
        return default

    if isinstance(value, Decimal):
        return value

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _datetime_from_payload(value: Any) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    if isinstance(value, (int, float)):
        try:
            timestamp = float(value)
            if timestamp > 10_000_000_000:
                timestamp /= 1000
            return datetime.utcfromtimestamp(timestamp)
        except (OverflowError, OSError, ValueError):
            return None

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None

        try:
            timestamp = float(raw)
            if timestamp > 10_000_000_000:
                timestamp /= 1000
            return datetime.utcfromtimestamp(timestamp)
        except (OverflowError, OSError, ValueError):
            pass

        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    return None


def _first_present(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload.get(key) is not None:
            return payload.get(key)
    return None


def _nested_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else None


def _scope_value(
    payload: Mapping[str, Any],
    scope_key: str,
    field_name: str,
    *fallback_keys: str,
) -> Any:
    scope = _nested_mapping(payload, scope_key)
    if scope is not None and scope.get(field_name) is not None:
        return scope.get(field_name)

    return _first_present(payload, *fallback_keys)


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw)


def _safe_str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    return raw or None


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
        self.min_confidence = _to_decimal(self.min_confidence, DECIMAL_ZERO) or DECIMAL_ZERO
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
        self.entry_value = _to_decimal(self.entry_value)
        self.entry_zscore = _to_decimal(self.entry_zscore)
        self.entry_net_edge = _to_decimal(self.entry_net_edge)
        self.confidence = _to_decimal(self.confidence)
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
    - normalizing bridge для analytics.spreads dict/model payload;
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

        `_subscribe_payload()` автоматично нормалізує:
        - spot_futures/cross_exchange snapshot topic -> SpreadSnapshot;
        - spread signal topic -> SpreadSignal;
        - arbitrage opportunity topic -> ArbitrageOpportunity.

        Тому дочірні handlers можуть лишатися типізованими під analytics-моделі.
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

        Якщо unregister=True:
        - EventBus subscriptions знімаються;
        - cleanup job disable-иться.
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

        Важливо:
        analytics.spreads може публікувати як dataclass-модель, так і dict payload.
        Цей wrapper нормалізує dict payload назад у canonical analytics model.
        """
        if not event_name or not event_name.strip():
            raise ValueError("event_name must not be empty")

        event_name = event_name.strip()
        handler_name = name or getattr(handler, "__name__", "payload_handler")

        async def _event_wrapper(event: Event) -> None:
            try:
                payload = self._normalize_analytics_payload(
                    topic=event_name,
                    payload=event.payload,
                    event=event,
                )
                result = handler(payload)
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
                    payload_type=type(getattr(event, "payload", None)).__name__,
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

        event_name = event_name.strip()
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
        Backward-compatible alias.
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
    # Analytics payload normalization bridge
    # ------------------------------------------------------------------

    def _normalize_analytics_payload(
        self,
        *,
        topic: str,
        payload: Any,
        event: Event | None = None,
    ) -> Any:
        """
        Нормалізує EventBus payload із analytics.spreads.

        Чому це потрібно:
        - analyzer-и можуть емiтити dataclass-модель;
        - BaseSpreadAnalyzer перед EventBus часто конвертує модель у dict через to_payload();
        - strategy handlers мають працювати стабільно в обох випадках.
        """
        try:
            if topic == SPOT_FUTURES_SNAPSHOT_EVENT:
                return self._normalize_spread_snapshot_payload(payload)

            if topic == CROSS_EXCHANGE_SNAPSHOT_EVENT:
                return self._normalize_spread_snapshot_payload(payload)

            if topic == SPREAD_SIGNAL_EVENT:
                return self._normalize_spread_signal_payload(payload)

            if topic == ARBITRAGE_OPPORTUNITY_EVENT:
                return self._normalize_arbitrage_opportunity_payload(payload)

            return payload

        except Exception as exc:
            self._stats["payload_normalization_failures"] += 1
            self._mark_exception(
                "Failed to normalize analytics.spreads payload",
                exc,
                topic=topic,
                event_id=getattr(event, "event_id", None),
                correlation_id=getattr(event, "correlation_id", None),
                payload_type=type(payload).__name__,
            )
            return payload

    def _normalize_spread_snapshot_payload(self, payload: Any) -> SpreadSnapshot:
        if isinstance(payload, SpreadSnapshot):
            return payload

        if not isinstance(payload, Mapping):
            raise TypeError(f"SpreadSnapshot payload must be Mapping, got {type(payload)!r}")

        stats_payload = payload.get("stats")
        stats = self._normalize_rolling_stats_payload(stats_payload)

        metadata = _metadata_copy(payload.get("metadata"))

        return SpreadSnapshot(
            spread_type=parse_spread_type(payload.get("spread_type")),
            symbol=str(payload["symbol"]),
            timeframe=str(payload.get("timeframe") or DEFAULT_TIMEFRAME),
            leg_a_exchange=str(
                _scope_value(payload, "leg_a_scope", "exchange", "leg_a_exchange", "exchange_a")
            ),
            leg_b_exchange=str(
                _scope_value(payload, "leg_b_scope", "exchange", "leg_b_exchange", "exchange_b")
            ),
            leg_a_type=parse_instrument_type(
                _first_present(payload, "leg_a_type", "instrument_type_a", "instrument_type")
            ),
            leg_b_type=parse_instrument_type(
                _first_present(payload, "leg_b_type", "instrument_type_b", "instrument_type")
            ),
            leg_a_market_type=_safe_str_or_none(
                _scope_value(
                    payload,
                    "leg_a_scope",
                    "market_type",
                    "leg_a_market_type",
                    "market_type_a",
                    "market_type",
                )
            ),
            leg_b_market_type=_safe_str_or_none(
                _scope_value(
                    payload,
                    "leg_b_scope",
                    "market_type",
                    "leg_b_market_type",
                    "market_type_b",
                    "market_type",
                )
            ),
            leg_a_exchange_symbol=_safe_str_or_none(
                _first_present(payload, "leg_a_exchange_symbol", "exchange_symbol_a")
            ),
            leg_b_exchange_symbol=_safe_str_or_none(
                _first_present(payload, "leg_b_exchange_symbol", "exchange_symbol_b")
            ),
            pricing_source=parse_pricing_source(payload.get("pricing_source")),
            raw_spread=_to_decimal(payload.get("raw_spread")),
            spread_pct=_to_decimal(payload.get("spread_pct")),
            spread_bps=_to_decimal(payload.get("spread_bps")),
            net_spread=_to_decimal(payload.get("net_spread")),
            basis=_to_decimal(payload.get("basis")),
            funding_adjusted_spread=_to_decimal(payload.get("funding_adjusted_spread")),
            direction=SpreadDirection.from_value(
                payload.get("direction"),
                default=SpreadDirection.FLAT,
            ),
            regime=SpreadRegime.from_value(
                payload.get("regime"),
                default=SpreadRegime.NORMAL,
            ),
            stats=stats,
            leg_a_bid=_to_decimal(payload.get("leg_a_bid")),
            leg_a_ask=_to_decimal(payload.get("leg_a_ask")),
            leg_b_bid=_to_decimal(payload.get("leg_b_bid")),
            leg_b_ask=_to_decimal(payload.get("leg_b_ask")),
            leg_a_mid=_to_decimal(payload.get("leg_a_mid")),
            leg_b_mid=_to_decimal(payload.get("leg_b_mid")),
            estimated_fees=_to_decimal(payload.get("estimated_fees")),
            estimated_slippage=_to_decimal(payload.get("estimated_slippage")),
            quote_validity=QuoteValidity.from_value(
                payload.get("quote_validity"),
                default=QuoteValidity.VALID,
            ),
            timestamp=_datetime_from_payload(payload.get("timestamp")) or self._utcnow(),
            metadata=metadata,
        )

    def _normalize_spread_signal_payload(self, payload: Any) -> SpreadSignal:
        if isinstance(payload, SpreadSignal):
            return payload

        if not isinstance(payload, Mapping):
            raise TypeError(f"SpreadSignal payload must be Mapping, got {type(payload)!r}")

        metadata = _metadata_copy(payload.get("metadata"))

        return SpreadSignal(
            signal_type=SpreadSignalType.from_value(payload.get("signal_type"), strict=True),
            spread_type=parse_spread_type(payload.get("spread_type")),
            symbol=str(payload["symbol"]),
            message=str(payload.get("message") or payload.get("reason") or "Spread signal"),
            value=_to_decimal(payload.get("value")),
            threshold=_to_decimal(payload.get("threshold")),
            confidence=_to_decimal(payload.get("confidence")),
            exchange_a=_safe_str_or_none(payload.get("exchange_a")),
            exchange_b=_safe_str_or_none(payload.get("exchange_b")),
            market_type_a=_safe_str_or_none(
                _scope_value(payload, "leg_a_scope", "market_type", "market_type_a")
            ),
            market_type_b=_safe_str_or_none(
                _scope_value(payload, "leg_b_scope", "market_type", "market_type_b")
            ),
            timeframe=str(payload.get("timeframe") or DEFAULT_TIMEFRAME),
            exchange_symbol_a=_safe_str_or_none(payload.get("exchange_symbol_a")),
            exchange_symbol_b=_safe_str_or_none(payload.get("exchange_symbol_b")),
            timestamp=_datetime_from_payload(payload.get("timestamp")) or self._utcnow(),
            metadata=metadata,
        )

    def _normalize_arbitrage_opportunity_payload(self, payload: Any) -> ArbitrageOpportunity:
        if isinstance(payload, ArbitrageOpportunity):
            return payload

        if isinstance(payload, Mapping) and isinstance(payload.get("opportunity"), Mapping):
            payload = payload["opportunity"]

        if not isinstance(payload, Mapping):
            raise TypeError(
                f"ArbitrageOpportunity payload must be Mapping, got {type(payload)!r}"
            )

        metadata = _metadata_copy(payload.get("metadata"))

        buy_scope = _nested_mapping(payload, "buy_scope")
        sell_scope = _nested_mapping(payload, "sell_scope")

        buy_market_type = _safe_str_or_none(
            _first_present(payload, "buy_market_type")
            or (buy_scope.get("market_type") if buy_scope else None)
        )
        sell_market_type = _safe_str_or_none(
            _first_present(payload, "sell_market_type")
            or (sell_scope.get("market_type") if sell_scope else None)
        )

        timeframe = str(
            payload.get("timeframe")
            or (buy_scope.get("timeframe") if buy_scope else None)
            or (sell_scope.get("timeframe") if sell_scope else None)
            or DEFAULT_TIMEFRAME
        )

        return ArbitrageOpportunity(
            symbol=str(payload["symbol"]),
            buy_exchange=str(payload["buy_exchange"]),
            sell_exchange=str(payload["sell_exchange"]),
            buy_instrument_type=parse_instrument_type(payload.get("buy_instrument_type")),
            sell_instrument_type=parse_instrument_type(payload.get("sell_instrument_type")),
            buy_price=_to_decimal(payload.get("buy_price"), DECIMAL_ZERO) or DECIMAL_ZERO,
            sell_price=_to_decimal(payload.get("sell_price"), DECIMAL_ZERO) or DECIMAL_ZERO,
            gross_edge=_to_decimal(payload.get("gross_edge"), DECIMAL_ZERO) or DECIMAL_ZERO,
            estimated_fees=(
                _to_decimal(payload.get("estimated_fees"), DECIMAL_ZERO) or DECIMAL_ZERO
            ),
            estimated_slippage=(
                _to_decimal(payload.get("estimated_slippage"), DECIMAL_ZERO)
                or DECIMAL_ZERO
            ),
            net_edge=_to_decimal(payload.get("net_edge"), DECIMAL_ZERO) or DECIMAL_ZERO,
            spread_pct=_to_decimal(payload.get("spread_pct")),
            spread_bps=_to_decimal(payload.get("spread_bps")),
            confidence=_to_decimal(payload.get("confidence")),
            status=OpportunityStatus.from_value(
                payload.get("status"),
                default=OpportunityStatus.ACTIVE,
            ),
            timestamp=_datetime_from_payload(payload.get("timestamp")) or self._utcnow(),
            expires_at=_datetime_from_payload(payload.get("expires_at")),
            buy_market_type=buy_market_type,
            sell_market_type=sell_market_type,
            timeframe=timeframe,
            buy_exchange_symbol=_safe_str_or_none(payload.get("buy_exchange_symbol")),
            sell_exchange_symbol=_safe_str_or_none(payload.get("sell_exchange_symbol")),
            metadata=metadata,
        )

    def _normalize_rolling_stats_payload(self, payload: Any) -> RollingStats | None:
        if payload is None:
            return None

        if isinstance(payload, RollingStats):
            return payload

        if not isinstance(payload, Mapping):
            raise TypeError(f"RollingStats payload must be Mapping, got {type(payload)!r}")

        return RollingStats(
            count=_to_int(payload.get("count"), default=0),
            mean=_to_decimal(payload.get("mean")),
            std=_to_decimal(payload.get("std")),
            min_value=_to_decimal(payload.get("min_value")),
            max_value=_to_decimal(payload.get("max_value")),
            ema=_to_decimal(payload.get("ema")),
            last_value=_to_decimal(payload.get("last_value")),
            zscore=_to_decimal(payload.get("zscore")),
            percentile_rank=_to_decimal(payload.get("percentile_rank")),
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

    def _build_scoped_state_key(
        self,
        *,
        spread_type: SpreadType | str | None,
        symbol: str,
        exchange_a: str | None,
        exchange_b: str | None,
        market_type_a: str | None = None,
        market_type_b: str | None = None,
        timeframe: str | None = None,
        suffix: str | None = None,
    ) -> str:
        """
        Новий preferred key format для узгодження з analytics scope:
        spread_type | symbol | exchange_a | market_type_a | exchange_b | market_type_b | timeframe | suffix

        Старий _build_state_key лишається для backward compatibility.
        """
        return self._build_state_key(
            _enum_value(spread_type) or "spread",
            self._normalize_symbol(symbol),
            self._normalize_exchange(exchange_a),
            market_type_a or "na",
            self._normalize_exchange(exchange_b),
            market_type_b or "na",
            timeframe or DEFAULT_TIMEFRAME,
            suffix,
        )

    def _build_snapshot_state_key(self, snapshot: SpreadSnapshot, *, suffix: str | None = None) -> str:
        return self._build_scoped_state_key(
            spread_type=snapshot.spread_type,
            symbol=snapshot.symbol,
            exchange_a=snapshot.leg_a_exchange,
            exchange_b=snapshot.leg_b_exchange,
            market_type_a=snapshot.leg_a_market_type,
            market_type_b=snapshot.leg_b_market_type,
            timeframe=snapshot.timeframe,
            suffix=suffix,
        )

    def _build_signal_state_key(self, signal: SpreadSignal, *, suffix: str | None = None) -> str:
        return self._build_scoped_state_key(
            spread_type=signal.spread_type,
            symbol=signal.symbol,
            exchange_a=signal.exchange_a,
            exchange_b=signal.exchange_b,
            market_type_a=signal.market_type_a,
            market_type_b=signal.market_type_b,
            timeframe=signal.timeframe,
            suffix=suffix,
        )

    def _build_opportunity_state_key(
        self,
        opportunity: ArbitrageOpportunity,
        *,
        use_opportunity_key: bool = False,
    ) -> str:
        if use_opportunity_key:
            opportunity_key = getattr(opportunity, "opportunity_key", None)
            if opportunity_key:
                return str(opportunity_key)

        return self._build_scoped_state_key(
            spread_type=SpreadType.CROSS_EXCHANGE,
            symbol=opportunity.symbol,
            exchange_a=opportunity.buy_exchange,
            exchange_b=opportunity.sell_exchange,
            market_type_a=getattr(opportunity, "buy_market_type", None),
            market_type_b=getattr(opportunity, "sell_market_type", None),
            timeframe=getattr(opportunity, "timeframe", DEFAULT_TIMEFRAME),
            suffix=opportunity.buy_instrument_type.value,
        )

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

    def get_base_stats(self) -> dict[str, Any]:
        return {
            **self._stats.copy(),
            "strategy": self.STRATEGY_NAME,
            "running": self._running,
            "registered": self._registered,
            "subscriptions": len(self._subscriptions),
            "cleanup_job_id": self._cleanup_job_id,
            "states_total": len(self._states),
            "states_active": len(self.active_states),
            "states_closed": len(self.closed_states),
        }

    def _build_base_stats(self) -> dict[str, int]:
        return {
            "events_received": 0,
            "events_rejected": 0,
            "events_failed": 0,
            "exceptions": 0,
            "payload_normalization_failures": 0,
            "state_created": 0,
            "state_updated": 0,
            "state_closed": 0,
            "signals_generated": 0,
            "signals_updated": 0,
            "signals_rejected": 0,
            "signals_cancelled": 0,
            "signals_closed": 0,
            "cooldown_skips": 0,
            "duplicate_skips": 0,
            "disabled_skips": 0,
            "symbol_skips": 0,
            "exchange_skips": 0,
            "confidence_skips": 0,
            "freshness_skips": 0,
            "cleanup_runs": 0,
            "cleanup_removed_states": 0,
        }

    def _build_start_log_extra(self) -> dict[str, Any]:
        return {
            "enabled": self._config.enabled,
            "cooldown_seconds": self._config.cooldown_seconds,
            "min_confidence": self._to_decimal_str(self._config.min_confidence),
            "max_snapshot_age_ms": self._config.max_snapshot_age_ms,
            "max_signal_age_ms": self._config.max_signal_age_ms,
            "allowed_symbols": sorted(self._config.allowed_symbols),
            "allowed_exchanges": sorted(self._config.allowed_exchanges),
            "subscriptions": len(self._subscriptions),
            "cleanup_job_id": self._cleanup_job_id,
            "analytics_input_topics": [
                SPOT_FUTURES_SNAPSHOT_EVENT,
                CROSS_EXCHANGE_SNAPSHOT_EVENT,
                SPREAD_SIGNAL_EVENT,
                ARBITRAGE_OPPORTUNITY_EVENT,
            ],
            "scope": "spread_type:symbol:exchange_a:market_type_a:exchange_b:market_type_b:timeframe",
        }

    def _build_stop_log_extra(self) -> dict[str, Any]:
        return {
            "stats": self._stats.copy(),
            "subscriptions": len(self._subscriptions),
            "cleanup_job_id": self._cleanup_job_id,
            "states_total": len(self._states),
        }

    def _mark_exception(
        self,
        message: str,
        exc: Exception,
        **extra: Any,
    ) -> None:
        self._stats["exceptions"] += 1
        self._logger.exception(
            message,
            extra={
                "error": str(exc),
                "strategy": self.STRATEGY_NAME,
                **extra,
            },
        )

    async def _safe_cleanup_after_failed_register(self) -> None:
        try:
            await self.unregister()
        except Exception as exc:
            self._mark_exception(
                "Failed to cleanup spread strategy after failed register",
                exc,
                strategy=self.STRATEGY_NAME,
            )