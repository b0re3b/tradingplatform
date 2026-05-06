from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Iterable, Mapping

from analytics.funding.enums import (
    FundingBias,
    FundingDataSource,
    FundingDivergenceType,
    FundingExtremeType,
    FundingFlipType,
    FundingPressureDirection,
    FundingPressureLevel,
    FundingRegime,
    FundingSignalType,
    FundingTimeframe,
)
from analytics.funding.models import (
    FundingDivergenceEvent,
    FundingExtremeEvent,
    FundingFlipEvent,
    FundingPressureState,
    FundingRegimeState,
    FundingSignal,
    FundingStatistics,
    FundingSnapshot,
)
from core.event_bus import Event, EventBus, EventPriority, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler


# =============================================================================
# Generic helpers
# =============================================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        return ensure_utc(value)

    if isinstance(value, (int, float)):
        try:
            timestamp = float(value) / 1000.0 if float(value) > 10_000_000_000 else float(value)
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return ensure_utc(datetime.fromisoformat(raw.replace("Z", "+00:00")))
        except ValueError:
            try:
                numeric = float(raw)
                return parse_datetime(numeric)
            except ValueError:
                return None

    return None


def serialize_for_event(value: Any) -> Any:
    """Convert common non-JSON-safe values before putting them into EventBus payloads."""
    if isinstance(value, datetime):
        normalized = ensure_utc(value)
        return normalized.isoformat() if normalized else None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return serialize_for_event(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): serialize_for_event(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_for_event(item) for item in value]
    return value


def unwrap_analytics_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """
    Accept both direct analytics payloads and FundingAnalyticsEvent.to_dict() envelopes.

    Examples accepted:
    - {"symbol": "BTCUSDT", "regime": "positive", ...}
    - {"event_type": "regime", "payload": {"symbol": "BTCUSDT", ...}}
    - {"payload": {"regime_state": {...}, "pressure_state": {...}}}
    """
    raw = dict(payload)
    inner = raw.get("payload")
    if isinstance(inner, Mapping):
        inner_dict = dict(inner)
        for key in ("snapshot", "statistics", "regime_state", "pressure_state", "extreme_event", "divergence_event", "flip_event", "signal"):
            if isinstance(inner_dict.get(key), Mapping):
                nested = dict(inner_dict[key])
                nested.setdefault("_envelope", raw)
                return nested
        inner_dict.setdefault("_envelope", raw)
        return inner_dict
    return raw


# =============================================================================
# Enums / Dataclasses
# =============================================================================


class FundingSetupStatus(str, Enum):
    """Current lifecycle status of a funding strategy setup."""

    IDLE = "idle"
    SETUP_DETECTED = "setup_detected"
    CONFIRMED = "confirmed"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"
    COOLDOWN = "cooldown"


class FundingStrategyDirection(str, Enum):
    """Expected trade direction produced by the strategy layer."""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass(slots=True)
class BaseFundingStrategyConfig:
    """
    Base config for strategy/funding/* runtime modules.

    Core-aligned responsibilities:
    - EventBus topic names and priorities
    - Scheduler-managed cleanup
    - optional funding analytics aggregate/signal subscriptions
    - state lifecycle limits and stale-event protection
    """

    setup_ttl_sec: float = 15 * 60.0
    cooldown_sec: float = 5 * 60.0
    event_stale_after_sec: float = 10 * 60.0
    state_lock_timeout_sec: float = 3.0

    allow_reconfirm: bool = False
    emit_setup_events: bool = True
    emit_invalidation_events: bool = True
    emit_expiration_events: bool = True
    emit_confirmation_events: bool = True

    cleanup_expired_on_access: bool = True
    attach_full_state_on_emit: bool = True
    attach_full_analytics_context_on_emit: bool = True

    strategy_namespace: str = "strategy.funding.base"
    source_name: str = "funding_strategy_base"
    service_name: str = "funding_strategy_base"

    setup_priority: EventPriority = EventPriority.NORMAL
    confirmation_priority: EventPriority = EventPriority.HIGH
    invalidation_priority: EventPriority = EventPriority.NORMAL
    expiration_priority: EventPriority = EventPriority.LOW

    enable_scheduler_cleanup: bool = True
    cleanup_interval_sec: float = 30.0
    cleanup_job_timeout_sec: float = 10.0

    # Optional base-level analytics subscriptions. Concrete strategies may enable these
    # to receive atomic context updates and normalized funding signals.
    enable_funding_updated_subscription: bool = False
    enable_funding_signal_subscription: bool = False
    funding_updated_event_name: str = "analytics.funding.updated"
    funding_signal_event_name: str = "analytics.funding.signal"

    def validate(self) -> None:
        if self.setup_ttl_sec <= 0:
            raise ValueError("setup_ttl_sec must be > 0")
        if self.cooldown_sec < 0:
            raise ValueError("cooldown_sec must be >= 0")
        if self.event_stale_after_sec <= 0:
            raise ValueError("event_stale_after_sec must be > 0")
        if self.state_lock_timeout_sec <= 0:
            raise ValueError("state_lock_timeout_sec must be > 0")
        if self.cleanup_interval_sec <= 0:
            raise ValueError("cleanup_interval_sec must be > 0")
        if self.cleanup_job_timeout_sec <= 0:
            raise ValueError("cleanup_job_timeout_sec must be > 0")
        if not self.strategy_namespace.strip():
            raise ValueError("strategy_namespace must not be empty")
        if not self.source_name.strip():
            raise ValueError("source_name must not be empty")
        if not self.service_name.strip():
            raise ValueError("service_name must not be empty")
        if self.enable_funding_updated_subscription and not self.funding_updated_event_name.strip():
            raise ValueError("funding_updated_event_name must not be empty")
        if self.enable_funding_signal_subscription and not self.funding_signal_event_name.strip():
            raise ValueError("funding_signal_event_name must not be empty")


@dataclass(slots=True)
class FundingStrategyState:
    """Local per-symbol/exchange state for funding strategies."""

    symbol: str
    exchange: str

    status: FundingSetupStatus = FundingSetupStatus.IDLE
    direction: FundingStrategyDirection = FundingStrategyDirection.NEUTRAL

    strategy_name: str = ""
    setup_type: str | None = None

    score: float = 0.0
    confidence: float = 0.0

    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    confirmed_at: datetime | None = None
    invalidated_at: datetime | None = None
    expires_at: datetime | None = None
    cooldown_until: datetime | None = None

    reason: str | None = None
    reasons: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    last_snapshot: FundingSnapshot | dict[str, Any] | None = None
    last_statistics: FundingStatistics | dict[str, Any] | None = None
    last_regime: FundingRegimeState | dict[str, Any] | None = None
    last_pressure: FundingPressureState | dict[str, Any] | None = None
    last_extreme: FundingExtremeEvent | dict[str, Any] | None = None
    last_divergence: FundingDivergenceEvent | dict[str, Any] | None = None
    last_flip: FundingFlipEvent | dict[str, Any] | None = None
    last_signal: FundingSignal | dict[str, Any] | None = None
    last_funding_updated_payload: dict[str, Any] | None = None

    setup_event_time: datetime | None = None
    confirmation_event_time: datetime | None = None
    last_analytics_update_time: datetime | None = None
    last_signal_time: datetime | None = None
    last_emit_time: datetime | None = None

    emit_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper().strip()
        self.exchange = str(self.exchange).lower().strip()

        self.created_at = ensure_utc(self.created_at) or utc_now()
        self.updated_at = ensure_utc(self.updated_at) or utc_now()
        self.confirmed_at = ensure_utc(self.confirmed_at)
        self.invalidated_at = ensure_utc(self.invalidated_at)
        self.expires_at = ensure_utc(self.expires_at)
        self.cooldown_until = ensure_utc(self.cooldown_until)
        self.setup_event_time = ensure_utc(self.setup_event_time)
        self.confirmation_event_time = ensure_utc(self.confirmation_event_time)
        self.last_analytics_update_time = ensure_utc(self.last_analytics_update_time)
        self.last_signal_time = ensure_utc(self.last_signal_time)
        self.last_emit_time = ensure_utc(self.last_emit_time)

        self.score = max(0.0, min(1.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))

    @property
    def key(self) -> str:
        return f"{self.symbol}:{self.exchange}"

    def is_active(self) -> bool:
        return self.status in {
            FundingSetupStatus.SETUP_DETECTED,
            FundingSetupStatus.CONFIRMED,
        }

    def to_dict(self) -> dict[str, Any]:
        return serialize_for_event(
            {
                "symbol": self.symbol,
                "exchange": self.exchange,
                "status": self.status,
                "direction": self.direction,
                "strategy_name": self.strategy_name,
                "setup_type": self.setup_type,
                "score": self.score,
                "confidence": self.confidence,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "confirmed_at": self.confirmed_at,
                "invalidated_at": self.invalidated_at,
                "expires_at": self.expires_at,
                "cooldown_until": self.cooldown_until,
                "reason": self.reason,
                "reasons": list(self.reasons),
                "tags": list(self.tags),
                "setup_event_time": self.setup_event_time,
                "confirmation_event_time": self.confirmation_event_time,
                "last_analytics_update_time": self.last_analytics_update_time,
                "last_signal_time": self.last_signal_time,
                "last_emit_time": self.last_emit_time,
                "emit_count": self.emit_count,
                "metadata": dict(self.metadata),
            }
        )


# =============================================================================
# Base Strategy
# =============================================================================


class BaseFundingStrategy(ABC):
    """
    Base class for strategy/funding/* modules.

    Core-aligned behavior:
    - typed EventBus / Event / Subscription / EventPriority
    - constructor dependency injection for EventBus, Scheduler, config
    - lifecycle start/stop/restart + backward-compatible register()
    - subscription tracking and graceful unsubscribe
    - optional Scheduler cleanup via add_interval_job()
    - centralized logging via core.logger.get_logger()
    - EventBus.emit with priority/source/correlation_id/headers
    - normalized funding analytics attachment helpers
    - optional handlers for analytics.funding.updated and analytics.funding.signal
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        config: BaseFundingStrategyConfig | None = None,
        scheduler: Scheduler | None = None,
        service_name: str | None = None,
    ) -> None:
        if event_bus is None:
            raise ValueError("event_bus is required")

        self.event_bus = event_bus
        self.config = config or BaseFundingStrategyConfig()
        self.config.validate()
        self.scheduler = scheduler
        self.service_name = service_name or self.config.service_name

        self.logger = get_logger(
            __name__,
            service_name=self.service_name,
            event_type="strategy",
            strategies=self.config.source_name,
        )

        self._states: dict[str, FundingStrategyState] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._subscriptions: list[Subscription] = []
        self._cleanup_job_id: str | None = None
        self._registered: bool = False
        self._running: bool = False
        self._stopping: bool = False

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            self.logger.warning("Funding strategy already running | strategy=%s", self.strategy_name)
            return

        self._running = True
        self._stopping = False
        self.register()
        self._register_scheduler_jobs()

        self.logger.info(
            "Funding strategy started | strategy=%s namespace=%s subscriptions=%s cleanup_job_id=%s",
            self.strategy_name,
            self.config.strategy_namespace,
            len(self._subscriptions),
            self._cleanup_job_id,
        )

    async def stop(self) -> None:
        if not self._running and not self._registered:
            return

        self._stopping = True
        self._running = False

        self.unregister()
        self._unregister_scheduler_jobs()

        self.logger.info(
            "Funding strategy stopped | strategy=%s states=%s",
            self.strategy_name,
            len(self._states),
        )
        self._stopping = False

    async def restart(self) -> None:
        await self.stop()
        await self.start()

    def register(self) -> None:
        """Backward-compatible sync registration API."""
        if self._registered:
            self.logger.warning("%s already registered", self.__class__.__name__)
            return

        before_ids = self._snapshot_event_bus_subscription_ids()
        self.register_base_subscriptions()
        self.register_subscriptions()
        self._capture_new_event_bus_subscriptions(before_ids)
        self._registered = True

        self.logger.info(
            "%s registered successfully | namespace=%s subscriptions=%s",
            self.__class__.__name__,
            self.config.strategy_namespace,
            len(self._subscriptions),
        )

    def unregister(self) -> None:
        if not self._registered and not self._subscriptions:
            return

        for subscription in list(self._subscriptions):
            try:
                self.event_bus.unsubscribe(subscription)
            except Exception:
                self.logger.exception(
                    "Failed to unsubscribe funding strategy handler | strategy=%s pattern=%s",
                    self.strategy_name,
                    getattr(subscription, "pattern", "unknown"),
                )

        self._subscriptions.clear()
        self._registered = False

    def register_base_subscriptions(self) -> None:
        if self.config.enable_funding_updated_subscription:
            self.subscribe(
                self.config.funding_updated_event_name,
                self.on_funding_updated,
                name=f"{self.strategy_name}.on_funding_updated",
            )

        if self.config.enable_funding_signal_subscription:
            self.subscribe(
                self.config.funding_signal_event_name,
                self.on_funding_signal,
                name=f"{self.strategy_name}.on_funding_signal",
            )

    def subscribe(
        self,
        pattern: str,
        handler: Any,
        *,
        name: str | None = None,
    ) -> Subscription:
        subscription = self.event_bus.subscribe(
            pattern,
            handler,
            name=name or f"{self.strategy_name}.{getattr(handler, '__name__', 'handler')}",
        )
        if subscription not in self._subscriptions:
            self._subscriptions.append(subscription)
        return subscription

    @abstractmethod
    def register_subscriptions(self) -> None:
        """Child strategies must subscribe through self.subscribe(...)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Base analytics handlers
    # -------------------------------------------------------------------------

    async def on_funding_updated(self, event: Event) -> None:
        payload = self.extract_payload(event)
        normalized = self._normalize_updated_payload(payload)
        symbol, exchange = self.extract_symbol_exchange(normalized)
        if not symbol:
            return

        lock = await self.acquire_symbol_lock(symbol, exchange)
        if lock is None:
            return

        try:
            state = self.get_state(symbol, exchange)
            self.attach_updated_context(state, normalized)
            self._expire_state_if_needed(state)
            await self.on_after_funding_updated(state, normalized, event)
        except Exception:
            self.logger.exception(
                "Failed to process funding updated event | strategy=%s symbol=%s exchange=%s",
                self.strategy_name,
                symbol,
                exchange,
            )
        finally:
            self.release_symbol_lock(lock)

    async def on_funding_signal(self, event: Event) -> None:
        payload = self.extract_payload(event)
        normalized = self._normalize_signal_payload(payload)
        symbol, exchange = self.extract_symbol_exchange(normalized)
        if not symbol:
            return

        lock = await self.acquire_symbol_lock(symbol, exchange)
        if lock is None:
            return

        try:
            state = self.get_state(symbol, exchange)
            signal = self._build_signal(normalized)
            self.attach_signal(state, signal)
            self._expire_state_if_needed(state)
            await self.on_after_funding_signal(state, signal, event)
        except Exception:
            self.logger.exception(
                "Failed to process funding signal event | strategy=%s symbol=%s exchange=%s",
                self.strategy_name,
                symbol,
                exchange,
            )
        finally:
            self.release_symbol_lock(lock)

    async def on_after_funding_updated(
        self,
        state: FundingStrategyState,
        payload: dict[str, Any],
        event: Event,
    ) -> None:
        """Optional hook for descendants."""
        return None

    async def on_after_funding_signal(
        self,
        state: FundingStrategyState,
        signal: FundingSignal | dict[str, Any],
        event: Event,
    ) -> None:
        """Optional hook for descendants."""
        return None

    # -------------------------------------------------------------------------
    # State access
    # -------------------------------------------------------------------------

    def get_state(self, symbol: str, exchange: str = "unknown") -> FundingStrategyState:
        key = self._make_key(symbol, exchange)
        state = self._states.get(key)

        if state is None:
            state = FundingStrategyState(
                symbol=symbol,
                exchange=exchange,
                strategy_name=self.strategy_name,
            )
            self._states[key] = state

        if self.config.cleanup_expired_on_access:
            self._expire_state_if_needed(state)

        return state

    def get_all_states(self) -> dict[str, FundingStrategyState]:
        return dict(self._states)

    def get_active_states(self) -> dict[str, FundingStrategyState]:
        return {key: state for key, state in self._states.items() if self.is_state_active(state)}

    def reset_state(
        self,
        symbol: str,
        exchange: str = "unknown",
        preserve_cooldown: bool = True,
        preserve_context: bool = True,
    ) -> FundingStrategyState:
        previous = self.get_state(symbol, exchange)
        cooldown_until = previous.cooldown_until if preserve_cooldown else None

        new_state = FundingStrategyState(
            symbol=symbol,
            exchange=exchange,
            strategy_name=self.strategy_name,
            cooldown_until=cooldown_until,
            status=(
                FundingSetupStatus.COOLDOWN
                if cooldown_until and cooldown_until > utc_now()
                else FundingSetupStatus.IDLE
            ),
        )

        if preserve_context:
            new_state.last_snapshot = previous.last_snapshot
            new_state.last_statistics = previous.last_statistics
            new_state.last_regime = previous.last_regime
            new_state.last_pressure = previous.last_pressure
            new_state.last_extreme = previous.last_extreme
            new_state.last_divergence = previous.last_divergence
            new_state.last_flip = previous.last_flip
            new_state.last_signal = previous.last_signal
            new_state.last_funding_updated_payload = previous.last_funding_updated_payload
            new_state.last_analytics_update_time = previous.last_analytics_update_time
            new_state.last_signal_time = previous.last_signal_time

        self._states[new_state.key] = new_state
        return new_state

    def stats(self) -> dict[str, Any]:
        active = sum(1 for state in self._states.values() if state.is_active())
        return {
            "strategy": self.strategy_name,
            "namespace": self.config.strategy_namespace,
            "registered": self._registered,
            "running": self._running,
            "stopping": self._stopping,
            "subscriptions": len(self._subscriptions),
            "states_total": len(self._states),
            "states_active": active,
            "locks": len(self._locks),
            "cleanup_job_id": self._cleanup_job_id,
        }

    # -------------------------------------------------------------------------
    # Locking
    # -------------------------------------------------------------------------

    async def acquire_symbol_lock(self, symbol: str, exchange: str = "unknown") -> asyncio.Lock | None:
        key = self._make_key(symbol, exchange)
        lock = self._locks.setdefault(key, asyncio.Lock())

        try:
            await asyncio.wait_for(lock.acquire(), timeout=self.config.state_lock_timeout_sec)
            return lock
        except asyncio.TimeoutError:
            get_logger(
                __name__,
                event_type="strategy",
                strategies=self.strategy_name,
                symbol=str(symbol).upper().strip(),
                exchange=str(exchange).lower().strip(),
            ).warning("Funding strategy lock timeout | strategy=%s", self.strategy_name)
            return None

    @staticmethod
    def release_symbol_lock(lock: asyncio.Lock | None) -> None:
        if lock is not None and lock.locked():
            lock.release()

    # -------------------------------------------------------------------------
    # State lifecycle helpers
    # -------------------------------------------------------------------------

    def set_setup_detected(
        self,
        state: FundingStrategyState,
        *,
        direction: FundingStrategyDirection,
        setup_type: str,
        score: float,
        confidence: float,
        reason: str | None = None,
        reasons: list[str] | None = None,
        tags: list[str] | None = None,
        event_time: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_sec: float | None = None,
    ) -> FundingStrategyState:
        now = utc_now()
        ttl = ttl_sec if ttl_sec is not None else self.config.setup_ttl_sec

        state.status = FundingSetupStatus.SETUP_DETECTED
        state.direction = direction
        state.setup_type = setup_type
        state.score = self._clip_score(score)
        state.confidence = self._clip_score(confidence)
        state.reason = reason
        state.reasons = list(reasons or ([] if reason is None else [reason]))
        state.tags = list(tags or [])
        state.updated_at = now
        state.invalidated_at = None
        state.confirmed_at = None
        state.setup_event_time = ensure_utc(event_time) or now
        state.expires_at = now + timedelta(seconds=ttl)
        state.cooldown_until = None
        state.metadata = serialize_for_event(dict(metadata or {}))

        get_logger(
            __name__,
            event_type="strategy",
            strategies=self.strategy_name,
            symbol=state.symbol,
            exchange=state.exchange,
        ).debug(
            "%s setup detected | type=%s direction=%s score=%.4f confidence=%.4f",
            self.strategy_name,
            setup_type,
            direction.value,
            state.score,
            state.confidence,
        )
        return state

    def set_confirmed(
        self,
        state: FundingStrategyState,
        *,
        score: float | None = None,
        confidence: float | None = None,
        reason: str | None = None,
        append_reason: bool = True,
        tags: list[str] | None = None,
        event_time: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FundingStrategyState:
        now = utc_now()

        if state.status == FundingSetupStatus.CONFIRMED and not self.config.allow_reconfirm:
            return state

        state.status = FundingSetupStatus.CONFIRMED
        state.updated_at = now
        state.confirmed_at = now
        state.confirmation_event_time = ensure_utc(event_time) or now

        if score is not None:
            state.score = self._clip_score(score)
        if confidence is not None:
            state.confidence = self._clip_score(confidence)

        if reason:
            state.reason = reason
            if append_reason and reason not in state.reasons:
                state.reasons.append(reason)

        if tags:
            for tag in tags:
                if tag not in state.tags:
                    state.tags.append(tag)

        if metadata:
            state.metadata.update(serialize_for_event(metadata))

        get_logger(
            __name__,
            event_type="strategy",
            strategies=self.strategy_name,
            symbol=state.symbol,
            exchange=state.exchange,
        ).debug(
            "%s setup confirmed | type=%s direction=%s score=%.4f confidence=%.4f",
            self.strategy_name,
            state.setup_type,
            state.direction.value,
            state.score,
            state.confidence,
        )
        return state

    def set_invalidated(
        self,
        state: FundingStrategyState,
        *,
        reason: str,
        cooldown: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> FundingStrategyState:
        now = utc_now()

        state.status = FundingSetupStatus.INVALIDATED
        state.updated_at = now
        state.invalidated_at = now
        state.reason = reason

        if reason and reason not in state.reasons:
            state.reasons.append(reason)

        if metadata:
            state.metadata.update(serialize_for_event(metadata))

        if cooldown:
            state.cooldown_until = now + timedelta(seconds=self.config.cooldown_sec)
            state.status = FundingSetupStatus.COOLDOWN

        get_logger(
            __name__,
            event_type="strategy",
            strategies=self.strategy_name,
            symbol=state.symbol,
            exchange=state.exchange,
        ).debug(
            "%s setup invalidated | type=%s reason=%s cooldown=%s",
            self.strategy_name,
            state.setup_type,
            reason,
            cooldown,
        )
        return state

    def set_expired(
        self,
        state: FundingStrategyState,
        *,
        reason: str = "setup_expired",
        cooldown: bool = True,
    ) -> FundingStrategyState:
        now = utc_now()

        state.status = FundingSetupStatus.EXPIRED
        state.updated_at = now
        state.invalidated_at = now
        state.reason = reason

        if reason not in state.reasons:
            state.reasons.append(reason)

        if cooldown:
            state.cooldown_until = now + timedelta(seconds=self.config.cooldown_sec)
            state.status = FundingSetupStatus.COOLDOWN

        get_logger(
            __name__,
            event_type="strategy",
            strategies=self.strategy_name,
            symbol=state.symbol,
            exchange=state.exchange,
        ).debug(
            "%s setup expired | type=%s cooldown=%s",
            self.strategy_name,
            state.setup_type,
            cooldown,
        )
        return state

    def set_cooldown(
        self,
        state: FundingStrategyState,
        *,
        cooldown_sec: float | None = None,
        reason: str | None = None,
    ) -> FundingStrategyState:
        now = utc_now()
        duration = cooldown_sec if cooldown_sec is not None else self.config.cooldown_sec

        state.status = FundingSetupStatus.COOLDOWN
        state.updated_at = now
        state.cooldown_until = now + timedelta(seconds=duration)

        if reason:
            state.reason = reason
            if reason not in state.reasons:
                state.reasons.append(reason)

        return state

    def set_idle(self, state: FundingStrategyState) -> FundingStrategyState:
        return self.reset_state(state.symbol, state.exchange, preserve_cooldown=False, preserve_context=True)

    # -------------------------------------------------------------------------
    # State freshness / cleanup
    # -------------------------------------------------------------------------

    def is_in_cooldown(self, state: FundingStrategyState) -> bool:
        if state.cooldown_until is None:
            return False
        if state.cooldown_until <= utc_now():
            if state.status == FundingSetupStatus.COOLDOWN:
                state.status = FundingSetupStatus.IDLE
            state.cooldown_until = None
            return False
        return True

    def is_expired(self, state: FundingStrategyState) -> bool:
        if state.expires_at is None:
            return False
        return state.expires_at <= utc_now()

    def is_state_active(self, state: FundingStrategyState) -> bool:
        if self.is_in_cooldown(state):
            return False
        if self.is_expired(state):
            return False
        return state.status in {
            FundingSetupStatus.SETUP_DETECTED,
            FundingSetupStatus.CONFIRMED,
        }

    def _expire_state_if_needed(self, state: FundingStrategyState) -> None:
        if self.is_expired(state) and state.status in {
            FundingSetupStatus.SETUP_DETECTED,
            FundingSetupStatus.CONFIRMED,
        }:
            self.set_expired(state)

    async def cleanup_expired_states(self, *, emit_events: bool = True) -> int:
        expired_count = 0
        for state in list(self._states.values()):
            previous_status = state.status
            self._expire_state_if_needed(state)
            if previous_status != state.status and state.status == FundingSetupStatus.COOLDOWN:
                expired_count += 1
                if emit_events and self.config.emit_expiration_events:
                    await self.emit_expired(state, extra_payload={"trigger": "scheduler_cleanup"})
        return expired_count

    # -------------------------------------------------------------------------
    # Analytics attachment helpers
    # -------------------------------------------------------------------------

    def attach_updated_context(self, state: FundingStrategyState, payload: Mapping[str, Any]) -> None:
        state.last_funding_updated_payload = serialize_for_event(dict(payload))
        state.last_analytics_update_time = self.extract_event_time(dict(payload)) or utc_now()
        state.updated_at = utc_now()

        snapshot_payload = self._nested_payload(payload, "snapshot")
        statistics_payload = self._nested_payload(payload, "statistics")
        regime_payload = self._nested_payload(payload, "regime_state")
        pressure_payload = self._nested_payload(payload, "pressure_state")
        extreme_payload = self._nested_payload(payload, "extreme_event")
        divergence_payload = self._nested_payload(payload, "divergence_event")
        flip_payload = self._nested_payload(payload, "flip_event")

        if snapshot_payload:
            state.last_snapshot = self._build_snapshot(snapshot_payload)
        if statistics_payload:
            state.last_statistics = self._build_statistics(statistics_payload)
        if regime_payload:
            state.last_regime = self._build_regime_state(regime_payload)
        if pressure_payload:
            state.last_pressure = self._build_pressure_state(pressure_payload)
        if extreme_payload:
            state.last_extreme = self._build_extreme_event(extreme_payload)
        if divergence_payload:
            state.last_divergence = self._build_divergence_event(divergence_payload)
        if flip_payload:
            state.last_flip = self._build_flip_event(flip_payload)

    def attach_regime(self, state: FundingStrategyState, regime_state: Any) -> None:
        state.last_regime = regime_state
        state.updated_at = utc_now()

    def attach_pressure(self, state: FundingStrategyState, pressure_state: Any) -> None:
        state.last_pressure = pressure_state
        state.updated_at = utc_now()

    def attach_extreme(self, state: FundingStrategyState, extreme_event: Any) -> None:
        state.last_extreme = extreme_event
        state.updated_at = utc_now()

    def attach_divergence(self, state: FundingStrategyState, divergence_event: Any) -> None:
        state.last_divergence = divergence_event
        state.updated_at = utc_now()

    def attach_flip(self, state: FundingStrategyState, flip_event: Any) -> None:
        state.last_flip = flip_event
        state.updated_at = utc_now()

    def attach_signal(self, state: FundingStrategyState, signal_event: Any) -> None:
        state.last_signal = signal_event
        state.last_signal_time = self._extract_event_time_from_normalized(signal_event) or utc_now()
        state.updated_at = utc_now()

    # -------------------------------------------------------------------------
    # Event helpers
    # -------------------------------------------------------------------------

    def extract_payload(self, event: Event | Mapping[str, Any] | Any) -> dict[str, Any]:
        if event is None:
            return {}

        if isinstance(event, Event):
            return event.payload if isinstance(event.payload, dict) else {}

        if isinstance(event, Mapping):
            if isinstance(event.get("payload"), Mapping):
                return dict(event["payload"])
            return dict(event)

        payload = getattr(event, "payload", None)
        if isinstance(payload, Mapping):
            return dict(payload)

        if hasattr(event, "__dict__"):
            raw = vars(event)
            if isinstance(raw.get("payload"), Mapping):
                return dict(raw["payload"])
            return dict(raw)

        return {}

    def extract_symbol_exchange(self, payload: Mapping[str, Any]) -> tuple[str, str]:
        data = unwrap_analytics_payload(payload)
        symbol = str(data.get("symbol") or self._nested_get(payload, "symbol") or "").upper().strip()
        exchange_value = data.get("exchange") or self._nested_get(payload, "exchange") or "unknown"
        exchange = self._enum_str(exchange_value) or "unknown"
        return symbol, str(exchange).lower().strip()

    def extract_event_time(self, payload: Mapping[str, Any]) -> datetime | None:
        data = unwrap_analytics_payload(payload)
        raw = data.get("event_time") or data.get("timestamp") or data.get("ts")
        if raw is None:
            envelope = data.get("_envelope")
            if isinstance(envelope, Mapping):
                raw = envelope.get("event_time") or envelope.get("timestamp") or envelope.get("ts")
        return parse_datetime(raw)

    def event_age_seconds(self, event_time: datetime | None) -> float | None:
        event_time = ensure_utc(event_time)
        if event_time is None:
            return None
        return max(0.0, (utc_now() - event_time).total_seconds())

    def is_stale_event(self, event_time: datetime | None) -> bool:
        age = self.event_age_seconds(event_time)
        if age is None:
            return False
        return age > self.config.event_stale_after_sec

    # -------------------------------------------------------------------------
    # Emit helpers
    # -------------------------------------------------------------------------

    async def emit_setup_event(
        self,
        state: FundingStrategyState,
        *,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        await self.emit_setup(state, extra_payload=extra_payload)

    async def emit_confirmation_event(
        self,
        state: FundingStrategyState,
        *,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        await self.emit_confirmed(state, extra_payload=extra_payload)

    async def emit_invalidation_event(
        self,
        state: FundingStrategyState,
        *,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        await self.emit_invalidated(state, extra_payload=extra_payload)

    async def emit_expiration_event(
        self,
        state: FundingStrategyState,
        *,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        await self.emit_expired(state, extra_payload=extra_payload)

    def build_base_signal_payload(
        self,
        state: FundingStrategyState,
        *,
        event_kind: str,
        extra_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbol": state.symbol,
            "exchange": state.exchange,
            "strategy": self.strategy_name,
            "strategy_name": self.strategy_name,
            "strategy_namespace": self.config.strategy_namespace,
            "event_kind": event_kind,
            "status": state.status.value,
            "direction": state.direction.value,
            "setup_type": state.setup_type,
            "score": state.score,
            "confidence": state.confidence,
            "reason": state.reason,
            "reasons": list(state.reasons),
            "tags": list(state.tags),
            "event_time": utc_now().isoformat(),
            "metadata": serialize_for_event(dict(state.metadata)),
        }

        if self.config.attach_full_state_on_emit:
            payload["state"] = state.to_dict()

        funding_context = self._build_funding_context(state)
        if funding_context:
            payload["funding_context"] = funding_context

        if self.config.attach_full_analytics_context_on_emit:
            analytics_context = self._build_full_analytics_context(state)
            if analytics_context:
                payload["analytics_context"] = analytics_context

        if extra_payload:
            payload.update(serialize_for_event(extra_payload))

        return serialize_for_event(payload)

    async def _emit(
        self,
        *,
        event_name: str,
        payload: dict[str, Any],
        priority: EventPriority | None = None,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        event_kind = str(payload.get("event_kind", "unknown"))
        resolved_priority = priority or self._priority_for_event_kind(event_kind)
        symbol = str(payload.get("symbol", "")).upper().strip()
        exchange = str(payload.get("exchange", "unknown")).lower().strip()
        resolved_correlation_id = correlation_id or str(
            payload.get("correlation_id")
            or payload.get("source_event_id")
            or f"{self.strategy_name}:{exchange}:{symbol}:{event_kind}:{payload.get('event_time', '')}"
        )
        resolved_headers = {
            "strategy": self.strategy_name,
            "strategy_namespace": self.config.strategy_namespace,
            "event_kind": event_kind,
            "symbol": symbol,
            "exchange": exchange,
        }
        if headers:
            resolved_headers.update(serialize_for_event(headers))

        try:
            accepted = await self.event_bus.emit(
                event_name,
                serialize_for_event(payload),
                priority=resolved_priority,
                source=self.config.source_name,
                correlation_id=resolved_correlation_id,
                headers=resolved_headers,
            )

            state = self._states.get(self._make_key(symbol, exchange))
            if state is not None:
                state.last_emit_time = utc_now()
                state.emit_count += 1

            return accepted

        except RuntimeError:
            self.logger.exception(
                "EventBus rejected funding strategy event | strategy=%s event_name=%s",
                self.strategy_name,
                event_name,
            )
            return False
        except Exception:
            self.logger.exception(
                "Failed to emit funding strategy event | strategy=%s event_name=%s",
                self.strategy_name,
                event_name,
            )
            return False

    async def emit_setup(self, state: FundingStrategyState, *, extra_payload: dict[str, Any] | None = None) -> None:
        if not self.config.emit_setup_events:
            return
        payload = self.build_base_signal_payload(state=state, event_kind="setup", extra_payload=extra_payload)
        payload = self.on_before_setup_emit(state, payload)
        await self._emit(event_name=f"{self.config.strategy_namespace}.setup", payload=payload, priority=self.config.setup_priority)

    async def emit_confirmed(self, state: FundingStrategyState, *, extra_payload: dict[str, Any] | None = None) -> None:
        if not self.config.emit_confirmation_events:
            return
        payload = self.build_base_signal_payload(state=state, event_kind="confirmed", extra_payload=extra_payload)
        payload = self.on_before_confirmation_emit(state, payload)
        await self._emit(event_name=f"{self.config.strategy_namespace}.confirmed", payload=payload, priority=self.config.confirmation_priority)

    async def emit_invalidated(self, state: FundingStrategyState, *, extra_payload: dict[str, Any] | None = None) -> None:
        if not self.config.emit_invalidation_events:
            return
        payload = self.build_base_signal_payload(state=state, event_kind="invalidated", extra_payload=extra_payload)
        payload = self.on_before_invalidation_emit(state, payload)
        await self._emit(event_name=f"{self.config.strategy_namespace}.invalidated", payload=payload, priority=self.config.invalidation_priority)

    async def emit_expired(self, state: FundingStrategyState, *, extra_payload: dict[str, Any] | None = None) -> None:
        if not self.config.emit_expiration_events:
            return
        payload = self.build_base_signal_payload(state=state, event_kind="expired", extra_payload=extra_payload)
        payload = self.on_before_expiration_emit(state, payload)
        await self._emit(event_name=f"{self.config.strategy_namespace}.expired", payload=payload, priority=self.config.expiration_priority)

    # -------------------------------------------------------------------------
    # Normalizers used by child strategies
    # -------------------------------------------------------------------------

    def _normalize_updated_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        raw = dict(payload)
        inner = raw.get("payload") if isinstance(raw.get("payload"), Mapping) else raw
        normalized = dict(inner)
        normalized.setdefault("symbol", raw.get("symbol") or self._nested_get(inner, "symbol"))
        normalized.setdefault("exchange", raw.get("exchange") or self._nested_get(inner, "exchange") or "unknown")
        normalized.setdefault("event_time", raw.get("event_time") or raw.get("timestamp") or raw.get("ts"))
        normalized["_envelope"] = raw
        return normalized

    def _normalize_signal_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return unwrap_analytics_payload(payload)

    def _normalize_regime_payload(self, payload: Mapping[str, Any]) -> FundingRegimeState | dict[str, Any]:
        data = unwrap_analytics_payload(payload)
        if "regime_state" in data and isinstance(data["regime_state"], Mapping):
            data = dict(data["regime_state"])
        try:
            return self._build_regime_state(data)
        except Exception:
            self.logger.exception("Failed to normalize funding regime payload")
            return dict(data)

    def _normalize_pressure_payload(self, payload: Mapping[str, Any]) -> FundingPressureState | dict[str, Any]:
        data = unwrap_analytics_payload(payload)
        if "pressure_state" in data and isinstance(data["pressure_state"], Mapping):
            data = dict(data["pressure_state"])
        try:
            return self._build_pressure_state(data)
        except Exception:
            self.logger.exception("Failed to normalize funding pressure payload")
            return dict(data)

    def _normalize_extreme_payload(self, payload: Mapping[str, Any]) -> FundingExtremeEvent | dict[str, Any]:
        data = unwrap_analytics_payload(payload)
        if "extreme_event" in data and isinstance(data["extreme_event"], Mapping):
            data = dict(data["extreme_event"])
        try:
            return self._build_extreme_event(data)
        except Exception:
            self.logger.exception("Failed to normalize funding extreme payload")
            return dict(data)

    def _normalize_divergence_payload(self, payload: Mapping[str, Any]) -> FundingDivergenceEvent | dict[str, Any]:
        data = unwrap_analytics_payload(payload)
        if "divergence_event" in data and isinstance(data["divergence_event"], Mapping):
            data = dict(data["divergence_event"])
        try:
            return self._build_divergence_event(data)
        except Exception:
            self.logger.exception("Failed to normalize funding divergence payload")
            return dict(data)

    def _normalize_flip_payload(self, payload: Mapping[str, Any]) -> FundingFlipEvent | dict[str, Any]:
        data = unwrap_analytics_payload(payload)
        if "flip_event" in data and isinstance(data["flip_event"], Mapping):
            data = dict(data["flip_event"])
        try:
            return self._build_flip_event(data)
        except Exception:
            self.logger.exception("Failed to normalize funding flip payload")
            return dict(data)

    # -------------------------------------------------------------------------
    # Model builders
    # -------------------------------------------------------------------------

    def _build_snapshot(self, data: Mapping[str, Any]) -> FundingSnapshot:
        return FundingSnapshot(
            symbol=str(data.get("symbol", "")).upper().strip(),
            exchange=self._parse_enum(FundingDataSource, data.get("exchange"), FundingDataSource.UNKNOWN),
            funding_rate=self._to_float(data.get("funding_rate"), 0.0),
            predicted_funding_rate=self._to_optional_float(data.get("predicted_funding_rate")),
            mark_price=self._to_optional_float(data.get("mark_price")),
            index_price=self._to_optional_float(data.get("index_price")),
            open_interest=self._to_optional_float(data.get("open_interest")),
            volume_24h=self._to_optional_float(data.get("volume_24h")),
            next_funding_time=parse_datetime(data.get("next_funding_time")),
            event_time=parse_datetime(data.get("event_time")) or utc_now(),
            received_at=parse_datetime(data.get("received_at")) or utc_now(),
            metadata=self._safe_metadata(data.get("metadata")),
        )

    def _build_statistics(self, data: Mapping[str, Any]) -> FundingStatistics:
        return FundingStatistics(
            symbol=str(data.get("symbol", "")).upper().strip(),
            exchange=self._parse_enum(FundingDataSource, data.get("exchange"), FundingDataSource.UNKNOWN),
            timeframe=self._parse_enum(FundingTimeframe, data.get("timeframe"), FundingTimeframe.H1),
            current_rate=self._to_float(data.get("current_rate"), 0.0),
            mean_rate=self._to_float(data.get("mean_rate"), 0.0),
            median_rate=self._to_float(data.get("median_rate"), 0.0),
            std_rate=self._to_float(data.get("std_rate"), 0.0),
            min_rate=self._to_float(data.get("min_rate"), 0.0),
            max_rate=self._to_float(data.get("max_rate"), 0.0),
            zscore=self._to_optional_float(data.get("zscore")),
            percentile=self._to_optional_float(data.get("percentile")),
            sample_size=int(self._to_float(data.get("sample_size"), 0.0)),
            window_start=parse_datetime(data.get("window_start")),
            window_end=parse_datetime(data.get("window_end")),
            updated_at=parse_datetime(data.get("updated_at")) or utc_now(),
        )

    def _build_regime_state(self, data: Mapping[str, Any]) -> FundingRegimeState:
        return FundingRegimeState(
            symbol=str(data.get("symbol", "")).upper().strip(),
            exchange=self._parse_enum(FundingDataSource, data.get("exchange"), FundingDataSource.UNKNOWN),
            timeframe=self._parse_enum(FundingTimeframe, data.get("timeframe"), FundingTimeframe.H1),
            regime=self._parse_enum(FundingRegime, data.get("regime"), FundingRegime.UNKNOWN),
            bias=self._parse_enum(FundingBias, data.get("bias"), FundingBias.NEUTRAL),
            current_rate=self._to_float(data.get("current_rate"), 0.0),
            mean_rate=self._to_optional_float(data.get("mean_rate")),
            zscore=self._to_optional_float(data.get("zscore")),
            percentile=self._to_optional_float(data.get("percentile")),
            confidence=self._clip_score(data.get("confidence")),
            changed=bool(data.get("changed", False)),
            previous_regime=self._parse_optional_enum(FundingRegime, data.get("previous_regime")),
            event_time=parse_datetime(data.get("event_time")) or utc_now(),
            metadata=self._safe_metadata(data.get("metadata")),
        )

    def _build_pressure_state(self, data: Mapping[str, Any]) -> FundingPressureState:
        return FundingPressureState(
            symbol=str(data.get("symbol", "")).upper().strip(),
            exchange=self._parse_enum(FundingDataSource, data.get("exchange"), FundingDataSource.UNKNOWN),
            timeframe=self._parse_enum(FundingTimeframe, data.get("timeframe"), FundingTimeframe.H1),
            direction=self._parse_enum(FundingPressureDirection, data.get("direction"), FundingPressureDirection.NEUTRAL),
            level=self._parse_enum(FundingPressureLevel, data.get("level"), FundingPressureLevel.UNKNOWN),
            bias=self._parse_enum(FundingBias, data.get("bias"), FundingBias.NEUTRAL),
            funding_rate=self._to_float(data.get("funding_rate"), 0.0),
            pressure_score=self._clip_score(data.get("pressure_score")),
            oi_confirmation=bool(data.get("oi_confirmation", False)),
            price_stall_confirmation=bool(data.get("price_stall_confirmation", False)),
            squeeze_probability=self._to_optional_float(data.get("squeeze_probability")),
            mean_reversion_probability=self._to_optional_float(data.get("mean_reversion_probability")),
            event_time=parse_datetime(data.get("event_time")) or utc_now(),
            metadata=self._safe_metadata(data.get("metadata")),
        )

    def _build_extreme_event(self, data: Mapping[str, Any]) -> FundingExtremeEvent:
        return FundingExtremeEvent(
            symbol=str(data.get("symbol", "")).upper().strip(),
            exchange=self._parse_enum(FundingDataSource, data.get("exchange"), FundingDataSource.UNKNOWN),
            timeframe=self._parse_enum(FundingTimeframe, data.get("timeframe"), FundingTimeframe.H1),
            extreme_type=self._parse_enum(FundingExtremeType, data.get("extreme_type"), FundingExtremeType.NONE),
            regime=self._parse_enum(FundingRegime, data.get("regime"), FundingRegime.UNKNOWN),
            funding_rate=self._to_float(data.get("funding_rate"), 0.0),
            zscore=self._to_optional_float(data.get("zscore")),
            percentile=self._to_optional_float(data.get("percentile")),
            severity=self._clip_score(data.get("severity")),
            is_reversal_risk=bool(data.get("is_reversal_risk", False)),
            is_squeeze_risk=bool(data.get("is_squeeze_risk", False)),
            event_time=parse_datetime(data.get("event_time")) or utc_now(),
            metadata=self._safe_metadata(data.get("metadata")),
        )

    def _build_divergence_event(self, data: Mapping[str, Any]) -> FundingDivergenceEvent:
        return FundingDivergenceEvent(
            symbol=str(data.get("symbol", "")).upper().strip(),
            exchange=self._parse_enum(FundingDataSource, data.get("exchange"), FundingDataSource.UNKNOWN),
            timeframe=self._parse_enum(FundingTimeframe, data.get("timeframe"), FundingTimeframe.H1),
            divergence_type=self._parse_enum(FundingDivergenceType, data.get("divergence_type"), FundingDivergenceType.NONE),
            funding_rate=self._to_float(data.get("funding_rate"), 0.0),
            price_change_pct=self._to_optional_float(data.get("price_change_pct")),
            oi_change_pct=self._to_optional_float(data.get("oi_change_pct")),
            cvd_change=self._to_optional_float(data.get("cvd_change")),
            long_liquidations=self._to_optional_float(data.get("long_liquidations")),
            short_liquidations=self._to_optional_float(data.get("short_liquidations")),
            confidence=self._clip_score(data.get("confidence")),
            event_time=parse_datetime(data.get("event_time")) or utc_now(),
            metadata=self._safe_metadata(data.get("metadata")),
        )

    def _build_flip_event(self, data: Mapping[str, Any]) -> FundingFlipEvent:
        return FundingFlipEvent(
            symbol=str(data.get("symbol", "")).upper().strip(),
            exchange=self._parse_enum(FundingDataSource, data.get("exchange"), FundingDataSource.UNKNOWN),
            timeframe=self._parse_enum(FundingTimeframe, data.get("timeframe"), FundingTimeframe.H1),
            flip_type=self._parse_enum(FundingFlipType, data.get("flip_type"), FundingFlipType.NONE),
            previous_rate=self._to_float(data.get("previous_rate"), 0.0),
            current_rate=self._to_float(data.get("current_rate"), 0.0),
            flip_magnitude=self._to_float(data.get("flip_magnitude"), 0.0),
            confidence=self._clip_score(data.get("confidence")),
            event_time=parse_datetime(data.get("event_time")) or utc_now(),
            metadata=self._safe_metadata(data.get("metadata")),
        )

    def _build_signal(self, data: Mapping[str, Any]) -> FundingSignal | dict[str, Any]:
        try:
            return FundingSignal(
                symbol=str(data.get("symbol", "")).upper().strip(),
                exchange=self._parse_enum(FundingDataSource, data.get("exchange"), FundingDataSource.UNKNOWN),
                timeframe=self._parse_enum(FundingTimeframe, data.get("timeframe"), FundingTimeframe.H1),
                signal_type=self._parse_enum(FundingSignalType, data.get("signal_type"), FundingSignalType.REVERSION_SETUP),
                bias=self._parse_enum(FundingBias, data.get("bias"), FundingBias.NEUTRAL),
                regime=self._parse_enum(FundingRegime, data.get("regime"), FundingRegime.UNKNOWN),
                score=self._to_float(data.get("score"), 0.0),
                confidence=self._clip_score(data.get("confidence")),
                description=str(data.get("description", "")),
                supporting_factors=list(data.get("supporting_factors") or []),
                tags=list(data.get("tags") or []),
                event_time=parse_datetime(data.get("event_time")) or utc_now(),
                metadata=self._safe_metadata(data.get("metadata")),
            )
        except Exception:
            self.logger.exception("Failed to build FundingSignal, keeping raw dict")
            return dict(data)

    # -------------------------------------------------------------------------
    # Score / direction / key helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _clip_score(value: float | int | str | None) -> float:
        if value is None:
            return 0.0
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, numeric))

    @classmethod
    def _average_scores(cls, *values: float | None) -> float:
        valid = [cls._clip_score(v) for v in values if v is not None]
        if not valid:
            return 0.0
        return sum(valid) / len(valid)

    @classmethod
    def _weighted_average(cls, weighted_values: Iterable[tuple[float | None, float]]) -> float:
        total_weight = 0.0
        total = 0.0
        for value, weight in weighted_values:
            if value is None or weight <= 0:
                continue
            total += cls._clip_score(value) * float(weight)
            total_weight += float(weight)
        if total_weight <= 0:
            return 0.0
        return cls._clip_score(total / total_weight)

    @staticmethod
    def _make_key(symbol: str, exchange: str = "unknown") -> str:
        return f"{str(symbol).upper().strip()}:{str(exchange).lower().strip()}"

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_metadata(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _enum_str(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, Enum):
            return str(value.value)
        return str(value)

    @staticmethod
    def _get_value(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, Mapping):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @classmethod
    def _parse_enum(cls, enum_cls: type[Enum], value: Any, default: Any) -> Any:
        if isinstance(value, enum_cls):
            return value
        if value is None:
            return default
        raw = cls._enum_str(value)
        try:
            return enum_cls(raw)
        except ValueError:
            return default

    @classmethod
    def _parse_optional_enum(cls, enum_cls: type[Enum], value: Any) -> Any | None:
        if value is None:
            return None
        if isinstance(value, enum_cls):
            return value
        raw = cls._enum_str(value)
        try:
            return enum_cls(raw)
        except ValueError:
            return None

    @staticmethod
    def _nested_payload(payload: Mapping[str, Any], key: str) -> dict[str, Any] | None:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return dict(value)
        inner = payload.get("payload")
        if isinstance(inner, Mapping) and isinstance(inner.get(key), Mapping):
            return dict(inner[key])
        return None

    @staticmethod
    def _nested_get(payload: Mapping[str, Any], key: str) -> Any:
        if key in payload:
            return payload[key]
        inner = payload.get("payload")
        if isinstance(inner, Mapping):
            if key in inner:
                return inner[key]
            for nested_key in ("snapshot", "statistics", "regime_state", "pressure_state", "extreme_event", "divergence_event", "flip_event", "signal"):
                nested = inner.get(nested_key)
                if isinstance(nested, Mapping) and key in nested:
                    return nested[key]
        return None

    @staticmethod
    def _pressure_level_rank(level: str | None) -> int:
        ranks = {
            FundingPressureLevel.UNKNOWN.value: 0,
            FundingPressureLevel.LOW.value: 1,
            FundingPressureLevel.MODERATE.value: 2,
            FundingPressureLevel.HIGH.value: 3,
            FundingPressureLevel.EXTREME.value: 4,
        }
        return ranks.get(str(level), 0)

    def _has_pressure_level_dropped_enough(self, previous_level: str | None, current_level: str | None) -> bool:
        previous_rank = self._pressure_level_rank(previous_level)
        current_rank = self._pressure_level_rank(current_level)
        return (previous_rank - current_rank) >= 1

    def _extract_event_time_from_normalized(self, obj: Any) -> datetime | None:
        return ensure_utc(self._get_value(obj, "event_time")) if isinstance(self._get_value(obj, "event_time"), datetime) else parse_datetime(self._get_value(obj, "event_time"))

    # -------------------------------------------------------------------------
    # Protected composition helpers
    # -------------------------------------------------------------------------

    def _build_funding_context(self, state: FundingStrategyState) -> dict[str, Any]:
        context: dict[str, Any] = {}

        regime = state.last_regime
        if regime is not None:
            context["regime"] = self._enum_str(self._get_value(regime, "regime"))
            context["bias"] = self._enum_str(self._get_value(regime, "bias"))
            context["regime_confidence"] = self._get_value(regime, "confidence")

        pressure = state.last_pressure
        if pressure is not None:
            context["pressure_direction"] = self._enum_str(self._get_value(pressure, "direction"))
            context["pressure_level"] = self._enum_str(self._get_value(pressure, "level"))
            context["pressure_score"] = self._get_value(pressure, "pressure_score")
            context["squeeze_probability"] = self._get_value(pressure, "squeeze_probability")
            context["mean_reversion_probability"] = self._get_value(pressure, "mean_reversion_probability")

        extreme = state.last_extreme
        if extreme is not None:
            context["extreme_type"] = self._enum_str(self._get_value(extreme, "extreme_type"))
            context["extreme_severity"] = self._get_value(extreme, "severity")
            context["is_reversal_risk"] = self._get_value(extreme, "is_reversal_risk")
            context["is_squeeze_risk"] = self._get_value(extreme, "is_squeeze_risk")

        divergence = state.last_divergence
        if divergence is not None:
            context["divergence_type"] = self._enum_str(self._get_value(divergence, "divergence_type"))
            context["divergence_confidence"] = self._get_value(divergence, "confidence")

        flip = state.last_flip
        if flip is not None:
            context["flip_type"] = self._enum_str(self._get_value(flip, "flip_type"))
            context["flip_confidence"] = self._get_value(flip, "confidence")

        signal = state.last_signal
        if signal is not None:
            context["funding_signal_type"] = self._enum_str(self._get_value(signal, "signal_type"))
            context["funding_signal_score"] = self._get_value(signal, "score")
            context["funding_signal_confidence"] = self._get_value(signal, "confidence")

        return serialize_for_event({k: v for k, v in context.items() if v is not None})

    def _build_full_analytics_context(self, state: FundingStrategyState) -> dict[str, Any]:
        return serialize_for_event(
            {
                "snapshot": state.last_snapshot,
                "statistics": state.last_statistics,
                "regime": state.last_regime,
                "pressure": state.last_pressure,
                "extreme": state.last_extreme,
                "divergence": state.last_divergence,
                "flip": state.last_flip,
                "signal": state.last_signal,
                "last_funding_updated_payload": state.last_funding_updated_payload,
            }
        )

    def _priority_for_event_kind(self, event_kind: str) -> EventPriority:
        if event_kind == "confirmed":
            return self.config.confirmation_priority
        if event_kind == "invalidated":
            return self.config.invalidation_priority
        if event_kind == "expired":
            return self.config.expiration_priority
        return self.config.setup_priority

    def _snapshot_event_bus_subscription_ids(self) -> set[int]:
        subscriptions = getattr(self.event_bus, "_subscriptions", None)
        if not isinstance(subscriptions, list):
            return set()
        return {id(item) for item in subscriptions}

    def _capture_new_event_bus_subscriptions(self, before_ids: set[int]) -> None:
        subscriptions = getattr(self.event_bus, "_subscriptions", None)
        if not isinstance(subscriptions, list):
            return
        for subscription in subscriptions:
            if id(subscription) not in before_ids and subscription not in self._subscriptions:
                self._subscriptions.append(subscription)

    def _register_scheduler_jobs(self) -> None:
        if self.scheduler is None or not self.config.enable_scheduler_cleanup:
            return
        if self._cleanup_job_id is not None:
            return

        existing = self.scheduler.get_job_by_name(f"{self.strategy_name}:cleanup_expired_states")
        if existing is not None:
            self._cleanup_job_id = existing.job_id
            return

        self._cleanup_job_id = self.scheduler.add_interval_job(
            name=f"{self.strategy_name}:cleanup_expired_states",
            func=self.cleanup_expired_states,
            interval=self.config.cleanup_interval_sec,
            kwargs={"emit_events": True},
            run_immediately=False,
            max_retries=1,
            retry_delay=1.0,
            timeout=self.config.cleanup_job_timeout_sec,
            allow_overlap=False,
            enabled=True,
        )

    def _unregister_scheduler_jobs(self) -> None:
        if self.scheduler is None or self._cleanup_job_id is None:
            self._cleanup_job_id = None
            return
        try:
            self.scheduler.remove_job(self._cleanup_job_id)
        except KeyError:
            pass
        finally:
            self._cleanup_job_id = None

    # -------------------------------------------------------------------------
    # Optional hooks for descendants
    # -------------------------------------------------------------------------

    def on_before_setup_emit(self, state: FundingStrategyState, payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    def on_before_confirmation_emit(self, state: FundingStrategyState, payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    def on_before_invalidation_emit(self, state: FundingStrategyState, payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    def on_before_expiration_emit(self, state: FundingStrategyState, payload: dict[str, Any]) -> dict[str, Any]:
        return payload
