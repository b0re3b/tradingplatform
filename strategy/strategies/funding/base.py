from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from core.event_bus import Event, EventBus, EventPriority, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler


# =============================================================================
# Helpers
# =============================================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def serialize_for_event(value: Any) -> Any:
    """Convert common non-JSON-safe values before putting them into EventBus payloads."""
    if isinstance(value, datetime):
        return ensure_utc(value).isoformat() if ensure_utc(value) else None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): serialize_for_event(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_for_event(item) for item in value]
    return value


# =============================================================================
# Enums / Dataclasses
# =============================================================================


class FundingSetupStatus(str, Enum):
    """Поточний життєвий цикл funding setup в strategy layer."""

    IDLE = "idle"
    SETUP_DETECTED = "setup_detected"
    CONFIRMED = "confirmed"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"
    COOLDOWN = "cooldown"


class FundingStrategyDirection(str, Enum):
    """Напрямок очікуваного трейду."""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass(slots=True)
class BaseFundingStrategyConfig:
    """
    Базовий конфіг для funding strategy layer.

    Узгоджено з core:
    - EventBus priority/header/correlation support
    - Scheduler-compatible cleanup
    - validate() як у core.config-style dataclass configs
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

    def validate(self) -> None:
        if self.setup_ttl_sec <= 0:
            raise ValueError("setup_ttl_sec must be > 0")
        if self.cooldown_sec < 0:
            raise ValueError("cooldown_sec must be >= 0")
        if self.event_stale_after_sec <= 0:
            raise ValueError("event_stale_after_sec must be > 0")
        if self.state_lock_timeout_sec <= 0:
            raise ValueError("state_lock_timeout_sec must be > 0")
        if not self.strategy_namespace.strip():
            raise ValueError("strategy_namespace must not be empty")
        if not self.source_name.strip():
            raise ValueError("source_name must not be empty")
        if not self.service_name.strip():
            raise ValueError("service_name must not be empty")
        if self.cleanup_interval_sec <= 0:
            raise ValueError("cleanup_interval_sec must be > 0")
        if self.cleanup_job_timeout_sec <= 0:
            raise ValueError("cleanup_job_timeout_sec must be > 0")


@dataclass(slots=True)
class FundingStrategyState:
    """Локальний strategy-state по конкретному symbol/exchange."""

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

    last_regime: Any | None = None
    last_pressure: Any | None = None
    last_extreme: Any | None = None
    last_divergence: Any | None = None
    last_flip: Any | None = None
    last_signal: Any | None = None

    setup_event_time: datetime | None = None
    confirmation_event_time: datetime | None = None
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
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "status": self.status.value,
            "direction": self.direction.value,
            "strategy_name": self.strategy_name,
            "setup_type": self.setup_type,
            "score": self.score,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "confirmed_at": self.confirmed_at.isoformat() if self.confirmed_at else None,
            "invalidated_at": self.invalidated_at.isoformat() if self.invalidated_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "reason": self.reason,
            "reasons": list(self.reasons),
            "tags": list(self.tags),
            "setup_event_time": self.setup_event_time.isoformat() if self.setup_event_time else None,
            "confirmation_event_time": (
                self.confirmation_event_time.isoformat()
                if self.confirmation_event_time
                else None
            ),
            "last_emit_time": self.last_emit_time.isoformat() if self.last_emit_time else None,
            "emit_count": self.emit_count,
            "metadata": serialize_for_event(dict(self.metadata)),
        }


# =============================================================================
# Base Strategy
# =============================================================================


class BaseFundingStrategy(ABC):
    """
    Базовий клас для strategy/funding/*.

    Відповідність core:
    - typed EventBus / Subscription / EventPriority
    - lifecycle start / stop / restart + backward-compatible register()
    - subscriptions зберігаються і коректно unsubscribe-яться
    - Scheduler cleanup optional
    - get_logger(__name__, event_type="strategy", strategies=...)
    - EventBus.emit з priority / source / correlation_id / headers
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: BaseFundingStrategyConfig | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        if event_bus is None:
            raise ValueError("event_bus is required")

        self.event_bus = event_bus
        self.config = config or BaseFundingStrategyConfig()
        self.config.validate()
        self.scheduler = scheduler

        self.logger = get_logger(
            __name__,
            service_name=self.config.service_name,
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
            "Funding strategy started | strategy=%s namespace=%s subscriptions=%s",
            self.strategy_name,
            self.config.strategy_namespace,
            len(self._subscriptions),
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
            except ValueError:
                pass
            except Exception:
                self.logger.exception(
                    "Failed to unsubscribe funding strategy handler | strategy=%s pattern=%s",
                    self.strategy_name,
                    getattr(subscription, "pattern", "unknown"),
                )

        self._subscriptions.clear()
        self._registered = False

    def subscribe(
        self,
        pattern: str,
        handler: Any,
        *,
        name: str | None = None,
    ) -> Subscription:
        """Preferred subscription helper for child classes."""
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
        """Дочірній клас має підписатися через self.subscribe(...)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        raise NotImplementedError

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

    def reset_state(
        self,
        symbol: str,
        exchange: str = "unknown",
        preserve_cooldown: bool = True,
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

        new_state.last_regime = previous.last_regime
        new_state.last_pressure = previous.last_pressure
        new_state.last_extreme = previous.last_extreme
        new_state.last_divergence = previous.last_divergence
        new_state.last_flip = previous.last_flip
        new_state.last_signal = previous.last_signal

        self._states[new_state.key] = new_state
        return new_state

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
            ).warning(
                "Funding strategy lock timeout | strategy=%s",
                self.strategy_name,
            )
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
        preserved_regime = state.last_regime
        preserved_pressure = state.last_pressure
        preserved_extreme = state.last_extreme
        preserved_divergence = state.last_divergence
        preserved_flip = state.last_flip
        preserved_signal = state.last_signal

        new_state = FundingStrategyState(
            symbol=state.symbol,
            exchange=state.exchange,
            strategy_name=state.strategy_name,
        )
        new_state.last_regime = preserved_regime
        new_state.last_pressure = preserved_pressure
        new_state.last_extreme = preserved_extreme
        new_state.last_divergence = preserved_divergence
        new_state.last_flip = preserved_flip
        new_state.last_signal = preserved_signal

        self._states[new_state.key] = new_state
        return new_state

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
        state.updated_at = utc_now()

    # -------------------------------------------------------------------------
    # Event helpers
    # -------------------------------------------------------------------------

    def extract_payload(self, event: Any) -> dict[str, Any]:
        if event is None:
            return {}

        if isinstance(event, Event):
            return event.payload if isinstance(event.payload, dict) else {}

        if isinstance(event, dict):
            if isinstance(event.get("payload"), dict):
                return event["payload"]
            return event

        payload = getattr(event, "payload", None)
        if isinstance(payload, dict):
            return payload

        if hasattr(event, "__dict__"):
            raw = vars(event)
            if isinstance(raw.get("payload"), dict):
                return raw["payload"]
            return raw

        return {}

    def extract_symbol_exchange(self, payload: dict[str, Any]) -> tuple[str, str]:
        symbol = str(payload.get("symbol", "")).upper().strip()
        exchange = str(payload.get("exchange", "unknown")).lower().strip()
        return symbol, exchange

    def extract_event_time(self, payload: dict[str, Any]) -> datetime | None:
        raw = payload.get("event_time") or payload.get("timestamp") or payload.get("ts")
        if raw is None:
            return None

        if isinstance(raw, datetime):
            return ensure_utc(raw)

        if isinstance(raw, (int, float)):
            # Accept seconds or milliseconds timestamps.
            timestamp = float(raw) / 1000.0 if raw > 10_000_000_000 else float(raw)
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)

        if isinstance(raw, str):
            try:
                return ensure_utc(datetime.fromisoformat(raw.replace("Z", "+00:00")))
            except ValueError:
                return None

        return None

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

    # -------------------------------------------------------------------------
    # Score / direction / key helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _clip_score(value: float | int | None) -> float:
        if value is None:
            return 0.0
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _average_scores(*values: float | None) -> float:
        valid = [max(0.0, min(1.0, float(v))) for v in values if v is not None]
        if not valid:
            return 0.0
        return sum(valid) / len(valid)

    @staticmethod
    def _make_key(symbol: str, exchange: str = "unknown") -> str:
        return f"{str(symbol).upper().strip()}:{str(exchange).lower().strip()}"

    # -------------------------------------------------------------------------
    # Protected composition helpers
    # -------------------------------------------------------------------------

    def _build_funding_context(self, state: FundingStrategyState) -> dict[str, Any]:
        context: dict[str, Any] = {}

        regime = state.last_regime
        if regime is not None:
            context["regime"] = getattr(getattr(regime, "regime", None), "value", None)
            context["bias"] = getattr(getattr(regime, "bias", None), "value", None)
            context["regime_confidence"] = getattr(regime, "confidence", None)

        pressure = state.last_pressure
        if pressure is not None:
            context["pressure_direction"] = getattr(getattr(pressure, "direction", None), "value", None)
            context["pressure_level"] = getattr(getattr(pressure, "level", None), "value", None)
            context["pressure_score"] = getattr(pressure, "pressure_score", None)
            context["squeeze_probability"] = getattr(pressure, "squeeze_probability", None)
            context["mean_reversion_probability"] = getattr(pressure, "mean_reversion_probability", None)

        extreme = state.last_extreme
        if extreme is not None:
            context["extreme_type"] = getattr(getattr(extreme, "extreme_type", None), "value", None)
            context["extreme_severity"] = getattr(extreme, "severity", None)
            context["is_reversal_risk"] = getattr(extreme, "is_reversal_risk", None)
            context["is_squeeze_risk"] = getattr(extreme, "is_squeeze_risk", None)

        divergence = state.last_divergence
        if divergence is not None:
            context["divergence_type"] = getattr(getattr(divergence, "divergence_type", None), "value", None)
            context["divergence_confidence"] = getattr(divergence, "confidence", None)

        flip = state.last_flip
        if flip is not None:
            context["flip_type"] = getattr(getattr(flip, "flip_type", None), "value", None)
            context["flip_confidence"] = getattr(flip, "confidence", None)

        return serialize_for_event({k: v for k, v in context.items() if v is not None})

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

    # -------------------------------------------------------------------------
    # High-level convenience wrappers
    # -------------------------------------------------------------------------

    async def emit_setup(
        self,
        state: FundingStrategyState,
        *,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        if not self.config.emit_setup_events:
            return
        payload = self.build_base_signal_payload(
            state=state,
            event_kind="setup",
            extra_payload=extra_payload,
        )
        payload = self.on_before_setup_emit(state, payload)
        await self._emit(
            event_name=f"{self.config.strategy_namespace}.setup",
            payload=payload,
            priority=self.config.setup_priority,
        )

    async def emit_confirmed(
        self,
        state: FundingStrategyState,
        *,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        if not self.config.emit_confirmation_events:
            return
        payload = self.build_base_signal_payload(
            state=state,
            event_kind="confirmed",
            extra_payload=extra_payload,
        )
        payload = self.on_before_confirmation_emit(state, payload)
        await self._emit(
            event_name=f"{self.config.strategy_namespace}.confirmed",
            payload=payload,
            priority=self.config.confirmation_priority,
        )

    async def emit_invalidated(
        self,
        state: FundingStrategyState,
        *,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        if not self.config.emit_invalidation_events:
            return
        payload = self.build_base_signal_payload(
            state=state,
            event_kind="invalidated",
            extra_payload=extra_payload,
        )
        payload = self.on_before_invalidation_emit(state, payload)
        await self._emit(
            event_name=f"{self.config.strategy_namespace}.invalidated",
            payload=payload,
            priority=self.config.invalidation_priority,
        )

    async def emit_expired(
        self,
        state: FundingStrategyState,
        *,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        if not self.config.emit_expiration_events:
            return
        payload = self.build_base_signal_payload(
            state=state,
            event_kind="expired",
            extra_payload=extra_payload,
        )
        payload = self.on_before_expiration_emit(state, payload)
        await self._emit(
            event_name=f"{self.config.strategy_namespace}.expired",
            payload=payload,
            priority=self.config.expiration_priority,
        )