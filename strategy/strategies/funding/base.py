from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from core.logger import get_logger


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


# =============================================================================
# Enums / Dataclasses
# =============================================================================


class FundingSetupStatus(str, Enum):
    """
    Поточний життєвий цикл funding setup в strategy layer.
    """

    IDLE = "idle"
    SETUP_DETECTED = "setup_detected"
    CONFIRMED = "confirmed"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"
    COOLDOWN = "cooldown"


class FundingStrategyDirection(str, Enum):
    """
    Напрямок очікуваного трейду.
    """

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass(slots=True)
class BaseFundingStrategyConfig:
    """
    Базовий конфіг для funding strategy layer.

    Це спільний набір налаштувань для будь-якої funding-стратегії:
    - TTL setup
    - cooldown
    - freshness checks
    - timeouts для lock
    - базова політика повторного emit
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


@dataclass(slots=True)
class FundingStrategyState:
    """
    Локальний strategy-state по конкретному symbol/exchange.

    Тут зберігається:
    - поточний статус setup
    - напрям
    - score/confidence
    - час створення / підтвердження / інвалідації
    - останні funding analytics events/state
    - довільні metadata / reasons
    """

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
            "metadata": dict(self.metadata),
        }


# =============================================================================
# Base Strategy
# =============================================================================


class BaseFundingStrategy(ABC):
    """
    Базовий клас для strategy/funding/*.

    Відповідальність:
    - тримати локальний state по symbol/exchange
    - дати helper-методи для setup lifecycle
    - забезпечити уніфікований emit strategy events у EventBus
    - прибрати дублювання між funding strategies

    НЕ відповідає за конкретну торгову логіку.
    Вся конкретна логіка живе в дочірніх класах.
    """

    def __init__(
        self,
        event_bus: Any,
        config: BaseFundingStrategyConfig | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.config = config or BaseFundingStrategyConfig()
        self.logger = get_logger(__name__)

        self._states: dict[str, FundingStrategyState] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._registered: bool = False

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def register(self) -> None:
        """
        Реєстрація підписок дочірнього класу.
        """
        if self._registered:
            self.logger.warning("%s already registered", self.__class__.__name__)
            return

        self.register_subscriptions()
        self._registered = True

        self.logger.info(
            "%s registered successfully: namespace=%s",
            self.__class__.__name__,
            self.config.strategy_namespace,
        )

    @abstractmethod
    def register_subscriptions(self) -> None:
        """
        Тут дочірній клас має викликати event_bus.subscribe(...).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """
        Коротка canonical назва стратегії.
        Наприклад:
        - funding_extreme_reversal
        - funding_divergence
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # State access
    # -------------------------------------------------------------------------

    def get_state(
        self,
        symbol: str,
        exchange: str = "unknown",
    ) -> FundingStrategyState:
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
        """
        Повний reset state для символу.

        Може бути корисно після hard invalidation, manual reset або restart recovery.
        """
        previous = self.get_state(symbol, exchange)
        cooldown_until = previous.cooldown_until if preserve_cooldown else None

        new_state = FundingStrategyState(
            symbol=symbol,
            exchange=exchange,
            strategy_name=self.strategy_name,
            cooldown_until=cooldown_until,
            status=FundingSetupStatus.COOLDOWN if cooldown_until and cooldown_until > utc_now() else FundingSetupStatus.IDLE,
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

    async def acquire_symbol_lock(
        self,
        symbol: str,
        exchange: str = "unknown",
    ) -> asyncio.Lock | None:
        """
        Helper для дочірніх стратегій, щоб серіалізувати оновлення стану по символу.
        """
        key = self._make_key(symbol, exchange)
        lock = self._locks.setdefault(key, asyncio.Lock())

        try:
            await asyncio.wait_for(lock.acquire(), timeout=self.config.state_lock_timeout_sec)
            return lock
        except asyncio.TimeoutError:
            self.logger.warning(
                "%s lock timeout: strategy=%s symbol=%s exchange=%s",
                self.__class__.__name__,
                self.strategy_name,
                symbol,
                exchange,
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
        state.metadata = dict(metadata or {})

        self.logger.debug(
            "%s setup detected: symbol=%s exchange=%s type=%s direction=%s score=%.4f confidence=%.4f",
            self.strategy_name,
            state.symbol,
            state.exchange,
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
            state.metadata.update(metadata)

        self.logger.debug(
            "%s setup confirmed: symbol=%s exchange=%s type=%s direction=%s score=%.4f confidence=%.4f",
            self.strategy_name,
            state.symbol,
            state.exchange,
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
            state.metadata.update(metadata)

        if cooldown:
            state.cooldown_until = now + timedelta(seconds=self.config.cooldown_sec)
            state.status = FundingSetupStatus.COOLDOWN

        self.logger.debug(
            "%s setup invalidated: symbol=%s exchange=%s type=%s reason=%s cooldown=%s",
            self.strategy_name,
            state.symbol,
            state.exchange,
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

        self.logger.debug(
            "%s setup expired: symbol=%s exchange=%s type=%s cooldown=%s",
            self.strategy_name,
            state.symbol,
            state.exchange,
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

    def set_idle(
        self,
        state: FundingStrategyState,
    ) -> FundingStrategyState:
        """
        М'який перехід в idle зі збереженням останніх analytics-посилань.
        """
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
        """
        Уніфіковане витягування payload з event bus event або plain dict.
        """
        if event is None:
            return {}

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

    def extract_symbol_exchange(
        self,
        payload: dict[str, Any],
    ) -> tuple[str, str]:
        symbol = str(payload.get("symbol", "")).upper().strip()
        exchange = str(payload.get("exchange", "unknown")).lower().strip()
        return symbol, exchange

    def extract_event_time(
        self,
        payload: dict[str, Any],
    ) -> datetime | None:
        """
        Підтримує datetime або ISO string.
        """
        raw = payload.get("event_time") or payload.get("timestamp") or payload.get("ts")
        if raw is None:
            return None

        if isinstance(raw, datetime):
            return ensure_utc(raw)

        if isinstance(raw, str):
            try:
                return ensure_utc(datetime.fromisoformat(raw.replace("Z", "+00:00")))
            except ValueError:
                return None

        return None

    def event_age_seconds(
        self,
        event_time: datetime | None,
    ) -> float | None:
        event_time = ensure_utc(event_time)
        if event_time is None:
            return None
        return max(0.0, (utc_now() - event_time).total_seconds())

    def is_stale_event(
        self,
        event_time: datetime | None,
    ) -> bool:
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
        if not self.config.emit_setup_events:
            return

        payload = self.build_base_signal_payload(
            state=state,
            event_kind="setup",
            extra_payload=extra_payload,
        )
        await self._emit(
            event_name=f"{self.config.strategy_namespace}.setup",
            payload=payload,
        )

    async def emit_confirmation_event(
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
        await self._emit(
            event_name=f"{self.config.strategy_namespace}.confirmed",
            payload=payload,
        )

    async def emit_invalidation_event(
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
        await self._emit(
            event_name=f"{self.config.strategy_namespace}.invalidated",
            payload=payload,
        )

    async def emit_expiration_event(
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
        await self._emit(
            event_name=f"{self.config.strategy_namespace}.expired",
            payload=payload,
        )

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
            "metadata": dict(state.metadata),
        }

        if self.config.attach_full_state_on_emit:
            payload["state"] = state.to_dict()

        funding_context = self._build_funding_context(state)
        if funding_context:
            payload["funding_context"] = funding_context

        if extra_payload:
            payload.update(extra_payload)

        return payload

    async def _emit(
        self,
        *,
        event_name: str,
        payload: dict[str, Any],
    ) -> None:
        try:
            await self.event_bus.emit(
                event_name,
                payload,
                source=self.config.source_name,
            )

            symbol = str(payload.get("symbol", "")).upper().strip()
            exchange = str(payload.get("exchange", "unknown")).lower().strip()
            state = self._states.get(self._make_key(symbol, exchange))
            if state is not None:
                state.last_emit_time = utc_now()
                state.emit_count += 1

        except Exception:
            self.logger.exception(
                "Failed to emit strategy event: strategy=%s event_name=%s",
                self.strategy_name,
                event_name,
            )

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

    def _build_funding_context(
        self,
        state: FundingStrategyState,
    ) -> dict[str, Any]:
        """
        Формує компактний funding context для emit payload.
        Працює максимально обережно через getattr, щоб не падати
        на різних типах funding events/state.
        """
        context: dict[str, Any] = {}

        regime = state.last_regime
        if regime is not None:
            context["regime"] = getattr(getattr(regime, "regime", None), "value", None)
            context["bias"] = getattr(getattr(regime, "bias", None), "value", None)
            context["regime_confidence"] = getattr(regime, "confidence", None)

        pressure = state.last_pressure
        if pressure is not None:
            context["pressure_direction"] = getattr(
                getattr(pressure, "direction", None),
                "value",
                None,
            )
            context["pressure_level"] = getattr(
                getattr(pressure, "level", None),
                "value",
                None,
            )
            context["pressure_score"] = getattr(pressure, "pressure_score", None)
            context["squeeze_probability"] = getattr(pressure, "squeeze_probability", None)
            context["mean_reversion_probability"] = getattr(
                pressure,
                "mean_reversion_probability",
                None,
            )

        extreme = state.last_extreme
        if extreme is not None:
            context["extreme_type"] = getattr(
                getattr(extreme, "extreme_type", None),
                "value",
                None,
            )
            context["extreme_severity"] = getattr(extreme, "severity", None)
            context["is_reversal_risk"] = getattr(extreme, "is_reversal_risk", None)
            context["is_squeeze_risk"] = getattr(extreme, "is_squeeze_risk", None)

        divergence = state.last_divergence
        if divergence is not None:
            context["divergence_type"] = getattr(
                getattr(divergence, "divergence_type", None),
                "value",
                None,
            )
            context["divergence_confidence"] = getattr(divergence, "confidence", None)

        flip = state.last_flip
        if flip is not None:
            context["flip_type"] = getattr(
                getattr(flip, "flip_type", None),
                "value",
                None,
            )
            context["flip_confidence"] = getattr(flip, "confidence", None)

        return {k: v for k, v in context.items() if v is not None}

    # -------------------------------------------------------------------------
    # Optional hooks for descendants
    # -------------------------------------------------------------------------

    def on_before_setup_emit(
        self,
        state: FundingStrategyState,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return payload

    def on_before_confirmation_emit(
        self,
        state: FundingStrategyState,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return payload

    def on_before_invalidation_emit(
        self,
        state: FundingStrategyState,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return payload

    def on_before_expiration_emit(
        self,
        state: FundingStrategyState,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
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
        payload = self.build_base_signal_payload(
            state=state,
            event_kind="setup",
            extra_payload=extra_payload,
        )
        payload = self.on_before_setup_emit(state, payload)
        if self.config.emit_setup_events:
            await self._emit(
                event_name=f"{self.config.strategy_namespace}.setup",
                payload=payload,
            )

    async def emit_confirmed(
        self,
        state: FundingStrategyState,
        *,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        payload = self.build_base_signal_payload(
            state=state,
            event_kind="confirmed",
            extra_payload=extra_payload,
        )
        payload = self.on_before_confirmation_emit(state, payload)
        if self.config.emit_confirmation_events:
            await self._emit(
                event_name=f"{self.config.strategy_namespace}.confirmed",
                payload=payload,
            )

    async def emit_invalidated(
        self,
        state: FundingStrategyState,
        *,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        payload = self.build_base_signal_payload(
            state=state,
            event_kind="invalidated",
            extra_payload=extra_payload,
        )
        payload = self.on_before_invalidation_emit(state, payload)
        if self.config.emit_invalidation_events:
            await self._emit(
                event_name=f"{self.config.strategy_namespace}.invalidated",
                payload=payload,
            )

    async def emit_expired(
        self,
        state: FundingStrategyState,
        *,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        payload = self.build_base_signal_payload(
            state=state,
            event_kind="expired",
            extra_payload=extra_payload,
        )
        payload = self.on_before_expiration_emit(state, payload)
        if self.config.emit_expiration_events:
            await self._emit(
                event_name=f"{self.config.strategy_namespace}.expired",
                payload=payload,
            )