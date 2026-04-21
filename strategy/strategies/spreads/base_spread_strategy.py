from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from core.logger import get_logger


@dataclass(slots=True)
class BaseSpreadStrategyConfig:
    """
    Базова конфігурація для spread-стратегій.

    Це strategy-layer config, а не analytics config.
    """

    enabled: bool = True

    cooldown_seconds: int = 10
    min_confidence: Decimal = Decimal("0")

    max_snapshot_age_ms: int = 3_000
    max_signal_age_ms: int = 5_000

    allowed_symbols: set[str] = field(default_factory=set)
    allowed_exchanges: set[str] = field(default_factory=set)

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SpreadStrategyState:
    """
    Уніфікований state для конкретного spread-setup.

    key:
        Унікальний ключ strategy instance для конкретної можливості / пари.
    status:
        idle / pending / open / blocked / closing / closed / cancelled / rejected
    bias:
        Напрямок ідеї, наприклад:
        - arb
        - long_basis
        - short_basis
    """

    key: str
    strategy: str
    symbol: str

    exchange_a: str
    exchange_b: str

    status: str = "idle"
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

    @property
    def is_active(self) -> bool:
        return self.status in {"pending", "open", "closing"}

    @property
    def is_closed(self) -> bool:
        return self.status in {"closed", "cancelled", "rejected"}


class BaseSpreadStrategy(ABC):
    """
    Базовий клас для spread strategy components.

    Відповідальність:
    - lifecycle: start / stop
    - інтеграція з EventBus
    - logger / lock / stats
    - state management для setup-ів
    - cooldown / dedup
    - базові filters:
        - enabled
        - symbol allowlist
        - exchange allowlist
        - confidence threshold
        - freshness
    - publish helpers для strategy-level events

    Не відповідає за:
    - spread analytics
    - розрахунок z-score / regime / basis / costs
    - побудову snapshot-ів
    - arbitrage detection
    - execution management
    """

    SIGNAL_GENERATED_EVENT = "signal.generated"
    SIGNAL_UPDATED_EVENT = "signal.updated"
    SIGNAL_REJECTED_EVENT = "signal.rejected"
    SIGNAL_CANCELLED_EVENT = "signal.cancelled"
    SIGNAL_CLOSED_EVENT = "signal.closed"

    STRATEGY_NAME = "base_spread_strategy"

    def __init__(
        self,
        *,
        event_bus: Any,
        config: BaseSpreadStrategyConfig | None = None,
        scheduler: Any | None = None,
        service_name: str | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._scheduler = scheduler
        self._config = config or BaseSpreadStrategyConfig()

        resolved_service_name = service_name or self.STRATEGY_NAME
        self._logger = get_logger(__name__, service_name=resolved_service_name)

        self._running = False
        self._lock = asyncio.Lock()

        self._states: dict[str, SpreadStrategyState] = {}
        self._last_signal_times: dict[str, datetime] = {}
        self._last_event_times: dict[str, datetime] = {}

        self._stats: dict[str, int] = self._build_base_stats()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def config(self) -> BaseSpreadStrategyConfig:
        return self._config

    @property
    def states(self) -> dict[str, SpreadStrategyState]:
        return self._states

    @abstractmethod
    async def _subscribe_events(self) -> None:
        """
        Конкретна стратегія сама визначає, які події слухати.
        """
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """
        Конкретна стратегія має повернути розширену статистику.
        """
        raise NotImplementedError

    async def start(self) -> None:
        if self._running:
            return

        self._running = True
        await self._subscribe_events()

        self._logger.info(
            "%s started",
            self.__class__.__name__,
            extra=self._build_start_log_extra(),
        )

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False

        self._logger.info(
            "%s stopped",
            self.__class__.__name__,
            extra=self._build_stop_log_extra(),
        )

    def _build_base_stats(self) -> dict[str, int]:
        return {
            "events_received": 0,
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
        }

    def _build_stop_log_extra(self) -> dict[str, Any]:
        return {
            "strategy": self.STRATEGY_NAME,
            "stats": self._stats.copy(),
            "active_states": sum(1 for state in self._states.values() if state.is_active),
            "total_states": len(self._states),
        }

    async def _maybe_await(self, value: Any) -> Any:
        if asyncio.iscoroutine(value) or isinstance(value, asyncio.Future):
            return await value
        return value

    async def _subscribe(
        self,
        event_name: str,
        handler: Any,
    ) -> None:
        subscribe = getattr(self._event_bus, "subscribe", None)
        if subscribe is None:
            self._logger.warning(
                "EventBus does not expose subscribe()",
                extra={
                    "strategy": self.STRATEGY_NAME,
                    "event_name": event_name,
                },
            )
            return

        await self._maybe_await(subscribe(event_name, handler))

    async def _publish(
        self,
        event_name: str,
        payload: Any,
    ) -> None:
        publish = getattr(self._event_bus, "publish", None)
        if publish is None:
            self._logger.warning(
                "EventBus does not expose publish()",
                extra={
                    "strategy": self.STRATEGY_NAME,
                    "event_name": event_name,
                },
            )
            return

        await self._maybe_await(publish(event_name, payload))

    def _utcnow(self) -> datetime:
        return datetime.utcnow()

    def _normalize_symbol(self, symbol: str | None) -> str:
        if not symbol:
            return ""
        return symbol.replace("-", "").replace("/", "").replace("_", "").upper().strip()

    def _normalize_exchange(self, exchange: str | None) -> str:
        if not exchange:
            return ""
        return exchange.strip().lower()

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
        return normalized_symbol in {self._normalize_symbol(item) for item in allowed}

    def _are_exchanges_allowed(self, *exchanges: str | None) -> bool:
        allowed = self._config.allowed_exchanges
        if not allowed:
            return True

        normalized_allowed = {self._normalize_exchange(item) for item in allowed}
        for exchange in exchanges:
            normalized_exchange = self._normalize_exchange(exchange)
            if normalized_exchange and normalized_exchange not in normalized_allowed:
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
        Повертає True, якщо подія дубльована по часу для цього key.
        """
        if timestamp is None:
            return False

        last_seen_at = self._last_event_times.get(key)
        if last_seen_at is not None and last_seen_at == timestamp:
            self._stats["duplicate_skips"] += 1
            return True

        self._last_event_times[key] = timestamp
        return False

    def _build_state_key(self, *parts: Any) -> str:
        cleaned: list[str] = []
        for part in parts:
            if part is None:
                cleaned.append("na")
                continue
            cleaned.append(str(part).strip())
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
            symbol=self._normalize_symbol(symbol),
            exchange_a=self._normalize_exchange(exchange_a),
            exchange_b=self._normalize_exchange(exchange_b),
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

        state.status = "open"
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

        state.status = "pending"
        state.bias = bias or state.bias
        state.updated_at = current_time
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

        state.status = "blocked"
        state.updated_at = current_time
        state.last_reason = reason or state.last_reason

        if metadata:
            state.metadata.update(metadata)

        self._stats["state_updated"] += 1

    def _set_state_closed(
        self,
        state: SpreadStrategyState,
        *,
        status: str = "closed",
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> None:
        current_time = now or self._utcnow()

        state.status = status
        state.updated_at = current_time
        state.closed_at = current_time
        state.last_reason = reason or state.last_reason

        if metadata:
            state.metadata.update(metadata)

        self._stats["state_closed"] += 1
        self._stats["state_updated"] += 1

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

        payload = {
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
        return payload

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
        await self._publish(event_name, payload)
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
        )

    async def _run_safely(
        self,
        operation_name: str,
        coro: Any,
        **context: Any,
    ) -> Any:
        try:
            return await coro
        except Exception as exc:
            self._stats["exceptions"] += 1
            self._logger.exception(
                "Strategy operation failed: %s",
                operation_name,
                extra={
                    "strategy": self.STRATEGY_NAME,
                    **context,
                    "error": str(exc),
                },
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

    def cleanup_closed_states(
        self,
        *,
        older_than_seconds: int = 3600,
        now: datetime | None = None,
    ) -> int:
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

        return len(keys_to_delete)

    def get_base_stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "running": self._running,
            "strategy": self.STRATEGY_NAME,
            "states_total": len(self._states),
            "states_active": sum(1 for state in self._states.values() if state.is_active),
            "states_closed": sum(1 for state in self._states.values() if state.is_closed),
        }