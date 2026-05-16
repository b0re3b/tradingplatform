from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from core.logger import get_logger

from risk.config import CircuitBreakerConfig
from risk.enums import CircuitBreakerReason, RiskDecisionType, RiskLevel, RiskViolationType
from risk.models import RiskCheckResult, RiskViolation
from risk.state import RiskState


@dataclass(slots=True)
class CircuitBreakerStats:
    """
    Runtime counters for CircuitBreaker.

    The object is intentionally local to CircuitBreaker. RiskManager may mirror
    important events into RiskMetrics and/or durable storage.

    The stats support snapshot/restore so a restart does not silently erase
    recent breaker history. Persistence itself stays outside this class.
    """

    consecutive_failures: int = 0
    execution_failures: int = 0
    data_failures: int = 0
    system_failures: int = 0
    cost_spikes: int = 0
    manual_triggers: int = 0
    emergency_triggers: int = 0
    total_failures: int = 0
    total_triggers: int = 0
    auto_releases: int = 0
    force_releases: int = 0

    last_failure_reason: str | None = None
    last_failure_at: float | None = None
    last_trigger_reason: str | None = None
    last_triggered_at: float | None = None
    last_release_reason: str | None = None
    last_released_at: float | None = None

    trigger_counts: dict[str, int] = field(default_factory=dict)
    failure_counts: dict[str, int] = field(default_factory=dict)

    def register_failure(self, reason: CircuitBreakerReason, *, now_ts: float | None = None) -> None:
        now_ts = _now(now_ts)
        reason_value = reason.value

        self.consecutive_failures += 1
        self.total_failures += 1
        self.last_failure_reason = reason_value
        self.last_failure_at = now_ts
        self.failure_counts[reason_value] = self.failure_counts.get(reason_value, 0) + 1

        if _is_execution_failure(reason):
            self.execution_failures += 1

        if _is_data_failure(reason):
            self.data_failures += 1

        if _is_system_failure(reason):
            self.system_failures += 1

        if _is_cost_spike(reason):
            self.cost_spikes += 1

    def register_trigger(self, reason: CircuitBreakerReason, *, now_ts: float | None = None) -> None:
        now_ts = _now(now_ts)
        reason_value = reason.value

        self.total_triggers += 1
        self.last_trigger_reason = reason_value
        self.last_triggered_at = now_ts
        self.trigger_counts[reason_value] = self.trigger_counts.get(reason_value, 0) + 1

        if reason is CircuitBreakerReason.MANUAL_HALT:
            self.manual_triggers += 1

        if reason is CircuitBreakerReason.EMERGENCY_STOP:
            self.emergency_triggers += 1

    def register_success(self) -> None:
        self.consecutive_failures = 0

    def register_release(self, *, forced: bool, reason: str | None = None, now_ts: float | None = None) -> None:
        now_ts = _now(now_ts)
        if forced:
            self.force_releases += 1
        else:
            self.auto_releases += 1
        self.last_release_reason = reason
        self.last_released_at = now_ts

    def reset_failure_counters(self) -> None:
        """
        Reset active failure counters but keep historical trigger/release data.
        """
        self.consecutive_failures = 0
        self.execution_failures = 0
        self.data_failures = 0
        self.system_failures = 0
        self.cost_spikes = 0
        self.last_failure_reason = None
        self.last_failure_at = None

    def reset_all(self) -> None:
        self.consecutive_failures = 0
        self.execution_failures = 0
        self.data_failures = 0
        self.system_failures = 0
        self.cost_spikes = 0
        self.manual_triggers = 0
        self.emergency_triggers = 0
        self.total_failures = 0
        self.total_triggers = 0
        self.auto_releases = 0
        self.force_releases = 0
        self.last_failure_reason = None
        self.last_failure_at = None
        self.last_trigger_reason = None
        self.last_triggered_at = None
        self.last_release_reason = None
        self.last_released_at = None
        self.trigger_counts.clear()
        self.failure_counts.clear()

    def snapshot(self, *, enabled: bool) -> dict[str, Any]:
        return {
            "enabled": enabled,
            "consecutive_failures": self.consecutive_failures,
            "execution_failures": self.execution_failures,
            "data_failures": self.data_failures,
            "system_failures": self.system_failures,
            "cost_spikes": self.cost_spikes,
            "manual_triggers": self.manual_triggers,
            "emergency_triggers": self.emergency_triggers,
            "total_failures": self.total_failures,
            "total_triggers": self.total_triggers,
            "auto_releases": self.auto_releases,
            "force_releases": self.force_releases,
            "last_failure_reason": self.last_failure_reason,
            "last_failure_at": self.last_failure_at,
            "last_trigger_reason": self.last_trigger_reason,
            "last_triggered_at": self.last_triggered_at,
            "last_release_reason": self.last_release_reason,
            "last_released_at": self.last_released_at,
            "trigger_counts": dict(self.trigger_counts),
            "failure_counts": dict(self.failure_counts),
        }

    @classmethod
    def from_snapshot(cls, payload: Mapping[str, Any]) -> CircuitBreakerStats:
        stats = cls()
        stats.consecutive_failures = _coerce_int(payload.get("consecutive_failures"), 0)
        stats.execution_failures = _coerce_int(payload.get("execution_failures"), 0)
        stats.data_failures = _coerce_int(payload.get("data_failures"), 0)
        stats.system_failures = _coerce_int(payload.get("system_failures"), 0)
        stats.cost_spikes = _coerce_int(payload.get("cost_spikes"), 0)
        stats.manual_triggers = _coerce_int(payload.get("manual_triggers"), 0)
        stats.emergency_triggers = _coerce_int(payload.get("emergency_triggers"), 0)
        stats.total_failures = _coerce_int(payload.get("total_failures"), 0)
        stats.total_triggers = _coerce_int(payload.get("total_triggers"), 0)
        stats.auto_releases = _coerce_int(payload.get("auto_releases"), 0)
        stats.force_releases = _coerce_int(payload.get("force_releases"), 0)

        stats.last_failure_reason = _coerce_optional_str(payload.get("last_failure_reason"))
        stats.last_failure_at = _coerce_optional_float(payload.get("last_failure_at"))
        stats.last_trigger_reason = _coerce_optional_str(payload.get("last_trigger_reason"))
        stats.last_triggered_at = _coerce_optional_float(payload.get("last_triggered_at"))
        stats.last_release_reason = _coerce_optional_str(payload.get("last_release_reason"))
        stats.last_released_at = _coerce_optional_float(payload.get("last_released_at"))

        stats.trigger_counts = _coerce_counter_dict(payload.get("trigger_counts"))
        stats.failure_counts = _coerce_counter_dict(payload.get("failure_counts"))
        return stats


class CircuitBreaker:
    """
    Emergency top-level risk blocker.

    Responsibilities:
    - block new risk while active;
    - trigger cooldown-based or manual-release circuit breaker;
    - count execution/data/system/cost failures;
    - support emergency stop and manual halt;
    - release after cooldown if manual release is not required;
    - expose snapshot/restore hooks for persistence.

    This class does not publish EventBus events and does not own lifecycle.
    RiskManager owns EventBus emits, locking, Scheduler jobs and storage.
    """

    def __init__(
        self,
        config: CircuitBreakerConfig,
        *,
        service_name: str = "risk.circuit_breaker",
        stats: CircuitBreakerStats | None = None,
    ) -> None:
        self._config = config
        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="circuit_breaker",
        )
        self._stats = stats or CircuitBreakerStats()

    @property
    def stats_state(self) -> CircuitBreakerStats:
        return self._stats

    def check(self, state: RiskState) -> RiskCheckResult:
        if not self._config.enabled:
            return RiskCheckResult(
                passed=True,
                decision=RiskDecisionType.ALLOW,
                metadata={"enabled": False},
            )

        if not state.is_circuit_breaker_active():
            return RiskCheckResult(
                passed=True,
                decision=RiskDecisionType.ALLOW,
                metadata={
                    "enabled": True,
                    "circuit_breaker_active": False,
                    "stats": self.stats(),
                },
            )

        reason = state.circuit_breaker.reason
        reason_value = reason.value if reason is not None else None

        decision = (
            RiskDecisionType.EMERGENCY_STOP
            if reason is CircuitBreakerReason.EMERGENCY_STOP or state.emergency_stop_active
            else RiskDecisionType.HALT_TRADING
        )

        violation_type = (
            RiskViolationType.EMERGENCY_STOP_TRIGGERED
            if decision is RiskDecisionType.EMERGENCY_STOP
            else RiskViolationType.CIRCUIT_BREAKER_TRIGGERED
        )

        metadata = {
            "enabled": True,
            "circuit_breaker_active": True,
            "reason": reason_value,
            "triggered_at": state.circuit_breaker.triggered_at,
            "cooldown_until": state.circuit_breaker.cooldown_until,
            "manual_release_required": state.circuit_breaker.manual_release_required,
            "stats": self.stats(),
        }

        return RiskCheckResult(
            passed=False,
            decision=decision,
            violations=[
                RiskViolation(
                    violation_type=violation_type,
                    level=RiskLevel.CRITICAL,
                    message=state.circuit_breaker.message or "Circuit breaker is active",
                    metadata=metadata,
                )
            ],
            risk_mode=state.risk_mode,
            reason=state.circuit_breaker.message or "Circuit breaker is active",
            metadata=metadata,
        )

    def register_failure(
        self,
        state: RiskState,
        *,
        reason: CircuitBreakerReason = CircuitBreakerReason.EXECUTION_FAILURES,
        message: str | None = None,
        count_as_failure: bool = True,
        now_ts: float | None = None,
    ) -> bool:
        """
        Register a failure and trigger the breaker if thresholds are breached.

        Returns True if this call triggered the breaker.
        """
        if not self._config.enabled:
            return False

        now_ts = _now(now_ts)

        if count_as_failure:
            self._stats.register_failure(reason, now_ts=now_ts)

        self._logger.warning(
            "Circuit breaker failure registered | reason=%s consecutive=%s execution=%s data=%s system=%s cost_spikes=%s",
            reason.value,
            self._stats.consecutive_failures,
            self._stats.execution_failures,
            self._stats.data_failures,
            self._stats.system_failures,
            self._stats.cost_spikes,
            extra={"reason": reason.value},
        )

        if not self._should_trigger(reason):
            return False

        self.trigger(
            state,
            reason=reason,
            message=message or self._default_trigger_message(reason),
            manual_release_required=self._requires_manual_release(reason),
            now_ts=now_ts,
        )
        return True

    def register_success(self) -> None:
        """
        Reset only the consecutive failure streak.

        Historical counters are intentionally kept for diagnostics and optional
        persistence.
        """
        self._stats.register_success()

    def trigger(
        self,
        state: RiskState,
        *,
        reason: CircuitBreakerReason,
        message: str | None = None,
        cooldown_seconds: float | None = None,
        manual_release_required: bool | None = None,
        now_ts: float | None = None,
    ) -> None:
        if not self._config.enabled:
            return

        now_ts = _now(now_ts)

        manual_release = (
            self._requires_manual_release(reason)
            if manual_release_required is None
            else manual_release_required
        )

        effective_cooldown = (
            self._config.cooldown_seconds
            if cooldown_seconds is None
            else cooldown_seconds
        )
        if not _is_finite_non_negative(effective_cooldown):
            effective_cooldown = 0.0

        cooldown_until = None
        if effective_cooldown > 0 and not manual_release:
            cooldown_until = now_ts + effective_cooldown

        trigger_message = message or self._default_trigger_message(reason)

        state.activate_circuit_breaker(
            reason,
            cooldown_until=cooldown_until,
            message=trigger_message,
            manual_release_required=manual_release,
        )

        if reason is CircuitBreakerReason.EMERGENCY_STOP:
            state.emergency_stop(trigger_message)

        self._stats.register_trigger(reason, now_ts=now_ts)

        self._logger.error(
            "Circuit breaker triggered | reason=%s cooldown_until=%s manual_release=%s",
            reason.value,
            cooldown_until,
            manual_release,
            extra={
                "reason": reason.value,
                "cooldown_until": cooldown_until,
                "manual_release_required": manual_release,
            },
        )

    def trigger_emergency_stop(
        self,
        state: RiskState,
        *,
        message: str | None = None,
    ) -> None:
        """Trigger non-auto-releasing emergency stop."""
        self.trigger(
            state,
            reason=CircuitBreakerReason.EMERGENCY_STOP,
            message=message or "Emergency stop triggered",
            cooldown_seconds=0.0,
            manual_release_required=True,
        )

    def trigger_manual_halt(
        self,
        state: RiskState,
        *,
        message: str | None = None,
    ) -> None:
        """Trigger manual halt which requires explicit release."""
        self.trigger(
            state,
            reason=CircuitBreakerReason.MANUAL_HALT,
            message=message or "Manual halt triggered",
            cooldown_seconds=0.0,
            manual_release_required=True,
        )

    def release_if_ready(
        self,
        state: RiskState,
        *,
        now_ts: float | None = None,
    ) -> bool:
        """
        Release breaker after cooldown, unless manual release is required.

        Returns True if breaker was released.
        """
        if not state.circuit_breaker.active:
            return False

        if state.circuit_breaker.manual_release_required:
            return False

        cooldown_until = state.circuit_breaker.cooldown_until
        if cooldown_until is None:
            return False

        now_ts = _now(now_ts)
        if now_ts < cooldown_until:
            return False

        state.deactivate_circuit_breaker()
        self._stats.register_release(
            forced=False,
            reason="cooldown_elapsed",
            now_ts=now_ts,
        )
        self.reset_failure_counters()

        self._logger.info(
            "Circuit breaker released after cooldown | cooldown_until=%s",
            cooldown_until,
        )
        return True

    def force_release(
        self,
        state: RiskState,
        *,
        clear_emergency_stop: bool = False,
        reason: str | None = None,
    ) -> bool:
        """
        Force release breaker.

        Intended for manual operator/recovery flow. RiskManager should decide
        who is allowed to call this and publish any EventBus notifications.
        """
        if not state.circuit_breaker.active:
            return False

        state.deactivate_circuit_breaker(force=True)

        if clear_emergency_stop:
            state.clear_emergency_stop()

        self._stats.register_release(
            forced=True,
            reason=reason or "force_release",
        )
        self.reset_failure_counters()

        self._logger.warning(
            "Circuit breaker force released | clear_emergency_stop=%s reason=%s",
            clear_emergency_stop,
            reason,
        )
        return True

    def reset_failure_counters(self) -> None:
        self._stats.reset_failure_counters()

    def reset_counters(self, *, include_history: bool = False) -> None:
        """
        Backward-compatible reset method.

        By default, only active failure counters are reset. Pass
        include_history=True for a full diagnostic reset.
        """
        if include_history:
            self._stats.reset_all()
        else:
            self._stats.reset_failure_counters()

    def stats(self) -> dict[str, Any]:
        return self._stats.snapshot(enabled=self._config.enabled)

    def snapshot(self, state: RiskState | None = None) -> dict[str, Any]:
        """
        Build a persistence-friendly snapshot.

        Storage is intentionally not handled here. RiskManager/storage layer may
        persist this payload and later call restore_from_snapshot().
        """
        payload: dict[str, Any] = {
            "enabled": self._config.enabled,
            "stats": self.stats(),
            "created_at": time.time(),
        }

        if state is not None:
            reason = state.circuit_breaker.reason
            payload["state"] = {
                "active": state.circuit_breaker.active,
                "reason": reason.value if reason is not None else None,
                "triggered_at": state.circuit_breaker.triggered_at,
                "cooldown_until": state.circuit_breaker.cooldown_until,
                "message": state.circuit_breaker.message,
                "manual_release_required": state.circuit_breaker.manual_release_required,
                "emergency_stop_active": state.emergency_stop_active,
                "trading_halted": state.trading_halted,
                "halt_reason": state.halt_reason,
            }

        return payload

    def restore_from_snapshot(self, payload: Mapping[str, Any]) -> None:
        """
        Restore local diagnostic counters from a previously saved snapshot.

        This method does not mutate RiskState because the breaker state belongs
        to RiskState and should be restored by the state/storage layer.
        """
        stats_payload = payload.get("stats")
        if isinstance(stats_payload, Mapping):
            # Support both payload from snapshot() and raw stats() payload.
            nested_stats = stats_payload.get("stats")
            if isinstance(nested_stats, Mapping):
                self._stats = CircuitBreakerStats.from_snapshot(nested_stats)
            else:
                self._stats = CircuitBreakerStats.from_snapshot(stats_payload)
        else:
            self._stats = CircuitBreakerStats.from_snapshot(payload)

    def _should_trigger(self, reason: CircuitBreakerReason) -> bool:
        if reason is CircuitBreakerReason.MANUAL_HALT:
            return True

        if reason is CircuitBreakerReason.EMERGENCY_STOP:
            return self._config.trigger_on_emergency_stop

        if _is_data_failure(reason):
            return self._config.trigger_on_data_feed_failure

        if _is_cost_spike(reason):
            return self._config.trigger_on_execution_cost_spike

        if self._stats.consecutive_failures >= self._config.max_consecutive_failures:
            return True

        if self._stats.execution_failures >= self._config.max_execution_failures:
            return True

        return False

    def _requires_manual_release(self, reason: CircuitBreakerReason) -> bool:
        if reason is CircuitBreakerReason.EMERGENCY_STOP:
            return self._config.require_manual_release_for_emergency

        if reason is CircuitBreakerReason.MANUAL_HALT:
            return True

        if reason in {
            CircuitBreakerReason.WEEKLY_LOSS_BREACH,
            CircuitBreakerReason.MONTHLY_LOSS_BREACH,
            CircuitBreakerReason.NEGATIVE_GLOBAL_EXPECTANCY,
        }:
            return True

        return False

    @staticmethod
    def _default_trigger_message(reason: CircuitBreakerReason) -> str:
        return f"Circuit breaker triggered: {reason.value}"


def _is_execution_failure(reason: CircuitBreakerReason) -> bool:
    return reason in {
        CircuitBreakerReason.EXECUTION_FAILURES,
        CircuitBreakerReason.EXECUTION_COST_SPIKE,
        CircuitBreakerReason.SLIPPAGE_SPIKE,
        CircuitBreakerReason.SPREAD_ABNORMAL,
    }


def _is_data_failure(reason: CircuitBreakerReason) -> bool:
    return reason in {
        CircuitBreakerReason.DATA_FEED_FAILURE,
        CircuitBreakerReason.DATA_STALE,
    }


def _is_system_failure(reason: CircuitBreakerReason) -> bool:
    return reason in {
        CircuitBreakerReason.SYSTEM_ERROR_RATE,
        CircuitBreakerReason.EXCHANGE_UNSTABLE,
    }


def _is_cost_spike(reason: CircuitBreakerReason) -> bool:
    return reason in {
        CircuitBreakerReason.EXECUTION_COST_SPIKE,
        CircuitBreakerReason.SLIPPAGE_SPIKE,
        CircuitBreakerReason.SPREAD_ABNORMAL,
    }


def _now(value: float | None = None) -> float:
    if value is None:
        return time.time()
    if not math.isfinite(value):
        return time.time()
    return value


def _is_finite_non_negative(value: float | int) -> bool:
    return math.isfinite(float(value)) and value >= 0


def _coerce_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float) and math.isfinite(value):
        return max(0, int(value))
    return default


def _coerce_optional_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    return None


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_counter_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}

    result: dict[str, int] = {}
    for raw_key, raw_count in value.items():
        key = str(raw_key)
        count = _coerce_int(raw_count, 0)
        if count > 0:
            result[key] = count
    return result


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerStats",
]
