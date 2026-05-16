from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from core.logger import get_logger

from risk.config import CircuitBreakerConfig
from risk.enums import CircuitBreakerReason, RiskDecisionType, RiskLevel, RiskViolationType
from risk.models import RiskCheckResult, RiskViolation
from risk.state import RiskState


@dataclass(slots=True)
class CircuitBreakerStats:
    """
    Runtime counters for circuit breaker.

    This is intentionally local to CircuitBreaker. RiskMetrics may additionally
    aggregate triggered events at RiskManager level.
    """

    consecutive_failures: int = 0
    execution_failures: int = 0
    data_failures: int = 0
    system_failures: int = 0
    cost_spikes: int = 0
    manual_triggers: int = 0
    emergency_triggers: int = 0

    last_failure_reason: str | None = None
    last_trigger_reason: str | None = None
    last_triggered_at: float | None = None

    trigger_counts: dict[str, int] = field(default_factory=dict)

    def register_failure(self, reason: CircuitBreakerReason) -> None:
        self.consecutive_failures += 1
        self.last_failure_reason = reason.value

        if reason in {
            CircuitBreakerReason.EXECUTION_FAILURES,
            CircuitBreakerReason.EXECUTION_COST_SPIKE,
            CircuitBreakerReason.SLIPPAGE_SPIKE,
            CircuitBreakerReason.SPREAD_ABNORMAL,
        }:
            self.execution_failures += 1

        if reason in {
            CircuitBreakerReason.DATA_FEED_FAILURE,
            CircuitBreakerReason.DATA_STALE,
        }:
            self.data_failures += 1

        if reason in {
            CircuitBreakerReason.SYSTEM_ERROR_RATE,
            CircuitBreakerReason.EXCHANGE_UNSTABLE,
        }:
            self.system_failures += 1

        if reason in {
            CircuitBreakerReason.EXECUTION_COST_SPIKE,
            CircuitBreakerReason.SLIPPAGE_SPIKE,
            CircuitBreakerReason.SPREAD_ABNORMAL,
        }:
            self.cost_spikes += 1

    def register_trigger(self, reason: CircuitBreakerReason) -> None:
        self.last_trigger_reason = reason.value
        self.last_triggered_at = time.time()
        self.trigger_counts[reason.value] = self.trigger_counts.get(reason.value, 0) + 1

        if reason is CircuitBreakerReason.MANUAL_HALT:
            self.manual_triggers += 1

        if reason is CircuitBreakerReason.EMERGENCY_STOP:
            self.emergency_triggers += 1

    def register_success(self) -> None:
        self.consecutive_failures = 0

    def reset(self) -> None:
        self.consecutive_failures = 0
        self.execution_failures = 0
        self.data_failures = 0
        self.system_failures = 0
        self.cost_spikes = 0
        self.last_failure_reason = None

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
            "last_failure_reason": self.last_failure_reason,
            "last_trigger_reason": self.last_trigger_reason,
            "last_triggered_at": self.last_triggered_at,
            "trigger_counts": dict(self.trigger_counts),
        }


class CircuitBreaker:
    """
    Emergency top-level risk blocker.

    Responsibilities:
    - block new risk when active;
    - trigger cooldown-based or manual-release circuit breaker;
    - count execution/data/system/cost failures;
    - support emergency stop;
    - release after cooldown if manual release is not required.

    This class does not publish EventBus events and does not own lifecycle.
    RiskManager is responsible for EventBus emits, locking and Scheduler.
    """

    def __init__(
        self,
        config: CircuitBreakerConfig,
        *,
        service_name: str = "risk.circuit_breaker",
    ) -> None:
        self._config = config
        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="circuit_breaker",
        )
        self._stats = CircuitBreakerStats()

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

        return RiskCheckResult(
            passed=False,
            decision=decision,
            violations=[
                RiskViolation(
                    violation_type=violation_type,
                    level=RiskLevel.CRITICAL,
                    message=state.circuit_breaker.message or "Circuit breaker is active",
                    metadata={
                        "reason": reason_value,
                        "triggered_at": state.circuit_breaker.triggered_at,
                        "cooldown_until": state.circuit_breaker.cooldown_until,
                        "manual_release_required": state.circuit_breaker.manual_release_required,
                    },
                )
            ],
            risk_mode=state.risk_mode,
            reason=state.circuit_breaker.message or "Circuit breaker is active",
            metadata={
                "enabled": True,
                "circuit_breaker_active": True,
                "reason": reason_value,
                "triggered_at": state.circuit_breaker.triggered_at,
                "cooldown_until": state.circuit_breaker.cooldown_until,
                "manual_release_required": state.circuit_breaker.manual_release_required,
            },
        )

    def register_failure(
        self,
        state: RiskState,
        *,
        reason: CircuitBreakerReason = CircuitBreakerReason.EXECUTION_FAILURES,
        message: str | None = None,
        count_as_failure: bool = True,
    ) -> bool:
        """
        Register a failure and trigger circuit breaker if thresholds are breached.

        Returns True if this call triggered the breaker.
        """
        if not self._config.enabled:
            return False

        if count_as_failure:
            self._stats.register_failure(reason)

        self._logger.warning(
            "Circuit breaker failure registered | reason=%s consecutive=%s execution=%s data=%s system=%s cost_spikes=%s",
            reason.value,
            self._stats.consecutive_failures,
            self._stats.execution_failures,
            self._stats.data_failures,
            self._stats.system_failures,
            self._stats.cost_spikes,
        )

        if not self._should_trigger(reason):
            return False

        self.trigger(
            state,
            reason=reason,
            message=message or self._default_trigger_message(reason),
            manual_release_required=self._requires_manual_release(reason),
        )
        return True

    def register_success(self) -> None:
        """
        Reset only consecutive failures.

        Historical counters are kept for diagnostics.
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
    ) -> None:
        if not self._config.enabled:
            return

        manual_release = (
            self._requires_manual_release(reason)
            if manual_release_required is None
            else manual_release_required
        )

        cooldown_until = None
        effective_cooldown = (
            self._config.cooldown_seconds
            if cooldown_seconds is None
            else cooldown_seconds
        )

        if effective_cooldown > 0 and not manual_release:
            cooldown_until = time.time() + effective_cooldown

        state.activate_circuit_breaker(
            reason,
            cooldown_until=cooldown_until,
            message=message or self._default_trigger_message(reason),
            manual_release_required=manual_release,
        )

        if reason is CircuitBreakerReason.EMERGENCY_STOP:
            state.emergency_stop(message or "Emergency stop triggered")

        self._stats.register_trigger(reason)

        self._logger.error(
            "Circuit breaker triggered | reason=%s cooldown_until=%s manual_release=%s",
            reason.value,
            cooldown_until,
            manual_release,
        )

    def trigger_emergency_stop(
        self,
        state: RiskState,
        *,
        message: str | None = None,
    ) -> None:
        """
        Trigger non-auto-releasing emergency stop.
        """
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
        """
        Trigger manual halt which requires explicit release.
        """
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

        now_ts = now_ts or time.time()
        if now_ts < cooldown_until:
            return False

        state.deactivate_circuit_breaker()
        self.reset_counters()

        self._logger.info("Circuit breaker released after cooldown")
        return True

    def force_release(
        self,
        state: RiskState,
        *,
        clear_emergency_stop: bool = False,
    ) -> bool:
        """
        Force release breaker.

        Intended for manual operator/recovery flow. RiskManager should decide
        who is allowed to call this.
        """
        if not state.circuit_breaker.active:
            return False

        state.deactivate_circuit_breaker(force=True)

        if clear_emergency_stop:
            state.clear_emergency_stop()

        self.reset_counters()

        self._logger.warning(
            "Circuit breaker force released | clear_emergency_stop=%s",
            clear_emergency_stop,
        )
        return True

    def reset_counters(self) -> None:
        self._stats.reset()

    def stats(self) -> dict[str, Any]:
        return self._stats.snapshot(enabled=self._config.enabled)

    def _should_trigger(self, reason: CircuitBreakerReason) -> bool:
        if reason is CircuitBreakerReason.MANUAL_HALT:
            return True

        if reason is CircuitBreakerReason.EMERGENCY_STOP:
            return self._config.trigger_on_emergency_stop

        if reason in {
            CircuitBreakerReason.DATA_FEED_FAILURE,
            CircuitBreakerReason.DATA_STALE,
        }:
            return self._config.trigger_on_data_feed_failure

        if reason in {
            CircuitBreakerReason.EXECUTION_COST_SPIKE,
            CircuitBreakerReason.SLIPPAGE_SPIKE,
            CircuitBreakerReason.SPREAD_ABNORMAL,
        }:
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


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerStats",
]