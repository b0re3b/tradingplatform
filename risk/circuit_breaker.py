from __future__ import annotations

import time

from core.logger import get_logger

from risk.config import CircuitBreakerConfig
from risk.enums import CircuitBreakerReason, RiskDecisionType, RiskLevel, RiskViolationType
from risk.models import RiskCheckResult, RiskViolation
from risk.state import RiskState


class CircuitBreaker:
    """
    Аварійний верхньорівневий блокер.
    Якщо активний, нові трейди блокуються одразу.
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

        self._consecutive_failures = 0
        self._execution_failures = 0

    def check(self, state: RiskState) -> RiskCheckResult:
        if not self._config.enabled:
            return RiskCheckResult(
                passed=True,
                decision=RiskDecisionType.ALLOW,
                metadata={"enabled": False},
            )

        if state.is_circuit_breaker_active():
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.HALT_TRADING,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.CIRCUIT_BREAKER_TRIGGERED,
                        level=RiskLevel.CRITICAL,
                        message=state.circuit_breaker.message or "Circuit breaker is active",
                        metadata={
                            "reason": (
                                state.circuit_breaker.reason.value
                                if state.circuit_breaker.reason is not None
                                else None
                            ),
                            "cooldown_until": state.circuit_breaker.cooldown_until,
                        },
                    )
                ],
                metadata={
                    "circuit_breaker_active": True,
                    "reason": (
                        state.circuit_breaker.reason.value
                        if state.circuit_breaker.reason is not None
                        else None
                    ),
                    "cooldown_until": state.circuit_breaker.cooldown_until,
                },
            )

        return RiskCheckResult(
            passed=True,
            decision=RiskDecisionType.ALLOW,
            metadata={
                "circuit_breaker_active": False,
            },
        )

    def register_failure(
        self,
        state: RiskState,
        *,
        reason: CircuitBreakerReason = CircuitBreakerReason.EXECUTION_FAILURES,
        message: str | None = None,
        count_as_execution_failure: bool = True,
    ) -> None:
        if not self._config.enabled:
            return

        self._consecutive_failures += 1
        if count_as_execution_failure:
            self._execution_failures += 1

        self._logger.warning(
            "Circuit breaker failure registered | consecutive=%s execution_failures=%s",
            self._consecutive_failures,
            self._execution_failures,
        )

        should_trigger = (
            self._consecutive_failures >= self._config.max_consecutive_failures
            or self._execution_failures >= self._config.max_execution_failures
        )

        if should_trigger:
            self.trigger(
                state,
                reason=reason,
                message=message or "Circuit breaker triggered by failure thresholds",
            )

    def register_success(self) -> None:
        self._consecutive_failures = 0

    def trigger(
        self,
        state: RiskState,
        *,
        reason: CircuitBreakerReason,
        message: str | None = None,
    ) -> None:
        if not self._config.enabled:
            return

        cooldown_until = None
        if self._config.cooldown_seconds > 0:
            cooldown_until = time.time() + self._config.cooldown_seconds

        state.activate_circuit_breaker(
            reason,
            cooldown_until=cooldown_until,
            message=message or f"Circuit breaker triggered: {reason.value}",
        )

        self._logger.error(
            "Circuit breaker triggered | reason=%s cooldown_until=%s",
            reason.value,
            cooldown_until,
        )

    def release_if_ready(self, state: RiskState, *, now_ts: float | None = None) -> bool:
        if not state.circuit_breaker.active:
            return False

        now_ts = now_ts or time.time()
        cooldown_until = state.circuit_breaker.cooldown_until

        if cooldown_until is None or now_ts < cooldown_until:
            return False

        state.deactivate_circuit_breaker()
        self.reset_counters()

        self._logger.info("Circuit breaker released after cooldown")
        return True

    def reset_counters(self) -> None:
        self._consecutive_failures = 0
        self._execution_failures = 0

    def stats(self) -> dict[str, int | bool]:
        return {
            "enabled": self._config.enabled,
            "consecutive_failures": self._consecutive_failures,
            "execution_failures": self._execution_failures,
        }