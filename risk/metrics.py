from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from risk.models import RiskDecision
from risk.state import RiskState


@dataclass(slots=True)
class RiskMetrics:
    approvals: int = 0
    rejections: int = 0
    size_adjustments: int = 0
    halts: int = 0
    force_close_requests: int = 0
    circuit_breaker_triggers: int = 0

    last_decision: str | None = None
    last_reason: str | None = None

    decision_counts: dict[str, int] = field(default_factory=dict)
    violation_counts: dict[str, int] = field(default_factory=dict)

    def register_decision(self, decision: RiskDecision) -> None:
        decision_name = decision.decision.value
        self.last_decision = decision_name
        self.last_reason = decision.reason

        self.decision_counts[decision_name] = self.decision_counts.get(decision_name, 0) + 1

        if decision.allowed:
            self.approvals += 1
        else:
            self.rejections += 1

        if decision.decision.value == "reduce_size":
            self.size_adjustments += 1
        elif decision.decision.value == "halt_trading":
            self.halts += 1
        elif decision.decision.value == "force_close":
            self.force_close_requests += 1

        for violation in decision.violations:
            key = violation.violation_type.value
            self.violation_counts[key] = self.violation_counts.get(key, 0) + 1

    def register_circuit_breaker_trigger(self) -> None:
        self.circuit_breaker_triggers += 1

    def snapshot(self, state: RiskState | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "approvals": self.approvals,
            "rejections": self.rejections,
            "size_adjustments": self.size_adjustments,
            "halts": self.halts,
            "force_close_requests": self.force_close_requests,
            "circuit_breaker_triggers": self.circuit_breaker_triggers,
            "last_decision": self.last_decision,
            "last_reason": self.last_reason,
            "decision_counts": dict(self.decision_counts),
            "violation_counts": dict(self.violation_counts),
        }

        if state is not None:
            payload["state"] = {
                "equity": state.equity,
                "free_balance": state.free_balance,
                "positions_count": len(state.positions),
                "trading_mode": state.trading_mode.value,
                "trading_halted": state.trading_halted,
                "daily_pnl": state.get_daily_pnl(),
                "drawdown_pct": state.get_drawdown_snapshot().drawdown_percent,
                "circuit_breaker_active": state.is_circuit_breaker_active(),
            }

        return payload