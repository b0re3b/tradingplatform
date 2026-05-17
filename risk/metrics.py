from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from risk.enums import (
    RiskDecisionType,
    TradeTier,
)
from risk.models import RiskDecision, RiskViolation
from risk.utils import safe_div


def _violation_type_value(violation: RiskViolation) -> str:
    """Return a stable string value for a violation type."""
    violation_type = getattr(violation, "violation_type", None)
    value = getattr(violation_type, "value", violation_type)
    return str(value or "")


def _decision_has_tier_downgrade(decision: RiskDecision) -> bool:
    """Detect actual tier downgrades even when final decision has another priority."""
    if decision.decision is RiskDecisionType.DOWNGRADE_TIER:
        return True

    for violation in decision.violations:
        value = _violation_type_value(violation)
        if value in {"tier_downgraded", "trade_tier_downgraded"}:
            return True
        if "tier" in value and "downgrad" in value:
            return True

    return False


def _decision_has_risk_reduction(decision: RiskDecision) -> bool:
    """Detect risk reductions independently from the final decision priority."""
    if decision.decision is RiskDecisionType.REDUCE_RISK:
        return True

    explicit_values = {
        "risk_reduced",
        "risk_amount_reduced",
        "risk_unit_reduced",
        "risk_budget_capped",
        "daily_budget_capped",
        "open_risk_capped",
        "strategy_budget_capped",
        "symbol_budget_capped",
    }
    for violation in decision.violations:
        value = _violation_type_value(violation)
        if value in explicit_values:
            return True

    return False


@dataclass(slots=True)
class MetricStats:
    """
    Generic numeric accumulator.

    Used for RR, EV, cost-to-reward, latency and other numeric metrics.
    Ignores None, NaN and infinite values to avoid poisoning long-lived
    runtime metrics.
    """

    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    last: float | None = None

    def add(self, value: float | None) -> None:
        if value is None:
            return

        if not math.isfinite(value):
            return

        self.count += 1
        self.total += value
        self.last = value

        if self.minimum is None or value < self.minimum:
            self.minimum = value

        if self.maximum is None or value > self.maximum:
            self.maximum = value

    @property
    def average(self) -> float | None:
        if self.count <= 0:
            return None
        return self.total / self.count

    def snapshot(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "total": self.total,
            "min": self.minimum,
            "max": self.maximum,
            "last": self.last,
            "avg": self.average,
        }


@dataclass(slots=True)
class ReservationMetrics:
    """
    Runtime metrics for pending risk reservations.

    Reservations represent risk approved by RiskManager but not yet confirmed
    by a real opened position. They are critical for detecting execution stalls,
    stuck pending orders and hidden projected exposure.
    """

    created: int = 0
    confirmed: int = 0
    released: int = 0
    expired: int = 0
    failed: int = 0

    active: int = 0
    peak_active: int = 0

    reserved_open_risk: float = 0.0
    reserved_margin: float = 0.0
    reserved_notional: float = 0.0

    confirmed_open_risk: float = 0.0
    released_open_risk: float = 0.0
    expired_open_risk: float = 0.0
    failed_open_risk: float = 0.0

    reservation_age_ms: MetricStats = field(default_factory=MetricStats)

    last_reservation_id: str | None = None
    last_event: str | None = None
    last_reason: str | None = None
    updated_at: float = field(default_factory=time.time)

    @property
    def completion_rate(self) -> float:
        return safe_div(self.confirmed, self.created)

    @property
    def expiration_rate(self) -> float:
        return safe_div(self.expired, self.created)

    @property
    def release_rate(self) -> float:
        return safe_div(self.released, self.created)

    @property
    def failure_rate(self) -> float:
        return safe_div(self.failed, self.created)

    def register_created(
        self,
        *,
        reservation_id: str | None = None,
        open_risk: float = 0.0,
        margin: float = 0.0,
        notional: float = 0.0,
    ) -> None:
        self.created += 1
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)

        self.reserved_open_risk += max(0.0, open_risk)
        self.reserved_margin += max(0.0, margin)
        self.reserved_notional += max(0.0, notional)

        self.last_reservation_id = reservation_id
        self.last_event = "created"
        self.last_reason = None
        self.updated_at = time.time()

    def register_confirmed(
        self,
        *,
        reservation_id: str | None = None,
        open_risk: float = 0.0,
        margin: float = 0.0,
        notional: float = 0.0,
        age_ms: float | None = None,
    ) -> None:
        self.confirmed += 1
        self._decrement_active()
        self._release_reserved_amounts(open_risk=open_risk, margin=margin, notional=notional)
        self.confirmed_open_risk += max(0.0, open_risk)
        self.reservation_age_ms.add(age_ms)

        self.last_reservation_id = reservation_id
        self.last_event = "confirmed"
        self.last_reason = None
        self.updated_at = time.time()

    def register_released(
        self,
        *,
        reservation_id: str | None = None,
        open_risk: float = 0.0,
        margin: float = 0.0,
        notional: float = 0.0,
        reason: str | None = None,
        age_ms: float | None = None,
    ) -> None:
        self.released += 1
        self._decrement_active()
        self._release_reserved_amounts(open_risk=open_risk, margin=margin, notional=notional)
        self.released_open_risk += max(0.0, open_risk)
        self.reservation_age_ms.add(age_ms)

        self.last_reservation_id = reservation_id
        self.last_event = "released"
        self.last_reason = reason
        self.updated_at = time.time()

    def register_expired(
        self,
        *,
        reservation_id: str | None = None,
        open_risk: float = 0.0,
        margin: float = 0.0,
        notional: float = 0.0,
        age_ms: float | None = None,
    ) -> None:
        self.expired += 1
        self._decrement_active()
        self._release_reserved_amounts(open_risk=open_risk, margin=margin, notional=notional)
        self.expired_open_risk += max(0.0, open_risk)
        self.reservation_age_ms.add(age_ms)

        self.last_reservation_id = reservation_id
        self.last_event = "expired"
        self.last_reason = "ttl_expired"
        self.updated_at = time.time()

    def register_failed(
        self,
        *,
        reservation_id: str | None = None,
        open_risk: float = 0.0,
        margin: float = 0.0,
        notional: float = 0.0,
        reason: str | None = None,
        age_ms: float | None = None,
    ) -> None:
        self.failed += 1
        self._decrement_active()
        self._release_reserved_amounts(open_risk=open_risk, margin=margin, notional=notional)
        self.failed_open_risk += max(0.0, open_risk)
        self.reservation_age_ms.add(age_ms)

        self.last_reservation_id = reservation_id
        self.last_event = "failed"
        self.last_reason = reason
        self.updated_at = time.time()

    def set_active_snapshot(
        self,
        *,
        active: int | None = None,
        reserved_open_risk: float | None = None,
        reserved_margin: float | None = None,
        reserved_notional: float | None = None,
    ) -> None:
        """
        Reconcile counters with RiskState snapshot.

        Useful after startup restore, manual cleanup or batch expiration where
        RiskManager has authoritative state and metrics must follow it.
        """
        if active is not None:
            self.active = max(0, active)
            self.peak_active = max(self.peak_active, self.active)
        if reserved_open_risk is not None:
            self.reserved_open_risk = max(0.0, reserved_open_risk)
        if reserved_margin is not None:
            self.reserved_margin = max(0.0, reserved_margin)
        if reserved_notional is not None:
            self.reserved_notional = max(0.0, reserved_notional)
        self.updated_at = time.time()

    def snapshot(self) -> dict[str, Any]:
        return {
            "created": self.created,
            "confirmed": self.confirmed,
            "released": self.released,
            "expired": self.expired,
            "failed": self.failed,
            "active": self.active,
            "peak_active": self.peak_active,
            "completion_rate": self.completion_rate,
            "expiration_rate": self.expiration_rate,
            "release_rate": self.release_rate,
            "failure_rate": self.failure_rate,
            "reserved_open_risk": self.reserved_open_risk,
            "reserved_margin": self.reserved_margin,
            "reserved_notional": self.reserved_notional,
            "confirmed_open_risk": self.confirmed_open_risk,
            "released_open_risk": self.released_open_risk,
            "expired_open_risk": self.expired_open_risk,
            "failed_open_risk": self.failed_open_risk,
            "reservation_age_ms": self.reservation_age_ms.snapshot(),
            "last_reservation_id": self.last_reservation_id,
            "last_event": self.last_event,
            "last_reason": self.last_reason,
            "updated_at": self.updated_at,
        }

    def _decrement_active(self) -> None:
        self.active = max(0, self.active - 1)

    def _release_reserved_amounts(
        self,
        *,
        open_risk: float = 0.0,
        margin: float = 0.0,
        notional: float = 0.0,
    ) -> None:
        self.reserved_open_risk = max(0.0, self.reserved_open_risk - max(0.0, open_risk))
        self.reserved_margin = max(0.0, self.reserved_margin - max(0.0, margin))
        self.reserved_notional = max(0.0, self.reserved_notional - max(0.0, notional))


@dataclass(slots=True)
class GroupMetrics:
    """
    Metrics grouped by tier, symbol or strategy.
    """

    decisions: int = 0
    approvals: int = 0
    rejections: int = 0
    size_adjustments: int = 0
    tier_downgrades: int = 0
    risk_reductions: int = 0
    halts: int = 0
    emergency_stops: int = 0

    realized_pnl: float = 0.0
    open_risk: float = 0.0
    pending_open_risk: float = 0.0
    pending_reservations: int = 0

    rr: MetricStats = field(default_factory=MetricStats)
    expected_value: MetricStats = field(default_factory=MetricStats)
    expected_value_after_cost: MetricStats = field(default_factory=MetricStats)
    cost_to_reward: MetricStats = field(default_factory=MetricStats)

    violation_counts: dict[str, int] = field(default_factory=dict)

    last_decision: str | None = None
    last_reason: str | None = None
    updated_at: float = field(default_factory=time.time)

    def register_decision(self, decision: RiskDecision) -> None:
        self.decisions += 1
        self.last_decision = decision.decision.value
        self.last_reason = decision.reason

        if decision.allowed:
            self.approvals += 1
        else:
            self.rejections += 1

        if decision.decision is RiskDecisionType.REDUCE_SIZE:
            self.size_adjustments += 1
        if _decision_has_tier_downgrade(decision):
            self.tier_downgrades += 1
        if _decision_has_risk_reduction(decision):
            self.risk_reductions += 1
        if decision.decision is RiskDecisionType.HALT_TRADING:
            self.halts += 1
        if decision.decision is RiskDecisionType.EMERGENCY_STOP:
            self.emergency_stops += 1

        self.rr.add(decision.risk_reward_ratio)
        self.expected_value.add(decision.expected_value)
        self.expected_value_after_cost.add(decision.expected_value_after_cost)
        self.cost_to_reward.add(decision.cost_to_reward_ratio)

        for violation in decision.violations:
            self.register_violation(violation)

        self.updated_at = time.time()

    def register_violation(self, violation: RiskViolation) -> None:
        key = violation.violation_type.value
        self.violation_counts[key] = self.violation_counts.get(key, 0) + 1
        self.updated_at = time.time()

    def register_closed_pnl(self, pnl: float, *, released_risk: float = 0.0) -> None:
        self.realized_pnl += pnl
        self.open_risk = max(0.0, self.open_risk - max(0.0, released_risk))
        self.updated_at = time.time()

    def register_open_risk(self, open_risk: float) -> None:
        self.open_risk += max(0.0, open_risk)
        self.updated_at = time.time()

    def register_pending_reservation(self, *, open_risk: float = 0.0) -> None:
        self.pending_reservations += 1
        self.pending_open_risk += max(0.0, open_risk)
        self.updated_at = time.time()

    def release_pending_reservation(self, *, open_risk: float = 0.0) -> None:
        self.pending_reservations = max(0, self.pending_reservations - 1)
        self.pending_open_risk = max(0.0, self.pending_open_risk - max(0.0, open_risk))
        self.updated_at = time.time()

    def snapshot(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "approvals": self.approvals,
            "rejections": self.rejections,
            "approval_rate": safe_div(self.approvals, self.decisions),
            "rejection_rate": safe_div(self.rejections, self.decisions),
            "size_adjustments": self.size_adjustments,
            "tier_downgrades": self.tier_downgrades,
            "risk_reductions": self.risk_reductions,
            "halts": self.halts,
            "emergency_stops": self.emergency_stops,
            "realized_pnl": self.realized_pnl,
            "open_risk": self.open_risk,
            "pending_open_risk": self.pending_open_risk,
            "projected_open_risk": self.open_risk + self.pending_open_risk,
            "pending_reservations": self.pending_reservations,
            "rr": self.rr.snapshot(),
            "expected_value": self.expected_value.snapshot(),
            "expected_value_after_cost": self.expected_value_after_cost.snapshot(),
            "cost_to_reward": self.cost_to_reward.snapshot(),
            "violation_counts": dict(self.violation_counts),
            "last_decision": self.last_decision,
            "last_reason": self.last_reason,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class RiskMetrics:
    """
    Runtime metrics for risk layer.

    This class intentionally has no EventBus, Scheduler or logger dependency.
    RiskManager owns lifecycle, locking and event publishing.
    """

    approvals: int = 0
    rejections: int = 0
    size_adjustments: int = 0
    tier_downgrades: int = 0
    risk_reductions: int = 0
    halts: int = 0
    emergency_stops: int = 0
    force_close_requests: int = 0
    only_reduce_decisions: int = 0

    circuit_breaker_triggers: int = 0
    manual_review_triggers: int = 0

    strategy_disables: int = 0
    strategy_reductions: int = 0
    strategy_cooldowns: int = 0

    symbol_disables: int = 0
    symbol_reductions: int = 0
    symbol_cooldowns: int = 0

    total_decisions: int = 0

    last_decision: str | None = None
    last_reason: str | None = None
    last_symbol: str | None = None
    last_strategy_name: str | None = None
    last_tier: str | None = None
    last_reservation_id: str | None = None

    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    decision_counts: dict[str, int] = field(default_factory=dict)
    violation_counts: dict[str, int] = field(default_factory=dict)

    decisions_by_tier: dict[str, GroupMetrics] = field(default_factory=dict)
    decisions_by_symbol: dict[str, GroupMetrics] = field(default_factory=dict)
    decisions_by_strategy: dict[str, GroupMetrics] = field(default_factory=dict)

    rr: MetricStats = field(default_factory=MetricStats)
    expected_value: MetricStats = field(default_factory=MetricStats)
    expected_value_after_cost: MetricStats = field(default_factory=MetricStats)
    cost_to_reward: MetricStats = field(default_factory=MetricStats)
    decision_latency_ms: MetricStats = field(default_factory=MetricStats)

    reservations: ReservationMetrics = field(default_factory=ReservationMetrics)

    def register_decision(
        self,
        decision: RiskDecision,
        *,
        latency_ms: float | None = None,
    ) -> None:
        self.total_decisions += 1

        decision_name = decision.decision.value
        self.last_decision = decision_name
        self.last_reason = decision.reason
        self.last_symbol = decision.symbol
        self.last_strategy_name = decision.strategy_name
        self.last_tier = decision.final_tier.value if decision.final_tier else None
        self.last_reservation_id = getattr(decision, "reservation_id", None)

        self.decision_counts[decision_name] = self.decision_counts.get(decision_name, 0) + 1

        if decision.allowed:
            self.approvals += 1
        else:
            self.rejections += 1

        if decision.decision is RiskDecisionType.REDUCE_SIZE:
            self.size_adjustments += 1
        if _decision_has_tier_downgrade(decision):
            self.tier_downgrades += 1
        if _decision_has_risk_reduction(decision):
            self.risk_reductions += 1
        if decision.decision is RiskDecisionType.HALT_TRADING:
            self.halts += 1
        if decision.decision is RiskDecisionType.EMERGENCY_STOP:
            self.emergency_stops += 1
        if decision.decision is RiskDecisionType.FORCE_CLOSE:
            self.force_close_requests += 1
        if decision.decision is RiskDecisionType.ONLY_REDUCE:
            self.only_reduce_decisions += 1

        self.rr.add(decision.risk_reward_ratio)
        self.expected_value.add(decision.expected_value)
        self.expected_value_after_cost.add(decision.expected_value_after_cost)
        self.cost_to_reward.add(decision.cost_to_reward_ratio)
        self.decision_latency_ms.add(latency_ms)

        for violation in decision.violations:
            self.register_violation(violation)

        if decision.final_tier is not None:
            tier_metrics = self._get_tier_metrics(decision.final_tier)
            tier_metrics.register_decision(decision)

        if decision.symbol:
            symbol_metrics = self._get_symbol_metrics(decision.symbol)
            symbol_metrics.register_decision(decision)

        if decision.strategy_name:
            strategy_metrics = self._get_strategy_metrics(decision.strategy_name)
            strategy_metrics.register_decision(decision)

        self.updated_at = time.time()

    def register_violation(self, violation: RiskViolation) -> None:
        key = violation.violation_type.value
        self.violation_counts[key] = self.violation_counts.get(key, 0) + 1

        if violation.tier is not None:
            self._get_tier_metrics(violation.tier).register_violation(violation)

        if violation.symbol:
            self._get_symbol_metrics(violation.symbol).register_violation(violation)

        if violation.strategy_name:
            self._get_strategy_metrics(violation.strategy_name).register_violation(violation)

        self.updated_at = time.time()

    def register_reservation_created(
        self,
        *,
        reservation_id: str | None = None,
        symbol: str | None = None,
        tier: TradeTier | None = None,
        strategy_name: str | None = None,
        open_risk: float = 0.0,
        margin: float = 0.0,
        notional: float = 0.0,
    ) -> None:
        self.reservations.register_created(
            reservation_id=reservation_id,
            open_risk=open_risk,
            margin=margin,
            notional=notional,
        )
        self._register_group_pending(
            symbol=symbol,
            tier=tier,
            strategy_name=strategy_name,
            open_risk=open_risk,
        )
        self.last_reservation_id = reservation_id
        self.updated_at = time.time()

    def register_reservation_confirmed(
        self,
        *,
        reservation_id: str | None = None,
        symbol: str | None = None,
        tier: TradeTier | None = None,
        strategy_name: str | None = None,
        open_risk: float = 0.0,
        margin: float = 0.0,
        notional: float = 0.0,
        age_ms: float | None = None,
    ) -> None:
        self.reservations.register_confirmed(
            reservation_id=reservation_id,
            open_risk=open_risk,
            margin=margin,
            notional=notional,
            age_ms=age_ms,
        )
        self._release_group_pending(
            symbol=symbol,
            tier=tier,
            strategy_name=strategy_name,
            open_risk=open_risk,
        )
        self.last_reservation_id = reservation_id
        self.updated_at = time.time()

    def register_reservation_released(
        self,
        *,
        reservation_id: str | None = None,
        symbol: str | None = None,
        tier: TradeTier | None = None,
        strategy_name: str | None = None,
        open_risk: float = 0.0,
        margin: float = 0.0,
        notional: float = 0.0,
        reason: str | None = None,
        age_ms: float | None = None,
    ) -> None:
        self.reservations.register_released(
            reservation_id=reservation_id,
            open_risk=open_risk,
            margin=margin,
            notional=notional,
            reason=reason,
            age_ms=age_ms,
        )
        self._release_group_pending(
            symbol=symbol,
            tier=tier,
            strategy_name=strategy_name,
            open_risk=open_risk,
        )
        self.last_reservation_id = reservation_id
        self.updated_at = time.time()

    def register_reservation_expired(
        self,
        *,
        reservation_id: str | None = None,
        symbol: str | None = None,
        tier: TradeTier | None = None,
        strategy_name: str | None = None,
        open_risk: float = 0.0,
        margin: float = 0.0,
        notional: float = 0.0,
        age_ms: float | None = None,
    ) -> None:
        self.reservations.register_expired(
            reservation_id=reservation_id,
            open_risk=open_risk,
            margin=margin,
            notional=notional,
            age_ms=age_ms,
        )
        self._release_group_pending(
            symbol=symbol,
            tier=tier,
            strategy_name=strategy_name,
            open_risk=open_risk,
        )
        self.last_reservation_id = reservation_id
        self.updated_at = time.time()

    def register_reservation_failed(
        self,
        *,
        reservation_id: str | None = None,
        symbol: str | None = None,
        tier: TradeTier | None = None,
        strategy_name: str | None = None,
        open_risk: float = 0.0,
        margin: float = 0.0,
        notional: float = 0.0,
        reason: str | None = None,
        age_ms: float | None = None,
    ) -> None:
        self.reservations.register_failed(
            reservation_id=reservation_id,
            open_risk=open_risk,
            margin=margin,
            notional=notional,
            reason=reason,
            age_ms=age_ms,
        )
        self._release_group_pending(
            symbol=symbol,
            tier=tier,
            strategy_name=strategy_name,
            open_risk=open_risk,
        )
        self.last_reservation_id = reservation_id
        self.updated_at = time.time()

    def reconcile_reservations(
        self,
        *,
        active: int | None = None,
        reserved_open_risk: float | None = None,
        reserved_margin: float | None = None,
        reserved_notional: float | None = None,
    ) -> None:
        self.reservations.set_active_snapshot(
            active=active,
            reserved_open_risk=reserved_open_risk,
            reserved_margin=reserved_margin,
            reserved_notional=reserved_notional,
        )
        self.updated_at = time.time()

    def register_circuit_breaker_trigger(self) -> None:
        self.circuit_breaker_triggers += 1
        self.updated_at = time.time()

    def register_manual_review_trigger(self) -> None:
        self.manual_review_triggers += 1
        self.updated_at = time.time()

    def register_strategy_disabled(self, strategy_name: str) -> None:
        self.strategy_disables += 1
        self._get_strategy_metrics(strategy_name).last_reason = "strategy_disabled"
        self.updated_at = time.time()

    def register_strategy_reduced(self, strategy_name: str) -> None:
        self.strategy_reductions += 1
        self._get_strategy_metrics(strategy_name).last_reason = "strategy_reduced"
        self.updated_at = time.time()

    def register_strategy_cooldown(self, strategy_name: str) -> None:
        self.strategy_cooldowns += 1
        self._get_strategy_metrics(strategy_name).last_reason = "strategy_cooldown"
        self.updated_at = time.time()

    def register_symbol_disabled(self, symbol: str) -> None:
        self.symbol_disables += 1
        self._get_symbol_metrics(symbol).last_reason = "symbol_disabled"
        self.updated_at = time.time()

    def register_symbol_reduced(self, symbol: str) -> None:
        self.symbol_reductions += 1
        self._get_symbol_metrics(symbol).last_reason = "symbol_reduced"
        self.updated_at = time.time()

    def register_symbol_cooldown(self, symbol: str) -> None:
        self.symbol_cooldowns += 1
        self._get_symbol_metrics(symbol).last_reason = "symbol_cooldown"
        self.updated_at = time.time()

    def register_position_opened(
        self,
        *,
        symbol: str,
        tier: TradeTier | None = None,
        strategy_name: str | None = None,
        open_risk: float = 0.0,
    ) -> None:
        if tier is not None:
            self._get_tier_metrics(tier).register_open_risk(open_risk)

        self._get_symbol_metrics(symbol).register_open_risk(open_risk)

        if strategy_name:
            self._get_strategy_metrics(strategy_name).register_open_risk(open_risk)

        self.updated_at = time.time()

    def register_position_closed(
        self,
        *,
        symbol: str,
        realized_pnl: float,
        released_risk: float = 0.0,
        tier: TradeTier | None = None,
        strategy_name: str | None = None,
    ) -> None:
        if tier is not None:
            self._get_tier_metrics(tier).register_closed_pnl(
                realized_pnl,
                released_risk=released_risk,
            )

        self._get_symbol_metrics(symbol).register_closed_pnl(
            realized_pnl,
            released_risk=released_risk,
        )

        if strategy_name:
            self._get_strategy_metrics(strategy_name).register_closed_pnl(
                realized_pnl,
                released_risk=released_risk,
            )

        self.updated_at = time.time()

    def snapshot(self, state_snapshot: Any | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "uptime_seconds": max(0.0, time.time() - self.started_at),
            "total_decisions": self.total_decisions,
            "approvals": self.approvals,
            "rejections": self.rejections,
            "approval_rate": safe_div(self.approvals, self.total_decisions),
            "rejection_rate": safe_div(self.rejections, self.total_decisions),
            "size_adjustments": self.size_adjustments,
            "tier_downgrades": self.tier_downgrades,
            "risk_reductions": self.risk_reductions,
            "halts": self.halts,
            "emergency_stops": self.emergency_stops,
            "force_close_requests": self.force_close_requests,
            "only_reduce_decisions": self.only_reduce_decisions,
            "circuit_breaker_triggers": self.circuit_breaker_triggers,
            "manual_review_triggers": self.manual_review_triggers,
            "strategy_disables": self.strategy_disables,
            "strategy_reductions": self.strategy_reductions,
            "strategy_cooldowns": self.strategy_cooldowns,
            "symbol_disables": self.symbol_disables,
            "symbol_reductions": self.symbol_reductions,
            "symbol_cooldowns": self.symbol_cooldowns,
            "last_decision": self.last_decision,
            "last_reason": self.last_reason,
            "last_symbol": self.last_symbol,
            "last_strategy_name": self.last_strategy_name,
            "last_tier": self.last_tier,
            "last_reservation_id": self.last_reservation_id,
            "decision_counts": dict(self.decision_counts),
            "violation_counts": dict(self.violation_counts),
            "rr": self.rr.snapshot(),
            "expected_value": self.expected_value.snapshot(),
            "expected_value_after_cost": self.expected_value_after_cost.snapshot(),
            "cost_to_reward": self.cost_to_reward.snapshot(),
            "decision_latency_ms": self.decision_latency_ms.snapshot(),
            "reservations": self.reservations.snapshot(),
            "by_tier": {
                tier: metrics.snapshot()
                for tier, metrics in self.decisions_by_tier.items()
            },
            "by_symbol": {
                symbol: metrics.snapshot()
                for symbol, metrics in self.decisions_by_symbol.items()
            },
            "by_strategy": {
                strategy_name: metrics.snapshot()
                for strategy_name, metrics in self.decisions_by_strategy.items()
            },
        }

        if state_snapshot is not None:
            payload["state"] = state_snapshot

        return payload

    def reset(self) -> None:
        self.approvals = 0
        self.rejections = 0
        self.size_adjustments = 0
        self.tier_downgrades = 0
        self.risk_reductions = 0
        self.halts = 0
        self.emergency_stops = 0
        self.force_close_requests = 0
        self.only_reduce_decisions = 0

        self.circuit_breaker_triggers = 0
        self.manual_review_triggers = 0

        self.strategy_disables = 0
        self.strategy_reductions = 0
        self.strategy_cooldowns = 0

        self.symbol_disables = 0
        self.symbol_reductions = 0
        self.symbol_cooldowns = 0

        self.total_decisions = 0

        self.last_decision = None
        self.last_reason = None
        self.last_symbol = None
        self.last_strategy_name = None
        self.last_tier = None
        self.last_reservation_id = None

        self.decision_counts.clear()
        self.violation_counts.clear()
        self.decisions_by_tier.clear()
        self.decisions_by_symbol.clear()
        self.decisions_by_strategy.clear()

        self.rr = MetricStats()
        self.expected_value = MetricStats()
        self.expected_value_after_cost = MetricStats()
        self.cost_to_reward = MetricStats()
        self.decision_latency_ms = MetricStats()
        self.reservations = ReservationMetrics()

        self.started_at = time.time()
        self.updated_at = self.started_at

    def _register_group_pending(
        self,
        *,
        symbol: str | None,
        tier: TradeTier | None,
        strategy_name: str | None,
        open_risk: float,
    ) -> None:
        if tier is not None:
            self._get_tier_metrics(tier).register_pending_reservation(open_risk=open_risk)
        if symbol:
            self._get_symbol_metrics(symbol).register_pending_reservation(open_risk=open_risk)
        if strategy_name:
            self._get_strategy_metrics(strategy_name).register_pending_reservation(open_risk=open_risk)

    def _release_group_pending(
        self,
        *,
        symbol: str | None,
        tier: TradeTier | None,
        strategy_name: str | None,
        open_risk: float,
    ) -> None:
        if tier is not None:
            self._get_tier_metrics(tier).release_pending_reservation(open_risk=open_risk)
        if symbol:
            self._get_symbol_metrics(symbol).release_pending_reservation(open_risk=open_risk)
        if strategy_name:
            self._get_strategy_metrics(strategy_name).release_pending_reservation(open_risk=open_risk)

    def _get_tier_metrics(self, tier: TradeTier) -> GroupMetrics:
        key = tier.value
        metrics = self.decisions_by_tier.get(key)
        if metrics is None:
            metrics = GroupMetrics()
            self.decisions_by_tier[key] = metrics
        return metrics

    def _get_symbol_metrics(self, symbol: str) -> GroupMetrics:
        metrics = self.decisions_by_symbol.get(symbol)
        if metrics is None:
            metrics = GroupMetrics()
            self.decisions_by_symbol[symbol] = metrics
        return metrics

    def _get_strategy_metrics(self, strategy_name: str) -> GroupMetrics:
        metrics = self.decisions_by_strategy.get(strategy_name)
        if metrics is None:
            metrics = GroupMetrics()
            self.decisions_by_strategy[strategy_name] = metrics
        return metrics


__all__ = [
    "GroupMetrics",
    "MetricStats",
    "ReservationMetrics",
    "RiskMetrics",
]