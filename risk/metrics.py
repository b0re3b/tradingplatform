from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from risk.enums import (
    RiskDecisionType,
    TradeTier,
)
from risk.models import RiskDecision, RiskViolation
from risk.utils import safe_div


@dataclass(slots=True)
class MetricStats:
    """
    Generic numeric accumulator.

    Used for RR, EV, cost-to-reward, latency and other numeric metrics.
    """

    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    last: float | None = None

    def add(self, value: float | None) -> None:
        if value is None:
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
        elif decision.decision is RiskDecisionType.DOWNGRADE_TIER:
            self.tier_downgrades += 1
        elif decision.decision is RiskDecisionType.REDUCE_RISK:
            self.risk_reductions += 1
        elif decision.decision is RiskDecisionType.HALT_TRADING:
            self.halts += 1
        elif decision.decision is RiskDecisionType.EMERGENCY_STOP:
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

        self.decision_counts[decision_name] = self.decision_counts.get(decision_name, 0) + 1

        if decision.allowed:
            self.approvals += 1
        else:
            self.rejections += 1

        if decision.decision is RiskDecisionType.REDUCE_SIZE:
            self.size_adjustments += 1
        elif decision.decision is RiskDecisionType.DOWNGRADE_TIER:
            self.tier_downgrades += 1
        elif decision.decision is RiskDecisionType.REDUCE_RISK:
            self.risk_reductions += 1
        elif decision.decision is RiskDecisionType.HALT_TRADING:
            self.halts += 1
        elif decision.decision is RiskDecisionType.EMERGENCY_STOP:
            self.emergency_stops += 1
        elif decision.decision is RiskDecisionType.FORCE_CLOSE:
            self.force_close_requests += 1
        elif decision.decision is RiskDecisionType.ONLY_REDUCE:
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
            "decision_counts": dict(self.decision_counts),
            "violation_counts": dict(self.violation_counts),
            "rr": self.rr.snapshot(),
            "expected_value": self.expected_value.snapshot(),
            "expected_value_after_cost": self.expected_value_after_cost.snapshot(),
            "cost_to_reward": self.cost_to_reward.snapshot(),
            "decision_latency_ms": self.decision_latency_ms.snapshot(),
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

        self.started_at = time.time()
        self.updated_at = self.started_at

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
    "RiskMetrics",
]