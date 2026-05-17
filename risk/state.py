from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from risk.enums import (
    CircuitBreakerReason,
    PositionSide,
    RiskMode,
    StrategyRiskStatus,
    SymbolRiskStatus,
    TradeTier,
    TradingMode,
)
from risk.models import (
    CircuitBreakerState,
    CorrelationSnapshot,
    DrawdownSnapshot,
    ExposureSnapshot,
    OpenRiskSnapshot,
    PortfolioPosition,
    RiskBudgetSnapshot,
    RiskStateSnapshot,
    StrategyRiskSnapshot,
    SymbolRiskSnapshot,
    TierStatsSnapshot,
)
from risk.utils import calculate_drawdown_pct, safe_div


@dataclass(slots=True)
class CooldownState:
    """
    Generic cooldown state for symbol/strategy throttling.
    """

    active: bool = False
    reason: str | None = None
    started_at: float | None = None
    cooldown_until: float | None = None

    def activate(
        self,
        *,
        cooldown_seconds: float,
        reason: str | None = None,
        now_ts: float | None = None,
    ) -> None:
        now_ts = now_ts or time.time()
        self.active = True
        self.reason = reason
        self.started_at = now_ts
        self.cooldown_until = now_ts + max(0.0, cooldown_seconds)

    def deactivate(self) -> None:
        self.active = False
        self.reason = None
        self.started_at = None
        self.cooldown_until = None

    def is_active(self, *, now_ts: float | None = None) -> bool:
        """
        Return whether cooldown is active without mutating state.

        Important: this method is intentionally side-effect free so snapshots,
        metrics and dashboard reads cannot accidentally deactivate cooldowns.
        Use expire_if_needed() or refresh_status() when mutation is desired.
        """
        if not self.active:
            return False

        if self.cooldown_until is None:
            return True

        now_ts = now_ts or time.time()
        return now_ts < self.cooldown_until

    def has_expired(self, *, now_ts: float | None = None) -> bool:
        """Return True when a finite cooldown is active but already expired."""
        if not self.active or self.cooldown_until is None:
            return False
        now_ts = now_ts or time.time()
        return now_ts >= self.cooldown_until

    def expire_if_needed(self, *, now_ts: float | None = None) -> bool:
        """
        Mutating counterpart of is_active().

        Returns True if the cooldown was active and got deactivated.
        """
        if not self.has_expired(now_ts=now_ts):
            return False
        self.deactivate()
        return True


@dataclass(slots=True)
class PendingRiskReservation:
    """
    Temporary risk reservation created after an ALLOW decision and before the
    exchange/execution layer confirms that a position was opened.

    RiskState owns only the reservation accounting. RiskManager is responsible
    for creating/releasing reservations under its async lock and for emitting
    EventBus events outside that lock.
    """

    reservation_id: str
    symbol: str
    side: PositionSide

    signal_id: str | None = None
    strategy_name: str | None = None
    tier: TradeTier | None = None
    position_id: str | None = None

    size: float = 0.0
    open_risk: float = 0.0
    margin: float = 0.0
    notional: float = 0.0

    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self, *, now_ts: float | None = None) -> bool:
        if self.expires_at is None:
            return False
        now_ts = now_ts or time.time()
        return now_ts >= self.expires_at

    def snapshot(self) -> dict[str, Any]:
        return {
            "reservation_id": self.reservation_id,
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "strategy_name": self.strategy_name,
            "tier": self.tier.value if self.tier else None,
            "position_id": self.position_id,
            "size": self.size,
            "open_risk": self.open_risk,
            "margin": self.margin,
            "notional": self.notional,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class SymbolRiskState:
    """
    Runtime state for one symbol.

    Tracks symbol-level PnL, open risk, trades count, losses and cooldowns.
    """

    symbol: str

    status: SymbolRiskStatus = SymbolRiskStatus.ACTIVE

    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0

    open_risk: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0

    cooldown: CooldownState = field(default_factory=CooldownState)

    disabled_until: float | None = None
    disabled_reason: str | None = None

    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def register_trade_opened(self, *, open_risk: float = 0.0) -> None:
        self.trades_today += 1
        self.open_risk = max(0.0, self.open_risk + max(0.0, open_risk))
        self.updated_at = time.time()

    def register_trade_closed(self, *, realized_pnl: float, released_risk: float = 0.0) -> None:
        self.daily_pnl += realized_pnl
        self.weekly_pnl += realized_pnl
        self.monthly_pnl += realized_pnl
        self.open_risk = max(0.0, self.open_risk - max(0.0, released_risk))

        if realized_pnl < 0:
            self.consecutive_losses += 1
        elif realized_pnl > 0:
            self.consecutive_losses = 0

        self.updated_at = time.time()

    def activate_cooldown(self, *, cooldown_seconds: float, reason: str | None = None) -> None:
        self.status = SymbolRiskStatus.COOLDOWN
        self.cooldown.activate(cooldown_seconds=cooldown_seconds, reason=reason)
        self.updated_at = time.time()

    def disable(self, *, reason: str | None = None, until: float | None = None) -> None:
        self.status = SymbolRiskStatus.DISABLED
        self.disabled_reason = reason
        self.disabled_until = until
        self.updated_at = time.time()

    def reduce(self, *, reason: str | None = None) -> None:
        self.status = SymbolRiskStatus.REDUCED
        self.metadata["reduced_reason"] = reason
        self.updated_at = time.time()

    def activate(self) -> None:
        self.status = SymbolRiskStatus.ACTIVE
        self.cooldown.deactivate()
        self.disabled_until = None
        self.disabled_reason = None
        self.updated_at = time.time()

    def refresh_status(self, *, now_ts: float | None = None) -> None:
        now_ts = now_ts or time.time()

        if self.status is SymbolRiskStatus.DISABLED and self.disabled_until is not None:
            if now_ts >= self.disabled_until:
                self.activate()

        if self.status is SymbolRiskStatus.COOLDOWN and self.cooldown.expire_if_needed(now_ts=now_ts):
            self.activate()

    def reset_daily(self) -> None:
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.consecutive_losses = 0
        self.updated_at = time.time()

    def reset_weekly(self) -> None:
        self.weekly_pnl = 0.0
        self.updated_at = time.time()

    def reset_monthly(self) -> None:
        self.monthly_pnl = 0.0
        self.updated_at = time.time()

    def snapshot(self, *, risk_unit: float) -> SymbolRiskSnapshot:
        now_ts = time.time()
        status = self.status
        cooldown_active = self.cooldown.is_active(now_ts=now_ts)

        if status is SymbolRiskStatus.DISABLED and self.disabled_until is not None:
            if now_ts >= self.disabled_until:
                status = SymbolRiskStatus.ACTIVE

        if status is SymbolRiskStatus.COOLDOWN and not cooldown_active:
            status = SymbolRiskStatus.ACTIVE

        return SymbolRiskSnapshot(
            symbol=self.symbol,
            status=status,
            daily_pnl=self.daily_pnl,
            daily_loss_r=safe_div(abs(min(0.0, self.daily_pnl)), risk_unit),
            open_risk=self.open_risk,
            open_risk_r=safe_div(self.open_risk, risk_unit),
            trades_today=self.trades_today,
            consecutive_losses=self.consecutive_losses,
            cooldown_until=self.cooldown.cooldown_until if cooldown_active else None,
            disabled_until=self.disabled_until if status is SymbolRiskStatus.DISABLED else None,
            metadata=dict(self.metadata),
        )


@dataclass(slots=True)
class StrategyRiskState:
    """
    Runtime state for one strategy.

    Tracks strategy-level PnL, open risk, rolling expectancy and throttling.
    """

    strategy_name: str

    status: StrategyRiskStatus = StrategyRiskStatus.ACTIVE

    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0

    open_risk: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0

    risk_multiplier: float = 1.0
    rolling_pnls: list[float] = field(default_factory=list)
    rolling_expectancy: float | None = None

    cooldown: CooldownState = field(default_factory=CooldownState)

    disabled_until: float | None = None
    disabled_reason: str | None = None

    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def register_trade_opened(self, *, open_risk: float = 0.0) -> None:
        self.trades_today += 1
        self.open_risk = max(0.0, self.open_risk + max(0.0, open_risk))
        self.updated_at = time.time()

    def register_trade_closed(
        self,
        *,
        realized_pnl: float,
        released_risk: float = 0.0,
        rolling_window: int | None = None,
    ) -> None:
        self.daily_pnl += realized_pnl
        self.weekly_pnl += realized_pnl
        self.monthly_pnl += realized_pnl
        self.open_risk = max(0.0, self.open_risk - max(0.0, released_risk))

        if realized_pnl < 0:
            self.consecutive_losses += 1
        elif realized_pnl > 0:
            self.consecutive_losses = 0

        self.rolling_pnls.append(realized_pnl)
        if rolling_window is not None and rolling_window > 0:
            self.rolling_pnls = self.rolling_pnls[-rolling_window:]

        self.rolling_expectancy = (
            sum(self.rolling_pnls) / len(self.rolling_pnls)
            if self.rolling_pnls
            else None
        )

        self.updated_at = time.time()

    def activate_cooldown(self, *, cooldown_seconds: float, reason: str | None = None) -> None:
        self.status = StrategyRiskStatus.COOLDOWN
        self.cooldown.activate(cooldown_seconds=cooldown_seconds, reason=reason)
        self.updated_at = time.time()

    def disable(self, *, reason: str | None = None, until: float | None = None) -> None:
        self.status = StrategyRiskStatus.DISABLED
        self.disabled_reason = reason
        self.disabled_until = until
        self.risk_multiplier = 0.0
        self.updated_at = time.time()

    def reduce(self, *, multiplier: float, reason: str | None = None) -> None:
        self.status = StrategyRiskStatus.REDUCED
        self.risk_multiplier = max(0.0, multiplier)
        self.metadata["reduced_reason"] = reason
        self.updated_at = time.time()

    def activate(self) -> None:
        self.status = StrategyRiskStatus.ACTIVE
        self.risk_multiplier = 1.0
        self.cooldown.deactivate()
        self.disabled_until = None
        self.disabled_reason = None
        self.updated_at = time.time()

    def refresh_status(self, *, now_ts: float | None = None) -> None:
        now_ts = now_ts or time.time()

        if self.status is StrategyRiskStatus.DISABLED and self.disabled_until is not None:
            if now_ts >= self.disabled_until:
                self.activate()

        if self.status is StrategyRiskStatus.COOLDOWN and self.cooldown.expire_if_needed(now_ts=now_ts):
            self.activate()

    def reset_daily(self) -> None:
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.consecutive_losses = 0
        self.updated_at = time.time()

    def reset_weekly(self) -> None:
        self.weekly_pnl = 0.0
        self.updated_at = time.time()

    def reset_monthly(self) -> None:
        self.monthly_pnl = 0.0
        self.updated_at = time.time()

    def snapshot(self, *, risk_unit: float) -> StrategyRiskSnapshot:
        now_ts = time.time()
        status = self.status
        cooldown_active = self.cooldown.is_active(now_ts=now_ts)
        risk_multiplier = self.risk_multiplier

        if status is StrategyRiskStatus.DISABLED and self.disabled_until is not None:
            if now_ts >= self.disabled_until:
                status = StrategyRiskStatus.ACTIVE
                risk_multiplier = 1.0

        if status is StrategyRiskStatus.COOLDOWN and not cooldown_active:
            status = StrategyRiskStatus.ACTIVE
            risk_multiplier = 1.0

        return StrategyRiskSnapshot(
            strategy_name=self.strategy_name,
            status=status,
            daily_pnl=self.daily_pnl,
            daily_loss_r=safe_div(abs(min(0.0, self.daily_pnl)), risk_unit),
            open_risk=self.open_risk,
            open_risk_r=safe_div(self.open_risk, risk_unit),
            trades_today=self.trades_today,
            consecutive_losses=self.consecutive_losses,
            rolling_expectancy=self.rolling_expectancy,
            rolling_trades=len(self.rolling_pnls),
            risk_multiplier=risk_multiplier,
            cooldown_until=self.cooldown.cooldown_until if cooldown_active else None,
            disabled_until=self.disabled_until if status is StrategyRiskStatus.DISABLED else None,
            metadata=dict(self.metadata),
        )


@dataclass(slots=True)
class TierRuntimeStats:
    """
    Runtime statistics grouped by TradeTier.
    """

    tier: TradeTier

    trades: int = 0
    approvals: int = 0
    rejections: int = 0

    realized_pnl: float = 0.0
    open_risk: float = 0.0

    rr_sum: float = 0.0
    rr_count: int = 0

    expected_value_sum: float = 0.0
    expected_value_count: int = 0

    cost_to_reward_sum: float = 0.0
    cost_to_reward_count: int = 0

    updated_at: float = field(default_factory=time.time)

    def register_approval(
        self,
        *,
        open_risk: float = 0.0,
        rr: float | None = None,
        expected_value: float | None = None,
        cost_to_reward: float | None = None,
    ) -> None:
        self.trades += 1
        self.approvals += 1
        self.open_risk += max(0.0, open_risk)

        if rr is not None:
            self.rr_sum += rr
            self.rr_count += 1

        if expected_value is not None:
            self.expected_value_sum += expected_value
            self.expected_value_count += 1

        if cost_to_reward is not None:
            self.cost_to_reward_sum += cost_to_reward
            self.cost_to_reward_count += 1

        self.updated_at = time.time()

    def register_rejection(self) -> None:
        self.rejections += 1
        self.updated_at = time.time()

    def register_close(self, *, realized_pnl: float, released_risk: float = 0.0) -> None:
        self.realized_pnl += realized_pnl
        self.open_risk = max(0.0, self.open_risk - max(0.0, released_risk))
        self.updated_at = time.time()

    def snapshot(self) -> TierStatsSnapshot:
        return TierStatsSnapshot(
            tier=self.tier,
            trades=self.trades,
            approvals=self.approvals,
            rejections=self.rejections,
            realized_pnl=self.realized_pnl,
            open_risk=self.open_risk,
            avg_rr=safe_div(self.rr_sum, self.rr_count, default=0.0)
            if self.rr_count
            else None,
            avg_expected_value=safe_div(
                self.expected_value_sum,
                self.expected_value_count,
                default=0.0,
            )
            if self.expected_value_count
            else None,
            avg_cost_to_reward=safe_div(
                self.cost_to_reward_sum,
                self.cost_to_reward_count,
                default=0.0,
            )
            if self.cost_to_reward_count
            else None,
        )


@dataclass(slots=True)
class RiskState:
    """
    Mutable runtime state of risk layer.

    This class intentionally has no EventBus, Scheduler or logger dependencies.
    RiskManager is responsible for locking, events, scheduling and lifecycle.
    """

    balance: float = 0.0
    equity: float = 0.0
    free_balance: float = 0.0
    used_margin: float = 0.0

    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    # Global realized PnL buckets used by RiskBudgetGuard/RiskModeResolver.
    # SymbolRiskState and StrategyRiskState keep their own independent buckets.
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0

    peak_equity: float = 0.0
    daily_start_equity: float = 0.0
    weekly_start_equity: float = 0.0
    monthly_start_equity: float = 0.0

    loss_streak: int = 0

    risk_mode: RiskMode = RiskMode.NORMAL
    trading_mode: TradingMode = TradingMode.NORMAL
    trading_halted: bool = False
    halt_reason: str | None = None

    positions: dict[str, PortfolioPosition] = field(default_factory=dict)
    pending_reservations: dict[str, PendingRiskReservation] = field(default_factory=dict)

    symbols: dict[str, SymbolRiskState] = field(default_factory=dict)
    strategies: dict[str, StrategyRiskState] = field(default_factory=dict)
    tiers: dict[TradeTier, TierRuntimeStats] = field(
        default_factory=lambda: {tier: TierRuntimeStats(tier=tier) for tier in TradeTier}
    )

    circuit_breaker: CircuitBreakerState = field(default_factory=CircuitBreakerState)

    manual_review_required: bool = False
    emergency_stop_active: bool = False

    last_rejected_reason: str | None = None
    updated_at: float = field(default_factory=time.time)

    def update_account(
        self,
        *,
        balance: float | None = None,
        equity: float | None = None,
        free_balance: float | None = None,
        used_margin: float | None = None,
        realized_pnl: float | None = None,
        unrealized_pnl: float | None = None,
    ) -> None:
        if balance is not None:
            self.balance = balance
        if equity is not None:
            self.equity = equity
        if free_balance is not None:
            self.free_balance = free_balance
        if used_margin is not None:
            self.used_margin = used_margin
        if realized_pnl is not None:
            self.realized_pnl = realized_pnl
        if unrealized_pnl is not None:
            self.unrealized_pnl = unrealized_pnl

        self._initialize_equity_anchors()
        self._update_peak_equity()
        self.updated_at = time.time()

    def set_risk_mode(self, mode: RiskMode, reason: str | None = None) -> None:
        self.risk_mode = mode
        self.trading_mode = self._risk_mode_to_trading_mode(mode)

        if mode in {RiskMode.HALTED, RiskMode.EMERGENCY_STOP}:
            self.trading_halted = True
            self.halt_reason = reason

        if mode is RiskMode.EMERGENCY_STOP:
            self.emergency_stop_active = True

        if mode in {RiskMode.NORMAL, RiskMode.CAUTION, RiskMode.SAFE_MODE}:
            self.trading_halted = False
            if not self.emergency_stop_active:
                self.halt_reason = None

        self.updated_at = time.time()

    def set_trading_mode(self, mode: TradingMode, reason: str | None = None) -> None:
        """
        Backward-compatible setter.
        Prefer set_risk_mode() in new code.
        """
        self.trading_mode = mode
        self.risk_mode = mode.to_risk_mode()

        if mode in {TradingMode.HALTED, TradingMode.EMERGENCY_STOP}:
            self.trading_halted = True
            self.halt_reason = reason

        if mode is TradingMode.EMERGENCY_STOP:
            self.emergency_stop_active = True

        self.updated_at = time.time()

    def halt_trading(self, reason: str) -> None:
        self.set_risk_mode(RiskMode.HALTED, reason=reason)

    def emergency_stop(self, reason: str) -> None:
        self.set_risk_mode(RiskMode.EMERGENCY_STOP, reason=reason)

    def activate_emergency_stop(self, *, reason: str | None = None) -> None:
        """Activate emergency stop using the explicit risk-state API."""
        self.set_risk_mode(
            RiskMode.EMERGENCY_STOP,
            reason=reason or "Emergency stop is active",
        )

    def resume_trading(self) -> None:
        if self.emergency_stop_active:
            return

        self.trading_halted = False
        self.halt_reason = None
        self.set_risk_mode(RiskMode.NORMAL)

    def enable_safe_mode(self, reason: str | None = None) -> None:
        self.set_risk_mode(RiskMode.SAFE_MODE, reason=reason)

    def enable_caution_mode(self, reason: str | None = None) -> None:
        self.set_risk_mode(RiskMode.CAUTION, reason=reason)

    def enable_reduce_only(self, reason: str | None = None) -> None:
        self.set_risk_mode(RiskMode.REDUCE_ONLY, reason=reason)

    def disable_protection_modes(self) -> None:
        if self.emergency_stop_active:
            return

        self.set_risk_mode(RiskMode.NORMAL)
        self.trading_halted = False
        self.halt_reason = None
        self.updated_at = time.time()

    def clear_manual_review(self) -> None:
        self.manual_review_required = False
        self.updated_at = time.time()

    def clear_emergency_stop(self) -> None:
        self.emergency_stop_active = False
        if self.risk_mode is RiskMode.EMERGENCY_STOP:
            self.set_risk_mode(RiskMode.NORMAL)
        self.updated_at = time.time()

    def reserve_risk(
        self,
        *,
        symbol: str,
        side: PositionSide,
        signal_id: str | None = None,
        strategy_name: str | None = None,
        tier: TradeTier | None = None,
        position_id: str | None = None,
        size: float = 0.0,
        open_risk: float = 0.0,
        margin: float = 0.0,
        notional: float = 0.0,
        ttl_seconds: float | None = 30.0,
        reservation_id: str | None = None,
        now_ts: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PendingRiskReservation:
        """
        Reserve candidate risk between risk approval and execution confirmation.

        The reservation must be released on execution rejection/cancellation/failure
        or confirmed when a real position is opened. This prevents multiple
        approved signals from temporarily exceeding open-risk/exposure budgets.
        """
        now_ts = now_ts or time.time()
        reservation = PendingRiskReservation(
            reservation_id=reservation_id or self._reservation_id(symbol, signal_id),
            signal_id=signal_id,
            symbol=symbol,
            side=side,
            strategy_name=strategy_name,
            tier=tier,
            position_id=position_id,
            size=max(0.0, size),
            open_risk=max(0.0, open_risk),
            margin=max(0.0, margin),
            notional=abs(notional),
            created_at=now_ts,
            expires_at=(now_ts + max(0.0, ttl_seconds)) if ttl_seconds is not None else None,
            metadata=dict(metadata or {}),
        )
        self.pending_reservations[reservation.reservation_id] = reservation
        self.updated_at = time.time()
        return reservation

    def get_pending_reservation(
        self,
        reservation_id: str | None = None,
        *,
        signal_id: str | None = None,
        symbol: str | None = None,
        position_id: str | None = None,
    ) -> PendingRiskReservation | None:
        if reservation_id is not None:
            return self.pending_reservations.get(reservation_id)

        for reservation in self.pending_reservations.values():
            if signal_id is not None and reservation.signal_id != signal_id:
                continue
            if symbol is not None and reservation.symbol != symbol:
                continue
            if position_id is not None and reservation.position_id != position_id:
                continue
            return reservation
        return None

    def release_risk_reservation(
        self,
        reservation_id: str | None = None,
        *,
        signal_id: str | None = None,
        symbol: str | None = None,
        position_id: str | None = None,
    ) -> PendingRiskReservation | None:
        reservation = self.get_pending_reservation(
            reservation_id,
            signal_id=signal_id,
            symbol=symbol,
            position_id=position_id,
        )
        if reservation is None:
            return None

        removed = self.pending_reservations.pop(reservation.reservation_id, None)
        self.updated_at = time.time()
        return removed

    def confirm_risk_reservation(
        self,
        reservation_id: str | None = None,
        *,
        signal_id: str | None = None,
        symbol: str | None = None,
        position_id: str | None = None,
    ) -> PendingRiskReservation | None:
        """
        Remove a reservation after execution confirms the position.

        This method intentionally does not call add_position(). RiskManager should
        confirm the reservation and then add/update the PortfolioPosition from the
        execution/position event payload so accounting remains explicit.
        """
        return self.release_risk_reservation(
            reservation_id,
            signal_id=signal_id,
            symbol=symbol,
            position_id=position_id,
        )

    def expire_pending_reservations(
        self,
        *,
        now_ts: float | None = None,
    ) -> list[PendingRiskReservation]:
        now_ts = now_ts or time.time()
        expired_ids = [
            reservation_id
            for reservation_id, reservation in self.pending_reservations.items()
            if reservation.is_expired(now_ts=now_ts)
        ]
        expired: list[PendingRiskReservation] = []
        for reservation_id in expired_ids:
            reservation = self.pending_reservations.pop(reservation_id, None)
            if reservation is not None:
                expired.append(reservation)

        if expired:
            self.updated_at = time.time()
        return expired

    def get_pending_open_risk(
        self,
        *,
        symbol: str | None = None,
        strategy_name: str | None = None,
        tier: TradeTier | None = None,
        now_ts: float | None = None,
        include_expired: bool = False,
    ) -> float:
        return sum(
            reservation.open_risk
            for reservation in self._iter_pending_reservations(
                symbol=symbol,
                strategy_name=strategy_name,
                tier=tier,
                now_ts=now_ts,
                include_expired=include_expired,
            )
        )

    def get_pending_margin(
        self,
        *,
        symbol: str | None = None,
        strategy_name: str | None = None,
        tier: TradeTier | None = None,
        now_ts: float | None = None,
        include_expired: bool = False,
    ) -> float:
        return sum(
            reservation.margin
            for reservation in self._iter_pending_reservations(
                symbol=symbol,
                strategy_name=strategy_name,
                tier=tier,
                now_ts=now_ts,
                include_expired=include_expired,
            )
        )

    def get_pending_notional(
        self,
        *,
        symbol: str | None = None,
        strategy_name: str | None = None,
        tier: TradeTier | None = None,
        side: PositionSide | None = None,
        now_ts: float | None = None,
        include_expired: bool = False,
    ) -> float:
        return sum(
            reservation.notional
            for reservation in self._iter_pending_reservations(
                symbol=symbol,
                strategy_name=strategy_name,
                tier=tier,
                side=side,
                now_ts=now_ts,
                include_expired=include_expired,
            )
        )

    def get_projected_open_risk(
        self,
        *,
        candidate_open_risk: float = 0.0,
        symbol: str | None = None,
        strategy_name: str | None = None,
        tier: TradeTier | None = None,
        now_ts: float | None = None,
    ) -> float:
        actual_open_risk = sum(
            position.open_risk
            for position in self.positions.values()
            if (symbol is None or position.symbol == symbol)
            and (strategy_name is None or position.strategy_name == strategy_name)
            and (tier is None or position.tier == tier)
        )
        return (
            actual_open_risk
            + self.get_pending_open_risk(
                symbol=symbol,
                strategy_name=strategy_name,
                tier=tier,
                now_ts=now_ts,
            )
            + max(0.0, candidate_open_risk)
        )

    def add_position(self, position: PortfolioPosition) -> None:
        key = self._position_key(position.symbol, position.position_id)
        self.positions[key] = position

        self.get_symbol_state(position.symbol).register_trade_opened(
            open_risk=position.open_risk,
        )

        if position.strategy_name:
            self.get_strategy_state(position.strategy_name).register_trade_opened(
                open_risk=position.open_risk,
            )

        if position.tier:
            self.get_tier_stats(position.tier).register_approval(
                open_risk=position.open_risk,
            )

        self.updated_at = time.time()

    def update_position(
        self,
        symbol: str,
        *,
        position_id: str | None = None,
        size: float | None = None,
        mark_price: float | None = None,
        notional_value: float | None = None,
        leverage: float | None = None,
        margin_used: float | None = None,
        risk_amount: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        unrealized_pnl: float | None = None,
    ) -> None:
        key = self._position_key(symbol, position_id)
        position = self.positions.get(key)
        if position is None:
            return

        if size is not None:
            position.size = size
        if mark_price is not None:
            position.mark_price = mark_price
        if notional_value is not None:
            position.notional_value = notional_value
        if leverage is not None:
            position.leverage = leverage
        if margin_used is not None:
            position.margin_used = margin_used
        if risk_amount is not None:
            position.risk_amount = risk_amount
        if stop_loss is not None:
            position.stop_loss = stop_loss
        if take_profit is not None:
            position.take_profit = take_profit
        if unrealized_pnl is not None:
            position.unrealized_pnl = unrealized_pnl

        position.updated_at = time.time()
        self.updated_at = time.time()

    def remove_position(
        self,
        symbol: str,
        *,
        position_id: str | None = None,
        realized_pnl: float = 0.0,
    ) -> PortfolioPosition | None:
        key = self._position_key(symbol, position_id)
        position = self.positions.pop(key, None)
        if position is None:
            return None

        released_risk = position.open_risk

        position.realized_pnl = realized_pnl
        self.register_trade_outcome(
            realized_pnl,
            symbol=position.symbol,
            strategy_name=position.strategy_name,
            tier=position.tier,
            released_risk=released_risk,
        )

        self.updated_at = time.time()
        return position

    def register_trade_outcome(
        self,
        pnl: float,
        *,
        symbol: str | None = None,
        strategy_name: str | None = None,
        tier: TradeTier | None = None,
        released_risk: float = 0.0,
        strategy_rolling_window: int | None = None,
    ) -> None:
        self.realized_pnl += pnl
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.monthly_pnl += pnl

        if pnl < 0:
            self.loss_streak += 1
        elif pnl > 0:
            self.loss_streak = 0

        if symbol:
            self.get_symbol_state(symbol).register_trade_closed(
                realized_pnl=pnl,
                released_risk=released_risk,
            )

        if strategy_name:
            self.get_strategy_state(strategy_name).register_trade_closed(
                realized_pnl=pnl,
                released_risk=released_risk,
                rolling_window=strategy_rolling_window,
            )

        if tier:
            self.get_tier_stats(tier).register_close(
                realized_pnl=pnl,
                released_risk=released_risk,
            )

        self.updated_at = time.time()

    def register_rejection(
        self,
        *,
        reason: str,
        symbol: str | None = None,
        strategy_name: str | None = None,
        tier: TradeTier | None = None,
    ) -> None:
        self.last_rejected_reason = reason

        if tier:
            self.get_tier_stats(tier).register_rejection()

        if symbol:
            self.get_symbol_state(symbol).metadata["last_rejection_reason"] = reason

        if strategy_name:
            self.get_strategy_state(strategy_name).metadata["last_rejection_reason"] = reason

        self.updated_at = time.time()

    def get_symbol_state(self, symbol: str) -> SymbolRiskState:
        state = self.symbols.get(symbol)
        if state is None:
            state = SymbolRiskState(symbol=symbol)
            self.symbols[symbol] = state
        return state

    def get_strategy_state(self, strategy_name: str) -> StrategyRiskState:
        state = self.strategies.get(strategy_name)
        if state is None:
            state = StrategyRiskState(strategy_name=strategy_name)
            self.strategies[strategy_name] = state
        return state

    def get_tier_stats(self, tier: TradeTier) -> TierRuntimeStats:
        stats = self.tiers.get(tier)
        if stats is None:
            stats = TierRuntimeStats(tier=tier)
            self.tiers[tier] = stats
        return stats

    def reset_daily_state(self) -> None:
        self.daily_start_equity = self.equity
        self.daily_pnl = 0.0
        self.loss_streak = 0

        for symbol_state in self.symbols.values():
            symbol_state.reset_daily()

        for strategy_state in self.strategies.values():
            strategy_state.reset_daily()

        self.updated_at = time.time()

    def reset_weekly_state(self) -> None:
        self.weekly_start_equity = self.equity
        self.weekly_pnl = 0.0

        for symbol_state in self.symbols.values():
            symbol_state.reset_weekly()

        for strategy_state in self.strategies.values():
            strategy_state.reset_weekly()

        self.updated_at = time.time()

    def reset_monthly_state(self) -> None:
        self.monthly_start_equity = self.equity
        self.monthly_pnl = 0.0
        self.manual_review_required = False

        for symbol_state in self.symbols.values():
            symbol_state.reset_monthly()

        for strategy_state in self.strategies.values():
            strategy_state.reset_monthly()

        self.updated_at = time.time()

    def activate_circuit_breaker(
        self,
        reason: CircuitBreakerReason,
        *,
        cooldown_until: float | None = None,
        message: str | None = None,
        manual_release_required: bool = False,
    ) -> None:
        self.circuit_breaker.active = True
        self.circuit_breaker.reason = reason
        self.circuit_breaker.triggered_at = time.time()
        self.circuit_breaker.cooldown_until = cooldown_until
        self.circuit_breaker.message = message
        self.circuit_breaker.manual_release_required = manual_release_required
        self.updated_at = time.time()

    def deactivate_circuit_breaker(self, *, force: bool = False) -> None:
        if self.circuit_breaker.manual_release_required and not force:
            return

        self.circuit_breaker.active = False
        self.circuit_breaker.reason = None
        self.circuit_breaker.triggered_at = None
        self.circuit_breaker.cooldown_until = None
        self.circuit_breaker.message = None
        self.circuit_breaker.manual_release_required = False
        self.circuit_breaker.metadata.clear()
        self.updated_at = time.time()

    def is_circuit_breaker_active(self, now_ts: float | None = None) -> bool:
        if not self.circuit_breaker.active:
            return False

        if self.circuit_breaker.manual_release_required:
            return True

        if self.circuit_breaker.cooldown_until is None:
            return True

        now_ts = now_ts or time.time()
        return now_ts < self.circuit_breaker.cooldown_until

    def get_daily_pnl(self) -> float:
        """
        Return global daily PnL for budget/risk-mode decisions.

        Supports both accounting styles used by the risk layer:
        - explicit realized daily bucket updated by register_trade_outcome();
        - equity-anchor delta used by account/equity based tests and snapshots.

        If the explicit bucket is non-zero, it is treated as the source of truth
        to avoid double-counting realized PnL and equity movement. Otherwise,
        the method falls back to the daily equity anchor when available.
        """
        if self.daily_pnl != 0.0:
            return self.daily_pnl + self.unrealized_pnl

        if self.daily_start_equity > 0:
            return self.equity - self.daily_start_equity

        return self.realized_pnl + self.unrealized_pnl

    def get_weekly_pnl(self) -> float:
        """Return global weekly PnL for budget/risk-mode decisions."""
        if self.weekly_pnl != 0.0:
            return self.weekly_pnl + self.unrealized_pnl

        if self.weekly_start_equity > 0:
            return self.equity - self.weekly_start_equity

        return self.realized_pnl + self.unrealized_pnl

    def get_monthly_pnl(self) -> float:
        """Return global monthly PnL for budget/risk-mode decisions."""
        if self.monthly_pnl != 0.0:
            return self.monthly_pnl + self.unrealized_pnl

        if self.monthly_start_equity > 0:
            return self.equity - self.monthly_start_equity

        return self.realized_pnl + self.unrealized_pnl

    def get_drawdown_snapshot(self) -> DrawdownSnapshot:
        current_equity = self.equity
        peak_equity = self.peak_equity if self.peak_equity > 0 else current_equity

        absolute_drawdown = max(0.0, peak_equity - current_equity)
        drawdown_percent = calculate_drawdown_pct(current_equity, peak_equity)

        return DrawdownSnapshot(
            peak_equity=peak_equity,
            current_equity=current_equity,
            absolute_drawdown=absolute_drawdown,
            drawdown_percent=drawdown_percent,
            daily_pnl=self.get_daily_pnl(),
            weekly_pnl=self.get_weekly_pnl(),
            monthly_pnl=self.get_monthly_pnl(),
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            loss_streak=self.loss_streak,
        )

    def get_exposure_snapshot(self) -> ExposureSnapshot:
        symbol_exposure: dict[str, float] = {}
        side_exposure: dict[str, float] = {
            PositionSide.LONG.value: 0.0,
            PositionSide.SHORT.value: 0.0,
        }

        gross_exposure = 0.0
        net_exposure = 0.0
        leverage_weighted_exposure = 0.0
        margin_used = 0.0
        actual_notional = 0.0

        for position in self.positions.values():
            notional = abs(position.notional_value)

            gross_exposure += notional
            actual_notional += notional
            net_exposure += position.signed_notional
            margin_used += max(0.0, position.margin_used)

            symbol_exposure[position.symbol] = (
                symbol_exposure.get(position.symbol, 0.0) + notional
            )
            side_exposure[position.side.value] += notional

            if position.leverage is not None:
                leverage_weighted_exposure += notional * position.leverage

        pending_margin = 0.0
        total_pending_notional = 0.0
        pending_symbol_exposure: dict[str, float] = {}
        pending_side_exposure: dict[str, float] = {
            PositionSide.LONG.value: 0.0,
            PositionSide.SHORT.value: 0.0,
        }

        for reservation in self._iter_pending_reservations():
            pending_notional = abs(reservation.notional)
            total_pending_notional += pending_notional
            gross_exposure += pending_notional

            if reservation.side is PositionSide.LONG:
                net_exposure += pending_notional
            elif reservation.side is PositionSide.SHORT:
                net_exposure -= pending_notional

            pending_margin += max(0.0, reservation.margin)
            symbol_exposure[reservation.symbol] = (
                symbol_exposure.get(reservation.symbol, 0.0) + pending_notional
            )
            pending_symbol_exposure[reservation.symbol] = (
                pending_symbol_exposure.get(reservation.symbol, 0.0) + pending_notional
            )
            side_exposure[reservation.side.value] = (
                side_exposure.get(reservation.side.value, 0.0) + pending_notional
            )
            pending_side_exposure[reservation.side.value] = (
                pending_side_exposure.get(reservation.side.value, 0.0) + pending_notional
            )

        effective_margin_used = (margin_used if margin_used > 0 else self.used_margin) + pending_margin

        return ExposureSnapshot(
            total_notional=gross_exposure,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            symbol_exposure=symbol_exposure,
            side_exposure=side_exposure,
            leverage_weighted_exposure=(
                leverage_weighted_exposure if leverage_weighted_exposure > 0 else None
            ),
            margin_used=effective_margin_used,
            margin_used_pct=safe_div(effective_margin_used, self.equity),
            actual_notional=actual_notional,
            pending_notional=total_pending_notional,
            pending_margin=pending_margin,
            pending_symbol_exposure=pending_symbol_exposure,
            pending_side_exposure=pending_side_exposure,
        )

    def get_open_risk_snapshot(self, *, risk_unit: float) -> OpenRiskSnapshot:
        symbol_open_risk: dict[str, float] = {}
        strategy_open_risk: dict[str, float] = {}
        tier_open_risk: dict[str, float] = {}

        actual_open_risk = 0.0

        for position in self.positions.values():
            open_risk = position.open_risk
            actual_open_risk += open_risk

            symbol_open_risk[position.symbol] = (
                symbol_open_risk.get(position.symbol, 0.0) + open_risk
            )

            if position.strategy_name:
                strategy_open_risk[position.strategy_name] = (
                    strategy_open_risk.get(position.strategy_name, 0.0) + open_risk
                )

            if position.tier:
                tier_key = position.tier.value
                tier_open_risk[tier_key] = tier_open_risk.get(tier_key, 0.0) + open_risk

        pending_orders_risk = 0.0
        for reservation in self._iter_pending_reservations():
            pending_orders_risk += reservation.open_risk

            symbol_open_risk[reservation.symbol] = (
                symbol_open_risk.get(reservation.symbol, 0.0) + reservation.open_risk
            )

            if reservation.strategy_name:
                strategy_open_risk[reservation.strategy_name] = (
                    strategy_open_risk.get(reservation.strategy_name, 0.0)
                    + reservation.open_risk
                )

            if reservation.tier:
                tier_key = reservation.tier.value
                tier_open_risk[tier_key] = (
                    tier_open_risk.get(tier_key, 0.0) + reservation.open_risk
                )

        total_open_risk = actual_open_risk + pending_orders_risk
        effective_margin_used = self.used_margin + self.get_pending_margin()

        pending_reservations_count = sum(1 for _ in self._iter_pending_reservations())

        return OpenRiskSnapshot(
            total_open_risk=total_open_risk,
            total_open_risk_r=safe_div(total_open_risk, risk_unit),
            used_margin=effective_margin_used,
            used_margin_pct=safe_div(effective_margin_used, self.equity),
            symbol_open_risk=symbol_open_risk,
            strategy_open_risk=strategy_open_risk,
            tier_open_risk=tier_open_risk,
            positions_count=len(self.positions) + pending_reservations_count,
            actual_open_risk=actual_open_risk,
            actual_open_risk_r=safe_div(actual_open_risk, risk_unit),
            pending_orders_risk=pending_orders_risk,
            pending_orders_risk_r=safe_div(pending_orders_risk, risk_unit),
            pending_margin=self.get_pending_margin(),
            pending_notional=self.get_pending_notional(),
            pending_reservations_count=pending_reservations_count,
        )

    def get_correlation_snapshot(self, groups: dict[str, list[str]]) -> CorrelationSnapshot:
        symbol_to_group: dict[str, str] = {}
        group_exposure: dict[str, float] = {group_name: 0.0 for group_name in groups}
        group_open_risk: dict[str, float] = {group_name: 0.0 for group_name in groups}

        for group_name, symbols in groups.items():
            for symbol in symbols:
                symbol_to_group[symbol] = group_name

        for position in self.positions.values():
            group_name = symbol_to_group.get(position.symbol)
            if group_name is not None:
                group_exposure[group_name] += abs(position.notional_value)
                group_open_risk[group_name] += position.open_risk

        for reservation in self._iter_pending_reservations():
            group_name = symbol_to_group.get(reservation.symbol)
            if group_name is not None:
                group_exposure[group_name] += abs(reservation.notional)
                group_open_risk[group_name] += reservation.open_risk

        return CorrelationSnapshot(
            groups=groups,
            symbol_to_group=symbol_to_group,
            group_exposure=group_exposure,
            group_open_risk=group_open_risk,
        )

    def get_budget_snapshot(
        self,
        *,
        risk_unit: float,
        caution_daily_loss_r: float,
        soft_daily_loss_r: float,
        hard_daily_loss_r: float,
        weekly_hard_loss_r: float,
        monthly_review_loss_r: float,
        emergency_stop_loss_r: float,
    ) -> RiskBudgetSnapshot:
        daily_pnl = self.get_daily_pnl()
        weekly_pnl = self.get_weekly_pnl()
        monthly_pnl = self.get_monthly_pnl()

        daily_loss_r = safe_div(abs(min(0.0, daily_pnl)), risk_unit)
        weekly_loss_r = safe_div(abs(min(0.0, weekly_pnl)), risk_unit)
        monthly_loss_r = safe_div(abs(min(0.0, monthly_pnl)), risk_unit)

        return RiskBudgetSnapshot(
            mode=self.risk_mode,
            daily_pnl=daily_pnl,
            weekly_pnl=weekly_pnl,
            monthly_pnl=monthly_pnl,
            daily_loss_r=daily_loss_r,
            weekly_loss_r=weekly_loss_r,
            monthly_loss_r=monthly_loss_r,
            caution_daily_loss_r=caution_daily_loss_r,
            soft_daily_loss_r=soft_daily_loss_r,
            hard_daily_loss_r=hard_daily_loss_r,
            weekly_hard_loss_r=weekly_hard_loss_r,
            monthly_review_loss_r=monthly_review_loss_r,
            emergency_stop_loss_r=emergency_stop_loss_r,
            remaining_daily_r=max(0.0, hard_daily_loss_r - daily_loss_r),
            remaining_weekly_r=max(0.0, weekly_hard_loss_r - weekly_loss_r),
            remaining_monthly_r=max(0.0, monthly_review_loss_r - monthly_loss_r),
            manual_review_required=self.manual_review_required,
            emergency_stop_active=self.emergency_stop_active,
        )

    def snapshot(
        self,
        *,
        risk_unit: float = 1.0,
        caution_daily_loss_r: float = 3.0,
        soft_daily_loss_r: float = 6.0,
        hard_daily_loss_r: float = 10.0,
        weekly_hard_loss_r: float = 25.0,
        monthly_review_loss_r: float = 40.0,
        emergency_stop_loss_r: float = 50.0,
    ) -> RiskStateSnapshot:
        """
        Build immutable snapshot for stats/dashboard/API.

        Defaults are safe fallback values. RiskManager should pass configured
        values from RiskConfig when producing production snapshots.
        """
        effective_risk_unit = risk_unit if risk_unit > 0 else 1.0

        return RiskStateSnapshot(
            balance=self.balance,
            equity=self.equity,
            free_balance=self.free_balance,
            used_margin=self.used_margin,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            peak_equity=self.peak_equity,
            daily_start_equity=self.daily_start_equity,
            daily_pnl=self.get_daily_pnl(),
            weekly_pnl=self.get_weekly_pnl(),
            monthly_pnl=self.get_monthly_pnl(),
            loss_streak=self.loss_streak,
            risk_mode=self.risk_mode,
            trading_mode=self.trading_mode,
            trading_halted=self.trading_halted,
            halt_reason=self.halt_reason,
            positions_count=len(self.positions),
            exposure=self.get_exposure_snapshot(),
            open_risk=self.get_open_risk_snapshot(risk_unit=effective_risk_unit),
            drawdown=self.get_drawdown_snapshot(),
            budget=self.get_budget_snapshot(
                risk_unit=effective_risk_unit,
                caution_daily_loss_r=caution_daily_loss_r,
                soft_daily_loss_r=soft_daily_loss_r,
                hard_daily_loss_r=hard_daily_loss_r,
                weekly_hard_loss_r=weekly_hard_loss_r,
                monthly_review_loss_r=monthly_review_loss_r,
                emergency_stop_loss_r=emergency_stop_loss_r,
            ),
            circuit_breaker=self.circuit_breaker,
            symbols={
                symbol: state.snapshot(risk_unit=effective_risk_unit)
                for symbol, state in self.symbols.items()
            },
            strategies={
                strategy_name: state.snapshot(risk_unit=effective_risk_unit)
                for strategy_name, state in self.strategies.items()
            },
            tiers={tier: stats.snapshot() for tier, stats in self.tiers.items()},
        )

    def _iter_pending_reservations(
        self,
        *,
        symbol: str | None = None,
        strategy_name: str | None = None,
        tier: TradeTier | None = None,
        side: PositionSide | None = None,
        now_ts: float | None = None,
        include_expired: bool = False,
    ):
        now_ts = now_ts or time.time()
        for reservation in self.pending_reservations.values():
            if not include_expired and reservation.is_expired(now_ts=now_ts):
                continue
            if symbol is not None and reservation.symbol != symbol:
                continue
            if strategy_name is not None and reservation.strategy_name != strategy_name:
                continue
            if tier is not None and reservation.tier != tier:
                continue
            if side is not None and reservation.side != side:
                continue
            yield reservation

    def _initialize_equity_anchors(self) -> None:
        if self.equity <= 0:
            return

        if self.peak_equity <= 0:
            self.peak_equity = self.equity

        if self.daily_start_equity <= 0:
            self.daily_start_equity = self.equity

        if self.weekly_start_equity <= 0:
            self.weekly_start_equity = self.equity

        if self.monthly_start_equity <= 0:
            self.monthly_start_equity = self.equity

    def _update_peak_equity(self) -> None:
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    @staticmethod
    def _risk_mode_to_trading_mode(mode: RiskMode) -> TradingMode:
        mapping = {
            RiskMode.NORMAL: TradingMode.NORMAL,
            RiskMode.CAUTION: TradingMode.NORMAL,
            RiskMode.SAFE_MODE: TradingMode.SAFE_MODE,
            RiskMode.REDUCE_ONLY: TradingMode.REDUCE_ONLY,
            RiskMode.HALTED: TradingMode.HALTED,
            RiskMode.EMERGENCY_STOP: TradingMode.EMERGENCY_STOP,
        }
        return mapping[mode]

    @staticmethod
    def _reservation_id(symbol: str, signal_id: str | None = None) -> str:
        suffix = signal_id or uuid4().hex
        return f"reservation:{symbol}:{suffix}"

    @staticmethod
    def _position_key(symbol: str, position_id: str | None = None) -> str:
        return f"{symbol}:{position_id}" if position_id else symbol


__all__ = [
    "CooldownState",
    "PendingRiskReservation",
    "RiskState",
    "StrategyRiskState",
    "SymbolRiskState",
    "TierRuntimeStats",
]