from __future__ import annotations

import time
from dataclasses import dataclass, field

from risk.enums import CircuitBreakerReason, PositionSide, TradingMode
from risk.models import (
    CircuitBreakerState,
    CorrelationSnapshot,
    DrawdownSnapshot,
    ExposureSnapshot,
    PortfolioPosition,
    RiskStateSnapshot,
)


@dataclass(slots=True)
class RiskState:
    balance: float = 0.0
    equity: float = 0.0
    free_balance: float = 0.0
    used_margin: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    peak_equity: float = 0.0
    daily_start_equity: float = 0.0
    loss_streak: int = 0

    trading_mode: TradingMode = TradingMode.NORMAL
    trading_halted: bool = False
    halt_reason: str | None = None

    positions: dict[str, PortfolioPosition] = field(default_factory=dict)
    circuit_breaker: CircuitBreakerState = field(default_factory=CircuitBreakerState)

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

        if self.peak_equity <= 0 and self.equity > 0:
            self.peak_equity = self.equity

        if self.daily_start_equity <= 0 and self.equity > 0:
            self.daily_start_equity = self.equity

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        self.updated_at = time.time()

    def set_trading_mode(self, mode: TradingMode, reason: str | None = None) -> None:
        self.trading_mode = mode
        self.halt_reason = reason if mode is TradingMode.HALTED else self.halt_reason
        self.updated_at = time.time()

    def halt_trading(self, reason: str) -> None:
        self.trading_halted = True
        self.trading_mode = TradingMode.HALTED
        self.halt_reason = reason
        self.updated_at = time.time()

    def resume_trading(self) -> None:
        self.trading_halted = False
        self.halt_reason = None
        if self.trading_mode is TradingMode.HALTED:
            self.trading_mode = TradingMode.NORMAL
        self.updated_at = time.time()

    def enable_safe_mode(self, reason: str | None = None) -> None:
        self.trading_mode = TradingMode.SAFE_MODE
        self.halt_reason = reason
        self.updated_at = time.time()

    def enable_reduce_only(self, reason: str | None = None) -> None:
        self.trading_mode = TradingMode.REDUCE_ONLY
        self.halt_reason = reason
        self.updated_at = time.time()

    def disable_protection_modes(self) -> None:
        self.trading_mode = TradingMode.NORMAL
        if not self.trading_halted:
            self.halt_reason = None
        self.updated_at = time.time()

    def add_position(self, position: PortfolioPosition) -> None:
        key = self._position_key(position.symbol, position.position_id)
        self.positions[key] = position
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
        if unrealized_pnl is not None:
            position.unrealized_pnl = unrealized_pnl

        self.updated_at = time.time()

    def remove_position(self, symbol: str, *, position_id: str | None = None) -> PortfolioPosition | None:
        key = self._position_key(symbol, position_id)
        position = self.positions.pop(key, None)
        self.updated_at = time.time()
        return position

    def register_trade_outcome(self, pnl: float) -> None:
        if pnl < 0:
            self.loss_streak += 1
        elif pnl > 0:
            self.loss_streak = 0
        self.updated_at = time.time()

    def reset_daily_state(self) -> None:
        self.daily_start_equity = self.equity
        self.loss_streak = 0
        self.updated_at = time.time()

    def activate_circuit_breaker(
        self,
        reason: CircuitBreakerReason,
        *,
        cooldown_until: float | None = None,
        message: str | None = None,
    ) -> None:
        self.circuit_breaker.active = True
        self.circuit_breaker.reason = reason
        self.circuit_breaker.triggered_at = time.time()
        self.circuit_breaker.cooldown_until = cooldown_until
        self.circuit_breaker.message = message
        self.updated_at = time.time()

    def deactivate_circuit_breaker(self) -> None:
        self.circuit_breaker.active = False
        self.circuit_breaker.reason = None
        self.circuit_breaker.triggered_at = None
        self.circuit_breaker.cooldown_until = None
        self.circuit_breaker.message = None
        self.updated_at = time.time()

    def is_circuit_breaker_active(self, now_ts: float | None = None) -> bool:
        if not self.circuit_breaker.active:
            return False

        if self.circuit_breaker.cooldown_until is None:
            return True

        now_ts = now_ts or time.time()
        return now_ts < self.circuit_breaker.cooldown_until

    def get_daily_pnl(self) -> float:
        if self.daily_start_equity <= 0:
            return 0.0
        return self.equity - self.daily_start_equity

    def get_drawdown_snapshot(self) -> DrawdownSnapshot:
        current_equity = self.equity
        peak_equity = self.peak_equity if self.peak_equity > 0 else current_equity

        absolute_drawdown = max(0.0, peak_equity - current_equity)
        drawdown_percent = (absolute_drawdown / peak_equity) if peak_equity > 0 else 0.0

        return DrawdownSnapshot(
            peak_equity=peak_equity,
            current_equity=current_equity,
            absolute_drawdown=absolute_drawdown,
            drawdown_percent=drawdown_percent,
            daily_pnl=self.get_daily_pnl(),
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

        for position in self.positions.values():
            notional = abs(position.notional_value)
            gross_exposure += notional
            net_exposure += position.signed_notional

            symbol_exposure[position.symbol] = symbol_exposure.get(position.symbol, 0.0) + notional
            side_exposure[position.side.value] += notional

            if position.leverage is not None:
                leverage_weighted_exposure += notional * position.leverage

        return ExposureSnapshot(
            total_notional=gross_exposure,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            symbol_exposure=symbol_exposure,
            side_exposure=side_exposure,
            leverage_weighted_exposure=leverage_weighted_exposure if leverage_weighted_exposure > 0 else None,
        )

    def get_correlation_snapshot(self, groups: dict[str, list[str]]) -> CorrelationSnapshot:
        symbol_to_group: dict[str, str] = {}
        group_exposure: dict[str, float] = {group_name: 0.0 for group_name in groups}

        for group_name, symbols in groups.items():
            for symbol in symbols:
                symbol_to_group[symbol] = group_name

        for position in self.positions.values():
            group_name = symbol_to_group.get(position.symbol)
            if group_name is not None:
                group_exposure[group_name] += abs(position.notional_value)

        return CorrelationSnapshot(
            groups=groups,
            symbol_to_group=symbol_to_group,
            group_exposure=group_exposure,
        )

    def snapshot(self) -> RiskStateSnapshot:
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
            loss_streak=self.loss_streak,
            trading_mode=self.trading_mode,
            trading_halted=self.trading_halted,
            halt_reason=self.halt_reason,
            positions_count=len(self.positions),
            exposure=self.get_exposure_snapshot(),
            drawdown=self.get_drawdown_snapshot(),
            circuit_breaker=self.circuit_breaker,
        )

    @staticmethod
    def _position_key(symbol: str, position_id: str | None = None) -> str:
        return f"{symbol}:{position_id}" if position_id else symbol