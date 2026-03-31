from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.logger import get_logger

from risk.config import PositionSizingConfig
from risk.enums import RiskDecisionType, RiskLevel, RiskViolationType, TradingMode
from risk.exceptions import InvalidPositionSizeError, InvalidRiskRequestError
from risk.models import (
    PositionSizeRequest,
    PositionSizeResult,
    RiskCheckResult,
    RiskViolation,
)
from risk.state import RiskState
from risk.utils import (
    apply_cap,
    apply_confidence_scale,
    calculate_margin_required,
    calculate_notional,
    calculate_stop_distance,
    clamp,
    round_down_to_step,
)


@dataclass(slots=True)
class SymbolConstraints:
    min_size: float | None = None
    max_size: float | None = None
    step_size: float | None = None
    min_notional: float | None = None


class PositionSizer:
    """
    Відповідає лише за розрахунок допустимого розміру позиції.
    Не приймає фінальне risk-рішення на рівні всього портфеля.
    """

    def __init__(
        self,
        config: PositionSizingConfig,
        *,
        symbol_constraints: dict[str, SymbolConstraints] | None = None,
        service_name: str = "risk.position_sizer",
    ) -> None:
        self._config = config
        self._symbol_constraints = symbol_constraints or {}
        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="position_sizer",
        )

    def calculate(
        self,
        request: PositionSizeRequest,
        state: RiskState,
    ) -> PositionSizeResult:
        self._validate_request(request, state)

        effective_risk_pct = min(
            request.risk_percent,
            self._config.max_risk_per_trade_pct,
        )

        if state.trading_mode is TradingMode.SAFE_MODE:
            effective_risk_pct *= 0.5

        risk_amount = state.equity * effective_risk_pct

        stop_distance = calculate_stop_distance(request.entry_price, request.stop_loss)

        if stop_distance is None:
            if self._config.require_stop_loss:
                raise InvalidPositionSizeError("Stop loss is required for position sizing")
            if self._config.fallback_stop_loss_pct is None or self._config.fallback_stop_loss_pct <= 0:
                raise InvalidPositionSizeError(
                    "Stop loss is missing and no valid fallback_stop_loss_pct configured"
                )
            stop_distance = request.entry_price * self._config.fallback_stop_loss_pct

        if stop_distance <= 0:
            raise InvalidPositionSizeError("Stop distance must be > 0")

        raw_size = risk_amount / stop_distance

        if self._config.use_confidence_adjustment:
            raw_size = apply_confidence_scale(
                raw_size,
                request.confidence,
                self._config.confidence_scale_min,
                self._config.confidence_scale_max,
            )

        if self._config.use_volatility_adjustment and request.volatility is not None and request.volatility > 0:
            # Чим вища волатильність, тим обережніше scale.
            volatility_scale = 1.0 / (1.0 + request.volatility)
            volatility_scale = clamp(volatility_scale, 0.25, 1.0)
            raw_size *= volatility_scale

        raw_size = max(0.0, raw_size)

        constraints = self._symbol_constraints.get(request.symbol, SymbolConstraints())

        if request.min_size is not None:
            constraints.min_size = max(constraints.min_size or 0.0, request.min_size)

        if request.max_size is not None:
            constraints.max_size = (
                min(constraints.max_size, request.max_size)
                if constraints.max_size is not None
                else request.max_size
            )

        normalized_size = self.normalize_size(request.symbol, raw_size, constraints)

        if normalized_size <= 0:
            raise InvalidPositionSizeError("Calculated normalized position size is <= 0")

        leverage_used = request.leverage
        notional_value = calculate_notional(request.entry_price, normalized_size)
        margin_required = calculate_margin_required(
            request.entry_price,
            normalized_size,
            leverage_used,
        )

        capped = False
        reason_parts: list[str] = []

        if constraints.max_size is not None and normalized_size >= constraints.max_size:
            capped = True
            reason_parts.append("max_size_cap_applied")

        if self._config.max_position_size is not None and normalized_size >= self._config.max_position_size:
            capped = True
            reason_parts.append("global_max_position_size_cap_applied")

        if margin_required > state.free_balance > 0:
            affordable_size = self._size_by_available_margin(
                entry_price=request.entry_price,
                free_balance=state.free_balance,
                leverage=leverage_used,
            )
            affordable_size = self.normalize_size(request.symbol, affordable_size, constraints)

            if affordable_size <= 0:
                raise InvalidPositionSizeError("Free balance is insufficient for even minimal position size")

            normalized_size = min(normalized_size, affordable_size)
            notional_value = calculate_notional(request.entry_price, normalized_size)
            margin_required = calculate_margin_required(
                request.entry_price,
                normalized_size,
                leverage_used,
            )
            capped = True
            reason_parts.append("free_balance_cap_applied")

        if constraints.min_notional is not None and notional_value < constraints.min_notional:
            raise InvalidPositionSizeError(
                f"Calculated notional {notional_value:.8f} is below min_notional {constraints.min_notional:.8f}"
            )

        self._logger.info(
            "Position size calculated | symbol=%s size=%s notional=%s risk_amount=%s leverage=%s",
            request.symbol,
            normalized_size,
            notional_value,
            risk_amount,
            leverage_used,
            extra={
                "symbol": request.symbol,
            },
        )

        return PositionSizeResult(
            size=normalized_size,
            notional_value=notional_value,
            risk_amount=risk_amount,
            risk_percent_used=effective_risk_pct,
            leverage_used=leverage_used,
            capped=capped,
            reason=", ".join(reason_parts) if reason_parts else None,
            metadata={
                "stop_distance": stop_distance,
                "margin_required": margin_required,
                "raw_size": raw_size,
            },
        )

    def check(
        self,
        request: PositionSizeRequest,
        state: RiskState,
    ) -> RiskCheckResult:
        try:
            result = self.calculate(request, state)
            decision = RiskDecisionType.REDUCE_SIZE if result.capped else RiskDecisionType.ALLOW
            return RiskCheckResult(
                passed=True,
                decision=decision,
                adjusted_size=result.size,
                adjusted_leverage=result.leverage_used,
                metadata={
                    "notional_value": result.notional_value,
                    "risk_amount": result.risk_amount,
                    "risk_percent_used": result.risk_percent_used,
                    "capped": result.capped,
                    "reason": result.reason,
                },
            )
        except InvalidPositionSizeError as exc:
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.POSITION_SIZE_INVALID,
                        level=RiskLevel.CRITICAL,
                        message=str(exc),
                    )
                ],
            )
        except InvalidRiskRequestError as exc:
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.INVALID_REQUEST,
                        level=RiskLevel.CRITICAL,
                        message=str(exc),
                    )
                ],
            )

    def normalize_size(
        self,
        symbol: str,
        raw_size: float,
        constraints: SymbolConstraints | None = None,
    ) -> float:
        constraints = constraints or self._symbol_constraints.get(symbol, SymbolConstraints())

        size = raw_size

        size = max(size, self._config.min_position_size)
        size = apply_cap(size, self._config.max_position_size)

        if constraints.min_size is not None:
            size = max(size, constraints.min_size)

        if constraints.max_size is not None:
            size = min(size, constraints.max_size)

        size = round_down_to_step(size, constraints.step_size)

        if size < 0:
            return 0.0

        return size

    @staticmethod
    def _size_by_available_margin(
        *,
        entry_price: float,
        free_balance: float,
        leverage: float | None,
    ) -> float:
        if entry_price <= 0:
            return 0.0

        effective_leverage = leverage if leverage is not None and leverage > 0 else 1.0
        max_notional = free_balance * effective_leverage
        return max_notional / entry_price

    @staticmethod
    def _validate_request(request: PositionSizeRequest, state: RiskState) -> None:
        if not request.symbol:
            raise InvalidRiskRequestError("symbol is required")

        if request.entry_price <= 0:
            raise InvalidRiskRequestError("entry_price must be > 0")

        if request.account_equity < 0:
            raise InvalidRiskRequestError("account_equity must be >= 0")

        if request.free_balance < 0:
            raise InvalidRiskRequestError("free_balance must be >= 0")

        if request.risk_percent <= 0:
            raise InvalidRiskRequestError("risk_percent must be > 0")

        if state.equity < 0:
            raise InvalidRiskRequestError("risk state equity must be >= 0")