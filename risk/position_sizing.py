from __future__ import annotations

from dataclasses import dataclass

from core.logger import get_logger
from risk.config import PositionSizingConfig, RiskUnitConfig
from risk.enums import RiskDecisionType, RiskLevel, RiskMode, RiskViolationType
from risk.exceptions import InvalidPositionSizeError, InvalidRiskRequestError
from risk.models import (
    PositionSizeRequest,
    PositionSizeResult,
    RiskCheckResult,
    RiskUnitSnapshot,
    RiskViolation,
)
from risk.state import RiskState
from risk.utils import (
    apply_cap,
    apply_confidence_scale,
    apply_volatility_scale,
    calculate_margin_required,
    calculate_notional,
    calculate_position_size_by_risk,
    calculate_side_aware_stop_distance,
    round_down_to_step,
)


@dataclass(slots=True)
class SymbolConstraints:
    """
    Exchange-level symbol constraints.

    Важливо: min_size/min_notional не повинні автоматично збільшувати
    позицію понад дозволений risk. Якщо calculated size нижче біржового
    мінімуму — це має бути DENY, а не auto-upsize.
    """

    min_size: float | None = None
    max_size: float | None = None
    step_size: float | None = None
    min_notional: float | None = None
    max_notional: float | None = None


class RiskUnitCalculator:
    """
    Calculates current effective R.

    R — базова одиниця ризику. Цей клас не приймає фінальне торгове
    рішення, а лише рахує effective_risk_unit для подальшого sizing.

    Не має EventBus/Scheduler залежностей. Це pure domain service,
    який викликається RiskManager або PositionSizer.
    """

    def __init__(
        self,
        config: RiskUnitConfig,
        *,
        service_name: str = "risk.risk_unit",
    ) -> None:
        self._config = config
        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="risk_unit",
        )

    def calculate(
        self,
        state: RiskState,
        *,
        mode: RiskMode | None = None,
        strategy_multiplier: float = 1.0,
        symbol_multiplier: float = 1.0,
        confidence_multiplier: float = 1.0,
        volatility_multiplier: float = 1.0,
        available_daily_r: float | None = None,
        available_open_r: float | None = None,
        available_strategy_r: float | None = None,
        available_symbol_r: float | None = None,
    ) -> RiskUnitSnapshot:
        if state.equity < 0:
            raise InvalidRiskRequestError("risk state equity must be >= 0")

        mode = mode or state.risk_mode

        base = self._calculate_base_risk_unit(state)
        mode_multiplier = self._mode_multiplier(mode)

        effective = base
        effective *= mode_multiplier
        effective *= max(0.0, strategy_multiplier)
        effective *= max(0.0, symbol_multiplier)
        effective *= max(0.0, confidence_multiplier)
        effective *= max(0.0, volatility_multiplier)

        capped_by_daily_budget = False
        capped_by_open_risk = False
        capped_by_strategy_budget = False
        capped_by_symbol_budget = False

        if self._config.use_available_budget_caps:
            if available_daily_r is not None and available_daily_r >= 0:
                cap = base * available_daily_r
                if effective > cap:
                    effective = cap
                    capped_by_daily_budget = True

            if available_open_r is not None and available_open_r >= 0:
                cap = base * available_open_r
                if effective > cap:
                    effective = cap
                    capped_by_open_risk = True

            if available_strategy_r is not None and available_strategy_r >= 0:
                cap = base * available_strategy_r
                if effective > cap:
                    effective = cap
                    capped_by_strategy_budget = True

            if available_symbol_r is not None and available_symbol_r >= 0:
                cap = base * available_symbol_r
                if effective > cap:
                    effective = cap
                    capped_by_symbol_budget = True

        if self._config.min_risk_unit is not None:
            effective = max(effective, self._config.min_risk_unit)

        if self._config.max_risk_unit is not None:
            effective = min(effective, self._config.max_risk_unit)

        effective = max(0.0, effective)

        return RiskUnitSnapshot(
            base_risk_unit=base,
            effective_risk_unit=effective,
            mode=mode,
            mode_multiplier=mode_multiplier,
            strategy_multiplier=max(0.0, strategy_multiplier),
            symbol_multiplier=max(0.0, symbol_multiplier),
            confidence_multiplier=max(0.0, confidence_multiplier),
            volatility_multiplier=max(0.0, volatility_multiplier),
            capped_by_daily_budget=capped_by_daily_budget,
            capped_by_open_risk=capped_by_open_risk,
            capped_by_strategy_budget=capped_by_strategy_budget,
            capped_by_symbol_budget=capped_by_symbol_budget,
            metadata={
                "available_daily_r": available_daily_r,
                "available_open_r": available_open_r,
                "available_strategy_r": available_strategy_r,
                "available_symbol_r": available_symbol_r,
            },
        )

    def _calculate_base_risk_unit(self, state: RiskState) -> float:
        if self._config.use_equity_for_r:
            base = state.equity * self._config.base_risk_unit_pct
        else:
            base = self._config.base_risk_unit_pct

        if self._config.min_risk_unit is not None:
            base = max(base, self._config.min_risk_unit)

        if self._config.max_risk_unit is not None:
            base = min(base, self._config.max_risk_unit)

        return max(0.0, base)

    def _mode_multiplier(self, mode: RiskMode) -> float:
        if mode is RiskMode.NORMAL:
            return 1.0
        if mode is RiskMode.CAUTION:
            return self._config.caution_multiplier
        if mode is RiskMode.SAFE_MODE:
            return self._config.safe_mode_multiplier
        if mode is RiskMode.REDUCE_ONLY:
            return self._config.reduce_only_multiplier
        if mode is RiskMode.HALTED:
            return self._config.halted_multiplier
        if mode is RiskMode.EMERGENCY_STOP:
            return self._config.emergency_stop_multiplier
        return 1.0


class PositionSizer:
    """
    Calculates position size from risk amount and stop distance.

    Відповідальність класу:
    - side-aware stop distance;
    - size = risk_amount / stop_distance;
    - confidence/volatility adjustment;
    - exchange constraints;
    - notional/margin calculation;
    - safe down-rounding.

    Клас не приймає фінального portfolio risk-рішення.
    Це робить RiskManager після guards/exposure/budget checks.
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

        stop_distance = self._resolve_stop_distance(request)

        raw_size = calculate_position_size_by_risk(
            risk_amount=request.risk_amount,
            stop_distance=stop_distance,
        )

        if self._config.use_confidence_adjustment:
            raw_size = apply_confidence_scale(
                raw_size,
                request.confidence,
                self._config.confidence_scale_min,
                self._config.confidence_scale_max,
            )

        if self._config.use_volatility_adjustment:
            raw_size = apply_volatility_scale(
                raw_size,
                request.volatility,
                scale_min=self._config.volatility_scale_min,
                scale_max=self._config.volatility_scale_max,
            )

        if request.requested_size is not None and request.requested_size > 0:
            raw_size = min(raw_size, request.requested_size)

        constraints = self._resolve_constraints(request)

        normalized_size, capped, reason_parts = self.normalize_size(
            request.symbol,
            raw_size,
            constraints,
        )

        if normalized_size <= 0:
            raise InvalidPositionSizeError("Calculated normalized position size is <= 0")

        notional_value = calculate_notional(request.entry_price, normalized_size)
        margin_required = calculate_margin_required(
            request.entry_price,
            normalized_size,
            request.leverage,
        )

        if request.requested_margin is not None and request.requested_margin > 0:
            if margin_required > request.requested_margin:
                affordable_size = self._size_by_available_margin(
                    entry_price=request.entry_price,
                    free_balance=request.requested_margin,
                    leverage=request.leverage,
                )
                normalized_size, margin_capped, margin_reasons = self.normalize_size(
                    request.symbol,
                    min(normalized_size, affordable_size),
                    constraints,
                )
                capped = capped or margin_capped
                reason_parts.extend(["requested_margin_cap_applied", *margin_reasons])

                notional_value = calculate_notional(request.entry_price, normalized_size)
                margin_required = calculate_margin_required(
                    request.entry_price,
                    normalized_size,
                    request.leverage,
                )

        if margin_required > state.free_balance > 0:
            affordable_size = self._size_by_available_margin(
                entry_price=request.entry_price,
                free_balance=state.free_balance,
                leverage=request.leverage,
            )
            normalized_size, margin_capped, margin_reasons = self.normalize_size(
                request.symbol,
                min(normalized_size, affordable_size),
                constraints,
            )
            capped = capped or margin_capped
            reason_parts.extend(["free_balance_cap_applied", *margin_reasons])

            if normalized_size <= 0:
                raise InvalidPositionSizeError(
                    "Free balance is insufficient for even minimal position size"
                )

            notional_value = calculate_notional(request.entry_price, normalized_size)
            margin_required = calculate_margin_required(
                request.entry_price,
                normalized_size,
                request.leverage,
            )

        self._validate_minimums(
            size=normalized_size,
            notional_value=notional_value,
            constraints=constraints,
        )

        self._logger.info(
            "Position size calculated | symbol=%s tier=%s size=%s notional=%s margin=%s risk_amount=%s leverage=%s",
            request.symbol,
            request.tier_profile.final_tier.value,
            normalized_size,
            notional_value,
            margin_required,
            request.risk_amount,
            request.leverage,
            extra={
                "symbol": request.symbol,
                "tier": request.tier_profile.final_tier.value,
            },
        )

        return PositionSizeResult(
            size=normalized_size,
            notional_value=notional_value,
            margin_required=margin_required,
            risk_amount=request.risk_amount,
            risk_unit_used=request.risk_unit_snapshot.effective_risk_unit,
            risk_units_used=request.tier_profile.risk_units,
            leverage_used=request.leverage,
            tier=request.tier_profile.final_tier,
            stop_distance=stop_distance,
            capped=capped,
            rejected_by_min_size=False,
            reason=", ".join(dict.fromkeys(reason_parts)) if reason_parts else None,
            metadata={
                "raw_size": raw_size,
                "account_equity": request.account_equity,
                "free_balance": request.free_balance,
                "requested_size": request.requested_size,
                "requested_margin": request.requested_margin,
                "risk_unit_snapshot": request.risk_unit_snapshot,
                "tier_profile": request.tier_profile,
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
                adjusted_margin=result.margin_required,
                adjusted_leverage=result.leverage_used,
                adjusted_risk_amount=result.risk_amount,
                adjusted_tier=result.tier,
                metadata={
                    "notional_value": result.notional_value,
                    "margin_required": result.margin_required,
                    "risk_amount": result.risk_amount,
                    "risk_unit_used": result.risk_unit_used,
                    "risk_units_used": result.risk_units_used,
                    "tier": result.tier.value,
                    "stop_distance": result.stop_distance,
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
                        symbol=request.symbol,
                        tier=request.tier_profile.final_tier,
                    )
                ],
                reason=str(exc),
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
                        symbol=getattr(request, "symbol", None),
                        tier=(
                            request.tier_profile.final_tier
                            if getattr(request, "tier_profile", None)
                            else None
                        ),
                    )
                ],
                reason=str(exc),
            )

        except ValueError as exc:
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.STOP_DISTANCE_INVALID,
                        level=RiskLevel.CRITICAL,
                        message=str(exc),
                        symbol=request.symbol,
                        tier=request.tier_profile.final_tier,
                    )
                ],
                reason=str(exc),
            )

    def normalize_size(
        self,
        symbol: str,
        raw_size: float,
        constraints: SymbolConstraints | None = None,
    ) -> tuple[float, bool, list[str]]:
        """
        Normalize size safely.

        Важливо: ця функція ніколи не піднімає size до min_size, якщо
        reject_if_below_min_size=True або never_increase_size_above_risk=True.
        Це захищає від непомітного збільшення risk.
        """
        if raw_size < 0:
            return 0.0, False, ["raw_size_negative"]

        constraints = constraints or self._symbol_constraints.get(symbol, SymbolConstraints())

        size = raw_size
        capped = False
        reasons: list[str] = []

        if self._config.max_position_size is not None and size > self._config.max_position_size:
            size = self._config.max_position_size
            capped = True
            reasons.append("global_max_position_size_cap_applied")

        if constraints.max_size is not None and size > constraints.max_size:
            size = constraints.max_size
            capped = True
            reasons.append("symbol_max_size_cap_applied")

        size = apply_cap(size, constraints.max_size)

        rounded_size = round_down_to_step(size, constraints.step_size)
        if rounded_size < size:
            capped = True
            reasons.append("step_size_round_down_applied")
        size = rounded_size

        min_size = self._effective_min_size(constraints)

        if min_size is not None and size < min_size:
            if self._config.reject_if_below_min_size or self._config.never_increase_size_above_risk:
                raise InvalidPositionSizeError(
                    f"Calculated size {size:.8f} is below min_size {min_size:.8f}"
                )

            size = min_size
            capped = True
            reasons.append("min_size_floor_applied")

        if size < 0:
            return 0.0, capped, reasons

        return size, capped, reasons

    def _resolve_stop_distance(self, request: PositionSizeRequest) -> float:
        stop_distance = calculate_side_aware_stop_distance(
            side=request.side,
            entry_price=request.entry_price,
            stop_loss=request.stop_loss,
        )

        if stop_distance is None:
            if self._config.require_stop_loss:
                raise InvalidPositionSizeError("Stop loss is required for position sizing")

            if (
                self._config.fallback_stop_loss_pct is None
                or self._config.fallback_stop_loss_pct <= 0
            ):
                raise InvalidPositionSizeError(
                    "Stop loss is missing and no valid fallback_stop_loss_pct configured"
                )

            stop_distance = request.entry_price * self._config.fallback_stop_loss_pct

        if stop_distance <= 0:
            raise InvalidPositionSizeError("Stop distance must be > 0")

        return stop_distance

    def _resolve_constraints(self, request: PositionSizeRequest) -> SymbolConstraints:
        base = self._symbol_constraints.get(request.symbol, SymbolConstraints())

        min_size = base.min_size
        max_size = base.max_size
        step_size = base.step_size
        min_notional = base.min_notional
        max_notional = base.max_notional

        if request.min_size is not None:
            min_size = max(min_size or 0.0, request.min_size)

        if request.max_size is not None:
            max_size = min(max_size, request.max_size) if max_size is not None else request.max_size

        if request.step_size is not None:
            step_size = request.step_size

        if request.min_notional is not None:
            min_notional = max(min_notional or 0.0, request.min_notional)

        return SymbolConstraints(
            min_size=min_size,
            max_size=max_size,
            step_size=step_size,
            min_notional=min_notional,
            max_notional=max_notional,
        )

    def _validate_minimums(
        self,
        *,
        size: float,
        notional_value: float,
        constraints: SymbolConstraints,
    ) -> None:
        min_size = self._effective_min_size(constraints)

        if min_size is not None and size < min_size:
            raise InvalidPositionSizeError(
                f"Calculated size {size:.8f} is below min_size {min_size:.8f}"
            )

        if constraints.min_notional is not None and notional_value < constraints.min_notional:
            raise InvalidPositionSizeError(
                f"Calculated notional {notional_value:.8f} is below min_notional "
                f"{constraints.min_notional:.8f}"
            )

        if constraints.max_notional is not None and notional_value > constraints.max_notional:
            raise InvalidPositionSizeError(
                f"Calculated notional {notional_value:.8f} exceeds max_notional "
                f"{constraints.max_notional:.8f}"
            )

    def _effective_min_size(self, constraints: SymbolConstraints) -> float | None:
        values = [
            self._config.min_position_size if self._config.min_position_size > 0 else None,
            constraints.min_size,
        ]
        values = [value for value in values if value is not None]
        if not values:
            return None
        return max(values)

    @staticmethod
    def _size_by_available_margin(
        *,
        entry_price: float,
        free_balance: float,
        leverage: float | None,
    ) -> float:
        if entry_price <= 0:
            return 0.0

        if free_balance <= 0:
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

        if state.equity < 0:
            raise InvalidRiskRequestError("risk state equity must be >= 0")

        if request.risk_amount <= 0:
            raise InvalidRiskRequestError("risk_amount must be > 0")

        if request.leverage <= 0:
            raise InvalidRiskRequestError("leverage must be > 0")

        if request.risk_unit_snapshot.effective_risk_unit < 0:
            raise InvalidRiskRequestError("effective_risk_unit must be >= 0")

        if request.tier_profile.risk_units <= 0:
            raise InvalidRiskRequestError("tier risk_units must be > 0")


__all__ = [
    "PositionSizer",
    "RiskUnitCalculator",
    "SymbolConstraints",
]