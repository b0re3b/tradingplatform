from __future__ import annotations

from typing import Any

from core.logger import get_logger

from risk.config import ExposureConfig
from risk.enums import OrderIntent, RiskDecisionType, RiskLevel, RiskMode, RiskViolationType
from risk.models import ExposureSnapshot, OpenRiskSnapshot, RiskCheckResult, RiskEvaluationRequest, RiskViolation
from risk.state import RiskState
from risk.utils import (
    calculate_margin_required,
    calculate_notional,
    is_finite_number,
    safe_div,
)


class ExposureControl:
    """
    Portfolio exposure and capital usage guard.

    Responsibilities:
    - portfolio open-risk limit in R;
    - used margin limit;
    - total notional exposure;
    - symbol exposure;
    - side exposure;
    - max open positions / pending risk reservations;
    - correlation group exposure and open risk.

    This guard is intentionally read-only. It does not mutate RiskState and
    does not publish EventBus events. RiskManager owns locking, reservations,
    EventBus emits and lifecycle.

    Important production invariant:
    state.get_exposure_snapshot(), state.get_open_risk_snapshot() and
    state.get_correlation_snapshot() are expected to include pending risk
    reservations. That makes this guard conservative between RiskManager ALLOW
    and the actual position.opened event.
    """

    def __init__(
        self,
        config: ExposureConfig,
        *,
        service_name: str = "risk.exposure_control",
    ) -> None:
        self._config = config
        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="exposure_control",
        )

    def check(
        self,
        request: RiskEvaluationRequest,
        state: RiskState,
        *,
        candidate_size: float,
        candidate_open_risk: float,
        candidate_leverage: float,
        risk_unit: float,
        mode: RiskMode | None = None,
        candidate_margin: float | None = None,
    ) -> RiskCheckResult:
        """
        Check projected portfolio exposure.

        candidate_open_risk represents max loss to stop-loss for the candidate
        order. In the R-based model this is more important than raw notional.

        Pending reservations are accounted through RiskState snapshots. The
        candidate order is then added on top of actual + pending exposure.
        """
        risk_mode = mode or state.risk_mode

        if request.order_intent.reduces_risk:
            return RiskCheckResult(
                passed=True,
                decision=RiskDecisionType.ALLOW,
                risk_mode=risk_mode,
                metadata={
                    "reduce_order_allowed": True,
                    "order_intent": request.order_intent.value,
                },
            )

        validation_error = self._validate_inputs(
            candidate_size=candidate_size,
            candidate_open_risk=candidate_open_risk,
            candidate_leverage=candidate_leverage,
            risk_unit=risk_unit,
            state=state,
        )
        if validation_error is not None:
            return validation_error

        candidate_notional = calculate_notional(request.entry_price, candidate_size)
        resolved_candidate_margin = (
            candidate_margin
            if candidate_margin is not None
            else calculate_margin_required(
                request.entry_price,
                candidate_size,
                candidate_leverage,
            )
        )

        validation_error = self._validate_projected_amounts(
            candidate_notional=candidate_notional,
            candidate_margin=resolved_candidate_margin,
        )
        if validation_error is not None:
            return validation_error

        exposure_snapshot = state.get_exposure_snapshot()
        open_risk_snapshot = state.get_open_risk_snapshot(risk_unit=risk_unit)

        # Snapshots are expected to include pending reservations. Candidate is
        # the only extra risk/exposure added here.
        current_open_risk = self._snapshot_total_open_risk(open_risk_snapshot)
        current_margin = self._snapshot_margin_used(exposure_snapshot, open_risk_snapshot)
        current_gross_exposure = exposure_snapshot.gross_exposure

        projected_open_risk = current_open_risk + candidate_open_risk
        projected_open_risk_r = safe_div(projected_open_risk, risk_unit)

        projected_margin = current_margin + resolved_candidate_margin
        projected_margin_pct = safe_div(projected_margin, state.equity)

        projected_total_exposure = current_gross_exposure + candidate_notional
        projected_total_exposure_pct = safe_div(projected_total_exposure, state.equity)

        projected_symbol_exposure = (
            exposure_snapshot.symbol_exposure.get(request.symbol, 0.0)
            + candidate_notional
        )
        projected_symbol_exposure_pct = safe_div(projected_symbol_exposure, state.equity)

        projected_side_exposure = (
            exposure_snapshot.side_exposure.get(request.side.value, 0.0)
            + candidate_notional
        )
        projected_side_exposure_pct = safe_div(projected_side_exposure, state.equity)

        projected_positions_count = self._projected_positions_count(request, state)

        violations: list[RiskViolation] = []

        self._check_open_risk(
            request=request,
            mode=risk_mode,
            projected_open_risk_r=projected_open_risk_r,
            violations=violations,
        )

        self._check_used_margin(
            request=request,
            mode=risk_mode,
            projected_margin_pct=projected_margin_pct,
            violations=violations,
        )

        self._check_open_positions(
            request=request,
            projected_positions_count=projected_positions_count,
            violations=violations,
        )

        self._check_notional_exposure(
            request=request,
            projected_total_exposure_pct=projected_total_exposure_pct,
            projected_symbol_exposure_pct=projected_symbol_exposure_pct,
            projected_side_exposure_pct=projected_side_exposure_pct,
            violations=violations,
        )

        correlation_metadata = self._check_correlation_group(
            request=request,
            state=state,
            candidate_notional=candidate_notional,
            candidate_open_risk=candidate_open_risk,
            risk_unit=risk_unit,
            violations=violations,
        )

        metadata = self._build_metadata(
            request=request,
            risk_mode=risk_mode,
            exposure_snapshot=exposure_snapshot,
            open_risk_snapshot=open_risk_snapshot,
            candidate_size=candidate_size,
            candidate_notional=candidate_notional,
            candidate_margin=resolved_candidate_margin,
            candidate_open_risk=candidate_open_risk,
            risk_unit=risk_unit,
            projected_open_risk=projected_open_risk,
            projected_open_risk_r=projected_open_risk_r,
            projected_margin=projected_margin,
            projected_margin_pct=projected_margin_pct,
            projected_total_exposure=projected_total_exposure,
            projected_total_exposure_pct=projected_total_exposure_pct,
            projected_symbol_exposure=projected_symbol_exposure,
            projected_symbol_exposure_pct=projected_symbol_exposure_pct,
            projected_side_exposure=projected_side_exposure,
            projected_side_exposure_pct=projected_side_exposure_pct,
            projected_positions_count=projected_positions_count,
            correlation_metadata=correlation_metadata,
        )

        if violations:
            self._logger.warning(
                "Exposure check failed | symbol=%s side=%s violations=%s open_risk_r=%s margin_pct=%s total_exposure_pct=%s pending_reservations=%s",
                request.symbol,
                request.side.value,
                len(violations),
                projected_open_risk_r,
                projected_margin_pct,
                projected_total_exposure_pct,
                self._pending_reservations_count(state),
                extra={
                    "symbol": request.symbol,
                    "side": request.side.value,
                    "risk_mode": risk_mode.value,
                    "pending_reservations": self._pending_reservations_count(state),
                },
            )
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=violations,
                risk_mode=risk_mode,
                reason="Exposure limits exceeded",
                metadata=metadata,
            )

        self._logger.info(
            "Exposure check passed | symbol=%s side=%s open_risk_r=%s margin_pct=%s total_exposure_pct=%s pending_reservations=%s",
            request.symbol,
            request.side.value,
            projected_open_risk_r,
            projected_margin_pct,
            projected_total_exposure_pct,
            self._pending_reservations_count(state),
            extra={
                "symbol": request.symbol,
                "side": request.side.value,
                "risk_mode": risk_mode.value,
                "pending_reservations": self._pending_reservations_count(state),
            },
        )

        return RiskCheckResult(
            passed=True,
            decision=RiskDecisionType.ALLOW,
            risk_mode=risk_mode,
            metadata=metadata,
        )

    def _check_open_risk(
        self,
        *,
        request: RiskEvaluationRequest,
        mode: RiskMode,
        projected_open_risk_r: float,
        violations: list[RiskViolation],
    ) -> None:
        limit = self._resolve_open_risk_limit(mode)

        if projected_open_risk_r > limit:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.OPEN_RISK_EXCEEDED,
                    level=RiskLevel.CRITICAL,
                    message="Maximum portfolio open risk would be exceeded",
                    current_value=projected_open_risk_r,
                    limit_value=limit,
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=request.tier,
                    metadata={"risk_mode": mode.value},
                )
            )

    def _check_used_margin(
        self,
        *,
        request: RiskEvaluationRequest,
        mode: RiskMode,
        projected_margin_pct: float,
        violations: list[RiskViolation],
    ) -> None:
        limit = self._resolve_used_margin_limit(mode)

        if projected_margin_pct > limit:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.USED_MARGIN_EXCEEDED,
                    level=RiskLevel.CRITICAL,
                    message="Maximum used margin would be exceeded",
                    current_value=projected_margin_pct,
                    limit_value=limit,
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=request.tier,
                    metadata={"risk_mode": mode.value},
                )
            )

    def _check_open_positions(
        self,
        *,
        request: RiskEvaluationRequest,
        projected_positions_count: int,
        violations: list[RiskViolation],
    ) -> None:
        if self._config.max_open_positions is None:
            return

        if projected_positions_count > self._config.max_open_positions:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.MAX_OPEN_POSITIONS_EXCEEDED,
                    level=RiskLevel.CRITICAL,
                    message="Maximum number of open or pending positions would be exceeded",
                    current_value=float(projected_positions_count),
                    limit_value=float(self._config.max_open_positions),
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=request.tier,
                )
            )

    def _check_notional_exposure(
        self,
        *,
        request: RiskEvaluationRequest,
        projected_total_exposure_pct: float,
        projected_symbol_exposure_pct: float,
        projected_side_exposure_pct: float,
        violations: list[RiskViolation],
    ) -> None:
        if projected_total_exposure_pct > self._config.max_total_exposure_pct:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.MAX_EXPOSURE_EXCEEDED,
                    level=RiskLevel.CRITICAL,
                    message="Maximum total portfolio notional exposure would be exceeded",
                    current_value=projected_total_exposure_pct,
                    limit_value=self._config.max_total_exposure_pct,
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=request.tier,
                )
            )

        if projected_symbol_exposure_pct > self._config.max_symbol_exposure_pct:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.MAX_SYMBOL_EXPOSURE_EXCEEDED,
                    level=RiskLevel.CRITICAL,
                    message="Maximum symbol notional exposure would be exceeded",
                    current_value=projected_symbol_exposure_pct,
                    limit_value=self._config.max_symbol_exposure_pct,
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=request.tier,
                )
            )

        if projected_side_exposure_pct > self._config.max_side_exposure_pct:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.MAX_SIDE_EXPOSURE_EXCEEDED,
                    level=RiskLevel.CRITICAL,
                    message="Maximum side notional exposure would be exceeded",
                    current_value=projected_side_exposure_pct,
                    limit_value=self._config.max_side_exposure_pct,
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=request.tier,
                    metadata={"side": request.side.value},
                )
            )

    def _check_correlation_group(
        self,
        *,
        request: RiskEvaluationRequest,
        state: RiskState,
        candidate_notional: float,
        candidate_open_risk: float,
        risk_unit: float,
        violations: list[RiskViolation],
    ) -> dict[str, object]:
        if not self._config.correlation_groups:
            return {
                "correlation_group": None,
                "current_correlation_group_exposure": None,
                "current_correlation_group_open_risk": None,
                "projected_correlation_group_exposure": None,
                "projected_correlation_group_open_risk": None,
                "projected_correlation_group_exposure_pct": None,
                "projected_correlation_group_open_risk_r": None,
            }

        # RiskState.get_correlation_snapshot() is expected to include pending
        # reservations. We only add the current candidate here.
        snapshot = state.get_correlation_snapshot(self._config.correlation_groups)
        group_name = snapshot.symbol_to_group.get(request.symbol)

        if group_name is None:
            return {
                "correlation_group": None,
                "current_correlation_group_exposure": None,
                "current_correlation_group_open_risk": None,
                "projected_correlation_group_exposure": None,
                "projected_correlation_group_open_risk": None,
                "projected_correlation_group_exposure_pct": None,
                "projected_correlation_group_open_risk_r": None,
            }

        current_group_exposure = snapshot.group_exposure.get(group_name, 0.0)
        current_group_open_risk = snapshot.group_open_risk.get(group_name, 0.0)

        projected_group_exposure = current_group_exposure + candidate_notional
        projected_group_open_risk = current_group_open_risk + candidate_open_risk

        projected_group_exposure_pct = safe_div(projected_group_exposure, state.equity)
        projected_group_open_risk_r = safe_div(projected_group_open_risk, risk_unit)

        if projected_group_exposure_pct > self._config.max_correlation_group_exposure_pct:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.CORRELATION_LIMIT_EXCEEDED,
                    level=RiskLevel.CRITICAL,
                    message="Correlation group notional exposure would be exceeded",
                    current_value=projected_group_exposure_pct,
                    limit_value=self._config.max_correlation_group_exposure_pct,
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=request.tier,
                    metadata={"group": group_name},
                )
            )

        if (
            self._config.max_correlation_group_open_risk_r is not None
            and projected_group_open_risk_r > self._config.max_correlation_group_open_risk_r
        ):
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.CORRELATION_LIMIT_EXCEEDED,
                    level=RiskLevel.CRITICAL,
                    message="Correlation group open risk would be exceeded",
                    current_value=projected_group_open_risk_r,
                    limit_value=self._config.max_correlation_group_open_risk_r,
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=request.tier,
                    metadata={"group": group_name},
                )
            )

        return {
            "correlation_group": group_name,
            "current_correlation_group_exposure": current_group_exposure,
            "current_correlation_group_open_risk": current_group_open_risk,
            "projected_correlation_group_exposure": projected_group_exposure,
            "projected_correlation_group_open_risk": projected_group_open_risk,
            "projected_correlation_group_exposure_pct": projected_group_exposure_pct,
            "projected_correlation_group_open_risk_r": projected_group_open_risk_r,
        }

    def _resolve_open_risk_limit(self, mode: RiskMode) -> float:
        if mode is RiskMode.SAFE_MODE:
            return self._config.safe_mode_max_open_risk_r

        if mode is RiskMode.CAUTION:
            return self._config.max_open_risk_r

        if mode is RiskMode.NORMAL:
            return self._config.max_open_risk_r

        if mode is RiskMode.REDUCE_ONLY:
            return 0.0

        if mode in {RiskMode.HALTED, RiskMode.EMERGENCY_STOP}:
            return 0.0

        return self._config.max_open_risk_r

    def _resolve_used_margin_limit(self, mode: RiskMode) -> float:
        if mode is RiskMode.SAFE_MODE:
            return self._config.safe_mode_max_used_margin_pct

        if mode is RiskMode.CAUTION:
            return self._config.max_used_margin_pct

        if mode is RiskMode.NORMAL:
            return self._config.max_used_margin_pct

        if mode is RiskMode.REDUCE_ONLY:
            return 0.0

        if mode in {RiskMode.HALTED, RiskMode.EMERGENCY_STOP}:
            return 0.0

        return self._config.max_used_margin_pct

    def _projected_positions_count(
        self,
        request: RiskEvaluationRequest,
        state: RiskState,
    ) -> int:
        """
        Conservative position count using actual positions + pending reservations.

        INCREASE does not add a new position, but OPEN does. Pending reservations
        are counted because they may become positions after execution accepts the
        already-approved signal.
        """
        current_count = len(state.positions) + self._pending_reservations_count(state)

        if request.order_intent in {OrderIntent.REDUCE, OrderIntent.CLOSE}:
            return current_count

        if request.order_intent is OrderIntent.INCREASE:
            return current_count

        return current_count + 1

    def _build_metadata(
        self,
        *,
        request: RiskEvaluationRequest,
        risk_mode: RiskMode,
        exposure_snapshot: ExposureSnapshot,
        open_risk_snapshot: OpenRiskSnapshot,
        candidate_size: float,
        candidate_notional: float,
        candidate_margin: float,
        candidate_open_risk: float,
        risk_unit: float,
        projected_open_risk: float,
        projected_open_risk_r: float,
        projected_margin: float,
        projected_margin_pct: float,
        projected_total_exposure: float,
        projected_total_exposure_pct: float,
        projected_symbol_exposure: float,
        projected_symbol_exposure_pct: float,
        projected_side_exposure: float,
        projected_side_exposure_pct: float,
        projected_positions_count: int,
        correlation_metadata: dict[str, object],
    ) -> dict[str, Any]:
        current_open_risk = self._snapshot_total_open_risk(open_risk_snapshot)
        actual_open_risk = getattr(open_risk_snapshot, "actual_open_risk", None)
        pending_open_risk = getattr(open_risk_snapshot, "pending_orders_risk", 0.0)
        pending_open_risk_r = getattr(open_risk_snapshot, "pending_orders_risk_r", None)

        return {
            "risk_mode": risk_mode.value,
            "order_intent": request.order_intent.value,
            "candidate_size": candidate_size,
            "candidate_notional": candidate_notional,
            "candidate_margin": candidate_margin,
            "candidate_open_risk": candidate_open_risk,
            "candidate_open_risk_r": safe_div(candidate_open_risk, risk_unit),
            "current_open_risk": current_open_risk,
            "actual_open_risk": actual_open_risk,
            "pending_open_risk": pending_open_risk,
            "pending_open_risk_r": pending_open_risk_r,
            "pending_margin": getattr(exposure_snapshot, "pending_margin", None),
            "pending_notional": getattr(exposure_snapshot, "pending_notional", None),
            "pending_reservations_count": getattr(
                open_risk_snapshot,
                "pending_reservations_count",
                None,
            ),
            "current_margin": self._snapshot_margin_used(exposure_snapshot, open_risk_snapshot),
            "current_total_exposure": exposure_snapshot.gross_exposure,
            "projected_open_risk": projected_open_risk,
            "projected_open_risk_r": projected_open_risk_r,
            "projected_margin": projected_margin,
            "projected_margin_pct": projected_margin_pct,
            "projected_total_exposure": projected_total_exposure,
            "projected_total_exposure_pct": projected_total_exposure_pct,
            "projected_symbol_exposure": projected_symbol_exposure,
            "projected_symbol_exposure_pct": projected_symbol_exposure_pct,
            "projected_side_exposure": projected_side_exposure,
            "projected_side_exposure_pct": projected_side_exposure_pct,
            "projected_positions_count": projected_positions_count,
            **correlation_metadata,
        }

    @staticmethod
    def _snapshot_total_open_risk(snapshot: OpenRiskSnapshot) -> float:
        # New model: total_open_risk already means actual + pending.
        return snapshot.total_open_risk

    @staticmethod
    def _snapshot_margin_used(
        exposure_snapshot: ExposureSnapshot,
        open_risk_snapshot: OpenRiskSnapshot,
    ) -> float:
        # Prefer ExposureSnapshot, because the fixed RiskState adds pending
        # reservation margin there as part of conservative exposure accounting.
        if exposure_snapshot.margin_used > 0:
            return exposure_snapshot.margin_used
        return open_risk_snapshot.used_margin

    @staticmethod
    def _pending_reservations_count(state: RiskState) -> int:
        reservations = getattr(state, "pending_reservations", None)
        if reservations is None:
            return 0
        return len(reservations)

    @staticmethod
    def _validate_inputs(
        *,
        candidate_size: float,
        candidate_open_risk: float,
        candidate_leverage: float,
        risk_unit: float,
        state: RiskState,
    ) -> RiskCheckResult | None:
        if candidate_size <= 0 or not is_finite_number(candidate_size):
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.POSITION_SIZE_INVALID,
                        level=RiskLevel.CRITICAL,
                        message="candidate_size must be finite and > 0",
                    )
                ],
                reason="Invalid candidate size",
            )

        if candidate_open_risk < 0 or not is_finite_number(candidate_open_risk):
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.OPEN_RISK_EXCEEDED,
                        level=RiskLevel.CRITICAL,
                        message="candidate_open_risk must be finite and >= 0",
                    )
                ],
                reason="Invalid candidate open risk",
            )

        if candidate_leverage <= 0 or not is_finite_number(candidate_leverage):
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.MAX_LEVERAGE_EXCEEDED,
                        level=RiskLevel.CRITICAL,
                        message="candidate_leverage must be finite and > 0",
                    )
                ],
                reason="Invalid candidate leverage",
            )

        if risk_unit <= 0 or not is_finite_number(risk_unit):
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.INVALID_REQUEST,
                        level=RiskLevel.CRITICAL,
                        message="risk_unit must be finite and > 0 for exposure checks",
                    )
                ],
                reason="Invalid risk unit",
            )

        if state.equity <= 0 or not is_finite_number(state.equity):
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.INVALID_REQUEST,
                        level=RiskLevel.CRITICAL,
                        message="risk state equity must be finite and > 0 for exposure checks",
                    )
                ],
                reason="Invalid state equity",
            )

        return None

    @staticmethod
    def _validate_projected_amounts(
        *,
        candidate_notional: float,
        candidate_margin: float,
    ) -> RiskCheckResult | None:
        if candidate_notional <= 0 or not is_finite_number(candidate_notional):
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.INVALID_REQUEST,
                        level=RiskLevel.CRITICAL,
                        message="candidate_notional must be finite and > 0",
                    )
                ],
                reason="Invalid candidate notional",
            )

        if candidate_margin < 0 or not is_finite_number(candidate_margin):
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.INVALID_REQUEST,
                        level=RiskLevel.CRITICAL,
                        message="candidate_margin must be finite and >= 0",
                    )
                ],
                reason="Invalid candidate margin",
            )

        return None


__all__ = ["ExposureControl"]
