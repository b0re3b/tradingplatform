from __future__ import annotations

from core.logger import get_logger

from risk.config import ExposureConfig
from risk.enums import OrderIntent, RiskDecisionType, RiskLevel, RiskMode, RiskViolationType
from risk.models import RiskCheckResult, RiskEvaluationRequest, RiskViolation
from risk.state import RiskState
from risk.utils import calculate_margin_required, calculate_notional, safe_div


class ExposureControl:
    """
    Portfolio exposure and capital usage guard.

    Responsibilities:
    - open risk limit in R;
    - used margin limit;
    - total notional exposure;
    - symbol exposure;
    - side exposure;
    - max open positions;
    - correlation group exposure and open risk.

    This class does not mutate RiskState and does not publish events.
    RiskManager is responsible for locking, EventBus emits and lifecycle.
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

        candidate_open_risk should represent max loss to stop-loss for
        the candidate order. It is more important than notional exposure
        in this adaptive R-based model.
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
        candidate_margin = (
            candidate_margin
            if candidate_margin is not None
            else calculate_margin_required(
                request.entry_price,
                candidate_size,
                candidate_leverage,
            )
        )

        exposure_snapshot = state.get_exposure_snapshot()
        open_risk_snapshot = state.get_open_risk_snapshot(risk_unit=risk_unit)

        projected_open_risk = open_risk_snapshot.total_open_risk + candidate_open_risk
        projected_open_risk_r = safe_div(projected_open_risk, risk_unit)

        projected_margin = exposure_snapshot.margin_used + candidate_margin
        projected_margin_pct = safe_div(projected_margin, state.equity)

        projected_total_exposure = exposure_snapshot.gross_exposure + candidate_notional
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

        projected_positions_count = self._projected_positions_count(
            request,
            state,
        )

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

        metadata = {
            "risk_mode": risk_mode.value,
            "candidate_size": candidate_size,
            "candidate_notional": candidate_notional,
            "candidate_margin": candidate_margin,
            "candidate_open_risk": candidate_open_risk,
            "candidate_open_risk_r": safe_div(candidate_open_risk, risk_unit),
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

        if violations:
            self._logger.warning(
                "Exposure check failed | symbol=%s side=%s violations=%s open_risk_r=%s margin_pct=%s total_exposure_pct=%s",
                request.symbol,
                request.side.value,
                len(violations),
                projected_open_risk_r,
                projected_margin_pct,
                projected_total_exposure_pct,
                extra={
                    "symbol": request.symbol,
                    "side": request.side.value,
                    "risk_mode": risk_mode.value,
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
            "Exposure check passed | symbol=%s side=%s open_risk_r=%s margin_pct=%s total_exposure_pct=%s",
            request.symbol,
            request.side.value,
            projected_open_risk_r,
            projected_margin_pct,
            projected_total_exposure_pct,
            extra={
                "symbol": request.symbol,
                "side": request.side.value,
                "risk_mode": risk_mode.value,
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
                    message="Maximum number of open positions would be exceeded",
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
                "projected_correlation_group_exposure_pct": None,
                "projected_correlation_group_open_risk_r": None,
            }

        snapshot = state.get_correlation_snapshot(self._config.correlation_groups)
        group_name = snapshot.symbol_to_group.get(request.symbol)

        if group_name is None:
            return {
                "correlation_group": None,
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
        if request.order_intent in {OrderIntent.REDUCE, OrderIntent.CLOSE}:
            return len(state.positions)

        if request.order_intent is OrderIntent.INCREASE:
            return len(state.positions)

        return len(state.positions) + 1

    @staticmethod
    def _validate_inputs(
        *,
        candidate_size: float,
        candidate_open_risk: float,
        candidate_leverage: float,
        risk_unit: float,
        state: RiskState,
    ) -> RiskCheckResult | None:
        if candidate_size <= 0:
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.POSITION_SIZE_INVALID,
                        level=RiskLevel.CRITICAL,
                        message="candidate_size must be > 0",
                    )
                ],
                reason="Invalid candidate size",
            )

        if candidate_open_risk < 0:
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.OPEN_RISK_EXCEEDED,
                        level=RiskLevel.CRITICAL,
                        message="candidate_open_risk must be >= 0",
                    )
                ],
                reason="Invalid candidate open risk",
            )

        if candidate_leverage <= 0:
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.MAX_LEVERAGE_EXCEEDED,
                        level=RiskLevel.CRITICAL,
                        message="candidate_leverage must be > 0",
                    )
                ],
                reason="Invalid candidate leverage",
            )

        if risk_unit <= 0:
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.INVALID_REQUEST,
                        level=RiskLevel.CRITICAL,
                        message="risk_unit must be > 0 for exposure checks",
                    )
                ],
                reason="Invalid risk unit",
            )

        if state.equity <= 0:
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.INVALID_REQUEST,
                        level=RiskLevel.CRITICAL,
                        message="risk state equity must be > 0 for exposure checks",
                    )
                ],
                reason="Invalid state equity",
            )

        return None


__all__ = ["ExposureControl"]