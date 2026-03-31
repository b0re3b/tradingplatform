from __future__ import annotations

from core.logger import get_logger

from risk.config import ExposureConfig
from risk.enums import RiskDecisionType, RiskLevel, RiskViolationType
from risk.models import RiskCheckResult, RiskEvaluationRequest, RiskViolation
from risk.state import RiskState
from risk.utils import calculate_notional, safe_div


class ExposureControl:
    """
    Контролює сумарний та projected exposure портфеля.
    Працює через notionals, нормалізовані відносно equity.
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
    ) -> RiskCheckResult:
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
            )

        equity = state.equity
        if equity <= 0:
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
            )

        snapshot = state.get_exposure_snapshot()
        new_notional = calculate_notional(request.entry_price, candidate_size)

        projected_total = snapshot.gross_exposure + new_notional
        projected_symbol = snapshot.symbol_exposure.get(request.symbol, 0.0) + new_notional
        projected_side = snapshot.side_exposure.get(request.side.value, 0.0) + new_notional
        projected_positions_count = len(state.positions) + 1

        total_exposure_pct = safe_div(projected_total, equity)
        symbol_exposure_pct = safe_div(projected_symbol, equity)
        side_exposure_pct = safe_div(projected_side, equity)

        violations: list[RiskViolation] = []

        if projected_positions_count > self._config.max_open_positions:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.MAX_OPEN_POSITIONS_EXCEEDED,
                    level=RiskLevel.CRITICAL,
                    message="Maximum number of open positions would be exceeded",
                    current_value=float(projected_positions_count),
                    limit_value=float(self._config.max_open_positions),
                )
            )

        if total_exposure_pct > self._config.max_total_exposure_pct:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.MAX_EXPOSURE_EXCEEDED,
                    level=RiskLevel.CRITICAL,
                    message="Maximum total portfolio exposure would be exceeded",
                    current_value=total_exposure_pct,
                    limit_value=self._config.max_total_exposure_pct,
                )
            )

        if symbol_exposure_pct > self._config.max_symbol_exposure_pct:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.MAX_SYMBOL_EXPOSURE_EXCEEDED,
                    level=RiskLevel.CRITICAL,
                    message=f"Maximum exposure for symbol {request.symbol} would be exceeded",
                    current_value=symbol_exposure_pct,
                    limit_value=self._config.max_symbol_exposure_pct,
                    metadata={"symbol": request.symbol},
                )
            )

        if side_exposure_pct > self._config.max_side_exposure_pct:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.MAX_SIDE_EXPOSURE_EXCEEDED,
                    level=RiskLevel.CRITICAL,
                    message=f"Maximum exposure for side {request.side.value} would be exceeded",
                    current_value=side_exposure_pct,
                    limit_value=self._config.max_side_exposure_pct,
                    metadata={"side": request.side.value},
                )
            )

        if violations:
            self._logger.warning(
                "Exposure check failed | symbol=%s side=%s projected_total_pct=%s projected_symbol_pct=%s projected_side_pct=%s",
                request.symbol,
                request.side.value,
                total_exposure_pct,
                symbol_exposure_pct,
                side_exposure_pct,
                extra={
                    "symbol": request.symbol,
                },
            )
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=violations,
                metadata={
                    "projected_total_exposure_pct": total_exposure_pct,
                    "projected_symbol_exposure_pct": symbol_exposure_pct,
                    "projected_side_exposure_pct": side_exposure_pct,
                    "projected_positions_count": projected_positions_count,
                    "new_notional": new_notional,
                },
            )

        self._logger.info(
            "Exposure check passed | symbol=%s side=%s projected_total_pct=%s",
            request.symbol,
            request.side.value,
            total_exposure_pct,
            extra={
                "symbol": request.symbol,
            },
        )

        return RiskCheckResult(
            passed=True,
            decision=RiskDecisionType.ALLOW,
            metadata={
                "projected_total_exposure_pct": total_exposure_pct,
                "projected_symbol_exposure_pct": symbol_exposure_pct,
                "projected_side_exposure_pct": side_exposure_pct,
                "projected_positions_count": projected_positions_count,
                "new_notional": new_notional,
            },
        )