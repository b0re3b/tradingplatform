from __future__ import annotations

from core.logger import get_logger

from risk.config import CorrelationConfig
from risk.enums import RiskDecisionType, RiskLevel, RiskViolationType
from risk.models import RiskCheckResult, RiskEvaluationRequest, RiskViolation
from risk.state import RiskState
from risk.utils import calculate_notional, safe_div


class CorrelationGuard:
    """
    Контролює cluster/group exposure для корельованих активів.
    Реалізація базується на статичних групах з config.
    """

    def __init__(
        self,
        config: CorrelationConfig,
        *,
        service_name: str = "risk.correlation_guard",
    ) -> None:
        self._config = config
        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="correlation_guard",
        )

    def check(
        self,
        request: RiskEvaluationRequest,
        state: RiskState,
        *,
        candidate_size: float,
    ) -> RiskCheckResult:
        if not self._config.enabled:
            return RiskCheckResult(
                passed=True,
                decision=RiskDecisionType.ALLOW,
                metadata={"enabled": False},
            )

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
                        message="risk state equity must be > 0 for correlation checks",
                    )
                ],
            )

        snapshot = state.get_correlation_snapshot(self._config.groups)
        group_name = snapshot.symbol_to_group.get(request.symbol)

        if group_name is None:
            return RiskCheckResult(
                passed=True,
                decision=RiskDecisionType.ALLOW,
                metadata={
                    "enabled": True,
                    "symbol_group": None,
                },
            )

        new_notional = calculate_notional(request.entry_price, candidate_size)
        projected_group_exposure = snapshot.group_exposure.get(group_name, 0.0) + new_notional
        projected_group_exposure_pct = safe_div(projected_group_exposure, equity)

        if projected_group_exposure_pct > self._config.max_group_exposure_pct:
            self._logger.warning(
                "Correlation/group limit exceeded | symbol=%s group=%s current_pct=%s limit=%s",
                request.symbol,
                group_name,
                projected_group_exposure_pct,
                self._config.max_group_exposure_pct,
                extra={"symbol": request.symbol},
            )
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.CORRELATION_LIMIT_EXCEEDED,
                        level=RiskLevel.CRITICAL,
                        message="Correlation group exposure would exceed configured limit",
                        current_value=projected_group_exposure_pct,
                        limit_value=self._config.max_group_exposure_pct,
                        metadata={
                            "symbol": request.symbol,
                            "group": group_name,
                        },
                    )
                ],
                metadata={
                    "group": group_name,
                    "projected_group_exposure": projected_group_exposure,
                    "projected_group_exposure_pct": projected_group_exposure_pct,
                    "new_notional": new_notional,
                },
            )

        return RiskCheckResult(
            passed=True,
            decision=RiskDecisionType.ALLOW,
            metadata={
                "group": group_name,
                "projected_group_exposure": projected_group_exposure,
                "projected_group_exposure_pct": projected_group_exposure_pct,
                "new_notional": new_notional,
            },
        )