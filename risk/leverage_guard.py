from __future__ import annotations

from core.logger import get_logger

from risk.config import LeverageConfig
from risk.enums import RiskDecisionType, RiskLevel, RiskViolationType, TradingMode
from risk.models import RiskCheckResult, RiskEvaluationRequest, RiskViolation
from risk.state import RiskState


class LeverageGuard:
    """
    Контролює допустиме плече:
    - глобальний max leverage
    - per-symbol leverage
    - safe mode leverage cap
    """

    def __init__(
        self,
        config: LeverageConfig,
        *,
        service_name: str = "risk.leverage_guard",
    ) -> None:
        self._config = config
        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="leverage_guard",
        )

    def check(
        self,
        request: RiskEvaluationRequest,
        state: RiskState,
        *,
        candidate_leverage: float | None,
    ) -> RiskCheckResult:
        requested_leverage = candidate_leverage or request.requested_leverage or 1.0

        if requested_leverage <= 0:
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.MAX_LEVERAGE_EXCEEDED,
                        level=RiskLevel.CRITICAL,
                        message="Requested leverage must be > 0",
                        current_value=requested_leverage,
                        limit_value=self._config.max_leverage,
                    )
                ],
            )

        max_allowed = self._config.max_leverage

        symbol_cap = self._config.max_leverage_per_symbol.get(request.symbol)
        if symbol_cap is not None:
            max_allowed = min(max_allowed, symbol_cap)

        if (
            state.trading_mode is TradingMode.SAFE_MODE
            and self._config.reduce_leverage_in_safe_mode
        ):
            max_allowed = min(max_allowed, self._config.safe_mode_max_leverage)

        if requested_leverage > max_allowed:
            self._logger.warning(
                "Leverage capped | symbol=%s requested=%s allowed=%s",
                request.symbol,
                requested_leverage,
                max_allowed,
                extra={"symbol": request.symbol},
            )
            return RiskCheckResult(
                passed=True,
                decision=RiskDecisionType.REDUCE_SIZE,
                adjusted_leverage=max_allowed,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.MAX_LEVERAGE_EXCEEDED,
                        level=RiskLevel.WARNING,
                        message="Requested leverage exceeds allowed limit and was capped",
                        current_value=requested_leverage,
                        limit_value=max_allowed,
                        metadata={"symbol": request.symbol},
                    )
                ],
                metadata={
                    "requested_leverage": requested_leverage,
                    "allowed_leverage": max_allowed,
                },
            )

        return RiskCheckResult(
            passed=True,
            decision=RiskDecisionType.ALLOW,
            adjusted_leverage=requested_leverage,
            metadata={
                "requested_leverage": requested_leverage,
                "allowed_leverage": max_allowed,
            },
        )