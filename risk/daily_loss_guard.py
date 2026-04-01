from __future__ import annotations

from core.logger import get_logger

from risk.config import DailyLossConfig
from risk.enums import RiskDecisionType, RiskLevel, RiskViolationType, TradingMode
from risk.models import RiskCheckResult, RiskViolation
from risk.state import RiskState


class DailyLossGuard:
    """
    Контролює денний ліміт втрат.
    Логіка базується на падінні equity від daily_start_equity.
    """

    def __init__(
        self,
        config: DailyLossConfig,
        *,
        service_name: str = "risk.daily_loss_guard",
    ) -> None:
        self._config = config
        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="daily_loss_guard",
        )

    def check(self, state: RiskState) -> RiskCheckResult:
        base_equity = state.daily_start_equity

        if base_equity <= 0:
            return RiskCheckResult(
                passed=True,
                decision=RiskDecisionType.ALLOW,
                metadata={
                    "daily_loss_pct": 0.0,
                    "daily_pnl": state.get_daily_pnl(),
                },
            )

        daily_pnl = state.get_daily_pnl()
        daily_loss_pct = abs(min(0.0, daily_pnl)) / base_equity

        if daily_loss_pct >= self._config.max_daily_loss_pct:
            message = (
                f"Daily loss limit exceeded: current={daily_loss_pct:.4f}, "
                f"limit={self._config.max_daily_loss_pct:.4f}"
            )

            self._logger.error(
                "Daily loss limit breached | daily_loss_pct=%s limit=%s daily_pnl=%s",
                daily_loss_pct,
                self._config.max_daily_loss_pct,
                daily_pnl,
            )

            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.HALT_TRADING,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.DAILY_LOSS_EXCEEDED,
                        level=RiskLevel.CRITICAL,
                        message=message,
                        current_value=daily_loss_pct,
                        limit_value=self._config.max_daily_loss_pct,
                    )
                ],
                metadata={
                    "trading_mode": TradingMode.HALTED.value,
                    "daily_loss_pct": daily_loss_pct,
                    "daily_pnl": daily_pnl,
                },
            )

        return RiskCheckResult(
            passed=True,
            decision=RiskDecisionType.ALLOW,
            metadata={
                "daily_loss_pct": daily_loss_pct,
                "daily_pnl": daily_pnl,
            },
        )