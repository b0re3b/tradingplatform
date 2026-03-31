from __future__ import annotations

from core.logger import get_logger

from risk.config import DrawdownConfig
from risk.enums import RiskDecisionType, RiskLevel, RiskViolationType, TradingMode
from risk.models import RiskCheckResult, RiskViolation
from risk.state import RiskState


class MaxDrawdownGuard:
    """
    Контролює глобальну просадку акаунта від peak equity.
    Може:
    - allow
    - reduce_size (safe mode)
    - halt_trading
    """

    def __init__(
        self,
        config: DrawdownConfig,
        *,
        service_name: str = "risk.max_drawdown_guard",
    ) -> None:
        self._config = config
        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="max_drawdown_guard",
        )

    def check(self, state: RiskState) -> RiskCheckResult:
        snapshot = state.get_drawdown_snapshot()
        drawdown_pct = snapshot.drawdown_percent

        if drawdown_pct >= self._config.max_total_drawdown_pct:
            message = (
                f"Maximum drawdown exceeded: current={drawdown_pct:.4f}, "
                f"limit={self._config.max_total_drawdown_pct:.4f}"
            )

            self._logger.error(
                "Hard drawdown limit breached | drawdown_pct=%s limit=%s",
                drawdown_pct,
                self._config.max_total_drawdown_pct,
            )

            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.HALT_TRADING,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.MAX_DRAWDOWN_EXCEEDED,
                        level=RiskLevel.CRITICAL,
                        message=message,
                        current_value=drawdown_pct,
                        limit_value=self._config.max_total_drawdown_pct,
                    )
                ],
                metadata={
                    "trading_mode": TradingMode.HALTED.value,
                    "drawdown_snapshot": snapshot,
                },
            )

        if drawdown_pct >= self._config.safe_mode_drawdown_pct:
            message = (
                f"Safe mode drawdown threshold reached: current={drawdown_pct:.4f}, "
                f"threshold={self._config.safe_mode_drawdown_pct:.4f}"
            )

            self._logger.warning(
                "Safe mode drawdown threshold reached | drawdown_pct=%s threshold=%s",
                drawdown_pct,
                self._config.safe_mode_drawdown_pct,
            )

            return RiskCheckResult(
                passed=True,
                decision=RiskDecisionType.REDUCE_SIZE,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.SAFE_MODE_ACTIVE,
                        level=RiskLevel.WARNING,
                        message=message,
                        current_value=drawdown_pct,
                        limit_value=self._config.safe_mode_drawdown_pct,
                    )
                ],
                metadata={
                    "trading_mode": TradingMode.SAFE_MODE.value,
                    "drawdown_snapshot": snapshot,
                    "size_multiplier": 0.5,
                },
            )

        if snapshot.loss_streak >= self._config.max_loss_streak:
            message = (
                f"Loss streak limit reached: current={snapshot.loss_streak}, "
                f"limit={self._config.max_loss_streak}"
            )

            self._logger.warning(
                "Loss streak threshold reached | loss_streak=%s limit=%s",
                snapshot.loss_streak,
                self._config.max_loss_streak,
            )

            return RiskCheckResult(
                passed=True,
                decision=RiskDecisionType.REDUCE_SIZE,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.SAFE_MODE_ACTIVE,
                        level=RiskLevel.WARNING,
                        message=message,
                        current_value=float(snapshot.loss_streak),
                        limit_value=float(self._config.max_loss_streak),
                    )
                ],
                metadata={
                    "trading_mode": TradingMode.SAFE_MODE.value,
                    "drawdown_snapshot": snapshot,
                    "size_multiplier": 0.5,
                },
            )

        return RiskCheckResult(
            passed=True,
            decision=RiskDecisionType.ALLOW,
            metadata={
                "drawdown_snapshot": snapshot,
            },
        )