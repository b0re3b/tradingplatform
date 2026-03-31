from __future__ import annotations

from dataclasses import dataclass, field

from risk.exceptions import RiskConfigurationError


@dataclass(slots=True)
class PositionSizingConfig:
    default_risk_per_trade_pct: float = 0.01
    max_risk_per_trade_pct: float = 0.02
    min_position_size: float = 0.0
    max_position_size: float | None = None
    use_volatility_adjustment: bool = True
    use_confidence_adjustment: bool = True
    confidence_scale_min: float = 0.5
    confidence_scale_max: float = 1.25
    fallback_stop_loss_pct: float | None = None
    require_stop_loss: bool = True


@dataclass(slots=True)
class ExposureConfig:
    max_total_exposure_pct: float = 1.0
    max_symbol_exposure_pct: float = 0.30
    max_side_exposure_pct: float = 0.60
    max_open_positions: int = 5
    include_pending_orders: bool = False


@dataclass(slots=True)
class DrawdownConfig:
    max_total_drawdown_pct: float = 0.15
    safe_mode_drawdown_pct: float = 0.10
    max_loss_streak: int = 5
    use_equity_for_drawdown: bool = True


@dataclass(slots=True)
class DailyLossConfig:
    max_daily_loss_pct: float = 0.05
    use_equity_for_daily_loss: bool = True
    reset_hour_utc: int = 0


@dataclass(slots=True)
class LeverageConfig:
    max_leverage: float = 5.0
    max_leverage_per_symbol: dict[str, float] = field(default_factory=dict)
    reduce_leverage_in_safe_mode: bool = True
    safe_mode_max_leverage: float = 2.0


@dataclass(slots=True)
class CorrelationConfig:
    enabled: bool = True
    groups: dict[str, list[str]] = field(default_factory=dict)
    max_group_exposure_pct: float = 0.50


@dataclass(slots=True)
class CircuitBreakerConfig:
    enabled: bool = True
    max_consecutive_failures: int = 5
    cooldown_seconds: float = 300.0
    max_execution_failures: int = 5
    extreme_volatility_threshold: float | None = None


@dataclass(slots=True)
class RiskConfig:
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    exposure: ExposureConfig = field(default_factory=ExposureConfig)
    drawdown: DrawdownConfig = field(default_factory=DrawdownConfig)
    daily_loss: DailyLossConfig = field(default_factory=DailyLossConfig)
    leverage: LeverageConfig = field(default_factory=LeverageConfig)
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    def validate(self) -> None:
        self._validate_pct(
            self.position_sizing.default_risk_per_trade_pct,
            "position_sizing.default_risk_per_trade_pct",
        )
        self._validate_pct(
            self.position_sizing.max_risk_per_trade_pct,
            "position_sizing.max_risk_per_trade_pct",
        )
        self._validate_pct(
            self.exposure.max_total_exposure_pct,
            "exposure.max_total_exposure_pct",
            upper_bound=None,
        )
        self._validate_pct(
            self.exposure.max_symbol_exposure_pct,
            "exposure.max_symbol_exposure_pct",
            upper_bound=None,
        )
        self._validate_pct(
            self.exposure.max_side_exposure_pct,
            "exposure.max_side_exposure_pct",
            upper_bound=None,
        )
        self._validate_pct(
            self.drawdown.max_total_drawdown_pct,
            "drawdown.max_total_drawdown_pct",
        )
        self._validate_pct(
            self.drawdown.safe_mode_drawdown_pct,
            "drawdown.safe_mode_drawdown_pct",
        )
        self._validate_pct(
            self.daily_loss.max_daily_loss_pct,
            "daily_loss.max_daily_loss_pct",
        )
        self._validate_pct(
            self.correlation.max_group_exposure_pct,
            "correlation.max_group_exposure_pct",
            upper_bound=None,
        )

        if self.position_sizing.max_risk_per_trade_pct < self.position_sizing.default_risk_per_trade_pct:
            raise RiskConfigurationError(
                "position_sizing.max_risk_per_trade_pct must be >= default_risk_per_trade_pct"
            )

        if self.drawdown.safe_mode_drawdown_pct > self.drawdown.max_total_drawdown_pct:
            raise RiskConfigurationError(
                "drawdown.safe_mode_drawdown_pct must be <= drawdown.max_total_drawdown_pct"
            )

        if self.position_sizing.min_position_size < 0:
            raise RiskConfigurationError("position_sizing.min_position_size must be >= 0")

        if (
            self.position_sizing.max_position_size is not None
            and self.position_sizing.max_position_size < self.position_sizing.min_position_size
        ):
            raise RiskConfigurationError(
                "position_sizing.max_position_size must be >= min_position_size"
            )

        if self.exposure.max_open_positions <= 0:
            raise RiskConfigurationError("exposure.max_open_positions must be > 0")

        if self.leverage.max_leverage <= 0:
            raise RiskConfigurationError("leverage.max_leverage must be > 0")

        if self.leverage.safe_mode_max_leverage <= 0:
            raise RiskConfigurationError("leverage.safe_mode_max_leverage must be > 0")

        if self.daily_loss.reset_hour_utc < 0 or self.daily_loss.reset_hour_utc > 23:
            raise RiskConfigurationError("daily_loss.reset_hour_utc must be in range [0, 23]")

        if self.circuit_breaker.cooldown_seconds < 0:
            raise RiskConfigurationError("circuit_breaker.cooldown_seconds must be >= 0")

    @staticmethod
    def _validate_pct(value: float, field_name: str, upper_bound: float | None = 1.0) -> None:
        if value < 0:
            raise RiskConfigurationError(f"{field_name} must be >= 0")
        if upper_bound is not None and value > upper_bound:
            raise RiskConfigurationError(f"{field_name} must be <= {upper_bound}")