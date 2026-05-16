from __future__ import annotations

from dataclasses import dataclass, field
import math

from risk.enums import (
    ExecutionQuality,
    LiquidityClass,
    RiskMode,
    TradeTier,
)
from risk.exceptions import RiskConfigurationError


@dataclass(slots=True)
class RiskUnitConfig:
    """
    Base R configuration.

    R — базова одиниця ризику. Вона може масштабуватись залежно від
    equity, risk mode, symbol/strategy budgets та open risk.
    """

    base_risk_unit_pct: float = 0.001
    min_risk_unit: float | None = None
    max_risk_unit: float | None = None

    caution_multiplier: float = 0.75
    safe_mode_multiplier: float = 0.50
    reduce_only_multiplier: float = 0.0
    halted_multiplier: float = 0.0
    emergency_stop_multiplier: float = 0.0

    aggressive_multiplier: float = 1.25

    use_equity_for_r: bool = True
    use_available_budget_caps: bool = True


@dataclass(slots=True)
class TierRiskConfig:
    """
    Risk profile for one trade tier.

    Tier не задає абсолютну суму. Він задає risk_units відносно R.
    """

    risk_units: float
    min_rr: float
    min_expected_value: float = 0.02
    max_cost_to_reward_pct: float = 0.10

    default_leverage: float = 5.0
    max_leverage: float = 5.0

    allow_in_caution: bool = True
    allow_in_safe_mode: bool = True
    allow_in_aggressive: bool = True

    require_take_profit: bool = True
    require_positive_ev_after_cost: bool = True


@dataclass(slots=True)
class TierModelConfig:
    """
    Tier map for adaptive position sizing and quality checks.
    """

    tiers: dict[TradeTier, TierRiskConfig] = field(
        default_factory=lambda: {
            TradeTier.T1: TierRiskConfig(
                risk_units=0.25,
                min_rr=1.50,
                min_expected_value=0.02,
                max_cost_to_reward_pct=0.08,
                default_leverage=5.0,
                max_leverage=10.0,
                allow_in_caution=True,
                allow_in_safe_mode=True,
                allow_in_aggressive=True,
            ),
            TradeTier.T2: TierRiskConfig(
                risk_units=0.50,
                min_rr=1.80,
                min_expected_value=0.03,
                max_cost_to_reward_pct=0.10,
                default_leverage=5.0,
                max_leverage=10.0,
                allow_in_caution=True,
                allow_in_safe_mode=True,
                allow_in_aggressive=True,
            ),
            TradeTier.T3: TierRiskConfig(
                risk_units=1.00,
                min_rr=2.00,
                min_expected_value=0.05,
                max_cost_to_reward_pct=0.12,
                default_leverage=5.0,
                max_leverage=5.0,
                allow_in_caution=True,
                allow_in_safe_mode=False,
                allow_in_aggressive=True,
            ),
            TradeTier.T4: TierRiskConfig(
                risk_units=1.50,
                min_rr=2.50,
                min_expected_value=0.07,
                max_cost_to_reward_pct=0.15,
                default_leverage=5.0,
                max_leverage=5.0,
                allow_in_caution=False,
                allow_in_safe_mode=False,
                allow_in_aggressive=True,
            ),
        }
    )

    default_tier: TradeTier = TradeTier.T2
    max_tier_by_mode: dict[RiskMode, TradeTier] = field(
        default_factory=lambda: {
            RiskMode.NORMAL: TradeTier.T4,
            RiskMode.CAUTION: TradeTier.T3,
            RiskMode.SAFE_MODE: TradeTier.T2,
            RiskMode.REDUCE_ONLY: TradeTier.T1,
            RiskMode.HALTED: TradeTier.T1,
            RiskMode.EMERGENCY_STOP: TradeTier.T1,
        }
    )

    downgrade_tier_in_caution: bool = True
    downgrade_tier_in_safe_mode: bool = True


@dataclass(slots=True)
class RiskBudgetConfig:
    """
    Global budget limits expressed in R.

    Ми не привʼязуємо модель до конкретного депозиту чи кількості угод.
    Daily/weekly/monthly limits задаються в R.
    """

    caution_daily_loss_r: float = 3.0
    soft_daily_loss_r: float = 6.0
    hard_daily_loss_r: float = 10.0

    weekly_hard_loss_r: float = 25.0
    monthly_review_loss_r: float = 40.0
    emergency_stop_loss_r: float = 50.0

    max_drawdown_r: float | None = None
    safe_mode_drawdown_r: float | None = None

    reset_hour_utc: int = 0
    weekly_reset_weekday_utc: int = 0

    allow_new_positions_after_soft_daily_loss: bool = True
    require_manual_review_after_monthly_limit: bool = True
    require_manual_reset_after_emergency_stop: bool = True


@dataclass(slots=True)
class SymbolRiskConfig:
    """
    Symbol-level throttling and budget.

    Контролює, щоб один symbol не зʼїв увесь денний/відкритий risk budget.
    """

    max_positions_per_symbol: int = 1
    max_trades_per_symbol_per_day: int | None = None

    max_symbol_daily_loss_r: float = 2.0
    max_symbol_open_risk_r: float = 2.0

    cooldown_after_consecutive_losses: int = 2
    cooldown_seconds: float = 1800.0

    disable_after_daily_loss_breach: bool = True
    disable_after_trade_limit: bool = False

    per_symbol_daily_loss_r: dict[str, float] = field(default_factory=dict)
    per_symbol_open_risk_r: dict[str, float] = field(default_factory=dict)
    per_symbol_trade_limit: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyRiskConfig:
    """
    Strategy-level budget and rolling expectancy control.
    """

    default_daily_loss_budget_r: float = 4.0
    default_open_risk_budget_r: float = 3.0

    max_consecutive_losses: int = 3
    cooldown_seconds: float = 1800.0

    rolling_expectancy_window: int = 30
    reduce_when_expectancy_below: float = 0.0
    disable_when_expectancy_below: float = -0.05
    reduced_risk_multiplier: float = 0.50

    disable_after_daily_loss_breach: bool = True
    disable_on_negative_expectancy: bool = True

    per_strategy_daily_loss_budget_r: dict[str, float] = field(default_factory=dict)
    per_strategy_open_risk_budget_r: dict[str, float] = field(default_factory=dict)
    per_strategy_trade_limit: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionCostConfig:
    """
    Spread/slippage/fee guard.

    Для micro-scalping це критичний guard: кількість сигналів може бути
    будь-якою, але cost не має зʼїдати edge.
    """

    enabled: bool = True

    default_max_cost_to_reward_pct: float = 0.10
    aggressive_max_cost_to_reward_pct: float = 0.15
    safe_mode_max_cost_to_reward_pct: float = 0.08

    require_spread_guard: bool = True
    require_slippage_guard: bool = True
    require_fee_estimate: bool = True
    require_positive_ev_after_cost: bool = True

    min_execution_quality: ExecutionQuality = ExecutionQuality.ACCEPTABLE

    max_spread_pct: float | None = None
    max_slippage_pct: float | None = None

    max_cost_to_reward_by_tier: dict[TradeTier, float] = field(
        default_factory=lambda: {
            TradeTier.T1: 0.08,
            TradeTier.T2: 0.10,
            TradeTier.T3: 0.12,
            TradeTier.T4: 0.15,
        }
    )


@dataclass(slots=True)
class LeveragePolicyConfig:
    """
    Adaptive leverage policy.

    x3 для risky/low-liquidity, x5 default, x10 тільки для top/high liquidity
    micro-scalp setups з хорошим execution quality.
    """

    default_leverage: float = 5.0

    low_liquidity_max_leverage: float = 3.0
    shitcoin_max_leverage: float = 3.0
    top_liquidity_max_leverage: float = 10.0

    safe_mode_max_leverage: float = 3.0
    caution_max_leverage: float = 5.0
    reduce_only_max_leverage: float = 1.0

    max_leverage_by_liquidity: dict[LiquidityClass, float] = field(
        default_factory=lambda: {
            LiquidityClass.TOP: 10.0,
            LiquidityClass.HIGH: 10.0,
            LiquidityClass.NORMAL: 5.0,
            LiquidityClass.LOW: 3.0,
            LiquidityClass.ILLIQUID: 2.0,
            LiquidityClass.SHITCOIN: 3.0,
        }
    )

    max_leverage_by_tier: dict[TradeTier, float] = field(
        default_factory=lambda: {
            TradeTier.T1: 10.0,
            TradeTier.T2: 10.0,
            TradeTier.T3: 5.0,
            TradeTier.T4: 5.0,
        }
    )

    per_symbol_max_leverage: dict[str, float] = field(default_factory=dict)
    per_strategy_max_leverage: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ExposureConfig:
    """
    Portfolio exposure and capital usage limits.

    Open risk є основним risk-limit, notional/margin exposure —
    додаткові захисні обмеження.
    """

    max_open_risk_r: float = 6.0
    aggressive_max_open_risk_r: float = 10.0
    safe_mode_max_open_risk_r: float = 2.0

    max_used_margin_pct: float = 0.25
    aggressive_max_used_margin_pct: float = 0.40
    safe_mode_max_used_margin_pct: float = 0.15

    max_total_exposure_pct: float = 1.00
    max_symbol_exposure_pct: float = 0.30
    max_side_exposure_pct: float = 0.60

    max_open_positions: int | None = None
    include_pending_orders: bool = True

    correlation_groups: dict[str, list[str]] = field(default_factory=dict)
    max_correlation_group_exposure_pct: float = 0.50
    max_correlation_group_open_risk_r: float | None = None


@dataclass(slots=True)
class PositionSizingConfig:
    """
    Exchange constraints and sizing behaviour.

    Сам risk amount береться не з fixed pct per trade, а з R * tier.risk_units.
    """

    min_position_size: float = 0.0
    max_position_size: float | None = None

    require_stop_loss: bool = True
    fallback_stop_loss_pct: float | None = None

    use_confidence_adjustment: bool = True
    confidence_scale_min: float = 0.50
    confidence_scale_max: float = 1.25

    use_volatility_adjustment: bool = True
    volatility_scale_min: float = 0.25
    volatility_scale_max: float = 1.00

    reject_if_below_min_size: bool = True
    never_increase_size_above_risk: bool = True


@dataclass(slots=True)
class RiskReservationConfig:
    """
    Pending risk reservation policy.

    Reservation закриває race-condition між risk ALLOW і фактичним
    position.opened/execution rejection. Якщо enabled=True, RiskManager
    має резервувати open risk одразу після фінального ALLOW і звільняти
    його після confirm/release/timeout.
    """

    enabled: bool = True
    ttl_seconds: float = 30.0
    cleanup_interval_seconds: float = 10.0

    max_pending_reservations: int = 100
    max_pending_per_symbol: int | None = 3
    max_pending_per_strategy: int | None = 10

    include_pending_in_exposure: bool = True
    include_pending_in_symbol_budget: bool = True
    include_pending_in_strategy_budget: bool = True
    include_pending_in_correlation: bool = True

    reserve_on_allow: bool = True
    fail_closed_on_reservation_error: bool = True
    auto_expire_on_evaluate: bool = True


@dataclass(slots=True)
class CircuitBreakerConfig:
    """
    Emergency risk blocker.
    """

    enabled: bool = True

    max_consecutive_failures: int = 5
    max_execution_failures: int = 5
    cooldown_seconds: float = 300.0

    extreme_volatility_threshold: float | None = None
    data_stale_after_seconds: float | None = None

    trigger_on_emergency_stop: bool = True
    trigger_on_execution_cost_spike: bool = True
    trigger_on_data_feed_failure: bool = True

    require_manual_release_for_emergency: bool = True


@dataclass(slots=True)
class RiskConfig:
    """
    Root config for adaptive tier-based risk management.

    Порядок логіки в RiskManager буде таким:
    budget → symbol/strategy → tier → RR/EV/cost → leverage →
    position sizing → exposure/open risk → circuit breaker/final decision.
    """

    risk_unit: RiskUnitConfig = field(default_factory=RiskUnitConfig)
    tiers: TierModelConfig = field(default_factory=TierModelConfig)
    budget: RiskBudgetConfig = field(default_factory=RiskBudgetConfig)
    symbol: SymbolRiskConfig = field(default_factory=SymbolRiskConfig)
    strategy: StrategyRiskConfig = field(default_factory=StrategyRiskConfig)
    execution_cost: ExecutionCostConfig = field(default_factory=ExecutionCostConfig)
    leverage: LeveragePolicyConfig = field(default_factory=LeveragePolicyConfig)
    exposure: ExposureConfig = field(default_factory=ExposureConfig)
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    reservation: RiskReservationConfig = field(default_factory=RiskReservationConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    def validate(self) -> None:
        self._validate_risk_unit()
        self._validate_tiers()
        self._validate_budget()
        self._validate_symbol()
        self._validate_strategy()
        self._validate_execution_cost()
        self._validate_leverage()
        self._validate_exposure()
        self._validate_position_sizing()
        self._validate_reservation()
        self._validate_circuit_breaker()

    def _validate_risk_unit(self) -> None:
        self._validate_pct(
            self.risk_unit.base_risk_unit_pct,
            "risk_unit.base_risk_unit_pct",
        )

        if self.risk_unit.min_risk_unit is not None and self.risk_unit.min_risk_unit < 0:
            raise RiskConfigurationError("risk_unit.min_risk_unit must be >= 0")

        if self.risk_unit.max_risk_unit is not None and self.risk_unit.max_risk_unit <= 0:
            raise RiskConfigurationError("risk_unit.max_risk_unit must be > 0")

        if (
            self.risk_unit.min_risk_unit is not None
            and self.risk_unit.max_risk_unit is not None
            and self.risk_unit.max_risk_unit < self.risk_unit.min_risk_unit
        ):
            raise RiskConfigurationError(
                "risk_unit.max_risk_unit must be >= risk_unit.min_risk_unit"
            )

        self._validate_non_negative(
            self.risk_unit.caution_multiplier,
            "risk_unit.caution_multiplier",
        )
        self._validate_non_negative(
            self.risk_unit.safe_mode_multiplier,
            "risk_unit.safe_mode_multiplier",
        )
        self._validate_non_negative(
            self.risk_unit.aggressive_multiplier,
            "risk_unit.aggressive_multiplier",
        )

    def _validate_tiers(self) -> None:
        if not self.tiers.tiers:
            raise RiskConfigurationError("tiers.tiers must not be empty")

        for tier in TradeTier:
            if tier not in self.tiers.tiers:
                raise RiskConfigurationError(f"Missing tier config for {tier.value}")

        if self.tiers.default_tier not in self.tiers.tiers:
            raise RiskConfigurationError("tiers.default_tier must exist in tiers.tiers")

        for mode in RiskMode:
            if mode not in self.tiers.max_tier_by_mode:
                raise RiskConfigurationError(
                    f"Missing max tier config for risk mode {mode.value}"
                )

        for tier, cfg in self.tiers.tiers.items():
            if cfg.risk_units <= 0:
                raise RiskConfigurationError(
                    f"tiers.{tier.value}.risk_units must be > 0"
                )
            if cfg.min_rr < 0:
                raise RiskConfigurationError(
                    f"tiers.{tier.value}.min_rr must be >= 0"
                )
            if not math.isfinite(cfg.min_expected_value):
                raise RiskConfigurationError(
                    f"tiers.{tier.value}.min_expected_value must be finite"
                )
            if cfg.default_leverage <= 0:
                raise RiskConfigurationError(
                    f"tiers.{tier.value}.default_leverage must be > 0"
                )
            if cfg.max_leverage <= 0:
                raise RiskConfigurationError(
                    f"tiers.{tier.value}.max_leverage must be > 0"
                )
            if cfg.default_leverage > cfg.max_leverage:
                raise RiskConfigurationError(
                    f"tiers.{tier.value}.default_leverage must be <= max_leverage"
                )
            self._validate_pct(
                cfg.max_cost_to_reward_pct,
                f"tiers.{tier.value}.max_cost_to_reward_pct",
                upper_bound=None,
            )

    def _validate_budget(self) -> None:
        ordered_limits = [
            ("budget.caution_daily_loss_r", self.budget.caution_daily_loss_r),
            ("budget.soft_daily_loss_r", self.budget.soft_daily_loss_r),
            ("budget.hard_daily_loss_r", self.budget.hard_daily_loss_r),
            ("budget.weekly_hard_loss_r", self.budget.weekly_hard_loss_r),
            ("budget.monthly_review_loss_r", self.budget.monthly_review_loss_r),
            ("budget.emergency_stop_loss_r", self.budget.emergency_stop_loss_r),
        ]

        for name, value in ordered_limits:
            self._validate_positive(value, name)

        if not (
            self.budget.caution_daily_loss_r
            <= self.budget.soft_daily_loss_r
            <= self.budget.hard_daily_loss_r
            <= self.budget.weekly_hard_loss_r
            <= self.budget.monthly_review_loss_r
            <= self.budget.emergency_stop_loss_r
        ):
            raise RiskConfigurationError(
                "budget loss limits must be ordered: "
                "caution_daily_loss_r <= soft_daily_loss_r <= hard_daily_loss_r "
                "<= weekly_hard_loss_r <= monthly_review_loss_r <= emergency_stop_loss_r"
            )

        if not 0 <= self.budget.reset_hour_utc <= 23:
            raise RiskConfigurationError("budget.reset_hour_utc must be in range [0, 23]")

        if not 0 <= self.budget.weekly_reset_weekday_utc <= 6:
            raise RiskConfigurationError(
                "budget.weekly_reset_weekday_utc must be in range [0, 6]"
            )

        if self.budget.max_drawdown_r is not None:
            self._validate_positive(
                self.budget.max_drawdown_r,
                "budget.max_drawdown_r",
            )

        if self.budget.safe_mode_drawdown_r is not None:
            self._validate_positive(
                self.budget.safe_mode_drawdown_r,
                "budget.safe_mode_drawdown_r",
            )

        if (
            self.budget.max_drawdown_r is not None
            and self.budget.safe_mode_drawdown_r is not None
            and self.budget.safe_mode_drawdown_r > self.budget.max_drawdown_r
        ):
            raise RiskConfigurationError(
                "budget.safe_mode_drawdown_r must be <= budget.max_drawdown_r"
            )

    def _validate_symbol(self) -> None:
        self._validate_positive_int(
            self.symbol.max_positions_per_symbol,
            "symbol.max_positions_per_symbol",
        )

        if self.symbol.max_trades_per_symbol_per_day is not None:
            self._validate_positive_int(
                self.symbol.max_trades_per_symbol_per_day,
                "symbol.max_trades_per_symbol_per_day",
            )

        self._validate_positive(
            self.symbol.max_symbol_daily_loss_r,
            "symbol.max_symbol_daily_loss_r",
        )
        self._validate_positive(
            self.symbol.max_symbol_open_risk_r,
            "symbol.max_symbol_open_risk_r",
        )
        self._validate_non_negative(
            self.symbol.cooldown_after_consecutive_losses,
            "symbol.cooldown_after_consecutive_losses",
        )
        self._validate_non_negative(
            self.symbol.cooldown_seconds,
            "symbol.cooldown_seconds",
        )

        for symbol, value in self.symbol.per_symbol_daily_loss_r.items():
            self._validate_positive(value, f"symbol.per_symbol_daily_loss_r[{symbol}]")

        for symbol, value in self.symbol.per_symbol_open_risk_r.items():
            self._validate_positive(value, f"symbol.per_symbol_open_risk_r[{symbol}]")

        for symbol, value in self.symbol.per_symbol_trade_limit.items():
            self._validate_positive_int(value, f"symbol.per_symbol_trade_limit[{symbol}]")

    def _validate_strategy(self) -> None:
        self._validate_positive(
            self.strategy.default_daily_loss_budget_r,
            "strategy.default_daily_loss_budget_r",
        )
        self._validate_positive(
            self.strategy.default_open_risk_budget_r,
            "strategy.default_open_risk_budget_r",
        )
        self._validate_non_negative(
            self.strategy.max_consecutive_losses,
            "strategy.max_consecutive_losses",
        )
        self._validate_non_negative(
            self.strategy.cooldown_seconds,
            "strategy.cooldown_seconds",
        )
        self._validate_positive_int(
            self.strategy.rolling_expectancy_window,
            "strategy.rolling_expectancy_window",
        )

        if self.strategy.reduced_risk_multiplier < 0:
            raise RiskConfigurationError(
                "strategy.reduced_risk_multiplier must be >= 0"
            )

        if self.strategy.reduced_risk_multiplier > 1:
            raise RiskConfigurationError(
                "strategy.reduced_risk_multiplier should be <= 1"
            )

        if self.strategy.disable_when_expectancy_below > self.strategy.reduce_when_expectancy_below:
            raise RiskConfigurationError(
                "strategy.disable_when_expectancy_below must be <= reduce_when_expectancy_below"
            )

        for strategy, value in self.strategy.per_strategy_daily_loss_budget_r.items():
            self._validate_positive(
                value,
                f"strategy.per_strategy_daily_loss_budget_r[{strategy}]",
            )

        for strategy, value in self.strategy.per_strategy_open_risk_budget_r.items():
            self._validate_positive(
                value,
                f"strategy.per_strategy_open_risk_budget_r[{strategy}]",
            )

        for strategy, value in self.strategy.per_strategy_trade_limit.items():
            self._validate_positive_int(
                value,
                f"strategy.per_strategy_trade_limit[{strategy}]",
            )

    def _validate_execution_cost(self) -> None:
        self._validate_pct(
            self.execution_cost.default_max_cost_to_reward_pct,
            "execution_cost.default_max_cost_to_reward_pct",
            upper_bound=None,
        )
        self._validate_pct(
            self.execution_cost.aggressive_max_cost_to_reward_pct,
            "execution_cost.aggressive_max_cost_to_reward_pct",
            upper_bound=None,
        )
        self._validate_pct(
            self.execution_cost.safe_mode_max_cost_to_reward_pct,
            "execution_cost.safe_mode_max_cost_to_reward_pct",
            upper_bound=None,
        )

        if (
            self.execution_cost.safe_mode_max_cost_to_reward_pct
            > self.execution_cost.default_max_cost_to_reward_pct
        ):
            raise RiskConfigurationError(
                "execution_cost.safe_mode_max_cost_to_reward_pct "
                "must be <= default_max_cost_to_reward_pct"
            )

        for tier, value in self.execution_cost.max_cost_to_reward_by_tier.items():
            if tier not in TradeTier:
                raise RiskConfigurationError(
                    f"Invalid tier in execution_cost.max_cost_to_reward_by_tier: {tier}"
                )
            self._validate_pct(
                value,
                f"execution_cost.max_cost_to_reward_by_tier[{tier.value}]",
                upper_bound=None,
            )

        if self.execution_cost.max_spread_pct is not None:
            self._validate_pct(
                self.execution_cost.max_spread_pct,
                "execution_cost.max_spread_pct",
                upper_bound=None,
            )

        if self.execution_cost.max_slippage_pct is not None:
            self._validate_pct(
                self.execution_cost.max_slippage_pct,
                "execution_cost.max_slippage_pct",
                upper_bound=None,
            )

    def _validate_leverage(self) -> None:
        self._validate_positive(
            self.leverage.default_leverage,
            "leverage.default_leverage",
        )
        self._validate_positive(
            self.leverage.low_liquidity_max_leverage,
            "leverage.low_liquidity_max_leverage",
        )
        self._validate_positive(
            self.leverage.shitcoin_max_leverage,
            "leverage.shitcoin_max_leverage",
        )
        self._validate_positive(
            self.leverage.top_liquidity_max_leverage,
            "leverage.top_liquidity_max_leverage",
        )
        self._validate_positive(
            self.leverage.safe_mode_max_leverage,
            "leverage.safe_mode_max_leverage",
        )
        self._validate_positive(
            self.leverage.caution_max_leverage,
            "leverage.caution_max_leverage",
        )
        self._validate_positive(
            self.leverage.reduce_only_max_leverage,
            "leverage.reduce_only_max_leverage",
        )

        for liquidity_class in LiquidityClass:
            if liquidity_class not in self.leverage.max_leverage_by_liquidity:
                raise RiskConfigurationError(
                    f"Missing leverage cap for liquidity class {liquidity_class.value}"
                )

        for tier in TradeTier:
            if tier not in self.leverage.max_leverage_by_tier:
                raise RiskConfigurationError(
                    f"Missing leverage cap for tier {tier.value}"
                )

        for liquidity_class, value in self.leverage.max_leverage_by_liquidity.items():
            self._validate_positive(
                value,
                f"leverage.max_leverage_by_liquidity[{liquidity_class.value}]",
            )

        for tier, value in self.leverage.max_leverage_by_tier.items():
            self._validate_positive(
                value,
                f"leverage.max_leverage_by_tier[{tier.value}]",
            )

        for symbol, value in self.leverage.per_symbol_max_leverage.items():
            self._validate_positive(value, f"leverage.per_symbol_max_leverage[{symbol}]")

        for strategy, value in self.leverage.per_strategy_max_leverage.items():
            self._validate_positive(
                value,
                f"leverage.per_strategy_max_leverage[{strategy}]",
            )

    def _validate_exposure(self) -> None:
        self._validate_positive(self.exposure.max_open_risk_r, "exposure.max_open_risk_r")
        self._validate_positive(
            self.exposure.aggressive_max_open_risk_r,
            "exposure.aggressive_max_open_risk_r",
        )
        self._validate_positive(
            self.exposure.safe_mode_max_open_risk_r,
            "exposure.safe_mode_max_open_risk_r",
        )

        if self.exposure.safe_mode_max_open_risk_r > self.exposure.max_open_risk_r:
            raise RiskConfigurationError(
                "exposure.safe_mode_max_open_risk_r must be <= exposure.max_open_risk_r"
            )

        if self.exposure.aggressive_max_open_risk_r < self.exposure.max_open_risk_r:
            raise RiskConfigurationError(
                "exposure.aggressive_max_open_risk_r must be >= exposure.max_open_risk_r"
            )

        self._validate_pct(
            self.exposure.max_used_margin_pct,
            "exposure.max_used_margin_pct",
            upper_bound=None,
        )
        self._validate_pct(
            self.exposure.aggressive_max_used_margin_pct,
            "exposure.aggressive_max_used_margin_pct",
            upper_bound=None,
        )
        self._validate_pct(
            self.exposure.safe_mode_max_used_margin_pct,
            "exposure.safe_mode_max_used_margin_pct",
            upper_bound=None,
        )

        if self.exposure.safe_mode_max_used_margin_pct > self.exposure.max_used_margin_pct:
            raise RiskConfigurationError(
                "exposure.safe_mode_max_used_margin_pct must be <= exposure.max_used_margin_pct"
            )

        if self.exposure.aggressive_max_used_margin_pct < self.exposure.max_used_margin_pct:
            raise RiskConfigurationError(
                "exposure.aggressive_max_used_margin_pct must be >= exposure.max_used_margin_pct"
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
            self.exposure.max_correlation_group_exposure_pct,
            "exposure.max_correlation_group_exposure_pct",
            upper_bound=None,
        )

        if self.exposure.max_open_positions is not None:
            self._validate_positive_int(
                self.exposure.max_open_positions,
                "exposure.max_open_positions",
            )

        if self.exposure.max_correlation_group_open_risk_r is not None:
            self._validate_positive(
                self.exposure.max_correlation_group_open_risk_r,
                "exposure.max_correlation_group_open_risk_r",
            )

    def _validate_position_sizing(self) -> None:
        self._validate_non_negative(
            self.position_sizing.min_position_size,
            "position_sizing.min_position_size",
        )

        if self.position_sizing.max_position_size is not None:
            self._validate_positive(
                self.position_sizing.max_position_size,
                "position_sizing.max_position_size",
            )

            if self.position_sizing.max_position_size < self.position_sizing.min_position_size:
                raise RiskConfigurationError(
                    "position_sizing.max_position_size must be >= min_position_size"
                )

        if self.position_sizing.fallback_stop_loss_pct is not None:
            self._validate_pct(
                self.position_sizing.fallback_stop_loss_pct,
                "position_sizing.fallback_stop_loss_pct",
                upper_bound=None,
            )
            if self.position_sizing.fallback_stop_loss_pct <= 0:
                raise RiskConfigurationError(
                    "position_sizing.fallback_stop_loss_pct must be > 0"
                )

        if self.position_sizing.confidence_scale_min < 0:
            raise RiskConfigurationError(
                "position_sizing.confidence_scale_min must be >= 0"
            )

        if self.position_sizing.confidence_scale_max < self.position_sizing.confidence_scale_min:
            raise RiskConfigurationError(
                "position_sizing.confidence_scale_max must be >= confidence_scale_min"
            )

        if self.position_sizing.volatility_scale_min < 0:
            raise RiskConfigurationError(
                "position_sizing.volatility_scale_min must be >= 0"
            )

        if self.position_sizing.volatility_scale_max < self.position_sizing.volatility_scale_min:
            raise RiskConfigurationError(
                "position_sizing.volatility_scale_max must be >= volatility_scale_min"
            )

    def _validate_reservation(self) -> None:
        self._validate_positive(
            self.reservation.ttl_seconds,
            "reservation.ttl_seconds",
        )
        self._validate_positive(
            self.reservation.cleanup_interval_seconds,
            "reservation.cleanup_interval_seconds",
        )
        self._validate_positive_int(
            self.reservation.max_pending_reservations,
            "reservation.max_pending_reservations",
        )

        if self.reservation.max_pending_per_symbol is not None:
            self._validate_positive_int(
                self.reservation.max_pending_per_symbol,
                "reservation.max_pending_per_symbol",
            )

        if self.reservation.max_pending_per_strategy is not None:
            self._validate_positive_int(
                self.reservation.max_pending_per_strategy,
                "reservation.max_pending_per_strategy",
            )

        if self.reservation.enabled and not self.reservation.reserve_on_allow:
            raise RiskConfigurationError(
                "reservation.reserve_on_allow must be True when reservation.enabled is True"
            )


    def _validate_circuit_breaker(self) -> None:
        self._validate_non_negative(
            self.circuit_breaker.max_consecutive_failures,
            "circuit_breaker.max_consecutive_failures",
        )
        self._validate_non_negative(
            self.circuit_breaker.max_execution_failures,
            "circuit_breaker.max_execution_failures",
        )
        self._validate_non_negative(
            self.circuit_breaker.cooldown_seconds,
            "circuit_breaker.cooldown_seconds",
        )

        if self.circuit_breaker.extreme_volatility_threshold is not None:
            self._validate_positive(
                self.circuit_breaker.extreme_volatility_threshold,
                "circuit_breaker.extreme_volatility_threshold",
            )

        if self.circuit_breaker.data_stale_after_seconds is not None:
            self._validate_positive(
                self.circuit_breaker.data_stale_after_seconds,
                "circuit_breaker.data_stale_after_seconds",
            )

    @staticmethod
    def _validate_pct(
        value: float,
        field_name: str,
        upper_bound: float | None = 1.0,
    ) -> None:
        if not math.isfinite(value):
            raise RiskConfigurationError(f"{field_name} must be finite")
        if value < 0:
            raise RiskConfigurationError(f"{field_name} must be >= 0")
        if upper_bound is not None and value > upper_bound:
            raise RiskConfigurationError(f"{field_name} must be <= {upper_bound}")

    @staticmethod
    def _validate_positive(value: float, field_name: str) -> None:
        if not math.isfinite(value):
            raise RiskConfigurationError(f"{field_name} must be finite")
        if value <= 0:
            raise RiskConfigurationError(f"{field_name} must be > 0")

    @staticmethod
    def _validate_non_negative(value: float | int, field_name: str) -> None:
        if not math.isfinite(float(value)):
            raise RiskConfigurationError(f"{field_name} must be finite")
        if value < 0:
            raise RiskConfigurationError(f"{field_name} must be >= 0")

    @staticmethod
    def _validate_positive_int(value: int, field_name: str) -> None:
        if not isinstance(value, int):
            raise RiskConfigurationError(f"{field_name} must be an int")
        if value <= 0:
            raise RiskConfigurationError(f"{field_name} must be > 0")


__all__ = [
    "CircuitBreakerConfig",
    "ExecutionCostConfig",
    "ExposureConfig",
    "LeveragePolicyConfig",
    "PositionSizingConfig",
    "RiskReservationConfig",
    "RiskBudgetConfig",
    "RiskConfig",
    "RiskUnitConfig",
    "StrategyRiskConfig",
    "SymbolRiskConfig",
    "TierModelConfig",
    "TierRiskConfig",
]