from __future__ import annotations

from enum import Enum


class RiskDecisionType(str, Enum):
    """
    Фінальне або проміжне рішення risk layer.

    Використовується RiskManager та guard-ами для уніфікованого
    рішення: дозволити, відхилити, зменшити ризик, перевести в
    reduce-only або зупинити торгівлю.
    """

    ALLOW = "allow"
    DENY = "deny"
    REDUCE_SIZE = "reduce_size"
    REDUCE_RISK = "reduce_risk"
    DOWNGRADE_TIER = "downgrade_tier"
    FORCE_CLOSE = "force_close"
    ONLY_REDUCE = "only_reduce"
    HALT_TRADING = "halt_trading"
    EMERGENCY_STOP = "emergency_stop"


class RiskViolationType(str, Enum):
    """
    Причини risk-відмов, зниження ризику або аварійного блокування.

    Enum навмисно широкий: він покриває legacy-guards і нову
    tier/budget/EV/execution-cost модель.
    """

    # Generic / request validation
    INVALID_REQUEST = "invalid_request"
    INSUFFICIENT_RISK_DATA = "insufficient_risk_data"

    # Position sizing / stop-loss / take-profit
    POSITION_SIZE_INVALID = "position_size_invalid"
    POSITION_SIZE_BELOW_MINIMUM = "position_size_below_minimum"
    POSITION_SIZE_ABOVE_MAXIMUM = "position_size_above_maximum"
    STOP_LOSS_MISSING = "stop_loss_missing"
    STOP_DISTANCE_INVALID = "stop_distance_invalid"
    STOP_LOSS_SIDE_INVALID = "stop_loss_side_invalid"
    TAKE_PROFIT_MISSING = "take_profit_missing"
    TAKE_PROFIT_SIDE_INVALID = "take_profit_side_invalid"

    # Risk/reward / expectancy / costs
    RISK_REWARD_TOO_LOW = "risk_reward_too_low"
    EXPECTED_VALUE_NEGATIVE = "expected_value_negative"
    EXPECTED_REWARD_TOO_LOW = "expected_reward_too_low"
    EXECUTION_COST_TOO_HIGH = "execution_cost_too_high"
    SPREAD_TOO_WIDE = "spread_too_wide"
    SLIPPAGE_TOO_HIGH = "slippage_too_high"
    FEES_TOO_HIGH = "fees_too_high"

    # Tier / leverage / liquidity policy
    TIER_NOT_ALLOWED = "tier_not_allowed"
    TIER_LIMIT_EXCEEDED = "tier_limit_exceeded"
    TIER_DOWNGRADED = "tier_downgraded"
    MAX_LEVERAGE_EXCEEDED = "max_leverage_exceeded"
    LEVERAGE_NOT_ALLOWED = "leverage_not_allowed"
    LIQUIDITY_TOO_LOW = "liquidity_too_low"
    EXECUTION_QUALITY_TOO_LOW = "execution_quality_too_low"

    # Margin / exposure / open risk
    INSUFFICIENT_MARGIN = "insufficient_margin"
    USED_MARGIN_EXCEEDED = "used_margin_exceeded"
    OPEN_RISK_EXCEEDED = "open_risk_exceeded"
    MAX_EXPOSURE_EXCEEDED = "max_exposure_exceeded"
    MAX_SYMBOL_EXPOSURE_EXCEEDED = "max_symbol_exposure_exceeded"
    MAX_SIDE_EXPOSURE_EXCEEDED = "max_side_exposure_exceeded"
    MAX_OPEN_POSITIONS_EXCEEDED = "max_open_positions_exceeded"
    CORRELATION_LIMIT_EXCEEDED = "correlation_limit_exceeded"

    # Global budgets / drawdown / daily-weekly-monthly limits
    DAILY_LOSS_EXCEEDED = "daily_loss_exceeded"
    SOFT_DAILY_LOSS_EXCEEDED = "soft_daily_loss_exceeded"
    HARD_DAILY_LOSS_EXCEEDED = "hard_daily_loss_exceeded"
    WEEKLY_LOSS_EXCEEDED = "weekly_loss_exceeded"
    MONTHLY_REVIEW_REQUIRED = "monthly_review_required"
    MAX_DRAWDOWN_EXCEEDED = "max_drawdown_exceeded"
    EMERGENCY_STOP_TRIGGERED = "emergency_stop_triggered"

    # Strategy budget / expectancy
    STRATEGY_BUDGET_EXCEEDED = "strategy_budget_exceeded"
    STRATEGY_DAILY_LOSS_EXCEEDED = "strategy_daily_loss_exceeded"
    STRATEGY_OPEN_RISK_EXCEEDED = "strategy_open_risk_exceeded"
    STRATEGY_DISABLED = "strategy_disabled"
    STRATEGY_COOLDOWN_ACTIVE = "strategy_cooldown_active"
    STRATEGY_EXPECTANCY_NEGATIVE = "strategy_expectancy_negative"

    # Symbol budget / throttling
    SYMBOL_BUDGET_EXCEEDED = "symbol_budget_exceeded"
    SYMBOL_DAILY_LOSS_EXCEEDED = "symbol_daily_loss_exceeded"
    SYMBOL_OPEN_RISK_EXCEEDED = "symbol_open_risk_exceeded"
    SYMBOL_POSITION_LIMIT_EXCEEDED = "symbol_position_limit_exceeded"
    SYMBOL_TRADE_LIMIT_EXCEEDED = "symbol_trade_limit_exceeded"
    SYMBOL_COOLDOWN_ACTIVE = "symbol_cooldown_active"
    SYMBOL_DISABLED = "symbol_disabled"

    # Trading state / circuit breaker
    SAFE_MODE_ACTIVE = "safe_mode_active"
    REDUCE_ONLY_ACTIVE = "reduce_only_active"
    TRADING_HALTED = "trading_halted"
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"


class RiskLevel(str, Enum):
    """
    Severity рівень risk-порушення.
    """

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class RiskMode(str, Enum):
    """
    Новий режим risk-системи.

    RiskMode ширший за старий TradingMode і використовується
    бюджетним шаром для adaptive risk allocation.
    """

    NORMAL = "normal"
    CAUTION = "caution"
    SAFE_MODE = "safe_mode"
    REDUCE_ONLY = "reduce_only"
    HALTED = "halted"
    EMERGENCY_STOP = "emergency_stop"

    @property
    def allows_new_positions(self) -> bool:
        return self in {
            RiskMode.NORMAL,
            RiskMode.CAUTION,
            RiskMode.SAFE_MODE,
        }

    @property
    def is_protective(self) -> bool:
        return self in {
            RiskMode.CAUTION,
            RiskMode.SAFE_MODE,
            RiskMode.REDUCE_ONLY,
            RiskMode.HALTED,
            RiskMode.EMERGENCY_STOP,
        }

    @property
    def is_terminal(self) -> bool:
        return self in {
            RiskMode.HALTED,
            RiskMode.EMERGENCY_STOP,
        }


class TradingMode(str, Enum):
    """
    Backward-compatible trading mode.

    Залишаємо для поточної сумісності зі старими state/guards.
    Надалі основним режимом стане RiskMode, а TradingMode можна
    буде або прибрати, або використовувати як alias у state/API.
    """

    NORMAL = "normal"
    SAFE_MODE = "safe_mode"
    REDUCE_ONLY = "reduce_only"
    HALTED = "halted"
    EMERGENCY_STOP = "emergency_stop"

    def to_risk_mode(self) -> RiskMode:
        mapping = {
            TradingMode.NORMAL: RiskMode.NORMAL,
            TradingMode.SAFE_MODE: RiskMode.SAFE_MODE,
            TradingMode.REDUCE_ONLY: RiskMode.REDUCE_ONLY,
            TradingMode.HALTED: RiskMode.HALTED,
            TradingMode.EMERGENCY_STOP: RiskMode.EMERGENCY_STOP,
        }
        return mapping[self]


class TradeTier(str, Enum):
    """
    Tier позиції в adaptive tier-based risk model.

    Tier не задає суму напряму. Він задає risk-units,
    які потім масштабуються через RiskUnitCalculator.
    """

    T1 = "t1"
    T2 = "t2"
    T3 = "t3"
    T4 = "t4"

    @property
    def rank(self) -> int:
        return {
            TradeTier.T1: 1,
            TradeTier.T2: 2,
            TradeTier.T3: 3,
            TradeTier.T4: 4,
        }[self]

    @classmethod
    def from_rank(cls, rank: int) -> TradeTier:
        normalized_rank = max(1, min(rank, 4))
        return {
            1: cls.T1,
            2: cls.T2,
            3: cls.T3,
            4: cls.T4,
        }[normalized_rank]

    def downgrade(self, steps: int = 1) -> TradeTier:
        return self.from_rank(self.rank - max(0, steps))


class LiquidityClass(str, Enum):
    """
    Клас ліквідності інструмента.

    Використовується LeverageGuard, ExecutionCostGuard,
    ExposureControl та SymbolRiskGuard.
    """

    TOP = "top"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    ILLIQUID = "illiquid"
    SHITCOIN = "shitcoin"

    @property
    def is_high_quality(self) -> bool:
        return self in {
            LiquidityClass.TOP,
            LiquidityClass.HIGH,
        }

    @property
    def is_risky(self) -> bool:
        return self in {
            LiquidityClass.LOW,
            LiquidityClass.ILLIQUID,
            LiquidityClass.SHITCOIN,
        }


class ExecutionQuality(str, Enum):
    """
    Оцінка якості execution перед входом.

    Дає змогу risk layer блокувати угоди, якщо spread/slippage/cost
    погіршились навіть при хорошому сигналі.
    """

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    BLOCKED = "blocked"

    @property
    def is_tradeable(self) -> bool:
        return self in {
            ExecutionQuality.EXCELLENT,
            ExecutionQuality.GOOD,
            ExecutionQuality.ACCEPTABLE,
        }


class OrderIntent(str, Enum):
    """
    Намір заявки відносно позиції.

    Критично для exposure/open-risk логіки: REDUCE/CLOSE не повинні
    блокуватись як новий ризик.
    """

    OPEN = "open"
    INCREASE = "increase"
    REDUCE = "reduce"
    CLOSE = "close"
    FLIP = "flip"

    @property
    def increases_risk(self) -> bool:
        return self in {
            OrderIntent.OPEN,
            OrderIntent.INCREASE,
            OrderIntent.FLIP,
        }

    @property
    def reduces_risk(self) -> bool:
        return self in {
            OrderIntent.REDUCE,
            OrderIntent.CLOSE,
        }


class PositionSide(str, Enum):
    """
    Напрям позиції.
    """

    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        return 1 if self is PositionSide.LONG else -1

    @property
    def opposite(self) -> PositionSide:
        return PositionSide.SHORT if self is PositionSide.LONG else PositionSide.LONG


class MarginMode(str, Enum):
    """
    Тип маржі.
    """

    ISOLATED = "isolated"
    CROSS = "cross"


class StrategyRiskStatus(str, Enum):
    """
    Runtime-статус strategy budget layer.
    """

    ACTIVE = "active"
    REDUCED = "reduced"
    COOLDOWN = "cooldown"
    DISABLED = "disabled"


class SymbolRiskStatus(str, Enum):
    """
    Runtime-статус symbol budget layer.
    """

    ACTIVE = "active"
    REDUCED = "reduced"
    COOLDOWN = "cooldown"
    DISABLED = "disabled"


class CircuitBreakerReason(str, Enum):
    """
    Причина активації circuit breaker / emergency stop.
    """

    EXTREME_VOLATILITY = "extreme_volatility"
    LOSS_STREAK = "loss_streak"
    SYSTEM_ERROR_RATE = "system_error_rate"
    LIQUIDITY_COLLAPSE = "liquidity_collapse"
    DRAWDOWN_BREACH = "drawdown_breach"
    DAILY_LOSS_BREACH = "daily_loss_breach"
    WEEKLY_LOSS_BREACH = "weekly_loss_breach"
    MONTHLY_LOSS_BREACH = "monthly_loss_breach"
    EMERGENCY_STOP = "emergency_stop"

    MANUAL_HALT = "manual_halt"
    EXECUTION_FAILURES = "execution_failures"
    DATA_FEED_FAILURE = "data_feed_failure"
    DATA_STALE = "data_stale"
    EXCHANGE_UNSTABLE = "exchange_unstable"

    EXECUTION_COST_SPIKE = "execution_cost_spike"
    SLIPPAGE_SPIKE = "slippage_spike"
    SPREAD_ABNORMAL = "spread_abnormal"

    NEGATIVE_GLOBAL_EXPECTANCY = "negative_global_expectancy"
    STRATEGY_FAILURES = "strategy_failures"
    SYMBOL_FAILURES = "symbol_failures"


__all__ = [
    "CircuitBreakerReason",
    "ExecutionQuality",
    "LiquidityClass",
    "MarginMode",
    "OrderIntent",
    "PositionSide",
    "RiskDecisionType",
    "RiskLevel",
    "RiskMode",
    "RiskViolationType",
    "StrategyRiskStatus",
    "SymbolRiskStatus",
    "TradeTier",
    "TradingMode",
]