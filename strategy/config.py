from __future__ import annotations

from dataclasses import dataclass, field

from .enums import (
    EntryType,
    MarketRegime,
    PresetMode,
    StrategyCategory,
    Timeframe,
)
from .exceptions import StrategyConfigError


@dataclass(slots=True)
class StrategyRuntimeConfig:
    enabled: bool = True
    symbols: list[str] = field(default_factory=list)
    timeframes: list[Timeframe] = field(default_factory=lambda: [Timeframe.M1])
    allowed_regimes: list[MarketRegime] = field(default_factory=lambda: [MarketRegime.UNKNOWN])
    cooldown_seconds: int = 0
    max_signal_age_seconds: int = 30
    min_confidence: float = 0.5
    min_score: float = 0.0

    def validate(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise StrategyConfigError("min_confidence must be between 0.0 and 1.0")
        if self.cooldown_seconds < 0:
            raise StrategyConfigError("cooldown_seconds must be >= 0")
        if self.max_signal_age_seconds <= 0:
            raise StrategyConfigError("max_signal_age_seconds must be > 0")


@dataclass(slots=True)
class StrategyDefinitionConfig:
    name: str
    category: StrategyCategory
    runtime: StrategyRuntimeConfig = field(default_factory=StrategyRuntimeConfig)
    required_features: set[str] = field(default_factory=set)
    weight: float = 1.0
    priority: int = 100
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name.strip():
            raise StrategyConfigError("StrategyDefinitionConfig.name cannot be empty")
        if self.weight < 0:
            raise StrategyConfigError("StrategyDefinitionConfig.weight must be >= 0")
        self.runtime.validate()


@dataclass(slots=True)
class RoutingConfig:
    reevaluate_on_any_update: bool = False
    route_hybrid_on_domain_signal: bool = True
    allow_partial_context: bool = True
    stale_feature_threshold_seconds: int = 30
    event_to_categories: dict[str, list[StrategyCategory]] = field(default_factory=dict)

    def validate(self) -> None:
        if self.stale_feature_threshold_seconds <= 0:
            raise StrategyConfigError("stale_feature_threshold_seconds must be > 0")


@dataclass(slots=True)
class ConfluenceConfig:
    enabled: bool = True
    min_agreement_count: int = 2
    min_confidence: float = 0.6
    min_score: float = 1.0
    conflict_penalty: float = 0.15
    confirmation_bonus: float = 0.10
    max_strategies_per_side: int = 10

    def validate(self) -> None:
        if self.min_agreement_count < 1:
            raise StrategyConfigError("min_agreement_count must be >= 1")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise StrategyConfigError("ConfluenceConfig.min_confidence must be between 0.0 and 1.0")
        if self.min_score < 0:
            raise StrategyConfigError("ConfluenceConfig.min_score must be >= 0")
        if self.conflict_penalty < 0:
            raise StrategyConfigError("conflict_penalty must be >= 0")
        if self.confirmation_bonus < 0:
            raise StrategyConfigError("confirmation_bonus must be >= 0")
        if self.max_strategies_per_side < 1:
            raise StrategyConfigError("max_strategies_per_side must be >= 1")


@dataclass(slots=True)
class ConfidenceConfig:
    very_low_threshold: float = 0.35
    low_threshold: float = 0.55
    medium_threshold: float = 0.75
    high_threshold: float = 0.90

    def validate(self) -> None:
        values = [
            self.very_low_threshold,
            self.low_threshold,
            self.medium_threshold,
            self.high_threshold,
        ]
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise StrategyConfigError("All confidence thresholds must be between 0.0 and 1.0")
        if values != sorted(values):
            raise StrategyConfigError("Confidence thresholds must be non-decreasing")


@dataclass(slots=True)
class WeightingConfig:
    category_weights: dict[StrategyCategory, float] = field(
        default_factory=lambda: {
            StrategyCategory.ORDERFLOW: 1.00,
            StrategyCategory.LIQUIDITY: 1.00,
            StrategyCategory.PRICE_ACTION: 0.85,
            StrategyCategory.LIQUIDATIONS: 0.95,
            StrategyCategory.WHALES: 0.90,
            StrategyCategory.SPOOFING: 0.80,
            StrategyCategory.SPREADS: 0.70,
            StrategyCategory.FUNDING: 0.70,
            StrategyCategory.OPEN_INTEREST: 0.80,
            StrategyCategory.HYBRID: 1.20,
        }
    )
    regime_adjustments: dict[MarketRegime, float] = field(
        default_factory=lambda: {
            MarketRegime.TRENDING_UP: 1.00,
            MarketRegime.TRENDING_DOWN: 1.00,
            MarketRegime.RANGING: 0.90,
            MarketRegime.BREAKOUT: 1.10,
            MarketRegime.SQUEEZE: 1.05,
            MarketRegime.HIGH_VOLATILITY: 0.85,
            MarketRegime.LOW_VOLATILITY: 0.85,
            MarketRegime.NEWS_DRIVEN: 0.70,
            MarketRegime.ILLIQUID: 0.50,
            MarketRegime.RISK_OFF: 0.60,
            MarketRegime.UNKNOWN: 1.00,
        }
    )

    def validate(self) -> None:
        for category, value in self.category_weights.items():
            if value < 0:
                raise StrategyConfigError(f"Category weight must be >= 0 for {category}")

        for regime, value in self.regime_adjustments.items():
            if value < 0:
                raise StrategyConfigError(f"Regime adjustment must be >= 0 for {regime}")


@dataclass(slots=True)
class VotingConfig:
    min_confirmations: int = 1
    min_total_votes: int = 1
    require_primary_trigger: bool = True
    allow_single_strategy_confirmation: bool = True

    def validate(self) -> None:
        if self.min_confirmations < 0:
            raise StrategyConfigError("min_confirmations must be >= 0")
        if self.min_total_votes < 1:
            raise StrategyConfigError("min_total_votes must be >= 1")


@dataclass(slots=True)
class ConflictConfig:
    reject_on_side_conflict: bool = False
    reject_on_regime_conflict: bool = False
    max_total_penalty: float = 0.5

    def validate(self) -> None:
        if not 0.0 <= self.max_total_penalty <= 10.0:
            raise StrategyConfigError("max_total_penalty must be between 0.0 and 10.0")


@dataclass(slots=True)
class FilterConfig:
    enable_regime_filter: bool = True
    enable_volatility_filter: bool = True
    enable_liquidity_filter: bool = True
    enable_spread_filter: bool = True
    enable_funding_filter: bool = True
    enable_session_filter: bool = False
    enable_news_filter: bool = False

    max_spread_bps: float = 20.0
    min_liquidity_score: float = 0.30
    max_volatility_zscore: float = 3.0
    min_funding_alignment: float = -1.0

    def validate(self) -> None:
        if self.max_spread_bps < 0:
            raise StrategyConfigError("max_spread_bps must be >= 0")
        if not 0.0 <= self.min_liquidity_score <= 1.0:
            raise StrategyConfigError("min_liquidity_score must be between 0.0 and 1.0")
        if self.max_volatility_zscore < 0:
            raise StrategyConfigError("max_volatility_zscore must be >= 0")


@dataclass(slots=True)
class BuilderConfig:
    default_entry_type: EntryType = EntryType.MARKET
    default_rr_ratio: float = 2.0
    enable_partial_take_profit: bool = True
    default_partial_tp_levels: list[float] = field(default_factory=lambda: [0.5, 0.5])
    require_invalidation: bool = True

    def validate(self) -> None:
        if self.default_rr_ratio <= 0:
            raise StrategyConfigError("default_rr_ratio must be > 0")
        if not self.default_partial_tp_levels:
            raise StrategyConfigError("default_partial_tp_levels cannot be empty")
        total = sum(self.default_partial_tp_levels)
        if total <= 0:
            raise StrategyConfigError("sum(default_partial_tp_levels) must be > 0")


@dataclass(slots=True)
class PortfolioCoordinatorConfig:
    enabled: bool = True
    max_signals_per_symbol: int = 3
    deduplicate_by_side: bool = True
    merge_similar_signals: bool = True
    correlation_guard_enabled: bool = True
    symbol_cooldown_seconds: int = 15

    def validate(self) -> None:
        if self.max_signals_per_symbol < 1:
            raise StrategyConfigError("max_signals_per_symbol must be >= 1")
        if self.symbol_cooldown_seconds < 0:
            raise StrategyConfigError("symbol_cooldown_seconds must be >= 0")


@dataclass(slots=True)
class FeatureFreshnessConfig:
    default_ttl_seconds: int = 30
    per_feature_ttl_seconds: dict[str, int] = field(default_factory=dict)

    def validate(self) -> None:
        if self.default_ttl_seconds <= 0:
            raise StrategyConfigError("default_ttl_seconds must be > 0")
        for feature_name, ttl in self.per_feature_ttl_seconds.items():
            if ttl <= 0:
                raise StrategyConfigError(f"TTL must be > 0 for feature '{feature_name}'")

    def get_ttl(self, feature_name: str) -> int:
        return self.per_feature_ttl_seconds.get(feature_name, self.default_ttl_seconds)


@dataclass(slots=True)
class PresetConfig:
    mode: PresetMode = PresetMode.INTRADAY
    enabled_strategy_names: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyConfig:
    runtime: StrategyRuntimeConfig = field(default_factory=StrategyRuntimeConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    confluence: ConfluenceConfig = field(default_factory=ConfluenceConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    voting: VotingConfig = field(default_factory=VotingConfig)
    conflict: ConflictConfig = field(default_factory=ConflictConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    builders: BuilderConfig = field(default_factory=BuilderConfig)
    freshness: FeatureFreshnessConfig = field(default_factory=FeatureFreshnessConfig)
    portfolio: PortfolioCoordinatorConfig = field(default_factory=PortfolioCoordinatorConfig)
    preset: PresetConfig = field(default_factory=PresetConfig)

    strategies: dict[str, StrategyDefinitionConfig] = field(default_factory=dict)

    def validate(self) -> None:
        self.runtime.validate()
        self.routing.validate()
        self.confluence.validate()
        self.confidence.validate()
        self.weighting.validate()
        self.voting.validate()
        self.conflict.validate()
        self.filters.validate()
        self.builders.validate()
        self.freshness.validate()
        self.portfolio.validate()

        for strategy_name, strategy_cfg in self.strategies.items():
            if strategy_name != strategy_cfg.name:
                raise StrategyConfigError(
                    f"Strategy config key '{strategy_name}' does not match embedded name '{strategy_cfg.name}'"
                )
            strategy_cfg.validate()

    def get_strategy(self, name: str) -> StrategyDefinitionConfig | None:
        return self.strategies.get(name)

    def is_strategy_enabled(self, name: str) -> bool:
        strategy = self.get_strategy(name)
        if strategy is None:
            return False
        return strategy.runtime.enabled

    def get_strategy_weight(self, name: str, default: float = 1.0) -> float:
        strategy = self.get_strategy(name)
        if strategy is None:
            return default
        return strategy.weight

    def get_category_weight(self, category: StrategyCategory) -> float:
        return self.weighting.category_weights.get(category, 1.0)

    def get_regime_adjustment(self, regime: MarketRegime) -> float:
        return self.weighting.regime_adjustments.get(regime, 1.0)
@dataclass(slots=True)
class PortfolioCoordinatorConfig:
    enabled: bool = True
    max_signals_per_symbol: int = 3
    deduplicate_by_side: bool = True
    merge_similar_signals: bool = True
    correlation_guard_enabled: bool = True
    symbol_cooldown_seconds: int = 15

    # NEW
    side_cooldown_seconds: int = 10
    repeated_signal_suppression_seconds: int = 30
    volatility_throttle_enabled: bool = True
    volatility_throttle_threshold: float = 3.0
    high_volatility_max_signals_per_symbol: int = 1

    max_signals_per_category: dict[StrategyCategory, int] = field(default_factory=dict)
    priority_overrides: dict[str, int] = field(default_factory=dict)

    exposure_bucket_limits: dict[str, int] = field(default_factory=dict)
    enable_correlation_direction_conflict: bool = True

    def validate(self) -> None:
        if self.max_signals_per_symbol < 1:
            raise StrategyConfigError("max_signals_per_symbol must be >= 1")
        if self.symbol_cooldown_seconds < 0:
            raise StrategyConfigError("symbol_cooldown_seconds must be >= 0")
        if self.side_cooldown_seconds < 0:
            raise StrategyConfigError("side_cooldown_seconds must be >= 0")
        if self.repeated_signal_suppression_seconds < 0:
            raise StrategyConfigError("repeated_signal_suppression_seconds must be >= 0")
        if self.volatility_throttle_threshold < 0:
            raise StrategyConfigError("volatility_throttle_threshold must be >= 0")
        if self.high_volatility_max_signals_per_symbol < 1:
            raise StrategyConfigError("high_volatility_max_signals_per_symbol must be >= 1")

        for category, value in self.max_signals_per_category.items():
            if value < 1:
                raise StrategyConfigError(f"max_signals_per_category[{category}] must be >= 1")

        for bucket_name, value in self.exposure_bucket_limits.items():
            if value < 1:
                raise StrategyConfigError(f"exposure_bucket_limits[{bucket_name}] must be >= 1")