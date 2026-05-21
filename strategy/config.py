# trading_system/strategy/config.py

from __future__ import annotations

from dataclasses import dataclass, field

from strategy.enums import (
    EntryType,
    MarketRegime,
    PresetMode,
    StrategyCategory,
    Timeframe,
)
from strategy.exceptions import StrategyConfigError


@dataclass(slots=True)
class StrategyRuntimeConfig:
    enabled: bool = True
    symbols: list[str] = field(default_factory=list)
    timeframes: list[Timeframe] = field(default_factory=lambda: [Timeframe.M1])
    allowed_regimes: list[MarketRegime] = field(
        default_factory=lambda: [MarketRegime.UNKNOWN]
    )
    cooldown_seconds: int = 0
    max_signal_age_seconds: int = 30
    min_confidence: float = 0.5
    min_score: float = 0.0

    def validate(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise StrategyConfigError(
                "StrategyRuntimeConfig.min_confidence must be between 0.0 and 1.0"
            )

        if self.min_score < 0:
            raise StrategyConfigError("StrategyRuntimeConfig.min_score must be >= 0")

        if self.cooldown_seconds < 0:
            raise StrategyConfigError(
                "StrategyRuntimeConfig.cooldown_seconds must be >= 0"
            )

        if self.max_signal_age_seconds <= 0:
            raise StrategyConfigError(
                "StrategyRuntimeConfig.max_signal_age_seconds must be > 0"
            )

        if any(not symbol.strip() for symbol in self.symbols):
            raise StrategyConfigError(
                "StrategyRuntimeConfig.symbols cannot contain empty symbols"
            )

        if not self.timeframes:
            raise StrategyConfigError(
                "StrategyRuntimeConfig.timeframes cannot be empty"
            )

        if not self.allowed_regimes:
            raise StrategyConfigError(
                "StrategyRuntimeConfig.allowed_regimes cannot be empty"
            )

    def allows_symbol(self, symbol: str) -> bool:
        if not self.symbols:
            return True
        return symbol in self.symbols

    def allows_timeframe(self, timeframe: Timeframe) -> bool:
        return timeframe in self.timeframes

    def allows_regime(self, regime: MarketRegime) -> bool:
        if MarketRegime.UNKNOWN in self.allowed_regimes:
            return True
        return regime in self.allowed_regimes


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
            raise StrategyConfigError(
                "StrategyDefinitionConfig.name cannot be empty"
            )

        if self.weight < 0:
            raise StrategyConfigError(
                "StrategyDefinitionConfig.weight must be >= 0"
            )

        if self.priority < 0:
            raise StrategyConfigError(
                "StrategyDefinitionConfig.priority must be >= 0"
            )

        if any(not feature.strip() for feature in self.required_features):
            raise StrategyConfigError(
                f"StrategyDefinitionConfig.required_features for '{self.name}' cannot contain empty names"
            )

        if any(not tag.strip() for tag in self.tags):
            raise StrategyConfigError(
                f"StrategyDefinitionConfig.tags for '{self.name}' cannot contain empty tags"
            )

        self.runtime.validate()

DEFAULT_ANALYTICS_EVENT_CATEGORY_PREFIXES: tuple[
    tuple[str, tuple[StrategyCategory, ...]],
    ...
] = (
    # Funding
    ("analytics.funding.", (StrategyCategory.FUNDING,)),

    # Liquidations
    ("analytics.liquidations.", (StrategyCategory.LIQUIDATIONS,)),
    ("analytics.liquidation.", (StrategyCategory.LIQUIDATIONS,)),

    # Liquidity
    ("analytics.liquidity.", (StrategyCategory.LIQUIDITY,)),

    # Open Interest
    ("analytics.oi.", (StrategyCategory.OPEN_INTEREST,)),
    ("analytics.open_interest.", (StrategyCategory.OPEN_INTEREST,)),

    # Orderflow
    ("analytics.orderflow.", (StrategyCategory.ORDERFLOW,)),

    # Price Action
    ("analytics.price_action.", (StrategyCategory.PRICE_ACTION,)),
    ("analytics.market_structure.", (StrategyCategory.PRICE_ACTION,)),
    ("analytics.support_resistance.", (StrategyCategory.PRICE_ACTION,)),
    ("analytics.fair_value_gap.", (StrategyCategory.PRICE_ACTION,)),
    ("analytics.fvg.", (StrategyCategory.PRICE_ACTION,)),
    ("analytics.trend.", (StrategyCategory.PRICE_ACTION,)),

    # Spoofing
    ("analytics.spoofing.", (StrategyCategory.SPOOFING,)),

    # Spreads
    ("analytics.spreads.", (StrategyCategory.SPREADS,)),
    ("analytics.spread.", (StrategyCategory.SPREADS,)),
    ("analytics.basis.", (StrategyCategory.SPREADS,)),

    # Whales
    ("analytics.whales.", (StrategyCategory.WHALES,)),
    ("analytics.whale.", (StrategyCategory.WHALES,)),
)
@dataclass(slots=True)
class RoutingConfig:
    reevaluate_on_any_update: bool = False
    route_hybrid_on_domain_signal: bool = True
    allow_partial_context: bool = True
    stale_feature_threshold_seconds: int = 30
    event_to_categories: dict[str, list[StrategyCategory]] = field(default_factory=dict)

    def validate(self) -> None:
        if self.stale_feature_threshold_seconds <= 0:
            raise StrategyConfigError(
                "RoutingConfig.stale_feature_threshold_seconds must be > 0"
            )

        for event_name, categories in self.event_to_categories.items():
            if not event_name.strip():
                raise StrategyConfigError(
                    "RoutingConfig.event_to_categories cannot contain empty event names"
                )

            if not categories:
                raise StrategyConfigError(
                    f"RoutingConfig.event_to_categories['{event_name}'] cannot be empty"
                )

    def categories_for_event(self, event_name: str) -> list[StrategyCategory]:
        """
        Resolve analytics topic into strategy categories.

        Resolution order:
        1. exact configured event_to_categories match;
        2. configured parent-prefix match;
        3. default production analytics-prefix match;
        4. empty list if event is not routable to strategy.
        """
        normalized = self._normalize_event_name(event_name)
        if not normalized:
            return []

        exact = self.event_to_categories.get(normalized)
        if exact:
            return self._normalize_categories(exact)

        configured_prefix_match = self._categories_from_configured_prefix(normalized)
        if configured_prefix_match:
            return configured_prefix_match

        default_prefix_match = self._categories_from_default_prefix(normalized)
        if default_prefix_match:
            return default_prefix_match

        return []

    @staticmethod
    def _normalize_event_name(event_name: object) -> str:
        if not isinstance(event_name, str):
            return ""
        return event_name.strip().lower()

    def _categories_from_configured_prefix(
            self,
            event_name: str,
    ) -> list[StrategyCategory]:
        matches: list[tuple[int, list[StrategyCategory]]] = []

        for configured_event, categories in self.event_to_categories.items():
            prefix = self._normalize_event_name(configured_event)
            if not prefix:
                continue

            dotted_prefix = prefix if prefix.endswith(".") else f"{prefix}."

            if event_name.startswith(dotted_prefix):
                matches.append((len(dotted_prefix), categories))

        if not matches:
            return []

        _, categories = max(matches, key=lambda item: item[0])
        return self._normalize_categories(categories)

    def _categories_from_default_prefix(
            self,
            event_name: str,
    ) -> list[StrategyCategory]:
        for prefix, categories in DEFAULT_ANALYTICS_EVENT_CATEGORY_PREFIXES:
            if event_name.startswith(prefix):
                return self._normalize_categories(categories)

        return []

    def _normalize_categories(
            self,
            categories: list[StrategyCategory] | tuple[StrategyCategory, ...],
    ) -> list[StrategyCategory]:
        result: list[StrategyCategory] = []

        for category in categories:
            if not isinstance(category, StrategyCategory):
                continue

            if category not in result:
                result.append(category)

        if (
                self.route_hybrid_on_domain_signal
                and result
                and StrategyCategory.HYBRID not in result
        ):
            result.append(StrategyCategory.HYBRID)

        return result


@dataclass(slots=True)
class ConfluenceConfig:
    enabled: bool = True
    min_agreement_count: int = 2
    min_confidence: float = 0.6
    min_score: float = 0.50
    conflict_penalty: float = 0.15
    confirmation_bonus: float = 0.10
    max_strategies_per_side: int = 10

    def validate(self) -> None:
        if self.min_agreement_count < 1:
            raise StrategyConfigError(
                "ConfluenceConfig.min_agreement_count must be >= 1"
            )

        if not 0.0 <= self.min_confidence <= 1.0:
            raise StrategyConfigError(
                "ConfluenceConfig.min_confidence must be between 0.0 and 1.0"
            )

        if self.min_score < 0:
            raise StrategyConfigError(
                "ConfluenceConfig.min_score must be >= 0"
            )

        if self.conflict_penalty < 0:
            raise StrategyConfigError(
                "ConfluenceConfig.conflict_penalty must be >= 0"
            )

        if self.confirmation_bonus < 0:
            raise StrategyConfigError(
                "ConfluenceConfig.confirmation_bonus must be >= 0"
            )

        if self.max_strategies_per_side < 1:
            raise StrategyConfigError(
                "ConfluenceConfig.max_strategies_per_side must be >= 1"
            )


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
            raise StrategyConfigError(
                "ConfidenceConfig thresholds must be between 0.0 and 1.0"
            )

        if values != sorted(values):
            raise StrategyConfigError(
                "ConfidenceConfig thresholds must be non-decreasing"
            )

    def grade_bounds(self) -> tuple[float, float, float, float]:
        return (
            self.very_low_threshold,
            self.low_threshold,
            self.medium_threshold,
            self.high_threshold,
        )


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
                raise StrategyConfigError(
                    f"WeightingConfig.category_weights[{category}] must be >= 0"
                )

        for regime, value in self.regime_adjustments.items():
            if value < 0:
                raise StrategyConfigError(
                    f"WeightingConfig.regime_adjustments[{regime}] must be >= 0"
                )


@dataclass(slots=True)
class VotingConfig:
    min_confirmations: int = 1
    min_total_votes: int = 1
    require_primary_trigger: bool = True
    allow_single_strategy_confirmation: bool = True

    def validate(self) -> None:
        if self.min_confirmations < 0:
            raise StrategyConfigError(
                "VotingConfig.min_confirmations must be >= 0"
            )

        if self.min_total_votes < 1:
            raise StrategyConfigError(
                "VotingConfig.min_total_votes must be >= 1"
            )

        if self.min_confirmations > self.min_total_votes:
            raise StrategyConfigError(
                "VotingConfig.min_confirmations cannot be greater than min_total_votes"
            )


@dataclass(slots=True)
class ConflictConfig:
    reject_on_side_conflict: bool = False
    reject_on_regime_conflict: bool = False
    max_total_penalty: float = 0.5

    def validate(self) -> None:
        if not 0.0 <= self.max_total_penalty <= 10.0:
            raise StrategyConfigError(
                "ConflictConfig.max_total_penalty must be between 0.0 and 10.0"
            )


@dataclass(slots=True)
class FilterConfig:
    enabled: bool = True

    # Runtime-level gates. None means: use StrategyRuntimeConfig fallback.
    min_signal_confidence: float | None = None
    min_signal_score: float | None = None
    min_risk_reward: float = 0.0

    # Optional filter groups. Safety filters such as symbol and directional side
    # remain enforced even when optional filters are disabled.
    enable_cooldown_filter: bool = True
    enable_freshness_filter: bool = True
    enable_portfolio_filter: bool = True
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
        if self.min_signal_confidence is not None and not 0.0 <= self.min_signal_confidence <= 1.0:
            raise StrategyConfigError(
                "FilterConfig.min_signal_confidence must be between 0.0 and 1.0"
            )

        if self.min_signal_score is not None and self.min_signal_score < 0:
            raise StrategyConfigError(
                "FilterConfig.min_signal_score must be >= 0"
            )

        if self.min_risk_reward < 0:
            raise StrategyConfigError(
                "FilterConfig.min_risk_reward must be >= 0"
            )

        if self.max_spread_bps < 0:
            raise StrategyConfigError(
                "FilterConfig.max_spread_bps must be >= 0"
            )

        if not 0.0 <= self.min_liquidity_score <= 1.0:
            raise StrategyConfigError(
                "FilterConfig.min_liquidity_score must be between 0.0 and 1.0"
            )

        if self.max_volatility_zscore < 0:
            raise StrategyConfigError(
                "FilterConfig.max_volatility_zscore must be >= 0"
            )

        if not -1.0 <= self.min_funding_alignment <= 1.0:
            raise StrategyConfigError(
                "FilterConfig.min_funding_alignment must be between -1.0 and 1.0"
            )


@dataclass(slots=True)
class BuilderConfig:
    default_entry_type: EntryType = EntryType.MARKET
    default_rr_ratio: float = 2.0
    enable_partial_take_profit: bool = True
    default_partial_tp_levels: list[float] = field(default_factory=lambda: [0.5, 0.5])
    require_invalidation: bool = True

    def validate(self) -> None:
        if self.default_rr_ratio <= 0:
            raise StrategyConfigError(
                "BuilderConfig.default_rr_ratio must be > 0"
            )

        if not self.default_partial_tp_levels:
            raise StrategyConfigError(
                "BuilderConfig.default_partial_tp_levels cannot be empty"
            )

        if any(level <= 0 for level in self.default_partial_tp_levels):
            raise StrategyConfigError(
                "BuilderConfig.default_partial_tp_levels must contain only positive values"
            )

        total = sum(self.default_partial_tp_levels)
        if total <= 0:
            raise StrategyConfigError(
                "sum(BuilderConfig.default_partial_tp_levels) must be > 0"
            )

        if total > 1.0:
            raise StrategyConfigError(
                "sum(BuilderConfig.default_partial_tp_levels) cannot be greater than 1.0"
            )


@dataclass(slots=True)
class PortfolioCoordinatorConfig:
    enabled: bool = True
    max_signals_per_symbol: int = 3
    deduplicate_by_side: bool = True
    merge_similar_signals: bool = True
    correlation_guard_enabled: bool = True
    symbol_cooldown_seconds: int = 15

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
            raise StrategyConfigError(
                "PortfolioCoordinatorConfig.max_signals_per_symbol must be >= 1"
            )

        if self.symbol_cooldown_seconds < 0:
            raise StrategyConfigError(
                "PortfolioCoordinatorConfig.symbol_cooldown_seconds must be >= 0"
            )

        if self.side_cooldown_seconds < 0:
            raise StrategyConfigError(
                "PortfolioCoordinatorConfig.side_cooldown_seconds must be >= 0"
            )

        if self.repeated_signal_suppression_seconds < 0:
            raise StrategyConfigError(
                "PortfolioCoordinatorConfig.repeated_signal_suppression_seconds must be >= 0"
            )

        if self.volatility_throttle_threshold < 0:
            raise StrategyConfigError(
                "PortfolioCoordinatorConfig.volatility_throttle_threshold must be >= 0"
            )

        if self.high_volatility_max_signals_per_symbol < 1:
            raise StrategyConfigError(
                "PortfolioCoordinatorConfig.high_volatility_max_signals_per_symbol must be >= 1"
            )

        for category, value in self.max_signals_per_category.items():
            if value < 1:
                raise StrategyConfigError(
                    f"PortfolioCoordinatorConfig.max_signals_per_category[{category}] must be >= 1"
                )

        for strategy_name, priority in self.priority_overrides.items():
            if not strategy_name.strip():
                raise StrategyConfigError(
                    "PortfolioCoordinatorConfig.priority_overrides cannot contain empty strategy names"
                )
            if priority < 0:
                raise StrategyConfigError(
                    f"PortfolioCoordinatorConfig.priority_overrides[{strategy_name}] must be >= 0"
                )

        for bucket_name, value in self.exposure_bucket_limits.items():
            if not bucket_name.strip():
                raise StrategyConfigError(
                    "PortfolioCoordinatorConfig.exposure_bucket_limits cannot contain empty bucket names"
                )
            if value < 1:
                raise StrategyConfigError(
                    f"PortfolioCoordinatorConfig.exposure_bucket_limits[{bucket_name}] must be >= 1"
                )


@dataclass(slots=True)
class FeatureFreshnessConfig:
    default_ttl_seconds: int = 30
    per_feature_ttl_seconds: dict[str, int] = field(default_factory=dict)

    def validate(self) -> None:
        if self.default_ttl_seconds <= 0:
            raise StrategyConfigError(
                "FeatureFreshnessConfig.default_ttl_seconds must be > 0"
            )

        for feature_name, ttl in self.per_feature_ttl_seconds.items():
            if not feature_name.strip():
                raise StrategyConfigError(
                    "FeatureFreshnessConfig.per_feature_ttl_seconds cannot contain empty feature names"
                )

            if ttl <= 0:
                raise StrategyConfigError(
                    f"FeatureFreshnessConfig.per_feature_ttl_seconds['{feature_name}'] must be > 0"
                )

    def get_ttl(self, feature_name: str) -> int:
        return self.per_feature_ttl_seconds.get(
            feature_name,
            self.default_ttl_seconds,
        )


@dataclass(slots=True)
class PresetConfig:
    mode: PresetMode = PresetMode.INTRADAY
    enabled_strategy_names: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if any(not name.strip() for name in self.enabled_strategy_names):
            raise StrategyConfigError(
                "PresetConfig.enabled_strategy_names cannot contain empty strategy names"
            )

    def is_strategy_allowed(self, name: str) -> bool:
        if not self.enabled_strategy_names:
            return True
        return name in self.enabled_strategy_names


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
        self.preset.validate()

        for strategy_name, strategy_cfg in self.strategies.items():
            if not strategy_name.strip():
                raise StrategyConfigError(
                    "StrategyConfig.strategies cannot contain empty keys"
                )

            if strategy_name != strategy_cfg.name:
                raise StrategyConfigError(
                    f"Strategy config key '{strategy_name}' does not match embedded name '{strategy_cfg.name}'"
                )

            strategy_cfg.validate()

    def get_strategy(self, name: str) -> StrategyDefinitionConfig | None:
        return self.strategies.get(name)

    def require_strategy(self, name: str) -> StrategyDefinitionConfig:
        strategy = self.get_strategy(name)
        if strategy is None:
            raise StrategyConfigError(f"Strategy '{name}' is not configured")
        return strategy

    def is_strategy_configured(self, name: str) -> bool:
        return name in self.strategies

    def is_strategy_enabled(self, name: str, default: bool = True) -> bool:
        strategy = self.get_strategy(name)
        if strategy is None:
            return default
        return strategy.runtime.enabled

    def is_strategy_allowed_by_preset(self, name: str) -> bool:
        return self.preset.is_strategy_allowed(name)

    def get_strategy_runtime(self, name: str) -> StrategyRuntimeConfig:
        strategy = self.get_strategy(name)
        if strategy is None:
            return self.runtime
        return strategy.runtime

    def get_strategy_weight(self, name: str, default: float = 1.0) -> float:
        strategy = self.get_strategy(name)
        if strategy is None:
            return default
        return strategy.weight

    def get_strategy_priority(self, name: str, default: int = 100) -> int:
        strategy = self.get_strategy(name)
        if strategy is None:
            return default
        return strategy.priority

    def get_strategy_required_features(self, name: str) -> set[str]:
        strategy = self.get_strategy(name)
        if strategy is None:
            return set()
        return set(strategy.required_features)

    def get_category_weight(self, category: StrategyCategory) -> float:
        return self.weighting.category_weights.get(category, 1.0)

    def get_regime_adjustment(self, regime: MarketRegime) -> float:
        return self.weighting.regime_adjustments.get(regime, 1.0)

    def get_feature_ttl(self, feature_name: str) -> int:
        return self.freshness.get_ttl(feature_name)

    def add_strategy(self, strategy: StrategyDefinitionConfig) -> None:
        strategy.validate()

        if strategy.name in self.strategies:
            raise StrategyConfigError(
                f"Strategy '{strategy.name}' is already configured"
            )

        self.strategies[strategy.name] = strategy

    def upsert_strategy(self, strategy: StrategyDefinitionConfig) -> None:
        strategy.validate()
        self.strategies[strategy.name] = strategy

    def remove_strategy(self, name: str) -> StrategyDefinitionConfig | None:
        return self.strategies.pop(name, None)

    def get_enabled_strategy_names(self) -> list[str]:
        return [
            name
            for name, strategy in self.strategies.items()
            if strategy.runtime.enabled and self.preset.is_strategy_allowed(name)
        ]

    def enabled_strategy_names(self) -> list[str]:
        """Backward-compatible method alias. Prefer get_enabled_strategy_names()."""
        return self.get_enabled_strategy_names()

    def strategies_by_category(
        self,
        category: StrategyCategory,
        *,
        enabled_only: bool = False,
    ) -> list[StrategyDefinitionConfig]:
        strategies = [
            strategy
            for strategy in self.strategies.values()
            if strategy.category == category
        ]

        if enabled_only:
            strategies = [
                strategy
                for strategy in strategies
                if strategy.runtime.enabled
                and self.preset.is_strategy_allowed(strategy.name)
            ]

        return sorted(strategies, key=lambda item: item.priority)