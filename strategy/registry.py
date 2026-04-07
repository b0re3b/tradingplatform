from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .base import StrategyComponent
from .context import StrategyContext
from .enums import MarketRegime, StrategyCategory, Timeframe
from .exceptions import StrategyRegistrationError, UnsupportedStrategyError
from .strategies.base_strategy import BaseStrategy


class StrategyRegistry(StrategyComponent):
    """
    Реєстр усіх strategy instances.

    Відповідає за:
    - register / unregister
    - lookup by name
    - grouping by category
    - selection by symbol/timeframe/regime
    - mapping required feature -> strategies
    """

    def __init__(self, config, event_bus=None, logger=None) -> None:
        super().__init__(config=config, event_bus=event_bus, logger=logger)
        self._strategies: dict[str, BaseStrategy] = {}
        self._by_category: dict[StrategyCategory, set[str]] = defaultdict(set)
        self._feature_index: dict[str, set[str]] = defaultdict(set)

    def register(self, strategy: BaseStrategy) -> None:
        if strategy is None:
            raise StrategyRegistrationError("strategy cannot be None")

        name = strategy.strategy_name
        if not name.strip():
            raise StrategyRegistrationError("strategy name cannot be empty")

        if name in self._strategies:
            raise StrategyRegistrationError(f"strategy '{name}' is already registered")

        strategy.validate_config()

        self._strategies[name] = strategy
        self._by_category[strategy.category].add(name)

        for feature_name in strategy.required_features():
            self._feature_index[feature_name].add(name)

        self.log_info(
            "Strategy registered",
            strategy_name=name,
            category=str(strategy.category),
        )

    def register_many(self, strategies: Iterable[BaseStrategy]) -> None:
        for strategy in strategies:
            self.register(strategy)

    def unregister(self, strategy_name: str) -> None:
        strategy = self._strategies.pop(strategy_name, None)
        if strategy is None:
            raise StrategyRegistrationError(f"strategy '{strategy_name}' is not registered")

        self._by_category[strategy.category].discard(strategy_name)

        for feature_name in strategy.required_features():
            names = self._feature_index.get(feature_name)
            if names is None:
                continue
            names.discard(strategy_name)
            if not names:
                self._feature_index.pop(feature_name, None)

        self.log_info("Strategy unregistered", strategy_name=strategy_name)

    def clear(self) -> None:
        self._strategies.clear()
        self._by_category.clear()
        self._feature_index.clear()
        self.log_info("Strategy registry cleared")

    def get(self, strategy_name: str) -> BaseStrategy | None:
        return self._strategies.get(strategy_name)

    def require(self, strategy_name: str) -> BaseStrategy:
        strategy = self.get(strategy_name)
        if strategy is None:
            raise UnsupportedStrategyError(f"strategy '{strategy_name}' is not registered")
        return strategy

    def has(self, strategy_name: str) -> bool:
        return strategy_name in self._strategies

    def list_all(self) -> list[BaseStrategy]:
        return list(self._strategies.values())

    def list_names(self) -> list[str]:
        return list(self._strategies.keys())

    def list_enabled(self) -> list[BaseStrategy]:
        return [strategy for strategy in self._strategies.values() if strategy.is_enabled()]

    def list_by_category(self, category: StrategyCategory) -> list[BaseStrategy]:
        names = self._by_category.get(category, set())
        return [self._strategies[name] for name in names if name in self._strategies]

    def find_by_required_feature(self, feature_name: str) -> list[BaseStrategy]:
        names = self._feature_index.get(feature_name, set())
        return [self._strategies[name] for name in names if name in self._strategies]

    def find_applicable(self, context: StrategyContext) -> list[BaseStrategy]:
        """
        Повертає всі стратегії, які теоретично можна оцінювати на цьому context.
        """
        context.validate()

        applicable: list[BaseStrategy] = []
        for strategy in self._strategies.values():
            try:
                if strategy.should_evaluate(context):
                    applicable.append(strategy)
            except Exception as exc:
                self.log_warning(
                    "Strategy applicability check failed",
                    strategy_name=strategy.strategy_name,
                    symbol=context.symbol,
                    error=str(exc),
                )

        applicable.sort(key=lambda item: item.priority)
        return applicable

    def find_by_categories(
        self,
        categories: Iterable[StrategyCategory],
        *,
        only_enabled: bool = True,
    ) -> list[BaseStrategy]:
        seen: set[str] = set()
        result: list[BaseStrategy] = []

        for category in categories:
            for strategy in self.list_by_category(category):
                if strategy.strategy_name in seen:
                    continue
                if only_enabled and not strategy.is_enabled():
                    continue
                result.append(strategy)
                seen.add(strategy.strategy_name)

        result.sort(key=lambda item: item.priority)
        return result

    def find_by_context_filters(
        self,
        *,
        symbol: str | None = None,
        timeframe: Timeframe | None = None,
        regime: MarketRegime | None = None,
        category: StrategyCategory | None = None,
        only_enabled: bool = True,
    ) -> list[BaseStrategy]:
        strategies = (
            self.list_by_category(category)
            if category is not None
            else self.list_all()
        )

        result: list[BaseStrategy] = []
        for strategy in strategies:
            if only_enabled and not strategy.is_enabled():
                continue

            allowed_symbols = strategy.allowed_symbols()
            if symbol is not None and allowed_symbols and symbol not in allowed_symbols:
                continue

            supported_timeframes = strategy.supported_timeframes()
            if timeframe is not None and supported_timeframes and timeframe not in supported_timeframes:
                continue

            supported_regimes = strategy.supported_regimes()
            if regime is not None and supported_regimes:
                if MarketRegime.UNKNOWN not in supported_regimes and regime not in supported_regimes:
                    continue

            result.append(strategy)

        result.sort(key=lambda item: item.priority)
        return result

    def required_feature_map(self) -> dict[str, list[str]]:
        return {
            feature_name: sorted(strategy_names)
            for feature_name, strategy_names in self._feature_index.items()
        }

    def category_map(self) -> dict[str, list[str]]:
        return {
            str(category): sorted(strategy_names)
            for category, strategy_names in self._by_category.items()
        }

    def summary(self) -> dict[str, object]:
        return {
            "total": len(self._strategies),
            "enabled": len(self.list_enabled()),
            "categories": self.category_map(),
            "required_features": self.required_feature_map(),
        }