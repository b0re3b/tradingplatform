# trading_system/strategy/registry.py

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from core.event_bus import EventBus, EventPriority
from core.scheduler import Scheduler

from .base import BaseStrategy, BaseStrategyComponent
from .config import StrategyConfig
from .enums import FeatureSource, MarketRegime, StrategyCategory, Timeframe
from .exceptions import (
    StrategyRegistrationError,
    UnsupportedStrategyError,
)
from .models import StrategyContext


class StrategyRegistry(BaseStrategyComponent):
    """
    Registry / selector for strategy instances.

    Responsibilities:
    - register / unregister concrete strategy instances;
    - lookup by strategy name;
    - grouping by StrategyCategory;
    - indexing by required feature names;
    - selecting strategies by StrategyContext, category, timeframe, regime,
      symbol and changed features.

    Forbidden responsibilities:
    - no signal generation;
    - no scoring;
    - no confluence;
    - no SignalBuilder logic;
    - no RiskReadySignalPayload creation;
    - no risk/trading/execution decisions.
    """

    component_namespace: str = "strategy.registry"

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
        )

        self._strategies: dict[str, BaseStrategy] = {}
        self._by_category: dict[StrategyCategory, set[str]] = defaultdict(set)
        self._by_timeframe: dict[Timeframe, set[str]] = defaultdict(set)
        self._feature_index: dict[str, set[str]] = defaultdict(set)
        self._by_symbol: dict[str, set[str]] = defaultdict(set)
        self._by_regime: dict[MarketRegime, set[str]] = defaultdict(set)

    def register(self) -> None:
        """
        Registry currently has no EventBus subscriptions.

        Kept for lifecycle compatibility with BaseStrategyComponent.
        """
        self._registered = True

    async def start(self) -> None:
        await super().start()
        await self.emit_event(
            "strategy.registry.started",
            self.summary(),
            priority=EventPriority.LOW,
            source=self.component_name,
        )

    async def stop(self) -> None:
        await self.emit_event(
            "strategy.registry.stopped",
            self.summary(),
            priority=EventPriority.LOW,
            source=self.component_name,
        )
        await super().stop()

    def register_strategy(
        self,
        strategy: BaseStrategy,
        *,
        replace: bool = False,
        emit_event: bool = True,
    ) -> None:
        """
        Register one concrete strategy instance.

        If replace=False and strategy already exists, raises
        StrategyRegistrationError.
        """
        self._validate_strategy_instance(strategy)

        name = strategy.strategy_name

        if name in self._strategies and not replace:
            raise StrategyRegistrationError(f"strategy '{name}' is already registered")

        if name in self._strategies and replace:
            self._remove_indexes(name, self._strategies[name])

        strategy.validate_config()

        self._strategies[name] = strategy
        self._add_indexes(name, strategy)

        self.log_info(
            "Strategy registered",
            strategy_name=name,
            category=strategy.category.value,
            priority=strategy.priority,
            required_features=sorted(strategy.required_features()),
        )

        if emit_event:
            self._emit_registry_event_nowait(
                "strategy.registry.strategy_registered",
                {
                    "strategy_name": name,
                    "category": strategy.category.value,
                    "priority": strategy.priority,
                    "required_features": sorted(strategy.required_features()),
                    "total": len(self._strategies),
                },
            )

    def register_many(
        self,
        strategies: Iterable[BaseStrategy],
        *,
        replace: bool = False,
        emit_event: bool = True,
    ) -> None:
        for strategy in strategies:
            self.register_strategy(
                strategy,
                replace=replace,
                emit_event=emit_event,
            )

    def unregister_strategy(
        self,
        strategy_name: str,
        *,
        emit_event: bool = True,
    ) -> BaseStrategy:
        name = self._require_strategy_name(strategy_name)

        strategy = self._strategies.pop(name, None)
        if strategy is None:
            raise StrategyRegistrationError(f"strategy '{name}' is not registered")

        self._remove_indexes(name, strategy)

        self.log_info(
            "Strategy unregistered",
            strategy_name=name,
            category=strategy.category.value,
        )

        if emit_event:
            self._emit_registry_event_nowait(
                "strategy.registry.strategy_unregistered",
                {
                    "strategy_name": name,
                    "category": strategy.category.value,
                    "total": len(self._strategies),
                },
            )

        return strategy

    def clear(self, *, emit_event: bool = True) -> None:
        total = len(self._strategies)

        self._strategies.clear()
        self._by_category.clear()
        self._by_timeframe.clear()
        self._feature_index.clear()
        self._by_symbol.clear()
        self._by_regime.clear()

        self.log_info("Strategy registry cleared", total_removed=total)

        if emit_event:
            self._emit_registry_event_nowait(
                "strategy.registry.cleared",
                {"total_removed": total},
            )

    def get(self, strategy_name: str) -> BaseStrategy | None:
        if not isinstance(strategy_name, str) or not strategy_name.strip():
            return None
        return self._strategies.get(strategy_name.strip())

    def require(self, strategy_name: str) -> BaseStrategy:
        name = self._require_strategy_name(strategy_name)

        strategy = self.get(name)
        if strategy is None:
            raise UnsupportedStrategyError(f"strategy '{name}' is not registered")

        return strategy

    def has(self, strategy_name: str) -> bool:
        return self.get(strategy_name) is not None

    def count(self) -> int:
        return len(self._strategies)

    def is_empty(self) -> bool:
        return not self._strategies

    def list_all(self) -> list[BaseStrategy]:
        return sorted(
            self._strategies.values(),
            key=lambda strategy: (strategy.priority, strategy.strategy_name),
        )

    def list_names(self) -> list[str]:
        return [strategy.strategy_name for strategy in self.list_all()]

    def list_by_category(
        self,
        category: StrategyCategory,
    ) -> list[BaseStrategy]:
        names = self._by_category.get(category, set())
        return self._strategies_by_names(names)

    def list_by_timeframe(
        self,
        timeframe: Timeframe,
    ) -> list[BaseStrategy]:
        names = self._by_timeframe.get(timeframe, set())
        return self._strategies_by_names(names)

    def list_by_feature(
        self,
        feature_name: str,
    ) -> list[BaseStrategy]:
        if not feature_name.strip():
            return []
        names = self._feature_index.get(feature_name.strip(), set())
        return self._strategies_by_names(names)

    def list_by_symbol(
        self,
        symbol: str,
    ) -> list[BaseStrategy]:
        if not symbol.strip():
            return []

        names = self._by_symbol.get(symbol.strip(), set())
        if not names:
            return self.list_all()

        return self._strategies_by_names(names)

    def list_by_regime(
        self,
        regime: MarketRegime,
    ) -> list[BaseStrategy]:
        names = self._by_regime.get(regime, set())

        # UNKNOWN in strategy config means "allowed in any regime".
        unknown_names = self._by_regime.get(MarketRegime.UNKNOWN, set())
        return self._strategies_by_names(names | unknown_names)

    def select(
        self,
        *,
        context: StrategyContext,
        categories: list[StrategyCategory] | set[StrategyCategory] | None = None,
        changed_features: list[str] | set[str] | None = None,
        source: FeatureSource | None = None,
        include_disabled: bool = False,
    ) -> list[BaseStrategy]:
        """
        Select applicable strategies for the given StrategyContext.

        This method only selects strategy instances. It does not evaluate them.
        """
        context.validate()

        candidates = self._candidate_names(
            context=context,
            categories=set(categories or []),
            changed_features=set(changed_features or []),
            source=source,
        )

        selected: list[BaseStrategy] = []

        for name in candidates:
            strategy = self._strategies.get(name)
            if strategy is None:
                continue

            if not include_disabled and not strategy.is_enabled():
                continue

            if not self._strategy_matches_context(strategy, context):
                continue

            selected.append(strategy)

        return sorted(
            selected,
            key=lambda strategy: (strategy.priority, strategy.strategy_name),
        )

    def select_for_event(
        self,
        *,
        context: StrategyContext,
        event_name: str,
        categories: list[StrategyCategory] | set[StrategyCategory] | None = None,
        changed_features: list[str] | set[str] | None = None,
        source: FeatureSource | None = None,
        include_disabled: bool = False,
    ) -> list[BaseStrategy]:
        """
        Convenience selector for SignalRouter.

        event_name is stored only for metadata/debug compatibility. Registry
        does not parse trading logic from event names.
        """
        if not event_name.strip():
            raise StrategyRegistrationError("event_name cannot be empty")

        return self.select(
            context=context,
            categories=categories,
            changed_features=changed_features,
            source=source,
            include_disabled=include_disabled,
        )

    def categories(self) -> list[StrategyCategory]:
        return sorted(self._by_category.keys(), key=lambda item: item.value)

    def features(self) -> list[str]:
        return sorted(self._feature_index.keys())

    def summary(self) -> dict[str, Any]:
        return {
            "total": len(self._strategies),
            "strategies": self.list_names(),
            "by_category": {
                category.value: sorted(names)
                for category, names in self._by_category.items()
            },
            "by_timeframe": {
                timeframe.value: sorted(names)
                for timeframe, names in self._by_timeframe.items()
            },
            "feature_index": {
                feature: sorted(names)
                for feature, names in self._feature_index.items()
            },
            "by_symbol": {
                symbol: sorted(names)
                for symbol, names in self._by_symbol.items()
            },
            "by_regime": {
                regime.value: sorted(names)
                for regime, names in self._by_regime.items()
            },
            "registered": self.is_registered,
            "started": self.is_started,
        }

    def _candidate_names(
        self,
        *,
        context: StrategyContext,
        categories: set[StrategyCategory],
        changed_features: set[str],
        source: FeatureSource | None,
    ) -> set[str]:
        if not self._strategies:
            return set()

        candidate_sets: list[set[str]] = []

        if categories:
            category_names: set[str] = set()
            for category in categories:
                category_names.update(self._by_category.get(category, set()))
            candidate_sets.append(category_names)

        if source is not None:
            mapped_category = self._source_to_category(source)
            if mapped_category is not None:
                candidate_sets.append(set(self._by_category.get(mapped_category, set())))

        if changed_features:
            feature_names: set[str] = set()
            for feature in changed_features:
                if not isinstance(feature, str) or not feature.strip():
                    continue
                feature_names.update(self._feature_index.get(feature.strip(), set()))

            if feature_names:
                candidate_sets.append(feature_names)

        if context.timeframe in self._by_timeframe:
            candidate_sets.append(set(self._by_timeframe[context.timeframe]))

        symbol_specific = self._by_symbol.get(context.symbol)
        if symbol_specific:
            candidate_sets.append(set(symbol_specific))

        if not candidate_sets:
            return set(self._strategies.keys())

        result = candidate_sets[0]
        for candidate_set in candidate_sets[1:]:
            if candidate_set:
                result = result & candidate_set

        if not result:
            # Fallback: if strict intersection is empty, use broad context selector.
            result = set().union(*candidate_sets)

        return result

    def _strategy_matches_context(
        self,
        strategy: BaseStrategy,
        context: StrategyContext,
    ) -> bool:
        if not strategy.supports_symbol(context.symbol):
            return False

        if not strategy.supports_timeframe(context.timeframe):
            return False

        regime = context.current_regime
        if not strategy.supports_regime(regime):
            return False

        required = strategy.required_features()
        if required and not all(context.has_feature(feature) for feature in required):
            return False

        return True

    def _add_indexes(
        self,
        name: str,
        strategy: BaseStrategy,
    ) -> None:
        self._by_category[strategy.category].add(name)

        for timeframe in strategy.supported_timeframes():
            self._by_timeframe[timeframe].add(name)

        for feature in strategy.required_features():
            if feature.strip():
                self._feature_index[feature.strip()].add(name)

        for symbol in strategy.allowed_symbols():
            if symbol.strip():
                self._by_symbol[symbol.strip()].add(name)

        for regime in strategy.supported_regimes():
            self._by_regime[regime].add(name)

    def _remove_indexes(
        self,
        name: str,
        strategy: BaseStrategy,
    ) -> None:
        self._discard_from_index(self._by_category, strategy.category, name)

        for timeframe in strategy.supported_timeframes():
            self._discard_from_index(self._by_timeframe, timeframe, name)

        for feature in strategy.required_features():
            self._discard_from_index(self._feature_index, feature, name)

        for symbol in strategy.allowed_symbols():
            self._discard_from_index(self._by_symbol, symbol, name)

        for regime in strategy.supported_regimes():
            self._discard_from_index(self._by_regime, regime, name)

    @staticmethod
    def _discard_from_index(
        index: dict[Any, set[str]],
        key: Any,
        name: str,
    ) -> None:
        names = index.get(key)
        if names is None:
            return

        names.discard(name)

        if not names:
            index.pop(key, None)

    def _strategies_by_names(
        self,
        names: set[str],
    ) -> list[BaseStrategy]:
        result = [
            self._strategies[name]
            for name in names
            if name in self._strategies
        ]

        return sorted(
            result,
            key=lambda strategy: (strategy.priority, strategy.strategy_name),
        )

    def _emit_registry_event_nowait(
        self,
        topic: str,
        payload: dict[str, Any],
    ) -> None:
        if self.event_bus is None:
            return

        self.emit_event_nowait_best_effort(
            topic,
            payload,
            priority=EventPriority.LOW,
            source=self.component_name,
        )

    @staticmethod
    def _validate_strategy_instance(strategy: BaseStrategy) -> None:
        if strategy is None:
            raise StrategyRegistrationError("strategy cannot be None")

        if not isinstance(strategy, BaseStrategy):
            raise StrategyRegistrationError(
                f"strategy must be an instance of BaseStrategy, got {type(strategy)!r}"
            )

        if not strategy.strategy_name.strip():
            raise StrategyRegistrationError("strategy name cannot be empty")

    @staticmethod
    def _require_strategy_name(strategy_name: str) -> str:
        if not isinstance(strategy_name, str) or not strategy_name.strip():
            raise StrategyRegistrationError("strategy_name cannot be empty")
        return strategy_name.strip()

    @staticmethod
    def _source_to_category(source: FeatureSource) -> StrategyCategory | None:
        mapping: dict[FeatureSource, StrategyCategory] = {
            FeatureSource.ORDERFLOW: StrategyCategory.ORDERFLOW,
            FeatureSource.LIQUIDITY: StrategyCategory.LIQUIDITY,
            FeatureSource.PRICE_ACTION: StrategyCategory.PRICE_ACTION,
            FeatureSource.LIQUIDATIONS: StrategyCategory.LIQUIDATIONS,
            FeatureSource.WHALES: StrategyCategory.WHALES,
            FeatureSource.SPOOFING: StrategyCategory.SPOOFING,
            FeatureSource.SPREADS: StrategyCategory.SPREADS,
            FeatureSource.FUNDING: StrategyCategory.FUNDING,
            FeatureSource.OPEN_INTEREST: StrategyCategory.OPEN_INTEREST,
        }
        return mapping.get(source)


__all__ = [
    "StrategyRegistry",
]