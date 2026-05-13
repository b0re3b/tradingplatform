# trading_system/strategy/registry.py

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from core.event_bus import EventBus, EventPriority
from core.scheduler import Scheduler

from .base import BaseStrategy, BaseStrategyComponent
from .config import StrategyConfig
from .enums import MarketRegime, StrategyCategory, Timeframe
from .exceptions import (
    StrategyRegistrationError,
    UnsupportedStrategyError,
)
from .models import StrategyContext


class StrategyRegistry(BaseStrategyComponent):
    """
    Registry for all strategy instances.

    Відповідає за:
    - register / unregister;
    - lookup by strategy name;
    - grouping by category;
    - selection by symbol/timeframe/regime/context;
    - mapping required feature -> strategies.

    Registry не виконує торгову логіку і не оцінює сигнали.
    Він тільки зберігає strategy instances і допомагає engine/processor
    знайти відповідні стратегії.
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
        self._feature_index: dict[str, set[str]] = defaultdict(set)

    def register(self) -> None:
        """
        StrategyRegistry наразі не має обов'язкових EventBus subscriptions.

        Метод залишений для lifecycle-сумісності з BaseStrategyComponent.
        """
        self._registered = True

    async def start(self) -> None:
        await super().start()
        await self.emit_event(
            "strategy.registry.started",
            self.summary(),
            priority=EventPriority.LOW,
        )

    async def stop(self) -> None:
        await self.emit_event(
            "strategy.registry.stopped",
            self.summary(),
            priority=EventPriority.LOW,
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
        Register a concrete strategy instance.

        Якщо replace=False і strategy вже існує — кидає StrategyRegistrationError.
        Якщо replace=True — старий instance буде замінений.
        """
        if strategy is None:
            raise StrategyRegistrationError("strategy cannot be None")

        if not isinstance(strategy, BaseStrategy):
            raise StrategyRegistrationError(
                f"strategy must be an instance of BaseStrategy, got {type(strategy)!r}"
            )

        name = strategy.strategy_name
        if not name.strip():
            raise StrategyRegistrationError("strategy name cannot be empty")

        if name in self._strategies and not replace:
            raise StrategyRegistrationError(f"strategy '{name}' is already registered")

        strategy.validate_config()

        if name in self._strategies and replace:
            self._remove_indexes(name, self._strategies[name])

        self._strategies[name] = strategy
        self._add_indexes(name, strategy)

        self.log_info(
            "Strategy registered",
            strategy_name=name,
            category=str(strategy.category),
            priority=strategy.priority,
            required_features=sorted(strategy.required_features()),
        )

        if emit_event:
            self._emit_registry_event_nowait(
                "strategy.registry.strategy_registered",
                {
                    "strategy_name": name,
                    "category": str(strategy.category),
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
        if not strategy_name.strip():
            raise StrategyRegistrationError("strategy_name cannot be empty")

        strategy = self._strategies.pop(strategy_name, None)
        if strategy is None:
            raise StrategyRegistrationError(
                f"strategy '{strategy_name}' is not registered"
            )

        self._remove_indexes(strategy_name, strategy)

        self.log_info(
            "Strategy unregistered",
            strategy_name=strategy_name,
            category=str(strategy.category),
        )

        if emit_event:
            self._emit_registry_event_nowait(
                "strategy.registry.strategy_unregistered",
                {
                    "strategy_name": strategy_name,
                    "category": str(strategy.category),
                    "total": len(self._strategies),
                },
            )

        return strategy

    def clear(self, *, emit_event: bool = True) -> None:
        total = len(self._strategies)

        self._strategies.clear()
        self._by_category.clear()
        self._feature_index.clear()

        self.log_info("Strategy registry cleared", total_removed=total)

        if emit_event:
            self._emit_registry_event_nowait(
                "strategy.registry.cleared",
                {"total_removed": total},
            )

    def get(self, strategy_name: str) -> BaseStrategy | None:
        return self._strategies.get(strategy_name)

    def require(self, strategy_name: str) -> BaseStrategy:
        strategy = self.get(strategy_name)
        if strategy is None:
            raise UnsupportedStrategyError(
                f"strategy '{strategy_name}' is not registered"
            )
        return strategy

    def has(self, strategy_name: str) -> bool:
        return strategy_name in self._strategies

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
        return sorted(self._strategies.keys())

    def list_enabled(self) -> list[BaseStrategy]:
        return [
            strategy
            for strategy in self.list_all()
            if strategy.is_enabled()
        ]

    def list_disabled(self) -> list[BaseStrategy]:
        return [
            strategy
            for strategy in self.list_all()
            if not strategy.is_enabled()
        ]

    def list_by_category(
        self,
        category: StrategyCategory,
        *,
        only_enabled: bool = False,
    ) -> list[BaseStrategy]:
        names = self._by_category.get(category, set())

        strategies = [
            self._strategies[name]
            for name in names
            if name in self._strategies
        ]

        if only_enabled:
            strategies = [
                strategy
                for strategy in strategies
                if strategy.is_enabled()
            ]

        return sorted(
            strategies,
            key=lambda strategy: (strategy.priority, strategy.strategy_name),
        )

    def find_by_required_feature(
        self,
        feature_name: str,
        *,
        only_enabled: bool = True,
    ) -> list[BaseStrategy]:
        if not feature_name.strip():
            return []

        names = self._feature_index.get(feature_name, set())

        strategies = [
            self._strategies[name]
            for name in names
            if name in self._strategies
        ]

        if only_enabled:
            strategies = [
                strategy
                for strategy in strategies
                if strategy.is_enabled()
            ]

        return sorted(
            strategies,
            key=lambda strategy: (strategy.priority, strategy.strategy_name),
        )

    def find_by_required_features(
        self,
        feature_names: Iterable[str],
        *,
        only_enabled: bool = True,
        match_all: bool = False,
    ) -> list[BaseStrategy]:
        """
        Find strategies by required feature names.

        match_all=False:
            повертає стратегії, які потребують хоча б одну з features.

        match_all=True:
            повертає стратегії, у яких required_features повністю покривають
            передані feature_names.
        """
        feature_set = {name for name in feature_names if name.strip()}
        if not feature_set:
            return []

        result: list[BaseStrategy] = []

        for strategy in self._strategies.values():
            if only_enabled and not strategy.is_enabled():
                continue

            required = strategy.required_features()

            if match_all:
                if feature_set.issubset(required):
                    result.append(strategy)
            else:
                if required.intersection(feature_set):
                    result.append(strategy)

        return sorted(
            result,
            key=lambda strategy: (strategy.priority, strategy.strategy_name),
        )

    def find_applicable(
        self,
        context: StrategyContext,
        *,
        only_enabled: bool = True,
    ) -> list[BaseStrategy]:
        """
        Return strategies that can be evaluated on current StrategyContext.

        Не кидає exception, якщо конкретна strategy має помилку applicability check.
        Таку strategy пропускаємо і логимо warning.
        """
        context.validate()

        applicable: list[BaseStrategy] = []

        for strategy in self.list_all():
            try:
                if only_enabled and not strategy.is_enabled():
                    continue

                if strategy.should_evaluate(context):
                    applicable.append(strategy)

            except Exception as exc:
                self.log_warning(
                    "Strategy applicability check failed",
                    strategy_name=strategy.strategy_name,
                    symbol=context.symbol,
                    timeframe=str(context.timeframe),
                    regime=str(context.current_regime),
                    error=str(exc),
                )

        return sorted(
            applicable,
            key=lambda strategy: (strategy.priority, strategy.strategy_name),
        )

    def find_by_categories(
        self,
        categories: Iterable[StrategyCategory],
        *,
        only_enabled: bool = True,
    ) -> list[BaseStrategy]:
        seen: set[str] = set()
        result: list[BaseStrategy] = []

        for category in categories:
            for strategy in self.list_by_category(
                category,
                only_enabled=only_enabled,
            ):
                if strategy.strategy_name in seen:
                    continue

                result.append(strategy)
                seen.add(strategy.strategy_name)

        return sorted(
            result,
            key=lambda strategy: (strategy.priority, strategy.strategy_name),
        )

    def find_by_context_filters(
        self,
        *,
        symbol: str | None = None,
        timeframe: Timeframe | None = None,
        regime: MarketRegime | None = None,
        category: StrategyCategory | None = None,
        only_enabled: bool = True,
    ) -> list[BaseStrategy]:
        """
        Lightweight filtering without building full StrategyContext.
        """
        strategies = (
            self.list_by_category(category, only_enabled=only_enabled)
            if category is not None
            else self.list_enabled() if only_enabled else self.list_all()
        )

        result: list[BaseStrategy] = []

        for strategy in strategies:
            runtime = self.config.get_strategy_runtime(strategy.strategy_name)

            if symbol is not None and not runtime.allows_symbol(symbol):
                continue

            if timeframe is not None and not runtime.allows_timeframe(timeframe):
                continue

            if regime is not None and not runtime.allows_regime(regime):
                continue

            result.append(strategy)

        return sorted(
            result,
            key=lambda strategy: (strategy.priority, strategy.strategy_name),
        )

    def find_for_event(
        self,
        event_name: str,
        *,
        only_enabled: bool = True,
    ) -> list[BaseStrategy]:
        """
        Find strategies mapped to an incoming analytics event.

        Uses config.routing.event_to_categories.
        """
        if not event_name.strip():
            return []

        categories = self.config.routing.categories_for_event(event_name)

        if not categories:
            if self.config.routing.reevaluate_on_any_update:
                return self.list_enabled() if only_enabled else self.list_all()
            return []

        return self.find_by_categories(
            categories,
            only_enabled=only_enabled,
        )

    def required_feature_map(self) -> dict[str, list[str]]:
        return {
            feature_name: sorted(strategy_names)
            for feature_name, strategy_names in sorted(self._feature_index.items())
        }

    def category_map(self) -> dict[str, list[str]]:
        return {
            str(category): sorted(strategy_names)
            for category, strategy_names in sorted(
                self._by_category.items(),
                key=lambda item: str(item[0]),
            )
        }

    def strategy_metadata_map(self) -> dict[str, dict[str, object]]:
        return {
            strategy.strategy_name: {
                "category": str(strategy.category),
                "priority": strategy.priority,
                "enabled": strategy.is_enabled(),
                "required_features": sorted(strategy.required_features()),
                "supported_timeframes": [
                    str(timeframe)
                    for timeframe in sorted(
                        strategy.supported_timeframes(),
                        key=str,
                    )
                ],
                "supported_regimes": [
                    str(regime)
                    for regime in sorted(
                        strategy.supported_regimes(),
                        key=str,
                    )
                ],
                "allowed_symbols": sorted(strategy.allowed_symbols()),
                "min_confidence": strategy.min_confidence(),
                "min_score": strategy.min_score(),
                "cooldown_seconds": strategy.cooldown_seconds(),
                "max_signal_age_seconds": strategy.max_signal_age_seconds(),
            }
            for strategy in self.list_all()
        }

    def summary(self) -> dict[str, object]:
        return {
            "total": len(self._strategies),
            "enabled": len(self.list_enabled()),
            "disabled": len(self.list_disabled()),
            "categories": self.category_map(),
            "required_features": self.required_feature_map(),
            "strategies": self.strategy_metadata_map(),
        }

    def _add_indexes(self, name: str, strategy: BaseStrategy) -> None:
        self._by_category[strategy.category].add(name)

        for feature_name in strategy.required_features():
            if feature_name.strip():
                self._feature_index[feature_name].add(name)

    def _remove_indexes(self, name: str, strategy: BaseStrategy) -> None:
        self._by_category[strategy.category].discard(name)

        if not self._by_category[strategy.category]:
            self._by_category.pop(strategy.category, None)

        for feature_name in strategy.required_features():
            names = self._feature_index.get(feature_name)
            if names is None:
                continue

            names.discard(name)

            if not names:
                self._feature_index.pop(feature_name, None)

    def _emit_registry_event_nowait(
        self,
        topic: str,
        payload: dict[str, object],
    ) -> None:
        """
        Best-effort event publishing for sync registry operations.

        Registry methods are sync, тому тут використовуємо publish_nowait_best_effort(),
        якщо він доступний у core.EventBus.
        """
        if self.event_bus is None:
            return

        publish_nowait = getattr(
            self.event_bus,
            "publish_nowait_best_effort",
            None,
        )

        if callable(publish_nowait):
            publish_nowait(
                topic,
                payload,
                priority=EventPriority.LOW,
                source=self.component_name,
            )