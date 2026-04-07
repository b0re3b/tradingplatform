from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .base import StrategyComponent
from .config import RoutingConfig
from .context import StrategyContext
from .enums import FeatureSource, MarketRegime, StrategyCategory
from .exceptions import SignalRoutingError
from .models import FeatureSnapshot
from .registry import StrategyRegistry
from .strategies.base_strategy import BaseStrategy


def utcnow() -> datetime:
    return datetime.utcnow()


@dataclass(slots=True)
class RouteDecision:
    """
    Результат маршрутизації одного analytics event / normalized payload.
    """

    event_name: str
    symbol: str
    source: FeatureSource | None = None
    timestamp: datetime = field(default_factory=utcnow)

    selected: list[BaseStrategy] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    categories_used: list[StrategyCategory] = field(default_factory=list)
    matched_features: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def selected_names(self) -> list[str]:
        return [strategy.strategy_name for strategy in self.selected]

    @property
    def total_selected(self) -> int:
        return len(self.selected)

    @property
    def is_empty(self) -> bool:
        return not self.selected


class SignalRouter(StrategyComponent):
    """
    Strategy router.

    Відповідає за:
    - визначення релевантних strategy categories для event
    - вибір candidate strategies
    - відбір applicable strategies по context
    - fallback routing через source/required features
    - optional hybrid routing
    - partial-context policy
    - freshness / staleness filtering
    """

    def __init__(
        self,
        config,
        registry: StrategyRegistry,
        event_bus=None,
        logger=None,
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, logger=logger)
        self.registry = registry
        self.validate_config()

    @property
    def routing_config(self) -> RoutingConfig:
        return self.config.routing

    def route(
        self,
        *,
        event_name: str,
        context: StrategyContext,
        source: FeatureSource | None = None,
        changed_features: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RouteDecision:
        """
        Головна точка маршрутизації.

        Повертає RouteDecision з:
        - selected strategies
        - skipped strategies + reasons
        - matched categories/features
        """
        if not event_name.strip():
            raise SignalRoutingError("event_name cannot be empty")

        context.validate()

        resolved_source = source or self._resolve_source_from_event_name(event_name)
        changed_features = changed_features or []

        decision = RouteDecision(
            event_name=event_name,
            symbol=context.symbol,
            source=resolved_source,
            timestamp=context.timestamp,
            metadata=metadata or {},
        )

        candidates = self._collect_candidates(
            event_name=event_name,
            context=context,
            source=resolved_source,
            changed_features=changed_features,
            decision=decision,
        )

        applicable = self._filter_applicable(
            candidates=candidates,
            context=context,
            changed_features=changed_features,
            decision=decision,
        )

        decision.selected = applicable

        self.log_debug(
            "Routing completed",
            event_name=event_name,
            symbol=context.symbol,
            source=str(resolved_source) if resolved_source else None,
            selected=decision.selected_names,
            skipped=decision.skipped,
        )
        return decision

    def route_batch(
        self,
        *,
        events: list[dict[str, Any]],
        context: StrategyContext,
    ) -> list[RouteDecision]:
        """
        Маршрутизація списку подій для одного symbol context.
        Формат event:
        {
            "event_name": "...",
            "source": FeatureSource | None,
            "changed_features": [...],
            "metadata": {...},
        }
        """
        decisions: list[RouteDecision] = []

        for item in events:
            event_name = item.get("event_name")
            if not isinstance(event_name, str) or not event_name.strip():
                raise SignalRoutingError("Each event must contain non-empty 'event_name'")

            decision = self.route(
                event_name=event_name,
                context=context,
                source=item.get("source"),
                changed_features=item.get("changed_features", []),
                metadata=item.get("metadata", {}),
            )
            decisions.append(decision)

        return decisions

    def route_from_features(
        self,
        *,
        context: StrategyContext,
        features: list[FeatureSnapshot],
        event_name: str = "strategy.feature_update",
        metadata: dict[str, Any] | None = None,
    ) -> RouteDecision:
        """
        Helper, якщо ми роутимо не від raw event_name, а від конкретного набору features.
        """
        changed_features = [feature.name for feature in features]
        source = features[0].source if features else None

        return self.route(
            event_name=event_name,
            context=context,
            source=source,
            changed_features=changed_features,
            metadata=metadata,
        )

    def _collect_candidates(
        self,
        *,
        event_name: str,
        context: StrategyContext,
        source: FeatureSource | None,
        changed_features: list[str],
        decision: RouteDecision,
    ) -> list[BaseStrategy]:
        """
        Кандидати збираються з кількох джерел:
        1. config.event_to_categories
        2. source -> category fallback
        3. required feature index
        4. hybrid injection
        """
        candidates: dict[str, BaseStrategy] = {}

        categories = self._resolve_categories_for_event(event_name, source)
        if categories:
            decision.categories_used.extend(categories)
            for strategy in self.registry.find_by_categories(categories):
                candidates[strategy.strategy_name] = strategy

        if changed_features:
            for feature_name in changed_features:
                matched = self.registry.find_by_required_feature(feature_name)
                if matched:
                    decision.matched_features.append(feature_name)
                for strategy in matched:
                    candidates[strategy.strategy_name] = strategy

        if self.routing_config.reevaluate_on_any_update and not candidates:
            for strategy in self.registry.list_enabled():
                candidates[strategy.strategy_name] = strategy

        if (
            self.routing_config.route_hybrid_on_domain_signal
            and source is not None
            and source != FeatureSource.SYSTEM
        ):
            for strategy in self.registry.list_by_category(StrategyCategory.HYBRID):
                if strategy.is_enabled():
                    candidates[strategy.strategy_name] = strategy
                    if StrategyCategory.HYBRID not in decision.categories_used:
                        decision.categories_used.append(StrategyCategory.HYBRID)

        result = list(candidates.values())
        result.sort(key=lambda item: item.priority)
        return result

    def _filter_applicable(
        self,
        *,
        candidates: list[BaseStrategy],
        context: StrategyContext,
        changed_features: list[str],
        decision: RouteDecision,
    ) -> list[BaseStrategy]:
        applicable: list[BaseStrategy] = []

        for strategy in candidates:
            reason = self._skip_reason(
                strategy=strategy,
                context=context,
                changed_features=changed_features,
            )
            if reason is not None:
                decision.skipped[strategy.strategy_name] = reason
                continue

            applicable.append(strategy)

        applicable.sort(key=lambda item: item.priority)
        return applicable

    def _skip_reason(
        self,
        *,
        strategy: BaseStrategy,
        context: StrategyContext,
        changed_features: list[str],
    ) -> str | None:
        """
        Якщо треба пропустити strategy — повертає причину.
        Якщо можна оцінювати — повертає None.
        """
        if not strategy.is_enabled():
            return "strategy_disabled"

        allowed_symbols = strategy.allowed_symbols()
        if allowed_symbols and context.symbol not in allowed_symbols:
            return "symbol_not_allowed"

        supported_timeframes = strategy.supported_timeframes()
        if supported_timeframes and context.timeframe not in supported_timeframes:
            return "timeframe_not_supported"

        supported_regimes = strategy.supported_regimes()
        if supported_regimes:
            current_regime = context.current_regime
            if MarketRegime.UNKNOWN not in supported_regimes and current_regime not in supported_regimes:
                return "regime_not_supported"

        required_features = strategy.required_features()
        missing_required = [name for name in required_features if not context.has_feature(name)]
        if missing_required and not self.routing_config.allow_partial_context:
            return f"missing_required_features:{','.join(sorted(missing_required))}"

        stale_required = [
            name
            for name in required_features
            if context.has_feature(name) and self._feature_is_stale(context, name)
        ]
        if stale_required:
            return f"stale_required_features:{','.join(sorted(stale_required))}"

        if changed_features:
            if not self._strategy_relevant_for_feature_change(strategy, changed_features):
                return "no_relevant_feature_change"

        try:
            if not strategy.should_evaluate(context):
                return "strategy_should_evaluate_false"
        except Exception as exc:
            return f"strategy_should_evaluate_error:{exc}"

        return None

    def _feature_is_stale(self, context: StrategyContext, feature_name: str) -> bool:
        snapshot = context.get_feature_snapshot(feature_name)
        if snapshot is None:
            return True

        threshold = self.routing_config.stale_feature_threshold_seconds
        age = snapshot.age_seconds(context.timestamp)
        return age > threshold

    def _strategy_relevant_for_feature_change(
        self,
        strategy: BaseStrategy,
        changed_features: list[str],
    ) -> bool:
        """
        Якщо є changed_features:
        - strategy релевантна, якщо її required_features перетинаються зі змінами
        - або якщо required_features порожній набір
        """
        required = strategy.required_features()
        if not required:
            return True

        return bool(required.intersection(changed_features))

    def _resolve_categories_for_event(
        self,
        event_name: str,
        source: FeatureSource | None,
    ) -> list[StrategyCategory]:
        """
        Порядок:
        1. config.event_to_categories exact match
        2. config.event_to_categories prefix-ish heuristic
        3. source fallback mapping
        """
        categories: list[StrategyCategory] = []

        configured = self.routing_config.event_to_categories.get(event_name)
        if configured:
            return list(dict.fromkeys(configured))

        event_lower = event_name.lower()
        for configured_event, configured_categories in self.routing_config.event_to_categories.items():
            if configured_event and configured_event.lower() in event_lower:
                categories.extend(configured_categories)

        if categories:
            return list(dict.fromkeys(categories))

        if source is not None:
            mapped = self._map_source_to_category(source)
            if mapped is not None:
                categories.append(mapped)

        return list(dict.fromkeys(categories))

    def _resolve_source_from_event_name(self, event_name: str) -> FeatureSource | None:
        event_lower = event_name.lower()

        if "orderflow" in event_lower or "cvd" in event_lower or "imbalance" in event_lower:
            return FeatureSource.ORDERFLOW
        if "liquidity" in event_lower or "equal_high" in event_lower or "stop_cluster" in event_lower:
            return FeatureSource.LIQUIDITY
        if "price_action" in event_lower or "market_structure" in event_lower or "fvg" in event_lower:
            return FeatureSource.PRICE_ACTION
        if "liquidation" in event_lower or "squeeze" in event_lower:
            return FeatureSource.LIQUIDATIONS
        if "whale" in event_lower:
            return FeatureSource.WHALES
        if "spoof" in event_lower or "fake_liquidity" in event_lower:
            return FeatureSource.SPOOFING
        if "spread" in event_lower or "basis" in event_lower or "arb" in event_lower:
            return FeatureSource.SPREADS
        if "funding" in event_lower:
            return FeatureSource.FUNDING
        if "open_interest" in event_lower or "oi_" in event_lower or ".oi" in event_lower:
            return FeatureSource.OPEN_INTEREST

        return None

    def _map_source_to_category(self, source: FeatureSource) -> StrategyCategory | None:
        mapping = {
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

    def explain_route(
        self,
        decision: RouteDecision,
    ) -> dict[str, Any]:
        """
        Зручний explain/debug output для логування, тестів і dashboard.
        """
        return {
            "event_name": decision.event_name,
            "symbol": decision.symbol,
            "source": str(decision.source) if decision.source else None,
            "selected": decision.selected_names,
            "skipped": decision.skipped,
            "categories_used": [str(category) for category in decision.categories_used],
            "matched_features": decision.matched_features,
            "metadata": decision.metadata,
        }