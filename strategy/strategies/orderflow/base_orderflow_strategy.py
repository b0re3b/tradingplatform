from __future__ import annotations

from typing import Any

from analytics.orderflow import OrderFlowAnalyzer
from core.event_bus import EventBus, EventPriority
from core.logger import get_logger

from ...base import ContextAwareComponent, NamedEntityMixin, PrioritizedMixin
from ...config import StrategyConfig, StrategyDefinitionConfig
from ...enums import (
    ConfidenceGrade,
    MarketRegime,
    SignalPriority,
    SignalStrength,
    StrategyCategory,
)
from ...models import SignalContext, StrategyEvaluation


class OrderflowStrategyBase(
    ContextAwareComponent,
    NamedEntityMixin,
    PrioritizedMixin,
):
    """
    Спільна база для strategy/strategies/orderflow.

    Відповідає за інфраструктурну частину orderflow-стратегій:
    - core.logger через get_logger()
    - core.event_bus через async evaluate_async()/emit
    - StrategyConfig/runtime filters
    - config weighting helpers
    - mapping confidence -> priority/strength/grade
    - generic feature/value/normalization helpers

    Конкретні стратегії залишають у себе тільки торгову логіку:
    detection, scoring, confirmations, signal plan construction.
    """

    STRATEGY_NAME: str = "orderflow_strategy"
    CATEGORY: StrategyCategory = StrategyCategory.ORDERFLOW
    DEFAULT_REQUIRED_FEATURES: set[str] = set()
    REQUIRED_FEATURES: set[str] = set()

    def __init__(
        self,
        config: StrategyConfig,
        *,
        orderflow_analyzer: OrderFlowAnalyzer | None = None,
        event_bus: EventBus | None = None,
        logger: Any | None = None,
    ) -> None:
        resolved_logger = logger or get_logger(
            __name__,
            event_type="strategy",
            strategies=self.STRATEGY_NAME,
        )
        super().__init__(config=config, event_bus=event_bus, logger=resolved_logger)
        self.orderflow_analyzer = orderflow_analyzer

    @property
    def component_name(self) -> str:
        return self.STRATEGY_NAME

    @property
    def priority(self) -> int:
        strategy_cfg = self.strategy_definition
        if strategy_cfg is not None:
            return strategy_cfg.priority
        return 100

    @property
    def strategy_definition(self) -> StrategyDefinitionConfig | None:
        return self.config.get_strategy(self.STRATEGY_NAME)

    def is_enabled(self) -> bool:
        strategy_cfg = self.strategy_definition
        if strategy_cfg is None:
            return True
        return strategy_cfg.runtime.enabled

    def required_features(self) -> set[str]:
        strategy_cfg = self.strategy_definition
        if strategy_cfg is not None and strategy_cfg.required_features:
            return set(strategy_cfg.required_features)

        if self.DEFAULT_REQUIRED_FEATURES:
            return set(self.DEFAULT_REQUIRED_FEATURES)

        return set(self.REQUIRED_FEATURES)

    def _runtime_allows_context(self, context: SignalContext) -> bool:
        strategy_cfg = self.strategy_definition
        runtime_cfg = strategy_cfg.runtime if strategy_cfg is not None else self.config.runtime

        if runtime_cfg.symbols and context.symbol not in runtime_cfg.symbols:
            return False

        if runtime_cfg.timeframes and context.timeframe not in runtime_cfg.timeframes:
            return False

        regime = context.regime.regime if context.regime is not None else MarketRegime.UNKNOWN

        if runtime_cfg.allowed_regimes:
            if (
                regime not in runtime_cfg.allowed_regimes
                and MarketRegime.UNKNOWN not in runtime_cfg.allowed_regimes
            ):
                return False

        return True

    async def evaluate_async(self, context: SignalContext) -> StrategyEvaluation:
        """
        Async wrapper для event-driven pipeline.

        evaluate() лишається синхронним, щоб StrategyEngine міг швидко
        викликати його in-process. Якщо потрібні події, engine може викликати
        evaluate_async(), і результат буде опублікований в EventBus.
        """
        evaluation = self.evaluate(context)  # type: ignore[attr-defined]
        await self._emit_evaluation_event(context, evaluation)
        return evaluation

    async def _emit_evaluation_event(
        self,
        context: SignalContext,
        evaluation: StrategyEvaluation,
    ) -> None:
        if self.event_bus is None:
            return

        try:
            signal = getattr(evaluation, "signal", None)

            await self.event_bus.emit(
                "strategy.orderflow.evaluated",
                {
                    "strategy_name": self.STRATEGY_NAME,
                    "category": getattr(self.CATEGORY, "value", str(self.CATEGORY)),
                    "symbol": context.symbol,
                    "timeframe": str(context.timeframe),
                    "timestamp": (
                        context.timestamp.isoformat()
                        if hasattr(context.timestamp, "isoformat")
                        else context.timestamp
                    ),
                    "passed": evaluation.passed,
                    "score": evaluation.score,
                    "confidence": evaluation.confidence,
                    "reasons": list(evaluation.reasons),
                    "signal_id": getattr(signal, "signal_id", None) if signal is not None else None,
                    "side": str(getattr(signal, "side", None)) if signal is not None else None,
                },
                priority=EventPriority.NORMAL,
                source=self.STRATEGY_NAME,
            )

        except Exception:
            self.log_warning(
                "Failed to emit orderflow strategy evaluation event",
                symbol=context.symbol,
                strategy=self.STRATEGY_NAME,
            )

    def _get_min_confidence(self) -> float:
        strategy_cfg = self.strategy_definition
        if strategy_cfg is not None:
            return strategy_cfg.runtime.min_confidence
        return self.config.runtime.min_confidence

    def _get_min_score(self) -> float:
        strategy_cfg = self.strategy_definition
        if strategy_cfg is not None:
            return strategy_cfg.runtime.min_score
        return self.config.runtime.min_score

    def _category_weight(self) -> float:
        try:
            return float(self.config.weighting.category_weights.get(self.CATEGORY, 1.0))
        except Exception:
            return 1.0

    def _strategy_weight(self) -> float:
        strategy_cfg = self.strategy_definition
        if strategy_cfg is None:
            return 1.0

        try:
            return float(strategy_cfg.weight)
        except Exception:
            return 1.0

    def _regime_adjustment(self, context: SignalContext) -> float:
        regime = context.regime.regime if context.regime is not None else MarketRegime.UNKNOWN

        try:
            return float(self.config.weighting.regime_adjustments.get(regime, 1.0))
        except Exception:
            return 1.0

    def _resolve_priority(self, confidence: float) -> SignalPriority:
        cfg = self.config.confidence

        if confidence >= cfg.high_threshold:
            return SignalPriority.HIGH
        if confidence >= cfg.low_threshold:
            return SignalPriority.MEDIUM
        return SignalPriority.LOW

    def _map_strength(self, confidence: float) -> SignalStrength:
        cfg = self.config.confidence

        if confidence >= cfg.high_threshold:
            return SignalStrength.STRONG
        if confidence >= cfg.medium_threshold:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK

    def _map_confidence_grade(self, confidence: float) -> ConfidenceGrade:
        cfg = self.config.confidence

        if confidence >= cfg.high_threshold:
            return ConfidenceGrade.VERY_HIGH
        if confidence >= cfg.medium_threshold:
            return ConfidenceGrade.HIGH
        if confidence >= cfg.low_threshold:
            return ConfidenceGrade.MEDIUM
        if confidence >= cfg.very_low_threshold:
            return ConfidenceGrade.LOW
        return ConfidenceGrade.VERY_LOW

    def _resolve_reference_price(self, context: SignalContext, data: Any) -> float | None:
        if context.price is not None:
            if context.price.mid_price is not None:
                return context.price.mid_price
            if context.price.last_price is not None:
                return context.price.last_price

        return self._coalesce_float(getattr(data, "last_price", None))

    def _feature_value(self, context: SignalContext, name: str) -> Any:
        snapshot = context.get_feature_snapshot(name)
        if snapshot is None:
            return None
        return snapshot.value

    @staticmethod
    def _read(obj: Any, name: str, default: Any = None) -> Any:
        if obj is None:
            return default

        if isinstance(obj, dict):
            return obj.get(name, default)

        return getattr(obj, name, default)

    @staticmethod
    def _coalesce_float(*values: Any) -> float | None:
        for value in values:
            if value is None:
                continue

            try:
                return float(value)
            except (TypeError, ValueError):
                continue

        return None

    @staticmethod
    def _coalesce_int(*values: Any) -> int | None:
        for value in values:
            if value is None:
                continue

            try:
                return int(value)
            except (TypeError, ValueError):
                continue

        return None

    @staticmethod
    def _normalize_percent(value: float, *, scale: float = 2.0) -> float:
        if scale <= 0:
            return 0.0

        return max(0.0, min(abs(value) / scale, 1.0))

    @staticmethod
    def _normalize_ratio(value: float, *, scale: float = 1.0) -> float:
        if scale <= 0:
            return 0.0

        return max(0.0, min(abs(value) / scale, 1.0))

    @staticmethod
    def _normalize_magnitude(value: float, *, scale: float = 10.0) -> float:
        if value <= 0 or scale <= 0:
            return 0.0

        return max(0.0, min(value / scale, 1.0))

    @staticmethod
    def _safe_get_latest_stats(
        facade: OrderFlowAnalyzer,
        module_name: str,
        symbol: str,
    ) -> Any:
        module = getattr(facade, module_name, None)

        if module is None and hasattr(facade, "get_module"):
            module = facade.get_module(module_name)

        if module is None:
            return None

        getter = getattr(module, "get_latest_stats", None)
        if not callable(getter):
            return None

        return getter(symbol)