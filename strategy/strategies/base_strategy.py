from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from ..base import (
    ContextAwareComponent,
    EventEmitterMixin,
    NamedEntityMixin,
    PrioritizedMixin,
)
from ..config import StrategyConfig, StrategyDefinitionConfig
from ..context import StrategyContext
from ..enums import (
    MarketRegime,
    SetupType,
    SignalOrigin,
    SignalPriority,
    SignalSide,
    SignalStatus,
    SignalStrength,
    StrategyCategory,
    Timeframe,
    TriggerType,
)
from strategy.exceptions import StrategyConfigError, StrategyEvaluationError
from ..models import (
    StrategyEvaluation,
    StrategySignal,
    confidence_to_grade,
    confidence_to_strength,
)


class BaseStrategy(
    ContextAwareComponent,
    EventEmitterMixin,
    NamedEntityMixin,
    PrioritizedMixin,
    ABC,
):
    """
    Базовий клас для всіх concrete strategy classes.

    Контракт:
    - strategy читає тільки StrategyContext
    - strategy не працює напряму з analytics-модулями
    - strategy повертає StrategyEvaluation
    """

    category: StrategyCategory = StrategyCategory.HYBRID
    default_setup_type: SetupType = SetupType.UNKNOWN
    default_timeframe: Timeframe = Timeframe.M1

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: Any | None = None,
        logger: Any | None = None,
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, logger=logger)
        self.validate_config()

    @property
    def strategy_name(self) -> str:
        return self.__class__.__name__

    @property
    def name(self) -> str:
        return self.strategy_name

    @property
    def priority(self) -> int:
        definition = self.get_definition_config()
        if definition is None:
            return 100
        return definition.priority

    def get_definition_config(self) -> StrategyDefinitionConfig | None:
        return self.config.get_strategy(self.strategy_name)

    def is_enabled(self) -> bool:
        definition = self.get_definition_config()
        if definition is None:
            return True
        return definition.runtime.enabled

    def required_features(self) -> set[str]:
        definition = self.get_definition_config()
        if definition is None:
            return set()
        return set(definition.required_features)

    def supported_regimes(self) -> set[MarketRegime]:
        definition = self.get_definition_config()
        if definition is None:
            return set()
        return set(definition.runtime.allowed_regimes)

    def supported_timeframes(self) -> set[Timeframe]:
        definition = self.get_definition_config()
        if definition is None:
            return {self.default_timeframe}
        return set(definition.runtime.timeframes)

    def min_confidence(self) -> float:
        definition = self.get_definition_config()
        if definition is None:
            return self.config.runtime.min_confidence
        return definition.runtime.min_confidence

    def min_score(self) -> float:
        definition = self.get_definition_config()
        if definition is None:
            return self.config.runtime.min_score
        return definition.runtime.min_score

    def allowed_symbols(self) -> set[str]:
        definition = self.get_definition_config()
        if definition is None:
            return set()
        return set(definition.runtime.symbols)

    def cooldown_seconds(self) -> int:
        definition = self.get_definition_config()
        if definition is None:
            return self.config.runtime.cooldown_seconds
        return definition.runtime.cooldown_seconds

    def max_signal_age_seconds(self) -> int:
        definition = self.get_definition_config()
        if definition is None:
            return self.config.runtime.max_signal_age_seconds
        return definition.runtime.max_signal_age_seconds

    def validate_config(self) -> None:
        super().validate_config()
        definition = self.get_definition_config()
        if definition is not None:
            definition.validate()

    def validate_context_requirements(self, context: StrategyContext) -> None:
        self.validate_context(context)

        allowed_symbols = self.allowed_symbols()
        if allowed_symbols and context.symbol not in allowed_symbols:
            raise StrategyEvaluationError(
                f"{self.strategy_name}: symbol {context.symbol} is not allowed"
            )

        timeframes = self.supported_timeframes()
        if timeframes and context.timeframe not in timeframes:
            raise StrategyEvaluationError(
                f"{self.strategy_name}: timeframe {context.timeframe} is not supported"
            )

        required = self.required_features()
        missing = [feature for feature in required if not context.has_feature(feature)]
        if missing:
            raise StrategyEvaluationError(
                f"{self.strategy_name}: missing required features: {missing}"
            )

    def should_evaluate(self, context: StrategyContext) -> bool:
        if not self.is_enabled():
            return False

        allowed_symbols = self.allowed_symbols()
        if allowed_symbols and context.symbol not in allowed_symbols:
            return False

        supported_timeframes = self.supported_timeframes()
        if supported_timeframes and context.timeframe not in supported_timeframes:
            return False

        regimes = self.supported_regimes()
        if regimes:
            current_regime = context.current_regime
            if MarketRegime.UNKNOWN not in regimes and current_regime not in regimes:
                return False

        required = self.required_features()
        if required and not required.issubset(set(context.feature_map.keys())):
            if not self.config.routing.allow_partial_context:
                return False

        return True

    async def evaluate(self, context: StrategyContext) -> StrategyEvaluation:
        """
        Єдиний зовнішній метод оцінки стратегії.
        """
        self.validate_context(context)

        if not self.should_evaluate(context):
            return StrategyEvaluation(
                strategy_name=self.strategy_name,
                symbol=context.symbol,
                timestamp=context.timestamp,
                passed=False,
                reasons=["strategy_not_applicable"],
            )

        try:
            signal = await self.generate_signal(context)
        except Exception as exc:
            raise StrategyEvaluationError(
                f"{self.strategy_name}: evaluation failed: {exc}"
            ) from exc

        if signal is None:
            return StrategyEvaluation(
                strategy_name=self.strategy_name,
                symbol=context.symbol,
                timestamp=context.timestamp,
                passed=False,
                reasons=["no_signal_generated"],
            )

        signal.validate()

        passed = (
            signal.is_directional
            and signal.confidence >= self.min_confidence()
            and signal.score >= self.min_score()
        )

        return StrategyEvaluation(
            strategy_name=self.strategy_name,
            symbol=context.symbol,
            timestamp=context.timestamp,
            signal=signal,
            passed=passed,
            score=signal.score,
            confidence=signal.confidence,
            reasons=list(signal.reasons),
        )

    @abstractmethod
    async def generate_signal(self, context: StrategyContext) -> StrategySignal | None:
        """
        Concrete strategy must implement this method.
        """

    def build_signal(
        self,
        *,
        context: StrategyContext,
        side: SignalSide,
        confidence: float,
        score: float,
        reasons: list[str] | None = None,
        confirmations: list[str] | None = None,
        source_features: list[str] | None = None,
        combined_from: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        setup_type: SetupType | None = None,
        trigger_type: TriggerType = TriggerType.PRIMARY,
        origin: SignalOrigin = SignalOrigin.SINGLE_STRATEGY,
        priority: SignalPriority = SignalPriority.MEDIUM,
        status: SignalStatus = SignalStatus.NEW,
    ) -> StrategySignal:
        confidence = max(0.0, min(confidence, 1.0))

        signal = StrategySignal(
            symbol=context.symbol,
            side=side,
            strategy_name=self.strategy_name,
            category=self.category,
            timeframe=context.timeframe,
            setup_type=setup_type or self.default_setup_type,
            timestamp=context.timestamp,
            confidence=confidence,
            score=score,
            strength=confidence_to_strength(confidence),
            confidence_grade=confidence_to_grade(confidence),
            status=status,
            trigger_type=trigger_type,
            origin=origin,
            priority=priority,
            reasons=reasons or [],
            confirmations=confirmations or [],
            source_features=source_features or [],
            combined_from=combined_from or [],
            regime=context.current_regime,
            metadata=metadata or {},
        )
        return signal

    def build_no_signal_evaluation(
        self,
        context: StrategyContext,
        reason: str = "no_signal_generated",
    ) -> StrategyEvaluation:
        return StrategyEvaluation(
            strategy_name=self.strategy_name,
            symbol=context.symbol,
            timestamp=context.timestamp,
            passed=False,
            reasons=[reason],
        )

    def add_reason_if(self, reasons: list[str], condition: bool, reason: str) -> None:
        if condition and reason not in reasons:
            reasons.append(reason)

    def get_feature(self, context: StrategyContext, name: str, default: Any = None) -> Any:
        return context.get_feature(name, default)

    def get_normalized_feature(
        self,
        context: StrategyContext,
        name: str,
        default: float | None = None,
    ) -> float | None:
        return context.get_normalized_feature(name, default)

    def get_mid_price(self, context: StrategyContext) -> float | None:
        return context.mid_price

    def now(self) -> datetime:
        return datetime.utcnow()