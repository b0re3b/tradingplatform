# trading_system/strategy/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Awaitable, Callable

from core.event_bus import EventBus, EventPriority
from core.logger import get_logger
from core.scheduler import Scheduler

from .config import StrategyConfig, StrategyDefinitionConfig
from .enums import (
    MarketRegime,
    SetupType,
    SignalOrigin,
    SignalPriority,
    SignalSide,
    SignalStatus,
    StrategyCategory,
    Timeframe,
    TriggerType,
)
from .exceptions import StrategyConfigError, StrategyEvaluationError
from .models import (
    StrategyContext,
    StrategyEvaluation,
    StrategySignal,
    confidence_to_grade,
    confidence_to_strength,
    utcnow,
)


EventHandler = Callable[..., Awaitable[None]] | Callable[..., None]


class BaseStrategyComponent(ABC):
    """
    Базовий компонент strategy layer.

    Використовується для:
    - StrategyEngine
    - StrategyContextBuilder
    - StrategyEventHandler
    - StrategyLifecycleManager
    - SignalProcessor
    - SignalNormalizer
    - SignalRouter
    - ConfluenceEngine
    - PortfolioCoordinator
    - SignalScorer
    - SignalFilterChain
    - SignalBuilder
    - concrete strategies

    Правила:
    - config передається через constructor dependency injection;
    - EventBus передається через constructor dependency injection;
    - Scheduler передається через constructor dependency injection;
    - logger береться через core.logger.get_logger();
    - міжмодульна комунікація йде через EventBus.
    """

    component_namespace: str = "strategy"

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.logger = get_logger(__name__)

        self._started: bool = False
        self._registered: bool = False

        self.validate_config()

    @property
    def component_name(self) -> str:
        return self.__class__.__name__

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def is_registered(self) -> bool:
        return self._registered

    def validate_config(self) -> None:
        if self.config is None:
            raise StrategyConfigError(f"{self.component_name}: config is required")

        self.config.validate()

    def register(self) -> None:
        """
        Register EventBus subscriptions.

        Компоненти, які слухають події, перевизначають цей метод.
        """
        self._registered = True

    async def start(self) -> None:
        """
        Async lifecycle hook.
        """
        if self._started:
            return

        if not self._registered:
            self.register()

        self._started = True
        self.logger.info(
            "%s started",
            self.component_name,
            extra={"component": self.component_name},
        )

    async def stop(self) -> None:
        """
        Async cleanup hook.
        """
        if not self._started:
            return

        self._started = False
        self.logger.info(
            "%s stopped",
            self.component_name,
            extra={"component": self.component_name},
        )

    def ensure_event_bus(self) -> EventBus:
        if self.event_bus is None:
            raise RuntimeError(f"{self.component_name}: event_bus is not configured")
        return self.event_bus

    def ensure_scheduler(self) -> Scheduler:
        if self.scheduler is None:
            raise RuntimeError(f"{self.component_name}: scheduler is not configured")
        return self.scheduler

    async def emit_event(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: EventPriority = EventPriority.NORMAL,
        source: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Publish event through EventBus.

        Використовувати тільки для реальних domain/system events,
        а не для локальних helper-обчислень.
        """
        if self.event_bus is None:
            self.logger.debug(
                "Event skipped because event_bus is not configured",
                extra={
                    "component": self.component_name,
                    "topic": topic,
                },
            )
            return

        await self.event_bus.emit(
            topic,
            payload,
            priority=priority,
            source=source or self.component_name,
            **kwargs,
        )

    def subscribe_event(
        self,
        topic: str,
        handler: EventHandler,
        **kwargs: Any,
    ) -> None:
        """
        Subscribe component to EventBus topic.
        """
        bus = self.ensure_event_bus()
        bus.subscribe(topic, handler, **kwargs)

    def log_debug(self, message: str, **extra: Any) -> None:
        self.logger.debug(message, extra={"component": self.component_name, **extra})

    def log_info(self, message: str, **extra: Any) -> None:
        self.logger.info(message, extra={"component": self.component_name, **extra})

    def log_warning(self, message: str, **extra: Any) -> None:
        self.logger.warning(message, extra={"component": self.component_name, **extra})

    def log_error(self, message: str, **extra: Any) -> None:
        self.logger.error(message, extra={"component": self.component_name, **extra})

    def log_exception(self, message: str, **extra: Any) -> None:
        self.logger.exception(message, extra={"component": self.component_name, **extra})


class StatefulStrategyComponent(BaseStrategyComponent, ABC):
    """
    Base class for components that keep internal runtime state.
    """

    @abstractmethod
    def reset_state(self) -> None:
        """
        Reset internal component state.
        """


class ContextAwareStrategyComponent(BaseStrategyComponent, ABC):
    """
    Base class for components that consume StrategyContext.
    """

    def validate_context(self, context: StrategyContext) -> None:
        if context is None:
            raise StrategyEvaluationError(f"{self.component_name}: context is required")

        context.validate()


class NamedEntityMixin:
    """
    Unified name accessor for registry, metrics and logs.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__


class PrioritizedMixin:
    """
    Unified priority accessor for sorting components/strategies.
    """

    @property
    def priority(self) -> int:
        return 100


class BaseStrategy(
    ContextAwareStrategyComponent,
    NamedEntityMixin,
    PrioritizedMixin,
    ABC,
):
    """
    Base class for all concrete trading strategies.

    Contract:
    - strategy читає тільки StrategyContext;
    - strategy не викликає analytics/risk/execution напряму;
    - strategy повертає StrategyEvaluation;
    - публікація signal.generated має бути в StrategyEngine / SignalProcessor.
    """

    category: StrategyCategory = StrategyCategory.HYBRID
    default_setup_type: SetupType = SetupType.UNKNOWN
    default_timeframe: Timeframe = Timeframe.M1

    @property
    def strategy_name(self) -> str:
        return self.__class__.__name__

    @property
    def name(self) -> str:
        return self.strategy_name

    @property
    def priority(self) -> int:
        return self.config.get_strategy_priority(self.strategy_name, default=100)

    def get_definition_config(self) -> StrategyDefinitionConfig | None:
        return self.config.get_strategy(self.strategy_name)

    def is_enabled(self) -> bool:
        return (
            self.config.is_strategy_enabled(self.strategy_name, default=True)
            and self.config.is_strategy_allowed_by_preset(self.strategy_name)
        )

    def required_features(self) -> set[str]:
        return self.config.get_strategy_required_features(self.strategy_name)

    def supported_regimes(self) -> set[MarketRegime]:
        runtime = self.config.get_strategy_runtime(self.strategy_name)
        return set(runtime.allowed_regimes)

    def supported_timeframes(self) -> set[Timeframe]:
        runtime = self.config.get_strategy_runtime(self.strategy_name)
        return set(runtime.timeframes)

    def min_confidence(self) -> float:
        runtime = self.config.get_strategy_runtime(self.strategy_name)
        return runtime.min_confidence

    def min_score(self) -> float:
        runtime = self.config.get_strategy_runtime(self.strategy_name)
        return runtime.min_score

    def allowed_symbols(self) -> set[str]:
        runtime = self.config.get_strategy_runtime(self.strategy_name)
        return set(runtime.symbols)

    def cooldown_seconds(self) -> int:
        runtime = self.config.get_strategy_runtime(self.strategy_name)
        return runtime.cooldown_seconds

    def max_signal_age_seconds(self) -> int:
        runtime = self.config.get_strategy_runtime(self.strategy_name)
        return runtime.max_signal_age_seconds

    def validate_config(self) -> None:
        super().validate_config()

        definition = self.get_definition_config()
        if definition is not None:
            definition.validate()

    def validate_context_requirements(self, context: StrategyContext) -> None:
        self.validate_context(context)

        runtime = self.config.get_strategy_runtime(self.strategy_name)

        if not runtime.allows_symbol(context.symbol):
            raise StrategyEvaluationError(
                f"{self.strategy_name}: symbol {context.symbol} is not allowed"
            )

        if not runtime.allows_timeframe(context.timeframe):
            raise StrategyEvaluationError(
                f"{self.strategy_name}: timeframe {context.timeframe} is not supported"
            )

        if not runtime.allows_regime(context.current_regime):
            raise StrategyEvaluationError(
                f"{self.strategy_name}: regime {context.current_regime} is not supported"
            )

        required = self.required_features()
        missing = [
            feature
            for feature in required
            if not context.has_feature(feature)
        ]
        if missing:
            raise StrategyEvaluationError(
                f"{self.strategy_name}: missing required features: {missing}"
            )

    def should_evaluate(self, context: StrategyContext) -> bool:
        """
        Lightweight applicability check.

        Не кидає exception для нормальних випадків, коли стратегія просто
        не підходить під поточний context.
        """
        if not self.is_enabled():
            return False

        runtime = self.config.get_strategy_runtime(self.strategy_name)

        if not runtime.allows_symbol(context.symbol):
            return False

        if not runtime.allows_timeframe(context.timeframe):
            return False

        if not runtime.allows_regime(context.current_regime):
            return False

        required = self.required_features()
        if required and not required.issubset(set(context.feature_map.keys())):
            if not self.config.routing.allow_partial_context:
                return False

        return True

    async def evaluate(self, context: StrategyContext) -> StrategyEvaluation:
        """
        Єдиний зовнішній entrypoint для оцінки стратегії.

        Concrete strategies мають реалізовувати generate_signal(),
        а не дублювати evaluate().
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
        except StrategyEvaluationError:
            raise
        except Exception as exc:
            self.log_exception(
                "Strategy evaluation failed",
                strategy=self.strategy_name,
                symbol=context.symbol,
            )
            raise StrategyEvaluationError(
                f"{self.strategy_name}: evaluation failed: {exc}"
            ) from exc

        if signal is None:
            return self.build_no_signal_evaluation(context)

        signal.validate()

        passed = (
            signal.is_directional
            and signal.confidence >= self.min_confidence()
            and signal.score >= self.min_score()
            and signal.passed_filters
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
    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        """
        Concrete strategy must implement signal generation logic.
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
        confidence = self._clamp(confidence, 0.0, 1.0)

        return StrategySignal(
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

    def add_reason_if(
        self,
        reasons: list[str],
        condition: bool,
        reason: str,
    ) -> None:
        if condition and reason not in reasons:
            reasons.append(reason)

    def get_feature(
        self,
        context: StrategyContext,
        name: str,
        default: Any = None,
    ) -> Any:
        return context.get_feature(name, default)

    def get_normalized_feature(
        self,
        context: StrategyContext,
        name: str,
        default: float | None = None,
    ) -> float | None:
        return context.get_normalized_feature(name, default)

    def has_feature(self, context: StrategyContext, name: str) -> bool:
        return context.has_feature(name)

    def get_mid_price(self, context: StrategyContext) -> float | None:
        return context.mid_price

    def now(self) -> datetime:
        return utcnow()

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(value, max_value))