# trading_system/strategy/base.py

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from datetime import datetime
from typing import Any, Callable

from core.event_bus import Event, EventBus, EventPriority
from core.logger import get_logger
from core.scheduler import Scheduler
from strategy.config import StrategyConfig, StrategyDefinitionConfig, StrategyRuntimeConfig
from strategy.enums import (
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
from strategy.exceptions import StrategyConfigError, StrategyEvaluationError
from strategy.models import (
    StrategyContext,
    StrategyEvaluation,
    StrategySignal,
    confidence_to_grade,
    confidence_to_strength,
    ensure_aware_utc,
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
    - StrategyRegistry
    - SignalProcessor
    - SignalNormalizer
    - SignalRouter
    - ConfluenceEngine
    - PortfolioCoordinator
    - SignalScorer
    - SignalFilterChain
    - SignalBuilder
    - BaseStrategy / concrete strategies

    Правила:
    - config передається через constructor dependency injection;
    - EventBus передається через constructor dependency injection;
    - Scheduler передається через constructor dependency injection;
    - logger береться через core.logger.get_logger();
    - міжмодульна комунікація йде через EventBus;
    - periodic/background jobs мають іти через Scheduler, не через unmanaged loops.
    """

    component_namespace: str = "strategy"

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        service_name: str | None = None,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.scheduler = scheduler

        self._service_name = service_name or self.component_namespace
        self.logger = get_logger(
            __name__,
            service=self._service_name,
            event_type=self.component_name,
        )

        self._started: bool = False
        self._registered: bool = False
        self._subscriptions: list[Any] = []
        self._scheduler_jobs: list[Any] = []

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

    @property
    def subscriptions_count(self) -> int:
        return len(self._subscriptions)

    @property
    def scheduler_jobs_count(self) -> int:
        return len(self._scheduler_jobs)

    def validate_config(self) -> None:
        if self.config is None:
            raise StrategyConfigError(f"{self.component_name}: config is required")

        validate = getattr(self.config, "validate", None)
        if callable(validate):
            validate()

    async def _await_best_effort(awaitable: Awaitable[Any]) -> None:
        await awaitable

    def register(self) -> None:
        """
        Register EventBus subscriptions.

        Компоненти, які реально слухають події, мають перевизначити цей метод.
        Базова реалізація тільки позначає компонент як registered.
        """
        self._registered = True

    def unregister(self) -> None:
        """
        Unregister EventBus subscriptions owned by this component.

        Concrete components should use subscribe_event(), щоб підписки були
        збережені в _subscriptions і могли бути коректно зняті під час stop().
        """
        if self.event_bus is not None:
            for subscription in list(self._subscriptions):
                try:
                    self.event_bus.unsubscribe(subscription)
                except Exception:
                    self.log_exception(
                        "Failed to unsubscribe strategy component",
                        subscription=str(subscription),
                    )

        self._subscriptions.clear()
        self._registered = False

    async def start(self) -> None:
        """
        Async lifecycle hook.
        """
        if self._started:
            self.log_debug("Component already started")
            return

        if not self._registered:
            self.register()

        self._started = True

        self.log_info(
            "Component started",
            registered=self._registered,
            subscriptions=len(self._subscriptions),
            scheduler_jobs=len(self._scheduler_jobs),
        )

    async def stop(self) -> None:
        """
        Async cleanup hook.
        """
        if not self._started:
            self.log_debug("Component already stopped")
            return

        self.unregister()
        self._started = False

        self.log_info("Component stopped")

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

        Використовувати тільки для domain/system events, не для локальних
        helper-обчислень.
        """
        if not topic.strip():
            raise ValueError("topic cannot be empty")

        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")

        if self.event_bus is None:
            self.log_debug(
                "Event skipped because event_bus is not configured",
                topic=topic,
            )
            return

        await self.event_bus.emit(
            topic,
            payload,
            priority=priority,
            source=source or self.component_name,
            **kwargs,
        )

    def emit_event_nowait_best_effort(
            self,
            topic: str,
            payload: dict[str, Any],
            *,
            priority: EventPriority = EventPriority.NORMAL,
            source: str | None = None,
            **kwargs: Any,
    ) -> None:
        """
        Best-effort EventBus emit for sync code paths.

        Supports multiple EventBus contracts:
        - publish_nowait_best_effort(topic, payload, priority=..., source=...)
        - publish_nowait_best_effort(topic, payload)
        - publish_nowait_best_effort(Event(...))
        """
        if not topic.strip():
            raise ValueError("topic cannot be empty")

        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")

        if self.event_bus is None:
            self.log_debug(
                "Best-effort event skipped because event_bus is not configured",
                topic=topic,
            )
            return

        publish_nowait = getattr(self.event_bus, "publish_nowait_best_effort", None)
        if not callable(publish_nowait):
            self.log_warning(
                "EventBus does not support publish_nowait_best_effort",
                topic=topic,
            )
            return

        event_source = source or self.component_name

        enriched_payload = dict(payload)
        metadata = enriched_payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        metadata.setdefault("source", event_source)
        metadata.setdefault(
            "priority",
            priority.value if hasattr(priority, "value") else str(priority),
        )
        metadata.setdefault("best_effort", True)

        if kwargs:
            metadata.setdefault("event_kwargs", dict(kwargs))

        enriched_payload["metadata"] = metadata

        # 1) New/full EventBus contract:
        # publish_nowait_best_effort(topic, payload, priority=..., source=...)
        if self._try_publish_nowait_best_effort(
            publish_nowait,
            topic,
            topic,
            enriched_payload,
            priority=priority,
            source=event_source,
            **kwargs,
        ):
            return

        # 2) Simpler EventBus contract:
        # publish_nowait_best_effort(topic, payload)
        if self._try_publish_nowait_best_effort(
            publish_nowait,
            topic,
            topic,
            enriched_payload,
        ):
            return

        # 3) Current core contract in your project:
        # publish_nowait_best_effort(Event(...))
        event = Event(
            topic=topic,
            payload=enriched_payload,
            priority=priority,
            source=event_source,
        )
        if self._try_publish_nowait_best_effort(
            publish_nowait,
            topic,
            event,
        ):
            return

        # 4) Compatibility for Event dataclass variants that use name/event_name
        # instead of topic/source.
        for event_kwargs in (
                {
                    "name": topic,
                    "payload": enriched_payload,
                    "priority": priority,
                    "source": event_source,
                },
                {
                    "event_name": topic,
                    "payload": enriched_payload,
                    "priority": priority,
                    "source": event_source,
                },
                {
                    "type": topic,
                    "payload": enriched_payload,
                    "priority": priority,
                    "source": event_source,
                },
                {
                    "topic": topic,
                    "payload": enriched_payload,
                    "priority": priority,
                },
                {
                    "name": topic,
                    "payload": enriched_payload,
                    "priority": priority,
                },
                {
                    "event_name": topic,
                    "payload": enriched_payload,
                    "priority": priority,
                },
        ):
            try:
                event = Event(**event_kwargs)
            except TypeError:
                continue

            if self._try_publish_nowait_best_effort(
                publish_nowait,
                topic,
                event,
            ):
                return

        self.log_warning(
            "Failed to emit best-effort event due to incompatible EventBus contract",
            topic=topic,
        )

    def _complete_best_effort_publish(
            self,
            result: Any,
            *,
            topic: str,
    ) -> None:
        """
        Complete/schedule a best-effort EventBus publish result.

        Some EventBus implementations expose publish_nowait_best_effort as a
        normal synchronous method. Others expose it as async coroutine. This
        helper prevents "coroutine was never awaited" warnings and ensures the
        event is actually submitted.
        """
        if not inspect.isawaitable(result):
            return

        awaitable: Awaitable[Any] = result

        async def _runner() -> None:
            await awaitable

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(_runner())
            except Exception:
                self.log_exception(
                    "Best-effort async event publish failed",
                    topic=topic,
                )
            return

        task = loop.create_task(_runner())

        def _log_task_error(done_task: asyncio.Task[None]) -> None:
            try:
                done_task.result()
            except asyncio.CancelledError:
                self.log_debug(
                    "Best-effort async event publish cancelled",
                    topic=topic,
                )
            except Exception:
                self.log_exception(
                    "Best-effort async event publish failed",
                    topic=topic,
                )

        task.add_done_callback(_log_task_error)

    def _try_publish_nowait_best_effort(
        self,
        publish_nowait: Callable[..., Any],
        topic: str,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """
        Try one EventBus publish_nowait_best_effort call shape.

        Returns:
        - True when the call shape was accepted and the result was completed or
          scheduled;
        - False when only the call signature was incompatible.
        """
        try:
            result = publish_nowait(*args, **kwargs)
        except TypeError:
            return False

        self._complete_best_effort_publish(result, topic=topic)
        return True

    def subscribe_event(
        self,
        topic: str,
        handler: EventHandler,
        **kwargs: Any,
    ) -> Any:
        """
        Subscribe component to EventBus topic and remember subscription.
        """
        if not topic.strip():
            raise ValueError("topic cannot be empty")

        bus = self.ensure_event_bus()
        subscription = bus.subscribe(topic, handler, **kwargs)
        self._subscriptions.append(subscription)
        return subscription

    def remember_scheduler_job(self, job: Any) -> Any:
        """
        Store Scheduler job reference for stats/lifecycle visibility.
        """
        self._scheduler_jobs.append(job)
        return job

    def log_debug(self, message: str, **extra: Any) -> None:
        self.logger.debug(
            message,
            extra={"component": self.component_name, **extra},
        )

    def log_info(self, message: str, **extra: Any) -> None:
        self.logger.info(
            message,
            extra={"component": self.component_name, **extra},
        )

    def log_warning(self, message: str, **extra: Any) -> None:
        self.logger.warning(
            message,
            extra={"component": self.component_name, **extra},
        )

    def log_error(self, message: str, **extra: Any) -> None:
        self.logger.error(
            message,
            extra={"component": self.component_name, **extra},
        )

    def log_exception(self, message: str, **extra: Any) -> None:
        self.logger.exception(
            message,
            extra={"component": self.component_name, **extra},
        )


class StatefulStrategyComponent(BaseStrategyComponent, ABC):
    """
    Base class for strategy components that keep local runtime state.
    """

    @abstractmethod
    def reset_state(self) -> None:
        """
        Reset internal component state.
        """


class ContextAwareStrategyComponent(BaseStrategyComponent, ABC):
    """
    Base class for components that consume StrategyContext.

    Важливо:
    - StrategyContext живе в models.py;
    - старий context.py більше не потрібен;
    - concrete strategies читають тільки StrategyContext, а не analytics/data напряму.
    """

    def validate_context(self, context: StrategyContext) -> None:
        if context is None:
            raise StrategyEvaluationError(f"{self.component_name}: context is required")

        if not isinstance(context, StrategyContext):
            raise StrategyEvaluationError(
                f"{self.component_name}: context must be StrategyContext, got {type(context)!r}"
            )

        context.validate()

    def require_feature(self, context: StrategyContext, feature_name: str) -> Any:
        """
        Return required feature value or raise StrategyEvaluationError.
        """
        self.validate_context(context)

        if not feature_name.strip():
            raise StrategyEvaluationError("feature_name cannot be empty")

        if not context.has_feature(feature_name):
            raise StrategyEvaluationError(
                f"{self.component_name}: missing required feature '{feature_name}' "
                f"for symbol {context.symbol}"
            )

        return context.get_feature(feature_name)

    def optional_feature(
        self,
        context: StrategyContext,
        feature_name: str,
        default: Any = None,
    ) -> Any:
        """
        Return optional feature value from StrategyContext.
        """
        self.validate_context(context)

        if not feature_name.strip():
            raise StrategyEvaluationError("feature_name cannot be empty")

        if not context.has_feature(feature_name):
            return default

        return context.get_feature(feature_name)

    def has_required_features(
        self,
        context: StrategyContext,
        required_features: set[str],
    ) -> bool:
        self.validate_context(context)
        return all(context.has_feature(feature) for feature in required_features)


class BaseStrategy(ContextAwareStrategyComponent, ABC):
    """
    Фінальний базовий клас для всіх concrete strategy classes.

    Контракт:
    - strategy читає тільки StrategyContext;
    - strategy не читає analytics/data напряму;
    - strategy не викликає risk/execution напряму;
    - strategy не публікує signal.generated;
    - strategy повертає StrategyEvaluation;
    - SignalProcessor/SignalRouter перетворює StrategySignal у risk-ready payload.
    """

    component_namespace: str = "strategy.base_strategy"

    category: StrategyCategory = StrategyCategory.HYBRID
    default_setup_type: SetupType = SetupType.UNKNOWN
    default_timeframe: Timeframe = Timeframe.M1
    default_trigger_type: TriggerType = TriggerType.PRIMARY

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        self._definition_override = definition

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            service_name=service_name or f"strategy.{self.__class__.__name__}",
        )

        self._last_no_signal_reason: str | None = None
        self._last_no_signal_metadata: dict[str, Any] = {}
        self._last_not_applicable_reason: str | None = None
        self._last_not_applicable_metadata: dict[str, Any] = {}

    @property
    def strategy_name(self) -> str:
        definition = self.get_definition_config()
        if definition is not None and definition.name.strip():
            return definition.name
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

    @property
    def weight(self) -> float:
        definition = self.get_definition_config()
        if definition is None:
            return 1.0
        return definition.weight

    def get_definition_config(self) -> StrategyDefinitionConfig | None:
        """
        Resolve per-strategy definition config.

        Підтримує:
        - explicit definition override;
        - StrategyConfig.get_strategy(name);
        - StrategyConfig.strategies dict fallback.
        """
        if self._definition_override is not None:
            return self._definition_override

        getter = getattr(self.config, "get_strategy", None)

        if callable(getter):
            definition = getter(self.__class__.__name__)

            if definition is not None:
                if not isinstance(definition, StrategyDefinitionConfig):
                    raise StrategyConfigError(
                        f"{self.strategy_name}: get_strategy({self.__class__.__name__!r}) "
                        f"must return StrategyDefinitionConfig | None, "
                        f"got {type(definition).__name__}"
                    )
                return definition

            if self.strategy_name != self.__class__.__name__:
                definition = getter(self.strategy_name)

                if definition is not None:
                    if not isinstance(definition, StrategyDefinitionConfig):
                        raise StrategyConfigError(
                            f"{self.strategy_name}: get_strategy({self.strategy_name!r}) "
                            f"must return StrategyDefinitionConfig | None, "
                            f"got {type(definition).__name__}"
                        )
                    return definition

        strategies = getattr(self.config, "strategies", None)

        if isinstance(strategies, dict):
            definition = strategies.get(self.__class__.__name__)

            if definition is not None:
                if not isinstance(definition, StrategyDefinitionConfig):
                    raise StrategyConfigError(
                        f"{self.strategy_name}: strategies[{self.__class__.__name__!r}] "
                        f"must be StrategyDefinitionConfig, got {type(definition).__name__}"
                    )
                return definition

            definition = strategies.get(self.strategy_name)

            if definition is not None:
                if not isinstance(definition, StrategyDefinitionConfig):
                    raise StrategyConfigError(
                        f"{self.strategy_name}: strategies[{self.strategy_name!r}] "
                        f"must be StrategyDefinitionConfig, got {type(definition).__name__}"
                    )
                return definition

        return None

    def get_runtime_config(self) -> StrategyRuntimeConfig:
        definition = self.get_definition_config()
        if definition is not None:
            return definition.runtime

        runtime = getattr(self.config, "runtime", None)

        if runtime is None:
            raise StrategyConfigError(
                f"{self.strategy_name}: StrategyConfig.runtime is required"
            )

        if not isinstance(runtime, StrategyRuntimeConfig):
            raise StrategyConfigError(
                f"{self.strategy_name}: StrategyConfig.runtime must be StrategyRuntimeConfig, "
                f"got {type(runtime).__name__}"
            )

        return runtime

    def validate_config(self) -> None:
        super().validate_config()

        definition = self._definition_override
        if definition is not None:
            definition.validate()

        runtime = self.get_runtime_config()
        runtime.validate()

    def is_enabled(self) -> bool:
        return self.get_runtime_config().enabled

    def required_features(self) -> set[str]:
        definition = self.get_definition_config()
        if definition is not None:
            return set(definition.required_features)
        return set()

    def supported_regimes(self) -> set[MarketRegime]:
        runtime = self.get_runtime_config()
        return set(runtime.allowed_regimes)

    def supported_timeframes(self) -> set[Timeframe]:
        runtime = self.get_runtime_config()
        return set(runtime.timeframes)

    def allowed_symbols(self) -> set[str]:
        runtime = self.get_runtime_config()
        return set(runtime.symbols)

    def min_confidence(self) -> float:
        return float(self.get_runtime_config().min_confidence)

    def min_score(self) -> float:
        return float(self.get_runtime_config().min_score)

    def cooldown_seconds(self) -> int:
        return int(self.get_runtime_config().cooldown_seconds)

    def max_signal_age_seconds(self) -> int:
        return int(self.get_runtime_config().max_signal_age_seconds)

    def supports_symbol(self, symbol: str) -> bool:
        if not symbol.strip():
            return False

        allowed = self.allowed_symbols()
        if not allowed:
            return True

        return symbol in allowed

    def supports_timeframe(self, timeframe: Timeframe) -> bool:
        return timeframe in self.supported_timeframes()

    def supports_regime(self, regime: MarketRegime) -> bool:
        regimes = self.supported_regimes()

        if MarketRegime.UNKNOWN in regimes:
            return True

        return regime in regimes

    def validate_context_requirements(self, context: StrategyContext) -> None:
        self.validate_context(context)

        if not self.supports_symbol(context.symbol):
            raise StrategyEvaluationError(
                f"{self.strategy_name}: symbol {context.symbol} is not allowed"
            )

        if not self.supports_timeframe(context.timeframe):
            raise StrategyEvaluationError(
                f"{self.strategy_name}: timeframe {context.timeframe} is not supported"
            )

        regime = self._context_regime(context)
        if not self.supports_regime(regime):
            raise StrategyEvaluationError(
                f"{self.strategy_name}: regime {regime.value} is not supported"
            )

        missing = [
            feature
            for feature in self.required_features()
            if not context.has_feature(feature)
        ]
        if missing:
            raise StrategyEvaluationError(
                f"{self.strategy_name}: missing required features: {missing}"
            )

    def remember_no_signal(self, reason: str, **metadata: Any) -> None:
        """
        Store the exact reason why generate_signal() returned None.

        Concrete strategies still return StrategySignal | None, but failed
        StrategyEvaluation objects can now expose actionable diagnostics instead
        of the generic no_signal_generated reason.
        """
        normalized = str(reason or "").strip() or "no_signal_generated"
        self._last_no_signal_reason = normalized
        self._last_no_signal_metadata = dict(metadata)

    def clear_no_signal_reason(self) -> None:
        self._last_no_signal_reason = None
        self._last_no_signal_metadata = {}

    def consume_no_signal_reason(self) -> tuple[list[str], dict[str, Any]]:
        reason = self._last_no_signal_reason or "no_signal_generated"
        metadata = dict(self._last_no_signal_metadata or {})
        self.clear_no_signal_reason()
        return [reason], metadata

    def remember_not_applicable(self, reason: str, **metadata: Any) -> None:
        normalized = str(reason or "").strip() or "strategy_not_applicable"
        self._last_not_applicable_reason = normalized
        self._last_not_applicable_metadata = dict(metadata)

    def clear_not_applicable_reason(self) -> None:
        self._last_not_applicable_reason = None
        self._last_not_applicable_metadata = {}

    def consume_not_applicable_reason(self) -> tuple[list[str], dict[str, Any]]:
        reason = self._last_not_applicable_reason or "strategy_not_applicable"
        metadata = dict(self._last_not_applicable_metadata or {})
        self.clear_not_applicable_reason()
        return [reason], metadata

    def should_evaluate(self, context: StrategyContext) -> bool:
        """
        Fast applicability check.

        Не кидає exception для normal negative cases.
        """
        self.clear_not_applicable_reason()

        if not self.is_enabled():
            self.remember_not_applicable("strategy_disabled")
            return False

        try:
            self.validate_context_requirements(context)
        except StrategyEvaluationError as exc:
            detail = str(exc)
            reason = "strategy_not_applicable"
            if "missing required features" in detail:
                reason = "missing_required_features"
            elif "timeframe" in detail and "not supported" in detail:
                reason = "unsupported_timeframe"
            elif "symbol" in detail and "not allowed" in detail:
                reason = "unsupported_symbol"
            elif "regime" in detail and "not supported" in detail:
                reason = "unsupported_regime"

            self.remember_not_applicable(
                reason,
                detail=detail,
                required_features=sorted(self.required_features()),
                context_symbol=getattr(context, "symbol", None),
                context_timeframe=(
                    context.timeframe.value
                    if hasattr(getattr(context, "timeframe", None), "value")
                    else str(getattr(context, "timeframe", None))
                ),
            )
            return False

        return True

    async def evaluate(self, context: StrategyContext) -> StrategyEvaluation:
        """
        Evaluate one StrategyContext and return StrategyEvaluation.

        Concrete strategy implements only generate_signal().
        BaseStrategy handles:
        - context validation;
        - enabled/symbol/timeframe/regime/feature checks;
        - confidence/score thresholds;
        - signal enrichment;
        - consistent StrategyEvaluation construction.
        """
        timestamp = getattr(context, "timestamp", None) or utcnow()
        timestamp = ensure_aware_utc(timestamp)

        try:
            self.validate_context(context)

            if not self.should_evaluate(context):
                reasons, not_applicable_metadata = self.consume_not_applicable_reason()
                return self._build_evaluation(
                    context=context,
                    timestamp=timestamp,
                    passed=False,
                    signal=None,
                    reasons=reasons,
                    metadata={
                        "strategy_category": self.category.value,
                        "strategy_priority": self.priority,
                        "strategy_weight": self.weight,
                        "required_features": sorted(self.required_features()),
                        "not_applicable": not_applicable_metadata,
                    },
                )

            self.clear_no_signal_reason()
            signal = await self._call_generate_signal(context)

            if signal is None:
                reasons, no_signal_metadata = self.consume_no_signal_reason()
                return self._build_evaluation(
                    context=context,
                    timestamp=timestamp,
                    passed=False,
                    signal=None,
                    reasons=reasons,
                    metadata={
                        "strategy_category": self.category.value,
                        "strategy_priority": self.priority,
                        "strategy_weight": self.weight,
                        "required_features": sorted(self.required_features()),
                        "no_signal": no_signal_metadata,
                    },
                )

            self._prepare_signal(signal, context=context)
            signal.validate()

            reasons = list(signal.reasons)
            passed = True

            if not self._signal_is_directional(signal):
                passed = False
                reasons.append("signal_side_is_not_directional")

            if signal.confidence < self.min_confidence():
                passed = False
                reasons.append("confidence_below_strategy_minimum")

            if signal.score < self.min_score():
                passed = False
                reasons.append("score_below_strategy_minimum")

            if not passed:
                signal.to_rejected()
                for reason in reasons:
                    self._add_signal_reason(signal, reason)

            return self._build_evaluation(
                context=context,
                timestamp=timestamp,
                passed=passed,
                signal=signal,
                score=signal.score,
                confidence=signal.confidence,
                reasons=reasons,
                metadata={
                    "strategy_category": self.category.value,
                    "strategy_priority": self.priority,
                    "strategy_weight": self.weight,
                    "required_features": sorted(self.required_features()),
                },
            )

        except Exception as exc:
            self.log_exception(
                "Strategy evaluation failed",
                strategy_name=self.strategy_name,
                symbol=getattr(context, "symbol", None),
                error=str(exc),
            )

            return self._build_evaluation(
                context=context,
                timestamp=timestamp,
                passed=False,
                signal=None,
                reasons=[f"evaluation_error:{exc}"],
                metadata={
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            )

    @abstractmethod
    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        """
        Generate internal StrategySignal from StrategyContext.

        Concrete strategies implement only this method.

        Заборонено:
        - публікувати signal.generated;
        - викликати RiskManager;
        - викликати Execution;
        - рахувати final position size;
        - читати analytics/data напряму.
        """

    def build_signal(
        self,
        *,
        context: StrategyContext,
        side: SignalSide,
        confidence: float,
        score: float,
        setup_type: SetupType | None = None,
        reasons: list[str] | None = None,
        confirmations: list[str] | None = None,
        source_features: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        trigger_type: TriggerType | None = None,
        origin: SignalOrigin = SignalOrigin.SINGLE_STRATEGY,
        priority: SignalPriority = SignalPriority.MEDIUM,
        status: SignalStatus = SignalStatus.NEW,
    ) -> StrategySignal:
        """
        Convenience helper for concrete strategies.

        Стратегія може створювати signal через цей метод, а SignalProcessor
        пізніше добудує entry/exit/execution plan і risk-ready payload.
        """
        self.validate_context_requirements(context)

        signal = StrategySignal(
            symbol=context.symbol,
            side=side,
            strategy_name=self.strategy_name,
            category=self.category,
            timeframe=context.timeframe,
            setup_type=setup_type or self.default_setup_type,
            timestamp=ensure_aware_utc(context.timestamp),
            confidence=float(confidence),
            score=float(score),
            confidence_grade=confidence_to_grade(float(confidence)),
            strength=confidence_to_strength(float(confidence)),
            status=status,
            trigger_type=trigger_type or self.default_trigger_type,
            origin=origin,
            priority=priority,
            reasons=list(reasons or []),
            confirmations=list(confirmations or []),
            source_features=list(source_features or []),
            regime=self._context_regime(context),
            metadata=dict(metadata or {}),
        )

        self._prepare_signal(signal, context=context)
        signal.validate()
        return signal

    async def _call_generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        result = self.generate_signal(context)

        if inspect.isawaitable(result):
            return await result

        return result

    def _prepare_signal(
        self,
        signal: StrategySignal,
        *,
        context: StrategyContext,
    ) -> None:
        """
        Normalize/enrich StrategySignal before returning StrategyEvaluation.
        """
        if not signal.symbol:
            signal.symbol = context.symbol

        if not signal.strategy_name:
            signal.strategy_name = self.strategy_name

        signal.category = signal.category or self.category
        signal.timeframe = signal.timeframe or context.timeframe
        signal.setup_type = signal.setup_type or self.default_setup_type
        signal.timestamp = ensure_aware_utc(signal.timestamp or context.timestamp or utcnow())

        if signal.confidence_grade is None:
            signal.confidence_grade = confidence_to_grade(signal.confidence)

        if signal.strength is None:
            signal.strength = confidence_to_strength(signal.confidence)

        if signal.regime is None or signal.regime is MarketRegime.UNKNOWN:
            signal.regime = self._context_regime(context)

        signal.metadata.setdefault("strategy_name", self.strategy_name)
        signal.metadata.setdefault("category", self.category.value)
        signal.metadata.setdefault("timeframe", context.timeframe.value)
        signal.metadata.setdefault("setup_type", signal.setup_type.value)
        signal.metadata.setdefault("source", "strategy")
        signal.metadata.setdefault("signal_id", signal.signal_id)

    def _build_evaluation(
            self,
            *,
            context: StrategyContext | None,
            timestamp: datetime,
            passed: bool,
            signal: StrategySignal | None,
            score: float = 0.0,
            confidence: float = 0.0,
            reasons: list[str] | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> StrategyEvaluation:
        symbol = "unknown"

        if context is not None:
            raw_symbol = getattr(context, "symbol", None)
            if isinstance(raw_symbol, str) and raw_symbol.strip():
                symbol = raw_symbol.strip()

        if signal is not None:
            raw_signal_symbol = getattr(signal, "symbol", None)
            if isinstance(raw_signal_symbol, str) and raw_signal_symbol.strip():
                symbol = raw_signal_symbol.strip()

        return StrategyEvaluation(
            strategy_name=self.strategy_name,
            symbol=symbol,
            timestamp=ensure_aware_utc(timestamp),
            signal=signal,
            passed=passed,
            score=float(score),
            confidence=float(confidence),
            reasons=list(reasons or []),
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _signal_is_directional(signal: StrategySignal) -> bool:
        side = signal.side

        if isinstance(side, SignalSide):
            return side.is_directional

        return str(side) in {SignalSide.LONG.value, SignalSide.SHORT.value}

    @staticmethod
    def _add_signal_reason(signal: StrategySignal, reason: str) -> None:
        if not reason:
            return

        add_reason = getattr(signal, "add_reason", None)
        if callable(add_reason):
            add_reason(reason)
            return

        if reason not in signal.reasons:
            signal.reasons.append(reason)

    @staticmethod
    def _context_regime(context: StrategyContext) -> MarketRegime:
        current_regime = getattr(context, "current_regime", None)
        if isinstance(current_regime, MarketRegime):
            return current_regime

        regime = getattr(context, "regime", None)
        if regime is None:
            return MarketRegime.UNKNOWN

        value = getattr(regime, "regime", None)
        if isinstance(value, MarketRegime):
            return value

        if isinstance(value, str):
            try:
                return MarketRegime(value)
            except ValueError:
                return MarketRegime.UNKNOWN

        return MarketRegime.UNKNOWN