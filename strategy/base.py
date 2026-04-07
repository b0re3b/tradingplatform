from __future__ import annotations

from abc import ABC
from typing import Any

from .config import StrategyConfig
from strategy.exceptions import StrategyConfigError
from .models import SignalContext


class StrategyComponent(ABC):
    """
    Базовий компонент strategy layer.

    Дає уніфіковану точку для:
    - config
    - event_bus
    - logger
    - lifecycle hooks
    """

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: Any | None = None,
        logger: Any | None = None,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.logger = logger

    @property
    def component_name(self) -> str:
        return self.__class__.__name__

    def validate_config(self) -> None:
        if self.config is None:
            raise StrategyConfigError(f"{self.component_name}: config is required")
        self.config.validate()

    async def start(self) -> None:
        """Lifecycle hook for async initialization."""

    async def stop(self) -> None:
        """Lifecycle hook for async cleanup."""

    def log_debug(self, message: str, **extra: Any) -> None:
        if self.logger:
            try:
                self.logger.debug(message, extra=extra)
            except TypeError:
                self.logger.debug(message)

    def log_info(self, message: str, **extra: Any) -> None:
        if self.logger:
            try:
                self.logger.info(message, extra=extra)
            except TypeError:
                self.logger.info(message)

    def log_warning(self, message: str, **extra: Any) -> None:
        if self.logger:
            try:
                self.logger.warning(message, extra=extra)
            except TypeError:
                self.logger.warning(message)

    def log_error(self, message: str, **extra: Any) -> None:
        if self.logger:
            try:
                self.logger.error(message, extra=extra)
            except TypeError:
                self.logger.error(message)


class StatefulComponent(StrategyComponent, ABC):
    """
    Базовий клас для компонентів, які тримають внутрішній state.
    """

    def reset_state(self) -> None:
        """Reset internal state to defaults."""


class ContextAwareComponent(StrategyComponent, ABC):
    """
    Базовий клас для компонентів, які працюють з SignalContext.
    """

    def validate_context(self, context: SignalContext) -> None:
        if context is None:
            raise ValueError(f"{self.component_name}: context is required")
        context.validate()


class EventEmitterMixin:
    """
    Mixin для компонентів, які публікують події в EventBus.
    """

    event_bus: Any | None

    async def emit_event(
        self,
        event_name: str,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        if self.event_bus is None:
            return

        emit = getattr(self.event_bus, "emit", None)
        if emit is None:
            raise AttributeError("event_bus does not provide emit()")

        await emit(event_name, payload, **kwargs)


class EventSubscriberMixin:
    """
    Mixin для компонентів, які підписуються на події EventBus.
    """

    event_bus: Any | None

    def subscribe_event(self, event_name: str, handler: Any) -> None:
        if self.event_bus is None:
            return

        subscribe = getattr(self.event_bus, "subscribe", None)
        if subscribe is None:
            raise AttributeError("event_bus does not provide subscribe()")

        subscribe(event_name, handler)


class NamedEntityMixin:
    """
    Mixin з уніфікованим name/id для registry і логування.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__


class PrioritizedMixin:
    """
    Mixin для сутностей, у яких є пріоритет.
    """

    @property
    def priority(self) -> int:
        return 100