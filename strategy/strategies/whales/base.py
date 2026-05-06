# strategy/strategies/whales/base.py

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

from core.event_bus import Event, EventBus, EventPriority
from core.logger import TradingLoggerAdapter, get_logger
from core.scheduler import Scheduler
from strategy.base import ContextAwareComponent, NamedEntityMixin, PrioritizedMixin
from strategy.config import StrategyConfig
from strategy.enums import (
    ConfidenceGrade,
    FilterDecision,
    MarketRegime,
    SignalPriority,
    SignalStrength,
    StrategyCategory,
)
from strategy.exceptions import StrategyEvaluationError
from strategy.models import (
    FilterResult,
    SignalContext,
    StrategyEvaluation,
    StrategySignal,
)

LoggerLike = logging.Logger | TradingLoggerAdapter


@dataclass(slots=True)
class WhaleStrategyEventConfig:
    """
    Event-driven runtime config для whale-стратегій.

    Цей dataclass не замінює StrategyConfig, а лише описує,
    як конкретна strategy-evaluator інтегрується з EventBus.
    """

    context_topic: str = "strategy.context.whales"
    generated_signal_topic: str = "signal.generated"
    rejected_signal_topic: str = "signal.rejected"
    evaluation_completed_topic: str = "strategy.evaluation.completed"
    evaluation_failed_topic: str = "strategy.evaluation.failed"

    subscribe_enabled: bool = True
    publish_generated_signals: bool = True
    publish_rejected_signals: bool = True
    publish_evaluations: bool = False
    publish_failures: bool = True

    scheduler_healthcheck_enabled: bool = False
    scheduler_healthcheck_interval: float = 60.0


class WhaleStrategyBase(
    ContextAwareComponent,
    NamedEntityMixin,
    PrioritizedMixin,
    ABC,
):
    """
    Базовий event-driven клас для strategy/strategies/whales.

    Призначення:
        - уніфікує DI через config/event_bus/scheduler/logger;
        - дає register() для EventBus-підписки;
        - приймає SignalContext із EventBus-події;
        - викликає evaluate(context);
        - публікує signal.generated / signal.rejected / strategy.evaluation.*;
        - містить спільні helper-и для whale-стратегій;
        - не запускає власних uncontrolled asyncio loops.

    Важливо:
        Сама стратегія залишається evaluator-компонентом.
        Її можна викликати напряму через StrategyEngine.evaluate(...),
        або використовувати event-driven шлях через register().
    """

    DEFAULT_CONTEXT_TOPIC = "strategy.context.whales"

    def __init__(
        self,
        *,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        event_config: WhaleStrategyEventConfig | None = None,
        logger: LoggerLike | None = None,
        strategy_name: str,
    ) -> None:
        self.strategy_name = strategy_name
        self.event_bus: EventBus | None = event_bus
        self.scheduler: Scheduler | None = scheduler
        self.event_config = event_config or WhaleStrategyEventConfig()

        self._logger: LoggerLike = logger or get_logger(
            __name__,
            event_type="strategy",
            strategy_name=strategy_name,
            category=StrategyCategory.WHALES.value,
        )

        super().__init__(
            config=config,
            event_bus=event_bus,
            logger=self._logger,
        )

        self._subscription = None
        self._healthcheck_job_id: str | None = None

        self.validate_config()

    # =========================================================================
    # Required strategy contract
    # =========================================================================

    @property
    def name(self) -> str:
        return self.strategy_name

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.WHALES

    @property
    def priority(self) -> int:
        definition = self._strategy_definition
        if definition is None:
            return 100
        return definition.priority

    @property
    def required_features(self) -> set[str]:
        return set()

    @property
    def _strategy_definition(self):
        return self.config.get_strategy(self.strategy_name)

    @property
    def _runtime_config(self):
        definition = self._strategy_definition
        if definition is not None:
            return definition.runtime
        return self.config.runtime

    @property
    def _metadata(self) -> dict[str, Any]:
        definition = self._strategy_definition
        if definition is None:
            return {}
        return dict(definition.metadata)

    @abstractmethod
    def evaluate(self, context: SignalContext) -> StrategyEvaluation:
        """
        Основна sync-evaluation логіка конкретної стратегії.

        Дочірні класи реалізують:
            - extraction;
            - setup detection;
            - scoring;
            - signal building;
            - filters;
            - execution plan draft.
        """
        raise NotImplementedError

    # =========================================================================
    # Event-driven lifecycle
    # =========================================================================

    def register(self) -> None:
        """
        Реєструє strategy evaluator в EventBus.

        Очікувана подія:
            topic: strategy.context.whales
            payload:
                - SignalContext
                або
                - {"context": SignalContext}

        Результат:
            - signal.generated, якщо evaluation.passed=True і є signal;
            - signal.rejected, якщо evaluation.passed=False;
            - strategy.evaluation.failed, якщо evaluate() впав.
        """
        if self.event_bus is None:
            self._logger.warning(
                "Whale strategy register skipped: event_bus is not provided | strategy=%s",
                self.strategy_name,
            )
            return

        if not self.event_config.subscribe_enabled:
            self._logger.info(
                "Whale strategy subscription disabled | strategy=%s",
                self.strategy_name,
            )
            return

        self._subscription = self.event_bus.subscribe(
            self.event_config.context_topic,
            self.on_context_event,
            name=f"{self.strategy_name}.on_context_event",
        )

        self._register_scheduler_jobs()

        self._logger.info(
            "Whale strategy registered | strategy=%s topic=%s",
            self.strategy_name,
            self.event_config.context_topic,
        )

    async def on_context_event(self, event: Event) -> None:
        """
        EventBus handler.

        Цей метод не містить торгової логіки.
        Він тільки:
            - дістає SignalContext;
            - викликає evaluate();
            - публікує результат.
        """
        context = self._extract_context_from_event(event)

        if context is None:
            await self._emit_evaluation_failed(
                event=event,
                error="Invalid event payload: SignalContext is missing",
            )
            return

        try:
            evaluation = self.evaluate(context)
        except Exception as exc:
            self._logger.exception(
                "Whale strategy evaluation failed | strategy=%s symbol=%s event_id=%s",
                self.strategy_name,
                getattr(context, "symbol", None),
                event.event_id,
            )

            await self._emit_evaluation_failed(
                event=event,
                error=str(exc),
                context=context,
            )
            return

        await self._publish_evaluation_result(
            evaluation=evaluation,
            source_event=event,
        )

    def _extract_context_from_event(self, event: Event) -> SignalContext | None:
        payload = event.payload

        if isinstance(payload, SignalContext):
            return payload

        if isinstance(payload, Mapping):
            context = payload.get("context")
            if isinstance(context, SignalContext):
                return context

        return None

    def _register_scheduler_jobs(self) -> None:
        """
        Реєструє periodic jobs тільки через core.scheduler.Scheduler.

        За замовчуванням healthcheck вимкнений, бо strategy evaluator
        зазвичай не потребує background задач.
        """
        if self.scheduler is None:
            return

        if not self.event_config.scheduler_healthcheck_enabled:
            return

        existing = self.scheduler.get_job_by_name(f"{self.strategy_name}.healthcheck")
        if existing is not None:
            self._healthcheck_job_id = existing.job_id
            return

        self._healthcheck_job_id = self.scheduler.add_interval_job(
            name=f"{self.strategy_name}.healthcheck",
            func=self._scheduler_healthcheck,
            interval=self.event_config.scheduler_healthcheck_interval,
            run_immediately=False,
            max_retries=1,
            retry_delay=1.0,
            timeout=5.0,
            allow_overlap=False,
            enabled=True,
        )

        self._logger.info(
            "Whale strategy healthcheck job registered | strategy=%s job_id=%s",
            self.strategy_name,
            self._healthcheck_job_id,
        )

    async def _scheduler_healthcheck(self) -> None:
        """
        Легкий healthcheck для scheduler.

        Не запускає торгову логіку.
        Не polling-ить ринок.
        Не дублює StrategyEngine.
        """
        if self.event_bus is None:
            return

        await self.event_bus.emit(
            "system.strategy.healthcheck",
            {
                "strategy_name": self.strategy_name,
                "category": self.category.value,
                "registered": self._subscription is not None,
                "required_features": sorted(self.required_features),
            },
            source=self.strategy_name,
        )

    # =========================================================================
    # Event publishing
    # =========================================================================

    async def _publish_evaluation_result(
        self,
        *,
        evaluation: StrategyEvaluation,
        source_event: Event,
    ) -> None:
        if self.event_bus is None:
            return

        if self.event_config.publish_evaluations:
            await self.event_bus.emit(
                self.event_config.evaluation_completed_topic,
                self._evaluation_to_event_payload(evaluation),
                priority=EventPriority.NORMAL,
                source=self.strategy_name,
                correlation_id=source_event.correlation_id,
                headers=self._build_child_headers(source_event),
            )

        if evaluation.passed and evaluation.signal is not None:
            if self.event_config.publish_generated_signals:
                await self.event_bus.emit(
                    self.event_config.generated_signal_topic,
                    self._signal_to_event_payload(evaluation.signal),
                    priority=self._event_priority_from_signal(evaluation.signal),
                    source=self.strategy_name,
                    correlation_id=source_event.correlation_id,
                    headers=self._build_child_headers(source_event),
                )
            return

        if self.event_config.publish_rejected_signals:
            await self.event_bus.emit(
                self.event_config.rejected_signal_topic,
                self._evaluation_to_event_payload(evaluation),
                priority=EventPriority.LOW,
                source=self.strategy_name,
                correlation_id=source_event.correlation_id,
                headers=self._build_child_headers(source_event),
            )

    async def _emit_evaluation_failed(
        self,
        *,
        event: Event,
        error: str,
        context: SignalContext | None = None,
    ) -> None:
        if self.event_bus is None or not self.event_config.publish_failures:
            return

        payload: dict[str, Any] = {
            "strategy_name": self.strategy_name,
            "category": self.category.value,
            "error": error,
            "source_event_id": event.event_id,
            "source_topic": event.topic,
        }

        if context is not None:
            payload.update(
                {
                    "symbol": context.symbol,
                    "timeframe": str(context.timeframe),
                    "timestamp": context.timestamp,
                }
            )

        await self.event_bus.emit(
            self.event_config.evaluation_failed_topic,
            payload,
            priority=EventPriority.HIGH,
            source=self.strategy_name,
            correlation_id=event.correlation_id,
            headers=self._build_child_headers(event),
        )

    def _build_child_headers(self, event: Event) -> dict[str, Any]:
        headers = dict(event.headers)
        headers.update(
            {
                "parent_event_id": event.event_id,
                "parent_topic": event.topic,
                "strategy_name": self.strategy_name,
                "strategy_category": self.category.value,
            }
        )
        return headers

    def _signal_to_event_payload(self, signal: StrategySignal) -> dict[str, Any]:
        if hasattr(signal, "to_event") and callable(signal.to_event):
            result = signal.to_event()
            if isinstance(result, dict):
                return result

        if hasattr(signal, "to_dict") and callable(signal.to_dict):
            result = signal.to_dict()
            if isinstance(result, dict):
                return result

        return {
            "symbol": signal.symbol,
            "side": signal.side.value if hasattr(signal.side, "value") else str(signal.side),
            "strategy_name": signal.strategy_name,
            "category": signal.category.value if hasattr(signal.category, "value") else str(signal.category),
            "timeframe": str(signal.timeframe),
            "timestamp": signal.timestamp,
            "confidence": signal.confidence,
            "score": signal.score,
            "priority": signal.priority.value if hasattr(signal.priority, "value") else str(signal.priority),
            "metadata": dict(signal.metadata),
            "reasons": list(signal.reasons),
            "confirmations": list(signal.confirmations),
            "source_features": list(signal.source_features),
        }

    def _evaluation_to_event_payload(self, evaluation: StrategyEvaluation) -> dict[str, Any]:
        if hasattr(evaluation, "to_event") and callable(evaluation.to_event):
            result = evaluation.to_event()
            if isinstance(result, dict):
                return result

        if hasattr(evaluation, "to_dict") and callable(evaluation.to_dict):
            result = evaluation.to_dict()
            if isinstance(result, dict):
                return result

        return {
            "strategy_name": evaluation.strategy_name,
            "symbol": evaluation.symbol,
            "timestamp": evaluation.timestamp,
            "passed": evaluation.passed,
            "score": evaluation.score,
            "confidence": evaluation.confidence,
            "reasons": list(evaluation.reasons),
            "signal": (
                self._signal_to_event_payload(evaluation.signal)
                if evaluation.signal is not None
                else None
            ),
        }

    def _event_priority_from_signal(self, signal: StrategySignal) -> EventPriority:
        priority = signal.priority

        if priority == SignalPriority.CRITICAL:
            return EventPriority.CRITICAL
        if priority == SignalPriority.HIGH:
            return EventPriority.HIGH
        if priority == SignalPriority.MEDIUM:
            return EventPriority.NORMAL
        return EventPriority.LOW

    # =========================================================================
    # Shared payload helpers
    # =========================================================================

    def _resolve_payload(
        self,
        context: SignalContext,
        *,
        names: tuple[str, ...],
    ) -> dict[str, Any]:
        for name in names:
            value = context.whales.get(name)
            resolved = self._object_to_dict(value)
            if resolved:
                return resolved

        for name in names:
            feature_value = context.get_feature(name)
            resolved = self._object_to_dict(feature_value)
            if resolved:
                return resolved

            snapshot = context.get_feature_snapshot(name)
            if snapshot is not None:
                resolved = self._object_to_dict(snapshot.value)
                if resolved:
                    return resolved

        return {}

    def _object_to_dict(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {}

        if isinstance(value, dict):
            return dict(value)

        if hasattr(value, "to_event") and callable(value.to_event):
            try:
                result = value.to_event()
                if isinstance(result, dict):
                    return result
            except Exception:
                self._logger.debug(
                    "Failed to convert object using to_event() | strategy=%s type=%s",
                    self.strategy_name,
                    type(value).__name__,
                    exc_info=True,
                )

        if hasattr(value, "to_dict") and callable(value.to_dict):
            try:
                result = value.to_dict()
                if isinstance(result, dict):
                    return result
            except Exception:
                self._logger.debug(
                    "Failed to convert object using to_dict() | strategy=%s type=%s",
                    self.strategy_name,
                    type(value).__name__,
                    exc_info=True,
                )

        if hasattr(value, "__dict__"):
            try:
                return dict(vars(value))
            except Exception:
                self._logger.debug(
                    "Failed to convert object using vars() | strategy=%s type=%s",
                    self.strategy_name,
                    type(value).__name__,
                    exc_info=True,
                )

        return {}

    # =========================================================================
    # Shared filters
    # =========================================================================

    def _run_common_filters(
        self,
        *,
        context: SignalContext,
        signal: StrategySignal,
    ) -> list[FilterResult]:
        results: list[FilterResult] = []

        results.extend(self._run_regime_filter(context, signal))
        results.extend(self._run_spread_filter(context))
        results.extend(self._run_liquidity_filter(context))
        results.extend(self._run_volatility_filter(context))

        return results

    def _run_regime_filter(
        self,
        context: SignalContext,
        signal: StrategySignal,
    ) -> list[FilterResult]:
        if not self.config.filters.enable_regime_filter:
            return []

        allowed_regimes = set(self._runtime_config.allowed_regimes)
        regime = self._resolve_regime(context)

        if not allowed_regimes:
            return []

        if MarketRegime.UNKNOWN in allowed_regimes and regime == MarketRegime.UNKNOWN:
            return [
                FilterResult(
                    name="regime_filter",
                    decision=FilterDecision.WARN,
                    reason="Regime unknown but allowed by runtime config",
                )
            ]

        if regime not in allowed_regimes:
            return [
                FilterResult(
                    name="regime_filter",
                    decision=FilterDecision.BLOCK,
                    reason=f"Regime {regime.value} is not allowed",
                )
            ]

        return [
            FilterResult(
                name="regime_filter",
                decision=FilterDecision.PASS,
                reason=f"Regime {regime.value} allowed",
            )
        ]

    def _run_spread_filter(self, context: SignalContext) -> list[FilterResult]:
        if not self.config.filters.enable_spread_filter:
            return []

        spread_bps = None
        if context.price is not None:
            spread_bps = context.price.spread_bps

        if spread_bps is None:
            return [
                FilterResult(
                    name="spread_filter",
                    decision=FilterDecision.WARN,
                    reason="Spread unavailable",
                )
            ]

        if spread_bps > self.config.filters.max_spread_bps:
            return [
                FilterResult(
                    name="spread_filter",
                    decision=FilterDecision.BLOCK,
                    reason=f"Spread too high: {spread_bps:.4f} bps",
                )
            ]

        return [
            FilterResult(
                name="spread_filter",
                decision=FilterDecision.PASS,
                reason=f"Spread acceptable: {spread_bps:.4f} bps",
            )
        ]

    def _run_liquidity_filter(self, context: SignalContext) -> list[FilterResult]:
        if not self.config.filters.enable_liquidity_filter:
            return []

        liquidity_score = self._safe_float(
            context.get_feature("liquidity_score"),
            default=None,
        )

        if liquidity_score is None:
            return [
                FilterResult(
                    name="liquidity_filter",
                    decision=FilterDecision.WARN,
                    reason="Liquidity score unavailable",
                )
            ]

        if liquidity_score < self.config.filters.min_liquidity_score:
            return [
                FilterResult(
                    name="liquidity_filter",
                    decision=FilterDecision.BLOCK,
                    reason=f"Liquidity score too low: {liquidity_score:.4f}",
                )
            ]

        return [
            FilterResult(
                name="liquidity_filter",
                decision=FilterDecision.PASS,
                reason=f"Liquidity score acceptable: {liquidity_score:.4f}",
            )
        ]

    def _run_volatility_filter(self, context: SignalContext) -> list[FilterResult]:
        if not self.config.filters.enable_volatility_filter:
            return []

        volatility_zscore = self._safe_float(
            context.get_feature("volatility_zscore"),
            default=None,
        )

        if volatility_zscore is None:
            return [
                FilterResult(
                    name="volatility_filter",
                    decision=FilterDecision.WARN,
                    reason="Volatility z-score unavailable",
                )
            ]

        if volatility_zscore > self.config.filters.max_volatility_zscore:
            return [
                FilterResult(
                    name="volatility_filter",
                    decision=FilterDecision.BLOCK,
                    reason=f"Volatility too high: {volatility_zscore:.4f}",
                )
            ]

        return [
            FilterResult(
                name="volatility_filter",
                decision=FilterDecision.PASS,
                reason=f"Volatility acceptable: {volatility_zscore:.4f}",
            )
        ]

    # =========================================================================
    # Shared market/context helpers
    # =========================================================================

    def _resolve_regime(self, context: SignalContext) -> MarketRegime:
        if context.regime is None:
            return MarketRegime.UNKNOWN
        return context.regime.regime

    def _resolve_reference_price(self, context: SignalContext) -> float | None:
        if context.price is None:
            return None

        if context.price.mid_price is not None and context.price.mid_price > 0:
            return context.price.mid_price

        if context.price.last_price is not None and context.price.last_price > 0:
            return context.price.last_price

        if context.price.mark_price is not None and context.price.mark_price > 0:
            return context.price.mark_price

        return None

    def _suggest_holding_seconds(self, context: SignalContext) -> int:
        mapping = {
            "1s": 60,
            "5s": 180,
            "15s": 300,
            "1m": 900,
            "3m": 1800,
            "5m": 3600,
            "15m": 4 * 3600,
            "30m": 6 * 3600,
            "1h": 12 * 3600,
            "4h": 24 * 3600,
            "1d": 3 * 24 * 3600,
        }
        return mapping.get(str(context.timeframe), 1800)

    # =========================================================================
    # Shared scoring helpers
    # =========================================================================

    def _resolve_strength(self, score: float, confidence: float) -> SignalStrength:
        composite = (score + confidence) / 2.0

        if composite >= 0.90:
            return SignalStrength.EXTREME

        if composite >= 0.75:
            return SignalStrength.STRONG

        if composite >= 0.55:
            return SignalStrength.MODERATE

        return SignalStrength.WEAK

    def _resolve_confidence_grade(self, confidence: float) -> ConfidenceGrade:
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

    def _map_priority(self, priority: int) -> SignalPriority:
        if priority <= 25:
            return SignalPriority.CRITICAL

        if priority <= 50:
            return SignalPriority.HIGH

        if priority <= 100:
            return SignalPriority.MEDIUM

        return SignalPriority.LOW

    def _safe_float(
        self,
        value: Any,
        default: float | None = None,
    ) -> float | None:
        if value is None:
            return default

        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _safe_int(
        self,
        value: Any,
        default: int = 0,
    ) -> int:
        if value is None:
            return default

        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _clamp(
        self,
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        return max(minimum, min(value, maximum))

    # =========================================================================
    # Evaluation guard helper
    # =========================================================================

    def _wrap_evaluation_error(
        self,
        *,
        context: SignalContext,
        exc: Exception,
    ) -> StrategyEvaluationError:
        return StrategyEvaluationError(
            f"{self.strategy_name} failed for symbol={context.symbol}: {exc}"
        )