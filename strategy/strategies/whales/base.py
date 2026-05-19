# strategy/strategies/whales/base.py

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

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


FUTURES_MARKET_TYPES: frozenset[str] = frozenset(
    {
        "perpetual",
        "futures",
        "linear",
        "inverse",
        "swap",
        "usdm_futures",
        "coinm_futures",
    }
)


DEFAULT_WHALE_FEATURE_MAX_AGE_MS = 90_000
DEFAULT_WHALE_CONTEXT_TOPIC = "strategy.context.whales"


@dataclass(slots=True)
class WhaleStrategyEventConfig:
    """
    Тимчасовий event-driven adapter config для whale-стратегій.

    Важливо:
    - це проміжний контракт для конкретного strategy/strategies/whales пакету;
    - пізніше publishing/filtering/building треба буде винести у StrategyEngine /
      SignalProcessor;
    - зараз залишаємо цей шар, щоб узгодити strategy/strategies/* з analytics/*.
    """

    context_topic: str = DEFAULT_WHALE_CONTEXT_TOPIC

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

    # Domain validation toggles.
    validate_scope_enabled: bool = True
    validate_freshness_enabled: bool = True
    validate_futures_market_type_enabled: bool = True

    # Whale analytics often uses realtime aggregation while strategies can run on
    # 1m/5m/15m contexts, тому timeframe strict-match вимкнений за замовчуванням.
    strict_timeframe_scope: bool = False

    # Якщо payload не має exchange/market_type/symbol/timeframe, не блокуємо його,
    # бо StrategyContextBuilder може вже гарантувати scope на рівні context.
    require_payload_scope: bool = False

    # Якщо True, invalid whale payload повертається як {}, щоб стратегія не
    # будувала сигнал на stale/wrong-scope даних.
    drop_invalid_payloads: bool = True

    whale_feature_max_age_ms: int = DEFAULT_WHALE_FEATURE_MAX_AGE_MS


@dataclass(slots=True)
class WhalePayloadValidation:
    """
    Результат validation одного analytics.whales payload.
    """

    valid: bool
    present: bool
    fresh: bool = True
    scope_valid: bool = True
    futures_market_type_valid: bool = True
    reasons: list[str] = field(default_factory=list)

    def add_reason(self, reason: str) -> None:
        if reason:
            self.reasons.append(reason)


@dataclass(slots=True)
class WhaleFeaturePayload:
    """
    Нормалізований wrapper для одного whale feature payload.
    """

    canonical_name: str
    aliases: tuple[str, ...]
    payload: dict[str, Any]
    source_name: str | None = None
    source_location: str | None = None
    validation: WhalePayloadValidation = field(
        default_factory=lambda: WhalePayloadValidation(valid=False, present=False)
    )

    @property
    def present(self) -> bool:
        return bool(self.payload)

    @property
    def valid(self) -> bool:
        return self.validation.valid

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(slots=True)
class WhaleStrategyInputSnapshot:
    """
    Нормалізований набір whale inputs для конкретної стратегії.

    Це не заміна SignalContext. Це легкий domain snapshot над:
    - context.whales;
    - context.feature_map;
    - context feature snapshots.
    """

    strategy_name: str
    symbol: str
    timestamp_ms: int | None
    features: dict[str, WhaleFeaturePayload] = field(default_factory=dict)

    def get(self, name: str) -> dict[str, Any]:
        feature = self.features.get(name)
        if feature is None:
            return {}
        return feature.to_dict()

    def feature(self, name: str) -> WhaleFeaturePayload | None:
        return self.features.get(name)

    @property
    def missing_features(self) -> list[str]:
        return [
            name
            for name, feature in self.features.items()
            if not feature.present
        ]

    @property
    def invalid_features(self) -> list[str]:
        return [
            name
            for name, feature in self.features.items()
            if feature.present and not feature.valid
        ]

    @property
    def valid(self) -> bool:
        return not self.invalid_features


class WhaleStrategyBase(
    ContextAwareComponent,
    NamedEntityMixin,
    PrioritizedMixin,
    ABC,
):
    """
    Базовий клас для strategy/strategies/whales.

    Поточний тимчасовий scope:
    - конкретні whale-стратегії ще можуть працювати як самодостатні evaluator-и;
    - EventBus/Scheduler/config передаються всередину конкретних стратегій;
    - register()/on_context_event() залишені для compatibility;
    - пізніше publishing/filtering/final scoring буде винесено в StrategyEngine /
      SignalProcessor.

    Що цей base робить зараз:
    - DI через config/event_bus/scheduler/logger;
    - EventBus subscription на strategy.context.whales;
    - evaluate(context) dispatch;
    - signal.generated / signal.rejected publishing;
    - common whale payload resolving;
    - scope validation: exchange / market_type / symbol / timeframe;
    - freshness validation;
    - futures-only market_type guard;
    - common filters;
    - shared scoring/market helpers.

    Чого цей base НЕ має робити:
    - не читати exchange adapters напряму;
    - не читати raw market data напряму;
    - не запускати власні uncontrolled asyncio loops;
    - не містити analytics.whales detection logic;
    - не дублювати whale analytics calculations.
    """

    DEFAULT_CONTEXT_TOPIC = DEFAULT_WHALE_CONTEXT_TOPIC

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

        self._subscription: Any | None = None
        self._healthcheck_job_id: str | None = None
        self._last_validation_warnings: dict[str, float] = {}

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
        return int(definition.priority)

    @property
    def required_features(self) -> set[str]:
        return set()

    @property
    def _strategy_definition(self) -> Any | None:
        return self.config.get_strategy(self.strategy_name)

    @property
    def _runtime_config(self) -> Any:
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
        Основна sync-evaluation логіка конкретної whale-стратегії.

        Дочірній клас відповідає за:
        - extraction domain inputs;
        - setup detection;
        - domain score/confidence;
        - signal building;
        - temporary local filters;
        - temporary execution draft.
        """
        raise NotImplementedError

    # =========================================================================
    # Event-driven lifecycle
    # =========================================================================

    def register(self) -> None:
        """
        Реєструє strategy evaluator в EventBus.

        Тимчасовий compatibility path:
            strategy.context.whales -> evaluate(context) -> signal.generated/rejected

        Пізніше цей шлях має перейти у StrategyEventHandler / StrategyEngine.
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

        if self._subscription is not None:
            self._logger.debug(
                "Whale strategy already registered | strategy=%s topic=%s",
                self.strategy_name,
                self.event_config.context_topic,
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

    async def stop(self) -> None:
        """
        Best-effort cleanup для тимчасового event-driven режиму.
        """
        self._remove_scheduler_job()
        self._unsubscribe()

        self._logger.info(
            "Whale strategy stopped | strategy=%s",
            self.strategy_name,
        )

    async def on_context_event(self, event: Event) -> None:
        """
        EventBus handler.

        Не містить торгової логіки:
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
                getattr(event, "event_id", None),
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

        За замовчуванням healthcheck вимкнений, бо strategy evaluator зазвичай
        не потребує background задач.
        """
        if self.scheduler is None:
            return

        if not self.event_config.scheduler_healthcheck_enabled:
            return

        job_name = f"{self.strategy_name}.healthcheck"

        existing = self.scheduler.get_job_by_name(job_name)
        if existing is not None:
            self._healthcheck_job_id = existing.job_id
            return

        self._healthcheck_job_id = self.scheduler.add_interval_job(
            name=job_name,
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

    def _remove_scheduler_job(self) -> None:
        if self.scheduler is None or self._healthcheck_job_id is None:
            return

        try:
            self.scheduler.remove_job(self._healthcheck_job_id)
        except Exception:
            self._logger.debug(
                "Failed to remove whale strategy scheduler job | strategy=%s job_id=%s",
                self.strategy_name,
                self._healthcheck_job_id,
                exc_info=True,
            )
        finally:
            self._healthcheck_job_id = None

    def _unsubscribe(self) -> None:
        if self._subscription is None:
            return

        try:
            if hasattr(self._subscription, "unsubscribe"):
                self._subscription.unsubscribe()
            elif self.event_bus is not None and hasattr(self.event_bus, "unsubscribe"):
                self.event_bus.unsubscribe(self._subscription)
        except Exception:
            self._logger.debug(
                "Failed to unsubscribe whale strategy | strategy=%s",
                self.strategy_name,
                exc_info=True,
            )
        finally:
            self._subscription = None

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
                "event_config": {
                    "validate_scope_enabled": self.event_config.validate_scope_enabled,
                    "validate_freshness_enabled": self.event_config.validate_freshness_enabled,
                    "validate_futures_market_type_enabled": (
                        self.event_config.validate_futures_market_type_enabled
                    ),
                    "strict_timeframe_scope": self.event_config.strict_timeframe_scope,
                    "whale_feature_max_age_ms": self.event_config.whale_feature_max_age_ms,
                },
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
            "category": (
                signal.category.value
                if hasattr(signal.category, "value")
                else str(signal.category)
            ),
            "timeframe": str(signal.timeframe),
            "timestamp": signal.timestamp,
            "confidence": signal.confidence,
            "score": signal.score,
            "priority": (
                signal.priority.value
                if hasattr(signal.priority, "value")
                else str(signal.priority)
            ),
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
    # Whale input snapshot helpers
    # =========================================================================

    def build_whale_input_snapshot(
        self,
        context: SignalContext,
        *,
        feature_specs: Mapping[str, tuple[str, ...]],
    ) -> WhaleStrategyInputSnapshot:
        """
        Створює normalized whale input snapshot для конкретної стратегії.

        feature_specs example:
            {
                "pressure": (
                    "whale_pressure",
                    "whale_pressure_signal",
                    "analytics.whales.whale_pressure",
                ),
                ...
            }
        """
        snapshot = WhaleStrategyInputSnapshot(
            strategy_name=self.strategy_name,
            symbol=str(context.symbol),
            timestamp_ms=self._context_timestamp_ms(context),
        )

        for canonical_name, aliases in feature_specs.items():
            snapshot.features[canonical_name] = self._resolve_whale_feature(
                context,
                canonical_name=canonical_name,
                names=aliases,
            )

        return snapshot

    def _resolve_payload(
        self,
        context: SignalContext,
        *,
        names: tuple[str, ...],
    ) -> dict[str, Any]:
        """
        Backward-compatible helper для наявних конкретних стратегій.

        Старі WhaleAbsorptionStrategy / WhaleBreakoutStrategy очікують dict.
        Тепер цей dict уже проходить:
        - object -> dict conversion;
        - optional scope validation;
        - optional freshness validation;
        - optional futures market type guard.
        """
        feature = self._resolve_whale_feature(
            context,
            canonical_name=names[0] if names else "unknown",
            names=names,
        )

        if not feature.present:
            return {}

        if not feature.valid and self.event_config.drop_invalid_payloads:
            self._log_payload_validation_warning(feature)
            return {}

        return feature.to_dict()

    def _resolve_whale_feature(
        self,
        context: SignalContext,
        *,
        canonical_name: str,
        names: tuple[str, ...],
    ) -> WhaleFeaturePayload:
        if not names:
            return WhaleFeaturePayload(
                canonical_name=canonical_name,
                aliases=(),
                payload={},
                validation=WhalePayloadValidation(
                    valid=False,
                    present=False,
                    reasons=["No aliases provided"],
                ),
            )

        candidates = self._iter_feature_candidates(context, names)

        for source_name, source_location, raw_value in candidates:
            payload = self._object_to_dict(raw_value)
            if not payload:
                continue

            validation = self._validate_whale_payload(
                context=context,
                payload=payload,
                source_name=source_name,
                source_location=source_location,
            )

            feature = WhaleFeaturePayload(
                canonical_name=canonical_name,
                aliases=names,
                payload=payload,
                source_name=source_name,
                source_location=source_location,
                validation=validation,
            )

            if validation.valid or not self.event_config.drop_invalid_payloads:
                return feature

            self._log_payload_validation_warning(feature)

        return WhaleFeaturePayload(
            canonical_name=canonical_name,
            aliases=names,
            payload={},
            validation=WhalePayloadValidation(
                valid=False,
                present=False,
                reasons=["Feature payload not found"],
            ),
        )

    def _iter_feature_candidates(
        self,
        context: SignalContext,
        names: tuple[str, ...],
    ) -> list[tuple[str, str, Any]]:
        candidates: list[tuple[str, str, Any]] = []

        whales = getattr(context, "whales", None)
        if isinstance(whales, Mapping):
            for name in names:
                if name in whales:
                    candidates.append((name, "context.whales", whales[name]))

        for name in names:
            try:
                feature_value = context.get_feature(name)
            except Exception:
                feature_value = None

            if feature_value is not None:
                candidates.append((name, "context.feature_map", feature_value))

            try:
                snapshot = context.get_feature_snapshot(name)
            except Exception:
                snapshot = None

            if snapshot is not None:
                candidates.append((name, "context.feature_snapshot", snapshot.value))

        return candidates

    def _validate_whale_payload(
        self,
        *,
        context: SignalContext,
        payload: Mapping[str, Any],
        source_name: str,
        source_location: str,
    ) -> WhalePayloadValidation:
        validation = WhalePayloadValidation(
            valid=True,
            present=bool(payload),
        )

        if not payload:
            validation.valid = False
            validation.present = False
            validation.add_reason("Payload is empty")
            return validation

        scope = self._extract_payload_scope(payload)

        if self.event_config.require_payload_scope:
            required_scope_fields = ("symbol", "market_type")
            missing = [
                field_name
                for field_name in required_scope_fields
                if not scope.get(field_name)
            ]

            if missing:
                validation.scope_valid = False
                validation.add_reason(
                    f"Payload scope is missing required fields: {', '.join(missing)}"
                )

        if self.event_config.validate_scope_enabled:
            scope_valid, scope_reason = self._is_payload_scope_valid(
                context=context,
                payload_scope=scope,
            )
            validation.scope_valid = scope_valid
            if not scope_valid:
                validation.add_reason(scope_reason)

        if self.event_config.validate_freshness_enabled:
            fresh, freshness_reason = self._is_payload_fresh(
                context=context,
                payload=payload,
            )
            validation.fresh = fresh
            if not fresh:
                validation.add_reason(freshness_reason)

        if self.event_config.validate_futures_market_type_enabled:
            futures_valid, futures_reason = self._is_futures_market_type_payload(
                context=context,
                payload_scope=scope,
            )
            validation.futures_market_type_valid = futures_valid
            if not futures_valid:
                validation.add_reason(futures_reason)

        validation.valid = (
            validation.present
            and validation.scope_valid
            and validation.fresh
            and validation.futures_market_type_valid
        )

        if not validation.valid:
            validation.add_reason(
                f"Invalid whale payload source={source_location}.{source_name}"
            )

        return validation

    def _extract_payload_scope(self, payload: Mapping[str, Any]) -> dict[str, str | None]:
        nested_scope = payload.get("scope")
        scope_mapping = nested_scope if isinstance(nested_scope, Mapping) else {}

        exchange = payload.get("exchange", scope_mapping.get("exchange"))
        market_type = payload.get("market_type", scope_mapping.get("market_type"))
        symbol = payload.get("symbol", scope_mapping.get("symbol"))
        timeframe = payload.get("timeframe", scope_mapping.get("timeframe"))
        exchange_symbol = payload.get(
            "exchange_symbol",
            scope_mapping.get("exchange_symbol"),
        )

        return {
            "exchange": self._normalize_exchange(exchange),
            "market_type": self._normalize_market_type(market_type),
            "symbol": self._normalize_symbol_or_none(symbol),
            "timeframe": self._normalize_timeframe(timeframe),
            "exchange_symbol": (
                str(exchange_symbol).strip()
                if exchange_symbol is not None and str(exchange_symbol).strip()
                else None
            ),
        }

    def _is_payload_scope_valid(
        self,
        *,
        context: SignalContext,
        payload_scope: Mapping[str, str | None],
    ) -> tuple[bool, str]:
        context_scope = self._extract_context_scope(context)

        # Symbol mismatch is the most dangerous. If payload has symbol, it must
        # match context.symbol.
        payload_symbol = payload_scope.get("symbol")
        context_symbol = context_scope.get("symbol")
        if payload_symbol and context_symbol and payload_symbol != context_symbol:
            return (
                False,
                f"Whale payload symbol mismatch: payload={payload_symbol} context={context_symbol}",
            )

        payload_exchange = payload_scope.get("exchange")
        context_exchange = context_scope.get("exchange")
        if payload_exchange and context_exchange and payload_exchange != context_exchange:
            return (
                False,
                f"Whale payload exchange mismatch: payload={payload_exchange} context={context_exchange}",
            )

        payload_market_type = payload_scope.get("market_type")
        context_market_type = context_scope.get("market_type")
        if (
            payload_market_type
            and context_market_type
            and payload_market_type != context_market_type
        ):
            return (
                False,
                "Whale payload market_type mismatch: "
                f"payload={payload_market_type} context={context_market_type}",
            )

        if self.event_config.strict_timeframe_scope:
            payload_timeframe = payload_scope.get("timeframe")
            context_timeframe = context_scope.get("timeframe")
            if payload_timeframe and context_timeframe and payload_timeframe != context_timeframe:
                return (
                    False,
                    "Whale payload timeframe mismatch: "
                    f"payload={payload_timeframe} context={context_timeframe}",
                )

        return True, "Scope valid"

    def _is_payload_fresh(
        self,
        *,
        context: SignalContext,
        payload: Mapping[str, Any],
    ) -> tuple[bool, str]:
        payload_ts_ms = self._payload_timestamp_ms(payload)
        context_ts_ms = self._context_timestamp_ms(context)

        if payload_ts_ms is None or context_ts_ms is None:
            return True, "Freshness skipped: timestamp unavailable"

        age_ms = max(0, context_ts_ms - payload_ts_ms)
        max_age_ms = max(1, int(self.event_config.whale_feature_max_age_ms))

        if age_ms > max_age_ms:
            return (
                False,
                f"Whale payload stale: age_ms={age_ms} max_age_ms={max_age_ms}",
            )

        return True, "Freshness valid"

    def _is_futures_market_type_payload(
        self,
        *,
        context: SignalContext,
        payload_scope: Mapping[str, str | None],
    ) -> tuple[bool, str]:
        market_type = payload_scope.get("market_type")

        if not market_type:
            context_scope = self._extract_context_scope(context)
            market_type = context_scope.get("market_type")

        if not market_type:
            return True, "Futures market_type guard skipped: market_type unavailable"

        if market_type not in FUTURES_MARKET_TYPES:
            return (
                False,
                f"Non-futures whale payload market_type={market_type}",
            )

        return True, "Futures market_type valid"

    def _extract_context_scope(self, context: SignalContext) -> dict[str, str | None]:
        exchange = self._context_value(context, "exchange")
        market_type = self._context_value(context, "market_type")
        timeframe = self._context_value(context, "timeframe")
        symbol = getattr(context, "symbol", None)

        return {
            "exchange": self._normalize_exchange(exchange),
            "market_type": self._normalize_market_type(market_type),
            "symbol": self._normalize_symbol_or_none(symbol),
            "timeframe": self._normalize_timeframe(timeframe),
        }

    def _context_value(self, context: SignalContext, name: str) -> Any:
        if hasattr(context, name):
            value = getattr(context, name)
            if value is not None:
                return value

        try:
            value = context.get_feature(name)
            if value is not None:
                return value
        except Exception:
            pass

        try:
            snapshot = context.get_feature_snapshot(name)
            if snapshot is not None:
                return snapshot.value
        except Exception:
            pass

        metadata = getattr(context, "metadata", None)
        if isinstance(metadata, Mapping):
            value = metadata.get(name)
            if value is not None:
                return value

        return None

    def _log_payload_validation_warning(self, feature: WhaleFeaturePayload) -> None:
        if feature.validation.valid:
            return

        reason_key = "|".join(feature.validation.reasons) or feature.canonical_name
        now = time.monotonic()
        last_logged = self._last_validation_warnings.get(reason_key, 0.0)

        # Rate-limit noisy validation warnings.
        if now - last_logged < 30.0:
            return

        self._last_validation_warnings[reason_key] = now

        self._logger.warning(
            "Invalid whale feature payload dropped | strategy=%s feature=%s source=%s reasons=%s",
            self.strategy_name,
            feature.canonical_name,
            feature.source_location,
            "; ".join(feature.validation.reasons),
        )

    # =========================================================================
    # Shared payload conversion helpers
    # =========================================================================

    def _object_to_dict(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {}

        if isinstance(value, dict):
            return dict(value)

        if isinstance(value, Mapping):
            return dict(value)

        if hasattr(value, "to_event") and callable(value.to_event):
            try:
                result = value.to_event()
                if isinstance(result, Mapping):
                    return dict(result)
            except Exception:
                self._logger.debug(
                    "Failed to convert object using to_event() | strategy=%s type=%s",
                    self.strategy_name,
                    type(value).__name__,
                    exc_info=True,
                )

        if hasattr(value, "to_payload") and callable(value.to_payload):
            try:
                result = value.to_payload()
                if isinstance(result, Mapping):
                    return dict(result)
            except Exception:
                self._logger.debug(
                    "Failed to convert object using to_payload() | strategy=%s type=%s",
                    self.strategy_name,
                    type(value).__name__,
                    exc_info=True,
                )

        if hasattr(value, "to_dict") and callable(value.to_dict):
            try:
                result = value.to_dict()
                if isinstance(result, Mapping):
                    return dict(result)
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

    # =========================================================================
    # Safe conversion / normalization helpers
    # =========================================================================

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

    def _normalize_exchange(self, value: Any) -> str | None:
        if value is None:
            return None

        normalized = str(value).strip().lower()
        return normalized or None

    def _normalize_market_type(self, value: Any) -> str | None:
        if value is None:
            return None

        normalized = str(value).strip().lower()
        return normalized or None

    def _normalize_timeframe(self, value: Any) -> str | None:
        if value is None:
            return None

        normalized = str(value).strip()
        return normalized or None

    def _normalize_symbol_or_none(self, value: Any) -> str | None:
        if value is None:
            return None

        normalized = (
            str(value)
            .replace("-", "")
            .replace("/", "")
            .replace("_", "")
            .upper()
            .strip()
        )

        return normalized or None

    def _context_timestamp_ms(self, context: SignalContext) -> int | None:
        return self._to_timestamp_ms(getattr(context, "timestamp", None))

    def _payload_timestamp_ms(self, payload: Mapping[str, Any]) -> int | None:
        for field_name in (
            "timestamp_ms",
            "created_at_ms",
            "updated_at_ms",
            "event_time_ms",
            "received_at_ms",
            "last_seen_ms",
            "timestamp",
        ):
            if field_name in payload:
                ts = self._to_timestamp_ms(payload.get(field_name))
                if ts is not None:
                    return ts

        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            for field_name in (
                "timestamp_ms",
                "created_at_ms",
                "updated_at_ms",
                "event_time_ms",
                "received_at_ms",
                "last_seen_ms",
                "timestamp",
            ):
                if field_name in metadata:
                    ts = self._to_timestamp_ms(metadata.get(field_name))
                    if ts is not None:
                        return ts

        return None

    def _to_timestamp_ms(self, value: Any) -> int | None:
        if value is None:
            return None

        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return int(value.timestamp() * 1000)

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None

        if numeric <= 0:
            return None

        # Heuristic:
        # - seconds timestamp ~ 1_700_000_000
        # - milliseconds timestamp ~ 1_700_000_000_000
        if numeric < 10_000_000_000:
            numeric *= 1000

        return int(numeric)

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