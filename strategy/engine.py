# trading_system/strategy/engine.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.event_bus import Event, EventBus, EventPriority
from core.scheduler import Scheduler

from .base import BaseStrategy, BaseStrategyComponent
from .config import StrategyConfig
from .enums import FeatureSource, Timeframe
from .exceptions import StrategyEvaluationError, StrategyStateError
from .models import (
    FeatureSnapshot,
    PortfolioSnapshot,
    PriceSnapshot,
    RegimeSnapshot,
    StrategyContext,
    StrategyEvaluation,
    StrategySignal,
    ensure_aware_utc,
    utcnow,
)
from .processor import ProcessedSignalBatch, SignalProcessor
from .registry import StrategyRegistry
from .state import StrategyRuntimeState


@dataclass(slots=True)
class StrategyEngineStats:
    """
    Lightweight runtime stats for StrategyEngine.
    """

    events_received: int = 0
    events_processed: int = 0
    events_failed: int = 0

    contexts_built: int = 0
    contexts_updated: int = 0

    batches_accepted: int = 0
    batches_rejected: int = 0

    started_at: datetime | None = None
    stopped_at: datetime | None = None
    last_event_at: datetime | None = None
    last_error_at: datetime | None = None
    last_processed_at: datetime | None = None

    errors: list[str] = field(default_factory=list)

    def record_start(self) -> None:
        self.started_at = utcnow()
        self.stopped_at = None

    def record_stop(self) -> None:
        self.stopped_at = utcnow()

    def record_event_received(self) -> None:
        self.events_received += 1
        self.last_event_at = utcnow()

    def record_processed(self, accepted: bool) -> None:
        self.events_processed += 1
        self.last_processed_at = utcnow()

        if accepted:
            self.batches_accepted += 1
        else:
            self.batches_rejected += 1

    def record_error(self, error: Exception | str) -> None:
        self.events_failed += 1
        self.last_error_at = utcnow()

        message = str(error)
        self.errors.append(message)

        if len(self.errors) > 100:
            self.errors = self.errors[-100:]

    def summary(self) -> dict[str, Any]:
        return {
            "events_received": self.events_received,
            "events_processed": self.events_processed,
            "events_failed": self.events_failed,
            "contexts_built": self.contexts_built,
            "contexts_updated": self.contexts_updated,
            "batches_accepted": self.batches_accepted,
            "batches_rejected": self.batches_rejected,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "last_event_at": self.last_event_at,
            "last_error_at": self.last_error_at,
            "last_processed_at": self.last_processed_at,
            "recent_errors": list(self.errors[-10:]),
        }


class StrategyContextBuilder(BaseStrategyComponent):
    """
    Builds and updates StrategyContext objects.

    Це заміна старого strategy/context.py на рівні orchestration.
    Сам StrategyContext тепер живе в models.py.
    """

    component_namespace = "strategy.context_builder"

    def __init__(
        self,
        config: StrategyConfig,
        state: StrategyRuntimeState,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
        )
        self.state = state

    def build(
        self,
        *,
        symbol: str,
        timestamp: datetime | None = None,
        timeframe: Timeframe = Timeframe.M1,
        price: PriceSnapshot | None = None,
        regime: RegimeSnapshot | None = None,
        portfolio: PortfolioSnapshot | None = None,
        features: list[FeatureSnapshot] | None = None,
        domain_data: dict[FeatureSource, dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StrategyContext:
        if not symbol.strip():
            raise StrategyStateError("symbol cannot be empty")

        context = StrategyContext(
            symbol=symbol,
            timestamp=ensure_aware_utc(timestamp or utcnow()),
            timeframe=timeframe,
            price=price,
            regime=regime,
            portfolio=portfolio or self.state.contexts.portfolio,
            metadata=metadata or {},
        )

        if features:
            for snapshot in features:
                context.put_feature(snapshot)

                if snapshot.freshness_seconds is not None:
                    context.freshness_map[snapshot.name] = snapshot.freshness_seconds

        if domain_data:
            for source, values in domain_data.items():
                context.domain_dict(source).update(values)

        context.validate()
        return context

    def get_or_build(
        self,
        *,
        symbol: str,
        timestamp: datetime | None = None,
        timeframe: Timeframe = Timeframe.M1,
    ) -> StrategyContext:
        existing = self.state.contexts.get_context(symbol)

        if existing is not None:
            if timestamp is not None:
                existing.timestamp = ensure_aware_utc(timestamp)
            return existing

        context = self.state.build_context(
            symbol,
            timestamp=timestamp or utcnow(),
            include_regime=True,
            include_portfolio=True,
        )
        context.timeframe = timeframe
        context.validate()
        return context

    def update_from_payload(
        self,
        *,
        event_name: str,
        payload: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> StrategyContext:
        """
        Lightweight direct context update.

        Основна нормалізація analytics payload усе одно живе в SignalProcessor,
        але цей метод корисний для manual/system context updates.
        """
        if not isinstance(payload, dict):
            raise StrategyStateError("payload must be a dict")

        symbol = self._extract_symbol(payload)
        ts = self._extract_timestamp(payload, timestamp)
        timeframe = self._extract_timeframe(payload)

        context = self.get_or_build(
            symbol=symbol,
            timestamp=ts,
            timeframe=timeframe,
        )

        price = self._extract_price_snapshot(symbol=symbol, payload=payload, timestamp=ts)
        if price is not None:
            context.price = price

        regime = self._extract_regime_snapshot(symbol=symbol, payload=payload, timestamp=ts)
        if regime is not None:
            context.regime = regime

        portfolio = payload.get("portfolio")
        if isinstance(portfolio, PortfolioSnapshot):
            context.portfolio = portfolio

        context.metadata.update(
            {
                "source_event": event_name,
                "updated_by": self.component_name,
            }
        )

        context.validate()
        self.state.update_context(context)
        return context

    def persist(self, context: StrategyContext) -> None:
        context.validate()
        self.state.update_context(context)

    def _extract_symbol(self, payload: dict[str, Any]) -> str:
        raw = payload.get("symbol") or payload.get("instrument") or payload.get("market")
        if not isinstance(raw, str) or not raw.strip():
            raise StrategyStateError("payload does not contain valid symbol")
        return raw.strip()

    def _extract_timestamp(
        self,
        payload: dict[str, Any],
        fallback: datetime | None = None,
    ) -> datetime:
        raw = payload.get("timestamp") or payload.get("ts") or fallback

        if raw is None:
            return utcnow()

        if isinstance(raw, datetime):
            return ensure_aware_utc(raw)

        if isinstance(raw, (int, float)):
            if raw > 10_000_000_000:
                return datetime.fromtimestamp(raw / 1000.0).astimezone()
            return datetime.fromtimestamp(raw).astimezone()

        raise StrategyStateError("unsupported timestamp type in payload")

    def _extract_timeframe(self, payload: dict[str, Any]) -> Timeframe:
        raw = payload.get("timeframe")

        if isinstance(raw, Timeframe):
            return raw

        if isinstance(raw, str):
            try:
                return Timeframe(raw)
            except ValueError:
                return Timeframe.M1

        return Timeframe.M1

    def _extract_price_snapshot(
        self,
        *,
        symbol: str,
        payload: dict[str, Any],
        timestamp: datetime,
    ) -> PriceSnapshot | None:
        price_keys = {
            "last_price",
            "price",
            "bid",
            "ask",
            "mark_price",
            "index_price",
            "spread_bps",
        }

        if not any(key in payload for key in price_keys):
            return None

        last_price = payload.get("last_price", payload.get("price"))

        snapshot = PriceSnapshot(
            symbol=symbol,
            last_price=self._optional_float(last_price),
            bid=self._optional_float(payload.get("bid")),
            ask=self._optional_float(payload.get("ask")),
            mark_price=self._optional_float(payload.get("mark_price")),
            index_price=self._optional_float(payload.get("index_price")),
            spread_bps=self._optional_float(payload.get("spread_bps")),
            timestamp=timestamp,
        )
        snapshot.validate()
        return snapshot

    def _extract_regime_snapshot(
        self,
        *,
        symbol: str,
        payload: dict[str, Any],
        timestamp: datetime,
    ) -> RegimeSnapshot | None:
        raw = payload.get("regime") or payload.get("market_regime")
        if raw is None:
            return None

        from .enums import MarketRegime

        if isinstance(raw, MarketRegime):
            regime = raw
        elif isinstance(raw, str):
            try:
                regime = MarketRegime(raw)
            except ValueError:
                regime = MarketRegime.UNKNOWN
        else:
            regime = MarketRegime.UNKNOWN

        confidence = self._optional_float(payload.get("regime_confidence")) or 0.0

        snapshot = RegimeSnapshot(
            symbol=symbol,
            regime=regime,
            confidence=confidence,
            timestamp=timestamp,
            reasons=list(payload.get("regime_reasons") or []),
        )
        snapshot.validate()
        return snapshot

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None


class StrategyEventHandler(BaseStrategyComponent):
    """
    EventBus-facing handler for strategy layer.

    Слухає analytics/context/system events і передає їх у StrategyEngine.
    """

    component_namespace = "strategy.event_handler"

    def __init__(
        self,
        config: StrategyConfig,
        engine: StrategyEngineProtocol,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
        )
        self.engine = engine

    def register(self) -> None:
        bus = self.ensure_event_bus()

        bus.subscribe("analytics.*", self.on_analytics_event)
        bus.subscribe("strategy.context.update", self.on_context_update)
        bus.subscribe("strategy.command.evaluate_symbol", self.on_evaluate_symbol)
        bus.subscribe("strategy.command.prune", self.on_prune_command)

        self._registered = True

    async def on_analytics_event(self, event: Event) -> None:
        event_name = self._event_topic(event)
        payload = self._event_payload(event)

        await self.engine.process_analytics_event(
            event_name=event_name,
            payload=payload,
            event=event,
        )

    async def on_context_update(self, event: Event) -> None:
        event_name = self._event_topic(event)
        payload = self._event_payload(event)

        self.engine.update_context_from_payload(
            event_name=event_name,
            payload=payload,
        )

    async def on_evaluate_symbol(self, event: Event) -> None:
        payload = self._event_payload(event)

        symbol = payload.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            raise StrategyEvaluationError("strategy.command.evaluate_symbol requires symbol")

        await self.engine.evaluate_symbol(symbol.strip())

    async def on_prune_command(self, event: Event) -> None:
        self.engine.prune()

    def _event_topic(self, event: Event) -> str:
        return (
            getattr(event, "topic", None)
            or getattr(event, "name", None)
            or getattr(event, "event_name", None)
            or "analytics.unknown"
        )

    def _event_payload(self, event: Event) -> dict[str, Any]:
        payload = getattr(event, "payload", None)

        if payload is None:
            return {}

        if not isinstance(payload, dict):
            raise StrategyEvaluationError("Event payload must be a dict")

        return payload


class StrategyLifecycleManager(BaseStrategyComponent):
    """
    Lifecycle manager for strategy package.

    Відповідає за:
    - start/stop дочірніх компонентів;
    - Scheduler cleanup jobs;
    - lifecycle events.
    """

    component_namespace = "strategy.lifecycle"

    def __init__(
        self,
        config: StrategyConfig,
        state: StrategyRuntimeState,
        components: list[BaseStrategyComponent],
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
        )
        self.state = state
        self.components = components
        self._cleanup_job_name = "strategy_runtime_state_cleanup"

    async def start(self) -> None:
        if self.is_started:
            return

        for component in self.components:
            await component.start()

        self._schedule_cleanup_job()

        await super().start()

        await self.emit_event(
            "strategy.engine.started",
            {
                "component": self.component_name,
                "state": self.state.summary(),
            },
            priority=EventPriority.LOW,
            source=self.component_name,
        )

    async def stop(self) -> None:
        if not self.is_started:
            return

        await self.emit_event(
            "strategy.engine.stopping",
            {
                "component": self.component_name,
                "state": self.state.summary(),
            },
            priority=EventPriority.LOW,
            source=self.component_name,
        )

        for component in reversed(self.components):
            try:
                await component.stop()
            except Exception as exc:
                self.log_warning(
                    "Component stop failed",
                    component_name=component.component_name,
                    error=str(exc),
                )

        await super().stop()

        await self.emit_event(
            "strategy.engine.stopped",
            {
                "component": self.component_name,
            },
            priority=EventPriority.LOW,
            source=self.component_name,
        )

    def _schedule_cleanup_job(self) -> None:
        if self.scheduler is None:
            return

        cleanup_interval = max(
            5,
            int(self.config.routing.stale_feature_threshold_seconds),
        )

        try:
            self.scheduler.add_interval_job(
                name=self._cleanup_job_name,
                func=self._cleanup_state_job,
                interval_seconds=cleanup_interval,
                run_immediately=False,
            )
        except TypeError:
            # Fallback for Scheduler versions with a smaller signature.
            self.scheduler.add_interval_job(
                self._cleanup_job_name,
                self._cleanup_state_job,
                cleanup_interval,
            )

    async def _cleanup_state_job(self) -> None:
        removed = self.state.prune(
            max_signal_age_seconds=self.config.runtime.max_signal_age_seconds,
        )

        self.log_debug(
            "Strategy runtime state pruned",
            **removed,
        )


class StrategyEngineProtocol:
    """
    Lightweight protocol-like base for StrategyEventHandler typing.

    Не використовує typing.Protocol, щоб не ускладнювати runtime imports.
    """

    async def process_analytics_event(
        self,
        *,
        event_name: str,
        payload: dict[str, Any],
        event: Event | None = None,
    ) -> ProcessedSignalBatch:
        raise NotImplementedError

    def update_context_from_payload(
        self,
        *,
        event_name: str,
        payload: dict[str, Any],
    ) -> StrategyContext:
        raise NotImplementedError

    async def evaluate_symbol(self, symbol: str) -> list[StrategyEvaluation]:
        raise NotImplementedError

    def prune(self) -> dict[str, int]:
        raise NotImplementedError


class StrategyEngine(BaseStrategyComponent, StrategyEngineProtocol):
    """
    Main orchestration layer for strategy package.

    StrategyEngine:
    - owns StrategyRegistry;
    - owns StrategyRuntimeState;
    - owns SignalProcessor;
    - subscribes to analytics events through StrategyEventHandler;
    - delegates signal pipeline to SignalProcessor;
    - emits strategy/system events through EventBus;
    - does not contain trading execution or risk logic.
    """

    component_namespace = "strategy.engine"

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        registry: StrategyRegistry | None = None,
        state: StrategyRuntimeState | None = None,
        processor: SignalProcessor | None = None,
    ) -> None:
        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
        )

        self.state = state or StrategyRuntimeState()

        self.registry = registry or StrategyRegistry(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
        )

        self.context_builder = StrategyContextBuilder(
            config=config,
            state=self.state,
            event_bus=event_bus,
            scheduler=scheduler,
        )

        self.processor = processor or SignalProcessor(
            config=config,
            registry=self.registry,
            state=self.state,
            event_bus=event_bus,
            scheduler=scheduler,
        )

        self.event_handler = StrategyEventHandler(
            config=config,
            engine=self,
            event_bus=event_bus,
            scheduler=scheduler,
        )

        self.lifecycle = StrategyLifecycleManager(
            config=config,
            state=self.state,
            components=[
                self.registry,
                self.context_builder,
                self.processor,
                self.event_handler,
            ],
            event_bus=event_bus,
            scheduler=scheduler,
        )

        self.stats = StrategyEngineStats()

    def register(self) -> None:
        """
        StrategyEngine itself does not subscribe directly.
        Event subscriptions are owned by StrategyEventHandler.
        """
        self.event_handler.register()
        self._registered = True

    async def start(self) -> None:
        if self.is_started:
            return

        if not self.is_registered:
            self.register()

        self.stats.record_start()
        await self.lifecycle.start()

        self._started = True

        self.log_info(
            "StrategyEngine started",
            strategies=self.registry.count(),
        )

    async def stop(self) -> None:
        if not self.is_started:
            return

        await self.lifecycle.stop()

        self.stats.record_stop()
        self._started = False

        self.log_info("StrategyEngine stopped")

    def add_strategy(
        self,
        strategy: BaseStrategy,
        *,
        replace: bool = False,
    ) -> None:
        self.registry.register_strategy(strategy, replace=replace)

    def add_strategies(
        self,
        strategies: list[BaseStrategy],
        *,
        replace: bool = False,
    ) -> None:
        self.registry.register_many(strategies, replace=replace)

    def remove_strategy(self, strategy_name: str) -> BaseStrategy:
        return self.registry.unregister_strategy(strategy_name)

    async def process_analytics_event(
        self,
        *,
        event_name: str,
        payload: dict[str, Any],
        event: Event | None = None,
    ) -> ProcessedSignalBatch:
        self.stats.record_event_received()

        try:
            batch = await self.processor.process_event(
                event_name=event_name,
                payload=payload,
                timestamp=self._event_timestamp(event),
            )

            self.stats.record_processed(batch.accepted)

            await self.emit_event(
                "strategy.engine.batch_processed",
                {
                    "symbol": batch.symbol,
                    "accepted": batch.accepted,
                    "final_signal_count": len(batch.final_signals),
                    "reasons": batch.reasons,
                    "timestamp": batch.timestamp.isoformat(),
                },
                priority=EventPriority.LOW,
                source=self.component_name,
            )

            return batch

        except Exception as exc:
            self.stats.record_error(exc)
            self.state.metrics.record_error()

            self.log_exception(
                "StrategyEngine failed to process analytics event",
                event_name=event_name,
                error=str(exc),
            )

            await self.emit_event(
                "strategy.engine.error",
                {
                    "event_name": event_name,
                    "error": str(exc),
                    "payload_symbol": payload.get("symbol"),
                    "timestamp": utcnow().isoformat(),
                },
                priority=EventPriority.HIGH,
                source=self.component_name,
            )

            raise

    def update_context_from_payload(
        self,
        *,
        event_name: str,
        payload: dict[str, Any],
    ) -> StrategyContext:
        context = self.context_builder.update_from_payload(
            event_name=event_name,
            payload=payload,
        )
        self.stats.contexts_updated += 1
        return context

    def build_context(
        self,
        *,
        symbol: str,
        timestamp: datetime | None = None,
        timeframe: Timeframe = Timeframe.M1,
        price: PriceSnapshot | None = None,
        regime: RegimeSnapshot | None = None,
        portfolio: PortfolioSnapshot | None = None,
        features: list[FeatureSnapshot] | None = None,
        domain_data: dict[FeatureSource, dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> StrategyContext:
        context = self.context_builder.build(
            symbol=symbol,
            timestamp=timestamp,
            timeframe=timeframe,
            price=price,
            regime=regime,
            portfolio=portfolio,
            features=features,
            domain_data=domain_data,
            metadata=metadata,
        )

        self.stats.contexts_built += 1

        if persist:
            self.context_builder.persist(context)
            self.stats.contexts_updated += 1

        return context

    async def evaluate_symbol(self, symbol: str) -> list[StrategyEvaluation]:
        if not symbol.strip():
            raise StrategyEvaluationError("symbol cannot be empty")

        context = self.state.contexts.get_context(symbol)
        if context is None:
            context = self.state.build_context(symbol)

        strategies = self.registry.find_applicable(context)

        if not strategies:
            self.state.metrics.record_applicability_skip()
            return []

        evaluations = await self.processor.evaluate_strategies(
            strategies=strategies,
            context=context,
        )

        return evaluations

    async def evaluate_context(
        self,
        context: StrategyContext,
    ) -> list[StrategyEvaluation]:
        context.validate()
        self.state.update_context(context)

        strategies = self.registry.find_applicable(context)

        if not strategies:
            self.state.metrics.record_applicability_skip()
            return []

        return await self.processor.evaluate_strategies(
            strategies=strategies,
            context=context,
        )

    def update_price(self, snapshot: PriceSnapshot) -> None:
        snapshot.validate()

        context = self.context_builder.get_or_build(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
        )
        context.price = snapshot
        self.state.update_context(context)

    def update_regime(self, snapshot: RegimeSnapshot) -> None:
        snapshot.validate()
        self.state.set_regime(snapshot)

        context = self.context_builder.get_or_build(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
        )
        context.regime = snapshot
        self.state.update_context(context)

    def update_portfolio(self, snapshot: PortfolioSnapshot) -> None:
        snapshot.validate()
        self.state.set_portfolio_snapshot(snapshot)

    def upsert_feature(self, snapshot: FeatureSnapshot) -> None:
        snapshot.validate()

        context = self.context_builder.get_or_build(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
        )
        context.put_feature(snapshot)

        if snapshot.freshness_seconds is not None:
            context.freshness_map[snapshot.name] = snapshot.freshness_seconds

        self.state.update_context(context)

    def record_signal(
        self,
        signal: StrategySignal,
        *,
        active: bool | None = None,
    ) -> None:
        signal.validate()
        self.state.update_signal(signal, active=active)

    def prune(self) -> dict[str, int]:
        result = self.state.prune(
            max_signal_age_seconds=self.config.runtime.max_signal_age_seconds,
        )

        self.log_debug(
            "StrategyEngine state pruned",
            **result,
        )

        return result

    def summary(self) -> dict[str, Any]:
        return {
            "component": self.component_name,
            "started": self.is_started,
            "registered": self.is_registered,
            "registry": self.registry.summary(),
            "state": self.state.summary(),
            "stats": self.stats.summary(),
        }

    def _event_timestamp(self, event: Event | None) -> datetime | None:
        if event is None:
            return None

        raw = getattr(event, "timestamp", None)

        if raw is None:
            return None

        if isinstance(raw, datetime):
            return ensure_aware_utc(raw)

        return None