# trading_system/strategy/engine.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.event_bus import Event, EventBus, EventPriority
from core.scheduler import Scheduler

from strategy.base import BaseStrategy, BaseStrategyComponent
from strategy.config import StrategyConfig
from strategy.enums import FeatureSource, SignalStatus, Timeframe, MarketRegime
from strategy.exceptions import StrategyEvaluationError, StrategyStateError
from strategy.models import (
    FeatureSnapshot,
    PortfolioSnapshot,
    PriceSnapshot,
    RegimeSnapshot,
    StrategyContext,
    StrategyEvaluation,
    ensure_aware_utc,
    utcnow,
    clamp,
)
from strategy.processor import ProcessedSignalBatch, SignalProcessor
from strategy.registry import StrategyRegistry
from strategy.state import StrategyRuntimeState


def _payload_from_event(event: Event | Any) -> dict[str, Any]:
    payload = getattr(event, "payload", None)
    return payload if isinstance(payload, dict) else {}


def _event_name_from_event(event: Event | Any) -> str:
    for attr in ("topic", "name", "event_name", "type"):
        value = getattr(event, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _event_timestamp(event: Event | Any | None) -> datetime | None:
    if event is None:
        return None

    raw = getattr(event, "timestamp", None)
    if raw is None:
        raw = getattr(event, "created_at", None)

    if raw is None:
        return None

    if isinstance(raw, datetime):
        return ensure_aware_utc(raw)

    if isinstance(raw, (int, float)):
        if raw > 10_000_000_000:
            return datetime.fromtimestamp(raw / 1000.0, tz=timezone.utc)
        return datetime.fromtimestamp(raw, tz=timezone.utc)

    return None


@dataclass(slots=True)
class StrategyEngineStats:
    """
    Lightweight runtime stats for StrategyEngine.

    Це тільки engine-level лічильники. Детальна signal/scoring/build/routing
    статистика має залишатися в SignalProcessor/StrategyRuntimeState.
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
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "last_event_at": self.last_event_at.isoformat() if self.last_event_at else None,
            "last_error_at": self.last_error_at.isoformat() if self.last_error_at else None,
            "last_processed_at": self.last_processed_at.isoformat() if self.last_processed_at else None,
            "recent_errors": list(self.errors[-10:]),
        }


class StrategyContextBuilder(BaseStrategyComponent):
    """
    Builds and updates StrategyContext objects.

    Це orchestration helper. Він не запускає стратегії, не scoring-ить сигнали
    і не формує risk payload.
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
        persist: bool = True,
    ) -> StrategyContext:
        if not symbol.strip():
            raise StrategyStateError("symbol cannot be empty")

        context = StrategyContext(
            symbol=symbol.strip(),
            timestamp=ensure_aware_utc(timestamp or utcnow()),
            timeframe=timeframe,
            price=price,
            regime=regime,
            portfolio=portfolio or self.state.contexts.portfolio,
            metadata=dict(metadata or {}),
        )

        if features:
            for snapshot in features:
                snapshot.validate()
                context.put_feature(snapshot)

                if snapshot.freshness_seconds is not None:
                    context.freshness_map[snapshot.name] = snapshot.freshness_seconds

        if domain_data:
            for source, values in domain_data.items():
                context.domain_dict(source).update(dict(values))

        context.validate()

        if persist:
            self.state.update_context(context)

        return context

    def get_or_build(
        self,
        *,
        symbol: str,
        timestamp: datetime | None = None,
        timeframe: Timeframe = Timeframe.M1,
    ) -> StrategyContext:
        if not symbol.strip():
            raise StrategyStateError("symbol cannot be empty")

        existing = self.state.contexts.get_context(symbol.strip())

        if existing is not None:
            if timestamp is not None:
                existing.timestamp = ensure_aware_utc(timestamp)
            existing.timeframe = timeframe
            existing.validate()
            return existing

        context = self.state.build_context(
            symbol.strip(),
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
        Lightweight manual/system context update.

        Основна analytics normalization лишається в SignalProcessor.
        """
        if not event_name.strip():
            raise StrategyStateError("event_name cannot be empty")

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

        price = self._extract_price_snapshot(
            symbol=symbol,
            payload=payload,
            timestamp=ts,
        )
        if price is not None:
            context.price = price

        regime = self._extract_regime_snapshot(
            symbol=symbol,
            payload=payload,
            timestamp=ts,
        )
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

    @staticmethod
    def _extract_symbol(payload: dict[str, Any]) -> str:
        raw = payload.get("symbol") or payload.get("instrument") or payload.get("market")
        if not isinstance(raw, str) or not raw.strip():
            raise StrategyStateError("payload does not contain valid symbol")
        return raw.strip()

    @staticmethod
    def _extract_timestamp(
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
                return datetime.fromtimestamp(raw / 1000.0, tz=timezone.utc)
            return datetime.fromtimestamp(raw, tz=timezone.utc)

        raise StrategyStateError("unsupported timestamp type in payload")

    @staticmethod
    def _extract_timeframe(payload: dict[str, Any]) -> Timeframe:
        raw = payload.get("timeframe")

        if isinstance(raw, Timeframe):
            return raw

        if isinstance(raw, str):
            try:
                return Timeframe(raw)
            except ValueError:
                return Timeframe.M1

        return Timeframe.M1

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        if value is None:
            return None

        if isinstance(value, bool):
            return float(value)

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None

        return None

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
            "close",
            "bid",
            "ask",
            "mark_price",
            "index_price",
            "spread_bps",
        }

        if not any(key in payload for key in price_keys):
            return None

        last = (
            self._float_or_none(payload.get("last_price"))
            or self._float_or_none(payload.get("price"))
            or self._float_or_none(payload.get("close"))
            or self._float_or_none(payload.get("mark_price"))
        )

        bid = self._float_or_none(payload.get("bid"))
        ask = self._float_or_none(payload.get("ask"))
        mark_price = self._float_or_none(payload.get("mark_price"))
        index_price = self._float_or_none(payload.get("index_price"))
        spread_bps = self._float_or_none(payload.get("spread_bps"))

        snapshot = PriceSnapshot(
            symbol=symbol,
            timestamp=timestamp,
            last_price=last,
            bid=bid,
            ask=ask,
            mark_price=mark_price,
            index_price=index_price,
            spread_bps=spread_bps,
            metadata={
                "source": self.component_name,
            },
        )
        snapshot.validate()
        return snapshot

    @staticmethod
    def _extract_regime_snapshot(
            *,
            symbol: str,
            payload: dict[str, Any],
            timestamp: datetime,
    ) -> RegimeSnapshot | None:
        raw = payload.get("regime") or payload.get("market_regime")
        if raw is None:
            return None

        if isinstance(raw, RegimeSnapshot):
            return raw

        regime = (
            raw
            if isinstance(raw, MarketRegime)
            else MarketRegime.safe_parse(raw, MarketRegime.UNKNOWN)
        )

        raw_confidence = payload.get("regime_confidence", 0.0)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        snapshot = RegimeSnapshot(
            symbol=symbol,
            timestamp=timestamp,
            regime=regime,
            confidence=clamp(confidence, 0.0, 1.0),
            metadata={
                "source": "payload",
                "raw_regime": raw if isinstance(raw, str) else str(raw),
            },
        )
        snapshot.validate()
        return snapshot


class StrategyEventHandler(BaseStrategyComponent):
    """
    Owns EventBus subscriptions for StrategyEngine.

    Він тільки приймає events і делегує в StrategyEngine. Не містить scoring,
    building, confluence, risk-payload або execution logic.
    """

    component_namespace = "strategy.event_handler"

    NON_TRADING_ANALYTICS_TOPIC_PARTS: tuple[str, ...] = (
        ".heartbeat",
        ".metrics",
        ".state_cleaned",
        ".cleanup",
        ".cleaned",
        ".stats",
        ".health",
        ".diagnostics",
    )

    def __init__(
        self,
        config: StrategyConfig,
        engine: "StrategyEngineProtocol",
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
        )
        self.engine = engine
        self._subscriptions: list[Any] = []

    @classmethod
    def is_non_trading_analytics_topic(cls, topic: str) -> bool:
        """
        Return True for analytics lifecycle/diagnostic topics that must not
        enter SignalProcessor.

        Examples:
        - analytics.funding.analyzer.heartbeat
        - analytics.oi.metrics
        - analytics.oi.state_cleaned

        These events are useful for monitoring, but they usually do not contain
        symbol/timeframe/trading features, тому SignalNormalizer не має їх
        обробляти як strategy-routable analytics payload.
        """
        normalized = str(topic or "").strip().lower()
        if not normalized:
            return True

        return any(
            marker in normalized
            for marker in cls.NON_TRADING_ANALYTICS_TOPIC_PARTS
        )

    def register(self) -> None:
        if self.event_bus is None:
            self.log_warning(
                "Cannot register StrategyEventHandler: event_bus is not configured"
            )
            self._registered = True
            return

        if self._subscriptions:
            self._registered = True
            return

        topics = self._analytics_topics()
        for topic in topics:
            subscription = self.event_bus.subscribe(
                topic,
                self._handle_analytics_event,
                name=f"strategy_on_{topic.replace('*', 'wildcard').replace('.', '_')}",
            )
            self._subscriptions.append(subscription)

        # Lifecycle/status feedback. These handlers only update local signal state.
        feedback_topics = {
            "signal.confirmed": self._handle_signal_confirmed,
            "risk.position_blocked": self._handle_signal_blocked,
            "risk.limit_warning": self._handle_risk_limit_warning,
            "risk.size_adjusted": self._handle_risk_size_adjusted,
            "risk.kill_switch": self._handle_risk_kill_switch,
            "risk.trading_halted": self._handle_risk_trading_halted,
            "risk.trading_resumed": self._handle_risk_trading_resumed,
            "execution.order_rejected": self._handle_execution_rejected,
            "execution.order_failed": self._handle_execution_failed,
            "execution.order_filled": self._handle_execution_filled,
            "execution.order_cancelled": self._handle_execution_cancelled,
            "system.strategy.context_update": self._handle_context_update,
        }

        for topic, handler in feedback_topics.items():
            subscription = self.event_bus.subscribe(
                topic,
                handler,
                name=f"strategy_on_{topic.replace('.', '_')}",
            )
            self._subscriptions.append(subscription)

        self._registered = True

        self.log_info(
            "StrategyEventHandler registered",
            subscriptions=len(self._subscriptions),
        )

    def unregister(self) -> None:
        if self.event_bus is None:
            self._subscriptions.clear()
            self._registered = False
            return

        for subscription in list(self._subscriptions):
            try:
                self.event_bus.unsubscribe(subscription)
            except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
                self.log_warning(
                    "Failed to unsubscribe strategy event handler",
                    error=str(exc),
                )

        self._subscriptions.clear()
        self._registered = False

    async def stop(self) -> None:
        self.unregister()
        await super().stop()

    async def _handle_analytics_event(self, event: Event | Any) -> None:
        event_name = _event_name_from_event(event)

        if self.is_non_trading_analytics_topic(event_name):
            self.log_debug(
                "Skipping non-trading analytics event",
                topic=event_name,
            )
            return

        payload = _payload_from_event(event)

        await self.engine.process_analytics_event(
            event_name=event_name,
            payload=payload,
            event=event,
        )

    async def _handle_context_update(self, event: Event | Any) -> None:
        event_name = _event_name_from_event(event)
        payload = _payload_from_event(event)

        self.engine.update_context_from_payload(
            event_name=event_name,
            payload=payload,
        )

    async def _handle_signal_confirmed(self, event: Event | Any) -> None:
        payload = _payload_from_event(event)
        self._mark_signal_status(
            payload=payload,
            status=SignalStatus.CONFIRMED,
            reason=payload.get("reason") or "risk_confirmed",
        )

    async def _handle_signal_blocked(self, event: Event | Any) -> None:
        payload = _payload_from_event(event)
        self._mark_signal_status(
            payload=payload,
            status=SignalStatus.REJECTED,
            reason=payload.get("reason") or "risk_position_blocked",
        )

    async def _handle_risk_limit_warning(self, event: Event | Any) -> None:
        payload = _payload_from_event(event)
        self._mark_signal_status(
            payload=payload,
            status=SignalStatus.CONFIRMED,
            reason=payload.get("reason") or "risk_limit_warning",
        )

    async def _handle_risk_size_adjusted(self, event: Event | Any) -> None:
        payload = _payload_from_event(event)
        self._mark_signal_status(
            payload=payload,
            status=SignalStatus.CONFIRMED,
            reason=payload.get("reason") or "risk_size_adjusted",
        )

    async def _handle_risk_kill_switch(self, event: Event | Any) -> None:
        payload = _payload_from_event(event)
        state = getattr(self.engine, "state", None)
        if state is not None:
            set_halt = getattr(state, "set_risk_halt", None)
            if callable(set_halt):
                set_halt(active=True, reason=payload.get("reason") or "risk_kill_switch")

        self._mark_signal_status(
            payload=payload,
            status=SignalStatus.REJECTED,
            reason=payload.get("reason") or "risk_kill_switch",
        )

    async def _handle_risk_trading_halted(self, event: Event | Any) -> None:
        payload = _payload_from_event(event)
        state = getattr(self.engine, "state", None)
        if state is not None:
            set_halt = getattr(state, "set_risk_halt", None)
            if callable(set_halt):
                set_halt(active=True, reason=payload.get("reason") or "risk_trading_halted")

    async def _handle_risk_trading_resumed(self, event: Event | Any) -> None:
        payload = _payload_from_event(event)
        state = getattr(self.engine, "state", None)
        if state is not None:
            set_halt = getattr(state, "set_risk_halt", None)
            if callable(set_halt):
                set_halt(active=False, reason=payload.get("reason") or "risk_trading_resumed")

    async def _handle_execution_rejected(self, event: Event | Any) -> None:
        payload = _payload_from_event(event)
        self._mark_signal_status(
            payload=payload,
            status=SignalStatus.REJECTED,
            reason=payload.get("reason") or "execution_order_rejected",
        )

    async def _handle_execution_failed(self, event: Event | Any) -> None:
        payload = _payload_from_event(event)
        self._mark_signal_status(
            payload=payload,
            status=SignalStatus.FAILED,
            reason=payload.get("reason") or "execution_order_failed",
        )

    async def _handle_execution_cancelled(self, event: Event | Any) -> None:
        payload = _payload_from_event(event)
        self._mark_signal_status(
            payload=payload,
            status=SignalStatus.CANCELLED,
            reason=payload.get("reason") or "execution_order_cancelled",
        )

    async def _handle_execution_filled(self, event: Event | Any) -> None:
        payload = _payload_from_event(event)
        self._mark_signal_status(
            payload=payload,
            status=SignalStatus.EXECUTED,
            reason=payload.get("reason") or "execution_order_filled",
        )

    def _mark_signal_status(
        self,
        *,
        payload: dict[str, Any],
        status: SignalStatus,
        reason: str,
    ) -> None:
        state = getattr(self.engine, "state", None)
        if state is None:
            return

        # Delegate lookup/update to StrategyRuntimeState so downstream events are
        # matched by signal_id first and then by symbol/strategy/side. This avoids
        # accidentally updating the last signal for a symbol when several
        # strategies produced signals for the same instrument.
        marker = getattr(state, "mark_signal_status_from_payload", None)
        if callable(marker):
            marker(
                payload=payload,
                status=status,
                default_reason=reason,
            )

    def _analytics_topics(self) -> list[str]:
        """
        Analytics subscriptions for StrategyEngine.

        Important:
        - event_to_categories is a routing map, not a subscription allowlist;
        - StrategyEngine may listen to analytics.*;
        - non-trading analytics topics are skipped in _handle_analytics_event();
        - SignalProcessor must not receive heartbeat/metrics/cleanup events.
        """
        configured = getattr(self.config.routing, "event_to_categories", {}) or {}

        topics = [
            topic.strip()
            for topic in configured.keys()
            if isinstance(topic, str) and topic.strip()
        ]

        topics.append("analytics.*")

        return list(dict.fromkeys(topics))

class StrategyLifecycleManager(BaseStrategyComponent):
    """
    Starts/stops strategy subcomponents and owns scheduled maintenance jobs.

    Не запускає signal processing самостійно і не містить strategy logic.
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
        self._cleanup_job_name = "strategy.runtime.cleanup"

    async def start(self) -> None:
        if self.is_started:
            return

        for component in self.components:
            if not component.is_registered:
                component.register()

        for component in self.components:
            await component.start()

        self._schedule_cleanup_job()

        await super().start()

        await self.emit_event(
            "strategy.engine.started",
            {
                "component": self.component_name,
                "components": [component.component_name for component in self.components],
            },
            priority=EventPriority.LOW,
            source=self.component_name,
        )

    async def stop(self) -> None:
        if not self.is_started:
            return

        for component in reversed(self.components):
            try:
                await component.stop()
            except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
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
            self.log_debug(
                "Strategy cleanup job skipped because scheduler is not configured"
            )
            return

        cleanup_interval_seconds = max(
            5,
            int(self.config.routing.stale_feature_threshold_seconds),
        )

        job = self.scheduler.add_interval_job(
            name=self._cleanup_job_name,
            func=self._cleanup_state_job,
            interval=cleanup_interval_seconds,
            run_immediately=False,
        )

        self._scheduler_jobs.append(job)

        self.log_info(
            "Strategy cleanup job scheduled",
            job_name=self._cleanup_job_name,
            interval_seconds=cleanup_interval_seconds,
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
    Runtime protocol-like base for StrategyEventHandler typing.

    Не використовує typing.Protocol, щоб не ускладнювати runtime imports.
    """

    state: StrategyRuntimeState

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
    Main orchestration facade for strategy package.

    StrategyEngine:
    - owns StrategyRegistry;
    - owns StrategyRuntimeState;
    - owns StrategyContextBuilder;
    - owns SignalProcessor;
    - owns StrategyEventHandler;
    - owns StrategyLifecycleManager;
    - delegates signal pipeline to SignalProcessor;
    - does not contain scoring/building/risk-payload/execution logic.
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
        """
        Process one analytics event through SignalProcessor.

        Safety guard:
        - StrategyEventHandler normally filters non-trading analytics topics.
        - This method also skips them defensively in case it is called directly.
        """
        self.stats.record_event_received()

        if StrategyEventHandler.is_non_trading_analytics_topic(event_name):
            batch = ProcessedSignalBatch(
                symbol=str(payload.get("symbol") or "unknown"),
                timestamp=_event_timestamp(event) or utcnow(),
                accepted=False,
                emitted=False,
                reasons=[f"skipped_non_trading_analytics_topic:{event_name}"],
                metadata={
                    "event_name": event_name,
                    "skipped": True,
                    "skip_reason": "non_trading_analytics_topic",
                },
            )
            self.stats.record_processed(False)
            self.log_debug(
                "StrategyEngine skipped non-trading analytics event",
                event_name=event_name,
            )
            return batch

        try:
            batch = await self.processor.process_event(
                event_name=event_name,
                payload=payload,
                timestamp=_event_timestamp(event),
                emit=True,
            )

            self.stats.record_processed(batch.accepted)

            await self.emit_event(
                "strategy.engine.batch_processed",
                {
                    "symbol": batch.symbol,
                    "accepted": batch.accepted,
                    "emitted": batch.emitted,
                    "final_signal_count": len(batch.final_signals),
                    "risk_payload_count": len(batch.risk_payloads),
                    "reasons": list(batch.reasons),
                    "timestamp": batch.timestamp.isoformat(),
                },
                priority=EventPriority.LOW,
                source=self.component_name,
            )

            return batch

        except (
            StrategyEvaluationError,
            StrategyStateError,
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
        ) as exc:
            self.stats.record_error(exc)
            self._record_metric_error()

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
            persist=persist,
        )

        self.stats.contexts_built += 1
        return context

    async def evaluate_symbol(self, symbol: str) -> list[StrategyEvaluation]:
        """
        Manual strategy evaluation helper.

        Не публікує signal.generated і не будує risk payload.
        Для повного pipeline використовувати process_analytics_event().
        """
        context = self.context_builder.get_or_build(symbol=symbol)
        strategies = self.registry.select(context=context)

        result: list[StrategyEvaluation] = []

        for strategy in strategies:
            try:
                evaluation = await strategy.evaluate(context)
                evaluation.validate()
                result.append(evaluation)
                self.state.update_evaluation(evaluation)
            except (
                StrategyEvaluationError,
                RuntimeError,
                ValueError,
                TypeError,
                AttributeError,
            ) as exc:
                failed = StrategyEvaluation(
                    strategy_name=strategy.strategy_name,
                    symbol=symbol,
                    timestamp=context.timestamp,
                    passed=False,
                    reasons=[f"manual_evaluation_error:{exc}"],
                    metadata={
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    },
                )
                failed.validate()
                result.append(failed)
                self._record_metric_error(strategy_name=strategy.strategy_name)

        return result

    def prune(self) -> dict[str, int]:
        return self.state.prune(
            max_signal_age_seconds=self.config.runtime.max_signal_age_seconds,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "engine": {
                "started": self.is_started,
                "registered": self.is_registered,
                "stats": self.stats.summary(),
            },
            "registry": self.registry.summary(),
            "state": self.state.summary(),
        }

    def _record_metric_error(self, strategy_name: str | None = None) -> None:
        metrics = getattr(self.state, "metrics", None)
        if metrics is None:
            return

        record_error = getattr(metrics, "record_error", None)
        if not callable(record_error):
            return

        try:
            if strategy_name is not None:
                record_error(strategy_name=strategy_name)
            else:
                record_error()
        except TypeError:
            record_error()

__all__ = [
    "StrategyEngineStats",
    "StrategyContextBuilder",
    "StrategyEventHandler",
    "StrategyLifecycleManager",
    "StrategyEngineProtocol",
    "StrategyEngine",
]