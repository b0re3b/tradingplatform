from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Final

from core.event_bus import EventBus, EventPriority, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler

from .config import BaseSpreadConfig
from .models import SpreadSignal, SpreadSnapshot
from .spread_regime_detector import SpreadRegimeDetector
from .spread_signal_engine import SpreadSignalEngine


class BaseSpreadAnalyzer(ABC):
    """
    Production-grade базовий клас для analytics/spreads analyzer-компонентів.

    Відповідальність:
    - dependency injection через core.EventBus / core.Scheduler / BaseSpreadConfig;
    - register()/unregister() для EventBus subscriptions;
    - lifecycle start()/stop();
    - EventBus.emit() helpers;
    - Scheduler job registration helpers;
    - cooldown/throttling для сигналів і snapshot emit;
    - спільні stats;
    - production-grade logging через core.logger.get_logger;
    - shared SpreadRegimeDetector / SpreadSignalEngine.

    Не відповідає за:
    - конкретну бізнес-логіку spread-аналізу;
    - побудову конкретних SpreadSnapshot;
    - специфічну обробку market.quote / market.funding payload;
    - пряме execution/risk/strategy управління.
    """

    DEFAULT_SNAPSHOT_PRIORITY: Final[EventPriority] = EventPriority.NORMAL
    DEFAULT_SIGNAL_PRIORITY: Final[EventPriority] = EventPriority.HIGH

    def __init__(
        self,
        *,
        config: BaseSpreadConfig,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
        service_name: str = "spread_analyzer",
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._scheduler = scheduler
        self._service_name = service_name

        self._logger = get_logger(
            __name__,
            service_name=service_name,
            event_type="spread_analyzer",
        )

        self._regime_detector = SpreadRegimeDetector(config)
        self._signal_engine = SpreadSignalEngine(
            config=config,
            regime_detector=self._regime_detector,
        )

        self._running = False
        self._registered = False
        self._lock = asyncio.Lock()

        self._subscriptions: list[Subscription] = []
        self._scheduler_job_ids: list[str] = []

        self._last_signal_times: dict[str, datetime] = {}
        self._last_emit_times: dict[tuple[Any, ...], datetime] = {}

        self._stats: dict[str, int] = self._build_base_stats()

    # ------------------------------------------------------------------
    # Required subclass API
    # ------------------------------------------------------------------

    @abstractmethod
    def register(self) -> None:
        """
        Конкретний analyzer має підписатись на потрібні EventBus topics.

        Приклад у дочірньому класі:

            def register(self) -> None:
                if self._registered:
                    return

                self._subscribe(
                    self._config.quote_event_topic,
                    self.on_quote_update,
                    name="spot_futures.on_quote_update",
                )
                self._registered = True
        """
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """
        Конкретний analyzer має повернути розширену статистику.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Вмикає analyzer.

        Важливо:
        - start() не робить EventBus.subscribe();
        - підписки мають створюватись через register();
        - periodic jobs запускаються через Scheduler, якщо він переданий.
        """
        if self._running:
            self._logger.warning(
                "Analyzer already started | analyzer=%s",
                self.__class__.__name__,
            )
            return

        if not self._registered:
            self.register()

        self._running = True
        self._register_scheduler_jobs()

        self._logger.info(
            "Analyzer started | analyzer=%s",
            self.__class__.__name__,
            extra=self._build_start_log_extra(),
        )

        await self._emit_lifecycle_event(
            "analytics.spreads.analyzer.started",
            {
                "analyzer": self.__class__.__name__,
                "service_name": self._service_name,
            },
        )

    async def stop(self) -> None:
        """
        Вимикає analyzer.

        За замовчуванням stop() не видаляє EventBus subscriptions, а лише
        переводить analyzer у неактивний стан. Це дозволяє повторно start()
        без повторної реєстрації підписок.

        Якщо треба повністю прибрати підписки — викликати unregister().
        """
        if not self._running:
            self._logger.warning(
                "Analyzer already stopped | analyzer=%s",
                self.__class__.__name__,
            )
            return

        self._running = False
        self._clear_runtime_state_on_stop()

        self._logger.info(
            "Analyzer stopped | analyzer=%s",
            self.__class__.__name__,
            extra=self._build_stop_log_extra(),
        )

        await self._emit_lifecycle_event(
            "analytics.spreads.analyzer.stopped",
            {
                "analyzer": self.__class__.__name__,
                "service_name": self._service_name,
                "stats": self._stats.copy(),
            },
        )

    def unregister(self) -> None:
        """
        Повністю відписує analyzer від EventBus.

        Використовувати під час shutdown/reconfigure, коли об'єкт більше
        не має отримувати події.
        """
        if not self._subscriptions:
            self._registered = False
            return

        for subscription in list(self._subscriptions):
            try:
                self._event_bus.unsubscribe(subscription)
            except Exception as exc:
                self._mark_exception(
                    "Failed to unsubscribe EventBus subscription",
                    exc,
                    pattern=getattr(subscription, "pattern", None),
                    handler=getattr(subscription, "name", None),
                )

        self._subscriptions.clear()
        self._registered = False

        self._logger.info(
            "Analyzer unregistered | analyzer=%s",
            self.__class__.__name__,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_registered(self) -> bool:
        return self._registered

    # ------------------------------------------------------------------
    # EventBus helpers
    # ------------------------------------------------------------------

    def _subscribe(
        self,
        topic_pattern: str,
        handler: Any,
        *,
        name: str | None = None,
    ) -> Subscription:
        """
        Реєструє EventBus subscription і зберігає Subscription для unregister().
        """
        subscription = self._event_bus.subscribe(
            topic_pattern,
            handler,
            name=name or f"{self.__class__.__name__}.{getattr(handler, '__name__', 'handler')}",
        )
        self._subscriptions.append(subscription)

        self._logger.info(
            "Analyzer subscribed | analyzer=%s topic_pattern=%s handler=%s",
            self.__class__.__name__,
            topic_pattern,
            subscription.name,
        )

        return subscription

    async def _emit(
        self,
        topic: str,
        payload: Any,
        *,
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        """
        Єдиний helper для EventBus.emit().

        Саме emit(), а не publish(topic, payload), бо core.EventBus.publish()
        очікує готовий Event object.
        """
        try:
            accepted = await self._event_bus.emit(
                topic,
                payload,
                priority=priority,
                source=self._service_name,
                correlation_id=correlation_id,
                headers=headers or {},
            )
            if not accepted:
                self._stats["events_rejected"] += 1
                self._logger.warning(
                    "Event rejected by EventBus | topic=%s analyzer=%s",
                    topic,
                    self.__class__.__name__,
                )
            return accepted

        except Exception as exc:
            self._stats["events_failed"] += 1
            self._mark_exception(
                "Failed to emit EventBus event",
                exc,
                topic=topic,
                analyzer=self.__class__.__name__,
            )
            return False

    async def _emit_lifecycle_event(
        self,
        topic: str,
        payload: dict[str, Any],
    ) -> None:
        """
        Best-effort lifecycle event.

        Не піднімає exception назовні, щоб lifecycle analyzer-а не ламався
        через telemetry/event issue.
        """
        try:
            await self._event_bus.emit(
                topic,
                payload,
                priority=EventPriority.LOW,
                source=self._service_name,
            )
        except Exception:
            self._logger.exception(
                "Failed to emit analyzer lifecycle event | topic=%s analyzer=%s",
                topic,
                self.__class__.__name__,
            )

    # ------------------------------------------------------------------
    # Scheduler helpers
    # ------------------------------------------------------------------

    def _register_scheduler_jobs(self) -> None:
        """
        Hook для дочірніх analyzer-ів.

        Базовий клас не додає jobs сам, бо не знає конкретної state-cleanup
        логіки. Дочірній клас може override-нути цей метод і викликати
        _add_interval_job().
        """
        return

    def _add_interval_job(
        self,
        *,
        name: str,
        func: Any,
        interval: float,
        run_immediately: bool = False,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        timeout: float | None = None,
        allow_overlap: bool = False,
        enabled: bool = True,
    ) -> str | None:
        """
        Безпечний helper для Scheduler.add_interval_job().
        """
        if self._scheduler is None:
            self._logger.debug(
                "Scheduler is not configured; interval job skipped | job=%s analyzer=%s",
                name,
                self.__class__.__name__,
            )
            return None

        existing_job = self._scheduler.get_job_by_name(name)
        if existing_job is not None:
            self._logger.debug(
                "Scheduler job already exists | job=%s analyzer=%s",
                name,
                self.__class__.__name__,
            )
            return existing_job.job_id

        try:
            job_id = self._scheduler.add_interval_job(
                name=name,
                func=func,
                interval=interval,
                run_immediately=run_immediately,
                max_retries=max_retries,
                retry_delay=retry_delay,
                timeout=timeout,
                allow_overlap=allow_overlap,
                enabled=enabled,
            )
            self._scheduler_job_ids.append(job_id)

            self._logger.info(
                "Scheduler interval job added | job=%s interval=%s analyzer=%s",
                name,
                interval,
                self.__class__.__name__,
            )
            return job_id

        except Exception as exc:
            self._mark_exception(
                "Failed to add scheduler interval job",
                exc,
                job_name=name,
                interval=interval,
                analyzer=self.__class__.__name__,
            )
            return None

    # ------------------------------------------------------------------
    # Publish helpers
    # ------------------------------------------------------------------

    async def _publish_snapshot(
        self,
        topic: str,
        snapshot: SpreadSnapshot,
        *,
        priority: EventPriority = DEFAULT_SNAPSHOT_PRIORITY,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        accepted = await self._emit(
            topic,
            snapshot,
            priority=priority,
            correlation_id=correlation_id,
            headers=headers,
        )

        if not accepted:
            return False

        self._stats["snapshots_published"] += 1

        self._logger.debug(
            "Spread snapshot published | topic=%s symbol=%s spread_type=%s",
            topic,
            snapshot.symbol,
            snapshot.spread_type.value,
            extra={
                "symbol": snapshot.symbol,
                "event_type": topic,
                "spread_type": snapshot.spread_type.value,
                "exchange_a": snapshot.leg_a_exchange,
                "exchange_b": snapshot.leg_b_exchange,
                "spread_bps": self._to_str(snapshot.spread_bps),
                "net_spread": self._to_str(snapshot.net_spread),
                "regime": snapshot.regime.value,
            },
        )
        return True

    async def _publish_signal(
        self,
        topic: str,
        signal: SpreadSignal,
        *,
        priority: EventPriority = DEFAULT_SIGNAL_PRIORITY,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        if self._should_skip_signal(signal):
            self._stats["cooldown_skips"] += 1
            return False

        accepted = await self._emit(
            topic,
            signal,
            priority=priority,
            correlation_id=correlation_id,
            headers=headers,
        )

        if not accepted:
            return False

        self._stats["signals_published"] += 1

        self._logger.debug(
            "Spread signal published | topic=%s signal_type=%s symbol=%s",
            topic,
            signal.signal_type.value,
            signal.symbol,
            extra={
                "symbol": signal.symbol,
                "event_type": topic,
                "signal_type": signal.signal_type.value,
                "spread_type": signal.spread_type.value,
                "exchange_a": signal.exchange_a,
                "exchange_b": signal.exchange_b,
                "value": self._to_str(signal.value),
                "threshold": self._to_str(signal.threshold),
                "confidence": self._to_str(signal.confidence),
            },
        )
        return True

    async def _publish_signals(
        self,
        topic: str,
        signals: list[SpreadSignal],
        *,
        priority: EventPriority = DEFAULT_SIGNAL_PRIORITY,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> int:
        published_count = 0

        for signal in signals:
            published = await self._publish_signal(
                topic,
                signal,
                priority=priority,
                correlation_id=correlation_id,
                headers=headers,
            )
            if published:
                published_count += 1

        return published_count

    # ------------------------------------------------------------------
    # Signal / throttling helpers
    # ------------------------------------------------------------------

    def _evaluate_snapshot_signals(
        self,
        *,
        snapshot: SpreadSnapshot,
        previous_snapshot: SpreadSnapshot | None = None,
        opportunity: Any | None = None,
    ) -> list[SpreadSignal]:
        result = self._signal_engine.evaluate_snapshot(
            snapshot=snapshot,
            previous_snapshot=previous_snapshot,
            opportunity=opportunity,
        )
        return result.signals

    def _should_skip_emit(
        self,
        key: tuple[Any, ...],
        timestamp: datetime,
    ) -> bool:
        last_emit_at = self._last_emit_times.get(key)
        if last_emit_at is None:
            self._last_emit_times[key] = timestamp
            return False

        min_interval = timedelta(milliseconds=self._config.min_emit_interval_ms)
        if (timestamp - last_emit_at) < min_interval:
            return True

        self._last_emit_times[key] = timestamp
        return False

    def _should_skip_signal(
        self,
        signal: SpreadSignal,
    ) -> bool:
        signal_key = self._build_signal_key(signal)
        now = signal.timestamp

        last_signal_at = self._last_signal_times.get(signal_key)
        if last_signal_at is None:
            self._last_signal_times[signal_key] = now
            return False

        cooldown = timedelta(seconds=self._config.cooldown_seconds)
        if (now - last_signal_at) < cooldown:
            return True

        self._last_signal_times[signal_key] = now
        return False

    def _build_signal_key(self, signal: SpreadSignal) -> str:
        exchange_a = signal.exchange_a or "na"
        exchange_b = signal.exchange_b or "na"

        return (
            f"{signal.signal_type.value}|"
            f"{signal.spread_type.value}|"
            f"{signal.symbol}|"
            f"{exchange_a}|"
            f"{exchange_b}"
        )

    # ------------------------------------------------------------------
    # Stats / logging helpers
    # ------------------------------------------------------------------

    def _build_base_stats(self) -> dict[str, int]:
        return {
            "calculations_total": 0,
            "snapshots_published": 0,
            "signals_published": 0,
            "cooldown_skips": 0,
            "emit_skips": 0,
            "events_rejected": 0,
            "events_failed": 0,
            "exceptions": 0,
        }

    def _build_start_log_extra(self) -> dict[str, Any]:
        return {
            "enabled": self._config.enabled,
            "max_quote_age_ms": self._config.max_quote_age_ms,
            "max_quote_skew_ms": self._config.max_quote_skew_ms,
            "rolling_window_size": self._config.rolling_window_size,
            "min_emit_interval_ms": self._config.min_emit_interval_ms,
            "cooldown_seconds": self._config.cooldown_seconds,
            "subscriptions": len(self._subscriptions),
            "scheduler_jobs": len(self._scheduler_job_ids),
        }

    def _build_stop_log_extra(self) -> dict[str, Any]:
        return {
            "stats": self._stats.copy(),
            "subscriptions": len(self._subscriptions),
            "scheduler_jobs": len(self._scheduler_job_ids),
        }

    def _mark_exception(
        self,
        message: str,
        exc: Exception,
        **extra: Any,
    ) -> None:
        self._stats["exceptions"] += 1
        self._logger.exception(
            message,
            extra={
                "error": str(exc),
                "analyzer": self.__class__.__name__,
                **extra,
            },
        )

    def _clear_runtime_state_on_stop(self) -> None:
        """
        Очищає runtime-only throttling state.

        Не очищає market caches дочірніх analyzer-ів — це їхня відповідальність.
        """
        self._last_signal_times.clear()
        self._last_emit_times.clear()

    @staticmethod
    def _to_str(value: Decimal | None) -> str | None:
        return str(value) if value is not None else None