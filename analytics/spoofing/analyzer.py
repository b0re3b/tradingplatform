from __future__ import annotations
from core.logger import get_logger

import inspect
import math
from datetime import datetime
from typing import Any, Iterable, Mapping, Protocol

from core.event_bus import Event, EventBus, EventPriority, Subscription
from core.scheduler import Scheduler
from analytics.market_state_contract import build_state_backed_cache_bundle, MarketStateSnapshotSource

from .base import BaseSpoofingAnalyzer
from .config import SpoofingConfig
from .enums import SpoofingComponent, SpoofingSide, SpoofingStatus
from .fake_liquidity_detector import FakeLiquidityDetector
from .flip_pressure_detector import FlipPressureDetector
from .layering_detector import LayeringDetector
from .models import (
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    AnalyzerOutput,
    DetectorResult,
    LiquidityLifecycleEvent,
    OrderbookLevelSnapshot,
    SpoofingKey,
    SpoofingSignal,
    TrackedWall,
    spoofing_key_to_dict,
)
from .order_pull_detector import OrderPullDetector
from .orderbook_wall_detector import OrderbookWallDetector
from .persistence_tracker import PersistenceTracker
from .spoofing_score import SpoofingScoreEngine


class SupportsOrderBookCache(Protocol):
    """
    Мінімальний read-only contract для data/orderbook_cache.py.

    Реальна реалізація OrderBookCache може мати ширший API. Analyzer
    використовує duck typing, щоб не створювати жорстку залежність
    analytics -> data.
    """

    def get_book(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        depth: int | None = None,
    ) -> Any: ...


class SupportsTrackedWallAnalyzeMany(Protocol):
    def analyze_many(
        self,
        walls: Iterable[TrackedWall],
        *,
        key: SpoofingKey | None = None,
        exchange: str | None = None,
        symbol: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
        current_mid_price: float | None = None,
    ) -> list[DetectorResult]: ...


class SpoofingAnalyzer(BaseSpoofingAnalyzer):
    """
    Центральний orchestrator для analytics.spoofing.

    Відповідає за:
    - підписку на data-layer topics через EventBus;
    - роботу зі scoped key: exchange + market_type + symbol + timeframe;
    - читання normalized orderbook state з OrderBookCache або payload
      market.orderbook.updated;
    - оновлення PersistenceTracker;
    - запуск wall / pull / advanced detector-ів;
    - агрегацію результатів через SpoofingScoreEngine;
    - публікацію тільки analytics.spoofing.* подій;
    - cleanup через Scheduler.

    Correct production flow:
        exchange adapters
            -> market.orderbook
            -> data.OrderBookCache
            -> market.orderbook.updated
            -> SpoofingAnalyzer
            -> analytics.spoofing.*
    """

    component = SpoofingComponent.ANALYZER

    def __init__(
        self,
        *,
        event_bus: EventBus | None,
        scheduler: Scheduler | None,
        config: SpoofingConfig,
        orderbook_cache: SupportsOrderBookCache | None = None,
        market_state_store: Any | None = None,
        persistence_tracker: PersistenceTracker | None = None,
        wall_detector: OrderbookWallDetector | None = None,
        pull_detector: OrderPullDetector | None = None,
        score_engine: SpoofingScoreEngine | None = None,
        fake_liquidity_detector: FakeLiquidityDetector | None = None,
        flip_pressure_detector: FlipPressureDetector | None = None,
        layering_detector: LayeringDetector | None = None,
    ) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__init__", _analytics_args)
        except Exception:
            pass
        super().__init__(
            event_bus=event_bus,
            scheduler=scheduler,
            config=config,
        )

        self.config.validate()
        self._market_state_store = market_state_store
        state_cache_bundle = build_state_backed_cache_bundle(market_state_store)
        self.orderbook_cache = orderbook_cache or (state_cache_bundle.orderbook if state_cache_bundle is not None else None)
        self._state_snapshot_source = state_cache_bundle.source if state_cache_bundle is not None else None

        self.persistence_tracker = persistence_tracker or PersistenceTracker(
            event_bus=event_bus,
            scheduler=scheduler,
            config=config,
        )
        self.wall_detector = wall_detector or OrderbookWallDetector(
            event_bus=event_bus,
            scheduler=scheduler,
            config=config,
            persistence_tracker=self.persistence_tracker,
        )
        self.pull_detector = pull_detector or OrderPullDetector(
            event_bus=event_bus,
            scheduler=scheduler,
            config=config,
            persistence_tracker=self.persistence_tracker,
        )
        self.score_engine = score_engine or SpoofingScoreEngine(
            event_bus=event_bus,
            scheduler=scheduler,
            config=config,
        )

        self.fake_liquidity_detector = (
            fake_liquidity_detector
            if fake_liquidity_detector is not None
            else self._create_fake_liquidity_detector_if_enabled()
        )
        self.flip_pressure_detector = (
            flip_pressure_detector
            if flip_pressure_detector is not None
            else self._create_flip_pressure_detector_if_enabled()
        )
        self.layering_detector = (
            layering_detector
            if layering_detector is not None
            else self._create_layering_detector_if_enabled()
        )

        self._latest_output_by_key: dict[SpoofingKey, AnalyzerOutput] = {}
        self._latest_signal_by_key: dict[SpoofingKey, SpoofingSignal] = {}
        self._subscriptions: list[Subscription] = []
        self._cleanup_job_id: str | None = None
        self._registered = False

        # BaseSpoofingAnalyzer уже може мати _running. Явно синхронізуємо
        # lifecycle, щоб stop() не залежав від різних прапорців.
        self._running = False

    # -------------------------------------------------------------------------
    # Dependency factories
    # -------------------------------------------------------------------------

    def _create_fake_liquidity_detector_if_enabled(self) -> FakeLiquidityDetector | None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_create_fake_liquidity_detector_if_enabled", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled or not self.config.fake_liquidity.enabled:
            return None
        return FakeLiquidityDetector(
            event_bus=self.event_bus,
            scheduler=self.scheduler,
            config=self.config,
            persistence_tracker=self.persistence_tracker,
        )

    def _create_flip_pressure_detector_if_enabled(self) -> FlipPressureDetector | None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_create_flip_pressure_detector_if_enabled", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled or not self.config.flip_pressure.enabled:
            return None
        return FlipPressureDetector(
            event_bus=self.event_bus,
            scheduler=self.scheduler,
            config=self.config,
            persistence_tracker=self.persistence_tracker,
        )

    def _create_layering_detector_if_enabled(self) -> LayeringDetector | None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_create_layering_detector_if_enabled", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled or not self.config.layering.enabled:
            return None
        return LayeringDetector(
            event_bus=self.event_bus,
            scheduler=self.scheduler,
            config=self.config,
            persistence_tracker=self.persistence_tracker,
        )

    # -------------------------------------------------------------------------
    # EventBus / Scheduler registration
    # -------------------------------------------------------------------------

    def register(self) -> None:
        """
        Реєструє analyzer у core infrastructure:
        - EventBus subscriptions на data-layer topics;
        - Scheduler interval job для cleanup PersistenceTracker.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "register", _analytics_args)
        except Exception:
            pass
        if self._registered:
            self.log_warning("SpoofingAnalyzer already registered")
            return

        if not self.config.enabled or not self.config.analyzer.enabled:
            self.log_warning("SpoofingAnalyzer registration skipped: disabled by config")
            return

        self._register_eventbus_subscriptions()
        self._register_scheduler_jobs()

        self._registered = True
        self._running = True

        self.log_info(
            "SpoofingAnalyzer registered",
            source_topics=list(self.config.production_source_topics),
            cleanup_job_id=self._cleanup_job_id,
            publish_updates=self.config.analyzer.publish_updates,
            publish_detected_only=self.config.analyzer.publish_detected_only,
            scope="exchange:market_type:symbol:timeframe",
        )

    async def start(self) -> None:
        """
        Async lifecycle hook used by app/main.py.

        Older spoofing runtime only implemented register(), so the component was
        hard to observe in the new state-driven app bootstrap: it could be
        registered as a MarketScheduler evaluator, but no analytics.spoofing.*
        startup event was emitted.  Keep register() synchronous for backward
        compatibility, and use start() only for explicit runtime diagnostics.
        """
        if not self._registered:
            self.register()

        if getattr(self, "_started", False):
            return

        self._started = True
        self._running = True

        await self.emit_event(
            "analytics.spoofing.analyzer.started",
            {
                "service": "spoofing_analyzer",
                "component": self.component.value,
                "registered": self._registered,
                "running": self._running,
                "input_mode": "market_state",
                "input_topics": list(self.config.production_source_topics),
                "orderbook_cache_attached": self.orderbook_cache is not None,
                "market_state_store_attached": self.market_state_store is not None,
                "publish_updates": self.config.analyzer.publish_updates,
                "publish_detected_only": self.config.analyzer.publish_detected_only,
                "scope": "exchange:market_type:symbol:timeframe",
            },
            priority=EventPriority.LOW,
        )

    def stop(self) -> None:
        """
        Зупиняє analyzer lifecycle:
        - unsubscribe всіх EventBus subscriptions;
        - disable cleanup Scheduler job;
        - синхронізує _registered і _running.

        Метод навмисно sync, як і register(), щоб відповідати поточному
        core-style lifecycle у пакеті.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "stop", _analytics_args)
        except Exception:
            pass
        if not self._registered and not getattr(self, "_running", False):
            self.log_warning("SpoofingAnalyzer already stopped")
            return

        for subscription in list(self._subscriptions):
            try:
                unsubscribe = getattr(subscription, "unsubscribe", None)
                if callable(unsubscribe):
                    unsubscribe()
                    continue

                close = getattr(subscription, "close", None)
                if callable(close):
                    close()
                    continue

                cancel = getattr(subscription, "cancel", None)
                if callable(cancel):
                    cancel()

            except Exception as exc:
                self.log_exception(
                    "Failed to unsubscribe SpoofingAnalyzer subscription",
                    error=str(exc),
                    subscription=str(subscription),
                )

        self._subscriptions.clear()

        if self.scheduler is not None and self._cleanup_job_id is not None:
            try:
                disable_job = getattr(self.scheduler, "disable_job", None)
                if callable(disable_job):
                    disable_job(self._cleanup_job_id)
                else:
                    remove_job = getattr(self.scheduler, "remove_job", None)
                    if callable(remove_job):
                        remove_job(self._cleanup_job_id)
            except Exception as exc:
                self.log_exception(
                    "Failed to disable SpoofingAnalyzer cleanup job",
                    cleanup_job_id=self._cleanup_job_id,
                    error=str(exc),
                )

        self._cleanup_job_id = None
        self._registered = False
        self._running = False
        self._started = False

        self.log_info("SpoofingAnalyzer stopped")

    def _register_eventbus_subscriptions(self) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_register_eventbus_subscriptions", _analytics_args)
        except Exception:
            pass
        event_bus = self.require_event_bus()

        for topic in self.config.production_source_topics:
            subscription = event_bus.subscribe(
                topic,
                self._handle_event,
            )
            self._subscriptions.append(subscription)

        if self.config.analyzer.allow_legacy_raw_topics:
            for topic in self.config.analyzer.legacy_raw_topic_patterns:
                subscription = event_bus.subscribe(
                    topic,
                    self._handle_legacy_raw_event,
                )
                self._subscriptions.append(subscription)

    def _register_scheduler_jobs(self) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_register_scheduler_jobs", _analytics_args)
        except Exception:
            pass
        if self.scheduler is None:
            self.log_warning("Scheduler cleanup registration skipped: scheduler is None")
            return

        if not self.config.analyzer.scheduler_cleanup_enabled:
            return

        self._cleanup_job_id = self.scheduler.add_interval_job(
            name=self.config.analyzer.scheduler_cleanup_job_name,
            func=self.cleanup_job,
            interval=self.config.cleanup_interval_seconds,
            run_immediately=False,
            max_retries=1,
            retry_delay=0.5,
            timeout=5.0,
            allow_overlap=False,
            enabled=True,
        )

    async def _handle_event(self, event: Event) -> None:
        """
        Production EventBus callback для data-layer events.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_handle_event", _analytics_args)
        except Exception:
            pass
        payload = getattr(event, "payload", None)
        if not isinstance(payload, Mapping):
            self.log_warning("Spoofing event payload is not a mapping")
            return

        correlation_id = self._extract_event_correlation_id(event)

        try:
            key = self.extract_key_from_payload(payload)
            if key is None:
                self.log_warning(
                    "Spoofing event skipped: cannot extract key",
                    topic=getattr(event, "topic", None),
                    payload_keys=list(payload.keys()),
                )
                return

            if not self.should_process_key(key):
                return

            topic = str(getattr(event, "topic", ""))

            if topic in self.config.analyzer.source_topic_patterns_orderbook:
                output = await self.process_event_payload(
                    dict(payload),
                    key=key,
                    correlation_id=correlation_id,
                )
            elif topic in self.config.analyzer.source_topic_patterns_trade:
                output = await self.process_key(
                    key,
                    current_mid_price=self._extract_current_mid_price(dict(payload)),
                    correlation_id=correlation_id,
                    metadata={
                        "source_topic": topic,
                        "reason": "trade_update_confirmation_reprocess",
                    },
                )
            else:
                output = await self.process_event_payload(
                    dict(payload),
                    key=key,
                    correlation_id=correlation_id,
                )

            if output.signal is not None:
                self.log_debug(
                    "Spoofing signal processed from event",
                    symbol=output.symbol,
                    exchange=output.exchange,
                    market_type=output.market_type,
                    timeframe=output.timeframe,
                    signal_id=output.signal.signal_id,
                    score=output.signal.score,
                    confidence=output.signal.confidence,
                )

        except Exception as exc:
            self.log_exception(
                "Failed to process spoofing event",
                error=str(exc),
                payload_keys=list(payload.keys()),
            )

            if self.config.analyzer.publish_errors:
                await self._publish_error(
                    error=exc,
                    payload=dict(payload),
                    context={"handler": "_handle_event"},
                    correlation_id=correlation_id,
                )
            raise

    async def _handle_legacy_raw_event(self, event: Event) -> None:
        """
        Legacy/manual raw callback.

        Не використовується у production, якщо allow_legacy_raw_topics=False.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_handle_legacy_raw_event", _analytics_args)
        except Exception:
            pass
        payload = getattr(event, "payload", None)
        if not isinstance(payload, Mapping):
            return

        correlation_id = self._extract_event_correlation_id(event)

        try:
            await self.process_event_payload(
                dict(payload),
                correlation_id=correlation_id,
                metadata={"source": "legacy_raw_event"},
                allow_raw_payload=True,
            )
        except Exception as exc:
            self.log_exception(
                "Failed to process legacy raw spoofing event",
                error=str(exc),
                payload_keys=list(payload.keys()),
            )
            if self.config.analyzer.publish_errors:
                await self._publish_error(
                    error=exc,
                    payload=dict(payload),
                    context={"handler": "_handle_legacy_raw_event"},
                    correlation_id=correlation_id,
                )
            raise

    # -------------------------------------------------------------------------
    # Main processing API
    # -------------------------------------------------------------------------

    async def process_key(
        self,
        key: SpoofingKey,
        *,
        current_mid_price: float | None = None,
        metadata: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> AnalyzerOutput:
        """
        Основний production API.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "process_key", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled or not self.config.analyzer.enabled:
            return self._empty_output(
                key=key,
                reason="analyzer_disabled",
                metadata=metadata,
            )

        if not self.should_process_key(key):
            return self._empty_output(
                key=key,
                reason="key_not_allowed",
                metadata=metadata,
            )

        snapshots = await self._load_snapshots_from_orderbook_cache(key)
        if not snapshots:
            return self._empty_output(
                key=key,
                reason="empty_orderbook_cache_snapshot",
                metadata=metadata,
            )

        return await self.process_snapshots(
            snapshots=snapshots,
            key=key,
            current_mid_price=current_mid_price,
            metadata=metadata,
            correlation_id=correlation_id,
        )

    async def process_event_payload(
        self,
        payload: dict[str, Any],
        *,
        key: SpoofingKey | None = None,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        allow_raw_payload: bool = False,
    ) -> AnalyzerOutput:
        """
        Обробляє EventBus payload.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "process_event_payload", _analytics_args)
        except Exception:
            pass
        resolved_key = key or self.extract_key_from_payload(payload)
        if resolved_key is None:
            fallback_key = self._fallback_key_from_payload(payload)
            return self._empty_output(
                key=fallback_key,
                reason="cannot_extract_key",
                metadata={
                    "payload_keys": list(payload.keys()),
                    **self._safe_metadata(metadata),
                },
            )

        snapshots = self._extract_or_build_snapshots_from_payload(
            payload,
            key=resolved_key,
            allow_raw_payload=allow_raw_payload,
        )

        if not snapshots and self.orderbook_cache is not None:
            snapshots = await self._load_snapshots_from_orderbook_cache(resolved_key)

        return await self.process_snapshots(
            snapshots=snapshots,
            key=resolved_key,
            current_mid_price=self._extract_current_mid_price(payload),
            correlation_id=correlation_id,
            metadata={
                "event_payload": {
                    "symbol": payload.get("symbol"),
                    "exchange": payload.get("exchange"),
                    "market_type": payload.get("market_type"),
                    "timeframe": payload.get("timeframe"),
                    "sequence_id": payload.get("sequence_id"),
                },
                **self._safe_metadata(metadata),
            },
        )


    async def process_market_snapshot(
        self,
        snapshot: Any,
        *,
        correlation_id: str | None = None,
    ) -> AnalyzerOutput:
        """
        State-driven entrypoint for MarketScheduler.

        New production flow:
            WS/REST -> MarketIngestion -> MarketStateStore
            -> MarketScheduler -> SpoofingAnalyzer.process_market_snapshot()

        The legacy/event path still works, but spoofing must not depend on
        market.orderbook.updated EventBus events anymore.  This adapter unwraps
        MarketSnapshot.orderbook into the same normalized orderbook contract used
        by process_orderbook().
        """
        payload = self._orderbook_payload_from_market_snapshot(snapshot)
        if payload is None:
            scope = getattr(snapshot, "scope", None)
            key = self.make_key(
                exchange=getattr(scope, "exchange", "binance"),
                market_type=getattr(scope, "market_type", DEFAULT_MARKET_TYPE),
                symbol=getattr(scope, "symbol", "UNKNOWN"),
                timeframe=getattr(scope, "timeframe", DEFAULT_TIMEFRAME),
            )
            reason = "market_snapshot_without_orderbook"
            await self.emit_event(
                "analytics.spoofing.snapshot_skipped",
                {
                    "service": "spoofing_analyzer",
                    "reason": reason,
                    "scope": spoofing_key_to_dict(key),
                    "dirty_reasons": list(getattr(snapshot, "dirty_reasons", ()) or ()),
                    "has_orderbook": getattr(snapshot, "orderbook", None) is not None,
                    "entrypoint": "process_market_snapshot",
                },
                priority=EventPriority.LOW,
            )
            return self._empty_output(
                key=key,
                reason=reason,
                metadata={"source": "market_scheduler", "entrypoint": "process_market_snapshot"},
            )

        if len(payload.get("bids") or ()) < max(1, int(getattr(self.config.wall_detection, "min_levels_to_scan", 10))) or len(payload.get("asks") or ()) < max(1, int(getattr(self.config.wall_detection, "min_levels_to_scan", 10))):
            key = self.make_key(
                exchange=str(payload.get("exchange") or "binance"),
                market_type=str(payload.get("market_type") or DEFAULT_MARKET_TYPE),
                symbol=str(payload.get("symbol") or "UNKNOWN"),
                timeframe=str(payload.get("timeframe") or DEFAULT_TIMEFRAME),
            )
            await self.emit_event(
                "analytics.spoofing.snapshot_skipped",
                {
                    "service": "spoofing_analyzer",
                    "reason": "insufficient_orderbook_depth",
                    "scope": spoofing_key_to_dict(key),
                    "bids": len(payload.get("bids") or ()),
                    "asks": len(payload.get("asks") or ()),
                    "min_levels_to_scan": int(getattr(self.config.wall_detection, "min_levels_to_scan", 10)),
                    "dirty_reasons": payload.get("dirty_reasons") or [],
                    "entrypoint": "process_market_snapshot",
                },
                priority=EventPriority.LOW,
            )
            return self._empty_output(
                key=key,
                reason="insufficient_orderbook_depth",
                metadata={
                    "source": "market_scheduler",
                    "entrypoint": "process_market_snapshot",
                    "bids": len(payload.get("bids") or ()),
                    "asks": len(payload.get("asks") or ()),
                },
            )

        return await self.process_orderbook(
            symbol=str(payload["symbol"]),
            exchange=str(payload["exchange"]),
            market_type=str(payload.get("market_type") or DEFAULT_MARKET_TYPE),
            timeframe=str(payload.get("timeframe") or DEFAULT_TIMEFRAME),
            exchange_symbol=payload.get("exchange_symbol"),
            bids=payload.get("bids") or (),
            asks=payload.get("asks") or (),
            best_bid=payload.get("best_bid"),
            best_ask=payload.get("best_ask"),
            sequence_id=payload.get("sequence") or payload.get("sequence_id") or payload.get("last_update_id"),
            timestamp=payload.get("updated_at_ms") or payload.get("timestamp_ms") or payload.get("last_update_ms"),
            current_mid_price=payload.get("mid_price") or payload.get("current_price") or payload.get("reference_price"),
            metadata={
                "source": "market_scheduler",
                "source_topic": "market.state.snapshot",
                "entrypoint": "process_market_snapshot",
                "dirty_reasons": payload.get("dirty_reasons") or [],
            },
            correlation_id=correlation_id,
        )

    @staticmethod
    def _level_to_payload(level: Any) -> dict[str, Any] | list[Any] | tuple[Any, ...] | None:
        if level is None:
            return None
        if isinstance(level, dict):
            return dict(level)
        to_dict = getattr(level, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
            if isinstance(value, dict):
                return value
        price = getattr(level, "price", None)
        quantity = getattr(level, "quantity", getattr(level, "qty", getattr(level, "size", None)))
        if price is not None and quantity is not None:
            return {"price": price, "quantity": quantity, "size": quantity}
        return None

    def _orderbook_payload_from_market_snapshot(self, snapshot: Any) -> dict[str, Any] | None:
        if snapshot is None:
            return None

        if isinstance(snapshot, Mapping):
            raw = dict(snapshot)
            scope = raw.get("scope") if isinstance(raw.get("scope"), Mapping) else {}
            orderbook = raw.get("orderbook") or raw.get("book") or raw.get("depth")
        else:
            scope_obj = getattr(snapshot, "scope", None)
            scope = scope_obj.to_dict() if hasattr(scope_obj, "to_dict") else {
                "exchange": getattr(scope_obj, "exchange", None),
                "market_type": getattr(scope_obj, "market_type", None),
                "symbol": getattr(scope_obj, "symbol", None),
                "timeframe": getattr(scope_obj, "timeframe", None),
                "exchange_symbol": getattr(scope_obj, "exchange_symbol", None),
            }
            raw = snapshot.to_dict() if hasattr(snapshot, "to_dict") else {}
            orderbook = getattr(snapshot, "orderbook", None)

        if orderbook is None:
            return None

        if isinstance(orderbook, Mapping):
            book = dict(orderbook)
        elif hasattr(orderbook, "to_dict"):
            book = orderbook.to_dict()
        else:
            book = {
                "bids": getattr(orderbook, "bids", ()),
                "asks": getattr(orderbook, "asks", ()),
                "best_bid": getattr(orderbook, "best_bid", None),
                "best_ask": getattr(orderbook, "best_ask", None),
                "mid_price": getattr(orderbook, "mid_price", None),
                "spread": getattr(orderbook, "spread", None),
                "sequence": getattr(orderbook, "sequence", None),
                "last_update_ms": getattr(orderbook, "last_update_ms", None),
            }

        raw_bids = (
            book.get("bids")
            or book.get("bid_levels")
            or book.get("buy")
            or book.get("bid_depth")
            or ()
        )
        raw_asks = (
            book.get("asks")
            or book.get("ask_levels")
            or book.get("sell")
            or book.get("ask_depth")
            or ()
        )
        bids = [item for item in (self._level_to_payload(level) for level in raw_bids) if item is not None]
        asks = [item for item in (self._level_to_payload(level) for level in raw_asks) if item is not None]

        # Top-of-book fallback is safe only when size/quantity exists.  Spoofing
        # detectors need real depth, so never invent synthetic quantity.
        if not bids:
            bid_price = book.get("best_bid") or book.get("best_bid_price") or book.get("bid") or book.get("bid_price")
            bid_qty = book.get("best_bid_quantity") or book.get("best_bid_qty") or book.get("bid_quantity") or book.get("bid_qty") or book.get("bid_size")
            if bid_price is not None and bid_qty is not None:
                level = self._level_to_payload({"price": bid_price, "quantity": bid_qty, "size": bid_qty})
                if level is not None:
                    bids = [level]
        if not asks:
            ask_price = book.get("best_ask") or book.get("best_ask_price") or book.get("ask") or book.get("ask_price")
            ask_qty = book.get("best_ask_quantity") or book.get("best_ask_qty") or book.get("ask_quantity") or book.get("ask_qty") or book.get("ask_size")
            if ask_price is not None and ask_qty is not None:
                level = self._level_to_payload({"price": ask_price, "quantity": ask_qty, "size": ask_qty})
                if level is not None:
                    asks = [level]

        if not bids or not asks:
            return None

        payload = {
            **scope,
            "exchange": scope.get("exchange") or raw.get("exchange") or self.config.default_exchange,
            "market_type": scope.get("market_type") or raw.get("market_type") or DEFAULT_MARKET_TYPE,
            "symbol": scope.get("symbol") or raw.get("symbol"),
            "timeframe": scope.get("timeframe") or raw.get("timeframe") or DEFAULT_TIMEFRAME,
            "exchange_symbol": scope.get("exchange_symbol") or raw.get("exchange_symbol") or scope.get("symbol") or raw.get("symbol"),
            "best_bid": book.get("best_bid"),
            "best_ask": book.get("best_ask"),
            "mid_price": book.get("mid_price") or raw.get("current_price") or raw.get("reference_price"),
            "spread": book.get("spread"),
            "sequence": book.get("sequence") or book.get("last_update_id"),
            "last_update_ms": book.get("last_update_ms") or raw.get("updated_at_ms"),
            "updated_at_ms": raw.get("updated_at_ms") or book.get("last_update_ms"),
            "timestamp_ms": raw.get("updated_at_ms") or book.get("last_update_ms"),
            "bids": bids,
            "asks": asks,
            "dirty_reasons": raw.get("dirty_reasons") or [],
        }
        return payload if payload.get("symbol") and payload.get("exchange") else None

    async def process_orderbook(
        self,
        *,
        symbol: str,
        exchange: str,
        bids: Iterable[Any],
        asks: Iterable[Any],
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        exchange_symbol: str | None = None,
        best_bid: float | None = None,
        best_ask: float | None = None,
        sequence_id: int | None = None,
        timestamp: datetime | None = None,
        current_mid_price: float | None = None,
        metadata: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> AnalyzerOutput:
        """
        Manual/test helper для прямої роботи із bids/asks без EventBus.

        Production runtime має використовувати:
            market.orderbook.updated -> process_event_payload/process_key
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "process_orderbook", _analytics_args)
        except Exception:
            pass
        key = self.make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

        snapshots = self.wall_detector.build_snapshot_levels_from_orderbook(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
            exchange_symbol=exchange_symbol,
            bids=bids,
            asks=asks,
            best_bid=best_bid,
            best_ask=best_ask,
            sequence_id=sequence_id,
            timestamp=timestamp,
            metadata={
                **self._safe_metadata(metadata),
                "source": "manual_or_test_process_orderbook",
            },
        )

        resolved_mid = current_mid_price
        if resolved_mid is None:
            resolved_mid = self._resolve_mid_price_from_best_quotes(
                best_bid=best_bid,
                best_ask=best_ask,
            )

        return await self.process_snapshots(
            snapshots=snapshots,
            key=key,
            current_mid_price=resolved_mid,
            metadata=metadata,
            correlation_id=correlation_id,
        )

    async def process_snapshots(
        self,
        *,
        snapshots: Iterable[OrderbookLevelSnapshot],
        key: SpoofingKey,
        current_mid_price: float | None = None,
        metadata: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> AnalyzerOutput:
        """
        Основна точка входу для normalized orderbook snapshots одного scoped key.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "process_snapshots", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled or not self.config.analyzer.enabled:
            return self._empty_output(
                key=key,
                reason="analyzer_disabled",
                metadata=metadata,
            )

        if not self.should_process_key(key):
            return self._empty_output(
                key=key,
                reason="key_not_allowed",
                metadata=metadata,
            )

        levels = [
            level
            for level in snapshots
            if self._has_level_contract(level) and level.key == key
        ]
        if not levels:
            output = self._empty_output(
                key=key,
                reason="no_levels_for_key",
                metadata=metadata,
            )
            self._latest_output_by_key[key] = output
            return output

        resolved_mid_price = current_mid_price
        if resolved_mid_price is not None and not self._is_finite_positive(resolved_mid_price):
            resolved_mid_price = None
        if resolved_mid_price is None:
            resolved_mid_price = self._resolve_mid_price_from_levels(levels)

        self.persistence_tracker.maybe_cleanup(now=levels[0].timestamp)

        filtered_levels = self._filter_levels_for_analysis(levels, key=key)
        if not filtered_levels:
            output = self._empty_output(
                key=key,
                reason="no_levels_after_filtering",
                metadata={
                    "input_levels_count": len(levels),
                    **self._safe_metadata(metadata),
                },
            )
            self._latest_output_by_key[key] = output
            return output

        tracked_walls, lifecycle_events = self._update_tracker(filtered_levels)

        detector_results = self._run_base_detectors(
            snapshots=filtered_levels,
            key=key,
            current_mid_price=resolved_mid_price,
        )

        detector_results.extend(
            self._run_optional_detectors(
                tracked_walls=self.persistence_tracker.get_walls_for_key(key),
                key=key,
                current_mid_price=resolved_mid_price,
            )
        )

        detector_results = self._limit_detector_results(detector_results)

        signal = self._build_signal(
            detector_results=detector_results,
            key=key,
        )

        if signal is not None:
            self._latest_signal_by_key[key] = signal

        scope = spoofing_key_to_dict(key)

        output = AnalyzerOutput(
            symbol=scope["symbol"],
            exchange=scope["exchange"],
            market_type=scope["market_type"],
            timeframe=scope["timeframe"],
            signal=signal,
            detector_results=detector_results,
            tracked_walls=self.persistence_tracker.snapshot_state(key=key),
            lifecycle_events=lifecycle_events,
            metadata={
                "scope": scope,
                "lifecycle_events_count": len(lifecycle_events),
                "input_levels_count": len(levels),
                "filtered_levels_count": len(filtered_levels),
                "current_mid_price": resolved_mid_price,
                **self._safe_metadata(metadata),
            },
        )

        self._latest_output_by_key[key] = output

        await self._publish_outputs(
            output=output,
            lifecycle_events=lifecycle_events,
            correlation_id=correlation_id,
        )

        return output

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def cleanup(self) -> int:
        """
        Синхронний cleanup для ручного виклику або тестів.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "cleanup", _analytics_args)
        except Exception:
            pass
        expired_count = self.persistence_tracker.cleanup()
        self.log_debug(
            "SpoofingAnalyzer cleanup completed",
            expired_count=expired_count,
        )
        return expired_count

    async def cleanup_job(self) -> None:
        """
        Async-safe Scheduler callback.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "cleanup_job", _analytics_args)
        except Exception:
            pass
        self.cleanup()
        return None

    # -------------------------------------------------------------------------
    # Tracker + detectors
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_event_correlation_id(event: Event) -> str | None:
        try:
            _analytics_class_name = "SpoofingAnalyzer"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_extract_event_correlation_id", _analytics_args)
        except Exception:
            pass
        event_id = getattr(event, "event_id", None)
        return event_id if isinstance(event_id, str) and event_id else None

    def _filter_levels_for_analysis(
        self,
        levels: list[OrderbookLevelSnapshot],
        *,
        key: SpoofingKey,
    ) -> list[OrderbookLevelSnapshot]:
        """
        Попередній фільтр рівнів.

        Залишає:
        - валідні bid/ask рівні;
        - finite price/size;
        - дозволений scoped key;
        - top-N levels per side according to wall_detection.max_levels_to_scan.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_filter_levels_for_analysis", _analytics_args)
        except Exception:
            pass
        valid = [
            level
            for level in levels
            if self._has_level_contract(level)
            and level.key == key
            and self._is_finite_positive(level.price)
            and self._is_finite_positive(level.size)
            and level.side in {SpoofingSide.BID, SpoofingSide.ASK}
            and self.should_process_key(level.key)
        ]

        if not valid:
            return []

        bids = sorted(
            [level for level in valid if level.side == SpoofingSide.BID],
            key=lambda item: item.price,
            reverse=True,
        )
        asks = sorted(
            [level for level in valid if level.side == SpoofingSide.ASK],
            key=lambda item: item.price,
        )

        min_levels = max(0, self.config.wall_detection.min_levels_to_scan)
        if min_levels > 0 and len(valid) < min_levels:
            self.log_debug(
                "Orderbook levels skipped: below min_levels_to_scan",
                levels_count=len(valid),
                min_levels_to_scan=min_levels,
                key=spoofing_key_to_dict(key),
            )
            return []

        max_levels = max(0, self.config.wall_detection.max_levels_to_scan)
        if max_levels <= 0:
            return []

        return bids[:max_levels] + asks[:max_levels]

    def _update_tracker(
        self,
        levels: list[OrderbookLevelSnapshot],
    ) -> tuple[list[TrackedWall], list[LiquidityLifecycleEvent]]:
        """
        Оновлює PersistenceTracker по релевантних normalized рівнях.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_update_tracker", _analytics_args)
        except Exception:
            pass
        return self.persistence_tracker.upsert_many(levels)

    def _run_base_detectors(
        self,
        *,
        snapshots: list[OrderbookLevelSnapshot],
        key: SpoofingKey,
        current_mid_price: float | None,
    ) -> list[DetectorResult]:
        """
        Запускає базові detector-и.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_run_base_detectors", _analytics_args)
        except Exception:
            pass
        detector_results: list[DetectorResult] = []

        wall_results = self.wall_detector.analyze_key(
            snapshots=snapshots,
            key=key,
        )
        detector_results.extend(wall_results)

        pull_results = self.pull_detector.analyze_key(
            key=key,
            current_mid_price=current_mid_price,
        )
        detector_results.extend(pull_results)

        return detector_results

    def _run_optional_detectors(
        self,
        *,
        tracked_walls: list[TrackedWall],
        key: SpoofingKey,
        current_mid_price: float | None,
    ) -> list[DetectorResult]:
        """
        Запускає advanced detector-и, якщо вони підключені.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_run_optional_detectors", _analytics_args)
        except Exception:
            pass
        detector_results: list[DetectorResult] = []

        if self.fake_liquidity_detector is not None:
            detector_results.extend(
                self._safe_run_detector(
                    detector=self.fake_liquidity_detector,
                    tracked_walls=tracked_walls,
                    key=key,
                    current_mid_price=current_mid_price,
                    detector_name="fake_liquidity_detector",
                )
            )

        if self.flip_pressure_detector is not None:
            detector_results.extend(
                self._safe_run_detector(
                    detector=self.flip_pressure_detector,
                    tracked_walls=tracked_walls,
                    key=key,
                    current_mid_price=current_mid_price,
                    detector_name="flip_pressure_detector",
                )
            )

        if self.layering_detector is not None:
            detector_results.extend(
                self._safe_run_detector(
                    detector=self.layering_detector,
                    tracked_walls=tracked_walls,
                    key=key,
                    current_mid_price=current_mid_price,
                    detector_name="layering_detector",
                )
            )

        return detector_results

    def _safe_run_detector(
        self,
        *,
        detector: SupportsTrackedWallAnalyzeMany,
        tracked_walls: list[TrackedWall],
        key: SpoofingKey,
        current_mid_price: float | None,
        detector_name: str,
    ) -> list[DetectorResult]:
        """
        Захищений запуск optional detector-а через analyze_many().
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_safe_run_detector", _analytics_args)
        except Exception:
            pass
        try:
            result = detector.analyze_many(
                tracked_walls,
                key=key,
                current_mid_price=current_mid_price,
            )

            return [
                item
                for item in result
                if isinstance(item, DetectorResult) and item.is_positive()
            ]

        except Exception as exc:
            self.log_exception(
                "Detector failed",
                detector_name=detector_name,
                error=str(exc),
                key=spoofing_key_to_dict(key),
            )
            return []

    def _build_signal(
        self,
        *,
        detector_results: list[DetectorResult],
        key: SpoofingKey,
    ) -> SpoofingSignal | None:
        """
        Будує фінальний signal через score engine.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_signal", _analytics_args)
        except Exception:
            pass
        if not detector_results:
            return None

        return self.score_engine.build_signal(
            detector_results=detector_results,
            key=key,
            status=SpoofingStatus.DETECTED,
        )

    def _limit_detector_results(
        self,
        detector_results: list[DetectorResult],
    ) -> list[DetectorResult]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_limit_detector_results", _analytics_args)
        except Exception:
            pass
        if not detector_results:
            return []

        ordered = sorted(
            detector_results,
            key=lambda item: (item.score, item.confidence),
            reverse=True,
        )

        limit = self.config.analyzer.max_detector_results_per_cycle
        if limit <= 0:
            return ordered

        return ordered[:limit]

    # -------------------------------------------------------------------------
    # Publishing
    # -------------------------------------------------------------------------

    async def _publish_outputs(
        self,
        *,
        output: AnalyzerOutput,
        lifecycle_events: list[LiquidityLifecycleEvent],
        correlation_id: str | None,
    ) -> None:
        """
        Публікує результати в EventBus.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_publish_outputs", _analytics_args)
        except Exception:
            pass
        if self.event_bus is None:
            return

        signal = output.signal

        if self.config.analyzer.publish_lifecycle_events and lifecycle_events:
            await self.emit_event(
                self.config.analyzer.event_topic_lifecycle,
                self._build_lifecycle_payload(output, lifecycle_events),
                priority=EventPriority.LOW,
                correlation_id=correlation_id,
            )

        if self.config.analyzer.publish_updates and not self.config.analyzer.publish_detected_only:
            await self.emit_event(
                self.config.analyzer.event_topic_updated,
                self._build_updated_payload(output, lifecycle_events),
                priority=EventPriority.NORMAL,
                correlation_id=correlation_id,
            )

        if signal is not None and signal.score_breakdown is not None:
            if self.config.analyzer.publish_score_updates:
                await self.emit_event(
                    self.config.analyzer.event_topic_score_updated,
                    self._build_score_payload(signal),
                    priority=EventPriority.NORMAL,
                    correlation_id=correlation_id,
                )

            if signal.score_breakdown.passed:
                await self.emit_event(
                    self.config.analyzer.event_topic_detected,
                    self._build_detected_payload(output),
                    priority=EventPriority.HIGH,
                    correlation_id=correlation_id,
                )

    async def _publish_error(
        self,
        *,
        error: Exception,
        payload: dict[str, Any],
        context: dict[str, Any],
        correlation_id: str | None = None,
    ) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_publish_error", _analytics_args)
        except Exception:
            pass
        if self.event_bus is None:
            return

        await self.emit_event(
            self.config.analyzer.event_topic_error,
            {
                "component": self.component.value,
                "error": str(error),
                "error_type": type(error).__name__,
                "payload_keys": list(payload.keys()),
                "context": context,
            },
            priority=EventPriority.HIGH,
            correlation_id=correlation_id,
        )

    def _build_lifecycle_payload(
        self,
        output: AnalyzerOutput,
        lifecycle_events: list[LiquidityLifecycleEvent],
    ) -> dict[str, Any]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_lifecycle_payload", _analytics_args)
        except Exception:
            pass
        return {
            "symbol": output.symbol,
            "exchange": output.exchange,
            "market_type": output.market_type,
            "timeframe": output.timeframe,
            "scope": spoofing_key_to_dict(output.key),
            "lifecycle_events": [
                self.serialize_dataclass(item)
                for item in lifecycle_events
            ],
            "metadata": output.metadata,
        }

    def _build_updated_payload(
        self,
        output: AnalyzerOutput,
        lifecycle_events: list[LiquidityLifecycleEvent],
    ) -> dict[str, Any]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_updated_payload", _analytics_args)
        except Exception:
            pass
        return {
            "symbol": output.symbol,
            "exchange": output.exchange,
            "market_type": output.market_type,
            "timeframe": output.timeframe,
            "scope": spoofing_key_to_dict(output.key),
            "signal": self._serialize_signal(output.signal) if output.signal is not None else None,
            "detector_results": [
                self.detector_result_payload(item)
                for item in output.detector_results
            ],
            "tracked_walls": [
                self.serialize_dataclass(item)
                for item in output.tracked_walls
            ],
            "lifecycle_events": [
                self.serialize_dataclass(item)
                for item in lifecycle_events
            ],
            "metadata": output.metadata,
        }

    def _build_detected_payload(
        self,
        output: AnalyzerOutput,
    ) -> dict[str, Any]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_detected_payload", _analytics_args)
        except Exception:
            pass
        signal = output.signal
        assert signal is not None

        return {
            "symbol": output.symbol,
            "exchange": output.exchange,
            "market_type": output.market_type,
            "timeframe": output.timeframe,
            "scope": spoofing_key_to_dict(output.key),
            "signal": self._serialize_signal(signal),
            "score": signal.score,
            "confidence": signal.confidence,
            "severity": signal.severity.value,
            "pattern": signal.pattern.value,
            "spoofing_type": signal.spoofing_type.value,
            "metadata": output.metadata,
        }

    def _build_score_payload(
        self,
        signal: SpoofingSignal,
    ) -> dict[str, Any]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_score_payload", _analytics_args)
        except Exception:
            pass
        score = signal.score_breakdown
        assert score is not None

        return {
            "signal_id": signal.signal_id,
            "symbol": signal.symbol,
            "exchange": signal.exchange,
            "market_type": signal.market_type,
            "timeframe": signal.timeframe,
            "scope": spoofing_key_to_dict(signal.key),
            "score": score.total_score,
            "confidence": score.confidence,
            "severity": score.severity.value,
            "threshold": score.threshold,
            "passed": score.passed,
            "contributions": [
                self.serialize_dataclass(item)
                for item in score.contributions
            ],
            "metadata": score.metadata,
        }

    # -------------------------------------------------------------------------
    # Payload normalization helpers
    # -------------------------------------------------------------------------

    def _extract_or_build_snapshots_from_payload(
        self,
        payload: dict[str, Any],
        *,
        key: SpoofingKey,
        allow_raw_payload: bool = False,
    ) -> list[OrderbookLevelSnapshot]:
        """
        Production:
            payload["snapshots"] або normalized/cache-level bids/asks з
            market.orderbook.updated.

        Raw bids/asks з exchange adapter дозволені тільки якщо allow_raw_payload=True.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_extract_or_build_snapshots_from_payload", _analytics_args)
        except Exception:
            pass
        if "snapshots" in payload:
            return self._normalize_snapshot_list(payload["snapshots"], key=key)

        has_book_sides = "bids" in payload or "asks" in payload
        if not has_book_sides:
            return []

        if not allow_raw_payload and payload.get("source") == "exchange_adapter":
            raise ValueError(
                "Raw exchange adapter orderbook payload is not allowed in SpoofingAnalyzer "
                "production path. Use OrderBookCache -> market.orderbook.updated."
            )

        scope = spoofing_key_to_dict(key)

        return self.wall_detector.build_snapshot_levels_from_orderbook(
            symbol=scope["symbol"],
            exchange=scope["exchange"],
            market_type=scope["market_type"],
            timeframe=scope["timeframe"],
            exchange_symbol=self._optional_str(payload.get("exchange_symbol")),
            bids=self._normalize_book_side(payload.get("bids", [])),
            asks=self._normalize_book_side(payload.get("asks", [])),
            best_bid=self._optional_float(payload.get("best_bid")),
            best_ask=self._optional_float(payload.get("best_ask")),
            sequence_id=self._optional_int(payload.get("sequence_id")),
            timestamp=self._optional_datetime(payload.get("timestamp")),
            metadata={
                **self._optional_metadata(payload.get("metadata")),
                "payload_source": payload.get("source", "data_layer_updated_event"),
            },
        )

    def _normalize_snapshot_list(
        self,
        raw_snapshots: Any,
        *,
        key: SpoofingKey,
    ) -> list[OrderbookLevelSnapshot]:
        """
        Нормалізує list[OrderbookLevelSnapshot | dict] у list[OrderbookLevelSnapshot].
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_snapshot_list", _analytics_args)
        except Exception:
            pass
        snapshots: list[OrderbookLevelSnapshot] = []

        if not isinstance(raw_snapshots, list):
            return snapshots

        scope = spoofing_key_to_dict(key)

        for item in raw_snapshots:
            if isinstance(item, OrderbookLevelSnapshot):
                if item.key == key:
                    snapshots.append(item)
                continue

            if not isinstance(item, Mapping):
                continue

            try:
                side = item["side"]
                price = self._optional_float(item["price"])
                size = self._optional_float(item["size"])
                if price is None or size is None:
                    continue
                if price <= 0.0 or size <= 0.0:
                    continue

                snapshot = self.build_level_snapshot(
                    symbol=str(item.get("symbol") or scope["symbol"]),
                    exchange=str(item.get("exchange") or scope["exchange"]),
                    market_type=str(item.get("market_type") or scope["market_type"]),
                    timeframe=str(item.get("timeframe") or scope["timeframe"]),
                    exchange_symbol=self._optional_str(item.get("exchange_symbol")),
                    side=side,
                    price=price,
                    size=size,
                    best_bid=self._optional_float(item.get("best_bid")),
                    best_ask=self._optional_float(item.get("best_ask")),
                    mid_price=self._optional_float(item.get("mid_price")),
                    spread=self._optional_float(item.get("spread")),
                    sequence_id=self._optional_int(item.get("sequence_id")),
                    timestamp=self._optional_datetime(item.get("timestamp")),
                    metadata=self._optional_metadata(item.get("metadata")),
                )
                if snapshot.key == key:
                    snapshots.append(snapshot)

            except Exception as exc:
                self.log_warning(
                    "Failed to normalize snapshot item",
                    error=str(exc),
                    item=item,
                )

        return snapshots

    async def _call_orderbook_cache_get_book(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        depth: int | None,
    ) -> Any:
        """
        Compatibility wrapper for sync/async OrderBookCache.get_book variants.

        Some cache implementations expose async get_book(...), while older ones
        are synchronous and may not accept market_type or depth. This helper
        tries the known signatures and awaits coroutine results when needed.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_call_orderbook_cache_get_book", _analytics_args)
        except Exception:
            pass
        if self.orderbook_cache is None:
            return None

        method = getattr(self.orderbook_cache, "get_book", None)
        if method is None:
            raise AttributeError("OrderBookCache does not expose get_book()")

        call_variants: list[dict[str, Any]] = [
            {
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
                "depth": depth,
            },
            {
                "exchange": exchange,
                "symbol": symbol,
                "depth": depth,
            },
            {
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
            },
            {
                "exchange": exchange,
                "symbol": symbol,
            },
            {
                "symbol": symbol,
                "depth": depth,
            },
            {
                "symbol": symbol,
            },
        ]

        last_type_error: TypeError | None = None

        for kwargs in call_variants:
            clean_kwargs = {
                key: value
                for key, value in kwargs.items()
                if value is not None
            }

            try:
                result = method(**clean_kwargs)
            except TypeError as exc:
                last_type_error = exc
                continue

            if inspect.isawaitable(result):
                result = await result

            return result

        positional_variants: list[tuple[Any, ...]] = [
            (exchange, market_type, symbol),
            (exchange, symbol),
            (symbol,),
        ]

        for args in positional_variants:
            try:
                result = method(*args)
            except TypeError as exc:
                last_type_error = exc
                continue

            if inspect.isawaitable(result):
                result = await result

            return result

        if last_type_error is not None:
            raise last_type_error

        raise RuntimeError("Unable to call OrderBookCache.get_book()")

    async def _load_snapshots_from_orderbook_cache(
        self,
        key: SpoofingKey,
    ) -> list[OrderbookLevelSnapshot]:
        """
        Best-effort read-only access до OrderBookCache без жорсткої залежності
        analytics -> data.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_load_snapshots_from_orderbook_cache", _analytics_args)
        except Exception:
            pass
        if self.orderbook_cache is None:
            return []

        scope = spoofing_key_to_dict(key)

        try:
            book = await self._call_orderbook_cache_get_book(
                exchange=scope["exchange"],
                market_type=scope["market_type"],
                symbol=scope["symbol"],
                depth=self.config.wall_detection.max_levels_to_scan,
            )
        except Exception:
            self.log_exception(
                "Failed to read orderbook cache",
                key=scope,
            )
            return []

        payload = self._orderbook_cache_snapshot_to_payload(book, key=key)
        return self._extract_or_build_snapshots_from_payload(
            payload,
            key=key,
            allow_raw_payload=False,
        )

    def _orderbook_cache_snapshot_to_payload(
        self,
        book: Any,
        *,
        key: SpoofingKey,
    ) -> dict[str, Any]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_orderbook_cache_snapshot_to_payload", _analytics_args)
        except Exception:
            pass
        scope = spoofing_key_to_dict(key)

        if isinstance(book, Mapping):
            return {
                "exchange": book.get("exchange", scope["exchange"]),
                "market_type": book.get("market_type", scope["market_type"]),
                "symbol": book.get("symbol", scope["symbol"]),
                "timeframe": book.get("timeframe", scope["timeframe"]),
                "exchange_symbol": book.get("exchange_symbol"),
                "bids": book.get("bids", []),
                "asks": book.get("asks", []),
                "best_bid": book.get("best_bid"),
                "best_ask": book.get("best_ask"),
                "sequence_id": book.get("sequence_id"),
                "timestamp": book.get("timestamp"),
                "metadata": book.get("metadata", {}),
                "source": "orderbook_cache",
            }

        return {
            "exchange": getattr(book, "exchange", scope["exchange"]),
            "market_type": getattr(book, "market_type", scope["market_type"]),
            "symbol": getattr(book, "symbol", scope["symbol"]),
            "timeframe": getattr(book, "timeframe", scope["timeframe"]),
            "exchange_symbol": getattr(book, "exchange_symbol", None),
            "bids": getattr(book, "bids", []),
            "asks": getattr(book, "asks", []),
            "best_bid": getattr(book, "best_bid", None),
            "best_ask": getattr(book, "best_ask", None),
            "sequence_id": getattr(book, "sequence_id", None),
            "timestamp": getattr(book, "timestamp", None),
            "metadata": getattr(book, "metadata", {}),
            "source": "orderbook_cache",
        }

    @staticmethod
    def _normalize_book_side(raw_levels: Any) -> list[tuple[float, float]]:
        try:
            _analytics_class_name = "SpoofingAnalyzer"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_book_side", _analytics_args)
        except Exception:
            pass
        if raw_levels is None:
            return []

        if isinstance(raw_levels, (str, bytes, dict)):
            return []

        levels: list[tuple[float, float]] = []

        try:
            iterator = iter(raw_levels)
        except TypeError:
            return []

        for item in iterator:
            try:
                if isinstance(item, Mapping):
                    raw_price = item.get("price")
                    raw_size = item.get("size", item.get("qty", item.get("quantity")))
                    if raw_price is None or raw_size is None:
                        continue
                    price = float(raw_price)
                    size = float(raw_size)
                else:
                    price = float(item[0])
                    size = float(item[1])

                if (
                    math.isfinite(price)
                    and math.isfinite(size)
                    and price > 0.0
                    and size > 0.0
                ):
                    levels.append((price, size))
            except Exception:
                continue

        return levels

    def _fallback_key_from_payload(self, payload: Mapping[str, Any]) -> SpoofingKey:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_fallback_key_from_payload", _analytics_args)
        except Exception:
            pass
        exchange = payload.get("exchange") or self.config.default_exchange or "unknown"
        symbol = payload.get("symbol") or "UNKNOWN"
        market_type = payload.get("market_type") or self.config.default_market_type
        timeframe = payload.get("timeframe") or self.config.default_timeframe
        return self.make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        try:
            _analytics_class_name = "SpoofingAnalyzer"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_optional_str", _analytics_args)
        except Exception:
            pass
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return str(value)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        try:
            _analytics_class_name = "SpoofingAnalyzer"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_optional_int", _analytics_args)
        except Exception:
            pass
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _optional_datetime(value: Any) -> datetime | None:
        try:
            _analytics_class_name = "SpoofingAnalyzer"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_optional_datetime", _analytics_args)
        except Exception:
            pass
        return value if isinstance(value, datetime) else None

    @staticmethod
    def _optional_metadata(value: Any) -> dict[str, Any]:
        try:
            _analytics_class_name = "SpoofingAnalyzer"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_optional_metadata", _analytics_args)
        except Exception:
            pass
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    @staticmethod
    def _safe_metadata(value: dict[str, Any] | None) -> dict[str, Any]:
        try:
            _analytics_class_name = "SpoofingAnalyzer"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_safe_metadata", _analytics_args)
        except Exception:
            pass
        return dict(value) if isinstance(value, dict) else {}

    def _extract_current_mid_price(self, payload: dict[str, Any]) -> float | None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_extract_current_mid_price", _analytics_args)
        except Exception:
            pass
        explicit_mid = self._optional_float(payload.get("current_mid_price"))
        if explicit_mid is not None and explicit_mid > 0.0:
            return explicit_mid

        explicit_mid = self._optional_float(payload.get("mid_price"))
        if explicit_mid is not None and explicit_mid > 0.0:
            return explicit_mid

        return self._resolve_mid_price_from_best_quotes(
            best_bid=payload.get("best_bid"),
            best_ask=payload.get("best_ask"),
        )

    @staticmethod
    def _resolve_mid_price_from_best_quotes(
        *,
        best_bid: Any,
        best_ask: Any,
    ) -> float | None:
        try:
            _analytics_class_name = "SpoofingAnalyzer"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_resolve_mid_price_from_best_quotes", _analytics_args)
        except Exception:
            pass
        try:
            bid = float(best_bid)
            ask = float(best_ask)
        except (TypeError, ValueError, OverflowError):
            return None

        if not math.isfinite(bid) or not math.isfinite(ask):
            return None
        if bid <= 0.0 or ask <= 0.0:
            return None

        return (bid + ask) / 2.0

    @staticmethod
    def _resolve_mid_price_from_levels(
        levels: list[OrderbookLevelSnapshot],
    ) -> float | None:
        try:
            _analytics_class_name = "SpoofingAnalyzer"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_resolve_mid_price_from_levels", _analytics_args)
        except Exception:
            pass
        for level in levels:
            mid = getattr(level, "mid_price", None)
            if mid is not None:
                try:
                    value = float(mid)
                except (TypeError, ValueError, OverflowError):
                    continue
                if math.isfinite(value) and value > 0.0:
                    return value

        for level in levels:
            best_bid = getattr(level, "best_bid", None)
            best_ask = getattr(level, "best_ask", None)
            try:
                bid = float(best_bid)
                ask = float(best_ask)
            except (TypeError, ValueError, OverflowError):
                continue

            if math.isfinite(bid) and math.isfinite(ask) and bid > 0.0 and ask > 0.0:
                return (bid + ask) / 2.0

        return None

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        try:
            _analytics_class_name = "SpoofingAnalyzer"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_optional_float", _analytics_args)
        except Exception:
            pass
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None

        if not math.isfinite(result):
            return None

        return result

    @staticmethod
    def _is_finite_positive(value: Any) -> bool:
        try:
            _analytics_class_name = "SpoofingAnalyzer"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_is_finite_positive", _analytics_args)
        except Exception:
            pass
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return False

        return math.isfinite(result) and result > 0.0

    @staticmethod
    def _has_level_contract(level: Any) -> bool:
        try:
            _analytics_class_name = "SpoofingAnalyzer"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_has_level_contract", _analytics_args)
        except Exception:
            pass
        required_attrs = (
            "key",
            "price",
            "size",
            "side",
            "timestamp",
        )
        return all(hasattr(level, attr) for attr in required_attrs)

    # -------------------------------------------------------------------------
    # Output helpers
    # -------------------------------------------------------------------------

    def _empty_output(
        self,
        *,
        key: SpoofingKey,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> AnalyzerOutput:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_empty_output", _analytics_args)
        except Exception:
            pass
        scope = spoofing_key_to_dict(key)
        return AnalyzerOutput(
            symbol=scope["symbol"],
            exchange=scope["exchange"],
            market_type=scope["market_type"],
            timeframe=scope["timeframe"],
            signal=None,
            metadata={
                "scope": scope,
                "reason": reason,
                "exchange_symbol": scope["symbol"],
                **self._safe_metadata(metadata),
            },
        )

    def get_latest_output_by_key(self, key: SpoofingKey) -> AnalyzerOutput | None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_latest_output_by_key", _analytics_args)
        except Exception:
            pass
        return self._latest_output_by_key.get(key)

    def get_latest_signal_by_key(self, key: SpoofingKey) -> SpoofingSignal | None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_latest_signal_by_key", _analytics_args)
        except Exception:
            pass
        return self._latest_signal_by_key.get(key)

    def get_latest_output(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> AnalyzerOutput | None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_latest_output", _analytics_args)
        except Exception:
            pass
        key = self.make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        return self.get_latest_output_by_key(key)

    # -------------------------------------------------------------------------
    # Serialization helpers
    # -------------------------------------------------------------------------

    def _serialize_signal(
        self,
        signal: SpoofingSignal,
    ) -> dict[str, Any]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_serialize_signal", _analytics_args)
        except Exception:
            pass
        payload = self.serialize_dataclass(signal)

        if signal.features is not None:
            payload["features"] = self.serialize_dataclass(signal.features)

        if signal.score_breakdown is not None:
            payload["score_breakdown"] = self.serialize_dataclass(signal.score_breakdown)
            payload["score_breakdown"]["contributions"] = [
                self.serialize_dataclass(item)
                for item in signal.score_breakdown.contributions
            ]

        payload["detector_results"] = [
            self.detector_result_payload(item)
            for item in signal.detector_results
        ]
        return payload

    # -------------------------------------------------------------------------
    # Debug / diagnostics
    # -------------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """
        Зведена статистика analyzer + tracker.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "stats", _analytics_args)
        except Exception:
            pass
        return {
            "component": self.component.value,
            "registered": self._registered,
            "running": getattr(self, "_running", False),
            "cleanup_job_id": self._cleanup_job_id,
            "subscriptions": len(self._subscriptions),
            "tracker": self.persistence_tracker.stats(),
            "latest_outputs": len(self._latest_output_by_key),
            "latest_signals": len(self._latest_signal_by_key),
            "config": {
                "enabled": self.config.enabled,
                "analyzer_enabled": self.config.analyzer.enabled,
                "publish_updates": self.config.analyzer.publish_updates,
                "publish_detected_only": self.config.analyzer.publish_detected_only,
                "publish_lifecycle_events": self.config.analyzer.publish_lifecycle_events,
                "publish_score_updates": self.config.analyzer.publish_score_updates,
                "production_source_topics": list(self.config.production_source_topics),
                "allow_legacy_raw_topics": self.config.analyzer.allow_legacy_raw_topics,
                "updated_topic": self.config.analyzer.event_topic_updated,
                "detected_topic": self.config.analyzer.event_topic_detected,
                "score_topic": self.config.analyzer.event_topic_score_updated,
                "lifecycle_topic": self.config.analyzer.event_topic_lifecycle,
                "error_topic": self.config.analyzer.event_topic_error,
                "scope": "exchange:market_type:symbol:timeframe",
            },
            "optional_detectors": {
                "fake_liquidity": self.fake_liquidity_detector is not None,
                "flip_pressure": self.flip_pressure_detector is not None,
                "layering": self.layering_detector is not None,
            },
            "dependencies": {
                "orderbook_cache": self.orderbook_cache is not None,
            },
        }


__all__ = ["SpoofingAnalyzer"]