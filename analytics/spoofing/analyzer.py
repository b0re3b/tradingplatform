from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Iterable, Mapping, Protocol

from core.event_bus import Event, EventBus, EventPriority, Subscription
from core.scheduler import Scheduler

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
        persistence_tracker: PersistenceTracker | None = None,
        wall_detector: OrderbookWallDetector | None = None,
        pull_detector: OrderPullDetector | None = None,
        score_engine: SpoofingScoreEngine | None = None,
        fake_liquidity_detector: FakeLiquidityDetector | None = None,
        flip_pressure_detector: FlipPressureDetector | None = None,
        layering_detector: LayeringDetector | None = None,
    ) -> None:
        super().__init__(
            event_bus=event_bus,
            scheduler=scheduler,
            config=config,
        )

        self.config.validate()
        self.orderbook_cache = orderbook_cache

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
        if not self.config.enabled or not self.config.fake_liquidity.enabled:
            return None
        return FakeLiquidityDetector(
            event_bus=self.event_bus,
            scheduler=self.scheduler,
            config=self.config,
            persistence_tracker=self.persistence_tracker,
        )

    def _create_flip_pressure_detector_if_enabled(self) -> FlipPressureDetector | None:
        if not self.config.enabled or not self.config.flip_pressure.enabled:
            return None
        return FlipPressureDetector(
            event_bus=self.event_bus,
            scheduler=self.scheduler,
            config=self.config,
            persistence_tracker=self.persistence_tracker,
        )

    def _create_layering_detector_if_enabled(self) -> LayeringDetector | None:
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

    def stop(self) -> None:
        """
        Зупиняє analyzer lifecycle:
        - unsubscribe всіх EventBus subscriptions;
        - disable cleanup Scheduler job;
        - синхронізує _registered і _running.

        Метод навмисно sync, як і register(), щоб відповідати поточному
        core-style lifecycle у пакеті.
        """
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

        self.log_info("SpoofingAnalyzer stopped")

    def _register_eventbus_subscriptions(self) -> None:
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

        snapshots = self._load_snapshots_from_orderbook_cache(key)
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
            snapshots = self._load_snapshots_from_orderbook_cache(resolved_key)

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
        self.cleanup()
        return None

    # -------------------------------------------------------------------------
    # Tracker + detectors
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_event_correlation_id(event: Event) -> str | None:
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

    def _load_snapshots_from_orderbook_cache(
        self,
        key: SpoofingKey,
    ) -> list[OrderbookLevelSnapshot]:
        """
        Best-effort read-only access до OrderBookCache без жорсткої залежності
        analytics -> data.
        """
        if self.orderbook_cache is None:
            return []

        scope = spoofing_key_to_dict(key)

        try:
            book = self.orderbook_cache.get_book(
                exchange=scope["exchange"],
                market_type=scope["market_type"],
                symbol=scope["symbol"],
                depth=self.config.wall_detection.max_levels_to_scan,
            )
        except TypeError:
            try:
                book = self.orderbook_cache.get_book(
                    exchange=scope["exchange"],
                    symbol=scope["symbol"],
                    depth=self.config.wall_detection.max_levels_to_scan,
                )
            except Exception:
                self.log_exception(
                    "Failed to read orderbook cache",
                    key=scope,
                )
                return []
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
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return str(value)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _optional_datetime(value: Any) -> datetime | None:
        return value if isinstance(value, datetime) else None

    @staticmethod
    def _optional_metadata(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    @staticmethod
    def _safe_metadata(value: dict[str, Any] | None) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    def _extract_current_mid_price(self, payload: dict[str, Any]) -> float | None:
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
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return False

        return math.isfinite(result) and result > 0.0

    @staticmethod
    def _has_level_contract(level: Any) -> bool:
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
        return self._latest_output_by_key.get(key)

    def get_latest_signal_by_key(self, key: SpoofingKey) -> SpoofingSignal | None:
        return self._latest_signal_by_key.get(key)

    def get_latest_output(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> AnalyzerOutput | None:
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