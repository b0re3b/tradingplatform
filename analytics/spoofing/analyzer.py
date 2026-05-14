from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Protocol

from core.event_bus import Event, EventBus, EventPriority
from core.scheduler import Scheduler

from .base import BaseSpoofingModule
from .config import SpoofingConfig
from .enums import SpoofingComponent, SpoofingSide, SpoofingStatus
from .fake_liquidity_detector import FakeLiquidityDetector
from .flip_pressure_detector import FlipPressureDetector
from .layering_detector import LayeringDetector
from .models import (
    AnalyzerOutput,
    DetectorResult,
    LiquidityLifecycleEvent,
    OrderbookLevelSnapshot,
    SpoofingSignal,
    TrackedWall,
)
from .order_pull_detector import OrderPullDetector
from .orderbook_wall_detector import OrderbookWallDetector
from .persistence_tracker import PersistenceTracker
from .spoofing_score import SpoofingScoreEngine


class SupportsTrackedWallAnalyzeMany(Protocol):
    def analyze_many(
        self,
        walls: Iterable[TrackedWall],
        *,
        exchange: str | None = None,
        symbol: str | None = None,
        current_mid_price: float | None = None,
    ) -> list[DetectorResult]: ...


class SpoofingAnalyzer(BaseSpoofingModule):
    """
    Центральний orchestrator для analytics.spoofing.

    Відповідає за:
    - підписку на market.orderbook events через EventBus;
    - нормалізацію raw orderbook payload у OrderbookLevelSnapshot;
    - оновлення PersistenceTracker;
    - запуск wall / pull / advanced detector-ів;
    - агрегацію результатів через SpoofingScoreEngine;
    - публікацію analytics.spoofing.* подій;
    - реєстрацію periodic cleanup через Scheduler.

    Важливо:
    - analyzer не містить складну spoofing-логіку;
    - detector/scoring/tracker логіка лишається в окремих класах;
    - analyzer є integration boundary між EventBus/Scheduler і доменною логікою.
    """

    component = SpoofingComponent.ANALYZER

    def __init__(
        self,
        *,
        event_bus: EventBus | None,
        scheduler: Scheduler | None,
        config: SpoofingConfig,
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

        self._cleanup_job_id: str | None = None
        self._registered = False

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
        - EventBus subscription на orderbook topic;
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

        self.log_info(
            "SpoofingAnalyzer registered",
            orderbook_topic=self.config.analyzer.event_topic_orderbook,
            cleanup_job_id=self._cleanup_job_id,
            publish_updates=self.config.analyzer.publish_updates,
            publish_detected_only=self.config.analyzer.publish_detected_only,
        )

    def _register_eventbus_subscriptions(self) -> None:
        if self.event_bus is None:
            self.log_warning("EventBus subscription skipped: event_bus is None")
            return

        self.event_bus.subscribe(
            self.config.analyzer.event_topic_orderbook,
            self.on_orderbook_event,
        )

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

    async def on_orderbook_event(self, event: Event) -> None:
        """
        EventBus callback для market.orderbook.

        Очікується, що event.payload містить:
        - або snapshots;
        - або raw bids/asks + top-of-book metadata.
        """
        payload = getattr(event, "payload", None)
        if not isinstance(payload, dict):
            self.log_warning("Orderbook event payload is not a dict")
            return

        try:
            output = await self.process_event_payload(
                payload,
                correlation_id=self._extract_event_correlation_id(event),
            )

            if output.signal is not None:
                self.log_debug(
                    "Spoofing signal processed from event",
                    symbol=output.symbol,
                    exchange=output.exchange,
                    signal_id=output.signal.signal_id,
                    score=output.signal.score,
                    confidence=output.signal.confidence,
                )

        except Exception as exc:
            self.log_exception(
                "Failed to process orderbook event",
                error=str(exc),
                payload_keys=list(payload.keys()),
            )

            if self.config.analyzer.publish_errors:
                await self._publish_error(
                    error=exc,
                    payload=payload,
                    context={"handler": "on_orderbook_event"},
                )
            raise

    # -------------------------------------------------------------------------
    # Main processing API
    # -------------------------------------------------------------------------

    async def process_event_payload(
        self,
        payload: dict[str, Any],
        *,
        correlation_id: str | None = None,
    ) -> AnalyzerOutput:
        """
        Обробляє payload orderbook event.

        Підтримує:
        1. payload["snapshots"] з уже нормалізованими snapshots;
        2. raw payload із bids/asks/top-of-book metadata.
        """
        snapshots = self._extract_or_build_snapshots_from_payload(payload)

        return await self.process_snapshots(
            snapshots=snapshots,
            symbol=self._optional_str(payload.get("symbol")),
            exchange=self._optional_str(payload.get("exchange")),
            current_mid_price=self._extract_current_mid_price(payload),
            correlation_id=correlation_id,
            metadata={
                "event_payload": {
                    "symbol": payload.get("symbol"),
                    "exchange": payload.get("exchange"),
                    "sequence_id": payload.get("sequence_id"),
                }
            },
        )

    async def process_orderbook(
        self,
        *,
        symbol: str,
        exchange: str,
        bids: Iterable[tuple[float, float]],
        asks: Iterable[tuple[float, float]],
        best_bid: float | None = None,
        best_ask: float | None = None,
        sequence_id: int | None = None,
        timestamp: datetime | None = None,
        current_mid_price: float | None = None,
        metadata: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> AnalyzerOutput:
        """
        Helper для прямої роботи із сирими bids/asks без EventBus.
        """
        snapshots = self.wall_detector.build_snapshot_levels_from_orderbook(
            symbol=symbol,
            exchange=exchange,
            bids=bids,
            asks=asks,
            best_bid=best_bid,
            best_ask=best_ask,
            sequence_id=sequence_id,
            timestamp=timestamp,
            metadata=metadata,
        )

        resolved_mid = current_mid_price
        if resolved_mid is None and best_bid is not None and best_ask is not None:
            if best_bid > 0 and best_ask > 0:
                resolved_mid = (best_bid + best_ask) / 2.0

        return await self.process_snapshots(
            snapshots=snapshots,
            symbol=symbol,
            exchange=exchange,
            current_mid_price=resolved_mid,
            metadata=metadata,
            correlation_id=correlation_id,
        )

    async def process_snapshots(
        self,
        *,
        snapshots: Iterable[OrderbookLevelSnapshot],
        symbol: str | None = None,
        exchange: str | None = None,
        current_mid_price: float | None = None,
        metadata: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> AnalyzerOutput:
        """
        Основна точка входу для вже нормалізованих orderbook snapshots.
        """
        if not self.config.enabled or not self.config.analyzer.enabled:
            return AnalyzerOutput(
                symbol=symbol or "unknown",
                exchange=exchange or "unknown",
                signal=None,
                metadata={"reason": "analyzer_disabled", **self._safe_metadata(metadata)},
            )

        levels = list(snapshots)
        if not levels:
            return AnalyzerOutput(
                symbol=symbol or "unknown",
                exchange=exchange or "unknown",
                signal=None,
                metadata={"reason": "empty_snapshots", **self._safe_metadata(metadata)},
            )

        resolved_symbol, resolved_exchange = self._resolve_market_identity(
            levels=levels,
            symbol=symbol,
            exchange=exchange,
        )
        resolved_mid_price = current_mid_price or self._resolve_mid_price_from_levels(levels)

        scoped_levels = self._filter_levels_by_market(
            levels=levels,
            symbol=resolved_symbol,
            exchange=resolved_exchange,
        )
        if not scoped_levels:
            return AnalyzerOutput(
                symbol=resolved_symbol,
                exchange=resolved_exchange,
                signal=None,
                metadata={"reason": "no_levels_for_resolved_market", **self._safe_metadata(metadata)},
            )

        self.persistence_tracker.maybe_cleanup(now=scoped_levels[0].timestamp)

        filtered_levels = self._filter_levels_for_analysis(scoped_levels)
        if not filtered_levels:
            return AnalyzerOutput(
                symbol=resolved_symbol,
                exchange=resolved_exchange,
                signal=None,
                metadata={"reason": "no_levels_after_filtering", **self._safe_metadata(metadata)},
            )

        tracked_walls, lifecycle_events = self._update_tracker(filtered_levels)

        detector_results = self._run_base_detectors(
            snapshots=filtered_levels,
            exchange=resolved_exchange,
            symbol=resolved_symbol,
            current_mid_price=resolved_mid_price,
        )

        detector_results.extend(
            self._run_optional_detectors(
                tracked_walls=self.persistence_tracker.get_walls_for_symbol(
                    exchange=resolved_exchange,
                    symbol=resolved_symbol,
                ),
                exchange=resolved_exchange,
                symbol=resolved_symbol,
                current_mid_price=resolved_mid_price,
            )
        )

        detector_results = self._limit_detector_results(detector_results)

        signal = self._build_signal(
            detector_results=detector_results,
            exchange=resolved_exchange,
            symbol=resolved_symbol,
        )

        output = AnalyzerOutput(
            symbol=resolved_symbol,
            exchange=resolved_exchange,
            signal=signal,
            detector_results=detector_results,
            tracked_walls=self.persistence_tracker.snapshot_state(
                exchange=resolved_exchange,
                symbol=resolved_symbol,
            ),
            lifecycle_events=lifecycle_events,
            metadata={
                "lifecycle_events_count": len(lifecycle_events),
                "input_levels_count": len(levels),
                "filtered_levels_count": len(filtered_levels),
                "current_mid_price": resolved_mid_price,
                **self._safe_metadata(metadata),
            },
        )

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

        Scheduler не отримує цей method напряму, бо він повертає int,
        а core.scheduler очікує job-callback із return type None або Awaitable[None].
        """
        expired_count = self.persistence_tracker.cleanup()
        self.log_debug(
            "SpoofingAnalyzer cleanup completed",
            expired_count=expired_count,
        )
        return expired_count

    async def cleanup_job(self) -> None:
        """
        Async-safe Scheduler callback із сумісним return type.
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

    @staticmethod
    def _resolve_market_identity(
        *,
        levels: list[OrderbookLevelSnapshot],
        symbol: str | None,
        exchange: str | None,
    ) -> tuple[str, str]:
        resolved_symbol = symbol if symbol is not None and symbol else levels[0].symbol
        resolved_exchange = exchange if exchange is not None and exchange else levels[0].exchange
        return resolved_symbol, resolved_exchange

    @staticmethod
    def _filter_levels_by_market(
        *,
        levels: list[OrderbookLevelSnapshot],
        symbol: str,
        exchange: str,
    ) -> list[OrderbookLevelSnapshot]:
        return [
            level
            for level in levels
            if level.symbol == symbol and level.exchange == exchange
        ]

    def _filter_levels_for_analysis(
        self,
        levels: list[OrderbookLevelSnapshot],
    ) -> list[OrderbookLevelSnapshot]:
        """
        Попередній фільтр рівнів.

        Залишає:
        - валідні bid/ask рівні;
        - top-N levels per side according to wall_detection.max_levels_to_scan.
        """
        valid = [
            level
            for level in levels
            if level.price > 0
            and level.size > 0
            and level.side in {SpoofingSide.BID, SpoofingSide.ASK}
            and self.config.is_symbol_allowed(level.symbol)
            and (self.config.exchange is None or level.exchange == self.config.exchange)
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
        Оновлює PersistenceTracker по релевантних рівнях.
        """
        tracked_walls, lifecycle_events = self.persistence_tracker.upsert_many(levels)
        return tracked_walls, lifecycle_events

    def _run_base_detectors(
        self,
        *,
        snapshots: list[OrderbookLevelSnapshot],
        exchange: str,
        symbol: str,
        current_mid_price: float | None,
    ) -> list[DetectorResult]:
        """
        Запускає базові detector-и.
        """
        detector_results: list[DetectorResult] = []

        wall_results = self.wall_detector.analyze_many(
            snapshots=snapshots,
            symbol=symbol,
            exchange=exchange,
        )
        detector_results.extend(wall_results)

        symbol_walls = self.persistence_tracker.get_walls_for_symbol(
            exchange=exchange,
            symbol=symbol,
        )
        pull_results = self.pull_detector.analyze_many(
            walls=symbol_walls,
            exchange=exchange,
            symbol=symbol,
            current_mid_price=current_mid_price,
        )
        detector_results.extend(pull_results)

        return detector_results

    def _run_optional_detectors(
        self,
        *,
        tracked_walls: list[TrackedWall],
        exchange: str,
        symbol: str,
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
                    exchange=exchange,
                    symbol=symbol,
                    current_mid_price=current_mid_price,
                    detector_name="fake_liquidity_detector",
                )
            )

        if self.flip_pressure_detector is not None:
            detector_results.extend(
                self._safe_run_detector(
                    detector=self.flip_pressure_detector,
                    tracked_walls=tracked_walls,
                    exchange=exchange,
                    symbol=symbol,
                    current_mid_price=current_mid_price,
                    detector_name="flip_pressure_detector",
                )
            )

        if self.layering_detector is not None:
            detector_results.extend(
                self._safe_run_detector(
                    detector=self.layering_detector,
                    tracked_walls=tracked_walls,
                    exchange=exchange,
                    symbol=symbol,
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
        exchange: str,
        symbol: str,
        current_mid_price: float | None,
        detector_name: str,
    ) -> list[DetectorResult]:
        """
        Захищений запуск detector-а через analyze_many().
        """
        try:
            result = detector.analyze_many(
                tracked_walls,
                exchange=exchange,
                symbol=symbol,
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
            )
            return []

    def _build_signal(
        self,
        *,
        detector_results: list[DetectorResult],
        exchange: str,
        symbol: str,
    ) -> SpoofingSignal | None:
        """
        Будує фінальний signal через score engine.
        """
        if not detector_results:
            return None

        return self.score_engine.build_signal(
            detector_results=detector_results,
            exchange=exchange,
            symbol=symbol,
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
        )

    def _build_lifecycle_payload(
        self,
        output: AnalyzerOutput,
        lifecycle_events: list[LiquidityLifecycleEvent],
    ) -> dict[str, Any]:
        return {
            "symbol": output.symbol,
            "exchange": output.exchange,
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
    ) -> list[OrderbookLevelSnapshot]:
        """
        Підтримує:
        - payload["snapshots"] -> list[OrderbookLevelSnapshot | dict];
        - payload["bids"], payload["asks"] -> raw orderbook tuples/lists.
        """
        if "snapshots" in payload:
            return self._normalize_snapshot_list(payload["snapshots"])

        required = {"symbol", "exchange"}
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Missing required orderbook payload keys: {missing}")

        return self.wall_detector.build_snapshot_levels_from_orderbook(
            symbol=str(payload["symbol"]),
            exchange=str(payload["exchange"]),
            bids=self._normalize_book_side(payload.get("bids", [])),
            asks=self._normalize_book_side(payload.get("asks", [])),
            best_bid=self._optional_float(payload.get("best_bid")),
            best_ask=self._optional_float(payload.get("best_ask")),
            sequence_id=self._optional_int(payload.get("sequence_id")),
            timestamp=self._optional_datetime(payload.get("timestamp")),
            metadata=self._optional_metadata(payload.get("metadata")),
        )

    def _normalize_snapshot_list(
        self,
        raw_snapshots: Any,
    ) -> list[OrderbookLevelSnapshot]:
        """
        Нормалізує list[OrderbookLevelSnapshot | dict] у list[OrderbookLevelSnapshot].
        """
        snapshots: list[OrderbookLevelSnapshot] = []

        if not isinstance(raw_snapshots, list):
            return snapshots

        for item in raw_snapshots:
            if isinstance(item, OrderbookLevelSnapshot):
                snapshots.append(item)
                continue

            if not isinstance(item, dict):
                continue

            try:
                snapshot = self.build_level_snapshot(
                    symbol=str(item["symbol"]),
                    exchange=str(item["exchange"]),
                    side=item["side"],
                    price=self.safe_float(item["price"]),
                    size=self.safe_float(item["size"]),
                    best_bid=self._optional_float(item.get("best_bid")),
                    best_ask=self._optional_float(item.get("best_ask")),
                    mid_price=self._optional_float(item.get("mid_price")),
                    spread=self._optional_float(item.get("spread")),
                    sequence_id=self._optional_int(item.get("sequence_id")),
                    timestamp=self._optional_datetime(item.get("timestamp")),
                    metadata=self._optional_metadata(item.get("metadata")),
                )
                snapshots.append(snapshot)

            except Exception as exc:
                self.log_warning(
                    "Failed to normalize snapshot item",
                    error=str(exc),
                    item=item,
                )

        return snapshots

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
                if isinstance(item, dict):
                    raw_price = item.get("price")
                    raw_size = item.get("size", item.get("qty", item.get("quantity")))
                    if raw_price is None or raw_size is None:
                        continue
                    price = float(raw_price)
                    size = float(raw_size)
                else:
                    price = float(item[0])
                    size = float(item[1])

                if price > 0.0 and size > 0.0:
                    levels.append((price, size))
            except Exception:
                continue

        return levels

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
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_datetime(value: Any) -> datetime | None:
        return value if isinstance(value, datetime) else None

    @staticmethod
    def _optional_metadata(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return dict(value)
        return None

    @staticmethod
    def _safe_metadata(value: dict[str, Any] | None) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    def _extract_current_mid_price(self, payload: dict[str, Any]) -> float | None:
        explicit_mid = self._optional_float(payload.get("current_mid_price"))
        if explicit_mid is not None:
            return explicit_mid

        explicit_mid = self._optional_float(payload.get("mid_price"))
        if explicit_mid is not None:
            return explicit_mid

        best_bid = self._optional_float(payload.get("best_bid"))
        best_ask = self._optional_float(payload.get("best_ask"))
        if best_bid is not None and best_ask is not None and best_bid > 0 and best_ask > 0:
            return (best_bid + best_ask) / 2.0

        return None

    @staticmethod
    def _resolve_mid_price_from_levels(
        levels: list[OrderbookLevelSnapshot],
    ) -> float | None:
        for level in levels:
            if level.mid_price is not None and level.mid_price > 0:
                return level.mid_price

        for level in levels:
            if (
                level.best_bid is not None
                and level.best_ask is not None
                and level.best_bid > 0
                and level.best_ask > 0
            ):
                return (level.best_bid + level.best_ask) / 2.0

        return None

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

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
            "cleanup_job_id": self._cleanup_job_id,
            "tracker": self.persistence_tracker.stats(),
            "config": {
                "enabled": self.config.enabled,
                "analyzer_enabled": self.config.analyzer.enabled,
                "publish_updates": self.config.analyzer.publish_updates,
                "publish_detected_only": self.config.analyzer.publish_detected_only,
                "publish_lifecycle_events": self.config.analyzer.publish_lifecycle_events,
                "publish_score_updates": self.config.analyzer.publish_score_updates,
                "orderbook_topic": self.config.analyzer.event_topic_orderbook,
                "updated_topic": self.config.analyzer.event_topic_updated,
                "detected_topic": self.config.analyzer.event_topic_detected,
                "score_topic": self.config.analyzer.event_topic_score_updated,
                "lifecycle_topic": self.config.analyzer.event_topic_lifecycle,
                "error_topic": self.config.analyzer.event_topic_error,
            },
            "optional_detectors": {
                "fake_liquidity": self.fake_liquidity_detector is not None,
                "flip_pressure": self.flip_pressure_detector is not None,
                "layering": self.layering_detector is not None,
            },
        }


__all__ = ["SpoofingAnalyzer"]