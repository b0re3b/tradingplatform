from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from .base import BaseSpoofingModule
from .config import SpoofingConfig
from .enums import (
    SpoofingComponent,
    SpoofingStatus,
)
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


class SpoofingAnalyzer(BaseSpoofingModule):
    """
    Центральний orchestrator для пакета analytics.spoofing.

    Основні задачі:
    - приймати нормалізовані orderbook snapshots
    - оновлювати PersistenceTracker
    - запускати wall detector
    - запускати pull detector
    - агрегувати detector results у фінальний spoofing signal
    - публікувати події в EventBus

    Архітектурний принцип:
    - analyzer НЕ містить складну spoofing-логіку всередині себе
    - analyzer координує інші модулі
    - вся аналітика лишається в detector/scoring/tracker класах
    """

    component = SpoofingComponent.ANALYZER

    def __init__(
        self,
        event_bus: Any | None,
        config: SpoofingConfig,
        persistence_tracker: PersistenceTracker | None = None,
        wall_detector: OrderbookWallDetector | None = None,
        pull_detector: OrderPullDetector | None = None,
        score_engine: SpoofingScoreEngine | None = None,
        fake_liquidity_detector: Any | None = None,
        flip_pressure_detector: Any | None = None,
        layering_detector: Any | None = None,
    ) -> None:
        super().__init__(event_bus=event_bus, config=config)

        self.persistence_tracker = persistence_tracker or PersistenceTracker(
            event_bus=event_bus,
            config=config,
        )
        self.wall_detector = wall_detector or OrderbookWallDetector(
            event_bus=event_bus,
            config=config,
            persistence_tracker=self.persistence_tracker,
        )
        self.pull_detector = pull_detector or OrderPullDetector(
            event_bus=event_bus,
            config=config,
            persistence_tracker=self.persistence_tracker,
        )
        self.score_engine = score_engine or SpoofingScoreEngine(
            event_bus=event_bus,
            config=config,
        )

        # optional advanced detectors
        self.fake_liquidity_detector = fake_liquidity_detector
        self.flip_pressure_detector = flip_pressure_detector
        self.layering_detector = layering_detector

    # -------------------------------------------------------------------------
    # EventBus registration
    # -------------------------------------------------------------------------

    def register(self) -> None:
        """
        Реєструє analyzer на orderbook events.
        """
        if self.event_bus is None:
            self.log_warning("SpoofingAnalyzer register skipped: event_bus is None")
            return

        topic = self.config.analyzer.event_topic_orderbook
        self.event_bus.subscribe(topic, self.on_orderbook_event)

        self.log_info(
            "SpoofingAnalyzer registered",
            topic=topic,
            publish_updates=self.config.analyzer.publish_updates,
            publish_detected_only=self.config.analyzer.publish_detected_only,
        )

    async def on_orderbook_event(self, event: Any) -> None:
        """
        EventBus callback для market.orderbook.

        Очікується, що event.payload вже містить нормалізовані або легко
        нормалізовані дані книги.
        """
        payload = getattr(event, "payload", None)
        if not isinstance(payload, dict):
            self.log_warning("Orderbook event payload is not a dict")
            return

        try:
            output = await self.process_event_payload(payload)
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
            self.log_error(
                "Failed to process orderbook event",
                error=str(exc),
                payload_keys=list(payload.keys()),
            )
            raise

    # -------------------------------------------------------------------------
    # Main processing API
    # -------------------------------------------------------------------------

    async def process_event_payload(
        self,
        payload: dict[str, Any],
    ) -> AnalyzerOutput:
        """
        Обробляє payload orderbook event.

        Підтримує два варіанти:
        1. payload already contains normalized snapshots
        2. payload contains raw bids/asks and top-of-book metadata
        """
        snapshots = self._extract_or_build_snapshots_from_payload(payload)
        return await self.process_snapshots(
            snapshots=snapshots,
            symbol=payload.get("symbol"),
            exchange=payload.get("exchange"),
            metadata={
                "event_payload": {
                    "symbol": payload.get("symbol"),
                    "exchange": payload.get("exchange"),
                    "sequence_id": payload.get("sequence_id"),
                }
            },
        )

    async def process_snapshots(
        self,
        *,
        snapshots: Iterable[OrderbookLevelSnapshot],
        symbol: str | None = None,
        exchange: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AnalyzerOutput:
        """
        Основна точка входу для вже нормалізованих orderbook snapshots.
        """
        if not self.config.enabled or not self.config.analyzer.enabled:
            self.log_debug("SpoofingAnalyzer disabled")
            return AnalyzerOutput(
                symbol=symbol or "unknown",
                exchange=exchange or "unknown",
                signal=None,
                metadata={"reason": "analyzer_disabled"},
            )

        levels = list(snapshots)
        if not levels:
            return AnalyzerOutput(
                symbol=symbol or "unknown",
                exchange=exchange or "unknown",
                signal=None,
                metadata={"reason": "empty_snapshots"},
            )

        resolved_symbol = symbol or levels[0].symbol
        resolved_exchange = exchange or levels[0].exchange

        self.persistence_tracker.maybe_cleanup(now=levels[0].timestamp)

        filtered_levels = self._filter_levels_for_analysis(levels)
        if not filtered_levels:
            return AnalyzerOutput(
                symbol=resolved_symbol,
                exchange=resolved_exchange,
                signal=None,
                metadata={"reason": "no_levels_after_filtering"},
            )

        tracked_walls, lifecycle_events = self._update_tracker(filtered_levels)
        detector_results = self._run_base_detectors(
            snapshots=filtered_levels,
            tracked_walls=tracked_walls,
            exchange=resolved_exchange,
            symbol=resolved_symbol,
        )

        detector_results.extend(
            self._run_optional_detectors(
                snapshots=filtered_levels,
                tracked_walls=tracked_walls,
                exchange=resolved_exchange,
                symbol=resolved_symbol,
            )
        )

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
            metadata={
                "lifecycle_events_count": len(lifecycle_events),
                "input_levels_count": len(levels),
                "filtered_levels_count": len(filtered_levels),
                **(metadata or {}),
            },
        )

        await self._publish_outputs(
            output=output,
            lifecycle_events=lifecycle_events,
        )

        return output

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
        metadata: dict[str, Any] | None = None,
    ) -> AnalyzerOutput:
        """
        Helper для роботи із сирими bids/asks.
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
        return await self.process_snapshots(
            snapshots=snapshots,
            symbol=symbol,
            exchange=exchange,
            metadata=metadata,
        )

    # -------------------------------------------------------------------------
    # Tracker + detectors
    # -------------------------------------------------------------------------

    def _filter_levels_for_analysis(
        self,
        levels: list[OrderbookLevelSnapshot],
    ) -> list[OrderbookLevelSnapshot]:
        """
        Попередній фільтр рівнів.

        Залишаємо лише:
        - валідні значення
        - top-N levels per side according to config
        """
        valid = [
            level for level in levels
            if level.price > 0
            and level.size > 0
            and level.side.value in {"bid", "ask"}
        ]

        if not valid:
            return []

        bids = sorted(
            [level for level in valid if level.side.value == "bid"],
            key=lambda item: item.price,
            reverse=True,
        )
        asks = sorted(
            [level for level in valid if level.side.value == "ask"],
            key=lambda item: item.price,
        )

        max_levels = self.config.wall_detection.max_levels_to_scan
        return bids[:max_levels] + asks[:max_levels]

    def _update_tracker(
        self,
        levels: list[OrderbookLevelSnapshot],
    ) -> tuple[list[TrackedWall], list[LiquidityLifecycleEvent]]:
        """
        Оновлює PersistenceTracker по релевантних рівнях.
        """
        tracked_walls: list[TrackedWall] = []
        lifecycle_events: list[LiquidityLifecycleEvent] = []

        for level in levels:
            wall, events = self.persistence_tracker.upsert_snapshot(level)
            tracked_walls.append(wall)
            lifecycle_events.extend(events)

            # Якщо рівень фактично зник, відмічаємо як pulled.
            if level.size <= self.config.persistence.size_update_epsilon:
                pulled_wall, pulled_event = self.persistence_tracker.mark_pulled(
                    exchange=level.exchange,
                    symbol=level.symbol,
                    side=level.side,
                    price=level.price,
                    timestamp=level.timestamp,
                    metadata={"reason": "zero_size_snapshot"},
                )
                if pulled_wall is not None:
                    tracked_walls.append(pulled_wall)
                if pulled_event is not None:
                    lifecycle_events.append(pulled_event)

        return tracked_walls, lifecycle_events

    def _run_base_detectors(
        self,
        *,
        snapshots: list[OrderbookLevelSnapshot],
        tracked_walls: list[TrackedWall],
        exchange: str,
        symbol: str,
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
        )
        detector_results.extend(pull_results)

        return detector_results

    def _run_optional_detectors(
        self,
        *,
        snapshots: list[OrderbookLevelSnapshot],
        tracked_walls: list[TrackedWall],
        exchange: str,
        symbol: str,
    ) -> list[DetectorResult]:
        """
        Запускає додаткові detector-и, якщо вони підключені.

        Тут закладена м’яка інтеграція:
        - модуль може мати analyze_many(...)
        - або analyze_symbol(...)
        """
        detector_results: list[DetectorResult] = []

        if self.fake_liquidity_detector is not None and self.config.fake_liquidity.enabled:
            detector_results.extend(
                self._safe_run_optional_detector(
                    detector=self.fake_liquidity_detector,
                    snapshots=snapshots,
                    tracked_walls=tracked_walls,
                    exchange=exchange,
                    symbol=symbol,
                    detector_name="fake_liquidity_detector",
                )
            )

        if self.flip_pressure_detector is not None and self.config.flip_pressure.enabled:
            detector_results.extend(
                self._safe_run_optional_detector(
                    detector=self.flip_pressure_detector,
                    snapshots=snapshots,
                    tracked_walls=tracked_walls,
                    exchange=exchange,
                    symbol=symbol,
                    detector_name="flip_pressure_detector",
                )
            )

        if self.layering_detector is not None and self.config.layering.enabled:
            detector_results.extend(
                self._safe_run_optional_detector(
                    detector=self.layering_detector,
                    snapshots=snapshots,
                    tracked_walls=tracked_walls,
                    exchange=exchange,
                    symbol=symbol,
                    detector_name="layering_detector",
                )
            )

        return detector_results

    def _safe_run_optional_detector(
        self,
        *,
        detector: Any,
        snapshots: list[OrderbookLevelSnapshot],
        tracked_walls: list[TrackedWall],
        exchange: str,
        symbol: str,
        detector_name: str,
    ) -> list[DetectorResult]:
        """
        Захищений запуск optional detector-ів.
        """
        try:
            if hasattr(detector, "analyze_many"):
                try:
                    result = detector.analyze_many(
                        tracked_walls,
                        exchange=exchange,
                        symbol=symbol,
                    )
                    if isinstance(result, list):
                        return result
                except TypeError:
                    result = detector.analyze_many(snapshots)
                    if isinstance(result, list):
                        return result

            if hasattr(detector, "analyze_symbol"):
                result = detector.analyze_symbol(
                    exchange=exchange,
                    symbol=symbol,
                )
                if isinstance(result, list):
                    return result

            if hasattr(detector, "analyze"):
                results: list[DetectorResult] = []
                for wall in tracked_walls:
                    try:
                        item = detector.analyze(wall)
                    except TypeError:
                        item = detector.analyze(snapshots)
                    if item is not None:
                        results.append(item)
                return results

            self.log_warning(
                "Optional detector has no supported API",
                detector_name=detector_name,
            )
            return []
        except Exception as exc:
            self.log_error(
                "Optional detector failed",
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

        signal = self.score_engine.build_signal(
            detector_results=detector_results,
            exchange=exchange,
            symbol=symbol,
            status=SpoofingStatus.DETECTED,
        )
        return signal

    # -------------------------------------------------------------------------
    # Publishing
    # -------------------------------------------------------------------------

    async def _publish_outputs(
        self,
        *,
        output: AnalyzerOutput,
        lifecycle_events: list[LiquidityLifecycleEvent],
    ) -> None:
        """
        Публікація результатів у EventBus.
        """
        if self.event_bus is None:
            return

        signal = output.signal

        if self.config.analyzer.publish_updates:
            await self.emit_event(
                self.config.analyzer.event_topic_updated,
                self._build_updated_payload(output, lifecycle_events),
            )

        if signal is not None and signal.score_breakdown is not None:
            await self.emit_event(
                self.config.analyzer.event_topic_score_updated,
                self._build_score_payload(signal),
            )

        if signal is not None and signal.score_breakdown is not None and signal.score_breakdown.passed:
            await self.emit_event(
                self.config.analyzer.event_topic_detected,
                self._build_detected_payload(output),
            )

    def _build_updated_payload(
        self,
        output: AnalyzerOutput,
        lifecycle_events: list[LiquidityLifecycleEvent],
    ) -> dict[str, Any]:
        return {
            "symbol": output.symbol,
            "exchange": output.exchange,
            "signal": self._serialize_signal(output.signal) if output.signal is not None else None,
            "detector_results": [self.detector_result_payload(item) for item in output.detector_results],
            "tracked_walls": [self.serialize_dataclass(item) for item in output.tracked_walls],
            "lifecycle_events": [self.serialize_dataclass(item) for item in lifecycle_events],
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
            "contributions": [self.serialize_dataclass(item) for item in score.contributions],
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
        - payload["snapshots"] -> list[OrderbookLevelSnapshot | dict]
        - payload["bids"], payload["asks"] -> raw orderbook tuples/lists
        """
        if "snapshots" in payload:
            return self._normalize_snapshot_list(payload["snapshots"])

        symbol = payload["symbol"]
        exchange = payload["exchange"]
        bids = payload.get("bids", [])
        asks = payload.get("asks", [])
        best_bid = payload.get("best_bid")
        best_ask = payload.get("best_ask")
        sequence_id = payload.get("sequence_id")
        timestamp = payload.get("timestamp")
        metadata = payload.get("metadata")

        return self.wall_detector.build_snapshot_levels_from_orderbook(
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
                    symbol=item["symbol"],
                    exchange=item["exchange"],
                    side=item["side"],
                    price=self.safe_float(item["price"]),
                    size=self.safe_float(item["size"]),
                    best_bid=self.safe_float(item.get("best_bid"), None) if item.get("best_bid") is not None else None,
                    best_ask=self.safe_float(item.get("best_ask"), None) if item.get("best_ask") is not None else None,
                    mid_price=self.safe_float(item.get("mid_price"), None) if item.get("mid_price") is not None else None,
                    spread=self.safe_float(item.get("spread"), None) if item.get("spread") is not None else None,
                    sequence_id=item.get("sequence_id"),
                    timestamp=item.get("timestamp"),
                    metadata=item.get("metadata"),
                )
                snapshots.append(snapshot)
            except Exception as exc:
                self.log_warning(
                    "Failed to normalize snapshot item",
                    error=str(exc),
                    item=item,
                )

        return snapshots

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
            "tracker": self.persistence_tracker.stats(),
            "config": {
                "enabled": self.config.enabled,
                "analyzer_enabled": self.config.analyzer.enabled,
                "publish_updates": self.config.analyzer.publish_updates,
                "publish_detected_only": self.config.analyzer.publish_detected_only,
                "orderbook_topic": self.config.analyzer.event_topic_orderbook,
                "updated_topic": self.config.analyzer.event_topic_updated,
                "detected_topic": self.config.analyzer.event_topic_detected,
                "score_topic": self.config.analyzer.event_topic_score_updated,
            },
            "optional_detectors": {
                "fake_liquidity": self.fake_liquidity_detector is not None,
                "flip_pressure": self.flip_pressure_detector is not None,
                "layering": self.layering_detector is not None,
            },
        }