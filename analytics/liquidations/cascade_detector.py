from __future__ import annotations

import asyncio
import contextlib
from dataclasses import asdict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Awaitable, Callable, Protocol

from core.logger import get_logger

from .config import CascadeDetectorConfig
from .enums import CascadeSeverity, LiquidationSide, LiquidationStatus
from .metrics import LiquidationMetrics
from .models import (
    CascadeDetectionResult,
    LiquidationEvent,
    LiquidationWindowStats,
)
from .state import LiquidationState, SymbolLiquidationState
from .utils import (
    build_cluster_from_events,
    clamp_float,
    compute_acceleration_ratio,
    compute_window_stats,
    ensure_utc,
    filter_events_by_side,
    infer_severity,
    normalize_score,
    prune_events_older_than,
    utc_now,
)


class EventBusProtocol(Protocol):
    async def emit(self, topic: str, event: Any) -> None:
        ...

    async def subscribe(
        self,
        topic: str,
        handler: Callable[[Any], Awaitable[None]],
    ) -> Any:
        ...


class SchedulerProtocol(Protocol):
    def add_interval_job(
        self,
        func: Any,
        seconds: int,
        *,
        name: str | None = None,
        enabled: bool = True,
        run_immediately: bool = False,
        max_retries: int = 0,
        timeout: int | None = None,
        tags: list[str] | None = None,
    ) -> Any:
        ...


class CascadeDetector:
    """
    Detector liquidation cascades поверх normalized liquidation events.

    Задачі:
    - слухати normalized liquidation events із EventBus
    - брати window подій із LiquidationState
    - рахувати side dominance / notional burst / acceleration / compaction
    - виявляти cascade або exhaustion
    - публікувати analytics signals

    Цей клас НЕ:
    - тягне WebSocket
    - не шукає liquidity zones
    - не приймає торгових рішень
    """

    DEFAULT_INPUT_TOPIC = "market.liquidation.normalized"
    DEFAULT_SNAPSHOT_TOPIC = "analytics.liquidation.detector.snapshot"

    def __init__(
        self,
        *,
        event_bus: EventBusProtocol,
        config: CascadeDetectorConfig,
        state: LiquidationState,
        metrics: LiquidationMetrics | None = None,
        scheduler: SchedulerProtocol | None = None,
        service_name: str = "cascade_detector",
    ) -> None:
        self.event_bus = event_bus
        self.config = config
        self.state = state
        self.metrics = metrics or LiquidationMetrics()
        self.scheduler = scheduler
        self.service_name = service_name

        self.logger = get_logger(
            __name__,
            service_name=service_name,
            component="analytics.liquidations.cascade_detector",
        )

        self._running = False
        self._started_at: datetime | None = None
        self._stopped_at: datetime | None = None
        self._subscription: Any = None

        self._processed_events = 0
        self._cascade_signals_emitted = 0
        self._exhaustion_signals_emitted = 0
        self._cooldown_skips = 0
        self._empty_window_skips = 0
        self._threshold_skips = 0
        self._last_signal_at: datetime | None = None
        self._last_error_at: datetime | None = None
        self._last_error: str | None = None

        self._latest_signals: list[CascadeDetectionResult] = []
        self._latest_signals_limit = 200

        self._healthcheck_job_id: str | None = None

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            self.logger.warning("CascadeDetector already running.")
            return

        if not self.config.enabled:
            self.logger.warning("CascadeDetector is disabled by config.")
            return

        self._running = True
        self._started_at = utc_now()
        self._stopped_at = None

        self._subscription = await self.event_bus.subscribe(
            self.DEFAULT_INPUT_TOPIC,
            self.on_liquidation_event,
        )

        self._register_scheduler_jobs()

        self.logger.info(
            "CascadeDetector started.",
            extra={
                "window_seconds": self.config.window_seconds,
                "min_events": self.config.min_events,
                "min_total_notional_usd": str(self.config.min_total_notional_usd),
                "min_side_imbalance_ratio": self.config.min_side_imbalance_ratio,
            },
        )

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        self._stopped_at = utc_now()

        self.logger.info(
            "CascadeDetector stopped.",
            extra=self.get_stats(),
        )

    async def restart(self) -> None:
        await self.stop()
        await self.start()

    # -------------------------------------------------------------------------
    # Main event handler
    # -------------------------------------------------------------------------

    async def on_liquidation_event(self, event: Any) -> None:
        """
        Основний вхід detector-а.
        Очікує LiquidationEvent.
        """
        if not self._running:
            return

        if not isinstance(event, LiquidationEvent):
            self.logger.debug(
                "CascadeDetector received non-LiquidationEvent payload, ignored.",
                extra={"payload_type": type(event).__name__},
            )
            return

        try:
            self._processed_events += 1

            symbol_state = self.state.get(event.exchange, event.symbol)
            if symbol_state is None or not symbol_state.events:
                self._empty_window_skips += 1
                return

            if symbol_state.is_in_cooldown(event.timestamp):
                self._cooldown_skips += 1
                return

            result = await self._detect_for_symbol_state(symbol_state, trigger_event=event)
            if result is None:
                return

            await self._emit_detection_result(result)

        except Exception as exc:
            self._last_error_at = utc_now()
            self._last_error = repr(exc)
            self.logger.exception(
                "Unhandled error in CascadeDetector.on_liquidation_event.",
                extra={
                    "exchange": event.exchange,
                    "symbol": event.symbol,
                    "error": repr(exc),
                },
            )

    async def _detect_for_symbol_state(
        self,
        symbol_state: SymbolLiquidationState,
        *,
        trigger_event: LiquidationEvent,
    ) -> CascadeDetectionResult | None:
        window_events = self._get_window_events(symbol_state, now=trigger_event.timestamp)
        if not window_events:
            self._empty_window_skips += 1
            return None

        stats = compute_window_stats(
            exchange=trigger_event.exchange,
            symbol=trigger_event.symbol,
            events=window_events,
        )

        if not self._passes_base_thresholds(stats):
            self._threshold_skips += 1
            return None

        dominant_side_events = filter_events_by_side(window_events, stats.dominant_side)
        if len(dominant_side_events) < self.config.min_events:
            self._threshold_skips += 1
            return None

        acceleration_ratio = (
            compute_acceleration_ratio(dominant_side_events)
            if self.config.acceleration_enabled
            else 0.0
        )

        intensity_score = self._compute_intensity_score(stats, acceleration_ratio=acceleration_ratio)
        severity = infer_severity(
            intensity_score=intensity_score,
            low_threshold=self.config.low_severity_threshold,
            medium_threshold=self.config.medium_severity_threshold,
            high_threshold=self.config.high_severity_threshold,
            extreme_threshold=self.config.extreme_severity_threshold,
        )

        cluster = build_cluster_from_events(
            exchange=trigger_event.exchange,
            symbol=trigger_event.symbol,
            side=stats.dominant_side,
            events=dominant_side_events,
            severity=severity,
        )
        if cluster is None:
            return None

        cluster.status = LiquidationStatus.CONFIRMED

        continuation_bias = self._compute_continuation_bias(
            stats=stats,
            acceleration_ratio=acceleration_ratio,
        )
        exhaustion_bias = self._compute_exhaustion_bias(
            stats=stats,
            acceleration_ratio=acceleration_ratio,
        )
        confidence = self._compute_confidence(
            stats=stats,
            intensity_score=intensity_score,
            acceleration_ratio=acceleration_ratio,
        )

        result = CascadeDetectionResult(
            exchange=trigger_event.exchange,
            symbol=trigger_event.symbol,
            side=stats.dominant_side,
            direction=cluster.direction,
            detected_at=utc_now(),
            cluster=cluster,
            intensity_score=intensity_score,
            confidence=confidence,
            continuation_bias=continuation_bias,
            exhaustion_bias=exhaustion_bias,
            event_count=stats.total_events,
            total_notional_usd=stats.total_notional_usd,
            window_seconds=self.config.window_seconds,
            price_range_pct=stats.price_range_pct,
            severity=severity,
            status=LiquidationStatus.CONFIRMED,
            metadata={
                "trigger_event_timestamp": trigger_event.timestamp.isoformat(),
                "side_imbalance_ratio": stats.side_imbalance_ratio,
                "dominant_side": stats.dominant_side.value,
                "acceleration_ratio": acceleration_ratio,
                "long_events": stats.long_events,
                "short_events": stats.short_events,
                "long_notional_usd": str(stats.long_notional_usd),
                "short_notional_usd": str(stats.short_notional_usd),
            },
        )

        cooldown_until = result.detected_at + timedelta(seconds=self.config.cooldown_seconds)
        symbol_state.set_cascade_detected(result.detected_at, cooldown_until)

        self._remember_signal(result)
        return result

    # -------------------------------------------------------------------------
    # Window helpers
    # -------------------------------------------------------------------------

    def _get_window_events(
        self,
        symbol_state: SymbolLiquidationState,
        *,
        now: datetime,
    ) -> list[LiquidationEvent]:
        min_ts = ensure_utc(now) - timedelta(seconds=self.config.window_seconds)
        return prune_events_older_than(list(symbol_state.events), min_timestamp=min_ts)

    def _passes_base_thresholds(self, stats: LiquidationWindowStats) -> bool:
        if stats.total_events < self.config.min_events:
            return False

        if stats.total_notional_usd < self.config.min_total_notional_usd:
            return False

        if stats.dominant_side == LiquidationSide.UNKNOWN:
            return False

        if stats.side_imbalance_ratio < self.config.min_side_imbalance_ratio:
            return False

        if self.config.price_compaction_enabled:
            if stats.price_range_pct > self.config.max_price_range_pct:
                return False

        return True

    # -------------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------------

    def _compute_intensity_score(
        self,
        stats: LiquidationWindowStats,
        *,
        acceleration_ratio: float,
    ) -> float:
        """
        Композитний intensity score [0..1].
        """
        notional_score = normalize_score(
            float(stats.total_notional_usd),
            float(self.config.min_total_notional_usd) * 3.0,
        )

        imbalance_score = clamp_float(stats.side_imbalance_ratio)

        continuation_signal = 0.0
        if self.config.price_compaction_enabled:
            if self.config.max_price_range_pct <= 0:
                continuation_signal = 0.0
            else:
                continuation_signal = clamp_float(
                    1.0 - (stats.price_range_pct / max(self.config.max_price_range_pct, 1e-9))
                )
        else:
            continuation_signal = 0.5

        acceleration_score = 0.0
        if self.config.acceleration_enabled:
            acceleration_score = normalize_score(
                acceleration_ratio,
                max(self.config.min_acceleration_ratio * 1.5, 1.0),
            )

        total_weight = (
            self.config.continuation_score_weight
            + self.config.imbalance_score_weight
            + self.config.notional_score_weight
            + self.config.acceleration_score_weight
        )

        if total_weight <= 0:
            return 0.0

        weighted = (
            continuation_signal * self.config.continuation_score_weight
            + imbalance_score * self.config.imbalance_score_weight
            + notional_score * self.config.notional_score_weight
            + acceleration_score * self.config.acceleration_score_weight
        ) / total_weight

        return clamp_float(weighted)

    def _compute_continuation_bias(
        self,
        *,
        stats: LiquidationWindowStats,
        acceleration_ratio: float,
    ) -> float:
        imbalance_component = clamp_float(stats.side_imbalance_ratio)
        acceleration_component = 0.0

        if self.config.acceleration_enabled:
            acceleration_component = normalize_score(
                acceleration_ratio,
                max(self.config.min_acceleration_ratio * 1.5, 1.0),
            )

        compaction_component = 0.5
        if self.config.price_compaction_enabled and self.config.max_price_range_pct > 0:
            compaction_component = clamp_float(
                1.0 - (stats.price_range_pct / self.config.max_price_range_pct)
            )

        result = (imbalance_component * 0.45) + (acceleration_component * 0.35) + (compaction_component * 0.20)
        return clamp_float(result)

    def _compute_exhaustion_bias(
        self,
        *,
        stats: LiquidationWindowStats,
        acceleration_ratio: float,
    ) -> float:
        """
        Exhaustion bias:
        - великий notional burst уже є
        - acceleration слабка або знижується
        - price range розширений сильніше, ніж очікується для компактного каскаду
        """
        burst_component = normalize_score(
            float(stats.total_notional_usd),
            float(self.config.min_total_notional_usd) * 4.0,
        )

        weak_acceleration_component = 0.0
        if self.config.acceleration_enabled:
            weak_acceleration_component = clamp_float(
                1.0 - normalize_score(
                    acceleration_ratio,
                    max(self.config.min_acceleration_ratio, 1.0),
                )
            )

        range_expansion_component = 0.0
        if self.config.price_compaction_enabled and self.config.max_price_range_pct > 0:
            range_expansion_component = clamp_float(
                stats.price_range_pct / (self.config.max_price_range_pct * 2.0)
            )

        result = (
            burst_component * 0.40
            + weak_acceleration_component * 0.35
            + range_expansion_component * 0.25
        )
        return clamp_float(result)

    def _compute_confidence(
        self,
        *,
        stats: LiquidationWindowStats,
        intensity_score: float,
        acceleration_ratio: float,
    ) -> float:
        notional_component = normalize_score(
            float(stats.total_notional_usd),
            float(self.config.min_total_notional_usd) * 4.0,
        )
        imbalance_component = clamp_float(stats.side_imbalance_ratio)

        acceleration_component = 0.5
        if self.config.acceleration_enabled:
            acceleration_component = normalize_score(
                acceleration_ratio,
                max(self.config.min_acceleration_ratio * 1.5, 1.0),
            )

        result = (
            intensity_score * 0.40
            + notional_component * 0.25
            + imbalance_component * 0.25
            + acceleration_component * 0.10
        )
        return clamp_float(result)

    # -------------------------------------------------------------------------
    # Emit / signal handling
    # -------------------------------------------------------------------------

    async def _emit_detection_result(self, result: CascadeDetectionResult) -> None:
        await self.event_bus.emit(self.config.publish_topic_detected, result)
        self.metrics.observe_cascade(result)

        self._cascade_signals_emitted += 1
        self._last_signal_at = result.detected_at

        self.logger.info(
            "Liquidation cascade detected.",
            extra={
                "exchange": result.exchange,
                "symbol": result.symbol,
                "side": result.side.value,
                "severity": result.severity.value,
                "intensity_score": result.intensity_score,
                "confidence": result.confidence,
                "continuation_bias": result.continuation_bias,
                "exhaustion_bias": result.exhaustion_bias,
                "event_count": result.event_count,
                "total_notional_usd": str(result.total_notional_usd),
                "price_range_pct": result.price_range_pct,
            },
        )

        if result.favors_exhaustion:
            await self.event_bus.emit(self.config.publish_topic_exhaustion, result)
            self.metrics.observe_exhaustion(result)
            self._exhaustion_signals_emitted += 1

    def _remember_signal(self, result: CascadeDetectionResult) -> None:
        self._latest_signals.append(result)
        if len(self._latest_signals) > self._latest_signals_limit:
            self._latest_signals = self._latest_signals[-self._latest_signals_limit :]

    # -------------------------------------------------------------------------
    # Feature methods
    # -------------------------------------------------------------------------

    async def detect_now(self, exchange: str, symbol: str) -> CascadeDetectionResult | None:
        """
        Форсований manual detect для конкретного символу.
        Корисно для debug/admin/API.
        """
        symbol_state = self.state.get(exchange.lower(), symbol.upper())
        if symbol_state is None or not symbol_state.events:
            return None

        trigger_event = symbol_state.events[-1]
        result = await self._detect_for_symbol_state(symbol_state, trigger_event=trigger_event)

        if result is not None:
            await self._emit_detection_result(result)

        return result

    def get_recent_signals(
        self,
        *,
        symbol: str | None = None,
        exchange: str | None = None,
        limit: int = 50,
    ) -> list[CascadeDetectionResult]:
        target_symbol = symbol.upper() if symbol else None
        target_exchange = exchange.lower() if exchange else None

        result: list[CascadeDetectionResult] = []
        for signal in reversed(self._latest_signals):
            if target_symbol and signal.symbol != target_symbol:
                continue
            if target_exchange and signal.exchange != target_exchange:
                continue
            result.append(signal)
            if len(result) >= limit:
                break

        return result

    def get_symbol_last_signal(
        self,
        exchange: str,
        symbol: str,
    ) -> CascadeDetectionResult | None:
        target_exchange = exchange.lower()
        target_symbol = symbol.upper()

        for signal in reversed(self._latest_signals):
            if signal.exchange == target_exchange and signal.symbol == target_symbol:
                return signal
        return None

    def get_hot_symbols(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Топ символів за силою останніх cascade signals.
        """
        latest_by_key: dict[tuple[str, str], CascadeDetectionResult] = {}

        for signal in self._latest_signals:
            key = (signal.exchange, signal.symbol)
            previous = latest_by_key.get(key)
            if previous is None or signal.detected_at > previous.detected_at:
                latest_by_key[key] = signal

        rows = [
            {
                "exchange": signal.exchange,
                "symbol": signal.symbol,
                "severity": signal.severity.value,
                "intensity_score": signal.intensity_score,
                "confidence": signal.confidence,
                "continuation_bias": signal.continuation_bias,
                "exhaustion_bias": signal.exhaustion_bias,
                "detected_at": signal.detected_at.isoformat(),
                "total_notional_usd": str(signal.total_notional_usd),
            }
            for signal in latest_by_key.values()
        ]

        rows.sort(
            key=lambda row: (
                float(row["intensity_score"]),
                float(row["confidence"]),
            ),
            reverse=True,
        )
        return rows[:limit]

    def get_symbol_diagnostic(self, exchange: str, symbol: str) -> dict[str, Any]:
        """
        Детальна діагностика по символу:
        - buffer snapshot
        - window stats
        - cooldown
        - last signal
        """
        exchange = exchange.lower()
        symbol = symbol.upper()
        symbol_state = self.state.get(exchange, symbol)

        if symbol_state is None:
            return {
                "exchange": exchange,
                "symbol": symbol,
                "exists": False,
            }

        now = utc_now()
        window_events = self._get_window_events(symbol_state, now=now)
        stats = compute_window_stats(exchange=exchange, symbol=symbol, events=window_events)
        last_signal = self.get_symbol_last_signal(exchange, symbol)

        return {
            "exchange": exchange,
            "symbol": symbol,
            "exists": True,
            "buffer_snapshot": asdict(symbol_state.snapshot()),
            "window_stats": {
                "total_events": stats.total_events,
                "long_events": stats.long_events,
                "short_events": stats.short_events,
                "total_notional_usd": str(stats.total_notional_usd),
                "long_notional_usd": str(stats.long_notional_usd),
                "short_notional_usd": str(stats.short_notional_usd),
                "dominant_side": stats.dominant_side.value,
                "side_imbalance_ratio": stats.side_imbalance_ratio,
                "price_range_pct": stats.price_range_pct,
                "window_start": stats.window_start.isoformat(),
                "window_end": stats.window_end.isoformat(),
            },
            "cooldown_active": symbol_state.is_in_cooldown(now),
            "last_signal": asdict(last_signal) if last_signal else None,
        }

    async def emit_runtime_snapshot(
        self,
        topic: str = DEFAULT_SNAPSHOT_TOPIC,
    ) -> None:
        snapshot = {
            "service": self.service_name,
            "running": self._running,
            "stats": self.get_stats(),
            "health": self.get_health(),
            "latest_signals": [asdict(item) for item in self.get_recent_signals(limit=25)],
            "emitted_at": utc_now().isoformat(),
        }
        await self.event_bus.emit(topic, snapshot)

    # -------------------------------------------------------------------------
    # Scheduler / health
    # -------------------------------------------------------------------------

    def _register_scheduler_jobs(self) -> None:
        if self.scheduler is None:
            return

        try:
            job = self.scheduler.add_interval_job(
                self._scheduled_healthcheck,
                seconds=15,
                name="cascade_detector_healthcheck",
                enabled=True,
                run_immediately=False,
                max_retries=0,
                timeout=5,
                tags=["liquidations", "cascade_detector", "healthcheck"],
            )
            self._healthcheck_job_id = getattr(job, "job_id", None)
        except Exception as exc:
            self.logger.exception(
                "Failed to register scheduler jobs for CascadeDetector.",
                extra={"error": repr(exc)},
            )

    async def _scheduled_healthcheck(self) -> None:
        health = self.get_health()
        if health["status"] != "healthy":
            self.logger.warning("CascadeDetector health degraded.", extra=health)

    def get_health(self) -> dict[str, Any]:
        status = "healthy"
        if not self._running:
            status = "stopped"
        elif self._last_error is not None:
            status = "degraded"

        seconds_since_last_signal = (
            (utc_now() - self._last_signal_at).total_seconds()
            if self._last_signal_at
            else None
        )

        return {
            "status": status,
            "running": self._running,
            "last_signal_at": self._last_signal_at.isoformat() if self._last_signal_at else None,
            "seconds_since_last_signal": seconds_since_last_signal,
            "last_error": self._last_error,
            "last_error_at": self._last_error_at.isoformat() if self._last_error_at else None,
        }

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        uptime_seconds = (
            max(0.0, (utc_now() - self._started_at).total_seconds())
            if self._started_at
            else 0.0
        )

        return {
            "service_name": self.service_name,
            "running": self._running,
            "uptime_seconds": uptime_seconds,
            "processed_events": self._processed_events,
            "cascade_signals_emitted": self._cascade_signals_emitted,
            "exhaustion_signals_emitted": self._exhaustion_signals_emitted,
            "cooldown_skips": self._cooldown_skips,
            "empty_window_skips": self._empty_window_skips,
            "threshold_skips": self._threshold_skips,
            "tracked_symbols": len(self.state.symbols),
            "latest_signals_buffered": len(self._latest_signals),
            "last_signal_at": self._last_signal_at.isoformat() if self._last_signal_at else None,
            "last_error": self._last_error,
            "last_error_at": self._last_error_at.isoformat() if self._last_error_at else None,
        }