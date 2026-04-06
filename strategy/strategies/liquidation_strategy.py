from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Protocol

from core.logger import get_logger

from analytics.liquidations.enums import CascadeSeverity
from analytics.liquidations.models import CascadeDetectionResult
from analytics.liquidations.utils import utc_now


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


@dataclass(slots=True)
class LiquidationStrategyConfig:
    """
    Конфіг стратегії на базі liquidation cascades.

    Ідея:
    - беремо cascade detection result
    - фільтруємо по confidence / severity / direction bias
    - генеруємо strategy signal без execution-логіки
    """

    enabled: bool = True

    subscribe_topic_cascade: str = "analytics.liquidation.cascade_detected"
    subscribe_topic_exhaustion: str = "analytics.liquidation.exhaustion_detected"

    publish_topic_generated: str = "signal.generated"
    publish_topic_rejected: str = "signal.rejected"
    publish_topic_snapshot: str = "strategy.liquidation.snapshot"

    min_confidence: float = 0.65
    min_intensity_score: float = 0.60
    min_continuation_bias: float = 0.55
    max_exhaustion_bias_for_entry: float = 0.45

    allow_low_severity: bool = False
    allow_medium_severity: bool = True
    allow_high_severity: bool = True
    allow_extreme_severity: bool = True

    symbol_cooldown_seconds: int = 20
    max_signals_per_symbol_window: int = 2
    signal_window_seconds: int = 60

    emit_rejections: bool = True
    emit_exhaustion_rejections: bool = True

    healthcheck_interval_seconds: int = 15

    signal_source: str = "liquidation_strategy"
    signal_type: str = "liquidation_cascade"


class LiquidationStrategy:
    """
    Strategy-модуль для liquidation analytics.

    Задачі:
    - слухати cascade / exhaustion detection events
    - відбирати лише ті патерни, які придатні для continuation-entry
    - публікувати strategy signal у EventBus

    Цей клас НЕ:
    - не отримує market data напряму
    - не ставить ордери
    - не керує позиціями
    - не робить risk checks
    """

    DEFAULT_COMPONENT = "strategy.liquidation_strategy"

    def __init__(
        self,
        *,
        event_bus: EventBusProtocol,
        config: LiquidationStrategyConfig | None = None,
        scheduler: SchedulerProtocol | None = None,
        service_name: str = "liquidation_strategy",
    ) -> None:
        self.event_bus = event_bus
        self.config = config or LiquidationStrategyConfig()
        self.scheduler = scheduler
        self.service_name = service_name

        self.logger = get_logger(
            __name__,
            service_name=service_name,
            component=self.DEFAULT_COMPONENT,
        )

        self._running = False
        self._started_at: datetime | None = None
        self._stopped_at: datetime | None = None
        self._subscription_cascade: Any = None
        self._subscription_exhaustion: Any = None
        self._healthcheck_job_id: str | None = None

        self._processed_events = 0
        self._generated_signals = 0
        self._rejected_signals = 0
        self._cooldown_skips = 0
        self._duplicate_window_skips = 0
        self._last_signal_at: datetime | None = None
        self._last_error_at: datetime | None = None
        self._last_error: str | None = None

        self._recent_signals: list[dict[str, Any]] = []
        self._recent_signals_limit = 200

        self._symbol_cooldowns: dict[tuple[str, str], datetime] = {}
        self._symbol_signal_times: dict[tuple[str, str], list[datetime]] = {}

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            self.logger.warning("LiquidationStrategy already running.")
            return

        if not self.config.enabled:
            self.logger.warning("LiquidationStrategy is disabled by config.")
            return

        self._running = True
        self._started_at = utc_now()
        self._stopped_at = None

        self._subscription_cascade = await self.event_bus.subscribe(
            self.config.subscribe_topic_cascade,
            self.on_cascade_detected,
        )
        self._subscription_exhaustion = await self.event_bus.subscribe(
            self.config.subscribe_topic_exhaustion,
            self.on_exhaustion_detected,
        )

        self._register_scheduler_jobs()

        self.logger.info(
            "LiquidationStrategy started.",
            extra={
                "subscribe_topic_cascade": self.config.subscribe_topic_cascade,
                "subscribe_topic_exhaustion": self.config.subscribe_topic_exhaustion,
                "publish_topic_generated": self.config.publish_topic_generated,
                "min_confidence": self.config.min_confidence,
                "min_intensity_score": self.config.min_intensity_score,
                "min_continuation_bias": self.config.min_continuation_bias,
                "max_exhaustion_bias_for_entry": self.config.max_exhaustion_bias_for_entry,
            },
        )

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        self._stopped_at = utc_now()

        self.logger.info(
            "LiquidationStrategy stopped.",
            extra=self.get_stats(),
        )

    async def restart(self) -> None:
        await self.stop()
        await self.start()

    # -------------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------------

    async def on_cascade_detected(self, event: Any) -> None:
        if not self._running:
            return

        if not isinstance(event, CascadeDetectionResult):
            self.logger.debug(
                "LiquidationStrategy received non-CascadeDetectionResult payload, ignored.",
                extra={"payload_type": type(event).__name__},
            )
            return

        try:
            self._processed_events += 1

            if not self._is_allowed_severity(event.severity):
                await self._reject(
                    result=event,
                    reason="severity_not_allowed",
                    extra={"severity": event.severity.value},
                )
                return

            if self._is_symbol_in_cooldown(event.exchange, event.symbol, now=event.detected_at):
                self._cooldown_skips += 1
                await self._reject(
                    result=event,
                    reason="symbol_cooldown",
                    extra={"cooldown_until": self._get_symbol_cooldown_until(event.exchange, event.symbol)},
                )
                return

            if self._exceeds_signal_rate_limit(event.exchange, event.symbol, now=event.detected_at):
                self._duplicate_window_skips += 1
                await self._reject(
                    result=event,
                    reason="signal_rate_limit",
                    extra={
                        "signal_window_seconds": self.config.signal_window_seconds,
                        "max_signals_per_symbol_window": self.config.max_signals_per_symbol_window,
                    },
                )
                return

            decision = self._build_trade_decision(event)
            if not decision["accepted"]:
                await self._reject(
                    result=event,
                    reason=decision["reason"],
                    extra=decision,
                )
                return

            signal = self._build_signal_payload(event, decision=decision)
            await self.event_bus.emit(self.config.publish_topic_generated, signal)

            self._generated_signals += 1
            self._last_signal_at = utc_now()
            self._remember_signal(signal)
            self._mark_symbol_cooldown(event.exchange, event.symbol, now=event.detected_at)
            self._remember_symbol_signal_time(event.exchange, event.symbol, now=event.detected_at)

            self.logger.info(
                "Liquidation strategy signal generated.",
                extra={
                    "exchange": signal["exchange"],
                    "symbol": signal["symbol"],
                    "side": signal["side"],
                    "strategy": signal["strategy"],
                    "confidence": signal["confidence"],
                    "severity": signal["severity"],
                    "direction": signal["direction"],
                    "continuation_bias": signal["context"]["continuation_bias"],
                    "exhaustion_bias": signal["context"]["exhaustion_bias"],
                    "event_count": signal["context"]["event_count"],
                    "total_notional_usd": signal["context"]["total_notional_usd"],
                },
            )

        except Exception as exc:
            self._last_error_at = utc_now()
            self._last_error = repr(exc)
            self.logger.exception(
                "Unhandled error in LiquidationStrategy.on_cascade_detected.",
                extra={
                    "error": repr(exc),
                    "exchange": getattr(event, "exchange", None),
                    "symbol": getattr(event, "symbol", None),
                },
            )

    async def on_exhaustion_detected(self, event: Any) -> None:
        if not self._running:
            return

        if not isinstance(event, CascadeDetectionResult):
            self.logger.debug(
                "LiquidationStrategy received non-CascadeDetectionResult exhaustion payload, ignored.",
                extra={"payload_type": type(event).__name__},
            )
            return

        try:
            self._processed_events += 1

            if not self.config.emit_exhaustion_rejections:
                return

            await self._reject(
                result=event,
                reason="exhaustion_detected",
                extra={
                    "continuation_bias": event.continuation_bias,
                    "exhaustion_bias": event.exhaustion_bias,
                    "confidence": event.confidence,
                    "severity": event.severity.value,
                },
            )

        except Exception as exc:
            self._last_error_at = utc_now()
            self._last_error = repr(exc)
            self.logger.exception(
                "Unhandled error in LiquidationStrategy.on_exhaustion_detected.",
                extra={
                    "error": repr(exc),
                    "exchange": getattr(event, "exchange", None),
                    "symbol": getattr(event, "symbol", None),
                },
            )

    # -------------------------------------------------------------------------
    # Decision logic
    # -------------------------------------------------------------------------

    def _build_trade_decision(self, result: CascadeDetectionResult) -> dict[str, Any]:
        if result.confidence < self.config.min_confidence:
            return {
                "accepted": False,
                "reason": "confidence_too_low",
                "confidence": result.confidence,
                "threshold": self.config.min_confidence,
            }

        if result.intensity_score < self.config.min_intensity_score:
            return {
                "accepted": False,
                "reason": "intensity_too_low",
                "intensity_score": result.intensity_score,
                "threshold": self.config.min_intensity_score,
            }

        if result.continuation_bias < self.config.min_continuation_bias:
            return {
                "accepted": False,
                "reason": "continuation_bias_too_low",
                "continuation_bias": result.continuation_bias,
                "threshold": self.config.min_continuation_bias,
            }

        if result.exhaustion_bias > self.config.max_exhaustion_bias_for_entry:
            return {
                "accepted": False,
                "reason": "exhaustion_bias_too_high",
                "exhaustion_bias": result.exhaustion_bias,
                "threshold": self.config.max_exhaustion_bias_for_entry,
            }

        if result.favors_exhaustion:
            return {
                "accepted": False,
                "reason": "favors_exhaustion",
                "continuation_bias": result.continuation_bias,
                "exhaustion_bias": result.exhaustion_bias,
            }

        trade_side = self._map_direction_to_trade_side(result)

        if trade_side is None:
            return {
                "accepted": False,
                "reason": "unknown_trade_direction",
                "direction": result.direction.value,
            }

        return {
            "accepted": True,
            "reason": "ok",
            "trade_side": trade_side,
            "confidence": result.confidence,
            "severity": result.severity.value,
            "continuation_bias": result.continuation_bias,
            "exhaustion_bias": result.exhaustion_bias,
        }

    def _map_direction_to_trade_side(self, result: CascadeDetectionResult) -> str | None:
        """
        Liquidation semantics:
        - long liquidations => downward cascade => short continuation bias
        - short liquidations => upward cascade => long continuation bias
        """
        if result.direction.value == "down":
            return "SHORT"
        if result.direction.value == "up":
            return "LONG"
        return None

    def _is_allowed_severity(self, severity: CascadeSeverity) -> bool:
        mapping = {
            CascadeSeverity.LOW: self.config.allow_low_severity,
            CascadeSeverity.MEDIUM: self.config.allow_medium_severity,
            CascadeSeverity.HIGH: self.config.allow_high_severity,
            CascadeSeverity.EXTREME: self.config.allow_extreme_severity,
        }
        return mapping.get(severity, False)

    # -------------------------------------------------------------------------
    # Signal payload
    # -------------------------------------------------------------------------

    def _build_signal_payload(
        self,
        result: CascadeDetectionResult,
        *,
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "strategy": self.config.signal_type,
            "source": self.config.signal_source,
            "exchange": result.exchange,
            "symbol": result.symbol,
            "side": decision["trade_side"],
            "direction": result.direction.value,
            "confidence": result.confidence,
            "severity": result.severity.value,
            "timestamp": utc_now().isoformat(),
            "signal_type": "entry",
            "reason": "liquidation_continuation",
            "context": {
                "detected_at": result.detected_at.isoformat(),
                "event_count": result.event_count,
                "window_seconds": result.window_seconds,
                "price_range_pct": result.price_range_pct,
                "intensity_score": result.intensity_score,
                "continuation_bias": result.continuation_bias,
                "exhaustion_bias": result.exhaustion_bias,
                "total_notional_usd": str(result.total_notional_usd),
                "cluster_side": result.side.value,
                "cluster_direction": result.direction.value,
                "cluster_status": result.status.value,
                "cluster_severity": result.severity.value,
                "cluster_start_time": result.cluster.start_time.isoformat(),
                "cluster_end_time": result.cluster.end_time.isoformat(),
                "cluster_duration_seconds": result.cluster.duration_seconds,
                "cluster_avg_price": str(result.cluster.avg_price),
                "cluster_min_price": str(result.cluster.min_price),
                "cluster_max_price": str(result.cluster.max_price),
                "metadata": dict(result.metadata),
            },
        }

    async def _reject(
        self,
        *,
        result: CascadeDetectionResult,
        reason: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._rejected_signals += 1

        payload = {
            "strategy": self.config.signal_type,
            "source": self.config.signal_source,
            "exchange": result.exchange,
            "symbol": result.symbol,
            "timestamp": utc_now().isoformat(),
            "reason": reason,
            "confidence": result.confidence,
            "severity": result.severity.value,
            "direction": result.direction.value,
            "context": {
                "intensity_score": result.intensity_score,
                "continuation_bias": result.continuation_bias,
                "exhaustion_bias": result.exhaustion_bias,
                "event_count": result.event_count,
                "window_seconds": result.window_seconds,
                "price_range_pct": result.price_range_pct,
                "total_notional_usd": str(result.total_notional_usd),
                "metadata": dict(result.metadata),
            },
        }

        if extra:
            payload["decision"] = extra

        if self.config.emit_rejections:
            await self.event_bus.emit(self.config.publish_topic_rejected, payload)

        self.logger.debug(
            "Liquidation strategy signal rejected.",
            extra={
                "exchange": result.exchange,
                "symbol": result.symbol,
                "reason": reason,
                "confidence": result.confidence,
                "severity": result.severity.value,
            },
        )

    # -------------------------------------------------------------------------
    # Cooldown / rate-limit
    # -------------------------------------------------------------------------

    def _symbol_key(self, exchange: str, symbol: str) -> tuple[str, str]:
        return exchange.lower(), symbol.upper()

    def _mark_symbol_cooldown(self, exchange: str, symbol: str, *, now: datetime) -> None:
        if self.config.symbol_cooldown_seconds <= 0:
            return
        self._symbol_cooldowns[self._symbol_key(exchange, symbol)] = (
            now + timedelta(seconds=self.config.symbol_cooldown_seconds)
        )

    def _is_symbol_in_cooldown(self, exchange: str, symbol: str, *, now: datetime) -> bool:
        cooldown_until = self._symbol_cooldowns.get(self._symbol_key(exchange, symbol))
        return cooldown_until is not None and now < cooldown_until

    def _get_symbol_cooldown_until(self, exchange: str, symbol: str) -> str | None:
        value = self._symbol_cooldowns.get(self._symbol_key(exchange, symbol))
        return value.isoformat() if value else None

    def _remember_symbol_signal_time(self, exchange: str, symbol: str, *, now: datetime) -> None:
        key = self._symbol_key(exchange, symbol)
        self._symbol_signal_times.setdefault(key, [])
        self._symbol_signal_times[key].append(now)
        self._prune_symbol_signal_times(key, now=now)

    def _exceeds_signal_rate_limit(self, exchange: str, symbol: str, *, now: datetime) -> bool:
        key = self._symbol_key(exchange, symbol)
        self._prune_symbol_signal_times(key, now=now)
        return len(self._symbol_signal_times.get(key, [])) >= self.config.max_signals_per_symbol_window

    def _prune_symbol_signal_times(self, key: tuple[str, str], *, now: datetime) -> None:
        min_ts = now - timedelta(seconds=self.config.signal_window_seconds)
        values = self._symbol_signal_times.get(key, [])
        if not values:
            return
        self._symbol_signal_times[key] = [item for item in values if item >= min_ts]

    # -------------------------------------------------------------------------
    # Runtime snapshots / diagnostics
    # -------------------------------------------------------------------------

    def _remember_signal(self, signal: dict[str, Any]) -> None:
        self._recent_signals.append(signal)
        if len(self._recent_signals) > self._recent_signals_limit:
            self._recent_signals = self._recent_signals[-self._recent_signals_limit :]

    def get_recent_signals(
        self,
        *,
        symbol: str | None = None,
        exchange: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        target_symbol = symbol.upper() if symbol else None
        target_exchange = exchange.lower() if exchange else None

        result: list[dict[str, Any]] = []
        for signal in reversed(self._recent_signals):
            if target_symbol and signal["symbol"] != target_symbol:
                continue
            if target_exchange and signal["exchange"] != target_exchange:
                continue
            result.append(signal)
            if len(result) >= limit:
                break
        return result

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
            "generated_signals": self._generated_signals,
            "rejected_signals": self._rejected_signals,
            "cooldown_skips": self._cooldown_skips,
            "duplicate_window_skips": self._duplicate_window_skips,
            "tracked_symbol_cooldowns": len(self._symbol_cooldowns),
            "tracked_symbol_signal_windows": len(self._symbol_signal_times),
            "last_signal_at": self._last_signal_at.isoformat() if self._last_signal_at else None,
            "last_error": self._last_error,
            "last_error_at": self._last_error_at.isoformat() if self._last_error_at else None,
        }

    def get_health(self) -> dict[str, Any]:
        if not self._running:
            return {
                "status": "stopped",
                "service_name": self.service_name,
            }

        now = utc_now()
        stale_signal_seconds = (
            (now - self._last_signal_at).total_seconds()
            if self._last_signal_at is not None
            else None
        )

        status = "healthy"
        if self._last_error_at is not None:
            if (now - self._last_error_at).total_seconds() <= 60:
                status = "degraded"

        return {
            "status": status,
            "service_name": self.service_name,
            "running": self._running,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "last_signal_at": self._last_signal_at.isoformat() if self._last_signal_at else None,
            "seconds_since_last_signal": stale_signal_seconds,
            "last_error": self._last_error,
            "last_error_at": self._last_error_at.isoformat() if self._last_error_at else None,
        }

    async def emit_runtime_snapshot(self) -> None:
        snapshot = {
            "service_name": self.service_name,
            "health": self.get_health(),
            "stats": self.get_stats(),
            "recent_signals": self.get_recent_signals(limit=25),
            "emitted_at": utc_now().isoformat(),
        }
        await self.event_bus.emit(self.config.publish_topic_snapshot, snapshot)

    # -------------------------------------------------------------------------
    # Scheduler
    # -------------------------------------------------------------------------

    def _register_scheduler_jobs(self) -> None:
        if self.scheduler is None:
            return

        try:
            job = self.scheduler.add_interval_job(
                self._scheduled_healthcheck,
                seconds=self.config.healthcheck_interval_seconds,
                name="liquidation_strategy_healthcheck",
                enabled=True,
                run_immediately=False,
                max_retries=0,
                timeout=5,
                tags=["strategy", "liquidations", "healthcheck"],
            )
            self._healthcheck_job_id = getattr(job, "job_id", None)
        except Exception as exc:
            self.logger.exception(
                "Failed to register scheduler jobs for LiquidationStrategy.",
                extra={"error": repr(exc)},
            )

    async def _scheduled_healthcheck(self) -> None:
        health = self.get_health()
        if health["status"] == "degraded":
            self.logger.warning(
                "LiquidationStrategy health degraded.",
                extra=health,
            )