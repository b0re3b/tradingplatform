from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from core.event_bus import Event, EventBus, EventPriority, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler

from analytics.liquidations.enums import CascadeDirection, CascadeSeverity
from analytics.liquidations.models import CascadeDetectionResult
from analytics.liquidations.utils import clamp_float, ensure_utc, normalize_symbol, utc_now


@dataclass(slots=True)
class LiquidationCascadeStrategyConfig:
    """
    Continuation strategy поверх liquidation cascade detector.

    Strategy:
    - слухає analytics.liquidation.cascade_detected
    - фільтрує слабкі / шумні каскади
    - генерує continuation signal у напрямку каскаду
    """

    enabled: bool = True

    subscribe_topic: str = "analytics.liquidation.cascade_detected"
    publish_topic_signal_generated: str = "signal.generated"
    publish_topic_signal_rejected: str = "signal.rejected"

    publish_rejected_events: bool = False
    publish_diagnostics_snapshots: bool = False

    diagnostics_topic: str = "strategy.liquidations.cascade.snapshot"
    diagnostics_interval_seconds: float = 30.0

    strategy_name: str = "liquidation_cascade_strategy"
    signal_type: str = "continuation"
    service_name: str = "liquidation_cascade_strategy"

    signal_priority: EventPriority = EventPriority.HIGH
    rejection_priority: EventPriority = EventPriority.LOW
    diagnostics_priority: EventPriority = EventPriority.LOW

    allowed_exchanges: tuple[str, ...] = ()
    allowed_symbols: tuple[str, ...] = ()
    blocked_symbols: tuple[str, ...] = ()

    allowed_severities: tuple[CascadeSeverity, ...] = (
        CascadeSeverity.MEDIUM,
        CascadeSeverity.HIGH,
        CascadeSeverity.EXTREME,
    )

    min_confidence: float = 0.60
    min_intensity_score: float = 0.55
    min_continuation_bias: float = 0.60
    min_total_notional_usd: Decimal = Decimal("300000")
    min_event_count: int = 5
    max_price_range_pct: float | None = None

    require_favors_continuation: bool = True
    require_high_confidence_only: bool = False

    symbol_cooldown_seconds: int = 20
    min_seconds_between_same_side_signals: int = 10

    max_signals_per_symbol_window: int = 2
    signal_window_seconds: int = 60

    deduplicate_by_detected_at: bool = True
    deduplicate_same_cluster_signature: bool = True

    recent_signals_limit: int = 200
    recent_rejections_limit: int = 200

    score_confidence_weight: float = 0.35
    score_continuation_bias_weight: float = 0.35
    score_intensity_weight: float = 0.20
    score_severity_weight: float = 0.10

    def validate(self) -> None:
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be between 0 and 1")

        if not (0.0 <= self.min_intensity_score <= 1.0):
            raise ValueError("min_intensity_score must be between 0 and 1")

        if not (0.0 <= self.min_continuation_bias <= 1.0):
            raise ValueError("min_continuation_bias must be between 0 and 1")

        if self.min_total_notional_usd < 0:
            raise ValueError("min_total_notional_usd must be >= 0")

        if self.min_event_count < 0:
            raise ValueError("min_event_count must be >= 0")

        if self.max_price_range_pct is not None and self.max_price_range_pct < 0:
            raise ValueError("max_price_range_pct must be >= 0 or None")

        if self.symbol_cooldown_seconds < 0:
            raise ValueError("symbol_cooldown_seconds must be >= 0")

        if self.min_seconds_between_same_side_signals < 0:
            raise ValueError("min_seconds_between_same_side_signals must be >= 0")

        if self.max_signals_per_symbol_window < 0:
            raise ValueError("max_signals_per_symbol_window must be >= 0")

        if self.signal_window_seconds <= 0:
            raise ValueError("signal_window_seconds must be > 0")

        if self.diagnostics_interval_seconds <= 0:
            raise ValueError("diagnostics_interval_seconds must be > 0")

        total_weight = (
            self.score_confidence_weight
            + self.score_continuation_bias_weight
            + self.score_intensity_weight
            + self.score_severity_weight
        )
        if total_weight <= 0:
            raise ValueError("strategy score weights sum must be > 0")


@dataclass(slots=True)
class LiquidationCascadeSignal:
    strategy_name: str
    signal_type: str

    exchange: str
    symbol: str
    side: str

    confidence: float
    score: float

    generated_at: datetime
    detected_at: datetime

    reason: str
    source_topic: str

    severity: str
    cascade_direction: str
    liquidation_side: str

    event_count: int
    total_notional_usd: Decimal
    intensity_score: float
    continuation_bias: float
    exhaustion_bias: float
    price_range_pct: float

    correlation_id: str | None = None
    source_event_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyRejection:
    exchange: str
    symbol: str
    rejected_at: datetime
    reason: str
    source_topic: str
    correlation_id: str | None = None
    source_event_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SymbolCascadeStrategyState:
    exchange: str
    symbol: str

    last_signal_at: datetime | None = None
    cooldown_until: datetime | None = None
    last_signal_side: str | None = None
    last_detected_at: datetime | None = None

    last_cluster_signature: str | None = None
    last_signal_score: float | None = None

    total_signals_emitted: int = 0
    signal_timestamps: list[datetime] = field(default_factory=list)

    def is_in_cooldown(self, now: datetime) -> bool:
        return self.cooldown_until is not None and ensure_utc(now) < self.cooldown_until

    def remember_signal(
        self,
        *,
        signal_at: datetime,
        signal_side: str,
        score: float,
        cooldown_seconds: int,
        cluster_signature: str | None,
        detected_at: datetime,
    ) -> None:
        signal_at = ensure_utc(signal_at)
        self.last_signal_at = signal_at
        self.cooldown_until = (
            signal_at + timedelta(seconds=cooldown_seconds)
            if cooldown_seconds > 0
            else None
        )
        self.last_signal_side = signal_side
        self.last_signal_score = score
        self.last_cluster_signature = cluster_signature
        self.last_detected_at = ensure_utc(detected_at)
        self.total_signals_emitted += 1
        self.signal_timestamps.append(signal_at)

    def prune_old_signal_timestamps(self, now: datetime, window_seconds: int) -> None:
        min_ts = ensure_utc(now) - timedelta(seconds=window_seconds)
        self.signal_timestamps = [ts for ts in self.signal_timestamps if ts >= min_ts]

    def signals_in_window(self, now: datetime, window_seconds: int) -> int:
        self.prune_old_signal_timestamps(now, window_seconds)
        return len(self.signal_timestamps)


@dataclass(slots=True)
class LiquidationCascadeStrategyStats:
    started_at: datetime | None = None
    stopped_at: datetime | None = None

    processed_events: int = 0
    emitted_signals: int = 0
    rejected_events: int = 0

    duplicate_skips: int = 0
    cooldown_skips: int = 0
    rate_limit_skips: int = 0
    filter_skips: int = 0
    invalid_payload_skips: int = 0

    last_signal_at: datetime | None = None
    last_error_at: datetime | None = None
    last_error: str | None = None


class LiquidationCascadeStrategy:
    """
    Continuation strategy поверх analytics.liquidation.cascade_detected.

    Pipeline:
    EventBus Event(topic=analytics.liquidation.cascade_detected, payload=CascadeDetectionResult)
        -> filters
        -> continuation signal
        -> EventBus.emit(signal.generated, payload=LiquidationCascadeSignal)
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        config: LiquidationCascadeStrategyConfig | None = None,
        scheduler: Scheduler | None = None,
        service_name: str | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.config = config or LiquidationCascadeStrategyConfig()
        self.scheduler = scheduler
        self.service_name = service_name or self.config.service_name

        self.config.validate()

        self.logger = get_logger(
            __name__,
            service_name=self.service_name,
            component="strategy.liquidations.cascade_strategy",
            strategy=self.config.strategy_name,
        )

        self._running = False
        self._subscription: Subscription | None = None
        self._diagnostics_job_id: str | None = None

        self._states: dict[tuple[str, str], SymbolCascadeStrategyState] = {}
        self._recent_signals: list[LiquidationCascadeSignal] = []
        self._recent_rejections: list[StrategyRejection] = []

        self._stats = LiquidationCascadeStrategyStats()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            self.logger.warning("LiquidationCascadeStrategy already running.")
            return

        if not self.config.enabled:
            self.logger.warning("LiquidationCascadeStrategy is disabled by config.")
            return

        self._running = True
        self._stats.started_at = utc_now()
        self._stats.stopped_at = None

        self._subscription = self.event_bus.subscribe(
            self.config.subscribe_topic,
            self.on_cascade_detected,
            name=f"{self.config.strategy_name}.on_cascade_detected",
        )

        self._register_scheduler_jobs()

        self.logger.info(
            "LiquidationCascadeStrategy started.",
            extra={
                "topic": self.config.subscribe_topic,
                "min_confidence": self.config.min_confidence,
                "min_intensity_score": self.config.min_intensity_score,
                "min_continuation_bias": self.config.min_continuation_bias,
                "min_total_notional_usd": str(self.config.min_total_notional_usd),
                "allowed_severities": [item.value for item in self.config.allowed_severities],
            },
        )

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        self._stats.stopped_at = utc_now()

        if self._subscription is not None:
            self.event_bus.unsubscribe(self._subscription)
            self._subscription = None

        if self._diagnostics_job_id and self.scheduler is not None:
            try:
                self.scheduler.remove_job(self._diagnostics_job_id)
            except KeyError:
                pass
            self._diagnostics_job_id = None

        self.logger.info(
            "LiquidationCascadeStrategy stopped.",
            extra=self.get_stats(),
        )

    async def restart(self) -> None:
        await self.stop()
        await self.start()

    # ------------------------------------------------------------------
    # Main event handler
    # ------------------------------------------------------------------

    async def on_cascade_detected(self, bus_event: Event) -> None:
        if not self._running:
            return

        payload = bus_event.payload
        if not isinstance(payload, CascadeDetectionResult):
            self._stats.invalid_payload_skips += 1
            self.logger.debug(
                "Non-CascadeDetectionResult payload received, ignored.",
                extra={
                    "topic": bus_event.topic,
                    "payload_type": type(payload).__name__,
                    "event_id": bus_event.event_id,
                },
            )
            return

        try:
            self._stats.processed_events += 1

            result = payload
            state = self._get_or_create_state(result.exchange, result.symbol)
            now = utc_now()

            rejection_reason = self._get_rejection_reason(
                result=result,
                state=state,
                now=now,
            )
            if rejection_reason is not None:
                await self._reject_result(
                    result=result,
                    bus_event=bus_event,
                    reason=rejection_reason,
                )
                return

            signal = self._build_signal(result=result, bus_event=bus_event)
            await self._emit_signal(signal, bus_event=bus_event)

            cluster_signature = self._build_cluster_signature(result)
            state.remember_signal(
                signal_at=signal.generated_at,
                signal_side=signal.side,
                score=signal.score,
                cooldown_seconds=self.config.symbol_cooldown_seconds,
                cluster_signature=cluster_signature,
                detected_at=result.detected_at,
            )

            self._remember_signal(signal)

            self._stats.emitted_signals += 1
            self._stats.last_signal_at = signal.generated_at

            self.logger.info(
                "Liquidation cascade continuation signal emitted.",
                extra={
                    "exchange": signal.exchange,
                    "symbol": signal.symbol,
                    "side": signal.side,
                    "score": signal.score,
                    "confidence": signal.confidence,
                    "severity": signal.severity,
                    "intensity_score": signal.intensity_score,
                    "continuation_bias": signal.continuation_bias,
                    "event_count": signal.event_count,
                    "total_notional_usd": str(signal.total_notional_usd),
                    "event_id": bus_event.event_id,
                    "correlation_id": bus_event.correlation_id,
                },
            )

        except Exception as exc:
            self._stats.last_error_at = utc_now()
            self._stats.last_error = repr(exc)
            self.logger.exception(
                "Unhandled error in LiquidationCascadeStrategy.on_cascade_detected.",
                extra={
                    "topic": bus_event.topic,
                    "event_id": bus_event.event_id,
                    "correlation_id": bus_event.correlation_id,
                    "error": repr(exc),
                },
            )

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _get_rejection_reason(
        self,
        *,
        result: CascadeDetectionResult,
        state: SymbolCascadeStrategyState,
        now: datetime,
    ) -> str | None:
        if self.config.allowed_exchanges:
            allowed = {item.lower() for item in self.config.allowed_exchanges}
            if result.exchange.lower() not in allowed:
                self._stats.filter_skips += 1
                return "exchange_not_allowed"

        if self.config.allowed_symbols:
            allowed_symbols = {normalize_symbol(item) for item in self.config.allowed_symbols}
            if result.symbol.upper() not in allowed_symbols:
                self._stats.filter_skips += 1
                return "symbol_not_allowed"

        if self.config.blocked_symbols:
            blocked_symbols = {normalize_symbol(item) for item in self.config.blocked_symbols}
            if result.symbol.upper() in blocked_symbols:
                self._stats.filter_skips += 1
                return "symbol_blocked"

        if result.direction == CascadeDirection.UNKNOWN:
            self._stats.filter_skips += 1
            return "unknown_direction"

        if self.config.require_favors_continuation and not result.favors_continuation:
            self._stats.filter_skips += 1
            return "continuation_not_favored"

        if self.config.require_high_confidence_only and not result.is_high_confidence:
            self._stats.filter_skips += 1
            return "not_high_confidence"

        if result.confidence < self.config.min_confidence:
            self._stats.filter_skips += 1
            return "confidence_below_threshold"

        if result.intensity_score < self.config.min_intensity_score:
            self._stats.filter_skips += 1
            return "intensity_below_threshold"

        if result.continuation_bias < self.config.min_continuation_bias:
            self._stats.filter_skips += 1
            return "continuation_bias_below_threshold"

        if result.total_notional_usd < self.config.min_total_notional_usd:
            self._stats.filter_skips += 1
            return "notional_below_threshold"

        if result.event_count < self.config.min_event_count:
            self._stats.filter_skips += 1
            return "event_count_below_threshold"

        if result.severity not in self.config.allowed_severities:
            self._stats.filter_skips += 1
            return "severity_not_allowed"

        if (
            self.config.max_price_range_pct is not None
            and result.price_range_pct > self.config.max_price_range_pct
        ):
            self._stats.filter_skips += 1
            return "price_range_above_threshold"

        if state.is_in_cooldown(now):
            self._stats.cooldown_skips += 1
            return "symbol_in_cooldown"

        if self.config.deduplicate_by_detected_at:
            if state.last_detected_at is not None and ensure_utc(result.detected_at) <= state.last_detected_at:
                self._stats.duplicate_skips += 1
                return "duplicate_detected_at"

        cluster_signature = self._build_cluster_signature(result)
        if self.config.deduplicate_same_cluster_signature:
            if cluster_signature and state.last_cluster_signature == cluster_signature:
                self._stats.duplicate_skips += 1
                return "duplicate_cluster_signature"

        trade_side = self._direction_to_trade_side(result.direction)
        if (
            state.last_signal_at is not None
            and state.last_signal_side == trade_side
            and (ensure_utc(now) - state.last_signal_at).total_seconds()
            < self.config.min_seconds_between_same_side_signals
        ):
            self._stats.duplicate_skips += 1
            return "same_side_signal_too_soon"

        if self.config.max_signals_per_symbol_window > 0:
            signals_in_window = state.signals_in_window(
                now=now,
                window_seconds=self.config.signal_window_seconds,
            )
            if signals_in_window >= self.config.max_signals_per_symbol_window:
                self._stats.rate_limit_skips += 1
                return "symbol_signal_rate_limited"

        return None

    # ------------------------------------------------------------------
    # Signal building
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        *,
        result: CascadeDetectionResult,
        bus_event: Event,
    ) -> LiquidationCascadeSignal:
        trade_side = self._direction_to_trade_side(result.direction)
        generated_at = utc_now()
        score = self._compute_strategy_score(result)

        reason = (
            f"liquidation cascade continuation: "
            f"direction={result.direction.value}, "
            f"severity={result.severity.value}, "
            f"continuation_bias={result.continuation_bias:.3f}, "
            f"confidence={result.confidence:.3f}"
        )

        metadata = {
            "cluster": {
                "start_time": result.cluster.start_time.isoformat(),
                "end_time": result.cluster.end_time.isoformat(),
                "event_count": result.cluster.event_count,
                "total_notional_usd": str(result.cluster.total_notional_usd),
                "avg_price": str(result.cluster.avg_price),
                "min_price": str(result.cluster.min_price),
                "max_price": str(result.cluster.max_price),
                "duration_seconds": result.cluster.duration_seconds,
                "avg_notional_per_event": str(result.cluster.avg_notional_per_event),
            },
            "strategy": {
                "min_confidence": self.config.min_confidence,
                "min_intensity_score": self.config.min_intensity_score,
                "min_continuation_bias": self.config.min_continuation_bias,
                "allowed_severities": [item.value for item in self.config.allowed_severities],
            },
            "bus_event": {
                "topic": bus_event.topic,
                "event_id": bus_event.event_id,
                "source": bus_event.source,
                "priority": int(bus_event.priority),
                "correlation_id": bus_event.correlation_id,
                "headers": dict(bus_event.headers),
            },
            "detector_metadata": dict(result.metadata),
        }

        return LiquidationCascadeSignal(
            strategy_name=self.config.strategy_name,
            signal_type=self.config.signal_type,
            exchange=result.exchange,
            symbol=result.symbol,
            side=trade_side,
            confidence=clamp_float(result.confidence),
            score=score,
            generated_at=generated_at,
            detected_at=ensure_utc(result.detected_at),
            reason=reason,
            source_topic=bus_event.topic,
            severity=result.severity.value,
            cascade_direction=result.direction.value,
            liquidation_side=result.side.value,
            event_count=result.event_count,
            total_notional_usd=result.total_notional_usd,
            intensity_score=clamp_float(result.intensity_score),
            continuation_bias=clamp_float(result.continuation_bias),
            exhaustion_bias=clamp_float(result.exhaustion_bias),
            price_range_pct=result.price_range_pct,
            correlation_id=bus_event.correlation_id,
            source_event_id=bus_event.event_id,
            metadata=metadata,
        )

    def _compute_strategy_score(self, result: CascadeDetectionResult) -> float:
        severity_score = self._severity_to_score(result.severity)

        total_weight = (
            self.config.score_confidence_weight
            + self.config.score_continuation_bias_weight
            + self.config.score_intensity_weight
            + self.config.score_severity_weight
        )
        if total_weight <= 0:
            return 0.0

        weighted_score = (
            clamp_float(result.confidence) * self.config.score_confidence_weight
            + clamp_float(result.continuation_bias) * self.config.score_continuation_bias_weight
            + clamp_float(result.intensity_score) * self.config.score_intensity_weight
            + severity_score * self.config.score_severity_weight
        ) / total_weight

        return clamp_float(weighted_score)

    def _severity_to_score(self, severity: CascadeSeverity) -> float:
        if severity == CascadeSeverity.EXTREME:
            return 1.0
        if severity == CascadeSeverity.HIGH:
            return 0.8
        if severity == CascadeSeverity.MEDIUM:
            return 0.6
        return 0.4

    def _direction_to_trade_side(self, direction: CascadeDirection) -> str:
        if direction == CascadeDirection.DOWN:
            return "SHORT"
        if direction == CascadeDirection.UP:
            return "LONG"
        return "FLAT"

    def _build_cluster_signature(self, result: CascadeDetectionResult) -> str:
        cluster = result.cluster
        return (
            f"{result.exchange.lower()}|{result.symbol.upper()}|"
            f"{result.direction.value}|{result.side.value}|"
            f"{cluster.start_time.isoformat()}|{cluster.end_time.isoformat()}|"
            f"{cluster.event_count}|{cluster.total_notional_usd}|{result.detected_at.isoformat()}"
        )

    # ------------------------------------------------------------------
    # Emit / reject / memory
    # ------------------------------------------------------------------

    async def _emit_signal(
        self,
        signal: LiquidationCascadeSignal,
        *,
        bus_event: Event,
    ) -> None:
        await self.event_bus.emit(
            self.config.publish_topic_signal_generated,
            signal,
            priority=self.config.signal_priority,
            source=self.config.strategy_name,
            correlation_id=bus_event.correlation_id or bus_event.event_id,
            headers={
                "strategy": self.config.strategy_name,
                "signal_type": self.config.signal_type,
                "exchange": signal.exchange,
                "symbol": signal.symbol,
                "side": signal.side,
                "source_event_id": bus_event.event_id,
                "source_topic": bus_event.topic,
            },
        )

    async def _reject_result(
        self,
        *,
        result: CascadeDetectionResult,
        bus_event: Event,
        reason: str,
    ) -> None:
        self._stats.rejected_events += 1

        rejection = StrategyRejection(
            exchange=result.exchange,
            symbol=result.symbol,
            rejected_at=utc_now(),
            reason=reason,
            source_topic=bus_event.topic,
            correlation_id=bus_event.correlation_id,
            source_event_id=bus_event.event_id,
            details={
                "severity": result.severity.value,
                "direction": result.direction.value,
                "confidence": result.confidence,
                "intensity_score": result.intensity_score,
                "continuation_bias": result.continuation_bias,
                "exhaustion_bias": result.exhaustion_bias,
                "event_count": result.event_count,
                "total_notional_usd": str(result.total_notional_usd),
                "price_range_pct": result.price_range_pct,
            },
        )

        self._remember_rejection(rejection)

        self.logger.debug(
            "Liquidation cascade result rejected by strategy filters.",
            extra={
                "exchange": result.exchange,
                "symbol": result.symbol,
                "reason": reason,
                "severity": result.severity.value,
                "confidence": result.confidence,
                "continuation_bias": result.continuation_bias,
                "event_id": bus_event.event_id,
                "correlation_id": bus_event.correlation_id,
            },
        )

        if self.config.publish_rejected_events:
            await self.event_bus.emit(
                self.config.publish_topic_signal_rejected,
                rejection,
                priority=self.config.rejection_priority,
                source=self.config.strategy_name,
                correlation_id=bus_event.correlation_id or bus_event.event_id,
                headers={
                    "strategy": self.config.strategy_name,
                    "exchange": result.exchange,
                    "symbol": result.symbol,
                    "reason": reason,
                    "source_event_id": bus_event.event_id,
                    "source_topic": bus_event.topic,
                },
            )

    def _remember_signal(self, signal: LiquidationCascadeSignal) -> None:
        self._recent_signals.append(signal)
        if len(self._recent_signals) > self.config.recent_signals_limit:
            self._recent_signals = self._recent_signals[-self.config.recent_signals_limit :]

    def _remember_rejection(self, rejection: StrategyRejection) -> None:
        self._recent_rejections.append(rejection)
        if len(self._recent_rejections) > self.config.recent_rejections_limit:
            self._recent_rejections = self._recent_rejections[-self.config.recent_rejections_limit :]

    # ------------------------------------------------------------------
    # State / snapshots
    # ------------------------------------------------------------------

    def _get_or_create_state(self, exchange: str, symbol: str) -> SymbolCascadeStrategyState:
        key = (exchange.lower(), normalize_symbol(symbol))
        state = self._states.get(key)
        if state is None:
            state = SymbolCascadeStrategyState(
                exchange=exchange.lower(),
                symbol=normalize_symbol(symbol),
            )
            self._states[key] = state
        return state

    def get_recent_signals(
        self,
        *,
        exchange: str | None = None,
        symbol: str | None = None,
        limit: int = 50,
    ) -> list[LiquidationCascadeSignal]:
        target_exchange = exchange.lower() if exchange else None
        target_symbol = normalize_symbol(symbol) if symbol else None

        result: list[LiquidationCascadeSignal] = []
        for signal in reversed(self._recent_signals):
            if target_exchange and signal.exchange.lower() != target_exchange:
                continue
            if target_symbol and signal.symbol != target_symbol:
                continue
            result.append(signal)
            if len(result) >= limit:
                break

        return result

    def get_recent_rejections(
        self,
        *,
        exchange: str | None = None,
        symbol: str | None = None,
        limit: int = 50,
    ) -> list[StrategyRejection]:
        target_exchange = exchange.lower() if exchange else None
        target_symbol = normalize_symbol(symbol) if symbol else None

        result: list[StrategyRejection] = []
        for item in reversed(self._recent_rejections):
            if target_exchange and item.exchange.lower() != target_exchange:
                continue
            if target_symbol and item.symbol != target_symbol:
                continue
            result.append(item)
            if len(result) >= limit:
                break

        return result

    def get_symbol_state_snapshot(self, exchange: str, symbol: str) -> dict[str, Any]:
        key = (exchange.lower(), normalize_symbol(symbol))
        state = self._states.get(key)

        if state is None:
            return {
                "exchange": exchange.lower(),
                "symbol": normalize_symbol(symbol),
                "exists": False,
            }

        now = utc_now()
        state.prune_old_signal_timestamps(now, self.config.signal_window_seconds)

        return {
            "exchange": state.exchange,
            "symbol": state.symbol,
            "exists": True,
            "last_signal_at": state.last_signal_at.isoformat() if state.last_signal_at else None,
            "cooldown_until": state.cooldown_until.isoformat() if state.cooldown_until else None,
            "last_signal_side": state.last_signal_side,
            "last_detected_at": state.last_detected_at.isoformat() if state.last_detected_at else None,
            "last_cluster_signature": state.last_cluster_signature,
            "last_signal_score": state.last_signal_score,
            "total_signals_emitted": state.total_signals_emitted,
            "signals_in_window": len(state.signal_timestamps),
            "is_in_cooldown": state.is_in_cooldown(now),
        }

    def get_hot_symbols(self, limit: int = 10) -> list[dict[str, Any]]:
        latest_by_key: dict[tuple[str, str], LiquidationCascadeSignal] = {}

        for signal in self._recent_signals:
            key = (signal.exchange.lower(), signal.symbol)
            previous = latest_by_key.get(key)
            if previous is None or signal.generated_at > previous.generated_at:
                latest_by_key[key] = signal

        rows = [
            {
                "exchange": signal.exchange,
                "symbol": signal.symbol,
                "side": signal.side,
                "score": signal.score,
                "confidence": signal.confidence,
                "severity": signal.severity,
                "intensity_score": signal.intensity_score,
                "continuation_bias": signal.continuation_bias,
                "generated_at": signal.generated_at.isoformat(),
                "total_notional_usd": str(signal.total_notional_usd),
            }
            for signal in latest_by_key.values()
        ]

        rows.sort(
            key=lambda row: (
                float(row["score"]),
                float(row["confidence"]),
                float(row["intensity_score"]),
            ),
            reverse=True,
        )
        return rows[:limit]

    def get_stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "started_at": self._stats.started_at.isoformat() if self._stats.started_at else None,
            "stopped_at": self._stats.stopped_at.isoformat() if self._stats.stopped_at else None,
            "processed_events": self._stats.processed_events,
            "emitted_signals": self._stats.emitted_signals,
            "rejected_events": self._stats.rejected_events,
            "duplicate_skips": self._stats.duplicate_skips,
            "cooldown_skips": self._stats.cooldown_skips,
            "rate_limit_skips": self._stats.rate_limit_skips,
            "filter_skips": self._stats.filter_skips,
            "invalid_payload_skips": self._stats.invalid_payload_skips,
            "tracked_symbols": len(self._states),
            "recent_signals": len(self._recent_signals),
            "recent_rejections": len(self._recent_rejections),
            "last_signal_at": self._stats.last_signal_at.isoformat() if self._stats.last_signal_at else None,
            "last_error_at": self._stats.last_error_at.isoformat() if self._stats.last_error_at else None,
            "last_error": self._stats.last_error,
        }

    # ------------------------------------------------------------------
    # Scheduler / diagnostics
    # ------------------------------------------------------------------

    def _register_scheduler_jobs(self) -> None:
        if self.scheduler is None:
            return

        if not self.config.publish_diagnostics_snapshots:
            return

        self._diagnostics_job_id = self.scheduler.add_interval_job(
            name=f"{self.config.strategy_name}:diagnostics",
            func=self._publish_diagnostics_snapshot,
            interval=self.config.diagnostics_interval_seconds,
            run_immediately=False,
            max_retries=0,
            retry_delay=1.0,
            timeout=10.0,
            allow_overlap=False,
            enabled=True,
        )

    async def _publish_diagnostics_snapshot(self) -> None:
        if not self._running:
            return

        snapshot = {
            "strategy_name": self.config.strategy_name,
            "signal_type": self.config.signal_type,
            "created_at": utc_now().isoformat(),
            "stats": self.get_stats(),
            "hot_symbols": self.get_hot_symbols(limit=10),
        }

        await self.event_bus.emit(
            self.config.diagnostics_topic,
            snapshot,
            priority=self.config.diagnostics_priority,
            source=self.config.strategy_name,
            correlation_id=None,
            headers={
                "strategy": self.config.strategy_name,
                "snapshot_type": "diagnostics",
            },
        )