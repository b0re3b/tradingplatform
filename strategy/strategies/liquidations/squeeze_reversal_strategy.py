from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from core.event_bus import Event, EventBus, EventPriority
from core.scheduler import Scheduler

from analytics.liquidations.enums import CascadeDirection, CascadeSeverity
from analytics.liquidations.models import CascadeDetectionResult

from strategy.strategies.liquidations.base import (
    BaseAnalyticsStrategy,
    BaseStrategyStats,
    BaseSymbolStrategyState,
    FilterResult,
    StrategyRejection,
    clamp_float,
    ensure_utc,
    normalize_symbol,
    serialize_value,
    utc_now,
)


DECIMAL_ZERO = Decimal("0")


def _safe_decimal(value: Any, default: Decimal = DECIMAL_ZERO) -> Decimal:
    if value is None:
        return default

    if isinstance(value, Decimal):
        return value

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================================
# Config
# ============================================================================


@dataclass(slots=True)
class SqueezeReversalStrategyConfig:
    """
    Exhaustion/reversal strategy поверх analytics.liquidation.exhaustion_detected.

    Ідея:
    - analytics уже визначив exhaustion-сценарій після liquidation cascade;
    - strategy перевіряє якість exhaustion-сигналу;
    - використовує cluster, bias delta, imbalance, acceleration, severity, freshness;
    - ставить candidate у pending-confirmation;
    - після затримки підтверджує reversal, якщо сигнал не застарів і не був перебитий
      новішим/сильнішим analytics result.
    """

    enabled: bool = True

    subscribe_topic: str = "analytics.liquidation.exhaustion_detected"
    publish_topic_signal_generated: str = "signal.generated"
    publish_topic_signal_rejected: str = "signal.rejected"

    publish_topic_pending_created: str = "strategy.liquidations.squeeze.pending_created"
    publish_topic_pending_expired: str = "strategy.liquidations.squeeze.pending_expired"
    publish_topic_pending_replaced: str = "strategy.liquidations.squeeze.pending_replaced"
    publish_topic_pending_confirmed: str = "strategy.liquidations.squeeze.pending_confirmed"

    publish_rejected_events: bool = False
    publish_pending_events: bool = True
    publish_diagnostics_snapshots: bool = False

    diagnostics_topic: str = "strategy.liquidations.squeeze.snapshot"
    diagnostics_interval_seconds: float = 30.0

    strategy_name: str = "squeeze_reversal_strategy"
    signal_type: str = "reversal"
    service_name: str = "squeeze_reversal_strategy"

    signal_priority: EventPriority = EventPriority.HIGH
    rejection_priority: EventPriority = EventPriority.LOW
    pending_priority: EventPriority = EventPriority.LOW
    diagnostics_priority: EventPriority = EventPriority.LOW

    allowed_exchanges: tuple[str, ...] = ()
    allowed_symbols: tuple[str, ...] = ()
    blocked_symbols: tuple[str, ...] = ()

    allowed_severities: tuple[CascadeSeverity, ...] = (
        CascadeSeverity.HIGH,
        CascadeSeverity.EXTREME,
    )

    # Base quality filters
    min_confidence: float = 0.65
    min_intensity_score: float = 0.60
    min_total_notional_usd: Decimal = Decimal("400000")
    min_event_count: int = 6
    max_price_range_pct: float | None = None

    # Exhaustion-specific filters
    require_favors_exhaustion: bool = True
    require_high_confidence_only: bool = False
    require_actionable_severity: bool = True

    min_exhaustion_bias: float = 0.70
    min_bias_delta: float = 0.12
    max_continuation_bias_after_exhaustion: float | None = 0.55

    # Detector metadata filters
    min_side_imbalance_ratio: float | None = 0.70
    min_event_imbalance_ratio: float | None = None
    min_climax_acceleration_ratio: float | None = 1.10

    # Cluster-shape filters
    max_cluster_duration_seconds: float | None = 12.0
    min_avg_notional_per_event: Decimal | None = Decimal("50000")

    # Freshness / clock skew
    max_result_age_seconds: float = 20.0
    max_future_detected_at_seconds: float = 5.0

    # Pending confirmation model
    enable_pending_confirmation: bool = True
    confirmation_delay_seconds: float = 2.0
    pending_ttl_seconds: float = 8.0
    min_pending_age_seconds: float = 1.5
    pending_scan_interval_seconds: float = 0.5

    # Якщо після candidate прийшов новіший analytics result по тому ж символу,
    # старий reversal candidate краще скасувати.
    cancel_if_newer_detected_at: bool = True

    # Якщо новіший candidate сильніший — замінити pending.
    replace_pending_if_score_improves: bool = True
    min_replacement_score_delta: float = 0.03

    symbol_cooldown_seconds: int = 35
    min_seconds_between_same_side_signals: int = 20

    max_signals_per_symbol_window: int = 1
    signal_window_seconds: int = 90

    deduplicate_by_detected_at: bool = True
    deduplicate_same_cluster_signature: bool = True

    recent_signals_limit: int = 200
    recent_rejections_limit: int = 200
    recent_pending_limit: int = 200

    hot_symbols_window_seconds: int | None = 300

    # Scoring
    score_confidence_weight: float = 0.25
    score_exhaustion_bias_weight: float = 0.30
    score_bias_delta_weight: float = 0.15
    score_intensity_weight: float = 0.12
    score_severity_weight: float = 0.10
    score_cluster_quality_weight: float = 0.08

    def validate(self) -> None:
        bounded = {
            "min_confidence": self.min_confidence,
            "min_intensity_score": self.min_intensity_score,
            "min_exhaustion_bias": self.min_exhaustion_bias,
            "min_bias_delta": self.min_bias_delta,
        }

        for name, value in bounded.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be between 0 and 1")

        if self.max_continuation_bias_after_exhaustion is not None:
            if not (0.0 <= self.max_continuation_bias_after_exhaustion <= 1.0):
                raise ValueError("max_continuation_bias_after_exhaustion must be between 0 and 1 or None")

        if self.min_total_notional_usd < 0:
            raise ValueError("min_total_notional_usd must be >= 0")

        if self.min_event_count < 0:
            raise ValueError("min_event_count must be >= 0")

        if self.max_price_range_pct is not None and self.max_price_range_pct < 0:
            raise ValueError("max_price_range_pct must be >= 0 or None")

        if self.min_side_imbalance_ratio is not None and not (0.0 <= self.min_side_imbalance_ratio <= 1.0):
            raise ValueError("min_side_imbalance_ratio must be between 0 and 1 or None")

        if self.min_event_imbalance_ratio is not None and not (0.0 <= self.min_event_imbalance_ratio <= 1.0):
            raise ValueError("min_event_imbalance_ratio must be between 0 and 1 or None")

        if self.min_climax_acceleration_ratio is not None and self.min_climax_acceleration_ratio < 0:
            raise ValueError("min_climax_acceleration_ratio must be >= 0 or None")

        if self.max_cluster_duration_seconds is not None and self.max_cluster_duration_seconds <= 0:
            raise ValueError("max_cluster_duration_seconds must be > 0 or None")

        if self.min_avg_notional_per_event is not None and self.min_avg_notional_per_event < 0:
            raise ValueError("min_avg_notional_per_event must be >= 0 or None")

        if self.max_result_age_seconds <= 0:
            raise ValueError("max_result_age_seconds must be > 0")

        if self.max_future_detected_at_seconds < 0:
            raise ValueError("max_future_detected_at_seconds must be >= 0")

        if self.confirmation_delay_seconds < 0:
            raise ValueError("confirmation_delay_seconds must be >= 0")

        if self.pending_ttl_seconds <= 0:
            raise ValueError("pending_ttl_seconds must be > 0")

        if self.min_pending_age_seconds < 0:
            raise ValueError("min_pending_age_seconds must be >= 0")

        if self.pending_scan_interval_seconds <= 0:
            raise ValueError("pending_scan_interval_seconds must be > 0")

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

        if self.hot_symbols_window_seconds is not None and self.hot_symbols_window_seconds <= 0:
            raise ValueError("hot_symbols_window_seconds must be > 0 or None")

        weights = {
            "score_confidence_weight": self.score_confidence_weight,
            "score_exhaustion_bias_weight": self.score_exhaustion_bias_weight,
            "score_bias_delta_weight": self.score_bias_delta_weight,
            "score_intensity_weight": self.score_intensity_weight,
            "score_severity_weight": self.score_severity_weight,
            "score_cluster_quality_weight": self.score_cluster_quality_weight,
        }

        for name, weight in weights.items():
            if weight < 0:
                raise ValueError(f"{name} must be >= 0")

        if sum(weights.values()) <= 0:
            raise ValueError("strategy score weights sum must be > 0")


# ============================================================================
# Models
# ============================================================================


@dataclass(slots=True)
class SqueezeReversalSignal:
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
    bias_delta: float
    price_range_pct: float
    window_seconds: int

    cluster_duration_seconds: float
    cluster_avg_notional_per_event: Decimal

    side_imbalance_ratio: float | None = None
    event_imbalance_ratio: float | None = None
    acceleration_ratio: float | None = None

    pending_started_at: datetime | None = None
    pending_confirmed_at: datetime | None = None

    correlation_id: str | None = None
    source_event_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_pending_confirmed(self) -> bool:
        return self.pending_started_at is not None and self.pending_confirmed_at is not None

    @property
    def confirmation_delay_seconds(self) -> float | None:
        if self.pending_started_at is None or self.pending_confirmed_at is None:
            return None

        return max(
            0.0,
            (ensure_utc(self.pending_confirmed_at) - ensure_utc(self.pending_started_at)).total_seconds(),
        )

    @property
    def is_long(self) -> bool:
        return self.side.upper() == "LONG"

    @property
    def is_short(self) -> bool:
        return self.side.upper() == "SHORT"

    def to_dict(self, *, serialize: bool = True) -> dict[str, Any]:
        data = asdict(self)

        data["is_pending_confirmed"] = self.is_pending_confirmed
        data["confirmation_delay_seconds"] = self.confirmation_delay_seconds
        data["is_long"] = self.is_long
        data["is_short"] = self.is_short

        return serialize_value(data) if serialize else data


@dataclass(slots=True)
class PendingReversalCandidate:
    exchange: str
    symbol: str

    result: CascadeDetectionResult
    source_topic: str
    source_event_id: str | None
    correlation_id: str | None

    created_at: datetime
    confirm_after: datetime
    expires_at: datetime

    cluster_signature: str
    score_at_creation: float
    quality_snapshot: dict[str, Any] = field(default_factory=dict)

    cancelled: bool = False
    cancel_reason: str | None = None

    candidate_detected_at: datetime = field(init=False)

    def __post_init__(self) -> None:
        self.candidate_detected_at = ensure_utc(self.result.detected_at)

    def is_ready(self, now: datetime) -> bool:
        return ensure_utc(now) >= ensure_utc(self.confirm_after)

    def is_expired(self, now: datetime) -> bool:
        return ensure_utc(now) > ensure_utc(self.expires_at)


@dataclass(slots=True)
class SymbolSqueezeStrategyState(BaseSymbolStrategyState):
    pending: PendingReversalCandidate | None = None

    latest_seen_detected_at: datetime | None = None
    latest_seen_score: float | None = None

    def update_latest_seen(
        self,
        *,
        detected_at: datetime,
        score: float,
    ) -> None:
        ts = ensure_utc(detected_at)

        if self.latest_seen_detected_at is None or ts > ensure_utc(self.latest_seen_detected_at):
            self.latest_seen_detected_at = ts
            self.latest_seen_score = score


@dataclass(slots=True)
class SqueezeReversalStrategyStats(BaseStrategyStats):
    pending_created: int = 0
    pending_confirmed: int = 0
    pending_cancelled: int = 0
    pending_expired: int = 0
    pending_replaced: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = BaseStrategyStats.to_dict(self)
        data.update(
            {
                "pending_created": self.pending_created,
                "pending_confirmed": self.pending_confirmed,
                "pending_cancelled": self.pending_cancelled,
                "pending_expired": self.pending_expired,
                "pending_replaced": self.pending_replaced,
            }
        )
        return data


# ============================================================================
# Strategy
# ============================================================================


class SqueezeReversalStrategy(
    BaseAnalyticsStrategy[
        CascadeDetectionResult,
        SqueezeReversalSignal,
        SymbolSqueezeStrategyState,
        SqueezeReversalStrategyConfig,
    ]
):
    """
    Exhaustion reversal strategy.

    Pipeline:
        analytics.liquidation.exhaustion_detected
            -> common filters
            -> exhaustion-specific analytics filters
            -> pending confirmation
            -> SqueezeReversalSignal
            -> signal.generated

    Direction:
        CascadeDirection.DOWN -> LONG
        CascadeDirection.UP   -> SHORT
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        config: SqueezeReversalStrategyConfig | None = None,
        scheduler: Scheduler | None = None,
        service_name: str | None = None,
    ) -> None:
        super().__init__(
            event_bus=event_bus,
            scheduler=scheduler,
            config=config or SqueezeReversalStrategyConfig(),
            service_name=service_name,
            component="strategy.liquidations.squeeze_reversal_strategy",
            payload_type=CascadeDetectionResult,
        )

        self._stats = SqueezeReversalStrategyStats()
        self._pending_scan_job_id: str | None = None
        self._pending_keys: set[tuple[str, str]] = set()
        self._recent_pending: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Base hooks
    # ------------------------------------------------------------------

    def create_symbol_state(
        self,
        *,
        exchange: str,
        symbol: str,
    ) -> SymbolSqueezeStrategyState:
        return SymbolSqueezeStrategyState(
            exchange=exchange.lower(),
            symbol=normalize_symbol(symbol),
        )

    def direction_to_trade_side(self, result: CascadeDetectionResult) -> str:
        if result.direction is CascadeDirection.DOWN:
            return "LONG"

        if result.direction is CascadeDirection.UP:
            return "SHORT"

        return "FLAT"

    async def process_result(
        self,
        result: CascadeDetectionResult,
        *,
        bus_event: Event,
    ) -> None:
        state = self.get_or_create_state(result.exchange, result.symbol)
        now = utc_now()

        current_score = self.compute_strategy_score(result)
        state.update_latest_seen(
            detected_at=result.detected_at,
            score=current_score,
        )

        filter_result = self.evaluate_filters(
            result=result,
            state=state,
            now=now,
        )

        if filter_result.rejection_reason is not None:
            await self.reject_result(
                result=result,
                bus_event=bus_event,
                reason=filter_result.rejection_reason,
            )
            return

        if self.config.enable_pending_confirmation:
            await self.create_or_replace_pending_candidate(
                result=result,
                bus_event=bus_event,
                state=state,
                now=now,
                cluster_signature=filter_result.cluster_signature or self.build_cluster_signature(result),
                score=current_score,
            )
            return

        await self.emit_confirmed_signal(
            result=result,
            bus_event=bus_event,
            state=state,
            cluster_signature=filter_result.cluster_signature,
            pending_started_at=None,
            pending_confirmed_at=now,
            source_event_id=bus_event.event_id,
        )

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def evaluate_filters(
        self,
        *,
        result: CascadeDetectionResult,
        state: SymbolSqueezeStrategyState,
        now: datetime,
    ) -> FilterResult:
        common = self.evaluate_common_filters(
            result=result,
            state=state,
            now=now,
        )

        if common.rejection_reason is not None:
            return common

        custom_rejection = self.get_exhaustion_rejection_reason(
            result=result,
            now=now,
        )

        if custom_rejection is not None:
            return FilterResult(rejection_reason=custom_rejection)

        return common

    def get_exhaustion_rejection_reason(
        self,
        *,
        result: CascadeDetectionResult,
        now: datetime,
    ) -> str | None:
        if not result.is_confirmed:
            self._stats.filter_skips += 1
            return "result_not_confirmed"

        if self.config.require_actionable_severity and not result.is_actionable_severity:
            self._stats.filter_skips += 1
            return "severity_not_actionable"

        if self.config.require_favors_exhaustion and not result.favors_exhaustion:
            self._stats.filter_skips += 1
            return "exhaustion_not_favored"

        if result.exhaustion_bias < self.config.min_exhaustion_bias:
            self._stats.filter_skips += 1
            return "exhaustion_bias_below_threshold"

        if result.bias_delta < self.config.min_bias_delta:
            self._stats.filter_skips += 1
            return "bias_delta_below_threshold"

        if (
            self.config.max_continuation_bias_after_exhaustion is not None
            and result.continuation_bias > self.config.max_continuation_bias_after_exhaustion
        ):
            self._stats.filter_skips += 1
            return "continuation_bias_too_high_for_reversal"

        detected_at = ensure_utc(result.detected_at)

        if detected_at > ensure_utc(now) + timedelta(seconds=self.config.max_future_detected_at_seconds):
            self._stats.filter_skips += 1
            return "detected_at_in_future"

        age_seconds = (ensure_utc(now) - detected_at).total_seconds()
        if age_seconds > self.config.max_result_age_seconds:
            self._stats.filter_skips += 1
            return "result_too_old"

        if self.config.max_cluster_duration_seconds is not None:
            if result.cluster.duration_seconds > self.config.max_cluster_duration_seconds:
                self._stats.filter_skips += 1
                return "cluster_duration_too_long"

        if self.config.min_avg_notional_per_event is not None:
            if result.cluster.avg_notional_per_event < self.config.min_avg_notional_per_event:
                self._stats.filter_skips += 1
                return "avg_notional_per_event_below_threshold"

        analytics_meta = self.extract_analytics_metadata(result)

        if self.config.min_side_imbalance_ratio is not None:
            value = analytics_meta["side_imbalance_ratio"]
            if value is None or value < self.config.min_side_imbalance_ratio:
                self._stats.filter_skips += 1
                return "side_imbalance_below_threshold"

        if self.config.min_event_imbalance_ratio is not None:
            value = analytics_meta["event_imbalance_ratio"]
            if value is None or value < self.config.min_event_imbalance_ratio:
                self._stats.filter_skips += 1
                return "event_imbalance_below_threshold"

        if self.config.min_climax_acceleration_ratio is not None:
            value = analytics_meta["acceleration_ratio"]
            if value is None or value < self.config.min_climax_acceleration_ratio:
                self._stats.filter_skips += 1
                return "acceleration_below_climax_threshold"

        return None

    # ------------------------------------------------------------------
    # Pending confirmation
    # ------------------------------------------------------------------

    async def create_or_replace_pending_candidate(
        self,
        *,
        result: CascadeDetectionResult,
        bus_event: Event,
        state: SymbolSqueezeStrategyState,
        now: datetime,
        cluster_signature: str,
        score: float,
    ) -> None:
        if state.pending is not None:
            existing = state.pending

            incoming_is_newer = ensure_utc(result.detected_at) > ensure_utc(existing.result.detected_at)
            incoming_is_stronger = score >= existing.score_at_creation + self.config.min_replacement_score_delta

            if not incoming_is_newer:
                self._stats.duplicate_skips += 1
                await self.reject_result(
                    result=result,
                    bus_event=bus_event,
                    reason="older_than_existing_pending",
                )
                return

            if self.config.replace_pending_if_score_improves and not incoming_is_stronger:
                self._stats.duplicate_skips += 1
                await self.reject_result(
                    result=result,
                    bus_event=bus_event,
                    reason="newer_pending_not_stronger_enough",
                )
                return

            existing.cancelled = True
            existing.cancel_reason = "replaced_by_newer_stronger_pending"
            self._stats.pending_cancelled += 1
            self._stats.pending_replaced += 1

            if self.config.publish_pending_events:
                await self.emit_pending_event(
                    topic=self.config.publish_topic_pending_replaced,
                    candidate=existing,
                    reason=existing.cancel_reason,
                    correlation_id=bus_event.correlation_id or bus_event.event_id,
                )

        candidate = PendingReversalCandidate(
            exchange=result.exchange,
            symbol=result.symbol,
            result=result,
            source_topic=bus_event.topic,
            source_event_id=bus_event.event_id,
            correlation_id=bus_event.correlation_id,
            created_at=ensure_utc(now),
            confirm_after=ensure_utc(now) + timedelta(seconds=self.config.confirmation_delay_seconds),
            expires_at=ensure_utc(now) + timedelta(seconds=self.config.pending_ttl_seconds),
            cluster_signature=cluster_signature,
            score_at_creation=score,
            quality_snapshot=self.build_quality_snapshot(result),
        )

        state.pending = candidate
        self._pending_keys.add(self.state_key(result.exchange, result.symbol))

        self._stats.pending_created += 1
        self.remember_pending(candidate, state="created")

        self.logger.info(
            "Squeeze reversal candidate moved to pending",
            extra={
                "strategy": self.config.strategy_name,
                "exchange": result.exchange,
                "symbol": result.symbol,
                "score": score,
                "confidence": result.confidence,
                "severity": result.severity.value,
                "exhaustion_bias": result.exhaustion_bias,
                "bias_delta": result.bias_delta,
                "confirm_after": candidate.confirm_after.isoformat(),
                "expires_at": candidate.expires_at.isoformat(),
                "event_id": bus_event.event_id,
                "correlation_id": bus_event.correlation_id,
            },
        )

        if self.config.publish_pending_events:
            await self.emit_pending_event(
                topic=self.config.publish_topic_pending_created,
                candidate=candidate,
                reason="pending_created",
                correlation_id=bus_event.correlation_id or bus_event.event_id,
            )

    async def process_pending_candidates(self) -> None:
        if not self._running:
            return

        now = utc_now()

        for key in list(self._pending_keys):
            state = self._states.get(key)

            if state is None:
                self._pending_keys.discard(key)
                continue

            candidate = state.pending

            if candidate is None:
                self._pending_keys.discard(key)
                continue

            if candidate.cancelled:
                state.pending = None
                self._pending_keys.discard(key)
                continue

            if candidate.is_expired(now):
                await self.expire_pending_candidate(
                    state=state,
                    candidate=candidate,
                    reason="pending_expired",
                )
                continue

            if not candidate.is_ready(now):
                continue

            age_seconds = (ensure_utc(now) - ensure_utc(candidate.created_at)).total_seconds()
            if age_seconds < self.config.min_pending_age_seconds:
                continue

            if (
                self.config.cancel_if_newer_detected_at
                and state.latest_seen_detected_at is not None
                and ensure_utc(state.latest_seen_detected_at) > ensure_utc(candidate.candidate_detected_at)
            ):
                await self.expire_pending_candidate(
                    state=state,
                    candidate=candidate,
                    reason="newer_detected_at_exists",
                )
                continue

            late_rejection = self.get_exhaustion_rejection_reason(
                result=candidate.result,
                now=now,
            )

            if late_rejection is not None:
                await self.expire_pending_candidate(
                    state=state,
                    candidate=candidate,
                    reason=f"late_filter_failed:{late_rejection}",
                )
                continue

            synthetic_event = Event(
                topic=candidate.source_topic,
                payload=candidate.result,
                priority=EventPriority.NORMAL,
                source=self.config.strategy_name,
                correlation_id=candidate.correlation_id,
                headers={"source_event_id": candidate.source_event_id}
                if candidate.source_event_id
                else {},
            )

            await self.emit_confirmed_signal(
                result=candidate.result,
                bus_event=synthetic_event,
                state=state,
                cluster_signature=candidate.cluster_signature,
                pending_started_at=candidate.created_at,
                pending_confirmed_at=now,
                source_event_id=candidate.source_event_id,
                pending_confirmation=True,
            )

            self._stats.pending_confirmed += 1

            if self.config.publish_pending_events:
                await self.emit_pending_event(
                    topic=self.config.publish_topic_pending_confirmed,
                    candidate=candidate,
                    reason="pending_confirmed",
                    correlation_id=candidate.correlation_id or candidate.source_event_id,
                )

            state.pending = None
            self._pending_keys.discard(key)

    async def expire_pending_candidate(
        self,
        *,
        state: SymbolSqueezeStrategyState,
        candidate: PendingReversalCandidate,
        reason: str,
    ) -> None:
        self._stats.pending_expired += 1

        candidate.cancelled = True
        candidate.cancel_reason = reason

        state.pending = None
        self._pending_keys.discard(self.state_key(candidate.exchange, candidate.symbol))

        self.logger.info(
            "Pending squeeze reversal candidate expired",
            extra={
                "strategy": self.config.strategy_name,
                "exchange": candidate.exchange,
                "symbol": candidate.symbol,
                "reason": reason,
                "created_at": candidate.created_at.isoformat(),
                "expires_at": candidate.expires_at.isoformat(),
                "correlation_id": candidate.correlation_id,
                "source_event_id": candidate.source_event_id,
            },
        )

        if self.config.publish_pending_events:
            await self.emit_pending_event(
                topic=self.config.publish_topic_pending_expired,
                candidate=candidate,
                reason=reason,
                correlation_id=candidate.correlation_id or candidate.source_event_id,
            )

    async def emit_pending_event(
        self,
        *,
        topic: str,
        candidate: PendingReversalCandidate,
        reason: str,
        correlation_id: str | None,
    ) -> bool:
        payload = {
            "strategy_name": self.config.strategy_name,
            "signal_type": self.config.signal_type,
            "exchange": candidate.exchange,
            "symbol": candidate.symbol,
            "state": reason,
            "created_at": candidate.created_at,
            "confirm_after": candidate.confirm_after,
            "expires_at": candidate.expires_at,
            "score_at_creation": candidate.score_at_creation,
            "source_event_id": candidate.source_event_id,
            "correlation_id": candidate.correlation_id,
            "cluster_signature": candidate.cluster_signature,
            "quality_snapshot": candidate.quality_snapshot,
        }

        return await self.emit_event(
            topic,
            serialize_value(payload),
            priority=self.config.pending_priority,
            correlation_id=correlation_id,
            headers={
                "strategy": self.config.strategy_name,
                "signal_type": self.config.signal_type,
                "exchange": candidate.exchange,
                "symbol": candidate.symbol,
                "state": reason,
            },
        )

    # ------------------------------------------------------------------
    # Signal building
    # ------------------------------------------------------------------

    async def emit_confirmed_signal(
        self,
        *,
        result: CascadeDetectionResult,
        bus_event: Event,
        state: SymbolSqueezeStrategyState,
        cluster_signature: str | None,
        pending_started_at: datetime | None,
        pending_confirmed_at: datetime | None,
        source_event_id: str | None,
        pending_confirmation: bool = False,
    ) -> bool:
        signal = self.build_signal(
            result=result,
            bus_event=bus_event,
            pending_started_at=pending_started_at,
            pending_confirmed_at=pending_confirmed_at,
            source_event_id=source_event_id,
        )

        headers = {
            "pending_confirmation": "true" if pending_confirmation else "false",
            "analytics_event_type": result.event_type.value,
        }

        emitted = await self.emit_signal(
            signal,
            bus_event=bus_event,
            headers=headers,
        )

        if not emitted:
            return False

        self.remember_emitted_signal(
            signal=signal,
            state=state,
            result=result,
            signal_side=signal.side,
            score=signal.score,
            cluster_signature=cluster_signature,
        )

        self.logger.info(
            "Squeeze reversal signal emitted",
            extra={
                "strategy": self.config.strategy_name,
                "exchange": signal.exchange,
                "symbol": signal.symbol,
                "side": signal.side,
                "score": signal.score,
                "confidence": signal.confidence,
                "severity": signal.severity,
                "exhaustion_bias": signal.exhaustion_bias,
                "bias_delta": signal.bias_delta,
                "event_count": signal.event_count,
                "total_notional_usd": str(signal.total_notional_usd),
                "pending_confirmation": pending_confirmation,
                "source_event_id": source_event_id,
                "correlation_id": signal.correlation_id,
            },
        )

        return True

    def build_signal(
        self,
        *,
        result: CascadeDetectionResult,
        bus_event: Event,
        pending_started_at: datetime | None,
        pending_confirmed_at: datetime | None,
        source_event_id: str | None,
    ) -> SqueezeReversalSignal:
        trade_side = self.direction_to_trade_side(result)
        generated_at = utc_now()
        score = self.compute_strategy_score(result)
        analytics_meta = self.extract_analytics_metadata(result)

        reason = (
            "squeeze reversal after liquidation exhaustion: "
            f"direction={result.direction.value}, "
            f"side={result.side.value}, "
            f"severity={result.severity.value}, "
            f"exhaustion_bias={result.exhaustion_bias:.3f}, "
            f"continuation_bias={result.continuation_bias:.3f}, "
            f"bias_delta={result.bias_delta:.3f}, "
            f"confidence={result.confidence:.3f}"
        )

        metadata = self.build_common_signal_metadata(
            result=result,
            bus_event=bus_event,
        )

        metadata["squeeze_reversal"] = {
            "analytics_event_type": result.event_type.value,
            "favors_exhaustion": result.favors_exhaustion,
            "bias_delta": result.bias_delta,
            "score": score,
            "quality_snapshot": self.build_quality_snapshot(result),
            "pending": {
                "enabled": self.config.enable_pending_confirmation,
                "pending_started_at": pending_started_at.isoformat() if pending_started_at else None,
                "pending_confirmed_at": pending_confirmed_at.isoformat() if pending_confirmed_at else None,
            },
        }

        return SqueezeReversalSignal(
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
            bias_delta=clamp_float(result.bias_delta),
            price_range_pct=result.price_range_pct,
            window_seconds=result.window_seconds,
            cluster_duration_seconds=result.cluster.duration_seconds,
            cluster_avg_notional_per_event=result.cluster.avg_notional_per_event,
            side_imbalance_ratio=analytics_meta["side_imbalance_ratio"],
            event_imbalance_ratio=analytics_meta["event_imbalance_ratio"],
            acceleration_ratio=analytics_meta["acceleration_ratio"],
            pending_started_at=pending_started_at,
            pending_confirmed_at=pending_confirmed_at,
            correlation_id=bus_event.correlation_id or result.correlation_id,
            source_event_id=source_event_id or bus_event.event_id,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Scoring / analytics extraction
    # ------------------------------------------------------------------

    def compute_strategy_score(self, result: CascadeDetectionResult) -> float:
        total_weight = (
            self.config.score_confidence_weight
            + self.config.score_exhaustion_bias_weight
            + self.config.score_bias_delta_weight
            + self.config.score_intensity_weight
            + self.config.score_severity_weight
            + self.config.score_cluster_quality_weight
        )

        if total_weight <= 0:
            return 0.0

        cluster_quality = self.compute_cluster_quality_score(result)

        weighted_score = (
            clamp_float(result.confidence) * self.config.score_confidence_weight
            + clamp_float(result.exhaustion_bias) * self.config.score_exhaustion_bias_weight
            + clamp_float(result.bias_delta) * self.config.score_bias_delta_weight
            + clamp_float(result.intensity_score) * self.config.score_intensity_weight
            + self.severity_to_score(result.severity) * self.config.score_severity_weight
            + cluster_quality * self.config.score_cluster_quality_weight
        ) / total_weight

        return clamp_float(weighted_score)

    def compute_cluster_quality_score(self, result: CascadeDetectionResult) -> float:
        duration_score = 0.5
        avg_notional_score = 0.5
        price_range_score = 0.5

        if self.config.max_cluster_duration_seconds:
            duration_score = clamp_float(
                1.0 - (result.cluster.duration_seconds / self.config.max_cluster_duration_seconds)
            )

        if self.config.min_avg_notional_per_event and self.config.min_avg_notional_per_event > 0:
            avg_notional_score = clamp_float(
                float(result.cluster.avg_notional_per_event / self.config.min_avg_notional_per_event)
            )

        if self.config.max_price_range_pct and self.config.max_price_range_pct > 0:
            price_range_score = clamp_float(
                1.0 - (result.price_range_pct / self.config.max_price_range_pct)
            )

        return clamp_float((duration_score + avg_notional_score + price_range_score) / 3.0)

    def extract_analytics_metadata(
        self,
        result: CascadeDetectionResult,
    ) -> dict[str, float | None]:
        metadata = result.metadata or {}

        return {
            "side_imbalance_ratio": self.optional_float(metadata.get("side_imbalance_ratio")),
            "event_imbalance_ratio": self.optional_float(metadata.get("event_imbalance_ratio")),
            "acceleration_ratio": self.optional_float(metadata.get("acceleration_ratio")),
            "long_events": self.optional_float(metadata.get("long_events")),
            "short_events": self.optional_float(metadata.get("short_events")),
            "long_notional_usd": self.optional_float(metadata.get("long_notional_usd")),
            "short_notional_usd": self.optional_float(metadata.get("short_notional_usd")),
        }

    @staticmethod
    def optional_float(value: Any) -> float | None:
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def build_quality_snapshot(
        self,
        result: CascadeDetectionResult,
    ) -> dict[str, Any]:
        analytics_meta = self.extract_analytics_metadata(result)

        return {
            "confidence": result.confidence,
            "intensity_score": result.intensity_score,
            "continuation_bias": result.continuation_bias,
            "exhaustion_bias": result.exhaustion_bias,
            "bias_delta": result.bias_delta,
            "severity": result.severity.value,
            "event_type": result.event_type.value,
            "event_count": result.event_count,
            "total_notional_usd": str(result.total_notional_usd),
            "window_seconds": result.window_seconds,
            "price_range_pct": result.price_range_pct,
            "cluster": {
                "duration_seconds": result.cluster.duration_seconds,
                "avg_notional_per_event": str(result.cluster.avg_notional_per_event),
                "price_range_pct": result.cluster.price_range_pct,
                "event_count": result.cluster.event_count,
                "total_notional_usd": str(result.cluster.total_notional_usd),
            },
            "analytics_metadata": analytics_meta,
        }

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def _register_scheduler_jobs(self) -> None:
        super()._register_scheduler_jobs()

        if self.scheduler is None:
            return

        self._pending_scan_job_id = self.scheduler.add_interval_job(
            name=f"{self.config.strategy_name}:pending_scan",
            func=self.process_pending_candidates,
            interval=self.config.pending_scan_interval_seconds,
            run_immediately=False,
            max_retries=0,
            retry_delay=1.0,
            timeout=10.0,
            allow_overlap=False,
            enabled=True,
        )

    def _remove_scheduler_jobs(self) -> None:
        super()._remove_scheduler_jobs()

        if self.scheduler is None:
            self._pending_scan_job_id = None
            return

        if self._pending_scan_job_id is None:
            return

        try:
            self.scheduler.remove_job(self._pending_scan_job_id)
        except KeyError:
            pass
        finally:
            self._pending_scan_job_id = None

    # ------------------------------------------------------------------
    # Query / diagnostics
    # ------------------------------------------------------------------

    def remember_pending(
        self,
        candidate: PendingReversalCandidate,
        *,
        state: str,
    ) -> None:
        self._recent_pending.append(
            serialize_value(
                {
                    "state": state,
                    "exchange": candidate.exchange,
                    "symbol": candidate.symbol,
                    "created_at": candidate.created_at,
                    "confirm_after": candidate.confirm_after,
                    "expires_at": candidate.expires_at,
                    "score_at_creation": candidate.score_at_creation,
                    "source_event_id": candidate.source_event_id,
                    "correlation_id": candidate.correlation_id,
                    "quality_snapshot": candidate.quality_snapshot,
                }
            )
        )

        if len(self._recent_pending) > self.config.recent_pending_limit:
            self._recent_pending = self._recent_pending[-self.config.recent_pending_limit :]

    def get_recent_pending(
        self,
        *,
        exchange: str | None = None,
        symbol: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        target_exchange = exchange.lower() if exchange else None
        target_symbol = normalize_symbol(symbol) if symbol else None

        rows: list[dict[str, Any]] = []

        for item in reversed(self._recent_pending):
            if target_exchange and str(item.get("exchange", "")).lower() != target_exchange:
                continue

            if target_symbol and normalize_symbol(str(item.get("symbol", ""))) != target_symbol:
                continue

            rows.append(item)

            if len(rows) >= limit:
                break

        return rows

    def get_symbol_state_snapshot(
        self,
        exchange: str,
        symbol: str,
    ) -> dict[str, Any]:
        key = self.state_key(exchange, symbol)
        state = self._states.get(key)

        if state is None:
            return {
                "exchange": exchange.lower(),
                "symbol": normalize_symbol(symbol),
                "exists": False,
            }

        now = utc_now()
        state.prune_old_signal_timestamps(now, self.config.signal_window_seconds)

        pending = None
        if state.pending is not None:
            pending = serialize_value(
                {
                    "created_at": state.pending.created_at,
                    "confirm_after": state.pending.confirm_after,
                    "expires_at": state.pending.expires_at,
                    "cancelled": state.pending.cancelled,
                    "cancel_reason": state.pending.cancel_reason,
                    "source_event_id": state.pending.source_event_id,
                    "correlation_id": state.pending.correlation_id,
                    "cluster_signature": state.pending.cluster_signature,
                    "score_at_creation": state.pending.score_at_creation,
                    "quality_snapshot": state.pending.quality_snapshot,
                }
            )

        return serialize_value(
            {
                "exchange": state.exchange,
                "symbol": state.symbol,
                "exists": True,
                "last_signal_at": state.last_signal_at,
                "cooldown_until": state.cooldown_until,
                "last_signal_side": state.last_signal_side,
                "last_detected_at": state.last_detected_at,
                "latest_seen_detected_at": state.latest_seen_detected_at,
                "latest_seen_score": state.latest_seen_score,
                "last_cluster_signature": state.last_cluster_signature,
                "last_signal_score": state.last_signal_score,
                "total_signals_emitted": state.total_signals_emitted,
                "signals_in_window": len(state.signal_timestamps),
                "is_in_cooldown": state.is_in_cooldown(now),
                "pending": pending,
            }
        )

    def get_hot_symbols(self, *, limit: int = 20) -> list[dict[str, Any]]:
        now = utc_now()
        min_ts = None

        if self.config.hot_symbols_window_seconds is not None:
            min_ts = now - timedelta(seconds=self.config.hot_symbols_window_seconds)

        latest_by_key: dict[tuple[str, str], SqueezeReversalSignal] = {}

        for signal in self._recent_signals:
            if min_ts is not None and ensure_utc(signal.generated_at) < min_ts:
                continue

            key = (signal.exchange.lower(), normalize_symbol(signal.symbol))
            previous = latest_by_key.get(key)

            if previous is None or ensure_utc(signal.generated_at) > ensure_utc(previous.generated_at):
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
                "exhaustion_bias": signal.exhaustion_bias,
                "bias_delta": signal.bias_delta,
                "acceleration_ratio": signal.acceleration_ratio,
                "side_imbalance_ratio": signal.side_imbalance_ratio,
                "generated_at": signal.generated_at.isoformat(),
                "total_notional_usd": str(signal.total_notional_usd),
            }
            for signal in latest_by_key.values()
        ]

        rows.sort(
            key=lambda row: (
                float(row["score"]),
                float(row["confidence"]),
                float(row["exhaustion_bias"]),
                float(row["bias_delta"]),
            ),
            reverse=True,
        )

        return rows[: max(0, limit)]

    def get_stats(self) -> dict[str, Any]:
        data = super().get_stats()
        data.update(
            {
                "active_pending": len(self._pending_keys),
                "recent_pending": len(self._recent_pending),
                "pending_scan_job_registered": self._pending_scan_job_id is not None,
            }
        )
        return data

    async def publish_diagnostics_snapshot(self) -> None:
        if not self._running:
            return

        snapshot = {
            "strategy_name": self.config.strategy_name,
            "signal_type": self.config.signal_type,
            "created_at": utc_now().isoformat(),
            "stats": self.get_stats(),
            "hot_symbols": self.get_hot_symbols(limit=10),
            "pending": self.get_recent_pending(limit=10),
        }

        await self.emit_event(
            self.config.diagnostics_topic,
            snapshot,
            priority=self.config.diagnostics_priority,
            correlation_id=None,
            headers={
                "strategy": self.config.strategy_name,
                "signal_type": self.config.signal_type,
                "snapshot_type": "diagnostics",
            },
        )

    def _start_log_extra(self) -> dict[str, Any]:
        data = super()._start_log_extra()
        data.update(
            {
                "min_exhaustion_bias": self.config.min_exhaustion_bias,
                "min_bias_delta": self.config.min_bias_delta,
                "max_continuation_bias_after_exhaustion": self.config.max_continuation_bias_after_exhaustion,
                "min_side_imbalance_ratio": self.config.min_side_imbalance_ratio,
                "min_climax_acceleration_ratio": self.config.min_climax_acceleration_ratio,
                "enable_pending_confirmation": self.config.enable_pending_confirmation,
                "confirmation_delay_seconds": self.config.confirmation_delay_seconds,
                "pending_ttl_seconds": self.config.pending_ttl_seconds,
                "allowed_severities": [
                    severity.value for severity in self.config.allowed_severities
                ],
            }
        )
        return data

    def get_symbol_state(self, exchange: str, symbol: str) -> dict[str, Any]:
        key = self.state_key(exchange, symbol)
        state = self._states.get(key)

        normalized_exchange, normalized_symbol = key

        if state is None:
            return {
                "exists": False,
                "exchange": normalized_exchange,
                "symbol": normalized_symbol,
            }

        now = utc_now()

        pending = None
        if state.pending is not None:
            pending_candidate = state.pending
            pending = {
                "exchange": pending_candidate.exchange,
                "symbol": pending_candidate.symbol,
                "created_at": pending_candidate.created_at.isoformat(),
                "confirm_after": pending_candidate.confirm_after.isoformat(),
                "expires_at": pending_candidate.expires_at.isoformat(),
                "cluster_signature": pending_candidate.cluster_signature,
                "score_at_creation": pending_candidate.score_at_creation,
                "candidate_detected_at": pending_candidate.candidate_detected_at.isoformat(),
                "cancelled": pending_candidate.cancelled,
                "cancel_reason": pending_candidate.cancel_reason,
                "is_ready": pending_candidate.is_ready(now),
                "is_expired": pending_candidate.is_expired(now),
                "quality_snapshot": serialize_value(pending_candidate.quality_snapshot),
                "source_event_id": pending_candidate.source_event_id,
                "correlation_id": pending_candidate.correlation_id,
            }

        return {
            "exists": True,
            "exchange": state.exchange,
            "symbol": state.symbol,
            "last_signal_at": state.last_signal_at.isoformat() if state.last_signal_at else None,
            "cooldown_until": state.cooldown_until.isoformat() if state.cooldown_until else None,
            "last_signal_side": state.last_signal_side,
            "last_detected_at": state.last_detected_at.isoformat() if state.last_detected_at else None,
            "last_cluster_signature": state.last_cluster_signature,
            "last_signal_score": state.last_signal_score,
            "total_signals_emitted": state.total_signals_emitted,
            "signals_in_window": state.signals_in_window(
                now=now,
                window_seconds=self.config.signal_window_seconds,
            ),
            "is_in_cooldown": state.is_in_cooldown(now),
            "pending": pending,
            "latest_seen_detected_at": (
                state.latest_seen_detected_at.isoformat()
                if state.latest_seen_detected_at
                else None
            ),
            "latest_seen_score": state.latest_seen_score,
        }