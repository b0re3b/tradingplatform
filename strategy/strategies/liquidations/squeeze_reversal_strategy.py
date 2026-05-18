from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from core.event_bus import Event, EventBus, EventPriority
from core.scheduler import Scheduler

from analytics.liquidations.enums import CascadeDirection, CascadeSeverity, LiquidationStatus
from analytics.liquidations.models import (
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    CascadeDetectionResult,
    LiquidationKey,
    liquidation_key_to_dict,
)

from strategy.strategies.liquidations.base import (
    BaseAnalyticsStrategy,
    BaseStrategyStats,
    BaseSymbolStrategyState,
    FilterResult,
    StrategyRejection,
    clamp_float,
    ensure_utc,
    make_strategy_scope_key,
    normalize_exchange,
    normalize_exchange_symbol,
    normalize_market_type,
    normalize_symbol,
    normalize_timeframe,
    scoped_key_to_string,
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


# ============================================================================
# Config
# ============================================================================


@dataclass(slots=True)
class SqueezeReversalStrategyConfig:
    """
    Exhaustion/reversal strategy поверх analytics.liquidations.exhaustion_detected.

    Strategy:
    - слухає analytics.liquidations.exhaustion_detected;
    - приймає CascadeDetectionResult;
    - працює тільки з повним futures/liquidation scope:
        exchange + market_type + symbol + timeframe;
    - перевіряє exhaustion quality;
    - створює pending reversal candidate;
    - після confirmation delay генерує reversal signal;
    - не викликає risk/execution напряму.

    Reversal direction:
    - CascadeDirection.DOWN -> LONG
    - CascadeDirection.UP   -> SHORT
    """

    enabled: bool = True

    # Важливо: plural namespace, як у новому CascadeDetector.
    subscribe_topic: str = "analytics.liquidations.exhaustion_detected"
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

    # Full-scope filters.
    allowed_exchanges: tuple[str, ...] = ()
    allowed_market_types: tuple[str, ...] = ()
    allowed_symbols: tuple[str, ...] = ()
    allowed_timeframes: tuple[str, ...] = ()

    blocked_market_types: tuple[str, ...] = ()
    blocked_symbols: tuple[str, ...] = ()
    blocked_timeframes: tuple[str, ...] = ()

    allowed_severities: tuple[CascadeSeverity, ...] = (
        CascadeSeverity.HIGH,
        CascadeSeverity.EXTREME,
    )

    # Base quality filters.
    require_confirmed_result: bool = True
    require_actionable_direction: bool = True

    min_confidence: float = 0.65
    min_intensity_score: float = 0.60
    min_total_notional_usd: Decimal = Decimal("400000")
    min_event_count: int = 6
    max_price_range_pct: float | None = None

    # Exhaustion-specific filters.
    require_favors_exhaustion: bool = True
    require_high_confidence_only: bool = False
    require_actionable_severity: bool = True

    min_exhaustion_bias: float = 0.70
    min_bias_delta: float = 0.12
    max_continuation_bias_after_exhaustion: float | None = 0.55

    # Detector metadata filters.
    min_side_imbalance_ratio: float | None = 0.70
    min_event_imbalance_ratio: float | None = None
    min_climax_acceleration_ratio: float | None = 1.10

    # Cluster-shape filters.
    max_cluster_duration_seconds: float | None = 12.0
    min_avg_notional_per_event: Decimal | None = Decimal("50000")

    # Freshness / clock skew.
    max_result_age_seconds: float = 20.0
    max_future_detected_at_seconds: float = 5.0

    # Pending confirmation model.
    enable_pending_confirmation: bool = True
    confirmation_delay_seconds: float = 2.0
    pending_ttl_seconds: float = 8.0
    min_pending_age_seconds: float = 1.5
    pending_scan_interval_seconds: float = 0.5

    # Якщо після candidate прийшов новіший analytics result по тому ж full scope,
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

    # Scoring.
    score_confidence_weight: float = 0.23
    score_exhaustion_bias_weight: float = 0.28
    score_bias_delta_weight: float = 0.14
    score_intensity_weight: float = 0.12
    score_severity_weight: float = 0.08
    score_cluster_quality_weight: float = 0.08
    score_imbalance_weight: float = 0.04
    score_acceleration_weight: float = 0.03

    def validate(self) -> None:
        if not self.subscribe_topic:
            raise ValueError("subscribe_topic must not be empty")

        if not self.publish_topic_signal_generated:
            raise ValueError("publish_topic_signal_generated must not be empty")

        if not self.publish_topic_signal_rejected:
            raise ValueError("publish_topic_signal_rejected must not be empty")

        if not self.diagnostics_topic:
            raise ValueError("diagnostics_topic must not be empty")

        pending_topics = {
            "publish_topic_pending_created": self.publish_topic_pending_created,
            "publish_topic_pending_expired": self.publish_topic_pending_expired,
            "publish_topic_pending_replaced": self.publish_topic_pending_replaced,
            "publish_topic_pending_confirmed": self.publish_topic_pending_confirmed,
        }
        for name, topic in pending_topics.items():
            if not topic:
                raise ValueError(f"{name} must not be empty")

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

        if self.pending_ttl_seconds <= self.confirmation_delay_seconds:
            raise ValueError("pending_ttl_seconds must be greater than confirmation_delay_seconds")

        if self.min_pending_age_seconds > self.pending_ttl_seconds:
            raise ValueError("min_pending_age_seconds must be <= pending_ttl_seconds")

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

        if self.recent_pending_limit <= 0:
            raise ValueError("recent_pending_limit must be > 0")

        if self.hot_symbols_window_seconds is not None and self.hot_symbols_window_seconds <= 0:
            raise ValueError("hot_symbols_window_seconds must be > 0 or None")

        normalized_allowed_market_types = {
            normalize_market_type(item)
            for item in self.allowed_market_types
            if str(item).strip()
        }
        normalized_blocked_market_types = {
            normalize_market_type(item)
            for item in self.blocked_market_types
            if str(item).strip()
        }
        overlap_market_types = normalized_allowed_market_types & normalized_blocked_market_types
        if overlap_market_types:
            raise ValueError(f"allowed_market_types and blocked_market_types overlap: {sorted(overlap_market_types)}")

        normalized_allowed_timeframes = {
            normalize_timeframe(item)
            for item in self.allowed_timeframes
            if str(item).strip()
        }
        normalized_blocked_timeframes = {
            normalize_timeframe(item)
            for item in self.blocked_timeframes
            if str(item).strip()
        }
        overlap_timeframes = normalized_allowed_timeframes & normalized_blocked_timeframes
        if overlap_timeframes:
            raise ValueError(f"allowed_timeframes and blocked_timeframes overlap: {sorted(overlap_timeframes)}")

        normalized_allowed_symbols = {
            normalize_symbol(item)
            for item in self.allowed_symbols
            if str(item).strip()
        }
        normalized_blocked_symbols = {
            normalize_symbol(item)
            for item in self.blocked_symbols
            if str(item).strip()
        }
        overlap_symbols = normalized_allowed_symbols & normalized_blocked_symbols
        if overlap_symbols:
            raise ValueError(f"allowed_symbols and blocked_symbols overlap: {sorted(overlap_symbols)}")

        weights = {
            "score_confidence_weight": self.score_confidence_weight,
            "score_exhaustion_bias_weight": self.score_exhaustion_bias_weight,
            "score_bias_delta_weight": self.score_bias_delta_weight,
            "score_intensity_weight": self.score_intensity_weight,
            "score_severity_weight": self.score_severity_weight,
            "score_cluster_quality_weight": self.score_cluster_quality_weight,
            "score_imbalance_weight": self.score_imbalance_weight,
            "score_acceleration_weight": self.score_acceleration_weight,
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

    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    event_type: str | None = None
    status: str | None = None

    side_imbalance_ratio: float | None = None
    event_imbalance_ratio: float | None = None
    acceleration_ratio: float | None = None

    pending_started_at: datetime | None = None
    pending_confirmed_at: datetime | None = None

    correlation_id: str | None = None
    source_event_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.exchange = normalize_exchange(self.exchange)
        self.symbol = normalize_symbol(self.symbol)
        self.market_type = normalize_market_type(self.market_type)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )
        self.side = self.side.upper()

        self.confidence = clamp_float(self.confidence)
        self.score = clamp_float(self.score)
        self.intensity_score = clamp_float(self.intensity_score)
        self.continuation_bias = clamp_float(self.continuation_bias)
        self.exhaustion_bias = clamp_float(self.exhaustion_bias)
        self.bias_delta = clamp_float(self.bias_delta)

        self.generated_at = ensure_utc(self.generated_at)
        self.detected_at = ensure_utc(self.detected_at)

        if self.pending_started_at is not None:
            self.pending_started_at = ensure_utc(self.pending_started_at)

        if self.pending_confirmed_at is not None:
            self.pending_confirmed_at = ensure_utc(self.pending_confirmed_at)

    @property
    def key(self) -> LiquidationKey:
        return make_strategy_scope_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def scope(self) -> dict[str, str]:
        scope = liquidation_key_to_dict(self.key)
        scope["exchange_symbol"] = self.exchange_symbol or self.symbol
        return scope

    @property
    def scope_key(self) -> str:
        return scoped_key_to_string(self.key)

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
        return self.side == "LONG"

    @property
    def is_short(self) -> bool:
        return self.side == "SHORT"

    def to_dict(self, *, serialize: bool = True) -> dict[str, Any]:
        data = asdict(self)

        data["scope"] = self.scope
        data["scope_key"] = self.scope_key
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

    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    cancelled: bool = False
    cancel_reason: str | None = None

    candidate_detected_at: datetime = field(init=False)

    def __post_init__(self) -> None:
        self.exchange = normalize_exchange(self.exchange)
        self.symbol = normalize_symbol(self.symbol)
        self.market_type = normalize_market_type(self.market_type)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

        self.created_at = ensure_utc(self.created_at)
        self.confirm_after = ensure_utc(self.confirm_after)
        self.expires_at = ensure_utc(self.expires_at)
        self.candidate_detected_at = ensure_utc(self.result.detected_at)
        self.score_at_creation = clamp_float(self.score_at_creation)

    @property
    def key(self) -> LiquidationKey:
        return make_strategy_scope_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def scope(self) -> dict[str, str]:
        scope = liquidation_key_to_dict(self.key)
        scope["exchange_symbol"] = self.exchange_symbol or self.symbol
        return scope

    @property
    def scope_key(self) -> str:
        return scoped_key_to_string(self.key)

    def is_ready(self, now: datetime) -> bool:
        return ensure_utc(now) >= self.confirm_after

    def is_expired(self, now: datetime) -> bool:
        return ensure_utc(now) > self.expires_at

    def to_dict(self, *, serialize: bool = True) -> dict[str, Any]:
        data = {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "exchange_symbol": self.exchange_symbol,
            "scope": self.scope,
            "scope_key": self.scope_key,
            "source_topic": self.source_topic,
            "source_event_id": self.source_event_id,
            "correlation_id": self.correlation_id,
            "created_at": self.created_at,
            "confirm_after": self.confirm_after,
            "expires_at": self.expires_at,
            "candidate_detected_at": self.candidate_detected_at,
            "cluster_signature": self.cluster_signature,
            "score_at_creation": self.score_at_creation,
            "quality_snapshot": self.quality_snapshot,
            "cancelled": self.cancelled,
            "cancel_reason": self.cancel_reason,
        }

        return serialize_value(data) if serialize else data


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
            self.latest_seen_score = clamp_float(score)

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "latest_seen_detected_at": (
                    self.latest_seen_detected_at.isoformat()
                    if self.latest_seen_detected_at
                    else None
                ),
                "latest_seen_score": self.latest_seen_score,
                "has_pending": self.pending is not None,
                "pending": self.pending.to_dict() if self.pending is not None else None,
            }
        )
        return data


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
        analytics.liquidations.exhaustion_detected
            -> common full-scope filters
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
        self._pending_keys: set[LiquidationKey] = set()
        self._recent_pending: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Base hooks
    # ------------------------------------------------------------------

    def create_symbol_state(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        exchange_symbol: str | None = None,
    ) -> SymbolSqueezeStrategyState:
        return SymbolSqueezeStrategyState(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            exchange_symbol=exchange_symbol,
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
        state = self.get_or_create_state_for_result(result)
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
    # Scheduler
    # ------------------------------------------------------------------

    def _register_scheduler_jobs(self) -> None:
        super()._register_scheduler_jobs()

        if self.scheduler is None:
            return

        if not self.config.enable_pending_confirmation:
            return

        if self._pending_scan_job_id is not None:
            return

        self._pending_scan_job_id = self.scheduler.add_interval_job(
            name=f"{self.config.strategy_name}:pending_scan",
            func=self.process_pending_candidates,
            interval=self.config.pending_scan_interval_seconds,
            run_immediately=False,
            max_retries=0,
            retry_delay=1.0,
            timeout=max(1.0, self.config.pending_scan_interval_seconds),
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
        except Exception as exc:
            self._record_error(exc)
            self.logger.warning(
                "Failed to remove pending scan scheduler job",
                extra={
                    "strategy": self.config.strategy_name,
                    "job_id": self._pending_scan_job_id,
                    "error": repr(exc),
                },
            )
        finally:
            self._pending_scan_job_id = None

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
        if self.config.require_confirmed_result and not result.is_confirmed:
            self._stats.filter_skips += 1
            return "result_not_confirmed"

        if self.config.require_confirmed_result and result.status is not LiquidationStatus.CONFIRMED:
            self._stats.filter_skips += 1
            return "status_not_confirmed"

        if self.config.require_actionable_direction and not result.direction.is_known:
            self._stats.filter_skips += 1
            return "direction_not_actionable"

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
            market_type=result.market_type,
            symbol=result.symbol,
            timeframe=result.timeframe,
            exchange_symbol=result.exchange_symbol,
            result=result,
            source_topic=bus_event.topic,
            source_event_id=bus_event.event_id,
            correlation_id=bus_event.correlation_id or result.correlation_id,
            created_at=ensure_utc(now),
            confirm_after=ensure_utc(now) + timedelta(seconds=self.config.confirmation_delay_seconds),
            expires_at=ensure_utc(now) + timedelta(seconds=self.config.pending_ttl_seconds),
            cluster_signature=cluster_signature,
            score_at_creation=score,
            quality_snapshot=self.build_quality_snapshot(result),
        )

        state.pending = candidate
        self._pending_keys.add(candidate.key)

        self._stats.pending_created += 1
        self.remember_pending(candidate, state="created")

        self.logger.info(
            "Squeeze reversal candidate moved to pending",
            extra={
                "strategy": self.config.strategy_name,
                "exchange": result.exchange,
                "market_type": result.market_type,
                "symbol": result.symbol,
                "timeframe": result.timeframe,
                "exchange_symbol": result.exchange_symbol,
                "scope": result.scope,
                "score": score,
                "confidence": result.confidence,
                "severity": result.severity.value,
                "exhaustion_bias": result.exhaustion_bias,
                "bias_delta": result.bias_delta,
                "confirm_after": candidate.confirm_after.isoformat(),
                "expires_at": candidate.expires_at.isoformat(),
                "event_id": bus_event.event_id,
                "correlation_id": candidate.correlation_id,
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

            age_seconds = (ensure_utc(now) - candidate.created_at).total_seconds()
            if age_seconds < self.config.min_pending_age_seconds:
                continue

            if (
                self.config.cancel_if_newer_detected_at
                and state.latest_seen_detected_at is not None
                and ensure_utc(state.latest_seen_detected_at) > candidate.candidate_detected_at
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
                headers={
                    "source_event_id": candidate.source_event_id,
                    "exchange": candidate.exchange,
                    "market_type": candidate.market_type,
                    "symbol": candidate.symbol,
                    "timeframe": candidate.timeframe,
                    "exchange_symbol": candidate.exchange_symbol,
                    "scope": candidate.scope_key,
                },
            )

            emitted = await self.emit_confirmed_signal(
                result=candidate.result,
                bus_event=synthetic_event,
                state=state,
                cluster_signature=candidate.cluster_signature,
                pending_started_at=candidate.created_at,
                pending_confirmed_at=now,
                source_event_id=candidate.source_event_id,
                pending_confirmation=True,
            )

            if not emitted:
                continue

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
        self._pending_keys.discard(candidate.key)
        self.remember_pending(candidate, state=reason)

        self.logger.info(
            "Pending squeeze reversal candidate expired",
            extra={
                "strategy": self.config.strategy_name,
                "exchange": candidate.exchange,
                "market_type": candidate.market_type,
                "symbol": candidate.symbol,
                "timeframe": candidate.timeframe,
                "exchange_symbol": candidate.exchange_symbol,
                "scope": candidate.scope,
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
            "market_type": candidate.market_type,
            "symbol": candidate.symbol,
            "timeframe": candidate.timeframe,
            "exchange_symbol": candidate.exchange_symbol,
            "scope": candidate.scope,
            "scope_key": candidate.scope_key,
            "state": reason,
            "created_at": candidate.created_at,
            "confirm_after": candidate.confirm_after,
            "expires_at": candidate.expires_at,
            "candidate_detected_at": candidate.candidate_detected_at,
            "score_at_creation": candidate.score_at_creation,
            "source_event_id": candidate.source_event_id,
            "correlation_id": candidate.correlation_id,
            "cluster_signature": candidate.cluster_signature,
            "quality_snapshot": candidate.quality_snapshot,
            "cancelled": candidate.cancelled,
            "cancel_reason": candidate.cancel_reason,
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
                "market_type": candidate.market_type,
                "symbol": candidate.symbol,
                "timeframe": candidate.timeframe,
                "exchange_symbol": candidate.exchange_symbol,
                "scope": candidate.scope_key,
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
            "analytics_status": result.status.value,
            "analytics_scope": scoped_key_to_string(result.key),
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
                "market_type": signal.market_type,
                "symbol": signal.symbol,
                "timeframe": signal.timeframe,
                "exchange_symbol": signal.exchange_symbol,
                "scope": signal.scope,
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
            f"scope={scoped_key_to_string(result.key)}, "
            f"direction={result.direction.value}, "
            f"side={result.side.value}, "
            f"severity={result.severity.value}, "
            f"status={result.status.value}, "
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
            "strategy_model": "exhaustion_reversal",
            "analytics_event_type": result.event_type.value,
            "analytics_status": result.status.value,
            "favors_exhaustion": result.favors_exhaustion,
            "favors_continuation": result.favors_continuation,
            "bias_delta": result.bias_delta,
            "score": score,
            "quality_snapshot": self.build_quality_snapshot(result),
            "trade_side_mapping": {
                "cascade_down": "LONG",
                "cascade_up": "SHORT",
            },
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
            market_type=result.market_type,
            symbol=result.symbol,
            timeframe=result.timeframe,
            exchange_symbol=result.exchange_symbol,
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
            event_type=result.event_type.value,
            status=result.status.value,
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
            + self.config.score_imbalance_weight
            + self.config.score_acceleration_weight
        )

        if total_weight <= 0:
            return 0.0

        cluster_quality = self.compute_cluster_quality_score(result)
        analytics_meta = self.extract_analytics_metadata(result)

        imbalance_score = analytics_meta["side_imbalance_ratio"]
        if imbalance_score is None:
            imbalance_score = analytics_meta["event_imbalance_ratio"]
        if imbalance_score is None:
            imbalance_score = 0.5

        acceleration_score = self.acceleration_to_score(analytics_meta["acceleration_ratio"])

        weighted_score = (
            clamp_float(result.confidence) * self.config.score_confidence_weight
            + clamp_float(result.exhaustion_bias) * self.config.score_exhaustion_bias_weight
            + clamp_float(result.bias_delta) * self.config.score_bias_delta_weight
            + clamp_float(result.intensity_score) * self.config.score_intensity_weight
            + self.severity_to_score(result.severity) * self.config.score_severity_weight
            + cluster_quality * self.config.score_cluster_quality_weight
            + clamp_float(imbalance_score) * self.config.score_imbalance_weight
            + acceleration_score * self.config.score_acceleration_weight
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
            "acceleration_ratio": self.optional_float(
                metadata.get("acceleration_ratio")
                or metadata.get("climax_acceleration_ratio")
            ),
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

    @staticmethod
    def acceleration_to_score(value: float | None) -> float:
        if value is None:
            return 0.5

        if value <= 0:
            return 0.0

        # 1.0 = neutral, 2.0+ = strong climax acceleration.
        return clamp_float((value - 1.0) / 1.0)

    def build_quality_snapshot(
        self,
        result: CascadeDetectionResult,
    ) -> dict[str, Any]:
        analytics_meta = self.extract_analytics_metadata(result)

        return {
            "exchange": result.exchange,
            "market_type": result.market_type,
            "symbol": result.symbol,
            "timeframe": result.timeframe,
            "exchange_symbol": result.exchange_symbol,
            "scope": liquidation_key_to_dict(result.key),
            "scope_key": scoped_key_to_string(result.key),
            "confidence": result.confidence,
            "intensity_score": result.intensity_score,
            "continuation_bias": result.continuation_bias,
            "exhaustion_bias": result.exhaustion_bias,
            "bias_delta": result.bias_delta,
            "severity": result.severity.value,
            "status": result.status.value,
            "event_type": result.event_type.value,
            "event_count": result.event_count,
            "total_notional_usd": str(result.total_notional_usd),
            "window_seconds": result.window_seconds,
            "price_range_pct": result.price_range_pct,
            "cluster": {
                "exchange": result.cluster.exchange,
                "market_type": result.cluster.market_type,
                "symbol": result.cluster.symbol,
                "timeframe": result.cluster.timeframe,
                "exchange_symbol": result.cluster.exchange_symbol,
                "scope": liquidation_key_to_dict(result.cluster.key),
                "scope_key": scoped_key_to_string(result.cluster.key),
                "duration_seconds": result.cluster.duration_seconds,
                "avg_notional_per_event": str(result.cluster.avg_notional_per_event),
                "price_range_pct": result.cluster.price_range_pct,
                "event_count": result.cluster.event_count,
                "total_notional_usd": str(result.cluster.total_notional_usd),
            },
            "analytics_metadata": analytics_meta,
            "computed_score": self.compute_strategy_score(result),
            "cluster_quality_score": self.compute_cluster_quality_score(result),
        }

    # ------------------------------------------------------------------
    # Pending diagnostics / memory
    # ------------------------------------------------------------------

    def remember_pending(
        self,
        candidate: PendingReversalCandidate,
        *,
        state: str,
    ) -> None:
        item = {
            "state": state,
            "created_at": utc_now().isoformat(),
            "candidate": candidate.to_dict(),
        }

        self._recent_pending.append(item)

        limit = max(1, self.config.recent_pending_limit)
        if len(self._recent_pending) > limit:
            self._recent_pending = self._recent_pending[-limit:]

    def get_recent_pending(
        self,
        *,
        exchange: str | None = None,
        symbol: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []

        target_exchange = normalize_exchange(exchange) if exchange else None
        target_symbol = normalize_symbol(symbol) if symbol else None
        target_market_type = normalize_market_type(market_type) if market_type else None
        target_timeframe = normalize_timeframe(timeframe) if timeframe else None

        result: list[dict[str, Any]] = []

        for item in reversed(self._recent_pending):
            candidate = item.get("candidate", {})

            if target_exchange is not None and candidate.get("exchange") != target_exchange:
                continue

            if target_market_type is not None and candidate.get("market_type") != target_market_type:
                continue

            if target_symbol is not None and candidate.get("symbol") != target_symbol:
                continue

            if target_timeframe is not None and candidate.get("timeframe") != target_timeframe:
                continue

            result.append(item)

            if len(result) >= limit:
                break

        return result

    # ------------------------------------------------------------------
    # Public diagnostics overrides
    # ------------------------------------------------------------------

    def get_hot_symbols(
        self,
        *,
        exchange: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []

        now = utc_now()
        min_ts = None

        if self.config.hot_symbols_window_seconds is not None:
            min_ts = now - timedelta(seconds=self.config.hot_symbols_window_seconds)

        target_exchange = normalize_exchange(exchange) if exchange else None
        target_market_type = normalize_market_type(market_type) if market_type else None
        target_timeframe = normalize_timeframe(timeframe) if timeframe else None

        latest_by_key: dict[LiquidationKey, SqueezeReversalSignal] = {}

        for signal in self._recent_signals:
            if min_ts is not None and ensure_utc(signal.generated_at) < min_ts:
                continue

            if target_exchange is not None and signal.exchange != target_exchange:
                continue

            if target_market_type is not None and signal.market_type != target_market_type:
                continue

            if target_timeframe is not None and signal.timeframe != target_timeframe:
                continue

            previous = latest_by_key.get(signal.key)

            if previous is None or ensure_utc(signal.generated_at) > ensure_utc(previous.generated_at):
                latest_by_key[signal.key] = signal

        rows = [
            {
                "exchange": signal.exchange,
                "market_type": signal.market_type,
                "symbol": signal.symbol,
                "timeframe": signal.timeframe,
                "exchange_symbol": signal.exchange_symbol,
                "scope": liquidation_key_to_dict(signal.key),
                "scope_key": signal.scope_key,
                "side": signal.side,
                "score": signal.score,
                "confidence": signal.confidence,
                "severity": signal.severity,
                "cascade_direction": signal.cascade_direction,
                "liquidation_side": signal.liquidation_side,
                "intensity_score": signal.intensity_score,
                "continuation_bias": signal.continuation_bias,
                "exhaustion_bias": signal.exhaustion_bias,
                "bias_delta": signal.bias_delta,
                "side_imbalance_ratio": signal.side_imbalance_ratio,
                "event_imbalance_ratio": signal.event_imbalance_ratio,
                "acceleration_ratio": signal.acceleration_ratio,
                "pending_confirmed": signal.is_pending_confirmed,
                "confirmation_delay_seconds": signal.confirmation_delay_seconds,
                "generated_at": signal.generated_at.isoformat(),
                "detected_at": signal.detected_at.isoformat(),
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

        return rows[:limit]

    def get_symbol_state(
        self,
        exchange: str,
        symbol: str,
        *,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> dict[str, Any]:
        key = self.state_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        state = self._states.get(key)

        normalized_exchange = normalize_exchange(exchange)
        normalized_symbol = normalize_symbol(symbol)
        normalized_market_type = normalize_market_type(market_type)
        normalized_timeframe = normalize_timeframe(timeframe)

        if state is None:
            return {
                "exchange": normalized_exchange,
                "market_type": normalized_market_type,
                "symbol": normalized_symbol,
                "timeframe": normalized_timeframe,
                "scope": liquidation_key_to_dict(key),
                "scope_key": scoped_key_to_string(key),
                "exists": False,
            }

        now = utc_now()
        state.prune_old_signal_timestamps(now, self.config.signal_window_seconds)

        return {
            "exchange": state.exchange,
            "market_type": state.market_type,
            "symbol": state.symbol,
            "timeframe": state.timeframe,
            "exchange_symbol": state.exchange_symbol,
            "scope": state.scope,
            "scope_key": state.scope_key,
            "exists": True,
            "last_signal_at": state.last_signal_at.isoformat() if state.last_signal_at else None,
            "cooldown_until": state.cooldown_until.isoformat() if state.cooldown_until else None,
            "in_cooldown": state.is_in_cooldown(now),
            "last_signal_side": state.last_signal_side,
            "last_detected_at": state.last_detected_at.isoformat() if state.last_detected_at else None,
            "last_cluster_signature": state.last_cluster_signature,
            "last_signal_score": state.last_signal_score,
            "latest_seen_detected_at": (
                state.latest_seen_detected_at.isoformat()
                if state.latest_seen_detected_at
                else None
            ),
            "latest_seen_score": state.latest_seen_score,
            "total_signals_emitted": state.total_signals_emitted,
            "signals_in_window": state.signals_in_window(
                now=now,
                window_seconds=self.config.signal_window_seconds,
            ),
            "pending": state.pending.to_dict() if state.pending is not None else None,
        }

    def get_stats(self) -> dict[str, Any]:
        data = super().get_stats()
        data.update(
            {
                "pending_scan_job_registered": self._pending_scan_job_id is not None,
                "pending_scan_job_id": self._pending_scan_job_id,
                "pending_active": len(self._pending_keys),
                "recent_pending": len(self._recent_pending),
            }
        )
        return data