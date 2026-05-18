from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
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


# ============================================================================
# Config
# ============================================================================


@dataclass(slots=True)
class LiquidationCascadeStrategyConfig:
    """
    Continuation strategy поверх analytics.liquidations.cascade_detected.

    Strategy:
    - слухає analytics.liquidations.cascade_detected;
    - приймає CascadeDetectionResult;
    - працює тільки з повним futures/liquidation scope:
        exchange + market_type + symbol + timeframe;
    - фільтрує слабкі / шумні cascade results;
    - генерує continuation signal у напрямку каскаду;
    - не викликає risk/execution напряму.

    Continuation direction:
    - CascadeDirection.DOWN -> SHORT
    - CascadeDirection.UP   -> LONG
    """

    enabled: bool = True

    # Важливо: plural namespace, як у новому CascadeDetector.
    subscribe_topic: str = "analytics.liquidations.cascade_detected"
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

    # Full-scope filters.
    allowed_exchanges: tuple[str, ...] = ()
    allowed_market_types: tuple[str, ...] = ()
    allowed_symbols: tuple[str, ...] = ()
    allowed_timeframes: tuple[str, ...] = ()

    blocked_market_types: tuple[str, ...] = ()
    blocked_symbols: tuple[str, ...] = ()
    blocked_timeframes: tuple[str, ...] = ()

    allowed_severities: tuple[CascadeSeverity, ...] = (
        CascadeSeverity.MEDIUM,
        CascadeSeverity.HIGH,
        CascadeSeverity.EXTREME,
    )

    # Common quality filters.
    require_confirmed_result: bool = True
    require_actionable_direction: bool = True

    min_confidence: float = 0.60
    min_intensity_score: float = 0.55
    min_total_notional_usd: Decimal = Decimal("300000")
    min_event_count: int = 5
    max_price_range_pct: float | None = None

    max_future_detected_at_seconds: float = 5.0
    max_result_age_seconds: float | None = 30.0

    require_favors_continuation: bool = True
    require_high_confidence_only: bool = False

    min_continuation_bias: float = 0.60
    max_exhaustion_bias_for_continuation: float | None = None
    min_bias_delta: float | None = None

    # Analytics metadata / cluster-quality filters.
    min_side_imbalance_ratio: float | None = None
    min_event_imbalance_ratio: float | None = None
    min_acceleration_ratio: float | None = None

    max_cluster_duration_seconds: float | None = None
    min_avg_notional_per_event: Decimal | None = None

    symbol_cooldown_seconds: int = 20
    min_seconds_between_same_side_signals: int = 10

    max_signals_per_symbol_window: int = 2
    signal_window_seconds: int = 60

    deduplicate_by_detected_at: bool = True
    deduplicate_same_cluster_signature: bool = True

    recent_signals_limit: int = 200
    recent_rejections_limit: int = 200

    hot_symbols_window_seconds: int | None = 300

    # Scoring.
    score_confidence_weight: float = 0.30
    score_continuation_bias_weight: float = 0.30
    score_intensity_weight: float = 0.18
    score_severity_weight: float = 0.10
    score_imbalance_weight: float = 0.07
    score_acceleration_weight: float = 0.05

    def validate(self) -> None:
        if not self.subscribe_topic:
            raise ValueError("subscribe_topic must not be empty")

        if not self.publish_topic_signal_generated:
            raise ValueError("publish_topic_signal_generated must not be empty")

        if not self.publish_topic_signal_rejected:
            raise ValueError("publish_topic_signal_rejected must not be empty")

        if not self.diagnostics_topic:
            raise ValueError("diagnostics_topic must not be empty")

        bounded = {
            "min_confidence": self.min_confidence,
            "min_intensity_score": self.min_intensity_score,
            "min_continuation_bias": self.min_continuation_bias,
        }

        for name, value in bounded.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be between 0 and 1")

        if self.max_exhaustion_bias_for_continuation is not None:
            if not (0.0 <= self.max_exhaustion_bias_for_continuation <= 1.0):
                raise ValueError("max_exhaustion_bias_for_continuation must be between 0 and 1 or None")

        if self.min_bias_delta is not None and not (0.0 <= self.min_bias_delta <= 1.0):
            raise ValueError("min_bias_delta must be between 0 and 1 or None")

        if self.min_total_notional_usd < 0:
            raise ValueError("min_total_notional_usd must be >= 0")

        if self.min_event_count < 0:
            raise ValueError("min_event_count must be >= 0")

        if self.max_price_range_pct is not None and self.max_price_range_pct < 0:
            raise ValueError("max_price_range_pct must be >= 0 or None")

        if self.max_future_detected_at_seconds < 0:
            raise ValueError("max_future_detected_at_seconds must be >= 0")

        if self.max_result_age_seconds is not None and self.max_result_age_seconds <= 0:
            raise ValueError("max_result_age_seconds must be > 0 or None")

        if self.min_side_imbalance_ratio is not None and not (0.0 <= self.min_side_imbalance_ratio <= 1.0):
            raise ValueError("min_side_imbalance_ratio must be between 0 and 1 or None")

        if self.min_event_imbalance_ratio is not None and not (0.0 <= self.min_event_imbalance_ratio <= 1.0):
            raise ValueError("min_event_imbalance_ratio must be between 0 and 1 or None")

        if self.min_acceleration_ratio is not None and self.min_acceleration_ratio < 0:
            raise ValueError("min_acceleration_ratio must be >= 0 or None")

        if self.max_cluster_duration_seconds is not None and self.max_cluster_duration_seconds <= 0:
            raise ValueError("max_cluster_duration_seconds must be > 0 or None")

        if self.min_avg_notional_per_event is not None and self.min_avg_notional_per_event < 0:
            raise ValueError("min_avg_notional_per_event must be >= 0 or None")

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

        score_weights = {
            "score_confidence_weight": self.score_confidence_weight,
            "score_continuation_bias_weight": self.score_continuation_bias_weight,
            "score_intensity_weight": self.score_intensity_weight,
            "score_severity_weight": self.score_severity_weight,
            "score_imbalance_weight": self.score_imbalance_weight,
            "score_acceleration_weight": self.score_acceleration_weight,
        }

        for name, weight in score_weights.items():
            if weight < 0:
                raise ValueError(f"{name} must be >= 0")

        if sum(score_weights.values()) <= 0:
            raise ValueError("strategy score weights sum must be > 0")


# ============================================================================
# Models
# ============================================================================


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

    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None

    event_type: str | None = None
    status: str | None = None
    window_seconds: int | None = None

    bias_delta: float | None = None
    side_imbalance_ratio: float | None = None
    event_imbalance_ratio: float | None = None
    acceleration_ratio: float | None = None

    cluster_duration_seconds: float | None = None
    cluster_avg_notional_per_event: Decimal | None = None

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
        self.generated_at = ensure_utc(self.generated_at)
        self.detected_at = ensure_utc(self.detected_at)

        if self.bias_delta is not None:
            self.bias_delta = clamp_float(self.bias_delta)

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
    def is_long(self) -> bool:
        return self.side == "LONG"

    @property
    def is_short(self) -> bool:
        return self.side == "SHORT"

    def to_dict(self, *, serialize: bool = True) -> dict[str, Any]:
        data = {
            "strategy_name": self.strategy_name,
            "signal_type": self.signal_type,
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "exchange_symbol": self.exchange_symbol,
            "scope": self.scope,
            "scope_key": self.scope_key,
            "side": self.side,
            "is_long": self.is_long,
            "is_short": self.is_short,
            "confidence": self.confidence,
            "score": self.score,
            "generated_at": self.generated_at,
            "detected_at": self.detected_at,
            "reason": self.reason,
            "source_topic": self.source_topic,
            "severity": self.severity,
            "cascade_direction": self.cascade_direction,
            "liquidation_side": self.liquidation_side,
            "event_type": self.event_type,
            "status": self.status,
            "event_count": self.event_count,
            "total_notional_usd": self.total_notional_usd,
            "window_seconds": self.window_seconds,
            "intensity_score": self.intensity_score,
            "continuation_bias": self.continuation_bias,
            "exhaustion_bias": self.exhaustion_bias,
            "bias_delta": self.bias_delta,
            "price_range_pct": self.price_range_pct,
            "side_imbalance_ratio": self.side_imbalance_ratio,
            "event_imbalance_ratio": self.event_imbalance_ratio,
            "acceleration_ratio": self.acceleration_ratio,
            "cluster_duration_seconds": self.cluster_duration_seconds,
            "cluster_avg_notional_per_event": self.cluster_avg_notional_per_event,
            "correlation_id": self.correlation_id,
            "source_event_id": self.source_event_id,
            "metadata": self.metadata,
        }

        return serialize_value(data) if serialize else data


@dataclass(slots=True)
class SymbolCascadeStrategyState(BaseSymbolStrategyState):
    """
    Full-scope state для liquidation continuation strategy.

    Scope:
        exchange + market_type + symbol + timeframe
    """

    pass


# Backward-compatible alias для імпортів із __init__.py
LiquidationCascadeStrategyStats = BaseStrategyStats


# ============================================================================
# Main strategy
# ============================================================================


class LiquidationCascadeStrategy(
    BaseAnalyticsStrategy[
        CascadeDetectionResult,
        LiquidationCascadeSignal,
        SymbolCascadeStrategyState,
        LiquidationCascadeStrategyConfig,
    ]
):
    """
    Continuation strategy поверх analytics.liquidations.cascade_detected.

    Pipeline:
        analytics.liquidations.cascade_detected
            -> common full-scope filters from BaseAnalyticsStrategy
            -> liquidation continuation-specific filters
            -> LiquidationCascadeSignal
            -> signal.generated

    Цей клас НЕ:
    - не читає raw market data;
    - не викликає CascadeDetector напряму;
    - не викликає risk/execution напряму;
    - не дублює EventBus/Scheduler/logger lifecycle.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        config: LiquidationCascadeStrategyConfig | None = None,
        scheduler: Scheduler | None = None,
        service_name: str | None = None,
    ) -> None:
        super().__init__(
            event_bus=event_bus,
            scheduler=scheduler,
            config=config or LiquidationCascadeStrategyConfig(),
            service_name=service_name,
            component="strategy.liquidations.cascade_strategy",
            payload_type=CascadeDetectionResult,
        )

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
    ) -> SymbolCascadeStrategyState:
        return SymbolCascadeStrategyState(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            exchange_symbol=exchange_symbol,
        )

    async def process_result(
        self,
        result: CascadeDetectionResult,
        *,
        bus_event: Event,
    ) -> None:
        state = self.get_or_create_state_for_result(result)
        now = utc_now()

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

        signal = self.build_signal(result=result, bus_event=bus_event)

        emitted = await self.emit_signal(
            signal,
            bus_event=bus_event,
            headers={
                "analytics_event_type": result.event_type.value,
                "analytics_status": result.status.value,
                "analytics_scope": scoped_key_to_string(result.key),
            },
        )
        if not emitted:
            return

        self.remember_emitted_signal(
            signal=signal,
            state=state,
            result=result,
            signal_side=signal.side,
            score=signal.score,
            cluster_signature=filter_result.cluster_signature,
        )

        self.logger.info(
            "Liquidation cascade continuation signal emitted",
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
                "intensity_score": signal.intensity_score,
                "continuation_bias": signal.continuation_bias,
                "event_count": signal.event_count,
                "total_notional_usd": str(signal.total_notional_usd),
                "event_id": bus_event.event_id,
                "correlation_id": signal.correlation_id,
            },
        )

    def direction_to_trade_side(self, result: CascadeDetectionResult) -> str:
        """
        Continuation логіка:
        - CascadeDirection.DOWN -> SHORT
        - CascadeDirection.UP   -> LONG
        """
        if result.direction is CascadeDirection.DOWN:
            return "SHORT"

        if result.direction is CascadeDirection.UP:
            return "LONG"

        return "FLAT"

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def evaluate_filters(
        self,
        *,
        result: CascadeDetectionResult,
        state: SymbolCascadeStrategyState,
        now: datetime,
    ) -> FilterResult:
        """
        Об'єднує:
        - common full-scope filters з BaseAnalyticsStrategy;
        - liquidation continuation-specific filters.
        """
        common_result = self.evaluate_common_filters(
            result=result,
            state=state,
            now=now,
        )

        if common_result.rejection_reason is not None:
            return common_result

        custom_rejection = self.get_liquidation_rejection_reason(
            result=result,
            now=now,
        )

        if custom_rejection is not None:
            return FilterResult(
                rejection_reason=custom_rejection,
                cluster_signature=None,
            )

        return common_result

    def get_liquidation_rejection_reason(
        self,
        *,
        result: CascadeDetectionResult,
        now: datetime,
    ) -> str | None:
        if self.config.require_confirmed_result and not result.is_confirmed:
            self._stats.filter_skips += 1
            return "result_not_confirmed"

        if self.config.require_actionable_direction and not result.direction.is_known:
            self._stats.filter_skips += 1
            return "direction_not_actionable"

        if result.status is not LiquidationStatus.CONFIRMED and self.config.require_confirmed_result:
            self._stats.filter_skips += 1
            return "status_not_confirmed"

        if self.config.require_favors_continuation and not result.favors_continuation:
            self._stats.filter_skips += 1
            return "continuation_not_favored"

        if result.continuation_bias < self.config.min_continuation_bias:
            self._stats.filter_skips += 1
            return "continuation_bias_below_threshold"

        if (
            self.config.max_exhaustion_bias_for_continuation is not None
            and result.exhaustion_bias > self.config.max_exhaustion_bias_for_continuation
        ):
            self._stats.filter_skips += 1
            return "exhaustion_bias_too_high_for_continuation"

        if self.config.min_bias_delta is not None and result.bias_delta < self.config.min_bias_delta:
            self._stats.filter_skips += 1
            return "bias_delta_below_threshold"

        max_future = timedelta(seconds=self.config.max_future_detected_at_seconds)
        if ensure_utc(result.detected_at) > ensure_utc(now) + max_future:
            self._stats.filter_skips += 1
            return "detected_at_in_future"

        if self.config.max_result_age_seconds is not None:
            age_seconds = (ensure_utc(now) - ensure_utc(result.detected_at)).total_seconds()
            if age_seconds > self.config.max_result_age_seconds:
                self._stats.filter_skips += 1
                return "result_too_old"

        analytics_meta = self.extract_analytics_metadata(result)

        if self.config.min_side_imbalance_ratio is not None:
            value = self.optional_float(analytics_meta.get("side_imbalance_ratio"))
            if value is None or value < self.config.min_side_imbalance_ratio:
                self._stats.filter_skips += 1
                return "side_imbalance_below_threshold"

        if self.config.min_event_imbalance_ratio is not None:
            value = self.optional_float(analytics_meta.get("event_imbalance_ratio"))
            if value is None or value < self.config.min_event_imbalance_ratio:
                self._stats.filter_skips += 1
                return "event_imbalance_below_threshold"

        if self.config.min_acceleration_ratio is not None:
            value = self.optional_float(analytics_meta.get("acceleration_ratio"))
            if value is None or value < self.config.min_acceleration_ratio:
                self._stats.filter_skips += 1
                return "acceleration_below_threshold"

        cluster = result.cluster

        if cluster is not None:
            if self.config.max_cluster_duration_seconds is not None:
                if cluster.duration_seconds > self.config.max_cluster_duration_seconds:
                    self._stats.filter_skips += 1
                    return "cluster_duration_too_long"

            if self.config.min_avg_notional_per_event is not None:
                if cluster.avg_notional_per_event < self.config.min_avg_notional_per_event:
                    self._stats.filter_skips += 1
                    return "avg_notional_per_event_below_threshold"

        return None

    # ------------------------------------------------------------------
    # Signal building
    # ------------------------------------------------------------------

    def build_signal(
        self,
        *,
        result: CascadeDetectionResult,
        bus_event: Event,
    ) -> LiquidationCascadeSignal:
        trade_side = self.direction_to_trade_side(result)
        generated_at = utc_now()
        score = self.compute_strategy_score(result)
        analytics_meta = self.extract_analytics_metadata(result)

        reason = (
            "liquidation cascade continuation: "
            f"scope={scoped_key_to_string(result.key)}, "
            f"direction={result.direction.value}, "
            f"side={result.side.value}, "
            f"severity={result.severity.value}, "
            f"status={result.status.value}, "
            f"continuation_bias={result.continuation_bias:.3f}, "
            f"exhaustion_bias={result.exhaustion_bias:.3f}, "
            f"confidence={result.confidence:.3f}, "
            f"intensity={result.intensity_score:.3f}"
        )

        metadata = self.build_common_signal_metadata(
            result=result,
            bus_event=bus_event,
        )

        metadata["liquidation_cascade_strategy"] = {
            "strategy_model": "continuation",
            "trade_side_mapping": {
                "cascade_down": "SHORT",
                "cascade_up": "LONG",
            },
            "require_confirmed_result": self.config.require_confirmed_result,
            "require_favors_continuation": self.config.require_favors_continuation,
            "min_continuation_bias": self.config.min_continuation_bias,
            "max_exhaustion_bias_for_continuation": self.config.max_exhaustion_bias_for_continuation,
            "min_bias_delta": self.config.min_bias_delta,
            "max_future_detected_at_seconds": self.config.max_future_detected_at_seconds,
            "max_result_age_seconds": self.config.max_result_age_seconds,
            "min_side_imbalance_ratio": self.config.min_side_imbalance_ratio,
            "min_event_imbalance_ratio": self.config.min_event_imbalance_ratio,
            "min_acceleration_ratio": self.config.min_acceleration_ratio,
            "max_cluster_duration_seconds": self.config.max_cluster_duration_seconds,
            "min_avg_notional_per_event": str(self.config.min_avg_notional_per_event)
            if self.config.min_avg_notional_per_event is not None
            else None,
            "analytics_metadata": serialize_value(analytics_meta),
        }

        cluster = result.cluster

        return LiquidationCascadeSignal(
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
            window_seconds=result.window_seconds,
            intensity_score=clamp_float(result.intensity_score),
            continuation_bias=clamp_float(result.continuation_bias),
            exhaustion_bias=clamp_float(result.exhaustion_bias),
            bias_delta=clamp_float(result.bias_delta),
            price_range_pct=result.price_range_pct,
            side_imbalance_ratio=self.optional_float(analytics_meta.get("side_imbalance_ratio")),
            event_imbalance_ratio=self.optional_float(analytics_meta.get("event_imbalance_ratio")),
            acceleration_ratio=self.optional_float(analytics_meta.get("acceleration_ratio")),
            cluster_duration_seconds=cluster.duration_seconds if cluster is not None else None,
            cluster_avg_notional_per_event=cluster.avg_notional_per_event if cluster is not None else None,
            correlation_id=bus_event.correlation_id or result.correlation_id,
            source_event_id=bus_event.event_id,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def compute_strategy_score(self, result: CascadeDetectionResult) -> float:
        total_weight = (
            self.config.score_confidence_weight
            + self.config.score_continuation_bias_weight
            + self.config.score_intensity_weight
            + self.config.score_severity_weight
            + self.config.score_imbalance_weight
            + self.config.score_acceleration_weight
        )

        if total_weight <= 0:
            return 0.0

        analytics_meta = self.extract_analytics_metadata(result)

        imbalance_score = self.optional_float(analytics_meta.get("side_imbalance_ratio"))
        if imbalance_score is None:
            imbalance_score = self.optional_float(analytics_meta.get("event_imbalance_ratio"))
        if imbalance_score is None:
            imbalance_score = 0.5

        acceleration_score = self.acceleration_to_score(
            self.optional_float(analytics_meta.get("acceleration_ratio"))
        )

        weighted_score = (
            clamp_float(result.confidence) * self.config.score_confidence_weight
            + clamp_float(result.continuation_bias) * self.config.score_continuation_bias_weight
            + clamp_float(result.intensity_score) * self.config.score_intensity_weight
            + self.severity_to_score(result.severity) * self.config.score_severity_weight
            + clamp_float(imbalance_score) * self.config.score_imbalance_weight
            + acceleration_score * self.config.score_acceleration_weight
        ) / total_weight

        return clamp_float(weighted_score)

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

        # 1.0 = neutral, 2.0+ = strong acceleration.
        return clamp_float((value - 1.0) / 1.0)

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
        """
        Override base implementation, бо continuation strategy додає
        continuation_bias / exhaustion_bias / cascade_direction.
        """
        if limit <= 0:
            return []

        now = utc_now()
        min_ts = None

        if self.config.hot_symbols_window_seconds is not None:
            min_ts = now - timedelta(seconds=self.config.hot_symbols_window_seconds)

        target_exchange = normalize_exchange(exchange) if exchange else None
        target_market_type = normalize_market_type(market_type) if market_type else None
        target_timeframe = normalize_timeframe(timeframe) if timeframe else None

        latest_by_key: dict[LiquidationKey, LiquidationCascadeSignal] = {}

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
                float(row["intensity_score"]),
                float(row["continuation_bias"]),
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
            "total_signals_emitted": state.total_signals_emitted,
            "signals_in_window": state.signals_in_window(
                now=now,
                window_seconds=self.config.signal_window_seconds,
            ),
        }