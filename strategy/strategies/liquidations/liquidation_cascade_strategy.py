from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
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


# ============================================================================
# Config
# ============================================================================


@dataclass(slots=True)
class LiquidationCascadeStrategyConfig:
    """
    Continuation strategy поверх analytics.liquidation.cascade_detected.

    Strategy:
    - слухає analytics.liquidation.cascade_detected;
    - приймає CascadeDetectionResult;
    - фільтрує слабкі / шумні cascade results;
    - генерує continuation signal у напрямку каскаду;
    - не викликає risk/execution напряму.
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

    max_future_detected_at_seconds: float = 5.0

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

    hot_symbols_window_seconds: int | None = 300

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

        if self.max_future_detected_at_seconds < 0:
            raise ValueError("max_future_detected_at_seconds must be >= 0")

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

        score_weights = {
            "score_confidence_weight": self.score_confidence_weight,
            "score_continuation_bias_weight": self.score_continuation_bias_weight,
            "score_intensity_weight": self.score_intensity_weight,
            "score_severity_weight": self.score_severity_weight,
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

    correlation_id: str | None = None
    source_event_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, serialize: bool = True) -> dict[str, Any]:
        data = {
            "strategy_name": self.strategy_name,
            "signal_type": self.signal_type,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "side": self.side,
            "confidence": self.confidence,
            "score": self.score,
            "generated_at": self.generated_at,
            "detected_at": self.detected_at,
            "reason": self.reason,
            "source_topic": self.source_topic,
            "severity": self.severity,
            "cascade_direction": self.cascade_direction,
            "liquidation_side": self.liquidation_side,
            "event_count": self.event_count,
            "total_notional_usd": self.total_notional_usd,
            "intensity_score": self.intensity_score,
            "continuation_bias": self.continuation_bias,
            "exhaustion_bias": self.exhaustion_bias,
            "price_range_pct": self.price_range_pct,
            "correlation_id": self.correlation_id,
            "source_event_id": self.source_event_id,
            "metadata": self.metadata,
        }

        return serialize_value(data) if serialize else data


@dataclass(slots=True)
class SymbolCascadeStrategyState(BaseSymbolStrategyState):
    """
    State одного (exchange, symbol) для liquidation continuation strategy.

    Усе базове:
    - cooldown;
    - last_signal_at;
    - last_detected_at;
    - last_cluster_signature;
    - rate-limit timestamps;

    уже реалізовано в BaseSymbolStrategyState.
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
    Continuation strategy поверх analytics.liquidation.cascade_detected.

    Pipeline:
        analytics.liquidation.cascade_detected
            -> common filters from BaseAnalyticsStrategy
            -> liquidation continuation filters
            -> LiquidationCascadeSignal
            -> signal.generated

    Цей клас НЕ:
    - не читає market data;
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
    ) -> SymbolCascadeStrategyState:
        return SymbolCascadeStrategyState(
            exchange=exchange.lower(),
            symbol=normalize_symbol(symbol),
        )

    async def process_result(
        self,
        result: CascadeDetectionResult,
        *,
        bus_event: Event,
    ) -> None:
        state = self.get_or_create_state(result.exchange, result.symbol)
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

        emitted = await self.emit_signal(signal, bus_event=bus_event)
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
        - common filters з BaseAnalyticsStrategy;
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
        if self.config.require_favors_continuation and not result.favors_continuation:
            self._stats.filter_skips += 1
            return "continuation_not_favored"

        if result.continuation_bias < self.config.min_continuation_bias:
            self._stats.filter_skips += 1
            return "continuation_bias_below_threshold"

        max_future = timedelta(seconds=self.config.max_future_detected_at_seconds)
        if ensure_utc(result.detected_at) > ensure_utc(now) + max_future:
            self._stats.filter_skips += 1
            return "detected_at_in_future"

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

        reason = (
            "liquidation cascade continuation: "
            f"direction={result.direction.value}, "
            f"severity={result.severity.value}, "
            f"continuation_bias={result.continuation_bias:.3f}, "
            f"confidence={result.confidence:.3f}"
        )

        metadata = self.build_common_signal_metadata(
            result=result,
            bus_event=bus_event,
        )

        metadata["liquidation_strategy"] = {
            "min_continuation_bias": self.config.min_continuation_bias,
            "require_favors_continuation": self.config.require_favors_continuation,
            "max_future_detected_at_seconds": self.config.max_future_detected_at_seconds,
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

    def compute_strategy_score(self, result: CascadeDetectionResult) -> float:
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
            + self.severity_to_score(result.severity) * self.config.score_severity_weight
        ) / total_weight

        return clamp_float(weighted_score)

    # ------------------------------------------------------------------
    # Public diagnostics overrides
    # ------------------------------------------------------------------

    def get_hot_symbols(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """
        Override base implementation, бо continuation strategy має
        hot_symbols_window_seconds і continuation_bias у рядку.
        """
        now = utc_now()
        min_ts = None

        if self.config.hot_symbols_window_seconds is not None:
            min_ts = now - timedelta(seconds=self.config.hot_symbols_window_seconds)

        latest_by_key: dict[tuple[str, str], LiquidationCascadeSignal] = {}

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

        return rows[: max(0, limit)]

    def get_symbol_state(
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

    def signal_to_hot_symbol_row(
        self,
        signal: LiquidationCascadeSignal,
    ) -> dict[str, Any]:
        return {
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

    def _start_log_extra(self) -> dict[str, Any]:
        data = super()._start_log_extra()
        data.update(
            {
                "min_continuation_bias": self.config.min_continuation_bias,
                "require_favors_continuation": self.config.require_favors_continuation,
                "allowed_severities": [
                    severity.value for severity in self.config.allowed_severities
                ],
                "hot_symbols_window_seconds": self.config.hot_symbols_window_seconds,
            }
        )
        return data