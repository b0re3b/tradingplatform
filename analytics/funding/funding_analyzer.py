from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from uuid import uuid4
from typing import Any, Deque

from core.event_bus import Event, EventBus, EventPriority, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler

from .enums import FundingEventType, FundingTimeframe
from .funding_divergence import FundingDivergenceConfig, FundingDivergenceDetector
from .funding_extremes import FundingExtremesConfig, FundingExtremesDetector
from .funding_flip_detector import FundingFlipDetector, FundingFlipDetectorConfig
from .funding_pressure import FundingPressureAnalyzer, FundingPressureConfig
from .funding_regime_detector import FundingRegimeDetector, FundingRegimeDetectorConfig
from .models import (
    FundingAnalyticsEvent,
    FundingDivergenceEvent,
    FundingExtremeEvent,
    FundingFlipEvent,
    FundingPressureState,
    FundingRegimeState,
    FundingSignal,
    FundingSignalType,
    FundingSnapshot,
    FundingStatistics,
)


@dataclass(slots=True)
class FundingAnalyzerConfig:
    """
    Runtime/orchestration config for analytics.funding.

    This config intentionally owns EventBus topic names and orchestration flags.
    Pure detector thresholds remain in detector-specific config classes.
    """

    history_size: int = 500
    default_timeframe: FundingTimeframe = FundingTimeframe.H1

    publish_updated_event: bool = True
    publish_regime_event_on_every_update: bool = False
    publish_pressure_event_on_every_update: bool = False
    publish_signal_event: bool = True

    state_lock_timeout_sec: float = 3.0

    enable_cleanup_job: bool = True
    cleanup_interval_sec: float = 60.0
    cleanup_timeout_sec: float = 5.0
    stale_context_ttl_sec: float = 60 * 60.0
    stale_liquidation_ttl_sec: float = 5 * 60.0
    cleanup_job_name: str = "analytics.funding.cleanup"

    # FundingAnalyzer should consume normalized cache-level events, not raw exchange events.
    # Exchange adapters publish market.*; data caches normalize/store and publish market.*.updated.
    funding_event_name: str = "market.funding.updated"
    open_interest_event_name: str = "market.open_interest.updated"
    candle_event_name: str = "market.candle.closed"
    trade_event_name: str = "market.trades.updated"
    cvd_event_name: str = "analytics.orderflow.updated"
    liquidation_event_name: str = "market.liquidation"

    analytics_updated_event_name: str = "analytics.funding.updated"
    analytics_regime_event_name: str = "analytics.funding.regime"
    analytics_pressure_event_name: str = "analytics.funding.pressure"
    analytics_flip_event_name: str = "analytics.funding.flip"
    analytics_extreme_event_name: str = "analytics.funding.extreme"
    analytics_divergence_event_name: str = "analytics.funding.divergence"
    analytics_signal_event_name: str = "analytics.funding.signal"

    signal_on_regime_change: bool = True
    signal_on_high_pressure: bool = True
    signal_on_extreme: bool = True
    signal_on_divergence: bool = True
    signal_on_flip: bool = True

    # Historical storage. The analyzer keeps a small in-memory rolling window for
    # real-time statistics and stores full analytics history as parquet records.
    enable_parquet_history: bool = True
    parquet_base_path: str = "data/parquet"
    parquet_dataset_name: str = "analytics_funding"
    parquet_flush_interval_sec: float = 30.0
    parquet_flush_timeout_sec: float = 10.0
    parquet_flush_batch_size: int = 250
    parquet_flush_job_name: str = "analytics.funding.parquet_flush"
    load_history_from_parquet_on_start: bool = True
    parquet_max_load_records_per_key: int = 500

    def __post_init__(self) -> None:
        if self.history_size <= 0:
            raise ValueError("history_size must be > 0")
        if self.state_lock_timeout_sec <= 0:
            raise ValueError("state_lock_timeout_sec must be > 0")
        if self.cleanup_interval_sec <= 0:
            raise ValueError("cleanup_interval_sec must be > 0")
        if self.cleanup_timeout_sec <= 0:
            raise ValueError("cleanup_timeout_sec must be > 0")
        if self.stale_context_ttl_sec <= 0:
            raise ValueError("stale_context_ttl_sec must be > 0")
        if self.stale_liquidation_ttl_sec <= 0:
            raise ValueError("stale_liquidation_ttl_sec must be > 0")
        if self.parquet_flush_interval_sec <= 0:
            raise ValueError("parquet_flush_interval_sec must be > 0")
        if self.parquet_flush_timeout_sec <= 0:
            raise ValueError("parquet_flush_timeout_sec must be > 0")
        if self.parquet_flush_batch_size <= 0:
            raise ValueError("parquet_flush_batch_size must be > 0")
        if self.parquet_max_load_records_per_key <= 0:
            raise ValueError("parquet_max_load_records_per_key must be > 0")


@dataclass(slots=True)
class FundingMarketContext:
    """
    Local context cache used by funding divergence/pressure analysis.
    """

    latest_open_interest: float | None = None
    previous_open_interest: float | None = None

    latest_price: float | None = None
    previous_price: float | None = None

    latest_cvd: float | None = None
    previous_cvd: float | None = None

    long_liquidations: float | None = None
    short_liquidations: float | None = None

    updated_at: datetime | None = None
    liquidation_updated_at: datetime | None = None


class FundingAnalyzer:
    """
    Event-driven orchestration module for analytics.funding.

    Responsibilities:
    - subscribe to market/context events through EventBus
    - maintain local funding history and context cache
    - build funding statistics
    - call pure funding detectors/analyzers
    - publish normalized analytics.funding.* events through EventBus
    - register periodic cleanup through Scheduler, when provided
    """

    SOURCE = "analytics.funding.funding_analyzer"

    def __init__(
        self,
        *,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
        config: FundingAnalyzerConfig | None = None,
        regime_detector: FundingRegimeDetector | None = None,
        pressure_analyzer: FundingPressureAnalyzer | None = None,
        flip_detector: FundingFlipDetector | None = None,
        extremes_detector: FundingExtremesDetector | None = None,
        divergence_detector: FundingDivergenceDetector | None = None,
        parquet_storage: Any | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.config = config or FundingAnalyzerConfig()
        self.parquet_storage = parquet_storage

        self.logger = get_logger(
            __name__,
            event_type="funding_analyzer",
        )

        self.regime_detector = regime_detector or FundingRegimeDetector(
            FundingRegimeDetectorConfig(default_timeframe=self.config.default_timeframe)
        )
        self.pressure_analyzer = pressure_analyzer or FundingPressureAnalyzer(
            FundingPressureConfig(default_timeframe=self.config.default_timeframe)
        )
        self.flip_detector = flip_detector or FundingFlipDetector(
            FundingFlipDetectorConfig(default_timeframe=self.config.default_timeframe)
        )
        self.extremes_detector = extremes_detector or FundingExtremesDetector(
            FundingExtremesConfig(default_timeframe=self.config.default_timeframe)
        )
        self.divergence_detector = divergence_detector or FundingDivergenceDetector(
            FundingDivergenceConfig(default_timeframe=self.config.default_timeframe)
        )

        self._history: dict[str, Deque[FundingSnapshot]] = defaultdict(
            lambda: deque(maxlen=self.config.history_size)
        )
        self._market_context: dict[str, FundingMarketContext] = defaultdict(FundingMarketContext)

        self._latest_statistics: dict[str, FundingStatistics] = {}
        self._latest_regime_state: dict[str, FundingRegimeState] = {}
        self._latest_pressure_state: dict[str, FundingPressureState] = {}
        self._latest_flip_event: dict[str, FundingFlipEvent] = {}
        self._latest_extreme_event: dict[str, FundingExtremeEvent] = {}
        self._latest_divergence_event: dict[str, FundingDivergenceEvent] = {}

        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._subscriptions: list[Subscription] = []
        self._cleanup_job_id: str | None = None
        self._parquet_flush_job_id: str | None = None
        self._history_write_buffer: list[dict[str, Any]] = []
        self._history_buffer_lock = asyncio.Lock()
        self._parquet_unavailable_logged = False
        self._registered: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self) -> None:
        """
        Register EventBus subscriptions and Scheduler jobs.

        This method is intentionally sync because EventBus.subscribe() and
        Scheduler.add_interval_job() are sync APIs in core.
        """
        if self._registered:
            self.logger.warning("FundingAnalyzer already registered")
            return

        self._subscriptions.extend(
            [
                self.event_bus.subscribe(
                    self.config.funding_event_name,
                    self.on_funding,
                    name="funding_analyzer.on_funding",
                ),
                self.event_bus.subscribe(
                    self.config.open_interest_event_name,
                    self.on_open_interest,
                    name="funding_analyzer.on_open_interest",
                ),
                self.event_bus.subscribe(
                    self.config.candle_event_name,
                    self.on_candle,
                    name="funding_analyzer.on_candle",
                ),
                self.event_bus.subscribe(
                    self.config.trade_event_name,
                    self.on_trade,
                    name="funding_analyzer.on_trade",
                ),
                self.event_bus.subscribe(
                    self.config.cvd_event_name,
                    self.on_cvd_update,
                    name="funding_analyzer.on_cvd_update",
                ),
                self.event_bus.subscribe(
                    self.config.liquidation_event_name,
                    self.on_liquidation,
                    name="funding_analyzer.on_liquidation",
                ),
            ]
        )

        self._register_cleanup_job()
        self._register_parquet_flush_job()
        self._registered = True

        self.logger.info(
            "FundingAnalyzer registered | subscriptions=%s cleanup_job=%s",
            len(self._subscriptions),
            self._cleanup_job_id,
        )

    def unregister(self) -> None:
        """
        Remove EventBus subscriptions and disable Scheduler cleanup job.
        """
        if not self._registered:
            self.logger.warning("FundingAnalyzer is not registered")
            return

        for subscription in list(self._subscriptions):
            self.event_bus.unsubscribe(subscription)
        self._subscriptions.clear()

        if self.scheduler is not None and self._cleanup_job_id is not None:
            try:
                self.scheduler.disable_job(self._cleanup_job_id)
            except KeyError:
                self.logger.warning(
                    "Cleanup job not found during unregister | job_id=%s",
                    self._cleanup_job_id,
                )

        if self.scheduler is not None and self._parquet_flush_job_id is not None:
            try:
                self.scheduler.disable_job(self._parquet_flush_job_id)
            except KeyError:
                self.logger.warning(
                    "Parquet flush job not found during unregister | job_id=%s",
                    self._parquet_flush_job_id,
                )

        self._registered = False
        self.logger.info("FundingAnalyzer unregistered")


    async def start(self) -> None:
        """Load historical parquet state, then register EventBus/Scheduler integration."""
        if self.config.enable_parquet_history and self.config.load_history_from_parquet_on_start:
            await self.load_history_from_parquet()
        self.register()

    async def stop(self) -> None:
        """Flush buffered history and unregister EventBus/Scheduler integration."""
        await self.flush_history_to_parquet()
        self.unregister()

    def get_latest_snapshot(
        self,
        symbol: str,
        exchange: str = "unknown",
    ) -> FundingSnapshot | None:
        history = self._history.get(self._make_key(symbol, exchange))
        if not history:
            return None
        return history[-1]

    def get_statistics(
        self,
        symbol: str,
        exchange: str = "unknown",
    ) -> FundingStatistics | None:
        return self._latest_statistics.get(self._make_key(symbol, exchange))

    def get_regime_state(
        self,
        symbol: str,
        exchange: str = "unknown",
    ) -> FundingRegimeState | None:
        return self._latest_regime_state.get(self._make_key(symbol, exchange))

    def get_pressure_state(
        self,
        symbol: str,
        exchange: str = "unknown",
    ) -> FundingPressureState | None:
        return self._latest_pressure_state.get(self._make_key(symbol, exchange))

    def get_market_context(
        self,
        symbol: str,
        exchange: str = "unknown",
    ) -> FundingMarketContext | None:
        return self._market_context.get(self._make_key(symbol, exchange))

    def stats(self) -> dict[str, Any]:
        return {
            "registered": self._registered,
            "subscriptions": len(self._subscriptions),
            "cleanup_job_id": self._cleanup_job_id,
            "parquet_flush_job_id": self._parquet_flush_job_id,
            "parquet_history_enabled": self.config.enable_parquet_history,
            "parquet_buffer_size": len(self._history_write_buffer),
            "parquet_root": str(self._parquet_root()),
            "symbols_tracked": len(self._history),
            "contexts_tracked": len(self._market_context),
            "latest_statistics": len(self._latest_statistics),
            "latest_regime_states": len(self._latest_regime_state),
            "latest_pressure_states": len(self._latest_pressure_state),
            "latest_flip_events": len(self._latest_flip_event),
            "latest_extreme_events": len(self._latest_extreme_event),
            "latest_divergence_events": len(self._latest_divergence_event),
        }

    async def cleanup_stale_state(self) -> None:
        """
        Scheduler-managed cleanup for stale market context and liquidation context.
        """
        now = self._utc_now()
        removed_contexts = 0
        cleared_liquidations = 0

        for key, context in list(self._market_context.items()):
            if context.updated_at is not None:
                age = (now - context.updated_at).total_seconds()
                if age >= self.config.stale_context_ttl_sec and key not in self._history:
                    self._market_context.pop(key, None)
                    removed_contexts += 1
                    continue

            if context.liquidation_updated_at is not None:
                liq_age = (now - context.liquidation_updated_at).total_seconds()
                if liq_age >= self.config.stale_liquidation_ttl_sec:
                    context.long_liquidations = None
                    context.short_liquidations = None
                    context.liquidation_updated_at = None
                    cleared_liquidations += 1

        if removed_contexts or cleared_liquidations:
            self.logger.info(
                "FundingAnalyzer cleanup completed | removed_contexts=%s cleared_liquidations=%s",
                removed_contexts,
                cleared_liquidations,
            )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def on_funding(self, event: Event) -> None:
        try:
            payload = self._extract_payload(event)
            snapshot = self._parse_funding_snapshot(payload)
        except Exception:
            self.logger.exception("Failed to parse funding event")
            return

        key = self._make_key(snapshot.symbol, snapshot.exchange.value)
        lock = self._locks[key]
        lock_acquired = False

        try:
            await asyncio.wait_for(lock.acquire(), timeout=self.config.state_lock_timeout_sec)
            lock_acquired = True
        except asyncio.TimeoutError:
            self.logger.warning(
                "FundingAnalyzer lock timeout | key=%s timeout=%s",
                key,
                self.config.state_lock_timeout_sec,
            )
            return

        try:
            context = self._market_context[key]
            previous_snapshot = self._history[key][-1] if self._history[key] else None
            previous_regime_state = self._latest_regime_state.get(key)

            self._enrich_snapshot(snapshot, context)
            self._history[key].append(snapshot)

            statistics = self._build_statistics(
                symbol=snapshot.symbol,
                exchange=snapshot.exchange.value,
                history=self._history[key],
                timeframe=self.config.default_timeframe,
            )

            regime_state = self.regime_detector.detect(
                snapshot=snapshot,
                statistics=statistics,
                previous_state=previous_regime_state,
                timeframe=self.config.default_timeframe,
            )

            pressure_state = self.pressure_analyzer.analyze(
                snapshot=snapshot,
                statistics=statistics,
                regime_state=regime_state,
                previous_snapshot=previous_snapshot,
                previous_open_interest=context.previous_open_interest,
                current_price=context.latest_price,
                previous_price=context.previous_price,
                timeframe=self.config.default_timeframe,
            )

            flip_event = self.flip_detector.detect(
                current_snapshot=snapshot,
                previous_snapshot=previous_snapshot,
                statistics=statistics,
                timeframe=self.config.default_timeframe,
            )

            extreme_event = self.extremes_detector.detect(
                snapshot=snapshot,
                statistics=statistics,
                regime_state=regime_state,
                timeframe=self.config.default_timeframe,
            )

            divergence_event = self.divergence_detector.detect(
                snapshot=snapshot,
                statistics=statistics,
                price_change_pct=self._calc_price_change_pct(
                    context.previous_price,
                    context.latest_price,
                ),
                oi_change_pct=self._calc_change_pct(
                    context.previous_open_interest,
                    context.latest_open_interest,
                ),
                cvd_change=self._calc_delta(
                    context.previous_cvd,
                    context.latest_cvd,
                ),
                long_liquidations=context.long_liquidations,
                short_liquidations=context.short_liquidations,
                timeframe=self.config.default_timeframe,
            )

            self._latest_statistics[key] = statistics
            self._latest_regime_state[key] = regime_state
            self._latest_pressure_state[key] = pressure_state

            if flip_event is not None:
                self._latest_flip_event[key] = flip_event
            if extreme_event is not None:
                self._latest_extreme_event[key] = extreme_event
            if divergence_event is not None:
                self._latest_divergence_event[key] = divergence_event

            await self._buffer_history_record(
                snapshot=snapshot,
                statistics=statistics,
                regime_state=regime_state,
                pressure_state=pressure_state,
                flip_event=flip_event,
                extreme_event=extreme_event,
                divergence_event=divergence_event,
                context=context,
            )

            await self._publish_updated_event(
                snapshot=snapshot,
                statistics=statistics,
                regime_state=regime_state,
                pressure_state=pressure_state,
                flip_event=flip_event,
                extreme_event=extreme_event,
                divergence_event=divergence_event,
                correlation_id=event.correlation_id,
            )

            await self._publish_regime_event(regime_state, event.correlation_id)
            await self._publish_pressure_event(pressure_state, event.correlation_id)
            await self._publish_flip_event(flip_event, event.correlation_id)
            await self._publish_extreme_event(extreme_event, event.correlation_id)
            await self._publish_divergence_event(divergence_event, event.correlation_id)
            await self._publish_signal_events(
                snapshot=snapshot,
                regime_state=regime_state,
                pressure_state=pressure_state,
                flip_event=flip_event,
                extreme_event=extreme_event,
                divergence_event=divergence_event,
                correlation_id=event.correlation_id,
            )

        except Exception:
            self.logger.exception(
                "Failed to process funding event | symbol=%s exchange=%s",
                snapshot.symbol,
                snapshot.exchange.value,
            )
        finally:
            if lock_acquired:
                lock.release()

    async def on_open_interest(self, event: Event) -> None:
        try:
            payload = self._extract_payload(event)
            symbol = str(payload["symbol"]).upper().strip()
            exchange = str(payload.get("exchange", "unknown")).lower().strip()
            key = self._make_key(symbol, exchange)

            new_oi = self._to_optional_float(payload.get("open_interest"))
            if new_oi is None:
                return

            context = self._market_context[key]
            context.previous_open_interest = context.latest_open_interest
            context.latest_open_interest = new_oi
            context.updated_at = self._utc_now()
        except Exception:
            self.logger.exception("Failed to process open interest event")

    async def on_candle(self, event: Event) -> None:
        try:
            payload = self._extract_payload(event)
            symbol = str(payload["symbol"]).upper().strip()
            exchange = str(payload.get("exchange", "unknown")).lower().strip()
            key = self._make_key(symbol, exchange)

            price = self._to_optional_float(payload.get("close"))
            if price is None:
                price = self._to_optional_float(payload.get("price"))
            if price is None:
                return

            context = self._market_context[key]
            context.previous_price = context.latest_price
            context.latest_price = price
            context.updated_at = self._utc_now()
        except Exception:
            self.logger.exception("Failed to process candle event")

    async def on_trade(self, event: Event) -> None:
        try:
            payload = self._extract_payload(event)
            trade_payload = payload.get("trade") if isinstance(payload.get("trade"), dict) else payload

            symbol = str(payload.get("symbol") or trade_payload["symbol"]).upper().strip()
            exchange = str(payload.get("exchange") or trade_payload.get("exchange", "unknown")).lower().strip()
            key = self._make_key(symbol, exchange)

            price = self._to_optional_float(trade_payload.get("price"))
            if price is None:
                return

            context = self._market_context[key]
            context.previous_price = context.latest_price
            context.latest_price = price
            context.updated_at = self._utc_now()
        except Exception:
            self.logger.exception("Failed to process trade event")

    async def on_cvd_update(self, event: Event) -> None:
        try:
            payload = self._extract_payload(event)
            inner_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else payload

            symbol = str(inner_payload["symbol"]).upper().strip()
            exchange = str(inner_payload.get("exchange", "unknown")).lower().strip()
            key = self._make_key(symbol, exchange)

            cvd_value = self._to_optional_float(inner_payload.get("cvd"))
            if cvd_value is None:
                cvd_value = self._to_optional_float(inner_payload.get("cumulative_volume_delta"))
            if cvd_value is None:
                return

            context = self._market_context[key]
            context.previous_cvd = context.latest_cvd
            context.latest_cvd = cvd_value
            context.updated_at = self._utc_now()
        except Exception:
            self.logger.exception("Failed to process CVD update event")

    async def on_liquidation(self, event: Event) -> None:
        try:
            payload = self._extract_payload(event)
            symbol = str(payload["symbol"]).upper().strip()
            exchange = str(payload.get("exchange", "unknown")).lower().strip()
            key = self._make_key(symbol, exchange)

            side = str(payload.get("side", "")).lower().strip()
            quantity = self._to_optional_float(payload.get("qty"))
            price = self._to_optional_float(payload.get("price"))
            notional = self._to_optional_float(payload.get("notional"))

            liquidation_value = notional
            if liquidation_value is None and quantity is not None and price is not None:
                liquidation_value = quantity * price

            context = self._market_context[key]
            if liquidation_value is not None:
                if side == "long":
                    context.long_liquidations = liquidation_value
                elif side == "short":
                    context.short_liquidations = liquidation_value

            if side not in {"long", "short"}:
                long_liq = self._to_optional_float(payload.get("long_liquidations"))
                short_liq = self._to_optional_float(payload.get("short_liquidations"))
                if long_liq is not None:
                    context.long_liquidations = long_liq
                if short_liq is not None:
                    context.short_liquidations = short_liq

            context.updated_at = self._utc_now()
            context.liquidation_updated_at = context.updated_at
        except Exception:
            self.logger.exception("Failed to process liquidation event")

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _build_statistics(
        self,
        *,
        symbol: str,
        exchange: str,
        history: Deque[FundingSnapshot],
        timeframe: FundingTimeframe,
    ) -> FundingStatistics:
        if not history:
            raise ValueError("history must not be empty")

        rates = [item.funding_rate for item in history]
        current_rate = rates[-1]
        mean_rate = sum(rates) / len(rates)
        median_rate = median(rates)

        if len(rates) > 1:
            variance = sum((value - mean_rate) ** 2 for value in rates) / len(rates)
            std_rate = variance**0.5
        else:
            std_rate = 0.0

        zscore = (current_rate - mean_rate) / std_rate if std_rate > 0 else None
        percentile = self._calc_percentile(rates, current_rate)

        return FundingStatistics(
            symbol=symbol,
            exchange=self._parse_exchange(exchange),
            timeframe=timeframe,
            current_rate=current_rate,
            mean_rate=mean_rate,
            median_rate=median_rate,
            std_rate=std_rate,
            min_rate=min(rates),
            max_rate=max(rates),
            zscore=zscore,
            percentile=percentile,
            sample_size=len(rates),
            window_start=history[0].event_time,
            window_end=history[-1].event_time,
        )

    def _enrich_snapshot(
        self,
        snapshot: FundingSnapshot,
        context: FundingMarketContext,
    ) -> None:
        if snapshot.open_interest is None:
            snapshot.open_interest = context.latest_open_interest
        if snapshot.mark_price is None:
            snapshot.mark_price = context.latest_price

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

    async def _publish_updated_event(
        self,
        *,
        snapshot: FundingSnapshot,
        statistics: FundingStatistics,
        regime_state: FundingRegimeState,
        pressure_state: FundingPressureState,
        flip_event: FundingFlipEvent | None,
        extreme_event: FundingExtremeEvent | None,
        divergence_event: FundingDivergenceEvent | None,
        correlation_id: str | None,
    ) -> None:
        if not self.config.publish_updated_event:
            return

        event = FundingAnalyticsEvent(
            event_type=FundingEventType.SNAPSHOT,
            symbol=snapshot.symbol,
            exchange=snapshot.exchange,
            timeframe=self.config.default_timeframe,
            payload={
                "snapshot": snapshot.to_dict(),
                "statistics": statistics.to_dict(),
                "regime_state": regime_state.to_dict(),
                "pressure_state": pressure_state.to_dict(),
                "flip_event": flip_event.to_dict() if flip_event is not None else None,
                "extreme_event": extreme_event.to_dict() if extreme_event is not None else None,
                "divergence_event": divergence_event.to_dict() if divergence_event is not None else None,
            },
            event_time=snapshot.event_time,
            source=self.SOURCE,
        )
        await self._emit_analytics_event(
            topic=self.config.analytics_updated_event_name,
            payload=event.to_dict(),
            correlation_id=correlation_id,
        )

    async def _publish_regime_event(
        self,
        regime_state: FundingRegimeState,
        correlation_id: str | None,
    ) -> None:
        if not regime_state.changed and not self.config.publish_regime_event_on_every_update:
            return

        event = FundingAnalyticsEvent(
            event_type=FundingEventType.REGIME,
            symbol=regime_state.symbol,
            exchange=regime_state.exchange,
            timeframe=regime_state.timeframe,
            payload=regime_state.to_dict(),
            event_time=regime_state.event_time,
            source=self.SOURCE,
        )
        await self._emit_analytics_event(
            topic=self.config.analytics_regime_event_name,
            payload=event.to_dict(),
            correlation_id=correlation_id,
        )

    async def _publish_pressure_event(
        self,
        pressure_state: FundingPressureState,
        correlation_id: str | None,
    ) -> None:
        should_emit = (
            self.config.publish_pressure_event_on_every_update
            or self.pressure_analyzer.is_high_pressure(pressure_state)
        )
        if not should_emit:
            return

        event = FundingAnalyticsEvent(
            event_type=FundingEventType.PRESSURE,
            symbol=pressure_state.symbol,
            exchange=pressure_state.exchange,
            timeframe=pressure_state.timeframe,
            payload=pressure_state.to_dict(),
            event_time=pressure_state.event_time,
            source=self.SOURCE,
        )
        await self._emit_analytics_event(
            topic=self.config.analytics_pressure_event_name,
            payload=event.to_dict(),
            correlation_id=correlation_id,
        )

    async def _publish_flip_event(
        self,
        flip_event: FundingFlipEvent | None,
        correlation_id: str | None,
    ) -> None:
        if flip_event is None:
            return

        event = FundingAnalyticsEvent(
            event_type=FundingEventType.FLIP,
            symbol=flip_event.symbol,
            exchange=flip_event.exchange,
            timeframe=flip_event.timeframe,
            payload=flip_event.to_dict(),
            event_time=flip_event.event_time,
            source=self.SOURCE,
        )
        await self._emit_analytics_event(
            topic=self.config.analytics_flip_event_name,
            payload=event.to_dict(),
            correlation_id=correlation_id,
        )

    async def _publish_extreme_event(
        self,
        extreme_event: FundingExtremeEvent | None,
        correlation_id: str | None,
    ) -> None:
        if extreme_event is None:
            return

        event = FundingAnalyticsEvent(
            event_type=FundingEventType.EXTREME,
            symbol=extreme_event.symbol,
            exchange=extreme_event.exchange,
            timeframe=extreme_event.timeframe,
            payload=extreme_event.to_dict(),
            event_time=extreme_event.event_time,
            source=self.SOURCE,
        )
        await self._emit_analytics_event(
            topic=self.config.analytics_extreme_event_name,
            payload=event.to_dict(),
            correlation_id=correlation_id,
        )

    async def _publish_divergence_event(
        self,
        divergence_event: FundingDivergenceEvent | None,
        correlation_id: str | None,
    ) -> None:
        if divergence_event is None:
            return

        event = FundingAnalyticsEvent(
            event_type=FundingEventType.DIVERGENCE,
            symbol=divergence_event.symbol,
            exchange=divergence_event.exchange,
            timeframe=divergence_event.timeframe,
            payload=divergence_event.to_dict(),
            event_time=divergence_event.event_time,
            source=self.SOURCE,
        )
        await self._emit_analytics_event(
            topic=self.config.analytics_divergence_event_name,
            payload=event.to_dict(),
            correlation_id=correlation_id,
        )

    async def _publish_signal_events(
        self,
        *,
        snapshot: FundingSnapshot,
        regime_state: FundingRegimeState,
        pressure_state: FundingPressureState,
        flip_event: FundingFlipEvent | None,
        extreme_event: FundingExtremeEvent | None,
        divergence_event: FundingDivergenceEvent | None,
        correlation_id: str | None,
    ) -> None:
        if not self.config.publish_signal_event:
            return

        signals = self._build_signals(
            snapshot=snapshot,
            regime_state=regime_state,
            pressure_state=pressure_state,
            flip_event=flip_event,
            extreme_event=extreme_event,
            divergence_event=divergence_event,
        )

        for signal in signals:
            event = FundingAnalyticsEvent(
                event_type=FundingEventType.SIGNAL,
                symbol=signal.symbol,
                exchange=signal.exchange,
                timeframe=signal.timeframe,
                payload=signal.to_dict(),
                event_time=signal.event_time,
                source=self.SOURCE,
            )
            await self._emit_analytics_event(
                topic=self.config.analytics_signal_event_name,
                payload=event.to_dict(),
                correlation_id=correlation_id,
                priority=EventPriority.HIGH,
            )

    async def _emit_analytics_event(
        self,
        *,
        topic: str,
        payload: dict[str, Any],
        correlation_id: str | None,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> None:
        await self.event_bus.emit(
            topic,
            payload,
            priority=priority,
            source="funding_analyzer",
            correlation_id=correlation_id,
        )

    # ------------------------------------------------------------------
    # Signal builders
    # ------------------------------------------------------------------

    def _build_signals(
        self,
        *,
        snapshot: FundingSnapshot,
        regime_state: FundingRegimeState,
        pressure_state: FundingPressureState,
        flip_event: FundingFlipEvent | None,
        extreme_event: FundingExtremeEvent | None,
        divergence_event: FundingDivergenceEvent | None,
    ) -> list[FundingSignal]:
        signals: list[FundingSignal] = []

        if self.config.signal_on_regime_change and regime_state.changed:
            previous_regime = regime_state.previous_regime.value if regime_state.previous_regime else "unknown"
            signals.append(
                FundingSignal(
                    symbol=snapshot.symbol,
                    exchange=snapshot.exchange,
                    timeframe=self.config.default_timeframe,
                    signal_type=FundingSignalType.REGIME_CHANGE,
                    bias=regime_state.bias,
                    regime=regime_state.regime,
                    score=self._regime_signal_score(regime_state),
                    confidence=regime_state.confidence,
                    description=f"Funding regime changed from {previous_regime} to {regime_state.regime.value}",
                    supporting_factors=[
                        f"funding_rate={snapshot.funding_rate:.8f}",
                        f"percentile={regime_state.percentile:.2f}" if regime_state.percentile is not None else "percentile=None",
                        f"zscore={regime_state.zscore:.4f}" if regime_state.zscore is not None else "zscore=None",
                    ],
                    tags=["funding", "regime"],
                    event_time=snapshot.event_time,
                )
            )

        if self.config.signal_on_high_pressure and self.pressure_analyzer.is_high_pressure(pressure_state):
            signal_type = (
                FundingSignalType.SQUEEZE_WARNING
                if self.pressure_analyzer.is_squeeze_risk(pressure_state)
                else FundingSignalType.CROWDING_WARNING
            )
            signals.append(
                FundingSignal(
                    symbol=snapshot.symbol,
                    exchange=snapshot.exchange,
                    timeframe=self.config.default_timeframe,
                    signal_type=signal_type,
                    bias=pressure_state.bias,
                    regime=regime_state.regime,
                    score=self._pressure_signal_score(pressure_state),
                    confidence=max(
                        pressure_state.squeeze_probability or 0.0,
                        pressure_state.mean_reversion_probability or 0.0,
                    ),
                    description=self.pressure_analyzer.build_summary(pressure_state),
                    supporting_factors=[
                        f"pressure_score={pressure_state.pressure_score:.4f}",
                        f"squeeze_probability={pressure_state.squeeze_probability:.4f}" if pressure_state.squeeze_probability is not None else "squeeze_probability=None",
                        f"mean_reversion_probability={pressure_state.mean_reversion_probability:.4f}" if pressure_state.mean_reversion_probability is not None else "mean_reversion_probability=None",
                    ],
                    tags=["funding", "pressure", pressure_state.level.value],
                    event_time=snapshot.event_time,
                )
            )

        if self.config.signal_on_flip and flip_event is not None:
            signals.append(
                FundingSignal(
                    symbol=snapshot.symbol,
                    exchange=snapshot.exchange,
                    timeframe=self.config.default_timeframe,
                    signal_type=FundingSignalType.FLIP_DETECTED,
                    bias=regime_state.bias,
                    regime=regime_state.regime,
                    score=self._flip_signal_score(flip_event),
                    confidence=flip_event.confidence,
                    description=self.flip_detector.build_summary(flip_event),
                    supporting_factors=[
                        f"previous_rate={flip_event.previous_rate:.8f}",
                        f"current_rate={flip_event.current_rate:.8f}",
                        f"flip_magnitude={flip_event.flip_magnitude:.8f}",
                    ],
                    tags=["funding", "flip", flip_event.flip_type.value],
                    event_time=snapshot.event_time,
                )
            )

        if self.config.signal_on_extreme and extreme_event is not None:
            signals.append(
                FundingSignal(
                    symbol=snapshot.symbol,
                    exchange=snapshot.exchange,
                    timeframe=self.config.default_timeframe,
                    signal_type=FundingSignalType.SQUEEZE_WARNING if extreme_event.is_squeeze_risk else FundingSignalType.REVERSION_SETUP,
                    bias=regime_state.bias,
                    regime=regime_state.regime,
                    score=self._extreme_signal_score(extreme_event),
                    confidence=extreme_event.severity,
                    description=self.extremes_detector.build_summary(extreme_event),
                    supporting_factors=[
                        f"extreme_type={extreme_event.extreme_type.value}",
                        f"severity={extreme_event.severity:.4f}",
                        f"reversal_risk={extreme_event.is_reversal_risk}",
                        f"squeeze_risk={extreme_event.is_squeeze_risk}",
                    ],
                    tags=["funding", "extreme", extreme_event.extreme_type.value],
                    event_time=snapshot.event_time,
                )
            )

        if self.config.signal_on_divergence and divergence_event is not None:
            signals.append(
                FundingSignal(
                    symbol=snapshot.symbol,
                    exchange=snapshot.exchange,
                    timeframe=self.config.default_timeframe,
                    signal_type=FundingSignalType.DIVERGENCE_DETECTED,
                    bias=regime_state.bias,
                    regime=regime_state.regime,
                    score=self._divergence_signal_score(divergence_event),
                    confidence=divergence_event.confidence,
                    description=self.divergence_detector.build_summary(divergence_event),
                    supporting_factors=[
                        f"type={divergence_event.divergence_type.value}",
                        f"price_change_pct={divergence_event.price_change_pct}" if divergence_event.price_change_pct is not None else "price_change_pct=None",
                        f"oi_change_pct={divergence_event.oi_change_pct}" if divergence_event.oi_change_pct is not None else "oi_change_pct=None",
                        f"cvd_change={divergence_event.cvd_change}" if divergence_event.cvd_change is not None else "cvd_change=None",
                    ],
                    tags=["funding", "divergence", divergence_event.divergence_type.value],
                    event_time=snapshot.event_time,
                )
            )

        return signals

    # ------------------------------------------------------------------
    # Signal score helpers
    # ------------------------------------------------------------------

    def _regime_signal_score(self, regime_state: FundingRegimeState) -> float:
        if regime_state.current_rate > 0:
            return -regime_state.confidence
        if regime_state.current_rate < 0:
            return regime_state.confidence
        return 0.0

    def _pressure_signal_score(self, pressure_state: FundingPressureState) -> float:
        if pressure_state.direction.value == "long":
            return -pressure_state.pressure_score
        if pressure_state.direction.value == "short":
            return pressure_state.pressure_score
        return 0.0

    def _flip_signal_score(self, flip_event: FundingFlipEvent) -> float:
        if flip_event.flip_type.value == "negative_to_positive":
            return -flip_event.confidence
        if flip_event.flip_type.value == "positive_to_negative":
            return flip_event.confidence
        return 0.0

    def _extreme_signal_score(self, extreme_event: FundingExtremeEvent) -> float:
        if extreme_event.funding_rate > 0:
            return -extreme_event.severity
        if extreme_event.funding_rate < 0:
            return extreme_event.severity
        return 0.0

    def _divergence_signal_score(self, divergence_event: FundingDivergenceEvent) -> float:
        if self.divergence_detector.is_bullish_divergence(divergence_event):
            return divergence_event.confidence
        if self.divergence_detector.is_bearish_divergence(divergence_event):
            return -divergence_event.confidence
        return 0.0

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def _register_cleanup_job(self) -> None:
        if not self.config.enable_cleanup_job:
            return

        if self.scheduler is None:
            self.logger.info("FundingAnalyzer cleanup job disabled: scheduler not provided")
            return

        existing_job = self.scheduler.get_job_by_name(self.config.cleanup_job_name)
        if existing_job is not None:
            self._cleanup_job_id = existing_job.job_id
            self.logger.warning(
                "FundingAnalyzer cleanup job already exists | job_id=%s name=%s",
                existing_job.job_id,
                existing_job.name,
            )
            return

        self._cleanup_job_id = self.scheduler.add_interval_job(
            name=self.config.cleanup_job_name,
            func=self.cleanup_stale_state,
            interval=self.config.cleanup_interval_sec,
            timeout=self.config.cleanup_timeout_sec,
            max_retries=1,
            retry_delay=1.0,
            allow_overlap=False,
            run_immediately=False,
            enabled=True,
        )

    def _register_parquet_flush_job(self) -> None:
        if not self.config.enable_parquet_history:
            return

        if self.scheduler is None:
            self.logger.info("FundingAnalyzer parquet flush job disabled: scheduler not provided")
            return

        existing_job = self.scheduler.get_job_by_name(self.config.parquet_flush_job_name)
        if existing_job is not None:
            self._parquet_flush_job_id = existing_job.job_id
            self.logger.warning(
                "FundingAnalyzer parquet flush job already exists | job_id=%s name=%s",
                existing_job.job_id,
                existing_job.name,
            )
            return

        self._parquet_flush_job_id = self.scheduler.add_interval_job(
            name=self.config.parquet_flush_job_name,
            func=self.flush_history_to_parquet,
            interval=self.config.parquet_flush_interval_sec,
            timeout=self.config.parquet_flush_timeout_sec,
            max_retries=1,
            retry_delay=1.0,
            allow_overlap=False,
            run_immediately=False,
            enabled=True,
        )

    # ------------------------------------------------------------------
    # Parquet-backed analytics history
    # ------------------------------------------------------------------

    async def get_history(
        self,
        *,
        symbol: str,
        exchange: str = "unknown",
        limit: int = 100,
        include_parquet: bool = True,
    ) -> list[FundingSnapshot]:
        """Read recent funding snapshots from memory and, optionally, parquet."""
        if limit <= 0:
            return []

        key = self._make_key(symbol, exchange)
        in_memory = list(self._history.get(key, []))[-limit:]
        if len(in_memory) >= limit or not include_parquet or not self.config.enable_parquet_history:
            return in_memory[-limit:]

        records = await self.get_historical_records(
            symbol=symbol,
            exchange=exchange,
            limit=limit,
        )
        snapshots = [self._history_row_to_snapshot(row) for row in records]
        snapshots = [snapshot for snapshot in snapshots if snapshot is not None]

        merged: dict[str, FundingSnapshot] = {
            snapshot.event_time.isoformat(): snapshot
            for snapshot in snapshots + in_memory
        }
        return sorted(merged.values(), key=lambda item: item.event_time)[-limit:]

    async def get_historical_records(
        self,
        *,
        symbol: str | None = None,
        exchange: str | None = None,
        timeframe: FundingTimeframe | str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Read flattened analytics records from parquet history."""
        if not self.config.enable_parquet_history:
            return []

        return await asyncio.to_thread(
            self._read_history_rows_from_parquet,
            symbol,
            exchange,
            timeframe.value if isinstance(timeframe, FundingTimeframe) else timeframe,
            self._ensure_utc(since) if since is not None else None,
            self._ensure_utc(until) if until is not None else None,
            limit,
        )

    async def load_history_from_parquet(
        self,
        *,
        symbol: str | None = None,
        exchange: str | None = None,
    ) -> int:
        """Warm in-memory rolling windows from parquet history."""
        if not self.config.enable_parquet_history:
            return 0

        records = await self.get_historical_records(
            symbol=symbol,
            exchange=exchange,
            limit=self.config.parquet_max_load_records_per_key if symbol and exchange else None,
        )

        loaded = 0
        per_key_loaded: dict[str, int] = defaultdict(int)
        for record in records:
            snapshot = self._history_row_to_snapshot(record)
            if snapshot is None:
                continue
            key = self._make_key(snapshot.symbol, snapshot.exchange.value)
            if per_key_loaded[key] >= self.config.parquet_max_load_records_per_key:
                continue
            self._history[key].append(snapshot)
            per_key_loaded[key] += 1
            loaded += 1

        if loaded:
            self.logger.info(
                "FundingAnalyzer history loaded from parquet | records=%s symbols=%s",
                loaded,
                len(per_key_loaded),
            )
        return loaded

    async def flush_history_to_parquet(self) -> int:
        """Persist buffered funding analytics records to parquet."""
        if not self.config.enable_parquet_history:
            return 0

        async with self._history_buffer_lock:
            if not self._history_write_buffer:
                return 0
            rows = list(self._history_write_buffer)
            self._history_write_buffer.clear()

        try:
            written = await asyncio.to_thread(self._write_history_rows_to_parquet, rows)
            if written:
                self.logger.debug("FundingAnalyzer parquet history flushed | records=%s", written)
            return written
        except Exception:
            async with self._history_buffer_lock:
                self._history_write_buffer[0:0] = rows
            self.logger.exception("Failed to flush FundingAnalyzer history to parquet")
            return 0

    async def _buffer_history_record(
        self,
        *,
        snapshot: FundingSnapshot,
        statistics: FundingStatistics,
        regime_state: FundingRegimeState,
        pressure_state: FundingPressureState,
        flip_event: FundingFlipEvent | None,
        extreme_event: FundingExtremeEvent | None,
        divergence_event: FundingDivergenceEvent | None,
        context: FundingMarketContext,
    ) -> None:
        if not self.config.enable_parquet_history:
            return

        row = self._build_history_row(
            snapshot=snapshot,
            statistics=statistics,
            regime_state=regime_state,
            pressure_state=pressure_state,
            flip_event=flip_event,
            extreme_event=extreme_event,
            divergence_event=divergence_event,
            context=context,
        )

        should_flush = False
        async with self._history_buffer_lock:
            self._history_write_buffer.append(row)
            should_flush = len(self._history_write_buffer) >= self.config.parquet_flush_batch_size

        if should_flush:
            await self.flush_history_to_parquet()

    def _build_history_row(
        self,
        *,
        snapshot: FundingSnapshot,
        statistics: FundingStatistics,
        regime_state: FundingRegimeState,
        pressure_state: FundingPressureState,
        flip_event: FundingFlipEvent | None,
        extreme_event: FundingExtremeEvent | None,
        divergence_event: FundingDivergenceEvent | None,
        context: FundingMarketContext,
    ) -> dict[str, Any]:
        snapshot_dict = snapshot.to_dict()
        statistics_dict = statistics.to_dict()
        regime_dict = regime_state.to_dict()
        pressure_dict = pressure_state.to_dict()
        flip_dict = flip_event.to_dict() if flip_event is not None else None
        extreme_dict = extreme_event.to_dict() if extreme_event is not None else None
        divergence_dict = divergence_event.to_dict() if divergence_event is not None else None

        return {
            "event_kind": "funding_analysis",
            "event_time": snapshot_dict.get("event_time"),
            "received_at": snapshot_dict.get("received_at"),
            "symbol": snapshot.symbol,
            "exchange": snapshot.exchange.value,
            "timeframe": self.config.default_timeframe.value,
            "funding_rate": snapshot.funding_rate,
            "predicted_funding_rate": snapshot.predicted_funding_rate,
            "mark_price": snapshot.mark_price,
            "index_price": snapshot.index_price,
            "basis": snapshot.basis,
            "funding_sign": snapshot.funding_sign,
            "open_interest": snapshot.open_interest,
            "volume_24h": snapshot.volume_24h,
            "next_funding_time": snapshot_dict.get("next_funding_time"),
            "latest_open_interest": context.latest_open_interest,
            "previous_open_interest": context.previous_open_interest,
            "latest_price": context.latest_price,
            "previous_price": context.previous_price,
            "latest_cvd": context.latest_cvd,
            "previous_cvd": context.previous_cvd,
            "long_liquidations": context.long_liquidations,
            "short_liquidations": context.short_liquidations,
            "statistics_json": self._json_dumps(statistics_dict),
            "regime_json": self._json_dumps(regime_dict),
            "pressure_json": self._json_dumps(pressure_dict),
            "flip_json": self._json_dumps(flip_dict),
            "extreme_json": self._json_dumps(extreme_dict),
            "divergence_json": self._json_dumps(divergence_dict),
            "metadata_json": self._json_dumps(snapshot.metadata),
            "created_at": self._utc_now().isoformat(),
        }

    def _write_history_rows_to_parquet(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0

        external_writer = getattr(self.parquet_storage, "append_records", None) or getattr(self.parquet_storage, "write_records", None)
        if external_writer is not None:
            result = external_writer(dataset=self.config.parquet_dataset_name, records=rows)
            if inspectable := getattr(result, "__await__", None):
                raise RuntimeError("Async parquet_storage writers are not supported from sync flush thread")
            return len(rows)

        pd = self._import_pandas_for_parquet()
        if pd is None:
            return 0

        root = self._parquet_root()
        grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            event_time = str(row.get("event_time") or self._utc_now().isoformat())
            event_date = event_time[:10]
            grouped[(row["exchange"], row["symbol"], row["timeframe"], event_date)].append(row)

        written = 0
        for (exchange, symbol, timeframe, event_date), group_rows in grouped.items():
            output_dir = (
                root
                / "snapshots"
                / f"exchange={exchange}"
                / f"symbol={symbol}"
                / f"timeframe={timeframe}"
                / f"date={event_date}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"part-{int(self._utc_now().timestamp() * 1000)}-{uuid4().hex}.parquet"
            pd.DataFrame(group_rows).to_parquet(output_file, index=False)
            written += len(group_rows)
        return written

    def _read_history_rows_from_parquet(
        self,
        symbol: str | None,
        exchange: str | None,
        timeframe: str | None,
        since: datetime | None,
        until: datetime | None,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        external_reader = getattr(self.parquet_storage, "read_records", None)
        if external_reader is not None:
            rows = external_reader(
                dataset=self.config.parquet_dataset_name,
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                since=since,
                until=until,
                limit=limit,
            )
            return list(rows or [])

        pd = self._import_pandas_for_parquet()
        if pd is None:
            return []

        root = self._parquet_root() / "snapshots"
        if not root.exists():
            return []

        files = list(root.rglob("*.parquet"))
        if exchange is not None:
            exchange_part = f"exchange={exchange.lower().strip()}"
            files = [path for path in files if exchange_part in path.parts]
        if symbol is not None:
            symbol_part = f"symbol={symbol.upper().strip()}"
            files = [path for path in files if symbol_part in path.parts]
        if timeframe is not None:
            timeframe_part = f"timeframe={timeframe}"
            files = [path for path in files if timeframe_part in path.parts]

        frames = []
        for file_path in files:
            try:
                frames.append(pd.read_parquet(file_path))
            except Exception:
                self.logger.exception("Failed to read funding parquet file | path=%s", file_path)

        if not frames:
            return []

        df = pd.concat(frames, ignore_index=True)
        if "event_time" in df.columns:
            df["_event_dt"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
            if since is not None:
                df = df[df["_event_dt"] >= since]
            if until is not None:
                df = df[df["_event_dt"] <= until]
            df = df.sort_values("_event_dt")
            df = df.drop(columns=["_event_dt"])

        if limit is not None and limit > 0:
            df = df.tail(limit)
        return df.to_dict(orient="records")

    def _history_row_to_snapshot(self, row: dict[str, Any]) -> FundingSnapshot | None:
        try:
            metadata = self._json_loads(row.get("metadata_json")) or {}
            return FundingSnapshot(
                symbol=str(row["symbol"]),
                exchange=self._parse_exchange(row.get("exchange", "unknown")),
                funding_rate=float(row.get("funding_rate", 0.0)),
                predicted_funding_rate=self._to_optional_float(row.get("predicted_funding_rate")),
                mark_price=self._to_optional_float(row.get("mark_price")),
                index_price=self._to_optional_float(row.get("index_price")),
                open_interest=self._to_optional_float(row.get("open_interest")),
                volume_24h=self._to_optional_float(row.get("volume_24h")),
                next_funding_time=self._parse_datetime(row["next_funding_time"]) if row.get("next_funding_time") else None,
                event_time=self._parse_datetime(row.get("event_time")) if row.get("event_time") else self._utc_now(),
                received_at=self._parse_datetime(row.get("received_at")) if row.get("received_at") else self._utc_now(),
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        except Exception:
            self.logger.exception("Failed to restore FundingSnapshot from parquet row")
            return None

    def _parquet_root(self) -> Path:
        return Path(self.config.parquet_base_path).expanduser() / self.config.parquet_dataset_name

    def _import_pandas_for_parquet(self):
        try:
            import pandas as pd  # type: ignore
            return pd
        except Exception:
            if not self._parquet_unavailable_logged:
                self.logger.warning(
                    "Parquet history is enabled but pandas/pyarrow/fastparquet is unavailable; "
                    "install pandas with pyarrow or inject parquet_storage"
                )
                self._parquet_unavailable_logged = True
            return None

    def _json_dumps(self, value: Any) -> str | None:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

    def _json_loads(self, value: Any) -> Any:
        if value is None or value == "":
            return None
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(str(value))
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    # Parsing / utils
    # ------------------------------------------------------------------

    def _parse_funding_snapshot(self, payload: dict[str, Any]) -> FundingSnapshot:
        symbol = str(payload["symbol"]).upper().strip()
        exchange = self._parse_exchange(payload.get("exchange", "unknown"))

        next_funding_time_raw = (
            payload.get("next_funding_time")
            or payload.get("next_funding_time_ms")
            or payload.get("next_funding_time_ms")
            or payload.get("next_funding_time")
        )
        next_funding_time = (
            self._parse_datetime(next_funding_time_raw)
            if next_funding_time_raw is not None
            else None
        )

        event_time_raw = (
            payload.get("event_time")
            or payload.get("timestamp_ms")
            or payload.get("timestamp")
            or payload.get("ts")
            or payload.get("funding_time")
        )
        received_at_raw = payload.get("received_at") or payload.get("received_at_ms")

        event_time = self._parse_datetime(event_time_raw) if event_time_raw is not None else self._utc_now()
        received_at = self._parse_datetime(received_at_raw) if received_at_raw is not None else self._utc_now()

        raw_metadata = payload.get("metadata")
        metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
        metadata.setdefault("market_type", payload.get("market_type") or payload.get("category") or "perpetual")

        return FundingSnapshot(
            symbol=symbol,
            exchange=exchange,
            funding_rate=float(payload.get("funding_rate", payload.get("rate", 0.0))),
            predicted_funding_rate=self._to_optional_float(
                payload.get("predicted_funding_rate")
                if payload.get("predicted_funding_rate") is not None
                else payload.get("predicted_rate")
            ),
            mark_price=self._to_optional_float(payload.get("mark_price")),
            index_price=self._to_optional_float(payload.get("index_price")),
            open_interest=self._to_optional_float(payload.get("open_interest")),
            volume_24h=self._to_optional_float(payload.get("volume_24h")),
            next_funding_time=next_funding_time,
            event_time=event_time,
            received_at=received_at,
            metadata=metadata,
        )

    def _extract_payload(self, event: Event) -> dict[str, Any]:
        payload = event.payload
        if not isinstance(payload, dict):
            raise TypeError(f"Event payload must be dict, got: {type(payload)!r}")
        return payload

    def _make_key(self, symbol: str, exchange: str) -> str:
        return f"{symbol.upper().strip()}::{exchange.lower().strip()}"

    def _parse_exchange(self, value: Any):
        from .enums import FundingDataSource

        normalized = str(value).strip().lower()
        try:
            return FundingDataSource(normalized)
        except ValueError:
            return FundingDataSource.UNKNOWN

    def _parse_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return self._ensure_utc(value)

        if isinstance(value, (int, float)):
            if value > 1_000_000_000_000:
                return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
            return datetime.fromtimestamp(value, tz=timezone.utc)

        if isinstance(value, str):
            normalized = value.strip().replace("Z", "+00:00")
            return self._ensure_utc(datetime.fromisoformat(normalized))

        raise TypeError(f"Unsupported datetime value: {value!r}")

    def _ensure_utc(self, dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _utc_now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _to_optional_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _calc_percentile(self, values: list[float], current_value: float) -> float | None:
        if not values:
            return None

        sorted_values = sorted(values)
        count = len(sorted_values)
        less_or_equal = sum(1 for value in sorted_values if value <= current_value)
        return max(0.0, min(100.0, (less_or_equal / count) * 100.0))

    def _calc_change_pct(
        self,
        previous: float | None,
        current: float | None,
    ) -> float | None:
        if previous is None or current is None or previous == 0:
            return None
        return (current - previous) / previous

    def _calc_price_change_pct(
        self,
        previous_price: float | None,
        current_price: float | None,
    ) -> float | None:
        return self._calc_change_pct(previous_price, current_price)

    def _calc_delta(
        self,
        previous: float | None,
        current: float | None,
    ) -> float | None:
        if previous is None or current is None:
            return None
        return current - previous