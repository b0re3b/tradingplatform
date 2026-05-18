from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.event_bus import Event, EventBus, EventPriority, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler

from .config import OIAnalyzerConfig
from .enums import OIAnomalyType, OIEventType, OIMarketEventType, OIRegime
from .models import (
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    OIAnalysisResult,
    OIFeatures,
    OIKey,
    OIMarketContext,
    OISnapshot,
    OIState,
    make_oi_key,
)
from .oi_anomaly_detector import OIAnomalyDetector
from .oi_divergence import OIDivergenceDetector
from .oi_features import OIFeatureBuilder
from .oi_regime_detector import OIRegimeDetector


@dataclass(slots=True)
class OIInstrumentBuffers:
    """
    Rolling history for one futures scope:

        exchange + market_type + symbol + timeframe

    All series are stored in chronological order:
        oldest -> newest.
    """

    oi_values: deque[float]
    oi_timestamps: deque[float]

    price_values: deque[float]
    price_timestamps: deque[float]

    volume_values: deque[float]
    volume_timestamps: deque[float]

    feature_history: deque[OIFeatures]

    def append_oi(self, oi: float, timestamp: float) -> None:
        self.oi_values.append(float(oi))
        self.oi_timestamps.append(float(timestamp))

    def append_price(self, price: float, timestamp: float) -> None:
        self.price_values.append(float(price))
        self.price_timestamps.append(float(timestamp))

    def append_volume(self, volume: float, timestamp: float) -> None:
        self.volume_values.append(float(volume))
        self.volume_timestamps.append(float(timestamp))


class OIAnalyzer:
    """
    Event-driven orchestration layer for futures Open Interest analytics.

    Responsibilities:
    - subscribe to normalized data-layer / analytics-layer events;
    - maintain per-scope context/state/history;
    - build OI features;
    - detect regimes, divergences, and anomalies;
    - emit analytics.oi.* events through EventBus;
    - schedule cleanup/metrics jobs through Scheduler.

    This class is infrastructure-aware and is the only Open Interest analytics
    class that should depend on core.EventBus / core.Scheduler.

    Data flow:
        OpenInterestCache   -> market.open_interest.updated
        CandlesCache        -> market.candle.closed / market.candles.updated
        TradesCache         -> market.trades.updated
        FundingCache        -> market.funding.updated
        OrderflowAnalyzer   -> analytics.orderflow.updated
        LiquidationsAnalyzer -> analytics.liquidations.updated

        OIAnalyzer -> analytics.oi.*

    Scope:
        exchange + market_type + symbol + timeframe

    Futures-only examples:
        ("binance", "usdm_futures", "BTCUSDT", "1m")
        ("bybit", "linear", "BTCUSDT", "1m")
        ("okx", "swap", "BTCUSDT", "1m")
        ("mexc", "usdm_futures", "BTCUSDT", "1m")
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
        config: OIAnalyzerConfig | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.config = config or OIAnalyzerConfig()

        self.logger = get_logger(
            __name__,
            service_name=self.config.source_name,
            event_type="analytics_open_interest",
        )

        self.feature_builder = OIFeatureBuilder(self.config)
        self.regime_detector = OIRegimeDetector(self.config)
        self.divergence_detector = OIDivergenceDetector(self.config)
        self.anomaly_detector = OIAnomalyDetector(self.config)

        self._history_size = self.config.windows.history_size

        self._buffers: dict[OIKey, OIInstrumentBuffers] = {}
        self._states: dict[OIKey, OIState] = {}
        self._cooldowns: dict[tuple[str, str, str, str, str], float] = {}
        self._last_context_ts: dict[OIKey, float] = {}

        self._subscriptions: list[Subscription] = []
        self._cleanup_job_id: str | None = None
        self._metrics_job_id: str | None = None

        self._registered = False

    # ------------------------------------------------------------------
    # Lifecycle / registration
    # ------------------------------------------------------------------

    def register(self) -> None:
        """
        Register EventBus subscriptions and optional Scheduler jobs.

        EventBus and Scheduler lifecycles are managed by the application
        bootstrap/container, not by this class.
        """
        if self._registered:
            self.logger.warning("OIAnalyzer already registered")
            return

        self._subscriptions.extend(
            [
                self.event_bus.subscribe(
                    OIMarketEventType.OPEN_INTEREST_UPDATED.topic,
                    self.on_open_interest,
                    name="oi_analyzer.on_open_interest_updated",
                ),
                self.event_bus.subscribe(
                    OIMarketEventType.CANDLE_CLOSED.topic,
                    self.on_candle,
                    name="oi_analyzer.on_candle_closed",
                ),
                self.event_bus.subscribe(
                    OIMarketEventType.CANDLES_UPDATED.topic,
                    self.on_candles_updated,
                    name="oi_analyzer.on_candles_updated",
                ),
                self.event_bus.subscribe(
                    OIMarketEventType.TRADES_UPDATED.topic,
                    self.on_trades_updated,
                    name="oi_analyzer.on_trades_updated",
                ),
                self.event_bus.subscribe(
                    OIMarketEventType.FUNDING_UPDATED.topic,
                    self.on_funding,
                    name="oi_analyzer.on_funding_updated",
                ),
                self.event_bus.subscribe(
                    OIMarketEventType.LIQUIDATIONS_UPDATED.topic,
                    self.on_liquidation,
                    name="oi_analyzer.on_liquidations_updated",
                ),
                self.event_bus.subscribe(
                    OIMarketEventType.ORDERFLOW_UPDATED.topic,
                    self.on_orderflow_update,
                    name="oi_analyzer.on_orderflow_update",
                ),
            ]
        )

        self._register_scheduler_jobs()

        self._registered = True
        self.logger.info(
            "OIAnalyzer registered",
            extra={
                "subscriptions": len(self._subscriptions),
                "scheduler_enabled": self.scheduler is not None,
            },
        )

    def unregister(self) -> None:
        """
        Remove EventBus subscriptions.

        Scheduler jobs are disabled if Scheduler is available. The Scheduler
        remains owned by the application container.
        """
        for subscription in self._subscriptions:
            self.event_bus.unsubscribe(subscription)

        self._subscriptions.clear()

        if self.scheduler is not None:
            for job_id in (self._cleanup_job_id, self._metrics_job_id):
                if job_id is not None and self.scheduler.get_job(job_id) is not None:
                    self.scheduler.disable_job(job_id)

        self._cleanup_job_id = None
        self._metrics_job_id = None
        self._registered = False

        self.logger.info("OIAnalyzer unregistered")

    def _register_scheduler_jobs(self) -> None:
        if self.scheduler is None:
            if (
                self.config.maintenance.enable_periodic_cleanup
                or self.config.maintenance.enable_metrics_emit
            ):
                self.logger.warning(
                    "OIAnalyzer maintenance jobs are enabled but scheduler is not provided"
                )
            return

        maintenance = self.config.maintenance

        if maintenance.enable_periodic_cleanup:
            self._cleanup_job_id = self.scheduler.add_interval_job(
                name=maintenance.cleanup_job_name,
                func=self.cleanup_stale_state,
                interval=maintenance.cleanup_interval_sec,
                run_immediately=False,
                timeout=maintenance.cleanup_job_timeout_sec,
                max_retries=1,
                retry_delay=1.0,
                allow_overlap=False,
            )

        if maintenance.enable_metrics_emit:
            self._metrics_job_id = self.scheduler.add_interval_job(
                name=maintenance.metrics_job_name,
                func=self.emit_metrics,
                interval=maintenance.metrics_interval_sec,
                run_immediately=False,
                timeout=maintenance.metrics_job_timeout_sec,
                max_retries=1,
                retry_delay=1.0,
                allow_overlap=False,
            )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def on_open_interest(self, event: Event) -> None:
        """
        Main OI analysis trigger.

        Expected source:
            OpenInterestCache -> market.open_interest.updated

        Expected payload:
            exchange, market_type, symbol, open_interest, timestamp_ms/received_at_ms
        """
        if not self.config.enabled:
            return

        try:
            payload = self._extract_payload(event)
            snapshot = self._parse_open_interest_payload(payload)
            if snapshot is None:
                return

            key = snapshot.key
            buffers = self._get_or_create_buffers(key)
            state = self._get_or_create_state(key)

            buffers.append_oi(snapshot.oi, snapshot.timestamp)
            state.apply_snapshot(snapshot)

            context = self._get_context_for_key(key)
            if self.config.require_price_context and context is None:
                self.logger.debug(
                    "Skipping OI analysis: price context is required but missing",
                    extra=self._key_payload(key),
                )
                return

            analysis_context = context or self._build_empty_context(snapshot)

            features = self.feature_builder.build_from_raw_inputs(
                snapshot=snapshot,
                context=analysis_context,
                oi_values=list(buffers.oi_values),
                oi_timestamps=list(buffers.oi_timestamps),
                price_values=list(buffers.price_values),
                price_timestamps=list(buffers.price_timestamps),
                volume_values=list(buffers.volume_values),
                volume_timestamps=list(buffers.volume_timestamps),
            )

            buffers.feature_history.append(features)
            state.apply_features(features)

            regime_result = self.regime_detector.detect(features)
            divergence_result = self._detect_divergence_if_possible(key)
            anomaly_result = self.anomaly_detector.detect(features)

            analysis_result = OIAnalysisResult(
                exchange=snapshot.exchange,
                market_type=snapshot.market_type,
                symbol=snapshot.symbol,
                timeframe=snapshot.timeframe,
                exchange_symbol=snapshot.exchange_symbol,
                timestamp=snapshot.timestamp,
                snapshot=snapshot,
                context=analysis_context,
                features=features,
                regime=regime_result,
                divergence=divergence_result,
                anomaly=anomaly_result,
                metadata={
                    "feature_history_size": len(buffers.feature_history),
                    "oi_history_size": len(buffers.oi_values),
                    "price_history_size": len(buffers.price_values),
                    "volume_history_size": len(buffers.volume_values),
                    "source_event_id": getattr(event, "event_id", None),
                    "source_topic": getattr(event, "topic", None),
                    "source": getattr(event, "source", None),
                    "correlation_id": getattr(event, "correlation_id", None),
                },
            )

            previous_regime = state.last_regime
            state.apply_analysis(analysis_result)

            await self._emit_analysis_events(
                key=key,
                previous_regime=previous_regime,
                analysis=analysis_result,
                correlation_id=getattr(event, "correlation_id", None),
            )

        except Exception as exc:
            self.logger.exception(
                "Failed to process market.open_interest.updated event",
                extra={
                    "error": str(exc),
                    "topic": getattr(event, "topic", None),
                    "event_id": getattr(event, "event_id", None),
                },
            )

    async def on_candle(self, event: Event) -> None:
        """
        Price/volume context from CandlesCache.

        Expected source:
            CandlesCache -> market.candle.closed
        """
        if not self.config.enabled:
            return

        try:
            payload = self._extract_payload(event)
            await self._apply_candle_payload(payload)

        except Exception as exc:
            self.logger.exception(
                "Failed to process market.candle.closed event",
                extra={
                    "error": str(exc),
                    "topic": getattr(event, "topic", None),
                    "event_id": getattr(event, "event_id", None),
                },
            )

    async def on_candles_updated(self, event: Event) -> None:
        """
        Optional batch/snapshot candle context from CandlesCache.

        Supports:
            payload["candles"] = [...]
        or:
            payload as one candle-like mapping.
        """
        if not self.config.enabled:
            return

        try:
            payload = self._extract_payload(event)
            candles = payload.get("candles")

            if isinstance(candles, list):
                for candle in candles:
                    if isinstance(candle, Mapping):
                        merged = {
                            **payload,
                            **candle,
                            "exchange": candle.get("exchange", payload.get("exchange")),
                            "market_type": candle.get(
                                "market_type",
                                payload.get("market_type"),
                            ),
                            "symbol": candle.get("symbol", payload.get("symbol")),
                            "timeframe": candle.get(
                                "timeframe",
                                payload.get("timeframe"),
                            ),
                            "exchange_symbol": candle.get(
                                "exchange_symbol",
                                payload.get("exchange_symbol"),
                            ),
                        }
                        await self._apply_candle_payload(merged)
                return

            await self._apply_candle_payload(payload)

        except Exception as exc:
            self.logger.exception(
                "Failed to process market.candles.updated event",
                extra={
                    "error": str(exc),
                    "topic": getattr(event, "topic", None),
                    "event_id": getattr(event, "event_id", None),
                },
            )

    async def on_trades_updated(self, event: Event) -> None:
        """
        Optional fallback context from TradesCache.

        Preferred source for orderflow is analytics.orderflow.updated, but
        market.trades.updated can still provide:
        - last price;
        - volume;
        - basic aggressive buy/sell flow.
        """
        if not self.config.enabled:
            return

        try:
            payload = self._extract_payload(event)
            trades = payload.get("trades")

            if isinstance(trades, list):
                for trade in trades:
                    if isinstance(trade, Mapping):
                        merged = {
                            **payload,
                            **trade,
                            "exchange": trade.get("exchange", payload.get("exchange")),
                            "market_type": trade.get(
                                "market_type",
                                payload.get("market_type"),
                            ),
                            "symbol": trade.get("symbol", payload.get("symbol")),
                            "timeframe": trade.get(
                                "timeframe",
                                payload.get("timeframe"),
                            ),
                            "exchange_symbol": trade.get(
                                "exchange_symbol",
                                payload.get("exchange_symbol"),
                            ),
                        }
                        self._apply_trade_payload(merged)
                return

            self._apply_trade_payload(payload)

        except Exception as exc:
            self.logger.exception(
                "Failed to process market.trades.updated event",
                extra={
                    "error": str(exc),
                    "topic": getattr(event, "topic", None),
                    "event_id": getattr(event, "event_id", None),
                },
            )

    async def on_funding(self, event: Event) -> None:
        """
        Funding context from FundingCache.

        Expected source:
            FundingCache -> market.funding.updated
        """
        if not self.config.enabled:
            return

        try:
            payload = self._extract_payload(event)
            key = self._extract_key_from_payload(payload)
            if key is None:
                return

            timestamp = self._extract_timestamp(payload)
            funding_rate = self._extract_float(
                payload,
                "funding_rate",
                "funding",
                "rate",
            )
            predicted_rate = self._extract_float(
                payload,
                "predicted_rate",
                "predicted_funding_rate",
                "next_funding_rate",
            )
            next_funding_time_ms = self._extract_float(
                payload,
                "next_funding_time_ms",
                "nextFundingTime",
                "funding_time",
            )

            if funding_rate is None and predicted_rate is None:
                return

            context = self._get_or_create_context(key, timestamp)

            if funding_rate is not None:
                context.funding_rate = funding_rate

            if predicted_rate is not None:
                context.predicted_funding_rate = predicted_rate

            if next_funding_time_ms is not None:
                context.next_funding_time_ms = next_funding_time_ms

            context.timestamp = timestamp
            context.source = str(payload.get("source") or "funding_cache")
            self._last_context_ts[key] = timestamp

            state = self._get_or_create_state(key)
            state.apply_context(context)

        except Exception as exc:
            self.logger.exception(
                "Failed to process market.funding.updated event",
                extra={
                    "error": str(exc),
                    "topic": getattr(event, "topic", None),
                    "event_id": getattr(event, "event_id", None),
                },
            )

    async def on_liquidation(self, event: Event) -> None:
        """
        Liquidation context from analytics liquidations layer.

        Expected source:
            LiquidationsAnalyzer -> analytics.liquidations.updated
        """
        if not self.config.enabled:
            return

        try:
            payload = self._extract_payload(event)
            key = self._extract_key_from_payload(payload)
            if key is None:
                return

            timestamp = self._extract_timestamp(payload)

            long_liq = self._extract_float(
                payload,
                "long_liquidations",
                "long_liq",
                "liquidated_longs",
            )
            short_liq = self._extract_float(
                payload,
                "short_liquidations",
                "short_liq",
                "liquidated_shorts",
            )

            side = self._extract_str(payload, "side", "liquidation_side")
            qty = self._extract_float(payload, "qty", "quantity", "size", "volume")

            context = self._get_or_create_context(key, timestamp)

            if long_liq is not None:
                context.long_liquidations = long_liq

            if short_liq is not None:
                context.short_liquidations = short_liq

            if qty is not None and qty >= 0 and side:
                normalized_side = side.lower().strip()
                if normalized_side in {"long", "buy"}:
                    context.long_liquidations = qty
                elif normalized_side in {"short", "sell"}:
                    context.short_liquidations = qty

            context.timestamp = timestamp
            context.source = str(payload.get("source") or "liquidations_analytics")
            self._last_context_ts[key] = timestamp

            state = self._get_or_create_state(key)
            state.apply_context(context)

        except Exception as exc:
            self.logger.exception(
                "Failed to process analytics.liquidations.updated event",
                extra={
                    "error": str(exc),
                    "topic": getattr(event, "topic", None),
                    "event_id": getattr(event, "event_id", None),
                },
            )

    async def on_orderflow_update(self, event: Event) -> None:
        """
        Preferred aggressive flow / CVD context.

        Expected source:
            OrderflowAnalyzer -> analytics.orderflow.updated
        """
        if not self.config.enabled:
            return

        try:
            payload = self._extract_payload(event)
            key = self._extract_key_from_payload(payload)
            if key is None:
                return

            timestamp = self._extract_timestamp(payload)
            context = self._get_or_create_context(key, timestamp)

            cvd_delta = self._extract_float(payload, "cvd_delta", "delta", "cvd")
            aggressive_buy_volume = self._extract_float(
                payload,
                "aggressive_buy_volume",
                "buy_volume",
                "aggressive_buys",
            )
            aggressive_sell_volume = self._extract_float(
                payload,
                "aggressive_sell_volume",
                "sell_volume",
                "aggressive_sells",
            )

            if cvd_delta is not None:
                context.cvd_delta = cvd_delta

            if aggressive_buy_volume is not None:
                context.aggressive_buy_volume = aggressive_buy_volume

            if aggressive_sell_volume is not None:
                context.aggressive_sell_volume = aggressive_sell_volume

            context.timestamp = timestamp
            context.source = str(payload.get("source") or "orderflow_analytics")
            self._last_context_ts[key] = timestamp

            state = self._get_or_create_state(key)
            state.apply_context(context)

        except Exception as exc:
            self.logger.exception(
                "Failed to process analytics.orderflow.updated event",
                extra={
                    "error": str(exc),
                    "topic": getattr(event, "topic", None),
                    "event_id": getattr(event, "event_id", None),
                },
            )

    # ------------------------------------------------------------------
    # Scheduled maintenance jobs
    # ------------------------------------------------------------------

    async def cleanup_stale_state(self) -> None:
        """
        Scheduled cleanup job.

        This should be run by core.scheduler.Scheduler.add_interval_job().
        """
        now_ts = self._now()
        stale_after = self.config.stale_state_cleanup_after_sec
        keys_to_delete: list[OIKey] = []

        for key, state in list(self._states.items()):
            if state.is_stale(now_ts, stale_after):
                keys_to_delete.append(key)

        for key in keys_to_delete:
            self._states.pop(key, None)
            self._buffers.pop(key, None)
            self._last_context_ts.pop(key, None)

            cooldown_keys = [cd_key for cd_key in self._cooldowns if cd_key[:4] == key]
            for cooldown_key in cooldown_keys:
                self._cooldowns.pop(cooldown_key, None)

        if keys_to_delete:
            self.logger.info(
                "Cleaned stale OI state",
                extra={"removed_count": len(keys_to_delete)},
            )

            await self._emit(
                OIEventType.STATE_CLEANED.topic,
                {
                    "timestamp": now_ts,
                    "removed_count": len(keys_to_delete),
                    "removed_keys": [
                        self._key_payload(key)
                        for key in keys_to_delete
                    ],
                },
                priority=EventPriority.LOW,
            )

    async def emit_metrics(self) -> None:
        """
        Scheduled metrics job.

        This should be run by core.scheduler.Scheduler.add_interval_job().
        """
        await self._emit(
            OIEventType.METRICS.topic,
            self.stats(),
            priority=EventPriority.LOW,
        )

    async def emit_health(self) -> None:
        await self._emit(
            OIEventType.HEALTH.topic,
            {
                "timestamp": self._now(),
                "registered": self._registered,
                "enabled": self.config.enabled,
                "states": len(self._states),
                "buffers": len(self._buffers),
                "subscriptions": len(self._subscriptions),
                "scheduler_available": self.scheduler is not None,
                "cleanup_job_id": self._cleanup_job_id,
                "metrics_job_id": self._metrics_job_id,
                "scope": "exchange:market_type:symbol:timeframe",
            },
            priority=EventPriority.LOW,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_state(
        self,
        exchange: str,
        symbol: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> OIState | None:
        return self._states.get(
            self._normalize_key(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            )
        )

    def get_last_analysis(
        self,
        exchange: str,
        symbol: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> OIAnalysisResult | None:
        state = self.get_state(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        if state is None:
            return None
        return state.last_analysis

    def get_feature_history(
        self,
        exchange: str,
        symbol: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> list[OIFeatures]:
        key = self._normalize_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        buffers = self._buffers.get(key)
        if buffers is None:
            return []
        return list(buffers.feature_history)

    def stats(self) -> dict[str, Any]:
        return {
            "timestamp": self._now(),
            "registered": self._registered,
            "enabled": self.config.enabled,
            "states": len(self._states),
            "buffers": len(self._buffers),
            "cooldowns": len(self._cooldowns),
            "subscriptions": len(self._subscriptions),
            "history_size": self._history_size,
            "cleanup_job_registered": self._cleanup_job_id is not None,
            "metrics_job_registered": self._metrics_job_id is not None,
            "scope": "exchange:market_type:symbol:timeframe",
            "instruments": [
                {
                    **self._key_payload(key),
                    "oi_history_size": len(buffers.oi_values),
                    "price_history_size": len(buffers.price_values),
                    "volume_history_size": len(buffers.volume_values),
                    "feature_history_size": len(buffers.feature_history),
                    "has_state": key in self._states,
                }
                for key, buffers in self._buffers.items()
            ],
        }

    # ------------------------------------------------------------------
    # Analytics flow helpers
    # ------------------------------------------------------------------

    def _detect_divergence_if_possible(
        self,
        key: OIKey,
    ) -> Any | None:
        buffers = self._buffers.get(key)
        if buffers is None or len(buffers.feature_history) < 3:
            return None

        try:
            return self.divergence_detector.detect(list(buffers.feature_history))
        except Exception as exc:
            self.logger.exception(
                "Failed to detect OI divergence",
                extra={**self._key_payload(key), "error": str(exc)},
            )
            return None

    async def _emit_analysis_events(
        self,
        *,
        key: OIKey,
        previous_regime: OIRegime,
        analysis: OIAnalysisResult,
        correlation_id: str | None,
    ) -> None:
        if self.config.emit_updates:
            await self._emit_oi_updated(
                analysis,
                correlation_id=correlation_id,
            )

        if (
            self.config.emit_regime_changes
            and analysis.regime.regime != previous_regime
            and self._cooldown_passed(
                key,
                "regime_change",
                self.config.cooldowns.regime_change_cooldown_sec,
                analysis.timestamp,
            )
        ):
            await self._emit_regime_changed(
                previous_regime=previous_regime,
                analysis=analysis,
                correlation_id=correlation_id,
            )

        if (
            self.config.emit_divergences
            and analysis.divergence is not None
            and analysis.divergence.detected
            and self._cooldown_passed(
                key,
                "divergence",
                self.config.cooldowns.divergence_event_cooldown_sec,
                analysis.timestamp,
            )
        ):
            await self._emit_divergence_detected(
                analysis,
                correlation_id=correlation_id,
            )

        if (
            self.config.emit_anomalies
            and analysis.anomaly is not None
            and analysis.anomaly.detected
            and self._cooldown_passed(
                key,
                "anomaly",
                self.config.cooldowns.anomaly_event_cooldown_sec,
                analysis.timestamp,
            )
        ):
            await self._emit_anomaly_detected(
                analysis,
                correlation_id=correlation_id,
            )

        if (
            self.config.emit_squeeze_events
            and analysis.regime.regime == OIRegime.SQUEEZE_SETUP
            and self._cooldown_passed(
                key,
                "squeeze_setup",
                self.config.cooldowns.squeeze_event_cooldown_sec,
                analysis.timestamp,
            )
        ):
            await self._emit_squeeze_setup(
                analysis,
                correlation_id=correlation_id,
            )

        if (
            self.config.emit_capitulation_events
            and self._is_capitulation_event(analysis)
            and self._cooldown_passed(
                key,
                "capitulation",
                self.config.cooldowns.capitulation_event_cooldown_sec,
                analysis.timestamp,
            )
        ):
            await self._emit_capitulation_detected(
                analysis,
                correlation_id=correlation_id,
            )

    @staticmethod
    def _is_capitulation_event(analysis: OIAnalysisResult) -> bool:
        if analysis.regime.regime == OIRegime.CAPITULATION:
            return True

        if analysis.anomaly is None or not analysis.anomaly.detected:
            return False

        return analysis.anomaly.anomaly_type in {
            OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP,
            OIAnomalyType.SUDDEN_DELEVERAGING,
            OIAnomalyType.OI_COLLAPSE,
        }

    # ------------------------------------------------------------------
    # Emit helpers
    # ------------------------------------------------------------------

    async def _emit_oi_updated(
        self,
        analysis: OIAnalysisResult,
        *,
        correlation_id: str | None,
    ) -> None:
        await self._emit(
            OIEventType.UPDATED.topic,
            self._analysis_payload(analysis),
            priority=EventPriority.NORMAL,
            correlation_id=correlation_id,
        )

    async def _emit_regime_changed(
        self,
        *,
        previous_regime: OIRegime,
        analysis: OIAnalysisResult,
        correlation_id: str | None,
    ) -> None:
        await self._emit(
            OIEventType.REGIME_CHANGED.topic,
            {
                **self._analysis_scope_payload(analysis),
                "timestamp": analysis.timestamp,
                "previous_regime": previous_regime.value,
                "new_regime": analysis.regime.regime.value,
                "confidence": analysis.regime.confidence,
                "score": analysis.regime.score,
                "reasons": list(analysis.regime.reasons),
                "features": analysis.features.to_dict(),
            },
            priority=EventPriority.HIGH,
            correlation_id=correlation_id,
        )

    async def _emit_divergence_detected(
        self,
        analysis: OIAnalysisResult,
        *,
        correlation_id: str | None,
    ) -> None:
        if analysis.divergence is None:
            return

        await self._emit(
            OIEventType.DIVERGENCE_DETECTED.topic,
            {
                **self._analysis_scope_payload(analysis),
                "timestamp": analysis.timestamp,
                "divergence_type": analysis.divergence.divergence_type.value,
                "confidence": analysis.divergence.confidence,
                "score": analysis.divergence.score,
                "window_size": analysis.divergence.window_size,
                "reasons": list(analysis.divergence.reasons),
                "regime": analysis.regime.regime.value,
                "features": analysis.features.to_dict(),
            },
            priority=EventPriority.HIGH,
            correlation_id=correlation_id,
        )

    async def _emit_anomaly_detected(
        self,
        analysis: OIAnalysisResult,
        *,
        correlation_id: str | None,
    ) -> None:
        if analysis.anomaly is None:
            return

        await self._emit(
            OIEventType.ANOMALY_DETECTED.topic,
            {
                **self._analysis_scope_payload(analysis),
                "timestamp": analysis.timestamp,
                "anomaly_type": analysis.anomaly.anomaly_type.value,
                "strength": analysis.anomaly.strength.value,
                "confidence": analysis.anomaly.confidence,
                "score": analysis.anomaly.score,
                "reasons": list(analysis.anomaly.reasons),
                "regime": analysis.regime.regime.value,
                "features": analysis.features.to_dict(),
            },
            priority=EventPriority.HIGH,
            correlation_id=correlation_id,
        )

    async def _emit_squeeze_setup(
        self,
        analysis: OIAnalysisResult,
        *,
        correlation_id: str | None,
    ) -> None:
        await self._emit(
            OIEventType.SQUEEZE_SETUP.topic,
            {
                **self._analysis_scope_payload(analysis),
                "timestamp": analysis.timestamp,
                "regime": analysis.regime.regime.value,
                "confidence": analysis.regime.confidence,
                "score": analysis.regime.score,
                "reasons": list(analysis.regime.reasons),
                "features": analysis.features.to_dict(),
            },
            priority=EventPriority.HIGH,
            correlation_id=correlation_id,
        )

    async def _emit_capitulation_detected(
        self,
        analysis: OIAnalysisResult,
        *,
        correlation_id: str | None,
    ) -> None:
        anomaly_type = (
            analysis.anomaly.anomaly_type.value
            if analysis.anomaly is not None and analysis.anomaly.detected
            else None
        )

        await self._emit(
            OIEventType.CAPITULATION_DETECTED.topic,
            {
                **self._analysis_scope_payload(analysis),
                "timestamp": analysis.timestamp,
                "regime": analysis.regime.regime.value,
                "regime_confidence": analysis.regime.confidence,
                "anomaly_type": anomaly_type,
                "features": analysis.features.to_dict(),
                "reasons": self._collect_capitulation_reasons(analysis),
            },
            priority=EventPriority.CRITICAL,
            correlation_id=correlation_id,
        )

    async def _emit(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: str | None = None,
    ) -> bool:
        return await self.event_bus.emit(
            topic,
            payload,
            priority=priority,
            source=self.config.source_name,
            correlation_id=correlation_id,
        )

    def _analysis_payload(self, analysis: OIAnalysisResult) -> dict[str, Any]:
        return analysis.to_dict()

    @staticmethod
    def _analysis_scope_payload(analysis: OIAnalysisResult) -> dict[str, Any]:
        return {
            "exchange": analysis.exchange,
            "market_type": analysis.market_type,
            "symbol": analysis.symbol,
            "exchange_symbol": analysis.exchange_symbol,
            "timeframe": analysis.timeframe,
            "key": list(analysis.key),
        }

    @staticmethod
    def _collect_capitulation_reasons(analysis: OIAnalysisResult) -> list[str]:
        reasons: list[str] = []
        reasons.extend(analysis.regime.reasons)

        if analysis.anomaly is not None:
            reasons.extend(analysis.anomaly.reasons)

        return list(dict.fromkeys(reasons))

    # ------------------------------------------------------------------
    # Context / state / buffers
    # ------------------------------------------------------------------

    def _get_or_create_buffers(
        self,
        key: OIKey,
    ) -> OIInstrumentBuffers:
        buffers = self._buffers.get(key)
        if buffers is not None:
            return buffers

        buffers = OIInstrumentBuffers(
            oi_values=deque(maxlen=self._history_size),
            oi_timestamps=deque(maxlen=self._history_size),
            price_values=deque(maxlen=self._history_size),
            price_timestamps=deque(maxlen=self._history_size),
            volume_values=deque(maxlen=self._history_size),
            volume_timestamps=deque(maxlen=self._history_size),
            feature_history=deque(maxlen=self._history_size),
        )
        self._buffers[key] = buffers
        return buffers

    def _get_or_create_state(
        self,
        key: OIKey,
    ) -> OIState:
        state = self._states.get(key)
        if state is not None:
            return state

        exchange, market_type, symbol, timeframe = key
        state = OIState(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        self._states[key] = state
        return state

    def _get_or_create_context(
        self,
        key: OIKey,
        timestamp: float,
    ) -> OIMarketContext:
        state = self._get_or_create_state(key)

        if state.last_context is None:
            exchange, market_type, symbol, timeframe = key
            state.last_context = OIMarketContext(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
            )

        return state.last_context

    def _get_context_for_key(
        self,
        key: OIKey,
    ) -> OIMarketContext | None:
        state = self._states.get(key)
        if state is None or state.last_context is None:
            return None

        if state.last_context.is_stale(
            self._now(),
            self.config.stale_context_after_sec,
        ):
            return None

        return state.last_context

    @staticmethod
    def _build_empty_context(snapshot: OISnapshot) -> OIMarketContext:
        return OIMarketContext(
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            symbol=snapshot.symbol,
            exchange_symbol=snapshot.exchange_symbol,
            timeframe=snapshot.timeframe,
            timestamp=snapshot.timestamp,
            mark_price=snapshot.mark_price,
            index_price=snapshot.index_price,
            source="empty_context",
        )

    def _cooldown_passed(
        self,
        key: OIKey,
        event_kind: str,
        cooldown_sec: float,
        now_ts: float,
    ) -> bool:
        cooldown_key = (*key, event_kind)
        last_ts = self._cooldowns.get(cooldown_key)

        if last_ts is None or (now_ts - last_ts) >= cooldown_sec:
            self._cooldowns[cooldown_key] = now_ts
            return True

        return False

    # ------------------------------------------------------------------
    # Context update helpers
    # ------------------------------------------------------------------

    async def _apply_candle_payload(self, payload: Mapping[str, Any]) -> None:
        key = self._extract_key_from_payload(payload)
        if key is None:
            return

        timestamp = self._extract_timestamp(payload)
        close_price = self._extract_float(
            payload,
            "close",
            "c",
            "price",
            "last_price",
        )
        mark_price = self._extract_float(payload, "mark_price")
        index_price = self._extract_float(payload, "index_price")

        volume = self._extract_float(payload, "volume", "v", "base_volume")
        quote_volume = self._extract_float(
            payload,
            "quote_volume",
            "quoteVolume",
            "quote_volume_24h",
        )

        buffers = self._get_or_create_buffers(key)
        context = self._get_or_create_context(key, timestamp)

        if close_price is not None:
            buffers.append_price(close_price, timestamp)
            self._update_price_context(
                context=context,
                buffers=buffers,
                price=close_price,
            )

        if mark_price is not None:
            context.mark_price = mark_price

        if index_price is not None:
            context.index_price = index_price

        if volume is not None and volume >= 0:
            buffers.append_volume(volume, timestamp)
            self._update_volume_context(
                context=context,
                buffers=buffers,
                volume=volume,
            )

        if quote_volume is not None and quote_volume >= 0:
            context.quote_volume = quote_volume

        context.timestamp = timestamp
        context.source = str(payload.get("source") or "candles_cache")
        self._last_context_ts[key] = timestamp

        state = self._get_or_create_state(key)
        state.apply_context(context)

    def _apply_trade_payload(self, payload: Mapping[str, Any]) -> None:
        key = self._extract_key_from_payload(payload)
        if key is None:
            return

        timestamp = self._extract_timestamp(payload)
        price = self._extract_float(payload, "price", "p")
        qty = self._extract_float(payload, "qty", "quantity", "q", "size", "volume")
        side = self._extract_str(payload, "side", "taker_side", "aggressor_side")

        buffers = self._get_or_create_buffers(key)
        context = self._get_or_create_context(key, timestamp)

        if price is not None:
            buffers.append_price(price, timestamp)
            self._update_price_context(
                context=context,
                buffers=buffers,
                price=price,
            )

        if qty is not None and qty >= 0:
            buffers.append_volume(qty, timestamp)
            self._update_volume_context(
                context=context,
                buffers=buffers,
                volume=qty,
            )
            self._update_aggressive_flow_context(
                context=context,
                side=side,
                qty=qty,
            )

        context.timestamp = timestamp
        context.source = str(payload.get("source") or "trades_cache")
        self._last_context_ts[key] = timestamp

        state = self._get_or_create_state(key)
        state.apply_context(context)

    def _update_price_context(
        self,
        *,
        context: OIMarketContext,
        buffers: OIInstrumentBuffers,
        price: float,
    ) -> None:
        previous_price = (
            float(buffers.price_values[-2])
            if len(buffers.price_values) >= 2
            else None
        )

        context.price = price

        if previous_price is not None:
            context.price_delta = price - previous_price
            if abs(previous_price) > 1e-12:
                context.price_delta_pct = (
                    (price - previous_price) / abs(previous_price)
                ) * 100.0

    def _update_volume_context(
        self,
        *,
        context: OIMarketContext,
        buffers: OIInstrumentBuffers,
        volume: float,
    ) -> None:
        context.volume = volume

        volume_ma = self.feature_builder.compute_moving_average(
            list(buffers.volume_values),
            self.config.windows.volume_window,
        )
        context.volume_ma = volume_ma
        context.volume_ratio = self.feature_builder.compute_volume_ratio(
            volume,
            volume_ma,
        )

    @staticmethod
    def _update_aggressive_flow_context(
        *,
        context: OIMarketContext,
        side: str | None,
        qty: float,
    ) -> None:
        if not side:
            return

        normalized_side = side.lower().strip()
        if normalized_side in {"buy", "bid", "long"}:
            context.aggressive_buy_volume = qty
        elif normalized_side in {"sell", "ask", "short"}:
            context.aggressive_sell_volume = qty

    # ------------------------------------------------------------------
    # Parsing / normalization
    # ------------------------------------------------------------------

    def _parse_open_interest_payload(
        self,
        payload: Mapping[str, Any],
    ) -> OISnapshot | None:
        key = self._extract_key_from_payload(payload)
        if key is None:
            self.logger.warning("OI payload missing exchange/market_type/symbol")
            return None

        timestamp = self._extract_timestamp(payload)
        oi = self._extract_float(
            payload,
            "oi",
            "open_interest",
            "openInterest",
            "value",
        )

        if oi is None:
            self.logger.warning(
                "OI payload missing OI value",
                extra=self._key_payload(key),
            )
            return None

        exchange, market_type, symbol, timeframe = key

        return OISnapshot(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            exchange_symbol=self._extract_str(payload, "exchange_symbol"),
            timestamp=timestamp,
            oi=oi,
            open_interest_value=self._extract_float(
                payload,
                "open_interest_value",
                "oi_value",
                "notional_value",
            ),
            mark_price=self._extract_float(payload, "mark_price", "markPrice"),
            index_price=self._extract_float(payload, "index_price", "indexPrice"),
            source=self._extract_str(payload, "source") or "open_interest_cache",
            metadata={
                "raw_topic_source": payload.get("source"),
                "received_at_ms": payload.get("received_at_ms"),
                "timestamp_ms": payload.get("timestamp_ms"),
            },
        )

    @staticmethod
    def _extract_payload(event: Event | Mapping[str, Any]) -> Mapping[str, Any]:
        payload = event.payload if isinstance(event, Event) else event

        if isinstance(payload, Mapping):
            return payload

        raise ValueError("Event payload must be a mapping-like object")

    def _extract_key_from_payload(
        self,
        payload: Mapping[str, Any],
    ) -> OIKey | None:
        exchange = self._extract_str(payload, "exchange", "venue", "source_exchange")
        market_type = self._extract_str(
            payload,
            "market_type",
            "category",
            "inst_type",
            "instrument_type",
        )
        symbol = self._extract_str(payload, "symbol", "instrument", "market")
        timeframe = self._extract_str(payload, "timeframe", "tf", "interval")

        if not exchange or not symbol:
            return None

        return self._normalize_key(
            exchange=exchange,
            market_type=market_type or DEFAULT_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe or DEFAULT_TIMEFRAME,
        )

    def _normalize_key(
        self,
        *,
        exchange: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        symbol: str,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> OIKey:
        normalized_symbol = (
            symbol.upper().strip()
            if self.config.normalize_symbol
            else symbol.strip()
        )

        return make_oi_key(
            exchange=exchange,
            market_type=market_type,
            symbol=normalized_symbol,
            timeframe=timeframe,
        )

    def _extract_timestamp(self, payload: Mapping[str, Any]) -> float:
        timestamp = self._extract_float(
            payload,
            "timestamp",
            "timestamp_ms",
            "received_at_ms",
            "ts",
            "time",
            "event_time",
            "T",
            "close_time_ms",
            "open_time_ms",
            "last_update_ts_ms",
        )

        if timestamp is None:
            return self._now()

        if timestamp > 10_000_000_000:
            return timestamp / 1000.0

        return timestamp

    @staticmethod
    def _extract_float(
        payload: Mapping[str, Any],
        *keys: str,
    ) -> float | None:
        for key in keys:
            if key not in payload or payload[key] is None:
                continue

            try:
                value = float(payload[key])
            except (TypeError, ValueError):
                continue

            if math.isfinite(value):
                return value

        return None

    @staticmethod
    def _extract_str(
        payload: Mapping[str, Any],
        *keys: str,
    ) -> str | None:
        for key in keys:
            if key not in payload or payload[key] is None:
                continue

            value = str(payload[key]).strip()
            if value:
                return value

        return None

    @staticmethod
    def _key_payload(key: OIKey) -> dict[str, Any]:
        exchange, market_type, symbol, timeframe = key
        return {
            "exchange": exchange,
            "market_type": market_type,
            "symbol": symbol,
            "timeframe": timeframe,
            "key": list(key),
        }

    @staticmethod
    def _now() -> float:
        return time.time()