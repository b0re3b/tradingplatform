from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import median
from typing import Any, Deque

from core.logger import get_logger
from .enums import (
    FundingEventType,
    FundingTimeframe,
)
from .funding_divergence import (
    FundingDivergenceConfig,
    FundingDivergenceDetector,
)
from .funding_extremes import (
    FundingExtremesConfig,
    FundingExtremesDetector,
)
from .funding_flip_detector import (
    FundingFlipDetector,
    FundingFlipDetectorConfig,
)
from .funding_pressure import (
    FundingPressureAnalyzer,
    FundingPressureConfig,
)
from .funding_regime_detector import (
    FundingRegimeDetector,
    FundingRegimeDetectorConfig,
)
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
    Конфігурація orchestration-рівня funding analyzer.
    """

    history_size: int = 500
    default_timeframe: FundingTimeframe = FundingTimeframe.H1

    publish_updated_event: bool = True
    publish_regime_event_on_every_update: bool = False
    publish_pressure_event_on_every_update: bool = False
    publish_signal_event: bool = True

    state_lock_timeout_sec: float = 3.0

    funding_event_name: str = "market.funding"
    open_interest_event_name: str = "market.open_interest"
    candle_event_name: str = "market.candle"
    trade_event_name: str = "market.trade"
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


@dataclass(slots=True)
class FundingMarketContext:
    """
    Локальний кеш зовнішнього контексту, який потрібен для divergence/pressure analysis.
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


class FundingAnalyzer:
    """
    Центральний orchestration-клас для analytics.funding.

    Відповідальність:
    - прийом funding та пов'язаних market events
    - збереження локальної funding history
    - побудова статистики
    - виклик детекторів/аналізаторів
    - публікація normalized analytics events у EventBus
    """

    def __init__(
        self,
        event_bus: Any,
        config: FundingAnalyzerConfig | None = None,
        regime_detector: FundingRegimeDetector | None = None,
        pressure_analyzer: FundingPressureAnalyzer | None = None,
        flip_detector: FundingFlipDetector | None = None,
        extremes_detector: FundingExtremesDetector | None = None,
        divergence_detector: FundingDivergenceDetector | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.config = config or FundingAnalyzerConfig()
        self.logger = get_logger(__name__)

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
        self._registered: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self) -> None:
        if self._registered:
            self.logger.warning("FundingAnalyzer already registered")
            return

        self.event_bus.subscribe(self.config.funding_event_name, self.on_funding)
        self.event_bus.subscribe(self.config.open_interest_event_name, self.on_open_interest)
        self.event_bus.subscribe(self.config.candle_event_name, self.on_candle)
        self.event_bus.subscribe(self.config.trade_event_name, self.on_trade)
        self.event_bus.subscribe(self.config.cvd_event_name, self.on_cvd_update)
        self.event_bus.subscribe(self.config.liquidation_event_name, self.on_liquidation)

        self._registered = True
        self.logger.info("FundingAnalyzer registered successfully")

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

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def on_funding(self, event: Any) -> None:
        payload = self._extract_payload(event)
        snapshot = self._parse_funding_snapshot(payload)
        key = self._make_key(snapshot.symbol, snapshot.exchange.value)

        lock = self._locks[key]
        try:
            await asyncio.wait_for(lock.acquire(), timeout=self.config.state_lock_timeout_sec)
        except asyncio.TimeoutError:
            self.logger.warning("FundingAnalyzer lock timeout for %s", key)
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

            await self._publish_updated_event(
                snapshot=snapshot,
                statistics=statistics,
                regime_state=regime_state,
                pressure_state=pressure_state,
                flip_event=flip_event,
                extreme_event=extreme_event,
                divergence_event=divergence_event,
            )

            await self._publish_regime_event(regime_state)
            await self._publish_pressure_event(pressure_state)
            await self._publish_flip_event(flip_event)
            await self._publish_extreme_event(extreme_event)
            await self._publish_divergence_event(divergence_event)
            await self._publish_signal_events(
                snapshot=snapshot,
                regime_state=regime_state,
                pressure_state=pressure_state,
                flip_event=flip_event,
                extreme_event=extreme_event,
                divergence_event=divergence_event,
            )

        except Exception:
            self.logger.exception(
                "Failed to process funding event: symbol=%s exchange=%s",
                snapshot.symbol,
                snapshot.exchange.value,
            )
        finally:
            lock.release()

    async def on_open_interest(self, event: Any) -> None:
        try:
            payload = self._extract_payload(event)
            symbol = str(payload["symbol"]).upper().strip()
            exchange = str(payload.get("exchange", "unknown")).lower().strip()
            key = self._make_key(symbol, exchange)

            context = self._market_context[key]
            new_oi = self._to_optional_float(payload.get("open_interest"))
            if new_oi is None:
                return

            context.previous_open_interest = context.latest_open_interest
            context.latest_open_interest = new_oi
            context.updated_at = self._utc_now()
        except Exception:
            self.logger.exception("Failed to process open interest event")

    async def on_candle(self, event: Any) -> None:
        try:
            payload = self._extract_payload(event)
            symbol = str(payload["symbol"]).upper().strip()
            exchange = str(payload.get("exchange", "unknown")).lower().strip()
            key = self._make_key(symbol, exchange)

            price = (
                self._to_optional_float(payload.get("close"))
                or self._to_optional_float(payload.get("price"))
            )
            if price is None:
                return

            context = self._market_context[key]
            context.previous_price = context.latest_price
            context.latest_price = price
            context.updated_at = self._utc_now()
        except Exception:
            self.logger.exception("Failed to process candle event")

    async def on_trade(self, event: Any) -> None:
        try:
            payload = self._extract_payload(event)
            symbol = str(payload["symbol"]).upper().strip()
            exchange = str(payload.get("exchange", "unknown")).lower().strip()
            key = self._make_key(symbol, exchange)

            price = self._to_optional_float(payload.get("price"))
            if price is None:
                return

            context = self._market_context[key]
            context.previous_price = context.latest_price
            context.latest_price = price
            context.updated_at = self._utc_now()
        except Exception:
            self.logger.exception("Failed to process trade event")

    async def on_cvd_update(self, event: Any) -> None:
        try:
            payload = self._extract_payload(event)

            inner_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else payload
            symbol = str(inner_payload["symbol"]).upper().strip()
            exchange = str(inner_payload.get("exchange", "unknown")).lower().strip()
            key = self._make_key(symbol, exchange)

            cvd_value = (
                self._to_optional_float(inner_payload.get("cvd"))
                or self._to_optional_float(inner_payload.get("cumulative_volume_delta"))
            )
            if cvd_value is None:
                return

            context = self._market_context[key]
            context.previous_cvd = context.latest_cvd
            context.latest_cvd = cvd_value
            context.updated_at = self._utc_now()
        except Exception:
            self.logger.exception("Failed to process CVD update event")

    async def on_liquidation(self, event: Any) -> None:
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

            if liquidation_value is None:
                return

            context = self._market_context[key]
            if side == "long":
                context.long_liquidations = liquidation_value
            elif side == "short":
                context.short_liquidations = liquidation_value
            else:
                long_liq = self._to_optional_float(payload.get("long_liquidations"))
                short_liq = self._to_optional_float(payload.get("short_liquidations"))
                if long_liq is not None:
                    context.long_liquidations = long_liq
                if short_liq is not None:
                    context.short_liquidations = short_liq

            context.updated_at = self._utc_now()
        except Exception:
            self.logger.exception("Failed to process liquidation event")

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _build_statistics(
        self,
        symbol: str,
        exchange: str,
        history: Deque[FundingSnapshot],
        timeframe: FundingTimeframe,
    ) -> FundingStatistics:
        rates = [item.funding_rate for item in history]
        current_rate = rates[-1]
        mean_rate = sum(rates) / len(rates)
        median_rate = median(rates)

        if len(rates) > 1:
            variance = sum((x - mean_rate) ** 2 for x in rates) / len(rates)
            std_rate = variance ** 0.5
        else:
            std_rate = 0.0

        zscore = None
        if std_rate > 0:
            zscore = (current_rate - mean_rate) / std_rate

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
            window_start=history[0].event_time if history else None,
            window_end=history[-1].event_time if history else None,
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
        snapshot: FundingSnapshot,
        statistics: FundingStatistics,
        regime_state: FundingRegimeState,
        pressure_state: FundingPressureState,
        flip_event: FundingFlipEvent | None,
        extreme_event: FundingExtremeEvent | None,
        divergence_event: FundingDivergenceEvent | None,
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
            source="analytics.funding.funding_analyzer",
        )
        await self.event_bus.emit(
            self.config.analytics_updated_event_name,
            event.to_dict(),
            source="funding_analyzer",
        )

    async def _publish_regime_event(self, regime_state: FundingRegimeState) -> None:
        if not regime_state.changed and not self.config.publish_regime_event_on_every_update:
            return

        event = FundingAnalyticsEvent(
            event_type=FundingEventType.REGIME,
            symbol=regime_state.symbol,
            exchange=regime_state.exchange,
            timeframe=regime_state.timeframe,
            payload=regime_state.to_dict(),
            event_time=regime_state.event_time,
            source="analytics.funding.funding_analyzer",
        )
        await self.event_bus.emit(
            self.config.analytics_regime_event_name,
            event.to_dict(),
            source="funding_analyzer",
        )

    async def _publish_pressure_event(self, pressure_state: FundingPressureState) -> None:
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
            source="analytics.funding.funding_analyzer",
        )
        await self.event_bus.emit(
            self.config.analytics_pressure_event_name,
            event.to_dict(),
            source="funding_analyzer",
        )

    async def _publish_flip_event(self, flip_event: FundingFlipEvent | None) -> None:
        if flip_event is None:
            return

        event = FundingAnalyticsEvent(
            event_type=FundingEventType.FLIP,
            symbol=flip_event.symbol,
            exchange=flip_event.exchange,
            timeframe=flip_event.timeframe,
            payload=flip_event.to_dict(),
            event_time=flip_event.event_time,
            source="analytics.funding.funding_analyzer",
        )
        await self.event_bus.emit(
            self.config.analytics_flip_event_name,
            event.to_dict(),
            source="funding_analyzer",
        )

    async def _publish_extreme_event(self, extreme_event: FundingExtremeEvent | None) -> None:
        if extreme_event is None:
            return

        event = FundingAnalyticsEvent(
            event_type=FundingEventType.EXTREME,
            symbol=extreme_event.symbol,
            exchange=extreme_event.exchange,
            timeframe=extreme_event.timeframe,
            payload=extreme_event.to_dict(),
            event_time=extreme_event.event_time,
            source="analytics.funding.funding_analyzer",
        )
        await self.event_bus.emit(
            self.config.analytics_extreme_event_name,
            event.to_dict(),
            source="funding_analyzer",
        )

    async def _publish_divergence_event(self, divergence_event: FundingDivergenceEvent | None) -> None:
        if divergence_event is None:
            return

        event = FundingAnalyticsEvent(
            event_type=FundingEventType.DIVERGENCE,
            symbol=divergence_event.symbol,
            exchange=divergence_event.exchange,
            timeframe=divergence_event.timeframe,
            payload=divergence_event.to_dict(),
            event_time=divergence_event.event_time,
            source="analytics.funding.funding_analyzer",
        )
        await self.event_bus.emit(
            self.config.analytics_divergence_event_name,
            event.to_dict(),
            source="funding_analyzer",
        )

    async def _publish_signal_events(
        self,
        snapshot: FundingSnapshot,
        regime_state: FundingRegimeState,
        pressure_state: FundingPressureState,
        flip_event: FundingFlipEvent | None,
        extreme_event: FundingExtremeEvent | None,
        divergence_event: FundingDivergenceEvent | None,
    ) -> None:
        if not self.config.publish_signal_event:
            return

        signals: list[FundingSignal] = []

        if self.config.signal_on_regime_change and regime_state.changed:
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
                    description=(
                        f"Funding regime changed from "
                        f"{regime_state.previous_regime.value if regime_state.previous_regime else 'unknown'} "
                        f"to {regime_state.regime.value}"
                    ),
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
            signals.append(
                FundingSignal(
                    symbol=snapshot.symbol,
                    exchange=snapshot.exchange,
                    timeframe=self.config.default_timeframe,
                    signal_type=FundingSignalType.SQUEEZE_WARNING
                    if self.pressure_analyzer.is_squeeze_risk(pressure_state)
                    else FundingSignalType.CROWDING_WARNING,
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
                        f"squeeze_probability={pressure_state.squeeze_probability:.4f}"
                        if pressure_state.squeeze_probability is not None else "squeeze_probability=None",
                        f"mean_reversion_probability={pressure_state.mean_reversion_probability:.4f}"
                        if pressure_state.mean_reversion_probability is not None else "mean_reversion_probability=None",
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
                    signal_type=FundingSignalType.SQUEEZE_WARNING
                    if extreme_event.is_squeeze_risk else FundingSignalType.REVERSION_SETUP,
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
                        f"price_change_pct={divergence_event.price_change_pct}"
                        if divergence_event.price_change_pct is not None else "price_change_pct=None",
                        f"oi_change_pct={divergence_event.oi_change_pct}"
                        if divergence_event.oi_change_pct is not None else "oi_change_pct=None",
                        f"cvd_change={divergence_event.cvd_change}"
                        if divergence_event.cvd_change is not None else "cvd_change=None",
                    ],
                    tags=["funding", "divergence", divergence_event.divergence_type.value],
                    event_time=snapshot.event_time,
                )
            )

        for signal in signals:
            event = FundingAnalyticsEvent(
                event_type=FundingEventType.SIGNAL,
                symbol=signal.symbol,
                exchange=signal.exchange,
                timeframe=signal.timeframe,
                payload=signal.to_dict(),
                event_time=signal.event_time,
                source="analytics.funding.funding_analyzer",
            )
            await self.event_bus.emit(
                self.config.analytics_signal_event_name,
                event.to_dict(),
                source="funding_analyzer",
            )

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
    # Parsing / utils
    # ------------------------------------------------------------------

    def _parse_funding_snapshot(self, payload: dict[str, Any]) -> FundingSnapshot:
        symbol = str(payload["symbol"]).upper().strip()
        exchange = self._parse_exchange(payload.get("exchange", "unknown"))

        next_funding_time = payload.get("next_funding_time")
        if next_funding_time is not None:
            next_funding_time = self._parse_datetime(next_funding_time)

        event_time_raw = payload.get("event_time") or payload.get("ts") or payload.get("timestamp")
        received_at_raw = payload.get("received_at")

        event_time = self._parse_datetime(event_time_raw) if event_time_raw is not None else self._utc_now()
        received_at = self._parse_datetime(received_at_raw) if received_at_raw is not None else self._utc_now()

        raw_metadata = payload.get("metadata")

        metadata: dict[str, Any] = (
            raw_metadata if isinstance(raw_metadata, dict) else {}
        )

        return FundingSnapshot(
            symbol=symbol,
            exchange=exchange,
            funding_rate=float(payload.get("funding_rate", 0.0)),
            predicted_funding_rate=self._to_optional_float(payload.get("predicted_funding_rate")),
            mark_price=self._to_optional_float(payload.get("mark_price")),
            index_price=self._to_optional_float(payload.get("index_price")),
            open_interest=self._to_optional_float(payload.get("open_interest")),
            volume_24h=self._to_optional_float(payload.get("volume_24h")),
            next_funding_time=next_funding_time,
            event_time=event_time,
            received_at=received_at,
            metadata=metadata,
        )

    def _extract_payload(self, event: Any) -> dict[str, Any]:
        if isinstance(event, dict):
            return event
        if hasattr(event, "payload"):
            return getattr(event, "payload")
        raise TypeError(f"Unsupported event type: {type(event)!r}")

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
        return float(value)

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