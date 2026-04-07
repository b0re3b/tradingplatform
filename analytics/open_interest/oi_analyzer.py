from __future__ import annotations

import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.logger import get_logger
from .config import OIAnalyzerConfig
from .enums import OIAnomalyType, OIEventType, OIRegime
from .models import (
    OIAnalysisResult,
    OIMarketContext,
    OISnapshot,
    OIState,
)
from .oi_anomaly_detector import OIAnomalyDetector
from .oi_divergence import OIDivergenceDetector
from .oi_features import OIFeatureBuilder
from .oi_regime_detector import OIRegimeDetector


@dataclass(slots=True)
class OIInstrumentBuffers:
    """
    Історія по конкретному (exchange, symbol).
    Усі значення зберігаються у хронологічному порядку.
    """

    oi_values: deque[float]
    oi_timestamps: deque[float]

    price_values: deque[float]
    price_timestamps: deque[float]

    volume_values: deque[float]
    volume_timestamps: deque[float]

    feature_history: deque[Any]

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
    Orchestration layer для Open Interest analytics.

    Відповідальність:
    - слухати market events
    - підтримувати state/history
    - будувати OI features
    - визначати regime
    - визначати divergence
    - визначати anomalies
    - публікувати analytics.oi.* події в EventBus
    """

    def __init__(
        self,
        event_bus: Any,
        config: OIAnalyzerConfig | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.config = config or OIAnalyzerConfig()
        self.logger = get_logger(__name__, service_name="oi_analyzer")

        self.feature_builder = OIFeatureBuilder(self.config)
        self.regime_detector = OIRegimeDetector(self.config)
        self.divergence_detector = OIDivergenceDetector(self.config)
        self.anomaly_detector = OIAnomalyDetector(self.config)

        history_size = self.config.windows.history_size

        self._buffers: dict[tuple[str, str], OIInstrumentBuffers] = {}
        self._states: dict[tuple[str, str], OIState] = {}

        self._cooldowns: dict[tuple[str, str, str], float] = {}

        self._last_context_ts: dict[tuple[str, str], float] = {}
        self._last_cleanup_ts: float = 0.0

        self._history_size = history_size

        self._subscribed = False

    # -------------------------------------------------------------------------
    # Lifecycle / registration
    # -------------------------------------------------------------------------

    def register(self) -> None:
        """
        Реєструє підписки на EventBus.
        """
        if self._subscribed:
            return

        self.event_bus.subscribe("market.open_interest", self.on_open_interest)
        self.event_bus.subscribe("market.candle", self.on_candle)
        self.event_bus.subscribe("market.trade", self.on_trade)
        self.event_bus.subscribe("market.funding", self.on_funding)
        self.event_bus.subscribe("market.liquidation", self.on_liquidation)

        # Опціонально підтягуємо orderflow context, якщо він є в системі
        self.event_bus.subscribe("analytics.orderflow.updated", self.on_orderflow_update)

        self._subscribed = True
        self.logger.info("OIAnalyzer registered event subscriptions")

    # -------------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------------

    async def on_open_interest(self, event: Any) -> None:
        if not self.config.enabled:
            return

        try:
            payload = self._extract_payload(event)
            snapshot = self._parse_open_interest_payload(payload)
            if snapshot is None:
                return

            key = snapshot.key
            now_ts = self._now()

            self._maybe_cleanup_stale_state(now_ts)

            buffers = self._get_or_create_buffers(key)
            state = self._get_or_create_state(key)

            buffers.append_oi(snapshot.oi, snapshot.timestamp)

            context = self._get_context_for_key(key)
            if self.config.require_price_context and context is None:
                self.logger.debug(
                    "Skipping OI analysis because price context is required but missing",
                    extra={"exchange": snapshot.exchange, "symbol": snapshot.symbol},
                )
                state.last_snapshot = snapshot
                state.touch(snapshot.timestamp)
                return

            features = self.feature_builder.build_from_raw_inputs(
                snapshot=snapshot,
                context=context,
                oi_values=list(buffers.oi_values),
                oi_timestamps=list(buffers.oi_timestamps),
                price_values=list(buffers.price_values),
                price_timestamps=list(buffers.price_timestamps),
                volume_values=list(buffers.volume_values),
                volume_timestamps=list(buffers.volume_timestamps),
            )

            buffers.feature_history.append(features)

            regime_result = self.regime_detector.detect(features)
            divergence_result = self._detect_divergence_if_possible(key)
            anomaly_result = self.anomaly_detector.detect(features)

            analysis_result = OIAnalysisResult(
                symbol=snapshot.symbol,
                exchange=snapshot.exchange,
                timestamp=snapshot.timestamp,
                snapshot=snapshot,
                context=context or self._build_empty_context(snapshot),
                features=features,
                regime=regime_result,
                divergence=divergence_result,
                anomaly=anomaly_result,
                metadata={
                    "feature_history_size": len(buffers.feature_history),
                    "oi_history_size": len(buffers.oi_values),
                },
            )

            previous_regime = state.last_regime

            state.last_snapshot = snapshot
            state.last_features = features
            state.last_analysis = analysis_result if self.config.store_full_analysis else None
            state.last_regime = regime_result.regime
            state.touch(snapshot.timestamp)

            if self.config.emit_updates:
                await self._emit_oi_updated(analysis_result)

            if (
                self.config.emit_regime_changes
                and regime_result.regime != previous_regime
                and self._cooldown_passed(
                    key,
                    "regime_change",
                    self.config.cooldowns.regime_change_cooldown_sec,
                    snapshot.timestamp,
                )
            ):
                await self._emit_regime_changed(
                    key=key,
                    timestamp=snapshot.timestamp,
                    previous_regime=previous_regime,
                    analysis=analysis_result,
                )

            if (
                self.config.emit_divergences
                and divergence_result is not None
                and divergence_result.detected
                and self._cooldown_passed(
                    key,
                    "divergence",
                    self.config.cooldowns.divergence_event_cooldown_sec,
                    snapshot.timestamp,
                )
            ):
                await self._emit_divergence_detected(analysis_result)

            if (
                self.config.emit_anomalies
                and anomaly_result is not None
                and anomaly_result.detected
                and self._cooldown_passed(
                    key,
                    "anomaly",
                    self.config.cooldowns.anomaly_event_cooldown_sec,
                    snapshot.timestamp,
                )
            ):
                await self._emit_anomaly_detected(analysis_result)

            if (
                self.config.emit_squeeze_events
                and regime_result.regime == OIRegime.SQUEEZE_SETUP
                and self._cooldown_passed(
                    key,
                    "squeeze_setup",
                    self.config.cooldowns.squeeze_event_cooldown_sec,
                    snapshot.timestamp,
                )
            ):
                await self._emit_squeeze_setup(analysis_result)

            if (
                self.config.emit_capitulation_events
                and (
                    regime_result.regime == OIRegime.CAPITULATION
                    or (
                        anomaly_result is not None
                        and anomaly_result.detected
                        and anomaly_result.anomaly_type
                        in {
                            OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP,
                            OIAnomalyType.SUDDEN_DELEVERAGING,
                            OIAnomalyType.OI_COLLAPSE,
                        }
                    )
                )
                and self._cooldown_passed(
                    key,
                    "capitulation",
                    self.config.cooldowns.capitulation_event_cooldown_sec,
                    snapshot.timestamp,
                )
            ):
                await self._emit_capitulation_detected(analysis_result)

        except Exception as exc:
            self.logger.exception(
                "Failed to process market.open_interest event",
                extra={"error": str(exc)},
            )

    async def on_candle(self, event: Any) -> None:
        try:
            payload = self._extract_payload(event)
            key = self._extract_key_from_payload(payload)
            if key is None:
                return

            exchange, symbol = key
            timestamp = self._extract_timestamp(payload)
            close_price = self._extract_float(
                payload,
                "close",
                "c",
                "price",
                "last_price",
            )
            volume = self._extract_float(payload, "volume", "v", "base_volume")

            if close_price is not None:
                buffers = self._get_or_create_buffers(key)
                buffers.append_price(close_price, timestamp)

            if volume is not None:
                buffers = self._get_or_create_buffers(key)
                buffers.append_volume(volume, timestamp)

            context = self._get_or_create_context(key, timestamp)
            context.price = close_price if close_price is not None else context.price
            context.volume = volume if volume is not None else context.volume

            previous_price = None
            buffers = self._get_or_create_buffers(key)
            if len(buffers.price_values) >= 2:
                previous_price = float(buffers.price_values[-2])

            if close_price is not None and previous_price is not None:
                context.price_delta = close_price - previous_price
                if previous_price != 0:
                    context.price_delta_pct = ((close_price - previous_price) / abs(previous_price)) * 100.0

            volume_ma = self.feature_builder.compute_moving_average(
                list(buffers.volume_values),
                self.config.windows.volume_window,
            )
            context.volume_ma = volume_ma
            context.volume_ratio = self.feature_builder.compute_volume_ratio(volume, volume_ma)

            context.timestamp = timestamp
            self._last_context_ts[key] = timestamp

        except Exception as exc:
            self.logger.exception(
                "Failed to process market.candle event",
                extra={"error": str(exc)},
            )

    async def on_trade(self, event: Any) -> None:
        """
        Trade event може слугувати fallback-джерелом для:
        - last price
        - агресивного buy/sell flow
        - опціонально volume
        """
        try:
            payload = self._extract_payload(event)
            key = self._extract_key_from_payload(payload)
            if key is None:
                return

            timestamp = self._extract_timestamp(payload)
            price = self._extract_float(payload, "price", "p")
            qty = self._extract_float(payload, "qty", "quantity", "q", "size")
            side = self._extract_str(payload, "side", "taker_side")

            buffers = self._get_or_create_buffers(key)
            context = self._get_or_create_context(key, timestamp)

            if price is not None:
                buffers.append_price(price, timestamp)
                previous_price = float(buffers.price_values[-2]) if len(buffers.price_values) >= 2 else None

                context.price = price
                if previous_price is not None:
                    context.price_delta = price - previous_price
                    if previous_price != 0:
                        context.price_delta_pct = ((price - previous_price) / abs(previous_price)) * 100.0

            if qty is not None and qty >= 0:
                buffers.append_volume(qty, timestamp)
                context.volume = qty

                volume_ma = self.feature_builder.compute_moving_average(
                    list(buffers.volume_values),
                    self.config.windows.volume_window,
                )
                context.volume_ma = volume_ma
                context.volume_ratio = self.feature_builder.compute_volume_ratio(qty, volume_ma)

            if qty is not None and side:
                normalized_side = side.lower().strip()
                if normalized_side in {"buy", "bid", "long"}:
                    context.aggressive_buy_volume = float(qty)
                elif normalized_side in {"sell", "ask", "short"}:
                    context.aggressive_sell_volume = float(qty)

            context.timestamp = timestamp
            self._last_context_ts[key] = timestamp

        except Exception as exc:
            self.logger.exception(
                "Failed to process market.trade event",
                extra={"error": str(exc)},
            )

    async def on_funding(self, event: Any) -> None:
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
            if funding_rate is None:
                return

            context = self._get_or_create_context(key, timestamp)
            context.funding_rate = funding_rate
            context.timestamp = timestamp
            self._last_context_ts[key] = timestamp

        except Exception as exc:
            self.logger.exception(
                "Failed to process market.funding event",
                extra={"error": str(exc)},
            )

    async def on_liquidation(self, event: Any) -> None:
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

            # fallback: інколи приходить одна liquidation event з side + qty
            side = self._extract_str(payload, "side")
            qty = self._extract_float(payload, "qty", "quantity", "size", "volume")

            context = self._get_or_create_context(key, timestamp)

            if long_liq is not None:
                context.long_liquidations = long_liq
            if short_liq is not None:
                context.short_liquidations = short_liq

            if qty is not None and side:
                normalized_side = side.lower().strip()
                if normalized_side in {"long", "buy"}:
                    context.long_liquidations = qty
                elif normalized_side in {"short", "sell"}:
                    context.short_liquidations = qty

            context.timestamp = timestamp
            self._last_context_ts[key] = timestamp

        except Exception as exc:
            self.logger.exception(
                "Failed to process market.liquidation event",
                extra={"error": str(exc)},
            )

    async def on_orderflow_update(self, event: Any) -> None:
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
            )
            aggressive_sell_volume = self._extract_float(
                payload,
                "aggressive_sell_volume",
                "sell_volume",
            )

            if cvd_delta is not None:
                context.cvd_delta = cvd_delta
            if aggressive_buy_volume is not None:
                context.aggressive_buy_volume = aggressive_buy_volume
            if aggressive_sell_volume is not None:
                context.aggressive_sell_volume = aggressive_sell_volume

            context.timestamp = timestamp
            self._last_context_ts[key] = timestamp

        except Exception as exc:
            self.logger.exception(
                "Failed to process analytics.orderflow.updated event",
                extra={"error": str(exc)},
            )

    # -------------------------------------------------------------------------
    # Public helpers
    # -------------------------------------------------------------------------

    def get_state(self, exchange: str, symbol: str) -> OIState | None:
        return self._states.get(self._normalize_key(exchange, symbol))

    def get_last_analysis(self, exchange: str, symbol: str) -> OIAnalysisResult | None:
        state = self.get_state(exchange, symbol)
        if state is None:
            return None
        return state.last_analysis

    def get_feature_history(self, exchange: str, symbol: str) -> list[Any]:
        key = self._normalize_key(exchange, symbol)
        buffers = self._buffers.get(key)
        if buffers is None:
            return []
        return list(buffers.feature_history)

    # -------------------------------------------------------------------------
    # Internal analytics flow helpers
    # -------------------------------------------------------------------------

    def _detect_divergence_if_possible(
        self,
        key: tuple[str, str],
    ) -> Any | None:
        buffers = self._buffers.get(key)
        if buffers is None or len(buffers.feature_history) < 3:
            return None

        try:
            return self.divergence_detector.detect(list(buffers.feature_history))
        except Exception as exc:
            self.logger.exception(
                "Failed to detect OI divergence",
                extra={"exchange": key[0], "symbol": key[1], "error": str(exc)},
            )
            return None

    async def _emit_oi_updated(self, analysis: OIAnalysisResult) -> None:
        payload = self._analysis_payload(analysis)
        await self._emit(
            OIEventType.UPDATED.value,
            payload,
            source="oi_analyzer",
        )

    async def _emit_regime_changed(
        self,
        *,
        key: tuple[str, str],
        timestamp: float,
        previous_regime: OIRegime,
        analysis: OIAnalysisResult,
    ) -> None:
        payload = {
            "symbol": analysis.symbol,
            "exchange": analysis.exchange,
            "timestamp": timestamp,
            "previous_regime": previous_regime.value,
            "new_regime": analysis.regime.regime.value,
            "confidence": analysis.regime.confidence,
            "score": analysis.regime.score,
            "reasons": list(analysis.regime.reasons),
            "features": analysis.features.to_dict(),
        }
        await self._emit(
            OIEventType.REGIME_CHANGED.value,
            payload,
            source="oi_analyzer",
        )

    async def _emit_divergence_detected(self, analysis: OIAnalysisResult) -> None:
        divergence = analysis.divergence
        if divergence is None:
            return

        payload = {
            "symbol": analysis.symbol,
            "exchange": analysis.exchange,
            "timestamp": analysis.timestamp,
            "divergence_type": divergence.divergence_type.value,
            "confidence": divergence.confidence,
            "score": divergence.score,
            "window_size": divergence.window_size,
            "reasons": list(divergence.reasons),
            "regime": analysis.regime.regime.value,
            "features": analysis.features.to_dict(),
        }
        await self._emit(
            OIEventType.DIVERGENCE_DETECTED.value,
            payload,
            source="oi_analyzer",
        )

    async def _emit_anomaly_detected(self, analysis: OIAnalysisResult) -> None:
        anomaly = analysis.anomaly
        if anomaly is None:
            return

        payload = {
            "symbol": analysis.symbol,
            "exchange": analysis.exchange,
            "timestamp": analysis.timestamp,
            "anomaly_type": anomaly.anomaly_type.value,
            "strength": anomaly.strength.value,
            "confidence": anomaly.confidence,
            "score": anomaly.score,
            "reasons": list(anomaly.reasons),
            "regime": analysis.regime.regime.value,
            "features": analysis.features.to_dict(),
        }
        await self._emit(
            OIEventType.ANOMALY_DETECTED.value,
            payload,
            source="oi_analyzer",
        )

    async def _emit_squeeze_setup(self, analysis: OIAnalysisResult) -> None:
        payload = {
            "symbol": analysis.symbol,
            "exchange": analysis.exchange,
            "timestamp": analysis.timestamp,
            "regime": analysis.regime.regime.value,
            "confidence": analysis.regime.confidence,
            "score": analysis.regime.score,
            "reasons": list(analysis.regime.reasons),
            "features": analysis.features.to_dict(),
        }
        await self._emit(
            OIEventType.SQUEEZE_SETUP.value,
            payload,
            source="oi_analyzer",
        )

    async def _emit_capitulation_detected(self, analysis: OIAnalysisResult) -> None:
        anomaly_type = (
            analysis.anomaly.anomaly_type.value
            if analysis.anomaly is not None and analysis.anomaly.detected
            else None
        )

        payload = {
            "symbol": analysis.symbol,
            "exchange": analysis.exchange,
            "timestamp": analysis.timestamp,
            "regime": analysis.regime.regime.value,
            "regime_confidence": analysis.regime.confidence,
            "anomaly_type": anomaly_type,
            "features": analysis.features.to_dict(),
            "reasons": self._collect_capitulation_reasons(analysis),
        }
        await self._emit(
            OIEventType.CAPITULATION_DETECTED.value,
            payload,
            source="oi_analyzer",
        )

    async def _emit(
        self,
        event_name: str,
        payload: dict[str, Any],
        *,
        source: str,
    ) -> None:
        await self.event_bus.emit(
            event_name,
            payload,
            source=source,
        )

    def _analysis_payload(self, analysis: OIAnalysisResult) -> dict[str, Any]:
        return {
            "symbol": analysis.symbol,
            "exchange": analysis.exchange,
            "timestamp": analysis.timestamp,
            "snapshot": analysis.snapshot.to_dict(),
            "context": analysis.context.to_dict(),
            "features": analysis.features.to_dict(),
            "regime": analysis.regime.to_dict(),
            "divergence": analysis.divergence.to_dict() if analysis.divergence else None,
            "anomaly": analysis.anomaly.to_dict() if analysis.anomaly else None,
            "metadata": dict(analysis.metadata),
        }

    def _collect_capitulation_reasons(self, analysis: OIAnalysisResult) -> list[str]:
        reasons: list[str] = []

        reasons.extend(analysis.regime.reasons)

        if analysis.anomaly is not None:
            reasons.extend(analysis.anomaly.reasons)

        # дедуплікація зі збереженням порядку
        seen: set[str] = set()
        unique_reasons: list[str] = []
        for item in reasons:
            if item not in seen:
                seen.add(item)
                unique_reasons.append(item)

        return unique_reasons

    # -------------------------------------------------------------------------
    # Context / state / buffers
    # -------------------------------------------------------------------------

    def _get_or_create_buffers(
        self,
        key: tuple[str, str],
    ) -> OIInstrumentBuffers:
        buffers = self._buffers.get(key)
        if buffers is None:
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
        key: tuple[str, str],
    ) -> OIState:
        state = self._states.get(key)
        if state is None:
            state = OIState(
                exchange=key[0],
                symbol=key[1],
            )
            self._states[key] = state
        return state

    def _get_or_create_context(
            self,
            key: tuple[str, str],
            timestamp: float,
    ) -> OIMarketContext:
        state = self._get_or_create_state(key)

        context = state.last_context
        if context is None:
            context = OIMarketContext(
                symbol=key[1],
                exchange=key[0],
                timestamp=timestamp,
            )
            state.last_context = context

        return context

    def _get_context_for_key(
        self,
        key: tuple[str, str],
    ) -> OIMarketContext | None:
        state = self._states.get(key)
        if state is None or state.last_context is None:
            return None

        last_ts = self._last_context_ts.get(key)
        if last_ts is None:
            return state.last_context

        if (self._now() - last_ts) > self.config.stale_context_after_sec:
            return None

        return state.last_context

    def _build_empty_context(self, snapshot: OISnapshot) -> OIMarketContext:
        return OIMarketContext(
            symbol=snapshot.symbol,
            exchange=snapshot.exchange,
            timestamp=snapshot.timestamp,
        )

    def _maybe_cleanup_stale_state(self, now_ts: float) -> None:
        if (now_ts - self._last_cleanup_ts) < 30.0:
            return

        stale_after = self.config.stale_state_cleanup_after_sec
        keys_to_delete: list[tuple[str, str]] = []

        for key, state in self._states.items():
            if state.last_update_ts is None:
                continue
            if (now_ts - state.last_update_ts) > stale_after:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            self._states.pop(key, None)
            self._buffers.pop(key, None)

            cooldown_keys = [cd_key for cd_key in self._cooldowns if cd_key[:2] == key]
            for cd_key in cooldown_keys:
                self._cooldowns.pop(cd_key, None)

            self._last_context_ts.pop(key, None)

        if keys_to_delete:
            self.logger.info(
                "Cleaned stale OI state",
                extra={"removed_count": len(keys_to_delete)},
            )

        self._last_cleanup_ts = now_ts

    def _cooldown_passed(
        self,
        key: tuple[str, str],
        event_kind: str,
        cooldown_sec: float,
        now_ts: float,
    ) -> bool:
        cd_key = (key[0], key[1], event_kind)
        last_ts = self._cooldowns.get(cd_key)
        if last_ts is None or (now_ts - last_ts) >= cooldown_sec:
            self._cooldowns[cd_key] = now_ts
            return True
        return False

    # -------------------------------------------------------------------------
    # Parsing / normalization
    # -------------------------------------------------------------------------

    def _parse_open_interest_payload(
        self,
        payload: Mapping[str, Any],
    ) -> OISnapshot | None:
        key = self._extract_key_from_payload(payload)
        if key is None:
            self.logger.warning("OI payload missing exchange/symbol")
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
                extra={"exchange": key[0], "symbol": key[1]},
            )
            return None

        return OISnapshot(
            symbol=key[1],
            exchange=key[0],
            timestamp=timestamp,
            oi=oi,
        )

    def _extract_payload(self, event: Any) -> Mapping[str, Any]:
        payload = getattr(event, "payload", event)
        if isinstance(payload, Mapping):
            return payload
        raise ValueError("Event payload must be a mapping-like object")

    def _extract_key_from_payload(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[str, str] | None:
        exchange = self._extract_str(payload, "exchange", "venue", "source_exchange")
        symbol = self._extract_str(payload, "symbol", "instrument", "market")

        if not exchange or not symbol:
            return None

        return self._normalize_key(exchange, symbol)

    def _normalize_key(self, exchange: str, symbol: str) -> tuple[str, str]:
        normalized_exchange = exchange.lower().strip()
        normalized_symbol = symbol.upper().strip() if self.config.normalize_symbol else symbol.strip()
        return normalized_exchange, normalized_symbol

    def _extract_timestamp(self, payload: Mapping[str, Any]) -> float:
        ts = self._extract_float(
            payload,
            "timestamp",
            "ts",
            "time",
            "event_time",
            "T",
        )
        if ts is None:
            return self._now()

        # Якщо timestamp виглядає як milliseconds
        if ts > 10_000_000_000:
            return ts / 1000.0
        return ts

    def _extract_float(
        self,
        payload: Mapping[str, Any],
        *keys: str,
    ) -> float | None:
        for key in keys:
            if key in payload and payload[key] is not None:
                try:
                    return float(payload[key])
                except (TypeError, ValueError):
                    continue
        return None

    def _extract_str(
        self,
        payload: Mapping[str, Any],
        *keys: str,
    ) -> str | None:
        for key in keys:
            if key in payload and payload[key] is not None:
                value = str(payload[key]).strip()
                if value:
                    return value
        return None

    def _now(self) -> float:
        return time.time()