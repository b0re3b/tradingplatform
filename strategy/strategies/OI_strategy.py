from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from core.logger import get_logger

from analytics.open_interest.enums import (
    OIAnomalyType,
    OIDivergenceType,
    OIEventType,
    OIRegime,
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


@dataclass(slots=True)
class OIStrategyConfig:
    enabled: bool = True

    emit_rejections: bool = True
    allow_longs: bool = True
    allow_shorts: bool = True

    min_signal_confidence: float = 0.60
    strong_signal_confidence: float = 0.75

    regime_weight: float = 0.34
    divergence_weight: float = 0.23
    anomaly_weight: float = 0.21
    pressure_weight: float = 0.12
    flow_weight: float = 0.10

    min_pressure_for_trend_trade: float = 0.10
    min_volume_ratio: float = 1.00
    max_signal_age_sec: float = 20.0

    cooldown_sec: float = 15.0
    symbol_cooldown_sec: float = 10.0

    emit_on_updated: bool = True
    emit_on_regime_changed: bool = True
    emit_on_divergence: bool = True
    emit_on_anomaly: bool = True
    emit_on_squeeze: bool = True
    emit_on_capitulation: bool = True

    def __post_init__(self) -> None:
        self.min_signal_confidence = _clamp(self.min_signal_confidence)
        self.strong_signal_confidence = _clamp(self.strong_signal_confidence)

        if self.cooldown_sec < 0:
            raise ValueError("cooldown_sec must be >= 0")
        if self.symbol_cooldown_sec < 0:
            raise ValueError("symbol_cooldown_sec must be >= 0")
        if self.max_signal_age_sec <= 0:
            raise ValueError("max_signal_age_sec must be > 0")
        if self.min_volume_ratio < 0:
            raise ValueError("min_volume_ratio must be >= 0")


@dataclass(slots=True)
class OIStrategyDecision:
    symbol: str
    exchange: str
    timestamp: float

    side: str
    confidence: float
    score: float

    strategy_name: str = "oi_strategy"
    regime: str = OIRegime.NEUTRAL.value
    divergence_type: str | None = None
    anomaly_type: str | None = None

    reasons: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_signal_payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timestamp": self.timestamp,
            "side": self.side,
            "confidence": self.confidence,
            "score": self.score,
            "strategy": self.strategy_name,
            "reason": self.reasons[0] if self.reasons else "oi_signal",
            "reasons": list(self.reasons),
            "tags": list(self.tags),
            "regime": self.regime,
            "divergence_type": self.divergence_type,
            "anomaly_type": self.anomaly_type,
            "features": dict(self.features),
            "context": dict(self.context),
            "metadata": dict(self.metadata),
        }


class OIStrategy:
    """
    Strategy module, який перетворює OI analytics events у торгові сигнали.

    Логіка:
    - слухає analytics.oi.* події
    - нормалізує payload
    - оцінює bullish / bearish score
    - якщо score > threshold -> emit signal.generated
    - інакше -> emit signal.rejected
    """

    def __init__(
        self,
        event_bus: Any,
        config: OIStrategyConfig | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.config = config or OIStrategyConfig()
        self.logger = get_logger(__name__, service_name="oi_strategy")

        self._subscribed = False
        self._last_emit_ts_by_key: dict[tuple[str, str, str], float] = {}
        self._last_symbol_emit_ts: dict[tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register(self) -> None:
        if self._subscribed:
            return

        if self.config.emit_on_updated:
            self.event_bus.subscribe(OIEventType.UPDATED.value, self.on_oi_updated)

        if self.config.emit_on_regime_changed:
            self.event_bus.subscribe(
                OIEventType.REGIME_CHANGED.value,
                self.on_regime_changed,
            )

        if self.config.emit_on_divergence:
            self.event_bus.subscribe(
                OIEventType.DIVERGENCE_DETECTED.value,
                self.on_divergence_detected,
            )

        if self.config.emit_on_anomaly:
            self.event_bus.subscribe(
                OIEventType.ANOMALY_DETECTED.value,
                self.on_anomaly_detected,
            )

        if self.config.emit_on_squeeze:
            self.event_bus.subscribe(
                OIEventType.SQUEEZE_SETUP.value,
                self.on_squeeze_setup,
            )

        if self.config.emit_on_capitulation:
            self.event_bus.subscribe(
                OIEventType.CAPITULATION_DETECTED.value,
                self.on_capitulation_detected,
            )

        self._subscribed = True
        self.logger.info("OIStrategy registered event subscriptions")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def on_oi_updated(self, event: Any) -> None:
        await self._process_event(event, trigger="oi_updated")

    async def on_regime_changed(self, event: Any) -> None:
        await self._process_event(event, trigger="regime_changed")

    async def on_divergence_detected(self, event: Any) -> None:
        await self._process_event(event, trigger="divergence_detected")

    async def on_anomaly_detected(self, event: Any) -> None:
        await self._process_event(event, trigger="anomaly_detected")

    async def on_squeeze_setup(self, event: Any) -> None:
        await self._process_event(event, trigger="squeeze_setup")

    async def on_capitulation_detected(self, event: Any) -> None:
        await self._process_event(event, trigger="capitulation_detected")

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    async def _process_event(self, event: Any, *, trigger: str) -> None:
        if not self.config.enabled:
            return

        try:
            payload = self._extract_payload(event)
            normalized = self._normalize_payload(payload)
            if normalized is None:
                return

            if self._is_stale(normalized):
                self.logger.debug(
                    "Skipping stale OI strategy payload",
                    extra={
                        "symbol": normalized["symbol"],
                        "exchange": normalized["exchange"],
                        "trigger": trigger,
                    },
                )
                return

            decision = self._build_decision(normalized, trigger=trigger)
            if decision is None:
                if self.config.emit_rejections:
                    await self._emit_rejection(normalized, trigger, "no_clear_oi_edge")
                return

            if not self._cooldown_passed(decision, trigger):
                return

            await self._emit_signal(decision, trigger)

        except Exception as exc:
            self.logger.exception(
                "Failed to process OI strategy event",
                extra={"trigger": trigger, "error": str(exc)},
            )

    def _build_decision(
        self,
        data: dict[str, Any],
        *,
        trigger: str,
    ) -> OIStrategyDecision | None:
        symbol = data["symbol"]
        exchange = data["exchange"]
        timestamp = data["timestamp"]

        features = _safe_dict(data.get("features"))
        context = _safe_dict(data.get("context"))
        regime_block = _safe_dict(data.get("regime"))
        divergence_block = _safe_dict(data.get("divergence"))
        anomaly_block = _safe_dict(data.get("anomaly"))

        regime = str(regime_block.get("regime") or data.get("regime") or OIRegime.NEUTRAL.value)
        divergence_type = divergence_block.get("divergence_type") or data.get("divergence_type")
        anomaly_type = anomaly_block.get("anomaly_type") or data.get("anomaly_type")

        bullish_score = 0.0
        bearish_score = 0.0
        reasons_long: list[str] = []
        reasons_short: list[str] = []
        tags: list[str] = [f"trigger:{trigger}", "module:open_interest"]

        regime_conf = _safe_float(regime_block.get("confidence"), 0.0) or 0.0
        divergence_conf = _safe_float(divergence_block.get("confidence"), 0.0) or _safe_float(
            data.get("confidence"),
            0.0,
        ) or 0.0
        anomaly_conf = _safe_float(anomaly_block.get("confidence"), 0.0) or _safe_float(
            data.get("confidence"),
            0.0,
        ) or 0.0

        oi_pressure = _safe_float(features.get("oi_pressure_score"), 0.0) or 0.0
        volume_ratio = _safe_float(features.get("volume_ratio"))
        price_delta_pct = _safe_float(features.get("price_delta_pct"))
        oi_delta_pct = _safe_float(features.get("oi_delta_pct"))
        liq_imbalance = _safe_float(features.get("liquidation_imbalance"))
        flow_imbalance = _safe_float(features.get("aggressive_flow_imbalance"))
        funding_rate = _safe_float(features.get("funding_rate"))
        oi_zscore = _safe_float(features.get("oi_zscore"))

        # --------------------------------------------------------------
        # Regime scoring
        # --------------------------------------------------------------
        if regime in {OIRegime.LONG_BUILDUP.value, OIRegime.SHORT_COVERING.value, OIRegime.TREND_CONFIRMATION.value}:
            bullish_score += self.config.regime_weight * max(regime_conf, 0.55)
            reasons_long.append(f"bullish_regime:{regime}")

        if regime in {OIRegime.SHORT_BUILDUP.value, OIRegime.LONG_UNWIND.value}:
            bearish_score += self.config.regime_weight * max(regime_conf, 0.55)
            reasons_short.append(f"bearish_regime:{regime}")

        if regime == OIRegime.SQUEEZE_SETUP.value:
            if (funding_rate is not None and funding_rate < 0) or (
                liq_imbalance is not None and liq_imbalance > 0
            ):
                bullish_score += self.config.regime_weight * max(regime_conf, 0.60)
                reasons_long.append("squeeze_setup_bias_long")
            elif (funding_rate is not None and funding_rate > 0) or (
                liq_imbalance is not None and liq_imbalance < 0
            ):
                bearish_score += self.config.regime_weight * max(regime_conf, 0.60)
                reasons_short.append("squeeze_setup_bias_short")

        if regime in {OIRegime.CAPITULATION.value, OIRegime.TREND_EXHAUSTION.value, OIRegime.OVERHEATED.value}:
            if price_delta_pct is not None and price_delta_pct < 0:
                bullish_score += self.config.regime_weight * 0.55
                reasons_long.append(f"reversal_regime_after_drop:{regime}")
            elif price_delta_pct is not None and price_delta_pct > 0:
                bearish_score += self.config.regime_weight * 0.55
                reasons_short.append(f"reversal_regime_after_rally:{regime}")

        # --------------------------------------------------------------
        # Divergence scoring
        # --------------------------------------------------------------
        if divergence_type in {
            OIDivergenceType.BULLISH.value,
            OIDivergenceType.PRICE_DOWN_OI_DOWN.value,
            OIDivergenceType.PRICE_DOWN_OI_FLAT.value,
            OIDivergenceType.EXHAUSTION_DOWN.value,
        }:
            bullish_score += self.config.divergence_weight * max(divergence_conf, 0.55)
            reasons_long.append(f"bullish_divergence:{divergence_type}")

        if divergence_type in {
            OIDivergenceType.BEARISH.value,
            OIDivergenceType.PRICE_UP_OI_DOWN.value,
            OIDivergenceType.PRICE_UP_OI_FLAT.value,
            OIDivergenceType.EXHAUSTION_UP.value,
        }:
            bearish_score += self.config.divergence_weight * max(divergence_conf, 0.55)
            reasons_short.append(f"bearish_divergence:{divergence_type}")

        # --------------------------------------------------------------
        # Anomaly scoring
        # --------------------------------------------------------------
        if anomaly_type in {
            OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP.value,
            OIAnomalyType.SUDDEN_DELEVERAGING.value,
            OIAnomalyType.OI_COLLAPSE.value,
        }:
            if price_delta_pct is not None and price_delta_pct < 0:
                bullish_score += self.config.anomaly_weight * max(anomaly_conf, 0.60)
                reasons_long.append(f"flush_anomaly_reversal:{anomaly_type}")
            elif price_delta_pct is not None and price_delta_pct > 0:
                bearish_score += self.config.anomaly_weight * max(anomaly_conf, 0.60)
                reasons_short.append(f"blowoff_anomaly_reversal:{anomaly_type}")

        if anomaly_type in {
            OIAnomalyType.OVERHEATED_BUILDUP.value,
            OIAnomalyType.EXTREME_CROWDING.value,
            OIAnomalyType.FUNDING_OI_IMBALANCE.value,
        }:
            if funding_rate is not None and funding_rate > 0:
                bearish_score += self.config.anomaly_weight * max(anomaly_conf, 0.55)
                reasons_short.append(f"crowded_longs:{anomaly_type}")
            elif funding_rate is not None and funding_rate < 0:
                bullish_score += self.config.anomaly_weight * max(anomaly_conf, 0.55)
                reasons_long.append(f"crowded_shorts:{anomaly_type}")

        if anomaly_type == OIAnomalyType.OI_SPIKE.value:
            if price_delta_pct is not None and price_delta_pct > 0:
                bullish_score += self.config.anomaly_weight * 0.45
                reasons_long.append("oi_spike_supporting_up_move")
            elif price_delta_pct is not None and price_delta_pct < 0:
                bearish_score += self.config.anomaly_weight * 0.45
                reasons_short.append("oi_spike_supporting_down_move")

        # --------------------------------------------------------------
        # Pressure / flow modifiers
        # --------------------------------------------------------------
        if oi_pressure >= self.config.min_pressure_for_trend_trade:
            bullish_score += self.config.pressure_weight * min(abs(oi_pressure), 1.0)
            reasons_long.append("positive_oi_pressure")

        if oi_pressure <= -self.config.min_pressure_for_trend_trade:
            bearish_score += self.config.pressure_weight * min(abs(oi_pressure), 1.0)
            reasons_short.append("negative_oi_pressure")

        if flow_imbalance is not None and flow_imbalance > 0:
            bullish_score += self.config.flow_weight * min(abs(flow_imbalance), 1.0)
            reasons_long.append("aggressive_buy_flow")

        if flow_imbalance is not None and flow_imbalance < 0:
            bearish_score += self.config.flow_weight * min(abs(flow_imbalance), 1.0)
            reasons_short.append("aggressive_sell_flow")

        # --------------------------------------------------------------
        # Filters
        # --------------------------------------------------------------
        if volume_ratio is not None and volume_ratio < self.config.min_volume_ratio:
            bullish_score *= 0.88
            bearish_score *= 0.88
            tags.append("low_volume_penalty")

        if oi_zscore is not None and abs(oi_zscore) >= 3.5:
            tags.append("extreme_oi_zscore")

        if oi_delta_pct is not None:
            if oi_delta_pct > 0:
                tags.append("oi_expanding")
            elif oi_delta_pct < 0:
                tags.append("oi_contracting")

        bullish_score = _clamp(bullish_score)
        bearish_score = _clamp(bearish_score)

        chosen_side: str | None = None
        chosen_score = 0.0
        chosen_reasons: list[str] = []

        if bullish_score >= self.config.min_signal_confidence and bullish_score > bearish_score:
            chosen_side = "LONG"
            chosen_score = bullish_score
            chosen_reasons = reasons_long

        elif bearish_score >= self.config.min_signal_confidence and bearish_score > bullish_score:
            chosen_side = "SHORT"
            chosen_score = bearish_score
            chosen_reasons = reasons_short

        elif (
            max(bullish_score, bearish_score) >= self.config.strong_signal_confidence
            and abs(bullish_score - bearish_score) >= 0.15
        ):
            if bullish_score > bearish_score:
                chosen_side = "LONG"
                chosen_score = bullish_score
                chosen_reasons = reasons_long
            else:
                chosen_side = "SHORT"
                chosen_score = bearish_score
                chosen_reasons = reasons_short

        if chosen_side == "LONG" and not self.config.allow_longs:
            return None
        if chosen_side == "SHORT" and not self.config.allow_shorts:
            return None
        if chosen_side is None:
            return None

        metadata = {
            "trigger": trigger,
            "bullish_score": bullish_score,
            "bearish_score": bearish_score,
            "regime_confidence": regime_conf,
            "divergence_confidence": divergence_conf,
            "anomaly_confidence": anomaly_conf,
        }

        return OIStrategyDecision(
            symbol=symbol,
            exchange=exchange,
            timestamp=timestamp,
            side=chosen_side,
            confidence=chosen_score,
            score=chosen_score,
            regime=regime,
            divergence_type=str(divergence_type) if divergence_type else None,
            anomaly_type=str(anomaly_type) if anomaly_type else None,
            reasons=self._deduplicate_preserve_order(chosen_reasons),
            tags=self._deduplicate_preserve_order(tags),
            features=features,
            context=context,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Emitters
    # ------------------------------------------------------------------

    async def _emit_signal(self, decision: OIStrategyDecision, trigger: str) -> None:
        payload = decision.to_signal_payload()
        payload["metadata"]["event"] = "signal.generated"

        await self.event_bus.emit(
            "signal.generated",
            payload,
            source="oi_strategy",
        )

        self._mark_emitted(decision, trigger)

        self.logger.info(
            "Generated OI strategy signal",
            extra={
                "symbol": decision.symbol,
                "exchange": decision.exchange,
                "side": decision.side,
                "confidence": decision.confidence,
                "trigger": trigger,
                "regime": decision.regime,
                "divergence_type": decision.divergence_type,
                "anomaly_type": decision.anomaly_type,
            },
        )

    async def _emit_rejection(
        self,
        data: dict[str, Any],
        trigger: str,
        reason: str,
    ) -> None:
        payload = {
            "symbol": data["symbol"],
            "exchange": data["exchange"],
            "timestamp": data["timestamp"],
            "strategy": "oi_strategy",
            "reason": reason,
            "trigger": trigger,
            "features": _safe_dict(data.get("features")),
            "context": _safe_dict(data.get("context")),
        }

        await self.event_bus.emit(
            "signal.rejected",
            payload,
            source="oi_strategy",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_payload(self, event: Any) -> dict[str, Any]:
        payload = getattr(event, "payload", event)
        return payload if isinstance(payload, dict) else {}

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        symbol = payload.get("symbol")
        exchange = payload.get("exchange")
        timestamp = _safe_float(payload.get("timestamp"), time.time())

        if not symbol or not exchange or timestamp is None:
            return None

        symbol = str(symbol).upper().strip()
        exchange = str(exchange).lower().strip()

        normalized = {
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": timestamp,
            "features": _safe_dict(payload.get("features")),
            "context": _safe_dict(payload.get("context")),
            "regime": payload.get("regime"),
            "divergence": payload.get("divergence"),
            "anomaly": payload.get("anomaly"),
            "metadata": _safe_dict(payload.get("metadata")),
            "reasons": _safe_list(payload.get("reasons")),
            "raw": dict(payload),
        }

        # Для regime_changed / anomaly_detected / divergence_detected подій
        if isinstance(normalized["regime"], str):
            normalized["regime"] = {
                "regime": normalized["regime"],
                "confidence": _safe_float(payload.get("confidence")),
                "score": _safe_float(payload.get("score")),
                "reasons": _safe_list(payload.get("reasons")),
            }

        if normalized["divergence"] is None and payload.get("divergence_type") is not None:
            normalized["divergence"] = {
                "detected": True,
                "divergence_type": payload.get("divergence_type"),
                "confidence": _safe_float(payload.get("confidence")),
                "score": _safe_float(payload.get("score")),
                "window_size": payload.get("window_size"),
                "reasons": _safe_list(payload.get("reasons")),
            }

        if normalized["anomaly"] is None and payload.get("anomaly_type") is not None:
            normalized["anomaly"] = {
                "detected": True,
                "anomaly_type": payload.get("anomaly_type"),
                "confidence": _safe_float(payload.get("confidence")),
                "score": _safe_float(payload.get("score")),
                "reasons": _safe_list(payload.get("reasons")),
            }

        if not normalized["context"]:
            normalized["context"] = _safe_dict(payload.get("snapshot"))

        return normalized

    def _is_stale(self, data: dict[str, Any]) -> bool:
        now_ts = time.time()
        return (now_ts - data["timestamp"]) > self.config.max_signal_age_sec

    def _cooldown_passed(self, decision: OIStrategyDecision, trigger: str) -> bool:
        now_ts = time.time()
        symbol_key = (decision.exchange, decision.symbol)
        detailed_key = (decision.exchange, decision.symbol, decision.side)

        last_symbol_ts = self._last_symbol_emit_ts.get(symbol_key)
        if last_symbol_ts is not None and (now_ts - last_symbol_ts) < self.config.symbol_cooldown_sec:
            self.logger.debug(
                "OI strategy symbol cooldown active",
                extra={
                    "symbol": decision.symbol,
                    "exchange": decision.exchange,
                    "trigger": trigger,
                },
            )
            return False

        last_emit_ts = self._last_emit_ts_by_key.get(detailed_key)
        if last_emit_ts is not None and (now_ts - last_emit_ts) < self.config.cooldown_sec:
            self.logger.debug(
                "OI strategy side cooldown active",
                extra={
                    "symbol": decision.symbol,
                    "exchange": decision.exchange,
                    "side": decision.side,
                    "trigger": trigger,
                },
            )
            return False

        return True

    def _mark_emitted(self, decision: OIStrategyDecision, trigger: str) -> None:
        now_ts = time.time()
        self._last_symbol_emit_ts[(decision.exchange, decision.symbol)] = now_ts
        self._last_emit_ts_by_key[(decision.exchange, decision.symbol, decision.side)] = now_ts

    @staticmethod
    def _deduplicate_preserve_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                result.append(item)
        return result