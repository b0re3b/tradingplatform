from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.logger import get_logger

from analytics.open_interest.enums import OIDivergenceType, OIRegime
from analytics.open_interest.models import OIFeatures

from strategy.base import ContextAwareComponent, NamedEntityMixin, PrioritizedMixin
from strategy.config import StrategyConfig
from strategy.exceptions import StrategyEvaluationError
from strategy.models import SignalContext, StrategyEvaluation, StrategyMetadata, StrategySignal
from strategy.enums import (
    ConfidenceGrade,
    MarketRegime,
    SetupType,
    SignalOrigin,
    SignalPriority,
    SignalSide,
    SignalStatus,
    SignalStrength,
    StrategyCategory,
    Timeframe,
    TriggerType,
)


@dataclass(slots=True)
class OIBreakoutConfirmationPayload:
    regime: OIRegime = OIRegime.NEUTRAL
    regime_confidence: float = 0.0
    regime_score: float = 0.0

    divergence_type: OIDivergenceType = OIDivergenceType.NONE
    divergence_detected: bool = False
    divergence_confidence: float = 0.0
    divergence_score: float = 0.0

    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _normalize_reasons(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, tuple):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


class OIBreakoutConfirmationStrategy(
    ContextAwareComponent,
    NamedEntityMixin,
    PrioritizedMixin,
):
    """
    Стратегія підтвердження breakout через поведінку open interest.

    Ідея:
    - strategy НЕ шукає breakout по price action сама
    - strategy відповідає на питання:
      "чи підтверджує open interest, volume, pressure та flow цей breakout?"

    Сильні сценарії:
    - TREND_CONFIRMATION
    - LONG_BUILDUP / SHORT_BUILDUP
    - SQUEEZE_SETUP

    Слабкі / відхиляючі сценарії:
    - WEAK_BREAKOUT_UP / WEAK_BREAKOUT_DOWN
    - EXHAUSTION_UP / EXHAUSTION_DOWN
    - LONG_UNWIND / SHORT_COVERING у невідповідному напрямку
    """

    STRATEGY_NAME = "oi_breakout_confirmation_strategy"

    REQUIRED_FEATURES: set[str] = {
        "oi.regime.type",
        "oi.regime.confidence",
    }

    LONG_CONFIRMATION_REGIMES: set[OIRegime] = {
        OIRegime.LONG_BUILDUP,
        OIRegime.TREND_CONFIRMATION,
        OIRegime.SQUEEZE_SETUP,
    }

    SHORT_CONFIRMATION_REGIMES: set[OIRegime] = {
        OIRegime.SHORT_BUILDUP,
        OIRegime.TREND_CONFIRMATION,
        OIRegime.SQUEEZE_SETUP,
    }

    LONG_REJECTION_DIVERGENCES: set[OIDivergenceType] = {
        OIDivergenceType.WEAK_BREAKOUT_UP,
        OIDivergenceType.EXHAUSTION_UP,
        OIDivergenceType.PRICE_UP_OI_DOWN,
        OIDivergenceType.PRICE_UP_OI_FLAT,
        OIDivergenceType.BEARISH,
    }

    SHORT_REJECTION_DIVERGENCES: set[OIDivergenceType] = {
        OIDivergenceType.WEAK_BREAKOUT_DOWN,
        OIDivergenceType.EXHAUSTION_DOWN,
        OIDivergenceType.PRICE_DOWN_OI_DOWN,
        OIDivergenceType.PRICE_DOWN_OI_FLAT,
        OIDivergenceType.BULLISH,
    }

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: Any | None = None,
        logger: Any | None = None,
    ) -> None:
        super().__init__(
            config=config,
            event_bus=event_bus,
            logger=logger or get_logger(__name__, service_name="strategy_oi_breakout_confirmation"),
        )
        self.validate_config()

    @property
    def priority(self) -> int:
        strategy_cfg = self.config.get_strategy(self.STRATEGY_NAME)
        if strategy_cfg is not None:
            return strategy_cfg.priority
        return 80

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_name=self.STRATEGY_NAME,
            category=StrategyCategory.OPEN_INTEREST,
            timeframe=Timeframe.M1,
            tags=[
                "open_interest",
                "breakout",
                "confirmation",
                "continuation",
                "pressure",
            ],
            version="1.0.0",
            description=(
                "Підтверджує breakout/continuation через OI regime, pressure, "
                "volume, aggressive flow та divergence context."
            ),
            required_features=set(self.REQUIRED_FEATURES),
            supported_regimes={
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.BREAKOUT,
                MarketRegime.SQUEEZE,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.UNKNOWN,
            },
            metadata={
                "source": "analytics.open_interest",
                "strategy_type": "confirmation",
            },
        )

    def evaluate(self, context: SignalContext) -> StrategyEvaluation:
        try:
            self.validate_context(context)

            evaluation = StrategyEvaluation(
                strategy_name=self.STRATEGY_NAME,
                symbol=context.symbol,
                timestamp=context.timestamp,
                signal=None,
                passed=False,
                score=0.0,
                confidence=0.0,
                reasons=[],
                metadata={},
            )

            strategy_cfg = self.config.get_strategy(self.STRATEGY_NAME)
            if strategy_cfg is None:
                evaluation.reasons.append("strategy_config_not_found")
                return evaluation

            if not strategy_cfg.runtime.enabled:
                evaluation.reasons.append("strategy_disabled")
                return evaluation

            if strategy_cfg.runtime.symbols and context.symbol not in strategy_cfg.runtime.symbols:
                evaluation.reasons.append("symbol_not_enabled_for_strategy")
                return evaluation

            if strategy_cfg.runtime.timeframes and context.timeframe not in strategy_cfg.runtime.timeframes:
                evaluation.reasons.append("timeframe_not_enabled_for_strategy")
                return evaluation

            if not self._context_has_minimum_oi_data(context):
                evaluation.reasons.append("missing_open_interest_context")
                return evaluation

            if self._has_stale_required_features(context):
                evaluation.reasons.append("required_oi_features_are_stale")
                return evaluation

            payload = self._extract_payload(context)
            features = self._extract_oi_features(context)

            evaluation.metadata["oi_payload"] = payload.raw
            evaluation.metadata["oi_regime"] = payload.regime.value
            evaluation.metadata["oi_divergence_type"] = payload.divergence_type.value

            regime = context.regime.regime if context.regime is not None else MarketRegime.UNKNOWN
            if (
                strategy_cfg.runtime.allowed_regimes
                and regime not in strategy_cfg.runtime.allowed_regimes
            ):
                evaluation.reasons.append("regime_not_allowed")
                evaluation.metadata["market_regime"] = regime.value
                return evaluation

            side = self._infer_side(payload=payload, features=features)
            if side == SignalSide.UNKNOWN:
                evaluation.reasons.append("breakout_direction_not_confirmed")
                return evaluation

            blocked_reason = self._check_directional_rejection(side=side, payload=payload, features=features)
            if blocked_reason is not None:
                evaluation.reasons.append(blocked_reason)
                return evaluation

            setup_type = self._infer_setup_type(payload)
            score = self._compute_score(context=context, payload=payload, features=features, side=side)
            confidence = self._compute_confidence(context=context, payload=payload, features=features, side=side)

            evaluation.score = score
            evaluation.confidence = confidence

            if confidence < strategy_cfg.runtime.min_confidence:
                evaluation.reasons.append("confidence_below_strategy_threshold")
                evaluation.metadata["min_confidence_required"] = strategy_cfg.runtime.min_confidence
                return evaluation

            if score < strategy_cfg.runtime.min_score:
                evaluation.reasons.append("score_below_strategy_threshold")
                evaluation.metadata["min_score_required"] = strategy_cfg.runtime.min_score
                return evaluation

            signal = self._build_signal(
                context=context,
                payload=payload,
                features=features,
                side=side,
                setup_type=setup_type,
                score=score,
                confidence=confidence,
            )
            signal.validate()

            evaluation.signal = signal
            evaluation.passed = True
            evaluation.reasons.extend(payload.reasons)
            evaluation.reasons.append("oi_breakout_confirmation_signal_generated")
            return evaluation

        except Exception as exc:
            raise StrategyEvaluationError(
                f"{self.STRATEGY_NAME}: failed to evaluate context for "
                f"{getattr(context, 'symbol', 'UNKNOWN')}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Validation / extraction
    # ------------------------------------------------------------------

    def _context_has_minimum_oi_data(self, context: SignalContext) -> bool:
        if context.open_interest:
            return True

        keys = (
            "oi.regime.type",
            "oi.regime.confidence",
            "oi.features.oi_delta_pct",
            "oi.features.price_delta_pct",
        )
        return any(context.has_feature(name) for name in keys)

    def _has_stale_required_features(self, context: SignalContext) -> bool:
        for name in self.REQUIRED_FEATURES:
            if context.has_feature(name) and context.feature_is_stale(name):
                return True
        return False

    def _extract_payload(self, context: SignalContext) -> OIBreakoutConfirmationPayload:
        raw: dict[str, Any] = {}

        domain = context.open_interest or {}

        regime_section = domain.get("regime")
        if isinstance(regime_section, dict):
            raw.update({f"regime_{k}": v for k, v in regime_section.items()})

        divergence_section = domain.get("divergence")
        if isinstance(divergence_section, dict):
            raw.update({f"divergence_{k}": v for k, v in divergence_section.items()})

        aliases = {
            "regime": "regime_type",
            "regime_type": "regime_type",
            "oi_regime": "regime_type",
            "oi_regime_type": "regime_type",
            "regime_confidence": "regime_confidence",
            "oi_regime_confidence": "regime_confidence",
            "regime_score": "regime_score",
            "oi_regime_score": "regime_score",
            "reasons": "reasons",
            "regime_reasons": "reasons",
            "divergence_type": "divergence_type",
            "oi_divergence_type": "divergence_type",
            "divergence_detected": "divergence_detected",
            "oi_divergence_detected": "divergence_detected",
            "divergence_confidence": "divergence_confidence",
            "oi_divergence_confidence": "divergence_confidence",
            "divergence_score": "divergence_score",
            "oi_divergence_score": "divergence_score",
        }

        for src, dst in aliases.items():
            if src in domain and dst not in raw:
                raw[dst] = domain[src]

        feature_aliases = {
            "oi.regime.type": "regime_type",
            "oi.regime.confidence": "regime_confidence",
            "oi.regime.score": "regime_score",
            "oi.regime.reasons": "reasons",
            "oi.divergence.type": "divergence_type",
            "oi.divergence.detected": "divergence_detected",
            "oi.divergence.confidence": "divergence_confidence",
            "oi.divergence.score": "divergence_score",
        }

        for feature_name, dst_key in feature_aliases.items():
            if dst_key not in raw and context.has_feature(feature_name):
                raw[dst_key] = context.get_feature(feature_name)

        regime = self._parse_oi_regime(raw.get("regime_type") or raw.get("regime_regime"))
        regime_confidence = _clamp(_safe_float(raw.get("regime_confidence"), 0.0))
        regime_score = _clamp(_safe_float(raw.get("regime_score"), regime_confidence))

        divergence_type = self._parse_divergence_type(raw.get("divergence_type"))
        divergence_detected = _safe_bool(
            raw.get("divergence_detected"),
            default=divergence_type != OIDivergenceType.NONE,
        )
        divergence_confidence = _clamp(_safe_float(raw.get("divergence_confidence"), 0.0))
        divergence_score = _clamp(_safe_float(raw.get("divergence_score"), divergence_confidence))

        reasons = _normalize_reasons(raw.get("reasons") or raw.get("regime_reasons"))

        return OIBreakoutConfirmationPayload(
            regime=regime,
            regime_confidence=regime_confidence,
            regime_score=regime_score,
            divergence_type=divergence_type,
            divergence_detected=divergence_detected,
            divergence_confidence=divergence_confidence,
            divergence_score=divergence_score,
            reasons=reasons,
            raw=raw,
        )

    def _extract_oi_features(self, context: SignalContext) -> OIFeatures | None:
        raw = context.open_interest.get("features") if context.open_interest else None
        if isinstance(raw, OIFeatures):
            return raw

        if not isinstance(raw, dict):
            return None

        try:
            return OIFeatures(**raw)
        except Exception:
            return None

    def _parse_oi_regime(self, value: Any) -> OIRegime:
        if isinstance(value, OIRegime):
            return value

        text = _safe_str(value).upper()
        if not text:
            return OIRegime.NEUTRAL

        for item in OIRegime:
            if item.value.upper() == text:
                return item

        return OIRegime.NEUTRAL

    def _parse_divergence_type(self, value: Any) -> OIDivergenceType:
        if isinstance(value, OIDivergenceType):
            return value

        text = _safe_str(value).upper()
        if not text:
            return OIDivergenceType.NONE

        for item in OIDivergenceType:
            if item.value.upper() == text:
                return item

        return OIDivergenceType.NONE

    # ------------------------------------------------------------------
    # Direction / setup
    # ------------------------------------------------------------------

    def _infer_side(
        self,
        *,
        payload: OIBreakoutConfirmationPayload,
        features: OIFeatures | None,
    ) -> SignalSide:
        if payload.regime == OIRegime.LONG_BUILDUP:
            return SignalSide.LONG

        if payload.regime == OIRegime.SHORT_BUILDUP:
            return SignalSide.SHORT

        if payload.regime == OIRegime.TREND_CONFIRMATION:
            if features is not None and features.price_delta_pct is not None:
                if features.price_delta_pct > 0:
                    return SignalSide.LONG
                if features.price_delta_pct < 0:
                    return SignalSide.SHORT

        if payload.regime == OIRegime.SQUEEZE_SETUP:
            if features is not None:
                if features.funding_rate is not None and features.funding_rate < 0:
                    return SignalSide.LONG
                if features.funding_rate is not None and features.funding_rate > 0:
                    return SignalSide.SHORT
                if features.price_delta_pct is not None:
                    if features.price_delta_pct > 0:
                        return SignalSide.LONG
                    if features.price_delta_pct < 0:
                        return SignalSide.SHORT

        if features is not None and features.price_delta_pct is not None:
            if features.price_delta_pct > 0 and self._positive_breakout_support(features):
                return SignalSide.LONG
            if features.price_delta_pct < 0 and self._negative_breakout_support(features):
                return SignalSide.SHORT

        return SignalSide.UNKNOWN

    def _infer_setup_type(self, payload: OIBreakoutConfirmationPayload) -> SetupType:
        if payload.regime == OIRegime.SQUEEZE_SETUP:
            return SetupType.SQUEEZE
        if payload.regime in {OIRegime.LONG_BUILDUP, OIRegime.SHORT_BUILDUP, OIRegime.TREND_CONFIRMATION}:
            return SetupType.CONTINUATION
        return SetupType.BREAKOUT

    def _check_directional_rejection(
        self,
        *,
        side: SignalSide,
        payload: OIBreakoutConfirmationPayload,
        features: OIFeatures | None,
    ) -> str | None:
        if side == SignalSide.LONG and payload.divergence_type in self.LONG_REJECTION_DIVERGENCES:
            return f"long_breakout_rejected_by_divergence:{payload.divergence_type.value}"

        if side == SignalSide.SHORT and payload.divergence_type in self.SHORT_REJECTION_DIVERGENCES:
            return f"short_breakout_rejected_by_divergence:{payload.divergence_type.value}"

        if features is None:
            return None

        if side == SignalSide.LONG:
            if features.oi_delta_pct <= 0:
                return "long_breakout_not_supported_by_oi_growth"
            if features.oi_price_efficiency is not None and features.oi_price_efficiency <= 0:
                return "long_breakout_not_supported_by_oi_price_efficiency"
            if features.oi_pressure_score is not None and features.oi_pressure_score < -0.10:
                return "long_breakout_negative_pressure"

        if side == SignalSide.SHORT:
            if features.oi_delta_pct <= 0:
                return "short_breakout_not_supported_by_oi_growth"
            if features.oi_price_efficiency is not None and features.oi_price_efficiency <= 0:
                return "short_breakout_not_supported_by_oi_price_efficiency"
            if features.oi_pressure_score is not None and features.oi_pressure_score > 0.10:
                return "short_breakout_positive_pressure"

        return None

    # ------------------------------------------------------------------
    # Feature support
    # ------------------------------------------------------------------

    def _positive_breakout_support(self, features: OIFeatures) -> bool:
        return (
            features.oi_delta_pct > 0
            and (features.volume_ratio is None or features.volume_ratio >= 1.0)
            and (features.oi_pressure_score is None or features.oi_pressure_score >= 0.15)
        )

    def _negative_breakout_support(self, features: OIFeatures) -> bool:
        return (
            features.oi_delta_pct > 0
            and (features.volume_ratio is None or features.volume_ratio >= 1.0)
            and (features.oi_pressure_score is None or features.oi_pressure_score <= -0.15)
        )

    # ------------------------------------------------------------------
    # Scoring / confidence
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        *,
        context: SignalContext,
        payload: OIBreakoutConfirmationPayload,
        features: OIFeatures | None,
        side: SignalSide,
    ) -> float:
        category_weight = self.config.weighting.category_weights.get(
            StrategyCategory.OPEN_INTEREST,
            1.0,
        )
        regime = context.regime.regime if context.regime else MarketRegime.UNKNOWN
        regime_adj = self.config.weighting.regime_adjustments.get(regime, 1.0)

        strategy_cfg = self.config.get_strategy(self.STRATEGY_NAME)
        strategy_weight = strategy_cfg.weight if strategy_cfg is not None else 1.0

        base_score = payload.regime_score if payload.regime_score > 0 else payload.regime_confidence
        score = base_score * category_weight * regime_adj * strategy_weight

        if features is not None:
            if features.volume_ratio is not None and features.volume_ratio >= 1.0:
                score += 0.06
            if features.oi_delta_pct >= 0.25:
                score += 0.06
            if features.oi_zscore is not None and abs(features.oi_zscore) >= 1.0:
                score += 0.04
            if features.oi_price_efficiency is not None and features.oi_price_efficiency > 0.5:
                score += 0.05

            if side == SignalSide.LONG:
                if features.oi_pressure_score is not None and features.oi_pressure_score >= 0.25:
                    score += 0.06
                if (
                    features.aggressive_flow_imbalance is not None
                    and features.aggressive_flow_imbalance >= 0.10
                ):
                    score += 0.04
                if features.funding_rate is not None and 0 <= features.funding_rate < 0.01:
                    score += 0.03

            if side == SignalSide.SHORT:
                if features.oi_pressure_score is not None and features.oi_pressure_score <= -0.25:
                    score += 0.06
                if (
                    features.aggressive_flow_imbalance is not None
                    and features.aggressive_flow_imbalance <= -0.10
                ):
                    score += 0.04
                if features.funding_rate is not None and -0.01 < features.funding_rate <= 0:
                    score += 0.03

        if payload.divergence_detected:
            score -= min(0.15, payload.divergence_score * 0.20)

        return _clamp(score)

    def _compute_confidence(
        self,
        *,
        context: SignalContext,
        payload: OIBreakoutConfirmationPayload,
        features: OIFeatures | None,
        side: SignalSide,
    ) -> float:
        confidence = payload.regime_confidence if payload.regime_confidence > 0 else payload.regime_score

        if features is not None:
            if features.volume_ratio is not None and features.volume_ratio >= 1.0:
                confidence += 0.04
            if features.oi_delta_pct >= 0.25:
                confidence += 0.04
            if features.oi_ma_fast is not None and features.oi_ma_slow is not None:
                if features.oi_ma_fast > features.oi_ma_slow:
                    confidence += 0.03
            if features.oi_price_efficiency is not None and features.oi_price_efficiency > 0.5:
                confidence += 0.03

            if side == SignalSide.LONG:
                if features.oi_pressure_score is not None and features.oi_pressure_score >= 0.25:
                    confidence += 0.04
            elif side == SignalSide.SHORT:
                if features.oi_pressure_score is not None and features.oi_pressure_score <= -0.25:
                    confidence += 0.04

        if payload.divergence_detected:
            confidence -= min(0.12, payload.divergence_confidence * 0.20)

        return _clamp(confidence)

    def _confidence_grade(self, confidence: float) -> ConfidenceGrade:
        cfg = self.config.confidence
        if confidence >= cfg.high_threshold:
            return ConfidenceGrade.VERY_HIGH
        if confidence >= cfg.medium_threshold:
            return ConfidenceGrade.HIGH
        if confidence >= cfg.low_threshold:
            return ConfidenceGrade.MEDIUM
        if confidence >= cfg.very_low_threshold:
            return ConfidenceGrade.LOW
        return ConfidenceGrade.VERY_LOW

    def _strength_from_score(self, score: float) -> SignalStrength:
        if score >= 0.90:
            return SignalStrength.EXTREME
        if score >= 0.75:
            return SignalStrength.STRONG
        if score >= 0.55:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK

    def _priority_from_score(self, score: float) -> SignalPriority:
        if score >= 0.90:
            return SignalPriority.CRITICAL
        if score >= 0.75:
            return SignalPriority.HIGH
        if score >= 0.50:
            return SignalPriority.MEDIUM
        return SignalPriority.LOW

    # ------------------------------------------------------------------
    # Signal building
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        *,
        context: SignalContext,
        payload: OIBreakoutConfirmationPayload,
        features: OIFeatures | None,
        side: SignalSide,
        setup_type: SetupType,
        score: float,
        confidence: float,
    ) -> StrategySignal:
        signal = StrategySignal(
            symbol=context.symbol,
            side=side,
            strategy_name=self.STRATEGY_NAME,
            category=StrategyCategory.OPEN_INTEREST,
            timeframe=context.timeframe,
            setup_type=setup_type,
            timestamp=context.timestamp,
            confidence=confidence,
            score=score,
            strength=self._strength_from_score(score),
            confidence_grade=self._confidence_grade(confidence),
            status=SignalStatus.NEW,
            trigger_type=TriggerType.CONFIRMATION,
            origin=SignalOrigin.SINGLE_STRATEGY,
            priority=self._priority_from_score(score),
            regime=context.regime.regime if context.regime else MarketRegime.UNKNOWN,
            metadata={
                "oi_regime": payload.regime.value,
                "oi_regime_confidence": payload.regime_confidence,
                "oi_regime_score": payload.regime_score,
                "oi_divergence_type": payload.divergence_type.value,
                "oi_divergence_detected": payload.divergence_detected,
                "oi_raw_payload": payload.raw,
            },
        )

        signal.add_reason(f"oi_regime:{payload.regime.value}")
        signal.add_reason(f"setup_type:{setup_type.value}")

        for reason in payload.reasons:
            signal.add_reason(reason)

        for feature_name in self.REQUIRED_FEATURES:
            if context.has_feature(feature_name):
                signal.add_source_feature(feature_name)

        if features is not None:
            self._append_feature_metadata(signal=signal, features=features)
            self._append_feature_reasons(signal=signal, features=features)
            for confirmation in self._build_confirmations(side=side, features=features, payload=payload):
                signal.add_confirmation(confirmation)

        return signal

    def _append_feature_reasons(
        self,
        *,
        signal: StrategySignal,
        features: OIFeatures,
    ) -> None:
        if features.oi_delta_pct is not None:
            signal.add_reason(f"oi_delta_pct:{features.oi_delta_pct:.4f}")
        if features.price_delta_pct is not None:
            signal.add_reason(f"price_delta_pct:{features.price_delta_pct:.4f}")
        if features.volume_ratio is not None:
            signal.add_reason(f"volume_ratio:{features.volume_ratio:.4f}")
        if features.oi_pressure_score is not None:
            signal.add_reason(f"oi_pressure_score:{features.oi_pressure_score:.4f}")
        if features.oi_price_efficiency is not None:
            signal.add_reason(f"oi_price_efficiency:{features.oi_price_efficiency:.4f}")

    def _append_feature_metadata(
        self,
        *,
        signal: StrategySignal,
        features: OIFeatures,
    ) -> None:
        signal.metadata.update(
            {
                "oi": features.oi,
                "oi_delta": features.oi_delta,
                "oi_delta_pct": features.oi_delta_pct,
                "oi_ma_fast": features.oi_ma_fast,
                "oi_ma_slow": features.oi_ma_slow,
                "oi_zscore": features.oi_zscore,
                "oi_velocity": features.oi_velocity,
                "oi_acceleration": features.oi_acceleration,
                "price": features.price,
                "price_delta": features.price_delta,
                "price_delta_pct": features.price_delta_pct,
                "volume": features.volume,
                "volume_ratio": features.volume_ratio,
                "funding_rate": features.funding_rate,
                "liquidation_imbalance": features.liquidation_imbalance,
                "aggressive_flow_imbalance": features.aggressive_flow_imbalance,
                "oi_change_per_volume": features.oi_change_per_volume,
                "oi_price_efficiency": features.oi_price_efficiency,
                "oi_pressure_score": features.oi_pressure_score,
                "oi_direction": getattr(features.oi_direction, "value", None),
                "price_direction": getattr(features.price_direction, "value", None),
            }
        )

    def _build_confirmations(
        self,
        *,
        side: SignalSide,
        features: OIFeatures,
        payload: OIBreakoutConfirmationPayload,
    ) -> list[str]:
        confirmations: list[str] = [f"regime:{payload.regime.value}"]

        if features.volume_ratio is not None and features.volume_ratio >= 1.0:
            confirmations.append("volume_confirmation")

        if features.oi_delta_pct >= 0.25:
            confirmations.append("oi_build_confirmation")

        if features.oi_ma_fast is not None and features.oi_ma_slow is not None:
            if features.oi_ma_fast > features.oi_ma_slow:
                confirmations.append("fast_oi_above_slow_oi")

        if features.oi_price_efficiency is not None and features.oi_price_efficiency > 0.5:
            confirmations.append("price_move_supported_by_oi")

        if side == SignalSide.LONG:
            if features.oi_pressure_score is not None and features.oi_pressure_score >= 0.25:
                confirmations.append("positive_pressure_confirmation")
            if (
                features.aggressive_flow_imbalance is not None
                and features.aggressive_flow_imbalance >= 0.10
            ):
                confirmations.append("aggressive_buy_confirmation")

        if side == SignalSide.SHORT:
            if features.oi_pressure_score is not None and features.oi_pressure_score <= -0.25:
                confirmations.append("negative_pressure_confirmation")
            if (
                features.aggressive_flow_imbalance is not None
                and features.aggressive_flow_imbalance <= -0.10
            ):
                confirmations.append("aggressive_sell_confirmation")

        return confirmations