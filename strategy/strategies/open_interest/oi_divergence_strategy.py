from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from analytics.open_interest.enums import OIDivergenceType
from analytics.open_interest.models import OIFeatures
from core.logger import get_logger
from strategy.base import ContextAwareComponent, NamedEntityMixin, PrioritizedMixin
from strategy.config import StrategyConfig
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
from strategy.exceptions import StrategyEvaluationError
from strategy.models import (
    SignalContext,
    StrategyEvaluation,
    StrategyMetadata,
    StrategySignal,
)


@dataclass(slots=True)
class OIDivergencePayload:
    divergence_type: OIDivergenceType = OIDivergenceType.NONE
    detected: bool = False
    confidence: float = 0.0
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    window_size: int | None = None
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


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


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


class OIDivergenceStrategy(
    ContextAwareComponent,
    NamedEntityMixin,
    PrioritizedMixin,
):
    """
    Strategy, що інтерпретує divergence сигнали з analytics/open_interest
    і перетворює їх у StrategySignal strategy layer.

    Основна ідея:
    - беремо з SignalContext already-normalized OI divergence data
    - валідуємо freshness / confidence / regime / symbol availability
    - мапимо divergence_type -> SignalSide + SetupType
    - формуємо StrategyEvaluation і StrategySignal

    Strategy не рахує divergence самостійно.
    Вона використовує результат analytics/open_interest.
    """

    STRATEGY_NAME = "oi_divergence_strategy"

    REQUIRED_FEATURES: set[str] = {
        "oi.divergence.type",
        "oi.divergence.detected",
        "oi.divergence.confidence",
    }

    BULLISH_DIVERGENCES: set[OIDivergenceType] = {
        OIDivergenceType.BULLISH,
        OIDivergenceType.EXHAUSTION_DOWN,
    }

    BEARISH_DIVERGENCES: set[OIDivergenceType] = {
        OIDivergenceType.BEARISH,
        OIDivergenceType.EXHAUSTION_UP,
        OIDivergenceType.PRICE_UP_OI_DOWN,
        OIDivergenceType.PRICE_UP_OI_FLAT,
        OIDivergenceType.WEAK_BREAKOUT_UP,
    }

    NEUTRAL_OR_CONTEXTUAL_DIVERGENCES: set[OIDivergenceType] = {
        OIDivergenceType.PRICE_DOWN_OI_DOWN,
        OIDivergenceType.PRICE_DOWN_OI_FLAT,
        OIDivergenceType.WEAK_BREAKOUT_DOWN,
        OIDivergenceType.NONE,
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
            logger=logger or get_logger(__name__, service_name="strategy_oi_divergence"),
        )
        self.validate_config()

    @property
    def priority(self) -> int:
        strategy_cfg = self.config.get_strategy(self.STRATEGY_NAME)
        if strategy_cfg is not None:
            return strategy_cfg.priority
        return 85

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_name=self.STRATEGY_NAME,
            category=StrategyCategory.OPEN_INTEREST,
            timeframe=Timeframe.M1,
            tags=[
                "open_interest",
                "divergence",
                "reversal",
                "exhaustion",
                "contextual",
            ],
            version="1.0.0",
            description=(
                "Інтерпретує divergence між price та open interest і будує "
                "directional strategy signals."
            ),
            required_features=set(self.REQUIRED_FEATURES),
            supported_regimes={
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.RANGING,
                MarketRegime.BREAKOUT,
                MarketRegime.SQUEEZE,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.LOW_VOLATILITY,
                MarketRegime.UNKNOWN,
            },
            metadata={
                "source": "analytics.open_interest",
                "strategy_type": "single_factor_contextual",
            },
        )

    def evaluate(self, context: SignalContext) -> StrategyEvaluation:
        """
        Головна точка входу для strategy engine.
        Повертає StrategyEvaluation, який може містити signal або бути rejected.
        """
        try:
            self.validate_context(context)

            evaluation = StrategyEvaluation(
                strategy_name=self.STRATEGY_NAME,
                symbol=context.symbol,
                timestamp=context.timestamp,
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

            payload = self._extract_divergence_payload(context)
            evaluation.metadata["divergence"] = payload.raw

            if not payload.detected:
                evaluation.reasons.append("divergence_not_detected")
                evaluation.score = payload.score
                evaluation.confidence = payload.confidence
                return evaluation

            if payload.divergence_type == OIDivergenceType.NONE:
                evaluation.reasons.append("divergence_type_none")
                evaluation.score = payload.score
                evaluation.confidence = payload.confidence
                return evaluation

            regime = context.regime.regime if context.regime is not None else MarketRegime.UNKNOWN
            if (
                strategy_cfg.runtime.allowed_regimes
                and regime not in strategy_cfg.runtime.allowed_regimes
            ):
                evaluation.reasons.append("regime_not_allowed")
                evaluation.metadata["regime"] = regime.value
                return evaluation

            side = self._map_divergence_to_side(payload.divergence_type)
            if side == SignalSide.UNKNOWN:
                evaluation.reasons.append("divergence_not_directional")
                evaluation.score = payload.score
                evaluation.confidence = payload.confidence
                return evaluation

            setup_type = self._map_divergence_to_setup_type(payload.divergence_type)
            score = self._compute_score(context=context, payload=payload)
            confidence = self._compute_confidence(context=context, payload=payload)

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
                side=side,
                setup_type=setup_type,
                score=score,
                confidence=confidence,
            )

            signal.validate()

            evaluation.signal = signal
            evaluation.passed = True
            evaluation.reasons.extend(payload.reasons or [])
            evaluation.reasons.append("oi_divergence_signal_generated")
            evaluation.metadata.update(
                {
                    "divergence_type": payload.divergence_type.value,
                    "signal_side": side.value,
                    "setup_type": setup_type.value,
                    "regime": regime.value,
                }
            )
            return evaluation

        except Exception as exc:
            raise StrategyEvaluationError(
                f"{self.STRATEGY_NAME}: failed to evaluate context for {getattr(context, 'symbol', 'UNKNOWN')}: {exc}"
            ) from exc

    # ---------------------------------------------------------------------
    # Extraction / validation
    # ---------------------------------------------------------------------

    def _context_has_minimum_oi_data(self, context: SignalContext) -> bool:
        if context.open_interest:
            return True

        keys = (
            "oi.divergence.type",
            "oi.divergence.detected",
            "oi.divergence.confidence",
        )
        return any(context.has_feature(name) for name in keys)

    def _has_stale_required_features(self, context: SignalContext) -> bool:
        for name in self.REQUIRED_FEATURES:
            if context.has_feature(name) and context.feature_is_stale(name):
                return True
        return False

    def _extract_divergence_payload(self, context: SignalContext) -> OIDivergencePayload:
        """
        Підтримує кілька форматів:
        1) context.open_interest["divergence"] = {...}
        2) flat keys у context.open_interest
        3) feature_map keys типу oi.divergence.*
        """
        raw: dict[str, Any] = {}

        domain = context.open_interest or {}
        nested = domain.get("divergence")
        if isinstance(nested, dict):
            raw.update(nested)

        for key in (
            "divergence_type",
            "detected",
            "confidence",
            "score",
            "reasons",
            "window_size",
        ):
            if key in domain and key not in raw:
                raw[key] = domain[key]

        aliases = {
            "type": "divergence_type",
            "oi_divergence_type": "divergence_type",
            "open_interest_divergence_type": "divergence_type",
            "is_detected": "detected",
            "oi_divergence_detected": "detected",
            "oi_divergence_confidence": "confidence",
            "oi_divergence_score": "score",
            "oi_divergence_reasons": "reasons",
        }
        for src_key, dst_key in aliases.items():
            if src_key in domain and dst_key not in raw:
                raw[dst_key] = domain[src_key]

        feature_aliases = {
            "oi.divergence.type": "divergence_type",
            "oi.divergence.detected": "detected",
            "oi.divergence.confidence": "confidence",
            "oi.divergence.score": "score",
            "oi.divergence.reasons": "reasons",
            "oi.divergence.window_size": "window_size",
            "open_interest.divergence.type": "divergence_type",
            "open_interest.divergence.detected": "detected",
            "open_interest.divergence.confidence": "confidence",
            "open_interest.divergence.score": "score",
            "open_interest.divergence.reasons": "reasons",
            "open_interest.divergence.window_size": "window_size",
        }

        for feature_name, dst_key in feature_aliases.items():
            if dst_key not in raw and context.has_feature(feature_name):
                raw[dst_key] = context.get_feature(feature_name)

        divergence_type = self._parse_divergence_type(raw.get("divergence_type"))
        detected = bool(raw.get("detected", divergence_type != OIDivergenceType.NONE))
        confidence = _clamp(_safe_float(raw.get("confidence"), 0.0))
        score = _clamp(_safe_float(raw.get("score"), confidence))
        reasons = _normalize_reasons(raw.get("reasons"))
        window_size = _safe_int(raw.get("window_size"))

        return OIDivergencePayload(
            divergence_type=divergence_type,
            detected=detected,
            confidence=confidence,
            score=score,
            reasons=reasons,
            window_size=window_size,
            raw=raw,
        )

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

    # ---------------------------------------------------------------------
    # Mapping
    # ---------------------------------------------------------------------

    def _map_divergence_to_side(self, divergence_type: OIDivergenceType) -> SignalSide:
        if divergence_type in self.BULLISH_DIVERGENCES:
            return SignalSide.LONG
        if divergence_type in self.BEARISH_DIVERGENCES:
            return SignalSide.SHORT

        if divergence_type == OIDivergenceType.PRICE_DOWN_OI_FLAT:
            return SignalSide.LONG
        if divergence_type == OIDivergenceType.WEAK_BREAKOUT_DOWN:
            return SignalSide.LONG
        if divergence_type == OIDivergenceType.PRICE_DOWN_OI_DOWN:
            return SignalSide.LONG

        return SignalSide.UNKNOWN

    def _map_divergence_to_setup_type(self, divergence_type: OIDivergenceType) -> SetupType:
        if divergence_type in {
            OIDivergenceType.BULLISH,
            OIDivergenceType.BEARISH,
        }:
            return SetupType.REVERSAL

        if divergence_type in {
            OIDivergenceType.EXHAUSTION_UP,
            OIDivergenceType.EXHAUSTION_DOWN,
        }:
            return SetupType.EXHAUSTION

        if divergence_type in {
            OIDivergenceType.WEAK_BREAKOUT_UP,
            OIDivergenceType.WEAK_BREAKOUT_DOWN,
        }:
            return SetupType.BREAKOUT

        if divergence_type in {
            OIDivergenceType.PRICE_UP_OI_DOWN,
            OIDivergenceType.PRICE_UP_OI_FLAT,
            OIDivergenceType.PRICE_DOWN_OI_DOWN,
            OIDivergenceType.PRICE_DOWN_OI_FLAT,
        }:
            return SetupType.MEAN_REVERSION

        return SetupType.UNKNOWN

    # ---------------------------------------------------------------------
    # Scoring / confidence
    # ---------------------------------------------------------------------

    def _compute_score(self, context: SignalContext, payload: OIDivergencePayload) -> float:
        strategy_cfg = self.config.get_strategy(self.STRATEGY_NAME)
        category_weight = self.config.get_category_weight(StrategyCategory.OPEN_INTEREST)
        strategy_weight = self.config.get_strategy_weight(self.STRATEGY_NAME, default=1.0)
        regime = context.regime.regime if context.regime else MarketRegime.UNKNOWN
        regime_adj = self.config.get_regime_adjustment(regime)

        base_score = payload.score if payload.score > 0 else payload.confidence
        score = base_score * category_weight * strategy_weight * regime_adj

        features = self._extract_oi_features(context)

        if features is not None:
            if features.volume_ratio is not None and features.volume_ratio >= 1.0:
                score += 0.05
            if features.oi_price_efficiency is not None and abs(features.oi_price_efficiency) < 0.50:
                score += 0.05
            if features.oi_pressure_score is not None and abs(features.oi_pressure_score) < 0.20:
                score += 0.04
            if features.liquidation_imbalance is not None and abs(features.liquidation_imbalance) >= 0.20:
                score += 0.03
            if features.aggressive_flow_imbalance is not None and abs(features.aggressive_flow_imbalance) < 0.10:
                score += 0.03

        if strategy_cfg is not None and payload.window_size is not None and payload.window_size >= 5:
            score += 0.02

        return _clamp(score)

    def _compute_confidence(self, context: SignalContext, payload: OIDivergencePayload) -> float:
        confidence = payload.confidence if payload.confidence > 0 else payload.score

        features = self._extract_oi_features(context)
        if features is not None:
            if features.volume_ratio is not None and features.volume_ratio >= 1.0:
                confidence += 0.04
            if features.oi_zscore is not None and abs(features.oi_zscore) >= 1.0:
                confidence += 0.03
            if features.oi_price_efficiency is not None and abs(features.oi_price_efficiency) < 0.50:
                confidence += 0.03

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

    # ---------------------------------------------------------------------
    # Signal building
    # ---------------------------------------------------------------------

    def _build_signal(
        self,
        *,
        context: SignalContext,
        payload: OIDivergencePayload,
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
            trigger_type=TriggerType.PRIMARY,
            origin=SignalOrigin.SINGLE_STRATEGY,
            priority=self._priority_from_score(score),
            regime=context.regime.regime if context.regime else MarketRegime.UNKNOWN,
            metadata={
                "oi_divergence_type": payload.divergence_type.value,
                "oi_divergence_window_size": payload.window_size,
                "oi_raw_payload": payload.raw,
            },
        )

        for reason in payload.reasons:
            signal.add_reason(reason)

        signal.add_reason(f"oi_divergence_type:{payload.divergence_type.value}")
        signal.add_reason(f"setup_type:{setup_type.value}")

        for feature_name in self.REQUIRED_FEATURES:
            if context.has_feature(feature_name):
                signal.add_source_feature(feature_name)

        features = self._extract_oi_features(context)
        if features is not None:
            self._append_feature_reasons(signal=signal, features=features)
            self._append_feature_metadata(signal=signal, features=features)

        confirmations = self._build_confirmations(
            side=side,
            divergence_type=payload.divergence_type,
            features=features,
        )
        for confirmation in confirmations:
            signal.add_confirmation(confirmation)

        return signal

    def _append_feature_reasons(
        self,
        *,
        signal: StrategySignal,
        features: OIFeatures,
    ) -> None:
        if features.volume_ratio is not None:
            signal.add_reason(f"volume_ratio:{features.volume_ratio:.4f}")
        if features.oi_delta_pct is not None:
            signal.add_reason(f"oi_delta_pct:{features.oi_delta_pct:.4f}")
        if features.price_delta_pct is not None:
            signal.add_reason(f"price_delta_pct:{features.price_delta_pct:.4f}")
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
                "oi_zscore": features.oi_zscore,
                "oi_velocity": features.oi_velocity,
                "oi_acceleration": features.oi_acceleration,
                "price": features.price,
                "price_delta_pct": features.price_delta_pct,
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
        divergence_type: OIDivergenceType,
        features: OIFeatures | None,
    ) -> list[str]:
        confirmations: list[str] = [f"divergence:{divergence_type.value}"]

        if features is None:
            return confirmations

        if side == SignalSide.LONG:
            if features.liquidation_imbalance is not None and features.liquidation_imbalance <= -0.20:
                confirmations.append("long_flush_context")
            if features.aggressive_flow_imbalance is not None and features.aggressive_flow_imbalance > -0.05:
                confirmations.append("sell_aggression_fading")
            if features.funding_rate is not None and features.funding_rate < 0:
                confirmations.append("negative_funding_context")

        if side == SignalSide.SHORT:
            if features.aggressive_flow_imbalance is not None and features.aggressive_flow_imbalance < 0.05:
                confirmations.append("buy_aggression_fading")
            if features.funding_rate is not None and features.funding_rate > 0:
                confirmations.append("positive_funding_context")
            if features.oi_price_efficiency is not None and abs(features.oi_price_efficiency) < 0.50:
                confirmations.append("price_move_not_supported_by_oi")

        if features.volume_ratio is not None and features.volume_ratio >= 1.0:
            confirmations.append("volume_context_present")

        return confirmations

    # ---------------------------------------------------------------------
    # Feature extraction
    # ---------------------------------------------------------------------

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