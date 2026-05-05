from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from analytics.open_interest.enums import OIDivergenceType, OIRegime
from analytics.open_interest.models import OIFeatures
from strategy.config import StrategyConfig
from strategy.enums import (
    MarketRegime,
    SetupType,
    SignalOrigin,
    SignalSide,
    SignalStatus,
    StrategyCategory,
    Timeframe,
    TriggerType,
)
from strategy.exceptions import StrategyEvaluationError
from strategy.models import SignalContext, StrategyEvaluation, StrategyMetadata, StrategySignal

from .base import OpenInterestStrategyBase


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


class OIBreakoutConfirmationStrategy(OpenInterestStrategyBase):
    """
    Стратегія підтвердження breakout через поведінку open interest.

    Вона не шукає breakout самостійно, а перевіряє, чи підтверджує
    open interest, volume, pressure та aggressive flow вже наявний breakout context.
    """

    STRATEGY_NAME = "oi_breakout_confirmation_strategy"
    DEFAULT_PRIORITY = 80

    REQUIRED_FEATURES: set[str] = {
        "oi.regime.type",
        "oi.regime.confidence",
    }

    MINIMUM_OI_CONTEXT_KEYS: tuple[str, ...] = (
        "oi.regime.type",
        "oi.regime.confidence",
        "oi.features.oi_delta_pct",
        "oi.features.price_delta_pct",
    )

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
    ) -> None:
        super().__init__(config=config, event_bus=event_bus)

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
            version="1.1.0",
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
                "base_class": "OpenInterestStrategyBase",
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

            runtime_allowed, runtime_reason = self.is_strategy_runtime_allowed(context)
            if not runtime_allowed:
                evaluation.reasons.append(runtime_reason or "strategy_runtime_not_allowed")
                return evaluation

            if strategy_cfg is None:
                evaluation.reasons.append("strategy_config_not_found")
                return evaluation

            if not self.context_has_any_oi_data(context, self.MINIMUM_OI_CONTEXT_KEYS):
                evaluation.reasons.append("missing_open_interest_context")
                return evaluation

            if self.has_stale_required_features(context):
                evaluation.reasons.append("required_oi_features_are_stale")
                return evaluation

            payload = self._extract_payload(context)
            features = self.extract_oi_features(context)
            regime = self.get_market_regime(context)

            evaluation.metadata.update(
                {
                    "oi_payload": payload.raw,
                    "oi_regime": payload.regime.value,
                    "oi_divergence_type": payload.divergence_type.value,
                    "market_regime": regime.value,
                }
            )

            regime_allowed, regime_reason = self.is_market_regime_allowed(context)
            if not regime_allowed:
                evaluation.reasons.append(regime_reason or "regime_not_allowed")
                return evaluation

            side = self._infer_side(payload=payload, features=features)
            if side == SignalSide.UNKNOWN:
                evaluation.reasons.append("breakout_direction_not_confirmed")
                return evaluation

            blocked_reason = self._check_directional_rejection(
                side=side,
                payload=payload,
                features=features,
            )
            if blocked_reason is not None:
                evaluation.reasons.append(blocked_reason)
                return evaluation

            setup_type = self._infer_setup_type(payload)
            score = self._compute_score(
                context=context,
                payload=payload,
                features=features,
                side=side,
            )
            confidence = self._compute_confidence(
                payload=payload,
                features=features,
                side=side,
            )

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

        except StrategyEvaluationError:
            raise
        except Exception as exc:
            self.logger.exception(
                "Strategy evaluation failed | strategy=%s symbol=%s",
                self.STRATEGY_NAME,
                getattr(context, "symbol", "UNKNOWN"),
            )
            raise StrategyEvaluationError(
                f"{self.STRATEGY_NAME}: failed to evaluate context for "
                f"{getattr(context, 'symbol', 'UNKNOWN')}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_payload(self, context: SignalContext) -> OIBreakoutConfirmationPayload:
        domain = context.open_interest or {}
        raw: dict[str, Any] = {}

        regime_section = self.extract_domain_section(
            context,
            "regime",
            prefix_nested_keys=True,
        )
        divergence_section = self.extract_domain_section(
            context,
            "divergence",
            prefix_nested_keys=True,
        )
        raw.update(regime_section)
        raw.update(divergence_section)

        self.merge_aliases(
            raw=raw,
            domain=domain,
            aliases={
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
            },
        )

        self.merge_feature_aliases(
            raw=raw,
            context=context,
            feature_aliases={
                "oi.regime.type": "regime_type",
                "oi.regime.confidence": "regime_confidence",
                "oi.regime.score": "regime_score",
                "oi.regime.reasons": "reasons",
                "oi.divergence.type": "divergence_type",
                "oi.divergence.detected": "divergence_detected",
                "oi.divergence.confidence": "divergence_confidence",
                "oi.divergence.score": "divergence_score",
            },
        )

        regime = self.parse_oi_regime(raw.get("regime_type") or raw.get("regime_regime"))
        regime_confidence = self.clamp(self.safe_float(raw.get("regime_confidence"), 0.0))
        regime_score = self.clamp(self.safe_float(raw.get("regime_score"), regime_confidence))

        divergence_type = self.parse_divergence_type(raw.get("divergence_type"))
        divergence_detected = self.safe_bool(
            raw.get("divergence_detected"),
            default=divergence_type != OIDivergenceType.NONE,
        )
        divergence_confidence = self.clamp(self.safe_float(raw.get("divergence_confidence"), 0.0))
        divergence_score = self.clamp(self.safe_float(raw.get("divergence_score"), divergence_confidence))
        reasons = self.normalize_reasons(raw.get("reasons") or raw.get("regime_reasons"))

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

    @staticmethod
    def _infer_setup_type(payload: OIBreakoutConfirmationPayload) -> SetupType:
        if payload.regime == OIRegime.SQUEEZE_SETUP:
            return SetupType.SQUEEZE

        if payload.regime in {
            OIRegime.LONG_BUILDUP,
            OIRegime.SHORT_BUILDUP,
            OIRegime.TREND_CONFIRMATION,
        }:
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
            if features.oi_delta_pct is None or features.oi_delta_pct <= 0:
                return "long_breakout_not_supported_by_oi_growth"
            if features.oi_price_efficiency is not None and features.oi_price_efficiency <= 0:
                return "long_breakout_not_supported_by_oi_price_efficiency"
            if features.oi_pressure_score is not None and features.oi_pressure_score < -0.10:
                return "long_breakout_negative_pressure"

        if side == SignalSide.SHORT:
            if features.oi_delta_pct is None or features.oi_delta_pct <= 0:
                return "short_breakout_not_supported_by_oi_growth"
            if features.oi_price_efficiency is not None and features.oi_price_efficiency <= 0:
                return "short_breakout_not_supported_by_oi_price_efficiency"
            if features.oi_pressure_score is not None and features.oi_pressure_score > 0.10:
                return "short_breakout_positive_pressure"

        return None

    # ------------------------------------------------------------------
    # Feature support
    # ------------------------------------------------------------------

    @staticmethod
    def _positive_breakout_support(features: OIFeatures) -> bool:
        return (
            features.oi_delta_pct is not None
            and features.oi_delta_pct > 0
            and (features.volume_ratio is None or features.volume_ratio >= 1.0)
            and (features.oi_pressure_score is None or features.oi_pressure_score >= 0.15)
        )

    @staticmethod
    def _negative_breakout_support(features: OIFeatures) -> bool:
        return (
            features.oi_delta_pct is not None
            and features.oi_delta_pct > 0
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
        regime = self.get_market_regime(context)
        base_score = payload.regime_score if payload.regime_score > 0 else payload.regime_confidence
        score = self.weighted_score(base_score, regime)

        if features is not None:
            if features.volume_ratio is not None and features.volume_ratio >= 1.0:
                score += 0.06
            if features.oi_delta_pct is not None and features.oi_delta_pct >= 0.25:
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

        return self.clamp(score)

    def _compute_confidence(
        self,
        *,
        payload: OIBreakoutConfirmationPayload,
        features: OIFeatures | None,
        side: SignalSide,
    ) -> float:
        confidence = payload.regime_confidence if payload.regime_confidence > 0 else payload.regime_score

        if features is not None:
            if features.volume_ratio is not None and features.volume_ratio >= 1.0:
                confidence += 0.04
            if features.oi_delta_pct is not None and features.oi_delta_pct >= 0.25:
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

        return self.clamp(confidence)

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
            strength=self.strength_from_score(score),
            confidence_grade=self.confidence_grade(confidence),
            status=SignalStatus.NEW,
            trigger_type=TriggerType.CONFIRMATION,
            origin=SignalOrigin.SINGLE_STRATEGY,
            priority=self.priority_from_score(score),
            regime=self.get_market_regime(context),
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

        self.add_required_source_features(signal, context)

        if features is not None:
            self.append_oi_feature_metadata(signal, features)
            self.append_oi_feature_reasons(signal, features)
            for confirmation in self._build_confirmations(
                side=side,
                features=features,
                payload=payload,
            ):
                signal.add_confirmation(confirmation)

        return signal

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

        if features.oi_delta_pct is not None and features.oi_delta_pct >= 0.25:
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
