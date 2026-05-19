# trading_system/strategy/strategies/funding/funding_extreme_reversal_strategy.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.event_bus import EventBus
from core.scheduler import Scheduler

from ...config import StrategyConfig, StrategyDefinitionConfig
from ...enums import (
    SetupType,
    SignalPriority,
    SignalSide,
    StrategyCategory,
)
from ...exceptions import StrategyConfigError
from ...models import StrategyContext, StrategySignal, clamp
from .base import (
    FUNDING_FEATURES,
    FundingStrategyConfig,
    FundingTradingStrategy,
)
from .utils import (
    ScoreBreakdown,
    alignment_score,
    confidence_from_components,
    extract_bias,
    extract_confidence,
    extract_event_time,
    extract_score,
    first_present,
    freshness_score,
    funding_item,
    is_directional_side,
    is_stale,
    normalize_label,
    opposite_side,
    serialize_for_metadata,
    side_from_bias,
    to_float,
    unit_score,
    weighted_score,
)


_PRESSURE_NEUTRALIZATION_THRESHOLD_RATIO: float = 0.70
_SIGNAL_CONFIRMATION_SCORE_WEIGHT: float = 0.35
_DIVERGENCE_CONFIRMATION_SCORE_WEIGHT: float = 0.30
_CONTEXT_CONFIRMATION_SCORE_WEIGHT: float = 0.35


@dataclass(slots=True)
class FundingExtremeReversalStrategyConfig(FundingStrategyConfig):
    """
    Unified funding extreme reversal strategy config.

    Strategy idea:
    - read normalized funding extreme context from StrategyContext;
    - detect overcrowded positive/negative funding extremes;
    - build contrarian reversal signal;
    - use pressure, regime, flip, divergence and funding.signal as confluence;
    - leave routing, filtering, confluence, portfolio coordination and risk-ready
      conversion to SignalProcessor.
    """

    min_extreme_severity: float = 0.60
    min_pressure_score: float = 0.55
    min_regime_confidence: float = 0.15
    min_mean_reversion_probability: float = 0.50
    min_squeeze_probability: float = 0.50
    min_divergence_confidence: float = 0.45
    min_signal_confidence: float = 0.45
    min_signal_abs_score: float = 0.35

    require_reversal_risk: bool = True
    require_squeeze_risk_or_reversion_probability: bool = True
    require_high_pressure_level: bool = True
    require_fresh_extreme: bool = True

    allow_flip_confirmation: bool = True
    allow_pressure_release_confirmation: bool = True
    allow_divergence_confirmation: bool = True
    allow_signal_confirmation: bool = True
    allow_regime_confirmation: bool = True

    bearish_setup_label: str = "extreme_positive_reversal"
    bullish_setup_label: str = "extreme_negative_reversal"

    tag_extreme: str = "funding_extreme"
    tag_reversal: str = "reversal"
    tag_crowding: str = "crowding"
    tag_squeeze: str = "squeeze_risk"
    tag_divergence: str = "funding_divergence"
    tag_signal: str = "funding_signal"
    tag_global_extreme: str = "global_extreme"
    tag_percentile_extreme: str = "percentile_extreme"
    tag_zscore_extreme: str = "zscore_extreme"
    tag_local_extreme: str = "local_extreme"

    tag_confirmed_by_flip: str = "confirmed_by_flip"
    tag_confirmed_by_release: str = "confirmed_by_pressure_release"
    tag_confirmed_by_divergence: str = "confirmed_by_divergence"
    tag_confirmed_by_signal: str = "confirmed_by_funding_signal"
    tag_confirmed_by_regime: str = "confirmed_by_regime"
    tag_atomic_context: str = "atomic_funding_context"

    score_weight_extreme: float = 0.40
    score_weight_pressure: float = 0.20
    score_weight_regime: float = 0.12
    score_weight_reversion_probability: float = 0.14
    score_weight_squeeze_probability: float = 0.08
    score_weight_confirmation: float = 0.06

    global_extreme_bonus: float = 0.12
    percentile_extreme_bonus: float = 0.09
    zscore_extreme_bonus: float = 0.08
    local_extreme_bonus: float = 0.04
    liquidation_divergence_bonus: float = 0.10
    cvd_divergence_bonus: float = 0.08
    oi_divergence_bonus: float = 0.06
    price_divergence_bonus: float = 0.04

    preferred_signal_origins_for_confirmation: tuple[str, ...] = (
        "extreme_reversion",
        "pressure_reversion",
        "divergence",
        "flip",
        "regime",
    )
    signal_origin_confirmation_weight: dict[str, float] = field(
        default_factory=lambda: {
            "extreme_reversion": 1.00,
            "pressure_reversion": 0.95,
            "divergence": 0.90,
            "flip": 0.85,
            "regime": 0.65,
            "extreme": 0.55,
            "pressure": 0.45,
            "extreme_squeeze": 0.35,
        }
    )
    signal_origin_alignment_weight: dict[str, float] = field(
        default_factory=lambda: {
            "extreme_reversion": 1.00,
            "pressure_reversion": 0.95,
            "divergence": 0.90,
            "flip": 0.80,
            "regime": 0.60,
            "extreme": 0.50,
            "pressure": 0.45,
            "extreme_squeeze": 0.30,
        }
    )

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.FUNDING_EXTREME

    required_funding_features: tuple[str, ...] = (
        FUNDING_FEATURES.EXTREME,
    )

    def validate(self) -> None:
        super().validate()

        bounded_fields = {
            "min_extreme_severity": self.min_extreme_severity,
            "min_pressure_score": self.min_pressure_score,
            "min_regime_confidence": self.min_regime_confidence,
            "min_mean_reversion_probability": self.min_mean_reversion_probability,
            "min_squeeze_probability": self.min_squeeze_probability,
            "min_divergence_confidence": self.min_divergence_confidence,
            "min_signal_confidence": self.min_signal_confidence,
            "min_signal_abs_score": self.min_signal_abs_score,
            "global_extreme_bonus": self.global_extreme_bonus,
            "percentile_extreme_bonus": self.percentile_extreme_bonus,
            "zscore_extreme_bonus": self.zscore_extreme_bonus,
            "local_extreme_bonus": self.local_extreme_bonus,
            "liquidation_divergence_bonus": self.liquidation_divergence_bonus,
            "cvd_divergence_bonus": self.cvd_divergence_bonus,
            "oi_divergence_bonus": self.oi_divergence_bonus,
            "price_divergence_bonus": self.price_divergence_bonus,
        }

        for field_name, value in bounded_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        for attr in (
            "score_weight_extreme",
            "score_weight_pressure",
            "score_weight_regime",
            "score_weight_reversion_probability",
            "score_weight_squeeze_probability",
            "score_weight_confirmation",
        ):
            value = getattr(self, attr)
            if value < 0.0:
                raise StrategyConfigError(f"{attr} must be >= 0")

        for attr in (
            "bearish_setup_label",
            "bullish_setup_label",
            "tag_extreme",
            "tag_reversal",
            "tag_crowding",
            "tag_squeeze",
            "tag_divergence",
            "tag_signal",
            "tag_global_extreme",
            "tag_percentile_extreme",
            "tag_zscore_extreme",
            "tag_local_extreme",
            "tag_confirmed_by_flip",
            "tag_confirmed_by_release",
            "tag_confirmed_by_divergence",
            "tag_confirmed_by_signal",
            "tag_confirmed_by_regime",
            "tag_atomic_context",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")

        for mapping_name in (
            "signal_origin_confirmation_weight",
            "signal_origin_alignment_weight",
        ):
            mapping = getattr(self, mapping_name)
            for key, value in mapping.items():
                if not isinstance(key, str) or not key.strip():
                    raise StrategyConfigError(
                        f"{mapping_name} keys must be non-empty strings"
                    )
                if not 0.0 <= float(value) <= 1.0:
                    raise StrategyConfigError(
                        f"{mapping_name}[{key!r}] must be between 0.0 and 1.0"
                    )

        if not self.required_funding_features:
            raise StrategyConfigError("required_funding_features cannot be empty")

        for feature in self.required_funding_features:
            if not isinstance(feature, str) or not feature.strip():
                raise StrategyConfigError(
                    "required_funding_features cannot contain empty feature names"
                )


class FundingExtremeReversalStrategy(FundingTradingStrategy):
    """
    Unified contrarian reversal strategy for funding extremes.

    Input:
        StrategyContext with FeatureSource.FUNDING domain data / FeatureSnapshot.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, filters, confluence, building and risk payloads.
    """

    component_namespace = "strategy.funding.extreme_reversal"
    category: StrategyCategory = StrategyCategory.FUNDING
    default_setup_type: SetupType = SetupType.FUNDING_EXTREME

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        funding_config: FundingExtremeReversalStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_funding_config = (
            funding_config or FundingExtremeReversalStrategyConfig()
        )
        resolved_funding_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            funding_config=resolved_funding_config,
            service_name=service_name,
        )

        self.extreme_config: FundingExtremeReversalStrategyConfig = (
            resolved_funding_config
        )

    @property
    def strategy_name(self) -> str:
        return "funding_extreme_reversal"

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(self.extreme_config.required_funding_features)

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        self.validate_context_requirements(context)

        extreme = funding_item(context, "extreme")
        if extreme is None:
            return None

        event_time = extract_event_time(extreme)
        if (
            self.extreme_config.require_fresh_extreme
            and is_stale(
                event_time=event_time,
                now=context.timestamp,
                stale_after_seconds=self.extreme_config.stale_feature_max_age_seconds,
            )
        ):
            return None

        side = self._derive_contrarian_side_from_extreme(extreme)
        if not is_directional_side(side):
            return None

        if not self._passes_extreme_thresholds(extreme):
            return None

        pressure = funding_item(context, "pressure")
        regime = funding_item(context, "regime")
        divergence = funding_item(context, "divergence")
        flip = funding_item(context, "flip")
        funding_signal = funding_item(context, "signal")

        if not self._passes_pressure_filter(side=side, pressure=pressure):
            return None

        if not self._passes_regime_filter(regime):
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            side=side,
            extreme=extreme,
            pressure=pressure,
            regime=regime,
            divergence=divergence,
            flip=flip,
            funding_signal=funding_signal,
        )

        if breakdown.score < self.extreme_config.min_signal_score:
            return None

        if breakdown.confidence < self.extreme_config.min_signal_confidence:
            return None

        setup_label = (
            self.extreme_config.bullish_setup_label
            if side is SignalSide.LONG
            else self.extreme_config.bearish_setup_label
        )

        source_features = self._source_features(
            extreme=extreme,
            pressure=pressure,
            regime=regime,
            divergence=divergence,
            flip=flip,
            funding_signal=funding_signal,
        )

        reasons = list(
            dict.fromkeys(
                [
                    setup_label,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "funding_setup_label": setup_label,
            "funding_setup_family": "funding_extreme_reversal",
            "funding_strategy_version": "2.0.0",
            "score_breakdown": breakdown.to_dict(),
            "extreme": serialize_for_metadata(extreme),
            "pressure": serialize_for_metadata(pressure),
            "regime": serialize_for_metadata(regime),
            "divergence": serialize_for_metadata(divergence),
            "flip": serialize_for_metadata(flip),
            "funding_signal": serialize_for_metadata(funding_signal),
            "event_time": event_time.isoformat() if event_time else None,
            "tags": self._tags(
                extreme=extreme,
                pressure=pressure,
                regime=regime,
                divergence=divergence,
                funding_signal=funding_signal,
            ),
        }

        return self.build_funding_signal(
            context=context,
            side=side,
            confidence=breakdown.confidence,
            score=breakdown.score,
            setup_type=self.extreme_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.extreme_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _passes_extreme_thresholds(self, extreme: Any) -> bool:
        severity = self._extreme_severity(extreme)
        if severity < self.extreme_config.min_extreme_severity:
            return False

        if self.extreme_config.require_reversal_risk:
            if not self._has_reversal_risk(extreme):
                return False

        if self.extreme_config.require_squeeze_risk_or_reversion_probability:
            mean_reversion_probability = self._mean_reversion_probability(extreme)
            squeeze_probability = self._squeeze_probability(extreme)

            if (
                mean_reversion_probability
                < self.extreme_config.min_mean_reversion_probability
                and squeeze_probability < self.extreme_config.min_squeeze_probability
            ):
                return False

        return True

    def _passes_pressure_filter(
        self,
        *,
        side: SignalSide,
        pressure: Any,
    ) -> bool:
        if pressure is None:
            return not self.extreme_config.require_high_pressure_level

        pressure_score = self._pressure_score(pressure)
        if self.extreme_config.require_high_pressure_level:
            if pressure_score < self.extreme_config.min_pressure_score:
                return False

        pressure_side = self._pressure_side(pressure)
        if pressure_side is SignalSide.UNKNOWN:
            return True

        # Funding extreme reversal is contrarian. Pressure can either already
        # support reversal side or still represent overcrowding opposite side.
        return pressure_side in {side, opposite_side(side)}

    def _passes_regime_filter(self, regime: Any) -> bool:
        if regime is None:
            return True

        confidence = extract_confidence(regime)
        if confidence < self.extreme_config.min_regime_confidence:
            return False

        return True

    # ------------------------------------------------------------------
    # Direction
    # ------------------------------------------------------------------

    def _derive_contrarian_side_from_extreme(self, extreme: Any) -> SignalSide:
        explicit_side = side_from_bias(
            first_present(
                extreme,
                (
                    "reversal_side",
                    "expected_side",
                    "target_side",
                    "signal_side",
                    "side",
                    "metadata.reversal_side",
                    "metadata.expected_side",
                    "metadata.side",
                ),
                default=None,
            )
        )
        if is_directional_side(explicit_side):
            return explicit_side

        extreme_type = normalize_label(
            first_present(
                extreme,
                (
                    "extreme_type",
                    "type",
                    "kind",
                    "metadata.extreme_type",
                    "metadata.type",
                ),
                default=None,
            )
        )

        # Positive funding extreme usually means crowded longs -> contrarian short.
        positive_extreme_tokens = {
            "positive",
            "positive_extreme",
            "high_positive",
            "overheated_positive",
            "long_crowding",
            "crowded_longs",
            "positive_funding",
            "funding_positive_extreme",
            "extreme_positive",
        }

        # Negative funding extreme usually means crowded shorts -> contrarian long.
        negative_extreme_tokens = {
            "negative",
            "negative_extreme",
            "high_negative",
            "overheated_negative",
            "short_crowding",
            "crowded_shorts",
            "negative_funding",
            "funding_negative_extreme",
            "extreme_negative",
        }

        if extreme_type in positive_extreme_tokens:
            return SignalSide.SHORT

        if extreme_type in negative_extreme_tokens:
            return SignalSide.LONG

        if "positive" in extreme_type or "long_crowd" in extreme_type:
            return SignalSide.SHORT

        if "negative" in extreme_type or "short_crowd" in extreme_type:
            return SignalSide.LONG

        funding_rate = to_float(
            first_present(
                extreme,
                (
                    "funding_rate",
                    "rate",
                    "current_rate",
                    "snapshot.funding_rate",
                    "metadata.funding_rate",
                ),
                default=None,
            )
        )
        if funding_rate is not None:
            if funding_rate > 0:
                return SignalSide.SHORT
            if funding_rate < 0:
                return SignalSide.LONG

        bias_side = side_from_bias(extract_bias(extreme))
        if is_directional_side(bias_side):
            return opposite_side(bias_side)

        return SignalSide.UNKNOWN

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        side: SignalSide,
        extreme: Any,
        pressure: Any,
        regime: Any,
        divergence: Any,
        flip: Any,
        funding_signal: Any,
    ) -> ScoreBreakdown:
        extreme_score = self._extreme_score(extreme)
        extreme_confidence = self._extreme_confidence(extreme)

        pressure_score = self._pressure_score(pressure)
        regime_score = self._regime_score(regime)
        reversion_probability = self._mean_reversion_probability(extreme)
        squeeze_probability = self._squeeze_probability(extreme)

        pressure_confirmation = self._pressure_reversal_confirmation(
            side=side,
            pressure=pressure,
        )
        regime_confirmation = self._regime_confirmation(
            side=side,
            regime=regime,
        )
        divergence_confirmation = self._divergence_confirmation(
            side=side,
            divergence=divergence,
        )
        signal_confirmation = self._funding_signal_confirmation(
            side=side,
            funding_signal=funding_signal,
        )
        flip_confirmation = self._flip_confirmation(
            side=side,
            flip=flip,
        )

        confirmation_score = weighted_score(
            {
                "pressure": pressure_confirmation,
                "regime": regime_confirmation,
                "divergence": divergence_confirmation,
                "signal": signal_confirmation,
                "flip": flip_confirmation,
            },
            {
                "pressure": _CONTEXT_CONFIRMATION_SCORE_WEIGHT,
                "regime": 0.15,
                "divergence": _DIVERGENCE_CONFIRMATION_SCORE_WEIGHT,
                "signal": _SIGNAL_CONFIRMATION_SCORE_WEIGHT,
                "flip": 0.20,
            },
            default=0.0,
        )

        score = weighted_score(
            {
                "extreme": extreme_score,
                "pressure": pressure_score,
                "regime": regime_score,
                "reversion_probability": reversion_probability,
                "squeeze_probability": squeeze_probability,
                "confirmation": confirmation_score,
            },
            {
                "extreme": self.extreme_config.score_weight_extreme,
                "pressure": self.extreme_config.score_weight_pressure,
                "regime": self.extreme_config.score_weight_regime,
                "reversion_probability": (
                    self.extreme_config.score_weight_reversion_probability
                ),
                "squeeze_probability": (
                    self.extreme_config.score_weight_squeeze_probability
                ),
                "confirmation": self.extreme_config.score_weight_confirmation,
            },
            default=extreme_score,
        )

        event_time = extract_event_time(extreme)
        fresh_score = freshness_score(
            event_time=event_time,
            now=context.timestamp,
            stale_after_seconds=self.extreme_config.stale_feature_max_age_seconds,
        )

        context_score = weighted_score(
            {
                "pressure": pressure_score,
                "regime": regime_score,
                "reversion_probability": reversion_probability,
                "squeeze_probability": squeeze_probability,
            },
            {
                "pressure": 0.35,
                "regime": 0.15,
                "reversion_probability": 0.30,
                "squeeze_probability": 0.20,
            },
            default=0.0,
        )

        confidence = confidence_from_components(
            primary=extreme_confidence,
            context=context_score,
            confirmation=confirmation_score,
            freshness=fresh_score,
        )

        reasons = [
            f"extreme_score:{extreme_score:.3f}",
            f"extreme_confidence:{extreme_confidence:.3f}",
            f"mean_reversion_probability:{reversion_probability:.3f}",
            f"squeeze_probability:{squeeze_probability:.3f}",
        ]

        confirmations: list[str] = []
        if pressure_confirmation > 0:
            confirmations.append(self.extreme_config.tag_confirmed_by_release)
        if regime_confirmation > 0:
            confirmations.append(self.extreme_config.tag_confirmed_by_regime)
        if divergence_confirmation > 0:
            confirmations.append(self.extreme_config.tag_confirmed_by_divergence)
        if signal_confirmation > 0:
            confirmations.append(self.extreme_config.tag_confirmed_by_signal)
        if flip_confirmation > 0:
            confirmations.append(self.extreme_config.tag_confirmed_by_flip)

        return ScoreBreakdown(
            score=score,
            confidence=confidence,
            components={
                "extreme_score": extreme_score,
                "extreme_confidence": extreme_confidence,
                "pressure_score": pressure_score,
                "regime_score": regime_score,
                "mean_reversion_probability": reversion_probability,
                "squeeze_probability": squeeze_probability,
                "pressure_confirmation": pressure_confirmation,
                "regime_confirmation": regime_confirmation,
                "divergence_confirmation": divergence_confirmation,
                "signal_confirmation": signal_confirmation,
                "flip_confirmation": flip_confirmation,
                "confirmation_score": confirmation_score,
                "context_score": context_score,
                "freshness_score": fresh_score,
            },
            weights={
                "score_weight_extreme": self.extreme_config.score_weight_extreme,
                "score_weight_pressure": self.extreme_config.score_weight_pressure,
                "score_weight_regime": self.extreme_config.score_weight_regime,
                "score_weight_reversion_probability": (
                    self.extreme_config.score_weight_reversion_probability
                ),
                "score_weight_squeeze_probability": (
                    self.extreme_config.score_weight_squeeze_probability
                ),
                "score_weight_confirmation": (
                    self.extreme_config.score_weight_confirmation
                ),
            },
            reasons=reasons,
            confirmations=confirmations,
        ).normalize()

    def _extreme_score(self, extreme: Any) -> float:
        base = unit_score(
            first_present(
                extreme,
                (
                    "score",
                    "severity",
                    "strength",
                    "normalized_score",
                    "metadata.score",
                    "metadata.severity",
                ),
                default=extract_score(extreme),
            )
        )

        bonus = self._extreme_type_bonus(extreme)
        return unit_score(base + bonus)

    def _extreme_confidence(self, extreme: Any) -> float:
        confidence = unit_score(
            first_present(
                extreme,
                (
                    "confidence",
                    "probability",
                    "mean_reversion_probability",
                    "metadata.confidence",
                ),
                default=extract_confidence(extreme, default=0.5),
            )
        )
        bonus = self._extreme_type_bonus(extreme)
        return unit_score(confidence + bonus)

    def _extreme_severity(self, extreme: Any) -> float:
        return unit_score(
            first_present(
                extreme,
                (
                    "severity",
                    "score",
                    "strength",
                    "normalized_score",
                    "metadata.severity",
                    "metadata.score",
                ),
                default=0.0,
            )
        )

    def _mean_reversion_probability(self, extreme: Any) -> float:
        return unit_score(
            first_present(
                extreme,
                (
                    "mean_reversion_probability",
                    "reversion_probability",
                    "reversal_probability",
                    "metadata.mean_reversion_probability",
                    "metadata.reversion_probability",
                ),
                default=0.0,
            )
        )

    def _squeeze_probability(self, extreme: Any) -> float:
        return unit_score(
            first_present(
                extreme,
                (
                    "squeeze_probability",
                    "squeeze_risk",
                    "short_squeeze_probability",
                    "long_squeeze_probability",
                    "metadata.squeeze_probability",
                    "metadata.squeeze_risk",
                ),
                default=0.0,
            )
        )

    def _has_reversal_risk(self, extreme: Any) -> bool:
        explicit = first_present(
            extreme,
            (
                "reversal_risk",
                "has_reversal_risk",
                "mean_reversion_risk",
                "metadata.reversal_risk",
                "metadata.mean_reversion_risk",
            ),
            default=None,
        )
        if explicit is not None:
            label = normalize_label(explicit)
            if label in {"true", "1", "yes", "y", "on"}:
                return True
            if label in {"false", "0", "no", "n", "off"}:
                return False

        return (
            self._mean_reversion_probability(extreme)
            >= self.extreme_config.min_mean_reversion_probability
        )

    def _pressure_score(self, pressure: Any) -> float:
        if pressure is None:
            return 0.0

        return unit_score(
            first_present(
                pressure,
                (
                    "pressure_score",
                    "score",
                    "strength",
                    "normalized_score",
                    "metadata.pressure_score",
                    "metadata.score",
                ),
                default=0.0,
            )
        )

    def _regime_score(self, regime: Any) -> float:
        if regime is None:
            return 0.0
        return extract_confidence(regime)

    def _pressure_side(self, pressure: Any) -> SignalSide:
        if pressure is None:
            return SignalSide.UNKNOWN

        return side_from_bias(
            first_present(
                pressure,
                (
                    "direction",
                    "pressure_direction",
                    "side",
                    "bias",
                    "metadata.direction",
                    "metadata.bias",
                ),
                default=None,
            )
        )

    def _regime_side(self, regime: Any) -> SignalSide:
        if regime is None:
            return SignalSide.UNKNOWN
        return side_from_bias(extract_bias(regime))

    def _pressure_reversal_confirmation(
        self,
        *,
        side: SignalSide,
        pressure: Any,
    ) -> float:
        if not self.extreme_config.allow_pressure_release_confirmation:
            return 0.0

        if pressure is None:
            return 0.0

        pressure_score = self._pressure_score(pressure)
        pressure_side = self._pressure_side(pressure)

        # Best case: pressure already aligns with reversal side.
        aligned = alignment_score(
            target_side=side,
            observed_side=pressure_side,
            score=pressure_score,
        )
        if aligned > 0:
            return aligned

        # Accept partial confirmation if previous overcrowding pressure is
        # neutralizing/releasing.
        release_score = unit_score(
            first_present(
                pressure,
                (
                    "release_score",
                    "neutralization_score",
                    "pressure_release_score",
                    "metadata.release_score",
                    "metadata.neutralization_score",
                ),
                default=0.0,
            )
        )
        if release_score > 0:
            return release_score

        previous_score = to_float(
            first_present(
                pressure,
                (
                    "previous_score",
                    "prev_score",
                    "metadata.previous_score",
                    "metadata.prev_score",
                ),
                default=None,
            )
        )
        if previous_score is not None and previous_score > 0:
            drop_ratio = 1.0 - clamp(pressure_score / previous_score, 0.0, 1.0)
            if drop_ratio >= (1.0 - _PRESSURE_NEUTRALIZATION_THRESHOLD_RATIO):
                return unit_score(drop_ratio)

        return 0.0

    def _regime_confirmation(
        self,
        *,
        side: SignalSide,
        regime: Any,
    ) -> float:
        if not self.extreme_config.allow_regime_confirmation:
            return 0.0

        if regime is None:
            return 0.0

        return alignment_score(
            target_side=side,
            observed_side=self._regime_side(regime),
            score=self._regime_score(regime),
        )

    def _divergence_confirmation(
        self,
        *,
        side: SignalSide,
        divergence: Any,
    ) -> float:
        if not self.extreme_config.allow_divergence_confirmation:
            return 0.0

        if divergence is None:
            return 0.0

        confidence = extract_confidence(divergence)
        if confidence < self.extreme_config.min_divergence_confidence:
            return 0.0

        divergence_side = side_from_bias(
            first_present(
                divergence,
                (
                    "side",
                    "signal_side",
                    "expected_side",
                    "target_side",
                    "direction",
                    "bias",
                    "metadata.side",
                    "metadata.bias",
                ),
                default=None,
            )
        )

        if divergence_side is SignalSide.UNKNOWN:
            divergence_side = self._side_from_divergence_type(divergence)

        score = unit_score(
            0.5 * confidence + 0.5 * extract_score(divergence, default=confidence)
        )
        score = unit_score(score + self._divergence_type_bonus(divergence))

        return alignment_score(
            target_side=side,
            observed_side=divergence_side,
            score=score,
        )

    def _funding_signal_confirmation(
        self,
        *,
        side: SignalSide,
        funding_signal: Any,
    ) -> float:
        if not self.extreme_config.allow_signal_confirmation:
            return 0.0

        if funding_signal is None:
            return 0.0

        signal_confidence = extract_confidence(funding_signal)
        if signal_confidence < self.extreme_config.min_signal_confidence:
            return 0.0

        raw_score = unit_score(
            abs(
                to_float(
                    first_present(
                        funding_signal,
                        (
                            "score",
                            "signed_score",
                            "strength",
                            "normalized_score",
                            "metadata.score",
                        ),
                        default=0.0,
                    ),
                    default=0.0,
                )
                or 0.0
            )
        )
        if raw_score < self.extreme_config.min_signal_abs_score:
            return 0.0

        signal_side = side_from_bias(extract_bias(funding_signal))
        origin_weight = self._signal_origin_alignment_weight(funding_signal)

        return alignment_score(
            target_side=side,
            observed_side=signal_side,
            score=unit_score(0.5 * signal_confidence + 0.5 * raw_score)
            * origin_weight,
        )

    def _flip_confirmation(
        self,
        *,
        side: SignalSide,
        flip: Any,
    ) -> float:
        if not self.extreme_config.allow_flip_confirmation:
            return 0.0

        if flip is None:
            return 0.0

        flip_label = normalize_label(
            first_present(
                flip,
                (
                    "flip_type",
                    "type",
                    "direction",
                    "bias",
                    "metadata.flip_type",
                    "metadata.type",
                ),
                default=None,
            )
        )
        if not flip_label:
            return 0.0

        long_flip_tokens = {
            "negative_to_positive",
            "bearish_to_bullish",
            "short_to_long",
            "to_positive",
            "to_bullish",
            "long",
            "bullish",
        }
        short_flip_tokens = {
            "positive_to_negative",
            "bullish_to_bearish",
            "long_to_short",
            "to_negative",
            "to_bearish",
            "short",
            "bearish",
        }

        confidence = extract_confidence(flip, default=1.0)

        if side is SignalSide.LONG and (
            flip_label in long_flip_tokens or "negative_to_positive" in flip_label
        ):
            return confidence

        if side is SignalSide.SHORT and (
            flip_label in short_flip_tokens or "positive_to_negative" in flip_label
        ):
            return confidence

        return 0.0

    # ------------------------------------------------------------------
    # Labels / bonuses / metadata
    # ------------------------------------------------------------------

    def _extreme_type_bonus(self, extreme: Any) -> float:
        labels = self._extreme_labels(extreme)

        bonus = 0.0

        if labels.intersection({"global", "global_extreme"}):
            bonus += self.extreme_config.global_extreme_bonus

        if labels.intersection({"percentile", "percentile_extreme"}):
            bonus += self.extreme_config.percentile_extreme_bonus

        if labels.intersection({"zscore", "z_score", "zscore_extreme"}):
            bonus += self.extreme_config.zscore_extreme_bonus

        if labels.intersection({"local", "local_extreme"}):
            bonus += self.extreme_config.local_extreme_bonus

        return unit_score(bonus)

    def _divergence_type_bonus(self, divergence: Any) -> float:
        labels = self._divergence_labels(divergence)

        bonus = 0.0

        if labels.intersection({"price", "price_divergence"}):
            bonus += self.extreme_config.price_divergence_bonus

        if labels.intersection({"oi", "open_interest", "open_interest_divergence"}):
            bonus += self.extreme_config.oi_divergence_bonus

        if labels.intersection({"cvd", "cvd_divergence", "volume_delta"}):
            bonus += self.extreme_config.cvd_divergence_bonus

        if labels.intersection(
            {
                "liquidation",
                "liquidations",
                "liquidation_divergence",
                "cascade",
            }
        ):
            bonus += self.extreme_config.liquidation_divergence_bonus

        return unit_score(bonus)

    def _extreme_labels(self, extreme: Any) -> set[str]:
        raw_values = [
            first_present(
                extreme,
                (
                    "extreme_type",
                    "type",
                    "kind",
                    "source",
                    "origin",
                    "metadata.extreme_type",
                    "metadata.type",
                    "metadata.source",
                    "metadata.origin",
                ),
                default=None,
            )
        ]

        tags = first_present(
            extreme,
            (
                "tags",
                "labels",
                "metadata.tags",
                "metadata.labels",
            ),
            default=None,
        )
        if isinstance(tags, (list, tuple, set)):
            raw_values.extend(tags)

        return self._labels_from_raw(raw_values)

    def _divergence_labels(self, divergence: Any) -> set[str]:
        raw_values = [
            first_present(
                divergence,
                (
                    "divergence_type",
                    "type",
                    "kind",
                    "source",
                    "origin",
                    "metadata.divergence_type",
                    "metadata.type",
                    "metadata.source",
                    "metadata.origin",
                ),
                default=None,
            )
        ]

        tags = first_present(
            divergence,
            (
                "tags",
                "labels",
                "metadata.tags",
                "metadata.labels",
            ),
            default=None,
        )
        if isinstance(tags, (list, tuple, set)):
            raw_values.extend(tags)

        return self._labels_from_raw(raw_values)

    @staticmethod
    def _labels_from_raw(raw_values: list[Any]) -> set[str]:
        result: set[str] = set()

        for value in raw_values:
            label = normalize_label(value)
            if not label:
                continue

            normalized = label.replace("-", "_")
            result.add(normalized)

            for chunk in normalized.split("_"):
                if chunk:
                    result.add(chunk)

        return result

    def _side_from_divergence_type(self, divergence: Any) -> SignalSide:
        divergence_type = normalize_label(
            first_present(
                divergence,
                (
                    "divergence_type",
                    "type",
                    "kind",
                    "metadata.divergence_type",
                    "metadata.type",
                ),
                default=None,
            )
        )

        if "bull" in divergence_type or divergence_type.endswith("_long"):
            return SignalSide.LONG

        if "bear" in divergence_type or divergence_type.endswith("_short"):
            return SignalSide.SHORT

        return SignalSide.UNKNOWN

    def _signal_origin_alignment_weight(self, funding_signal: Any) -> float:
        origin = normalize_label(
            first_present(
                funding_signal,
                (
                    "origin",
                    "signal_origin",
                    "source",
                    "metadata.origin",
                    "metadata.signal_origin",
                    "metadata.source",
                ),
                default=None,
            )
        )

        if not origin:
            return 1.0

        return unit_score(
            self.extreme_config.signal_origin_alignment_weight.get(origin, 0.5)
        )

    def _source_features(
        self,
        *,
        extreme: Any,
        pressure: Any,
        regime: Any,
        divergence: Any,
        flip: Any,
        funding_signal: Any,
    ) -> list[str]:
        features = [
            FUNDING_FEATURES.EXTREME,
            FUNDING_FEATURES.EXTREME_SEVERITY,
            FUNDING_FEATURES.EXTREME_MEAN_REVERSION_PROBABILITY,
            FUNDING_FEATURES.EXTREME_SQUEEZE_PROBABILITY,
        ]

        if pressure is not None:
            features.append(FUNDING_FEATURES.PRESSURE)
            features.append(FUNDING_FEATURES.PRESSURE_SCORE)

        if regime is not None:
            features.append(FUNDING_FEATURES.REGIME)
            features.append(FUNDING_FEATURES.REGIME_CONFIDENCE)

        if divergence is not None:
            features.append(FUNDING_FEATURES.DIVERGENCE)
            features.append(FUNDING_FEATURES.DIVERGENCE_CONFIDENCE)

        if flip is not None:
            features.append(FUNDING_FEATURES.FLIP)

        if funding_signal is not None:
            features.append(FUNDING_FEATURES.SIGNAL)
            features.append(FUNDING_FEATURES.SIGNAL_SCORE)
            features.append(FUNDING_FEATURES.SIGNAL_CONFIDENCE)

        return list(dict.fromkeys(features))

    def _tags(
        self,
        *,
        extreme: Any,
        pressure: Any,
        regime: Any,
        divergence: Any,
        funding_signal: Any,
    ) -> list[str]:
        tags = [
            self.extreme_config.tag_funding,
            self.extreme_config.tag_extreme,
            self.extreme_config.tag_reversal,
            self.extreme_config.tag_crowding,
        ]

        extreme_labels = self._extreme_labels(extreme)

        if extreme_labels.intersection({"global", "global_extreme"}):
            tags.append(self.extreme_config.tag_global_extreme)

        if extreme_labels.intersection({"percentile", "percentile_extreme"}):
            tags.append(self.extreme_config.tag_percentile_extreme)

        if extreme_labels.intersection({"zscore", "z_score", "zscore_extreme"}):
            tags.append(self.extreme_config.tag_zscore_extreme)

        if extreme_labels.intersection({"local", "local_extreme"}):
            tags.append(self.extreme_config.tag_local_extreme)

        if self._squeeze_probability(extreme) >= self.extreme_config.min_squeeze_probability:
            tags.append(self.extreme_config.tag_squeeze)

        if pressure is not None or regime is not None:
            tags.append(self.extreme_config.tag_atomic_context)

        if divergence is not None:
            tags.append(self.extreme_config.tag_divergence)

        if funding_signal is not None:
            tags.append(self.extreme_config.tag_signal)

        return list(dict.fromkeys(tags))


__all__ = [
    "FundingExtremeReversalStrategy",
    "FundingExtremeReversalStrategyConfig",
]