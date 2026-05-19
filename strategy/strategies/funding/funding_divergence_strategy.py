# trading_system/strategy/strategies/funding/funding_divergence_strategy.py

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from core.event_bus import EventBus
from core.scheduler import Scheduler

from ...config import StrategyConfig, StrategyDefinitionConfig
from ...enums import (
    FeatureSource,
    SetupType,
    SignalPriority,
    SignalSide,
    StrategyCategory,
)
from ...exceptions import StrategyConfigError, StrategyEvaluationError
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
    funding_path,
    is_directional_side,
    is_stale,
    normalize_label,
    serialize_for_metadata,
    side_from_bias,
    to_bool,
    to_float,
    unit_score,
    weighted_score,
)


_ALIGNMENT_BONUS_BASE: float = 0.50
_ALIGNMENT_BONUS_PER_DIMENSION: float = 0.25
_EXTREME_ALIGNMENT_WEIGHT: float = 0.08
_SIGNAL_ALIGNMENT_WEIGHT: float = 0.07


@dataclass(slots=True)
class FundingDivergenceStrategyConfig(FundingStrategyConfig):
    """
    Unified funding divergence strategy config.

    Strategy idea:
    - read normalized funding divergence context from StrategyContext;
    - use regime, pressure, extreme and funding.signal as confluence layers;
    - generate an internal StrategySignal only when divergence context is
      directional, fresh and strong enough;
    - leave filtering, confluence, build and risk-ready conversion to
      SignalProcessor.
    """

    min_divergence_confidence: float = 0.50
    min_pressure_score: float = 0.35
    min_regime_confidence: float = 0.10
    min_extreme_severity: float = 0.45
    min_signal_confidence: float = 0.45
    min_signal_abs_score: float = 0.30

    require_non_neutral_regime: bool = True
    require_pressure_alignment: bool = False
    require_pressure_present: bool = False
    require_fresh_divergence: bool = True

    score_weight_divergence: float = 0.40
    score_weight_pressure: float = 0.22
    score_weight_regime: float = 0.18
    score_weight_alignment: float = 0.15
    score_weight_extreme_alignment: float = _EXTREME_ALIGNMENT_WEIGHT
    score_weight_signal_alignment: float = _SIGNAL_ALIGNMENT_WEIGHT

    allow_flip_confirmation: bool = True
    allow_extreme_confirmation: bool = True
    allow_signal_confirmation: bool = True
    allow_regime_confirmation: bool = True
    allow_pressure_confirmation: bool = True

    bullish_setup_label: str = "funding_bullish_divergence"
    bearish_setup_label: str = "funding_bearish_divergence"

    tag_divergence: str = "funding_divergence"
    tag_dislocation: str = "dislocation"
    tag_reversal: str = "reversal"
    tag_extreme: str = "funding_extreme"
    tag_signal: str = "funding_signal"
    tag_atomic_context: str = "atomic_funding_context"
    tag_liquidation: str = "liquidation_divergence"
    tag_cvd: str = "cvd_divergence"
    tag_oi: str = "open_interest_divergence"
    tag_price: str = "price_divergence"

    tag_confirmed_by_flip: str = "confirmed_by_flip"
    tag_confirmed_by_pressure: str = "confirmed_by_pressure"
    tag_confirmed_by_regime: str = "confirmed_by_regime"
    tag_confirmed_by_extreme: str = "confirmed_by_extreme"
    tag_confirmed_by_signal: str = "confirmed_by_funding_signal"

    price_divergence_bonus: float = 0.04
    oi_divergence_bonus: float = 0.06
    cvd_divergence_bonus: float = 0.08
    liquidation_divergence_bonus: float = 0.12

    preferred_signal_origins_for_confirmation: tuple[str, ...] = (
        "divergence",
        "pressure_reversion",
        "extreme_reversion",
        "flip",
    )
    signal_origin_confirmation_weight: dict[str, float] = field(
        default_factory=lambda: {
            "divergence": 1.00,
            "pressure_reversion": 0.90,
            "extreme_reversion": 0.85,
            "flip": 0.80,
            "regime": 0.60,
            "pressure": 0.50,
            "extreme": 0.45,
            "extreme_squeeze": 0.35,
        }
    )
    signal_origin_alignment_weight: dict[str, float] = field(
        default_factory=lambda: {
            "divergence": 1.00,
            "pressure_reversion": 0.90,
            "extreme_reversion": 0.85,
            "flip": 0.75,
            "regime": 0.55,
            "pressure": 0.45,
            "extreme": 0.40,
            "extreme_squeeze": 0.30,
        }
    )

    default_priority: SignalPriority = SignalPriority.MEDIUM
    default_setup_type: SetupType = SetupType.MEAN_REVERSION

    required_funding_features: tuple[str, ...] = (
        FUNDING_FEATURES.DIVERGENCE,
    )

    def validate(self) -> None:
        FundingStrategyConfig.validate(self)

        bounded_fields = {
            "min_divergence_confidence": self.min_divergence_confidence,
            "min_pressure_score": self.min_pressure_score,
            "min_regime_confidence": self.min_regime_confidence,
            "min_extreme_severity": self.min_extreme_severity,
            "min_signal_confidence": self.min_signal_confidence,
            "min_signal_abs_score": self.min_signal_abs_score,
            "price_divergence_bonus": self.price_divergence_bonus,
            "oi_divergence_bonus": self.oi_divergence_bonus,
            "cvd_divergence_bonus": self.cvd_divergence_bonus,
            "liquidation_divergence_bonus": self.liquidation_divergence_bonus,
        }

        for field_name, value in bounded_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        for attr in (
            "score_weight_divergence",
            "score_weight_pressure",
            "score_weight_regime",
            "score_weight_alignment",
            "score_weight_extreme_alignment",
            "score_weight_signal_alignment",
        ):
            value = getattr(self, attr)
            if value < 0.0:
                raise StrategyConfigError(f"{attr} must be >= 0")

        for attr in (
            "bullish_setup_label",
            "bearish_setup_label",
            "tag_divergence",
            "tag_dislocation",
            "tag_reversal",
            "tag_extreme",
            "tag_signal",
            "tag_atomic_context",
            "tag_liquidation",
            "tag_cvd",
            "tag_oi",
            "tag_price",
            "tag_confirmed_by_flip",
            "tag_confirmed_by_pressure",
            "tag_confirmed_by_regime",
            "tag_confirmed_by_extreme",
            "tag_confirmed_by_signal",
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


class FundingDivergenceStrategy(FundingTradingStrategy):
    """
    Unified funding divergence strategy.

    Input:
        StrategyContext with FeatureSource.FUNDING domain data / FeatureSnapshot.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, confluence, filters, building and risk payloads.
    """

    component_namespace = "strategy.funding.divergence"
    category: StrategyCategory = StrategyCategory.FUNDING
    default_setup_type: SetupType = SetupType.MEAN_REVERSION

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        funding_config: FundingDivergenceStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_funding_config = funding_config or FundingDivergenceStrategyConfig()
        resolved_funding_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            funding_config=resolved_funding_config,
            service_name=service_name,
        )

        self.divergence_config: FundingDivergenceStrategyConfig = (
            resolved_funding_config
        )

    @property
    def strategy_name(self) -> str:
        return "funding_divergence"

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(self.divergence_config.required_funding_features)

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        self.validate_context_requirements(context)

        divergence = funding_item(context, "divergence")
        if divergence is None:
            return None

        event_time = extract_event_time(divergence)
        if (
            self.divergence_config.require_fresh_divergence
            and is_stale(
                event_time=event_time,
                now=context.timestamp,
                stale_after_seconds=self.divergence_config.stale_feature_max_age_seconds,
            )
        ):
            return None

        side = self._derive_side_from_divergence(divergence)
        if not is_directional_side(side):
            return None

        divergence_confidence = self._divergence_confidence(divergence)
        if divergence_confidence < self.divergence_config.min_divergence_confidence:
            return None

        pressure = funding_item(context, "pressure")
        regime = funding_item(context, "regime")
        extreme = funding_item(context, "extreme")
        flip = funding_item(context, "flip")
        funding_signal = funding_item(context, "signal")

        if self.divergence_config.require_pressure_present and pressure is None:
            return None

        if not self._passes_regime_filter(regime):
            return None

        if not self._passes_pressure_filter(side=side, pressure=pressure):
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            side=side,
            divergence=divergence,
            pressure=pressure,
            regime=regime,
            extreme=extreme,
            flip=flip,
            funding_signal=funding_signal,
        )

        if breakdown.score < self.divergence_config.min_signal_score:
            return None

        if breakdown.confidence < self.divergence_config.min_signal_confidence:
            return None

        setup_label = (
            self.divergence_config.bullish_setup_label
            if side is SignalSide.LONG
            else self.divergence_config.bearish_setup_label
        )

        source_features = self._source_features(
            divergence=divergence,
            pressure=pressure,
            regime=regime,
            extreme=extreme,
            flip=flip,
            funding_signal=funding_signal,
        )

        reasons = list(dict.fromkeys([
            setup_label,
            *breakdown.reasons,
        ]))

        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "funding_setup_label": setup_label,
            "funding_setup_family": "funding_divergence",
            "funding_strategy_version": "2.0.0",
            "score_breakdown": breakdown.to_dict(),
            "divergence": serialize_for_metadata(divergence),
            "pressure": serialize_for_metadata(pressure),
            "regime": serialize_for_metadata(regime),
            "extreme": serialize_for_metadata(extreme),
            "flip": serialize_for_metadata(flip),
            "funding_signal": serialize_for_metadata(funding_signal),
            "event_time": event_time.isoformat() if event_time else None,
            "tags": self._tags(
                divergence=divergence,
                pressure=pressure,
                regime=regime,
                extreme=extreme,
                funding_signal=funding_signal,
            ),
        }

        return self.build_funding_signal(
            context=context,
            side=side,
            confidence=breakdown.confidence,
            score=breakdown.score,
            setup_type=self.divergence_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.divergence_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _passes_regime_filter(self, regime: Any) -> bool:
        if regime is None:
            return not self.divergence_config.require_non_neutral_regime

        confidence = extract_confidence(regime)
        if confidence < self.divergence_config.min_regime_confidence:
            return False

        if not self.divergence_config.require_non_neutral_regime:
            return True

        label = normalize_label(
            first_present(
                regime,
                (
                    "regime",
                    "bias",
                    "state",
                    "name",
                    "metadata.regime",
                    "metadata.bias",
                ),
                default=None,
            )
        )

        if not label:
            return False

        return label not in {"neutral", "flat", "unknown", "mixed", "none"}

    def _passes_pressure_filter(
        self,
        *,
        side: SignalSide,
        pressure: Any,
    ) -> bool:
        if pressure is None:
            return not self.divergence_config.require_pressure_alignment

        pressure_score = self._pressure_score(pressure)
        if pressure_score < self.divergence_config.min_pressure_score:
            return False

        if not self.divergence_config.require_pressure_alignment:
            return True

        pressure_side = self._pressure_side(pressure)
        return pressure_side in {SignalSide.UNKNOWN, side}

    # ------------------------------------------------------------------
    # Direction
    # ------------------------------------------------------------------

    def _derive_side_from_divergence(self, divergence: Any) -> SignalSide:
        explicit_side = side_from_bias(
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
        if is_directional_side(explicit_side):
            return explicit_side

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

        bullish_tokens = {
            "bullish",
            "bullish_divergence",
            "positive_bullish",
            "funding_bullish_divergence",
            "price_down_funding_up",
            "negative_funding_bullish",
            "long",
        }
        bearish_tokens = {
            "bearish",
            "bearish_divergence",
            "negative_bearish",
            "funding_bearish_divergence",
            "price_up_funding_down",
            "positive_funding_bearish",
            "short",
        }

        if divergence_type in bullish_tokens:
            return SignalSide.LONG

        if divergence_type in bearish_tokens:
            return SignalSide.SHORT

        if "bull" in divergence_type or divergence_type.endswith("_long"):
            return SignalSide.LONG

        if "bear" in divergence_type or divergence_type.endswith("_short"):
            return SignalSide.SHORT

        signed = first_present(
            divergence,
            (
                "signed_score",
                "score",
                "normalized_value",
                "funding_price_spread",
                "metadata.signed_score",
            ),
            default=None,
        )
        signed_value = to_float(signed)
        if signed_value is None:
            return SignalSide.UNKNOWN

        if signed_value > 0:
            return SignalSide.LONG
        if signed_value < 0:
            return SignalSide.SHORT

        return SignalSide.UNKNOWN

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        side: SignalSide,
        divergence: Any,
        pressure: Any,
        regime: Any,
        extreme: Any,
        flip: Any,
        funding_signal: Any,
    ) -> ScoreBreakdown:
        divergence_score = self._divergence_score(divergence)
        divergence_confidence = self._divergence_confidence(divergence)

        pressure_score = self._pressure_score(pressure)
        regime_score = self._regime_score(regime)

        pressure_alignment = self._pressure_alignment(side=side, pressure=pressure)
        regime_alignment = self._regime_alignment(side=side, regime=regime)
        alignment = clamp(
            _ALIGNMENT_BONUS_BASE
            + _ALIGNMENT_BONUS_PER_DIMENSION * pressure_alignment
            + _ALIGNMENT_BONUS_PER_DIMENSION * regime_alignment,
            0.0,
            1.0,
        )

        extreme_alignment = self._extreme_alignment(side=side, extreme=extreme)
        signal_alignment = self._funding_signal_alignment(
            side=side,
            funding_signal=funding_signal,
        )
        flip_confirmation = self._flip_confirmation(side=side, flip=flip)

        score = weighted_score(
            {
                "divergence": divergence_score,
                "pressure": pressure_score,
                "regime": regime_score,
                "alignment": alignment,
                "extreme_alignment": extreme_alignment,
                "signal_alignment": signal_alignment,
            },
            {
                "divergence": self.divergence_config.score_weight_divergence,
                "pressure": self.divergence_config.score_weight_pressure,
                "regime": self.divergence_config.score_weight_regime,
                "alignment": self.divergence_config.score_weight_alignment,
                "extreme_alignment": (
                    self.divergence_config.score_weight_extreme_alignment
                ),
                "signal_alignment": (
                    self.divergence_config.score_weight_signal_alignment
                ),
            },
            default=divergence_score,
        )

        confirmation_score = weighted_score(
            {
                "flip": flip_confirmation,
                "extreme": extreme_alignment,
                "signal": signal_alignment,
                "pressure": pressure_alignment,
                "regime": regime_alignment,
            },
            {
                "flip": 0.25,
                "extreme": 0.20,
                "signal": 0.25,
                "pressure": 0.15,
                "regime": 0.15,
            },
            default=0.0,
        )

        event_time = extract_event_time(divergence)
        fresh_score = freshness_score(
            event_time=event_time,
            now=context.timestamp,
            stale_after_seconds=self.divergence_config.stale_feature_max_age_seconds,
        )

        confidence = confidence_from_components(
            primary=divergence_confidence,
            context=weighted_score(
                {
                    "pressure": pressure_score,
                    "regime": regime_score,
                    "alignment": alignment,
                },
                {
                    "pressure": 0.35,
                    "regime": 0.30,
                    "alignment": 0.35,
                },
                default=0.0,
            ),
            confirmation=confirmation_score,
            freshness=fresh_score,
        )

        reasons = [
            f"divergence_score:{divergence_score:.3f}",
            f"divergence_confidence:{divergence_confidence:.3f}",
        ]
        confirmations: list[str] = []

        if pressure_alignment > 0:
            confirmations.append(self.divergence_config.tag_confirmed_by_pressure)
        if regime_alignment > 0:
            confirmations.append(self.divergence_config.tag_confirmed_by_regime)
        if extreme_alignment > 0:
            confirmations.append(self.divergence_config.tag_confirmed_by_extreme)
        if signal_alignment > 0:
            confirmations.append(self.divergence_config.tag_confirmed_by_signal)
        if flip_confirmation > 0:
            confirmations.append(self.divergence_config.tag_confirmed_by_flip)

        return ScoreBreakdown(
            score=score,
            confidence=confidence,
            components={
                "divergence_score": divergence_score,
                "divergence_confidence": divergence_confidence,
                "pressure_score": pressure_score,
                "regime_score": regime_score,
                "pressure_alignment": pressure_alignment,
                "regime_alignment": regime_alignment,
                "alignment": alignment,
                "extreme_alignment": extreme_alignment,
                "signal_alignment": signal_alignment,
                "flip_confirmation": flip_confirmation,
                "confirmation_score": confirmation_score,
                "freshness_score": fresh_score,
            },
            weights={
                "score_weight_divergence": self.divergence_config.score_weight_divergence,
                "score_weight_pressure": self.divergence_config.score_weight_pressure,
                "score_weight_regime": self.divergence_config.score_weight_regime,
                "score_weight_alignment": self.divergence_config.score_weight_alignment,
                "score_weight_extreme_alignment": (
                    self.divergence_config.score_weight_extreme_alignment
                ),
                "score_weight_signal_alignment": (
                    self.divergence_config.score_weight_signal_alignment
                ),
            },
            reasons=reasons,
            confirmations=confirmations,
        ).normalize()

    def _divergence_score(self, divergence: Any) -> float:
        base = unit_score(
            first_present(
                divergence,
                (
                    "score",
                    "strength",
                    "severity",
                    "normalized_score",
                    "metadata.score",
                ),
                default=None,
            ),
            default=extract_score(divergence),
        )

        bonus = self._divergence_type_bonus(divergence)
        return unit_score(base + bonus)

    def _divergence_confidence(self, divergence: Any) -> float:
        base = extract_confidence(divergence)
        bonus = self._divergence_type_bonus(divergence)
        return unit_score(base + bonus)

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

    def _pressure_alignment(
        self,
        *,
        side: SignalSide,
        pressure: Any,
    ) -> float:
        if pressure is None:
            return 0.0

        return alignment_score(
            target_side=side,
            observed_side=self._pressure_side(pressure),
            score=self._pressure_score(pressure),
        )

    def _regime_alignment(
        self,
        *,
        side: SignalSide,
        regime: Any,
    ) -> float:
        if regime is None:
            return 0.0

        return alignment_score(
            target_side=side,
            observed_side=self._regime_side(regime),
            score=self._regime_score(regime),
        )

    def _extreme_alignment(
        self,
        *,
        side: SignalSide,
        extreme: Any,
    ) -> float:
        if not self.divergence_config.allow_extreme_confirmation or extreme is None:
            return 0.0

        severity = unit_score(
            first_present(
                extreme,
                (
                    "severity",
                    "score",
                    "strength",
                    "metadata.severity",
                    "metadata.score",
                ),
                default=0.0,
            )
        )
        if severity < self.divergence_config.min_extreme_severity:
            return 0.0

        extreme_side = side_from_bias(
            first_present(
                extreme,
                (
                    "expected_side",
                    "reversal_side",
                    "side",
                    "bias",
                    "direction",
                    "metadata.side",
                    "metadata.bias",
                ),
                default=None,
            )
        )

        return alignment_score(
            target_side=side,
            observed_side=extreme_side,
            score=severity,
        )

    def _funding_signal_alignment(
        self,
        *,
        side: SignalSide,
        funding_signal: Any,
    ) -> float:
        if not self.divergence_config.allow_signal_confirmation:
            return 0.0

        if funding_signal is None:
            return 0.0

        signal_confidence = extract_confidence(funding_signal)
        if signal_confidence < self.divergence_config.min_signal_confidence:
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
        if raw_score < self.divergence_config.min_signal_abs_score:
            return 0.0

        signal_side = side_from_bias(extract_bias(funding_signal))
        origin_weight = self._signal_origin_alignment_weight(funding_signal)

        return alignment_score(
            target_side=side,
            observed_side=signal_side,
            score=unit_score(0.5 * signal_confidence + 0.5 * raw_score) * origin_weight,
        )

    def _flip_confirmation(
        self,
        *,
        side: SignalSide,
        flip: Any,
    ) -> float:
        if not self.divergence_config.allow_flip_confirmation:
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
    # Bonuses / tags / metadata
    # ------------------------------------------------------------------

    def _divergence_type_bonus(self, divergence: Any) -> float:
        labels = self._divergence_labels(divergence)

        bonus = 0.0

        if labels.intersection({"price", "price_divergence"}):
            bonus += self.divergence_config.price_divergence_bonus

        if labels.intersection({"oi", "open_interest", "open_interest_divergence"}):
            bonus += self.divergence_config.oi_divergence_bonus

        if labels.intersection({"cvd", "cvd_divergence", "volume_delta"}):
            bonus += self.divergence_config.cvd_divergence_bonus

        if labels.intersection(
            {
                "liquidation",
                "liquidations",
                "liquidation_divergence",
                "cascade",
            }
        ):
            bonus += self.divergence_config.liquidation_divergence_bonus

        return unit_score(bonus)

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

        result: set[str] = set()
        for value in raw_values:
            label = normalize_label(value)
            if not label:
                continue

            result.add(label)
            for chunk in label.replace("-", "_").split("_"):
                if chunk:
                    result.add(chunk)

        return result

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
            self.divergence_config.signal_origin_alignment_weight.get(origin, 0.5)
        )

    def _source_features(
        self,
        *,
        divergence: Any,
        pressure: Any,
        regime: Any,
        extreme: Any,
        flip: Any,
        funding_signal: Any,
    ) -> list[str]:
        features = [FUNDING_FEATURES.DIVERGENCE]

        if pressure is not None:
            features.append(FUNDING_FEATURES.PRESSURE)
            features.append(FUNDING_FEATURES.PRESSURE_SCORE)

        if regime is not None:
            features.append(FUNDING_FEATURES.REGIME)
            features.append(FUNDING_FEATURES.REGIME_CONFIDENCE)

        if extreme is not None:
            features.append(FUNDING_FEATURES.EXTREME)
            features.append(FUNDING_FEATURES.EXTREME_SEVERITY)

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
        divergence: Any,
        pressure: Any,
        regime: Any,
        extreme: Any,
        funding_signal: Any,
    ) -> list[str]:
        tags = [
            self.divergence_config.tag_funding,
            self.divergence_config.tag_divergence,
            self.divergence_config.tag_dislocation,
            self.divergence_config.tag_reversal,
        ]

        labels = self._divergence_labels(divergence)

        if labels.intersection({"price", "price_divergence"}):
            tags.append(self.divergence_config.tag_price)

        if labels.intersection({"oi", "open_interest", "open_interest_divergence"}):
            tags.append(self.divergence_config.tag_oi)

        if labels.intersection({"cvd", "cvd_divergence", "volume_delta"}):
            tags.append(self.divergence_config.tag_cvd)

        if labels.intersection({"liquidation", "liquidations", "liquidation_divergence"}):
            tags.append(self.divergence_config.tag_liquidation)

        if pressure is not None or regime is not None:
            tags.append(self.divergence_config.tag_atomic_context)

        if extreme is not None:
            tags.append(self.divergence_config.tag_extreme)

        if funding_signal is not None:
            tags.append(self.divergence_config.tag_signal)

        return list(dict.fromkeys(tags))


__all__ = [
    "FundingDivergenceStrategy",
    "FundingDivergenceStrategyConfig",
]