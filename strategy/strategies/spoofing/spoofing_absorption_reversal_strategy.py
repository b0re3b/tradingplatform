# trading_system/strategy/strategies/spoofing/spoofing_absorption_reversal_strategy.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from analytics.spoofing.enums import SpoofingComponent, SpoofingPattern, SpoofingType
from core.event_bus import EventBus
from core.scheduler import Scheduler

from ...config import StrategyConfig, StrategyDefinitionConfig
from ...enums import (
    MarketRegime,
    SetupType,
    SignalPriority,
    SignalSide,
    StrategyCategory,
    Timeframe,
)
from ...exceptions import StrategyConfigError
from ...models import StrategyContext, StrategyMetadata, StrategySignal
from .base import (
    SPOOFING_FEATURES,
    SpoofingCompositeSnapshot,
    SpoofingStrategyConfig,
    SpoofingTradingStrategy,
)
from .utils import (
    ScoreBreakdown,
    average_score,
    base_spoofing_source_features,
    confidence_from_components,
    detector_agreement_ratio,
    detector_average_confidence,
    detector_confidence,
    detector_count,
    detector_passed,
    detector_score,
    extract_cancel_to_fill_ratio,
    extract_event_time,
    extract_fill_ratio,
    extract_price_reaction_bps,
    extract_pull_ratio,
    extract_pulled_notional,
    extract_score,
    extract_wall_notional,
    fake_liquidity_source_features,
    freshness_score,
    is_directional_side,
    is_fake_liquidity_signal,
    is_stale,
    normalize_label,
    quality_filter_reason,
    reaction_aligns_with_side,
    serialize_for_metadata,
    spoofing_side_to_signal_side,
    unit_score,
    weighted_score,
)


@dataclass(slots=True)
class SpoofingAbsorptionReversalPayload:
    """
    Normalized strategy-level payload для fake absorption reversal.

    Direction convention:
    - fake BID absorption/support -> buyers were not real -> SHORT;
    - fake ASK absorption/resistance -> sellers were not real -> LONG.
    """

    snapshot: SpoofingCompositeSnapshot
    side: SignalSide

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def pull_ratio(self) -> float:
        return extract_pull_ratio(self.snapshot.raw_signal)

    @property
    def fill_ratio(self) -> float:
        return extract_fill_ratio(self.snapshot.raw_signal)

    @property
    def cancel_to_fill_ratio(self) -> float:
        return extract_cancel_to_fill_ratio(self.snapshot.raw_signal)

    @property
    def price_reaction_bps(self) -> float:
        return extract_price_reaction_bps(self.snapshot.raw_signal)

    @property
    def wall_notional(self) -> float:
        return extract_wall_notional(self.snapshot.raw_signal)

    @property
    def pulled_notional(self) -> float:
        return extract_pulled_notional(self.snapshot.raw_signal)


@dataclass(slots=True)
class SpoofingAbsorptionReversalStrategyConfig(SpoofingStrategyConfig):
    """
    Unified fake-absorption reversal strategy config.

    Strategy idea:
    - analytics.spoofing detects fake absorption / fake liquidity wall;
    - wall looked like support/resistance but had high cancel-to-fill and low fill;
    - pulled liquidity invalidates absorption narrative;
    - strategy returns internal StrategySignal only.
    """

    min_absorption_score: float = 0.70
    min_absorption_confidence: float = 0.60

    min_pull_ratio: float = 0.55
    max_fill_ratio: float = 0.25
    min_cancel_to_fill_ratio: float = 0.65
    min_wall_notional: float = 0.0
    min_pulled_notional: float = 0.0

    require_fake_absorption_pattern: bool = True
    require_fake_liquidity_detector: bool = False
    require_fake_liquidity_detector_passed: bool = False
    require_market_reaction: bool = False
    require_directional_reaction_alignment: bool = False

    min_price_reaction_bps: float = 1.0
    min_fake_liquidity_detector_score: float = 0.0
    min_fake_liquidity_detector_confidence: float = 0.0

    score_base_weight: float = 0.24
    score_absorption_weight: float = 0.22
    score_cancel_fill_weight: float = 0.18
    score_fill_weight: float = 0.14
    score_reaction_weight: float = 0.10
    score_detector_weight: float = 0.08
    score_freshness_weight: float = 0.04

    confidence_primary_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    strong_cancel_to_fill_bonus: float = 0.04
    low_fill_bonus: float = 0.04
    strong_pull_bonus: float = 0.04
    detector_bonus: float = 0.04
    reaction_bonus: float = 0.03
    directional_reaction_bonus: float = 0.04
    notional_bonus: float = 0.03

    strong_cancel_to_fill_threshold: float = 0.85
    very_low_fill_ratio_threshold: float = 0.10
    strong_pull_ratio_threshold: float = 0.75

    entry_offset_bps_hint: float | None = None
    stop_buffer_bps_hint: float | None = None
    take_profit_bps_hint: float | None = None
    absorption_tp_multiplier_hint: float | None = None

    tag_spoofing_absorption_reversal: str = "spoofing_absorption_reversal"
    tag_fake_absorption: str = "fake_absorption"
    tag_absorption_failed: str = "absorption_failed"
    tag_cancel_to_fill: str = "cancel_to_fill"
    tag_low_fill: str = "low_fill"
    tag_unwind: str = "unwind"

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.REVERSAL

    required_spoofing_features: tuple[str, ...] = (
        SPOOFING_FEATURES.SIGNAL,
    )

    def validate(self) -> None:
        SpoofingStrategyConfig.validate(self)

        unit_fields = {
            "min_absorption_score": self.min_absorption_score,
            "min_absorption_confidence": self.min_absorption_confidence,
            "min_pull_ratio": self.min_pull_ratio,
            "max_fill_ratio": self.max_fill_ratio,
            "min_cancel_to_fill_ratio": self.min_cancel_to_fill_ratio,
            "min_fake_liquidity_detector_score": self.min_fake_liquidity_detector_score,
            "min_fake_liquidity_detector_confidence": self.min_fake_liquidity_detector_confidence,
            "strong_cancel_to_fill_bonus": self.strong_cancel_to_fill_bonus,
            "low_fill_bonus": self.low_fill_bonus,
            "strong_pull_bonus": self.strong_pull_bonus,
            "detector_bonus": self.detector_bonus,
            "reaction_bonus": self.reaction_bonus,
            "directional_reaction_bonus": self.directional_reaction_bonus,
            "notional_bonus": self.notional_bonus,
            "strong_cancel_to_fill_threshold": self.strong_cancel_to_fill_threshold,
            "very_low_fill_ratio_threshold": self.very_low_fill_ratio_threshold,
            "strong_pull_ratio_threshold": self.strong_pull_ratio_threshold,
        }
        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        non_negative_fields = {
            "min_wall_notional": self.min_wall_notional,
            "min_pulled_notional": self.min_pulled_notional,
            "min_price_reaction_bps": self.min_price_reaction_bps,
        }
        for field_name, value in non_negative_fields.items():
            if float(value) < 0.0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        hint_fields = {
            "entry_offset_bps_hint": self.entry_offset_bps_hint,
            "stop_buffer_bps_hint": self.stop_buffer_bps_hint,
            "take_profit_bps_hint": self.take_profit_bps_hint,
            "absorption_tp_multiplier_hint": self.absorption_tp_multiplier_hint,
        }
        for field_name, value in hint_fields.items():
            if value is not None and value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        score_weights = {
            "score_base_weight": self.score_base_weight,
            "score_absorption_weight": self.score_absorption_weight,
            "score_cancel_fill_weight": self.score_cancel_fill_weight,
            "score_fill_weight": self.score_fill_weight,
            "score_reaction_weight": self.score_reaction_weight,
            "score_detector_weight": self.score_detector_weight,
            "score_freshness_weight": self.score_freshness_weight,
        }
        confidence_weights = {
            "confidence_primary_weight": self.confidence_primary_weight,
            "confidence_context_weight": self.confidence_context_weight,
            "confidence_confirmation_weight": self.confidence_confirmation_weight,
            "confidence_freshness_weight": self.confidence_freshness_weight,
        }

        for field_name, value in {**score_weights, **confidence_weights}.items():
            if float(value) < 0.0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        if sum(score_weights.values()) <= 0:
            raise StrategyConfigError("score weights sum must be > 0")

        if sum(confidence_weights.values()) <= 0:
            raise StrategyConfigError("confidence weights sum must be > 0")

        for attr in (
            "tag_spoofing_absorption_reversal",
            "tag_fake_absorption",
            "tag_absorption_failed",
            "tag_cancel_to_fill",
            "tag_low_fill",
            "tag_unwind",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")

        if not self.required_spoofing_features:
            raise StrategyConfigError("required_spoofing_features cannot be empty")


class SpoofingAbsorptionReversalStrategy(SpoofingTradingStrategy):
    """
    Unified fake-absorption reversal strategy.

    Input:
        StrategyContext with FeatureSource.SPOOFING domain data / features.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, filters, confluence, building and risk payloads.
    """

    component_namespace = "strategy.spoofing.absorption_reversal"
    category: StrategyCategory = StrategyCategory.SPOOFING
    default_setup_type: SetupType = SetupType.REVERSAL

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        spoofing_config: SpoofingAbsorptionReversalStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_spoofing_config = (
            spoofing_config or SpoofingAbsorptionReversalStrategyConfig()
        )
        resolved_spoofing_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            spoofing_config=resolved_spoofing_config,
            service_name=service_name,
        )

        self.absorption_config: SpoofingAbsorptionReversalStrategyConfig = (
            resolved_spoofing_config
        )

    @property
    def strategy_name(self) -> str:
        return "spoofing_absorption_reversal"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.SPOOFING,
            timeframe=Timeframe.M1,
            tags=[
                self.absorption_config.tag_spoofing,
                self.absorption_config.tag_absorption,
                self.absorption_config.tag_fake_absorption,
                self.absorption_config.tag_reversal,
                self.absorption_config.tag_spoofing_absorption_reversal,
                "analytics_spoofing",
            ],
            version="2.0.0",
            description=(
                "Interprets fake absorption / fake liquidity wall failure from "
                "normalized StrategyContext and returns internal StrategySignal."
            ),
            required_features=set(self.required_features()),
            supported_regimes={
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.BREAKOUT,
                MarketRegime.SQUEEZE,
                MarketRegime.RANGING,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.UNKNOWN,
            },
            metadata={
                "source": "analytics.spoofing",
                "strategy_type": "spoofing_absorption_reversal",
                "base_class": "SpoofingTradingStrategy",
                "canonical_payload": "SpoofingCompositeSnapshot",
                "uses_fake_absorption": True,
                "uses_fake_liquidity_detector": True,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(
            self.absorption_config.required_spoofing_features
        )

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        self.validate_context_requirements(context)

        if not self.has_any_spoofing_data(
            context,
            tuple(self.absorption_config.required_spoofing_features),
        ):
            return None

        if self.has_stale_spoofing_features(
            context,
            tuple(self.absorption_config.required_spoofing_features),
        ):
            return None

        payload = self._extract_payload(context)
        if payload is None:
            return None

        if is_stale(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.absorption_config.stale_feature_max_age_seconds,
        ):
            return None

        rejection = quality_filter_reason(
            payload.snapshot.raw_signal,
            min_score=max(
                self.absorption_config.min_score,
                self.absorption_config.min_absorption_score,
            ),
            min_confidence=max(
                self.absorption_config.min_confidence,
                self.absorption_config.min_absorption_confidence,
            ),
            allowed_severities=self.absorption_config.allowed_severities,
            min_detector_count=self.absorption_config.min_detector_count,
            min_agreement_ratio=self.absorption_config.min_agreement_ratio,
            min_average_confidence=self.absorption_config.min_average_confidence,
            require_score_passed=self.absorption_config.require_score_passed,
            stale_after_seconds=self.absorption_config.stale_feature_max_age_seconds,
            now=context.timestamp,
        )
        if rejection is not None:
            return None

        if not self.accepts_spoofing_snapshot(payload.snapshot):
            return None

        if not self._supports_snapshot(payload.snapshot):
            return None

        if not self._passes_absorption_filters(payload):
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
        )

        if breakdown.score < self.absorption_config.min_absorption_score:
            return None

        if breakdown.confidence < self.absorption_config.min_absorption_confidence:
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload)

        reasons = list(
            dict.fromkeys(
                [
                    "spoofing_absorption_reversal_signal",
                    f"side:{payload.side.value}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "spoofing_setup_family": "spoofing_absorption_reversal",
            "spoofing_strategy_version": "2.0.0",
            "contract": "spoofing",
            "contract_version": "strategy-domain-v1",
            "primary_section": "signal",
            "secondary_section": "features",
            "strategy_contract_role": "decision_module",
            "risk_ready_payload_owner": "SignalProcessor",
            "score_breakdown": breakdown.to_dict(),
            "tags": tags,
            "snapshot": serialize_for_metadata(payload.snapshot.to_dict()),
            "raw": serialize_for_metadata(payload.raw),
            "event_time": payload.event_time.isoformat() if payload.event_time else None,
            "spoofing_type": normalize_label(payload.snapshot.spoofing_type),
            "pattern": normalize_label(payload.snapshot.pattern),
            "spoofing_side": normalize_label(payload.snapshot.side),
            "mapped_side": payload.side.value,
            "score": payload.snapshot.score,
            "confidence": payload.snapshot.confidence,
            "pull_ratio": payload.pull_ratio,
            "fill_ratio": payload.fill_ratio,
            "cancel_to_fill_ratio": payload.cancel_to_fill_ratio,
            "price_reaction_bps": payload.price_reaction_bps,
            "signed_price_reaction_bps": payload.snapshot.signed_price_reaction_bps,
            "wall_notional": payload.wall_notional,
            "pulled_notional": payload.pulled_notional,
            "detector_count": detector_count(payload.snapshot.raw_signal),
            "detector_agreement_ratio": detector_agreement_ratio(payload.snapshot.raw_signal),
            "detector_average_confidence": detector_average_confidence(payload.snapshot.raw_signal),
            "execution_hints": self._execution_hints(),
        }

        return self.build_spoofing_signal(
            context=context,
            side=payload.side,
            confidence=breakdown.confidence,
            score=breakdown.score,
            setup_type=self.absorption_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.absorption_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_payload(
        self,
        context: StrategyContext,
    ) -> SpoofingAbsorptionReversalPayload | None:
        snapshot = self.resolve_spoofing_snapshot(context)
        if snapshot is None or not snapshot.has_minimum_data():
            return None

        side = spoofing_side_to_signal_side(snapshot.side)
        if not is_directional_side(side):
            return None

        event_time = (
            extract_event_time(snapshot.raw_signal)
            or snapshot.timestamp
            or context.timestamp
        )

        reasons = [
            "spoofing_absorption_context",
            f"spoofing_type:{normalize_label(snapshot.spoofing_type)}",
            f"pattern:{normalize_label(snapshot.pattern)}",
            f"spoofing_side:{normalize_label(snapshot.side)}",
            f"score:{snapshot.score:.4f}",
            f"confidence:{snapshot.confidence:.4f}",
        ]

        if snapshot.pattern is SpoofingPattern.FAKE_ABSORPTION:
            reasons.append("fake_absorption_pattern_detected")

        if is_fake_liquidity_signal(snapshot.raw_signal):
            reasons.append("fake_liquidity_context")

        return SpoofingAbsorptionReversalPayload(
            snapshot=snapshot,
            side=side,
            event_time=event_time,
            reasons=list(dict.fromkeys(reasons)),
            raw=snapshot.raw_signal,
        )

    # ------------------------------------------------------------------
    # Support / filters
    # ------------------------------------------------------------------

    def _supports_snapshot(
        self,
        snapshot: SpoofingCompositeSnapshot,
    ) -> bool:
        if snapshot.pattern is SpoofingPattern.FAKE_ABSORPTION:
            return True

        if snapshot.spoofing_type is SpoofingType.FAKE_LIQUIDITY:
            return True

        if snapshot.has_detector(SpoofingComponent.FAKE_LIQUIDITY_DETECTOR):
            return True

        return is_fake_liquidity_signal(snapshot.raw_signal)

    def _passes_absorption_filters(
        self,
        payload: SpoofingAbsorptionReversalPayload,
    ) -> bool:
        snapshot = payload.snapshot

        if self.absorption_config.require_fake_absorption_pattern:
            if snapshot.pattern is not SpoofingPattern.FAKE_ABSORPTION:
                return False

        if self.absorption_config.require_fake_liquidity_detector:
            if not snapshot.has_detector(SpoofingComponent.FAKE_LIQUIDITY_DETECTOR):
                return False

        if self.absorption_config.require_fake_liquidity_detector_passed:
            if not detector_passed(
                snapshot.raw_signal,
                SpoofingComponent.FAKE_LIQUIDITY_DETECTOR,
            ):
                return False

        if payload.pull_ratio < self.absorption_config.min_pull_ratio:
            return False

        if payload.fill_ratio > self.absorption_config.max_fill_ratio:
            return False

        if payload.cancel_to_fill_ratio < self.absorption_config.min_cancel_to_fill_ratio:
            return False

        if payload.wall_notional < self.absorption_config.min_wall_notional:
            return False

        if payload.pulled_notional < self.absorption_config.min_pulled_notional:
            return False

        if self.absorption_config.require_market_reaction:
            if payload.price_reaction_bps < self.absorption_config.min_price_reaction_bps:
                return False

        if self.absorption_config.require_directional_reaction_alignment:
            if not reaction_aligns_with_side(
                signed_reaction_bps=snapshot.signed_price_reaction_bps,
                side=payload.side,
                min_reaction_bps=self.absorption_config.min_price_reaction_bps,
            ):
                return False

        if detector_score(
            snapshot.raw_signal,
            SpoofingComponent.FAKE_LIQUIDITY_DETECTOR,
        ) < self.absorption_config.min_fake_liquidity_detector_score:
            if self.absorption_config.require_fake_liquidity_detector:
                return False

        if detector_confidence(
            snapshot.raw_signal,
            SpoofingComponent.FAKE_LIQUIDITY_DETECTOR,
        ) < self.absorption_config.min_fake_liquidity_detector_confidence:
            if self.absorption_config.require_fake_liquidity_detector:
                return False

        return True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: SpoofingAbsorptionReversalPayload,
    ) -> ScoreBreakdown:
        snapshot = payload.snapshot

        base_component = average_score(
            extract_score(snapshot.raw_signal),
            snapshot.score,
            snapshot.confidence,
        )
        absorption_component = average_score(
            payload.pull_ratio,
            payload.cancel_to_fill_ratio,
            payload.pulled_notional / max(payload.wall_notional, 1.0)
            if payload.wall_notional > 0
            else 0.0,
        )
        cancel_fill_component = payload.cancel_to_fill_ratio
        fill_component = unit_score(1.0 - payload.fill_ratio)
        reaction_component = unit_score(
            payload.price_reaction_bps
            / max(self.absorption_config.min_price_reaction_bps * 4.0, 0.01)
        )
        detector_component = average_score(
            detector_score(snapshot.raw_signal, SpoofingComponent.FAKE_LIQUIDITY_DETECTOR),
            detector_confidence(snapshot.raw_signal, SpoofingComponent.FAKE_LIQUIDITY_DETECTOR),
            detector_agreement_ratio(snapshot.raw_signal),
            detector_average_confidence(snapshot.raw_signal),
        )
        fresh_component = freshness_score(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.absorption_config.stale_feature_max_age_seconds,
        )

        components = {
            "base": base_component,
            "absorption": absorption_component,
            "cancel_fill": cancel_fill_component,
            "fill": fill_component,
            "reaction": reaction_component,
            "detector": detector_component,
            "freshness": fresh_component,
        }
        weights = {
            "base": self.absorption_config.score_base_weight,
            "absorption": self.absorption_config.score_absorption_weight,
            "cancel_fill": self.absorption_config.score_cancel_fill_weight,
            "fill": self.absorption_config.score_fill_weight,
            "reaction": self.absorption_config.score_reaction_weight,
            "detector": self.absorption_config.score_detector_weight,
            "freshness": self.absorption_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=base_component)
        confidence = confidence_from_components(
            primary=base_component,
            context=average_score(absorption_component, detector_component),
            confirmation=average_score(cancel_fill_component, fill_component, reaction_component),
            freshness=fresh_component,
            primary_weight=self.absorption_config.confidence_primary_weight,
            context_weight=self.absorption_config.confidence_context_weight,
            confirmation_weight=self.absorption_config.confidence_confirmation_weight,
            freshness_weight=self.absorption_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            "fake_absorption_reversal_context",
            f"side:{payload.side.value}",
            f"pull_ratio:{payload.pull_ratio:.4f}",
            f"fill_ratio:{payload.fill_ratio:.4f}",
            f"cancel_to_fill_ratio:{payload.cancel_to_fill_ratio:.4f}",
        ]

        if payload.cancel_to_fill_ratio >= self.absorption_config.strong_cancel_to_fill_threshold:
            score += self.absorption_config.strong_cancel_to_fill_bonus
            confirmations.append("strong_cancel_to_fill_ratio")

        if payload.fill_ratio <= self.absorption_config.very_low_fill_ratio_threshold:
            score += self.absorption_config.low_fill_bonus
            confirmations.append("very_low_fill_ratio")

        if payload.pull_ratio >= self.absorption_config.strong_pull_ratio_threshold:
            score += self.absorption_config.strong_pull_bonus
            confirmations.append("strong_pull_ratio")

        if snapshot.has_detector(SpoofingComponent.FAKE_LIQUIDITY_DETECTOR):
            score += self.absorption_config.detector_bonus
            confirmations.append("fake_liquidity_detector_context")

        if payload.price_reaction_bps >= self.absorption_config.min_price_reaction_bps:
            score += self.absorption_config.reaction_bonus
            confirmations.append("market_reaction_context")

        if reaction_aligns_with_side(
            signed_reaction_bps=snapshot.signed_price_reaction_bps,
            side=payload.side,
            min_reaction_bps=self.absorption_config.min_price_reaction_bps,
        ):
            score += self.absorption_config.directional_reaction_bonus
            confidence += min(0.03, self.absorption_config.directional_reaction_bonus)
            confirmations.append("directional_absorption_failure_reaction")

        if payload.pulled_notional > 0:
            score += self.absorption_config.notional_bonus
            confirmations.append("pulled_notional_confirmed")

        if payload.wall_notional > 0:
            reasons.append(f"wall_notional:{payload.wall_notional:.4f}")

        if payload.pulled_notional > 0:
            reasons.append(f"pulled_notional:{payload.pulled_notional:.4f}")

        return ScoreBreakdown(
            score=unit_score(score),
            confidence=unit_score(confidence),
            components=components,
            weights=weights,
            reasons=reasons,
            confirmations=list(dict.fromkeys(confirmations)),
        ).normalize()

    # ------------------------------------------------------------------
    # Source features / tags / execution hints
    # ------------------------------------------------------------------

    def _source_features(
        self,
        payload: SpoofingAbsorptionReversalPayload,
    ) -> list[str]:
        features = [
            *base_spoofing_source_features(),
            *fake_liquidity_source_features(),
            SPOOFING_FEATURES.SIGNAL,
            SPOOFING_FEATURES.SPOOFING_TYPE,
            SPOOFING_FEATURES.PATTERN,
            SPOOFING_FEATURES.SIDE,
            SPOOFING_FEATURES.SCORE,
            SPOOFING_FEATURES.CONFIDENCE,
            SPOOFING_FEATURES.PULL_RATIO,
            SPOOFING_FEATURES.FILL_RATIO,
            SPOOFING_FEATURES.CANCEL_TO_FILL_RATIO,
            SPOOFING_FEATURES.PRICE_REACTION_BPS,
            SPOOFING_FEATURES.WALL_NOTIONAL,
            SPOOFING_FEATURES.PULLED_NOTIONAL,
            SPOOFING_FEATURES.DETECTOR_RESULTS,
            SPOOFING_FEATURES.SCORE_BREAKDOWN,
        ]

        return list(dict.fromkeys(features))

    def _tags(
        self,
        payload: SpoofingAbsorptionReversalPayload,
    ) -> list[str]:
        tags = [
            self.absorption_config.tag_spoofing,
            self.absorption_config.tag_absorption,
            self.absorption_config.tag_fake_absorption,
            self.absorption_config.tag_reversal,
            self.absorption_config.tag_spoofing_absorption_reversal,
            self.absorption_config.tag_absorption_failed,
            self.absorption_config.tag_unwind,
            f"side:{payload.side.value}",
        ]

        if payload.cancel_to_fill_ratio >= self.absorption_config.min_cancel_to_fill_ratio:
            tags.append(self.absorption_config.tag_cancel_to_fill)

        if payload.fill_ratio <= self.absorption_config.very_low_fill_ratio_threshold:
            tags.append(self.absorption_config.tag_low_fill)

        if payload.snapshot.spoofing_type is not None:
            tags.append(f"type:{normalize_label(payload.snapshot.spoofing_type)}")

        if payload.snapshot.pattern is not None:
            tags.append(f"pattern:{normalize_label(payload.snapshot.pattern)}")

        return list(dict.fromkeys(tags))

    def _execution_hints(self) -> dict[str, Any]:
        """
        Execution hints only. Final EntryPlan/ExitPlan/RiskReadySignalPayload
        is owned by SignalProcessor / SignalBuilder.
        """
        return {
            "entry_offset_bps": self.absorption_config.entry_offset_bps_hint,
            "stop_buffer_bps": self.absorption_config.stop_buffer_bps_hint,
            "take_profit_bps": self.absorption_config.take_profit_bps_hint,
            "absorption_tp_multiplier": (
                self.absorption_config.absorption_tp_multiplier_hint
            ),
        }