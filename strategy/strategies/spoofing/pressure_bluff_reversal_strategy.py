# trading_system/strategy/strategies/spoofing/pressure_bluff_reversal_strategy.py

from __future__ import annotations
import logging

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
    confidence_from_components,
    detector_agreement_ratio,
    detector_average_confidence,
    detector_confidence,
    detector_count,
    detector_passed,
    detector_score,
    extract_distance_from_mid_bps,
    extract_event_time,
    extract_fill_ratio,
    extract_price_reaction_bps,
    extract_pressure_flip_strength,
    extract_pull_ratio,
    extract_score,
    freshness_score,
    is_directional_side,
    is_pressure_bluff_signal,
    is_stale,
    normalize_label,
    pressure_bluff_source_features,
    quality_filter_reason,
    reaction_aligns_with_side,
    serialize_for_metadata,
    spoofing_side_to_signal_side,
    unit_score,
    weighted_score,
)


@dataclass(slots=True)
class PressureBluffReversalPayload:
    """
    Normalized strategy-level payload для pressure-bluff reversal.

    Direction convention:
    - fake ASK pressure disappears / flips -> fake resistance removed -> LONG;
    - fake BID pressure disappears / flips -> fake support removed -> SHORT.
    """
    _logger = logging.getLogger(__name__ + ".PressureBluffReversalPayload")

    snapshot: SpoofingCompositeSnapshot
    side: SignalSide

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def pressure_flip_strength(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PressureBluffReversalPayload.pressure_flip_strength")
        return extract_pressure_flip_strength(self.snapshot.raw_signal)

    @property
    def price_reaction_bps(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PressureBluffReversalPayload.price_reaction_bps")
        return extract_price_reaction_bps(self.snapshot.raw_signal)

    @property
    def distance_from_mid_bps(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PressureBluffReversalPayload.distance_from_mid_bps")
        return extract_distance_from_mid_bps(self.snapshot.raw_signal)

    @property
    def pull_ratio(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PressureBluffReversalPayload.pull_ratio")
        return extract_pull_ratio(self.snapshot.raw_signal)

    @property
    def fill_ratio(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PressureBluffReversalPayload.fill_ratio")
        return extract_fill_ratio(self.snapshot.raw_signal)


@dataclass(slots=True)
class PressureBluffReversalStrategyConfig(SpoofingStrategyConfig):
    """
    Unified pressure-bluff reversal strategy config.

    Strategy idea:
    - analytics.spoofing detects fake pressure / pressure flip;
    - fake bid/ask pressure disappears or flips;
    - market reaction confirms unwind/reversal direction;
    - strategy returns internal StrategySignal only.
    """
    _logger = logging.getLogger(__name__ + ".PressureBluffReversalStrategyConfig")

    min_pressure_bluff_score: float = 0.70
    min_pressure_bluff_confidence: float = 0.60

    min_pressure_flip_strength: float = 0.40
    min_price_reaction_bps: float = 1.5
    max_distance_from_mid_bps: float = 4.0

    min_pull_ratio: float = 0.0
    max_fill_ratio: float = 1.0

    require_flip_pressure_detector: bool = False
    require_flip_pressure_detector_passed: bool = False
    require_reaction_for_pressure_bluff: bool = True
    require_has_reversal_for_pressure_bluff: bool = False
    require_directional_reaction_alignment: bool = False

    min_flip_pressure_detector_score: float = 0.0
    min_flip_pressure_detector_confidence: float = 0.0

    score_base_weight: float = 0.26
    score_pressure_weight: float = 0.24
    score_reaction_weight: float = 0.18
    score_distance_weight: float = 0.10
    score_detector_weight: float = 0.12
    score_pull_fill_weight: float = 0.04
    score_freshness_weight: float = 0.06

    confidence_primary_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    strong_pressure_flip_bonus: float = 0.05
    detector_bonus: float = 0.04
    directional_reaction_bonus: float = 0.04
    close_to_mid_bonus: float = 0.03
    pull_confirmation_bonus: float = 0.03
    low_fill_bonus: float = 0.03

    strong_pressure_flip_threshold: float = 0.65
    close_to_mid_threshold_bps: float = 2.0
    strong_pull_ratio_threshold: float = 0.55
    low_fill_ratio_threshold: float = 0.35

    entry_offset_bps_hint: float | None = None
    stop_buffer_bps_hint: float | None = None
    take_profit_bps_hint: float | None = None
    reaction_tp_multiplier_hint: float | None = None

    tag_pressure_bluff_reversal: str = "pressure_bluff_reversal"
    tag_pressure_flip: str = "pressure_flip"
    tag_fake_pressure: str = "fake_pressure"
    tag_reaction_confirmed: str = "reaction_confirmed"
    tag_close_to_mid: str = "close_to_mid"

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.REVERSAL

    required_spoofing_features: tuple[str, ...] = (
        SPOOFING_FEATURES.SIGNAL,
    )

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PressureBluffReversalStrategyConfig.validate")
        SpoofingStrategyConfig.validate(self)

        unit_fields = {
            "min_pressure_bluff_score": self.min_pressure_bluff_score,
            "min_pressure_bluff_confidence": self.min_pressure_bluff_confidence,
            "min_pressure_flip_strength": self.min_pressure_flip_strength,
            "min_pull_ratio": self.min_pull_ratio,
            "max_fill_ratio": self.max_fill_ratio,
            "min_flip_pressure_detector_score": self.min_flip_pressure_detector_score,
            "min_flip_pressure_detector_confidence": self.min_flip_pressure_detector_confidence,
            "strong_pressure_flip_bonus": self.strong_pressure_flip_bonus,
            "detector_bonus": self.detector_bonus,
            "directional_reaction_bonus": self.directional_reaction_bonus,
            "close_to_mid_bonus": self.close_to_mid_bonus,
            "pull_confirmation_bonus": self.pull_confirmation_bonus,
            "low_fill_bonus": self.low_fill_bonus,
            "strong_pressure_flip_threshold": self.strong_pressure_flip_threshold,
            "strong_pull_ratio_threshold": self.strong_pull_ratio_threshold,
            "low_fill_ratio_threshold": self.low_fill_ratio_threshold,
        }
        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        non_negative_fields = {
            "min_price_reaction_bps": self.min_price_reaction_bps,
            "max_distance_from_mid_bps": self.max_distance_from_mid_bps,
            "close_to_mid_threshold_bps": self.close_to_mid_threshold_bps,
        }
        for field_name, value in non_negative_fields.items():
            if float(value) < 0.0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        hint_fields = {
            "entry_offset_bps_hint": self.entry_offset_bps_hint,
            "stop_buffer_bps_hint": self.stop_buffer_bps_hint,
            "take_profit_bps_hint": self.take_profit_bps_hint,
            "reaction_tp_multiplier_hint": self.reaction_tp_multiplier_hint,
        }
        for field_name, value in hint_fields.items():
            if value is not None and value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        score_weights = {
            "score_base_weight": self.score_base_weight,
            "score_pressure_weight": self.score_pressure_weight,
            "score_reaction_weight": self.score_reaction_weight,
            "score_distance_weight": self.score_distance_weight,
            "score_detector_weight": self.score_detector_weight,
            "score_pull_fill_weight": self.score_pull_fill_weight,
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
            "tag_pressure_bluff_reversal",
            "tag_pressure_flip",
            "tag_fake_pressure",
            "tag_reaction_confirmed",
            "tag_close_to_mid",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")

        if not self.required_spoofing_features:
            raise StrategyConfigError("required_spoofing_features cannot be empty")


class PressureBluffReversalStrategy(SpoofingTradingStrategy):
    """
    Unified pressure-bluff reversal strategy.

    Input:
        StrategyContext with FeatureSource.SPOOFING domain data / features.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, filters, confluence, building and risk payloads.
    """
    _logger = logging.getLogger(__name__ + ".PressureBluffReversalStrategy")

    component_namespace = "strategy.spoofing.pressure_bluff_reversal"
    category: StrategyCategory = StrategyCategory.SPOOFING
    default_setup_type: SetupType = SetupType.REVERSAL

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        spoofing_config: PressureBluffReversalStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PressureBluffReversalStrategy.__init__")
        resolved_spoofing_config = (
            spoofing_config or PressureBluffReversalStrategyConfig()
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

        self.pressure_config: PressureBluffReversalStrategyConfig = (
            resolved_spoofing_config
        )

    @property
    def strategy_name(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PressureBluffReversalStrategy.strategy_name")
        return "pressure_bluff_reversal"

    @property
    def metadata(self) -> StrategyMetadata:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PressureBluffReversalStrategy.metadata")
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.SPOOFING,
            timeframe=Timeframe.M1,
            tags=[
                self.pressure_config.tag_spoofing,
                self.pressure_config.tag_pressure_bluff,
                self.pressure_config.tag_reversal,
                self.pressure_config.tag_pressure_bluff_reversal,
                self.pressure_config.tag_pressure_flip,
                "analytics_spoofing",
            ],
            version="2.0.0",
            description=(
                "Interprets flip-pressure / pressure-bluff spoofing signals "
                "from normalized StrategyContext and returns internal StrategySignal."
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
                "strategy_type": "pressure_bluff_reversal",
                "base_class": "SpoofingTradingStrategy",
                "canonical_payload": "SpoofingCompositeSnapshot",
                "uses_flip_pressure_detector": True,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PressureBluffReversalStrategy.required_features")
        base_required = super().required_features()
        return set(base_required).union(self.pressure_config.required_spoofing_features)

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PressureBluffReversalStrategy.generate_signal")
        self.validate_context_requirements(context)

        if not self.has_any_spoofing_data(
            context,
            tuple(self.pressure_config.required_spoofing_features),
        ):
            return None

        if self.has_stale_spoofing_features(
            context,
            tuple(self.pressure_config.required_spoofing_features),
        ):
            return None

        payload = self._extract_payload(context)
        if payload is None:
            return None

        if is_stale(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.pressure_config.stale_feature_max_age_seconds,
        ):
            return None

        rejection = quality_filter_reason(
            payload.snapshot.raw_signal,
            min_score=max(
                self.pressure_config.min_score,
                self.pressure_config.min_pressure_bluff_score,
            ),
            min_confidence=max(
                self.pressure_config.min_confidence,
                self.pressure_config.min_pressure_bluff_confidence,
            ),
            allowed_severities=self.pressure_config.allowed_severities,
            min_detector_count=self.pressure_config.min_detector_count,
            min_agreement_ratio=self.pressure_config.min_agreement_ratio,
            min_average_confidence=self.pressure_config.min_average_confidence,
            require_score_passed=self.pressure_config.require_score_passed,
            stale_after_seconds=self.pressure_config.stale_feature_max_age_seconds,
            now=context.timestamp,
        )
        if rejection is not None:
            return None

        if not self.accepts_spoofing_snapshot(payload.snapshot):
            return None

        if not self._supports_snapshot(payload.snapshot):
            return None

        if not self._passes_pressure_filters(payload):
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
        )

        if breakdown.score < self.pressure_config.min_pressure_bluff_score:
            return None

        if breakdown.confidence < self.pressure_config.min_pressure_bluff_confidence:
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload)

        reasons = list(
            dict.fromkeys(
                [
                    "pressure_bluff_reversal_signal",
                    f"side:{payload.side.value}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "spoofing_setup_family": "pressure_bluff_reversal",
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
            "pressure_flip_strength": payload.pressure_flip_strength,
            "price_reaction_bps": payload.price_reaction_bps,
            "signed_price_reaction_bps": payload.snapshot.signed_price_reaction_bps,
            "distance_from_mid_bps": payload.distance_from_mid_bps,
            "pull_ratio": payload.pull_ratio,
            "fill_ratio": payload.fill_ratio,
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
            setup_type=self.pressure_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.pressure_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_payload(
        self,
        context: StrategyContext,
    ) -> PressureBluffReversalPayload | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PressureBluffReversalStrategy._extract_payload")
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
            "pressure_bluff_context",
            f"spoofing_type:{normalize_label(snapshot.spoofing_type)}",
            f"pattern:{normalize_label(snapshot.pattern)}",
            f"spoofing_side:{normalize_label(snapshot.side)}",
            f"score:{snapshot.score:.4f}",
            f"confidence:{snapshot.confidence:.4f}",
        ]

        if is_pressure_bluff_signal(snapshot.raw_signal):
            reasons.append("pressure_bluff_signal_detected")

        return PressureBluffReversalPayload(
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PressureBluffReversalStrategy._supports_snapshot")
        return (
            snapshot.spoofing_type is SpoofingType.FLIP_PRESSURE
            or snapshot.pattern is SpoofingPattern.PRESSURE_BLUFF
            or snapshot.has_detector(SpoofingComponent.FLIP_PRESSURE_DETECTOR)
            or is_pressure_bluff_signal(snapshot.raw_signal)
        )

    def _passes_pressure_filters(
        self,
        payload: PressureBluffReversalPayload,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PressureBluffReversalStrategy._passes_pressure_filters")
        snapshot = payload.snapshot

        if self.pressure_config.require_flip_pressure_detector:
            if not snapshot.has_detector(SpoofingComponent.FLIP_PRESSURE_DETECTOR):
                return False

        if self.pressure_config.require_flip_pressure_detector_passed:
            if not detector_passed(
                snapshot.raw_signal,
                SpoofingComponent.FLIP_PRESSURE_DETECTOR,
            ):
                return False

        if payload.pressure_flip_strength < self.pressure_config.min_pressure_flip_strength:
            return False

        if self.pressure_config.require_reaction_for_pressure_bluff:
            if payload.price_reaction_bps < self.pressure_config.min_price_reaction_bps:
                return False

        if payload.distance_from_mid_bps > self.pressure_config.max_distance_from_mid_bps:
            return False

        if payload.pull_ratio < self.pressure_config.min_pull_ratio:
            return False

        if payload.fill_ratio > self.pressure_config.max_fill_ratio:
            return False

        if self.pressure_config.require_directional_reaction_alignment:
            if not reaction_aligns_with_side(
                signed_reaction_bps=snapshot.signed_price_reaction_bps,
                side=payload.side,
                min_reaction_bps=self.pressure_config.min_price_reaction_bps,
            ):
                return False

        if self.pressure_config.require_has_reversal_for_pressure_bluff:
            if not reaction_aligns_with_side(
                signed_reaction_bps=snapshot.signed_price_reaction_bps,
                side=payload.side,
                min_reaction_bps=0.0,
            ):
                return False

        if detector_score(
            snapshot.raw_signal,
            SpoofingComponent.FLIP_PRESSURE_DETECTOR,
        ) < self.pressure_config.min_flip_pressure_detector_score:
            if self.pressure_config.require_flip_pressure_detector:
                return False

        if detector_confidence(
            snapshot.raw_signal,
            SpoofingComponent.FLIP_PRESSURE_DETECTOR,
        ) < self.pressure_config.min_flip_pressure_detector_confidence:
            if self.pressure_config.require_flip_pressure_detector:
                return False

        return True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: PressureBluffReversalPayload,
    ) -> ScoreBreakdown:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PressureBluffReversalStrategy._build_score_breakdown")
        snapshot = payload.snapshot

        base_component = average_score(
            extract_score(snapshot.raw_signal),
            snapshot.score,
            snapshot.confidence,
        )
        pressure_component = payload.pressure_flip_strength
        reaction_component = unit_score(
            payload.price_reaction_bps
            / max(self.pressure_config.min_price_reaction_bps * 4.0, 0.01)
        )
        distance_component = self._distance_component(payload)
        detector_component = average_score(
            detector_score(
                snapshot.raw_signal,
                SpoofingComponent.FLIP_PRESSURE_DETECTOR,
            ),
            detector_confidence(
                snapshot.raw_signal,
                SpoofingComponent.FLIP_PRESSURE_DETECTOR,
            ),
            detector_agreement_ratio(snapshot.raw_signal),
            detector_average_confidence(snapshot.raw_signal),
        )
        pull_fill_component = average_score(
            payload.pull_ratio,
            1.0 - payload.fill_ratio,
        )
        fresh_component = freshness_score(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.pressure_config.stale_feature_max_age_seconds,
        )

        components = {
            "base": base_component,
            "pressure": pressure_component,
            "reaction": reaction_component,
            "distance": distance_component,
            "detector": detector_component,
            "pull_fill": pull_fill_component,
            "freshness": fresh_component,
        }
        weights = {
            "base": self.pressure_config.score_base_weight,
            "pressure": self.pressure_config.score_pressure_weight,
            "reaction": self.pressure_config.score_reaction_weight,
            "distance": self.pressure_config.score_distance_weight,
            "detector": self.pressure_config.score_detector_weight,
            "pull_fill": self.pressure_config.score_pull_fill_weight,
            "freshness": self.pressure_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=base_component)
        confidence = confidence_from_components(
            primary=base_component,
            context=average_score(pressure_component, detector_component),
            confirmation=average_score(reaction_component, distance_component),
            freshness=fresh_component,
            primary_weight=self.pressure_config.confidence_primary_weight,
            context_weight=self.pressure_config.confidence_context_weight,
            confirmation_weight=self.pressure_config.confidence_confirmation_weight,
            freshness_weight=self.pressure_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            "pressure_bluff_reversal_context",
            f"side:{payload.side.value}",
            f"pressure_flip_strength:{payload.pressure_flip_strength:.4f}",
            f"price_reaction_bps:{payload.price_reaction_bps:.4f}",
            f"distance_from_mid_bps:{payload.distance_from_mid_bps:.4f}",
        ]

        if payload.pressure_flip_strength >= self.pressure_config.strong_pressure_flip_threshold:
            score += self.pressure_config.strong_pressure_flip_bonus
            confirmations.append("strong_pressure_flip")

        if payload.distance_from_mid_bps <= self.pressure_config.close_to_mid_threshold_bps:
            score += self.pressure_config.close_to_mid_bonus
            confirmations.append("pressure_close_to_mid")

        if snapshot.has_detector(SpoofingComponent.FLIP_PRESSURE_DETECTOR):
            score += self.pressure_config.detector_bonus
            confirmations.append("flip_pressure_detector_context")

        if reaction_aligns_with_side(
            signed_reaction_bps=snapshot.signed_price_reaction_bps,
            side=payload.side,
            min_reaction_bps=self.pressure_config.min_price_reaction_bps,
        ):
            score += self.pressure_config.directional_reaction_bonus
            confidence += min(0.03, self.pressure_config.directional_reaction_bonus)
            confirmations.append("directional_reaction_alignment")

        if payload.pull_ratio >= self.pressure_config.strong_pull_ratio_threshold:
            score += self.pressure_config.pull_confirmation_bonus
            confirmations.append("pull_confirms_pressure_bluff")

        if payload.fill_ratio <= self.pressure_config.low_fill_ratio_threshold:
            score += self.pressure_config.low_fill_bonus
            confirmations.append("low_fill_confirms_fake_pressure")

        if payload.pull_ratio > 0:
            reasons.append(f"pull_ratio:{payload.pull_ratio:.4f}")

        if payload.fill_ratio > 0:
            reasons.append(f"fill_ratio:{payload.fill_ratio:.4f}")

        return ScoreBreakdown(
            score=unit_score(score),
            confidence=unit_score(confidence),
            components=components,
            weights=weights,
            reasons=reasons,
            confirmations=list(dict.fromkeys(confirmations)),
        ).normalize()

    def _distance_component(
        self,
        payload: PressureBluffReversalPayload,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PressureBluffReversalStrategy._distance_component")
        max_distance = max(self.pressure_config.max_distance_from_mid_bps, 0.0001)
        return unit_score(1.0 - (payload.distance_from_mid_bps / max_distance))

    # ------------------------------------------------------------------
    # Source features / tags / execution hints
    # ------------------------------------------------------------------

    def _source_features(
        self,
        payload: PressureBluffReversalPayload,
    ) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PressureBluffReversalStrategy._source_features")
        features = [
            *pressure_bluff_source_features(),
            SPOOFING_FEATURES.SIGNAL,
            SPOOFING_FEATURES.SPOOFING_TYPE,
            SPOOFING_FEATURES.PATTERN,
            SPOOFING_FEATURES.SIDE,
            SPOOFING_FEATURES.SCORE,
            SPOOFING_FEATURES.CONFIDENCE,
            SPOOFING_FEATURES.PRESSURE_FLIP_STRENGTH,
            SPOOFING_FEATURES.PRICE_REACTION_BPS,
            SPOOFING_FEATURES.DISTANCE_FROM_MID_BPS,
            SPOOFING_FEATURES.PULL_RATIO,
            SPOOFING_FEATURES.FILL_RATIO,
            SPOOFING_FEATURES.DETECTOR_RESULTS,
            SPOOFING_FEATURES.SCORE_BREAKDOWN,
        ]

        return list(dict.fromkeys(features))

    def _tags(
        self,
        payload: PressureBluffReversalPayload,
    ) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PressureBluffReversalStrategy._tags")
        tags = [
            self.pressure_config.tag_spoofing,
            self.pressure_config.tag_pressure_bluff,
            self.pressure_config.tag_reversal,
            self.pressure_config.tag_pressure_bluff_reversal,
            self.pressure_config.tag_pressure_flip,
            self.pressure_config.tag_fake_pressure,
            f"side:{payload.side.value}",
        ]

        if payload.price_reaction_bps >= self.pressure_config.min_price_reaction_bps:
            tags.append(self.pressure_config.tag_reaction_confirmed)

        if payload.distance_from_mid_bps <= self.pressure_config.close_to_mid_threshold_bps:
            tags.append(self.pressure_config.tag_close_to_mid)

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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PressureBluffReversalStrategy._execution_hints")
        return {
            "entry_offset_bps": self.pressure_config.entry_offset_bps_hint,
            "stop_buffer_bps": self.pressure_config.stop_buffer_bps_hint,
            "take_profit_bps": self.pressure_config.take_profit_bps_hint,
            "reaction_tp_multiplier": self.pressure_config.reaction_tp_multiplier_hint,
        }