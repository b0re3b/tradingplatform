# trading_system/strategy/strategies/spoofing/composite_spoofing_strategy.py

from __future__ import annotations
import logging

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from analytics.spoofing.enums import SpoofingComponent, SpoofingType
from core.event_bus import EventBus
from core.scheduler import Scheduler
from .base import (
    SPOOFING_FEATURES,
    SpoofingCompositeSnapshot,
    SpoofingStrategyConfig,
    SpoofingTradingStrategy,
)
from .utils import (
    ScoreBreakdown,
    average_score,
    composite_spoofing_source_features,
    confidence_from_components,
    detector_agreement_ratio,
    detector_average_confidence,
    detector_confidence,
    detector_count,
    detector_passed,
    detector_score,
    extract_cancel_to_fill_ratio,
    extract_distance_from_mid_bps,
    extract_event_time,
    extract_fill_ratio,
    extract_layer_count,
    extract_layer_price_span_bps,
    extract_lifetime_ms,
    extract_price_reaction_bps,
    extract_pressure_flip_strength,
    extract_pull_ratio,
    extract_pulled_notional,
    extract_score,
    extract_wall_notional,
    fake_liquidity_source_features,
    freshness_score,
    is_composite_signal,
    is_directional_side,
    is_fake_liquidity_signal,
    is_layering_signal,
    is_order_pull_signal,
    is_pressure_bluff_signal,
    is_stale,
    layering_source_features,
    normalize_label,
    order_pull_source_features,
    pressure_bluff_source_features,
    quality_filter_reason,
    reaction_aligns_with_side,
    serialize_for_metadata,
    spoofing_side_to_signal_side,
    unit_score,
    weighted_score,
)
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


@dataclass(slots=True)
class CompositeSpoofingPayload:
    """
    Normalized strategy-level payload для analytics-level composite spoofing.

    Важливо: це не strategy confluence. Цей клас читає лише composite signal,
    який уже сформував analytics.spoofing.
    """
    _logger = logging.getLogger(__name__ + ".CompositeSpoofingPayload")

    snapshot: SpoofingCompositeSnapshot
    side: SignalSide

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def pull_ratio(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingPayload.pull_ratio")
        return extract_pull_ratio(self.snapshot.raw_signal)

    @property
    def fill_ratio(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingPayload.fill_ratio")
        return extract_fill_ratio(self.snapshot.raw_signal)

    @property
    def price_reaction_bps(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingPayload.price_reaction_bps")
        return extract_price_reaction_bps(self.snapshot.raw_signal)

    @property
    def lifetime_ms(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingPayload.lifetime_ms")
        return extract_lifetime_ms(self.snapshot.raw_signal)

    @property
    def wall_notional(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingPayload.wall_notional")
        return extract_wall_notional(self.snapshot.raw_signal)

    @property
    def pulled_notional(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingPayload.pulled_notional")
        return extract_pulled_notional(self.snapshot.raw_signal)

    @property
    def cancel_to_fill_ratio(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingPayload.cancel_to_fill_ratio")
        return extract_cancel_to_fill_ratio(self.snapshot.raw_signal)

    @property
    def distance_from_mid_bps(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingPayload.distance_from_mid_bps")
        return extract_distance_from_mid_bps(self.snapshot.raw_signal)

    @property
    def layer_count(self) -> int:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingPayload.layer_count")
        return extract_layer_count(self.snapshot.raw_signal)

    @property
    def layer_price_span_bps(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingPayload.layer_price_span_bps")
        return extract_layer_price_span_bps(self.snapshot.raw_signal)

    @property
    def pressure_flip_strength(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingPayload.pressure_flip_strength")
        return extract_pressure_flip_strength(self.snapshot.raw_signal)


@dataclass(slots=True)
class CompositeSpoofingStrategyConfig(SpoofingStrategyConfig):
    """
    Unified composite spoofing strategy config.

    Strategy idea:
    - analytics.spoofing already produced a composite spoofing signal;
    - several spoofing detectors agree on direction and quality;
    - this strategy maps analytics composite signal to internal StrategySignal;
    - SignalProcessor still owns strategy confluence, portfolio coordination,
      filtering and risk-ready conversion.
    """
    _logger = logging.getLogger(__name__ + ".CompositeSpoofingStrategyConfig")

    min_composite_score: float = 0.72
    min_composite_confidence: float = 0.62

    min_composite_detector_count: int = 2
    min_composite_agreement_ratio: float = 0.50
    min_composite_average_confidence: float = 0.55

    require_composite_type_or_pattern: bool = False
    require_score_passed: bool = False
    require_directional_side: bool = True
    require_directional_reaction_alignment: bool = False

    allow_order_pull_component: bool = True
    allow_fake_liquidity_component: bool = True
    allow_flip_pressure_component: bool = True
    allow_layering_component: bool = True

    require_any_allowed_component: bool = True
    require_market_reaction: bool = False
    min_price_reaction_bps: float = 0.0

    min_pull_ratio: float = 0.0
    max_fill_ratio: float = 1.0
    min_cancel_to_fill_ratio: float = 0.0

    score_base_weight: float = 0.26
    score_detector_weight: float = 0.24
    score_agreement_weight: float = 0.18
    score_features_weight: float = 0.14
    score_reaction_weight: float = 0.10
    score_freshness_weight: float = 0.08

    confidence_primary_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    composite_bonus: float = 0.05
    multi_detector_bonus: float = 0.04
    high_agreement_bonus: float = 0.04
    order_pull_component_bonus: float = 0.03
    fake_liquidity_component_bonus: float = 0.03
    pressure_bluff_component_bonus: float = 0.03
    layering_component_bonus: float = 0.03
    directional_reaction_bonus: float = 0.04

    high_agreement_threshold: float = 0.75
    strong_detector_count_threshold: int = 3

    entry_offset_bps_hint: float | None = None
    stop_buffer_bps_hint: float | None = None
    take_profit_bps_hint: float | None = None
    composite_tp_multiplier_hint: float | None = None

    tag_composite_spoofing: str = "composite_spoofing"
    tag_multi_detector: str = "multi_detector"
    tag_detector_agreement: str = "detector_agreement"
    tag_order_pull_component: str = "order_pull_component"
    tag_fake_liquidity_component: str = "fake_liquidity_component"
    tag_pressure_bluff_component: str = "pressure_bluff_component"
    tag_layering_component: str = "layering_component"

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.REVERSAL

    required_spoofing_features: tuple[str, ...] = (
        SPOOFING_FEATURES.SIGNAL,
    )

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingStrategyConfig.validate")
        SpoofingStrategyConfig.validate(self)

        unit_fields = {
            "min_composite_score": self.min_composite_score,
            "min_composite_confidence": self.min_composite_confidence,
            "min_composite_agreement_ratio": self.min_composite_agreement_ratio,
            "min_composite_average_confidence": self.min_composite_average_confidence,
            "min_pull_ratio": self.min_pull_ratio,
            "max_fill_ratio": self.max_fill_ratio,
            "min_cancel_to_fill_ratio": self.min_cancel_to_fill_ratio,
            "composite_bonus": self.composite_bonus,
            "multi_detector_bonus": self.multi_detector_bonus,
            "high_agreement_bonus": self.high_agreement_bonus,
            "order_pull_component_bonus": self.order_pull_component_bonus,
            "fake_liquidity_component_bonus": self.fake_liquidity_component_bonus,
            "pressure_bluff_component_bonus": self.pressure_bluff_component_bonus,
            "layering_component_bonus": self.layering_component_bonus,
            "directional_reaction_bonus": self.directional_reaction_bonus,
            "high_agreement_threshold": self.high_agreement_threshold,
        }
        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        if self.min_composite_detector_count < 0:
            raise StrategyConfigError("min_composite_detector_count must be >= 0")

        if self.strong_detector_count_threshold < 1:
            raise StrategyConfigError("strong_detector_count_threshold must be >= 1")

        if self.min_price_reaction_bps < 0:
            raise StrategyConfigError("min_price_reaction_bps must be >= 0")

        hint_fields = {
            "entry_offset_bps_hint": self.entry_offset_bps_hint,
            "stop_buffer_bps_hint": self.stop_buffer_bps_hint,
            "take_profit_bps_hint": self.take_profit_bps_hint,
            "composite_tp_multiplier_hint": self.composite_tp_multiplier_hint,
        }
        for field_name, value in hint_fields.items():
            if value is not None and value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        score_weights = {
            "score_base_weight": self.score_base_weight,
            "score_detector_weight": self.score_detector_weight,
            "score_agreement_weight": self.score_agreement_weight,
            "score_features_weight": self.score_features_weight,
            "score_reaction_weight": self.score_reaction_weight,
            "score_freshness_weight": self.score_freshness_weight,
        }
        confidence_weights = {
            "confidence_primary_weight": self.confidence_primary_weight,
            "confidence_context_weight": self.confidence_context_weight,
            "confidence_confirmation_weight": self.confidence_confirmation_weight,
            "confidence_freshness_weight": self.confidence_freshness_weight,
        }

        for field_name, value in {**score_weights, **confidence_weights}.items():
            if float(value) < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        if sum(score_weights.values()) <= 0:
            raise StrategyConfigError("score weights sum must be > 0")

        if sum(confidence_weights.values()) <= 0:
            raise StrategyConfigError("confidence weights sum must be > 0")

        for attr in (
            "tag_composite_spoofing",
            "tag_multi_detector",
            "tag_detector_agreement",
            "tag_order_pull_component",
            "tag_fake_liquidity_component",
            "tag_pressure_bluff_component",
            "tag_layering_component",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")

        if not self.required_spoofing_features:
            raise StrategyConfigError("required_spoofing_features cannot be empty")


class CompositeSpoofingStrategy(SpoofingTradingStrategy):
    """
    Unified analytics-level composite spoofing strategy.

    Input:
        StrategyContext with FeatureSource.SPOOFING domain data / features.

    Output:
        StrategySignal | None.

    This class does not perform strategy confluence. It only interprets
    analytics.spoofing composite signals. SignalProcessor still owns routing,
    filters, confluence, portfolio coordination and risk-ready payloads.
    """
    _logger = logging.getLogger(__name__ + ".CompositeSpoofingStrategy")

    component_namespace = "strategy.spoofing.composite"
    category: StrategyCategory = StrategyCategory.SPOOFING
    default_setup_type: SetupType = SetupType.REVERSAL

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        spoofing_config: CompositeSpoofingStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingStrategy.__init__")
        resolved_spoofing_config = (
            spoofing_config or CompositeSpoofingStrategyConfig()
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

        self.composite_config: CompositeSpoofingStrategyConfig = (
            resolved_spoofing_config
        )

    @property
    def strategy_name(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingStrategy.strategy_name")
        return "composite_spoofing"

    @property
    def metadata(self) -> StrategyMetadata:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingStrategy.metadata")
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.SPOOFING,
            timeframe=Timeframe.M1,
            tags=[
                self.composite_config.tag_spoofing,
                self.composite_config.tag_composite,
                self.composite_config.tag_composite_spoofing,
                self.composite_config.tag_multi_detector,
                self.composite_config.tag_detector_agreement,
                "analytics_spoofing",
            ],
            version="2.0.0",
            description=(
                "Interprets analytics-level composite spoofing signals from "
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
                "strategy_type": "composite_spoofing",
                "base_class": "SpoofingTradingStrategy",
                "canonical_payload": "SpoofingCompositeSnapshot",
                "uses_composite_analytics_signal": True,
                "duplicates_signal_processor_confluence": False,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingStrategy.required_features")
        base_required = super().required_features()
        return set(base_required).union(
            self.composite_config.required_spoofing_features
        )

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingStrategy.generate_signal")
        self.validate_context_requirements(context)

        if not self.has_any_spoofing_data(
            context,
            tuple(self.composite_config.required_spoofing_features),
        ):
            return None

        if self.has_stale_spoofing_features(
            context,
            tuple(self.composite_config.required_spoofing_features),
        ):
            return None

        payload = self._extract_payload(context)
        if payload is None:
            return None

        if is_stale(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.composite_config.stale_feature_max_age_seconds,
        ):
            return None

        rejection = quality_filter_reason(
            payload.snapshot.raw_signal,
            min_score=max(
                self.composite_config.min_score,
                self.composite_config.min_composite_score,
            ),
            min_confidence=max(
                self.composite_config.min_confidence,
                self.composite_config.min_composite_confidence,
            ),
            allowed_severities=self.composite_config.allowed_severities,
            min_detector_count=max(
                self.composite_config.min_detector_count,
                self.composite_config.min_composite_detector_count,
            ),
            min_agreement_ratio=max(
                self.composite_config.min_agreement_ratio,
                self.composite_config.min_composite_agreement_ratio,
            ),
            min_average_confidence=max(
                self.composite_config.min_average_confidence,
                self.composite_config.min_composite_average_confidence,
            ),
            require_score_passed=(
                self.composite_config.require_score_passed
                or self.composite_config.require_score_passed
            ),
            stale_after_seconds=self.composite_config.stale_feature_max_age_seconds,
            now=context.timestamp,
        )
        if rejection is not None:
            return None

        if not self.accepts_spoofing_snapshot(payload.snapshot):
            return None

        if not self._supports_snapshot(payload.snapshot):
            return None

        if not self._passes_composite_filters(payload):
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
        )

        if breakdown.score < self.composite_config.min_composite_score:
            return None

        if breakdown.confidence < self.composite_config.min_composite_confidence:
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload)

        reasons = list(
            dict.fromkeys(
                [
                    "composite_spoofing_signal",
                    f"side:{payload.side.value}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "spoofing_setup_family": "composite_spoofing",
            "spoofing_strategy_version": "2.0.0",
            "contract": "spoofing",
            "contract_version": "strategy-domain-v1",
            "primary_section": "composite",
            "secondary_section": "signal",
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
            "detector_count": detector_count(payload.snapshot.raw_signal),
            "detector_agreement_ratio": detector_agreement_ratio(payload.snapshot.raw_signal),
            "detector_average_confidence": detector_average_confidence(payload.snapshot.raw_signal),
            "components": self._component_metadata(payload),
            "features": {
                "pull_ratio": payload.pull_ratio,
                "fill_ratio": payload.fill_ratio,
                "price_reaction_bps": payload.price_reaction_bps,
                "signed_price_reaction_bps": payload.snapshot.signed_price_reaction_bps,
                "cancel_to_fill_ratio": payload.cancel_to_fill_ratio,
                "distance_from_mid_bps": payload.distance_from_mid_bps,
                "lifetime_ms": payload.lifetime_ms,
                "wall_notional": payload.wall_notional,
                "pulled_notional": payload.pulled_notional,
                "layer_count": payload.layer_count,
                "layer_price_span_bps": payload.layer_price_span_bps,
                "pressure_flip_strength": payload.pressure_flip_strength,
            },
            "execution_hints": self._execution_hints(),
        }

        return self.build_spoofing_signal(
            context=context,
            side=payload.side,
            confidence=breakdown.confidence,
            score=breakdown.score,
            setup_type=self.composite_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.composite_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_payload(
        self,
        context: StrategyContext,
    ) -> CompositeSpoofingPayload | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingStrategy._extract_payload")
        snapshot = self.resolve_spoofing_snapshot(context)
        if snapshot is None or not snapshot.has_minimum_data():
            return None

        side = spoofing_side_to_signal_side(snapshot.side)
        if self.composite_config.require_directional_side and not is_directional_side(side):
            return None

        event_time = (
            extract_event_time(snapshot.raw_signal)
            or snapshot.timestamp
            or context.timestamp
        )

        reasons = [
            "composite_spoofing_context",
            f"spoofing_type:{normalize_label(snapshot.spoofing_type)}",
            f"pattern:{normalize_label(snapshot.pattern)}",
            f"spoofing_side:{normalize_label(snapshot.side)}",
            f"score:{snapshot.score:.4f}",
            f"confidence:{snapshot.confidence:.4f}",
            f"detector_count:{detector_count(snapshot.raw_signal)}",
            f"agreement_ratio:{detector_agreement_ratio(snapshot.raw_signal):.4f}",
        ]

        if is_order_pull_signal(snapshot.raw_signal):
            reasons.append("order_pull_component_detected")

        if is_fake_liquidity_signal(snapshot.raw_signal):
            reasons.append("fake_liquidity_component_detected")

        if is_pressure_bluff_signal(snapshot.raw_signal):
            reasons.append("pressure_bluff_component_detected")

        if is_layering_signal(snapshot.raw_signal):
            reasons.append("layering_component_detected")

        return CompositeSpoofingPayload(
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
            _strategy_logger.debug("Entering CompositeSpoofingStrategy._supports_snapshot")
        if self.composite_config.require_composite_type_or_pattern:
            return (
                snapshot.spoofing_type is SpoofingType.COMPOSITE
                or is_composite_signal(snapshot.raw_signal)

            )

        return is_composite_signal(snapshot.raw_signal)

    def _passes_composite_filters(
        self,
        payload: CompositeSpoofingPayload,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingStrategy._passes_composite_filters")
        snapshot = payload.snapshot

        if detector_count(snapshot.raw_signal) < self.composite_config.min_composite_detector_count:
            return False

        if detector_agreement_ratio(snapshot.raw_signal) < self.composite_config.min_composite_agreement_ratio:
            return False

        if detector_average_confidence(snapshot.raw_signal) < self.composite_config.min_composite_average_confidence:
            return False

        if self.composite_config.require_any_allowed_component:
            if not self._has_allowed_component(snapshot):
                return False

        if payload.pull_ratio < self.composite_config.min_pull_ratio:
            return False

        if payload.fill_ratio > self.composite_config.max_fill_ratio:
            return False

        if payload.cancel_to_fill_ratio < self.composite_config.min_cancel_to_fill_ratio:
            return False

        if self.composite_config.require_market_reaction:
            if payload.price_reaction_bps < self.composite_config.min_price_reaction_bps:
                return False

        if self.composite_config.require_directional_reaction_alignment:
            if not reaction_aligns_with_side(
                signed_reaction_bps=snapshot.signed_price_reaction_bps,
                side=payload.side,
                min_reaction_bps=self.composite_config.min_price_reaction_bps,
            ):
                return False

        return True

    def _has_allowed_component(
        self,
        snapshot: SpoofingCompositeSnapshot,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingStrategy._has_allowed_component")
        return any(
            (
                self.composite_config.allow_order_pull_component
                and is_order_pull_signal(snapshot.raw_signal),
                self.composite_config.allow_fake_liquidity_component
                and is_fake_liquidity_signal(snapshot.raw_signal),
                self.composite_config.allow_flip_pressure_component
                and is_pressure_bluff_signal(snapshot.raw_signal),
                self.composite_config.allow_layering_component
                and is_layering_signal(snapshot.raw_signal),
            )
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: CompositeSpoofingPayload,
    ) -> ScoreBreakdown:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingStrategy._build_score_breakdown")
        snapshot = payload.snapshot

        base_component = average_score(
            extract_score(snapshot.raw_signal),
            snapshot.score,
            snapshot.confidence,
        )
        detector_component = self._detector_component(snapshot)
        agreement_component = detector_agreement_ratio(snapshot.raw_signal)
        features_component = self._features_component(payload)
        reaction_component = unit_score(
            payload.price_reaction_bps
            / max(self.composite_config.min_price_reaction_bps * 4.0, 0.01)
        ) if self.composite_config.min_price_reaction_bps > 0 else unit_score(
            payload.price_reaction_bps / 10.0
        )
        fresh_component = freshness_score(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.composite_config.stale_feature_max_age_seconds,
        )

        components = {
            "base": base_component,
            "detector": detector_component,
            "agreement": agreement_component,
            "features": features_component,
            "reaction": reaction_component,
            "freshness": fresh_component,
        }
        weights = {
            "base": self.composite_config.score_base_weight,
            "detector": self.composite_config.score_detector_weight,
            "agreement": self.composite_config.score_agreement_weight,
            "features": self.composite_config.score_features_weight,
            "reaction": self.composite_config.score_reaction_weight,
            "freshness": self.composite_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=base_component)
        confidence = confidence_from_components(
            primary=base_component,
            context=average_score(detector_component, agreement_component),
            confirmation=average_score(features_component, reaction_component),
            freshness=fresh_component,
            primary_weight=self.composite_config.confidence_primary_weight,
            context_weight=self.composite_config.confidence_context_weight,
            confirmation_weight=self.composite_config.confidence_confirmation_weight,
            freshness_weight=self.composite_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            "analytics_composite_spoofing_context",
            f"side:{payload.side.value}",
            f"detector_count:{detector_count(snapshot.raw_signal)}",
            f"agreement_ratio:{detector_agreement_ratio(snapshot.raw_signal):.4f}",
            f"average_detector_confidence:{detector_average_confidence(snapshot.raw_signal):.4f}",
        ]

        if is_composite_signal(snapshot.raw_signal):
            score += self.composite_config.composite_bonus
            confirmations.append("composite_spoofing_signal")

        if detector_count(snapshot.raw_signal) >= self.composite_config.strong_detector_count_threshold:
            score += self.composite_config.multi_detector_bonus
            confirmations.append("strong_multi_detector_support")

        if detector_agreement_ratio(snapshot.raw_signal) >= self.composite_config.high_agreement_threshold:
            score += self.composite_config.high_agreement_bonus
            confirmations.append("high_detector_agreement")

        if is_order_pull_signal(snapshot.raw_signal):
            score += self.composite_config.order_pull_component_bonus
            confirmations.append("order_pull_component")

        if is_fake_liquidity_signal(snapshot.raw_signal):
            score += self.composite_config.fake_liquidity_component_bonus
            confirmations.append("fake_liquidity_component")

        if is_pressure_bluff_signal(snapshot.raw_signal):
            score += self.composite_config.pressure_bluff_component_bonus
            confirmations.append("pressure_bluff_component")

        if is_layering_signal(snapshot.raw_signal):
            score += self.composite_config.layering_component_bonus
            confirmations.append("layering_component")

        if reaction_aligns_with_side(
            signed_reaction_bps=snapshot.signed_price_reaction_bps,
            side=payload.side,
            min_reaction_bps=self.composite_config.min_price_reaction_bps,
        ):
            score += self.composite_config.directional_reaction_bonus
            confidence += min(0.03, self.composite_config.directional_reaction_bonus)
            confirmations.append("directional_composite_reaction")

        if payload.price_reaction_bps > 0:
            reasons.append(f"price_reaction_bps:{payload.price_reaction_bps:.4f}")

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

    def _detector_component(
        self,
        snapshot: SpoofingCompositeSnapshot,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingStrategy._detector_component")
        detector_scores = [
            detector_agreement_ratio(snapshot.raw_signal),
            detector_average_confidence(snapshot.raw_signal),
        ]

        for component in (
            SpoofingComponent.ORDER_PULL_DETECTOR,
            SpoofingComponent.FAKE_LIQUIDITY_DETECTOR,
            SpoofingComponent.FLIP_PRESSURE_DETECTOR,
            SpoofingComponent.LAYERING_DETECTOR,
        ):
            if snapshot.has_detector(component):
                detector_scores.append(detector_score(snapshot.raw_signal, component))
                detector_scores.append(detector_confidence(snapshot.raw_signal, component))

                if detector_passed(snapshot.raw_signal, component):
                    detector_scores.append(1.0)

        return average_score(*detector_scores)

    def _features_component(
        self,
        payload: CompositeSpoofingPayload,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingStrategy._features_component")
        notional_ratio = (
            payload.pulled_notional / max(payload.wall_notional, 1.0)
            if payload.wall_notional > 0
            else 0.0
        )

        layer_component = unit_score(payload.layer_count / 5.0)
        pressure_component = payload.pressure_flip_strength
        fill_quality = unit_score(1.0 - payload.fill_ratio)

        return weighted_score(
            {
                "pull": payload.pull_ratio,
                "fill_quality": fill_quality,
                "cancel_fill": payload.cancel_to_fill_ratio,
                "notional": unit_score(notional_ratio),
                "layers": layer_component,
                "pressure": pressure_component,
            },
            {
                "pull": 0.25,
                "fill_quality": 0.20,
                "cancel_fill": 0.18,
                "notional": 0.12,
                "layers": 0.12,
                "pressure": 0.13,
            },
        )

    # ------------------------------------------------------------------
    # Source features / tags / metadata / execution hints
    # ------------------------------------------------------------------

    def _source_features(
        self,
        payload: CompositeSpoofingPayload,
    ) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingStrategy._source_features")
        features = [
            *composite_spoofing_source_features(),
            SPOOFING_FEATURES.SIGNAL,
            SPOOFING_FEATURES.SPOOFING_TYPE,
            SPOOFING_FEATURES.PATTERN,
            SPOOFING_FEATURES.SIDE,
            SPOOFING_FEATURES.SCORE,
            SPOOFING_FEATURES.CONFIDENCE,
            SPOOFING_FEATURES.DETECTOR_RESULTS,
            SPOOFING_FEATURES.SCORE_BREAKDOWN,
            SPOOFING_FEATURES.ANALYTICS_METADATA,
        ]

        if is_order_pull_signal(payload.snapshot.raw_signal):
            features.extend(order_pull_source_features())

        if is_fake_liquidity_signal(payload.snapshot.raw_signal):
            features.extend(fake_liquidity_source_features())

        if is_pressure_bluff_signal(payload.snapshot.raw_signal):
            features.extend(pressure_bluff_source_features())

        if is_layering_signal(payload.snapshot.raw_signal):
            features.extend(layering_source_features())

        return list(dict.fromkeys(features))

    def _tags(
        self,
        payload: CompositeSpoofingPayload,
    ) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingStrategy._tags")
        tags = [
            self.composite_config.tag_spoofing,
            self.composite_config.tag_composite,
            self.composite_config.tag_composite_spoofing,
            self.composite_config.tag_multi_detector,
            self.composite_config.tag_detector_agreement,
            f"side:{payload.side.value}",
        ]

        if is_order_pull_signal(payload.snapshot.raw_signal):
            tags.append(self.composite_config.tag_order_pull_component)

        if is_fake_liquidity_signal(payload.snapshot.raw_signal):
            tags.append(self.composite_config.tag_fake_liquidity_component)

        if is_pressure_bluff_signal(payload.snapshot.raw_signal):
            tags.append(self.composite_config.tag_pressure_bluff_component)

        if is_layering_signal(payload.snapshot.raw_signal):
            tags.append(self.composite_config.tag_layering_component)

        if payload.snapshot.spoofing_type is not None:
            tags.append(f"type:{normalize_label(payload.snapshot.spoofing_type)}")

        if payload.snapshot.pattern is not None:
            tags.append(f"pattern:{normalize_label(payload.snapshot.pattern)}")

        return list(dict.fromkeys(tags))

    def _component_metadata(
        self,
        payload: CompositeSpoofingPayload,
    ) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingStrategy._component_metadata")
        snapshot = payload.snapshot

        return {
            "has_order_pull_component": is_order_pull_signal(snapshot.raw_signal),
            "has_fake_liquidity_component": is_fake_liquidity_signal(snapshot.raw_signal),
            "has_pressure_bluff_component": is_pressure_bluff_signal(snapshot.raw_signal),
            "has_layering_component": is_layering_signal(snapshot.raw_signal),
            "order_pull_detector": serialize_for_metadata(
                snapshot.detector_payload(SpoofingComponent.ORDER_PULL_DETECTOR)
            ),
            "fake_liquidity_detector": serialize_for_metadata(
                snapshot.detector_payload(SpoofingComponent.FAKE_LIQUIDITY_DETECTOR)
            ),
            "flip_pressure_detector": serialize_for_metadata(
                snapshot.detector_payload(SpoofingComponent.FLIP_PRESSURE_DETECTOR)
            ),
            "layering_detector": serialize_for_metadata(
                snapshot.detector_payload(SpoofingComponent.LAYERING_DETECTOR)
            ),
        }

    def _execution_hints(self) -> dict[str, Any]:
        """
        Execution hints only. Final EntryPlan/ExitPlan/RiskReadySignalPayload
        is owned by SignalProcessor / SignalBuilder.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CompositeSpoofingStrategy._execution_hints")
        return {
            "entry_offset_bps": self.composite_config.entry_offset_bps_hint,
            "stop_buffer_bps": self.composite_config.stop_buffer_bps_hint,
            "take_profit_bps": self.composite_config.take_profit_bps_hint,
            "composite_tp_multiplier": self.composite_config.composite_tp_multiplier_hint,
        }