# trading_system/strategy/strategies/liquidations/liquidation_cascade_strategy.py

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from core.event_bus import EventBus
from core.scheduler import Scheduler
from .base import (
    LIQUIDATIONS_FEATURES,
    LiquidationsStrategyConfig,
    LiquidationsTradingStrategy,
)
from .utils import (
    ScoreBreakdown,
    confidence_from_components,
    continuation_side_from_direction,
    extract_acceleration_ratio,
    extract_bias_delta,
    extract_cluster_avg_notional_per_event,
    extract_cluster_duration_seconds,
    extract_confidence,
    extract_continuation_bias,
    extract_direction,
    extract_event_count,
    extract_event_imbalance_ratio,
    extract_event_time,
    extract_exhaustion_bias,
    extract_intensity_score,
    extract_notional_usd,
    extract_price_range_pct,
    extract_score,
    extract_severity_label,
    extract_side_imbalance_ratio,
    freshness_score,
    is_directional_side,
    is_stale,
    liquidations_item,
    quality_filter_reason,
    serialize_for_metadata,
    severity_score,
    unit_score,
    weighted_score,
)
from ...config import StrategyConfig, StrategyDefinitionConfig
from ...enums import (
    SetupType,
    SignalPriority,
    SignalSide,
    StrategyCategory,
)
from ...exceptions import StrategyConfigError
from ...models import StrategyContext, StrategySignal


@dataclass(slots=True)
class LiquidationCascadeStrategyConfig(LiquidationsStrategyConfig):
    """
    Unified liquidation cascade continuation strategy config.

    Strategy idea:
    - read normalized liquidation cascade context from StrategyContext;
    - accept strong cascade / forced liquidation flow;
    - generate continuation signal in cascade direction:
        cascade UP   -> LONG
        cascade DOWN -> SHORT;
    - leave routing, filtering, confluence, portfolio coordination and
      risk-ready conversion to SignalProcessor.
    """

    require_confirmed_result: bool = True
    require_actionable_direction: bool = True
    require_fresh_cascade: bool = True

    allowed_severities: tuple[str, ...] = (
        "medium",
        "high",
        "extreme",
    )

    min_confidence: float = 0.60
    min_intensity_score: float = 0.55
    min_total_notional_usd: Decimal = Decimal("300000")
    min_event_count: int = 5
    max_price_range_pct: float | None = None

    require_favors_continuation: bool = True
    min_continuation_bias: float = 0.60
    max_exhaustion_bias_for_continuation: float | None = None
    min_bias_delta: float | None = None

    min_side_imbalance_ratio: float | None = None
    min_event_imbalance_ratio: float | None = None
    min_acceleration_ratio: float | None = None

    max_cluster_duration_seconds: float | None = None
    min_avg_notional_per_event: Decimal | None = None

    score_confidence_weight: float = 0.30
    score_continuation_bias_weight: float = 0.30
    score_intensity_weight: float = 0.18
    score_severity_weight: float = 0.10
    score_imbalance_weight: float = 0.07
    score_acceleration_weight: float = 0.05

    tag_cascade_continuation: str = "liquidation_cascade_continuation"
    tag_forced_flow: str = "forced_liquidation_flow"
    tag_continuation: str = "continuation"
    tag_high_intensity: str = "high_intensity"
    tag_large_notional: str = "large_notional"
    tag_imbalanced_cluster: str = "imbalanced_cluster"
    tag_acceleration: str = "liquidation_acceleration"

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.CONTINUATION

    required_liquidations_features: tuple[str, ...] = (
        LIQUIDATIONS_FEATURES.CASCADE,
    )

    def validate(self) -> None:
        LiquidationsStrategyConfig.validate(self)

        bounded_fields = {
            "min_confidence": self.min_confidence,
            "min_intensity_score": self.min_intensity_score,
            "min_continuation_bias": self.min_continuation_bias,
        }

        for field_name, value in bounded_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        if self.max_exhaustion_bias_for_continuation is not None:
            if not 0.0 <= self.max_exhaustion_bias_for_continuation <= 1.0:
                raise StrategyConfigError(
                    "max_exhaustion_bias_for_continuation must be between 0.0 and 1.0"
                )

        if self.min_bias_delta is not None:
            if not 0.0 <= self.min_bias_delta <= 1.0:
                raise StrategyConfigError("min_bias_delta must be between 0.0 and 1.0")

        if self.min_total_notional_usd < 0:
            raise StrategyConfigError("min_total_notional_usd must be >= 0")

        if self.min_event_count < 0:
            raise StrategyConfigError("min_event_count must be >= 0")

        if self.max_price_range_pct is not None and self.max_price_range_pct < 0:
            raise StrategyConfigError("max_price_range_pct must be >= 0")

        if self.min_side_imbalance_ratio is not None:
            if not 0.0 <= self.min_side_imbalance_ratio <= 1.0:
                raise StrategyConfigError(
                    "min_side_imbalance_ratio must be between 0.0 and 1.0"
                )

        if self.min_event_imbalance_ratio is not None:
            if not 0.0 <= self.min_event_imbalance_ratio <= 1.0:
                raise StrategyConfigError(
                    "min_event_imbalance_ratio must be between 0.0 and 1.0"
                )

        if self.min_acceleration_ratio is not None and self.min_acceleration_ratio < 0:
            raise StrategyConfigError("min_acceleration_ratio must be >= 0")

        if (
            self.max_cluster_duration_seconds is not None
            and self.max_cluster_duration_seconds <= 0
        ):
            raise StrategyConfigError("max_cluster_duration_seconds must be > 0")

        if (
            self.min_avg_notional_per_event is not None
            and self.min_avg_notional_per_event < 0
        ):
            raise StrategyConfigError("min_avg_notional_per_event must be >= 0")

        score_weights = {
            "score_confidence_weight": self.score_confidence_weight,
            "score_continuation_bias_weight": self.score_continuation_bias_weight,
            "score_intensity_weight": self.score_intensity_weight,
            "score_severity_weight": self.score_severity_weight,
            "score_imbalance_weight": self.score_imbalance_weight,
            "score_acceleration_weight": self.score_acceleration_weight,
        }

        for field_name, value in score_weights.items():
            if value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        if sum(score_weights.values()) <= 0:
            raise StrategyConfigError("strategy score weights sum must be > 0")

        for attr in (
            "tag_cascade_continuation",
            "tag_forced_flow",
            "tag_continuation",
            "tag_high_intensity",
            "tag_large_notional",
            "tag_imbalanced_cluster",
            "tag_acceleration",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")

        if not self.allowed_severities:
            raise StrategyConfigError("allowed_severities cannot be empty")

        for severity in self.allowed_severities:
            if not isinstance(severity, str) or not severity.strip():
                raise StrategyConfigError(
                    "allowed_severities cannot contain empty values"
                )

        if not self.required_liquidations_features:
            raise StrategyConfigError("required_liquidations_features cannot be empty")

        for feature in self.required_liquidations_features:
            if not isinstance(feature, str) or not feature.strip():
                raise StrategyConfigError(
                    "required_liquidations_features cannot contain empty feature names"
                )


class LiquidationCascadeStrategy(LiquidationsTradingStrategy):
    """
    Unified liquidation cascade continuation strategy.

    Input:
        StrategyContext with FeatureSource.LIQUIDATIONS domain data / features.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, filters, confluence, building and risk payloads.
    """

    component_namespace = "strategy.liquidations.cascade"
    category: StrategyCategory = StrategyCategory.LIQUIDATIONS
    default_setup_type: SetupType = SetupType.CONTINUATION

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        liquidations_config: LiquidationCascadeStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_liquidations_config = (
            liquidations_config or LiquidationCascadeStrategyConfig()
        )
        resolved_liquidations_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            liquidations_config=resolved_liquidations_config,
            service_name=service_name,
        )

        self.cascade_config: LiquidationCascadeStrategyConfig = (
            resolved_liquidations_config
        )

    @property
    def strategy_name(self) -> str:
        return "liquidation_cascade"

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(
            self.cascade_config.required_liquidations_features
        )

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        self.validate_context_requirements(context)

        cascade = liquidations_item(context, "cascade")
        if cascade is None:
            return None

        event_time = extract_event_time(cascade)
        if (
            self.cascade_config.require_fresh_cascade
            and is_stale(
                event_time=event_time,
                now=context.timestamp,
                stale_after_seconds=self.cascade_config.stale_feature_max_age_seconds,
            )
        ):
            return None

        common_rejection = quality_filter_reason(
            cascade,
            min_confidence=self.cascade_config.min_confidence,
            min_intensity_score=self.cascade_config.min_intensity_score,
            min_total_notional_usd=self.cascade_config.min_total_notional_usd,
            min_event_count=self.cascade_config.min_event_count,
            max_price_range_pct=self.cascade_config.max_price_range_pct,
            require_confirmed=self.cascade_config.require_confirmed_result,
            require_actionable_direction=self.cascade_config.require_actionable_direction,
        )
        if common_rejection is not None:
            return None

        if not self._passes_cascade_filters(cascade):
            return None

        side = self._derive_continuation_side(cascade)
        if not is_directional_side(side):
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            cascade=cascade,
        )

        if breakdown.score < self.cascade_config.min_signal_score:
            return None

        if breakdown.confidence < self.cascade_config.min_signal_confidence:
            return None

        source_features = self._source_features(cascade)
        tags = self._tags(cascade)

        reasons = list(
            dict.fromkeys(
                [
                    "liquidation_cascade_continuation",
                    f"side:{side.value}",
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "liquidations_setup_family": "liquidation_cascade_continuation",
            "liquidations_strategy_version": "2.0.0",
            "score_breakdown": breakdown.to_dict(),
            "cascade": serialize_for_metadata(cascade),
            "event_time": event_time.isoformat() if event_time else None,
            "tags": tags,
            "continuation_side": side.value,
            "cascade_direction": serialize_for_metadata(extract_direction(cascade)),
            "severity": self._severity(cascade),
            "event_count": extract_event_count(cascade),
            "total_notional_usd": str(extract_notional_usd(cascade)),
            "intensity_score": extract_intensity_score(cascade),
            "continuation_bias": extract_continuation_bias(cascade),
            "exhaustion_bias": extract_exhaustion_bias(cascade),
            "bias_delta": extract_bias_delta(cascade),
            "price_range_pct": extract_price_range_pct(cascade),
            "side_imbalance_ratio": extract_side_imbalance_ratio(cascade),
            "event_imbalance_ratio": extract_event_imbalance_ratio(cascade),
            "acceleration_ratio": extract_acceleration_ratio(cascade),
            "cluster_duration_seconds": extract_cluster_duration_seconds(cascade),
            "cluster_avg_notional_per_event": str(
                extract_cluster_avg_notional_per_event(cascade)
            ),
        }

        return self.build_liquidations_signal(
            context=context,
            side=side,
            confidence=breakdown.confidence,
            score=breakdown.score,
            setup_type=self.cascade_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.cascade_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _passes_cascade_filters(self, cascade: Any) -> bool:
        if not self._severity_allowed(cascade):
            return False

        if self.cascade_config.require_favors_continuation:
            continuation_bias = extract_continuation_bias(cascade)
            if continuation_bias < self.cascade_config.min_continuation_bias:
                return False

        if self.cascade_config.max_exhaustion_bias_for_continuation is not None:
            exhaustion_bias = extract_exhaustion_bias(cascade)
            if exhaustion_bias > self.cascade_config.max_exhaustion_bias_for_continuation:
                return False

        if self.cascade_config.min_bias_delta is not None:
            bias_delta = extract_bias_delta(cascade)
            if bias_delta < self.cascade_config.min_bias_delta:
                return False

        if self.cascade_config.min_side_imbalance_ratio is not None:
            imbalance = extract_side_imbalance_ratio(cascade)
            if imbalance < self.cascade_config.min_side_imbalance_ratio:
                return False

        if self.cascade_config.min_event_imbalance_ratio is not None:
            imbalance = extract_event_imbalance_ratio(cascade)
            if imbalance < self.cascade_config.min_event_imbalance_ratio:
                return False

        if self.cascade_config.min_acceleration_ratio is not None:
            acceleration = extract_acceleration_ratio(cascade)
            if acceleration < self.cascade_config.min_acceleration_ratio:
                return False

        if self.cascade_config.max_cluster_duration_seconds is not None:
            duration = extract_cluster_duration_seconds(cascade)
            if duration > self.cascade_config.max_cluster_duration_seconds:
                return False

        if self.cascade_config.min_avg_notional_per_event is not None:
            avg_notional = extract_cluster_avg_notional_per_event(cascade)
            if avg_notional < self.cascade_config.min_avg_notional_per_event:
                return False

        return True

    def _severity_allowed(self, cascade: Any) -> bool:
        severity = self._severity(cascade)
        allowed = {
            item.strip().lower()
            for item in self.cascade_config.allowed_severities
            if item.strip()
        }

        return severity in allowed

    # ------------------------------------------------------------------
    # Direction
    # ------------------------------------------------------------------

    def _derive_continuation_side(self, cascade: Any) -> SignalSide:
        return continuation_side_from_direction(extract_direction(cascade))

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        cascade: Any,
    ) -> ScoreBreakdown:
        confidence_component = extract_confidence(cascade)
        continuation_bias = extract_continuation_bias(cascade)
        intensity_score = extract_intensity_score(cascade)
        severity_component = severity_score(cascade)
        imbalance_component = self._imbalance_score(cascade)
        acceleration_component = self._acceleration_score(cascade)

        score = weighted_score(
            {
                "confidence": confidence_component,
                "continuation_bias": continuation_bias,
                "intensity": intensity_score,
                "severity": severity_component,
                "imbalance": imbalance_component,
                "acceleration": acceleration_component,
            },
            {
                "confidence": self.cascade_config.score_confidence_weight,
                "continuation_bias": (
                    self.cascade_config.score_continuation_bias_weight
                ),
                "intensity": self.cascade_config.score_intensity_weight,
                "severity": self.cascade_config.score_severity_weight,
                "imbalance": self.cascade_config.score_imbalance_weight,
                "acceleration": self.cascade_config.score_acceleration_weight,
            },
            default=extract_score(cascade),
        )

        event_time = extract_event_time(cascade)
        fresh_score = freshness_score(
            event_time=event_time,
            now=context.timestamp,
            stale_after_seconds=self.cascade_config.stale_feature_max_age_seconds,
        )

        context_score = weighted_score(
            {
                "continuation_bias": continuation_bias,
                "intensity": intensity_score,
                "severity": severity_component,
                "imbalance": imbalance_component,
                "acceleration": acceleration_component,
            },
            {
                "continuation_bias": 0.30,
                "intensity": 0.25,
                "severity": 0.15,
                "imbalance": 0.15,
                "acceleration": 0.15,
            },
            default=0.0,
        )

        confidence = confidence_from_components(
            primary=confidence_component,
            context=context_score,
            confirmation=continuation_bias,
            freshness=fresh_score,
        )

        reasons = [
            f"confidence:{confidence_component:.3f}",
            f"continuation_bias:{continuation_bias:.3f}",
            f"intensity_score:{intensity_score:.3f}",
            f"severity_score:{severity_component:.3f}",
        ]

        confirmations: list[str] = []

        if continuation_bias >= self.cascade_config.min_continuation_bias:
            confirmations.append("continuation_bias_confirmed")

        if intensity_score >= self.cascade_config.min_intensity_score:
            confirmations.append("intensity_confirmed")

        if extract_notional_usd(cascade) >= self.cascade_config.min_total_notional_usd:
            confirmations.append("notional_confirmed")

        if imbalance_component >= 0.60:
            confirmations.append("imbalance_confirmed")

        if acceleration_component >= 0.60:
            confirmations.append("acceleration_confirmed")

        return ScoreBreakdown(
            score=score,
            confidence=confidence,
            components={
                "confidence": confidence_component,
                "continuation_bias": continuation_bias,
                "intensity_score": intensity_score,
                "severity_score": severity_component,
                "imbalance_score": imbalance_component,
                "acceleration_score": acceleration_component,
                "context_score": context_score,
                "freshness_score": fresh_score,
            },
            weights={
                "score_confidence_weight": self.cascade_config.score_confidence_weight,
                "score_continuation_bias_weight": (
                    self.cascade_config.score_continuation_bias_weight
                ),
                "score_intensity_weight": self.cascade_config.score_intensity_weight,
                "score_severity_weight": self.cascade_config.score_severity_weight,
                "score_imbalance_weight": self.cascade_config.score_imbalance_weight,
                "score_acceleration_weight": self.cascade_config.score_acceleration_weight,
            },
            reasons=reasons,
            confirmations=confirmations,
        ).normalize()

    @staticmethod
    def _imbalance_score(cascade: Any) -> float:
        side_imbalance = extract_side_imbalance_ratio(cascade)
        event_imbalance = extract_event_imbalance_ratio(cascade)

        if side_imbalance > 0:
            return side_imbalance

        if event_imbalance > 0:
            return event_imbalance

        return 0.50

    @staticmethod
    def _acceleration_score(cascade: Any) -> float:
        acceleration = extract_acceleration_ratio(cascade)

        if acceleration <= 0:
            return 0.50

        # 1.0 = neutral, 2.0+ = strong acceleration.
        return unit_score((acceleration - 1.0) / 1.0)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _severity(self, cascade: Any) -> str:
        return extract_severity_label(cascade, default="unknown")

    def _source_features(self, cascade: Any) -> list[str]:
        features = [
            LIQUIDATIONS_FEATURES.CASCADE,
            LIQUIDATIONS_FEATURES.CASCADE_CONFIDENCE,
            LIQUIDATIONS_FEATURES.CASCADE_INTENSITY,
            LIQUIDATIONS_FEATURES.CASCADE_DIRECTION,
            LIQUIDATIONS_FEATURES.CASCADE_SEVERITY,
            LIQUIDATIONS_FEATURES.CASCADE_CONTINUATION_BIAS,
            LIQUIDATIONS_FEATURES.CASCADE_EXHAUSTION_BIAS,
            LIQUIDATIONS_FEATURES.CASCADE_NOTIONAL_USD,
            LIQUIDATIONS_FEATURES.CASCADE_EVENT_COUNT,
        ]

        if extract_side_imbalance_ratio(cascade) > 0:
            features.append(LIQUIDATIONS_FEATURES.CLUSTER_SIDE_IMBALANCE_RATIO)

        if extract_event_imbalance_ratio(cascade) > 0:
            features.append(LIQUIDATIONS_FEATURES.CLUSTER_EVENT_IMBALANCE_RATIO)

        if extract_acceleration_ratio(cascade) > 0:
            features.append(LIQUIDATIONS_FEATURES.CLUSTER_ACCELERATION_RATIO)

        if extract_cluster_duration_seconds(cascade) > 0:
            features.append(LIQUIDATIONS_FEATURES.CLUSTER_DURATION_SECONDS)

        if extract_cluster_avg_notional_per_event(cascade) > Decimal("0"):
            features.append(LIQUIDATIONS_FEATURES.CLUSTER_AVG_NOTIONAL_PER_EVENT)

        return list(dict.fromkeys(features))

    def _tags(self, cascade: Any) -> list[str]:
        tags = [
            self.cascade_config.tag_liquidations,
            self.cascade_config.tag_cascade,
            self.cascade_config.tag_cascade_continuation,
            self.cascade_config.tag_continuation,
            self.cascade_config.tag_forced_flow,
        ]

        if extract_intensity_score(cascade) >= 0.75:
            tags.append(self.cascade_config.tag_high_intensity)

        if extract_notional_usd(cascade) >= self.cascade_config.min_total_notional_usd:
            tags.append(self.cascade_config.tag_large_notional)

        if (
            extract_side_imbalance_ratio(cascade) >= 0.60
            or extract_event_imbalance_ratio(cascade) >= 0.60
        ):
            tags.append(self.cascade_config.tag_imbalanced_cluster)

        if extract_acceleration_ratio(cascade) >= 1.50:
            tags.append(self.cascade_config.tag_acceleration)

        tags.append(f"severity:{self._severity(cascade)}")

        return list(dict.fromkeys(tags))


__all__ = [
    "LiquidationCascadeStrategy",
    "LiquidationCascadeStrategyConfig",
]