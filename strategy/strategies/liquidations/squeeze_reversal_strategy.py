# trading_system/strategy/strategies/liquidations/squeeze_reversal_strategy.py

from __future__ import annotations
import logging

from dataclasses import dataclass
from decimal import Decimal
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
from ...models import StrategyContext, StrategySignal
from .base import (
    LIQUIDATIONS_FEATURES,
    LiquidationsStrategyConfig,
    LiquidationsTradingStrategy,
)
from .utils import (
    ScoreBreakdown,
    confidence_from_components,
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
    extract_reversal_side,
    extract_score,
    extract_severity_label,
    extract_side_imbalance_ratio,
    freshness_score,
    is_confirmed_status,
    is_directional_side,
    is_stale,
    liquidations_item,
    quality_filter_reason,
    serialize_for_metadata,
    severity_score,
    unit_score,
    weighted_score,
)


@dataclass(slots=True)
class SqueezeReversalStrategyConfig(LiquidationsStrategyConfig):
    """
    Unified liquidation squeeze / exhaustion reversal strategy config.

    Strategy idea:
    - read normalized liquidation exhaustion/squeeze context from StrategyContext;
    - detect cascade exhaustion after forced liquidation flow;
    - generate reversal signal:
        cascade DOWN / long liquidations exhaustion -> LONG
        cascade UP / short liquidations exhaustion -> SHORT;
    - do not keep pending candidates inside strategy;
    - leave routing, confluence, filtering, portfolio coordination and
      risk-ready conversion to SignalProcessor.

    Pending confirmation:
    - old strategy handled pending confirmation internally with Scheduler;
    - new strategy expects analytics/context to provide confirmed exhaustion or
      squeeze context if confirmation is required.
    """
    _logger = logging.getLogger(__name__ + ".SqueezeReversalStrategyConfig")

    require_confirmed_result: bool = True
    require_actionable_direction: bool = True
    require_fresh_exhaustion: bool = True

    # Якщо True, strategy генерує сигнал тільки коли context має confirmed/status.
    # Це замінює старий internal pending-confirmation scheduler.
    require_confirmed_exhaustion_context: bool = True

    allowed_severities: tuple[str, ...] = (
        "high",
        "extreme",
    )

    min_confidence: float = 0.65
    min_intensity_score: float = 0.60
    min_total_notional_usd: Decimal = Decimal("400000")
    min_event_count: int = 6
    max_price_range_pct: float | None = None

    require_favors_exhaustion: bool = True
    require_actionable_severity: bool = True

    min_exhaustion_bias: float = 0.70
    min_bias_delta: float = 0.12
    max_continuation_bias_after_exhaustion: float | None = 0.55

    min_side_imbalance_ratio: float | None = 0.70
    min_event_imbalance_ratio: float | None = None
    min_climax_acceleration_ratio: float | None = 1.10

    max_cluster_duration_seconds: float | None = 12.0
    min_avg_notional_per_event: Decimal | None = Decimal("50000")

    score_confidence_weight: float = 0.23
    score_exhaustion_bias_weight: float = 0.28
    score_bias_delta_weight: float = 0.14
    score_intensity_weight: float = 0.12
    score_severity_weight: float = 0.08
    score_cluster_quality_weight: float = 0.08
    score_imbalance_weight: float = 0.04
    score_acceleration_weight: float = 0.03

    tag_squeeze_reversal: str = "liquidation_squeeze_reversal"
    tag_exhaustion: str = "exhaustion"
    tag_reversal: str = "reversal"
    tag_climax: str = "liquidation_climax"
    tag_imbalanced_cluster: str = "imbalanced_cluster"
    tag_acceleration: str = "climax_acceleration"
    tag_confirmed_context: str = "confirmed_context"

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.SQUEEZE

    required_liquidations_features: tuple[str, ...] = (
        LIQUIDATIONS_FEATURES.EXHAUSTION,
    )

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SqueezeReversalStrategyConfig.validate")
        LiquidationsStrategyConfig.validate(self)

        bounded_fields = {
            "min_confidence": self.min_confidence,
            "min_intensity_score": self.min_intensity_score,
            "min_exhaustion_bias": self.min_exhaustion_bias,
            "min_bias_delta": self.min_bias_delta,
        }

        for field_name, value in bounded_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        if self.max_continuation_bias_after_exhaustion is not None:
            if not 0.0 <= self.max_continuation_bias_after_exhaustion <= 1.0:
                raise StrategyConfigError(
                    "max_continuation_bias_after_exhaustion must be between 0.0 and 1.0"
                )

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

        if (
            self.min_climax_acceleration_ratio is not None
            and self.min_climax_acceleration_ratio < 0
        ):
            raise StrategyConfigError("min_climax_acceleration_ratio must be >= 0")

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
            "score_exhaustion_bias_weight": self.score_exhaustion_bias_weight,
            "score_bias_delta_weight": self.score_bias_delta_weight,
            "score_intensity_weight": self.score_intensity_weight,
            "score_severity_weight": self.score_severity_weight,
            "score_cluster_quality_weight": self.score_cluster_quality_weight,
            "score_imbalance_weight": self.score_imbalance_weight,
            "score_acceleration_weight": self.score_acceleration_weight,
        }

        for field_name, value in score_weights.items():
            if value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        if sum(score_weights.values()) <= 0:
            raise StrategyConfigError("strategy score weights sum must be > 0")

        for attr in (
            "tag_squeeze_reversal",
            "tag_exhaustion",
            "tag_reversal",
            "tag_climax",
            "tag_imbalanced_cluster",
            "tag_acceleration",
            "tag_confirmed_context",
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


class SqueezeReversalStrategy(LiquidationsTradingStrategy):
    """
    Unified squeeze / exhaustion reversal strategy.

    Input:
        StrategyContext with FeatureSource.LIQUIDATIONS domain data / features.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus, does not run pending scheduler
    jobs and does not emit signal.generated. SignalProcessor owns routing,
    filtering, confluence, building and risk payloads.
    """
    _logger = logging.getLogger(__name__ + ".SqueezeReversalStrategy")

    component_namespace = "strategy.liquidations.squeeze_reversal"
    category: StrategyCategory = StrategyCategory.LIQUIDATIONS
    default_setup_type: SetupType = SetupType.SQUEEZE

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        liquidations_config: SqueezeReversalStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SqueezeReversalStrategy.__init__")
        resolved_liquidations_config = (
            liquidations_config or SqueezeReversalStrategyConfig()
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

        self.squeeze_config: SqueezeReversalStrategyConfig = (
            resolved_liquidations_config
        )

    @property
    def strategy_name(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SqueezeReversalStrategy.strategy_name")
        return "squeeze_reversal"

    def required_features(self) -> set[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SqueezeReversalStrategy.required_features")
        base_required = super().required_features()
        return set(base_required).union(
            self.squeeze_config.required_liquidations_features
        )

    async def generate_signal(
            self,
            context: StrategyContext,
    ) -> StrategySignal | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SqueezeReversalStrategy.generate_signal")
        self.validate_context_requirements(context)

        exhaustion = self._resolve_exhaustion_context(context)
        if exhaustion is None:
            self.remember_no_signal(
                "missing_liquidations_squeeze_or_exhaustion_contract",
                liquidations_domain_keys=sorted(self.liquidations_domain(context).keys()),
                required_features=sorted(self.required_features()),
            )
            return None

        event_time = extract_event_time(exhaustion)
        if (
                self.squeeze_config.require_fresh_exhaustion
                and is_stale(
            event_time=event_time,
            now=context.timestamp,
            stale_after_seconds=self.squeeze_config.stale_feature_max_age_seconds,
        )
        ):
            self.remember_no_signal(
                "stale_liquidations_exhaustion",
                event_time=event_time.isoformat() if event_time else None,
                context_timestamp=context.timestamp.isoformat(),
                stale_after_seconds=self.squeeze_config.stale_feature_max_age_seconds,
                exhaustion=serialize_for_metadata(exhaustion),
            )
            return None

        common_rejection = quality_filter_reason(
            exhaustion,
            min_confidence=self.squeeze_config.min_confidence,
            min_intensity_score=self.squeeze_config.min_intensity_score,
            min_total_notional_usd=self.squeeze_config.min_total_notional_usd,
            min_event_count=self.squeeze_config.min_event_count,
            max_price_range_pct=self.squeeze_config.max_price_range_pct,
            require_confirmed=self.squeeze_config.require_confirmed_result,
            require_actionable_direction=self.squeeze_config.require_actionable_direction,
        )
        if common_rejection is not None:
            self.remember_no_signal(
                "liquidations_squeeze_quality_filter_failed",
                filter_reason=common_rejection,
                exhaustion=serialize_for_metadata(exhaustion),
                confidence=extract_confidence(exhaustion),
                score=extract_score(exhaustion),
                intensity_score=extract_intensity_score(exhaustion),
                total_notional_usd=str(extract_notional_usd(exhaustion)),
                event_count=extract_event_count(exhaustion),
                min_confidence=self.squeeze_config.min_confidence,
                min_intensity_score=self.squeeze_config.min_intensity_score,
                min_total_notional_usd=str(self.squeeze_config.min_total_notional_usd),
                min_event_count=self.squeeze_config.min_event_count,
            )
            return None

        if not self._passes_squeeze_filters(exhaustion):
            self.remember_no_signal(
                "liquidations_squeeze_strategy_filters_failed",
                exhaustion=serialize_for_metadata(exhaustion),
                confirmed_context=is_confirmed_status(exhaustion),
                require_confirmed_exhaustion_context=(
                    self.squeeze_config.require_confirmed_exhaustion_context
                ),
                severity=self._severity(exhaustion),
                allowed_severities=list(self.squeeze_config.allowed_severities),
                exhaustion_bias=extract_exhaustion_bias(exhaustion),
                min_exhaustion_bias=self.squeeze_config.min_exhaustion_bias,
                bias_delta=extract_bias_delta(exhaustion),
                min_bias_delta=self.squeeze_config.min_bias_delta,
                continuation_bias=extract_continuation_bias(exhaustion),
                max_continuation_bias_after_exhaustion=(
                    self.squeeze_config.max_continuation_bias_after_exhaustion
                ),
                side_imbalance_ratio=extract_side_imbalance_ratio(exhaustion),
                min_side_imbalance_ratio=self.squeeze_config.min_side_imbalance_ratio,
                event_imbalance_ratio=extract_event_imbalance_ratio(exhaustion),
                min_event_imbalance_ratio=self.squeeze_config.min_event_imbalance_ratio,
                acceleration_ratio=extract_acceleration_ratio(exhaustion),
                min_climax_acceleration_ratio=(
                    self.squeeze_config.min_climax_acceleration_ratio
                ),
                cluster_duration_seconds=extract_cluster_duration_seconds(exhaustion),
                max_cluster_duration_seconds=self.squeeze_config.max_cluster_duration_seconds,
                cluster_avg_notional_per_event=str(
                    extract_cluster_avg_notional_per_event(exhaustion)
                ),
                min_avg_notional_per_event=(
                    str(self.squeeze_config.min_avg_notional_per_event)
                    if self.squeeze_config.min_avg_notional_per_event is not None
                    else None
                ),
            )
            return None

        side = self._derive_reversal_side(exhaustion)
        if not is_directional_side(side):
            self.remember_no_signal(
                "liquidations_squeeze_side_not_directional",
                exhaustion=serialize_for_metadata(exhaustion),
                cascade_direction=serialize_for_metadata(extract_direction(exhaustion)),
                reversal_side=serialize_for_metadata(extract_reversal_side(exhaustion)),
            )
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            exhaustion=exhaustion,
        )

        if breakdown.score < self.squeeze_config.min_signal_score:
            self.remember_no_signal(
                "liquidations_squeeze_score_below_minimum",
                score=breakdown.score,
                confidence=breakdown.confidence,
                min_signal_score=self.squeeze_config.min_signal_score,
                score_breakdown=breakdown.to_dict(),
            )
            return None

        if breakdown.confidence < self.squeeze_config.min_signal_confidence:
            self.remember_no_signal(
                "liquidations_squeeze_confidence_below_minimum",
                score=breakdown.score,
                confidence=breakdown.confidence,
                min_signal_confidence=self.squeeze_config.min_signal_confidence,
                score_breakdown=breakdown.to_dict(),
            )
            return None

        source_features = self._source_features(exhaustion)
        tags = self._tags(exhaustion)

        reasons = list(
            dict.fromkeys(
                [
                    "liquidation_squeeze_reversal",
                    f"side:{side.value}",
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "liquidations_setup_family": "liquidation_squeeze_reversal",
            "liquidations_strategy_version": "2.0.0",
            "contract": "liquidations",
            "primary_section": "exhaustion",
            "score_breakdown": breakdown.to_dict(),
            "exhaustion": serialize_for_metadata(exhaustion),
            "event_time": event_time.isoformat() if event_time else None,
            "tags": tags,
            "reversal_side": side.value,
            "cascade_direction": serialize_for_metadata(extract_direction(exhaustion)),
            "severity": self._severity(exhaustion),
            "event_count": extract_event_count(exhaustion),
            "total_notional_usd": str(extract_notional_usd(exhaustion)),
            "intensity_score": extract_intensity_score(exhaustion),
            "continuation_bias": extract_continuation_bias(exhaustion),
            "exhaustion_bias": extract_exhaustion_bias(exhaustion),
            "bias_delta": extract_bias_delta(exhaustion),
            "price_range_pct": extract_price_range_pct(exhaustion),
            "side_imbalance_ratio": extract_side_imbalance_ratio(exhaustion),
            "event_imbalance_ratio": extract_event_imbalance_ratio(exhaustion),
            "acceleration_ratio": extract_acceleration_ratio(exhaustion),
            "cluster_duration_seconds": extract_cluster_duration_seconds(exhaustion),
            "cluster_avg_notional_per_event": str(
                extract_cluster_avg_notional_per_event(exhaustion)
            ),
            "confirmed_context": is_confirmed_status(exhaustion),
        }

        return self.build_liquidations_signal(
            context=context,
            side=side,
            confidence=breakdown.confidence,
            score=breakdown.score,
            setup_type=self.squeeze_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.squeeze_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Context resolution
    # ------------------------------------------------------------------

    def _resolve_exhaustion_context(self, context: StrategyContext) -> Any:
        """
        Resolve liquidation exhaustion/squeeze context from StrategyContext.

        Preferred:
            domain_data["squeeze"] with confirmed=True/status=confirmed.

        Fallback:
            domain_data["exhaustion"].

        Last fallback:
            domain_data["cascade"] if analytics encodes exhaustion fields into
            cascade result.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SqueezeReversalStrategy._resolve_exhaustion_context")
        squeeze = liquidations_item(context, "squeeze")
        exhaustion = liquidations_item(context, "exhaustion")
        cascade = liquidations_item(context, "cascade")

        if squeeze is not None:
            return squeeze

        if exhaustion is not None:
            return exhaustion

        if cascade is not None:
            # Generic fallback for analytics implementations that publish
            # exhaustion_detected using the same CascadeDetectionResult shape.
            if extract_exhaustion_bias(cascade) > 0 or extract_bias_delta(cascade) > 0:
                return cascade

        return None

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _passes_squeeze_filters(self, exhaustion: Any) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SqueezeReversalStrategy._passes_squeeze_filters")
        if self.squeeze_config.require_confirmed_exhaustion_context:
            if not is_confirmed_status(exhaustion):
                return False

        if self.squeeze_config.require_actionable_severity:
            if not self._severity_allowed(exhaustion):
                return False

        if self.squeeze_config.require_favors_exhaustion:
            exhaustion_bias = extract_exhaustion_bias(exhaustion)
            if exhaustion_bias < self.squeeze_config.min_exhaustion_bias:
                return False

        bias_delta = extract_bias_delta(exhaustion)
        if bias_delta < self.squeeze_config.min_bias_delta:
            return False

        if self.squeeze_config.max_continuation_bias_after_exhaustion is not None:
            continuation_bias = extract_continuation_bias(exhaustion)
            if (
                continuation_bias
                > self.squeeze_config.max_continuation_bias_after_exhaustion
            ):
                return False

        if self.squeeze_config.min_side_imbalance_ratio is not None:
            imbalance = extract_side_imbalance_ratio(exhaustion)
            if imbalance < self.squeeze_config.min_side_imbalance_ratio:
                return False

        if self.squeeze_config.min_event_imbalance_ratio is not None:
            imbalance = extract_event_imbalance_ratio(exhaustion)
            if imbalance < self.squeeze_config.min_event_imbalance_ratio:
                return False

        if self.squeeze_config.min_climax_acceleration_ratio is not None:
            acceleration = extract_acceleration_ratio(exhaustion)
            if acceleration < self.squeeze_config.min_climax_acceleration_ratio:
                return False

        if self.squeeze_config.max_cluster_duration_seconds is not None:
            duration = extract_cluster_duration_seconds(exhaustion)
            if duration > self.squeeze_config.max_cluster_duration_seconds:
                return False

        if self.squeeze_config.min_avg_notional_per_event is not None:
            avg_notional = extract_cluster_avg_notional_per_event(exhaustion)
            if avg_notional < self.squeeze_config.min_avg_notional_per_event:
                return False

        return True

    def _severity_allowed(self, exhaustion: Any) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SqueezeReversalStrategy._severity_allowed")
        severity = self._severity(exhaustion)
        allowed = {
            item.strip().lower()
            for item in self.squeeze_config.allowed_severities
            if item.strip()
        }

        return severity in allowed

    # ------------------------------------------------------------------
    # Direction
    # ------------------------------------------------------------------

    def _derive_reversal_side(self, exhaustion: Any) -> SignalSide:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SqueezeReversalStrategy._derive_reversal_side")
        return extract_reversal_side(exhaustion)

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        exhaustion: Any,
    ) -> ScoreBreakdown:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SqueezeReversalStrategy._build_score_breakdown")
        confidence_component = extract_confidence(exhaustion)
        exhaustion_bias = extract_exhaustion_bias(exhaustion)
        bias_delta = extract_bias_delta(exhaustion)
        intensity_score = extract_intensity_score(exhaustion)
        severity_component = severity_score(exhaustion)
        cluster_quality = self._cluster_quality_score(exhaustion)
        imbalance_component = self._imbalance_score(exhaustion)
        acceleration_component = self._acceleration_score(exhaustion)

        score = weighted_score(
            {
                "confidence": confidence_component,
                "exhaustion_bias": exhaustion_bias,
                "bias_delta": bias_delta,
                "intensity": intensity_score,
                "severity": severity_component,
                "cluster_quality": cluster_quality,
                "imbalance": imbalance_component,
                "acceleration": acceleration_component,
            },
            {
                "confidence": self.squeeze_config.score_confidence_weight,
                "exhaustion_bias": self.squeeze_config.score_exhaustion_bias_weight,
                "bias_delta": self.squeeze_config.score_bias_delta_weight,
                "intensity": self.squeeze_config.score_intensity_weight,
                "severity": self.squeeze_config.score_severity_weight,
                "cluster_quality": self.squeeze_config.score_cluster_quality_weight,
                "imbalance": self.squeeze_config.score_imbalance_weight,
                "acceleration": self.squeeze_config.score_acceleration_weight,
            },
            default=extract_score(exhaustion),
        )

        event_time = extract_event_time(exhaustion)
        fresh_score = freshness_score(
            event_time=event_time,
            now=context.timestamp,
            stale_after_seconds=self.squeeze_config.stale_feature_max_age_seconds,
        )

        context_score = weighted_score(
            {
                "exhaustion_bias": exhaustion_bias,
                "bias_delta": bias_delta,
                "cluster_quality": cluster_quality,
                "imbalance": imbalance_component,
                "acceleration": acceleration_component,
            },
            {
                "exhaustion_bias": 0.35,
                "bias_delta": 0.20,
                "cluster_quality": 0.20,
                "imbalance": 0.15,
                "acceleration": 0.10,
            },
            default=0.0,
        )

        confirmation_score = 1.0 if is_confirmed_status(exhaustion) else 0.0

        confidence = confidence_from_components(
            primary=confidence_component,
            context=context_score,
            confirmation=confirmation_score,
            freshness=fresh_score,
        )

        reasons = [
            f"confidence:{confidence_component:.3f}",
            f"exhaustion_bias:{exhaustion_bias:.3f}",
            f"bias_delta:{bias_delta:.3f}",
            f"intensity_score:{intensity_score:.3f}",
            f"severity_score:{severity_component:.3f}",
        ]

        confirmations: list[str] = []

        if is_confirmed_status(exhaustion):
            confirmations.append(self.squeeze_config.tag_confirmed_context)

        if exhaustion_bias >= self.squeeze_config.min_exhaustion_bias:
            confirmations.append("exhaustion_bias_confirmed")

        if bias_delta >= self.squeeze_config.min_bias_delta:
            confirmations.append("bias_delta_confirmed")

        if intensity_score >= self.squeeze_config.min_intensity_score:
            confirmations.append("intensity_confirmed")

        if extract_notional_usd(exhaustion) >= self.squeeze_config.min_total_notional_usd:
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
                "exhaustion_bias": exhaustion_bias,
                "bias_delta": bias_delta,
                "intensity_score": intensity_score,
                "severity_score": severity_component,
                "cluster_quality": cluster_quality,
                "imbalance_score": imbalance_component,
                "acceleration_score": acceleration_component,
                "context_score": context_score,
                "confirmation_score": confirmation_score,
                "freshness_score": fresh_score,
            },
            weights={
                "score_confidence_weight": self.squeeze_config.score_confidence_weight,
                "score_exhaustion_bias_weight": (
                    self.squeeze_config.score_exhaustion_bias_weight
                ),
                "score_bias_delta_weight": self.squeeze_config.score_bias_delta_weight,
                "score_intensity_weight": self.squeeze_config.score_intensity_weight,
                "score_severity_weight": self.squeeze_config.score_severity_weight,
                "score_cluster_quality_weight": (
                    self.squeeze_config.score_cluster_quality_weight
                ),
                "score_imbalance_weight": self.squeeze_config.score_imbalance_weight,
                "score_acceleration_weight": self.squeeze_config.score_acceleration_weight,
            },
            reasons=reasons,
            confirmations=confirmations,
        ).normalize()

    @staticmethod
    def _imbalance_score(exhaustion: Any) -> float:
        _strategy_logger = logging.getLogger(__name__ + ".SqueezeReversalStrategy._imbalance_score")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SqueezeReversalStrategy._imbalance_score")
        side_imbalance = extract_side_imbalance_ratio(exhaustion)
        event_imbalance = extract_event_imbalance_ratio(exhaustion)

        if side_imbalance > 0:
            return side_imbalance

        if event_imbalance > 0:
            return event_imbalance

        return 0.50

    @staticmethod
    def _acceleration_score(exhaustion: Any) -> float:
        _strategy_logger = logging.getLogger(__name__ + ".SqueezeReversalStrategy._acceleration_score")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SqueezeReversalStrategy._acceleration_score")
        acceleration = extract_acceleration_ratio(exhaustion)

        if acceleration <= 0:
            return 0.50

        # 1.0 = neutral, 2.0+ = strong climax acceleration.
        return unit_score((acceleration - 1.0) / 1.0)

    @staticmethod
    def _cluster_quality_score(exhaustion: Any) -> float:
        _strategy_logger = logging.getLogger(__name__ + ".SqueezeReversalStrategy._cluster_quality_score")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SqueezeReversalStrategy._cluster_quality_score")
        duration = extract_cluster_duration_seconds(exhaustion)
        avg_notional = extract_cluster_avg_notional_per_event(exhaustion)

        duration_score = 0.50
        if duration > 0:
            # Shorter liquidation climax clusters are better for squeeze reversal.
            if duration <= 5:
                duration_score = 1.00
            elif duration <= 10:
                duration_score = 0.80
            elif duration <= 20:
                duration_score = 0.55
            else:
                duration_score = 0.30

        notional_score = 0.50
        if avg_notional > Decimal("0"):
            if avg_notional >= Decimal("250000"):
                notional_score = 1.00
            elif avg_notional >= Decimal("100000"):
                notional_score = 0.80
            elif avg_notional >= Decimal("50000"):
                notional_score = 0.60
            else:
                notional_score = 0.35

        return unit_score(0.5 * duration_score + 0.5 * notional_score)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _severity(self, exhaustion: Any) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SqueezeReversalStrategy._severity")
        return extract_severity_label(exhaustion, default="unknown")

    def _source_features(self, exhaustion: Any) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SqueezeReversalStrategy._source_features")
        features = [
            LIQUIDATIONS_FEATURES.EXHAUSTION,
            LIQUIDATIONS_FEATURES.EXHAUSTION_CONFIDENCE,
            LIQUIDATIONS_FEATURES.EXHAUSTION_BIAS,
            LIQUIDATIONS_FEATURES.EXHAUSTION_BIAS_DELTA,
            LIQUIDATIONS_FEATURES.EXHAUSTION_CONFIRMED,
            LIQUIDATIONS_FEATURES.CASCADE_DIRECTION,
            LIQUIDATIONS_FEATURES.CASCADE_INTENSITY,
            LIQUIDATIONS_FEATURES.CASCADE_SEVERITY,
            LIQUIDATIONS_FEATURES.CASCADE_NOTIONAL_USD,
            LIQUIDATIONS_FEATURES.CASCADE_EVENT_COUNT,
        ]

        if extract_side_imbalance_ratio(exhaustion) > 0:
            features.append(LIQUIDATIONS_FEATURES.CLUSTER_SIDE_IMBALANCE_RATIO)

        if extract_event_imbalance_ratio(exhaustion) > 0:
            features.append(LIQUIDATIONS_FEATURES.CLUSTER_EVENT_IMBALANCE_RATIO)

        if extract_acceleration_ratio(exhaustion) > 0:
            features.append(LIQUIDATIONS_FEATURES.CLUSTER_ACCELERATION_RATIO)

        if extract_cluster_duration_seconds(exhaustion) > 0:
            features.append(LIQUIDATIONS_FEATURES.CLUSTER_DURATION_SECONDS)

        if extract_cluster_avg_notional_per_event(exhaustion) > Decimal("0"):
            features.append(LIQUIDATIONS_FEATURES.CLUSTER_AVG_NOTIONAL_PER_EVENT)

        return list(dict.fromkeys(features))

    def _tags(self, exhaustion: Any) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SqueezeReversalStrategy._tags")
        tags = [
            self.squeeze_config.tag_liquidations,
            self.squeeze_config.tag_squeeze,
            self.squeeze_config.tag_squeeze_reversal,
            self.squeeze_config.tag_exhaustion,
            self.squeeze_config.tag_reversal,
            self.squeeze_config.tag_climax,
        ]

        if is_confirmed_status(exhaustion):
            tags.append(self.squeeze_config.tag_confirmed_context)

        if (
            extract_side_imbalance_ratio(exhaustion) >= 0.60
            or extract_event_imbalance_ratio(exhaustion) >= 0.60
        ):
            tags.append(self.squeeze_config.tag_imbalanced_cluster)

        if extract_acceleration_ratio(exhaustion) >= 1.50:
            tags.append(self.squeeze_config.tag_acceleration)

        tags.append(f"severity:{self._severity(exhaustion)}")

        return list(dict.fromkeys(tags))


__all__ = [
    "SqueezeReversalStrategy",
    "SqueezeReversalStrategyConfig",
]