# trading_system/strategy/strategies/liquidity/liquidity_map_bias_strategy.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from analytics.liquidity.enums import LiquidityBias, LiquiditySide, SweepStatus
from analytics.liquidity.models import (
    LiquidityLevel,
    LiquidityMapSnapshot,
    StopCluster,
)
from core.event_bus import EventBus
from core.scheduler import Scheduler
from .base import (
    LIQUIDITY_FEATURES,
    LiquidityStrategyConfig,
    LiquidityTradingStrategy,
)
from .utils import (
    ScoreBreakdown,
    analytics_signal_confidence,
    best_zone_for_side,
    collect_targets_above,
    collect_targets_below,
    confidence_from_components,
    distance_pct,
    distance_score,
    downside_bias_edge,
    freshness_score,
    is_directional_side,
    is_terminal_item,
    magnet_score_down,
    magnet_score_up,
    reference_price,
    serialize_for_metadata,
    signed_score,
    snapshot_liquidity_strength,
    sweep_risk_down,
    sweep_risk_up,
    unit_score,
    upside_bias_edge,
    weighted_score,
)
from ...config import StrategyConfig, StrategyDefinitionConfig
from ...enums import (
    FilterDecision,
    SetupType,
    SignalPriority,
    SignalSide,
    StrategyCategory,
)
from ...exceptions import StrategyConfigError
from ...models import FilterResult, StrategyContext, StrategySignal, TargetPlan


@dataclass(slots=True)
class LiquidityMapBiasStrategyConfig(LiquidityStrategyConfig):
    """
    Unified liquidity map directional bias strategy config.

    Strategy idea:
    - read LiquidityMapSnapshot from StrategyContext;
    - infer directional bias from liquidity map pressure, upside/downside edges,
      liquidity magnets, sweep risks and analytics signal confidence;
    - return internal StrategySignal as confluence/context signal;
    - target is optional;
    - leave routing, filtering, confluence and risk-ready conversion to SignalProcessor.
    """

    min_directional_edge: float = 0.42
    min_edge_delta: float = 0.10
    min_pressure_abs: float = 0.12
    min_analytics_confidence: float = 0.35

    min_target_distance_pct: float = 0.0010
    max_target_distance_pct: float = 0.0800

    fallback_stop_pct: float = 0.0050
    long_stop_offset: float = 0.9980
    short_stop_offset: float = 1.0020

    require_directional_bias: bool = True
    allow_signal_without_target: bool = True
    reject_terminal_targets: bool = True

    score_directional_edge_weight: float = 0.32
    score_edge_delta_weight: float = 0.16
    score_pressure_weight: float = 0.16
    score_analytics_confidence_weight: float = 0.16
    score_target_weight: float = 0.12
    score_zone_weight: float = 0.08

    confidence_directional_edge_weight: float = 0.30
    confidence_pressure_weight: float = 0.18
    confidence_analytics_weight: float = 0.22
    confidence_target_weight: float = 0.14
    confidence_zone_weight: float = 0.10
    confidence_strength_weight: float = 0.06

    high_priority_score: float = 0.82
    critical_priority_score: float = 0.92

    tag_map_bias: str = "liquidity_map_bias"
    tag_directional_context: str = "directional_context"
    tag_confluence_signal: str = "confluence_signal"
    tag_upside_bias: str = "upside_liquidity_bias"
    tag_downside_bias: str = "downside_liquidity_bias"
    tag_target_available: str = "target_available"

    default_priority: SignalPriority = SignalPriority.MEDIUM
    default_setup_type: SetupType = SetupType.CONTINUATION

    required_liquidity_features: tuple[str, ...] = (
        LIQUIDITY_FEATURES.SNAPSHOT,
    )

    def validate(self) -> None:
        LiquidityStrategyConfig.validate(self)

        bounded_fields = {
            "min_directional_edge": self.min_directional_edge,
            "min_edge_delta": self.min_edge_delta,
            "min_pressure_abs": self.min_pressure_abs,
            "min_analytics_confidence": self.min_analytics_confidence,
            "min_target_distance_pct": self.min_target_distance_pct,
            "max_target_distance_pct": self.max_target_distance_pct,
            "fallback_stop_pct": self.fallback_stop_pct,
            "high_priority_score": self.high_priority_score,
            "critical_priority_score": self.critical_priority_score,
        }

        for field_name, value in bounded_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        if self.min_target_distance_pct > self.max_target_distance_pct:
            raise StrategyConfigError(
                "min_target_distance_pct cannot be greater than max_target_distance_pct"
            )

        if self.long_stop_offset <= 0:
            raise StrategyConfigError("long_stop_offset must be > 0")

        if self.short_stop_offset <= 0:
            raise StrategyConfigError("short_stop_offset must be > 0")

        weights = {
            "score_directional_edge_weight": self.score_directional_edge_weight,
            "score_edge_delta_weight": self.score_edge_delta_weight,
            "score_pressure_weight": self.score_pressure_weight,
            "score_analytics_confidence_weight": (
                self.score_analytics_confidence_weight
            ),
            "score_target_weight": self.score_target_weight,
            "score_zone_weight": self.score_zone_weight,
            "confidence_directional_edge_weight": (
                self.confidence_directional_edge_weight
            ),
            "confidence_pressure_weight": self.confidence_pressure_weight,
            "confidence_analytics_weight": self.confidence_analytics_weight,
            "confidence_target_weight": self.confidence_target_weight,
            "confidence_zone_weight": self.confidence_zone_weight,
            "confidence_strength_weight": self.confidence_strength_weight,
        }

        for field_name, value in weights.items():
            if value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        if (
            self.score_directional_edge_weight
            + self.score_edge_delta_weight
            + self.score_pressure_weight
            + self.score_analytics_confidence_weight
            + self.score_target_weight
            + self.score_zone_weight
        ) <= 0:
            raise StrategyConfigError("score weights sum must be > 0")

        if (
            self.confidence_directional_edge_weight
            + self.confidence_pressure_weight
            + self.confidence_analytics_weight
            + self.confidence_target_weight
            + self.confidence_zone_weight
            + self.confidence_strength_weight
        ) <= 0:
            raise StrategyConfigError("confidence weights sum must be > 0")

        for attr in (
            "tag_map_bias",
            "tag_directional_context",
            "tag_confluence_signal",
            "tag_upside_bias",
            "tag_downside_bias",
            "tag_target_available",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")

        if not self.required_liquidity_features:
            raise StrategyConfigError("required_liquidity_features cannot be empty")

        for feature in self.required_liquidity_features:
            if not isinstance(feature, str) or not feature.strip():
                raise StrategyConfigError(
                    "required_liquidity_features cannot contain empty feature names"
                )


class LiquidityMapBiasStrategy(LiquidityTradingStrategy):
    """
    Unified directional liquidity map bias strategy.

    Input:
        StrategyContext with FeatureSource.LIQUIDITY domain data and LiquidityMapSnapshot.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    It returns an internal StrategySignal for the shared strategy pipeline.
    """

    component_namespace = "strategy.liquidity.map_bias"
    category: StrategyCategory = StrategyCategory.LIQUIDITY
    default_setup_type: SetupType = SetupType.CONTINUATION

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        liquidity_config: LiquidityMapBiasStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_liquidity_config = liquidity_config or LiquidityMapBiasStrategyConfig()
        resolved_liquidity_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            liquidity_config=resolved_liquidity_config,
            service_name=service_name,
        )

        self.bias_config: LiquidityMapBiasStrategyConfig = resolved_liquidity_config

    @property
    def strategy_name(self) -> str:
        return "liquidity_map_bias_strategy"

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(self.bias_config.required_liquidity_features)

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        self.validate_context_requirements(context)

        snapshot = self.liquidity_snapshot(context)
        if snapshot is None:
            return None

        if not self.base_context_is_valid(context=context, snapshot=snapshot):
            return None

        current_price = self.current_price(context=context, snapshot=snapshot)
        if current_price is None or current_price <= 0:
            return None

        filters = self._run_pre_filters(
            context=context,
            snapshot=snapshot,
            current_price=current_price,
        )
        if any(item.blocked for item in filters):
            return None

        side = self._infer_side(snapshot)
        if not is_directional_side(side):
            return None

        target = self._target_for_side(
            snapshot=snapshot,
            current_price=current_price,
            side=side,
        )

        if target is None and not self.bias_config.allow_signal_without_target:
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            snapshot=snapshot,
            current_price=current_price,
            side=side,
            target=target,
        )

        if breakdown.score < self.bias_config.min_signal_score:
            return None

        if breakdown.confidence < self.bias_config.min_signal_confidence:
            return None

        stop_loss = self._resolve_stop_price(
            side=side,
            current_price=current_price,
            invalidation_level=self._invalidation_level_for_side(
                snapshot=snapshot,
                current_price=current_price,
                side=side,
            ),
        )
        take_profit = reference_price(target) if target is not None else None

        target_plans = self._target_plans(
            snapshot=snapshot,
            current_price=current_price,
            side=side,
            target=target,
            stop_loss=stop_loss,
        )

        reasons = list(
            dict.fromkeys(
                [
                    self._primary_reason(snapshot=snapshot, side=side),
                    self._target_reason(target),
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))
        source_features = self._source_features(side=side, target=target)

        metadata = {
            "liquidity_setup_family": "liquidity_map_bias",
            "liquidity_strategy_version": "2.0.0",
            "score_breakdown": breakdown.to_dict(),
            "tags": self._tags(side=side, target=target, snapshot=snapshot),
            "side": side.value,
            "current_price": current_price,
            "target": self._target_metadata(target),
            "target_price": take_profit,
            "stop_loss": stop_loss,
            "upside_edge": upside_bias_edge(snapshot),
            "downside_edge": downside_bias_edge(snapshot),
            "edge_delta": upside_bias_edge(snapshot) - downside_bias_edge(snapshot),
            "bias": serialize_for_metadata(getattr(snapshot, "bias", None)),
            "analytics_confidence": analytics_signal_confidence(snapshot),
            "liquidity_strength": snapshot_liquidity_strength(snapshot),
            "liquidity_pressure_score": signed_score(
                getattr(snapshot, "liquidity_pressure_score", 0.0)
            ),
            "above_liquidity_score": unit_score(
                getattr(snapshot, "above_liquidity_score", 0.0)
            ),
            "below_liquidity_score": unit_score(
                getattr(snapshot, "below_liquidity_score", 0.0)
            ),
            "magnet_score_up": magnet_score_up(snapshot),
            "magnet_score_down": magnet_score_down(snapshot),
            "sweep_risk_up": sweep_risk_up(snapshot),
            "sweep_risk_down": sweep_risk_down(snapshot),
            "filters": [item.to_dict() for item in filters],
            "target_plans": [plan.to_dict() for plan in target_plans],
        }

        return self.build_liquidity_signal(
            context=context,
            side=side,
            confidence=breakdown.confidence,
            score=breakdown.score,
            setup_type=self.bias_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self._resolve_priority(
                score=breakdown.score,
                confidence=breakdown.confidence,
            ),
            snapshot=snapshot,
            current_price=current_price,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _run_pre_filters(
        self,
        *,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> list[FilterResult]:
        results = self.run_common_pre_filters(
            context=context,
            snapshot=snapshot,
            current_price=current_price,
        )

        upside_edge = upside_bias_edge(snapshot)
        downside_edge = downside_bias_edge(snapshot)
        edge_delta = abs(upside_edge - downside_edge)
        directional_edge = max(upside_edge, downside_edge)
        pressure_abs = abs(signed_score(getattr(snapshot, "liquidity_pressure_score", 0.0)))

        if directional_edge < self.bias_config.min_directional_edge:
            results.append(
                FilterResult(
                    name="liquidity_map_bias_directional_edge",
                    decision=FilterDecision.BLOCK,
                    reason=(
                        "Directional liquidity map edge too weak: "
                        f"upside={upside_edge:.4f}, downside={downside_edge:.4f}, "
                        f"required={self.bias_config.min_directional_edge:.4f}"
                    ),
                )
            )
        else:
            results.append(
                FilterResult(
                    name="liquidity_map_bias_directional_edge",
                    decision=FilterDecision.PASS,
                    reason=(
                        "Directional liquidity map edge accepted: "
                        f"upside={upside_edge:.4f}, downside={downside_edge:.4f}"
                    ),
                )
            )

        if (
            edge_delta < self.bias_config.min_edge_delta
            and pressure_abs < self.bias_config.min_pressure_abs
        ):
            results.append(
                FilterResult(
                    name="liquidity_map_bias_separation",
                    decision=FilterDecision.BLOCK,
                    reason=(
                        "Liquidity map bias separation too weak: "
                        f"edge_delta={edge_delta:.4f}, pressure_abs={pressure_abs:.4f}"
                    ),
                )
            )
        else:
            results.append(
                FilterResult(
                    name="liquidity_map_bias_separation",
                    decision=FilterDecision.PASS,
                    reason=(
                        "Liquidity map bias separation accepted: "
                        f"edge_delta={edge_delta:.4f}, pressure_abs={pressure_abs:.4f}"
                    ),
                )
            )

        analytics_confidence = analytics_signal_confidence(snapshot)
        if analytics_confidence < self.bias_config.min_analytics_confidence:
            results.append(
                FilterResult(
                    name="liquidity_map_bias_analytics_confidence",
                    decision=FilterDecision.BLOCK,
                    reason=(
                        "Analytics liquidity signal confidence too weak: "
                        f"{analytics_confidence:.4f}"
                    ),
                )
            )
        else:
            results.append(
                FilterResult(
                    name="liquidity_map_bias_analytics_confidence",
                    decision=FilterDecision.PASS,
                    reason=(
                        "Analytics liquidity signal confidence accepted: "
                        f"{analytics_confidence:.4f}"
                    ),
                )
            )

        for item in results:
            item.validate()

        return results

    # ------------------------------------------------------------------
    # Direction
    # ------------------------------------------------------------------

    def _infer_side(self, snapshot: LiquidityMapSnapshot) -> SignalSide:
        upside_edge = upside_bias_edge(snapshot)
        downside_edge = downside_bias_edge(snapshot)
        delta = upside_edge - downside_edge
        pressure = signed_score(getattr(snapshot, "liquidity_pressure_score", 0.0))

        if getattr(snapshot, "bias", None) == LiquidityBias.UP:
            if upside_edge >= self.bias_config.min_directional_edge:
                return SignalSide.LONG

        if getattr(snapshot, "bias", None) == LiquidityBias.DOWN:
            if downside_edge >= self.bias_config.min_directional_edge:
                return SignalSide.SHORT

        if (
            delta >= self.bias_config.min_edge_delta
            and upside_edge >= self.bias_config.min_directional_edge
            and pressure >= -0.05
        ):
            return SignalSide.LONG

        if (
            delta <= -self.bias_config.min_edge_delta
            and downside_edge >= self.bias_config.min_directional_edge
            and pressure <= 0.05
        ):
            return SignalSide.SHORT

        if (
            pressure >= self.bias_config.min_pressure_abs
            and upside_edge >= self.bias_config.min_directional_edge * 0.85
        ):
            return SignalSide.LONG

        if (
            pressure <= -self.bias_config.min_pressure_abs
            and downside_edge >= self.bias_config.min_directional_edge * 0.85
        ):
            return SignalSide.SHORT

        return SignalSide.UNKNOWN

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------

    def _target_for_side(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
    ) -> LiquidityLevel | StopCluster | None:
        if side is SignalSide.LONG:
            candidates = [
                getattr(snapshot, "nearest_above_level", None),
                getattr(snapshot, "strongest_cluster_above", None),
                *self._collect_valid_targets_above(snapshot, current_price),
            ]
            liquidity_side = LiquiditySide.BUY_SIDE
        elif side is SignalSide.SHORT:
            candidates = [
                getattr(snapshot, "nearest_below_level", None),
                getattr(snapshot, "strongest_cluster_below", None),
                *self._collect_valid_targets_below(snapshot, current_price),
            ]
            liquidity_side = LiquiditySide.SELL_SIDE
        else:
            return None

        valid = [
            item
            for item in candidates
            if item is not None
            and self._is_valid_bias_target(
                item=item,
                current_price=current_price,
                side=side,
            )
        ]

        valid = list(self.dedupe_liquidity_items(valid))
        if not valid:
            return None

        zone = best_zone_for_side(
            snapshot=snapshot,
            side=liquidity_side,
            current_price=current_price,
        )

        if zone is not None:
            zone_center = reference_price(zone)
            zone_aligned = [
                item
                for item in valid
                if distance_pct(reference_price(item), zone_center) <= 0.01
            ]
            if zone_aligned:
                valid = zone_aligned

        if side is SignalSide.LONG:
            return min(valid, key=reference_price)

        return max(valid, key=reference_price)

    def _collect_valid_targets_above(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> list[LiquidityLevel | StopCluster]:
        return [
            item
            for item in collect_targets_above(snapshot, current_price)
            if self._is_valid_bias_target(
                item=item,
                current_price=current_price,
                side=SignalSide.LONG,
            )
        ]

    def _collect_valid_targets_below(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> list[LiquidityLevel | StopCluster]:
        return [
            item
            for item in collect_targets_below(snapshot, current_price)
            if self._is_valid_bias_target(
                item=item,
                current_price=current_price,
                side=SignalSide.SHORT,
            )
        ]

    def _is_valid_bias_target(
        self,
        *,
        item: LiquidityLevel | StopCluster,
        current_price: float,
        side: SignalSide,
    ) -> bool:
        ref_price = reference_price(item)
        if ref_price <= 0 or current_price <= 0:
            return False

        if side is SignalSide.LONG and ref_price <= current_price:
            return False

        if side is SignalSide.SHORT and ref_price >= current_price:
            return False

        target_distance = distance_pct(ref_price, current_price)

        if target_distance < self.bias_config.min_target_distance_pct:
            return False

        if target_distance > self.bias_config.max_target_distance_pct:
            return False

        if self.bias_config.reject_terminal_targets and is_terminal_item(item):
            return False

        sweep_status = getattr(item, "sweep_status", None)
        if sweep_status in {SweepStatus.SWEPT, SweepStatus.PARTIALLY_SWEPT}:
            return False

        return True

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
        target: LiquidityLevel | StopCluster | None,
    ) -> ScoreBreakdown:
        directional_edge = self._directional_edge(snapshot=snapshot, side=side)
        opposite_edge = self._opposite_edge(snapshot=snapshot, side=side)
        edge_delta = unit_score(abs(directional_edge - opposite_edge))
        pressure = self._pressure_alignment(snapshot=snapshot, side=side)
        analytics_confidence = analytics_signal_confidence(snapshot)
        target_score = self._target_quality_score(
            target=target,
            current_price=current_price,
        )
        zone_alignment = self._zone_alignment_score(
            snapshot=snapshot,
            current_price=current_price,
            side=side,
        )
        liquidity_strength = snapshot_liquidity_strength(snapshot)

        score = weighted_score(
            {
                "directional_edge": directional_edge,
                "edge_delta": edge_delta,
                "pressure": pressure,
                "analytics_confidence": analytics_confidence,
                "target": target_score,
                "zone": zone_alignment,
            },
            {
                "directional_edge": self.bias_config.score_directional_edge_weight,
                "edge_delta": self.bias_config.score_edge_delta_weight,
                "pressure": self.bias_config.score_pressure_weight,
                "analytics_confidence": (
                    self.bias_config.score_analytics_confidence_weight
                ),
                "target": self.bias_config.score_target_weight,
                "zone": self.bias_config.score_zone_weight,
            },
            default=directional_edge,
        )

        snapshot_time = getattr(snapshot, "timestamp", None)
        fresh_score = freshness_score(
            event_time=snapshot_time,
            now=context.timestamp,
            stale_after_seconds=self.bias_config.max_snapshot_age_seconds,
        )

        confidence_primary = weighted_score(
            {
                "directional_edge": directional_edge,
                "pressure": pressure,
                "analytics": analytics_confidence,
                "target": target_score,
                "zone": zone_alignment,
                "strength": liquidity_strength,
            },
            {
                "directional_edge": self.bias_config.confidence_directional_edge_weight,
                "pressure": self.bias_config.confidence_pressure_weight,
                "analytics": self.bias_config.confidence_analytics_weight,
                "target": self.bias_config.confidence_target_weight,
                "zone": self.bias_config.confidence_zone_weight,
                "strength": self.bias_config.confidence_strength_weight,
            },
            default=directional_edge,
        )

        confidence = confidence_from_components(
            primary=confidence_primary,
            context=weighted_score(
                {
                    "edge_delta": edge_delta,
                    "pressure": pressure,
                    "strength": liquidity_strength,
                },
                {
                    "edge_delta": 0.35,
                    "pressure": 0.35,
                    "strength": 0.30,
                },
                default=0.0,
            ),
            confirmation=analytics_confidence,
            freshness=fresh_score,
        )

        reasons = [
            f"directional_edge:{directional_edge:.3f}",
            f"opposite_edge:{opposite_edge:.3f}",
            f"edge_delta:{edge_delta:.3f}",
            f"pressure_alignment:{pressure:.3f}",
            f"analytics_confidence:{analytics_confidence:.3f}",
        ]

        confirmations: list[str] = []

        if directional_edge >= self.bias_config.min_directional_edge:
            confirmations.append("directional_edge_confirmed")

        if edge_delta >= self.bias_config.min_edge_delta:
            confirmations.append("edge_separation_confirmed")

        if pressure >= self.bias_config.min_pressure_abs:
            confirmations.append("pressure_alignment_confirmed")

        if analytics_confidence >= self.bias_config.min_analytics_confidence:
            confirmations.append("analytics_confidence_confirmed")

        if target is not None:
            confirmations.append("liquidity_target_available")

        return ScoreBreakdown(
            score=score,
            confidence=confidence,
            components={
                "directional_edge": directional_edge,
                "opposite_edge": opposite_edge,
                "edge_delta": edge_delta,
                "pressure_alignment": pressure,
                "analytics_confidence": analytics_confidence,
                "target_score": target_score,
                "zone_alignment": zone_alignment,
                "liquidity_strength": liquidity_strength,
                "freshness_score": fresh_score,
            },
            weights={
                "score_directional_edge_weight": (
                    self.bias_config.score_directional_edge_weight
                ),
                "score_edge_delta_weight": self.bias_config.score_edge_delta_weight,
                "score_pressure_weight": self.bias_config.score_pressure_weight,
                "score_analytics_confidence_weight": (
                    self.bias_config.score_analytics_confidence_weight
                ),
                "score_target_weight": self.bias_config.score_target_weight,
                "score_zone_weight": self.bias_config.score_zone_weight,
            },
            reasons=reasons,
            confirmations=confirmations,
        ).normalize()

    def _directional_edge(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        side: SignalSide,
    ) -> float:
        if side is SignalSide.LONG:
            return upside_bias_edge(snapshot)

        if side is SignalSide.SHORT:
            return downside_bias_edge(snapshot)

        return 0.0

    def _opposite_edge(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        side: SignalSide,
    ) -> float:
        if side is SignalSide.LONG:
            return downside_bias_edge(snapshot)

        if side is SignalSide.SHORT:
            return upside_bias_edge(snapshot)

        return 0.0

    def _pressure_alignment(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        side: SignalSide,
    ) -> float:
        pressure = signed_score(getattr(snapshot, "liquidity_pressure_score", 0.0))

        if side is SignalSide.LONG:
            return unit_score(max(pressure, 0.0))

        if side is SignalSide.SHORT:
            return unit_score(max(-pressure, 0.0))

        return 0.0

    def _zone_alignment_score(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
    ) -> float:
        liquidity_side = (
            LiquiditySide.BUY_SIDE
            if side is SignalSide.LONG
            else LiquiditySide.SELL_SIDE
        )
        zone = best_zone_for_side(
            snapshot=snapshot,
            side=liquidity_side,
            current_price=current_price,
        )
        if zone is None:
            return 0.0

        return unit_score(getattr(zone, "score", 0.0))

    def _target_quality_score(
        self,
        *,
        target: LiquidityLevel | StopCluster | None,
        current_price: float,
    ) -> float:
        if target is None:
            return 0.0

        target_distance_score = distance_score(
            price=reference_price(target),
            current_price=current_price,
            min_distance_pct=self.bias_config.min_target_distance_pct,
            max_distance_pct=self.bias_config.max_target_distance_pct,
        )

        if isinstance(target, StopCluster):
            return unit_score(
                0.45 * unit_score(getattr(target, "confidence", 0.0))
                + 0.30 * unit_score(getattr(target, "estimated_stop_density", 0.0))
                + 0.25 * target_distance_score
            )

        if isinstance(target, LiquidityLevel):
            touches = max(int(getattr(target, "touches_count", 0) or 0), 0)
            reactions = max(int(getattr(target, "reaction_count", 0) or 0), 0)

            return unit_score(
                0.48 * unit_score(getattr(target, "confidence", 0.0))
                + 0.24 * target_distance_score
                + 0.14 * min(touches / 6.0, 1.0)
                + 0.14 * min(reactions / 4.0, 1.0)
            )

        return 0.0

    # ------------------------------------------------------------------
    # Trade levels
    # ------------------------------------------------------------------

    def _resolve_stop_price(
        self,
        *,
        side: SignalSide,
        current_price: float,
        invalidation_level: LiquidityLevel | StopCluster | None,
    ) -> float | None:
        if current_price <= 0:
            return None

        anchor = reference_price(invalidation_level) if invalidation_level is not None else 0.0

        if side is SignalSide.LONG:
            if anchor > 0 and anchor < current_price:
                return anchor * self.bias_config.long_stop_offset
            return current_price * (1.0 - self.bias_config.fallback_stop_pct)

        if side is SignalSide.SHORT:
            if anchor > 0 and anchor > current_price:
                return anchor * self.bias_config.short_stop_offset
            return current_price * (1.0 + self.bias_config.fallback_stop_pct)

        return None

    def _invalidation_level_for_side(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
    ) -> LiquidityLevel | StopCluster | None:
        if side is SignalSide.LONG:
            candidates = collect_targets_below(snapshot, current_price)
            return candidates[0] if candidates else None

        if side is SignalSide.SHORT:
            candidates = collect_targets_above(snapshot, current_price)
            return candidates[0] if candidates else None

        return None

    def _target_plans(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
        target: LiquidityLevel | StopCluster | None,
        stop_loss: float | None,
    ) -> list[TargetPlan]:
        result: list[TargetPlan] = []

        primary_price = reference_price(target) if target is not None else 0.0
        if primary_price > 0:
            result.append(
                TargetPlan(
                    price=primary_price,
                    size_fraction=1.0,
                    rr=self._compute_rr(
                        current_price=current_price,
                        stop_price=stop_loss,
                        target_price=primary_price,
                        side=side,
                    ),
                    label="liquidity_bias_target",
                    metadata={
                        "source": "liquidity_map_bias_target",
                        "target_type": self._target_type(target),
                    },
                )
            )

        for item in result:
            item.validate()

        return result

    @staticmethod
    def _compute_rr(
        *,
        current_price: float,
        stop_price: float | None,
        target_price: float | None,
        side: SignalSide,
    ) -> float | None:
        if current_price <= 0 or stop_price is None or target_price is None:
            return None

        risk = abs(current_price - stop_price)
        reward = (
            target_price - current_price
            if side is SignalSide.LONG
            else current_price - target_price
        )

        if risk <= 0 or reward <= 0:
            return None

        return reward / risk

    @staticmethod
    def _target_type(target: LiquidityLevel | StopCluster | None) -> str | None:
        if target is None:
            return None

        if isinstance(target, StopCluster):
            return "stop_cluster"

        if isinstance(target, LiquidityLevel):
            level_type = getattr(target, "level_type", None)
            if hasattr(level_type, "value"):
                return str(level_type.value)
            return str(level_type)

        return target.__class__.__name__

    # ------------------------------------------------------------------
    # Metadata / reasons
    # ------------------------------------------------------------------

    def _resolve_priority(
        self,
        *,
        score: float,
        confidence: float,
    ) -> SignalPriority:
        combined = unit_score(0.55 * score + 0.45 * confidence)

        if combined >= self.bias_config.critical_priority_score:
            return SignalPriority.URGENT

        if combined >= self.bias_config.high_priority_score:
            return SignalPriority.HIGH

        return self.bias_config.default_priority

    def _source_features(
        self,
        *,
        side: SignalSide,
        target: LiquidityLevel | StopCluster | None,
    ) -> list[str]:
        features = [
            LIQUIDITY_FEATURES.SNAPSHOT,
            LIQUIDITY_FEATURES.MAP_SNAPSHOT,
            LIQUIDITY_FEATURES.PRESSURE_SCORE,
            LIQUIDITY_FEATURES.BIAS,
        ]

        if side is SignalSide.LONG:
            features.extend(
                [
                    LIQUIDITY_FEATURES.ABOVE_LIQUIDITY_SCORE,
                    LIQUIDITY_FEATURES.MAGNET_UP,
                    LIQUIDITY_FEATURES.SWEEP_RISK_UP,
                    LIQUIDITY_FEATURES.NEAREST_ABOVE_LEVEL,
                    LIQUIDITY_FEATURES.STRONGEST_CLUSTER_ABOVE,
                ]
            )

        if side is SignalSide.SHORT:
            features.extend(
                [
                    LIQUIDITY_FEATURES.BELOW_LIQUIDITY_SCORE,
                    LIQUIDITY_FEATURES.MAGNET_DOWN,
                    LIQUIDITY_FEATURES.SWEEP_RISK_DOWN,
                    LIQUIDITY_FEATURES.NEAREST_BELOW_LEVEL,
                    LIQUIDITY_FEATURES.STRONGEST_CLUSTER_BELOW,
                ]
            )

        if target is not None:
            if isinstance(target, StopCluster):
                features.append(LIQUIDITY_FEATURES.STOP_CLUSTERS)
            elif isinstance(target, LiquidityLevel):
                features.append(LIQUIDITY_FEATURES.ACTIVE_LEVELS)

        return list(dict.fromkeys(features))

    def _tags(
        self,
        *,
        side: SignalSide,
        target: LiquidityLevel | StopCluster | None,
        snapshot: LiquidityMapSnapshot,
    ) -> list[str]:
        tags = [
            self.bias_config.tag_liquidity,
            self.bias_config.tag_bias,
            self.bias_config.tag_map_bias,
            self.bias_config.tag_directional_context,
            self.bias_config.tag_confluence_signal,
        ]

        if side is SignalSide.LONG:
            tags.append(self.bias_config.tag_upside_bias)

        if side is SignalSide.SHORT:
            tags.append(self.bias_config.tag_downside_bias)

        if target is not None:
            tags.append(self.bias_config.tag_target_available)
            tags.append(f"target_type:{self._target_type(target)}")

        bias = getattr(snapshot, "bias", None)
        if bias is not None:
            tags.append(f"bias:{serialize_for_metadata(bias)}")

        return list(dict.fromkeys(tags))

    def _primary_reason(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        side: SignalSide,
    ) -> str:
        if side is SignalSide.LONG:
            return (
                "Liquidity map favors upside: "
                f"upside_edge={upside_bias_edge(snapshot):.3f}, "
                f"downside_edge={downside_bias_edge(snapshot):.3f}, "
                f"pressure={signed_score(getattr(snapshot, 'liquidity_pressure_score', 0.0)):.3f}, "
                f"analytics_confidence={analytics_signal_confidence(snapshot):.3f}"
            )

        return (
            "Liquidity map favors downside: "
            f"downside_edge={downside_bias_edge(snapshot):.3f}, "
            f"upside_edge={upside_bias_edge(snapshot):.3f}, "
            f"pressure={signed_score(getattr(snapshot, 'liquidity_pressure_score', 0.0)):.3f}, "
            f"analytics_confidence={analytics_signal_confidence(snapshot):.3f}"
        )

    def _target_reason(
        self,
        target: LiquidityLevel | StopCluster | None,
    ) -> str:
        if target is None:
            return "Directional liquidity bias accepted without explicit target"

        if isinstance(target, StopCluster):
            return (
                f"Optional target is stop cluster at {reference_price(target):.6f} "
                f"(confidence={unit_score(getattr(target, 'confidence', 0.0)):.3f})"
            )

        if isinstance(target, LiquidityLevel):
            return (
                f"Optional target is liquidity level at {reference_price(target):.6f} "
                f"(confidence={unit_score(getattr(target, 'confidence', 0.0)):.3f})"
            )

        return f"Optional target selected: {target.__class__.__name__}"

    def _target_metadata(
        self,
        target: LiquidityLevel | StopCluster | None,
    ) -> dict[str, Any] | None:
        if target is None:
            return None

        return {
            "type": self._target_type(target),
            "price": reference_price(target),
            "confidence": unit_score(getattr(target, "confidence", 0.0)),
            "strength": serialize_for_metadata(getattr(target, "strength", None)),
            "side": serialize_for_metadata(getattr(target, "side", None)),
            "sweep_status": serialize_for_metadata(getattr(target, "sweep_status", None)),
            "raw": serialize_for_metadata(target),
        }


__all__ = [
    "LiquidityMapBiasStrategy",
    "LiquidityMapBiasStrategyConfig",
]