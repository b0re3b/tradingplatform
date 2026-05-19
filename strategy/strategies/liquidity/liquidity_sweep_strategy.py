# trading_system/strategy/strategies/liquidity/liquidity_sweep_strategy.py

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
from .base import (
    LIQUIDITY_FEATURES,
    LiquidityStrategyConfig,
    LiquidityTradingStrategy,
)
from .utils import (
    ScoreBreakdown,
    best_zone_for_side,
    collect_targets_above,
    collect_targets_below,
    confidence_from_components,
    distance_pct,
    distance_score,
    freshness_score,
    is_directional_side,
    is_terminal_item,
    magnet_score_down,
    magnet_score_up,
    reference_price,
    serialize_for_metadata,
    signed_score,
    sweep_edge_down,
    sweep_edge_up,
    sweep_risk_down,
    sweep_risk_up,
    unit_score,
    weighted_score,
    zone_score,
)


@dataclass(slots=True)
class LiquiditySweepStrategyConfig(LiquidityStrategyConfig):
    """
    Unified liquidity magnet / sweep continuation strategy config.

    Strategy idea:
    - read LiquidityMapSnapshot from StrategyContext;
    - detect directional liquidity magnet / sweep path;
    - LONG when upside liquidity magnet dominates;
    - SHORT when downside liquidity magnet dominates;
    - use only non-terminal liquidity targets;
    - return internal StrategySignal and leave risk-ready conversion to SignalProcessor.
    """

    edge_threshold: float = 0.45
    edge_delta_threshold: float = 0.12

    min_target_distance_pct: float = 0.0010
    max_target_distance_pct: float = 0.0600

    fallback_stop_pct: float = 0.0040
    long_stop_offset: float = 0.9985
    short_stop_offset: float = 1.0015

    require_directional_target: bool = True
    reject_terminal_targets: bool = True

    score_edge_weight: float = 0.36
    score_target_weight: float = 0.24
    score_magnet_weight: float = 0.18
    score_sweep_risk_weight: float = 0.14
    score_pressure_weight: float = 0.08

    confidence_edge_weight: float = 0.34
    confidence_target_weight: float = 0.24
    confidence_magnet_weight: float = 0.18
    confidence_sweep_risk_weight: float = 0.14
    confidence_zone_weight: float = 0.10

    high_priority_score: float = 0.82
    critical_priority_score: float = 0.92

    tag_sweep_continuation: str = "liquidity_sweep_continuation"
    tag_liquidity_magnet: str = "liquidity_magnet"
    tag_follow_through: str = "follow_through"
    tag_directional_target: str = "directional_target"
    tag_upside_sweep: str = "upside_sweep"
    tag_downside_sweep: str = "downside_sweep"

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.CONTINUATION

    required_liquidity_features: tuple[str, ...] = (
        LIQUIDITY_FEATURES.SNAPSHOT,
    )

    def validate(self) -> None:
        LiquidityStrategyConfig.validate(self)

        bounded_fields = {
            "edge_threshold": self.edge_threshold,
            "edge_delta_threshold": self.edge_delta_threshold,
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
            "score_edge_weight": self.score_edge_weight,
            "score_target_weight": self.score_target_weight,
            "score_magnet_weight": self.score_magnet_weight,
            "score_sweep_risk_weight": self.score_sweep_risk_weight,
            "score_pressure_weight": self.score_pressure_weight,
            "confidence_edge_weight": self.confidence_edge_weight,
            "confidence_target_weight": self.confidence_target_weight,
            "confidence_magnet_weight": self.confidence_magnet_weight,
            "confidence_sweep_risk_weight": self.confidence_sweep_risk_weight,
            "confidence_zone_weight": self.confidence_zone_weight,
        }

        for field_name, value in weights.items():
            if value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        if (
            self.score_edge_weight
            + self.score_target_weight
            + self.score_magnet_weight
            + self.score_sweep_risk_weight
            + self.score_pressure_weight
        ) <= 0:
            raise StrategyConfigError("score weights sum must be > 0")

        if (
            self.confidence_edge_weight
            + self.confidence_target_weight
            + self.confidence_magnet_weight
            + self.confidence_sweep_risk_weight
            + self.confidence_zone_weight
        ) <= 0:
            raise StrategyConfigError("confidence weights sum must be > 0")

        for attr in (
            "tag_sweep_continuation",
            "tag_liquidity_magnet",
            "tag_follow_through",
            "tag_directional_target",
            "tag_upside_sweep",
            "tag_downside_sweep",
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


class LiquiditySweepStrategy(LiquidityTradingStrategy):
    """
    Unified liquidity sweep / magnet continuation strategy.

    Input:
        StrategyContext with FeatureSource.LIQUIDITY domain data and LiquidityMapSnapshot.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, confluence, filtering, building and risk payloads.
    """

    component_namespace = "strategy.liquidity.sweep"
    category: StrategyCategory = StrategyCategory.LIQUIDITY
    default_setup_type: SetupType = SetupType.CONTINUATION

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        liquidity_config: LiquiditySweepStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_liquidity_config = liquidity_config or LiquiditySweepStrategyConfig()
        resolved_liquidity_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            liquidity_config=resolved_liquidity_config,
            service_name=service_name,
        )

        self.sweep_config: LiquiditySweepStrategyConfig = resolved_liquidity_config

    @property
    def strategy_name(self) -> str:
        return "liquidity_sweep_strategy"

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(self.sweep_config.required_liquidity_features)

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
        if target is None and self.sweep_config.require_directional_target:
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            snapshot=snapshot,
            current_price=current_price,
            side=side,
            target=target,
        )

        if breakdown.score < self.sweep_config.min_signal_score:
            return None

        if breakdown.confidence < self.sweep_config.min_signal_confidence:
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
            "liquidity_setup_family": "liquidity_sweep_continuation",
            "liquidity_strategy_version": "2.0.0",
            "score_breakdown": breakdown.to_dict(),
            "tags": self._tags(side=side, target=target, snapshot=snapshot),
            "side": side.value,
            "current_price": current_price,
            "target": self._target_metadata(target),
            "target_price": take_profit,
            "stop_loss": stop_loss,
            "up_edge": sweep_edge_up(snapshot),
            "down_edge": sweep_edge_down(snapshot),
            "edge_delta": sweep_edge_up(snapshot) - sweep_edge_down(snapshot),
            "bias": serialize_for_metadata(getattr(snapshot, "bias", None)),
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
            setup_type=self.sweep_config.default_setup_type,
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

        up_edge = sweep_edge_up(snapshot)
        down_edge = sweep_edge_down(snapshot)
        directional_edge = max(up_edge, down_edge)
        edge_delta = abs(up_edge - down_edge)

        if (
            directional_edge < self.sweep_config.edge_threshold
            and edge_delta < self.sweep_config.edge_delta_threshold
        ):
            results.append(
                FilterResult(
                    name="liquidity_sweep_directional_edge",
                    decision=FilterDecision.BLOCK,
                    reason=(
                        "No strong directional liquidity sweep edge: "
                        f"up_edge={up_edge:.4f}, down_edge={down_edge:.4f}, "
                        f"delta={edge_delta:.4f}"
                    ),
                )
            )
        else:
            results.append(
                FilterResult(
                    name="liquidity_sweep_directional_edge",
                    decision=FilterDecision.PASS,
                    reason=(
                        "Directional liquidity sweep edge present: "
                        f"up_edge={up_edge:.4f}, down_edge={down_edge:.4f}, "
                        f"delta={edge_delta:.4f}"
                    ),
                )
            )

        has_targets = self._snapshot_has_usable_targets(
            snapshot=snapshot,
            current_price=current_price,
        )
        results.append(
            FilterResult(
                name="liquidity_sweep_target_presence",
                decision=FilterDecision.PASS if has_targets else FilterDecision.BLOCK,
                reason=(
                    "Valid non-terminal directional liquidity target exists"
                    if has_targets
                    else "No valid non-terminal directional liquidity target exists"
                ),
            )
        )

        for item in results:
            item.validate()

        return results

    def _snapshot_has_usable_targets(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> bool:
        return bool(
            self._collect_valid_targets_above(snapshot, current_price)
            or self._collect_valid_targets_below(snapshot, current_price)
        )

    # ------------------------------------------------------------------
    # Direction
    # ------------------------------------------------------------------

    def _infer_side(self, snapshot: LiquidityMapSnapshot) -> SignalSide:
        up_edge = sweep_edge_up(snapshot)
        down_edge = sweep_edge_down(snapshot)
        delta = up_edge - down_edge

        if getattr(snapshot, "bias", None) == LiquidityBias.UP:
            if up_edge >= self.sweep_config.edge_threshold:
                return SignalSide.LONG

        if getattr(snapshot, "bias", None) == LiquidityBias.DOWN:
            if down_edge >= self.sweep_config.edge_threshold:
                return SignalSide.SHORT

        if (
            delta >= self.sweep_config.edge_delta_threshold
            and up_edge >= self.sweep_config.edge_threshold * 0.80
        ):
            return SignalSide.LONG

        if (
            delta <= -self.sweep_config.edge_delta_threshold
            and down_edge >= self.sweep_config.edge_threshold * 0.80
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
            and self._is_valid_follow_through_target(
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

    def _find_extended_target(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
        exclude: LiquidityLevel | StopCluster | None = None,
    ) -> LiquidityLevel | StopCluster | None:
        if side is SignalSide.LONG:
            candidates = self._collect_valid_targets_above(snapshot, current_price)
        elif side is SignalSide.SHORT:
            candidates = self._collect_valid_targets_below(snapshot, current_price)
        else:
            return None

        if exclude is not None:
            exclude_price = reference_price(exclude)
            candidates = [
                item
                for item in candidates
                if abs(reference_price(item) - exclude_price) > 1e-12
            ]

        return candidates[0] if candidates else None

    def _collect_valid_targets_above(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> list[LiquidityLevel | StopCluster]:
        return [
            item
            for item in collect_targets_above(snapshot, current_price)
            if self._is_valid_follow_through_target(
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
            if self._is_valid_follow_through_target(
                item=item,
                current_price=current_price,
                side=SignalSide.SHORT,
            )
        ]

    def _is_valid_follow_through_target(
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

        if target_distance < self.sweep_config.min_target_distance_pct:
            return False

        if target_distance > self.sweep_config.max_target_distance_pct:
            return False

        if self.sweep_config.reject_terminal_targets and is_terminal_item(item):
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
        edge = self._edge_for_side(snapshot=snapshot, side=side)
        target_score = self._target_quality_score(target=target, current_price=current_price)
        magnet = self._magnet_for_side(snapshot=snapshot, side=side)
        sweep_risk = self._sweep_risk_for_side(snapshot=snapshot, side=side)
        pressure = self._pressure_alignment(snapshot=snapshot, side=side)
        zone_alignment = self._zone_alignment_score(
            snapshot=snapshot,
            current_price=current_price,
            side=side,
        )

        score = weighted_score(
            {
                "edge": edge,
                "target": target_score,
                "magnet": magnet,
                "sweep_risk": sweep_risk,
                "pressure": pressure,
            },
            {
                "edge": self.sweep_config.score_edge_weight,
                "target": self.sweep_config.score_target_weight,
                "magnet": self.sweep_config.score_magnet_weight,
                "sweep_risk": self.sweep_config.score_sweep_risk_weight,
                "pressure": self.sweep_config.score_pressure_weight,
            },
            default=edge,
        )

        snapshot_time = getattr(snapshot, "timestamp", None)
        fresh_score = freshness_score(
            event_time=snapshot_time,
            now=context.timestamp,
            stale_after_seconds=self.sweep_config.max_snapshot_age_seconds,
        )

        confidence = confidence_from_components(
            primary=weighted_score(
                {
                    "edge": edge,
                    "target": target_score,
                    "magnet": magnet,
                    "sweep_risk": sweep_risk,
                    "zone": zone_alignment,
                },
                {
                    "edge": self.sweep_config.confidence_edge_weight,
                    "target": self.sweep_config.confidence_target_weight,
                    "magnet": self.sweep_config.confidence_magnet_weight,
                    "sweep_risk": self.sweep_config.confidence_sweep_risk_weight,
                    "zone": self.sweep_config.confidence_zone_weight,
                },
                default=edge,
            ),
            context=weighted_score(
                {
                    "pressure": pressure,
                    "zone": zone_alignment,
                    "target": target_score,
                },
                {
                    "pressure": 0.40,
                    "zone": 0.30,
                    "target": 0.30,
                },
                default=0.0,
            ),
            confirmation=target_score,
            freshness=fresh_score,
        )

        reasons = [
            f"edge:{edge:.3f}",
            f"target_score:{target_score:.3f}",
            f"magnet:{magnet:.3f}",
            f"sweep_risk:{sweep_risk:.3f}",
        ]

        confirmations: list[str] = []

        if edge >= self.sweep_config.edge_threshold:
            confirmations.append("directional_edge_confirmed")

        if target is not None:
            confirmations.append("directional_liquidity_target_confirmed")

        if magnet >= 0.50:
            confirmations.append("liquidity_magnet_confirmed")

        if sweep_risk >= 0.50:
            confirmations.append("sweep_risk_confirmed")

        if zone_alignment > 0:
            confirmations.append("zone_alignment_confirmed")

        return ScoreBreakdown(
            score=score,
            confidence=confidence,
            components={
                "edge": edge,
                "target_score": target_score,
                "magnet": magnet,
                "sweep_risk": sweep_risk,
                "pressure": pressure,
                "zone_alignment": zone_alignment,
                "freshness_score": fresh_score,
            },
            weights={
                "score_edge_weight": self.sweep_config.score_edge_weight,
                "score_target_weight": self.sweep_config.score_target_weight,
                "score_magnet_weight": self.sweep_config.score_magnet_weight,
                "score_sweep_risk_weight": self.sweep_config.score_sweep_risk_weight,
                "score_pressure_weight": self.sweep_config.score_pressure_weight,
            },
            reasons=reasons,
            confirmations=confirmations,
        ).normalize()

    def _edge_for_side(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        side: SignalSide,
    ) -> float:
        if side is SignalSide.LONG:
            return sweep_edge_up(snapshot)

        if side is SignalSide.SHORT:
            return sweep_edge_down(snapshot)

        return 0.0

    def _magnet_for_side(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        side: SignalSide,
    ) -> float:
        if side is SignalSide.LONG:
            return magnet_score_up(snapshot)

        if side is SignalSide.SHORT:
            return magnet_score_down(snapshot)

        return 0.0

    def _sweep_risk_for_side(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        side: SignalSide,
    ) -> float:
        if side is SignalSide.LONG:
            return sweep_risk_up(snapshot)

        if side is SignalSide.SHORT:
            return sweep_risk_down(snapshot)

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
        liquidity_side = LiquiditySide.BUY_SIDE if side is SignalSide.LONG else LiquiditySide.SELL_SIDE
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

        bonus = 0.0

        if isinstance(target, StopCluster):
            bonus += 0.36 * unit_score(getattr(target, "confidence", 0.0))
            bonus += 0.24 * unit_score(getattr(target, "estimated_stop_density", 0.0))
            bonus += 0.20 * self._target_distance_score(target, current_price)

            strength = getattr(target, "strength", None)
            strength_value = strength.value if hasattr(strength, "value") else str(strength or "")
            if strength_value == "medium":
                bonus += 0.04
            elif strength_value == "high":
                bonus += 0.07
            elif strength_value == "extreme":
                bonus += 0.10

        elif isinstance(target, LiquidityLevel):
            bonus += 0.42 * unit_score(getattr(target, "confidence", 0.0))
            bonus += 0.24 * self._target_distance_score(target, current_price)

            touches = max(int(getattr(target, "touches_count", 0) or 0), 0)
            reactions = max(int(getattr(target, "reaction_count", 0) or 0), 0)

            bonus += min(touches / 6.0, 1.0) * 0.08
            bonus += min(reactions / 4.0, 1.0) * 0.06

            if getattr(target, "sweep_status", None) == SweepStatus.PARTIALLY_SWEPT:
                bonus -= 0.03
            elif getattr(target, "sweep_status", None) == SweepStatus.SWEPT:
                bonus -= 0.10

        return unit_score(bonus)

    def _target_distance_score(
        self,
        target: LiquidityLevel | StopCluster | None,
        current_price: float,
    ) -> float:
        if target is None or current_price <= 0:
            return 0.0

        ref_price = reference_price(target)
        if ref_price <= 0:
            return 0.0

        target_distance = distance_pct(ref_price, current_price)

        if target_distance < self.sweep_config.min_target_distance_pct:
            return 0.0

        if target_distance <= 0.003:
            return 0.38
        if target_distance <= 0.010:
            return 0.90
        if target_distance <= 0.020:
            return 0.70
        if target_distance <= 0.040:
            return 0.40
        if target_distance <= self.sweep_config.max_target_distance_pct:
            return 0.18

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
                return anchor * self.sweep_config.long_stop_offset
            return current_price * (1.0 - self.sweep_config.fallback_stop_pct)

        if side is SignalSide.SHORT:
            if anchor > 0 and anchor > current_price:
                return anchor * self.sweep_config.short_stop_offset
            return current_price * (1.0 + self.sweep_config.fallback_stop_pct)

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
                    size_fraction=0.70,
                    rr=self._compute_rr(
                        current_price=current_price,
                        stop_price=stop_loss,
                        target_price=primary_price,
                        side=side,
                    ),
                    label="primary_liquidity_target",
                    metadata={
                        "source": "directional_liquidity_target",
                        "target_type": self._target_type(target),
                    },
                )
            )

        extended = self._find_extended_target(
            snapshot=snapshot,
            current_price=current_price,
            side=side,
            exclude=target,
        )
        extended_price = reference_price(extended) if extended is not None else 0.0

        if extended_price > 0:
            result.append(
                TargetPlan(
                    price=extended_price,
                    size_fraction=0.30,
                    rr=self._compute_rr(
                        current_price=current_price,
                        stop_price=stop_loss,
                        target_price=extended_price,
                        side=side,
                    ),
                    label="secondary_liquidity_target",
                    metadata={
                        "source": "extended_directional_liquidity",
                        "target_type": self._target_type(extended),
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

        if combined >= self.sweep_config.critical_priority_score:
            return SignalPriority.CRITICAL

        if combined >= self.sweep_config.high_priority_score:
            return SignalPriority.HIGH

        return self.sweep_config.default_priority

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
            self.sweep_config.tag_liquidity,
            self.sweep_config.tag_sweep,
            self.sweep_config.tag_sweep_continuation,
            self.sweep_config.tag_liquidity_magnet,
            self.sweep_config.tag_follow_through,
        ]

        if side is SignalSide.LONG:
            tags.append(self.sweep_config.tag_upside_sweep)

        if side is SignalSide.SHORT:
            tags.append(self.sweep_config.tag_downside_sweep)

        if target is not None:
            tags.append(self.sweep_config.tag_directional_target)
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
                "Upside liquidity magnet dominates: "
                f"magnet_up={magnet_score_up(snapshot):.3f}, "
                f"sweep_risk_up={sweep_risk_up(snapshot):.3f}, "
                f"above_liquidity={unit_score(getattr(snapshot, 'above_liquidity_score', 0.0)):.3f}, "
                f"pressure={signed_score(getattr(snapshot, 'liquidity_pressure_score', 0.0)):.3f}"
            )

        return (
            "Downside liquidity magnet dominates: "
            f"magnet_down={magnet_score_down(snapshot):.3f}, "
            f"sweep_risk_down={sweep_risk_down(snapshot):.3f}, "
            f"below_liquidity={unit_score(getattr(snapshot, 'below_liquidity_score', 0.0)):.3f}, "
            f"pressure={signed_score(getattr(snapshot, 'liquidity_pressure_score', 0.0)):.3f}"
        )

    def _target_reason(
        self,
        target: LiquidityLevel | StopCluster | None,
    ) -> str:
        if target is None:
            return "No explicit directional liquidity target found"

        if isinstance(target, StopCluster):
            return (
                f"Primary target is stop cluster at {reference_price(target):.6f} "
                f"(confidence={unit_score(getattr(target, 'confidence', 0.0)):.3f})"
            )

        if isinstance(target, LiquidityLevel):
            return (
                f"Primary target is liquidity level at {reference_price(target):.6f} "
                f"(confidence={unit_score(getattr(target, 'confidence', 0.0)):.3f})"
            )

        return f"Primary target selected: {target.__class__.__name__}"

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
    "LiquiditySweepStrategy",
    "LiquiditySweepStrategyConfig",
]