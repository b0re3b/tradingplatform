# trading_system/strategy/strategies/liquidity/stop_hunt_reversal_strategy.py

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from analytics.liquidity.enums import LiquidityBias, LiquiditySide
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
    collect_targets_above,
    collect_targets_below,
    confidence_from_components,
    distance_pct,
    distance_score,
    evidence_type,
    is_directional_side,
    item_strength,
    magnet_score_down,
    magnet_score_up,
    reclaim_score_from_reference,
    reference_price,
    safe_decimal,
    serialize_for_metadata,
    signed_score,
    sweep_risk_down,
    sweep_risk_up,
    swept_clusters,
    swept_evidence_rank,
    swept_levels,
    unit_score,
    weighted_score,
)


@dataclass(slots=True)
class StopHuntReversalStrategyConfig(LiquidityStrategyConfig):
    """
    Unified stop-hunt reversal strategy config.

    Strategy idea:
    - LONG after sell-side liquidity was swept/partially swept below current price
      and price reclaimed above sweep reference;
    - SHORT after buy-side liquidity was swept/partially swept above current price
      and price rejected below sweep reference;
    - unswept liquidity is not enough;
    - return internal StrategySignal only; SignalProcessor owns final emission.
    """

    min_edge: float = 0.18
    min_reclaim_score: float = 0.04

    max_evidence_distance_pct: float = 0.0350
    max_target_distance_pct: float = 0.0700

    fallback_stop_pct: float = 0.0045
    long_stop_offset: float = 0.9985
    short_stop_offset: float = 1.0015

    require_swept_evidence: bool = True
    require_reclaim_or_rejection: bool = True
    allow_partially_swept_evidence: bool = True

    score_edge_weight: float = 0.30
    score_reclaim_weight: float = 0.22
    score_evidence_weight: float = 0.20
    score_pressure_weight: float = 0.10
    score_sweep_risk_weight: float = 0.10
    score_target_weight: float = 0.08

    confidence_edge_weight: float = 0.28
    confidence_evidence_weight: float = 0.24
    confidence_reclaim_weight: float = 0.20
    confidence_context_weight: float = 0.18
    confidence_target_weight: float = 0.10

    high_priority_score: float = 0.82
    critical_priority_score: float = 0.92

    tag_stop_hunt_reversal: str = "stop_hunt_reversal"
    tag_sell_side_hunt: str = "sell_side_stop_hunt"
    tag_buy_side_hunt: str = "buy_side_stop_hunt"
    tag_swept_evidence: str = "swept_liquidity_evidence"
    tag_reclaim: str = "reclaim"
    tag_rejection: str = "rejection"
    tag_reversal: str = "reversal"

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.MEAN_REVERSION

    required_liquidity_features: tuple[str, ...] = (
        LIQUIDITY_FEATURES.SNAPSHOT,
    )

    def validate(self) -> None:
        LiquidityStrategyConfig.validate(self)

        bounded_fields = {
            "min_edge": self.min_edge,
            "min_reclaim_score": self.min_reclaim_score,
            "max_evidence_distance_pct": self.max_evidence_distance_pct,
            "max_target_distance_pct": self.max_target_distance_pct,
            "fallback_stop_pct": self.fallback_stop_pct,
            "high_priority_score": self.high_priority_score,
            "critical_priority_score": self.critical_priority_score,
        }

        for field_name, value in bounded_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        if self.long_stop_offset <= 0:
            raise StrategyConfigError("long_stop_offset must be > 0")

        if self.short_stop_offset <= 0:
            raise StrategyConfigError("short_stop_offset must be > 0")

        weights = {
            "score_edge_weight": self.score_edge_weight,
            "score_reclaim_weight": self.score_reclaim_weight,
            "score_evidence_weight": self.score_evidence_weight,
            "score_pressure_weight": self.score_pressure_weight,
            "score_sweep_risk_weight": self.score_sweep_risk_weight,
            "score_target_weight": self.score_target_weight,
            "confidence_edge_weight": self.confidence_edge_weight,
            "confidence_evidence_weight": self.confidence_evidence_weight,
            "confidence_reclaim_weight": self.confidence_reclaim_weight,
            "confidence_context_weight": self.confidence_context_weight,
            "confidence_target_weight": self.confidence_target_weight,
        }

        for field_name, value in weights.items():
            if value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        if (
            self.score_edge_weight
            + self.score_reclaim_weight
            + self.score_evidence_weight
            + self.score_pressure_weight
            + self.score_sweep_risk_weight
            + self.score_target_weight
        ) <= 0:
            raise StrategyConfigError("score weights sum must be > 0")

        if (
            self.confidence_edge_weight
            + self.confidence_evidence_weight
            + self.confidence_reclaim_weight
            + self.confidence_context_weight
            + self.confidence_target_weight
        ) <= 0:
            raise StrategyConfigError("confidence weights sum must be > 0")

        for attr in (
            "tag_stop_hunt_reversal",
            "tag_sell_side_hunt",
            "tag_buy_side_hunt",
            "tag_swept_evidence",
            "tag_reclaim",
            "tag_rejection",
            "tag_reversal",
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


class StopHuntReversalStrategy(LiquidityTradingStrategy):
    """
    Unified stop-hunt reversal strategy.

    Input:
        StrategyContext with FeatureSource.LIQUIDITY domain data and LiquidityMapSnapshot.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    """

    component_namespace = "strategy.liquidity.stop_hunt_reversal"
    category: StrategyCategory = StrategyCategory.LIQUIDITY
    default_setup_type: SetupType = SetupType.MEAN_REVERSION

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        liquidity_config: StopHuntReversalStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_liquidity_config = liquidity_config or StopHuntReversalStrategyConfig()
        resolved_liquidity_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            liquidity_config=resolved_liquidity_config,
            service_name=service_name,
        )

        self.stop_hunt_config: StopHuntReversalStrategyConfig = (
            resolved_liquidity_config
        )

    @property
    def strategy_name(self) -> str:
        return "stop_hunt_reversal_strategy"

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(
            self.stop_hunt_config.required_liquidity_features
        )

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

        candidate = self._find_reversal_candidate(
            snapshot=snapshot,
            current_price=current_price,
        )
        if candidate is None:
            return None

        side = candidate["side"]
        if not is_directional_side(side):
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            snapshot=snapshot,
            current_price=current_price,
            candidate=candidate,
        )

        if breakdown.score < self.stop_hunt_config.min_signal_score:
            return None

        if breakdown.confidence < self.stop_hunt_config.min_signal_confidence:
            return None

        evidence = candidate.get("evidence")
        target = candidate.get("target")
        reference = float(candidate["reference_price"])

        stop_loss = self._resolve_stop_price(
            side=side,
            current_price=current_price,
            reference_price_value=reference,
        )
        take_profit = reference_price(target) if target is not None else None

        target_plans = self._target_plans(
            current_price=current_price,
            side=side,
            target=target,
            stop_loss=stop_loss,
        )

        reasons = list(
            dict.fromkeys(
                [
                    self._primary_reason(candidate),
                    self._target_reason(target),
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))
        source_features = self._source_features(candidate)

        metadata = {
            "liquidity_setup_family": "stop_hunt_reversal",
            "liquidity_strategy_version": "2.0.0",
            "score_breakdown": breakdown.to_dict(),
            "tags": self._tags(candidate=candidate, snapshot=snapshot),
            "side": side.value,
            "direction": candidate.get("direction"),
            "current_price": current_price,
            "reference_price": reference,
            "target": self._target_metadata(target),
            "target_price": take_profit,
            "stop_loss": stop_loss,
            "evidence": self._evidence_metadata(evidence),
            "evidence_type": candidate.get("evidence_type"),
            "edge": candidate.get("edge"),
            "reclaim_score": candidate.get("reclaim_score"),
            "level_score": candidate.get("level_score"),
            "cluster_score": candidate.get("cluster_score"),
            "pressure_bonus": candidate.get("pressure_bonus"),
            "anti_bias_bonus": candidate.get("anti_bias_bonus"),
            "sweep_risk_bonus": candidate.get("sweep_risk_bonus"),
            "magnet_bonus": candidate.get("magnet_bonus"),
            "bias": serialize_for_metadata(getattr(snapshot, "bias", None)),
            "liquidity_pressure_score": signed_score(
                getattr(snapshot, "liquidity_pressure_score", 0.0)
            ),
            "filters": [item.to_dict() for item in filters],
            "target_plans": [plan.to_dict() for plan in target_plans],
        }

        return self.build_liquidity_signal(
            context=context,
            side=side,
            confidence=breakdown.confidence,
            score=breakdown.score,
            setup_type=self.stop_hunt_config.default_setup_type,
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

        swept_evidence = self._collect_swept_evidence(snapshot)

        if swept_evidence:
            results.append(
                FilterResult(
                    name="stop_hunt_swept_evidence",
                    decision=FilterDecision.PASS,
                    reason=f"Swept liquidity evidence found: {len(swept_evidence)}",
                )
            )
        else:
            results.append(
                FilterResult(
                    name="stop_hunt_swept_evidence",
                    decision=FilterDecision.BLOCK,
                    reason="No swept or partially swept liquidity evidence",
                )
            )

        for item in results:
            item.validate()

        return results

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    def _find_reversal_candidate(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> dict[str, Any] | None:
        sell_side = self._find_sell_side_stop_hunt(
            snapshot=snapshot,
            current_price=current_price,
        )
        buy_side = self._find_buy_side_stop_hunt(
            snapshot=snapshot,
            current_price=current_price,
        )

        if sell_side is None and buy_side is None:
            return None

        if sell_side is None:
            return buy_side

        if buy_side is None:
            return sell_side

        return (
            sell_side
            if float(sell_side["edge"]) >= float(buy_side["edge"])
            else buy_side
        )

    def _find_sell_side_stop_hunt(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> dict[str, Any] | None:
        """
        LONG reversal after sell-side liquidity sweep.
        """
        swept_sell_levels = [
            level
            for level in swept_levels(snapshot, side=LiquiditySide.SELL_SIDE)
            if reference_price(level) < current_price
            and self._evidence_distance_ok(level, current_price)
            and self._swept_evidence_allowed(level)
        ]
        swept_sell_clusters = [
            cluster
            for cluster in swept_clusters(snapshot, side=LiquiditySide.SELL_SIDE)
            if reference_price(cluster) < current_price
            and self._evidence_distance_ok(cluster, current_price)
        ]

        if not swept_sell_levels and not swept_sell_clusters:
            return None

        level = self._pick_best_level(swept_sell_levels, current_price)
        cluster = self._pick_best_cluster(swept_sell_clusters, current_price)
        evidence = self._pick_best_evidence(
            level=level,
            cluster=cluster,
            current_price=current_price,
        )
        if evidence is None:
            return None

        ref_price = reference_price(evidence)
        reclaim_score = reclaim_score_from_reference(
            current_price=current_price,
            reference_price_value=ref_price,
            side=SignalSide.LONG,
        )
        if (
            self.stop_hunt_config.require_reclaim_or_rejection
            and reclaim_score < self.stop_hunt_config.min_reclaim_score
        ):
            return None

        level_score = self._level_reversal_score(level, current_price)
        cluster_score = self._cluster_reversal_score(cluster, current_price)

        pressure_bonus = max(
            -signed_score(getattr(snapshot, "liquidity_pressure_score", 0.0)),
            0.0,
        ) * 0.18
        anti_bias_bonus = 0.24 if getattr(snapshot, "bias", None) == LiquidityBias.DOWN else 0.0
        sweep_risk_bonus = sweep_risk_down(snapshot) * 0.16
        magnet_bonus = magnet_score_up(snapshot) * 0.08

        edge = (
            max(level_score, cluster_score)
            + reclaim_score
            + pressure_bonus
            + anti_bias_bonus
            + sweep_risk_bonus
            + magnet_bonus
        )

        if edge <= self.stop_hunt_config.min_edge:
            return None

        target = self._nearest_opposite_target(
            snapshot=snapshot,
            current_price=current_price,
            side=SignalSide.LONG,
        )

        return {
            "direction": "sell_side_stop_hunt_reversal",
            "side": SignalSide.LONG,
            "hunted_level": level,
            "swept_cluster": cluster,
            "support_cluster": cluster,
            "resistance_cluster": None,
            "evidence": evidence,
            "evidence_type": evidence_type(evidence),
            "reference_price": ref_price,
            "target": target,
            "edge": unit_score(edge),
            "has_swept_level": level is not None,
            "has_swept_cluster": cluster is not None,
            "reclaim_score": unit_score(reclaim_score),
            "level_score": unit_score(level_score),
            "cluster_score": unit_score(cluster_score),
            "pressure_bonus": unit_score(pressure_bonus),
            "anti_bias_bonus": unit_score(anti_bias_bonus),
            "sweep_risk_bonus": unit_score(sweep_risk_bonus),
            "magnet_bonus": unit_score(magnet_bonus),
        }

    def _find_buy_side_stop_hunt(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> dict[str, Any] | None:
        """
        SHORT reversal after buy-side liquidity sweep.
        """
        swept_buy_levels = [
            level
            for level in swept_levels(snapshot, side=LiquiditySide.BUY_SIDE)
            if reference_price(level) > current_price
            and self._evidence_distance_ok(level, current_price)
            and self._swept_evidence_allowed(level)
        ]
        swept_buy_clusters = [
            cluster
            for cluster in swept_clusters(snapshot, side=LiquiditySide.BUY_SIDE)
            if reference_price(cluster) > current_price
            and self._evidence_distance_ok(cluster, current_price)
        ]

        if not swept_buy_levels and not swept_buy_clusters:
            return None

        level = self._pick_best_level(swept_buy_levels, current_price)
        cluster = self._pick_best_cluster(swept_buy_clusters, current_price)
        evidence = self._pick_best_evidence(
            level=level,
            cluster=cluster,
            current_price=current_price,
        )
        if evidence is None:
            return None

        ref_price = reference_price(evidence)
        reclaim_score = reclaim_score_from_reference(
            current_price=current_price,
            reference_price_value=ref_price,
            side=SignalSide.SHORT,
        )
        if (
            self.stop_hunt_config.require_reclaim_or_rejection
            and reclaim_score < self.stop_hunt_config.min_reclaim_score
        ):
            return None

        level_score = self._level_reversal_score(level, current_price)
        cluster_score = self._cluster_reversal_score(cluster, current_price)

        pressure_bonus = max(
            signed_score(getattr(snapshot, "liquidity_pressure_score", 0.0)),
            0.0,
        ) * 0.18
        anti_bias_bonus = 0.24 if getattr(snapshot, "bias", None) == LiquidityBias.UP else 0.0
        sweep_risk_bonus = sweep_risk_up(snapshot) * 0.16
        magnet_bonus = magnet_score_down(snapshot) * 0.08

        edge = (
            max(level_score, cluster_score)
            + reclaim_score
            + pressure_bonus
            + anti_bias_bonus
            + sweep_risk_bonus
            + magnet_bonus
        )

        if edge <= self.stop_hunt_config.min_edge:
            return None

        target = self._nearest_opposite_target(
            snapshot=snapshot,
            current_price=current_price,
            side=SignalSide.SHORT,
        )

        return {
            "direction": "buy_side_stop_hunt_reversal",
            "side": SignalSide.SHORT,
            "hunted_level": level,
            "swept_cluster": cluster,
            "support_cluster": None,
            "resistance_cluster": cluster,
            "evidence": evidence,
            "evidence_type": evidence_type(evidence),
            "reference_price": ref_price,
            "target": target,
            "edge": unit_score(edge),
            "has_swept_level": level is not None,
            "has_swept_cluster": cluster is not None,
            "reclaim_score": unit_score(reclaim_score),
            "level_score": unit_score(level_score),
            "cluster_score": unit_score(cluster_score),
            "pressure_bonus": unit_score(pressure_bonus),
            "anti_bias_bonus": unit_score(anti_bias_bonus),
            "sweep_risk_bonus": unit_score(sweep_risk_bonus),
            "magnet_bonus": unit_score(magnet_bonus),
        }

    # ------------------------------------------------------------------
    # Evidence helpers
    # ------------------------------------------------------------------

    def _collect_swept_evidence(
        self,
        snapshot: LiquidityMapSnapshot,
    ) -> list[LiquidityLevel | StopCluster]:
        return [
            *[
                item
                for item in swept_levels(snapshot, side=LiquiditySide.BUY_SIDE)
                if self._swept_evidence_allowed(item)
            ],
            *[
                item
                for item in swept_levels(snapshot, side=LiquiditySide.SELL_SIDE)
                if self._swept_evidence_allowed(item)
            ],
            *swept_clusters(snapshot, side=LiquiditySide.BUY_SIDE),
            *swept_clusters(snapshot, side=LiquiditySide.SELL_SIDE),
        ]

    def _swept_evidence_allowed(self, item: Any) -> bool:
        if self.stop_hunt_config.allow_partially_swept_evidence:
            return True

        method = getattr(item, "is_swept", None)
        if callable(method):
            try:
                return bool(method())
            except Exception:
                return False

        status = str(getattr(item, "sweep_status", "")).lower()
        return status.endswith("swept") and "partially" not in status

    def _evidence_distance_ok(
        self,
        evidence: LiquidityLevel | StopCluster,
        current_price: float,
    ) -> bool:
        if current_price <= 0:
            return False

        ref_price = reference_price(evidence)
        if ref_price <= 0:
            return False

        return (
            distance_pct(ref_price, current_price)
            <= self.stop_hunt_config.max_evidence_distance_pct
        )

    def _pick_best_level(
        self,
        levels: list[LiquidityLevel],
        current_price: float,
    ) -> LiquidityLevel | None:
        if not levels:
            return None

        def rank(level: LiquidityLevel) -> tuple[float, float, float]:
            return (
                self._level_reversal_score(level, current_price),
                item_strength(level),
                -distance_pct(reference_price(level), current_price),
            )

        return max(levels, key=rank)

    def _pick_best_cluster(
        self,
        clusters: list[StopCluster],
        current_price: float,
    ) -> StopCluster | None:
        if not clusters:
            return None

        def rank(cluster: StopCluster) -> tuple[float, float, float]:
            return (
                self._cluster_reversal_score(cluster, current_price),
                item_strength(cluster),
                -distance_pct(reference_price(cluster), current_price),
            )

        return max(clusters, key=rank)

    def _pick_best_evidence(
        self,
        *,
        level: LiquidityLevel | None,
        cluster: StopCluster | None,
        current_price: float,
    ) -> LiquidityLevel | StopCluster | None:
        if level is None and cluster is None:
            return None

        if level is None:
            return cluster

        if cluster is None:
            return level

        level_rank = (
            self._level_reversal_score(level, current_price),
            swept_evidence_rank(level),
        )
        cluster_rank = (
            self._cluster_reversal_score(cluster, current_price),
            swept_evidence_rank(cluster),
        )

        return level if level_rank >= cluster_rank else cluster

    def _level_reversal_score(
        self,
        level: LiquidityLevel | None,
        current_price: float,
    ) -> float:
        if level is None:
            return 0.0

        ref_price = reference_price(level)
        if ref_price <= 0 or current_price <= 0:
            return 0.0

        confidence = unit_score(getattr(level, "confidence", 0.0))
        strength = item_strength(level)
        proximity = unit_score(
            1.0
            - min(
                distance_pct(ref_price, current_price)
                / self.stop_hunt_config.max_evidence_distance_pct,
                1.0,
            )
        )

        touches = max(int(getattr(level, "touches_count", 0) or 0), 0)
        reactions = max(int(getattr(level, "reaction_count", 0) or 0), 0)

        return unit_score(
            0.34 * confidence
            + 0.24 * strength
            + 0.22 * proximity
            + 0.10 * min(touches / 6.0, 1.0)
            + 0.10 * min(reactions / 4.0, 1.0)
        )

    def _cluster_reversal_score(
        self,
        cluster: StopCluster | None,
        current_price: float,
    ) -> float:
        if cluster is None:
            return 0.0

        ref_price = reference_price(cluster)
        if ref_price <= 0 or current_price <= 0:
            return 0.0

        confidence = unit_score(getattr(cluster, "confidence", 0.0))
        strength = item_strength(cluster)
        density = unit_score(getattr(cluster, "estimated_stop_density", 0.0))
        proximity = unit_score(
            1.0
            - min(
                distance_pct(ref_price, current_price)
                / self.stop_hunt_config.max_evidence_distance_pct,
                1.0,
            )
        )

        total_notional = safe_decimal(getattr(cluster, "total_notional", None))
        notional_score = unit_score(
            float(min(total_notional / safe_decimal("1000000", Decimal("1000000")), Decimal("1")))
        )

        return unit_score(
            0.30 * confidence
            + 0.24 * strength
            + 0.20 * density
            + 0.16 * proximity
            + 0.10 * notional_score
        )

    # ------------------------------------------------------------------
    # Target / trade levels
    # ------------------------------------------------------------------

    def _nearest_opposite_target(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
    ) -> LiquidityLevel | StopCluster | None:
        if side is SignalSide.LONG:
            candidates = collect_targets_above(snapshot, current_price)
        elif side is SignalSide.SHORT:
            candidates = collect_targets_below(snapshot, current_price)
        else:
            return None

        valid = [
            item
            for item in candidates
            if reference_price(item) > 0
            and distance_pct(reference_price(item), current_price)
            <= self.stop_hunt_config.max_target_distance_pct
        ]

        if not valid:
            return None

        return valid[0]

    def _resolve_stop_price(
        self,
        *,
        side: SignalSide,
        current_price: float,
        reference_price_value: float,
    ) -> float | None:
        if current_price <= 0 or reference_price_value <= 0:
            return None

        if side is SignalSide.LONG:
            if reference_price_value < current_price:
                return reference_price_value * self.stop_hunt_config.long_stop_offset
            return current_price * (1.0 - self.stop_hunt_config.fallback_stop_pct)

        if side is SignalSide.SHORT:
            if reference_price_value > current_price:
                return reference_price_value * self.stop_hunt_config.short_stop_offset
            return current_price * (1.0 + self.stop_hunt_config.fallback_stop_pct)

        return None

    def _target_plans(
        self,
        *,
        current_price: float,
        side: SignalSide,
        target: LiquidityLevel | StopCluster | None,
        stop_loss: float | None,
    ) -> list[TargetPlan]:
        result: list[TargetPlan] = []

        target_price = reference_price(target) if target is not None else 0.0
        if target_price > 0:
            result.append(
                TargetPlan(
                    price=target_price,
                    size_fraction=1.0,
                    rr=self._compute_rr(
                        current_price=current_price,
                        stop_price=stop_loss,
                        target_price=target_price,
                        side=side,
                    ),
                    label="stop_hunt_reversal_target",
                    metadata={
                        "source": "opposite_liquidity_target",
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

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        candidate: dict[str, Any],
    ) -> ScoreBreakdown:
        edge = unit_score(candidate.get("edge", 0.0))
        reclaim = unit_score(candidate.get("reclaim_score", 0.0))
        evidence_score = self._evidence_score(candidate)
        pressure = unit_score(candidate.get("pressure_bonus", 0.0) / 0.18)
        sweep_risk = unit_score(candidate.get("sweep_risk_bonus", 0.0) / 0.16)
        target_score = self._target_quality_score(
            target=candidate.get("target"),
            current_price=current_price,
        )
        context_score = unit_score(
            candidate.get("anti_bias_bonus", 0.0)
            + candidate.get("magnet_bonus", 0.0)
            + candidate.get("pressure_bonus", 0.0)
            + candidate.get("sweep_risk_bonus", 0.0)
        )

        score = weighted_score(
            {
                "edge": edge,
                "reclaim": reclaim,
                "evidence": evidence_score,
                "pressure": pressure,
                "sweep_risk": sweep_risk,
                "target": target_score,
            },
            {
                "edge": self.stop_hunt_config.score_edge_weight,
                "reclaim": self.stop_hunt_config.score_reclaim_weight,
                "evidence": self.stop_hunt_config.score_evidence_weight,
                "pressure": self.stop_hunt_config.score_pressure_weight,
                "sweep_risk": self.stop_hunt_config.score_sweep_risk_weight,
                "target": self.stop_hunt_config.score_target_weight,
            },
            default=edge,
        )

        confidence_primary = weighted_score(
            {
                "edge": edge,
                "evidence": evidence_score,
                "reclaim": reclaim,
                "context": context_score,
                "target": target_score,
            },
            {
                "edge": self.stop_hunt_config.confidence_edge_weight,
                "evidence": self.stop_hunt_config.confidence_evidence_weight,
                "reclaim": self.stop_hunt_config.confidence_reclaim_weight,
                "context": self.stop_hunt_config.confidence_context_weight,
                "target": self.stop_hunt_config.confidence_target_weight,
            },
            default=edge,
        )

        confidence = confidence_from_components(
            primary=confidence_primary,
            context=context_score,
            confirmation=reclaim,
            freshness=1.0,
        )

        reasons = [
            f"edge:{edge:.3f}",
            f"reclaim_score:{reclaim:.3f}",
            f"evidence_score:{evidence_score:.3f}",
            f"target_score:{target_score:.3f}",
        ]

        confirmations: list[str] = []

        if reclaim >= self.stop_hunt_config.min_reclaim_score:
            confirmations.append("reclaim_or_rejection_confirmed")

        if candidate.get("has_swept_level"):
            confirmations.append("swept_level_confirmed")

        if candidate.get("has_swept_cluster"):
            confirmations.append("swept_cluster_confirmed")

        if candidate.get("target") is not None:
            confirmations.append("opposite_liquidity_target_confirmed")

        if context.timestamp:
            confirmations.append("context_timestamp_available")

        return ScoreBreakdown(
            score=score,
            confidence=confidence,
            components={
                "edge": edge,
                "reclaim_score": reclaim,
                "evidence_score": evidence_score,
                "pressure": pressure,
                "sweep_risk": sweep_risk,
                "target_score": target_score,
                "context_score": context_score,
            },
            weights={
                "score_edge_weight": self.stop_hunt_config.score_edge_weight,
                "score_reclaim_weight": self.stop_hunt_config.score_reclaim_weight,
                "score_evidence_weight": self.stop_hunt_config.score_evidence_weight,
                "score_pressure_weight": self.stop_hunt_config.score_pressure_weight,
                "score_sweep_risk_weight": self.stop_hunt_config.score_sweep_risk_weight,
                "score_target_weight": self.stop_hunt_config.score_target_weight,
            },
            reasons=reasons,
            confirmations=confirmations,
        ).normalize()

    @staticmethod
    def _evidence_score(candidate: dict[str, Any]) -> float:
        level_score = unit_score(candidate.get("level_score", 0.0))
        cluster_score = unit_score(candidate.get("cluster_score", 0.0))
        return max(level_score, cluster_score)

    def _target_quality_score(
        self,
        *,
        target: LiquidityLevel | StopCluster | None,
        current_price: float,
    ) -> float:
        if target is None:
            return 0.0

        ref_price = reference_price(target)
        if ref_price <= 0 or current_price <= 0:
            return 0.0

        target_distance = distance_pct(ref_price, current_price)
        distance_component = unit_score(
            1.0
            - min(
                target_distance / self.stop_hunt_config.max_target_distance_pct,
                1.0,
            )
        )

        if isinstance(target, StopCluster):
            return unit_score(
                0.42 * unit_score(getattr(target, "confidence", 0.0))
                + 0.32 * unit_score(getattr(target, "estimated_stop_density", 0.0))
                + 0.26 * distance_component
            )

        if isinstance(target, LiquidityLevel):
            touches = max(int(getattr(target, "touches_count", 0) or 0), 0)
            reactions = max(int(getattr(target, "reaction_count", 0) or 0), 0)

            return unit_score(
                0.44 * unit_score(getattr(target, "confidence", 0.0))
                + 0.24 * distance_component
                + 0.16 * min(touches / 6.0, 1.0)
                + 0.16 * min(reactions / 4.0, 1.0)
            )

        return 0.0

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

        if combined >= self.stop_hunt_config.critical_priority_score:
            return SignalPriority.CRITICAL

        if combined >= self.stop_hunt_config.high_priority_score:
            return SignalPriority.HIGH

        return self.stop_hunt_config.default_priority

    def _source_features(self, candidate: dict[str, Any]) -> list[str]:
        features = [
            LIQUIDITY_FEATURES.SNAPSHOT,
            LIQUIDITY_FEATURES.MAP_SNAPSHOT,
            LIQUIDITY_FEATURES.PRESSURE_SCORE,
            LIQUIDITY_FEATURES.BIAS,
            LIQUIDITY_FEATURES.EQUAL_LEVELS,
            LIQUIDITY_FEATURES.ACTIVE_LEVELS,
            LIQUIDITY_FEATURES.STOP_CLUSTERS,
        ]

        if candidate.get("side") is SignalSide.LONG:
            features.extend(
                [
                    LIQUIDITY_FEATURES.BELOW_LIQUIDITY_SCORE,
                    LIQUIDITY_FEATURES.SWEEP_RISK_DOWN,
                    LIQUIDITY_FEATURES.MAGNET_UP,
                    LIQUIDITY_FEATURES.NEAREST_ABOVE_LEVEL,
                ]
            )

        if candidate.get("side") is SignalSide.SHORT:
            features.extend(
                [
                    LIQUIDITY_FEATURES.ABOVE_LIQUIDITY_SCORE,
                    LIQUIDITY_FEATURES.SWEEP_RISK_UP,
                    LIQUIDITY_FEATURES.MAGNET_DOWN,
                    LIQUIDITY_FEATURES.NEAREST_BELOW_LEVEL,
                ]
            )

        return list(dict.fromkeys(features))

    def _tags(
        self,
        *,
        candidate: dict[str, Any],
        snapshot: LiquidityMapSnapshot,
    ) -> list[str]:
        side = candidate.get("side")

        tags = [
            self.stop_hunt_config.tag_liquidity,
            self.stop_hunt_config.tag_stop_hunt,
            self.stop_hunt_config.tag_stop_hunt_reversal,
            self.stop_hunt_config.tag_reversal,
            self.stop_hunt_config.tag_swept_evidence,
        ]

        if side is SignalSide.LONG:
            tags.extend(
                [
                    self.stop_hunt_config.tag_sell_side_hunt,
                    self.stop_hunt_config.tag_reclaim,
                ]
            )

        if side is SignalSide.SHORT:
            tags.extend(
                [
                    self.stop_hunt_config.tag_buy_side_hunt,
                    self.stop_hunt_config.tag_rejection,
                ]
            )

        evidence = candidate.get("evidence")
        if evidence is not None:
            tags.append(f"evidence_type:{evidence_type(evidence)}")

        bias = getattr(snapshot, "bias", None)
        if bias is not None:
            tags.append(f"bias:{serialize_for_metadata(bias)}")

        return list(dict.fromkeys(tags))

    def _primary_reason(self, candidate: dict[str, Any]) -> str:
        side = candidate.get("side")
        reference = float(candidate.get("reference_price", 0.0))
        reclaim = float(candidate.get("reclaim_score", 0.0))

        if side is SignalSide.LONG:
            return (
                "Sell-side liquidity was swept and price reclaimed above sweep reference: "
                f"reference={reference:.6f}, reclaim_score={reclaim:.3f}"
            )

        return (
            "Buy-side liquidity was swept and price rejected below sweep reference: "
            f"reference={reference:.6f}, rejection_score={reclaim:.3f}"
        )

    def _target_reason(
        self,
        target: LiquidityLevel | StopCluster | None,
    ) -> str:
        if target is None:
            return "No explicit opposite liquidity target found"

        if isinstance(target, StopCluster):
            return (
                f"Opposite target is stop cluster at {reference_price(target):.6f} "
                f"(confidence={unit_score(getattr(target, 'confidence', 0.0)):.3f})"
            )

        if isinstance(target, LiquidityLevel):
            return (
                f"Opposite target is liquidity level at {reference_price(target):.6f} "
                f"(confidence={unit_score(getattr(target, 'confidence', 0.0)):.3f})"
            )

        return f"Opposite target selected: {target.__class__.__name__}"

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

    def _evidence_metadata(
        self,
        evidence: LiquidityLevel | StopCluster | None,
    ) -> dict[str, Any] | None:
        if evidence is None:
            return None

        return {
            "type": evidence_type(evidence),
            "price": reference_price(evidence),
            "confidence": unit_score(getattr(evidence, "confidence", 0.0)),
            "strength": serialize_for_metadata(getattr(evidence, "strength", None)),
            "side": serialize_for_metadata(getattr(evidence, "side", None)),
            "sweep_status": serialize_for_metadata(getattr(evidence, "sweep_status", None)),
            "rank": serialize_for_metadata(swept_evidence_rank(evidence)),
            "raw": serialize_for_metadata(evidence),
        }


__all__ = [
    "StopHuntReversalStrategy",
    "StopHuntReversalStrategyConfig",
]