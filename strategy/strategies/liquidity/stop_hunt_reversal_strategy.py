from __future__ import annotations

from typing import Any

from analytics.liquidity.enums import (
    LiquidityBias,
    LiquidityLevelType,
    LiquiditySide,
    SweepStatus,
)
from analytics.liquidity.models import (
    EqualLevel,
    LiquidityLevel,
    LiquidityMapSnapshot,
    LiquidityZone,
    StopCluster,
)
from strategy.context import StrategyContext
from strategy.enums import (
    EntryType,
    ExitType,
    SetupType,
    SignalOrigin,
    SignalPriority,
    SignalSide,
    SignalStatus,
    TriggerType,
)
from strategy.models import (
    EntryPlan,
    ExecutionPlanDraft,
    ExitPlan,
    FilterResult,
    InvalidationPlan,
    StrategySignal,
    TargetPlan,
    confidence_to_grade,
    confidence_to_strength,
)
from strategy.strategies.liquidity.base_liquidity_strategy import BaseLiquidityStrategy


class StopHuntReversalStrategy(BaseLiquidityStrategy):
    """
    Stop-hunt reversal strategy.

    LONG:
        sell-side liquidity swept below current price -> reversal up

    SHORT:
        buy-side liquidity swept above current price -> reversal down
    """

    @property
    def strategy_name(self) -> str:
        return "stop_hunt_reversal_strategy"

    def evaluate(self, context: StrategyContext) -> StrategySignal | None:
        self.validate_context(context)

        if not self.is_enabled():
            self.log_debug(
                "StopHuntReversalStrategy skipped: disabled",
                symbol=context.symbol,
                timeframe=str(context.timeframe),
            )
            return None

        snapshot = self._extract_snapshot(context)
        if snapshot is None:
            self.log_debug(
                "StopHuntReversalStrategy skipped: liquidity snapshot not found",
                symbol=context.symbol,
                timeframe=str(context.timeframe),
            )
            return None

        if not self._base_context_is_valid(context, snapshot):
            return None

        current_price = self._resolve_current_price(context, snapshot)
        if current_price is None or current_price <= 0:
            self.log_warning(
                "StopHuntReversalStrategy skipped: current price unavailable",
                symbol=context.symbol,
                timeframe=str(context.timeframe),
            )
            return None

        filters = self._run_pre_filters(context, snapshot, current_price)
        if any(result.blocked for result in filters):
            self.log_debug(
                "StopHuntReversalStrategy blocked by filters",
                symbol=context.symbol,
                timeframe=str(snapshot.timeframe),
                filters=[f"{f.name}:{f.decision.value}" for f in filters],
            )
            return None

        candidate = self._find_reversal_candidate(snapshot, current_price)
        if candidate is None:
            self.log_debug(
                "StopHuntReversalStrategy skipped: no stop-hunt reversal candidate",
                symbol=context.symbol,
                timeframe=str(snapshot.timeframe),
            )
            return None

        side = candidate["side"]
        confidence = self._compute_confidence(snapshot, current_price, candidate)
        score = self._compute_score(snapshot, current_price, candidate, confidence)

        runtime = self._runtime
        if confidence < runtime.min_confidence or score < runtime.min_score:
            self.log_debug(
                "StopHuntReversalStrategy skipped: below thresholds",
                symbol=context.symbol,
                timeframe=str(snapshot.timeframe),
                confidence=confidence,
                score=score,
                min_confidence=runtime.min_confidence,
                min_score=runtime.min_score,
            )
            return None

        signal = self._build_signal(
            context=context,
            snapshot=snapshot,
            current_price=current_price,
            candidate=candidate,
            side=side,
            confidence=confidence,
            score=score,
            filters=filters,
        )
        signal.validate()
        return signal

    async def evaluate_and_emit(
        self,
        context: StrategyContext,
        **emit_kwargs: Any,
    ) -> StrategySignal | None:
        signal = self.evaluate(context)
        if signal is None:
            return None

        return await self.emit_signal(
            signal=signal,
            context=context,
            **emit_kwargs,
        )

    def _run_pre_filters(
        self,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> list[FilterResult]:
        return self._run_common_pre_filters(
            context=context,
            snapshot=snapshot,
            current_price=current_price,
        )

    def _find_reversal_candidate(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> dict[str, Any] | None:
        sell_side = self._find_sell_side_stop_hunt(snapshot, current_price)
        buy_side = self._find_buy_side_stop_hunt(snapshot, current_price)

        if sell_side is None and buy_side is None:
            return None
        if sell_side is None:
            return buy_side
        if buy_side is None:
            return sell_side

        return sell_side if float(sell_side["edge"]) >= float(buy_side["edge"]) else buy_side

    def _find_sell_side_stop_hunt(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> dict[str, Any] | None:
        level_candidates = [
            level
            for level in snapshot.equal_levels
            if level.side == LiquiditySide.SELL_SIDE
            and level.price < current_price
        ]
        cluster_candidates = [
            cluster
            for cluster in snapshot.stop_clusters
            if cluster.side == LiquiditySide.SELL_SIDE
            and cluster.center_price < current_price
        ]

        swept_levels = [
            level
            for level in level_candidates
            if level.sweep_status in {SweepStatus.SWEPT, SweepStatus.PARTIALLY_SWEPT}
        ]

        if not swept_levels and not cluster_candidates:
            return None

        level = self._pick_best_level(swept_levels or level_candidates, current_price)
        cluster = self._pick_best_cluster(cluster_candidates, current_price)

        if level is None and cluster is None:
            return None

        reference = level if level is not None else cluster
        reference_price = self._reference_price(reference)

        level_score = self._level_reversal_score(level, current_price)
        cluster_score = self._cluster_reversal_score(cluster, current_price)
        reclaim_score = self._reclaim_score_from_reference(
            current_price=current_price,
            reference_price=reference_price,
            side=SignalSide.LONG,
        )
        pressure_bonus = max(-snapshot.liquidity_pressure_score, 0.0) * 0.20
        bias_penalty = 0.08 if snapshot.bias == LiquidityBias.DOWN else 0.0

        edge = max(level_score, cluster_score) + reclaim_score + pressure_bonus - bias_penalty
        if edge <= 0.15:
            return None

        return {
            "direction": "sell_side_stop_hunt",
            "side": SignalSide.LONG,
            "hunted_level": level,
            "support_cluster": cluster,
            "target": self._nearest_opposite_target(snapshot, SignalSide.LONG),
            "edge": max(0.0, edge),
        }

    def _find_buy_side_stop_hunt(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> dict[str, Any] | None:
        level_candidates = [
            level
            for level in snapshot.equal_levels
            if level.side == LiquiditySide.BUY_SIDE
            and level.price > current_price
        ]
        cluster_candidates = [
            cluster
            for cluster in snapshot.stop_clusters
            if cluster.side == LiquiditySide.BUY_SIDE
            and cluster.center_price > current_price
        ]

        swept_levels = [
            level
            for level in level_candidates
            if level.sweep_status in {SweepStatus.SWEPT, SweepStatus.PARTIALLY_SWEPT}
        ]

        if not swept_levels and not cluster_candidates:
            return None

        level = self._pick_best_level(swept_levels or level_candidates, current_price)
        cluster = self._pick_best_cluster(cluster_candidates, current_price)

        if level is None and cluster is None:
            return None

        reference = level if level is not None else cluster
        reference_price = self._reference_price(reference)

        level_score = self._level_reversal_score(level, current_price)
        cluster_score = self._cluster_reversal_score(cluster, current_price)
        reclaim_score = self._reclaim_score_from_reference(
            current_price=current_price,
            reference_price=reference_price,
            side=SignalSide.SHORT,
        )
        pressure_bonus = max(snapshot.liquidity_pressure_score, 0.0) * 0.20
        bias_penalty = 0.08 if snapshot.bias == LiquidityBias.UP else 0.0

        edge = max(level_score, cluster_score) + reclaim_score + pressure_bonus - bias_penalty
        if edge <= 0.15:
            return None

        return {
            "direction": "buy_side_stop_hunt",
            "side": SignalSide.SHORT,
            "hunted_level": level,
            "resistance_cluster": cluster,
            "target": self._nearest_opposite_target(snapshot, SignalSide.SHORT),
            "edge": max(0.0, edge),
        }

    def _pick_best_level(
        self,
        levels: list[EqualLevel | LiquidityLevel],
        current_price: float,
    ) -> EqualLevel | LiquidityLevel | None:
        if not levels:
            return None

        def score(level: EqualLevel | LiquidityLevel) -> float:
            distance_bonus = self._distance_bonus(level.price, current_price)

            sweep_bonus = 0.0
            if level.sweep_status == SweepStatus.SWEPT:
                sweep_bonus = 0.18
            elif level.sweep_status == SweepStatus.PARTIALLY_SWEPT:
                sweep_bonus = 0.10

            touches_bonus = min(level.touches_count / 6.0, 1.0) * 0.20
            reaction_bonus = min(level.reaction_count / 4.0, 1.0) * 0.15
            confidence_bonus = min(max(level.confidence, 0.0), 1.0) * 0.35

            type_bonus = 0.0
            if level.level_type in {
                LiquidityLevelType.EQUAL_HIGHS,
                LiquidityLevelType.EQUAL_LOWS,
            }:
                type_bonus = 0.12

            return (
                confidence_bonus
                + touches_bonus
                + reaction_bonus
                + distance_bonus
                + sweep_bonus
                + type_bonus
            )

        return max(levels, key=score)

    def _pick_best_cluster(
        self,
        clusters: list[StopCluster],
        current_price: float,
    ) -> StopCluster | None:
        if not clusters:
            return None

        def score(cluster: StopCluster) -> float:
            strength_value = getattr(cluster.strength, "value", str(cluster.strength))

            strength_bonus = 0.0
            if strength_value == "medium":
                strength_bonus = 0.06
            elif strength_value == "high":
                strength_bonus = 0.12
            elif strength_value == "extreme":
                strength_bonus = 0.18

            return (
                self._distance_bonus(cluster.center_price, current_price)
                + min(max(cluster.confidence, 0.0), 1.0) * 0.45
                + min(max(cluster.estimated_stop_density, 0.0), 1.0) * 0.25
                + min(cluster.touches_count / 6.0, 1.0) * 0.15
                + strength_bonus
            )

        return max(clusters, key=score)

    def _distance_bonus(self, level_price: float, current_price: float) -> float:
        if current_price <= 0 or level_price <= 0:
            return 0.0

        distance_pct = abs(level_price - current_price) / current_price

        if distance_pct <= 0.0015:
            return 0.22
        if distance_pct <= 0.0040:
            return 0.18
        if distance_pct <= 0.0100:
            return 0.12
        if distance_pct <= 0.0200:
            return 0.06
        return 0.0

    def _level_reversal_score(
        self,
        level: EqualLevel | LiquidityLevel | None,
        current_price: float,
    ) -> float:
        if level is None:
            return 0.0

        score = min(max(level.confidence, 0.0), 1.0) * 0.45
        score += min(level.touches_count / 6.0, 1.0) * 0.15
        score += min(level.reaction_count / 4.0, 1.0) * 0.10
        score += self._distance_bonus(level.price, current_price)

        if level.sweep_status == SweepStatus.SWEPT:
            score += 0.18
        elif level.sweep_status == SweepStatus.PARTIALLY_SWEPT:
            score += 0.10

        if level.level_type in {
            LiquidityLevelType.EQUAL_HIGHS,
            LiquidityLevelType.EQUAL_LOWS,
        }:
            score += 0.10

        return score

    def _cluster_reversal_score(
        self,
        cluster: StopCluster | None,
        current_price: float,
    ) -> float:
        if cluster is None:
            return 0.0

        score = min(max(cluster.confidence, 0.0), 1.0) * 0.40
        score += min(max(cluster.estimated_stop_density, 0.0), 1.0) * 0.20
        score += min(cluster.touches_count / 6.0, 1.0) * 0.10
        score += self._distance_bonus(cluster.center_price, current_price)

        strength_value = getattr(cluster.strength, "value", str(cluster.strength))
        if strength_value == "medium":
            score += 0.05
        elif strength_value == "high":
            score += 0.10
        elif strength_value == "extreme":
            score += 0.15

        return score

    def _reclaim_score_from_reference(
        self,
        current_price: float,
        reference_price: float,
        side: SignalSide,
    ) -> float:
        if current_price <= 0 or reference_price <= 0:
            return 0.0

        if side == SignalSide.LONG:
            if current_price <= reference_price:
                return 0.0
            reclaim_pct = (current_price - reference_price) / reference_price
        else:
            if current_price >= reference_price:
                return 0.0
            reclaim_pct = (reference_price - current_price) / reference_price

        if reclaim_pct <= 0.0005:
            return 0.06
        if reclaim_pct <= 0.0020:
            return 0.16
        if reclaim_pct <= 0.0060:
            return 0.22
        return 0.12

    def _nearest_opposite_target(
        self,
        snapshot: LiquidityMapSnapshot,
        side: SignalSide,
    ) -> LiquidityLevel | StopCluster | None:
        if side == SignalSide.LONG:
            return snapshot.nearest_above_level
        return snapshot.nearest_below_level

    def _compute_confidence(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        candidate: dict[str, Any],
    ) -> float:
        side: SignalSide = candidate["side"]
        edge = float(candidate["edge"])

        hunted_level = candidate.get("hunted_level")
        support_cluster = candidate.get("support_cluster")
        resistance_cluster = candidate.get("resistance_cluster")
        target = candidate.get("target")

        confidence = min(edge, 1.0) * 0.55

        if hunted_level is not None:
            confidence += self._level_reversal_score(hunted_level, current_price) * 0.20

        cluster = support_cluster if support_cluster is not None else resistance_cluster
        if cluster is not None:
            confidence += self._cluster_reversal_score(cluster, current_price) * 0.15

        if target is not None:
            confidence += self._target_quality_bonus(target, current_price) * 0.10

        confidence += self._reversal_zone_bonus(snapshot.zones, side, current_price)

        if snapshot.signal is not None:
            if side == SignalSide.LONG and snapshot.signal.bias == LiquidityBias.DOWN:
                confidence += 0.05
            elif side == SignalSide.SHORT and snapshot.signal.bias == LiquidityBias.UP:
                confidence += 0.05

        return max(0.0, min(confidence, 1.0))

    def _compute_score(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        candidate: dict[str, Any],
        confidence: float,
    ) -> float:
        side: SignalSide = candidate["side"]
        edge = float(candidate["edge"])
        target = candidate.get("target")

        anti_bias_bonus = 0.0
        if side == SignalSide.LONG and snapshot.bias == LiquidityBias.DOWN:
            anti_bias_bonus = 0.30
        elif side == SignalSide.SHORT and snapshot.bias == LiquidityBias.UP:
            anti_bias_bonus = 0.30

        return max(
            0.0,
            confidence * 1.20
            + edge * 0.90
            + self._target_distance_score(target, current_price) * 0.45
            + anti_bias_bonus,
        )

    def _target_quality_bonus(
        self,
        target: LiquidityLevel | StopCluster | None,
        current_price: float,
    ) -> float:
        if target is None:
            return 0.0

        ref_price = self._reference_price(target)
        if current_price <= 0 or ref_price <= 0:
            return 0.0

        distance_pct = abs(ref_price - current_price) / current_price

        bonus = 0.0
        if 0.0010 <= distance_pct <= 0.0150:
            bonus += 0.06
        elif distance_pct <= 0.0300:
            bonus += 0.03

        if isinstance(target, StopCluster):
            bonus += 0.05 * min(max(target.confidence, 0.0), 1.0)
        elif isinstance(target, LiquidityLevel):
            bonus += 0.04 * min(max(target.confidence, 0.0), 1.0)

        return bonus

    def _target_distance_score(
        self,
        target: LiquidityLevel | StopCluster | None,
        current_price: float,
    ) -> float:
        if target is None or current_price <= 0:
            return 0.0

        ref_price = self._reference_price(target)
        if ref_price <= 0:
            return 0.0

        distance_pct = abs(ref_price - current_price) / current_price

        if distance_pct <= 0.001:
            return 0.10
        if distance_pct <= 0.003:
            return 0.45
        if distance_pct <= 0.010:
            return 1.00
        if distance_pct <= 0.020:
            return 0.70
        if distance_pct <= 0.040:
            return 0.35
        return 0.10

    def _reversal_zone_bonus(
        self,
        zones: list[LiquidityZone],
        side: SignalSide,
        current_price: float,
    ) -> float:
        if not zones:
            return 0.0

        if side == SignalSide.LONG:
            relevant = [
                zone
                for zone in zones
                if zone.side in {LiquiditySide.BUY_SIDE, LiquiditySide.BOTH}
                and zone.center_price > current_price
            ]
        else:
            relevant = [
                zone
                for zone in zones
                if zone.side in {LiquiditySide.SELL_SIDE, LiquiditySide.BOTH}
                and zone.center_price < current_price
            ]

        if not relevant:
            return 0.0

        best = max(relevant, key=lambda zone: zone.score)
        return 0.05 * min(max(best.score, 0.0), 1.0)

    def _build_signal(
        self,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        candidate: dict[str, Any],
        side: SignalSide,
        confidence: float,
        score: float,
        filters: list[FilterResult],
    ) -> StrategySignal:
        hunted_level = candidate.get("hunted_level")
        support_cluster = candidate.get("support_cluster")
        resistance_cluster = candidate.get("resistance_cluster")
        target = candidate.get("target")

        invalidation_anchor = (
            hunted_level or support_cluster
            if side == SignalSide.LONG
            else hunted_level or resistance_cluster
        )

        entry_plan = self._build_entry_plan(
            side=side,
            current_price=current_price,
            target=target,
        )
        exit_plan = self._build_exit_plan(
            side=side,
            current_price=current_price,
            target=target,
            invalidation_anchor=invalidation_anchor,
            snapshot=snapshot,
        )
        invalidation_plan = self._build_invalidation_plan(
            side=side,
            current_price=current_price,
            invalidation_anchor=invalidation_anchor,
        )
        execution_plan = self._build_execution_plan(
            symbol=context.symbol,
            side=side,
            entry_plan=entry_plan,
            exit_plan=exit_plan,
            invalidation_plan=invalidation_plan,
        )

        priority = SignalPriority.MEDIUM
        if score >= 1.75 or confidence >= 0.85:
            priority = SignalPriority.HIGH
        if score >= 2.15 and confidence >= 0.90:
            priority = SignalPriority.CRITICAL

        signal = StrategySignal(
            symbol=context.symbol,
            side=side,
            strategy_name=self.strategy_name,
            category=self.category,
            timeframe=context.timeframe,
            setup_type=SetupType.REVERSAL,
            timestamp=context.timestamp,
            confidence=confidence,
            score=score,
            strength=confidence_to_strength(confidence),
            confidence_grade=confidence_to_grade(confidence),
            status=SignalStatus.NEW,
            trigger_type=TriggerType.PRIMARY,
            origin=SignalOrigin.SINGLE_STRATEGY,
            priority=priority,
            entry_plan=entry_plan,
            exit_plan=exit_plan,
            invalidation_plan=invalidation_plan,
            execution_plan=execution_plan,
            regime=context.regime.regime if context.regime is not None else self._unknown_regime(),
            metadata={
                "snapshot_timestamp": snapshot.timestamp.isoformat(),
                "snapshot_timeframe": snapshot.timeframe,
                "current_price": current_price,
                "bias": snapshot.bias.value,
                "above_liquidity_score": snapshot.above_liquidity_score,
                "below_liquidity_score": snapshot.below_liquidity_score,
                "liquidity_pressure_score": snapshot.liquidity_pressure_score,
                "sweep_risk_up": snapshot.signal.sweep_risk_up
                if snapshot.signal is not None
                else None,
                "sweep_risk_down": snapshot.signal.sweep_risk_down
                if snapshot.signal is not None
                else None,
                "direction": candidate["direction"],
                "strategy_weight": self.config.get_strategy_weight(
                    self.strategy_name,
                    default=1.0,
                ),
            },
        )

        signal.add_reason(self._build_primary_reason(candidate, current_price))
        signal.add_reason(self._build_target_reason(target))
        signal.add_reason(f"Liquidity pressure score = {snapshot.liquidity_pressure_score:.3f}")

        if snapshot.signal is not None and snapshot.signal.explanation:
            signal.add_reason(snapshot.signal.explanation)

        for confirmation in self._build_confirmations(
            snapshot,
            candidate,
            current_price,
        ):
            signal.add_confirmation(confirmation)

        signal.add_source_feature("liquidity_map_snapshot")
        signal.add_source_feature("liquidity")
        signal.add_source_feature("liquidity.stop_hunt")
        signal.add_source_feature("liquidity.equal_levels")
        signal.add_source_feature("liquidity.stop_clusters")

        for filter_result in filters:
            signal.add_filter_result(filter_result)

        signal.validate()
        return signal

    def _build_entry_plan(
        self,
        side: SignalSide,
        current_price: float,
        target: LiquidityLevel | StopCluster | None,
    ) -> EntryPlan:
        notes = ["Enter after stop-hunt reversal thesis is active"]

        if target is not None:
            notes.append(f"Primary target liquidity at {self._reference_price(target):.6f}")

        entry_type = self.config.builders.default_entry_type or EntryType.MARKET

        return EntryPlan(
            entry_type=entry_type,
            price=current_price if entry_type == EntryType.LIMIT else None,
            timeout_seconds=self.config.runtime.max_signal_age_seconds,
            max_slippage_bps=8.0,
            confirmation_required=False,
            notes=notes,
            metadata={"entry_logic": "stop_hunt_reversal"},
        )

    def _build_exit_plan(
        self,
        side: SignalSide,
        current_price: float,
        target: LiquidityLevel | StopCluster | None,
        invalidation_anchor: LiquidityLevel | StopCluster | None,
        snapshot: LiquidityMapSnapshot,
    ) -> ExitPlan:
        stop_price = self._resolve_stop_price(side, current_price, invalidation_anchor)
        tp_levels: list[TargetPlan] = []

        primary_target_price = self._reference_price(target) if target is not None else None
        if primary_target_price is not None and primary_target_price > 0:
            tp_levels.append(
                TargetPlan(
                    price=primary_target_price,
                    size_fraction=0.70
                    if self.config.builders.enable_partial_take_profit
                    else 1.0,
                    rr=self._compute_rr(
                        current_price=current_price,
                        stop_price=stop_price,
                        target_price=primary_target_price,
                        side=side,
                    ),
                    label="reversal_target_liquidity",
                    metadata={"source": "nearest_opposite_side_liquidity"},
                )
            )

        secondary_target = self._find_extended_target(
            snapshot,
            current_price,
            side,
            exclude=target,
        )
        secondary_price = (
            self._reference_price(secondary_target)
            if secondary_target is not None
            else None
        )

        if (
            self.config.builders.enable_partial_take_profit
            and secondary_price is not None
            and secondary_price > 0
        ):
            tp_levels.append(
                TargetPlan(
                    price=secondary_price,
                    size_fraction=0.30,
                    rr=self._compute_rr(
                        current_price=current_price,
                        stop_price=stop_price,
                        target_price=secondary_price,
                        side=side,
                    ),
                    label="extended_reversal_target",
                    metadata={"source": "extended_opposite_side_liquidity"},
                )
            )

        return ExitPlan(
            exit_types=[
                ExitType.TAKE_PROFIT,
                ExitType.STOP_LOSS,
                ExitType.INVALIDATION,
            ],
            stop_loss=stop_price,
            take_profit_levels=tp_levels,
            trailing_distance=None,
            max_holding_seconds=max(self.config.runtime.max_signal_age_seconds * 3, 60),
            partial_exit_enabled=self.config.builders.enable_partial_take_profit,
            metadata={"exit_logic": "reversal_to_opposite_liquidity"},
        )

    def _build_invalidation_plan(
        self,
        side: SignalSide,
        current_price: float,
        invalidation_anchor: LiquidityLevel | StopCluster | None,
    ) -> InvalidationPlan:
        price = self._resolve_stop_price(side, current_price, invalidation_anchor)

        reason = None
        if self.config.builders.require_invalidation:
            reason = (
                "Swept sell-side liquidity failed to hold as support"
                if side == SignalSide.LONG
                else "Swept buy-side liquidity failed to hold as resistance"
            )

        return InvalidationPlan(
            price=price,
            reason=reason,
            timeout_seconds=max(self.config.runtime.max_signal_age_seconds, 30),
            conditions=[
                "signal_age_expired",
                "reversal_failed_reclaim",
                "opposite_liquidity_pressure_domination",
            ],
            metadata={"invalidation_source": "stop_hunt_reversal_anchor"},
        )

    def _build_execution_plan(
        self,
        symbol: str,
        side: SignalSide,
        entry_plan: EntryPlan,
        exit_plan: ExitPlan,
        invalidation_plan: InvalidationPlan,
    ) -> ExecutionPlanDraft:
        return ExecutionPlanDraft(
            symbol=symbol,
            side=side,
            entry=entry_plan,
            exit=exit_plan,
            invalidation=invalidation_plan,
            leverage=None,
            reduce_only=False,
            post_only=entry_plan.entry_type == EntryType.PASSIVE,
            expected_holding_seconds=exit_plan.max_holding_seconds,
            notes=[
                "Generated by StopHuntReversalStrategy",
                "Risk manager must validate portfolio/correlation constraints before execution",
            ],
            metadata={
                "strategy_name": self.strategy_name,
                "category": self.category.value,
            },
        )

    def _build_primary_reason(
        self,
        candidate: dict[str, Any],
        current_price: float,
    ) -> str:
        side: SignalSide = candidate["side"]
        hunted_level = candidate.get("hunted_level")
        cluster = candidate.get("support_cluster") or candidate.get("resistance_cluster")
        edge = float(candidate["edge"])

        parts = [f"Stop-hunt reversal edge = {edge:.3f}"]

        if hunted_level is not None:
            parts.append(
                f"hunted_level={hunted_level.price:.6f} "
                f"({hunted_level.level_type.value}, {hunted_level.sweep_status.value})"
            )

        if cluster is not None:
            parts.append(
                f"cluster={cluster.center_price:.6f} "
                f"(conf={cluster.confidence:.3f}, density={cluster.estimated_stop_density:.3f})"
            )

        parts.append(f"current_price={current_price:.6f}")

        prefix = (
            "Sell-side stop hunt reclaimed -> long reversal"
            if side == SignalSide.LONG
            else "Buy-side stop hunt rejected -> short reversal"
        )

        return f"{prefix}: {', '.join(parts)}"

    def _build_target_reason(self, target: LiquidityLevel | StopCluster | None) -> str:
        if target is None:
            return "No explicit reversal target found; signal based on reclaimed liquidity structure"

        if isinstance(target, StopCluster):
            return (
                f"Nearest target is stop cluster at {target.center_price:.6f} "
                f"(confidence={target.confidence:.3f}, strength={target.strength.value})"
            )

        return (
            f"Nearest target is liquidity level at {target.price:.6f} "
            f"(type={target.level_type.value}, confidence={target.confidence:.3f}, "
            f"sweep_status={target.sweep_status.value})"
        )

    def _build_confirmations(
        self,
        snapshot: LiquidityMapSnapshot,
        candidate: dict[str, Any],
        current_price: float,
    ) -> list[str]:
        confirmations: list[str] = []

        side: SignalSide = candidate["side"]
        hunted_level = candidate.get("hunted_level")
        cluster = candidate.get("support_cluster") or candidate.get("resistance_cluster")
        target = candidate.get("target")

        if hunted_level is not None:
            if hunted_level.sweep_status == SweepStatus.SWEPT:
                confirmations.append("Liquidity level fully swept")
            elif hunted_level.sweep_status == SweepStatus.PARTIALLY_SWEPT:
                confirmations.append("Liquidity level partially swept")

            if hunted_level.level_type in {
                LiquidityLevelType.EQUAL_HIGHS,
                LiquidityLevelType.EQUAL_LOWS,
            }:
                confirmations.append("Equal highs/lows structure involved in stop hunt")

        if cluster is not None and cluster.confidence >= 0.65:
            confirmations.append("Strong nearby stop cluster supports reversal thesis")

        if side == SignalSide.LONG:
            if snapshot.bias == LiquidityBias.DOWN:
                confirmations.append("Counter-bias reversal setup after sell-side sweep")

            if self._has_high_quality_zone(
                snapshot.zones,
                LiquiditySide.BUY_SIDE,
                current_price,
            ):
                confirmations.append("High-quality buy-side target zone ahead")
        else:
            if snapshot.bias == LiquidityBias.UP:
                confirmations.append("Counter-bias reversal setup after buy-side sweep")

            if self._has_high_quality_zone(
                snapshot.zones,
                LiquiditySide.SELL_SIDE,
                current_price,
            ):
                confirmations.append("High-quality sell-side target zone ahead")

        if target is not None:
            confirmations.append("Clear opposite-side liquidity target available")

        return confirmations

    def _has_high_quality_zone(
        self,
        zones: list[LiquidityZone],
        side: LiquiditySide,
        current_price: float,
    ) -> bool:
        for zone in zones:
            if zone.side not in {side, LiquiditySide.BOTH}:
                continue

            if side == LiquiditySide.BUY_SIDE and zone.center_price <= current_price:
                continue

            if side == LiquiditySide.SELL_SIDE and zone.center_price >= current_price:
                continue

            if zone.score >= 0.65:
                return True

        return False

    def _resolve_stop_price(
        self,
        side: SignalSide,
        current_price: float,
        invalidation_anchor: LiquidityLevel | StopCluster | None,
    ) -> float | None:
        if current_price <= 0:
            return None

        anchor_price = (
            self._reference_price(invalidation_anchor)
            if invalidation_anchor is not None
            else None
        )
        fallback_pct = 0.0045

        if side == SignalSide.LONG:
            if anchor_price is not None and anchor_price < current_price:
                return anchor_price * 0.9985
            return current_price * (1.0 - fallback_pct)

        if side == SignalSide.SHORT:
            if anchor_price is not None and anchor_price > current_price:
                return anchor_price * 1.0015
            return current_price * (1.0 + fallback_pct)

        return None

    def _find_extended_target(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
        exclude: LiquidityLevel | StopCluster | None = None,
    ) -> LiquidityLevel | StopCluster | None:
        candidates: list[LiquidityLevel | StopCluster] = []

        if side == SignalSide.LONG:
            candidates.extend(level for level in snapshot.active_levels if level.price > current_price)
            candidates.extend(cluster for cluster in snapshot.stop_clusters if cluster.center_price > current_price)
            candidates.sort(key=self._reference_price)

        elif side == SignalSide.SHORT:
            candidates.extend(level for level in snapshot.active_levels if level.price < current_price)
            candidates.extend(cluster for cluster in snapshot.stop_clusters if cluster.center_price < current_price)
            candidates.sort(key=self._reference_price, reverse=True)

        if exclude is not None:
            exclude_price = self._reference_price(exclude)
            candidates = [
                item
                for item in candidates
                if abs(self._reference_price(item) - exclude_price) > 1e-12
            ]

        return candidates[0] if candidates else None

    def _compute_rr(
        self,
        current_price: float,
        stop_price: float | None,
        target_price: float | None,
        side: SignalSide,
    ) -> float | None:
        if stop_price is None or target_price is None or current_price <= 0:
            return None

        if side == SignalSide.LONG:
            risk = current_price - stop_price
            reward = target_price - current_price
        else:
            risk = stop_price - current_price
            reward = current_price - target_price

        if risk <= 0 or reward <= 0:
            return None

        return reward / risk

    def _reference_price(self, item: LiquidityLevel | StopCluster | None) -> float:
        if item is None:
            return 0.0

        if isinstance(item, StopCluster):
            return float(item.center_price)

        return float(item.price)

    def _unknown_regime(self):
        from strategy.enums import MarketRegime

        return MarketRegime.UNKNOWN