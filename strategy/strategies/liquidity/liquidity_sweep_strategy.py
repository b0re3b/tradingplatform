from __future__ import annotations

from typing import Any

from analytics.liquidity.enums import (
    LiquidityBias,
    LiquidityLevelType,
    LiquiditySide,
    SweepStatus,
)
from analytics.liquidity.models import (
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


class LiquiditySweepStrategy(BaseLiquidityStrategy):
    """
    Strategy для торгівлі в напрямку dominant liquidity magnet / sweep path.

    Semantics:
    - LONG означає очікування руху до buy-side liquidity вище current price;
    - SHORT означає очікування руху до sell-side liquidity нижче current price;
    - це НЕ stop-hunt reversal strategy;
    - swept/terminal liquidity не повинна бути основним target-ом;
    - strategy не виконує угоди напряму, а лише формує StrategySignal.
    """

    EDGE_THRESHOLD: float = 0.45
    EDGE_DELTA_THRESHOLD: float = 0.12

    HIGH_PRIORITY_SCORE: float = 1.80
    HIGH_PRIORITY_CONFIDENCE: float = 0.85
    CRITICAL_PRIORITY_SCORE: float = 2.20
    CRITICAL_PRIORITY_CONFIDENCE: float = 0.90

    FALLBACK_STOP_PCT: float = 0.0040
    LONG_STOP_OFFSET: float = 0.9985
    SHORT_STOP_OFFSET: float = 1.0015

    @property
    def strategy_name(self) -> str:
        return "liquidity_sweep_strategy"

    def evaluate(self, context: StrategyContext) -> StrategySignal | None:
        self.validate_context(context)

        if not self.is_enabled():
            self.log_debug(
                "LiquiditySweepStrategy skipped: disabled",
                symbol=context.symbol,
                timeframe=self._value(context.timeframe),
            )
            return None

        snapshot = self._extract_snapshot(context)
        if snapshot is None:
            self.log_debug(
                "LiquiditySweepStrategy skipped: liquidity snapshot not found",
                symbol=context.symbol,
                timeframe=self._value(context.timeframe),
            )
            return None

        if not self._base_context_is_valid(context, snapshot):
            return None

        current_price = self._resolve_current_price(context, snapshot)
        if current_price is None:
            self.log_warning(
                "LiquiditySweepStrategy skipped: current price unavailable",
                symbol=context.symbol,
                timeframe=self._value(context.timeframe),
            )
            return None

        filters = self._run_pre_filters(
            context=context,
            snapshot=snapshot,
            current_price=current_price,
        )
        if any(result.blocked for result in filters):
            self.log_debug(
                "LiquiditySweepStrategy blocked by filters",
                symbol=context.symbol,
                timeframe=self._value(snapshot.timeframe),
                filters=[f"{item.name}:{item.decision.value}" for item in filters],
            )
            return None

        side = self._infer_side(snapshot)
        if side == SignalSide.UNKNOWN:
            self.log_debug(
                "LiquiditySweepStrategy skipped: no directional sweep edge",
                symbol=context.symbol,
                timeframe=self._value(snapshot.timeframe),
                up_edge=self._upside_edge(snapshot),
                down_edge=self._downside_edge(snapshot),
            )
            return None

        target = self._target_for_side(
            snapshot=snapshot,
            current_price=current_price,
            side=side,
        )
        if target is None:
            self.log_debug(
                "LiquiditySweepStrategy skipped: no directional liquidity target",
                symbol=context.symbol,
                timeframe=self._value(snapshot.timeframe),
                side=side.value,
            )
            return None

        confidence = self._compute_confidence(
            snapshot=snapshot,
            current_price=current_price,
            side=side,
            target=target,
        )
        score = self._compute_score(
            snapshot=snapshot,
            current_price=current_price,
            side=side,
            confidence=confidence,
            target=target,
        )

        runtime = self._runtime
        if confidence < runtime.min_confidence or score < runtime.min_score:
            self.log_debug(
                "LiquiditySweepStrategy skipped: below thresholds",
                symbol=context.symbol,
                timeframe=self._value(snapshot.timeframe),
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
            side=side,
            target=target,
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
        """
        Використовує common-фільтри з BaseLiquidityStrategy.

        Тут не дублюємо portfolio/spread/price/snapshot checks.
        Додаємо тільки sweep-specific перевірку directional edge.
        """
        results = self._run_common_pre_filters(
            context=context,
            snapshot=snapshot,
            current_price=current_price,
        )

        up_edge = self._upside_edge(snapshot)
        down_edge = self._downside_edge(snapshot)
        directional_edge = max(up_edge, down_edge)
        edge_delta = abs(up_edge - down_edge)

        if directional_edge < self.EDGE_THRESHOLD and edge_delta < self.EDGE_DELTA_THRESHOLD:
            results.append(
                FilterResult(
                    name="liquidity_sweep_directional_edge",
                    decision="block",
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
                    decision="pass",
                    reason=(
                        "Directional liquidity sweep edge present: "
                        f"up_edge={up_edge:.4f}, down_edge={down_edge:.4f}, "
                        f"delta={edge_delta:.4f}"
                    ),
                )
            )

        return results

    def _infer_side(self, snapshot: LiquidityMapSnapshot) -> SignalSide:
        up_edge = self._upside_edge(snapshot)
        down_edge = self._downside_edge(snapshot)
        delta = up_edge - down_edge

        if snapshot.bias == LiquidityBias.UP and up_edge >= self.EDGE_THRESHOLD:
            return SignalSide.LONG

        if snapshot.bias == LiquidityBias.DOWN and down_edge >= self.EDGE_THRESHOLD:
            return SignalSide.SHORT

        if delta >= self.EDGE_DELTA_THRESHOLD:
            return SignalSide.LONG

        if delta <= -self.EDGE_DELTA_THRESHOLD:
            return SignalSide.SHORT

        return SignalSide.UNKNOWN

    def _compute_confidence(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
        target: LiquidityLevel | StopCluster,
    ) -> float:
        if side == SignalSide.LONG:
            base = (
                0.32 * self._magnet_score_up(snapshot)
                + 0.28 * self._sweep_risk_up(snapshot)
                + 0.20 * snapshot.above_liquidity_score
                + 0.20 * max(snapshot.liquidity_pressure_score, 0.0)
            )
            zone_bonus = self._zone_alignment_bonus(
                zones=snapshot.zones,
                side=LiquiditySide.BUY_SIDE,
                current_price=current_price,
            )

        elif side == SignalSide.SHORT:
            base = (
                0.32 * self._magnet_score_down(snapshot)
                + 0.28 * self._sweep_risk_down(snapshot)
                + 0.20 * snapshot.below_liquidity_score
                + 0.20 * max(-snapshot.liquidity_pressure_score, 0.0)
            )
            zone_bonus = self._zone_alignment_bonus(
                zones=snapshot.zones,
                side=LiquiditySide.SELL_SIDE,
                current_price=current_price,
            )

        else:
            return 0.0

        target_bonus = self._target_quality_bonus(target, current_price)
        return self._clamp01(base + target_bonus + zone_bonus)

    def _compute_score(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
        confidence: float,
        target: LiquidityLevel | StopCluster,
    ) -> float:
        if side == SignalSide.LONG:
            directional_edge = self._upside_edge(snapshot) - self._downside_edge(snapshot)
        else:
            directional_edge = self._downside_edge(snapshot) - self._upside_edge(snapshot)

        return max(
            0.0,
            1.30 * confidence
            + 0.80 * max(directional_edge, 0.0)
            + 0.40 * self._target_distance_score(target, current_price),
        )

    def _upside_edge(self, snapshot: LiquidityMapSnapshot) -> float:
        return self._clamp01(
            0.40 * self._magnet_score_up(snapshot)
            + 0.35 * self._sweep_risk_up(snapshot)
            + 0.25 * snapshot.above_liquidity_score
        )

    def _downside_edge(self, snapshot: LiquidityMapSnapshot) -> float:
        return self._clamp01(
            0.40 * self._magnet_score_down(snapshot)
            + 0.35 * self._sweep_risk_down(snapshot)
            + 0.25 * snapshot.below_liquidity_score
        )

    def _magnet_score_up(self, snapshot: LiquidityMapSnapshot) -> float:
        return self._snapshot_metric(
            snapshot=snapshot,
            signal_attr="magnet_score_up",
            metadata_key="magnet_score_up",
        )

    def _magnet_score_down(self, snapshot: LiquidityMapSnapshot) -> float:
        return self._snapshot_metric(
            snapshot=snapshot,
            signal_attr="magnet_score_down",
            metadata_key="magnet_score_down",
        )

    def _sweep_risk_up(self, snapshot: LiquidityMapSnapshot) -> float:
        return self._snapshot_metric(
            snapshot=snapshot,
            signal_attr="sweep_risk_up",
            metadata_key="sweep_risk_up",
        )

    def _sweep_risk_down(self, snapshot: LiquidityMapSnapshot) -> float:
        return self._snapshot_metric(
            snapshot=snapshot,
            signal_attr="sweep_risk_down",
            metadata_key="sweep_risk_down",
        )

    def _snapshot_metric(
        self,
        snapshot: LiquidityMapSnapshot,
        signal_attr: str,
        metadata_key: str,
    ) -> float:
        """
        Читає analytics-derived metric із snapshot.signal,
        а якщо signal=None — fallback у snapshot.metadata.
        """
        if snapshot.signal is not None:
            value = getattr(snapshot.signal, signal_attr, None)
            resolved = self._safe_float(value)
            if resolved is not None:
                return self._clamp01(resolved)

        metadata = getattr(snapshot, "metadata", None) or {}
        value = metadata.get(metadata_key)
        resolved = self._safe_float(value)

        return self._clamp01(resolved or 0.0)

    def _target_for_side(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
    ) -> LiquidityLevel | StopCluster | None:
        """
        Обирає directional target для magnet-follow-through.

        Пріоритет:
        1. nearest directional liquidity із snapshot;
        2. strongest directional cluster;
        3. fallback scan по active levels / active clusters.

        Terminal/swept targets не використовуються як primary target для цієї
        strategy, бо це задача StopHuntReversalStrategy.
        """
        if side == SignalSide.LONG:
            candidates = [
                snapshot.nearest_above_level,
                snapshot.strongest_cluster_above,
                *self._collect_directional_targets(
                    snapshot=snapshot,
                    current_price=current_price,
                    side=side,
                ),
            ]
        elif side == SignalSide.SHORT:
            candidates = [
                snapshot.nearest_below_level,
                snapshot.strongest_cluster_below,
                *self._collect_directional_targets(
                    snapshot=snapshot,
                    current_price=current_price,
                    side=side,
                ),
            ]
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

        if not valid:
            return None

        if side == SignalSide.LONG:
            return min(valid, key=self._reference_price)

        return max(valid, key=self._reference_price)

    def _collect_directional_targets(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
    ) -> list[LiquidityLevel | StopCluster]:
        candidates: list[LiquidityLevel | StopCluster] = []

        if side == SignalSide.LONG:
            candidates.extend(
                level
                for level in snapshot.active_levels
                if level.price > current_price
            )
            candidates.extend(
                cluster
                for cluster in snapshot.stop_clusters
                if cluster.center_price > current_price
            )
            candidates.sort(key=self._reference_price)

        elif side == SignalSide.SHORT:
            candidates.extend(
                level
                for level in snapshot.active_levels
                if level.price < current_price
            )
            candidates.extend(
                cluster
                for cluster in snapshot.stop_clusters
                if cluster.center_price < current_price
            )
            candidates.sort(key=self._reference_price, reverse=True)

        return candidates

    def _is_valid_follow_through_target(
        self,
        item: LiquidityLevel | StopCluster,
        current_price: float,
        side: SignalSide,
    ) -> bool:
        ref_price = self._reference_price(item)
        if ref_price <= 0 or current_price <= 0:
            return False

        if side == SignalSide.LONG and ref_price <= current_price:
            return False

        if side == SignalSide.SHORT and ref_price >= current_price:
            return False

        if isinstance(item, LiquidityLevel):
            if item.is_invalidated() or item.is_expired():
                return False

            if item.sweep_status == SweepStatus.SWEPT:
                return False

        if isinstance(item, StopCluster):
            if item.is_swept():
                return False

        return True

    def _target_quality_bonus(
        self,
        target: LiquidityLevel | StopCluster | None,
        current_price: float,
    ) -> float:
        if target is None:
            return 0.0

        ref_price = self._reference_price(target)
        if ref_price <= 0 or current_price <= 0:
            return 0.0

        distance_pct = abs(ref_price - current_price) / current_price

        bonus = 0.0
        if 0.0010 <= distance_pct <= 0.0150:
            bonus += 0.06
        elif distance_pct <= 0.0300:
            bonus += 0.03

        if isinstance(target, StopCluster):
            bonus += 0.05 * self._clamp01(target.confidence)
            if target.strength.value in {"high", "extreme"}:
                bonus += 0.04

        elif isinstance(target, LiquidityLevel):
            bonus += 0.04 * self._clamp01(target.confidence)

            if target.level_type in {
                LiquidityLevelType.EQUAL_HIGHS,
                LiquidityLevelType.EQUAL_LOWS,
            }:
                bonus += 0.03

            if target.sweep_status == SweepStatus.PARTIALLY_SWEPT:
                bonus -= 0.03
            elif target.sweep_status == SweepStatus.SWEPT:
                bonus -= 0.07

        return max(0.0, bonus)

    def _zone_alignment_bonus(
        self,
        zones: list[LiquidityZone],
        side: LiquiditySide,
        current_price: float,
    ) -> float:
        relevant = [
            zone
            for zone in zones
            if zone.side in {side, LiquiditySide.BOTH}
            and (
                (side == LiquiditySide.BUY_SIDE and zone.center_price > current_price)
                or (side == LiquiditySide.SELL_SIDE and zone.center_price < current_price)
            )
        ]

        if not relevant:
            return 0.0

        best = max(relevant, key=lambda zone: zone.score)
        return 0.05 * self._clamp01(best.score)

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
            return 0.15
        if distance_pct <= 0.003:
            return 0.55
        if distance_pct <= 0.010:
            return 1.00
        if distance_pct <= 0.020:
            return 0.70
        if distance_pct <= 0.040:
            return 0.35

        return 0.10

    def _build_signal(
        self,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
        target: LiquidityLevel | StopCluster,
        confidence: float,
        score: float,
        filters: list[FilterResult],
    ) -> StrategySignal:
        invalidation_level = self._invalidation_level_for_side(
            snapshot=snapshot,
            current_price=current_price,
            side=side,
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
            invalidation_level=invalidation_level,
            snapshot=snapshot,
        )
        invalidation_plan = self._build_invalidation_plan(
            side=side,
            current_price=current_price,
            invalidation_level=invalidation_level,
        )
        execution_plan = self._build_execution_plan(
            symbol=context.symbol,
            side=side,
            entry_plan=entry_plan,
            exit_plan=exit_plan,
            invalidation_plan=invalidation_plan,
        )

        priority = self._resolve_priority(score=score, confidence=confidence)
        strategy_cfg = self._strategy_cfg

        signal = StrategySignal(
            symbol=context.symbol,
            side=side,
            strategy_name=self.strategy_name,
            category=self.category,
            timeframe=context.timeframe,
            setup_type=SetupType.BREAKOUT,
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
            regime=self._resolve_regime(context),
            metadata={
                "snapshot_timestamp": snapshot.timestamp.isoformat(),
                "snapshot_timeframe": snapshot.timeframe,
                "current_price": current_price,
                "above_liquidity_score": snapshot.above_liquidity_score,
                "below_liquidity_score": snapshot.below_liquidity_score,
                "liquidity_pressure_score": snapshot.liquidity_pressure_score,
                "bias": snapshot.bias.value,
                "magnet_score_up": self._magnet_score_up(snapshot),
                "magnet_score_down": self._magnet_score_down(snapshot),
                "sweep_risk_up": self._sweep_risk_up(snapshot),
                "sweep_risk_down": self._sweep_risk_down(snapshot),
                "upside_edge": self._upside_edge(snapshot),
                "downside_edge": self._downside_edge(snapshot),
                "target_price": self._reference_price(target),
                "target_type": self._target_type(target),
                "target_confidence": self._target_confidence(target),
                "levels_count": len(snapshot.active_levels),
                "zones_count": len(snapshot.zones),
                "clusters_count": len(snapshot.stop_clusters),
                "strategy_weight": strategy_cfg.weight if strategy_cfg is not None else 1.0,
                "strategy_semantics": "liquidity_magnet_follow_through",
            },
        )

        signal.add_reason(self._build_primary_reason(snapshot, side))
        signal.add_reason(self._build_target_reason(target))
        signal.add_reason(
            f"Liquidity pressure score = {snapshot.liquidity_pressure_score:.3f}"
        )

        if snapshot.signal is not None and snapshot.signal.explanation:
            signal.add_reason(snapshot.signal.explanation)

        for confirmation in self._build_confirmations(
            snapshot=snapshot,
            side=side,
            current_price=current_price,
        ):
            signal.add_confirmation(confirmation)

        signal.add_source_feature("liquidity_map_snapshot")
        signal.add_source_feature("liquidity")
        signal.add_source_feature("liquidity.signal")
        signal.add_source_feature("liquidity.magnet_follow_through")

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
        notes = ["Enter in direction of dominant liquidity magnet/sweep pressure"]

        if target is not None:
            notes.append(
                f"Primary target liquidity at {self._reference_price(target):.6f}"
            )

        entry_type = self.config.builders.default_entry_type or EntryType.MARKET

        return EntryPlan(
            entry_type=entry_type,
            price=current_price if entry_type == EntryType.LIMIT else None,
            timeout_seconds=self.config.runtime.max_signal_age_seconds,
            max_slippage_bps=8.0 if side in {SignalSide.LONG, SignalSide.SHORT} else None,
            confirmation_required=False,
            notes=notes,
            metadata={
                "entry_logic": "liquidity_sweep_follow_through",
                "target_price": self._reference_price(target),
            },
        )

    def _build_exit_plan(
        self,
        side: SignalSide,
        current_price: float,
        target: LiquidityLevel | StopCluster | None,
        invalidation_level: LiquidityLevel | StopCluster | None,
        snapshot: LiquidityMapSnapshot,
    ) -> ExitPlan:
        target_price = self._reference_price(target) if target is not None else None
        stop_price = self._resolve_stop_price(side, current_price, invalidation_level)

        tp_levels: list[TargetPlan] = []

        if target_price is not None and target_price > 0:
            tp_levels.append(
                TargetPlan(
                    price=target_price,
                    size_fraction=(
                        0.70
                        if self.config.builders.enable_partial_take_profit
                        else 1.0
                    ),
                    rr=self._compute_rr(
                        current_price=current_price,
                        stop_price=stop_price,
                        target_price=target_price,
                        side=side,
                    ),
                    label="primary_liquidity_target",
                    metadata={"source": "nearest_directional_liquidity"},
                )
            )

        extended_target = self._find_extended_target(
            snapshot=snapshot,
            current_price=current_price,
            side=side,
            exclude=target,
        )
        extended_target_price = (
            self._reference_price(extended_target)
            if extended_target is not None
            else None
        )

        if (
            self.config.builders.enable_partial_take_profit
            and extended_target_price is not None
            and extended_target_price > 0
        ):
            tp_levels.append(
                TargetPlan(
                    price=extended_target_price,
                    size_fraction=0.30,
                    rr=self._compute_rr(
                        current_price=current_price,
                        stop_price=stop_price,
                        target_price=extended_target_price,
                        side=side,
                    ),
                    label="secondary_liquidity_target",
                    metadata={"source": "extended_liquidity_target"},
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
            metadata={"exit_logic": "liquidity_target_then_invalidation"},
        )

    def _build_invalidation_plan(
        self,
        side: SignalSide,
        current_price: float,
        invalidation_level: LiquidityLevel | StopCluster | None,
    ) -> InvalidationPlan:
        price = self._resolve_stop_price(side, current_price, invalidation_level)

        reason = None
        if self.config.builders.require_invalidation:
            reason = (
                "Downside liquidity reclaimed against long follow-through thesis"
                if side == SignalSide.LONG
                else "Upside liquidity reclaimed against short follow-through thesis"
            )

        return InvalidationPlan(
            price=price,
            reason=reason,
            timeout_seconds=max(self.config.runtime.max_signal_age_seconds, 30),
            conditions=[
                "signal_age_expired",
                "opposite_liquidity_pressure_domination",
            ],
            metadata={"invalidation_source": "opposite_side_liquidity"},
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
                "Generated by LiquiditySweepStrategy",
                "Execution should validate risk and portfolio constraints before submit",
            ],
            metadata={
                "strategy_name": self.strategy_name,
                "category": self.category.value,
            },
        )

    def _build_primary_reason(
        self,
        snapshot: LiquidityMapSnapshot,
        side: SignalSide,
    ) -> str:
        if side == SignalSide.LONG:
            return (
                "Upside liquidity sweep path dominates: "
                f"magnet_up={self._magnet_score_up(snapshot):.3f}, "
                f"sweep_up={self._sweep_risk_up(snapshot):.3f}, "
                f"above_liquidity={snapshot.above_liquidity_score:.3f}"
            )

        return (
            "Downside liquidity sweep path dominates: "
            f"magnet_down={self._magnet_score_down(snapshot):.3f}, "
            f"sweep_down={self._sweep_risk_down(snapshot):.3f}, "
            f"below_liquidity={snapshot.below_liquidity_score:.3f}"
        )

    def _build_target_reason(
        self,
        target: LiquidityLevel | StopCluster | None,
    ) -> str:
        if target is None:
            return (
                "No explicit nearest liquidity target found; signal based on "
                "aggregate sweep pressure"
            )

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
        side: SignalSide,
        current_price: float,
    ) -> list[str]:
        confirmations: list[str] = []

        if side == SignalSide.LONG:
            if snapshot.bias == LiquidityBias.UP:
                confirmations.append("Liquidity bias aligned up")

            if (
                snapshot.strongest_cluster_above is not None
                and snapshot.strongest_cluster_above.confidence >= 0.65
                and not snapshot.strongest_cluster_above.is_swept()
            ):
                confirmations.append("Strong active stop cluster above current price")

            if self._magnet_score_up(snapshot) >= 0.70:
                confirmations.append("Strong upside liquidity magnet")

            if self._sweep_risk_up(snapshot) >= 0.65:
                confirmations.append("Elevated upside sweep probability")

            if self._has_high_quality_zone(
                zones=snapshot.zones,
                side=LiquiditySide.BUY_SIDE,
                current_price=current_price,
            ):
                confirmations.append("High-quality buy-side liquidity zone ahead")

        elif side == SignalSide.SHORT:
            if snapshot.bias == LiquidityBias.DOWN:
                confirmations.append("Liquidity bias aligned down")

            if (
                snapshot.strongest_cluster_below is not None
                and snapshot.strongest_cluster_below.confidence >= 0.65
                and not snapshot.strongest_cluster_below.is_swept()
            ):
                confirmations.append("Strong active stop cluster below current price")

            if self._magnet_score_down(snapshot) >= 0.70:
                confirmations.append("Strong downside liquidity magnet")

            if self._sweep_risk_down(snapshot) >= 0.65:
                confirmations.append("Elevated downside sweep probability")

            if self._has_high_quality_zone(
                zones=snapshot.zones,
                side=LiquiditySide.SELL_SIDE,
                current_price=current_price,
            ):
                confirmations.append("High-quality sell-side liquidity zone ahead")

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

    def _invalidation_level_for_side(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
    ) -> LiquidityLevel | StopCluster | None:
        if side == SignalSide.LONG:
            candidates = [
                item
                for item in [
                    snapshot.nearest_below_level,
                    snapshot.strongest_cluster_below,
                    *self._collect_opposite_liquidity(snapshot, current_price, side),
                ]
                if item is not None and self._reference_price(item) < current_price
            ]
            return max(candidates, key=self._reference_price) if candidates else None

        if side == SignalSide.SHORT:
            candidates = [
                item
                for item in [
                    snapshot.nearest_above_level,
                    snapshot.strongest_cluster_above,
                    *self._collect_opposite_liquidity(snapshot, current_price, side),
                ]
                if item is not None and self._reference_price(item) > current_price
            ]
            return min(candidates, key=self._reference_price) if candidates else None

        return None

    def _collect_opposite_liquidity(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
    ) -> list[LiquidityLevel | StopCluster]:
        candidates: list[LiquidityLevel | StopCluster] = []

        if side == SignalSide.LONG:
            candidates.extend(
                level
                for level in snapshot.active_levels
                if level.price < current_price
            )
            candidates.extend(
                cluster
                for cluster in snapshot.stop_clusters
                if cluster.center_price < current_price
            )

        elif side == SignalSide.SHORT:
            candidates.extend(
                level
                for level in snapshot.active_levels
                if level.price > current_price
            )
            candidates.extend(
                cluster
                for cluster in snapshot.stop_clusters
                if cluster.center_price > current_price
            )

        return candidates

    def _resolve_stop_price(
        self,
        side: SignalSide,
        current_price: float,
        invalidation_level: LiquidityLevel | StopCluster | None,
    ) -> float | None:
        if current_price <= 0:
            return None

        ref = (
            self._reference_price(invalidation_level)
            if invalidation_level is not None
            else None
        )

        if side == SignalSide.LONG:
            if ref is not None and 0 < ref < current_price:
                return ref * self.LONG_STOP_OFFSET
            return current_price * (1.0 - self.FALLBACK_STOP_PCT)

        if side == SignalSide.SHORT:
            if ref is not None and ref > current_price:
                return ref * self.SHORT_STOP_OFFSET
            return current_price * (1.0 + self.FALLBACK_STOP_PCT)

        return None

    def _find_extended_target(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
        exclude: LiquidityLevel | StopCluster | None = None,
    ) -> LiquidityLevel | StopCluster | None:
        candidates = self._collect_directional_targets(
            snapshot=snapshot,
            current_price=current_price,
            side=side,
        )

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

    def _resolve_priority(self, score: float, confidence: float) -> SignalPriority:
        if (
            score >= self.CRITICAL_PRIORITY_SCORE
            and confidence >= self.CRITICAL_PRIORITY_CONFIDENCE
        ):
            return SignalPriority.CRITICAL

        if score >= self.HIGH_PRIORITY_SCORE or confidence >= self.HIGH_PRIORITY_CONFIDENCE:
            return SignalPriority.HIGH

        return SignalPriority.MEDIUM

    def _resolve_regime(self, context: StrategyContext):
        if context.regime is not None:
            return context.regime.regime

        metadata = getattr(context, "metadata", None)
        regime = getattr(metadata, "regime", None)
        if regime is not None:
            return regime

        return self._unknown_regime()

    def _target_type(self, target: LiquidityLevel | StopCluster | None) -> str | None:
        if target is None:
            return None

        if isinstance(target, StopCluster):
            return "stop_cluster"

        return target.level_type.value

    def _target_confidence(self, target: LiquidityLevel | StopCluster | None) -> float | None:
        if target is None:
            return None

        return self._clamp01(target.confidence)

    def _reference_price(self, item: LiquidityLevel | StopCluster | None) -> float:
        if item is None:
            return 0.0

        if isinstance(item, StopCluster):
            return float(item.center_price)

        return float(item.price)

    @staticmethod
    def _clamp01(value: Any) -> float:
        try:
            resolved = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(0.0, min(resolved, 1.0))

    def _unknown_regime(self):
        from strategy.enums import MarketRegime

        return MarketRegime.UNKNOWN