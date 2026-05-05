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
    FilterDecision,
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
    Стратегія торгівлі в напрямку найбільш імовірного liquidity sweep / magnet move.

    Стратегія не виконує угоди напряму, а лише формує StrategySignal.
    """

    @property
    def strategy_name(self) -> str:
        return "liquidity_sweep_strategy"

    def evaluate(self, context: StrategyContext) -> StrategySignal | None:
        self.validate_context(context)

        if not self.is_enabled():
            self.log_debug(
                "LiquiditySweepStrategy skipped: disabled",
                symbol=context.symbol,
                timeframe=str(context.timeframe),
            )
            return None

        snapshot = self._extract_snapshot(context)
        if snapshot is None:
            self.log_debug(
                "LiquiditySweepStrategy skipped: liquidity snapshot not found",
                symbol=context.symbol,
                timeframe=str(context.timeframe),
            )
            return None

        if not self._base_context_is_valid(context, snapshot):
            return None

        current_price = self._resolve_current_price(context, snapshot)
        if current_price is None or current_price <= 0:
            self.log_warning(
                "LiquiditySweepStrategy skipped: current price unavailable",
                symbol=context.symbol,
                timeframe=str(context.timeframe),
            )
            return None

        filters = self._run_pre_filters(context, snapshot, current_price)
        if any(result.blocked for result in filters):
            self.log_debug(
                "LiquiditySweepStrategy blocked by filters",
                symbol=context.symbol,
                timeframe=str(snapshot.timeframe),
                filters=[f"{f.name}:{f.decision.value}" for f in filters],
            )
            return None

        side = self._infer_side(snapshot)
        if side == SignalSide.UNKNOWN:
            self.log_debug(
                "LiquiditySweepStrategy skipped: no directional sweep edge",
                symbol=context.symbol,
                timeframe=str(snapshot.timeframe),
            )
            return None

        confidence = self._compute_confidence(snapshot, current_price, side)
        score = self._compute_score(snapshot, current_price, side, confidence)

        runtime = self._runtime
        if confidence < runtime.min_confidence or score < runtime.min_score:
            self.log_debug(
                "LiquiditySweepStrategy skipped: below thresholds",
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
        results: list[FilterResult] = []

        if context.portfolio is not None and context.symbol in context.portfolio.blocked_symbols:
            results.append(
                FilterResult(
                    name="portfolio_blocked_symbol",
                    decision=FilterDecision.BLOCK,
                    reason=f"Symbol {context.symbol} is blocked by portfolio snapshot",
                )
            )

        if self.config.filters.enable_spread_filter and context.price is not None:
            spread_bps = context.price.spread_bps
            if spread_bps is not None and spread_bps > self.config.filters.max_spread_bps:
                results.append(
                    FilterResult(
                        name="spread_filter",
                        decision=FilterDecision.BLOCK,
                        reason=f"Spread too high: {spread_bps:.2f} bps",
                    )
                )
            else:
                results.append(
                    FilterResult(
                        name="spread_filter",
                        decision=FilterDecision.PASS,
                        reason="Spread within threshold",
                    )
                )

        if self.config.filters.enable_liquidity_filter:
            side = self._infer_side(snapshot)
            liquidity_score = (
                max(snapshot.above_liquidity_score, snapshot.below_liquidity_score)
                if side == SignalSide.UNKNOWN
                else (
                    snapshot.above_liquidity_score
                    if side == SignalSide.LONG
                    else snapshot.below_liquidity_score
                )
            )

            if liquidity_score < self.config.filters.min_liquidity_score:
                results.append(
                    FilterResult(
                        name="liquidity_strength_filter",
                        decision=FilterDecision.BLOCK,
                        reason=f"Liquidity score too low: {liquidity_score:.4f}",
                    )
                )
            else:
                results.append(
                    FilterResult(
                        name="liquidity_strength_filter",
                        decision=FilterDecision.PASS,
                        reason=f"Liquidity score OK: {liquidity_score:.4f}",
                    )
                )

        if not snapshot.has_levels():
            results.append(
                FilterResult(
                    name="liquidity_snapshot_presence",
                    decision=FilterDecision.BLOCK,
                    reason="Liquidity snapshot has no active levels or clusters",
                )
            )
        else:
            results.append(
                FilterResult(
                    name="liquidity_snapshot_presence",
                    decision=FilterDecision.PASS,
                    reason="Liquidity snapshot contains levels/clusters",
                )
            )

        if current_price <= 0:
            results.append(
                FilterResult(
                    name="price_validation",
                    decision=FilterDecision.BLOCK,
                    reason="Current price must be positive",
                )
            )
        else:
            results.append(
                FilterResult(
                    name="price_validation",
                    decision=FilterDecision.PASS,
                    reason="Current price is valid",
                )
            )

        return results

    def _infer_side(self, snapshot: LiquidityMapSnapshot) -> SignalSide:
        up_edge = self._upside_edge(snapshot)
        down_edge = self._downside_edge(snapshot)
        delta = up_edge - down_edge

        if snapshot.bias == LiquidityBias.UP and up_edge >= 0.45:
            return SignalSide.LONG

        if snapshot.bias == LiquidityBias.DOWN and down_edge >= 0.45:
            return SignalSide.SHORT

        if delta >= 0.12:
            return SignalSide.LONG

        if delta <= -0.12:
            return SignalSide.SHORT

        return SignalSide.UNKNOWN

    def _compute_confidence(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
    ) -> float:
        if side == SignalSide.LONG:
            base = (
                0.32 * self._magnet_score_up(snapshot)
                + 0.28 * self._sweep_risk_up(snapshot)
                + 0.20 * snapshot.above_liquidity_score
                + 0.20 * max(snapshot.liquidity_pressure_score, 0.0)
            )
            target = snapshot.nearest_above_level
            bonus = self._target_quality_bonus(target, current_price)
            zone_bonus = self._zone_alignment_bonus(
                snapshot.zones,
                LiquiditySide.BUY_SIDE,
                current_price,
            )

        elif side == SignalSide.SHORT:
            base = (
                0.32 * self._magnet_score_down(snapshot)
                + 0.28 * self._sweep_risk_down(snapshot)
                + 0.20 * snapshot.below_liquidity_score
                + 0.20 * max(-snapshot.liquidity_pressure_score, 0.0)
            )
            target = snapshot.nearest_below_level
            bonus = self._target_quality_bonus(target, current_price)
            zone_bonus = self._zone_alignment_bonus(
                snapshot.zones,
                LiquiditySide.SELL_SIDE,
                current_price,
            )

        else:
            return 0.0

        return max(0.0, min(1.0, base + bonus + zone_bonus))

    def _compute_score(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
        confidence: float,
    ) -> float:
        if side == SignalSide.LONG:
            directional_edge = self._upside_edge(snapshot) - self._downside_edge(snapshot)
            target = snapshot.nearest_above_level
        else:
            directional_edge = self._downside_edge(snapshot) - self._upside_edge(snapshot)
            target = snapshot.nearest_below_level

        return max(
            0.0,
            1.30 * confidence
            + 0.80 * max(directional_edge, 0.0)
            + 0.40 * self._target_distance_score(target, current_price),
        )

    def _upside_edge(self, snapshot: LiquidityMapSnapshot) -> float:
        return (
            0.40 * self._magnet_score_up(snapshot)
            + 0.35 * self._sweep_risk_up(snapshot)
            + 0.25 * snapshot.above_liquidity_score
        )

    def _downside_edge(self, snapshot: LiquidityMapSnapshot) -> float:
        return (
            0.40 * self._magnet_score_down(snapshot)
            + 0.35 * self._sweep_risk_down(snapshot)
            + 0.25 * snapshot.below_liquidity_score
        )

    def _magnet_score_up(self, snapshot: LiquidityMapSnapshot) -> float:
        return snapshot.signal.magnet_score_up if snapshot.signal is not None else 0.0

    def _magnet_score_down(self, snapshot: LiquidityMapSnapshot) -> float:
        return snapshot.signal.magnet_score_down if snapshot.signal is not None else 0.0

    def _sweep_risk_up(self, snapshot: LiquidityMapSnapshot) -> float:
        return snapshot.signal.sweep_risk_up if snapshot.signal is not None else 0.0

    def _sweep_risk_down(self, snapshot: LiquidityMapSnapshot) -> float:
        return snapshot.signal.sweep_risk_down if snapshot.signal is not None else 0.0

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
            bonus += 0.05 * min(max(target.confidence, 0.0), 1.0)
            if target.strength.value in {"high", "extreme"}:
                bonus += 0.04

        if isinstance(target, LiquidityLevel):
            bonus += 0.04 * min(max(target.confidence, 0.0), 1.0)

            if target.level_type in {
                LiquidityLevelType.EQUAL_HIGHS,
                LiquidityLevelType.EQUAL_LOWS,
            }:
                bonus += 0.03

            if target.sweep_status == SweepStatus.PARTIALLY_SWEPT:
                bonus -= 0.03
            elif target.sweep_status == SweepStatus.SWEPT:
                bonus -= 0.07

        return bonus

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
        return 0.05 * min(max(best.score, 0.0), 1.0)

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
            return 1.0
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
        confidence: float,
        score: float,
        filters: list[FilterResult],
    ) -> StrategySignal:
        target = snapshot.nearest_above_level if side == SignalSide.LONG else snapshot.nearest_below_level
        invalidation_level = snapshot.nearest_below_level if side == SignalSide.LONG else snapshot.nearest_above_level

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

        strategy_cfg = self._strategy_cfg

        priority = SignalPriority.MEDIUM
        if score >= 1.8 or confidence >= 0.85:
            priority = SignalPriority.HIGH
        if score >= 2.2 and confidence >= 0.90:
            priority = SignalPriority.CRITICAL

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
            regime=(
                context.regime.regime
                if context.regime is not None
                else context.metadata.regime
                if context.metadata
                else self._unknown_regime()
            ),
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
                "levels_count": len(snapshot.active_levels),
                "zones_count": len(snapshot.zones),
                "clusters_count": len(snapshot.stop_clusters),
                "strategy_weight": strategy_cfg.weight if strategy_cfg is not None else 1.0,
            },
        )

        signal.add_reason(self._build_primary_reason(snapshot, side))
        signal.add_reason(self._build_target_reason(target))
        signal.add_reason(f"Liquidity pressure score = {snapshot.liquidity_pressure_score:.3f}")

        if snapshot.signal is not None and snapshot.signal.explanation:
            signal.add_reason(snapshot.signal.explanation)

        for confirmation in self._build_confirmations(snapshot, side, current_price):
            signal.add_confirmation(confirmation)

        signal.add_source_feature("liquidity_map_snapshot")
        signal.add_source_feature("liquidity")
        signal.add_source_feature("liquidity.signal")

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
            notes.append(f"Primary target liquidity at {self._reference_price(target):.6f}")

        entry_type = self.config.builders.default_entry_type or EntryType.MARKET

        return EntryPlan(
            entry_type=entry_type,
            price=current_price if entry_type == EntryType.LIMIT else None,
            timeout_seconds=self.config.runtime.max_signal_age_seconds,
            max_slippage_bps=8.0 if side in {SignalSide.LONG, SignalSide.SHORT} else None,
            confirmation_required=False,
            notes=notes,
            metadata={"entry_logic": "liquidity_sweep_follow_through"},
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
                    size_fraction=0.70
                    if self.config.builders.enable_partial_take_profit
                    else 1.0,
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
            snapshot,
            current_price,
            side,
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
                "Downside liquidity reclaimed against long thesis"
                if side == SignalSide.LONG
                else "Upside liquidity reclaimed against short thesis"
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
            return "No explicit nearest liquidity target found; signal based on aggregate sweep pressure"

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
            ):
                confirmations.append("Strong stop cluster above current price")

            if self._magnet_score_up(snapshot) >= 0.70:
                confirmations.append("Strong upside liquidity magnet")

            if self._sweep_risk_up(snapshot) >= 0.65:
                confirmations.append("Elevated upside sweep probability")

            if self._has_high_quality_zone(
                snapshot.zones,
                LiquiditySide.BUY_SIDE,
                current_price,
            ):
                confirmations.append("High-quality buy-side liquidity zone ahead")

        elif side == SignalSide.SHORT:
            if snapshot.bias == LiquidityBias.DOWN:
                confirmations.append("Liquidity bias aligned down")

            if (
                snapshot.strongest_cluster_below is not None
                and snapshot.strongest_cluster_below.confidence >= 0.65
            ):
                confirmations.append("Strong stop cluster below current price")

            if self._magnet_score_down(snapshot) >= 0.70:
                confirmations.append("Strong downside liquidity magnet")

            if self._sweep_risk_down(snapshot) >= 0.65:
                confirmations.append("Elevated downside sweep probability")

            if self._has_high_quality_zone(
                snapshot.zones,
                LiquiditySide.SELL_SIDE,
                current_price,
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

    def _resolve_stop_price(
        self,
        side: SignalSide,
        current_price: float,
        invalidation_level: LiquidityLevel | StopCluster | None,
    ) -> float | None:
        if current_price <= 0:
            return None

        ref = self._reference_price(invalidation_level) if invalidation_level is not None else None
        fallback_pct = 0.0040

        if side == SignalSide.LONG:
            if ref is not None and ref < current_price:
                return ref * 0.9985
            return current_price * (1.0 - fallback_pct)

        if side == SignalSide.SHORT:
            if ref is not None and ref > current_price:
                return ref * 1.0015
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