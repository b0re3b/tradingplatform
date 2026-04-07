from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from core.logger import get_logger

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

from strategy.base import (
    ContextAwareComponent,
    EventEmitterMixin,
    NamedEntityMixin,
    PrioritizedMixin,
    StrategyComponent,
)
from strategy.config import StrategyConfig, StrategyDefinitionConfig
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
    StrategyCategory,
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


class EqualHighLowStrategy(
    StrategyComponent,
    ContextAwareComponent,
    EventEmitterMixin,
    NamedEntityMixin,
    PrioritizedMixin,
):
    """
    Strategy для торгівлі від equal highs / equal lows.

    Основна логіка:
    - шукає найсильніший equal highs або equal lows у snapshot;
    - оцінює його релевантність відносно current price;
    - формує directional signal:
        * LONG від equal lows
        * SHORT від equal highs
    - враховує sweep / partial sweep / confidence / touches / reactions;
    - будує execution-ready signal.
    """

    SIGNAL_TOPIC = "signal.generated"

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: Any | None = None,
        logger: Any | None = None,
    ) -> None:
        super().__init__(
            config=config,
            event_bus=event_bus,
            logger=logger or get_logger(__name__, service_name="strategy"),
        )
        self.validate_config()
        self._last_emitted_at: dict[str, datetime] = {}

    @property
    def strategy_name(self) -> str:
        return "equal_high_low_strategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.LIQUIDITY

    @property
    def _strategy_cfg(self) -> StrategyDefinitionConfig | None:
        return self.config.get_strategy(self.strategy_name)

    @property
    def priority(self) -> int:
        strategy_cfg = self._strategy_cfg
        return strategy_cfg.priority if strategy_cfg is not None else 100

    def is_enabled(self) -> bool:
        strategy_cfg = self._strategy_cfg
        if strategy_cfg is None:
            return True
        return strategy_cfg.runtime.enabled

    def evaluate(self, context: StrategyContext) -> StrategySignal | None:
        self.validate_context(context)

        if not self.is_enabled():
            self.log_debug(
                "EqualHighLowStrategy skipped: disabled",
                symbol=context.symbol,
                timeframe=str(context.timeframe),
            )
            return None

        snapshot = self._extract_snapshot(context)
        if snapshot is None:
            self.log_debug(
                "EqualHighLowStrategy skipped: liquidity snapshot not found",
                symbol=context.symbol,
                timeframe=str(context.timeframe),
            )
            return None

        if not self._is_symbol_allowed(context.symbol):
            return None

        if not self._is_timeframe_allowed(context.timeframe):
            return None

        if not self._is_regime_allowed(context):
            return None

        if self._snapshot_is_stale(context, snapshot):
            self.log_debug(
                "EqualHighLowStrategy skipped: stale liquidity snapshot",
                symbol=context.symbol,
                timeframe=snapshot.timeframe,
                snapshot_ts=snapshot.timestamp.isoformat(),
                context_ts=context.timestamp.isoformat(),
            )
            return None

        current_price = self._resolve_current_price(context, snapshot)
        if current_price is None or current_price <= 0:
            self.log_warning(
                "EqualHighLowStrategy skipped: current price unavailable",
                symbol=context.symbol,
                timeframe=str(context.timeframe),
            )
            return None

        filters = self._run_pre_filters(context, snapshot, current_price)
        if any(result.blocked for result in filters):
            self.log_debug(
                "EqualHighLowStrategy blocked by filters",
                symbol=context.symbol,
                timeframe=snapshot.timeframe,
                filters=[f"{f.name}:{f.decision.value}" for f in filters],
            )
            return None

        candidate = self._find_best_equal_level_candidate(snapshot, current_price)
        if candidate is None:
            self.log_debug(
                "EqualHighLowStrategy skipped: no valid equal highs/lows candidate",
                symbol=context.symbol,
                timeframe=snapshot.timeframe,
            )
            return None

        side = candidate["side"]
        confidence = self._compute_confidence(snapshot, current_price, candidate)
        score = self._compute_score(snapshot, current_price, candidate, confidence)

        strategy_cfg = self._strategy_cfg
        runtime = strategy_cfg.runtime if strategy_cfg is not None else self.config.runtime

        if confidence < runtime.min_confidence or score < runtime.min_score:
            self.log_debug(
                "EqualHighLowStrategy skipped: below thresholds",
                symbol=context.symbol,
                timeframe=snapshot.timeframe,
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

        if self._is_on_emit_cooldown(signal.symbol, context.timestamp):
            self.log_debug(
                "EqualHighLowStrategy emit suppressed by cooldown",
                symbol=signal.symbol,
                timeframe=signal.timeframe.value,
            )
            return None

        payload = {
            "symbol": signal.symbol,
            "strategy_name": signal.strategy_name,
            "category": signal.category.value,
            "timeframe": signal.timeframe.value,
            "side": signal.side.value,
            "score": signal.score,
            "confidence": signal.confidence,
            "status": signal.status.value,
            "signal": signal,
        }

        await self.emit_event(
            self.SIGNAL_TOPIC,
            payload,
            source=self.strategy_name,
            **emit_kwargs,
        )
        self._last_emitted_at[signal.symbol] = context.timestamp

        self.log_info(
            "Equal highs/lows signal emitted",
            symbol=signal.symbol,
            timeframe=signal.timeframe.value,
            side=signal.side.value,
            confidence=signal.confidence,
            score=signal.score,
        )
        return signal

    def _extract_snapshot(self, context: StrategyContext) -> LiquidityMapSnapshot | None:
        domain_candidates = [
            context.liquidity.get("snapshot"),
            context.liquidity.get("liquidity_map_snapshot"),
            context.liquidity.get("map_snapshot"),
            context.liquidity.get("last_snapshot"),
        ]
        feature_candidates = [
            context.get_feature("liquidity_map_snapshot"),
            context.get_feature("liquidity.snapshot"),
            context.get_feature("liquidity_snapshot"),
            context.get_feature("liquidity.map.snapshot"),
        ]

        for candidate in [*domain_candidates, *feature_candidates]:
            if isinstance(candidate, LiquidityMapSnapshot):
                return candidate

        return None

    def _is_symbol_allowed(self, symbol: str) -> bool:
        strategy_cfg = self._strategy_cfg
        runtime = strategy_cfg.runtime if strategy_cfg is not None else self.config.runtime
        return not runtime.symbols or symbol in runtime.symbols

    def _is_timeframe_allowed(self, timeframe) -> bool:
        strategy_cfg = self._strategy_cfg
        runtime = strategy_cfg.runtime if strategy_cfg is not None else self.config.runtime
        return not runtime.timeframes or timeframe in runtime.timeframes

    def _is_regime_allowed(self, context: StrategyContext) -> bool:
        strategy_cfg = self._strategy_cfg
        runtime = strategy_cfg.runtime if strategy_cfg is not None else self.config.runtime

        if not runtime.allowed_regimes:
            return True

        regime = context.regime.regime if context.regime is not None else None
        if regime is None:
            return True

        return regime in runtime.allowed_regimes

    def _snapshot_is_stale(self, context: StrategyContext, snapshot: LiquidityMapSnapshot) -> bool:
        feature = context.get_feature_snapshot("liquidity_map_snapshot")
        if feature is not None:
            return feature.is_stale(context.timestamp)

        ttl = self.config.freshness.get_ttl("liquidity_map_snapshot")
        age_seconds = abs((context.timestamp - snapshot.timestamp).total_seconds())
        return age_seconds > ttl

    def _resolve_current_price(
        self,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot,
    ) -> float | None:
        if context.price is not None:
            if context.price.mid_price is not None:
                return float(context.price.mid_price)
            if context.price.last_price is not None:
                return float(context.price.last_price)

        return float(snapshot.current_price) if snapshot.current_price > 0 else None

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
            strongest_liquidity = max(
                snapshot.above_liquidity_score,
                snapshot.below_liquidity_score,
            )
            if strongest_liquidity < self.config.filters.min_liquidity_score:
                results.append(
                    FilterResult(
                        name="liquidity_strength_filter",
                        decision=FilterDecision.BLOCK,
                        reason=f"Liquidity score too low: {strongest_liquidity:.4f}",
                    )
                )
            else:
                results.append(
                    FilterResult(
                        name="liquidity_strength_filter",
                        decision=FilterDecision.PASS,
                        reason=f"Liquidity score OK: {strongest_liquidity:.4f}",
                    )
                )

        if not snapshot.equal_levels:
            results.append(
                FilterResult(
                    name="equal_levels_presence",
                    decision=FilterDecision.BLOCK,
                    reason="Liquidity snapshot has no equal highs/lows",
                )
            )
        else:
            results.append(
                FilterResult(
                    name="equal_levels_presence",
                    decision=FilterDecision.PASS,
                    reason="Liquidity snapshot contains equal highs/lows",
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

    def _find_best_equal_level_candidate(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> dict[str, Any] | None:
        long_candidates = [
            level
            for level in snapshot.equal_levels
            if level.side == LiquiditySide.SELL_SIDE and level.price <= current_price * 1.003
        ]
        short_candidates = [
            level
            for level in snapshot.equal_levels
            if level.side == LiquiditySide.BUY_SIDE and level.price >= current_price * 0.997
        ]

        best_long = self._pick_best_equal_level(long_candidates, current_price, SignalSide.LONG)
        best_short = self._pick_best_equal_level(short_candidates, current_price, SignalSide.SHORT)

        if best_long is None and best_short is None:
            return None
        if best_long is None:
            return best_short
        if best_short is None:
            return best_long

        if float(best_long["edge"]) >= float(best_short["edge"]):
            return best_long
        return best_short

    def _pick_best_equal_level(
        self,
        levels: list[EqualLevel],
        current_price: float,
        side: SignalSide,
    ) -> dict[str, Any] | None:
        if not levels:
            return None

        ranked: list[dict[str, Any]] = []

        for level in levels:
            if level.level_type not in {
                LiquidityLevelType.EQUAL_HIGHS,
                LiquidityLevelType.EQUAL_LOWS,
            }:
                continue

            edge = self._equal_level_edge(level, current_price, side)
            if edge <= 0:
                continue

            target = None
            ranked.append(
                {
                    "side": side,
                    "level": level,
                    "edge": edge,
                    "target": target,
                }
            )

        if not ranked:
            return None

        return max(ranked, key=lambda item: float(item["edge"]))

    def _equal_level_edge(
        self,
        level: EqualLevel,
        current_price: float,
        side: SignalSide,
    ) -> float:
        if current_price <= 0 or level.price <= 0:
            return 0.0

        distance_pct = abs(level.price - current_price) / current_price

        confidence_part = min(max(level.confidence, 0.0), 1.0) * 0.40
        touches_part = min(level.touches_count / 6.0, 1.0) * 0.20
        reactions_part = min(level.reaction_count / 4.0, 1.0) * 0.12

        compactness_part = 0.0
        if level.cluster_low is not None and level.cluster_high is not None and level.price > 0:
            width_pct = abs(level.cluster_high - level.cluster_low) / level.price
            if width_pct <= 0.0008:
                compactness_part = 0.16
            elif width_pct <= 0.0015:
                compactness_part = 0.10
            elif width_pct <= 0.0030:
                compactness_part = 0.05

        distance_part = 0.0
        if distance_pct <= 0.0015:
            distance_part = 0.22
        elif distance_pct <= 0.0040:
            distance_part = 0.18
        elif distance_pct <= 0.0100:
            distance_part = 0.12
        elif distance_pct <= 0.0200:
            distance_part = 0.06

        sweep_part = 0.0
        if level.sweep_status == SweepStatus.NOT_SWEPT:
            sweep_part = 0.10
        elif level.sweep_status == SweepStatus.PARTIALLY_SWEPT:
            sweep_part = 0.14
        elif level.sweep_status == SweepStatus.SWEPT:
            sweep_part = 0.18

        directional_part = 0.0
        if side == SignalSide.LONG and level.side == LiquiditySide.SELL_SIDE:
            directional_part = 0.10
        elif side == SignalSide.SHORT and level.side == LiquiditySide.BUY_SIDE:
            directional_part = 0.10

        return (
            confidence_part
            + touches_part
            + reactions_part
            + compactness_part
            + distance_part
            + sweep_part
            + directional_part
        )

    def _compute_confidence(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        candidate: dict[str, Any],
    ) -> float:
        level: EqualLevel = candidate["level"]
        side: SignalSide = candidate["side"]
        edge = float(candidate["edge"])

        base = min(edge, 1.0) * 0.62
        base += min(max(level.confidence, 0.0), 1.0) * 0.18
        base += min(level.touches_count / 6.0, 1.0) * 0.08
        base += min(level.reaction_count / 4.0, 1.0) * 0.05

        if level.sweep_status == SweepStatus.SWEPT:
            base += 0.06
        elif level.sweep_status == SweepStatus.PARTIALLY_SWEPT:
            base += 0.04

        if side == SignalSide.LONG and snapshot.bias == LiquidityBias.DOWN:
            base += 0.04
        elif side == SignalSide.SHORT and snapshot.bias == LiquidityBias.UP:
            base += 0.04

        base += self._zone_bonus(snapshot.zones, side, current_price)

        return max(0.0, min(base, 1.0))

    def _compute_score(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        candidate: dict[str, Any],
        confidence: float,
    ) -> float:
        side: SignalSide = candidate["side"]
        edge = float(candidate["edge"])
        level: EqualLevel = candidate["level"]

        structure_distance_score = self._level_distance_score(level, current_price)

        anti_bias_bonus = 0.0
        if side == SignalSide.LONG and snapshot.bias == LiquidityBias.DOWN:
            anti_bias_bonus = 0.25
        elif side == SignalSide.SHORT and snapshot.bias == LiquidityBias.UP:
            anti_bias_bonus = 0.25

        return max(
            0.0,
            (confidence * 1.25) + (edge * 0.85) + (structure_distance_score * 0.40) + anti_bias_bonus,
        )

    def _zone_bonus(
        self,
        zones: list[LiquidityZone],
        side: SignalSide,
        current_price: float,
    ) -> float:
        if not zones:
            return 0.0

        if side == SignalSide.LONG:
            relevant = [
                z for z in zones
                if z.side in {LiquiditySide.BUY_SIDE, LiquiditySide.BOTH} and z.center_price > current_price
            ]
        else:
            relevant = [
                z for z in zones
                if z.side in {LiquiditySide.SELL_SIDE, LiquiditySide.BOTH} and z.center_price < current_price
            ]

        if not relevant:
            return 0.0

        best = max(relevant, key=lambda z: z.score)
        return 0.05 * min(max(best.score, 0.0), 1.0)

    def _level_distance_score(
        self,
        level: EqualLevel,
        current_price: float,
    ) -> float:
        if current_price <= 0 or level.price <= 0:
            return 0.0

        distance_pct = abs(level.price - current_price) / current_price
        if distance_pct <= 0.001:
            return 0.18
        if distance_pct <= 0.003:
            return 0.55
        if distance_pct <= 0.010:
            return 1.00
        if distance_pct <= 0.020:
            return 0.65
        if distance_pct <= 0.040:
            return 0.30
        return 0.08

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
        level: EqualLevel = candidate["level"]
        target = self._find_target(snapshot, current_price, side, anchor_level=level)

        entry_plan = self._build_entry_plan(side=side, current_price=current_price, target=target)
        exit_plan = self._build_exit_plan(
            side=side,
            current_price=current_price,
            target=target,
            anchor_level=level,
            snapshot=snapshot,
        )
        invalidation_plan = self._build_invalidation_plan(
            side=side,
            current_price=current_price,
            anchor_level=level,
        )
        execution_plan = self._build_execution_plan(
            symbol=context.symbol,
            side=side,
            entry_plan=entry_plan,
            exit_plan=exit_plan,
            invalidation_plan=invalidation_plan,
        )

        priority = SignalPriority.MEDIUM
        if score >= 1.70 or confidence >= 0.84:
            priority = SignalPriority.HIGH
        if score >= 2.10 and confidence >= 0.90:
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
                "level_price": level.price,
                "level_type": level.level_type.value,
                "level_side": level.side.value,
                "level_confidence": level.confidence,
                "touches_count": level.touches_count,
                "reaction_count": level.reaction_count,
                "sweep_status": level.sweep_status.value,
                "strategy_weight": self.config.get_strategy_weight(self.strategy_name, default=1.0),
            },
        )

        signal.add_reason(self._build_primary_reason(level, side, current_price))
        signal.add_reason(self._build_target_reason(target))
        signal.add_reason(
            f"Liquidity pressure score = {snapshot.liquidity_pressure_score:.3f}"
        )

        if snapshot.signal is not None and getattr(snapshot.signal, "explanation", None):
            signal.add_reason(snapshot.signal.explanation)

        for confirmation in self._build_confirmations(snapshot, level, side, current_price, target):
            signal.add_confirmation(confirmation)

        signal.add_source_feature("liquidity_map_snapshot")
        signal.add_source_feature("liquidity.equal_levels")
        signal.add_source_feature("liquidity")
        signal.add_source_feature("liquidity.equal_high_low")

        for filter_result in filters:
            signal.add_filter_result(filter_result)

        signal.validate()
        return signal

    def _find_target(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
        anchor_level: EqualLevel,
    ) -> LiquidityLevel | StopCluster | None:
        candidates: list[LiquidityLevel | StopCluster] = []

        if side == SignalSide.LONG:
            for level in snapshot.active_levels:
                if level.price > current_price:
                    candidates.append(level)
            for cluster in snapshot.stop_clusters:
                if cluster.center_price > current_price:
                    candidates.append(cluster)
            candidates.sort(key=self._reference_price)

        elif side == SignalSide.SHORT:
            for level in snapshot.active_levels:
                if level.price < current_price:
                    candidates.append(level)
            for cluster in snapshot.stop_clusters:
                if cluster.center_price < current_price:
                    candidates.append(cluster)
            candidates.sort(key=self._reference_price, reverse=True)

        anchor_price = anchor_level.price
        filtered = [
            item for item in candidates
            if abs(self._reference_price(item) - anchor_price) > 1e-12
        ]
        return filtered[0] if filtered else None

    def _build_entry_plan(
        self,
        side: SignalSide,
        current_price: float,
        target: LiquidityLevel | StopCluster | None,
    ) -> EntryPlan:
        notes = ["Enter from equal highs/lows structure reaction"]
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
            metadata={"entry_logic": "equal_high_low_reaction"},
        )

    def _build_exit_plan(
        self,
        side: SignalSide,
        current_price: float,
        target: LiquidityLevel | StopCluster | None,
        anchor_level: EqualLevel,
        snapshot: LiquidityMapSnapshot,
    ) -> ExitPlan:
        stop_price = self._resolve_stop_price(side, current_price, anchor_level)
        tp_levels: list[TargetPlan] = []

        primary_target_price = self._reference_price(target) if target is not None else None
        if primary_target_price is not None and primary_target_price > 0:
            rr = self._compute_rr(
                current_price=current_price,
                stop_price=stop_price,
                target_price=primary_target_price,
                side=side,
            )
            tp_levels.append(
                TargetPlan(
                    price=primary_target_price,
                    size_fraction=0.70 if self.config.builders.enable_partial_take_profit else 1.0,
                    rr=rr,
                    label="equal_level_primary_target",
                    metadata={"source": "nearest_directional_liquidity"},
                )
            )

        secondary_target = self._find_extended_target(snapshot, current_price, side, exclude=target)
        secondary_price = self._reference_price(secondary_target) if secondary_target is not None else None
        if (
            self.config.builders.enable_partial_take_profit
            and secondary_price is not None
            and secondary_price > 0
        ):
            rr = self._compute_rr(
                current_price=current_price,
                stop_price=stop_price,
                target_price=secondary_price,
                side=side,
            )
            tp_levels.append(
                TargetPlan(
                    price=secondary_price,
                    size_fraction=0.30,
                    rr=rr,
                    label="equal_level_secondary_target",
                    metadata={"source": "extended_directional_liquidity"},
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
            metadata={"exit_logic": "equal_high_low_to_liquidity_target"},
        )

    def _build_invalidation_plan(
        self,
        side: SignalSide,
        current_price: float,
        anchor_level: EqualLevel,
    ) -> InvalidationPlan:
        price = self._resolve_stop_price(side, current_price, anchor_level)

        if self.config.builders.require_invalidation:
            reason = (
                "Equal lows support failed"
                if side == SignalSide.LONG
                else "Equal highs resistance failed"
            )
        else:
            reason = None

        return InvalidationPlan(
            price=price,
            reason=reason,
            timeout_seconds=max(self.config.runtime.max_signal_age_seconds, 30),
            conditions=[
                "signal_age_expired",
                "equal_level_structure_failed",
                "opposite_liquidity_pressure_domination",
            ],
            metadata={"invalidation_source": "equal_high_low_anchor"},
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
                "Generated by EqualHighLowStrategy",
                "Risk manager must validate portfolio/correlation constraints before execution",
            ],
            metadata={
                "strategy_name": self.strategy_name,
                "category": self.category.value,
            },
        )

    def _build_primary_reason(
        self,
        level: EqualLevel,
        side: SignalSide,
        current_price: float,
    ) -> str:
        prefix = "Equal lows reaction -> long setup" if side == SignalSide.LONG else "Equal highs reaction -> short setup"
        return (
            f"{prefix}: level={level.price:.6f}, current_price={current_price:.6f}, "
            f"confidence={level.confidence:.3f}, touches={level.touches_count}, "
            f"reactions={level.reaction_count}, sweep_status={level.sweep_status.value}"
        )

    def _build_target_reason(self, target: LiquidityLevel | StopCluster | None) -> str:
        if target is None:
            return "No explicit target found; signal based on equal highs/lows structure quality"

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
        level: EqualLevel,
        side: SignalSide,
        current_price: float,
        target: LiquidityLevel | StopCluster | None,
    ) -> list[str]:
        confirmations: list[str] = []

        if level.touches_count >= 3:
            confirmations.append("Multiple touches confirm equal level importance")

        if level.reaction_count >= 2:
            confirmations.append("Repeated reactions confirm structure validity")

        if level.sweep_status == SweepStatus.SWEPT:
            confirmations.append("Equal level has been swept")
        elif level.sweep_status == SweepStatus.PARTIALLY_SWEPT:
            confirmations.append("Equal level has been partially swept")

        if side == SignalSide.LONG:
            if snapshot.bias == LiquidityBias.DOWN:
                confirmations.append("Counter-bias long setup from equal lows")
            if self._has_high_quality_zone(snapshot.zones, LiquiditySide.BUY_SIDE, current_price):
                confirmations.append("High-quality buy-side zone ahead")
        else:
            if snapshot.bias == LiquidityBias.UP:
                confirmations.append("Counter-bias short setup from equal highs")
            if self._has_high_quality_zone(snapshot.zones, LiquiditySide.SELL_SIDE, current_price):
                confirmations.append("High-quality sell-side zone ahead")

        if target is not None:
            confirmations.append("Clear liquidity target available")

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
        anchor_level: EqualLevel,
    ) -> float | None:
        if current_price <= 0:
            return None

        anchor_price = anchor_level.price
        fallback_pct = 0.0040

        if side == SignalSide.LONG:
            if anchor_price < current_price:
                return anchor_price * 0.9985
            return current_price * (1.0 - fallback_pct)

        if side == SignalSide.SHORT:
            if anchor_price > current_price:
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
            for level in snapshot.active_levels:
                if level.price > current_price:
                    candidates.append(level)
            for cluster in snapshot.stop_clusters:
                if cluster.center_price > current_price:
                    candidates.append(cluster)
            candidates.sort(key=self._reference_price)

        elif side == SignalSide.SHORT:
            for level in snapshot.active_levels:
                if level.price < current_price:
                    candidates.append(level)
            for cluster in snapshot.stop_clusters:
                if cluster.center_price < current_price:
                    candidates.append(cluster)
            candidates.sort(key=self._reference_price, reverse=True)

        if exclude is not None:
            exclude_price = self._reference_price(exclude)
            candidates = [
                item for item in candidates
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

    def _is_on_emit_cooldown(self, symbol: str, now: datetime) -> bool:
        strategy_cfg = self._strategy_cfg
        cooldown_seconds = (
            strategy_cfg.runtime.cooldown_seconds
            if strategy_cfg is not None
            else self.config.runtime.cooldown_seconds
        )
        if cooldown_seconds <= 0:
            return False

        last = self._last_emitted_at.get(symbol)
        if last is None:
            return False

        return now < last + timedelta(seconds=cooldown_seconds)