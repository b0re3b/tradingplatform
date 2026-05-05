from __future__ import annotations

from dataclasses import dataclass, field
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
    FilterDecision,
    MarketRegime,
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

# ---------------------------------------------------------------------------
# Proximity thresholds
# ---------------------------------------------------------------------------

# Максимальне перевищення рівня над поточною ціною для LONG-кандидата (0.3%)
_LONG_CANDIDATE_MAX_OVERSHOOT: float = 1.003

# Максимальне заниження рівня під поточною ціною для SHORT-кандидата (0.3%)
_SHORT_CANDIDATE_MIN_UNDERSHOOT: float = 0.997

# ---------------------------------------------------------------------------
# Stop-price offsets
# ---------------------------------------------------------------------------

# Зсув стоп-ціни нижче anchor для LONG (0.15%)
_LONG_STOP_OFFSET: float = 0.9985

# Зсув стоп-ціни вище anchor для SHORT (0.15%)
_SHORT_STOP_OFFSET: float = 1.0015

# Fallback відстань стопу від поточної ціни (0.4%)
_FALLBACK_STOP_PCT: float = 0.0040

# ---------------------------------------------------------------------------
# Edge weight constants  (raw score, NOT a probability → max > 1.0 is fine)
# ---------------------------------------------------------------------------

_EDGE_W_CONFIDENCE: float = 0.40
_EDGE_W_TOUCHES: float = 0.20
_EDGE_W_REACTIONS: float = 0.12
_EDGE_W_COMPACTNESS_TIGHT: float = 0.16    # width_pct <= 0.08%
_EDGE_W_COMPACTNESS_MED: float = 0.10      # width_pct <= 0.15%
_EDGE_W_COMPACTNESS_WIDE: float = 0.05     # width_pct <= 0.30%
_EDGE_W_DISTANCE_VERY_CLOSE: float = 0.22  # dist <= 0.15%
_EDGE_W_DISTANCE_CLOSE: float = 0.18       # dist <= 0.40%
_EDGE_W_DISTANCE_MED: float = 0.12         # dist <= 1.00%
_EDGE_W_DISTANCE_FAR: float = 0.06         # dist <= 2.00%
_EDGE_W_SWEEP_NOT: float = 0.10
_EDGE_W_SWEEP_PARTIAL: float = 0.14
_EDGE_W_SWEEP_FULL: float = 0.18
_EDGE_W_DIRECTIONAL: float = 0.10

# Normalisation denominators
_TOUCHES_NORM: float = 6.0
_REACTIONS_NORM: float = 4.0

# Compactness thresholds (as fraction of price)
_COMPACTNESS_TIGHT_PCT: float = 0.0008
_COMPACTNESS_MED_PCT: float = 0.0015
_COMPACTNESS_WIDE_PCT: float = 0.0030

# Distance thresholds (as fraction of price)
_DIST_VERY_CLOSE_PCT: float = 0.0015
_DIST_CLOSE_PCT: float = 0.0040
_DIST_MED_PCT: float = 0.0100
_DIST_FAR_PCT: float = 0.0200

# ---------------------------------------------------------------------------
# Signal priority thresholds
# ---------------------------------------------------------------------------

_PRIORITY_HIGH_SCORE: float = 1.70
_PRIORITY_HIGH_CONFIDENCE: float = 0.84
_PRIORITY_CRITICAL_SCORE: float = 2.10
_PRIORITY_CRITICAL_CONFIDENCE: float = 0.90

# ---------------------------------------------------------------------------
# Typed candidate container
# ---------------------------------------------------------------------------


@dataclass
class _EqualLevelCandidate:
    """Внутрішній контейнер для кандидата на сигнал."""

    side: SignalSide
    level: EqualLevel
    edge: float
    target: LiquidityLevel | StopCluster | None = field(default=None)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class EqualHighLowStrategy(BaseLiquidityStrategy):
    """
    Strategy для торгівлі від equal highs / equal lows.

    Логіка:
    - бере LiquidityMapSnapshot з StrategyContext;
    - шукає найкращий equal highs / equal lows candidate;
    - LONG від sell-side equal lows;
    - SHORT від buy-side equal highs;
    - формує StrategySignal без прямого execution.
    """

    @property
    def strategy_name(self) -> str:
        return "equal_high_low_strategy"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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

        if not self._base_context_is_valid(context, snapshot):
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
                timeframe=str(snapshot.timeframe),
                filters=[f"{f.name}:{f.decision.value}" for f in filters],
            )
            return None

        candidate = self._find_best_equal_level_candidate(snapshot, current_price)
        if candidate is None:
            self.log_debug(
                "EqualHighLowStrategy skipped: no valid equal highs/lows candidate",
                symbol=context.symbol,
                timeframe=str(snapshot.timeframe),
            )
            return None

        confidence = self._compute_confidence(snapshot, current_price, candidate)
        score = self._compute_score(snapshot, current_price, candidate, confidence)

        runtime = self._runtime
        if confidence < runtime.min_confidence or score < runtime.min_score:
            self.log_debug(
                "EqualHighLowStrategy skipped: below thresholds",
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

    # ------------------------------------------------------------------
    # Pre-filters
    # ------------------------------------------------------------------

    def _run_pre_filters(
        self,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> list[FilterResult]:
        results = self._run_common_pre_filters(
            context=context,
            snapshot=snapshot,
            current_price=current_price,
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

        return results

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    def _find_best_equal_level_candidate(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> _EqualLevelCandidate | None:
        long_levels = [
            level
            for level in snapshot.equal_levels
            if level.side == LiquiditySide.SELL_SIDE
            and level.price <= current_price * _LONG_CANDIDATE_MAX_OVERSHOOT
        ]
        short_levels = [
            level
            for level in snapshot.equal_levels
            if level.side == LiquiditySide.BUY_SIDE
            and level.price >= current_price * _SHORT_CANDIDATE_MIN_UNDERSHOOT
        ]

        best_long = self._pick_best_equal_level(long_levels, current_price, SignalSide.LONG)
        best_short = self._pick_best_equal_level(short_levels, current_price, SignalSide.SHORT)

        if best_long is None and best_short is None:
            return None
        if best_long is None:
            return best_short
        if best_short is None:
            return best_long

        return best_long if best_long.edge >= best_short.edge else best_short

    def _pick_best_equal_level(
        self,
        levels: list[EqualLevel],
        current_price: float,
        side: SignalSide,
    ) -> _EqualLevelCandidate | None:
        ranked: list[_EqualLevelCandidate] = []

        for level in levels:
            if level.level_type not in {
                LiquidityLevelType.EQUAL_HIGHS,
                LiquidityLevelType.EQUAL_LOWS,
            }:
                continue

            edge = self._equal_level_edge(level, current_price, side)
            if edge <= 0:
                continue

            ranked.append(_EqualLevelCandidate(side=side, level=level, edge=edge))

        if not ranked:
            return None

        return max(ranked, key=lambda c: c.edge)

    # ------------------------------------------------------------------
    # Edge / confidence / score
    # ------------------------------------------------------------------

    def _equal_level_edge(
        self,
        level: EqualLevel,
        current_price: float,
        side: SignalSide,
    ) -> float:
        """
        Зважена raw-оцінка якості рівня (НЕ імовірність; теоретичний максимум > 1.0).
        Складові та їхні максимальні внески задокументовані константами _EDGE_W_*.
        """
        if current_price <= 0 or level.price <= 0:
            return 0.0

        distance_pct = abs(level.price - current_price) / current_price

        confidence_part = min(max(level.confidence, 0.0), 1.0) * _EDGE_W_CONFIDENCE
        touches_part = min(level.touches_count / _TOUCHES_NORM, 1.0) * _EDGE_W_TOUCHES
        reactions_part = min(level.reaction_count / _REACTIONS_NORM, 1.0) * _EDGE_W_REACTIONS
        compactness_part = self._compactness_score(level)
        distance_part = self._distance_score_edge(distance_pct)
        sweep_part = self._sweep_score(level.sweep_status)
        directional_part = self._directional_score(level.side, side)

        return (
            confidence_part
            + touches_part
            + reactions_part
            + compactness_part
            + distance_part
            + sweep_part
            + directional_part
        )

    def _compactness_score(self, level: EqualLevel) -> float:
        if level.cluster_low is None or level.cluster_high is None:
            return 0.0

        width_pct = abs(level.cluster_high - level.cluster_low) / level.price
        if width_pct <= _COMPACTNESS_TIGHT_PCT:
            return _EDGE_W_COMPACTNESS_TIGHT
        if width_pct <= _COMPACTNESS_MED_PCT:
            return _EDGE_W_COMPACTNESS_MED
        if width_pct <= _COMPACTNESS_WIDE_PCT:
            return _EDGE_W_COMPACTNESS_WIDE
        return 0.0

    def _distance_score_edge(self, distance_pct: float) -> float:
        if distance_pct <= _DIST_VERY_CLOSE_PCT:
            return _EDGE_W_DISTANCE_VERY_CLOSE
        if distance_pct <= _DIST_CLOSE_PCT:
            return _EDGE_W_DISTANCE_CLOSE
        if distance_pct <= _DIST_MED_PCT:
            return _EDGE_W_DISTANCE_MED
        if distance_pct <= _DIST_FAR_PCT:
            return _EDGE_W_DISTANCE_FAR
        return 0.0

    def _sweep_score(self, sweep_status: SweepStatus) -> float:
        if sweep_status == SweepStatus.NOT_SWEPT:
            return _EDGE_W_SWEEP_NOT
        if sweep_status == SweepStatus.PARTIALLY_SWEPT:
            return _EDGE_W_SWEEP_PARTIAL
        if sweep_status == SweepStatus.SWEPT:
            return _EDGE_W_SWEEP_FULL
        return 0.0

    def _directional_score(self, level_side: LiquiditySide, signal_side: SignalSide) -> float:
        if signal_side == SignalSide.LONG and level_side == LiquiditySide.SELL_SIDE:
            return _EDGE_W_DIRECTIONAL
        if signal_side == SignalSide.SHORT and level_side == LiquiditySide.BUY_SIDE:
            return _EDGE_W_DIRECTIONAL
        return 0.0

    def _compute_confidence(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        candidate: _EqualLevelCandidate,
    ) -> float:
        level = candidate.level
        side = candidate.side

        confidence = min(candidate.edge, 1.0) * 0.62
        confidence += min(max(level.confidence, 0.0), 1.0) * 0.18
        confidence += min(level.touches_count / _TOUCHES_NORM, 1.0) * 0.08
        confidence += min(level.reaction_count / _REACTIONS_NORM, 1.0) * 0.05

        if level.sweep_status == SweepStatus.SWEPT:
            confidence += 0.06
        elif level.sweep_status == SweepStatus.PARTIALLY_SWEPT:
            confidence += 0.04

        if side == SignalSide.LONG and snapshot.bias == LiquidityBias.DOWN:
            confidence += 0.04
        elif side == SignalSide.SHORT and snapshot.bias == LiquidityBias.UP:
            confidence += 0.04

        confidence += self._zone_bonus(snapshot.zones, side, current_price)

        return max(0.0, min(confidence, 1.0))

    def _compute_score(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        candidate: _EqualLevelCandidate,
        confidence: float,
    ) -> float:
        anti_bias_bonus = 0.0
        if candidate.side == SignalSide.LONG and snapshot.bias == LiquidityBias.DOWN:
            anti_bias_bonus = 0.25
        elif candidate.side == SignalSide.SHORT and snapshot.bias == LiquidityBias.UP:
            anti_bias_bonus = 0.25

        structure_distance_score = self._level_distance_score(candidate.level, current_price)

        return max(
            0.0,
            confidence * 1.25
            + candidate.edge * 0.85
            + structure_distance_score * 0.40
            + anti_bias_bonus,
        )

    # ------------------------------------------------------------------
    # Helpers: scoring sub-components
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Signal building
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        candidate: _EqualLevelCandidate,
        confidence: float,
        score: float,
        filters: list[FilterResult],
    ) -> StrategySignal:
        level = candidate.level
        side = candidate.side

        target = self._find_target(snapshot, current_price, side, anchor_level=level)
        # Зберігаємо знайдений target у кандидаті для подальшого використання
        candidate.target = target

        entry_plan = self._build_entry_plan(
            side=side,
            current_price=current_price,
            target=target,
        )
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

        priority = self._resolve_priority(score, confidence)

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
            regime=context.regime.regime if context.regime is not None else MarketRegime.UNKNOWN,
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
                "strategy_weight": self.config.get_strategy_weight(
                    self.strategy_name,
                    default=1.0,
                ),
            },
        )

        signal.add_reason(self._build_primary_reason(level, side, current_price))
        signal.add_reason(self._build_target_reason(target))
        signal.add_reason(f"Liquidity pressure score = {snapshot.liquidity_pressure_score:.3f}")

        if snapshot.signal is not None and getattr(snapshot.signal, "explanation", None):
            signal.add_reason(snapshot.signal.explanation)

        for confirmation in self._build_confirmations(
            snapshot,
            level,
            side,
            current_price,
            target,
        ):
            signal.add_confirmation(confirmation)

        signal.add_source_feature("liquidity_map_snapshot")
        signal.add_source_feature("liquidity.equal_levels")
        signal.add_source_feature("liquidity")
        signal.add_source_feature("liquidity.equal_high_low")

        for filter_result in filters:
            signal.add_filter_result(filter_result)

        signal.validate()
        return signal

    @staticmethod
    def _resolve_priority(score: float, confidence: float) -> SignalPriority:
        if score >= _PRIORITY_CRITICAL_SCORE and confidence >= _PRIORITY_CRITICAL_CONFIDENCE:
            return SignalPriority.CRITICAL
        if score >= _PRIORITY_HIGH_SCORE or confidence >= _PRIORITY_HIGH_CONFIDENCE:
            return SignalPriority.HIGH
        return SignalPriority.MEDIUM

    # ------------------------------------------------------------------
    # Target search  (єдиний приватний хелпер замість двох дублікатів)
    # ------------------------------------------------------------------

    def _collect_directional_candidates(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
    ) -> list[LiquidityLevel | StopCluster]:
        """Збирає та сортує кандидати-цілі у напрямку сигналу."""
        candidates: list[LiquidityLevel | StopCluster] = []

        if side == SignalSide.LONG:
            candidates.extend(lvl for lvl in snapshot.active_levels if lvl.price > current_price)
            candidates.extend(cl for cl in snapshot.stop_clusters if cl.center_price > current_price)
            candidates.sort(key=self._reference_price)
        elif side == SignalSide.SHORT:
            candidates.extend(lvl for lvl in snapshot.active_levels if lvl.price < current_price)
            candidates.extend(cl for cl in snapshot.stop_clusters if cl.center_price < current_price)
            candidates.sort(key=self._reference_price, reverse=True)

        return candidates

    def _find_target(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
        anchor_level: EqualLevel,
    ) -> LiquidityLevel | StopCluster | None:
        candidates = self._collect_directional_candidates(snapshot, current_price, side)
        anchor_price = anchor_level.price
        filtered = [
            item
            for item in candidates
            if abs(self._reference_price(item) - anchor_price) > 1e-12
        ]
        return filtered[0] if filtered else None

    def _find_extended_target(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
        exclude: LiquidityLevel | StopCluster | None = None,
    ) -> LiquidityLevel | StopCluster | None:
        candidates = self._collect_directional_candidates(snapshot, current_price, side)

        if exclude is not None:
            exclude_price = self._reference_price(exclude)
            candidates = [
                item
                for item in candidates
                if abs(self._reference_price(item) - exclude_price) > 1e-12
            ]

        return candidates[0] if candidates else None

    # ------------------------------------------------------------------
    # Plan builders
    # ------------------------------------------------------------------

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
                    label="equal_level_primary_target",
                    metadata={"source": "nearest_directional_liquidity"},
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

        reason = None
        if self.config.builders.require_invalidation:
            reason = (
                "Equal lows support failed"
                if side == SignalSide.LONG
                else "Equal highs resistance failed"
            )

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

    # ------------------------------------------------------------------
    # Reason / confirmation builders
    # ------------------------------------------------------------------

    def _build_primary_reason(
        self,
        level: EqualLevel,
        side: SignalSide,
        current_price: float,
    ) -> str:
        prefix = (
            "Equal lows reaction -> long setup"
            if side == SignalSide.LONG
            else "Equal highs reaction -> short setup"
        )
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

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _resolve_stop_price(
        self,
        side: SignalSide,
        current_price: float,
        anchor_level: EqualLevel,
    ) -> float | None:
        if current_price <= 0:
            return None

        anchor_price = anchor_level.price

        if side == SignalSide.LONG:
            if anchor_price < current_price:
                return anchor_price * _LONG_STOP_OFFSET
            return current_price * (1.0 - _FALLBACK_STOP_PCT)

        if side == SignalSide.SHORT:
            if anchor_price > current_price:
                return anchor_price * _SHORT_STOP_OFFSET
            return current_price * (1.0 + _FALLBACK_STOP_PCT)

        return None

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