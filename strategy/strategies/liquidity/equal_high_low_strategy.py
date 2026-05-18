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
    StopCluster,
)

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


@dataclass(slots=True)
class _EqualLevelCandidate:
    """
    Internal candidate для equal highs / equal lows reaction setup.
    """

    side: SignalSide
    level: EqualLevel | LiquidityLevel
    edge: float
    target: LiquidityLevel | StopCluster | None = field(default=None)


class EqualHighLowStrategy(BaseLiquidityStrategy):
    """
    Production-ready equal highs / equal lows reaction strategy.

    Семантика:
    - LONG: reaction від активних sell-side equal lows нижче/біля current_price.
    - SHORT: reaction від активних buy-side equal highs вище/біля current_price.
    - Swept / partially swept equal levels за замовчуванням НЕ використовуються,
      бо це зона відповідальності StopHuntReversalStrategy.
    - Strategy не викликає analytics detectors.
    - Strategy не читає raw market data.
    - Strategy не виконує execution.
    - Full scope/futures/freshness validation делеговано BaseLiquidityStrategy.

    Очікуваний input:
    - StrategyContext-like object із LiquidityMapSnapshot.
    - Snapshot сформований analytics/liquidity.
    - Scope збігається: exchange + market_type + symbol + timeframe.
    """

    ALLOW_SWEPT_EQUAL_LEVELS: bool = False

    LONG_CANDIDATE_MAX_OVERSHOOT: float = 1.0030
    SHORT_CANDIDATE_MIN_UNDERSHOOT: float = 0.9970

    MIN_EDGE: float = 0.20
    MAX_LEVEL_DISTANCE_PCT: float = 0.0450
    MAX_TARGET_DISTANCE_PCT: float = 0.0800

    HIGH_PRIORITY_SCORE: float = 1.70
    HIGH_PRIORITY_CONFIDENCE: float = 0.84
    CRITICAL_PRIORITY_SCORE: float = 2.10
    CRITICAL_PRIORITY_CONFIDENCE: float = 0.90

    LONG_STOP_OFFSET: float = 0.9985
    SHORT_STOP_OFFSET: float = 1.0015
    FALLBACK_STOP_PCT: float = 0.0040

    @property
    def strategy_name(self) -> str:
        return "equal_high_low_strategy"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, context: Any) -> StrategySignal | None:
        self.validate_context(context)

        if not self.is_enabled():
            self.log_debug(
                "EqualHighLowStrategy skipped: disabled",
                symbol=getattr(context, "symbol", None),
                timeframe=self._value(getattr(context, "timeframe", None)),
            )
            return None

        snapshot = self._extract_snapshot(context)
        if snapshot is None:
            self.log_debug(
                "EqualHighLowStrategy skipped: liquidity snapshot not found",
                symbol=getattr(context, "symbol", None),
                timeframe=self._value(getattr(context, "timeframe", None)),
            )
            return None

        if not self._base_context_is_valid(context, snapshot):
            return None

        current_price = self._resolve_current_price(context, snapshot)
        if current_price is None:
            self.log_warning(
                "EqualHighLowStrategy skipped: current price unavailable",
                exchange=snapshot.exchange,
                market_type=snapshot.market_type,
                symbol=snapshot.symbol,
                timeframe=snapshot.timeframe,
            )
            return None

        filters = self._run_pre_filters(
            context=context,
            snapshot=snapshot,
            current_price=current_price,
        )
        if any(result.blocked for result in filters):
            self.log_debug(
                "EqualHighLowStrategy blocked by filters",
                exchange=snapshot.exchange,
                market_type=snapshot.market_type,
                symbol=snapshot.symbol,
                timeframe=snapshot.timeframe,
                filters=[f"{item.name}:{self._value(item.decision)}" for item in filters],
            )
            return None

        candidate = self._find_best_equal_level_candidate(
            snapshot=snapshot,
            current_price=current_price,
        )
        if candidate is None:
            self.log_debug(
                "EqualHighLowStrategy skipped: no valid equal highs/lows reaction candidate",
                exchange=snapshot.exchange,
                market_type=snapshot.market_type,
                symbol=snapshot.symbol,
                timeframe=snapshot.timeframe,
            )
            return None

        confidence = self._compute_confidence(
            snapshot=snapshot,
            current_price=current_price,
            candidate=candidate,
        )
        score = self._compute_score(
            snapshot=snapshot,
            current_price=current_price,
            candidate=candidate,
            confidence=confidence,
        )

        runtime = self._runtime
        min_confidence = self._safe_float(getattr(runtime, "min_confidence", 0.0), 0.0)
        min_score = self._safe_float(getattr(runtime, "min_score", 0.0), 0.0)

        if confidence < min_confidence or score < min_score:
            self.log_debug(
                "EqualHighLowStrategy skipped: below thresholds",
                exchange=snapshot.exchange,
                market_type=snapshot.market_type,
                symbol=snapshot.symbol,
                timeframe=snapshot.timeframe,
                confidence=confidence,
                score=score,
                min_confidence=min_confidence,
                min_score=min_score,
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
        context: Any,
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
    # Filters
    # ------------------------------------------------------------------

    def _run_pre_filters(
        self,
        context: Any,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> list[FilterResult]:
        results = self._run_common_pre_filters(
            context=context,
            snapshot=snapshot,
            current_price=current_price,
        )

        valid_equal_levels = self._valid_equal_reaction_levels(snapshot)

        if valid_equal_levels:
            results.append(
                FilterResult(
                    name="equal_levels_presence",
                    decision=FilterDecision.PASS,
                    reason=(
                        "Liquidity snapshot contains valid active equal highs/lows: "
                        f"{len(valid_equal_levels)}"
                    ),
                )
            )
        else:
            results.append(
                FilterResult(
                    name="equal_levels_presence",
                    decision=FilterDecision.BLOCK,
                    reason="Liquidity snapshot has no valid active equal highs/lows",
                )
            )

        near_levels = [
            level
            for level in valid_equal_levels
            if self._level_distance_ok(level, current_price)
        ]

        if near_levels:
            results.append(
                FilterResult(
                    name="equal_level_distance",
                    decision=FilterDecision.PASS,
                    reason=f"Valid equal levels near current price: {len(near_levels)}",
                )
            )
        else:
            results.append(
                FilterResult(
                    name="equal_level_distance",
                    decision=FilterDecision.BLOCK,
                    reason=(
                        "No valid equal highs/lows within max reaction distance: "
                        f"{self.MAX_LEVEL_DISTANCE_PCT:.4f}"
                    ),
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
        valid_levels = self._valid_equal_reaction_levels(snapshot)

        long_levels = [
            level
            for level in valid_levels
            if level.side == LiquiditySide.SELL_SIDE
            and level.price <= current_price * self.LONG_CANDIDATE_MAX_OVERSHOOT
            and self._level_distance_ok(level, current_price)
        ]

        short_levels = [
            level
            for level in valid_levels
            if level.side == LiquiditySide.BUY_SIDE
            and level.price >= current_price * self.SHORT_CANDIDATE_MIN_UNDERSHOOT
            and self._level_distance_ok(level, current_price)
        ]

        best_long = self._pick_best_equal_level(
            levels=long_levels,
            current_price=current_price,
            side=SignalSide.LONG,
        )
        best_short = self._pick_best_equal_level(
            levels=short_levels,
            current_price=current_price,
            side=SignalSide.SHORT,
        )

        if best_long is None and best_short is None:
            return None
        if best_long is None:
            return best_short
        if best_short is None:
            return best_long

        return best_long if best_long.edge >= best_short.edge else best_short

    def _valid_equal_reaction_levels(
        self,
        snapshot: LiquidityMapSnapshot,
    ) -> list[EqualLevel | LiquidityLevel]:
        """
        Бере equal levels із snapshot.equal_levels і, як fallback,
        equal-type levels із snapshot.active_levels.

        Це важливо для сумісності з різними версіями analytics/liquidity,
        де equal levels можуть бути представлені в обох списках.
        """
        candidates: list[EqualLevel | LiquidityLevel] = []

        candidates.extend(snapshot.equal_levels)
        candidates.extend(
            level
            for level in snapshot.active_levels
            if level.level_type in {
                LiquidityLevelType.EQUAL_HIGHS,
                LiquidityLevelType.EQUAL_LOWS,
            }
        )

        result: dict[str, EqualLevel | LiquidityLevel] = {}

        for level in candidates:
            if not self._is_valid_equal_reaction_level(level):
                continue

            existing = result.get(level.key)
            if existing is None:
                result[level.key] = level
                continue

            if self._equal_level_quality(level) > self._equal_level_quality(existing):
                result[level.key] = level

        return list(result.values())

    def _pick_best_equal_level(
        self,
        levels: list[EqualLevel | LiquidityLevel],
        current_price: float,
        side: SignalSide,
    ) -> _EqualLevelCandidate | None:
        ranked: list[_EqualLevelCandidate] = []

        for level in levels:
            edge = self._equal_level_edge(
                level=level,
                current_price=current_price,
                side=side,
            )
            if edge < self.MIN_EDGE:
                continue

            ranked.append(
                _EqualLevelCandidate(
                    side=side,
                    level=level,
                    edge=edge,
                )
            )

        if not ranked:
            return None

        return max(ranked, key=lambda candidate: candidate.edge)

    def _is_valid_equal_reaction_level(
        self,
        level: EqualLevel | LiquidityLevel,
    ) -> bool:
        if level.level_type not in {
            LiquidityLevelType.EQUAL_HIGHS,
            LiquidityLevelType.EQUAL_LOWS,
        }:
            return False

        if level.price <= 0:
            return False

        if level.is_invalidated() or level.is_expired():
            return False

        if not self.ALLOW_SWEPT_EQUAL_LEVELS and level.sweep_status in {
            SweepStatus.SWEPT,
            SweepStatus.PARTIALLY_SWEPT,
        }:
            return False

        expected_side = (
            LiquiditySide.BUY_SIDE
            if level.level_type == LiquidityLevelType.EQUAL_HIGHS
            else LiquiditySide.SELL_SIDE
        )

        if level.side != expected_side:
            return False

        return True

    def _level_distance_ok(
        self,
        level: EqualLevel | LiquidityLevel,
        current_price: float,
    ) -> bool:
        if current_price <= 0 or level.price <= 0:
            return False

        return self._distance_pct(level.price, current_price) <= self.MAX_LEVEL_DISTANCE_PCT

    def _equal_level_quality(
        self,
        level: EqualLevel | LiquidityLevel,
    ) -> tuple[float, int, int, float]:
        return (
            self._clamp01(level.confidence),
            int(max(level.touches_count, 0)),
            int(max(level.reaction_count, 0)),
            -self._compactness_width_pct(level),
        )

    # ------------------------------------------------------------------
    # Edge / confidence / score
    # ------------------------------------------------------------------

    def _equal_level_edge(
        self,
        level: EqualLevel | LiquidityLevel,
        current_price: float,
        side: SignalSide,
    ) -> float:
        if current_price <= 0 or level.price <= 0:
            return 0.0

        distance_pct = self._distance_pct(level.price, current_price)

        confidence_part = self._clamp01(level.confidence) * 0.34
        touches_part = min(max(level.touches_count, 0) / 6.0, 1.0) * 0.16
        reactions_part = min(max(level.reaction_count, 0) / 4.0, 1.0) * 0.10
        compactness_part = self._compactness_score(level) * 0.12
        distance_part = self._distance_score_edge(distance_pct) * 0.18
        directional_part = self._directional_score(level.side, side) * 0.10

        return self._clamp01(
            confidence_part
            + touches_part
            + reactions_part
            + compactness_part
            + distance_part
            + directional_part
        )

    def _compute_confidence(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        candidate: _EqualLevelCandidate,
    ) -> float:
        level = candidate.level
        side = candidate.side

        confidence = candidate.edge * 0.42
        confidence += self._clamp01(level.confidence) * 0.20
        confidence += min(max(level.touches_count, 0) / 6.0, 1.0) * 0.10
        confidence += min(max(level.reaction_count, 0) / 4.0, 1.0) * 0.07
        confidence += self._compactness_score(level) * 0.06

        if side == SignalSide.LONG:
            confidence += max(-self._clamp_signed(snapshot.liquidity_pressure_score), 0.0) * 0.06
            confidence += self._sweep_risk_down(snapshot) * 0.04
            confidence += self._zone_bonus(
                snapshot=snapshot,
                side=LiquiditySide.SELL_SIDE,
                current_price=current_price,
            )

            if snapshot.bias == LiquidityBias.DOWN:
                confidence += 0.04

        elif side == SignalSide.SHORT:
            confidence += max(self._clamp_signed(snapshot.liquidity_pressure_score), 0.0) * 0.06
            confidence += self._sweep_risk_up(snapshot) * 0.04
            confidence += self._zone_bonus(
                snapshot=snapshot,
                side=LiquiditySide.BUY_SIDE,
                current_price=current_price,
            )

            if snapshot.bias == LiquidityBias.UP:
                confidence += 0.04

        if snapshot.signal is not None:
            confidence += self._clamp01(getattr(snapshot.signal, "confidence", 0.0)) * 0.04

        return self._clamp01(confidence)

    def _compute_score(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        candidate: _EqualLevelCandidate,
        confidence: float,
    ) -> float:
        level = candidate.level
        side = candidate.side

        anti_bias_bonus = 0.0
        if side == SignalSide.LONG and snapshot.bias == LiquidityBias.DOWN:
            anti_bias_bonus = 0.22
        elif side == SignalSide.SHORT and snapshot.bias == LiquidityBias.UP:
            anti_bias_bonus = 0.22

        structure_distance_score = self._level_distance_score(
            level=level,
            current_price=current_price,
        )

        return max(
            0.0,
            confidence * 1.18
            + candidate.edge * 0.82
            + structure_distance_score * 0.36
            + anti_bias_bonus
        )

    def _compactness_score(self, level: EqualLevel | LiquidityLevel) -> float:
        width_pct = self._compactness_width_pct(level)

        if width_pct <= 0:
            return 0.0
        if width_pct <= 0.0008:
            return 1.0
        if width_pct <= 0.0015:
            return 0.70
        if width_pct <= 0.0030:
            return 0.40

        return 0.15

    def _compactness_width_pct(self, level: EqualLevel | LiquidityLevel) -> float:
        if level.price <= 0:
            return 0.0

        cluster_low = getattr(level, "cluster_low", None)
        cluster_high = getattr(level, "cluster_high", None)

        low = self._safe_float(cluster_low)
        high = self._safe_float(cluster_high)

        if low is None or high is None:
            return 0.0

        width = abs(high - low)
        return width / level.price

    def _distance_score_edge(self, distance_pct: float) -> float:
        if distance_pct <= 0.0015:
            return 1.0
        if distance_pct <= 0.0040:
            return 0.85
        if distance_pct <= 0.0100:
            return 0.62
        if distance_pct <= 0.0200:
            return 0.35
        if distance_pct <= self.MAX_LEVEL_DISTANCE_PCT:
            return 0.12

        return 0.0

    def _level_distance_score(
        self,
        level: EqualLevel | LiquidityLevel,
        current_price: float,
    ) -> float:
        if current_price <= 0 or level.price <= 0:
            return 0.0

        distance_pct = self._distance_pct(level.price, current_price)

        if distance_pct <= 0.001:
            return 0.20
        if distance_pct <= 0.003:
            return 0.55
        if distance_pct <= 0.010:
            return 1.00
        if distance_pct <= 0.020:
            return 0.65
        if distance_pct <= self.MAX_LEVEL_DISTANCE_PCT:
            return 0.28

        return 0.0

    def _directional_score(
        self,
        level_side: LiquiditySide,
        signal_side: SignalSide,
    ) -> float:
        if signal_side == SignalSide.LONG and level_side == LiquiditySide.SELL_SIDE:
            return 1.0

        if signal_side == SignalSide.SHORT and level_side == LiquiditySide.BUY_SIDE:
            return 1.0

        return 0.0

    def _zone_bonus(
        self,
        snapshot: LiquidityMapSnapshot,
        side: LiquiditySide,
        current_price: float,
    ) -> float:
        """
        Для LONG від equal lows корисна sell-side zone під/біля ціни.
        Для SHORT від equal highs корисна buy-side zone над/біля ціни.
        """
        zones = [
            zone
            for zone in self._directional_zones(snapshot, side)
            if (
                side == LiquiditySide.SELL_SIDE
                and zone.center_price <= current_price * self.LONG_CANDIDATE_MAX_OVERSHOOT
            )
            or (
                side == LiquiditySide.BUY_SIDE
                and zone.center_price >= current_price * self.SHORT_CANDIDATE_MIN_UNDERSHOOT
            )
        ]

        if not zones:
            return 0.0

        best = max(zones, key=lambda zone: self._clamp01(zone.score))
        return 0.05 * self._clamp01(best.score)

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------

    def _find_target(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
        anchor_level: EqualLevel | LiquidityLevel,
    ) -> LiquidityLevel | StopCluster | None:
        if side == SignalSide.LONG:
            candidates = self._collect_targets_above(snapshot, current_price)
        elif side == SignalSide.SHORT:
            candidates = self._collect_targets_below(snapshot, current_price)
        else:
            return None

        valid = [
            item
            for item in candidates
            if self._target_distance_ok(item, current_price)
            and self._reference_price(item) != anchor_level.price
        ]

        return valid[0] if valid else None

    def _find_extended_target(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
        exclude: LiquidityLevel | StopCluster | None = None,
    ) -> LiquidityLevel | StopCluster | None:
        if side == SignalSide.LONG:
            candidates = self._collect_targets_above(snapshot, current_price)
        elif side == SignalSide.SHORT:
            candidates = self._collect_targets_below(snapshot, current_price)
        else:
            return None

        if exclude is not None:
            exclude_price = self._reference_price(exclude)
            candidates = [
                item
                for item in candidates
                if abs(self._reference_price(item) - exclude_price) > 1e-12
            ]

        candidates = [
            item
            for item in candidates
            if self._target_distance_ok(item, current_price)
        ]

        return candidates[0] if candidates else None

    def _target_distance_ok(
        self,
        target: LiquidityLevel | StopCluster,
        current_price: float,
    ) -> bool:
        ref_price = self._reference_price(target)
        if ref_price <= 0 or current_price <= 0:
            return False

        return self._distance_pct(ref_price, current_price) <= self.MAX_TARGET_DISTANCE_PCT

    # ------------------------------------------------------------------
    # Signal building
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        context: Any,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        candidate: _EqualLevelCandidate,
        confidence: float,
        score: float,
        filters: list[FilterResult],
    ) -> StrategySignal:
        level = candidate.level
        side = candidate.side

        target = self._find_target(
            snapshot=snapshot,
            current_price=current_price,
            side=side,
            anchor_level=level,
        )
        candidate.target = target

        entry_plan = self._build_entry_plan(
            side=side,
            current_price=current_price,
            anchor_level=level,
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
            symbol=snapshot.symbol,
            side=side,
            entry_plan=entry_plan,
            exit_plan=exit_plan,
            invalidation_plan=invalidation_plan,
            snapshot=snapshot,
        )

        priority = self._resolve_priority(score=score, confidence=confidence)
        strategy_cfg = self._strategy_cfg

        analytics_metadata = self._build_liquidity_signal_metadata(
            snapshot=snapshot,
            current_price=current_price,
            target=target,
            evidence=level,
            setup_name="equal_high_low_reaction",
            extra={
                "side": self._value(side),
                "edge": candidate.edge,
                "level_price": level.price,
                "level_type": self._value(level.level_type),
                "level_side": self._value(level.side),
                "level_confidence": self._clamp01(level.confidence),
                "touches_count": level.touches_count,
                "reaction_count": level.reaction_count,
                "sweep_status": self._value(level.sweep_status),
                "cluster_low": getattr(level, "cluster_low", None),
                "cluster_high": getattr(level, "cluster_high", None),
                "compactness_width_pct": self._compactness_width_pct(level),
                "level_distance_pct": self._distance_pct(level.price, current_price),
                "target_price": self._reference_price(target),
                "target_type": self._target_type(target),
                "target_confidence": self._target_confidence(target),
                "target_distance_pct": (
                    self._distance_pct(self._reference_price(target), current_price)
                    if target is not None
                    else None
                ),
                "allow_swept_equal_levels": self.ALLOW_SWEPT_EQUAL_LEVELS,
                "strategy_weight": (
                    strategy_cfg.weight if strategy_cfg is not None else 1.0
                ),
                "strategy_semantics": "equal_high_low_reaction",
            },
        )

        signal = StrategySignal(
            symbol=snapshot.symbol,
            side=side,
            strategy_name=self.strategy_name,
            category=self.category,
            timeframe=snapshot.timeframe,
            setup_type=SetupType.REVERSAL,
            timestamp=self._context_timestamp(context),
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
            metadata=analytics_metadata,
        )

        signal.add_reason(self._build_primary_reason(candidate, current_price))
        signal.add_reason(self._build_target_reason(target))
        signal.add_reason(
            f"Equal level edge={candidate.edge:.3f}, confidence={confidence:.3f}"
        )

        if snapshot.signal is not None and getattr(snapshot.signal, "explanation", None):
            signal.add_reason(snapshot.signal.explanation)

        for confirmation in self._build_confirmations(
            snapshot=snapshot,
            candidate=candidate,
            current_price=current_price,
            target=target,
        ):
            signal.add_confirmation(confirmation)

        signal.add_source_feature("liquidity_map_snapshot")
        signal.add_source_feature("analytics.liquidity")
        signal.add_source_feature("analytics.liquidity.equal_levels")
        signal.add_source_feature("liquidity.equal_high_low_reaction")

        for filter_result in filters:
            signal.add_filter_result(filter_result)

        signal.validate()
        return signal

    # ------------------------------------------------------------------
    # Plan builders
    # ------------------------------------------------------------------

    def _build_entry_plan(
        self,
        side: SignalSide,
        current_price: float,
        anchor_level: EqualLevel | LiquidityLevel,
        target: LiquidityLevel | StopCluster | None,
    ) -> EntryPlan:
        notes = [
            "Enter on reaction from active equal highs/lows liquidity structure",
            f"Anchor level at {anchor_level.price:.6f}",
        ]

        if target is not None:
            notes.append(f"Primary opposite liquidity target at {self._reference_price(target):.6f}")

        entry_type = (
            getattr(self.config.builders, "default_entry_type", None)
            or EntryType.MARKET
        )

        return EntryPlan(
            entry_type=entry_type,
            price=current_price if entry_type == EntryType.LIMIT else None,
            timeout_seconds=getattr(self.config.runtime, "max_signal_age_seconds", 60),
            max_slippage_bps=8.0,
            confirmation_required=False,
            notes=notes,
            metadata={
                "entry_logic": "equal_high_low_reaction",
                "anchor_level_price": anchor_level.price,
                "anchor_level_type": self._value(anchor_level.level_type),
                "target_price": self._reference_price(target),
                "target_type": self._target_type(target),
            },
        )

    def _build_exit_plan(
        self,
        side: SignalSide,
        current_price: float,
        target: LiquidityLevel | StopCluster | None,
        anchor_level: EqualLevel | LiquidityLevel,
        snapshot: LiquidityMapSnapshot,
    ) -> ExitPlan:
        target_price = self._reference_price(target) if target is not None else None
        stop_price = self._resolve_stop_price(
            side=side,
            current_price=current_price,
            anchor_level=anchor_level,
        )

        tp_levels: list[TargetPlan] = []

        enable_partial_tp = bool(
            getattr(self.config.builders, "enable_partial_take_profit", True)
        )

        if target_price is not None and target_price > 0:
            tp_levels.append(
                TargetPlan(
                    price=target_price,
                    size_fraction=0.70 if enable_partial_tp else 1.0,
                    rr=self._compute_rr(
                        current_price=current_price,
                        stop_price=stop_price,
                        target_price=target_price,
                        side=side,
                    ),
                    label="equal_level_reaction_primary_target",
                    metadata={
                        "source": "opposite_side_liquidity",
                        "target_type": self._target_type(target),
                    },
                )
            )

        secondary_target = self._find_extended_target(
            snapshot=snapshot,
            current_price=current_price,
            side=side,
            exclude=target,
        )
        secondary_price = (
            self._reference_price(secondary_target)
            if secondary_target is not None
            else None
        )

        if enable_partial_tp and secondary_price is not None and secondary_price > 0:
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
                    label="equal_level_reaction_extended_target",
                    metadata={
                        "source": "extended_opposite_side_liquidity",
                        "target_type": self._target_type(secondary_target),
                    },
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
            max_holding_seconds=max(
                getattr(self.config.runtime, "max_signal_age_seconds", 60) * 3,
                60,
            ),
            partial_exit_enabled=enable_partial_tp,
            metadata={
                "exit_logic": "equal_level_reaction_to_opposite_liquidity",
                "primary_target_price": target_price,
                "secondary_target_price": secondary_price,
                "stop_price": stop_price,
                "anchor_level_price": anchor_level.price,
            },
        )

    def _build_invalidation_plan(
        self,
        side: SignalSide,
        current_price: float,
        anchor_level: EqualLevel | LiquidityLevel,
    ) -> InvalidationPlan:
        price = self._resolve_stop_price(
            side=side,
            current_price=current_price,
            anchor_level=anchor_level,
        )

        reason = None
        if bool(getattr(self.config.builders, "require_invalidation", True)):
            reason = (
                "Sell-side equal lows failed to hold as reaction support"
                if side == SignalSide.LONG
                else "Buy-side equal highs failed to hold as reaction resistance"
            )

        return InvalidationPlan(
            price=price,
            reason=reason,
            timeout_seconds=max(
                getattr(self.config.runtime, "max_signal_age_seconds", 60),
                30,
            ),
            conditions=[
                "signal_age_expired",
                "equal_level_invalidated",
                "equal_level_swept_against_reaction",
                "opposite_liquidity_pressure_domination",
                "snapshot_stale",
            ],
            metadata={
                "invalidation_source": "equal_level_anchor",
                "invalidation_price": price,
                "anchor_level_price": anchor_level.price,
                "anchor_level_type": self._value(anchor_level.level_type),
            },
        )

    def _build_execution_plan(
        self,
        symbol: str,
        side: SignalSide,
        entry_plan: EntryPlan,
        exit_plan: ExitPlan,
        invalidation_plan: InvalidationPlan,
        snapshot: LiquidityMapSnapshot,
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
                "Signal uses active equal highs/lows reaction, not swept stop-hunt evidence",
                "Risk manager must validate exposure, leverage, drawdown and correlation constraints before execution",
                "Execution should only consume risk-confirmed signals",
            ],
            metadata={
                "strategy_name": self.strategy_name,
                "category": self._value(self.category),
                "exchange": snapshot.exchange,
                "market_type": snapshot.market_type,
                "scope": snapshot.scope,
                "scope_key": snapshot.scope_key,
                "strategy_semantics": "equal_high_low_reaction",
            },
        )

    # ------------------------------------------------------------------
    # Reason / confirmation builders
    # ------------------------------------------------------------------

    def _build_primary_reason(
        self,
        candidate: _EqualLevelCandidate,
        current_price: float,
    ) -> str:
        level = candidate.level

        if candidate.side == SignalSide.LONG:
            prefix = "Sell-side equal lows reaction -> long setup"
        else:
            prefix = "Buy-side equal highs reaction -> short setup"

        return (
            f"{prefix}: level={level.price:.6f}, "
            f"current_price={current_price:.6f}, "
            f"type={self._value(level.level_type)}, "
            f"side={self._value(level.side)}, "
            f"confidence={self._clamp01(level.confidence):.3f}, "
            f"touches={level.touches_count}, "
            f"reactions={level.reaction_count}, "
            f"edge={candidate.edge:.3f}"
        )

    def _build_target_reason(
        self,
        target: LiquidityLevel | StopCluster | None,
    ) -> str:
        if target is None:
            return (
                "No explicit opposite liquidity target found; signal is based on "
                "equal highs/lows reaction structure"
            )

        if isinstance(target, StopCluster):
            return (
                f"Nearest opposite target is stop cluster at {target.center_price:.6f} "
                f"(confidence={target.confidence:.3f}, "
                f"density={target.estimated_stop_density:.3f}, "
                f"strength={self._value(target.strength)})"
            )

        return (
            f"Nearest opposite target is liquidity level at {target.price:.6f} "
            f"(type={self._value(target.level_type)}, "
            f"confidence={target.confidence:.3f}, "
            f"sweep_status={self._value(target.sweep_status)})"
        )

    def _build_confirmations(
        self,
        snapshot: LiquidityMapSnapshot,
        candidate: _EqualLevelCandidate,
        current_price: float,
        target: LiquidityLevel | StopCluster | None,
    ) -> list[str]:
        confirmations: list[str] = []

        level = candidate.level
        side = candidate.side

        if level.touches_count >= 3:
            confirmations.append("Equal level has multiple touches")

        if level.reaction_count >= 2:
            confirmations.append("Equal level has prior reactions")

        if self._compactness_score(level) >= 0.70:
            confirmations.append("Equal level cluster is compact")

        if side == SignalSide.LONG:
            if snapshot.bias == LiquidityBias.DOWN:
                confirmations.append("Prior downside liquidity bias supports sell-side reaction context")

            if snapshot.liquidity_pressure_score < 0:
                confirmations.append("Liquidity pressure favors sell-side liquidity below price")

            if self._sweep_risk_down(snapshot) >= 0.55:
                confirmations.append("Downside sweep risk is elevated near equal lows")

            if self._has_reaction_zone_confirmation(
                snapshot=snapshot,
                liquidity_side=LiquiditySide.SELL_SIDE,
                current_price=current_price,
            ):
                confirmations.append("Sell-side liquidity zone confirms equal lows reaction area")

        elif side == SignalSide.SHORT:
            if snapshot.bias == LiquidityBias.UP:
                confirmations.append("Prior upside liquidity bias supports buy-side reaction context")

            if snapshot.liquidity_pressure_score > 0:
                confirmations.append("Liquidity pressure favors buy-side liquidity above price")

            if self._sweep_risk_up(snapshot) >= 0.55:
                confirmations.append("Upside sweep risk is elevated near equal highs")

            if self._has_reaction_zone_confirmation(
                snapshot=snapshot,
                liquidity_side=LiquiditySide.BUY_SIDE,
                current_price=current_price,
            ):
                confirmations.append("Buy-side liquidity zone confirms equal highs reaction area")

        if target is not None:
            confirmations.append("Opposite-side liquidity target available")

        if snapshot.signal is not None and getattr(snapshot.signal, "confidence", 0.0) >= 0.65:
            confirmations.append("Analytics liquidity signal confidence is strong")

        return confirmations

    def _has_reaction_zone_confirmation(
        self,
        snapshot: LiquidityMapSnapshot,
        liquidity_side: LiquiditySide,
        current_price: float,
    ) -> bool:
        zones = [
            zone
            for zone in self._directional_zones(snapshot, liquidity_side)
            if (
                liquidity_side == LiquiditySide.SELL_SIDE
                and zone.center_price <= current_price * self.LONG_CANDIDATE_MAX_OVERSHOOT
            )
            or (
                liquidity_side == LiquiditySide.BUY_SIDE
                and zone.center_price >= current_price * self.SHORT_CANDIDATE_MIN_UNDERSHOOT
            )
        ]

        if not zones:
            return False

        best = max(zones, key=lambda zone: self._clamp01(zone.score))
        return self._clamp01(best.score) >= 0.60

    # ------------------------------------------------------------------
    # Stop / RR / priority / misc
    # ------------------------------------------------------------------

    def _resolve_stop_price(
        self,
        side: SignalSide,
        current_price: float,
        anchor_level: EqualLevel | LiquidityLevel,
    ) -> float:
        anchor_price = anchor_level.price

        if side == SignalSide.LONG:
            if anchor_price > 0 and anchor_price <= current_price * self.LONG_CANDIDATE_MAX_OVERSHOOT:
                return anchor_price * self.LONG_STOP_OFFSET
            return current_price * (1.0 - self.FALLBACK_STOP_PCT)

        if side == SignalSide.SHORT:
            if anchor_price > 0 and anchor_price >= current_price * self.SHORT_CANDIDATE_MIN_UNDERSHOOT:
                return anchor_price * self.SHORT_STOP_OFFSET
            return current_price * (1.0 + self.FALLBACK_STOP_PCT)

        return current_price

    def _compute_rr(
        self,
        current_price: float,
        stop_price: float,
        target_price: float,
        side: SignalSide,
    ) -> float:
        if current_price <= 0 or stop_price <= 0 or target_price <= 0:
            return 0.0

        if side == SignalSide.LONG:
            risk = current_price - stop_price
            reward = target_price - current_price
        elif side == SignalSide.SHORT:
            risk = stop_price - current_price
            reward = current_price - target_price
        else:
            return 0.0

        if risk <= 0 or reward <= 0:
            return 0.0

        return reward / risk

    def _resolve_priority(
        self,
        score: float,
        confidence: float,
    ) -> SignalPriority:
        if (
            score >= self.CRITICAL_PRIORITY_SCORE
            and confidence >= self.CRITICAL_PRIORITY_CONFIDENCE
        ):
            return SignalPriority.CRITICAL

        if score >= self.HIGH_PRIORITY_SCORE and confidence >= self.HIGH_PRIORITY_CONFIDENCE:
            return SignalPriority.HIGH

        return SignalPriority.NORMAL

    def _target_type(
        self,
        target: LiquidityLevel | StopCluster | None,
    ) -> str | None:
        if target is None:
            return None

        if isinstance(target, StopCluster):
            return "stop_cluster"

        if isinstance(target, LiquidityLevel):
            return self._value(target.level_type)

        return target.__class__.__name__

    def _target_confidence(
        self,
        target: LiquidityLevel | StopCluster | None,
    ) -> float:
        if target is None:
            return 0.0

        return self._clamp01(getattr(target, "confidence", 0.0))