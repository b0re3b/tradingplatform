from __future__ import annotations

from typing import Any

from analytics.liquidity.enums import LiquidityBias, LiquiditySide, SweepStatus
from analytics.liquidity.models import (
    LiquidityLevel,
    LiquidityMapSnapshot,
    LiquidityZone,
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


class LiquidityMapBiasStrategy(BaseLiquidityStrategy):
    """
    Production-ready directional bias strategy на основі повної liquidity map.

    Семантика:
    - LONG: liquidity landscape явно зміщений до upside / buy-side liquidity.
    - SHORT: liquidity landscape явно зміщений до downside / sell-side liquidity.
    - Strategy не є stop-hunt reversal і не є aggressive sweep-entry trigger.
    - Основна роль: дати directional context / bias signal для confluence,
      hybrid strategies, AI/risk scoring, dashboard і downstream signal processor.

    Важливо:
    - не викликає analytics detectors;
    - не читає raw market data;
    - працює тільки з LiquidityMapSnapshot;
    - перевіряє full scope через BaseLiquidityStrategy:
      exchange + market_type + symbol + timeframe;
    - futures/perpetual only;
    - не виконує угоди напряму.
    """

    MIN_DIRECTIONAL_EDGE: float = 0.42
    MIN_EDGE_DELTA: float = 0.10
    MIN_PRESSURE_ABS: float = 0.12
    MIN_ANALYTICS_CONFIDENCE: float = 0.35

    MIN_TARGET_DISTANCE_PCT: float = 0.0010
    MAX_TARGET_DISTANCE_PCT: float = 0.0800

    HIGH_PRIORITY_SCORE: float = 1.65
    HIGH_PRIORITY_CONFIDENCE: float = 0.82
    CRITICAL_PRIORITY_SCORE: float = 2.05
    CRITICAL_PRIORITY_CONFIDENCE: float = 0.90

    FALLBACK_STOP_PCT: float = 0.0050
    LONG_STOP_OFFSET: float = 0.9980
    SHORT_STOP_OFFSET: float = 1.0020

    @property
    def strategy_name(self) -> str:
        return "liquidity_map_bias_strategy"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, context: Any) -> StrategySignal | None:
        self.validate_context(context)

        if not self.is_enabled():
            self.log_debug(
                "LiquidityMapBiasStrategy skipped: disabled",
                symbol=getattr(context, "symbol", None),
                timeframe=self._value(getattr(context, "timeframe", None)),
            )
            return None

        snapshot = self._extract_snapshot(context)
        if snapshot is None:
            self.log_debug(
                "LiquidityMapBiasStrategy skipped: liquidity snapshot not found",
                symbol=getattr(context, "symbol", None),
                timeframe=self._value(getattr(context, "timeframe", None)),
            )
            return None

        if not self._base_context_is_valid(context, snapshot):
            return None

        current_price = self._resolve_current_price(context, snapshot)
        if current_price is None:
            self.log_warning(
                "LiquidityMapBiasStrategy skipped: current price unavailable",
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
                "LiquidityMapBiasStrategy blocked by filters",
                exchange=snapshot.exchange,
                market_type=snapshot.market_type,
                symbol=snapshot.symbol,
                timeframe=snapshot.timeframe,
                filters=[f"{item.name}:{self._value(item.decision)}" for item in filters],
            )
            return None

        side = self._infer_side(snapshot)
        if side == SignalSide.UNKNOWN:
            self.log_debug(
                "LiquidityMapBiasStrategy skipped: no strong liquidity map bias",
                exchange=snapshot.exchange,
                market_type=snapshot.market_type,
                symbol=snapshot.symbol,
                timeframe=snapshot.timeframe,
                bias=self._value(snapshot.bias),
                upside_edge=self._upside_bias_edge(snapshot),
                downside_edge=self._downside_bias_edge(snapshot),
                pressure=snapshot.liquidity_pressure_score,
            )
            return None

        target = self._target_for_side(
            snapshot=snapshot,
            current_price=current_price,
            side=side,
        )

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
        min_confidence = self._safe_float(getattr(runtime, "min_confidence", 0.0), 0.0)
        min_score = self._safe_float(getattr(runtime, "min_score", 0.0), 0.0)

        if confidence < min_confidence or score < min_score:
            self.log_debug(
                "LiquidityMapBiasStrategy skipped: below thresholds",
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

        upside_edge = self._upside_bias_edge(snapshot)
        downside_edge = self._downside_bias_edge(snapshot)
        edge_delta = abs(upside_edge - downside_edge)
        directional_edge = max(upside_edge, downside_edge)
        pressure_abs = abs(self._clamp_signed(snapshot.liquidity_pressure_score))

        if directional_edge < self.MIN_DIRECTIONAL_EDGE:
            results.append(
                FilterResult(
                    name="liquidity_map_bias_directional_edge",
                    decision=FilterDecision.BLOCK,
                    reason=(
                        "Directional liquidity map edge too weak: "
                        f"upside={upside_edge:.4f}, downside={downside_edge:.4f}, "
                        f"required={self.MIN_DIRECTIONAL_EDGE:.4f}"
                    ),
                )
            )
        else:
            results.append(
                FilterResult(
                    name="liquidity_map_bias_directional_edge",
                    decision=FilterDecision.PASS,
                    reason=(
                        "Directional liquidity map edge present: "
                        f"upside={upside_edge:.4f}, downside={downside_edge:.4f}"
                    ),
                )
            )

        if edge_delta < self.MIN_EDGE_DELTA and pressure_abs < self.MIN_PRESSURE_ABS:
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

        analytics_confidence = self._analytics_signal_confidence(snapshot)
        if analytics_confidence < self.MIN_ANALYTICS_CONFIDENCE:
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

        return results

    # ------------------------------------------------------------------
    # Bias inference
    # ------------------------------------------------------------------

    def _infer_side(self, snapshot: LiquidityMapSnapshot) -> SignalSide:
        upside_edge = self._upside_bias_edge(snapshot)
        downside_edge = self._downside_bias_edge(snapshot)
        delta = upside_edge - downside_edge
        pressure = self._clamp_signed(snapshot.liquidity_pressure_score)

        if snapshot.bias == LiquidityBias.UP:
            if upside_edge >= self.MIN_DIRECTIONAL_EDGE:
                return SignalSide.LONG

        if snapshot.bias == LiquidityBias.DOWN:
            if downside_edge >= self.MIN_DIRECTIONAL_EDGE:
                return SignalSide.SHORT

        if (
            delta >= self.MIN_EDGE_DELTA
            and upside_edge >= self.MIN_DIRECTIONAL_EDGE
            and pressure >= -0.05
        ):
            return SignalSide.LONG

        if (
            delta <= -self.MIN_EDGE_DELTA
            and downside_edge >= self.MIN_DIRECTIONAL_EDGE
            and pressure <= 0.05
        ):
            return SignalSide.SHORT

        if pressure >= self.MIN_PRESSURE_ABS and upside_edge >= self.MIN_DIRECTIONAL_EDGE * 0.85:
            return SignalSide.LONG

        if pressure <= -self.MIN_PRESSURE_ABS and downside_edge >= self.MIN_DIRECTIONAL_EDGE * 0.85:
            return SignalSide.SHORT

        return SignalSide.UNKNOWN

    def _upside_bias_edge(self, snapshot: LiquidityMapSnapshot) -> float:
        pressure = max(self._clamp_signed(snapshot.liquidity_pressure_score), 0.0)

        bias_bonus = 0.10 if snapshot.bias == LiquidityBias.UP else 0.0
        signal_bias_bonus = 0.05 if self._snapshot_signal_bias(snapshot) == LiquidityBias.UP else 0.0

        return self._clamp01(
            0.24 * self._clamp01(snapshot.above_liquidity_score)
            + 0.22 * self._magnet_score_up(snapshot)
            + 0.18 * self._sweep_risk_up(snapshot)
            + 0.16 * pressure
            + 0.10 * self._zone_score(snapshot, LiquiditySide.BUY_SIDE)
            + bias_bonus
            + signal_bias_bonus
        )

    def _downside_bias_edge(self, snapshot: LiquidityMapSnapshot) -> float:
        pressure = max(-self._clamp_signed(snapshot.liquidity_pressure_score), 0.0)

        bias_bonus = 0.10 if snapshot.bias == LiquidityBias.DOWN else 0.0
        signal_bias_bonus = 0.05 if self._snapshot_signal_bias(snapshot) == LiquidityBias.DOWN else 0.0

        return self._clamp01(
            0.24 * self._clamp01(snapshot.below_liquidity_score)
            + 0.22 * self._magnet_score_down(snapshot)
            + 0.18 * self._sweep_risk_down(snapshot)
            + 0.16 * pressure
            + 0.10 * self._zone_score(snapshot, LiquiditySide.SELL_SIDE)
            + bias_bonus
            + signal_bias_bonus
        )

    def _snapshot_signal_bias(self, snapshot: LiquidityMapSnapshot) -> LiquidityBias | None:
        if snapshot.signal is None:
            return None

        bias = getattr(snapshot.signal, "bias", None)
        if isinstance(bias, LiquidityBias):
            return bias

        try:
            return LiquidityBias(str(bias))
        except Exception:
            return None

    def _analytics_signal_confidence(self, snapshot: LiquidityMapSnapshot) -> float:
        if snapshot.signal is None:
            metadata_confidence = (snapshot.metadata or {}).get("confidence")
            return self._clamp01(metadata_confidence or 0.0)

        return self._clamp01(getattr(snapshot.signal, "confidence", 0.0))

    def _zone_score(
        self,
        snapshot: LiquidityMapSnapshot,
        side: LiquiditySide,
    ) -> float:
        zones = self._directional_zones(snapshot, side)
        if not zones:
            return 0.0

        return self._clamp01(max(self._clamp01(zone.score) for zone in zones))

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------

    def _target_for_side(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
    ) -> LiquidityLevel | StopCluster | None:
        """
        Bias strategy може мати target, але target не є обов'язковим.

        Для confluence-сигналу достатньо сильного directional bias.
        Якщо target є — він посилює confidence/score і додається в metadata.
        """

        if side == SignalSide.LONG:
            candidates = [
                snapshot.nearest_above_level,
                snapshot.strongest_cluster_above,
                *self._collect_targets_above(snapshot, current_price),
            ]
        elif side == SignalSide.SHORT:
            candidates = [
                snapshot.nearest_below_level,
                snapshot.strongest_cluster_below,
                *self._collect_targets_below(snapshot, current_price),
            ]
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

        valid = self._dedupe_liquidity_items(valid)

        if not valid:
            return None

        if side == SignalSide.LONG:
            return min(valid, key=self._reference_price)

        return max(valid, key=self._reference_price)

    def _is_valid_bias_target(
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

        distance_pct = self._distance_pct(ref_price, current_price)
        if distance_pct < self.MIN_TARGET_DISTANCE_PCT:
            return False

        if distance_pct > self.MAX_TARGET_DISTANCE_PCT:
            return False

        if isinstance(item, LiquidityLevel):
            if item.is_invalidated() or item.is_expired():
                return False

            if item.sweep_status == SweepStatus.SWEPT:
                return False

        if isinstance(item, StopCluster):
            if self._cluster_is_swept(item):
                return False

        return True

    # ------------------------------------------------------------------
    # Confidence / score
    # ------------------------------------------------------------------

    def _compute_confidence(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
        target: LiquidityLevel | StopCluster | None,
    ) -> float:
        analytics_confidence = self._analytics_signal_confidence(snapshot)

        if side == SignalSide.LONG:
            edge = self._upside_bias_edge(snapshot)
            opposite_edge = self._downside_bias_edge(snapshot)
            pressure = max(self._clamp_signed(snapshot.liquidity_pressure_score), 0.0)
            zone_bonus = self._zone_alignment_bonus(
                snapshot=snapshot,
                side=LiquiditySide.BUY_SIDE,
                current_price=current_price,
            )
            bias_bonus = 0.06 if snapshot.bias == LiquidityBias.UP else 0.0

        elif side == SignalSide.SHORT:
            edge = self._downside_bias_edge(snapshot)
            opposite_edge = self._upside_bias_edge(snapshot)
            pressure = max(-self._clamp_signed(snapshot.liquidity_pressure_score), 0.0)
            zone_bonus = self._zone_alignment_bonus(
                snapshot=snapshot,
                side=LiquiditySide.SELL_SIDE,
                current_price=current_price,
            )
            bias_bonus = 0.06 if snapshot.bias == LiquidityBias.DOWN else 0.0

        else:
            return 0.0

        separation = max(edge - opposite_edge, 0.0)
        target_bonus = self._target_quality_bonus(target, current_price)

        return self._clamp01(
            0.34 * edge
            + 0.18 * separation
            + 0.16 * analytics_confidence
            + 0.12 * pressure
            + 0.10 * target_bonus
            + zone_bonus
            + bias_bonus
        )

    def _compute_score(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
        confidence: float,
        target: LiquidityLevel | StopCluster | None,
    ) -> float:
        if side == SignalSide.LONG:
            edge = self._upside_bias_edge(snapshot)
            opposite_edge = self._downside_bias_edge(snapshot)
        elif side == SignalSide.SHORT:
            edge = self._downside_bias_edge(snapshot)
            opposite_edge = self._upside_bias_edge(snapshot)
        else:
            return 0.0

        separation = max(edge - opposite_edge, 0.0)
        pressure_abs = abs(self._clamp_signed(snapshot.liquidity_pressure_score))

        return max(
            0.0,
            1.20 * confidence
            + 0.75 * edge
            + 0.45 * separation
            + 0.30 * pressure_abs
            + 0.25 * self._target_distance_score(target, current_price)
        )

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

        distance_pct = self._distance_pct(ref_price, current_price)

        bonus = 0.0

        if 0.0010 <= distance_pct <= 0.0150:
            bonus += 0.14
        elif distance_pct <= 0.0300:
            bonus += 0.09
        elif distance_pct <= self.MAX_TARGET_DISTANCE_PCT:
            bonus += 0.04

        if isinstance(target, StopCluster):
            bonus += 0.32 * self._clamp01(target.confidence)
            bonus += 0.20 * self._clamp01(target.estimated_stop_density)

            strength = self._value(getattr(target, "strength", None))
            if strength == "medium":
                bonus += 0.04
            elif strength == "high":
                bonus += 0.07
            elif strength == "extreme":
                bonus += 0.10

        elif isinstance(target, LiquidityLevel):
            bonus += 0.32 * self._clamp01(target.confidence)

            if target.sweep_status == SweepStatus.PARTIALLY_SWEPT:
                bonus -= 0.03
            elif target.sweep_status == SweepStatus.SWEPT:
                bonus -= 0.10

        return self._clamp01(bonus)

    def _zone_alignment_bonus(
        self,
        snapshot: LiquidityMapSnapshot,
        side: LiquiditySide,
        current_price: float,
    ) -> float:
        zone = self._best_zone_for_side(
            snapshot=snapshot,
            side=side,
            current_price=current_price,
        )
        if zone is None:
            return 0.0

        return 0.06 * self._clamp01(zone.score)

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

        distance_pct = self._distance_pct(ref_price, current_price)

        if distance_pct <= 0.001:
            return 0.08
        if distance_pct <= 0.003:
            return 0.38
        if distance_pct <= 0.010:
            return 0.90
        if distance_pct <= 0.020:
            return 0.70
        if distance_pct <= 0.040:
            return 0.40
        if distance_pct <= self.MAX_TARGET_DISTANCE_PCT:
            return 0.18

        return 0.0

    # ------------------------------------------------------------------
    # Signal building
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        context: Any,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
        target: LiquidityLevel | StopCluster | None,
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
            evidence=target,
            setup_name="liquidity_map_directional_bias",
            extra={
                "side": self._value(side),
                "upside_bias_edge": self._upside_bias_edge(snapshot),
                "downside_bias_edge": self._downside_bias_edge(snapshot),
                "bias_edge_delta": self._upside_bias_edge(snapshot) - self._downside_bias_edge(snapshot),
                "analytics_signal_confidence": self._analytics_signal_confidence(snapshot),
                "target_price": self._reference_price(target),
                "target_type": self._target_type(target),
                "target_confidence": self._target_confidence(target),
                "target_distance_pct": (
                    self._distance_pct(self._reference_price(target), current_price)
                    if target is not None
                    else None
                ),
                "target_quality_bonus": self._target_quality_bonus(target, current_price),
                "strategy_weight": (
                    strategy_cfg.weight if strategy_cfg is not None else 1.0
                ),
                "strategy_semantics": "liquidity_map_directional_bias",
            },
        )

        signal = StrategySignal(
            symbol=snapshot.symbol,
            side=side,
            strategy_name=self.strategy_name,
            category=self.category,
            timeframe=snapshot.timeframe,
            setup_type=SetupType.CONTINUATION,
            timestamp=self._context_timestamp(context),
            confidence=confidence,
            score=score,
            strength=confidence_to_strength(confidence),
            confidence_grade=confidence_to_grade(confidence),
            status=SignalStatus.NEW,
            trigger_type=TriggerType.CONFIRMATION,
            origin=SignalOrigin.SINGLE_STRATEGY,
            priority=priority,
            entry_plan=entry_plan,
            exit_plan=exit_plan,
            invalidation_plan=invalidation_plan,
            execution_plan=execution_plan,
            regime=self._resolve_regime(context),
            metadata=analytics_metadata,
        )

        signal.add_reason(self._build_primary_reason(snapshot, side))
        signal.add_reason(
            f"Liquidity map bias edge: up={self._upside_bias_edge(snapshot):.3f}, "
            f"down={self._downside_bias_edge(snapshot):.3f}"
        )
        signal.add_reason(
            f"Signed liquidity pressure={snapshot.liquidity_pressure_score:.3f}"
        )

        if target is not None:
            signal.add_reason(self._build_target_reason(target))
        else:
            signal.add_reason(
                "No explicit primary target selected; signal is based on aggregate liquidity map bias"
            )

        if snapshot.signal is not None and getattr(snapshot.signal, "explanation", None):
            signal.add_reason(snapshot.signal.explanation)

        for confirmation in self._build_confirmations(
            snapshot=snapshot,
            side=side,
            current_price=current_price,
            target=target,
        ):
            signal.add_confirmation(confirmation)

        signal.add_source_feature("liquidity_map_snapshot")
        signal.add_source_feature("analytics.liquidity")
        signal.add_source_feature("analytics.liquidity.signal")
        signal.add_source_feature("liquidity.map.bias")

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
        target: LiquidityLevel | StopCluster | None,
    ) -> EntryPlan:
        notes = [
            "Directional bias signal from aggregate liquidity map",
            "Use as confluence input unless signal processor confirms trade execution path",
        ]

        if target is not None:
            notes.append(f"Nearest directional liquidity target at {self._reference_price(target):.6f}")

        entry_type = (
            getattr(self.config.builders, "default_entry_type", None)
            or EntryType.MARKET
        )

        return EntryPlan(
            entry_type=entry_type,
            price=current_price if entry_type == EntryType.LIMIT else None,
            timeout_seconds=getattr(self.config.runtime, "max_signal_age_seconds", 60),
            max_slippage_bps=6.0,
            confirmation_required=True,
            notes=notes,
            metadata={
                "entry_logic": "liquidity_map_directional_bias",
                "target_price": self._reference_price(target),
                "target_type": self._target_type(target),
                "bias_signal": True,
                "requires_confluence": True,
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
        stop_price = self._resolve_stop_price(
            side=side,
            current_price=current_price,
            invalidation_level=invalidation_level,
        )

        tp_levels: list[TargetPlan] = []

        if target_price is not None and target_price > 0:
            tp_levels.append(
                TargetPlan(
                    price=target_price,
                    size_fraction=1.0,
                    rr=self._compute_rr(
                        current_price=current_price,
                        stop_price=stop_price,
                        target_price=target_price,
                        side=side,
                    ),
                    label="directional_bias_liquidity_target",
                    metadata={
                        "source": "liquidity_map_bias_target",
                        "target_type": self._target_type(target),
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
                getattr(self.config.runtime, "max_signal_age_seconds", 60) * 4,
                120,
            ),
            partial_exit_enabled=False,
            metadata={
                "exit_logic": "bias_invalidated_or_target_reached",
                "primary_target_price": target_price,
                "stop_price": stop_price,
                "bias": self._value(snapshot.bias),
            },
        )

    def _build_invalidation_plan(
        self,
        side: SignalSide,
        current_price: float,
        invalidation_level: LiquidityLevel | StopCluster | None,
    ) -> InvalidationPlan:
        price = self._resolve_stop_price(
            side=side,
            current_price=current_price,
            invalidation_level=invalidation_level,
        )

        reason = (
            "Liquidity map bias invalidated by opposite-side liquidity reclaim"
            if bool(getattr(self.config.builders, "require_invalidation", True))
            else None
        )

        return InvalidationPlan(
            price=price,
            reason=reason,
            timeout_seconds=max(
                getattr(self.config.runtime, "max_signal_age_seconds", 60),
                60,
            ),
            conditions=[
                "liquidity_map_bias_flipped",
                "opposite_liquidity_pressure_domination",
                "snapshot_stale",
                "signal_age_expired",
            ],
            metadata={
                "invalidation_source": "opposite_liquidity_bias",
                "invalidation_price": price,
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
                "Generated by LiquidityMapBiasStrategy",
                "This is a directional liquidity bias signal, not standalone execution permission",
                "SignalProcessor / ConfluenceEngine should combine it with other analytics before risk confirmation",
                "Risk manager must validate exposure, drawdown, leverage and correlation constraints",
            ],
            metadata={
                "strategy_name": self.strategy_name,
                "category": self._value(self.category),
                "exchange": snapshot.exchange,
                "market_type": snapshot.market_type,
                "scope": snapshot.scope,
                "scope_key": snapshot.scope_key,
                "bias_signal": True,
                "requires_confluence": True,
            },
        )

    # ------------------------------------------------------------------
    # Reason / confirmation builders
    # ------------------------------------------------------------------

    def _build_primary_reason(
        self,
        snapshot: LiquidityMapSnapshot,
        side: SignalSide,
    ) -> str:
        if side == SignalSide.LONG:
            return (
                "Liquidity map directional bias favors upside: "
                f"bias={self._value(snapshot.bias)}, "
                f"above_score={snapshot.above_liquidity_score:.3f}, "
                f"magnet_up={self._magnet_score_up(snapshot):.3f}, "
                f"sweep_risk_up={self._sweep_risk_up(snapshot):.3f}, "
                f"pressure={snapshot.liquidity_pressure_score:.3f}"
            )

        return (
            "Liquidity map directional bias favors downside: "
            f"bias={self._value(snapshot.bias)}, "
            f"below_score={snapshot.below_liquidity_score:.3f}, "
            f"magnet_down={self._magnet_score_down(snapshot):.3f}, "
            f"sweep_risk_down={self._sweep_risk_down(snapshot):.3f}, "
            f"pressure={snapshot.liquidity_pressure_score:.3f}"
        )

    def _build_target_reason(
        self,
        target: LiquidityLevel | StopCluster,
    ) -> str:
        if isinstance(target, StopCluster):
            return (
                f"Directional bias target is stop cluster at {target.center_price:.6f} "
                f"(confidence={target.confidence:.3f}, "
                f"density={target.estimated_stop_density:.3f}, "
                f"strength={self._value(target.strength)})"
            )

        return (
            f"Directional bias target is liquidity level at {target.price:.6f} "
            f"(type={self._value(target.level_type)}, "
            f"confidence={target.confidence:.3f}, "
            f"sweep_status={self._value(target.sweep_status)})"
        )

    def _build_confirmations(
        self,
        snapshot: LiquidityMapSnapshot,
        side: SignalSide,
        current_price: float,
        target: LiquidityLevel | StopCluster | None,
    ) -> list[str]:
        confirmations: list[str] = []

        if side == SignalSide.LONG:
            if snapshot.bias == LiquidityBias.UP:
                confirmations.append("Snapshot bias is UP")

            if snapshot.liquidity_pressure_score > 0:
                confirmations.append("Signed liquidity pressure confirms upside")

            if self._upside_bias_edge(snapshot) > self._downside_bias_edge(snapshot):
                confirmations.append("Upside bias edge dominates downside edge")

            if self._magnet_score_up(snapshot) >= 0.60:
                confirmations.append("Upside magnet score is strong")

            if self._sweep_risk_up(snapshot) >= 0.60:
                confirmations.append("Upside sweep risk supports directional bias")

            if self._has_high_quality_zone(
                snapshot=snapshot,
                side=LiquiditySide.BUY_SIDE,
                current_price=current_price,
            ):
                confirmations.append("High-quality buy-side liquidity zone supports bias")

        elif side == SignalSide.SHORT:
            if snapshot.bias == LiquidityBias.DOWN:
                confirmations.append("Snapshot bias is DOWN")

            if snapshot.liquidity_pressure_score < 0:
                confirmations.append("Signed liquidity pressure confirms downside")

            if self._downside_bias_edge(snapshot) > self._upside_bias_edge(snapshot):
                confirmations.append("Downside bias edge dominates upside edge")

            if self._magnet_score_down(snapshot) >= 0.60:
                confirmations.append("Downside magnet score is strong")

            if self._sweep_risk_down(snapshot) >= 0.60:
                confirmations.append("Downside sweep risk supports directional bias")

            if self._has_high_quality_zone(
                snapshot=snapshot,
                side=LiquiditySide.SELL_SIDE,
                current_price=current_price,
            ):
                confirmations.append("High-quality sell-side liquidity zone supports bias")

        if target is not None:
            confirmations.append("Directional liquidity target is available")

        if self._analytics_signal_confidence(snapshot) >= 0.65:
            confirmations.append("Analytics liquidity signal confidence is strong")

        return confirmations

    def _has_high_quality_zone(
        self,
        snapshot: LiquidityMapSnapshot,
        side: LiquiditySide,
        current_price: float,
    ) -> bool:
        zone = self._best_zone_for_side(
            snapshot=snapshot,
            side=side,
            current_price=current_price,
        )
        return zone is not None and self._clamp01(zone.score) >= 0.65

    # ------------------------------------------------------------------
    # Invalidation / RR / priority helpers
    # ------------------------------------------------------------------

    def _invalidation_level_for_side(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
    ) -> LiquidityLevel | StopCluster | None:
        if side == SignalSide.LONG:
            candidates = self._collect_targets_below(snapshot, current_price)
            return candidates[0] if candidates else snapshot.nearest_below_level

        if side == SignalSide.SHORT:
            candidates = self._collect_targets_above(snapshot, current_price)
            return candidates[0] if candidates else snapshot.nearest_above_level

        return None

    def _resolve_stop_price(
        self,
        side: SignalSide,
        current_price: float,
        invalidation_level: LiquidityLevel | StopCluster | None,
    ) -> float:
        anchor_price = self._reference_price(invalidation_level)

        if side == SignalSide.LONG:
            if anchor_price > 0 and anchor_price < current_price:
                return anchor_price * self.LONG_STOP_OFFSET
            return current_price * (1.0 - self.FALLBACK_STOP_PCT)

        if side == SignalSide.SHORT:
            if anchor_price > 0 and anchor_price > current_price:
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

    def _target_type(self, target: LiquidityLevel | StopCluster | None) -> str | None:
        if target is None:
            return None

        if isinstance(target, StopCluster):
            return "stop_cluster"

        if isinstance(target, LiquidityLevel):
            return self._value(target.level_type)

        return target.__class__.__name__

    def _target_confidence(self, target: LiquidityLevel | StopCluster | None) -> float:
        if target is None:
            return 0.0

        return self._clamp01(getattr(target, "confidence", 0.0))