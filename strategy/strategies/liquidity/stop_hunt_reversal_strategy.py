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
    StopCluster,
)

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
    Production-ready stop-hunt reversal strategy.

    Семантика:
    - LONG: sell-side liquidity була swept / partially swept нижче current_price,
      після чого ціна reclaim-нулася вище sweep reference.
    - SHORT: buy-side liquidity була swept / partially swept вище current_price,
      після чого ціна rejected нижче sweep reference.

    Важливо:
    - unswept liquidity НЕ є достатньою підставою для reversal signal;
    - потрібен swept або partially swept level/cluster;
    - strategy не викликає analytics detectors;
    - strategy не читає raw market data;
    - strategy не виконує order execution;
    - full scope/futures/freshness перевіряються в BaseLiquidityStrategy.
    """

    MIN_EDGE: float = 0.18
    MIN_RECLAIM_SCORE: float = 0.04

    HIGH_PRIORITY_SCORE: float = 1.75
    HIGH_PRIORITY_CONFIDENCE: float = 0.85
    CRITICAL_PRIORITY_SCORE: float = 2.15
    CRITICAL_PRIORITY_CONFIDENCE: float = 0.90

    FALLBACK_STOP_PCT: float = 0.0045
    LONG_STOP_OFFSET: float = 0.9985
    SHORT_STOP_OFFSET: float = 1.0015

    MAX_EVIDENCE_DISTANCE_PCT: float = 0.0350
    MAX_TARGET_DISTANCE_PCT: float = 0.0700

    @property
    def strategy_name(self) -> str:
        return "stop_hunt_reversal_strategy"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, context: Any) -> StrategySignal | None:
        self.validate_context(context)

        if not self.is_enabled():
            self.log_debug(
                "StopHuntReversalStrategy skipped: disabled",
                symbol=getattr(context, "symbol", None),
                timeframe=self._value(getattr(context, "timeframe", None)),
            )
            return None

        snapshot = self._extract_snapshot(context)
        if snapshot is None:
            self.log_debug(
                "StopHuntReversalStrategy skipped: liquidity snapshot not found",
                symbol=getattr(context, "symbol", None),
                timeframe=self._value(getattr(context, "timeframe", None)),
            )
            return None

        if not self._base_context_is_valid(context, snapshot):
            return None

        current_price = self._resolve_current_price(context, snapshot)
        if current_price is None:
            self.log_warning(
                "StopHuntReversalStrategy skipped: current price unavailable",
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
                "StopHuntReversalStrategy blocked by filters",
                exchange=snapshot.exchange,
                market_type=snapshot.market_type,
                symbol=snapshot.symbol,
                timeframe=snapshot.timeframe,
                filters=[f"{item.name}:{self._value(item.decision)}" for item in filters],
            )
            return None

        candidate = self._find_reversal_candidate(
            snapshot=snapshot,
            current_price=current_price,
        )
        if candidate is None:
            self.log_debug(
                "StopHuntReversalStrategy skipped: no valid swept liquidity evidence",
                exchange=snapshot.exchange,
                market_type=snapshot.market_type,
                symbol=snapshot.symbol,
                timeframe=snapshot.timeframe,
            )
            return None

        side: SignalSide = candidate["side"]

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
                "StopHuntReversalStrategy skipped: below thresholds",
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
            side=side,
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

        swept_evidence = self._collect_swept_evidence(snapshot)

        if swept_evidence:
            results.append(
                FilterResult(
                    name="stop_hunt_swept_evidence",
                    decision="pass",
                    reason=f"Swept liquidity evidence found: {len(swept_evidence)}",
                )
            )
        else:
            results.append(
                FilterResult(
                    name="stop_hunt_swept_evidence",
                    decision="block",
                    reason="No swept or partially swept liquidity evidence",
                )
            )

        return results

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    def _find_reversal_candidate(
        self,
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

        return sell_side if float(sell_side["edge"]) >= float(buy_side["edge"]) else buy_side

    def _find_sell_side_stop_hunt(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> dict[str, Any] | None:
        """
        LONG reversal after sell-side liquidity sweep.

        Потрібно:
        - swept/partially swept sell-side level нижче current_price; або
        - swept sell-side cluster нижче current_price;
        - current_price має бути вище sweep reference.
        """
        swept_levels = [
            level
            for level in self._swept_levels(snapshot, side=LiquiditySide.SELL_SIDE)
            if level.price < current_price
            and self._evidence_distance_ok(level, current_price)
        ]

        swept_clusters = [
            cluster
            for cluster in self._swept_clusters(snapshot, side=LiquiditySide.SELL_SIDE)
            if cluster.center_price < current_price
            and self._evidence_distance_ok(cluster, current_price)
        ]

        if not swept_levels and not swept_clusters:
            return None

        level = self._pick_best_level(swept_levels, current_price)
        swept_cluster = self._pick_best_cluster(swept_clusters, current_price)

        evidence = self._pick_best_evidence(
            level=level,
            cluster=swept_cluster,
            current_price=current_price,
        )
        if evidence is None:
            return None

        reference_price = self._reference_price(evidence)

        reclaim_score = self._reclaim_score_from_reference(
            current_price=current_price,
            reference_price=reference_price,
            side=SignalSide.LONG,
        )
        if reclaim_score < self.MIN_RECLAIM_SCORE:
            return None

        level_score = self._level_reversal_score(level, current_price)
        cluster_score = self._cluster_reversal_score(swept_cluster, current_price)

        pressure_bonus = max(-self._clamp_signed(snapshot.liquidity_pressure_score), 0.0) * 0.18
        anti_bias_bonus = 0.24 if snapshot.bias == LiquidityBias.DOWN else 0.0
        sweep_risk_bonus = self._sweep_risk_down(snapshot) * 0.16
        magnet_bonus = self._magnet_score_up(snapshot) * 0.08

        edge = (
            max(level_score, cluster_score)
            + reclaim_score
            + pressure_bonus
            + anti_bias_bonus
            + sweep_risk_bonus
            + magnet_bonus
        )
        if edge <= self.MIN_EDGE:
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
            "swept_cluster": swept_cluster,
            "support_cluster": swept_cluster,
            "resistance_cluster": None,
            "evidence": evidence,
            "evidence_type": self._evidence_type(evidence),
            "reference_price": reference_price,
            "target": target,
            "edge": max(0.0, edge),
            "has_swept_level": level is not None,
            "has_swept_cluster": swept_cluster is not None,
            "reclaim_score": reclaim_score,
            "level_score": level_score,
            "cluster_score": cluster_score,
            "pressure_bonus": pressure_bonus,
            "anti_bias_bonus": anti_bias_bonus,
            "sweep_risk_bonus": sweep_risk_bonus,
            "magnet_bonus": magnet_bonus,
        }

    def _find_buy_side_stop_hunt(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> dict[str, Any] | None:
        """
        SHORT reversal after buy-side liquidity sweep.

        Потрібно:
        - swept/partially swept buy-side level вище current_price; або
        - swept buy-side cluster вище current_price;
        - current_price має бути нижче sweep reference.
        """
        swept_levels = [
            level
            for level in self._swept_levels(snapshot, side=LiquiditySide.BUY_SIDE)
            if level.price > current_price
            and self._evidence_distance_ok(level, current_price)
        ]

        swept_clusters = [
            cluster
            for cluster in self._swept_clusters(snapshot, side=LiquiditySide.BUY_SIDE)
            if cluster.center_price > current_price
            and self._evidence_distance_ok(cluster, current_price)
        ]

        if not swept_levels and not swept_clusters:
            return None

        level = self._pick_best_level(swept_levels, current_price)
        swept_cluster = self._pick_best_cluster(swept_clusters, current_price)

        evidence = self._pick_best_evidence(
            level=level,
            cluster=swept_cluster,
            current_price=current_price,
        )
        if evidence is None:
            return None

        reference_price = self._reference_price(evidence)

        reclaim_score = self._reclaim_score_from_reference(
            current_price=current_price,
            reference_price=reference_price,
            side=SignalSide.SHORT,
        )
        if reclaim_score < self.MIN_RECLAIM_SCORE:
            return None

        level_score = self._level_reversal_score(level, current_price)
        cluster_score = self._cluster_reversal_score(swept_cluster, current_price)

        pressure_bonus = max(self._clamp_signed(snapshot.liquidity_pressure_score), 0.0) * 0.18
        anti_bias_bonus = 0.24 if snapshot.bias == LiquidityBias.UP else 0.0
        sweep_risk_bonus = self._sweep_risk_up(snapshot) * 0.16
        magnet_bonus = self._magnet_score_down(snapshot) * 0.08

        edge = (
            max(level_score, cluster_score)
            + reclaim_score
            + pressure_bonus
            + anti_bias_bonus
            + sweep_risk_bonus
            + magnet_bonus
        )
        if edge <= self.MIN_EDGE:
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
            "swept_cluster": swept_cluster,
            "support_cluster": None,
            "resistance_cluster": swept_cluster,
            "evidence": evidence,
            "evidence_type": self._evidence_type(evidence),
            "reference_price": reference_price,
            "target": target,
            "edge": max(0.0, edge),
            "has_swept_level": level is not None,
            "has_swept_cluster": swept_cluster is not None,
            "reclaim_score": reclaim_score,
            "level_score": level_score,
            "cluster_score": cluster_score,
            "pressure_bonus": pressure_bonus,
            "anti_bias_bonus": anti_bias_bonus,
            "sweep_risk_bonus": sweep_risk_bonus,
            "magnet_bonus": magnet_bonus,
        }

    # ------------------------------------------------------------------
    # Evidence collection
    # ------------------------------------------------------------------

    def _collect_swept_evidence(
        self,
        snapshot: LiquidityMapSnapshot,
    ) -> list[LiquidityLevel | StopCluster]:
        return [
            *self._swept_levels(snapshot, side=LiquiditySide.BUY_SIDE),
            *self._swept_levels(snapshot, side=LiquiditySide.SELL_SIDE),
            *self._swept_clusters(snapshot, side=LiquiditySide.BUY_SIDE),
            *self._swept_clusters(snapshot, side=LiquiditySide.SELL_SIDE),
        ]

    def _swept_levels(
        self,
        snapshot: LiquidityMapSnapshot,
        side: LiquiditySide,
    ) -> list[LiquidityLevel]:
        candidates: list[LiquidityLevel] = []

        for level in [*snapshot.equal_levels, *snapshot.active_levels]:
            if level.side != side:
                continue

            if not self._is_valid_swept_level(level):
                continue

            candidates.append(level)

        return self._dedupe_levels(candidates)

    def _swept_clusters(
        self,
        snapshot: LiquidityMapSnapshot,
        side: LiquiditySide,
    ) -> list[StopCluster]:
        getter = getattr(snapshot, "get_swept_clusters", None)
        if callable(getter):
            try:
                raw_clusters = list(getter())
            except Exception:
                raw_clusters = []
        else:
            raw_clusters = [
                cluster
                for cluster in snapshot.stop_clusters
                if self._cluster_is_swept(cluster)
            ]

        return [
            cluster
            for cluster in raw_clusters
            if cluster.side == side and self._cluster_is_swept(cluster)
        ]

    def _dedupe_levels(
        self,
        levels: list[LiquidityLevel],
    ) -> list[LiquidityLevel]:
        result: dict[str, LiquidityLevel] = {}

        for level in levels:
            existing = result.get(level.key)
            if existing is None:
                result[level.key] = level
                continue

            if self._level_evidence_rank(level) > self._level_evidence_rank(existing):
                result[level.key] = level

        return list(result.values())

    def _level_evidence_rank(
        self,
        level: LiquidityLevel,
    ) -> tuple[int, int, float, int, int]:
        if level.is_swept():
            sweep_rank = 3
        elif level.is_partially_swept():
            sweep_rank = 2
        elif level.is_active():
            sweep_rank = 1
        else:
            sweep_rank = 0

        explicit_sweep_rank = 1 if level.swept_at is not None else 0

        return (
            sweep_rank,
            explicit_sweep_rank,
            self._clamp01(level.confidence),
            int(level.touches_count),
            int(level.reaction_count),
        )

    def _is_valid_swept_level(self, level: LiquidityLevel) -> bool:
        if level.is_invalidated() or level.is_expired():
            return False

        return level.sweep_status in {
            SweepStatus.SWEPT,
            SweepStatus.PARTIALLY_SWEPT,
        }

    def _evidence_distance_ok(
        self,
        item: LiquidityLevel | StopCluster,
        current_price: float,
    ) -> bool:
        reference_price = self._reference_price(item)
        if reference_price <= 0 or current_price <= 0:
            return False

        return self._distance_pct(reference_price, current_price) <= self.MAX_EVIDENCE_DISTANCE_PCT

    # ------------------------------------------------------------------
    # Ranking / scoring
    # ------------------------------------------------------------------

    def _pick_best_level(
        self,
        levels: list[LiquidityLevel],
        current_price: float,
    ) -> LiquidityLevel | None:
        if not levels:
            return None

        return max(
            levels,
            key=lambda level: self._level_reversal_score(level, current_price),
        )

    def _pick_best_cluster(
        self,
        clusters: list[StopCluster],
        current_price: float,
    ) -> StopCluster | None:
        if not clusters:
            return None

        return max(
            clusters,
            key=lambda cluster: self._cluster_reversal_score(cluster, current_price),
        )

    def _pick_best_evidence(
        self,
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

        level_score = self._level_reversal_score(level, current_price)
        cluster_score = self._cluster_reversal_score(cluster, current_price)

        return level if level_score >= cluster_score else cluster

    def _level_reversal_score(
        self,
        level: LiquidityLevel | None,
        current_price: float,
    ) -> float:
        if level is None or not self._is_valid_swept_level(level):
            return 0.0

        score = self._clamp01(level.confidence) * 0.38
        score += min(max(level.touches_count, 0) / 6.0, 1.0) * 0.14
        score += min(max(level.reaction_count, 0) / 4.0, 1.0) * 0.10
        score += self._evidence_distance_bonus(level, current_price)

        if level.sweep_status == SweepStatus.SWEPT:
            score += 0.18
        elif level.sweep_status == SweepStatus.PARTIALLY_SWEPT:
            score += 0.11

        if level.level_type in {
            LiquidityLevelType.EQUAL_HIGHS,
            LiquidityLevelType.EQUAL_LOWS,
        }:
            score += 0.08

        if level.swept_at is not None:
            score += 0.04

        return max(0.0, score)

    def _cluster_reversal_score(
        self,
        cluster: StopCluster | None,
        current_price: float,
    ) -> float:
        if cluster is None or not self._cluster_is_swept(cluster):
            return 0.0

        score = self._clamp01(cluster.confidence) * 0.34
        score += self._clamp01(cluster.estimated_stop_density) * 0.22
        score += min(max(cluster.touches_count, 0) / 6.0, 1.0) * 0.10
        score += self._evidence_distance_bonus(cluster, current_price)

        strength_value = self._value(getattr(cluster, "strength", None))
        if strength_value == "medium":
            score += 0.04
        elif strength_value == "high":
            score += 0.08
        elif strength_value == "extreme":
            score += 0.12

        if getattr(cluster, "swept_at", None) is not None:
            score += 0.04

        return max(0.0, score)

    def _evidence_distance_bonus(
        self,
        item: LiquidityLevel | StopCluster,
        current_price: float,
    ) -> float:
        ref_price = self._reference_price(item)
        if current_price <= 0 or ref_price <= 0:
            return 0.0

        distance_pct = self._distance_pct(ref_price, current_price)

        if distance_pct <= 0.0015:
            return 0.20
        if distance_pct <= 0.0040:
            return 0.17
        if distance_pct <= 0.0100:
            return 0.12
        if distance_pct <= 0.0200:
            return 0.07
        if distance_pct <= self.MAX_EVIDENCE_DISTANCE_PCT:
            return 0.03

        return 0.0

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
        elif side == SignalSide.SHORT:
            if current_price >= reference_price:
                return 0.0
            reclaim_pct = (reference_price - current_price) / reference_price
        else:
            return 0.0

        if reclaim_pct <= 0.0005:
            return 0.06
        if reclaim_pct <= 0.0020:
            return 0.16
        if reclaim_pct <= 0.0060:
            return 0.22
        if reclaim_pct <= 0.0150:
            return 0.15

        return 0.08

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------

    def _nearest_opposite_target(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
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
    # Confidence / score
    # ------------------------------------------------------------------

    def _compute_confidence(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        candidate: dict[str, Any],
    ) -> float:
        side: SignalSide = candidate["side"]
        edge = self._safe_float(candidate.get("edge"), 0.0) or 0.0

        hunted_level = candidate.get("hunted_level")
        swept_cluster = candidate.get("swept_cluster")
        target = candidate.get("target")

        confidence = min(edge, 1.0) * 0.46

        if hunted_level is not None:
            confidence += self._level_reversal_score(hunted_level, current_price) * 0.18

        if swept_cluster is not None:
            confidence += self._cluster_reversal_score(swept_cluster, current_price) * 0.16

        if target is not None:
            confidence += self._target_quality_bonus(target, current_price) * 0.08

        if side == SignalSide.LONG:
            confidence += self._reversal_zone_bonus(
                snapshot=snapshot,
                side=LiquiditySide.SELL_SIDE,
                current_price=current_price,
            )
            confidence += self._sweep_risk_down(snapshot) * 0.06

            if snapshot.bias == LiquidityBias.DOWN:
                confidence += 0.04

        elif side == SignalSide.SHORT:
            confidence += self._reversal_zone_bonus(
                snapshot=snapshot,
                side=LiquiditySide.BUY_SIDE,
                current_price=current_price,
            )
            confidence += self._sweep_risk_up(snapshot) * 0.06

            if snapshot.bias == LiquidityBias.UP:
                confidence += 0.04

        if snapshot.signal is not None:
            confidence += self._clamp01(getattr(snapshot.signal, "confidence", 0.0)) * 0.06

        return self._clamp01(confidence)

    def _compute_score(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        candidate: dict[str, Any],
        confidence: float,
    ) -> float:
        side: SignalSide = candidate["side"]
        edge = self._safe_float(candidate.get("edge"), 0.0) or 0.0
        target = candidate.get("target")

        anti_bias_bonus = 0.0
        if side == SignalSide.LONG and snapshot.bias == LiquidityBias.DOWN:
            anti_bias_bonus = 0.26
        elif side == SignalSide.SHORT and snapshot.bias == LiquidityBias.UP:
            anti_bias_bonus = 0.26

        return max(
            0.0,
            confidence * 1.18
            + edge * 0.88
            + self._target_distance_score(target, current_price) * 0.40
            + anti_bias_bonus
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

        distance_pct = self._distance_pct(ref_price, current_price)

        bonus = 0.0
        if 0.0010 <= distance_pct <= 0.0150:
            bonus += 0.14
        elif distance_pct <= 0.0300:
            bonus += 0.09
        elif distance_pct <= self.MAX_TARGET_DISTANCE_PCT:
            bonus += 0.04

        if isinstance(target, StopCluster):
            bonus += 0.30 * self._clamp01(target.confidence)
            bonus += 0.16 * self._clamp01(target.estimated_stop_density)
        elif isinstance(target, LiquidityLevel):
            bonus += 0.30 * self._clamp01(target.confidence)

        return self._clamp01(bonus)

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
            return 0.10
        if distance_pct <= 0.003:
            return 0.40
        if distance_pct <= 0.010:
            return 1.00
        if distance_pct <= 0.020:
            return 0.72
        if distance_pct <= 0.040:
            return 0.38
        if distance_pct <= self.MAX_TARGET_DISTANCE_PCT:
            return 0.15

        return 0.0

    def _reversal_zone_bonus(
        self,
        snapshot: LiquidityMapSnapshot,
        side: LiquiditySide,
        current_price: float,
    ) -> float:
        """
        Для LONG після sell-side hunt корисна якісна sell-side zone під ціною.
        Для SHORT після buy-side hunt корисна якісна buy-side zone над ціною.
        """
        zones = [
            zone
            for zone in self._directional_zones(snapshot, side)
            if (
                side == LiquiditySide.SELL_SIDE
                and zone.center_price < current_price
            )
            or (
                side == LiquiditySide.BUY_SIDE
                and zone.center_price > current_price
            )
        ]

        if not zones:
            return 0.0

        best = max(zones, key=lambda zone: self._clamp01(zone.score))
        return 0.06 * self._clamp01(best.score)

    # ------------------------------------------------------------------
    # Signal building
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        context: Any,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        candidate: dict[str, Any],
        side: SignalSide,
        confidence: float,
        score: float,
        filters: list[FilterResult],
    ) -> StrategySignal:
        evidence = candidate.get("evidence")
        target = candidate.get("target")

        entry_plan = self._build_entry_plan(
            side=side,
            current_price=current_price,
            evidence=evidence,
        )
        exit_plan = self._build_exit_plan(
            side=side,
            current_price=current_price,
            target=target,
            invalidation_anchor=evidence,
            snapshot=snapshot,
        )
        invalidation_plan = self._build_invalidation_plan(
            side=side,
            current_price=current_price,
            invalidation_anchor=evidence,
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
            evidence=evidence,
            setup_name="stop_hunt_reversal",
            extra={
                "side": self._value(side),
                "direction": candidate.get("direction"),
                "evidence_type": candidate.get("evidence_type"),
                "reference_price": candidate.get("reference_price"),
                "edge": candidate.get("edge"),
                "reclaim_score": candidate.get("reclaim_score"),
                "level_score": candidate.get("level_score"),
                "cluster_score": candidate.get("cluster_score"),
                "pressure_bonus": candidate.get("pressure_bonus"),
                "anti_bias_bonus": candidate.get("anti_bias_bonus"),
                "sweep_risk_bonus": candidate.get("sweep_risk_bonus"),
                "magnet_bonus": candidate.get("magnet_bonus"),
                "has_swept_level": candidate.get("has_swept_level"),
                "has_swept_cluster": candidate.get("has_swept_cluster"),
                "hunted_level": self._to_payload(candidate.get("hunted_level")),
                "swept_cluster": self._to_payload(candidate.get("swept_cluster")),
                "target_price": self._reference_price(target),
                "target_type": self._target_type(target),
                "target_confidence": self._target_confidence(target),
                "target_distance_pct": (
                    self._distance_pct(self._reference_price(target), current_price)
                    if target is not None
                    else None
                ),
                "strategy_weight": (
                    strategy_cfg.weight if strategy_cfg is not None else 1.0
                ),
                "strategy_semantics": "stop_hunt_reversal",
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
            f"Liquidity pressure score = {snapshot.liquidity_pressure_score:.3f}"
        )

        if snapshot.signal is not None and getattr(snapshot.signal, "explanation", None):
            signal.add_reason(snapshot.signal.explanation)

        for confirmation in self._build_confirmations(
            snapshot=snapshot,
            candidate=candidate,
            current_price=current_price,
        ):
            signal.add_confirmation(confirmation)

        signal.add_source_feature("liquidity_map_snapshot")
        signal.add_source_feature("analytics.liquidity")
        signal.add_source_feature("analytics.liquidity.signal")
        signal.add_source_feature("liquidity.stop_hunt_reversal")
        signal.add_source_feature("liquidity.swept_evidence")

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
        evidence: LiquidityLevel | StopCluster | None,
    ) -> EntryPlan:
        notes = ["Enter after swept liquidity reclaim / rejection confirmation"]

        if evidence is not None:
            notes.append(
                f"Swept evidence reference at {self._reference_price(evidence):.6f}"
            )

        entry_type = (
            getattr(self.config.builders, "default_entry_type", None)
            or EntryType.MARKET
        )

        return EntryPlan(
            entry_type=entry_type,
            price=current_price if entry_type == EntryType.LIMIT else None,
            timeout_seconds=getattr(self.config.runtime, "max_signal_age_seconds", 60),
            max_slippage_bps=10.0,
            confirmation_required=False,
            notes=notes,
            metadata={
                "entry_logic": "stop_hunt_reversal_reclaim",
                "evidence_price": self._reference_price(evidence),
                "evidence_type": self._evidence_type(evidence),
            },
        )

    def _build_exit_plan(
        self,
        side: SignalSide,
        current_price: float,
        target: LiquidityLevel | StopCluster | None,
        invalidation_anchor: LiquidityLevel | StopCluster | None,
        snapshot: LiquidityMapSnapshot,
    ) -> ExitPlan:
        target_price = self._reference_price(target) if target is not None else None
        stop_price = self._resolve_stop_price(
            side=side,
            current_price=current_price,
            invalidation_anchor=invalidation_anchor,
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
                    label="reversal_primary_liquidity_target",
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
                    label="reversal_extended_liquidity_target",
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
                "exit_logic": "stop_hunt_reversal_to_opposite_liquidity",
                "primary_target_price": target_price,
                "secondary_target_price": secondary_price,
                "stop_price": stop_price,
            },
        )

    def _build_invalidation_plan(
        self,
        side: SignalSide,
        current_price: float,
        invalidation_anchor: LiquidityLevel | StopCluster | None,
    ) -> InvalidationPlan:
        price = self._resolve_stop_price(
            side=side,
            current_price=current_price,
            invalidation_anchor=invalidation_anchor,
        )

        reason = None
        if bool(getattr(self.config.builders, "require_invalidation", True)):
            reason = (
                "Swept sell-side liquidity failed to hold as support"
                if side == SignalSide.LONG
                else "Swept buy-side liquidity failed to hold as resistance"
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
                "reversal_failed_reclaim",
                "opposite_liquidity_pressure_domination",
                "swept_evidence_invalidated",
                "snapshot_stale",
            ],
            metadata={
                "invalidation_source": "stop_hunt_reversal_anchor",
                "invalidation_price": price,
                "anchor_price": self._reference_price(invalidation_anchor),
                "anchor_type": self._evidence_type(invalidation_anchor),
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
                "Generated by StopHuntReversalStrategy",
                "Signal requires swept/partially swept liquidity evidence",
                "Risk manager must validate portfolio, leverage, drawdown and correlation constraints before execution",
                "Execution should only consume risk-confirmed signals",
            ],
            metadata={
                "strategy_name": self.strategy_name,
                "category": self._value(self.category),
                "exchange": snapshot.exchange,
                "market_type": snapshot.market_type,
                "scope": snapshot.scope,
                "scope_key": snapshot.scope_key,
                "strategy_semantics": "stop_hunt_reversal",
            },
        )

    # ------------------------------------------------------------------
    # Reason / confirmation builders
    # ------------------------------------------------------------------

    def _build_primary_reason(
        self,
        candidate: dict[str, Any],
        current_price: float,
    ) -> str:
        side: SignalSide = candidate["side"]
        hunted_level = candidate.get("hunted_level")
        swept_cluster = candidate.get("swept_cluster")
        edge = self._safe_float(candidate.get("edge"), 0.0) or 0.0

        parts = [f"stop_hunt_edge={edge:.3f}"]

        if hunted_level is not None:
            parts.append(
                f"hunted_level={hunted_level.price:.6f} "
                f"({self._value(hunted_level.level_type)}, "
                f"{self._value(hunted_level.sweep_status)})"
            )

        if swept_cluster is not None:
            parts.append(
                f"swept_cluster={swept_cluster.center_price:.6f} "
                f"(conf={swept_cluster.confidence:.3f}, "
                f"density={swept_cluster.estimated_stop_density:.3f}, "
                f"strength={self._value(swept_cluster.strength)})"
            )

        parts.append(f"current_price={current_price:.6f}")

        prefix = (
            "Sell-side stop hunt reclaimed -> long reversal"
            if side == SignalSide.LONG
            else "Buy-side stop hunt rejected -> short reversal"
        )

        return f"{prefix}: {', '.join(parts)}"

    def _build_target_reason(
        self,
        target: LiquidityLevel | StopCluster | None,
    ) -> str:
        if target is None:
            return (
                "No explicit reversal target found; signal is based on reclaimed "
                "swept liquidity structure"
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
        candidate: dict[str, Any],
        current_price: float,
    ) -> list[str]:
        confirmations: list[str] = []

        side: SignalSide = candidate["side"]
        hunted_level = candidate.get("hunted_level")
        swept_cluster = candidate.get("swept_cluster")
        target = candidate.get("target")

        if hunted_level is not None:
            if hunted_level.sweep_status == SweepStatus.SWEPT:
                confirmations.append("Liquidity level fully swept")
            elif hunted_level.sweep_status == SweepStatus.PARTIALLY_SWEPT:
                confirmations.append("Liquidity level partially swept")

        if swept_cluster is not None:
            confirmations.append("Swept stop cluster confirms stop-hunt context")

        reclaim_score = self._safe_float(candidate.get("reclaim_score"), 0.0) or 0.0
        if reclaim_score >= 0.12:
            confirmations.append("Reclaim / rejection distance is meaningful")

        if side == SignalSide.LONG:
            if snapshot.bias == LiquidityBias.DOWN:
                confirmations.append("Prior downside liquidity bias supports sell-side hunt context")

            if snapshot.liquidity_pressure_score < 0:
                confirmations.append("Liquidity pressure was downside before long reversal")

            if self._sweep_risk_down(snapshot) >= 0.60:
                confirmations.append("Downside sweep risk was elevated")

            if self._magnet_score_up(snapshot) >= 0.50:
                confirmations.append("Upside magnet supports reversal follow-through")

        elif side == SignalSide.SHORT:
            if snapshot.bias == LiquidityBias.UP:
                confirmations.append("Prior upside liquidity bias supports buy-side hunt context")

            if snapshot.liquidity_pressure_score > 0:
                confirmations.append("Liquidity pressure was upside before short reversal")

            if self._sweep_risk_up(snapshot) >= 0.60:
                confirmations.append("Upside sweep risk was elevated")

            if self._magnet_score_down(snapshot) >= 0.50:
                confirmations.append("Downside magnet supports reversal follow-through")

        if target is not None:
            confirmations.append("Opposite-side liquidity target available")

        if snapshot.signal is not None and getattr(snapshot.signal, "confidence", 0.0) >= 0.65:
            confirmations.append("Analytics liquidity signal confidence is strong")

        if self._has_reversal_zone_confirmation(
            snapshot=snapshot,
            side=side,
            current_price=current_price,
        ):
            confirmations.append("Liquidity zone confirms stop-hunt reversal area")

        return confirmations

    def _has_reversal_zone_confirmation(
        self,
        snapshot: LiquidityMapSnapshot,
        side: SignalSide,
        current_price: float,
    ) -> bool:
        liquidity_side = (
            LiquiditySide.SELL_SIDE
            if side == SignalSide.LONG
            else LiquiditySide.BUY_SIDE
        )

        zones = [
            zone
            for zone in self._directional_zones(snapshot, liquidity_side)
            if (
                side == SignalSide.LONG
                and zone.center_price < current_price
            )
            or (
                side == SignalSide.SHORT
                and zone.center_price > current_price
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
        invalidation_anchor: LiquidityLevel | StopCluster | None,
    ) -> float:
        anchor_price = self._reference_price(invalidation_anchor)

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

    def _evidence_type(
        self,
        evidence: LiquidityLevel | StopCluster | None,
    ) -> str | None:
        if evidence is None:
            return None

        if isinstance(evidence, StopCluster):
            return "stop_cluster"

        if isinstance(evidence, LiquidityLevel):
            return self._value(evidence.level_type)

        return evidence.__class__.__name__

    def _target_type(
        self,
        target: LiquidityLevel | StopCluster | None,
    ) -> str | None:
        return self._evidence_type(target)

    def _target_confidence(
        self,
        target: LiquidityLevel | StopCluster | None,
    ) -> float:
        if target is None:
            return 0.0

        return self._clamp01(getattr(target, "confidence", 0.0))