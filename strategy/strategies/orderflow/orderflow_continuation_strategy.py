from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from analytics.orderflow import OrderFlowAnalyzer
from core.event_bus import EventBus

from ...config import StrategyConfig
from ...enums import (
    EntryType,
    ExitType,
    MarketRegime,
    SignalOrigin,
    SignalSide,
    SignalStatus,
    StrategyCategory,
    Timeframe,
    TriggerType,
    SetupType,
)
from ...models import (
    EntryPlan,
    ExecutionPlanDraft,
    ExitPlan,
    InvalidationPlan,
    SignalContext,
    StrategyEvaluation,
    StrategySignal,
    TargetPlan,
)

from .base_orderflow_strategy import (
    OrderflowCompositeSnapshot,
    OrderflowStrategyBase,
)


@dataclass(slots=True)
class OrderflowContinuationThresholds:
    """
    Strategy-level thresholds for orderflow continuation.

    Analytics calculates facts. This strategy decides whether those facts show
    tradable continuation in the same direction as price movement.
    """

    min_trades_count: int = 10
    min_total_volume: float = 0.0

    min_abs_price_change_pct: float = 0.03

    min_cvd_delta_ratio: float = 0.08
    min_cvd_change_pct: float = 0.03
    min_cvd_slope: float = 0.0

    min_volume_delta_ratio: float = 0.10
    min_volume_delta_abs: float = 0.0
    min_cumulative_volume_delta_abs: float = 0.0

    min_aggressive_buy_ratio: float = 0.55
    min_aggressive_sell_ratio: float = 0.55
    min_aggressive_burst_score: float = 0.0
    min_large_aggressive_trades: int = 0

    min_orderbook_imbalance_ratio: float = 0.05

    min_score_for_signal: float = 0.45
    min_confidence_for_signal: float = 0.50

    preferred_entry_offset_pct: float = 0.0010
    stop_buffer_pct: float = 0.0030
    fallback_rr_ratio: float = 2.0
    max_expected_holding_seconds: int = 300

    def validate(self) -> None:
        if self.min_trades_count < 1:
            raise ValueError("min_trades_count must be >= 1")
        if self.min_total_volume < 0:
            raise ValueError("min_total_volume must be >= 0")
        if self.min_abs_price_change_pct < 0:
            raise ValueError("min_abs_price_change_pct must be >= 0")
        if self.min_cvd_delta_ratio < 0:
            raise ValueError("min_cvd_delta_ratio must be >= 0")
        if self.min_cvd_change_pct < 0:
            raise ValueError("min_cvd_change_pct must be >= 0")
        if self.min_cvd_slope < 0:
            raise ValueError("min_cvd_slope must be >= 0")
        if self.min_volume_delta_ratio < 0:
            raise ValueError("min_volume_delta_ratio must be >= 0")
        if self.min_volume_delta_abs < 0:
            raise ValueError("min_volume_delta_abs must be >= 0")
        if self.min_cumulative_volume_delta_abs < 0:
            raise ValueError("min_cumulative_volume_delta_abs must be >= 0")
        if not 0 <= self.min_aggressive_buy_ratio <= 1:
            raise ValueError("min_aggressive_buy_ratio must be between 0 and 1")
        if not 0 <= self.min_aggressive_sell_ratio <= 1:
            raise ValueError("min_aggressive_sell_ratio must be between 0 and 1")
        if self.min_aggressive_burst_score < 0:
            raise ValueError("min_aggressive_burst_score must be >= 0")
        if self.min_large_aggressive_trades < 0:
            raise ValueError("min_large_aggressive_trades must be >= 0")
        if self.min_orderbook_imbalance_ratio < 0:
            raise ValueError("min_orderbook_imbalance_ratio must be >= 0")
        if self.min_score_for_signal < 0:
            raise ValueError("min_score_for_signal must be >= 0")
        if not 0 <= self.min_confidence_for_signal <= 1:
            raise ValueError("min_confidence_for_signal must be between 0 and 1")
        if self.preferred_entry_offset_pct < 0:
            raise ValueError("preferred_entry_offset_pct must be >= 0")
        if self.stop_buffer_pct <= 0:
            raise ValueError("stop_buffer_pct must be > 0")
        if self.fallback_rr_ratio <= 0:
            raise ValueError("fallback_rr_ratio must be > 0")
        if self.max_expected_holding_seconds <= 0:
            raise ValueError("max_expected_holding_seconds must be > 0")


class OrderflowContinuationStrategy(OrderflowStrategyBase):
    """
    Orderflow continuation strategy.

    LONG continuation:
    - price already moves up;
    - CVD confirms upward flow;
    - volume delta confirms upward flow;
    - aggressive buyers dominate;
    - orderbook imbalance does not contradict / ideally supports continuation.

    SHORT continuation:
    - price already moves down;
    - CVD confirms downward flow;
    - volume delta confirms downward flow;
    - aggressive sellers dominate;
    - orderbook imbalance does not contradict / ideally supports continuation.
    """

    STRATEGY_NAME = "orderflow_continuation_strategy"
    CATEGORY = StrategyCategory.ORDERFLOW
    DEFAULT_TIMEFRAME = Timeframe.M1

    DEFAULT_REQUIRED_FEATURES = {
        "orderflow.cvd.delta_ratio",
        "orderflow.volume_delta.delta_ratio",
        "orderflow.aggressive_trades.buy_ratio",
        "orderflow.aggressive_trades.sell_ratio",
    }

    def __init__(
        self,
        config: StrategyConfig,
        *,
        orderflow_analyzer: OrderFlowAnalyzer | None = None,
        thresholds: OrderflowContinuationThresholds | None = None,
        event_bus: EventBus | None = None,
        logger: Any | None = None,
    ) -> None:
        super().__init__(
            config=config,
            orderflow_analyzer=orderflow_analyzer,
            event_bus=event_bus,
            logger=logger,
        )
        self.thresholds = thresholds or OrderflowContinuationThresholds()

        self.validate_config()
        self.thresholds.validate()

    @property
    def supported_regimes(self) -> set[MarketRegime]:
        return {
            MarketRegime.TRENDING_UP,
            MarketRegime.TRENDING_DOWN,
            MarketRegime.BREAKOUT,
            MarketRegime.SQUEEZE,
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.UNKNOWN,
        }

    def can_evaluate(self, context: SignalContext) -> bool:
        self.validate_context(context)

        if not self.is_enabled():
            return False

        if not self._runtime_allows_context(context):
            return False

        snapshot = self._resolve_snapshot(context)
        if snapshot is None or not snapshot.has_minimum_data():
            return False

        if snapshot.trades_count < self.thresholds.min_trades_count:
            return False

        if snapshot.total_volume < self.thresholds.min_total_volume:
            return False

        return True

    def evaluate(self, context: SignalContext) -> StrategyEvaluation:
        self.validate_context(context)

        evaluation = StrategyEvaluation(
            strategy_name=self.STRATEGY_NAME,
            symbol=context.symbol,
            timestamp=context.timestamp,
            passed=False,
            score=0.0,
            confidence=0.0,
        )

        if not self.can_evaluate(context):
            evaluation.reasons.append("strategy_cannot_evaluate_context")
            return evaluation

        snapshot = self._resolve_snapshot(context)
        if snapshot is None:
            evaluation.reasons.append("orderflow_snapshot_unavailable")
            return evaluation

        side = self._detect_continuation_side(snapshot)
        if side == SignalSide.UNKNOWN:
            evaluation.reasons.append("no_orderflow_continuation_detected")
            return evaluation

        score = self._calculate_score(snapshot, side, context)
        confidence = self._calculate_confidence(snapshot, side, context)
        reasons = self._build_reasons(snapshot, side)
        confirmations = self._build_confirmations(snapshot, side, context)

        evaluation.score = score
        evaluation.confidence = confidence
        evaluation.reasons.extend(reasons)

        min_score = max(self._get_min_score(), self.thresholds.min_score_for_signal)
        min_confidence = max(
            self._get_min_confidence(),
            self.thresholds.min_confidence_for_signal,
        )

        if score < min_score:
            evaluation.reasons.append("score_below_threshold")
            return evaluation

        if confidence < min_confidence:
            evaluation.reasons.append("confidence_below_threshold")
            return evaluation

        signal = self._build_signal(
            context=context,
            snapshot=snapshot,
            side=side,
            score=score,
            confidence=confidence,
            reasons=reasons,
            confirmations=confirmations,
        )

        evaluation.signal = signal
        evaluation.passed = True
        return evaluation

    def build_signal(self, context: SignalContext) -> StrategySignal | None:
        evaluation = self.evaluate(context)
        return evaluation.signal if evaluation.passed else None

    # ------------------------------------------------------------------
    # Snapshot resolution
    # ------------------------------------------------------------------

    def _resolve_snapshot(
        self,
        context: SignalContext,
    ) -> OrderflowCompositeSnapshot | None:
        return self._resolve_orderflow_composite_snapshot(context)

    # ------------------------------------------------------------------
    # Continuation detection
    # ------------------------------------------------------------------

    def _detect_continuation_side(
        self,
        snapshot: OrderflowCompositeSnapshot,
    ) -> SignalSide:
        long_ok = self._is_long_continuation(snapshot)
        short_ok = self._is_short_continuation(snapshot)

        if long_ok and not short_ok:
            return SignalSide.LONG

        if short_ok and not long_ok:
            return SignalSide.SHORT

        return SignalSide.UNKNOWN

    def _is_long_continuation(
        self,
        snapshot: OrderflowCompositeSnapshot,
    ) -> bool:
        return (
            snapshot.price_change_pct >= self.thresholds.min_abs_price_change_pct
            and snapshot.cvd_delta_ratio >= self.thresholds.min_cvd_delta_ratio
            and snapshot.cvd_change_pct >= self.thresholds.min_cvd_change_pct
            and snapshot.cvd_slope >= self.thresholds.min_cvd_slope
            and snapshot.volume_delta_ratio >= self.thresholds.min_volume_delta_ratio
            and snapshot.volume_delta >= self.thresholds.min_volume_delta_abs
            and snapshot.cumulative_volume_delta
            >= self.thresholds.min_cumulative_volume_delta_abs
            and snapshot.aggressive_buy_ratio >= self.thresholds.min_aggressive_buy_ratio
            and snapshot.aggressive_buy_ratio > snapshot.aggressive_sell_ratio
            and snapshot.aggressive_burst_score
            >= self.thresholds.min_aggressive_burst_score
            and snapshot.large_buy_trades >= self.thresholds.min_large_aggressive_trades
            and snapshot.signed_orderbook_imbalance
            >= -self.thresholds.min_orderbook_imbalance_ratio
        )

    def _is_short_continuation(
        self,
        snapshot: OrderflowCompositeSnapshot,
    ) -> bool:
        return (
            snapshot.price_change_pct <= -self.thresholds.min_abs_price_change_pct
            and snapshot.cvd_delta_ratio <= -self.thresholds.min_cvd_delta_ratio
            and snapshot.cvd_change_pct <= -self.thresholds.min_cvd_change_pct
            and snapshot.cvd_slope <= -self.thresholds.min_cvd_slope
            and snapshot.volume_delta_ratio <= -self.thresholds.min_volume_delta_ratio
            and snapshot.volume_delta <= -self.thresholds.min_volume_delta_abs
            and snapshot.cumulative_volume_delta
            <= -self.thresholds.min_cumulative_volume_delta_abs
            and snapshot.aggressive_sell_ratio >= self.thresholds.min_aggressive_sell_ratio
            and snapshot.aggressive_sell_ratio > snapshot.aggressive_buy_ratio
            and snapshot.aggressive_burst_score
            >= self.thresholds.min_aggressive_burst_score
            and snapshot.large_sell_trades >= self.thresholds.min_large_aggressive_trades
            and snapshot.signed_orderbook_imbalance
            <= self.thresholds.min_orderbook_imbalance_ratio
        )

    # ------------------------------------------------------------------
    # Scoring / confidence
    # ------------------------------------------------------------------

    def _calculate_score(
        self,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
        context: SignalContext,
    ) -> float:
        price_component = self._normalize_percent(
            abs(snapshot.price_change_pct),
            scale=1.5,
        )
        cvd_component = self._normalize_ratio(
            abs(snapshot.cvd_delta_ratio),
            scale=0.45,
        )
        cvd_change_component = self._normalize_percent(
            abs(snapshot.cvd_change_pct),
            scale=1.5,
        )
        volume_delta_component = self._normalize_ratio(
            abs(snapshot.volume_delta_ratio),
            scale=0.45,
        )
        cumulative_component = self._normalize_magnitude(
            abs(snapshot.cumulative_volume_delta),
            scale=max(abs(snapshot.total_volume), 1.0),
        )
        notional_component = self._normalize_ratio(
            abs(self._notional_delta_ratio(snapshot)),
            scale=0.45,
        )
        cumulative_notional_component = self._normalize_magnitude(
            abs(snapshot.cumulative_notional_delta),
            scale=max(abs(snapshot.total_notional), 1.0),
        )
        aggression_component = (
            snapshot.aggressive_buy_ratio
            if side == SignalSide.LONG
            else snapshot.aggressive_sell_ratio
        )
        aggressive_net_component = self._normalize_ratio(
            abs(self._directional_aggressive_notional_ratio(snapshot, side)),
            scale=0.45,
        )
        large_trade_component = self._large_trade_component(snapshot, side)
        orderbook_component = self._orderbook_component(snapshot, side)
        burst_component = self._normalize_ratio(
            snapshot.aggressive_burst_score,
            scale=1.0,
        )

        raw_score = (
            (price_component * 0.11)
            + (cvd_component * 0.14)
            + (cvd_change_component * 0.09)
            + (volume_delta_component * 0.13)
            + (cumulative_component * 0.07)
            + (notional_component * 0.10)
            + (cumulative_notional_component * 0.05)
            + (aggression_component * 0.12)
            + (aggressive_net_component * 0.08)
            + (large_trade_component * 0.04)
            + (orderbook_component * 0.04)
            + (burst_component * 0.03)
        )

        weighted_score = raw_score
        weighted_score *= self._category_weight()
        weighted_score *= self._regime_adjustment(context)
        weighted_score *= self._strategy_weight()

        return max(0.0, weighted_score)

    def _calculate_confidence(
        self,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
        context: SignalContext,
    ) -> float:
        components: list[float] = [
            self._normalize_percent(abs(snapshot.price_change_pct), scale=1.5),
            self._normalize_ratio(abs(snapshot.cvd_delta_ratio), scale=0.35),
            self._normalize_percent(abs(snapshot.cvd_change_pct), scale=1.25),
            self._normalize_ratio(abs(snapshot.volume_delta_ratio), scale=0.35),
            self._normalize_ratio(abs(self._notional_delta_ratio(snapshot)), scale=0.35),
            self._normalize_ratio(
                abs(self._directional_aggressive_notional_ratio(snapshot, side)),
                scale=0.35,
            ),
            self._large_trade_component(snapshot, side),
            self._orderbook_component(snapshot, side),
            min(
                snapshot.trades_count / max(self.thresholds.min_trades_count * 2, 1),
                1.0,
            ),
        ]

        if side == SignalSide.LONG:
            components.append(snapshot.aggressive_buy_ratio)
            components.append(1.0 if snapshot.large_buy_trades >= snapshot.large_sell_trades else 0.45)
            components.append(1.0 if snapshot.notional_delta >= 0 else 0.35)
            components.append(1.0 if snapshot.aggressive_net_notional_delta >= 0 else 0.35)

        elif side == SignalSide.SHORT:
            components.append(snapshot.aggressive_sell_ratio)
            components.append(1.0 if snapshot.large_sell_trades >= snapshot.large_buy_trades else 0.45)
            components.append(1.0 if snapshot.notional_delta <= 0 else 0.35)
            components.append(1.0 if snapshot.aggressive_net_notional_delta <= 0 else 0.35)

        if snapshot.has_orderbook:
            components.append(0.75)
            if snapshot.depth_levels_used > 0:
                components.append(0.75)

        if snapshot.spread is not None and snapshot.mid_price:
            spread_ratio = snapshot.spread / max(snapshot.mid_price, 1e-12)
            components.append(1.0 if spread_ratio <= 0.001 else 0.55)

        if context.price is not None and context.price.spread_bps is not None:
            spread_ok = context.price.spread_bps <= self.config.filters.max_spread_bps
            components.append(1.0 if spread_ok else 0.35)

        if context.regime is not None and context.regime.regime in self.supported_regimes:
            components.append(0.80)

        confidence = sum(components) / len(components) if components else 0.0
        return max(0.0, min(confidence, 1.0))

    # ------------------------------------------------------------------
    # Reasons / confirmations
    # ------------------------------------------------------------------

    def _build_reasons(
        self,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
    ) -> list[str]:
        reasons: list[str] = []

        if side == SignalSide.LONG:
            reasons.extend(
                [
                    "price_continuation_up",
                    "cvd_confirms_long_continuation",
                    "volume_delta_confirms_long_continuation",
                    "aggressive_buyers_dominate",
                ]
            )

            if snapshot.notional_delta > 0:
                reasons.append("positive_notional_delta_confirmation")
            if snapshot.cumulative_notional_delta > 0:
                reasons.append("positive_cumulative_notional_delta")
            if snapshot.aggressive_net_notional_delta > 0:
                reasons.append("aggressive_buy_notional_dominance")
            if snapshot.large_buy_trades >= self.thresholds.min_large_aggressive_trades:
                reasons.append("large_buy_trades_support_continuation")
            if snapshot.signed_orderbook_imbalance > 0:
                reasons.append("bid_side_orderbook_support")

        elif side == SignalSide.SHORT:
            reasons.extend(
                [
                    "price_continuation_down",
                    "cvd_confirms_short_continuation",
                    "volume_delta_confirms_short_continuation",
                    "aggressive_sellers_dominate",
                ]
            )

            if snapshot.notional_delta < 0:
                reasons.append("negative_notional_delta_confirmation")
            if snapshot.cumulative_notional_delta < 0:
                reasons.append("negative_cumulative_notional_delta")
            if snapshot.aggressive_net_notional_delta < 0:
                reasons.append("aggressive_sell_notional_dominance")
            if snapshot.large_sell_trades >= self.thresholds.min_large_aggressive_trades:
                reasons.append("large_sell_trades_support_continuation")
            if snapshot.signed_orderbook_imbalance < 0:
                reasons.append("ask_side_orderbook_support")

        if snapshot.aggressive_burst_score > 0:
            reasons.append("aggressive_flow_burst_present")

        if snapshot.trades_count >= self.thresholds.min_trades_count:
            reasons.append("sufficient_trade_sample")

        if snapshot.total_notional > 0:
            reasons.append("notional_data_available")

        if snapshot.has_orderbook:
            reasons.append("orderbook_context_available")

        return reasons

    def _build_confirmations(
        self,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
        context: SignalContext,
    ) -> list[str]:
        confirmations: list[str] = []

        if side == SignalSide.LONG:
            if snapshot.cvd_slope >= 0:
                confirmations.append("positive_cvd_slope")
            if snapshot.volume_delta > 0:
                confirmations.append("positive_volume_delta")
            if snapshot.notional_delta > 0:
                confirmations.append("positive_notional_delta")
            if snapshot.aggressive_buy_ratio > snapshot.aggressive_sell_ratio:
                confirmations.append("buy_aggression_dominance")
            if snapshot.large_buy_trades >= snapshot.large_sell_trades:
                confirmations.append("large_buy_trades_dominate")
            if snapshot.signed_orderbook_imbalance >= 0:
                confirmations.append("non_bearish_orderbook_imbalance")

        elif side == SignalSide.SHORT:
            if snapshot.cvd_slope <= 0:
                confirmations.append("negative_cvd_slope")
            if snapshot.volume_delta < 0:
                confirmations.append("negative_volume_delta")
            if snapshot.notional_delta < 0:
                confirmations.append("negative_notional_delta")
            if snapshot.aggressive_sell_ratio > snapshot.aggressive_buy_ratio:
                confirmations.append("sell_aggression_dominance")
            if snapshot.large_sell_trades >= snapshot.large_buy_trades:
                confirmations.append("large_sell_trades_dominate")
            if snapshot.signed_orderbook_imbalance <= 0:
                confirmations.append("non_bullish_orderbook_imbalance")

        if snapshot.depth_levels_used > 0:
            confirmations.append("orderbook_depth_available")

        if snapshot.spread is not None:
            confirmations.append("orderbook_spread_available")

        if context.price is not None and context.price.spread_bps is not None:
            if context.price.spread_bps <= self.config.filters.max_spread_bps:
                confirmations.append("spread_filter_ok")

        if context.regime is not None and context.regime.regime in self.supported_regimes:
            confirmations.append("regime_alignment_ok")

        return confirmations

    # ------------------------------------------------------------------
    # Signal build
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        *,
        context: SignalContext,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
        score: float,
        confidence: float,
        reasons: list[str],
        confirmations: list[str],
    ) -> StrategySignal:
        entry_plan = self._build_entry_plan(context, snapshot, side)
        exit_plan = self._build_exit_plan(context, snapshot, side, entry_plan)
        invalidation_plan = self._build_invalidation_plan(
            context,
            snapshot,
            side,
            entry_plan,
        )
        execution_plan = self._build_execution_plan(
            context=context,
            snapshot=snapshot,
            side=side,
            entry_plan=entry_plan,
            exit_plan=exit_plan,
            invalidation_plan=invalidation_plan,
        )

        signal = StrategySignal(
            symbol=snapshot.symbol,
            side=side,
            strategy_name=self.STRATEGY_NAME,
            category=self.CATEGORY,
            timeframe=context.timeframe or self.DEFAULT_TIMEFRAME,
            setup_type=SetupType.CONTINUATION,
            timestamp=context.timestamp,
            confidence=confidence,
            score=score,
            strength=self._map_strength(confidence),
            confidence_grade=self._map_confidence_grade(confidence),
            status=SignalStatus.NEW,
            trigger_type=TriggerType.PRIMARY,
            origin=SignalOrigin.SINGLE_STRATEGY,
            priority=self._resolve_priority(confidence),
            regime=(
                context.regime.regime
                if context.regime is not None
                else MarketRegime.UNKNOWN
            ),
            entry_plan=entry_plan,
            exit_plan=exit_plan,
            invalidation_plan=invalidation_plan,
            execution_plan=execution_plan,
            metadata={
                "source": self.STRATEGY_NAME,
                "analytics_fallback_enabled": self.orderflow_analyzer is not None,
                "analytics_metrics": [
                    "cvd",
                    "volume_delta",
                    "aggressive_trades",
                    "orderbook_imbalance",
                ],
                "scope": snapshot.scope,
                "scope_key": snapshot.scope_key,
                "key": list(snapshot.key),
                "orderflow_snapshot": snapshot.to_dict(),
            },
        )

        for reason in reasons:
            signal.add_reason(reason)

        for confirmation in confirmations:
            signal.add_confirmation(confirmation)

        for feature_name in self.required_features():
            signal.add_source_feature(feature_name)

        signal.add_source_feature("orderflow.cvd")
        signal.add_source_feature("orderflow.volume_delta")
        signal.add_source_feature("orderflow.aggressive_trades")
        signal.add_source_feature("orderflow.orderbook_imbalance")

        return signal

    # ------------------------------------------------------------------
    # Plans
    # ------------------------------------------------------------------

    def _build_entry_plan(
        self,
        context: SignalContext,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
    ) -> EntryPlan:
        ref_price = self._resolve_reference_price(context, snapshot)
        entry_price = None

        if ref_price is not None:
            offset = ref_price * self.thresholds.preferred_entry_offset_pct

            if side == SignalSide.LONG:
                entry_price = ref_price + offset
            elif side == SignalSide.SHORT:
                entry_price = ref_price - offset

        return EntryPlan(
            entry_type=getattr(
                self.config.builders,
                "default_entry_type",
                EntryType.MARKET,
            ),
            price=entry_price,
            confirmation_required=False,
            notes=[
                "entry_generated_from_orderflow_continuation",
                "prefer_execution_with_momentum_alignment",
            ],
            metadata={
                "reference_price": ref_price,
                "entry_offset_pct": self.thresholds.preferred_entry_offset_pct,
                "scope": snapshot.scope,
                "scope_key": snapshot.scope_key,
            },
        )

    def _build_exit_plan(
        self,
        context: SignalContext,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
        entry_plan: EntryPlan,
    ) -> ExitPlan:
        ref_price = entry_plan.price or self._resolve_reference_price(context, snapshot)

        stop_loss = None
        tp_levels: list[TargetPlan] = []

        rr_ratio = getattr(
            self.config.builders,
            "default_rr_ratio",
            self.thresholds.fallback_rr_ratio,
        )
        rr_ratio = rr_ratio if rr_ratio and rr_ratio > 0 else self.thresholds.fallback_rr_ratio

        if ref_price is not None:
            stop_buffer = ref_price * self.thresholds.stop_buffer_pct

            if side == SignalSide.LONG:
                stop_loss = ref_price - stop_buffer
                risk = max(ref_price - stop_loss, 0.0)
                tp_price = ref_price + (risk * rr_ratio)
            else:
                stop_loss = ref_price + stop_buffer
                risk = max(stop_loss - ref_price, 0.0)
                tp_price = ref_price - (risk * rr_ratio)

            if tp_price > 0:
                tp_levels.append(
                    TargetPlan(
                        price=tp_price,
                        size_fraction=1.0,
                        rr=rr_ratio,
                        label="tp1",
                    )
                )

        return ExitPlan(
            exit_types=[
                ExitType.STOP_LOSS,
                ExitType.TAKE_PROFIT,
                ExitType.INVALIDATION,
            ],
            stop_loss=stop_loss,
            take_profit_levels=tp_levels,
            partial_exit_enabled=getattr(
                self.config.builders,
                "enable_partial_take_profit",
                True,
            ),
            metadata={
                "strategy": self.STRATEGY_NAME,
                "rr_ratio": rr_ratio,
                "scope": snapshot.scope,
                "scope_key": snapshot.scope_key,
            },
        )

    def _build_invalidation_plan(
        self,
        context: SignalContext,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
        entry_plan: EntryPlan,
    ) -> InvalidationPlan:
        ref_price = entry_plan.price or self._resolve_reference_price(context, snapshot)
        invalidation_price = None

        if ref_price is not None:
            buffer = ref_price * self.thresholds.stop_buffer_pct

            if side == SignalSide.LONG:
                invalidation_price = ref_price - buffer
            elif side == SignalSide.SHORT:
                invalidation_price = ref_price + buffer

        return InvalidationPlan(
            price=invalidation_price,
            reason="orderflow_continuation_failed",
            conditions=[
                "cvd_delta_ratio_reverses",
                "volume_delta_ratio_reverses",
                "aggressive_flow_loses_directional_dominance",
                "notional_delta_reverses",
                "orderbook_imbalance_flips_against_continuation",
            ],
            metadata={
                "strategy": self.STRATEGY_NAME,
                "scope": snapshot.scope,
                "scope_key": snapshot.scope_key,
            },
        )

    def _build_execution_plan(
        self,
        *,
        context: SignalContext,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
        entry_plan: EntryPlan,
        exit_plan: ExitPlan,
        invalidation_plan: InvalidationPlan,
    ) -> ExecutionPlanDraft:
        return ExecutionPlanDraft(
            symbol=snapshot.symbol,
            side=side,
            entry=entry_plan,
            exit=exit_plan,
            invalidation=invalidation_plan,
            expected_holding_seconds=self.thresholds.max_expected_holding_seconds,
            notes=[
                "generated_from_orderflow_continuation_strategy",
            ],
            metadata={
                "strategy_name": self.STRATEGY_NAME,
                "timeframe": str(context.timeframe),
                "scope": snapshot.scope,
                "scope_key": snapshot.scope_key,
            },
        )

    # ------------------------------------------------------------------
    # Derived analytics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _notional_delta_ratio(snapshot: OrderflowCompositeSnapshot) -> float:
        if snapshot.total_notional <= 0:
            return 0.0
        return snapshot.notional_delta / snapshot.total_notional

    @staticmethod
    def _directional_aggressive_notional_ratio(
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
    ) -> float:
        total = snapshot.aggressive_buy_notional + snapshot.aggressive_sell_notional
        if total <= 0:
            return 0.0

        if side == SignalSide.LONG:
            return snapshot.aggressive_net_notional_delta / total

        if side == SignalSide.SHORT:
            return -snapshot.aggressive_net_notional_delta / total

        return 0.0

    def _large_trade_component(
        self,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
    ) -> float:
        required = max(self.thresholds.min_large_aggressive_trades, 1)

        if side == SignalSide.LONG:
            directional = snapshot.large_buy_trades
            opposite = snapshot.large_sell_trades
        elif side == SignalSide.SHORT:
            directional = snapshot.large_sell_trades
            opposite = snapshot.large_buy_trades
        else:
            return 0.0

        base = min(directional / required, 1.0)
        dominance = 1.0 if directional >= opposite else 0.5
        return max(0.0, min(base * dominance, 1.0))

    def _orderbook_component(
        self,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
    ) -> float:
        if not snapshot.has_orderbook:
            return 0.5

        signed = snapshot.signed_orderbook_imbalance
        scale = max(self.thresholds.min_orderbook_imbalance_ratio, 0.01)

        if side == SignalSide.LONG:
            if signed < -scale:
                return 0.0
            return 0.5 + (0.5 * self._normalize_ratio(max(signed, 0.0), scale=scale))

        if side == SignalSide.SHORT:
            if signed > scale:
                return 0.0
            return 0.5 + (0.5 * self._normalize_ratio(abs(min(signed, 0.0)), scale=scale))

        return 0.0