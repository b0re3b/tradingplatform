from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from analytics.orderflow import CvdStats, OrderFlowAnalyzer
from analytics.orderflow.models import (
    OrderFlowKey,
    orderflow_key_to_dict,
    orderflow_key_to_string,
)
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

from .base_orderflow_strategy import OrderflowStrategyBase


@dataclass(slots=True)
class CvdDivergenceThresholds:
    """
    Strategy-level thresholds for CVD divergence logic.

    These thresholds are not analytics config duplicates.
    Analytics calculates CVD facts. Strategy decides whether those facts are
    tradable enough to produce a StrategySignal.
    """

    min_abs_price_change_pct: float = 0.05
    min_abs_cvd_change_pct: float = 0.05
    min_abs_delta_ratio: float = 0.08
    min_abs_cvd_slope: float = 0.0

    min_trades_count: int = 12
    min_strength_for_signal: float = 0.25

    bullish_divergence_score_threshold: float = 0.55
    bearish_divergence_score_threshold: float = 0.55

    max_entry_offset_pct: float = 0.0015
    default_stop_buffer_pct: float = 0.0035
    default_tp_rr: float = 2.0
    max_expected_holding_seconds: int = 300

    def validate(self) -> None:
        if self.min_abs_price_change_pct < 0:
            raise ValueError("min_abs_price_change_pct must be >= 0")
        if self.min_abs_cvd_change_pct < 0:
            raise ValueError("min_abs_cvd_change_pct must be >= 0")
        if self.min_abs_delta_ratio < 0:
            raise ValueError("min_abs_delta_ratio must be >= 0")
        if self.min_abs_cvd_slope < 0:
            raise ValueError("min_abs_cvd_slope must be >= 0")
        if self.min_trades_count < 1:
            raise ValueError("min_trades_count must be >= 1")
        if not 0.0 <= self.min_strength_for_signal <= 1.0:
            raise ValueError("min_strength_for_signal must be between 0.0 and 1.0")
        if self.bullish_divergence_score_threshold < 0:
            raise ValueError("bullish_divergence_score_threshold must be >= 0")
        if self.bearish_divergence_score_threshold < 0:
            raise ValueError("bearish_divergence_score_threshold must be >= 0")
        if self.max_entry_offset_pct < 0:
            raise ValueError("max_entry_offset_pct must be >= 0")
        if self.default_stop_buffer_pct <= 0:
            raise ValueError("default_stop_buffer_pct must be > 0")
        if self.default_tp_rr <= 0:
            raise ValueError("default_tp_rr must be > 0")
        if self.max_expected_holding_seconds <= 0:
            raise ValueError("max_expected_holding_seconds must be > 0")


class CvdDivergenceStrategy(OrderflowStrategyBase):
    """
    CVD divergence strategy.

    Main idea:
    - bullish divergence:
        price_change_pct < 0 while CVD strengthens;
    - bearish divergence:
        price_change_pct > 0 while CVD weakens.

    Data contract:
    - primary source: SignalContext.orderflow["cvd"] / feature snapshots;
    - fallback source: scoped analytics.orderflow CVD stats;
    - scope: exchange + market_type + symbol + timeframe.
    """

    STRATEGY_NAME = "cvd_divergence_strategy"
    CATEGORY = StrategyCategory.ORDERFLOW
    DEFAULT_TIMEFRAME = Timeframe.M1

    REQUIRED_FEATURES = {
        "orderflow.cvd.price_change_pct",
        "orderflow.cvd.cvd_change_pct",
        "orderflow.cvd.delta_ratio",
    }

    def __init__(
        self,
        config: StrategyConfig,
        *,
        orderflow_analyzer: OrderFlowAnalyzer | None = None,
        thresholds: CvdDivergenceThresholds | None = None,
        event_bus: EventBus | None = None,
        logger: Any | None = None,
    ) -> None:
        super().__init__(
            config=config,
            orderflow_analyzer=orderflow_analyzer,
            event_bus=event_bus,
            logger=logger,
        )
        self.thresholds = thresholds or CvdDivergenceThresholds()

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

        stats = self._resolve_cvd_stats(context)
        if stats is None:
            return False

        if stats.trades_count < self.thresholds.min_trades_count:
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

        stats = self._resolve_cvd_stats(context)
        if stats is None:
            evaluation.reasons.append("cvd_stats_unavailable")
            return evaluation

        side = self._detect_divergence_side(stats)
        if side == SignalSide.UNKNOWN:
            evaluation.reasons.append("no_cvd_divergence_detected")
            return evaluation

        score = self._calculate_score(stats, side, context)
        confidence = self._calculate_confidence(stats, side, context)
        reasons = self._build_reasons(stats, side)
        confirmations = self._build_confirmations(stats, side, context)

        evaluation.score = score
        evaluation.confidence = confidence
        evaluation.reasons.extend(reasons)

        side_score_threshold = (
            self.thresholds.bullish_divergence_score_threshold
            if side == SignalSide.LONG
            else self.thresholds.bearish_divergence_score_threshold
        )
        min_score = max(self._get_min_score(), side_score_threshold)
        min_confidence = self._get_min_confidence()

        if score < min_score:
            evaluation.reasons.append("score_below_threshold")
            return evaluation

        if confidence < min_confidence:
            evaluation.reasons.append("confidence_below_threshold")
            return evaluation

        signal = self._build_signal(
            context=context,
            stats=stats,
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
    # Analytics integration
    # ------------------------------------------------------------------

    def _resolve_cvd_stats(self, context: SignalContext) -> CvdStats | None:
        """
        Resolve CVD stats using the new scoped orderflow contract.

        Priority:
        1. SignalContext.orderflow["cvd"] / feature snapshots;
        2. OrderFlowAnalyzer / CvdAnalyzer by OrderFlowKey;
        3. temporary legacy fallback inside base only if enabled there.
        """
        key = self._resolve_orderflow_key(context)

        stats = self._build_stats_from_context(context, key=key)
        if stats is not None:
            return stats

        raw_stats = self._safe_get_metric_stats_by_key("cvd", key)
        return self._coerce_cvd_stats(
            raw_stats,
            context=context,
            key=key,
            source="facade",
        )

    def _build_stats_from_context(
        self,
        context: SignalContext,
        *,
        key: OrderFlowKey,
    ) -> CvdStats | None:
        payload = self._extract_metric_payload(context, "cvd")
        if not payload:
            return None

        return self._coerce_cvd_stats(
            payload,
            context=context,
            key=key,
            source="context",
        )

    def _coerce_cvd_stats(
        self,
        value: Any,
        *,
        context: SignalContext,
        key: OrderFlowKey,
        source: str,
    ) -> CvdStats | None:
        if value is None:
            return None

        if isinstance(value, CvdStats):
            return value

        if isinstance(value, Mapping):
            data = dict(value)
        else:
            data = self._model_to_plain_dict(value)

        if not data:
            return None

        exchange, market_type, symbol, timeframe = key

        price_from_context = None
        if context.price is not None:
            price_from_context = context.price.last_price or context.price.mid_price

        timestamp = self._coalesce_float(
            data.get("timestamp"),
            self._context_timestamp_float(context),
        )

        try:
            return CvdStats(
                exchange=self._coalesce_str(data.get("exchange"), exchange) or exchange,
                market_type=(
                    self._coalesce_str(data.get("market_type"), market_type)
                    or market_type
                ),
                symbol=self._coalesce_str(data.get("symbol"), symbol) or symbol,
                exchange_symbol=self._coalesce_str(
                    data.get("exchange_symbol"),
                    symbol,
                ),
                timeframe=self._coalesce_str(data.get("timeframe"), timeframe) or timeframe,
                metric=data.get("metric", "cvd"),
                source_type=data.get("source_type", "trades"),
                timestamp=float(timestamp or 0.0),
                window_seconds=float(
                    self._coalesce_float(data.get("window_seconds"), 0.0) or 0.0
                ),
                trades_count=int(
                    self._coalesce_int(data.get("trades_count"), 0) or 0
                ),
                buy_volume=float(
                    self._coalesce_float(data.get("buy_volume"), 0.0) or 0.0
                ),
                sell_volume=float(
                    self._coalesce_float(data.get("sell_volume"), 0.0) or 0.0
                ),
                volume_delta=float(
                    self._coalesce_float(data.get("volume_delta"), 0.0) or 0.0
                ),
                buy_notional=float(
                    self._coalesce_float(data.get("buy_notional"), 0.0) or 0.0
                ),
                sell_notional=float(
                    self._coalesce_float(data.get("sell_notional"), 0.0) or 0.0
                ),
                notional_delta=float(
                    self._coalesce_float(data.get("notional_delta"), 0.0) or 0.0
                ),
                cvd_value=float(
                    self._coalesce_float(
                        data.get("cvd_value"),
                        data.get("value"),
                        data.get("cvd_close"),
                        data.get("close"),
                        0.0,
                    )
                    or 0.0
                ),
                cvd_open=float(
                    self._coalesce_float(
                        data.get("cvd_open"),
                        data.get("open"),
                        0.0,
                    )
                    or 0.0
                ),
                cvd_high=float(
                    self._coalesce_float(
                        data.get("cvd_high"),
                        data.get("high"),
                        0.0,
                    )
                    or 0.0
                ),
                cvd_low=float(
                    self._coalesce_float(
                        data.get("cvd_low"),
                        data.get("low"),
                        0.0,
                    )
                    or 0.0
                ),
                cvd_close=float(
                    self._coalesce_float(
                        data.get("cvd_close"),
                        data.get("close"),
                        data.get("cvd_value"),
                        data.get("value"),
                        0.0,
                    )
                    or 0.0
                ),
                cvd_change=float(
                    self._coalesce_float(
                        data.get("cvd_change"),
                        data.get("change"),
                        0.0,
                    )
                    or 0.0
                ),
                cvd_change_pct=float(
                    self._coalesce_float(
                        data.get("cvd_change_pct"),
                        data.get("change_pct"),
                        0.0,
                    )
                    or 0.0
                ),
                cvd_slope=float(
                    self._coalesce_float(
                        data.get("cvd_slope"),
                        data.get("slope"),
                        0.0,
                    )
                    or 0.0
                ),
                delta_ratio=float(
                    self._coalesce_float(data.get("delta_ratio"), 0.0) or 0.0
                ),
                buy_ratio=float(
                    self._coalesce_float(data.get("buy_ratio"), 0.0) or 0.0
                ),
                sell_ratio=float(
                    self._coalesce_float(data.get("sell_ratio"), 0.0) or 0.0
                ),
                avg_trade_size=float(
                    self._coalesce_float(data.get("avg_trade_size"), 0.0) or 0.0
                ),
                avg_trade_notional=float(
                    self._coalesce_float(data.get("avg_trade_notional"), 0.0) or 0.0
                ),
                last_price=self._coalesce_float(
                    data.get("last_price"),
                    price_from_context,
                ),
                price_change=self._coalesce_float(data.get("price_change")),
                price_change_pct=self._coalesce_float(data.get("price_change_pct")),
                metadata={
                    **(
                        dict(data.get("metadata"))
                        if isinstance(data.get("metadata"), Mapping)
                        else {}
                    ),
                    "strategy_source": source,
                    "scope": "exchange:market_type:symbol:timeframe",
                    "scope_payload": orderflow_key_to_dict(key),
                    "scope_key": orderflow_key_to_string(key),
                },
            )
        except Exception:
            self.log_warning(
                "Failed to reconstruct scoped CvdStats",
                symbol=symbol,
                strategy=self.STRATEGY_NAME,
                source=source,
                scope_key=orderflow_key_to_string(key),
            )
            return None

    # ------------------------------------------------------------------
    # Core divergence logic
    # ------------------------------------------------------------------

    def _detect_divergence_side(self, stats: CvdStats) -> SignalSide:
        price_change_pct = float(stats.price_change_pct or 0.0)
        cvd_change_pct = float(stats.cvd_change_pct or 0.0)
        delta_ratio = float(stats.delta_ratio)
        cvd_slope = float(stats.cvd_slope)

        bullish_divergence = (
            price_change_pct <= -abs(self.thresholds.min_abs_price_change_pct)
            and cvd_change_pct >= abs(self.thresholds.min_abs_cvd_change_pct)
            and delta_ratio >= abs(self.thresholds.min_abs_delta_ratio)
            and cvd_slope >= self.thresholds.min_abs_cvd_slope
        )

        bearish_divergence = (
            price_change_pct >= abs(self.thresholds.min_abs_price_change_pct)
            and cvd_change_pct <= -abs(self.thresholds.min_abs_cvd_change_pct)
            and delta_ratio <= -abs(self.thresholds.min_abs_delta_ratio)
            and cvd_slope <= -abs(self.thresholds.min_abs_cvd_slope)
        )

        if bullish_divergence and not bearish_divergence:
            return SignalSide.LONG

        if bearish_divergence and not bullish_divergence:
            return SignalSide.SHORT

        return SignalSide.UNKNOWN

    def _calculate_score(
        self,
        stats: CvdStats,
        side: SignalSide,
        context: SignalContext,
    ) -> float:
        price_component = self._normalize_percent(
            abs(float(stats.price_change_pct or 0.0)),
            scale=2.0,
        )
        cvd_pct_component = self._normalize_percent(
            abs(float(stats.cvd_change_pct or 0.0)),
            scale=2.0,
        )
        delta_component = self._normalize_ratio(
            abs(float(stats.delta_ratio)),
            scale=0.50,
        )
        slope_component = self._normalize_magnitude(
            abs(float(stats.cvd_slope)),
            scale=10.0,
        )
        notional_delta_component = self._normalize_ratio(
            abs(self._notional_delta_ratio(stats)),
            scale=0.50,
        )
        cvd_range_component = self._cvd_range_component(stats)

        flow_balance_component = (
            min(max(float(stats.buy_ratio), 0.0), 1.0)
            if side == SignalSide.LONG
            else min(max(float(stats.sell_ratio), 0.0), 1.0)
        )

        trades_component = min(
            float(stats.trades_count) / max(self.thresholds.min_trades_count * 2, 1),
            1.0,
        )

        raw_score = (
            (price_component * 0.16)
            + (cvd_pct_component * 0.22)
            + (delta_component * 0.18)
            + (slope_component * 0.12)
            + (notional_delta_component * 0.12)
            + (cvd_range_component * 0.08)
            + (flow_balance_component * 0.06)
            + (trades_component * 0.06)
        )

        weighted_score = raw_score
        weighted_score *= self._category_weight()
        weighted_score *= self._regime_adjustment(context)
        weighted_score *= self._strategy_weight()

        return max(0.0, weighted_score)

    def _calculate_confidence(
        self,
        stats: CvdStats,
        side: SignalSide,
        context: SignalContext,
    ) -> float:
        components: list[float] = [
            self._normalize_ratio(abs(float(stats.delta_ratio)), scale=0.35),
            self._normalize_percent(abs(float(stats.cvd_change_pct or 0.0)), scale=2.0),
            self._normalize_percent(abs(float(stats.price_change_pct or 0.0)), scale=2.0),
            self._normalize_magnitude(abs(float(stats.cvd_slope)), scale=10.0),
            self._normalize_ratio(abs(self._notional_delta_ratio(stats)), scale=0.35),
            self._cvd_range_component(stats),
            min(
                float(stats.trades_count) / max(self.thresholds.min_trades_count * 2, 1),
                1.0,
            ),
        ]

        if side == SignalSide.LONG:
            components.append(min(max(float(stats.buy_ratio), 0.0), 1.0))
            components.append(1.0 if stats.notional_delta > 0 else 0.35)
        elif side == SignalSide.SHORT:
            components.append(min(max(float(stats.sell_ratio), 0.0), 1.0))
            components.append(1.0 if stats.notional_delta < 0 else 0.35)

        if stats.avg_trade_notional > 0:
            components.append(0.75)

        if context.regime is not None and context.regime.regime in self.supported_regimes:
            components.append(0.75)

        if context.price is not None and context.price.spread_bps is not None:
            spread_ok = context.price.spread_bps <= self.config.filters.max_spread_bps
            components.append(1.0 if spread_ok else 0.35)

        confidence = sum(components) / len(components) if components else 0.0
        return max(0.0, min(confidence, 1.0))

    def _build_reasons(self, stats: CvdStats, side: SignalSide) -> list[str]:
        reasons: list[str] = []

        if side == SignalSide.LONG:
            reasons.extend(
                [
                    "price_declining_while_cvd_strengthens",
                    "bullish_cvd_divergence_detected",
                ]
            )

            if stats.delta_ratio > 0:
                reasons.append("positive_delta_ratio_confirmation")
            if stats.cvd_slope > 0:
                reasons.append("positive_cvd_slope_confirmation")
            if stats.notional_delta > 0:
                reasons.append("positive_notional_delta_confirmation")
            if stats.buy_ratio > stats.sell_ratio:
                reasons.append("buy_flow_dominance")

        elif side == SignalSide.SHORT:
            reasons.extend(
                [
                    "price_rising_while_cvd_weakens",
                    "bearish_cvd_divergence_detected",
                ]
            )

            if stats.delta_ratio < 0:
                reasons.append("negative_delta_ratio_confirmation")
            if stats.cvd_slope < 0:
                reasons.append("negative_cvd_slope_confirmation")
            if stats.notional_delta < 0:
                reasons.append("negative_notional_delta_confirmation")
            if stats.sell_ratio > stats.buy_ratio:
                reasons.append("sell_flow_dominance")

        if stats.trades_count >= self.thresholds.min_trades_count:
            reasons.append("sufficient_trade_sample")

        if self._cvd_range_component(stats) > 0:
            reasons.append("cvd_range_available")

        if stats.avg_trade_notional > 0:
            reasons.append("avg_trade_notional_available")

        return reasons

    def _build_confirmations(
        self,
        stats: CvdStats,
        side: SignalSide,
        context: SignalContext,
    ) -> list[str]:
        confirmations: list[str] = []

        if side == SignalSide.LONG:
            if stats.buy_ratio > stats.sell_ratio:
                confirmations.append("buy_flow_dominance")
            if stats.delta_ratio > 0:
                confirmations.append("positive_volume_delta")
            if stats.notional_delta > 0:
                confirmations.append("positive_notional_delta")
            if stats.cvd_close >= stats.cvd_open:
                confirmations.append("cvd_close_above_open")

        elif side == SignalSide.SHORT:
            if stats.sell_ratio > stats.buy_ratio:
                confirmations.append("sell_flow_dominance")
            if stats.delta_ratio < 0:
                confirmations.append("negative_volume_delta")
            if stats.notional_delta < 0:
                confirmations.append("negative_notional_delta")
            if stats.cvd_close <= stats.cvd_open:
                confirmations.append("cvd_close_below_open")

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
        stats: CvdStats,
        side: SignalSide,
        score: float,
        confidence: float,
        reasons: list[str],
        confirmations: list[str],
    ) -> StrategySignal:
        entry_plan = self._build_entry_plan(context, stats, side)
        exit_plan = self._build_exit_plan(context, stats, side, entry_plan)
        invalidation_plan = self._build_invalidation_plan(context, stats, side, entry_plan)
        execution_plan = self._build_execution_plan(
            context=context,
            stats=stats,
            side=side,
            entry_plan=entry_plan,
            exit_plan=exit_plan,
            invalidation_plan=invalidation_plan,
        )

        signal = StrategySignal(
            symbol=stats.symbol,
            side=side,
            strategy_name=self.STRATEGY_NAME,
            category=self.CATEGORY,
            timeframe=context.timeframe or self.DEFAULT_TIMEFRAME,
            setup_type=SetupType.REVERSAL,
            timestamp=context.timestamp,
            confidence=confidence,
            score=score,
            strength=self._map_strength(confidence),
            confidence_grade=self._map_confidence_grade(confidence),
            status=SignalStatus.NEW,
            trigger_type=TriggerType.PRIMARY,
            origin=SignalOrigin.SINGLE_STRATEGY,
            priority=self._resolve_priority(confidence),
            regime=context.regime.regime if context.regime is not None else MarketRegime.UNKNOWN,
            entry_plan=entry_plan,
            exit_plan=exit_plan,
            invalidation_plan=invalidation_plan,
            execution_plan=execution_plan,
            metadata={
                "source": self.STRATEGY_NAME,
                "analytics_metric": "cvd",
                "scope": self._stats_scope_payload(stats),
                "scope_key": orderflow_key_to_string(stats.key),
                "key": list(stats.key),
                "uses_orderflow_analyzer_fallback": self.orderflow_analyzer is not None,
                "cvd_snapshot": self._cvd_snapshot_payload(stats),
            },
        )

        for reason in reasons:
            signal.add_reason(reason)

        for confirmation in confirmations:
            signal.add_confirmation(confirmation)

        for feature_name in self.required_features():
            signal.add_source_feature(feature_name)

        signal.add_source_feature("orderflow.cvd")

        return signal

    # ------------------------------------------------------------------
    # Plans
    # ------------------------------------------------------------------

    def _build_entry_plan(
        self,
        context: SignalContext,
        stats: CvdStats,
        side: SignalSide,
    ) -> EntryPlan:
        ref_price = self._resolve_reference_price(context, stats)
        entry_price = None

        if ref_price is not None:
            offset = ref_price * self.thresholds.max_entry_offset_pct

            if side == SignalSide.LONG:
                entry_price = ref_price - offset
            elif side == SignalSide.SHORT:
                entry_price = ref_price + offset

        return EntryPlan(
            entry_type=getattr(self.config.builders, "default_entry_type", EntryType.MARKET),
            price=entry_price,
            confirmation_required=False,
            notes=[
                "entry_based_on_cvd_divergence",
                "prefer_execution_near_reference_price",
            ],
            metadata={
                "reference_price": ref_price,
                "entry_offset_pct": self.thresholds.max_entry_offset_pct,
                "scope": self._stats_scope_payload(stats),
                "scope_key": orderflow_key_to_string(stats.key),
            },
        )

    def _build_exit_plan(
        self,
        context: SignalContext,
        stats: CvdStats,
        side: SignalSide,
        entry_plan: EntryPlan,
    ) -> ExitPlan:
        ref_price = entry_plan.price or self._resolve_reference_price(context, stats)

        stop_loss = None
        tp_price = None

        rr = getattr(
            self.config.builders,
            "default_rr_ratio",
            self.thresholds.default_tp_rr,
        )
        rr = rr if rr and rr > 0 else self.thresholds.default_tp_rr

        if ref_price is not None:
            stop_buffer = ref_price * self.thresholds.default_stop_buffer_pct

            if side == SignalSide.LONG:
                stop_loss = ref_price - stop_buffer
                risk = max(ref_price - stop_loss, 0.0)
                tp_price = ref_price + (risk * rr)

            elif side == SignalSide.SHORT:
                stop_loss = ref_price + stop_buffer
                risk = max(stop_loss - ref_price, 0.0)
                tp_price = ref_price - (risk * rr)

        targets: list[TargetPlan] = []

        if tp_price is not None and tp_price > 0:
            targets.append(
                TargetPlan(
                    price=tp_price,
                    size_fraction=1.0,
                    rr=rr,
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
            take_profit_levels=targets,
            partial_exit_enabled=getattr(
                self.config.builders,
                "enable_partial_take_profit",
                True,
            ),
            metadata={
                "rr_ratio": rr,
                "strategy": self.STRATEGY_NAME,
                "scope": self._stats_scope_payload(stats),
                "scope_key": orderflow_key_to_string(stats.key),
            },
        )

    def _build_invalidation_plan(
        self,
        context: SignalContext,
        stats: CvdStats,
        side: SignalSide,
        entry_plan: EntryPlan,
    ) -> InvalidationPlan:
        ref_price = entry_plan.price or self._resolve_reference_price(context, stats)
        invalidation_price = None

        if ref_price is not None:
            buffer = ref_price * self.thresholds.default_stop_buffer_pct

            if side == SignalSide.LONG:
                invalidation_price = ref_price - buffer
            elif side == SignalSide.SHORT:
                invalidation_price = ref_price + buffer

        return InvalidationPlan(
            price=invalidation_price,
            reason="cvd_divergence_failed",
            conditions=[
                "delta_ratio_flips_against_position",
                "cvd_slope_reverts_against_position",
                "cvd_change_pct_reverts_against_position",
                "notional_delta_confirms_failed_divergence",
            ],
            metadata={
                "strategy": self.STRATEGY_NAME,
                "scope": self._stats_scope_payload(stats),
                "scope_key": orderflow_key_to_string(stats.key),
            },
        )

    def _build_execution_plan(
        self,
        *,
        context: SignalContext,
        stats: CvdStats,
        side: SignalSide,
        entry_plan: EntryPlan,
        exit_plan: ExitPlan,
        invalidation_plan: InvalidationPlan,
    ) -> ExecutionPlanDraft:
        return ExecutionPlanDraft(
            symbol=stats.symbol,
            side=side,
            entry=entry_plan,
            exit=exit_plan,
            invalidation=invalidation_plan,
            expected_holding_seconds=self.thresholds.max_expected_holding_seconds,
            notes=[
                "generated_from_cvd_divergence_strategy",
            ],
            metadata={
                "timeframe": str(context.timeframe),
                "strategy_name": self.STRATEGY_NAME,
                "scope": self._stats_scope_payload(stats),
                "scope_key": orderflow_key_to_string(stats.key),
            },
        )

    # ------------------------------------------------------------------
    # CVD helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _total_volume(stats: CvdStats) -> float:
        return max(float(stats.buy_volume) + float(stats.sell_volume), 0.0)

    @staticmethod
    def _total_notional(stats: CvdStats) -> float:
        return max(float(stats.buy_notional) + float(stats.sell_notional), 0.0)

    def _notional_delta_ratio(self, stats: CvdStats) -> float:
        total_notional = self._total_notional(stats)
        if total_notional <= 0:
            return 0.0
        return float(stats.notional_delta) / total_notional

    @staticmethod
    def _cvd_range_component(stats: CvdStats) -> float:
        cvd_range = abs(float(stats.cvd_high) - float(stats.cvd_low))
        if cvd_range <= 0:
            return 0.0

        return max(0.0, min(abs(float(stats.cvd_change)) / cvd_range, 1.0))

    @staticmethod
    def _stats_scope_payload(stats: CvdStats) -> dict[str, Any]:
        return {
            "exchange": stats.exchange,
            "market_type": stats.market_type,
            "symbol": stats.symbol,
            "exchange_symbol": stats.exchange_symbol,
            "timeframe": stats.timeframe,
        }

    def _cvd_snapshot_payload(self, stats: CvdStats) -> dict[str, Any]:
        return {
            "exchange": stats.exchange,
            "market_type": stats.market_type,
            "symbol": stats.symbol,
            "exchange_symbol": stats.exchange_symbol,
            "timeframe": stats.timeframe,
            "key": list(stats.key),
            "scope": self._stats_scope_payload(stats),
            "scope_key": orderflow_key_to_string(stats.key),
            "timestamp": stats.timestamp,
            "window_seconds": stats.window_seconds,
            "trades_count": stats.trades_count,
            "buy_volume": stats.buy_volume,
            "sell_volume": stats.sell_volume,
            "total_volume": self._total_volume(stats),
            "volume_delta": stats.volume_delta,
            "buy_notional": stats.buy_notional,
            "sell_notional": stats.sell_notional,
            "total_notional": self._total_notional(stats),
            "notional_delta": stats.notional_delta,
            "notional_delta_ratio": self._notional_delta_ratio(stats),
            "cvd_value": stats.cvd_value,
            "cvd_open": stats.cvd_open,
            "cvd_high": stats.cvd_high,
            "cvd_low": stats.cvd_low,
            "cvd_close": stats.cvd_close,
            "cvd_change": stats.cvd_change,
            "cvd_change_pct": stats.cvd_change_pct,
            "cvd_slope": stats.cvd_slope,
            "cvd_range_component": self._cvd_range_component(stats),
            "delta_ratio": stats.delta_ratio,
            "buy_ratio": stats.buy_ratio,
            "sell_ratio": stats.sell_ratio,
            "avg_trade_size": stats.avg_trade_size,
            "avg_trade_notional": stats.avg_trade_notional,
            "last_price": stats.last_price,
            "price_change": stats.price_change,
            "price_change_pct": stats.price_change_pct,
        }