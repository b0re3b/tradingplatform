from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from analytics.orderflow import CvdStats, OrderFlowAnalyzer
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
    Внутрішні пороги саме для divergence-логіки strategy layer.

    Це НЕ дублювання config analytics.
    Це пороги вже на рівні прийняття рішення strategy.
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
    Strategy для пошуку divergence між ціною і CVD.

    Основна ідея:
    - bearish divergence:
        price_change > 0, а CVD weak/negative
    - bullish divergence:
        price_change < 0, а CVD strong/positive

    Джерела даних:
    1. SignalContext.feature_map / context.orderflow
    2. OrderFlowAnalyzer facade fallback
    3. CvdAnalyzer через facade.get_module("cvd") / facade.cvd
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

        min_score = self._get_min_score()
        min_confidence = self._get_min_confidence()

        side_score_threshold = (
            self.thresholds.bullish_divergence_score_threshold
            if side == SignalSide.LONG
            else self.thresholds.bearish_divergence_score_threshold
        )
        min_score = max(min_score, side_score_threshold)

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
        Пріоритет:
        1. context.feature_map / context.orderflow
        2. orderflow_analyzer.get_latest_stats(symbol)
        3. orderflow_analyzer.cvd.get_latest_stats(symbol)
        4. orderflow_analyzer.get_module("cvd").get_latest_stats(symbol)
        """
        stats = self._build_stats_from_context(context)
        if stats is not None:
            return stats

        facade = self.orderflow_analyzer
        if facade is None:
            return None

        try:
            get_latest_stats = getattr(facade, "get_latest_stats", None)
            if callable(get_latest_stats):
                result = get_latest_stats(context.symbol)
                if isinstance(result, dict):
                    cvd_result = result.get("cvd")
                    if isinstance(cvd_result, CvdStats):
                        return cvd_result
        except Exception:
            self.log_warning(
                "Failed to resolve latest stats from orderflow facade",
                symbol=context.symbol,
                strategy=self.STRATEGY_NAME,
            )

        try:
            cvd_analyzer = getattr(facade, "cvd", None)
            if cvd_analyzer is not None:
                result = cvd_analyzer.get_latest_stats(context.symbol)
                if isinstance(result, CvdStats):
                    return result
        except Exception:
            self.log_warning(
                "Failed to resolve latest stats from facade.cvd",
                symbol=context.symbol,
                strategy=self.STRATEGY_NAME,
            )

        try:
            module = facade.get_module("cvd") if hasattr(facade, "get_module") else None
            if module is not None:
                result = module.get_latest_stats(context.symbol)
                if isinstance(result, CvdStats):
                    return result
        except Exception:
            self.log_warning(
                "Failed to resolve latest stats from facade.get_module('cvd')",
                symbol=context.symbol,
                strategy=self.STRATEGY_NAME,
            )

        return None

    def _build_stats_from_context(self, context: SignalContext) -> CvdStats | None:
        """
        Відновлює CvdStats з SignalContext.

        Підтримує:
        - context.orderflow["cvd"]
        - context.get_feature_snapshot(...)
        """
        symbol = context.symbol
        timestamp = context.timestamp.timestamp()

        data = self._extract_orderflow_cvd_payload(context)
        if not data:
            return None

        price_from_context = None
        if context.price is not None:
            price_from_context = context.price.last_price or context.price.mid_price

        try:
            return CvdStats(
                symbol=symbol,
                metric=data.get("metric", "cvd"),
                source_type=data.get("source_type", "trades"),
                timestamp=float(data.get("timestamp", timestamp)),
                window_seconds=float(data.get("window_seconds", 0.0)),
                trades_count=int(data.get("trades_count", 0)),
                buy_volume=float(data.get("buy_volume", 0.0)),
                sell_volume=float(data.get("sell_volume", 0.0)),
                volume_delta=float(data.get("volume_delta", 0.0)),
                buy_notional=float(data.get("buy_notional", 0.0)),
                sell_notional=float(data.get("sell_notional", 0.0)),
                notional_delta=float(data.get("notional_delta", 0.0)),
                total_volume=float(data.get("total_volume", 0.0)),
                total_notional=float(data.get("total_notional", 0.0)),
                cvd_open=float(data.get("cvd_open", 0.0)),
                cvd_high=float(data.get("cvd_high", 0.0)),
                cvd_low=float(data.get("cvd_low", 0.0)),
                cvd_close=float(data.get("cvd_close", 0.0)),
                cvd_value=float(data.get("cvd_value", data.get("cvd_close", 0.0))),
                cvd_change=float(data.get("cvd_change", 0.0)),
                cvd_change_pct=float(data.get("cvd_change_pct", 0.0)),
                cvd_slope=float(data.get("cvd_slope", 0.0)),
                delta_ratio=float(data.get("delta_ratio", 0.0)),
                buy_ratio=float(data.get("buy_ratio", 0.0)),
                sell_ratio=float(data.get("sell_ratio", 0.0)),
                avg_trade_size=float(data.get("avg_trade_size", 0.0)),
                avg_trade_notional=float(data.get("avg_trade_notional", 0.0)),
                last_price=self._coalesce_float(
                    data.get("last_price"),
                    price_from_context,
                ),
                price_change=self._coalesce_float(data.get("price_change")),
                price_change_pct=self._coalesce_float(data.get("price_change_pct")),
            )
        except Exception:
            self.log_warning(
                "Failed to reconstruct CvdStats from context",
                symbol=context.symbol,
                strategy=self.STRATEGY_NAME,
            )
            return None

    def _extract_orderflow_cvd_payload(self, context: SignalContext) -> dict[str, Any]:
        payload: dict[str, Any] = {}

        orderflow = context.orderflow if isinstance(context.orderflow, dict) else {}
        raw_orderflow = orderflow.get("cvd")

        if isinstance(raw_orderflow, dict):
            payload.update(raw_orderflow)

        feature_aliases = {
            "window_seconds": ["orderflow.cvd.window_seconds"],
            "trades_count": ["orderflow.cvd.trades_count"],
            "buy_volume": ["orderflow.cvd.buy_volume"],
            "sell_volume": ["orderflow.cvd.sell_volume"],
            "volume_delta": ["orderflow.cvd.volume_delta"],
            "buy_notional": ["orderflow.cvd.buy_notional"],
            "sell_notional": ["orderflow.cvd.sell_notional"],
            "notional_delta": ["orderflow.cvd.notional_delta"],
            "total_volume": ["orderflow.cvd.total_volume"],
            "total_notional": ["orderflow.cvd.total_notional"],
            "cvd_open": ["orderflow.cvd.open", "orderflow.cvd.cvd_open"],
            "cvd_high": ["orderflow.cvd.high", "orderflow.cvd.cvd_high"],
            "cvd_low": ["orderflow.cvd.low", "orderflow.cvd.cvd_low"],
            "cvd_close": ["orderflow.cvd.close", "orderflow.cvd.cvd_close"],
            "cvd_value": ["orderflow.cvd.value", "orderflow.cvd.cvd_value"],
            "cvd_change": ["orderflow.cvd.change", "orderflow.cvd.cvd_change"],
            "cvd_change_pct": ["orderflow.cvd.change_pct", "orderflow.cvd.cvd_change_pct"],
            "cvd_slope": ["orderflow.cvd.slope", "orderflow.cvd.cvd_slope"],
            "delta_ratio": ["orderflow.cvd.delta_ratio"],
            "buy_ratio": ["orderflow.cvd.buy_ratio"],
            "sell_ratio": ["orderflow.cvd.sell_ratio"],
            "avg_trade_size": ["orderflow.cvd.avg_trade_size"],
            "avg_trade_notional": ["orderflow.cvd.avg_trade_notional"],
            "last_price": ["orderflow.cvd.last_price"],
            "price_change": ["orderflow.cvd.price_change"],
            "price_change_pct": ["orderflow.cvd.price_change_pct"],
        }

        for target_name, aliases in feature_aliases.items():
            if target_name in payload:
                continue

            for alias in aliases:
                snapshot = context.get_feature_snapshot(alias)
                if snapshot is not None:
                    payload[target_name] = snapshot.value
                    break

        return payload

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
        cvd_component = self._normalize_percent(
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
        flow_balance_component = max(
            min(float(stats.buy_ratio), 1.0),
            min(float(stats.sell_ratio), 1.0),
        )
        trades_component = min(
            float(stats.trades_count) / max(self.thresholds.min_trades_count * 2, 1),
            1.0,
        )

        raw_score = (
            (price_component * 0.18)
            + (cvd_component * 0.26)
            + (delta_component * 0.22)
            + (slope_component * 0.14)
            + (flow_balance_component * 0.10)
            + (trades_component * 0.10)
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
            min(
                float(stats.trades_count) / max(self.thresholds.min_trades_count * 2, 1),
                1.0,
            ),
        ]

        if side == SignalSide.LONG:
            components.append(min(max(float(stats.buy_ratio), 0.0), 1.0))
        elif side == SignalSide.SHORT:
            components.append(min(max(float(stats.sell_ratio), 0.0), 1.0))

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
            reasons.append("price_declining_while_cvd_strengthens")
            reasons.append("bullish_cvd_divergence_detected")

            if stats.delta_ratio > 0:
                reasons.append("positive_delta_ratio_confirmation")
            if stats.cvd_slope > 0:
                reasons.append("positive_cvd_slope_confirmation")

        elif side == SignalSide.SHORT:
            reasons.append("price_rising_while_cvd_weakens")
            reasons.append("bearish_cvd_divergence_detected")

            if stats.delta_ratio < 0:
                reasons.append("negative_delta_ratio_confirmation")
            if stats.cvd_slope < 0:
                reasons.append("negative_cvd_slope_confirmation")

        if stats.trades_count >= self.thresholds.min_trades_count:
            reasons.append("sufficient_trade_sample")

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

        elif side == SignalSide.SHORT:
            if stats.sell_ratio > stats.buy_ratio:
                confirmations.append("sell_flow_dominance")
            if stats.delta_ratio < 0:
                confirmations.append("negative_volume_delta")

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
            side=side,
            entry_plan=entry_plan,
            exit_plan=exit_plan,
            invalidation_plan=invalidation_plan,
        )

        signal = StrategySignal(
            symbol=context.symbol,
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
                "uses_orderflow_analyzer_fallback": self.orderflow_analyzer is not None,
                "cvd_snapshot": {
                    "window_seconds": stats.window_seconds,
                    "trades_count": stats.trades_count,
                    "cvd_value": stats.cvd_value,
                    "cvd_change": stats.cvd_change,
                    "cvd_change_pct": stats.cvd_change_pct,
                    "cvd_slope": stats.cvd_slope,
                    "delta_ratio": stats.delta_ratio,
                    "buy_ratio": stats.buy_ratio,
                    "sell_ratio": stats.sell_ratio,
                    "price_change": stats.price_change,
                    "price_change_pct": stats.price_change_pct,
                    "last_price": stats.last_price,
                },
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
            ],
            metadata={
                "strategy": self.STRATEGY_NAME,
            },
        )

    def _build_execution_plan(
        self,
        *,
        context: SignalContext,
        side: SignalSide,
        entry_plan: EntryPlan,
        exit_plan: ExitPlan,
        invalidation_plan: InvalidationPlan,
    ) -> ExecutionPlanDraft:
        return ExecutionPlanDraft(
            symbol=context.symbol,
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
            },
        )